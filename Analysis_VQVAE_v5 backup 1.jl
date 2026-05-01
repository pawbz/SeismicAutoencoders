### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ b90f919b-1e27-4a89-8a2c-c9dd990d5ec0
using JLD2,
    DSP,
    Statistics,
    LinearAlgebra,
    PlutoUI,
    PlutoLinks,
    PlutoHooks,
    PlutoPlotly,
    ColorSchemes,
    Colors,
StatsBase

# ╔═╡ f926472a-434d-4f57-b9ee-f3e8455db857
using FFTW

# ╔═╡ 1aee8298-a9f8-4ae8-a774-e07948c1419b
using Peaks, Printf

# ╔═╡ 9442424b-4d6d-435a-8d75-a80c359c6f6e


# ╔═╡ c0936154-0a83-41a6-82f3-a0ec01295820
vqvae = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/VQVAE_architecture_v5.jl")

# ╔═╡ 133b0599-8022-4dc6-9713-95b6ab5eb0ba
mft = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/MFT.jl")

# ╔═╡ 073120bf-7da5-4192-9440-f2a9cfa5e8a0
TableOfContents(include_definitions=true)

# ╔═╡ 6f130ae5-f43b-454a-9709-aca2d79cae40
md"""
# VQ-VAE v5 Analysis Notebook

Load a saved `analysis_cache.jld2` plus raw data and analyze one receiver pair at a time without retraining.
"""

# ╔═╡ 54157987-f4ee-4646-939c-92303f8a5322
md"## Paths"

# ╔═╡ d4f82278-14b8-4d5d-ba0a-618ebdaa1910
dt = 1.0

# ╔═╡ 3b7fec31-93ce-45a8-a8c8-893090f49b41
begin
    period_min = 20
    period_max = 80
    responsetype = Bandpass(inv(period_max), inv(period_min))
    designmethod = Butterworth(2)
    digfilter = digitalfilter(responsetype, designmethod; fs=inv(dt))
end

# ╔═╡ af11fe78-eeea-4126-a187-f191f1e50e8d
function taper(x)
    w = cat(tukey(size(x, 1), 0.1), dims=ndims(x))
    return w .* x
end

# ╔═╡ ac12596d-984c-4ac2-8093-68c086009468
function get_acausal_causal(pair::String, filepath::String)
    jldfile = load(filter(x -> occursin(pair, x), readdir(filepath, join=true))[1])
    correlations = jldfile["correlations"]
    headers = jldfile["headers"]
    distance = jldfile["dist"]
    return (; correlations, headers, distance)
end

# ╔═╡ edd2c342-ac56-4159-9d39-75271aa41656
function split_causal_acausal(X::AbstractMatrix, zero_lag::Bool, max_lag=nothing)
    nt, ntr = size(X)
    !isodd(nt) && error("nt should be odd")
    center = div(nt + 1, 2)
    half = div(nt - 1, 2)
    N = isnothing(max_lag) ? half : max(0, min(half, max_lag))
    if N == 0
        return similar(X, 0, ntr), similar(X, 0, ntr)
    end
    X_acausal = reverse(X[center-N:center-1, :], dims=1)
    X_causal = X[center+1:center+N, :]
    if zero_lag
        return vcat(zeros(1, size(X)[2:end]...), Array(X_acausal)),
            vcat(zeros(1, size(X)[2:end]...), Array(X_causal))
    else
        return Array(X_acausal), Array(X_causal)
    end
end

# ╔═╡ 296053a9-8834-4ab4-84a3-951458e5fbe0
function build_analysis_bundle(pair; filepath)
    pair_name = join(pair, "_")
    data_pair_local = get_acausal_causal(pair_name, filepath)
    D1 = data_pair_local.correlations
    D1 = normalise(D1, dims=1)
    D1ac, D1c = split_causal_acausal(D1, true)
    D1ac = taper(D1ac)
    D1c = taper(D1c)
    D1fac = filtfilt(digfilter, D1ac)
    D1fc = filtfilt(digfilter, D1c)
    D1fac = Float32.(normalise(D1fac[2:end, :], dims=1))
    D1fc = Float32.(normalise(D1fc[2:end, :], dims=1))
    return (; pair=pair, D1fac, D1fc, distance=data_pair_local.distance)
end

# ╔═╡ 2f06b9a3-cee3-45b6-bf23-b029199b4be5
function list_cache_runs(raw_data_dir::String)
    dirs = filter(path -> isdir(path) && startswith(basename(path), "vqvae_v5_run_"),
        readdir(raw_data_dir, join=true))
    sort!(dirs)
    return dirs
end

# ╔═╡ e23c2651-385e-4f6d-99af-183ec50d8ffa
raw_data_dir = "/mnt/NAS2/Sanket_data/California_2013_BK_CI_20032026/"

# ╔═╡ 7fe32a39-7679-4f62-ba75-c883f3ae0f64
cache_run_dirs = list_cache_runs(raw_data_dir)

# ╔═╡ 68439964-d258-4efd-8d53-fc9771dca4cb
md"Found **$(length(cache_run_dirs))** saved cache runs in $(raw_data_dir)"

# ╔═╡ fa401085-a4a3-4d66-9c2c-88a07a0f7895
begin
    isempty(cache_run_dirs) && error("No cache run directories found in $(raw_data_dir)")
    cache_options = basename.(cache_run_dirs)
    @bind selected_cache_run confirm(Select(cache_options, default=last(cache_options)))
end

# ╔═╡ 8d93dcb2-bafd-42e2-9880-7bc668e29462
selected_cache_dir = joinpath(raw_data_dir, selected_cache_run)

# ╔═╡ 3f6f634d-edf2-433c-9c33-e9b6b02994d0
analysis_cache = load(joinpath(selected_cache_dir, "analysis_cache.jld2"))["analysis_cache"]

# ╔═╡ 2963c4ff-800a-43d5-aeae-7c8939806fc6
loss_history = analysis_cache.loss_history

# ╔═╡ 8a482231-b1cb-4efc-8a90-4e5f640ee937
pair_metadata = analysis_cache.pair_metadata

# ╔═╡ b0ef58e8-a0b5-43dd-8122-07d6564c90e3
pair_names = [pm.pair_name for pm in pair_metadata]

# ╔═╡ f4fc8119-99f6-450b-b2d5-b715e256d6ab
md"## Pair Selection"

# ╔═╡ 61a7b9b3-1896-4e03-b597-df16f9d00059
begin
    @bind selected_pair_name confirm(Select(pair_names, default=first(pair_names)))
end

# ╔═╡ e0888c71-0249-4626-ae74-0f602b81c00f
selected_pair_id = findfirst(==(selected_pair_name), pair_names)

# ╔═╡ c93cb971-370d-4daf-aa63-49d104b7e375
selected_pair_meta = pair_metadata[selected_pair_id]

# ╔═╡ 04733c54-491a-4bc7-8a9f-ae8fa2b82147
selected_pair = Tuple(selected_pair_meta.pair)

# ╔═╡ 48135e48-036f-433a-8ec4-a541b5883ca1
encoded_cache = analysis_cache.all_pair_encoded_cache[selected_pair_id]

# ╔═╡ e09728b7-ac67-403c-a392-ed648437203d
data_bundle = build_analysis_bundle(selected_pair; filepath=raw_data_dir)

# ╔═╡ 23086044-d104-49f7-8439-f541f869fcc8
data = (;
    D_ac_all=data_bundle.D1fac,
    D_c_all=data_bundle.D1fc,
)

# ╔═╡ a39233a4-8cca-419d-831d-51905889da04
nth = size(data.D_ac_all, 1)

# ╔═╡ 0bc30aa0-069e-40ea-81fa-a3d60edac016
t_neg = [-(nth - i + 1) * dt for i in 1:nth]

# ╔═╡ 2b2d7ee0-6cd3-4b33-8956-fee6916bb4dd
t_pos = [i * dt for i in 1:nth]

# ╔═╡ d1597c31-bc78-47bf-ba2d-21adc4b60152
t_full = [t_neg; t_pos]

# ╔═╡ 532da58f-a1d4-4d61-b5c6-b97bf91b85ed
vqvae_para = analysis_cache.vqvae_para

# ╔═╡ dcc4775c-f909-4260-a1da-102ad1d4b3f5
combo_labels = vqvae.combination_labels(vqvae_para.Ksmall, vqvae_para.T)

# ╔═╡ 1713faaa-12ae-410c-b08f-1b40d713047c
function cluster_averages_from_codes(x_cpu, ci; K::Int)
    nt = size(x_cpu, 1)
    ncomb = K^size(ci, 1)
    out = zeros(Float32, nt, ncomb)
    counts = zeros(Int, ncomb)
    for j in 1:size(ci, 2)
        combo_idx = 1
        for t in 1:size(ci, 1)
            combo_idx += (ci[t, j] - 1) * (K^(t - 1))
        end
        out[:, combo_idx] .+= x_cpu[:, j]
        counts[combo_idx] += 1
    end
    for k in 1:ncomb
        counts[k] > 0 && (out[:, k] ./= counts[k])
    end
    return out
end

# ╔═╡ 353ca4f7-976e-4581-a0e2-95f57d83fa90
cluster_avg_ac = cluster_averages_from_codes(data.D_ac_all, encoded_cache.ci_ac; K=vqvae_para.Ksmall)

# ╔═╡ 4af4c843-8a28-4d62-ba63-9dda2061465e
cluster_avg_c = cluster_averages_from_codes(data.D_c_all, encoded_cache.ci_c; K=vqvae_para.Ksmall)

# ╔═╡ 980da5c6-a04c-483a-8c72-875ad73660fa
cross = let
    ci_ac = encoded_cache.ci_ac
    ci_c = encoded_cache.ci_c
    K = vqvae_para.Ksmall
    T = size(ci_ac, 1)
    ncomb = K^T
    counts_ac = zeros(Float32, ncomb)
    counts_c = zeros(Float32, ncomb)

    combo_index(digits) = begin
        idx = 1
        for t in 1:T
            idx += (digits[t] - 1) * (K^(t - 1))
        end
        idx
    end

    for j in 1:size(ci_ac, 2)
        counts_ac[combo_index(ci_ac[:, j])] += 1f0
    end
    for j in 1:size(ci_c, 2)
        counts_c[combo_index(ci_c[:, j])] += 1f0
    end

    nw = min(size(ci_ac, 2), size(ci_c, 2))
    idx_ac = Vector{Int}(undef, nw)
    idx_c = Vector{Int}(undef, nw)
    confusion = zeros(Float32, ncomb, ncomb)
    for w in 1:nw
        idx_aw = combo_index(ci_ac[:, w])
        idx_cw = combo_index(ci_c[:, w])
        idx_ac[w] = idx_aw
        idx_c[w] = idx_cw
        confusion[idx_aw, idx_cw] += 1f0
    end
    confusion ./= max(sum(confusion), 1f-10)
    pct_ac = counts_ac ./ max(sum(counts_ac), 1f-10) .* 100f0
    pct_c = counts_c ./ max(sum(counts_c), 1f-10) .* 100f0
    (; confusion, pct_ac, pct_c, labels=combo_labels)
end

# ╔═╡ cd25e99a-2a5c-484c-892c-02a6136a431c
md"## Training Summary"

# ╔═╡ 204ddd82-6536-42f1-a23c-8cff9e924af7
WideCell(vqvae.plot_training_dashboard(loss_history;
    title="VQ-VAE v5: $(selected_pair_name) loss history from saved cache"))

# ╔═╡ 65891fdf-1074-4450-a0f3-9e4b7743fc0c
md"## Source State Analysis"

# ╔═╡ c8445edf-206a-4a64-98f6-1cf53221b994
WideCell(vqvae.plot_state_usage(cross.pct_ac, cross.pct_c;
    labels=cross.labels,
    title="$(selected_pair_name) Source State Usage"))

# ╔═╡ 3dc0a19c-2670-4e97-bd4e-e93b2f53ad71
WideCell(vqvae.plot_codebook_confusion(cross.confusion;
    title="$(selected_pair_name) Code Confusion",
    labels=cross.labels))

# ╔═╡ 51e6d80b-a400-48b6-badf-2a1456d7c823
let
    global_avg_ac = vec(mean(data.D_ac_all; dims=2))
    global_avg_c = vec(mean(data.D_c_all; dims=2))
    global_full = [reverse(global_avg_ac); global_avg_c]
    ncomb = length(combo_labels)
    traces = AbstractTrace[]
    cs = ColorSchemes.rainbow
    colors = [Colors.hex(get(cs, (i - 1) / max(1, ncomb - 1))) for i in 1:ncomb]
    mean_ac = vec(mean(cluster_avg_ac; dims=1))
    mean_c = vec(mean(cluster_avg_c; dims=1))
    vertical_spacing = maximum(abs.(vcat(mean_ac, mean_c))) * 2.5 + 1e-3

    for combo_idx in 1:ncomb
        a = cluster_avg_ac[:, combo_idx]
        b = cluster_avg_c[:, combo_idx]
        joined = [reverse(a); b]
        offset = (combo_idx - 1) * vertical_spacing
        push!(traces, PlutoPlotly.scatter(
            x=t_full, y=joined .+ offset, mode="lines",
            line=attr(color=colors[combo_idx], width=2),
            name="State $(combo_labels[combo_idx])"))
    end

    push!(traces, PlutoPlotly.scatter(
        x=t_full, y=global_full, mode="lines",
        line=attr(color="black", width=2, dash="dot"),
        name="Global mean"))

    layout = Layout(
        title=attr(text="Source State Average Waveforms ($(selected_pair_name)) $(round(Int, data_bundle.distance))km"),
        height=550, width=950,
        xaxis=attr(title="Lag (s)"),
        yaxis=attr(title="Amplitude + offset"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ 09931af2-91c2-4ac0-be11-183fc3bbbad5
md"## MFT"

# ╔═╡ ebc826f7-df75-4d9c-87c4-ec7fab19e9f9
mft_analysis_all_states = let
    nstates = size(cluster_avg_ac, 2)
    ac_traces = [
        mft.SeismicTrace(data=vec(cluster_avg_ac[:, i]), dt=dt, distance=data_bundle.distance)
        for i in 1:nstates
    ]
    c_traces = [
        mft.SeismicTrace(data=vec(cluster_avg_c[:, i]), dt=dt, distance=data_bundle.distance)
        for i in 1:nstates
    ]
    mft.analyze_causal_acausal_branches(
        ac_traces,
        c_traces;
        state_labels=combo_labels,
        period_max=80.0,
        velocity_range=(1.0, 8.0),
        bandwidth_factor=0.15,
        zero_pad_factor=4,
    )
end

# ╔═╡ 25ad9c95-4def-4fca-b0b3-829f033cb011
@bind ui_period Slider(mft_analysis_all_states.periods, show_value=true)

# ╔═╡ 562c024b-1564-4a53-acf3-1c21df16b698
WideCell(
    mft.plot_filtered_traces_by_period(
        mft_analysis_all_states;
        period=ui_period,
        correlation_threshold=nothing,
        normalize_each=true,
        scale=0.7,
        spacing=2.2,
        title="$(selected_pair_name) Filtered Traces by Source State"
    )
)

# ╔═╡ a7604fea-de0a-4598-8e1c-777e709e5a5c
WideCell(mft.plot_branch_correlation(mft_analysis_all_states;
    title="$(selected_pair_name) Branch Correlation Across Source States"))

# ╔═╡ 0aca9aea-e2e0-4819-9bbe-3ba7bf0ae3bd
WideCell(mft.plot_all_highcorr_groupvelocity_picks(mft_analysis_all_states;
    correlation_threshold=0.9,
    title="Group Velocity Picks $(selected_pair_name)"))

# ╔═╡ 2e47641b-af82-47c0-82dc-78da299517b3
WideCell(mft.plot_all_highcorr_groupvelocity_picks(mft_analysis_all_states;
    correlation_threshold=0.85,
    pair_and_average=true,
    title="Group Velocity Picks $(selected_pair_name)"))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Peaks = "18e31ff7-3703-566c-8e60-38913d67486b"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
ColorSchemes = "~3.31.0"
Colors = "~0.13.1"
DSP = "~0.8.4"
FFTW = "~1.10.0"
JLD2 = "~0.6.4"
Peaks = "~0.6.2"
PlutoHooks = "~0.1.0"
PlutoLinks = "~0.1.8"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.80"
StatsBase = "~0.34.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "7a4e84f5cf26920ac69b79b4bda73dc7a6f7eecc"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "1a3ad7e16a321667698a19e77362b35a1e94c544"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.1"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "cee8104904c53d39eb94fd06cbe60cb5acde7177"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.0.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "REPL", "UUIDs"]
git-tree-sha1 = "cfb7a2e89e245a9d5016b70323db412b3a7438d5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "5989debfc3b38f736e69724818210c67ffee4352"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.4"

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

    [deps.DSP.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e86f4a2805f7f19bec5129bc9150c38208e5dc23"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.4"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6866aec60ef98e3164cd8d6855225684207e9dff"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.12+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6522cfb3b8fe97bec632252263057996cbd3de20"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.18.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.JLD2]]
deps = ["ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "941f87a0ae1b14d1ac2fa57245425b23a9d7a516"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.4"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "67c6f1f085cb2671c93fe34244c9cccde30f7a26"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.5.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "58927c485919bf17ea308d9d82156de1adf4b006"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.12"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "5d4278f755440f70648d80cc6225f51e78e94094"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.5.1"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Peaks]]
deps = ["SIMD"]
git-tree-sha1 = "a9b6680fb7fb097fb6eb1210c35549218d73da84"
uuid = "18e31ff7-3703-566c-8e60-38913d67486b"
version = "0.6.2"

    [deps.Peaks.extensions]
    MakieExt = "Makie"
    PlotsExt = "RecipesBase"

    [deps.Peaks.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "6256ab3ee24ef079b3afa310593817e069925eeb"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.23"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "844a829c8dc9fd0fe62eced22bc2d0dfd66a3f51"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.1.0"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "aea4eede5ab3ee188906d0cf3bbfa36eb543dccc"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.8"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "8acd04abc9a636ef57004f4c2e6f3f6ed4611099"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.5"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "fbc875044d82c113a9dee6fc14e16cf01fd48872"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.80"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "Setfield", "SparseArrays"]
git-tree-sha1 = "2d99b4c8a7845ab1342921733fa29366dae28b24"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.1"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"
    PolynomialsRecipesBaseExt = "RecipesBase"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "InteractiveUtils", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "5f4f629c085b87e71125eec6773f5f872c74a47a"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.14.2"

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

    [deps.Revise.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "ac4b837d89a58c848e85e698e2a2514e9d59d8f6"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2700b235561b0335d5bef7097a111dc513b8655e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.2"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "86f5831495301b2a1387476cb30f86af7ab99194"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.0"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╠═b90f919b-1e27-4a89-8a2c-c9dd990d5ec0
# ╠═f926472a-434d-4f57-b9ee-f3e8455db857
# ╠═1aee8298-a9f8-4ae8-a774-e07948c1419b
# ╠═9442424b-4d6d-435a-8d75-a80c359c6f6e
# ╠═c0936154-0a83-41a6-82f3-a0ec01295820
# ╠═133b0599-8022-4dc6-9713-95b6ab5eb0ba
# ╠═073120bf-7da5-4192-9440-f2a9cfa5e8a0
# ╠═6f130ae5-f43b-454a-9709-aca2d79cae40
# ╠═54157987-f4ee-4646-939c-92303f8a5322
# ╠═d4f82278-14b8-4d5d-ba0a-618ebdaa1910
# ╠═3b7fec31-93ce-45a8-a8c8-893090f49b41
# ╠═af11fe78-eeea-4126-a187-f191f1e50e8d
# ╠═ac12596d-984c-4ac2-8093-68c086009468
# ╠═edd2c342-ac56-4159-9d39-75271aa41656
# ╠═296053a9-8834-4ab4-84a3-951458e5fbe0
# ╠═2f06b9a3-cee3-45b6-bf23-b029199b4be5
# ╠═e23c2651-385e-4f6d-99af-183ec50d8ffa
# ╠═7fe32a39-7679-4f62-ba75-c883f3ae0f64
# ╠═68439964-d258-4efd-8d53-fc9771dca4cb
# ╠═fa401085-a4a3-4d66-9c2c-88a07a0f7895
# ╠═8d93dcb2-bafd-42e2-9880-7bc668e29462
# ╠═3f6f634d-edf2-433c-9c33-e9b6b02994d0
# ╠═2963c4ff-800a-43d5-aeae-7c8939806fc6
# ╠═8a482231-b1cb-4efc-8a90-4e5f640ee937
# ╠═b0ef58e8-a0b5-43dd-8122-07d6564c90e3
# ╠═f4fc8119-99f6-450b-b2d5-b715e256d6ab
# ╠═61a7b9b3-1896-4e03-b597-df16f9d00059
# ╠═e0888c71-0249-4626-ae74-0f602b81c00f
# ╠═c93cb971-370d-4daf-aa63-49d104b7e375
# ╠═04733c54-491a-4bc7-8a9f-ae8fa2b82147
# ╠═48135e48-036f-433a-8ec4-a541b5883ca1
# ╠═e09728b7-ac67-403c-a392-ed648437203d
# ╠═23086044-d104-49f7-8439-f541f869fcc8
# ╠═a39233a4-8cca-419d-831d-51905889da04
# ╠═0bc30aa0-069e-40ea-81fa-a3d60edac016
# ╠═2b2d7ee0-6cd3-4b33-8956-fee6916bb4dd
# ╠═d1597c31-bc78-47bf-ba2d-21adc4b60152
# ╠═532da58f-a1d4-4d61-b5c6-b97bf91b85ed
# ╠═dcc4775c-f909-4260-a1da-102ad1d4b3f5
# ╠═1713faaa-12ae-410c-b08f-1b40d713047c
# ╠═353ca4f7-976e-4581-a0e2-95f57d83fa90
# ╠═4af4c843-8a28-4d62-ba63-9dda2061465e
# ╠═980da5c6-a04c-483a-8c72-875ad73660fa
# ╠═cd25e99a-2a5c-484c-892c-02a6136a431c
# ╠═204ddd82-6536-42f1-a23c-8cff9e924af7
# ╠═65891fdf-1074-4450-a0f3-9e4b7743fc0c
# ╠═c8445edf-206a-4a64-98f6-1cf53221b994
# ╠═3dc0a19c-2670-4e97-bd4e-e93b2f53ad71
# ╠═51e6d80b-a400-48b6-badf-2a1456d7c823
# ╠═09931af2-91c2-4ac0-be11-183fc3bbbad5
# ╠═ebc826f7-df75-4d9c-87c4-ec7fab19e9f9
# ╠═25ad9c95-4def-4fca-b0b3-829f033cb011
# ╠═562c024b-1564-4a53-acf3-1c21df16b698
# ╠═a7604fea-de0a-4598-8e1c-777e709e5a5c
# ╠═0aca9aea-e2e0-4819-9bbe-3ba7bf0ae3bd
# ╠═2e47641b-af82-47c0-82dc-78da299517b3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
