### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

macro bind(def, element)
    return quote
        local iv = try Base.loaded_modules[
            Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")
        ].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

using JLD2,
    DSP,
    Statistics,
    LinearAlgebra,
    PlutoUI,
    PlutoLinks,
    PlutoHooks,
    PlutoPlotly,
    ColorSchemes,
    Colors

vqvae = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/VQVAE_architecture_v5.jl")
mft = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/MFT.jl")

TableOfContents(include_definitions=true)

md"""
# VQ-VAE v5 Analysis Notebook

Load a saved `analysis_cache.jld2` plus raw data and analyze one receiver pair at a time without retraining.
"""

md"## Paths"

dt = 1.0

begin
    period_min = 20
    period_max = 80
    responsetype = Bandpass(inv(period_max), inv(period_min))
    designmethod = Butterworth(2)
    digfilter = digitalfilter(responsetype, designmethod; fs=inv(dt))
end

function taper(x)
    w = cat(tukey(size(x, 1), 0.1), dims=ndims(x))
    return w .* x
end

function get_acausal_causal(pair::String, filepath::String)
    jldfile = load(filter(x -> occursin(pair, x), readdir(filepath, join=true))[1])
    correlations = jldfile["correlations"]
    headers = jldfile["headers"]
    distance = jldfile["dist"]
    return (; correlations, headers, distance)
end

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

function list_cache_runs(raw_data_dir::String)
    dirs = filter(path -> isdir(path) && startswith(basename(path), "vqvae_v5_run_"),
        readdir(raw_data_dir, join=true))
    sort!(dirs)
    return dirs
end

raw_data_dir = "/mnt/NAS2/Sanket_data/California_2013_BK_CI_20032026/"

cache_run_dirs = list_cache_runs(raw_data_dir)

md"Found **$(length(cache_run_dirs))** saved cache runs in $(raw_data_dir)"

begin
    isempty(cache_run_dirs) && error("No cache run directories found in $(raw_data_dir)")
    cache_options = basename.(cache_run_dirs)
    @bind selected_cache_run confirm(Select(cache_options, default=last(cache_options)))
end

selected_cache_dir = joinpath(raw_data_dir, selected_cache_run)

analysis_cache = load(joinpath(selected_cache_dir, "analysis_cache.jld2"))["analysis_cache"]
loss_history = analysis_cache.loss_history
pair_metadata = analysis_cache.pair_metadata
pair_names = [pm.pair_name for pm in pair_metadata]

md"## Pair Selection"

begin
    @bind selected_pair_name confirm(Select(pair_names, default=first(pair_names)))
end

selected_pair_id = findfirst(==(selected_pair_name), pair_names)
selected_pair_meta = pair_metadata[selected_pair_id]
selected_pair = Tuple(selected_pair_meta.pair)
encoded_cache = analysis_cache.all_pair_encoded_cache[selected_pair_id]

data_bundle = build_analysis_bundle(selected_pair; filepath=raw_data_dir)
data = (;
    D_ac_all=data_bundle.D1fac,
    D_c_all=data_bundle.D1fc,
)

nth = size(data.D_ac_all, 1)
t_neg = [-(nth - i + 1) * dt for i in 1:nth]
t_pos = [i * dt for i in 1:nth]
t_full = [t_neg; t_pos]

vqvae_para = analysis_cache.vqvae_para
combo_labels = vqvae.combination_labels(vqvae_para.Ksmall, vqvae_para.T)

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

cluster_avg_ac = cluster_averages_from_codes(data.D_ac_all, encoded_cache.ci_ac; K=vqvae_para.Ksmall)
cluster_avg_c = cluster_averages_from_codes(data.D_c_all, encoded_cache.ci_c; K=vqvae_para.Ksmall)

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

md"## Training Summary"

WideCell(vqvae.plot_training_dashboard(loss_history;
    title="VQ-VAE v5: $(selected_pair_name) loss history from saved cache"))

md"## Source State Analysis"

WideCell(vqvae.plot_state_usage(cross.pct_ac, cross.pct_c;
    labels=cross.labels,
    title="$(selected_pair_name) Source State Usage"))

WideCell(vqvae.plot_codebook_confusion(cross.confusion;
    title="$(selected_pair_name) Code Confusion",
    labels=cross.labels))

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

md"## MFT"

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

@bind ui_period Slider(mft_analysis_all_states.periods, show_value=true)

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

WideCell(mft.plot_branch_correlation(mft_analysis_all_states;
    title="$(selected_pair_name) Branch Correlation Across Source States"))

WideCell(mft.plot_all_highcorr_groupvelocity_picks(mft_analysis_all_states;
    correlation_threshold=0.9,
    title="Group Velocity Picks $(selected_pair_name)"))

WideCell(mft.plot_all_highcorr_groupvelocity_picks(mft_analysis_all_states;
    correlation_threshold=0.85,
    pair_and_average=true,
    title="Group Velocity Picks $(selected_pair_name)"))
