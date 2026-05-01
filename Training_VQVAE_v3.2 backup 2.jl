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

# ╔═╡ 02556dd0-4cb7-4251-a969-6bea09a41358
using Clustering

# ╔═╡ 9db29532-6d82-495c-bc3f-0daa882f5064
using ColorSchemes, Colors

# ╔═╡ 10000001-0000-0000-0000-000000000001
begin
    using CUDA, cuDNN,
        Flux,
        Distances,
        JLD2,
        Random,
        MLUtils,
        DSP,
        ProgressLogging,
        Statistics,
        LinearAlgebra,
        PlutoLinks,
        PlutoUI,
        PlutoHooks,
        PlutoPlotly,
        FFTW,
        StatsBase,
        Optimisers,
        ParameterSchedulers,
        Functors
    CUDA.device!(0)
end

# ╔═╡ 341dbed8-1d09-4b46-8434-eb332c332f75
using Peaks

# ╔═╡ c7f70869-8f84-4c33-a455-d79f78ac02ec
using Printf

# ╔═╡ da62431a-7cc6-4253-986d-5ba7d39e9f90
using Zygote

# ╔═╡ 53f17afb-91fb-4881-a9f4-9fa87a24fee6
using Enzyme

# ╔═╡ 418c15e5-8116-4d86-8c3e-aeac13cc3ef1
using BenchmarkTools

# ╔═╡ 10000002-0000-0000-0000-000000000001
TableOfContents(include_definitions=true)

# ╔═╡ 10000003-0000-0000-0000-000000000001
xpu = gpu

# ╔═╡ 10000004-0000-0000-0000-000000000001
md"""# VQ-VAE Training Notebook

Train a VQ-VAE model on a single station pair for clustering causal/acausal cross-correlation branches.
"""

# ╔═╡ 10000005-0000-0000-0000-000000000001
md"## Data Loading"

# ╔═╡ 10000007-0000-0000-0000-000000000001
dt = 1.0 # sanket

# ╔═╡ 10000006-0000-0000-0000-000000000001
begin
    period_min = 8
    period_max = 50
    responsetype = Bandpass(inv(period_max), inv(period_min))
    designmethod = Butterworth(2)
    digfilter = digitalfilter(responsetype, designmethod; fs=inv(dt))
end

# ╔═╡ 10000008-0000-0000-0000-000000000001
function taper(x)
    w = cat(tukey(size(x, 1), 0.1), dims=ndims(x))
    return w .* x
end

# ╔═╡ 10000009-0000-0000-0000-000000000001
# function get_acausal_causal(pair::String, filepath::String)
# 	jldfile = load(filter(x -> occursin(pair, x), readdir(filepath, join=true))[1])
# 		correlations = DSP.resample(jldfile["D"][1], 0.25, dims=1)
# 	headers = jldfile["headers"][1]
# 	distance = jldfile["Distances"][1]
# 	return (; correlations, headers, distance)
# end
function get_acausal_causal(pair::String, filepath::String)
    jldfile = load(filter(x -> occursin(pair, x), readdir(filepath, join=true))[1])
    correlations = jldfile["correlations"]
	correlations = randn(size(correlations)...)
    headers = jldfile["headers"]
    distance = jldfile["dist"] # sanket
    return (; correlations, headers, distance)
end

# ╔═╡ 1000000a-0000-0000-0000-000000000001
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

# ╔═╡ 1000000c-0000-0000-0000-000000000001
function build_training_bundle(pair;
    filepath="/mnt/NAS/Sanket_DRDO/station_pairs_12112025_30mins/")
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
    return (pair=pair, D1=Float32.(D1), D1fac=D1fac, D1fc=D1fc,
        distance=data_pair_local.distance)
end

# ╔═╡ 1000000d-0000-0000-0000-000000000001
md"### Select Station Pair"

# ╔═╡ 1000000e-0000-0000-0000-000000000001
# data_filepath = "/mnt/NAS2/Sanket_data/California_2013_BK_CI_20032026/"
data_filepath = "/mnt/NAS2/Sanket_data/California_TO_with_latlong/"
# data_filepath = "/mnt/NAS2/Sanket_data/California_XJ_13032026/"

# ╔═╡ 10000010-0000-0000-0000-000000000001
md"### Train/Test Split (Pooled)"

# ╔═╡ 10000011-0000-0000-0000-000000000001
function make_pooled_split(D1fac, D1fc; at=0.9, shuffle=true)
    # Pool causal and acausal into one matrix
    D_all = hcat(D1fac, D1fc)
    nw = size(D_all, 2)
    idx = collect(1:nw)
    shuffle && Random.shuffle!(idx)
    ntrain = round(Int, at * nw)
    train_idx = idx[1:ntrain]
    test_idx = idx[ntrain+1:end]
    return (
        D_train=xpu(D_all[:, train_idx]),
        D_test=xpu(D_all[:, test_idx]),
        D_all=xpu(D_all),
        # Keep separate branch references for post-hoc analysis
        D_ac_all=xpu(D1fac),
        D_c_all=xpu(D1fc),
    )
end

# ╔═╡ 10000015-0000-0000-0000-000000000001
md"""
## Data Visualization
"""

# ╔═╡ 10000017-0000-0000-0000-000000000001
md"## Load VQ-VAE Architecture"

# ╔═╡ 10000018-0000-0000-0000-000000000001
vqvae = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/VQVAE_architecture_v3.1.jl")

# ╔═╡ 6d7cebf7-fb3e-4134-a428-91dea9f272b4
mft = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/MFT.jl")

# ╔═╡ 10000019-0000-0000-0000-000000000001
md"## Model Setup"

# ╔═╡ 1000001a-0000-0000-0000-000000000001
reload_network_button = @bind reload_network CounterButton("Reload Network")

# ╔═╡ 1000001e-0000-0000-0000-000000000001
md"""
## Training
"""

# ╔═╡ 10000020-0000-0000-0000-000000000001
CUDA.pool_status()

# ╔═╡ 10000022-0000-0000-0000-000000000001
md"## Loss Curves"

# ╔═╡ 10000024-0000-0000-0000-000000000001
md"## Source State Analysis"

# ╔═╡ 10000028-0000-0000-0000-000000000001
md"### Confusion Matrix"

# ╔═╡ 1000002a-0000-0000-0000-000000000001
md"The diagonal shows window-pairs where causal and acausal branches share the same code. Off-diagonal entries reveal branch-specific clustering."

# ╔═╡ 1000002b-0000-0000-0000-000000000001
md"### Source State Averages"

# ╔═╡ 1000002f-0000-0000-0000-000000000001
md"""## Gather Plots

Select a source state to view all waveforms assigned to it.
"""

# ╔═╡ 10000032-0000-0000-0000-000000000001
md"""## Reconstruction Quality

Visualize a few reconstructions vs originals.
"""

# ╔═╡ 3da3c466-6d7f-4d8b-8ba2-1f7a66ff9a46
md"### Filtered reconstruction for selected state"

# ╔═╡ 10000034-0000-0000-0000-000000000001
md"## Saving"

# ╔═╡ db4ddb38-2938-11f1-b8e3-e5227df9322c
md"## Appendix"

# ╔═╡ e928b8ab-e159-427a-b525-c6d60e6d6015
resample_button = @bind resample CounterButton("Resample Waveforms")

# ╔═╡ 3bf7a033-db24-4e6c-93a9-8706d2ea57c3
resample_button

# ╔═╡ 84c4a49c-c6fb-45b4-ac33-a5510cd6618e
resample_button

# ╔═╡ 6a3e970f-a7e7-4c67-a9e6-dc79be22c547
resample_button

# ╔═╡ b1f70fd0-1a7a-4f8f-9e7c-8e3e8cd3f5e1
function list_station_pairs(filepath::String)
    files = readdir(filepath)
    pairs = Set{Tuple{String,String}}()
    for f in files
        m = match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)", basename(f))
        m === nothing && continue
        push!(pairs, (m.captures[1], m.captures[2]))
    end
    return sort!(collect(pairs), by=x -> (x[1], x[2]))
end

# ╔═╡ f3b8c867-f8f9-45e0-a41f-4b7d7a1b17f0
available_pairs = list_station_pairs(data_filepath)

# ╔═╡ 0f84f2f6-6403-4a2e-9c42-6f8a84a2bc3f
md"Found **$(length(available_pairs))** station pairs in $(data_filepath)"

# ╔═╡ 2a8a4d12-c96b-4d64-8f58-98f54f81a77b
available_pairs

# ╔═╡ eaa7d770-d2d0-42f0-a65b-99447f11649a
begin
    if isempty(available_pairs)
        error("No station pairs found in $(data_filepath)")
    end
    pair_options = ["$(p[1])-$(p[2])" for p in available_pairs]
    @bind selected_pair_name confirm(Select(pair_options, default=first(pair_options)))
end

# ╔═╡ 74d0f419-79d0-4a6a-ae65-18f6be9d64c8
selected_pair = let
    parts = split(selected_pair_name, "-", limit=2)
    length(parts) == 2 || error("Invalid selected pair format: $(selected_pair_name)")
    (parts[1], parts[2])
end

# ╔═╡ f86fec7f-f467-4411-80aa-c1621e3de063
data_bundle_pushkar = try
    build_training_bundle(selected_pair; filepath="/mnt/NAS2/Pushkar_Data/uttaranchal_data/jldfiles/30mins_dt_0p25_band_0p01_2p00_500maxlag/Z/")
catch
    nothing
end

# ╔═╡ 1000000f-0000-0000-0000-000000000001
data_bundle_cc = try
    build_training_bundle(selected_pair; filepath="/mnt/NAS2/Sanket_data/California_TO_with_latlong/")
catch
    nothing
end

# ╔═╡ e8ae6df6-fa04-42b7-a6e5-b2ade4322995
selected_pair |> typeof

# ╔═╡ c669f1d3-26c0-4028-b3d4-5a87bd696924
data_bundle = build_training_bundle(selected_pair; filepath=data_filepath)

# ╔═╡ 10000012-0000-0000-0000-000000000001
data = make_pooled_split(data_bundle.D1fac, data_bundle.D1fc)

# ╔═╡ 10000013-0000-0000-0000-000000000001
nth = size(data.D_train, 1)

# ╔═╡ 10000014-0000-0000-0000-000000000001
tgrid = collect(-nth:nth) .* dt

# ╔═╡ 83946706-1d00-4794-af60-8c65979236f8
data_bundle.distance / 1.2

# ╔═╡ 1000001b-0000-0000-0000-000000000001
vqvae_parameters = (;
    nt=nth,
    d=20,            # codebook embedding dimension (matches SymAE p)
    K=5,             # codebook size (matches SymAE k)
    T=1,             # quantized vectors per waveform
    beta_commit=0.25f0,    # ↓ from 0.25: less commitment pressure → encoder can spread out
    ema_decay=0.999f0,
    enc_kernels=[64, 32, 16, 8],
    enc_strides=[1, 2, 1, 2],
    dead_threshold=50,     # reset dead entries after this many batches unused
    entropy_weight=0.1f0,  # stronger anti-collapse regularization
    interstation_distance=Float64(data_bundle.distance),  # km; set to nothing for dense path
    dt=dt,                 # sampling interval (s)
    velocity_range=(1.5, 4),  # km/s; (vmin, vmax) group velocity range for window
    # epsilon=1e-1,
)

# ╔═╡ 1000001c-0000-0000-0000-000000000001
vqvae_para = vqvae.VQVAE_Para(; vqvae_parameters...)

# ╔═╡ f2369548-2d88-11f1-a737-85bad04c89cb
model, loss_history = @use_memo([reload_network, vqvae_parameters]) do
    reload_network
    model, loss_history = vqvae.get_vqvae(vqvae_para)
    model, loss_history
end

# ╔═╡ 1a623368-a2fc-4b7f-8d01-68e36d04a891
model.pre_vq

# ╔═╡ ed09ec19-f1b0-4819-9c9f-7ee796bd8f09
@bind islot Slider(1:model.T, show_value=true)

# ╔═╡ 10000021-0000-0000-0000-000000000001
trained = @use_memo([]) do
    training_para = vqvae.VQVAE_Training_Para(
        batchsize=256,
        nepoch=50,
        initial_learning_rate=0.001,  # ↑ from 0.001,
        # stop_on_recon_loss = 0.98,
        lr_decay=0.99,
    )

    vqvae.update(model, loss_history,
        data.D_train[:, :], data.D_test,
        vqvae_para, training_para)

	   # vqvae.update(model, loss_history,
    #     data.D_train[:, :], data.D_test,
    #     vqvae_para, training_para)

    randn(), loss_history
end

# ╔═╡ 10000025-0000-0000-0000-000000000001
begin
    trained
    cross = vqvae.codebook_cross_analysis(model, data.D_ac_all, data.D_c_all)
end

# ╔═╡ 01c3acc7-efc4-471a-8f4e-515c59ae3cf8
cross

# ╔═╡ 10000026-0000-0000-0000-000000000001
md"""
**Codebook Cross-Analysis:**
- **Agreement rate**: $(round(cross.agreement * 100; digits=1))% of windows map to same code
- **Shared codes** (both >5%): $(cross.shared_codes)
- **Acausal‐dominant codes**: $(cross.ac_only_codes)
- **Causal‐dominant codes**: $(cross.c_only_codes)
"""

# ╔═╡ 1000002c-0000-0000-0000-000000000001
begin
    trained
    cluster_avg_ac = vqvae.get_cluster_averages(model, data.D_ac_all)
    cluster_avg_c = vqvae.get_cluster_averages(model, data.D_c_all)
end;

# ╔═╡ 1000001d-0000-0000-0000-000000000001
begin
    trained
    combo_labels = vqvae.combination_labels(vqvae_para.K, vqvae_para.T)
    combo_count = length(combo_labels)
end;

# ╔═╡ 80f07cb0-0a48-49c8-8f35-476024d3b826
select_combo_button = @bind selected_combo Select(combo_labels)

# ╔═╡ 77766676-bc51-4554-80e3-65112f725fd2
select_combo_button

# ╔═╡ 474a66d1-8851-4146-bc65-2026196981ee
select_combo_button

# ╔═╡ 1cac8d93-75ed-4dfb-b330-ae26e5750ff2
select_combo_button

# ╔═╡ 6ad210ef-a2dc-47d8-bd7e-a06ab7ac1a32
mft_analysis_all_states = let
    nstates = size(cluster_avg_ac, 2)

    ac_traces = [
        mft.SeismicTrace(
            data=vec(cluster_avg_ac[:, i]),
            dt=dt,
            distance=data_bundle.distance
        )
        for i in 1:nstates
    ]

    c_traces = [
        mft.SeismicTrace(
            data=vec(cluster_avg_c[:, i]),
            dt=dt,
            distance=data_bundle.distance
        )
        for i in 1:nstates
    ]

    labels = string.(combo_labels)

    mft.analyze_causal_acausal_branches(
        ac_traces,
        c_traces,
        state_labels=labels,
		period_max=80.0,
		velocity_range=(1.0, 8.0),
        bandwidth_factor=0.15,
	 zero_pad_factor=4,
    )
end

# ╔═╡ b81bef7c-b237-4589-a10d-dd92f95232f3
@bind ui_period Slider(mft_analysis_all_states.periods, show_value=true)

# ╔═╡ b03589cd-99ba-44e6-88c8-27b733ae0e96
mft_analysis = let
	 ac_trace = mft.SeismicTrace(
		            data=vec(cluster_avg_ac[:, parse(Int, selected_combo)]),
		            dt=dt,
		            distance=data_bundle.distance
		        )
	 c_trace = mft.SeismicTrace(
		            data=vec(cluster_avg_c[:, parse(Int, selected_combo)]),
		            dt=dt,
		            distance=data_bundle.distance
		        )
	mft.analyze_causal_acausal_branches(ac_trace, c_trace, bandwidth_factor=0.15, velocity_range=(1.0, 8.0))
end

# ╔═╡ 993f8556-a7fb-45d4-b097-3e4420776e14
WideCell(mft.plot_all_highcorr_groupvelocity_picks(mft_analysis; correlation_threshold=0.85))

# ╔═╡ 40aec027-40cc-4901-9232-2091b69091de
WideCell(mft.plot_branch_comparison(mft_analysis))

# ╔═╡ aea9e546-cab6-4cae-a0d4-71990763a1ec
mft.plot_dispersion_curve([mft_analysis.causal_result, mft_analysis.acausal_result])

# ╔═╡ ae8aefcf-9397-451b-b5db-1afb922b9347
mft.plot_dispersion_curve(mft_analysis.acausal_result)

# ╔═╡ 60630929-d25c-46b4-adf6-08b5927fe3aa
inv(3.0/(data_bundle.distance/2))

# ╔═╡ 481e1558-d494-40cb-990e-45b7ad76a670
# After training (e.g. after `trained` is computed), add this cell:
let
    trained
    # Get codebook matrix
    E = cpu(model.quantizer.embedding)  # (d, K) or (K,d) depending implementation; adjust below if needed

    E = Array(E)  # ensure CPU Array

    K = size(E, 2)
    kmax = min(K, 20)
    Esel = E[:, 1:kmax]

    # axis labels
    xlabels = string.(1:kmax)
    ylabels = string.(1:size(Esel, 1))

    # Choose colormap; `Viridis` is readable
    cm = ColorSchemes.viridis

    # Build heatmap
    trace = PlutoPlotly.heatmap(
        z=Esel,
        x=xlabels,
        y=ylabels,
        # colorscale = [ [i, colorant"$(RGB(cm[i]))"] for i in range(0, stop=1, length=256) ],
        colorbar=attr(title="Embedding\nvalue", titleside="right"),
        zmid=0
    )

    layout = Layout(
        title=attr(text="Codebook Embedding Heatmap (first $kmax codes) — $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s", font=attr(size=16)),
        xaxis=attr(title="Code index"),
        yaxis=attr(title="Embedding dimension"),
        width=900,
        height=600,
        margin=attr(t=70, b=60, l=80, r=140)
    )

    WideCell(PlutoPlotly.plot([trace], layout))
end

# ╔═╡ 10000023-0000-0000-0000-000000000001
begin
    trained
    WideCell(vqvae.plot_training_dashboard(loss_history;
        title="VQ-VAE: $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s"))
end

# ╔═╡ 10000027-0000-0000-0000-000000000001
begin
    trained
    WideCell(vqvae.plot_cluster_histogram(cross.pct_ac, cross.pct_c;
        title="$(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s Source State Usage",
        labels=cross.labels))
end

# ╔═╡ 10000029-0000-0000-0000-000000000001
begin
    trained
    WideCell(vqvae.plot_codebook_confusion(cross.confusion;
        title="$(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s Code Confusion",
        labels=cross.labels))
end

# ╔═╡ a0000201-0000-0000-0000-000000000001
let
    trained
    labels = string.(combo_labels)
    n = length(labels)

    function norm_corr_matrix(A)
        # A: (nt, n); returns (n, n) normalized correlation matrix
        C = Matrix{Float32}(undef, n, n)
        cols = [begin v = vec(A[:, i]); v .- mean(v) end for i in 1:n]
        norms = [norm(c) + 1f-8 for c in cols]
        for i in 1:n, j in 1:n
            C[i, j] = dot(cols[i], cols[j]) / (norms[i] * norms[j])
        end
        C
    end

    C_ac = norm_corr_matrix(cluster_avg_ac)
    C_c  = norm_corr_matrix(cluster_avg_c)

    trace_ac = PlutoPlotly.heatmap(
        z=C_ac, x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=attr(title="Corr", len=0.9, x=0.46),
        xaxis="x1", yaxis="y1",
    )
    trace_c = PlutoPlotly.heatmap(
        z=C_c, x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=attr(title="Corr", len=0.9, x=1.01),
        xaxis="x2", yaxis="y2",
    )

    sz = max(350, n * 40)
    layout = Layout(
        title=attr(text="State–State Normalised Correlation — $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s",
            font=attr(size=16)),
        grid=attr(rows=1, columns=2, pattern="independent"),
        annotations=[
            attr(text="Acausal", x=0.22, xref="paper", y=1.05, yref="paper",
                 showarrow=false, font=attr(size=14)),
            attr(text="Causal",  x=0.78, xref="paper", y=1.05, yref="paper",
                 showarrow=false, font=attr(size=14)),
        ],
        xaxis=attr(title="State", tickangle=-45),
        yaxis=attr(title="State"),
        xaxis2=attr(title="State", tickangle=-45),
        yaxis2=attr(title="State"),
        width=900, height=sz + 80,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=attr(t=80, b=80, l=80, r=80),
    )
    WideCell(PlutoPlotly.plot([trace_ac, trace_c], layout))
end

# ╔═╡ 6050d016-1589-47bc-90a3-87c441a59acf
WideCell(
    mft.plot_filtered_traces_by_period(
        mft_analysis_all_states;
        period=ui_period,                   # or period_index=...
        correlation_threshold=nothing,      # e.g. 0.9 to keep only high-symmetry states
        normalize_each=true,
        scale=0.7,
        spacing=2.2,
        title="$(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s Filtered Traces by Source State"
    )
)

# ╔═╡ cec27a5a-f20c-459b-a4ad-8a8b0e792d13
WideCell(mft.plot_branch_correlation(mft_analysis_all_states; title="$(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s Branch Correlation Across Source States"))

# ╔═╡ 785a4efa-d336-458a-8c8b-aca5f24e27d3
WideCell(mft.plot_all_highcorr_groupvelocity_picks(mft_analysis_all_states; correlation_threshold=0.9, title="Group Velocity Picks $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s"))

# ╔═╡ 0825f860-0d47-4b0b-ab97-2436a0f97136
WideCell(mft.plot_all_highcorr_groupvelocity_picks(mft_analysis_all_states; correlation_threshold=0.85, pair_and_average=true, title="Group Velocity Picks $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s"))

# ╔═╡ 10000035-0000-0000-0000-000000000001
# ╠═╡ disabled = true
#=╠═╡
begin
	using Dates
	timestamp = now()
	pair_str = join(selected_pair, "_")
	jldsave("SavedModels/vqvae_model-$(pair_str)-$(timestamp).jld2",
		model_state = Flux.state(cpu(model)))
	jldsave("SavedModels/vqvae_para-$(pair_str)-$(timestamp).jld2";
		vqvae_parameters)
	jldsave("SavedModels/vqvae_loss-$(pair_str)-$(timestamp).jld2";
		loss_history)
end
  ╠═╡ =#

# ╔═╡ a0000101-0000-0000-0000-000000000001
# Velocity-range travel times for vertical lines on all lag-time plots
begin
    _vmin, _vmax = vqvae_para.velocity_range
    _dist = data_bundle.distance
    t_vmin = _dist / _vmin   # slowest velocity → latest arrival (s)
    t_vmax = _dist / _vmax   # fastest velocity → earliest arrival (s)

    """Return PlutoPlotly shape attrs for vertical lines at vmin/vmax arrivals.
    Pass `symmetric=true` for joined acausal+causal plots (adds negative-lag lines)."""
    function velocity_vlines(t_vmin, t_vmax; symmetric=false)
        lines = [
            attr(type="line", x0=t_vmax, x1=t_vmax, y0=0, y1=1, yref="paper",
                 line=attr(color="steelblue", dash="dash", width=1.5)),
            attr(type="line", x0=t_vmin, x1=t_vmin, y0=0, y1=1, yref="paper",
                 line=attr(color="tomato", dash="dash", width=1.5)),
        ]
        if symmetric
            append!(lines, [
                attr(type="line", x0=-t_vmax, x1=-t_vmax, y0=0, y1=1, yref="paper",
                     line=attr(color="steelblue", dash="dash", width=1.5)),
                attr(type="line", x0=-t_vmin, x1=-t_vmin, y0=0, y1=1, yref="paper",
                     line=attr(color="tomato", dash="dash", width=1.5)),
            ])
        end
        lines
    end
end

# ╔═╡ 1000002e-0000-0000-0000-000000000001
let
    K = vqvae_para.K
    # Time axes: acausal → negative lags (reversed), causal → positive lags
    t_neg = [-(nth - i + 1) * dt for i in 1:nth]  # -nth*dt … -dt
    t_pos = [i * dt for i in 1:nth]                # dt … nth*dt
    t_full = [t_neg; t_pos]

    # Global averages across all waveforms
    global_avg_ac = vec(mean(cpu(data.D_ac_all); dims=2))
    global_avg_c = vec(mean(cpu(data.D_c_all); dims=2))
    global_full = [reverse(global_avg_ac); global_avg_c]

    combo_labels_local = cross.labels
    ncomb = length(combo_labels_local)
    traces = AbstractTrace[]
    begin
        nc = max(ncomb, 1)
        cs = ColorSchemes.rainbow
        colors = [Colors.hex(get(cs, (i - 1) / max(1, nc - 1))) for i in 1:nc]
    end

    # Per-cluster joined CCF: acausal (negative lags) + causal (positive lags)
    total_ac = size(data.D_ac_all, 2)
    total_c = size(data.D_c_all, 2)
    # compute vertical spacing from typical amplitude
    mean_ac = vec(mean(cluster_avg_ac; dims=1))
    mean_c = vec(mean(cluster_avg_c; dims=1))
    amp_peak = maximum(abs.(vcat(mean_ac, mean_c)))
    vertical_spacing = amp_peak * 2.5 + 1e-3

    for combo_idx in 1:ncomb
        c = colors[mod1(combo_idx, length(colors))]
        # Build per-state joined CCF (acausal reversed to align with causal)
        a = cluster_avg_ac[:, combo_idx]
        a_rev = reverse(cluster_avg_ac[:, combo_idx])
        b = cluster_avg_c[:, combo_idx]
        full_k = [a_rev; b]
        # normalized cross-correlation (zero-mean cosine-like similarity)
        a0 = a .- mean(a)
        b0 = b .- mean(b)
        ncc = dot(a0, b0) / ((norm(a0) * norm(b0)) + 1e-8)
        # Get percentage of windows used for averaging in each cluster
        ks_tuple = Tuple(vqvae.combination_digits(combo_idx, K, vqvae_para.T))
        _, sel_ac = vqvae.filter_cluster(model, data.D_ac_all, ks_tuple)
        _, sel_c = vqvae.filter_cluster(model, data.D_c_all, ks_tuple)
        pct_ac = 100 * length(sel_ac) / max(total_ac, 1)
        pct_c = 100 * length(sel_c) / max(total_c, 1)
        legend_label = "State $(combo_labels_local[combo_idx]) (ac: $(round(pct_ac; digits=1))%, c: $(round(pct_c; digits=1))%, corr=$(round(ncc; digits=3)))"
        offset = (combo_idx - 1) * vertical_spacing
        push!(traces, PlutoPlotly.scatter(x=t_full, y=full_k .+ offset, mode="lines",
            name=legend_label,
            line=attr(color=c, width=2)))
    end

    # Global mean overlay
    push!(traces, PlutoPlotly.scatter(x=t_full, y=global_full, mode="lines",
        name="Global mean",
        line=attr(color="black", width=2, dash="dot")))

    layout = Layout(
        title=attr(text="Source State Average Waveforms ($(selected_pair[1])-$(selected_pair[2])) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s",
            font=attr(size=18, family="Computer Modern, serif")),
        height=500, width=900,
        xaxis=attr(title="Lag (s)", zeroline=true, zerolinecolor="rgba(0,0,0,0.3)"),
        yaxis=attr(title="Amplitude"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=attr(x=0.5, xanchor="center", y=-0.2, orientation="h",
            font=attr(size=12, family="Computer Modern, serif")),
        shapes=velocity_vlines(t_vmin, t_vmax; symmetric=true),
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ 10000031-0000-0000-0000-000000000001
let
    trained
    t_neg = [-(nth - i + 1) * dt for i in 1:nth]
    t_pos = [i * dt for i in 1:nth]
    t = [t_neg; t_pos]


    averaged_combinations = map(combo_labels) do selected_combo

        ks_tuple = Tuple(vqvae.combination_digits(findall(x -> x == selected_combo, combo_labels)[1], vqvae_para.K, vqvae_para.T))

        # Get indices for selected cluster (acausal)
        _, selected_ac = vqvae.filter_cluster(model, data.D_ac_all, ks_tuple)
        # Get indices for selected cluster (causal)
        _, selected_c = vqvae.filter_cluster(model, data.D_c_all, ks_tuple)

        if (selected_ac == [] || selected_c == [])
            @info "No waveforms assigned to state $(selected_combo)"
            return (; raw=nothing, recon=nothing)
        else
            # Get the actual waveforms for these indices
            x_ac_sel = xpu(data.D_ac_all[:, selected_ac])
            x_c_sel = xpu(data.D_c_all[:, selected_c])

            # Reconstruct all selected waveforms and get per-component contributions
            r_ac = model(x_ac_sel; training=false)
            r_c = model(x_c_sel; training=false)

            # x_ac_recon = cpu(r_ac.xhat_per_slot[:, 2, :])
            # x_c_recon = cpu(r_c.xhat_per_slot[:, 2, :])

            r_ac = normalise(cpu(vec(mean(r_ac.xhat, dims=2))), dims=1)
            r_c = normalise(cpu(vec(mean(r_c.xhat, dims=2))), dims=1)
			
            ac = normalise(cpu(vec(mean(x_ac_sel; dims=2))), dims=1)
            c = normalise(cpu(vec(mean(x_c_sel; dims=2))), dims=1)
            return (; raw=[reverse(ac); c], recon=[reverse(r_ac); r_c])
        end


    end


    traces = AbstractTrace[]

    cs = ColorSchemes.rainbow
    nc = length(combo_labels)
    colors = [Colors.hex(get(cs, (i - 1) / max(1, nc - 1))) for i in 1:nc]

    j = 1
    for i in 1:length(combo_labels)
        if(isnothing(averaged_combinations[i].raw))
             @info "Skipping state $(combo_labels[i]) due to no assigned waveforms."
            continue
        end
        j += 1
        offset = (j - 1) * 3.0
        c = colors[i]
        push!(traces, PlutoPlotly.scatter(x=t, y=averaged_combinations[i].raw .* 0.25 .+ offset, mode="lines", line=attr(color="black", width=2), opacity=0.5,
            showlegend=(i == 1), name="Raw $(combo_labels[i])"))

        push!(traces, PlutoPlotly.scatter(x=t, y=averaged_combinations[i].recon .* 0.25 .+ offset,
            mode="lines", line=attr(color=c, width=1),
            showlegend=(i == 1), name="Recon. $(combo_labels[i])"))
    end


    layout = Layout(
        title=attr(text="Source States ($(selected_pair[1])-$(selected_pair[2])) $(round(Int, data_bundle.distance))km $(period_min)-$(period_max)s",
            font=attr(size=18, family="Computer Modern, serif")),
        height=900, width=900,
        xaxis=attr(title="Lag Time (s)"),
        yaxis=attr(title="Source State"),
        legend=attr(orientation="h", x=0.5, xanchor="center", y=-0.15, font=attr(size=10)),
        plot_bgcolor="white", paper_bgcolor="white",
        shapes=velocity_vlines(t_vmin, t_vmax; symmetric=true),
    )
    WideCell(PlutoPlotly.plot(traces, layout))

end

# ╔═╡ 1ce2fb4a-ff49-4873-9f93-6decbf9b8e80
let
    trained
    resample
    nshow = 10
    combo_idx = selected_combo isa Integer ? selected_combo : findfirst(x -> x == selected_combo, combo_labels)
    if combo_idx === nothing
        md"### Could not resolve the selected source state."
    else
        ks_tuple = Tuple(vqvae.combination_digits(combo_idx, vqvae_para.K, vqvae_para.T))
        _, selected_ac = vqvae.filter_cluster(model, data.D_ac_all, ks_tuple)
        _, selected_c = vqvae.filter_cluster(model, data.D_c_all, ks_tuple)
        ac_ids = collect(selected_ac)
        c_ids = collect(selected_c)
        ac_count = min(nshow, length(ac_ids))
        c_count = min(nshow, length(c_ids))
        if ac_count == 0 || c_count == 0
            missing = String[]
            if ac_count == 0
                push!(missing, "acausal")
            end
            if c_count == 0
                push!(missing, "causal")
            end
            md"""### Selected state $(combo_labels[combo_idx]) has no $(join(missing, " and ")) windows to display."""
        else
            function random_sample(ids, count)
                if count >= length(ids)
                    collect(ids)
                else
                    perm = randperm(length(ids))
                    ids[perm[1:count]]
                end
            end
            ac_indices = random_sample(ac_ids, ac_count)
            c_indices = random_sample(c_ids, c_count)
            plot_n = min(length(ac_indices), length(c_indices))
            ac_indices = ac_indices[1:plot_n]
            c_indices = c_indices[1:plot_n]
            if plot_n == 0
                md"### Not enough windows to pair causal and acausal samples for plotting."
            else
                x_ac_input = xpu(data.D_ac_all[:, ac_indices])
                x_c_input = xpu(data.D_c_all[:, c_indices])
                r_ac = model(x_ac_input; training=false)
                r_c = model(x_c_input; training=false)
                x_ac_recon = cpu(r_ac.xhat)
                x_c_recon = cpu(r_c.xhat)
                x_ac_raw = cpu(data.D_ac_all[:, ac_indices])
                x_c_raw = cpu(data.D_c_all[:, c_indices])
                t_neg = [-(nth - i + 1) * dt for i in 1:nth]
                t_pos = [i * dt for i in 1:nth]
                t_full = [t_neg; t_pos]
                amplitude_pool = vcat(vec(mean(x_ac_raw; dims=2)), vec(mean(x_c_raw; dims=2)),
                    vec(mean(x_ac_recon[:, :, 1]; dims=2)), vec(mean(x_c_recon[:, :, 1]; dims=2)))
                vertical_spacing = maximum(abs.(amplitude_pool)) * 2.5 + 1e-3
                cs = ColorSchemes.rainbow
                colors = [Colors.hex(get(cs, (i - 1) / max(1, plot_n - 1))) for i in 1:plot_n]
                traces = AbstractTrace[]
                for i in 1:plot_n
                    raw_ac = x_ac_raw[:, i]
                    raw_c = x_c_raw[:, i]
                    recon_ac = x_ac_recon[:, i, 1]
                    recon_c = x_c_recon[:, i, 1]
                    raw_combo = [reverse(raw_ac); raw_c]
                    recon_combo = [reverse(recon_ac); recon_c]
                    offset = (i - 1) * vertical_spacing
                    push!(traces, PlutoPlotly.scatter(x=t_full, y=raw_combo .+ offset,
                        mode="lines", opacity=0.5, line=attr(color="black", width=1.),
                        name="Raw", showlegend=i == 1))
                    push!(traces, PlutoPlotly.scatter(x=t_full, y=recon_combo * 2 .+ offset,
                        mode="lines", line=attr(color="red", width=2,),
                        name="Recon", showlegend=i == 1))
                end
                layout = Layout(
                    title=attr(text="($(period_min)-$(period_max)s) Filtered $(combo_labels[combo_idx]): joined acausal+causal reconstructions $(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km",
                        font=attr(size=18, family="Computer Modern, serif")),
                    height=max(750, plot_n * 40), width=950,
                    xaxis=attr(title="Time Lag (s)"),
                    yaxis=attr(title="Trace + offset"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=attr(orientation="h", x=0.5, xanchor="center", y=-0.2, font=attr(size=10)),
                    shapes=velocity_vlines(t_vmin, t_vmax; symmetric=true))
                WideCell(PlutoPlotly.plot(traces, layout))
            end
        end
    end
end

# ╔═╡ 10000033-0000-0000-0000-000000000001
let
    trained
    resample
    nshow = 10
    x_sample = randobs(data.D_ac_all, nshow)
    result = model(x_sample; training=false)
    x_orig = cpu(x_sample)

    x_recon = cpu(result.xhat) * 2.5

    t = collect(1:nth) .* dt

    traces = AbstractTrace[]
    for i in 1:nshow
        offset = (i - 1) * 4
        push!(traces, PlutoPlotly.scatter(x=t, y=x_orig[:, i] .+ offset,
            mode="lines", line=attr(color="black", width=1), opacity=0.5,
            showlegend=i == 1, name="Original"))
        push!(traces, PlutoPlotly.scatter(x=t, y=x_recon[:, i, 1] .+ offset,
            mode="lines", line=attr(color="red", width=2),
            showlegend=i == 1, name="Reconstructed"))
    end

    layout = Layout(
        title=attr(text="$(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km Reconstruction Examples (Acausal) $(period_min)-$(period_max)s",
            font=attr(size=18, family="Computer Modern, serif")),
        height=750, width=900,
        xaxis=attr(title="Time Lag (s)"),
        legend=attr(orientation="h", x=0.5, xanchor="center", y=-0.15, font=attr(size=10)),
        plot_bgcolor="white", paper_bgcolor="white",
        shapes=velocity_vlines(t_vmin, t_vmax),
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ 41391781-fa8e-4479-a1a8-7132053026bf
let
    trained
    resample
    nshow = 10
    x_sample = randobs(data.D_c_all, nshow)
    result = model(x_sample; training=false)
    x_orig = cpu(x_sample)

    x_recon = cpu(result.xhat) * 2.5

    t = collect(1:nth) .* dt

    traces = AbstractTrace[]
    for i in 1:nshow
        offset = (i - 1) * 4
        push!(traces, PlutoPlotly.scatter(x=t, y=x_orig[:, i] .+ offset,
            mode="lines", line=attr(color="black", width=1), opacity=0.5,
            showlegend=i == 1, name="Original"))
        push!(traces, PlutoPlotly.scatter(x=t, y=x_recon[:, i, 1] .+ offset,
            mode="lines", line=attr(color="red", width=2),
            showlegend=i == 1, name="Reconstructed"))
    end

    layout = Layout(
        title=attr(text="$(selected_pair[1])-$(selected_pair[2]) $(round(Int, data_bundle.distance))km Reconstruction Examples (Causal) $(period_min)-$(period_max)s",
            font=attr(size=18, family="Computer Modern, serif")),
        height=750, width=900,
        xaxis=attr(title="Time Lag (s)"),
        legend=attr(orientation="h", x=0.5, xanchor="center", y=-0.15, font=attr(size=10)),
        plot_bgcolor="white", paper_bgcolor="white",
        shapes=velocity_vlines(t_vmin, t_vmax),
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ Cell order:
# ╠═02556dd0-4cb7-4251-a969-6bea09a41358
# ╠═9db29532-6d82-495c-bc3f-0daa882f5064
# ╠═10000001-0000-0000-0000-000000000001
# ╠═341dbed8-1d09-4b46-8434-eb332c332f75
# ╠═c7f70869-8f84-4c33-a455-d79f78ac02ec
# ╠═da62431a-7cc6-4253-986d-5ba7d39e9f90
# ╠═53f17afb-91fb-4881-a9f4-9fa87a24fee6
# ╠═418c15e5-8116-4d86-8c3e-aeac13cc3ef1
# ╠═10000002-0000-0000-0000-000000000001
# ╠═10000003-0000-0000-0000-000000000001
# ╟─10000004-0000-0000-0000-000000000001
# ╟─10000005-0000-0000-0000-000000000001
# ╠═10000006-0000-0000-0000-000000000001
# ╠═10000007-0000-0000-0000-000000000001
# ╠═10000008-0000-0000-0000-000000000001
# ╠═10000009-0000-0000-0000-000000000001
# ╠═1000000a-0000-0000-0000-000000000001
# ╠═1000000c-0000-0000-0000-000000000001
# ╟─1000000d-0000-0000-0000-000000000001
# ╠═f3b8c867-f8f9-45e0-a41f-4b7d7a1b17f0
# ╠═0f84f2f6-6403-4a2e-9c42-6f8a84a2bc3f
# ╠═2a8a4d12-c96b-4d64-8f58-98f54f81a77b
# ╠═eaa7d770-d2d0-42f0-a65b-99447f11649a
# ╠═1000000e-0000-0000-0000-000000000001
# ╠═f86fec7f-f467-4411-80aa-c1621e3de063
# ╠═1000000f-0000-0000-0000-000000000001
# ╠═e8ae6df6-fa04-42b7-a6e5-b2ade4322995
# ╠═c669f1d3-26c0-4028-b3d4-5a87bd696924
# ╟─10000010-0000-0000-0000-000000000001
# ╠═10000011-0000-0000-0000-000000000001
# ╠═10000012-0000-0000-0000-000000000001
# ╠═10000013-0000-0000-0000-000000000001
# ╠═10000014-0000-0000-0000-000000000001
# ╟─10000015-0000-0000-0000-000000000001
# ╟─10000017-0000-0000-0000-000000000001
# ╠═10000018-0000-0000-0000-000000000001
# ╠═6d7cebf7-fb3e-4134-a428-91dea9f272b4
# ╟─10000019-0000-0000-0000-000000000001
# ╠═1000001a-0000-0000-0000-000000000001
# ╠═83946706-1d00-4794-af60-8c65979236f8
# ╠═1000001b-0000-0000-0000-000000000001
# ╠═1000001c-0000-0000-0000-000000000001
# ╠═1a623368-a2fc-4b7f-8d01-68e36d04a891
# ╠═f2369548-2d88-11f1-a737-85bad04c89cb
# ╟─1000001e-0000-0000-0000-000000000001
# ╠═10000020-0000-0000-0000-000000000001
# ╠═10000021-0000-0000-0000-000000000001
# ╠═481e1558-d494-40cb-990e-45b7ad76a670
# ╟─10000022-0000-0000-0000-000000000001
# ╟─10000023-0000-0000-0000-000000000001
# ╟─10000024-0000-0000-0000-000000000001
# ╠═1000001d-0000-0000-0000-000000000001
# ╠═10000025-0000-0000-0000-000000000001
# ╠═01c3acc7-efc4-471a-8f4e-515c59ae3cf8
# ╟─10000026-0000-0000-0000-000000000001
# ╟─10000027-0000-0000-0000-000000000001
# ╟─10000028-0000-0000-0000-000000000001
# ╟─10000029-0000-0000-0000-000000000001
# ╟─1000002a-0000-0000-0000-000000000001
# ╟─1000002b-0000-0000-0000-000000000001
# ╠═1000002c-0000-0000-0000-000000000001
# ╟─77766676-bc51-4554-80e3-65112f725fd2
# ╟─1000002e-0000-0000-0000-000000000001
# ╟─1000002f-0000-0000-0000-000000000001
# ╟─ed09ec19-f1b0-4819-9c9f-7ee796bd8f09
# ╟─a0000201-0000-0000-0000-000000000001
# ╠═10000031-0000-0000-0000-000000000001
# ╟─10000032-0000-0000-0000-000000000001
# ╟─3da3c466-6d7f-4d8b-8ba2-1f7a66ff9a46
# ╟─474a66d1-8851-4146-bc65-2026196981ee
# ╟─3bf7a033-db24-4e6c-93a9-8706d2ea57c3
# ╟─1ce2fb4a-ff49-4873-9f93-6decbf9b8e80
# ╟─84c4a49c-c6fb-45b4-ac33-a5510cd6618e
# ╟─10000033-0000-0000-0000-000000000001
# ╟─6a3e970f-a7e7-4c67-a9e6-dc79be22c547
# ╟─41391781-fa8e-4479-a1a8-7132053026bf
# ╠═6ad210ef-a2dc-47d8-bd7e-a06ab7ac1a32
# ╠═b81bef7c-b237-4589-a10d-dd92f95232f3
# ╠═6050d016-1589-47bc-90a3-87c441a59acf
# ╠═b03589cd-99ba-44e6-88c8-27b733ae0e96
# ╟─1cac8d93-75ed-4dfb-b330-ae26e5750ff2
# ╠═993f8556-a7fb-45d4-b097-3e4420776e14
# ╠═40aec027-40cc-4901-9232-2091b69091de
# ╠═cec27a5a-f20c-459b-a4ad-8a8b0e792d13
# ╠═785a4efa-d336-458a-8c8b-aca5f24e27d3
# ╠═60630929-d25c-46b4-adf6-08b5927fe3aa
# ╠═0825f860-0d47-4b0b-ab97-2436a0f97136
# ╠═aea9e546-cab6-4cae-a0d4-71990763a1ec
# ╠═ae8aefcf-9397-451b-b5db-1afb922b9347
# ╟─10000034-0000-0000-0000-000000000001
# ╠═10000035-0000-0000-0000-000000000001
# ╟─db4ddb38-2938-11f1-b8e3-e5227df9322c
# ╠═80f07cb0-0a48-49c8-8f35-476024d3b826
# ╠═e928b8ab-e159-427a-b525-c6d60e6d6015
# ╠═b1f70fd0-1a7a-4f8f-9e7c-8e3e8cd3f5e1
# ╠═74d0f419-79d0-4a6a-ae65-18f6be9d64c8
# ╠═a0000101-0000-0000-0000-000000000001
