### A Pluto.jl notebook ###
# v0.20.21

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
md"""# VQ-VAE + Spatial Transformer — Training Notebook

Train a **VQ-VAE with Fourier time-shift alignment** on either
synthetic Ricker waveforms (for validation) or real station-pair data.

The model jointly learns:
- **Discrete codebook prototypes** (via VQ) in a canonical zero-shift frame
- **Per-waveform time shifts** (via differentiable Fourier interpolation)
"""

# ╔═╡ 10000005-0000-0000-0000-000000000001
md"## Load Architecture"

# ╔═╡ 10000006-0000-0000-0000-000000000001
vqvae = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/VQVAE_spatial_transformer_v1.jl")

# ╔═╡ st_dt_cell
dt = 1.0   # sampling interval in seconds

# ╔═╡ st_filter_cell
begin
    period_min = 3
    period_max = 19
    responsetype = Bandpass(inv(period_max), inv(period_min))
    designmethod = Butterworth(2)
    digfilter = digitalfilter(responsetype, designmethod; fs=inv(dt))
end

# ╔═╡ st_taper_cell
function taper(x)
    w = cat(tukey(size(x, 1), 0.1), dims=ndims(x))
    return w .* x
end

# ╔═╡ st_data_cell
md"---
## Ricker Synthetic Test

Use this section to validate that the spatial transformer correctly recovers
known time shifts on noiseless/noisy Ricker waveforms before training on real data.
"

# ╔═╡ st_ricker_def
md"""
### Ricker Wavelet Generator

`ricker(t, f0)` = (1 - 2π²f₀²t²) · exp(−π²f₀²t²), normalized to unit peak amplitude.
"""

# ╔═╡ st_ricker_fn
function ricker(t::AbstractVector{<:Real}, f0::Real)
    w = @. (1f0 - 2f0 * Float32(π)^2 * Float32(f0)^2 * Float32(t)^2) *
           exp(-Float32(π)^2 * Float32(f0)^2 * Float32(t)^2)
    return Float32.(w ./ max(maximum(abs.(w)), 1f-10))
end

# ╔═╡ st_syn_params
begin
    nt_syn = 256          # waveform length
    f0_syn = 0.05f0       # dominant frequency in (normalized) cycles per sample
    noise_std_syn = 0.05f0   # RMS noise level (relative to peak amplitude)
    N_syn = 600           # number of synthetic waveforms
    max_shift_syn = 40    # ±40 samples true shift range (uniform random)
    rng_syn = Random.MersenneTwister(42)
end

# ╔═╡ st_syn_gen
"""
    make_ricker_dataset(; nt, f0, noise_std, N, max_shift, rng)

Generate N Ricker waveforms with random time shifts applied via Fourier interpolation.
Returns `(X_shifted, tau_true)` where:
- `X_shifted`: Float32 matrix `(nt, N)` — noisy, shifted waveforms on CPU
- `tau_true`: Float32 vector `(N,)` — true shifts in samples
"""
function make_ricker_dataset(; nt=nt_syn, f0=f0_syn, noise_std=noise_std_syn,
    N=N_syn, max_shift=max_shift_syn, rng=rng_syn)
    # Unit Ricker at zero shift (centered)
    t = collect(Float32, range(-(nt / 2), (nt / 2); length=nt))
    w0 = ricker(t, f0)                              # (nt,)

    # True shifts: uniform in [-max_shift, +max_shift] samples
    tau_true = Float32.((rand(rng, Float64, N) .* 2 .- 1) .* Float64(max_shift))

    # Replicate wavelet into a matrix
    W = repeat(w0, 1, N)                            # (nt, N)

    # Sampling grid for Fourier shift
    sg = im .* Float32.(fftfreq(nt) .* (2π * nt))  # (nt,) complex

    tau_mat = reshape(tau_true, 1, N)               # (1, N) — matches VQVAE_ST convention

    # Apply true shifts (Fourier interpolation, CPU)
    W_shifted = vqvae.shift_traces_Fourier(W, tau_mat, sg)  # (nt, N)

    # Add noise
    noise = noise_std .* Float32.(randn(rng, Float32, nt, N))
    W_noisy = Float32.(W_shifted) .+ noise

    return W_noisy, tau_true
end

# ╔═╡ st_syn_make
X_syn, tau_true_syn = make_ricker_dataset()

# ╔═╡ st_syn_preview
let
    nshow = 10
    ts = collect(1:nt_syn)
    traces = [PlutoPlotly.scatter(
        x=ts, y=X_syn[:, i], mode="lines",
        name="τ=$(round(tau_true_syn[i]; digits=1))s",
        line=attr(width=1)) for i in 1:nshow]
    layout = Layout(
        title=attr(text="Ricker waveforms (first $nshow) with random true shifts",
            font=attr(size=16, family="Computer Modern, serif")),
        height=350, width=900,
        xaxis=attr(title="Sample"), yaxis=attr(title="Amplitude"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ st_syn_split
begin
    idx_syn = randperm(rng_syn, N_syn)
    ntrain_syn = round(Int, 0.85 * N_syn)
    X_syn_train = xpu(X_syn[:, idx_syn[1:ntrain_syn]])
    X_syn_test  = xpu(X_syn[:, idx_syn[ntrain_syn+1:end]])
    tau_true_train = tau_true_syn[idx_syn[1:ntrain_syn]]
    tau_true_test  = tau_true_syn[idx_syn[ntrain_syn+1:end]]
end;

# ╔═╡ st_syn_params_model
md"### Build Synthetic Model"

# ╔═╡ st_syn_vqvae_para
vqvae_syn_parameters = (;
    nt=nt_syn,
    d=16,
    K=4,             # small codebook: only ~2-3 "shapes" in Ricker data (same polarity etc.)
    T=1,
    beta_commit=0.25f0,
    ema_decay=0.99f0,
    enc_kernels=[32, 16, 8, 4],
    enc_filters=[8, 16, 32, 32],
    enc_strides=[2, 2, 2, 2],
    use_bn=true,
    dead_threshold=20,
    entropy_weight=0.05f0,
    interstation_distance=nothing,   # no physics window for synthetic test
    dt=1.0,
    reference_velocity=3.0,
    # ── Spatial transformer ──────────────────────────────────────────────
    gamma=0.001f0,
    max_shift_samples=max_shift_syn + 10,   # slightly wider than true range
    shift_penalty_type=:l2,
    cauchy_sigma=Float32(max_shift_syn) / 2f0,
    seed=42,
)

# ╔═╡ st_syn_reload
reload_syn_button = @bind reload_syn CounterButton("Reload Synthetic Model")

# ╔═╡ st_syn_model
model_syn, loss_history_syn = @use_memo([reload_syn, vqvae_syn_parameters]) do
    reload_syn
    vqvae.get_vqvae(vqvae.VQVAE_Para(; vqvae_syn_parameters...))
end

# ╔═╡ st_syn_train
trained_syn = @use_memo([]) do
    training_para = vqvae.VQVAE_Training_Para(
        batchsize=128,
        nepoch=80,
        initial_learning_rate=0.001,
        lr_decay=0.995,
    )
    para = vqvae.VQVAE_Para(; vqvae_syn_parameters...)
    vqvae.update(model_syn, loss_history_syn,
        X_syn_train, X_syn_test,
        para, training_para)
    randn(), loss_history_syn
end

# ╔═╡ st_syn_dashboard
begin
    trained_syn
    PlutoPlotly.plot(vqvae.plot_training_dashboard(loss_history_syn;
        title="Synthetic Ricker Test — VQ-VAE ST Training"))
end

# ╔═╡ st_syn_shift_scatter
md"### Shift Recovery: True vs Predicted"

# ╔═╡ st_syn_shift_pred
# Predict shifts on the test set (on original order test waveforms)
tau_pred_test = let
    trained_syn
    res = vqvae.encode(model_syn, X_syn_test)
    vec(cpu(res.shifts))
end

# ╔═╡ st_syn_shift_plot
let
    trained_syn
    τ_true = tau_true_test
    τ_pred = tau_pred_test

    corr_val = cor(τ_true, τ_pred)
    rmse_val = sqrt(mean((τ_true .- τ_pred).^2))

    # Perfect prediction reference line
    lim = Float64(max_shift_syn) * 1.1
    ref_line = PlutoPlotly.scatter(
        x=[-lim, lim], y=[-lim, lim],
        mode="lines", name="y = x (perfect)",
        line=attr(color="black", width=1.5, dash="dash"),
    )

    scatter_trace = PlutoPlotly.scatter(
        x=τ_true, y=τ_pred,
        mode="markers", name="Test waveforms",
        marker=attr(size=5, opacity=0.6,
            color=τ_true,
            colorscale="RdBu", cmid=0,
            colorbar=attr(title="True shift (samples)")),
        text=["τ_true=$(round(τ_true[i]; digits=1)), τ_pred=$(round(τ_pred[i]; digits=1))"
              for i in eachindex(τ_true)],
        hoverinfo="text",
    )

    layout = Layout(
        title=attr(
            text="Shift Recovery: True vs Predicted   |   r=$(round(corr_val; digits=3)),  RMSE=$(round(rmse_val; digits=2)) samples",
            font=attr(size=16, family="Computer Modern, serif")),
        height=500, width=600,
        xaxis=attr(title="True shift (samples)", range=[-lim, lim],
            zeroline=true, zerolinecolor="rgba(0,0,0,0.3)"),
        yaxis=attr(title="Predicted shift (samples)", range=[-lim, lim],
            zeroline=true, zerolinecolor="rgba(0,0,0,0.3)"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot([ref_line, scatter_trace], layout)
end

# ╔═╡ st_syn_reconstruction
md"### Sample Reconstructions"

# ╔═╡ st_syn_recon_plot
let
    trained_syn
    nshow = 4
    sidx = [1, 2, 3, 4]
    x_sub = X_syn_test[:, sidx]
    result = model_syn(x_sub; training=false)
    x_orig  = cpu(x_sub)
    x_recon = cpu(result.xhat)
    shifts_sub = cpu(result.shifts)

    ts = collect(1:nt_syn)
    traces = AbstractTrace[]
    for (i, j) in enumerate(sidx)
        col_orig  = "#1f77b4"
        col_recon = "#d62728"
        off = (i - 1) * 2.5f0
        τ_pred_i = shifts_sub[1, i]
        τ_true_i = tau_true_test[j]
        push!(traces, PlutoPlotly.scatter(
            x=ts, y=x_orig[:, i] .+ off, mode="lines",
            name="orig #$j (true τ=$(round(τ_true_i;digits=1)), pred τ=$(round(τ_pred_i;digits=1)))",
            line=attr(color=col_orig, width=1.5)))
        push!(traces, PlutoPlotly.scatter(
            x=ts, y=x_recon[:, i] .+ off, mode="lines",
            name="recon #$j", showlegend=false,
            line=attr(color=col_recon, width=1.5, dash="dash")))
    end

    layout = Layout(
        title=attr(text="Reconstruction quality (blue=original, red dashed=reconstructed)",
            font=attr(size=15, family="Computer Modern, serif")),
        height=500, width=900,
        xaxis=attr(title="Sample"),
        yaxis=attr(title="Amplitude (offset per trace)"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=attr(x=1.02, xanchor="left"),
        margin=attr(r=200),
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ st_real_data_start
md"""---
## Real Data Training

Load a station pair, apply band-pass filter, train the spatial-transformer VQ-VAE.
"""

# ╔═╡ st_real_get_ac
function get_acausal_causal(pair::String, filepath::String)
    jldfile = load(filter(x -> occursin(pair, x), readdir(filepath, join=true))[1])
    correlations = jldfile["correlations"]
    headers = jldfile["headers"]
    distance = jldfile["dist"]
    return (; correlations, headers, distance)
end

# ╔═╡ st_normalise_cell
function normalise(X; dims=1)
    m = mean(X; dims); s = std(X; dims)
    return (X .- m) ./ max.(s, 1e-10)
end

# ╔═╡ st_build_bundle
function build_training_bundle(pair::Tuple{String,String};
    filepath="/mnt/NAS2/Sanket_data/California_TO_with_latlong/")
    pair_name = join(pair, "_")
    data_pair_local = get_acausal_causal(pair_name, filepath)
    D1 = data_pair_local.correlations
    D1 = normalise(D1, dims=1)
    D1ac, D1c = split_causal_acausal(D1, true)
    D1ac = taper(D1ac)
    D1c  = taper(D1c)
    D1fac = filtfilt(digfilter, D1ac)
    D1fc  = filtfilt(digfilter, D1c)
    D1fac = Float32.(normalise(D1fac[2:end, :], dims=1))
    D1fc  = Float32.(normalise(D1fc[2:end, :], dims=1))
    return (; pair, D1=Float32.(D1), D1fac, D1fc,
        distance=data_pair_local.distance)
end

# ╔═╡ st_split_ac
function split_causal_acausal(X::AbstractMatrix, zero_lag::Bool, max_lag=nothing)
    nt, ntr = size(X)
    !isodd(nt) && error("nt should be odd")
    center = div(nt + 1, 2)
    half   = div(nt - 1, 2)
    N = isnothing(max_lag) ? half : max(0, min(half, max_lag))
    N == 0 && return similar(X, 0, ntr), similar(X, 0, ntr)
    X_acausal = reverse(X[center-N:center-1, :], dims=1)
    X_causal  = X[center+1:center+N, :]
    if zero_lag
        return vcat(zeros(1, size(X)[2:end]...), Array(X_acausal)),
               vcat(zeros(1, size(X)[2:end]...), Array(X_causal))
    else
        return Array(X_acausal), Array(X_causal)
    end
end

# ╔═╡ st_pair_sel
selected_pair = ("CC01", "CC35")   # ← change as needed

# ╔═╡ st_load_data
data_bundle = build_training_bundle(selected_pair)

# ╔═╡ st_pooled_split
function make_pooled_split(D1fac, D1fc; at=0.9, shuffle=true)
    D_all = hcat(D1fac, D1fc)
    nw  = size(D_all, 2)
    idx = collect(1:nw)
    shuffle && Random.shuffle!(idx)
    ntrain = round(Int, at * nw)
    return (;
        D_train  = xpu(D_all[:, idx[1:ntrain]]),
        D_test   = xpu(D_all[:, idx[ntrain+1:end]]),
        D_all    = xpu(D_all),
        D_ac_all = xpu(D1fac),
        D_c_all  = xpu(D1fc),
    )
end

# ╔═╡ st_make_data
data = make_pooled_split(data_bundle.D1fac, data_bundle.D1fc)

# ╔═╡ st_nth_cell
nth = size(data.D_train, 1)

# ╔═╡ st_tgrid_cell
tgrid = collect(-nth:nth) .* dt

# ╔═╡ st_real_model_params
md"## Model Setup"

# ╔═╡ st_reload_btn
reload_network_button = @bind reload_network CounterButton("Reload Network")

# ╔═╡ st_vqvae_params
vqvae_parameters = (;
    nt=nth,
    d=20,
    K=5,
    T=1,
    beta_commit=0.25f0,
    ema_decay=0.999f0,
    enc_kernels=[64, 32, 16, 8],
    enc_filters=[8, 16, 32, 64],
    enc_strides=[1, 1, 1, 1],
    use_bn=true,
    dead_threshold=50,
    entropy_weight=0.1f0,
    interstation_distance=Float64(data_bundle.distance),  # km
    dt=dt,
    reference_velocity=3.0,
    # ── Spatial transformer ──────────────────────────────────────────────────────
    gamma=0.001f0,              # shift regularization weight; start small
    max_shift_samples=80,       # ±80 samples; tune to max physical offset
    shift_penalty_type=:l2,     # :l1 | :l2 | :cauchy | :bounded
    cauchy_sigma=20f0,          # used only if shift_penalty_type = :cauchy
    seed=nothing,
)

# ╔═╡ st_para_cell
vqvae_para = vqvae.VQVAE_Para(; vqvae_parameters...)

# ╔═╡ st_model_cell
model, loss_history = @use_memo([reload_network, vqvae_parameters]) do
    reload_network
    vqvae.get_vqvae(vqvae.VQVAE_Para(; vqvae_parameters...))
end

# ╔═╡ st_model_info
model.locnet

# ╔═╡ st_training_section
md"## Training"

# ╔═╡ st_pool_status
CUDA.pool_status()

# ╔═╡ st_train_cell
trained = @use_memo([]) do
    training_para = vqvae.VQVAE_Training_Para(
        batchsize=512,
        nepoch=50,
        initial_learning_rate=0.001,
        lr_decay=0.99,
    )
    vqvae.update(model, loss_history,
        data.D_train[:, :], data.D_test,
        vqvae_para, training_para)
    randn(), loss_history
end

# ╔═╡ st_dashboard
begin
    trained
    vqvae.plot_training_dashboard(loss_history;
        title="VQ-VAE ST: $(selected_pair[1])-$(selected_pair[2])  $(period_min)-$(period_max)s")
end

# ╔═╡ st_shift_dist
md"## Shift Distribution on Real Data"

# ╔═╡ st_shift_dist_plot
let
    trained
    res_ac = vqvae.encode(model, data.D_ac_all)
    res_c  = vqvae.encode(model, data.D_c_all)
    sh_ac  = vec(cpu(res_ac.shifts))
    sh_c   = vec(cpu(res_c.shifts))

    bin_edges = range(-Float64(vqvae_para.max_shift_samples),
        Float64(vqvae_para.max_shift_samples); length=40)
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in 1:length(bin_edges)-1]

    function histcounts(v, edges)
        counts = zeros(Int, length(edges) - 1)
        for x in v
            idx = searchsortedfirst(edges, x) - 1
            idx = clamp(idx, 1, length(counts))
            counts[idx] += 1
        end
        return counts
    end

    h_ac = histcounts(sh_ac, collect(bin_edges))
    h_c  = histcounts(sh_c,  collect(bin_edges))

    traces = [
        PlutoPlotly.bar(x=bin_centers, y=h_ac, name="Acausal",
            marker=attr(color="rgba(31,119,180,0.7)")),
        PlutoPlotly.bar(x=bin_centers, y=h_c,  name="Causal",
            marker=attr(color="rgba(214,39,40,0.7)")),
    ]
    layout = Layout(
        title=attr(text="Learned shift distribution — $(selected_pair[1])-$(selected_pair[2])",
            font=attr(size=16, family="Computer Modern, serif")),
        barmode="overlay",
        height=350, width=700,
        xaxis=attr(title="Shift (samples)"),
        yaxis=attr(title="Count"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ st_analysis_section
md"## Source State Analysis"

# ╔═╡ st_combo_labels
begin
    trained
    combo_labels = vqvae.combination_labels(vqvae_para.K, vqvae_para.T)
    combo_count  = length(combo_labels)
end;

# ╔═╡ st_cross_analysis
begin
    trained
    cross = vqvae.codebook_cross_analysis(model, data.D_ac_all, data.D_c_all)
end

# ╔═╡ st_cross_summary
md"""
**Codebook Cross-Analysis:**
- **Agreement rate**: $(round(cross.agreement * 100; digits=1))% of windows map to same code
- **Shared codes** (both >5%): $(cross.shared_codes)
- **Acausal-dominant codes**: $(cross.ac_only_codes)
- **Causal-dominant codes**: $(cross.c_only_codes)
"""

# ╔═╡ st_cluster_hist
begin
    trained
    vqvae.plot_cluster_histogram(cross.pct_ac, cross.pct_c;
        title="$(selected_pair[1])-$(selected_pair[2]) Source State Usage",
        labels=cross.labels)
end

# ╔═╡ st_confusion_cell
begin
    trained
    vqvae.plot_codebook_confusion(cross.confusion;
        title="$(selected_pair[1])-$(selected_pair[2]) Code Confusion",
        labels=cross.labels)
end

# ╔═╡ st_avg_section
md"### Source State Average Waveforms"

# ╔═╡ st_cluster_avgs
begin
    trained
    cluster_avg_ac = vqvae.get_cluster_averages(model, data.D_ac_all)
    cluster_avg_c  = vqvae.get_cluster_averages(model, data.D_c_all)
end;

# ╔═╡ st_avg_plot
let
    trained
    K = vqvae_para.K
    t_neg  = [-(nth - i + 1) * dt for i in 1:nth]
    t_pos  = [i * dt for i in 1:nth]
    t_full = [t_neg; t_pos]

    global_avg_ac   = vec(mean(cpu(data.D_ac_all); dims=2))
    global_avg_c    = vec(mean(cpu(data.D_c_all);  dims=2))
    global_full     = [reverse(global_avg_ac); global_avg_c]

    combo_labels_local = cross.labels
    ncomb = length(combo_labels_local)
    cs = ColorSchemes.rainbow
    colors = [Colors.hex(get(cs, (i - 1) / max(1, ncomb - 1))) for i in 1:ncomb]

    total_ac = size(data.D_ac_all, 2)
    total_c  = size(data.D_c_all, 2)
    vertical_spacing = maximum(abs.(vcat(vec(cluster_avg_ac), vec(cluster_avg_c)))) * 2.5 + 1e-3

    traces = AbstractTrace[]
    for combo_idx in 1:ncomb
        c = colors[mod1(combo_idx, length(colors))]
        a = cluster_avg_ac[:, combo_idx]
        b = cluster_avg_c[:, combo_idx]
        full_k = [reverse(a); b]
        a0 = a .- mean(a); b0 = b .- mean(b)
        ncc = dot(a0, b0) / (norm(a0) * norm(b0) + 1e-8)
        ks_tuple = Tuple(vqvae.combination_digits(combo_idx, K, vqvae_para.T))
        _, sel_ac = vqvae.filter_cluster(model, data.D_ac_all, ks_tuple)
        _, sel_c  = vqvae.filter_cluster(model, data.D_c_all,  ks_tuple)
        pct_ac = 100 * length(sel_ac) / max(total_ac, 1)
        pct_c  = 100 * length(sel_c)  / max(total_c,  1)
        label  = "Combo $(combo_labels_local[combo_idx]) (ac: $(round(pct_ac;digits=1))%, c: $(round(pct_c;digits=1))%, r=$(round(ncc;digits=3)))"
        offset = (combo_idx - 1) * vertical_spacing
        push!(traces, PlutoPlotly.scatter(x=t_full, y=full_k .+ offset,
            mode="lines", name=label, line=attr(color=c, width=2)))
    end
    push!(traces, PlutoPlotly.scatter(x=t_full, y=global_full, mode="lines",
        name="Global mean", line=attr(color="black", width=2, dash="dot")))

    layout = Layout(
        title=attr(
            text="Source State Averages — $(selected_pair[1])-$(selected_pair[2])  $(period_min)-$(period_max)s",
            font=attr(size=18, family="Computer Modern, serif")),
        height=500, width=900,
        xaxis=attr(title="Lag (s)", zeroline=true, zerolinecolor="rgba(0,0,0,0.3)"),
        yaxis=attr(title="Amplitude"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=attr(x=0.5, xanchor="center", y=-0.2, orientation="h",
            font=attr(size=12, family="Computer Modern, serif")),
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
ParameterSchedulers = "d7d3b36b-41b8-4d0d-a2bf-768c6151755e"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[compat]
BenchmarkTools = "~1.7.0"
CUDA = "~5.11.0"
Clustering = "~0.15.8"
ColorSchemes = "~3.30.0"
Colors = "~0.13.0"
DSP = "~0.8.4"
Distances = "~0.10.12"
Enzyme = "~0.13.138"
FFTW = "~1.10.0"
Flux = "~0.16.9"
Functors = "~0.5.2"
JLD2 = "~0.6.4"
MLUtils = "~0.4.8"
Optimisers = "~0.4.7"
ParameterSchedulers = "~0.4.3"
PlutoHooks = "~0.1.0"
PlutoLinks = "~0.1.8"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.80"
ProgressLogging = "~0.1.6"
StatsBase = "~0.34.10"
Zygote = "~0.7.10"
cuDNN = "~1.4.7"
"""
