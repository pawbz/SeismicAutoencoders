### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        $(esc(def))
    end
    #! format: on
end

# ╔═╡ tr000001-0000-0000-0000-000000000001
begin
    using CUDA, cuDNN, Flux, Zygote, FFTW
    using Optimisers, Random, Statistics, LinearAlgebra
    using JLD2, DSP, ProgressLogging
    using PlutoUI, PlutoHooks, PlutoLinks, PlutoPlotly
    CUDA.device!(0)
end

# ╔═╡ tr000002-0000-0000-0000-000000000001
using PlutoLinks: @ingredients

# ╔═╡ tr000003-0000-0000-0000-000000000001
TableOfContents(include_definitions=true)

# ╔═╡ tr000004-0000-0000-0000-000000000001
xpu = gpu

# ╔═╡ tr000005-0000-0000-0000-000000000001
md"""# Phase Aligner — Training Notebook

Train a **Design C siamese phase aligner** to self-supervisedly align a set of
waveforms to a common frame, then compute the coherent stack.

## Workflow
1. Load / generate waveforms
2. Build model from `PhaseAligner_architecture.jl`
3. **Phase 1** — equivariance pre-training (no baseline needed)
4. **Phase 2** — EMA baseline alignment
5. Plot diagnostics: φ distribution, shift scatter, aligned stack
"""

# ╔═╡ tr000006-0000-0000-0000-000000000001
md"## Load Architecture"

# ╔═╡ tr000007-0000-0000-0000-000000000001
pa = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/PhaseAligner_architecture.jl")

# ╔═╡ tr000008-0000-0000-0000-000000000001
md"---
## Synthetic Validation

Generate Ricker waveforms with known shifts to verify the aligner works before
running on real data.
"

# ╔═╡ tr000009-0000-0000-0000-000000000001
begin
    # ── Synthetic dataset parameters ─────────────────────────────────────
    nt_syn        = 300
    f0_syn        = 0.05f0          # dominant frequency (cycles/sample)
    noise_std_syn = 0.3f0           # noise level relative to peak
    N_syn         = 800
    max_shift_syn = 40              # ±40 samples true shift range
    rng_syn       = Random.MersenneTwister(42)
end

# ╔═╡ tr000010-0000-0000-0000-000000000001
"""
Generate N Ricker waveforms with known true shifts.
Returns `(X_noisy, tau_true_samples)` — both on CPU, Float32.
"""
function make_ricker_dataset_syn(; nt=nt_syn, f0=f0_syn,
                                   noise_std=noise_std_syn,
                                   N=N_syn, max_shift=max_shift_syn,
                                   rng=rng_syn)
    t  = collect(Float32, range(-(nt/2), nt/2; length=nt))
    w0 = @. (1f0 - 2f0*Float32(π)^2*f0^2*t^2) * exp(-Float32(π)^2*f0^2*t^2)
    w0 ./= max(maximum(abs.(w0)), 1f-10)

    tau = Float32.(rand(rng, Float64, N) .* 2 .- 1) .* Float32(max_shift)

    # Fourier shift on CPU
    sg  = -im .* Float32.(fftfreq(nt) .* 2π)
    tau_mat = reshape(tau, 1, N)
    W   = repeat(w0, 1, N)
    W_s = real(ifft(fft(W, 1) .* exp.(sg .* tau_mat), 1))

    noise = noise_std .* randn(rng, Float32, nt, N)
    W_n   = W_s .+ noise

    # Per-trace zero-mean unit-std normalization
    m = mean(W_n; dims=1); s = std(W_n; dims=1)
    W_norm = (W_n .- m) ./ max.(s, 1f-8)
    return Float32.(W_norm), tau
end

# ╔═╡ tr000011-0000-0000-0000-000000000001
X_syn, tau_true_syn = make_ricker_dataset_syn()

# ╔═╡ tr000012-0000-0000-0000-000000000001
begin
    ntrain_syn = round(Int, 0.85 * N_syn)
    idx_syn    = randperm(rng_syn, N_syn)
    X_train_syn = xpu(X_syn[:, idx_syn[1:ntrain_syn]])
    X_test_syn  = xpu(X_syn[:, idx_syn[ntrain_syn+1:end]])
    tau_true_train_syn = tau_true_syn[idx_syn[1:ntrain_syn]]
    tau_true_test_syn  = tau_true_syn[idx_syn[ntrain_syn+1:end]]
end;

# ╔═╡ tr000013-0000-0000-0000-000000000001
md"## Model Parameters"

# ╔═╡ tr000014-0000-0000-0000-000000000001
pa_params_syn = pa.PhaseAligner_Para(
    nt              = nt_syn,
    max_shift_samples = max_shift_syn + 10,   # a bit wider than true range
    enc_kernels     = [64, 32, 16],
    enc_filters     = [8, 16, 32],
    gamma           = 0.0001f0,
    ema_decay       = 0.99f0,
    seed            = 42,
)

# ╔═╡ tr000015-0000-0000-0000-000000000001
reload_syn_button = @bind reload_syn_pa CounterButton("Reload Model")

# ╔═╡ tr000016-0000-0000-0000-000000000001
model_syn_pa = @use_memo([reload_syn_pa, pa_params_syn]) do
    reload_syn_pa
    pa.get_phase_aligner(pa.PhaseAligner_Para(; pa_params_syn...))
end

# ╔═╡ tr000017-0000-0000-0000-000000000001
md"## Training"

# ╔═╡ tr000018-0000-0000-0000-000000000001
pa_training_para_syn = pa.PhaseAligner_Training_Para(
    batchsize        = 256,
    nepoch_phase1    = 150,
    nepoch_phase2    = 400,
    initial_lr_phase1 = 0.001,
    initial_lr_phase2 = 0.0003,
    lr_decay         = 0.995,
    nprint           = 25,
)

# ╔═╡ tr000019-0000-0000-0000-000000000001
trained_syn_pa = @use_memo([]) do
    pa.train_phase_aligner(
        model_syn_pa,
        X_train_syn, X_test_syn,
        pa_params_syn, pa_training_para_syn,
    )
end

# ╔═╡ tr000020-0000-0000-0000-000000000001
baseline_syn, loss_history_syn_pa = trained_syn_pa

# ╔═╡ tr000021-0000-0000-0000-000000000001
md"## Diagnostics"

# ╔═╡ tr000022-0000-0000-0000-000000000001
md"### Loss curves"

# ╔═╡ tr000023-0000-0000-0000-000000000001
let
    h = loss_history_syn_pa
    n1 = pa_training_para_syn.nepoch_phase1
    n2 = pa_training_para_syn.nepoch_phase2
    e1 = 1:n1
    e2 = n1+1:n1+n2
    font = attr(family="Computer Modern, serif")
    traces = [
        PlutoPlotly.scatter(x=collect(e1), y=h.train_total[e1], mode="lines",
            name="Train Phase 1", line=attr(color="#1f77b4", width=1.5)),
        PlutoPlotly.scatter(x=collect(e1), y=h.test_total[e1], mode="lines",
            name="Test Phase 1", line=attr(color="#1f77b4", width=1.5, dash="dash")),
        PlutoPlotly.scatter(x=collect(e2), y=h.train_total[e2], mode="lines",
            name="Train Phase 2", line=attr(color="#d62728", width=1.5)),
        PlutoPlotly.scatter(x=collect(e2), y=h.test_total[e2], mode="lines",
            name="Test Phase 2", line=attr(color="#d62728", width=1.5, dash="dash")),
    ]
    layout = Layout(
        title=attr(text="Training loss (Phase 1 + Phase 2)", font=merge(font, attr(size=16))),
        xaxis=attr(title="Epoch", type="linear"),
        yaxis=attr(title="Loss", type="log"),
        height=350, width=900,
        plot_bgcolor="white", paper_bgcolor="white",
        shapes=[attr(type="line", x0=n1, x1=n1, y0=0, y1=1,
                     yref="paper", line=attr(color="grey", dash="dot", width=1))],
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ tr000024-0000-0000-0000-000000000001
md"### φ distribution"

# ╔═╡ tr000025-0000-0000-0000-000000000001
let
    phi_train = vec(cpu(model_syn_pa(X_train_syn)))
    bins = range(minimum(phi_train) - 1, maximum(phi_train) + 1; length=40)
    counts = [sum(bins[i] .<= phi_train .< bins[i+1]) for i in 1:length(bins)-1]
    centers = [(bins[i] + bins[i+1]) / 2 for i in 1:length(bins)-1]
    traces = [
        PlutoPlotly.bar(x=centers, y=counts, name="φ(x_i)",
            marker=attr(color="#1f77b4", opacity=0.7)),
        PlutoPlotly.scatter(x=[baseline_syn.phi_baseline, baseline_syn.phi_baseline],
            y=[0, maximum(counts)], mode="lines",
            name="φ_B (baseline)", line=attr(color="red", dash="dash", width=2)),
    ]
    layout = Layout(
        title=attr(text="Phase coordinate distribution (train set)"),
        xaxis=attr(title="φ (samples)"), yaxis=attr(title="Count"),
        height=300, width=800, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ tr000026-0000-0000-0000-000000000001
md"### True vs predicted shift scatter"

# ╔═╡ tr000027-0000-0000-0000-000000000001
let
    tau_pred_train, _ = pa.predict_shifts(model_syn_pa, X_train_syn, baseline_syn)
    tau_pred = vec(cpu(tau_pred_train))
    tau_true = tau_true_train_syn

    # Scatter: predicted vs true
    lim = max(maximum(abs.(tau_true)), maximum(abs.(tau_pred))) * 1.1
    ref = PlutoPlotly.scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
        name="ideal", line=attr(color="grey", dash="dot", width=1))
    sc  = PlutoPlotly.scatter(x=tau_true, y=tau_pred, mode="markers",
        name="train", marker=attr(size=4, color="#1f77b4", opacity=0.6))
    layout = Layout(
        title=attr(text="True shift vs predicted shift (samples)"),
        xaxis=attr(title="τ_true (samples)", range=[-lim, lim]),
        yaxis=attr(title="τ_pred (samples)", range=[-lim, lim]),
        height=400, width=450, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot([ref, sc], layout)
end

# ╔═╡ tr000028-0000-0000-0000-000000000001
md"### Coherent stack"

# ╔═╡ tr000029-0000-0000-0000-000000000001
let
    tau_all, X_aligned_all = pa.apply_aligner(model_syn_pa, X_syn, baseline_syn)
    stack_aligned = pa.coherent_stack(X_aligned_all)

    # Raw mean (unaligned) for comparison
    stack_raw = vec(mean(X_syn; dims=2))

    ts = 1:nt_syn
    traces = [
        PlutoPlotly.scatter(x=ts, y=stack_raw, mode="lines",
            name="Raw mean (unaligned)", line=attr(color="grey", width=1.5, dash="dash")),
        PlutoPlotly.scatter(x=ts, y=stack_aligned, mode="lines",
            name="Aligned stack (PhaseAligner)", line=attr(color="#1f77b4", width=2)),
    ]
    layout = Layout(
        title=attr(text="Coherent stack: aligned vs raw mean"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Amplitude"),
        height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ tr000030-0000-0000-0000-000000000001
md"---
## Real Data Section

Replace the synthetic data above with your real station waveforms.
The workflow is identical — just substitute `X_train_syn` / `X_test_syn`
with your preprocessed waveform matrix `(nt, N)`.
"

# ╔═╡ tr000031-0000-0000-0000-000000000001
begin
    fldir = "/mnt/NAS/EQData/RFData"
    dfile = "$(fldir)/Syn410Ps_snr1.5.jld2"
    EqR      = load(dfile, "Syn")["Data"]
    TrueRF   = load(dfile, "Syn")["TrueRF"]
    StaName  = load(dfile, "Syn")["Sta"][1]
    EvtLoc   = load(dfile, "Syn")["EventLoc"]
end

# ╔═╡ tr000032-0000-0000-0000-000000000001
begin
    s_sel = "1"
    stik  = findall(x -> x == s_sel, StaName)[1]
    raw_data = EqR[stik][451:750, :]   # (nt_real, N_real) trim for 410Ps window
    nt_real  = size(raw_data, 1)
    N_real   = size(raw_data, 2)
end

# ╔═╡ tr000033-0000-0000-0000-000000000001
begin
    # Per-trace normalization
    mr = mean(raw_data; dims=1); sr = std(raw_data; dims=1)
    X_real = Float32.((raw_data .- mr) ./ max.(sr, 1f-8))
end

# ╔═╡ tr000034-0000-0000-0000-000000000001
begin
    rng_real = Random.MersenneTwister(7)
    ntrain_real = round(Int, 0.85 * N_real)
    idx_real    = randperm(rng_real, N_real)
    X_train_real = xpu(X_real[:, idx_real[1:ntrain_real]])
    X_test_real  = xpu(X_real[:, idx_real[ntrain_real+1:end]])
end;

# ╔═╡ tr000035-0000-0000-0000-000000000001
pa_params_real = pa.PhaseAligner_Para(
    nt                = nt_real,
    max_shift_samples = 30,          # adjust to your expected lag range
    enc_kernels       = [64, 32, 16],
    enc_filters       = [8, 16, 32],
    gamma             = 0.0001f0,
    ema_decay         = 0.99f0,
    seed              = 42,
)

# ╔═╡ tr000036-0000-0000-0000-000000000001
reload_real_button = @bind reload_real_pa CounterButton("Reload Real Model")

# ╔═╡ tr000037-0000-0000-0000-000000000001
model_real_pa = @use_memo([reload_real_pa, pa_params_real]) do
    reload_real_pa
    pa.get_phase_aligner(pa.PhaseAligner_Para(; pa_params_real...))
end

# ╔═╡ tr000038-0000-0000-0000-000000000001
pa_training_para_real = pa.PhaseAligner_Training_Para(
    batchsize        = 128,
    nepoch_phase1    = 200,
    nepoch_phase2    = 500,
    initial_lr_phase1 = 0.001,
    initial_lr_phase2 = 0.0003,
    lr_decay         = 0.995,
    nprint           = 50,
)

# ╔═╡ tr000039-0000-0000-0000-000000000001
trained_real_pa = @use_memo([]) do
    pa.train_phase_aligner(
        model_real_pa,
        X_train_real, X_test_real,
        pa_params_real, pa_training_para_real,
    )
end

# ╔═╡ tr000040-0000-0000-0000-000000000001
baseline_real, loss_history_real_pa = trained_real_pa

# ╔═╡ tr000041-0000-0000-0000-000000000001
md"### Real Data: Aligned Stack"

# ╔═╡ tr000042-0000-0000-0000-000000000001
let
    tau_all, X_aligned_all = pa.apply_aligner(model_real_pa, X_real, baseline_real)
    stack_aligned = pa.coherent_stack(X_aligned_all)
    stack_raw     = vec(mean(X_real; dims=2))

    ts = 1:nt_real
    traces = [
        PlutoPlotly.scatter(x=ts, y=stack_raw, mode="lines",
            name="Raw mean", line=attr(color="grey", width=1.5, dash="dash")),
        PlutoPlotly.scatter(x=ts, y=stack_aligned, mode="lines",
            name="Aligned stack", line=attr(color="#d62728", width=2)),
    ]
    layout = Layout(
        title=attr(text="Real data: coherent stack"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Amplitude"),
        height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ tr000043-0000-0000-0000-000000000001
md"### Real Data: Shift Distribution"

# ╔═╡ tr000044-0000-0000-0000-000000000001
let
    tau_all, _ = pa.apply_aligner(model_real_pa, X_real, baseline_real)
    bins = range(minimum(tau_all) - 0.5, maximum(tau_all) + 0.5; length=41)
    counts  = [sum(bins[i] .<= tau_all .< bins[i+1]) for i in 1:length(bins)-1]
    centers = [(bins[i] + bins[i+1]) / 2 for i in 1:length(bins)-1]
    traces = [PlutoPlotly.bar(x=centers, y=counts,
        name="shifts", marker=attr(color="#d62728", opacity=0.7))]
    layout = Layout(
        title=attr(text="Predicted shift distribution (samples)"),
        xaxis=attr(title="τ (samples)"), yaxis=attr(title="Count"),
        height=300, width=700, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA    = "052768ef-5323-5732-b1bb-66c8b64840ba"
cuDNN   = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
DSP     = "717857b8-e6f2-59f4-9121-6e50c889abd2"
FFTW    = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
Flux    = "587475ba-b771-5e3f-ad9e-33799f191a9c"
JLD2    = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random  = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Zygote  = "e88e6eb3-aa80-5325-afca-941959d7151f"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised
julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "phasealigner_training_placeholder"
"""
