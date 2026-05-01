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

# ╔═╡ sa000001-0000-0000-0000-000000000001
begin
    using CUDA, cuDNN, Flux, Zygote, FFTW
    using Optimisers, Random, Statistics, LinearAlgebra
    using JLD2, DSP, ProgressLogging, StatsBase
    using PlutoUI, PlutoHooks, PlutoLinks, PlutoPlotly
    CUDA.device!(0)
end

# ╔═╡ sa000002-0000-0000-0000-000000000001
using PlutoLinks: @ingredients

# ╔═╡ sa000003-0000-0000-0000-000000000001
TableOfContents(include_definitions=true)

# ╔═╡ sa000004-0000-0000-0000-000000000001
xpu = gpu

# ╔═╡ sa000005-0000-0000-0000-000000000001
md"""# VQ-VAE with Shift Augmentation

## Idea

Train a standard VQ-VAE on a **shift-augmented** dataset:
for each waveform `x_i`, create `M` copies shifted by uniformly spaced `τ ∈ [-max_shift, +max_shift]`.

The hope is that the VQ-VAE codebook converges to **shift-invariant shape prototypes**:
all shifted copies of a given waveform type cluster to the same code `k`.

## Finding the canonical alignment

After training, for each original waveform `x_i`:
1. Apply all `M` discrete shifts and run through the encoder
2. Keep the shift `τ_j*` that gives the **lowest reconstruction loss**
3. That shift IS the alignment: `x_i_aligned = shift(x_i, -τ_j*)`

Stack = `mean_i( x_i_aligned )` within each code group.

## Two-step workflow
```
Step 1:  Augment ✕M  →  Train VQ-VAE on augmented data
Step 2:  Scan shifts  →  Pick τ* = argmin recon loss  →  Align  →  Stack
```
"""

# ╔═╡ sa000006-0000-0000-0000-000000000001
md"## Load Architecture"

# ╔═╡ sa000007-0000-0000-0000-000000000001
vqvae = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/VQVAE_architecture_v3.1.jl")

# ╔═╡ sa000008-0000-0000-0000-000000000001
md"## Fourier Shift Utility"

# ╔═╡ sa000009-0000-0000-0000-000000000001
begin
    """
    Differentiable Fourier time-shift.
    - `x`:    `(nt, B)` Float32
    - `τ`:    `(1, B)` or scalar Float32 — shift in **samples**
    - `grid`: `(nt,)` complex = `-im * 2π * fftfreq(nt)` (precomputed)
    """
    function shift_traces_Fourier(x::AbstractMatrix{Float32},
                                   τ::AbstractMatrix{Float32},
                                   grid::AbstractVector)
        return real(ifft(fft(x, 1) .* exp.(grid .* τ), 1))
    end

    function shift_traces_Fourier(x::AbstractMatrix{Float32},
                                   τ_scalar::Float32,
                                   grid::AbstractVector)
        return shift_traces_Fourier(x, fill(τ_scalar, 1, size(x, 2)), grid)
    end

    """Sampling grid for Fourier shift (CPU or GPU)."""
    make_shift_grid(nt::Int) = -im .* Float32.(fftfreq(nt) .* 2π)
end

# ╔═╡ sa000010-0000-0000-0000-000000000001
md"## Shift Augmentation"

# ╔═╡ sa000011-0000-0000-0000-000000000001
begin
    """
        augment_with_shifts(X, M, max_shift, grid) -> (X_aug, tau_applied, orig_idx)

    Create an augmented dataset by applying `M` uniformly spaced shifts
    in `[-max_shift, +max_shift]` samples to each waveform in `X` (nt, N).

    Returns:
    - `X_aug`       : `(nt, N*M)` Float32 CPU matrix — all augmented waveforms
    - `tau_applied` : `(N*M,)` Float32 — which shift was applied (in samples)
    - `orig_idx`    : `(N*M,)` Int    — which original waveform (1..N) each row came from
    - `shifts_grid` : `(M,)` Float32  — the M discrete shift values used
    """
    function augment_with_shifts(X::AbstractMatrix{Float32},
                                  M::Int,
                                  max_shift::Real,
                                  grid::AbstractVector)
        nt, N = size(X)
        shifts_grid = Float32.(range(-max_shift, max_shift; length=M))
        X_aug      = zeros(Float32, nt, N * M)
        tau_applied = zeros(Float32, N * M)
        orig_idx    = zeros(Int, N * M)
        for (j, τ) in enumerate(shifts_grid)
            τ_mat = fill(τ, 1, N)
            X_s = real(ifft(fft(X, 1) .* exp.(grid .* τ_mat), 1))
            X_aug[:, (j-1)*N+1 : j*N] .= Float32.(X_s)
            tau_applied[(j-1)*N+1 : j*N] .= τ
            orig_idx[(j-1)*N+1 : j*N]   .= 1:N
        end
        return X_aug, tau_applied, orig_idx, shifts_grid
    end

    """Per-trace zero-mean unit-std normalization."""
    function normalise_traces(X::AbstractMatrix{Float32}; dims=1)
        m = mean(X; dims=dims)
        s = std(X; dims=dims)
        return (X .- m) ./ max.(s, 1f-8)
    end
end

# ╔═╡ sa000012-0000-0000-0000-000000000001
md"## Post-training: Find Best Shift per Waveform"

# ╔═╡ sa000013-0000-0000-0000-000000000001
begin
    """
        find_best_shifts(model, X, shifts_grid, grid_gpu; batchsize=256) 
            -> (tau_star, codes_star, recon_losses_by_shift)

    For each waveform `x_i` in `X` (nt, N), scan all shifts in `shifts_grid` and
    find `τ*` that minimises the VQ-VAE reconstruction loss.

    Returns:
    - `tau_star`   : `(N,)` Float32 — best alignment shift per waveform (samples)
    - `codes_star` : `(N,)` Int     — codebook entry at best shift
    - `recon_mat`  : `(M, N)` Float32 — reconstruction loss at each shift
    """
    function find_best_shifts(model, X::AbstractMatrix{Float32},
                               shifts_grid::AbstractVector{Float32},
                               grid_gpu::AbstractVector;
                               batchsize::Int=256)
        nt, N = size(X)
        M = length(shifts_grid)
        recon_mat  = zeros(Float32, M, N)
        codes_mat  = zeros(Int,     M, N)

        for (j, τ) in enumerate(shifts_grid)
            τ_mat = fill(τ, 1, N)
            grid_cpu = cpu(grid_gpu)
            X_s = real(ifft(fft(X, 1) .* exp.(grid_cpu .* τ_mat), 1))
            loader = Flux.DataLoader(Float32.(X_s); batchsize=batchsize, shuffle=false)
            offset = 0
            for xb in loader
                nb = size(xb, 2)
                result = vqvae.encode(model, xpu(xb); training=false)
                # Decode from quantized
                z_q_for_dec = reshape(result.z_q, model.d * model.T, nb)
                xhat = cpu(model.decoder(z_q_for_dec))
                recon_mat[j, offset+1:offset+nb] .= vec(mean(abs2, cpu(xb) .- xhat; dims=1))
                codes_mat[j, offset+1:offset+nb] .= vec(result.codebook_indices[1, :])
                offset += nb
            end
        end

        # Best shift = argmin reconstruction loss over shifts axis
        best_j = [argmin(recon_mat[:, i]) for i in 1:N]
        tau_star   = Float32[shifts_grid[j] for j in best_j]
        codes_star = Int[    codes_mat[j, i] for (i, j) in enumerate(best_j)]
        return tau_star, codes_star, recon_mat
    end

    """
    Align waveforms using best shifts and compute per-code stacks.
    Returns `(X_aligned, stacks)` where `stacks` is a Dict(code => mean_waveform).
    """
    function align_and_stack(X::AbstractMatrix{Float32},
                              tau_star::AbstractVector{Float32},
                              codes_star::AbstractVector{Int},
                              grid::AbstractVector)
        nt, N = size(X)
        X_aligned = zeros(Float32, nt, N)
        for i in 1:N
            x_i   = X[:, i:i]
            τ_mat = fill(-tau_star[i], 1, 1)
            X_aligned[:, i] .= vec(real(ifft(fft(x_i, 1) .* exp.(grid .* τ_mat), 1)))
        end
        codes = sort(unique(codes_star))
        stacks = Dict{Int, Vector{Float32}}()
        for k in codes
            sel = findall(==(k), codes_star)
            stacks[k] = vec(mean(X_aligned[:, sel]; dims=2))
        end
        return X_aligned, stacks
    end
end

# ╔═╡ sa000014-0000-0000-0000-000000000001
md"---
## Synthetic Validation

Test at multiple SNR levels. Ground-truth shifts are known so we can measure
shift recovery accuracy directly.
"

# ╔═╡ sa000015-0000-0000-0000-000000000001
begin
    nt_syn        = 300
    f0_syn        = 0.05f0
    N_syn         = 400          # waveforms per SNR test
    max_shift_syn = 30           # ±30 samples true shift range
    M_aug_syn     = 15           # augmentation factor (M discrete shifts)
    snr_levels    = [5.0, 2.0, 1.0, 0.5]   # SNR = peak / noise_std
    rng_syn       = Random.MersenneTwister(42)
end

# ╔═╡ sa000016-0000-0000-0000-000000000001
begin
    """
    Generate N Ricker waveforms with known true shifts (in samples).
    Returns `(X_norm, tau_true)` on CPU, Float32.
    """
    function make_ricker_syn(; nt=nt_syn, f0=f0_syn, N=N_syn,
                               max_shift=max_shift_syn, snr=2.0, rng=rng_syn)
        t  = collect(Float32, range(-nt/2, nt/2; length=nt))
        w0 = @. (1f0 - 2f0*Float32(π)^2*f0^2*t^2) * exp(-Float32(π)^2*f0^2*t^2)
        w0 ./= max(maximum(abs.(w0)), 1f-10)

        tau = Float32.(rand(rng, Float64, N) .* 2 .- 1) .* Float32(max_shift)
        grid = make_shift_grid(nt)
        W     = repeat(w0, 1, N)
        W_s   = real(ifft(fft(W, 1) .* exp.(grid .* reshape(tau, 1, N)), 1))

        noise_std = 1f0 / Float32(snr)
        noise = noise_std .* randn(rng, Float32, nt, N)
        W_n   = Float32.(W_s) .+ noise
        return normalise_traces(W_n), tau
    end
end

# ╔═╡ sa000017-0000-0000-0000-000000000001
md"### Model & Augmentation Parameters (Synthetic)"

# ╔═╡ sa000018-0000-0000-0000-000000000001
vqvae_params_syn = vqvae.VQVAE_Para(
    nt             = nt_syn,
    d              = 16,
    K              = 4,            # small: Ricker has ~1 shape type
    T              = 1,
    beta_commit    = 0.25f0,
    enc_kernels    = [64, 32, 16, 8],
    enc_filters    = [8,  16, 32, 32],
    enc_strides    = [1,  1,  1,  1],   # no striding → keep spatial resolution
    dec_kernels    = [8, 16, 64],
    dec_filters    = [32, 16, 8, 1],
    use_bn         = true,
    ema_decay      = 0.99f0,
    dead_threshold = 30,
    entropy_weight = 0.05f0,
    interstation_distance = nothing,
    seed           = 42,
)

# ╔═╡ sa000019-0000-0000-0000-000000000001
training_para_syn = vqvae.VQVAE_Training_Para(
    batchsize             = 256,
    nepoch                = 300,
    initial_learning_rate = 0.001,
    lr_decay              = 0.997,
    nprint                = 50,
)

# ╔═╡ sa000020-0000-0000-0000-000000000001
md"### Run Synthetic Tests at Multiple SNRs"

# ╔═╡ sa000021-0000-0000-0000-000000000001
reload_syn_button = @bind reload_syn CounterButton("Reload Synthetic Models")

# ╔═╡ sa000022-0000-0000-0000-000000000001
syn_results = @use_memo([reload_syn, vqvae_params_syn, training_para_syn]) do
    reload_syn
    grid_cpu = make_shift_grid(nt_syn)
    grid_gpu = xpu(grid_cpu)
    results  = Dict{Float64, NamedTuple}()

    for snr in snr_levels
        @info "Testing SNR=$snr"
        X_raw, tau_true = make_ricker_syn(; snr=snr)

        # Build augmented dataset
        X_aug, tau_aug, orig_idx_aug, shifts_grid = augment_with_shifts(
            X_raw, M_aug_syn, max_shift_syn, grid_cpu)
        X_aug_norm = normalise_traces(X_aug)

        ntrain = round(Int, 0.85 * size(X_aug_norm, 2))
        idx    = randperm(rng_syn, size(X_aug_norm, 2))
        D_train = xpu(X_aug_norm[:, idx[1:ntrain]])
        D_test  = xpu(X_aug_norm[:, idx[ntrain+1:end]])

        model, lhist = vqvae.get_vqvae(vqvae_params_syn)
        vqvae.update(model, lhist, D_train, D_test, vqvae_params_syn, training_para_syn)

        # Post-training: find best shifts on original (non-augmented) data
        tau_star, codes_star, recon_mat = find_best_shifts(
            model, X_raw, shifts_grid, grid_gpu)
        X_aligned, stacks = align_and_stack(X_raw, tau_star, codes_star, grid_cpu)

        results[snr] = (; model, lhist, tau_true, tau_star, codes_star,
                          X_raw, X_aligned, stacks, shifts_grid, recon_mat)
    end
    results
end

# ╔═╡ sa000023-0000-0000-0000-000000000001
md"### Diagnostic: True vs Predicted Shift (per SNR)"

# ╔═╡ sa000024-0000-0000-0000-000000000001
@bind snr_sel Select(string.(snr_levels); default=string(snr_levels[1]))

# ╔═╡ sa000025-0000-0000-0000-000000000001
let
    snr  = parse(Float64, snr_sel)
    r    = syn_results[snr]
    tt   = r.tau_true
    tp   = r.tau_star
    lim  = Float64(max(maximum(abs.(tt)), maximum(abs.(tp)))) * 1.1
    ref  = PlutoPlotly.scatter(x=[-lim,lim], y=[-lim,lim], mode="lines",
               name="ideal", line=attr(color="grey", dash="dot"))
    sc   = PlutoPlotly.scatter(x=tt, y=tp, mode="markers",
               name="SNR=$snr", marker=attr(size=4, opacity=0.6, color="#1f77b4"))
    rmse = round(sqrt(mean(abs2, tt .- tp)); digits=2)
    layout = Layout(
        title=attr(text="True vs predicted shift (SNR=$snr) — RMSE=$rmse samples"),
        xaxis=attr(title="τ_true (samples)", range=[-lim,lim]),
        yaxis=attr(title="τ_pred (samples)", range=[-lim,lim]),
        height=400, width=450, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot([ref, sc], layout)
end

# ╔═╡ sa000026-0000-0000-0000-000000000001
md"### Coherent stack per SNR"

# ╔═╡ sa000027-0000-0000-0000-000000000001
let
    font = attr(family="Computer Modern, serif")
    traces = []
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
    ts = 1:nt_syn
    for (i, snr) in enumerate(snr_levels)
        r = syn_results[snr]
        # stack across ALL codes (treat as one shape type)
        stack_aligned = vec(mean(r.X_aligned; dims=2))
        stack_raw     = vec(mean(r.X_raw; dims=2))
        push!(traces, PlutoPlotly.scatter(x=ts, y=stack_raw, mode="lines",
            name="Raw SNR=$snr", line=attr(color=colors[i], width=1, dash="dot")))
        push!(traces, PlutoPlotly.scatter(x=ts, y=stack_aligned, mode="lines",
            name="Aligned SNR=$snr", line=attr(color=colors[i], width=2)))
    end
    layout = Layout(
        title=attr(text="Coherent stack: aligned vs raw mean (multiple SNRs)", font=merge(font, attr(size=16))),
        xaxis=attr(title="Sample"), yaxis=attr(title="Normalised amplitude"),
        height=400, width=950, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ sa000028-0000-0000-0000-000000000001
md"### Training Loss Curve"

# ╔═╡ sa000029-0000-0000-0000-000000000001
let
    snr    = parse(Float64, snr_sel)
    r      = syn_results[snr]
    epochs = eachindex(r.lhist.train_recon)
    traces = [
        PlutoPlotly.scatter(x=collect(epochs), y=r.lhist.train_recon, mode="lines",
            name="Train recon", line=attr(color="#1f77b4", width=1.5)),
        PlutoPlotly.scatter(x=collect(epochs), y=r.lhist.test_recon, mode="lines",
            name="Test recon", line=attr(color="#1f77b4", dash="dash", width=1.5)),
    ]
    layout = Layout(
        title=attr(text="Recon loss — SNR=$snr"),
        xaxis=attr(title="Epoch"), yaxis=attr(title="MSE", type="log"),
        height=300, width=800, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ sa000030-0000-0000-0000-000000000001
md"### Code Usage per SNR"

# ╔═╡ sa000031-0000-0000-0000-000000000001
let
    snr = parse(Float64, snr_sel)
    r   = syn_results[snr]
    K   = vqvae_params_syn.K
    counts = [count(==(k), r.codes_star) for k in 1:K]
    pct    = counts ./ sum(counts) .* 100
    layout = Layout(
        title=attr(text="Code assignment distribution — SNR=$snr"),
        xaxis=attr(title="Code"), yaxis=attr(title="% waveforms"),
        height=280, width=500, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot([PlutoPlotly.bar(x=1:K, y=pct, marker=attr(color="#1f77b4"))], layout)
end

# ╔═╡ sa000032-0000-0000-0000-000000000001
md"---
## Real Data Section
### Load 410Ps Receiver Functions
"

# ╔═╡ sa000033-0000-0000-0000-000000000001
begin
    fldir   = "/mnt/NAS/EQData/RFData"
    dfile   = "$(fldir)/Syn410Ps_snr1.5.jld2"
    EqR     = load(dfile, "Syn")["Data"]
    TrueRF  = load(dfile, "Syn")["TrueRF"]
    StaName = load(dfile, "Syn")["Sta"][1]
    EvtLoc  = load(dfile, "Syn")["EventLoc"]
end

# ╔═╡ sa000034-0000-0000-0000-000000000001
begin
    s_sel   = "1"
    stik    = findall(x -> x == s_sel, StaName)[1]
    raw_data   = EqR[stik][451:750, :]         # (nt_real, N_real)
    true_rf    = TrueRF[stik][451:750, :]
    nt_real    = size(raw_data, 1)
    N_real     = size(raw_data, 2)
    X_real_raw = normalise_traces(Float32.(raw_data))
    @info "Real data" nt=nt_real N=N_real
end

# ╔═╡ sa000035-0000-0000-0000-000000000001
md"### Augmentation & Model Parameters (Real Data)"

# ╔═╡ sa000036-0000-0000-0000-000000000001
begin
    M_aug_real     = 20          # augmentation factor
    max_shift_real = 20          # ±20 samples (adjust to your data's expected lag)
end

# ╔═╡ sa000037-0000-0000-0000-000000000001
vqvae_params_real = vqvae.VQVAE_Para(
    nt             = nt_real,
    d              = 32,
    K              = 6,
    T              = 1,
    beta_commit    = 0.25f0,
    enc_kernels    = [64, 32, 16, 8],
    enc_filters    = [8,  16, 32, 64],
    enc_strides    = [1,  1,  1,  1],
    dec_kernels    = [8, 16, 64],
    dec_filters    = [64, 32, 8, 1],
    use_bn         = true,
    ema_decay      = 0.99f0,
    dead_threshold = 30,
    entropy_weight = 0.02f0,
    interstation_distance = nothing,
    seed           = 7,
)

# ╔═╡ sa000038-0000-0000-0000-000000000001
training_para_real = vqvae.VQVAE_Training_Para(
    batchsize             = 256,
    nepoch                = 400,
    initial_learning_rate = 0.001,
    lr_decay              = 0.997,
    nprint                = 50,
)

# ╔═╡ sa000039-0000-0000-0000-000000000001
reload_real_button = @bind reload_real CounterButton("Reload Real Model")

# ╔═╡ sa000040-0000-0000-0000-000000000001
real_result = @use_memo([reload_real, vqvae_params_real, training_para_real]) do
    reload_real
    grid_cpu = make_shift_grid(nt_real)
    grid_gpu = xpu(grid_cpu)

    X_aug, tau_aug, orig_idx_aug, shifts_grid = augment_with_shifts(
        X_real_raw, M_aug_real, max_shift_real, grid_cpu)
    X_aug_norm = normalise_traces(X_aug)
    @info "Augmented dataset" size=size(X_aug_norm)

    ntrain = round(Int, 0.85 * size(X_aug_norm, 2))
    rng_r  = Random.MersenneTwister(99)
    idx    = randperm(rng_r, size(X_aug_norm, 2))
    D_train = xpu(X_aug_norm[:, idx[1:ntrain]])
    D_test  = xpu(X_aug_norm[:, idx[ntrain+1:end]])

    model, lhist = vqvae.get_vqvae(vqvae_params_real)
    vqvae.update(model, lhist, D_train, D_test, vqvae_params_real, training_para_real)

    tau_star, codes_star, recon_mat = find_best_shifts(
        model, X_real_raw, shifts_grid, grid_gpu)
    X_aligned, stacks = align_and_stack(X_real_raw, tau_star, codes_star, grid_cpu)

    (; model, lhist, tau_star, codes_star, X_aligned, stacks, shifts_grid, recon_mat)
end

# ╔═╡ sa000041-0000-0000-0000-000000000001
md"### Real Data: Training Loss"

# ╔═╡ sa000042-0000-0000-0000-000000000001
let
    r      = real_result
    epochs = eachindex(r.lhist.train_recon)
    traces = [
        PlutoPlotly.scatter(x=collect(epochs), y=r.lhist.train_recon, mode="lines",
            name="Train", line=attr(color="#d62728", width=1.5)),
        PlutoPlotly.scatter(x=collect(epochs), y=r.lhist.test_recon, mode="lines",
            name="Test", line=attr(color="#d62728", dash="dash", width=1.5)),
    ]
    layout = Layout(
        title=attr(text="Real data recon loss"),
        xaxis=attr(title="Epoch"), yaxis=attr(title="MSE", type="log"),
        height=300, width=800, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ sa000043-0000-0000-0000-000000000001
md"### Real Data: Code Distribution"

# ╔═╡ sa000044-0000-0000-0000-000000000001
let
    r = real_result
    K = vqvae_params_real.K
    counts = [count(==(k), r.codes_star) for k in 1:K]
    pct    = counts ./ sum(counts) .* 100
    layout = Layout(
        title=attr(text="Code assignment distribution (real data)"),
        xaxis=attr(title="Code"), yaxis=attr(title="% waveforms"),
        height=280, width=500, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot([PlutoPlotly.bar(x=1:K, y=pct, marker=attr(color="#d62728"))], layout)
end

# ╔═╡ sa000045-0000-0000-0000-000000000001
md"### Real Data: Shift Distribution"

# ╔═╡ sa000046-0000-0000-0000-000000000001
let
    r   = real_result
    tau = r.tau_star
    bins    = range(minimum(tau)-0.5, maximum(tau)+0.5; length=41)
    counts  = [sum(bins[i] .<= tau .< bins[i+1]) for i in 1:length(bins)-1]
    centers = [(bins[i]+bins[i+1])/2 for i in 1:length(bins)-1]
    layout = Layout(
        title=attr(text="Best-shift distribution (real data)"),
        xaxis=attr(title="τ* (samples)"), yaxis=attr(title="Count"),
        height=280, width=600, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot([PlutoPlotly.bar(x=centers, y=counts, marker=attr(color="#d62728", opacity=0.75))], layout)
end

# ╔═╡ sa000047-0000-0000-0000-000000000001
md"### Real Data: Coherent Stack per Code"

# ╔═╡ sa000048-0000-0000-0000-000000000001
@bind code_sel_real Slider(1:vqvae_params_real.K, show_value=true)

# ╔═╡ sa000049-0000-0000-0000-000000000001
let
    r    = real_result
    k    = code_sel_real
    ts   = 1:nt_real
    stk  = get(r.stacks, k, nothing)
    if isnothing(stk)
        md"No waveforms assigned to code $k"
    else
        n_k = count(==(k), r.codes_star)
        # Raw mean for comparison
        sel = findall(==(k), r.codes_star)
        stack_raw = vec(mean(X_real_raw[:, sel]; dims=2))
        traces = [
            PlutoPlotly.scatter(x=ts, y=stack_raw, mode="lines",
                name="Raw mean (code $k, N=$n_k)",
                line=attr(color="grey", dash="dash", width=1.5)),
            PlutoPlotly.scatter(x=ts, y=stk, mode="lines",
                name="Aligned stack (code $k)",
                line=attr(color="#d62728", width=2)),
        ]
        layout = Layout(
            title=attr(text="Code $k: aligned stack vs raw mean  (N=$n_k waveforms)"),
            xaxis=attr(title="Sample"), yaxis=attr(title="Amplitude"),
            height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
        )
        PlutoPlotly.plot(traces, layout)
    end
end

# ╔═╡ sa000050-0000-0000-0000-000000000001
md"### Real Data: All Aligned Waveforms for Selected Code"

# ╔═╡ sa000051-0000-0000-0000-000000000001
let
    r   = real_result
    k   = code_sel_real
    sel = findall(==(k), r.codes_star)
    isempty(sel) && (md"No waveforms for code $k")
    nshow = min(30, length(sel))
    ts = 1:nt_real
    traces = [PlutoPlotly.scatter(x=ts, y=r.X_aligned[:, sel[i]], mode="lines",
        name="w$(sel[i])", line=attr(width=0.8, color="#1f77b4"), opacity=0.4)
        for i in 1:nshow]
    # overlay stack
    stk = r.stacks[k]
    push!(traces, PlutoPlotly.scatter(x=ts, y=stk, mode="lines",
        name="Stack", line=attr(color="black", width=2.5)))
    layout = Layout(
        title=attr(text="Code $k: aligned waveforms (first $nshow of $(length(sel)))"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Normalised amplitude"),
        height=400, width=950, plot_bgcolor="white", paper_bgcolor="white",
        showlegend=false,
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ sa000052-0000-0000-0000-000000000001
md"### Overall Aligned Stack (all codes)"

# ╔═╡ sa000053-0000-0000-0000-000000000001
let
    r  = real_result
    ts = 1:nt_real
    stack_all_aligned = vec(mean(r.X_aligned; dims=2))
    stack_raw         = vec(mean(X_real_raw;  dims=2))
    true_stack        = vec(mean(Float32.(true_rf); dims=2))
    traces = [
        PlutoPlotly.scatter(x=ts, y=stack_raw, mode="lines",
            name="Raw mean (unaligned)", line=attr(color="grey", dash="dash", width=1.5)),
        PlutoPlotly.scatter(x=ts, y=stack_all_aligned, mode="lines",
            name="Aligned stack (ShiftAug)", line=attr(color="#d62728", width=2)),
        PlutoPlotly.scatter(x=ts, y=true_stack, mode="lines",
            name="True RF stack", line=attr(color="black", dash="dot", width=1.5)),
    ]
    layout = Layout(
        title=attr(text="Overall coherent stack: aligned vs raw mean vs true RF"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Amplitude"),
        height=380, width=950, plot_bgcolor="white", paper_bgcolor="white",
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
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Zygote  = "e88e6eb3-aa80-5325-afca-941959d7151f"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised
julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "shiftaug_placeholder"
"""
