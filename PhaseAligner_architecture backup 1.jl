### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ pa000001-0000-0000-0000-000000000001
using CUDA,
    cuDNN,
    Flux,
    Zygote,
    Random,
    Statistics,
    LinearAlgebra,
    FFTW,
    Optimisers,
    ProgressLogging

# ╔═╡ pa000002-0000-0000-0000-000000000001
md"""# Phase Aligner — Design C (Siamese Scalar Phase)

## Architecture Overview

**Self-supervised waveform aligner** based on a siamese scalar-phase network.

### Key Idea

Each waveform `x_i` is mapped to a scalar **phase coordinate** `φ(x_i)` by a
convolutional encoder. The relative shift between any two waveforms is then:

```
τ_ab = φ(x_a) − φ(x_b)
```

Anti-symmetry is exact by construction. Alignment to a running EMA baseline `B`
is:

```
τ_i = φ(x_i) − φ_B
```

where `φ_B` is one scalar (the phase of the current baseline), not a full waveform.

### Training (two phases)

**Phase 1 — Equivariance pre-training** (no baseline needed)

For each waveform, sample a random shift `δ` and enforce:

```
φ(shift(x, δ)) ≈ φ(x) + δ
```

This is a self-supervised equivariance constraint that gives `φ` a precise
physical meaning (time coordinate in samples) before any baseline is needed.

**Phase 2 — EMA baseline alignment**

With a bootstrapped `φ_B` (mean phase from Phase 1), train:

```
L = mean_i( (φ(x_i) − φ_B)² ) + γ · mean_i( φ(x_i)² )
φ_B ← (1−α)·φ_B + α·mean_batch(φ(x_i))   [EMA, outside gradient]
```

### Final output

```
τ_i   = φ(x_i) − φ_B           (shift in samples)
x̃_i   = shift(x_i, −τ_i)       (aligned to baseline frame)
stack = mean_i(x̃_i)            (the coherent stack)
```

### Why no decoder?

This is alignment-only. We do not need to reconstruct or cluster — just find the
best-fit time shift for each waveform. The architecture is therefore much smaller
and faster than VQ-VAE-ST.
"""

# ╔═╡ pa000003-0000-0000-0000-000000000001
xpu = gpu

# ╔═╡ pa000004-0000-0000-0000-000000000001
const activation_pa = x -> leakyrelu(x, 0.1f0)

# ╔═╡ pa000005-0000-0000-0000-000000000001
md"## Parameters"

# ╔═╡ pa000006-0000-0000-0000-000000000001
Base.@kwdef struct PhaseAligner_Para
    nt::Int                                  # waveform length (samples)
    max_shift_samples::Int = 50              # hard clamp ±max_shift (tanh squash)
    enc_kernels::Vector{Int} = [32, 16, 8]   # conv kernel widths
    enc_filters::Vector{Int} = [8, 16, 32]  # conv channel counts
    gamma::Float32 = 0.001f0                 # L2 regularization on φ
    ema_decay::Float32 = 0.99f0              # EMA decay for running phase baseline
    seed::Int = 42
end

# ╔═╡ pa000007-0000-0000-0000-000000000001
md"## Fourier Shift (reused from VQVAE-ST)"

# ╔═╡ pa000008-0000-0000-0000-000000000001
begin
    """
    Differentiable Fourier time-shift applied to a batch of waveforms.

    - `x`:    `(nt, B)` Float32 matrix
    - `τ`:    `(1, B)` Float32 matrix — shifts in **samples** (sub-sample accurate)
    - `grid`: `(nt,)` complex vector `= -im * 2π * fftfreq(nt)`, precomputed on GPU

    Phase: `exp(grid * τ) = exp(-im * 2π * k/N * τ)` — standard DFT shift theorem.
    Wrapping-free as long as |τ| < N/2.
    """
    function shift_traces_Fourier(x::AbstractMatrix{Float32},
                                  τ::AbstractMatrix{Float32},
                                  grid::AbstractVector)
        x_fft = fft(x, 1)
        phase = exp.(grid .* τ)
        return real(ifft(x_fft .* phase, 1))
    end

    """Convenience: shift by a scalar (broadcasts to all columns)."""
    function shift_traces_Fourier(x::AbstractMatrix{Float32},
                                  τ_scalar::Float32,
                                  grid::AbstractVector)
        τ_mat = fill(τ_scalar, 1, size(x, 2)) |> xpu
        return shift_traces_Fourier(x, τ_mat, grid)
    end
end

# ╔═╡ pa000009-0000-0000-0000-000000000001
md"## Phase Encoder (`φ`)"

# ╔═╡ pa000010-0000-0000-0000-000000000001
begin
    """
    `PhaseEncoder` maps each waveform `(nt,)` to a scalar phase coordinate `φ ∈ ℝ`.

    Architecture:
    - Strided 1D convolutions with global average pooling
    - Final `Dense(C, 1; init=Flux.zeros32)` — initializes to φ=0 for all inputs
    - Squashed with `tanh * max_shift` to keep φ within ±max_shift_samples
    """
    struct PhaseEncoder{C}
        conv_chain::C
        max_shift::Float32
    end
    Flux.@layer PhaseEncoder trainable = (conv_chain,)

    function (m::PhaseEncoder)(x::AbstractMatrix{Float32})
        # x: (nt, B) → add channel dim → (nt, 1, B)
        x3 = reshape(x, size(x, 1), 1, size(x, 2))
        feat = m.conv_chain(x3)          # (C, B)
        return m.max_shift .* tanh.(feat) # (1, B)
    end

    """
    Build a `PhaseEncoder` from architecture parameters.
    Final Dense layer is zero-initialized so φ(x)=0 at the start of training.
    """
    function build_phase_encoder(nt::Int, max_shift::Int;
                                  kernels::Vector{Int}=[32, 16, 8],
                                  filters::Vector{Int}=[8, 16, 32])
        layers = Any[]
        nin = 1
        for (i, k) in enumerate(kernels)
            nout = filters[i]
            push!(layers, Conv((k,), nin => nout, activation_pa; pad=SamePad(), stride=2))
            push!(layers, BatchNorm(nout))
            nin = nout
        end
        # Global average pool: (L, C, B) → (C, B) by averaging over time
        push!(layers, x -> dropdims(mean(x; dims=1); dims=1))
        # Scalar output, zero-initialized
        push!(layers, Dense(nin, 1; init=Flux.zeros32))
        return PhaseEncoder(Chain(layers...), Float32(max_shift))
    end
end

# ╔═╡ pa000011-0000-0000-0000-000000000001
md"## EMA Phase Baseline"

# ╔═╡ pa000012-0000-0000-0000-000000000001
begin
    """
    `EMAPhaseBaseline` tracks a running EMA of the mean phase coordinate.

    Unlike an EMA over full waveforms (Design B), this stores only a single float:
    the mean φ value over all training waveforms seen so far. This is numerically
    stable and never changes the input distribution of the encoder.

    Update rule (outside gradient):
        φ_B ← (1−α)·φ_B + α·mean_batch(φ(x_i))
    """
    mutable struct EMAPhaseBaseline
        phi_baseline::Float32    # running EMA of mean phase (CPU scalar)
        decay::Float32
        initialized::Bool
    end

    EMAPhaseBaseline(; decay::Float32=0.99f0) =
        EMAPhaseBaseline(0f0, decay, false)

    """Update the baseline with the batch mean phase (call outside gradient)."""
    function update_baseline!(b::EMAPhaseBaseline, phi_batch_mean::Float32)
        if !b.initialized
            b.phi_baseline = phi_batch_mean
            b.initialized  = true
        else
            b.phi_baseline = b.decay * b.phi_baseline + (1f0 - b.decay) * phi_batch_mean
        end
        return nothing
    end

    """Initialize the baseline directly (e.g. from Phase 1 result)."""
    function init_baseline!(b::EMAPhaseBaseline, phi_mean::Float32)
        b.phi_baseline = phi_mean
        b.initialized  = true
    end
end

# ╔═╡ pa000013-0000-0000-0000-000000000001
md"## PhaseAligner Model"

# ╔═╡ pa000014-0000-0000-0000-000000000001
begin
    """
    Full PhaseAligner model: encoder + sampling grid.

    `model(x)` returns `φ(x)` — shape `(1, B)`, in samples.

    Use `predict_shifts(model, x, baseline)` to get per-waveform shifts
    relative to the current EMA baseline.
    """
    struct PhaseAligner{E, G}
        encoder::E        # PhaseEncoder: (nt, B) → (1, B)
        grid::G           # -im * 2π * fftfreq(nt) on GPU
    end
    Flux.@layer PhaseAligner trainable = (encoder,)

    function (m::PhaseAligner)(x::AbstractMatrix{Float32})
        return m.encoder(x)   # (1, B) in samples
    end

    """
    Predict shifts relative to the EMA baseline.
    Returns `(τ, x_aligned)` where τ = φ(x) − φ_B.
    """
    function predict_shifts(m::PhaseAligner, x::AbstractMatrix{Float32},
                             baseline::EMAPhaseBaseline)
        phi = m(x)                                        # (1, B)
        τ   = phi .- baseline.phi_baseline                # (1, B)
        x_aligned = shift_traces_Fourier(x, -τ, m.grid)  # (nt, B)
        return τ, x_aligned
    end

    """Compute the coherent stack over aligned waveforms."""
    function aligned_stack(m::PhaseAligner, x::AbstractMatrix{Float32},
                            baseline::EMAPhaseBaseline)
        _, x_aligned = predict_shifts(m, x, baseline)
        return dropdims(mean(x_aligned; dims=2); dims=2)  # (nt,)
    end
end

# ╔═╡ pa000015-0000-0000-0000-000000000001
md"## Loss Functions"

# ╔═╡ pa000016-0000-0000-0000-000000000001
begin
    """
    **Phase 1 loss — equivariance**

    For each waveform, draw a random shift `δ ~ Uniform(±max_shift)` and enforce:
        φ(shift(x, δ)) ≈ φ(x) + δ

    Gradient flows only through the encoder, not through δ or the shift operation.
    """
    function equivariance_loss(model::PhaseAligner, x::AbstractMatrix{Float32},
                                gamma::Float32, max_shift::Int)
        N   = size(x, 2)
        δ   = (2f0 .* CUDA.rand(Float32, 1, N) .- 1f0) .* Float32(max_shift)
        x_s = Zygote.@ignore shift_traces_Fourier(x, δ, model.grid)
        φ_x = model(x)                      # (1, N)
        φ_xs = model(x_s)                   # (1, N)
        δ_stop = Zygote.@ignore(δ)           # no gradient through δ
        equiv = mean(abs2, φ_xs .- φ_x .- δ_stop)
        reg   = gamma * mean(abs2, φ_x)
        return equiv + reg, equiv, reg
    end

    """
    **Phase 2 loss — EMA baseline alignment**

    Push all φ(x_i) toward the current baseline `φ_B`, plus light L2 regularization.
    The baseline is updated (EMA) outside the gradient via `Zygote.@ignore`.
    """
    function alignment_loss(model::PhaseAligner, x::AbstractMatrix{Float32},
                             baseline::EMAPhaseBaseline, gamma::Float32;
                             training::Bool=true)
        phi = model(x)                          # (1, N)
        phi_B = baseline.phi_baseline            # scalar Float32

        recon  = mean(abs2, phi .- phi_B)        # pull toward baseline
        reg    = gamma * mean(abs2, phi)          # prevent drift

        Zygote.@ignore if training
            batch_mean = mean(cpu(phi))
            update_baseline!(baseline, Float32(batch_mean))
        end

        return recon + reg, recon, reg
    end
end

# ╔═╡ pa000017-0000-0000-0000-000000000001
md"## Training Loop"

# ╔═╡ pa000018-0000-0000-0000-000000000001
Base.@kwdef struct PhaseAligner_Training_Para
    batchsize::Int            = 128
    nepoch_phase1::Int        = 100   # equivariance pre-training epochs
    nepoch_phase2::Int        = 300   # EMA baseline alignment epochs
    initial_lr_phase1::Float64 = 0.001
    initial_lr_phase2::Float64 = 0.0003
    lr_decay::Float64         = 0.995
    nprint::Int               = 10
end

# ╔═╡ pa000019-0000-0000-0000-000000000001
begin
    """Record one epoch's metrics onto the loss history."""
    function record_pa_metrics!(h, train_total, train_equiv, train_reg,
                                 test_total, test_equiv, test_reg)
        push!(h.train_total,   train_total)
        push!(h.train_equiv,   train_equiv)
        push!(h.train_reg,     train_reg)
        push!(h.test_total,    test_total)
        push!(h.test_equiv,    test_equiv)
        push!(h.test_reg,      test_reg)
    end

    """Log if on print interval."""
    function log_pa_epoch(epoch::Int, phase::Int,
                           train_total, test_total, nprint::Int)
        mod(epoch, nprint) == 0 || return nothing
        @info "Phase $phase  Epoch $epoch" train=round(train_total; digits=6) test=round(test_total; digits=6)
        return nothing
    end

    """One epoch of gradient steps (equivariance phase)."""
    function phase1_train_epoch!(model, opt_state, loader,
                                  gamma::Float32, max_shift::Int)
        for x in loader
            grads = Flux.gradient(model) do m
                equivariance_loss(m, xpu(x), gamma, max_shift)[1]
            end
            Optimisers.update!(opt_state, model, grads[1])
        end
        GC.gc(false)
        CUDA.reclaim()
        return nothing
    end

    """One epoch of gradient steps (baseline alignment phase)."""
    function phase2_train_epoch!(model, opt_state, loader,
                                  baseline::EMAPhaseBaseline, gamma::Float32)
        for x in loader
            grads = Flux.gradient(model) do m
                alignment_loss(m, xpu(x), baseline, gamma; training=true)[1]
            end
            Optimisers.update!(opt_state, model, grads[1])
        end
        GC.gc(false)
        CUDA.reclaim()
        return nothing
    end

    """Evaluate equivariance loss on a batch (no gradient)."""
    function eval_equiv(model, x, gamma, max_shift)
        total, equiv, reg = equivariance_loss(model, xpu(x), gamma, max_shift)
        return Float32(total), Float32(equiv), Float32(reg)
    end

    """Evaluate alignment loss on a batch (no gradient, no EMA update)."""
    function eval_align(model, x, baseline, gamma)
        total, recon, reg = alignment_loss(model, xpu(x), baseline, gamma; training=false)
        return Float32(total), Float32(recon), Float32(reg)
    end
end

# ╔═╡ pa000020-0000-0000-0000-000000000001
begin
    """
        train_phase_aligner(model, X_train, X_test, para, training_para)

    Two-phase training of a `PhaseAligner`:
    - Phase 1: equivariance loss (self-supervised, no baseline)
    - Phase 2: EMA baseline alignment

    Returns `(baseline, loss_history)`.
    """
    function train_phase_aligner(model::PhaseAligner,
                                  X_train::AbstractMatrix{Float32},
                                  X_test::AbstractMatrix{Float32},
                                  para::PhaseAligner_Para,
                                  training_para::PhaseAligner_Training_Para=PhaseAligner_Training_Para())
        bs = min(training_para.batchsize, size(X_train, 2))
        train_loader = Flux.DataLoader(X_train; batchsize=bs, shuffle=true)
        test_loader  = Flux.DataLoader(X_test;  batchsize=min(bs, size(X_test, 2)), shuffle=false)
        monitor_train = cpu(first(train_loader))
        monitor_test  = cpu(first(test_loader))

        h = (;
            train_total=Float32[], train_equiv=Float32[], train_reg=Float32[],
            test_total=Float32[],  test_equiv=Float32[],  test_reg=Float32[],
        )

        # ── Phase 1: equivariance ───────────────────────────────────────────
        opt1 = Optimisers.setup(Optimisers.Adam(eta=training_para.initial_lr_phase1), model)
        @progress name = "Phase 1 (equivariance)" for epoch in 1:training_para.nepoch_phase1
            tr = eval_equiv(model, monitor_train, para.gamma, para.max_shift_samples)
            te = eval_equiv(model, monitor_test,  para.gamma, para.max_shift_samples)
            record_pa_metrics!(h, tr..., te...)
            phase1_train_epoch!(model, opt1, train_loader, para.gamma, para.max_shift_samples)
            log_pa_epoch(epoch, 1, tr[1], te[1], training_para.nprint)
            Optimisers.adjust!(opt1; eta=training_para.initial_lr_phase1 * training_para.lr_decay^epoch)
        end

        # ── Bootstrap baseline from Phase 1 result ─────────────────────────
        baseline = EMAPhaseBaseline(decay=para.ema_decay)
        phi_all  = vec(cpu(model(xpu(X_train))))   # (N,)
        init_baseline!(baseline, mean(phi_all))
        @info "Phase 1 done" phi_mean=baseline.phi_baseline phi_std=std(phi_all)

        # ── Phase 2: EMA baseline alignment ────────────────────────────────
        opt2 = Optimisers.setup(Optimisers.Adam(eta=training_para.initial_lr_phase2), model)
        @progress name = "Phase 2 (alignment)" for epoch in 1:training_para.nepoch_phase2
            tr = eval_align(model, monitor_train, baseline, para.gamma)
            te = eval_align(model, monitor_test,  baseline, para.gamma)
            record_pa_metrics!(h, tr..., te...)
            phase2_train_epoch!(model, opt2, train_loader, baseline, para.gamma)
            log_pa_epoch(epoch, 2, tr[1], te[1], training_para.nprint)
            Optimisers.adjust!(opt2; eta=training_para.initial_lr_phase2 * training_para.lr_decay^epoch)
        end

        return baseline, h
    end
end

# ╔═╡ pa000021-0000-0000-0000-000000000001
md"## Model Factory"

# ╔═╡ pa000022-0000-0000-0000-000000000001
"""
    get_phase_aligner(para::PhaseAligner_Para) -> model::PhaseAligner

Build a `PhaseAligner` from parameters and move to GPU.
"""
function get_phase_aligner(para::PhaseAligner_Para)
    Random.seed!(para.seed)
    encoder = build_phase_encoder(para.nt, para.max_shift_samples;
                                   kernels=para.enc_kernels,
                                   filters=para.enc_filters) |> xpu
    grid = xpu(-im .* Float32.(fftfreq(para.nt) .* 2π))
    model = PhaseAligner(encoder, grid)
    @info "PhaseAligner built" nt=para.nt max_shift=para.max_shift_samples enc_filters=para.enc_filters
    return model
end

# ╔═╡ pa000023-0000-0000-0000-000000000001
md"## Post-training Utilities"

# ╔═╡ pa000024-0000-0000-0000-000000000001
begin
    """
    Run the full aligner over a dataset in mini-batches (avoids OOM).
    Returns `(tau_all, X_aligned_all)` on CPU.
    """
    function apply_aligner(model::PhaseAligner, X::AbstractMatrix{Float32},
                            baseline::EMAPhaseBaseline; batchsize::Int=256)
        N  = size(X, 2)
        tau_all     = zeros(Float32, N)
        X_aligned   = zeros(Float32, size(X, 1), N)
        loader = Flux.DataLoader(X; batchsize=batchsize, shuffle=false)
        offset = 0
        for x in loader
            nb = size(x, 2)
            τ, xa = predict_shifts(model, xpu(x), baseline)
            tau_all[offset+1:offset+nb]       .= vec(cpu(τ))
            X_aligned[:, offset+1:offset+nb]  .= cpu(xa)
            offset += nb
        end
        return tau_all, X_aligned
    end

    """Compute the coherent stack from a full aligned dataset."""
    function coherent_stack(X_aligned::AbstractMatrix{Float32})
        return dropdims(mean(X_aligned; dims=2); dims=2)
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA    = "052768ef-5323-5732-b1bb-66c8b64840ba"
cuDNN   = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
Flux    = "587475ba-b771-5e3f-ad9e-33799f191a9c"
Zygote  = "e88e6eb3-aa80-5325-afca-941959d7151f"
FFTW    = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random  = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised
julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "phasealigner_placeholder"
"""
