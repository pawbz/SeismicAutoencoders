### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ cc11647d-1c56-4ceb-9677-703aca03c9f4
using Functors

# ╔═╡ d73472ff-9e09-45b0-8811-b7dd8d820358
using CUDA,
    cuDNN,
    Enzyme,
    Flux,
    Zygote,
    Distances,
    JLD2,
    Random,
    MLUtils,
    DSP,
    ProgressLogging,
    Statistics,
    LinearAlgebra,
    PlutoUI,
    PlutoHooks,
    FFTW,
    StatsBase,
    Optimisers

# ╔═╡ 76dbf599-a9b3-459f-992b-16ab2f7b74f1
using PlutoLinks

# ╔═╡ 4a95997e-5c12-4658-9b8e-a5065328e1c1
using BenchmarkTools

# ╔═╡ a0000018-0000-0000-0000-000000000001
using ParameterSchedulers

# ╔═╡ a0000021-0000-0000-0000-000000000001
using PlutoPlotly

# ╔═╡ 461f0505-2230-4b84-b6c6-1a9730808437
md"""# VQ-VAE with Fourier Spatial Transformer

## Architecture Overview

**VQ-VAE + Fourier Spatial Transformer** for jointly learning optimal time-shift
alignment and discrete waveform clustering.

### Key Ideas
- **Learnable Fourier shifts**: a localization network predicts a scalar time-shift per
  waveform; shifts are applied via Fourier interpolation (fully differentiable)
- **Shift-then-quantize**: the shifted waveform enters the VQ-VAE; codebook entries
  become canonical (zero-shift) prototypes
- **Un-shift after decode**: reconstruction is shifted back so the loss is always
  computed against the original, unshifted waveform
- **Shift regularization** discourages spuriously large shifts (see `gamma` and
  `shift_penalty_type`)
- All other VQ-VAE features from v3.1 are preserved (EMA codebook, dead-entry reset,
  latent-time window, entropy bonus)

### Why joint shift + quantization?
For noisy waveforms without a good reference, cross-correlation is unreliable.
By learning shifts jointly with the codebook:
1. As the codebook forms, the shift network learns to align waveforms to the nearest
   prototype — a stable, evolving reference.
2. Gradients flow end-to-end through both the Fourier shift and the VQ-VAE encoder.
3. The codebook collapse pathology is reduced because aligned waveforms have smaller
   inter-waveform variance, making the commitment loss more informative.

### Shift Regularization Options (set `shift_penalty_type`)
| Type | Formula | Effect |
|------|---------|--------|
| `:l2` | `γ · mean(τ²)` | Soft penalty; large shifts still possible |
| `:l1` | `γ · mean(|τ|)` | Promotes exactly-zero shifts (sparse timing) |
| `:cauchy` | `γ · mean(log(1 + (τ/σ₀)²))` | Robust; tolerates a few large shifts |
| `:bounded` | L2 inside `±max_shift`, hard wall outside | Strict physical constraint |

**Recommendation**: start with `:l2` and a moderate `gamma ≈ 0.001`. Use `:cauchy`
if a small fraction of waveforms legitimately need large alignment corrections (e.g.
teleseismic arrivals mixed with local events). Set `max_shift_samples` to the maximum
physically plausible lag (e.g. `round(distance / v_min / dt)`).

### Architecture
```
x (nt,B) ──→ LocalizationNet ──→ τ (1,B)
      │                                │
      └──→ shift_Fourier(x, τ) ──→ x̃ (nt,B)
                                        │
                          VQ-VAE(x̃) ──→ x̃_hat (nt,B)
                                        │
                    shift_Fourier(x̃_hat, -τ) ──→ x̂ (nt,B)
                                        │
                       loss = MSE(x̂, x) + γ·penalty(τ)
```
"""

# ╔═╡ 97ae4222-5a3e-4cbd-b4d1-aa028d3e4ca8
TableOfContents(include_definitions=true)

# ╔═╡ 26fb86d5-c844-469a-aef5-ed3c2a9ba949
xpu = gpu

# ╔═╡ 6affb3b3-9dc4-4bbc-a582-495fc1783a7a
activation = x -> leakyrelu(x, 0.1f0)

# ╔═╡ a0000001-0000-0000-0000-000000000001
md"## Utilities"

# ╔═╡ 80f77b52-84e0-4664-8aa0-3d79fded40de
"""
Instead of cat(x, dims=3)
"""
add_dim3_reshape(::Nothing) = nothing

# ╔═╡ 6ba143e2-50df-441a-8f38-3ea8d9edd4d8
begin
    function add_dim3_reshape(x)
        nd = ndims(x)
        if nd == 2
            return reshape(x, size(x, 1), 1, size(x, 2))
        elseif nd == 3
            return x
        else
            return x
        end
    end

    function flatten_batch(x)
        return reshape(x, size(x, 1), :)
    end
end

# ╔═╡ a0000002-0000-0000-0000-000000000001
md"## Parameters"

# ╔═╡ 91a25156-e121-4d53-a5a1-422f1230d235
Base.@kwdef struct VQVAE_Para
    nt::Int                                # waveform length (time samples)
    d::Int = 64                            # codebook embedding dimension
    K::Int = 8                             # codebook size (number of entries)
    T::Int = 1                             # quantized vectors per waveform
    beta_commit::Float32 = 0.25f0          # commitment loss weight
    enc_kernels::Vector{Int} = [32, 16, 8, 4]
    enc_filters::Vector{Int} = [8, 16, 32, 64]
    enc_strides::Vector{Int} = [2, 2, 2, 2]
    dec_kernels::Vector{Int} = [4, 8, 16]
    dec_filters::Vector{Int} = [64, 48, 16, 1]
    use_bn::Bool = true
    ema_decay::Float32 = 0.99f0
    epsilon::Float32 = 1f-5
    dead_threshold::Int = 2
    entropy_weight::Float32 = 0.01f0
    # ── Physics-based latent window (optional) ────────────────────────────
    interstation_distance::Union{Nothing,Float64} = nothing  # km
    dt::Float64 = 1.0                                        # sampling interval (s)
    reference_velocity::Float64 = 3.0                        # km/s
    # ── Spatial transformer parameters ───────────────────────────────────
    gamma::Float32 = 0.001f0               # shift regularization weight
    max_shift_samples::Int = 50            # hard clamp ±max_shift_samples
    shift_penalty_type::Symbol = :l2       # :l1 | :l2 | :cauchy | :bounded
    cauchy_sigma::Float32 = 10f0           # reference scale for :cauchy penalty (samples)
    seed = nothing
end

# ╔═╡ b1c2d3e4-f5a6-7890-abcd-ef1234567890
"""
    compute_latent_window(para) -> (latent_time_index, latent_time_window)

Compute the latent-space window from interstation distance and reference velocity.
Returns `(nothing, 1)` when `interstation_distance === nothing`.
"""
function compute_latent_window(para)
    para.interstation_distance === nothing && return (nothing, 1)
    total_stride = prod(para.enc_strides)
    t_center = para.interstation_distance / para.reference_velocity / para.dt
    t_slow   = para.interstation_distance / (para.reference_velocity * 0.7) / para.dt
    t_fast   = para.interstation_distance / (para.reference_velocity * 1.3) / para.dt
    lti = round(Int, t_center / total_stride) + 1
    ltw = max(1, round(Int, (t_slow - t_fast) / total_stride))
    return (lti, ltw)
end

# ╔═╡ a0000003-0000-0000-0000-000000000001
md"## Conv Encoder / Decoder"

# ╔═╡ 89599b3f-8c20-46c5-8f5c-ccbb71b26b36
begin
    struct ReshapeLayer
        dims::Tuple
    end
    Flux.@layer ReshapeLayer trainable = ()
    (m::ReshapeLayer)(x) = reshape(x, m.dims..., size(x)[end])

    struct Conv1DChain
        chain::Chain
    end
    Flux.@layer Conv1DChain trainable = (chain,)

    function (m::Conv1DChain)(::Nothing)
        return nothing
    end

    function (m::Conv1DChain)(x)
        x_flat = flatten_batch(x)
        x3 = add_dim3_reshape(x_flat)
        features = m.chain(x3)
        if ndims(features) == 3 && size(features, 2) == 1
            return reshape(features, size(features, 1), size(features, 3))
        elseif ndims(features) == 3
            return features
        end
        return reshape(features, size(features, 1), size(x_flat, 2))
    end

    struct SeqConv1DChain
        chain::Chain
    end
    Flux.@layer SeqConv1DChain trainable = (chain,)

    function (m::SeqConv1DChain)(::Nothing)
        return nothing
    end

    function (m::SeqConv1DChain)(x)
        x3 = ndims(x) == 3 ? x : add_dim3_reshape(flatten_batch(x))
        y = m.chain(x3)
        if ndims(y) == 3 && size(y, 2) == 1
            return reshape(y, size(y, 1), size(y, 3))
        end
        return y
    end
end

# ╔═╡ 2f7550d1-e854-4c2f-8efb-ad0bb70d5013
function get_vq_conv_encoder(nt; kernels=[32, 16, 8, 4], filters=[8, 16, 32, 64],
    strides=[2, 2, 2, 2], use_bn::Bool=true,
    flatten_output::Bool=true, return_outsize::Bool=false)
    @assert length(kernels) == length(filters)
    layers = Any[]
    nin = 1
    for (i, k) in enumerate(kernels)
        nout = filters[i]
        s = i <= length(strides) ? strides[i] : 1
        push!(layers, Conv((k,), nin => nout, activation; pad=SamePad(), stride=s))
        if use_bn && i < length(kernels)
            push!(layers, BatchNorm(nout))
        end
        nin = nout
    end
    trunk = Chain(layers...)
    outsize = Flux.outputsize(trunk, (nt, 1); padbatch=true)
    flat_len = prod(outsize)
    if flatten_output
        push!(layers, Flux.flatten)
        enc = Conv1DChain(Chain(layers...))
    else
        enc = SeqConv1DChain(trunk)
    end
    return return_outsize ? (enc, flat_len, outsize) : (enc, flat_len)
end

# ╔═╡ 44e9c4cc-d02b-4e68-ad49-24f173556cbd
function infer_dec_upstrides(enc_strides::AbstractVector{<:Integer}, n_dec_layers::Int)
    vals = reverse(Int[s for s in enc_strides if s > 1])
    isempty(vals) && (vals = [1])
    while length(vals) > n_dec_layers
        vals[2] *= vals[1]
        deleteat!(vals, 1)
    end
    while length(vals) < n_dec_layers
        push!(vals, 1)
    end
    return vals
end

# ╔═╡ 64430447-c267-4eec-8d38-63ccf91d82c4
function get_vq_conv_decoder(nt, d_in; kernels=[4, 8, 16], filters=[64, 48, 16, 1],
    upstrides=[2, 2, 1], use_bn=false)
    @assert length(kernels) == length(filters) - 1
    @assert length(upstrides) == length(kernels)

    bottleneck_len = nt
    for s in upstrides
        bottleneck_len = cld(bottleneck_len, s)
    end
    bottleneck_channels = filters[1]

    layers = Any[]
    push!(layers, Dense(d_in, bottleneck_len * bottleneck_channels, activation))
    push!(layers, ReshapeLayer((bottleneck_len, bottleneck_channels)))

    nin = bottleneck_channels
    for (i, k) in enumerate(kernels)
        nout = filters[i+1]
        s = upstrides[i]
        if i < length(kernels)
            push!(layers, ConvTranspose((k,), nin => nout, activation; stride=s, pad=SamePad()))
            if use_bn
                push!(layers, BatchNorm(nout))
            end
        else
            push!(layers, ConvTranspose((k,), nin => nout; stride=s, pad=SamePad()))
        end
        nin = nout
    end

    return Conv1DChain(Chain(layers...)), bottleneck_len, bottleneck_channels
end

# ╔═╡ 8eb0be68-99e1-4df9-97bf-b29b99d8f759
function auto_dec_upstrides_for_nt(nt::Int, latent_len::Int, d::Int;
    dec_kernels, dec_filters, use_bn, enc_strides)
    n_dec = length(dec_kernels)
    base = infer_dec_upstrides(enc_strides, n_dec)

    function outlen(strides)
        dec, _, _ = get_vq_conv_decoder(nt, d;
            kernels=collect(dec_kernels), filters=collect(dec_filters),
            upstrides=strides, use_bn=use_bn)
        Flux.outputsize(dec.chain, (d,); padbatch=true)[1]
    end

    outlen(base) == nt && return base

    kmax = collect(Int, dec_kernels)
    lo = max.(1, base .- 3)
    hi = min.(kmax, base .+ 3)
    best = copy(base)
    best_err = abs(outlen(base) - nt)
    for cand in Iterators.product([lo[i]:hi[i] for i in 1:n_dec]...)
        s = collect(Int, cand)
        any(s .> kmax) && continue
        olen = outlen(s)
        err = abs(olen - nt)
        if err < best_err; best = s; best_err = err; end
        err == 0 && return s
    end
    return best
end

# ╔═╡ a0000006-0000-0000-0000-000000000001
md"## Vector Quantizer (EMA)"

# ╔═╡ a0000007-0000-0000-0000-000000000001
begin
    mutable struct VectorQuantizerEMA
        K::Int
        d::Int
        embedding::AbstractMatrix{Float32}
        ema_cluster_size::AbstractVector{Float32}
        ema_dw::AbstractMatrix{Float32}
        decay::Float32
        epsilon::Float32
        dead_count::Vector{Int}
        dead_threshold::Int
    end
    Flux.@layer VectorQuantizerEMA trainable = ()

    function VectorQuantizerEMA(K::Int, d::Int;
        decay::Float32=0.99f0, epsilon::Float32=1f-5, dead_threshold::Int=2)
        embedding = xpu(randn(Float32, d, K) .* (1f0 / K))
        ema_cluster_size = xpu(ones(Float32, K))
        ema_dw = copy(embedding)
        dead_count = zeros(Int, K)
        return VectorQuantizerEMA(K, d, embedding, ema_cluster_size, ema_dw,
            decay, epsilon, dead_count, dead_threshold)
    end

    function (vq::VectorQuantizerEMA)(z_e::AbstractMatrix{Float32};
        beta_commit::Float32=0.25f0, training::Bool=true)
        d, N = size(z_e)
        @assert d == vq.d

        z_q_detached, indices, encodings, vq_loss_val, perplexity, entropy_loss =
            Zygote.@ignore begin
                z_sq = sum(abs2, z_e; dims=1)
                e_sq = sum(abs2, vq.embedding; dims=1)
                dist = e_sq' .+ z_sq .- 2f0 .* (vq.embedding' * z_e)

                indices_cart = dropdims(CUDA.argmin(dist; dims=1); dims=1)
                indices = getindex.(indices_cart, 1)
                encodings = xpu(Float32.(Flux.onehotbatch(cpu(indices), 1:vq.K)))
                z_q = vq.embedding * encodings

                if training
                    enc_sum_vec = vec(sum(encodings; dims=2))
                    vq.ema_cluster_size .= vq.decay .* vq.ema_cluster_size .+
                                          (1f0 - vq.decay) .* enc_sum_vec
                    n = sum(vq.ema_cluster_size)
                    vq.ema_cluster_size .= (vq.ema_cluster_size .+ vq.epsilon) ./
                                           (n .+ Float32(vq.K) .* vq.epsilon) .* n
                    dw = z_e * encodings'
                    vq.ema_dw .= vq.decay .* vq.ema_dw .+ (1f0 - vq.decay) .* dw
                    vq.embedding .= vq.ema_dw ./ reshape(vq.ema_cluster_size, 1, :)

                    counts_cpu = cpu(enc_sum_vec)
                    dead_mask_cpu = counts_cpu .< 0.5f0
                    vq.dead_count .= ifelse.(dead_mask_cpu, vq.dead_count .+ 1, 0)
                    reset_mask_cpu = vq.dead_count .>= vq.dead_threshold
                    n_reset = sum(reset_mask_cpu)
                    if n_reset > 0
                        reset_idxs = findall(reset_mask_cpu)
                        donor_js = rand(1:N, n_reset)
                        donor_cols = z_e[:, donor_js] .+
                                     xpu(randn(Float32, d, n_reset) .* 0.01f0)
                        emb_cpu = cpu(vq.embedding); dw_cpu = cpu(vq.ema_dw)
                        cs_cpu = cpu(vq.ema_cluster_size)
                        donor_cpu = cpu(donor_cols)
                        for (i, k) in enumerate(reset_idxs)
                            emb_cpu[:, k] .= donor_cpu[:, i]
                            dw_cpu[:, k]  .= donor_cpu[:, i]
                            cs_cpu[k] = 1f0
                        end
                        copyto!(vq.embedding, xpu(emb_cpu))
                        copyto!(vq.ema_dw, xpu(dw_cpu))
                        copyto!(vq.ema_cluster_size, xpu(cs_cpu))
                        vq.dead_count[reset_idxs] .= 0
                    end
                end

                vq_loss_val = Flux.mse(z_e, z_q)
                avg_probs = mean(cpu(encodings); dims=2) |> vec
                avg_probs_safe = clamp.(avg_probs, 1f-10, 1f0)
                perplexity = exp(-sum(avg_probs_safe .* log.(avg_probs_safe)))
                entropy_loss = sum(avg_probs_safe .* log.(avg_probs_safe))
                (z_q, indices, encodings, vq_loss_val, perplexity, entropy_loss)
            end

        st_residual = Zygote.@ignore(z_q_detached .- z_e)
        z_q_st = z_e .+ st_residual
        commit_loss = beta_commit * Flux.mse(z_e, Zygote.@ignore(z_q_detached))

        return (; z_q=z_q_st, indices, encodings, vq_loss=vq_loss_val,
            commit_loss, perplexity, entropy_loss)
    end
end

# ╔═╡ a0000008-0000-0000-0000-000000000001
md"## Fourier Spatial Transformer"

# ╔═╡ st_cell_fourier
md"""
### Fourier shift

`shift_traces_Fourier(x, τ, grid)` applies sub-sample, differentiable time shifts
to a batch of waveforms via Fourier interpolation.

- `x`:    `(nt, batch)` — waveforms
- `τ`:    `(1, batch)`  — shifts in **samples** (real-valued, sub-sample accurate)
- `grid`: `(nt,)` complex vector `im * 2π * fftfreq(nt) * nt` (precomputed, on GPU)
"""

# ╔═╡ st_cell_impl
begin
    """
    Differentiable Fourier time-shift.
    τ is in samples (real-valued). grid = im * 2π * fftfreq(nt) * nt (complex, GPU).
    """
    function shift_traces_Fourier(x::AbstractMatrix{Float32},
                                  τ::AbstractMatrix{Float32},
                                  grid::AbstractVector)
        x_fft = fft(x, 1)                        # (nt, B)  complex
        # phase vector: exp(i·2π·f·τ) for each (f, b)
        phase = exp.(grid .* τ)                   # (nt, B)  complex
        x_shifted_fft = x_fft .* phase
        return real(ifft(x_shifted_fft, 1))       # (nt, B)
    end

    """
    Localization network: waveform (nt, B) → scalar shift τ (1, B) in samples.
    Uses a lightweight conv network followed by a global average pool + Dense.
    Output is tanh-scaled to ±max_shift_samples.
    """
    struct LocalizationNet{C}
        chain::C
        max_shift::Float32
    end
    Flux.@layer LocalizationNet trainable = (chain,)

    function (m::LocalizationNet)(x::AbstractMatrix{Float32})
        # x: (nt, B)
        x3 = reshape(x, size(x, 1), 1, size(x, 2))  # (nt, 1, B)
        h = m.chain(x3)                               # (1, 1, B) or (1, B)
        h_flat = reshape(h, 1, size(x, 2))            # (1, B)
        return m.max_shift .* tanh.(h_flat)           # (1, B)  in samples
    end

    """
    Build localization network: (nt, 1, B) → scalar (1, B).
    Architecture: light conv encoder → global average pool → Dense(1).
    """
    function build_localization_net(nt::Int, max_shift::Int;
        kernels=[32, 16, 8], filters=[8, 16, 32])
        layers = Any[]
        nin = 1
        for (i, k) in enumerate(kernels)
            nout = filters[i]
            push!(layers, Conv((k,), nin => nout, activation; pad=SamePad(), stride=2))
            push!(layers, BatchNorm(nout))
            nin = nout
        end
        # Global average pool over time → (1, C, B) → flatten → Dense
        push!(layers, x -> dropdims(mean(x; dims=1); dims=1))  # (C, B)
        push!(layers, Dense(nin, 1))                            # (1, B)
        return LocalizationNet(Chain(layers...), Float32(max_shift))
    end
end

# ╔═╡ a0000009-0000-0000-0000-000000000001
md"## VQ-VAE Model (core, without transformer)"

# ╔═╡ vqvae_core_cell
begin
    struct VQVAE{E,P,VQ,D}
        encoder::E
        pre_vq::P
        quantizer::VQ
        decoder::D
        T::Int
        d::Int
        latent_time_index::Union{Nothing,Int}
        latent_time_window::Int
    end
    Flux.@layer VQVAE trainable = (encoder, pre_vq, decoder)

    codebook_size(m) = m.quantizer.K

    function select_latent_time_window(z_map::AbstractArray{Float32,3},
        latent_time_index::Int, latent_time_window::Int)
        L, C, B = size(z_map)
        latent_time_index < 1 || latent_time_index > L &&
            error("latent_time_index=$latent_time_index out of range $L")
        latent_time_window < 1 &&
            error("latent_time_window must be ≥ 1")
        left  = fld(latent_time_window - 1, 2)
        right = latent_time_window - 1 - left
        idxs = clamp.(collect(latent_time_index-left:latent_time_index+right), 1, L)
        return z_map[idxs, :, :]
    end

    function dense_slot_latents(m, feat)
        z_pre = m.pre_vq(feat)
        ndims(z_pre) != 2 && error("Dense path: expected 2D, got ndims=$(ndims(z_pre))")
        N_total = size(z_pre, 2)
        return reshape(z_pre, m.d, m.T, N_total), z_pre
    end

    function latent_time_window_slot_latents(m, feat)
        ndims(feat) != 3 && error("Latent-time path: expected 3D, got ndims=$(ndims(feat))")
        z_window = select_latent_time_window(feat, m.latent_time_index, m.latent_time_window)
        z_window_flat = reshape(z_window, :, size(z_window, 3))
        z_pre = m.pre_vq(z_window_flat)
        ndims(z_pre) != 2 && error("Pre-VQ expected 2D, got ndims=$(ndims(z_pre))")
        N_total = size(z_pre, 2)
        return reshape(z_pre, m.d, m.T, N_total), z_pre, feat
    end

    function decode_from_latents(m, result)
        N_total = size(result.z_q, 3)
        z_q_for_dec = reshape(result.z_q, m.d * m.T, N_total)
        xhat = m.decoder(z_q_for_dec)
        return merge(result, (; xhat))
    end

    function (m::VQVAE)(x; beta_commit::Float32=0.25f0, training::Bool=true)
        return decode_from_latents(m, encode(m, x; beta_commit, training))
    end

    function encode(m::VQVAE, x; beta_commit::Float32=0.25f0, training::Bool=false)
        x_flat = flatten_batch(x)
        feat_map = m.encoder(x_flat)
        if m.latent_time_index === nothing
            slot_latents, z_pre_flat = dense_slot_latents(m, feat_map)
            z_map = nothing
        else
            slot_latents, z_pre_flat, z_map = latent_time_window_slot_latents(m, feat_map)
        end
        N_total = size(slot_latents, 3)
        z_e = reshape(slot_latents, m.d, m.T * N_total)
        rt = m.quantizer(z_e; beta_commit, training)
        codebook_indices = training ? nothing : reshape(Int.(cpu(rt.indices)), m.T, N_total)
        z_q = reshape(rt.z_q, m.d, m.T, N_total)
        return (; z_map, z_pre_flat,
            z_e=reshape(z_e, m.d, m.T, N_total), z_q,
            z_q_flat=rt.z_q, codebook_indices,
            vq_loss=rt.vq_loss, commit_loss=rt.commit_loss,
            perplexity=rt.perplexity, entropy_loss=rt.entropy_loss)
    end

    get_codebook(m::VQVAE) = cpu(m.quantizer.embedding)
end

# ╔═╡ a0000008b-0000-0000-0000-00000000000b
md"## VQ-VAE + Spatial Transformer (wrapped model)"

# ╔═╡ vqvae_st_cell
begin
    """
    VQ-VAE wrapped with a Fourier spatial transformer.

    Forward pass:
      1. LocalizationNet(x) → τ   (scalar shift per waveform, in samples)
      2. shift_Fourier(x, τ)  → x̃  (aligned waveform)
      3. VQVAE(x̃)             → x̃_hat (reconstruction of aligned waveform)
      4. shift_Fourier(x̃_hat, -τ) → x̂   (reconstruction of original waveform)
      5. loss = MSE(x̂, x) + γ·shift_penalty(τ)

    The codebook entries are therefore prototypes in the *zero-shift* canonical frame.
    """
    struct VQVAE_ST{VQ, LN, G}
        vqvae::VQ               # inner VQVAE (no spatial transformer)
        locnet::LN              # LocalizationNet: x → τ
        sampling_grid::G        # im * 2π * fftfreq(nt) * nt, on GPU (not trainable)
    end
    Flux.@layer VQVAE_ST trainable = (vqvae, locnet)

    # Forward: full encode+decode with shift
    function (m::VQVAE_ST)(x; beta_commit::Float32=0.25f0, training::Bool=true)
        x_flat = flatten_batch(x)
        τ = m.locnet(x_flat)                                  # (1, B) in samples
        x_shifted = shift_traces_Fourier(x_flat, τ, m.sampling_grid)  # (nt, B)
        vq_result = m.vqvae(x_shifted; beta_commit, training)
        xhat_shifted = vq_result.xhat                         # (nt, B)
        xhat = shift_traces_Fourier(xhat_shifted, -τ, m.sampling_grid)  # (nt, B)
        return merge(vq_result, (; xhat, xhat_shifted, shifts=τ))
    end

    # Encode-only (for inference / codebook analysis):
    # shifts are predicted and applied, but reconstruction is NOT unshifted
    function encode(m::VQVAE_ST, x; beta_commit::Float32=0.25f0, training::Bool=false)
        x_flat = flatten_batch(x)
        τ = m.locnet(x_flat)
        x_shifted = shift_traces_Fourier(x_flat, τ, m.sampling_grid)
        result = encode(m.vqvae, x_shifted; beta_commit, training)
        return merge(result, (; shifts=τ))
    end

    get_codebook(m::VQVAE_ST) = get_codebook(m.vqvae)
    codebook_size(m::VQVAE_ST) = codebook_size(m.vqvae)

    # Delegate T and d to the inner vqvae
    Base.getproperty(m::VQVAE_ST, s::Symbol) =
        s in (:vqvae, :locnet, :sampling_grid) ? getfield(m, s) : getproperty(m.vqvae, s)
end

# ╔═╡ a0000010-0000-0000-0000-000000000001
md"## Model Factory"

# ╔═╡ a0000011-0000-0000-0000-000000000001
"""
    get_vqvae(para::VQVAE_Para)

Build a `VQVAE_ST` (VQ-VAE with Fourier spatial transformer) from parameters.
Returns `(model, loss_history)`.
"""
function get_vqvae(para)
    para.seed !== nothing && Random.seed!(para.seed)

    latent_time_index, latent_time_window = compute_latent_window(para)

    if latent_time_index === nothing
        encoder, flat_len, enc_outsize = get_vq_conv_encoder(para.nt;
            kernels=para.enc_kernels, filters=para.enc_filters,
            strides=para.enc_strides, use_bn=para.use_bn,
            flatten_output=true, return_outsize=true)
        latent_len = enc_outsize[1]
        pre_vq = Dense(flat_len, para.d * para.T) |> xpu
    else
        encoder, flat_len, enc_outsize = get_vq_conv_encoder(para.nt;
            kernels=para.enc_kernels, filters=para.enc_filters,
            strides=para.enc_strides, use_bn=para.use_bn,
            flatten_output=false, return_outsize=true)
        latent_len = enc_outsize[1]
        latent_time_index > latent_len &&
            error("latent_time_index=$latent_time_index > latent_len=$latent_len")
        enc_channels = enc_outsize[2]
        pre_vq = Dense(latent_time_window * enc_channels, para.d * para.T) |> xpu
    end

    quantizer = VectorQuantizerEMA(para.K, para.d;
        decay=para.ema_decay, epsilon=para.epsilon,
        dead_threshold=para.dead_threshold)

    d_in_dec = para.d * para.T
    dec_upstrides = auto_dec_upstrides_for_nt(para.nt, latent_len, para.d;
        dec_kernels=para.dec_kernels, dec_filters=para.dec_filters,
        use_bn=para.use_bn, enc_strides=para.enc_strides)
    decoder, _, _ = get_vq_conv_decoder(para.nt, d_in_dec;
        kernels=para.dec_kernels, filters=para.dec_filters,
        upstrides=dec_upstrides, use_bn=para.use_bn)

    dec_out_len = Flux.outputsize(decoder.chain, (d_in_dec,); padbatch=true)[1]
    dec_out_len != para.nt &&
        error("Decoder output $dec_out_len ≠ nt $(para.nt). Adjust strides/kernels.")

    inner_vqvae = VQVAE(xpu(encoder), xpu(pre_vq), quantizer, xpu(decoder),
        para.T, para.d, latent_time_index, latent_time_window)

    # Localization network
    locnet = build_localization_net(para.nt, para.max_shift_samples) |> xpu

    # Sampling grid: im * 2π * fftfreq(nt) * nt  (complex, on GPU)
    sampling_grid = xpu(im .* Float32.(fftfreq(para.nt) .* (2π * para.nt)))

    model = VQVAE_ST(inner_vqvae, locnet, sampling_grid)

    @info "VQVAE_ST geometry" nt=para.nt latent_len enc_outsize dec_upstrides dec_out_len \
        max_shift_samples=para.max_shift_samples latent_time_index latent_time_window

    loss_history = (;
        train_recon=Float32[],
        test_recon=Float32[],
        train_commit=Float32[],
        test_commit=Float32[],
        train_total=Float32[],
        test_total=Float32[],
        train_perplexity=Float32[],
        test_perplexity=Float32[],
        train_shift_penalty=Float32[],
        test_shift_penalty=Float32[],
        train_mean_shift=Float32[],
        test_mean_shift=Float32[],
    )

    return model, loss_history
end

# ╔═╡ a0000012-0000-0000-0000-000000000001
md"## Loss Functions"

# ╔═╡ st_shift_penalty_cell
"""
    shift_penalty(τ, para) -> scalar Float32

Compute shift regularization penalty from shifts `τ` (1, B).

| `shift_penalty_type` | Formula |
|---|---|
| `:l2`     | `mean(τ²)` |
| `:l1`     | `mean(|τ|)` |
| `:cauchy` | `mean(log(1 + (τ/σ₀)²))` |
| `:bounded`| L2 inside ±max_shift, strong quadratic wall outside |
"""
function shift_penalty(τ::AbstractArray{Float32}, para)
    t = para.shift_penalty_type
    if t == :l2
        return mean(abs2, τ)
    elseif t == :l1
        return mean(abs, τ)
    elseif t == :cauchy
        σ = para.cauchy_sigma
        return mean(@. log(1f0 + (τ / σ)^2))
    elseif t == :bounded
        M = Float32(para.max_shift_samples)
        inside  = abs.(τ) .<= M
        penalty_inside  = mean(abs2, τ .* inside)
        penalty_outside = mean(@. ifelse(!inside, (abs(τ) - M)^2 * 100f0, 0f0))
        return penalty_inside + penalty_outside
    else
        error("Unknown shift_penalty_type: $t. Choose :l1, :l2, :cauchy, or :bounded.")
    end
end

# ╔═╡ a0000013-0000-0000-0000-000000000001
"""
    loss_vqvae(model, x, para; training)

VQ-VAE loss with shift regularization:

    L = L_recon + L_commit + β_entropy·H + γ·shift_penalty(τ)
"""
function loss_vqvae(model, x, para; training::Bool=true)
    x_flat = xpu(x)
    result = model(x_flat; beta_commit=para.beta_commit, training)

    recon_loss   = Flux.mse(result.xhat, x_flat)
    commit_loss  = result.commit_loss
    entropy_loss = result.entropy_loss
    sp           = shift_penalty(result.shifts, para)

    total = recon_loss + commit_loss +
            para.entropy_weight * entropy_loss +
            para.gamma * sp

    mean_shift = mean(abs, cpu(result.shifts))

    return (; total, recon_loss, commit_loss, entropy_loss,
        shift_penalty=sp, mean_shift,
        perplexity=result.perplexity)
end

# ╔═╡ a0000015-0000-0000-0000-000000000001
md"## Training Loop"

# ╔═╡ eafe181e-19e9-409e-ad1d-ce859cf0e672
Base.@kwdef struct VQVAE_Training_Para
    batchsize::Int = 100
    nepoch::Int = 30
    nprint::Int = 1
    initial_learning_rate::Float64 = 0.001
    lr_decay::Float64 = 0.99
    stop_on_recon_loss::Union{Nothing,Float64} = nothing
end

# ╔═╡ a0000019-0000-0000-0000-000000000001
function update(model, loss_history, D_train, D_test, para,
    training_para=VQVAE_Training_Para())

    opt_state = Optimisers.setup(
        Optimisers.Adam(eta=Float64(training_para.initial_learning_rate)), model)

    bs   = min(training_para.batchsize, size(D_train, 2))
    bs_t = min(training_para.batchsize, size(D_test, 2))
    train_loader = Flux.DataLoader(D_train; batchsize=bs, shuffle=true)
    test_loader  = Flux.DataLoader(D_test;  batchsize=bs_t, shuffle=false)
    monitor_train = first(train_loader)
    monitor_test  = first(test_loader)

    @progress name="VQ-VAE ST training" for epoch = 1:training_para.nepoch
        train_m = loss_vqvae(model, monitor_train, para; training=false)
        test_m  = loss_vqvae(model, monitor_test,  para; training=false)

        push!(loss_history.train_recon,         train_m.recon_loss)
        push!(loss_history.test_recon,          test_m.recon_loss)
        push!(loss_history.train_commit,        train_m.commit_loss)
        push!(loss_history.test_commit,         test_m.commit_loss)
        push!(loss_history.train_total,         train_m.total)
        push!(loss_history.test_total,          test_m.total)
        push!(loss_history.train_perplexity,    train_m.perplexity)
        push!(loss_history.test_perplexity,     test_m.perplexity)
        push!(loss_history.train_shift_penalty, train_m.shift_penalty)
        push!(loss_history.test_shift_penalty,  test_m.shift_penalty)
        push!(loss_history.train_mean_shift,    train_m.mean_shift)
        push!(loss_history.test_mean_shift,     test_m.mean_shift)

        function train_loss(model, x)
            return loss_vqvae(model, x, para; training=true).total
        end
        for x in train_loader
            g = Flux.gradient(train_loss, model, x)[1]
            Optimisers.update!(opt_state, model, g)
        end

        if mod(epoch, training_para.nprint) == 0
            @info "Epoch $epoch" recon=train_m.recon_loss commit=train_m.commit_loss \
                perplexity=train_m.perplexity mean_shift=train_m.mean_shift \
                shift_penalty=train_m.shift_penalty test_recon=test_m.recon_loss
        end
        if !isnothing(training_para.stop_on_recon_loss) &&
            train_m.recon_loss < training_para.stop_on_recon_loss
            @info "Early stop at epoch $epoch (recon=$(train_m.recon_loss))"
            break
        end
    end
    return nothing
end

# ╔═╡ a0000020-0000-0000-0000-000000000001
md"## Plotting"

# ╔═╡ a0000022-0000-0000-0000-000000000001
"""
    plot_training_dashboard(loss_history; title)

4-panel dashboard: recon loss, perplexity, commitment loss, mean shift.
"""
function plot_training_dashboard(loss_history; title="VQ-VAE ST Training")
    epochs = collect(1:length(loss_history.train_recon))
    font_spec = attr(family="Computer Modern, Latin Modern Math, serif")
    gc = "rgba(128,128,128,0.2)"

    traces = [
        PlutoPlotly.scatter(x=epochs, y=loss_history.train_recon, mode="lines",
            name="Train recon", xaxis="x", yaxis="y",
            line=attr(color="#1f77b4", width=1.5)),
        PlutoPlotly.scatter(x=epochs, y=loss_history.test_recon, mode="lines",
            name="Test recon", xaxis="x", yaxis="y",
            line=attr(color="#1f77b4", width=1.5, dash="dash")),
        PlutoPlotly.scatter(x=epochs, y=loss_history.train_perplexity, mode="lines",
            name="Train perplexity", xaxis="x", yaxis="y2",
            line=attr(color="#2ca02c", width=1.5)),
        PlutoPlotly.scatter(x=epochs, y=loss_history.test_perplexity, mode="lines",
            name="Test perplexity", xaxis="x", yaxis="y2",
            line=attr(color="#2ca02c", width=1.5, dash="dash")),
        PlutoPlotly.scatter(x=epochs, y=loss_history.train_commit, mode="lines",
            name="Train commit", xaxis="x", yaxis="y3",
            line=attr(color="#d62728", width=1.5)),
        PlutoPlotly.scatter(x=epochs, y=loss_history.test_commit, mode="lines",
            name="Test commit", xaxis="x", yaxis="y3",
            line=attr(color="#d62728", width=1.5, dash="dash")),
        PlutoPlotly.scatter(x=epochs, y=loss_history.train_mean_shift, mode="lines",
            name="Train |shift|", xaxis="x", yaxis="y4",
            line=attr(color="#ff7f0e", width=1.5)),
        PlutoPlotly.scatter(x=epochs, y=loss_history.test_mean_shift, mode="lines",
            name="Test |shift|", xaxis="x", yaxis="y4",
            line=attr(color="#ff7f0e", width=1.5, dash="dash")),
    ]

    layout = Layout(
        title=attr(text=title, font=merge(font_spec, attr(size=18))),
        height=900, width=900,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=attr(title="Epoch", anchor="y4", showgrid=true, gridcolor=gc),
        yaxis=attr(title="Recon loss", domain=[0.77, 1.00], type="log",
            showgrid=true, gridcolor=gc, titlefont=attr(color="#1f77b4")),
        yaxis2=attr(title="Perplexity", domain=[0.53, 0.73],
            showgrid=true, gridcolor=gc, rangemode="tozero",
            titlefont=attr(color="#2ca02c")),
        yaxis3=attr(title="Commit loss", domain=[0.28, 0.49], type="log",
            showgrid=true, gridcolor=gc, titlefont=attr(color="#d62728")),
        yaxis4=attr(title="Mean |shift| (samp)", domain=[0.00, 0.24],
            showgrid=true, gridcolor=gc, rangemode="tozero",
            titlefont=attr(color="#ff7f0e")),
        legend=attr(x=1.02, xanchor="left", y=0.5,
            font=merge(font_spec, attr(size=11))),
        margin=attr(r=170),
    )
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ a0000023-0000-0000-0000-000000000001
function plot_cluster_histogram(pct_ac, pct_c; title="Cluster Usage", labels=nothing)
    xlabels = labels === nothing ? string.(1:length(pct_ac)) : labels
    traces = [
        PlutoPlotly.bar(x=xlabels, y=pct_ac, name="Acausal",
            marker=attr(color="rgba(31,119,180,0.7)")),
        PlutoPlotly.bar(x=xlabels, y=pct_c, name="Causal",
            marker=attr(color="rgba(214,39,40,0.7)")),
    ]
    layout = Layout(
        title=attr(text=title, font=attr(size=20,
            family="Computer Modern, Latin Modern Math, serif")),
        barmode="group", height=400, width=800,
        xaxis=attr(title="Cluster combination",
            tickangle=labels === nothing ? 0 : -30),
        yaxis=attr(title="Percentage (%)"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ a0000024-0000-0000-0000-000000000001
md"## Codebook Analysis"

# ╔═╡ combo_helpers_cell
begin
    function combination_multipliers(K, T)
        [K^(t - 1) for t in 1:T]
    end
    function combination_index(digits::AbstractVector{Int}, mults::AbstractVector{Int})
        return sum((digits .- 1) .* mults) + 1
    end
    function combination_digits(idx::Int, K::Int, T::Int)
        mults = combination_multipliers(K, T)
        digits = zeros(Int, T)
        n = idx - 1
        for t in 1:T
            digits[t] = mod(div(n, mults[t]), K) + 1
        end
        return digits
    end
    function combination_labels(K, T)
        T == 1 && return [string(k) for k in 1:K]
        total = K^T
        labels = Vector{String}(undef, total)
        for idx in 1:total
            labels[idx] = join(combination_digits(idx, K, T), "-")
        end
        return labels
    end

    function get_cluster_percentages(m, x; return_labels::Bool=false)
        result = encode(m, x)
        K = codebook_size(m)
        T = m.T
        counts = T == 1 ? zeros(Float32, K) : zeros(Float32, K^T)
        mults = T == 1 ? nothing : combination_multipliers(K, T)
        N_total = size(result.codebook_indices, 2)
        for j in 1:N_total
            if T == 1
                counts[result.codebook_indices[1, j]] += 1f0
            else
                combo_idx = combination_index(result.codebook_indices[:, j], mults)
                counts[combo_idx] += 1f0
            end
        end
        labels = T == 1 ? [string(k) for k in 1:K] : combination_labels(K, T)
        percentages = counts ./ max(sum(counts), 1f-10) .* 100f0
        return return_labels ? (; percentages, labels) : percentages
    end

    function filter_cluster(m, x, ks::NTuple{N,Int}) where {N}
        N != m.T && throw(ArgumentError("Expected tuple of length m.T=$(m.T)"))
        result = encode(m, x)
        x_flat = flatten_batch(x)
        ci_flat = result.codebook_indices
        ks_vec = collect(ks)
        selected = findall(j -> all(ci_flat[:, j] .== ks_vec), 1:size(ci_flat, 2))
        return x_flat[:, selected], selected
    end

    function get_cluster_averages(m, x)
        result = encode(m, x)
        K = codebook_size(m)
        T = m.T
        nt = size(x, 1)
        x_flat = cpu(flatten_batch(x))
        N_total = size(result.codebook_indices, 2)
        if T == 1
            indices = vec(result.codebook_indices)
            avgs = zeros(Float32, nt, K)
            counts = zeros(Int, K)
            for (j, k) in enumerate(indices)
                avgs[:, k] .+= x_flat[:, j]; counts[k] += 1
            end
            for k in 1:K; counts[k] > 0 && (avgs[:, k] ./= counts[k]); end
        else
            ci_flat = result.codebook_indices
            num_c = K^T
            avgs = zeros(Float32, nt, num_c)
            counts = zeros(Int, num_c)
            mults = [K^(t-1) for t in 1:T]
            for j in 1:N_total
                combo_idx = sum((ci_flat[:, j] .- 1) .* mults) + 1
                avgs[:, combo_idx] .+= x_flat[:, j]; counts[combo_idx] += 1
            end
            for idx in 1:num_c; counts[idx] > 0 && (avgs[:, idx] ./= counts[idx]); end
        end
        return avgs
    end

    function codebook_agreement(model, D_ac, D_c)
        res_ac = encode(model, D_ac)
        res_c  = encode(model, D_c)
        if model.T == 1
            return mean(vec(res_ac.codebook_indices) .== vec(res_c.codebook_indices))
        end
        K = codebook_size(model)
        function majority(ci)
            [begin
                counts = zeros(Int, K)
                for t in 1:model.T; counts[ci[t, j]] += 1; end
                argmax(counts)
            end for j in 1:size(ci, 2)]
        end
        return mean(
            majority(res_ac.codebook_indices) .==
            majority(res_c.codebook_indices))
    end

    function codebook_cross_analysis(model, D_ac, D_c)
        K = codebook_size(model); T = model.T
        pct_ac_res = get_cluster_percentages(model, D_ac; return_labels=true)
        pct_ac  = pct_ac_res.percentages
        labels  = pct_ac_res.labels
        pct_c   = get_cluster_percentages(model, D_c)
        num_c   = length(pct_ac)
        res_ac  = encode(model, D_ac)
        res_c   = encode(model, D_c)
        ci_ac   = reshape(res_ac.codebook_indices, T, :)
        ci_c    = reshape(res_c.codebook_indices, T, :)
        nw      = min(size(ci_ac, 2), size(ci_c, 2))
        agreement = mean([all(ci_ac[:, w] .== ci_c[:, w]) for w in 1:nw])
        confusion = zeros(Float32, num_c, num_c)
        mults = combination_multipliers(K, T)
        for w in 1:nw
            ia = combination_index(ci_ac[:, w], mults)
            ic = combination_index(ci_c[:, w], mults)
            confusion[ia, ic] += 1f0
        end
        confusion ./= max(sum(confusion), 1f-10)
        thresh = 5f0; ratio = 5f0
        shared = Int[]; ac_only = Int[]; c_only = Int[]
        for k in 1:num_c
            if pct_ac[k] > thresh && pct_c[k] > thresh
                push!(shared, k)
            elseif pct_ac[k] > ratio * max(pct_c[k], 0.1f0)
                push!(ac_only, k)
            elseif pct_c[k] > ratio * max(pct_ac[k], 0.1f0)
                push!(c_only, k)
            end
        end
        return (; pct_ac, pct_c, confusion, agreement,
            shared_codes=shared, ac_only_codes=ac_only, c_only_codes=c_only,
            labels)
    end

    function plot_codebook_confusion(confusion; title="Codebook Confusion", labels=nothing)
        KT = size(confusion, 1)
        xl = labels === nothing ? string.(1:KT) : labels
        text_vv = [[string(round(confusion[i, j] * 100; digits=1), "%")
                    for j in 1:KT] for i in 1:KT]
        trace = PlutoPlotly.heatmap(z=confusion, x=xl, y=xl,
            colorscale="Blues", text=text_vv, texttemplate="%{text}")
        layout = Layout(
            title=attr(text=title,
                font=attr(size=20, family="Computer Modern, Latin Modern Math, serif")),
            height=900, width=1000,
            xaxis=attr(title="Causal", dtick=1, constrain="domain"),
            yaxis=attr(title="Acausal", dtick=1, scaleanchor="x", constrain="domain"),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        return PlutoPlotly.plot([trace], layout)
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
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
