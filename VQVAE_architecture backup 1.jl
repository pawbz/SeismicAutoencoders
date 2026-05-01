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
using PlutoLinks, PlutoHooks

# ╔═╡ 4a95997e-5c12-4658-9b8e-a5065328e1c1
using BenchmarkTools

# ╔═╡ 97ae4222-5a3e-4cbd-b4d1-aa028d3e4ca8
TableOfContents(include_definitions=true)

# ╔═╡ 26fb86d5-c844-469a-aef5-ed3c2a9ba949
xpu = gpu

# ╔═╡ 6affb3b3-9dc4-4bbc-a582-495fc1783a7a
activation = x -> leakyrelu(x, 0.1f0)

# ╔═╡ dc7e0a28-2739-44c2-9a44-c66079aaae17
DG = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/data_generators.jl")

# ╔═╡ 461f0505-2230-4b84-b6c6-1a9730808437
md"""# VQ-VAE for Seismic Waveform Clustering

## Architecture Overview

**Vector Quantized Variational Autoencoder (VQ-VAE)** for extracting shared coherent features
between causal and acausal cross-correlation branches.

### Key Ideas
- **Shared encoder** processes both causal and acausal branches identically
- **Vector quantization** with finite codebook forces discrete clustering
- **EMA codebook updates** (no gradient through codebook — more stable)
- **Configurable T** (number of quantized vectors per waveform) controls information bottleneck
- **Symmetry loss** encourages encoder to extract direction-invariant features
- **Codebook reset** for dead entries prevents codebook collapse

### Architecture
```
waveform (nt,) → ConvEncoder → z_e (d, T) → VQ → z_q (d, T) → ConvDecoder → x̂ (nt,)
                                                    ↓
                                             codebook indices (T,)
```
"""

# ╔═╡ a0000001-0000-0000-0000-000000000001
md"## Utilities"

# ╔═╡ 80f77b52-84e0-4664-8aa0-3d79fded40de
"""
Instead of cat(x, dims=3)
"""
add_dim3_reshape(::Nothing) = nothing

# ╔═╡ 6ba143e2-50df-441a-8f38-3ea8d9edd4d8
function add_dim3_reshape(x)
    if ndims(x) == 1
        return reshape(x, :, 1, 1)
    elseif ndims(x) == 2
        return reshape(x, size(x, 1), size(x, 2), 1)
    else
        return x
    end
end

# ╔═╡ a0000002-0000-0000-0000-000000000001
md"## Parameters"

# ╔═╡ 91a25156-e121-4d53-a5a1-422f1230d235
Base.@kwdef struct VQVAE_Para
    nt::Int                                # waveform length (time samples)
    d::Int = 64                            # codebook embedding dimension
    K::Int = 8                             # codebook size (number of entries)
    T::Int = 1                             # quantized vectors per waveform (1 = single-vector VQ)
    beta_commit::Float32 = 0.25f0          # commitment loss weight
    beta_sym::Float32 = 1.0f0              # symmetry loss weight (‖z_ac - z_c‖²)
    enc_kernels::Vector{Int} = [32, 16, 8, 4]
    enc_filters::Vector{Int} = [8, 16, 32, 64]
    enc_strides::Vector{Int} = [2, 2, 2, 2]
    dec_kernels::Vector{Int} = [4, 8, 16]
    dec_filters::Vector{Int} = [64, 48, 16, 1]
    dec_upstrides::Vector{Int} = [2, 2, 1]
    use_bn::Bool = true
    ema_decay::Float32 = 0.99f0            # EMA decay for codebook updates
    epsilon::Float32 = 1f-5                # EMA Laplace smoothing
    dead_threshold::Int = 2                # reset codebook entry after this many batches unused
    entropy_weight::Float32 = 0.01f0       # entropy bonus to encourage uniform codebook usage
    seed = nothing
end

# ╔═╡ a0000003-0000-0000-0000-000000000001
md"## Conv Encoder"

# ╔═╡ 89599b3f-8c20-46c5-8f5c-ccbb71b26b36
begin
    struct Conv1DChain
        chain::Chain
    end
    Flux.@layer Conv1DChain trainable = (chain,)

    function (m::Conv1DChain)(::Nothing)
        return nothing
    end

    function (m::Conv1DChain)(x)
        x = add_dim3_reshape(x)
        n1, n2, n3 = size(x)
        X = reshape(x, n1, 1, n2 * n3)
        X = m.chain(X)
        return reshape(X, :, n2, n3)
    end
end

# ╔═╡ a0000004-0000-0000-0000-000000000001
"""
Build a 1D convolutional encoder.

Returns `(encoder, flat_output_length)`.
"""
function get_vq_conv_encoder(nt; kernels=[32, 16, 8, 4], filters=[8, 16, 32, 64],
                              strides=[2, 2, 2, 2], use_bn::Bool=true)
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
    push!(layers, Flux.flatten)
    return Conv1DChain(Chain(layers...)), flat_len
end

# ╔═╡ a0000005-0000-0000-0000-000000000001
md"## Conv Decoder"

# ╔═╡ 64430447-c267-4eec-8d38-63ccf91d82c4
"""
Build a 1D convolutional decoder (transposed convolutions).

Input dimension: `d_in` (latent dim fed to the decoder).
Output: reconstructed waveform of length `nt`.
"""
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
    # Project from latent to spatial
    push!(layers, Dense(d_in, bottleneck_len * bottleneck_channels, activation))
    push!(layers, x -> reshape(x, bottleneck_len, bottleneck_channels, :))

    nin = bottleneck_channels
    for (i, k) in enumerate(kernels)
        nout = filters[i + 1]
        s = upstrides[i]
        if i < length(kernels)
            push!(layers, ConvTranspose((k,), nin => nout, activation; stride=s, pad=SamePad()))
            if use_bn
                push!(layers, BatchNorm(nout))
            end
        else
            # Final layer: no activation, output 1 channel
            push!(layers, ConvTranspose((k,), nin => nout; stride=s, pad=SamePad()))
        end
        nin = nout
    end

    # Trim or pad to exact nt
    push!(layers, x -> x[1:min(nt, size(x, 1)), :, :])

    return Conv1DChain(Chain(layers...)), bottleneck_len, bottleneck_channels
end

# ╔═╡ a0000006-0000-0000-0000-000000000001
md"""## Vector Quantizer (EMA)

Core VQ layer with:
- **Exponential Moving Average** codebook updates (no codebook gradient needed)
- **Dead entry reset**: re-initializes unused codebook entries from encoder outputs
- **Straight-through estimator**: gradients pass directly from z_q to z_e
"""

# ╔═╡ a0000007-0000-0000-0000-000000000001
begin
    mutable struct VectorQuantizerEMA
        K::Int                                # number of codebook entries
        d::Int                                # embedding dimension
        embedding::AbstractMatrix{Float32}    # codebook: (d, K)
        ema_cluster_size::AbstractVector{Float32}   # (K,) EMA of assignment counts
        ema_dw::AbstractMatrix{Float32}       # (d, K) EMA of sum of assigned encoder outputs
        decay::Float32
        epsilon::Float32
        dead_count::Vector{Int}               # CPU counter: batches since last assignment
        dead_threshold::Int
    end
    Flux.@layer VectorQuantizerEMA trainable = ()  # no trainable params — EMA only

    function VectorQuantizerEMA(K::Int, d::Int;
                                 decay::Float32=0.99f0,
                                 epsilon::Float32=1f-5,
                                 dead_threshold::Int=2)
        # Initialize codebook from uniform [-1/K, 1/K]
        embedding = xpu(randn(Float32, d, K) .* (1f0 / K))
        ema_cluster_size = xpu(ones(Float32, K))
        ema_dw = copy(embedding)
        dead_count = zeros(Int, K)
        return VectorQuantizerEMA(K, d, embedding, ema_cluster_size, ema_dw,
                                   decay, epsilon, dead_count, dead_threshold)
    end

    """
    Quantize encoder output `z_e` of shape (d, N) where N = T * batch_size.

    Returns `(; z_q, indices, encodings, vq_loss, commit_loss, perplexity)`.
    - During training: updates codebook via EMA, resets dead entries.
    - `z_q` has straight-through gradient (gradient flows to encoder).
    """
    function (vq::VectorQuantizerEMA)(z_e::AbstractMatrix{Float32}; beta_commit::Float32=0.25f0, training::Bool=true)
        d, N = size(z_e)
        @assert d == vq.d "Expected embedding dim $(vq.d), got $d"

        # ── Nearest-neighbor lookup ───────────────────────────────────────
        # distances: (K, N) — squared L2 distance from each z_e to each codebook entry
        # ‖z - e‖² = ‖z‖² - 2zᵀe + ‖e‖²
        z_sq = sum(abs2, z_e; dims=1)         # (1, N)
        e_sq = sum(abs2, vq.embedding; dims=1) # (1, K)
        dist = z_sq' .+ e_sq .- 2f0 .* (vq.embedding' * z_e)  # (K, N)

        # Hard assignment: argmin over K
        indices_cart = dropdims(CUDA.argmin(dist; dims=1); dims=1)  # CartesianIndex (N,)
        indices = getindex.(indices_cart, 1)  # Int indices (N,)

        # One-hot encodings: (K, N)
        encodings = xpu(Float32.(Flux.onehotbatch(cpu(indices), 1:vq.K)))

        # Quantized vectors: lookup from codebook
        z_q = vq.embedding * encodings  # (d, K) × (K, N) = (d, N)

        # ── EMA update (training only) ────────────────────────────────────
        if training
            enc_sum = encodings * ones(Float32, N, 1) |> xpu  # (K, 1): count per entry
            enc_sum_vec = vec(enc_sum)  # (K,)

            # EMA cluster size
            vq.ema_cluster_size .= vq.decay .* vq.ema_cluster_size .+ (1f0 - vq.decay) .* enc_sum_vec

            # Laplace smoothing
            n = sum(vq.ema_cluster_size)
            vq.ema_cluster_size .= (vq.ema_cluster_size .+ vq.epsilon) ./ (n .+ Float32(vq.K) .* vq.epsilon) .* n

            # EMA sum of assigned encoder outputs
            dw = z_e * encodings'  # (d, K)
            vq.ema_dw .= vq.decay .* vq.ema_dw .+ (1f0 - vq.decay) .* dw

            # Update codebook
            vq.embedding .= vq.ema_dw ./ reshape(vq.ema_cluster_size, 1, :)

            # ── Dead entry reset ──────────────────────────────────────────
            counts_cpu = cpu(enc_sum_vec)
            for k in 1:vq.K
                if counts_cpu[k] < 0.5f0
                    vq.dead_count[k] += 1
                else
                    vq.dead_count[k] = 0
                end
                if vq.dead_count[k] >= vq.dead_threshold
                    # Reset to random encoder output + small noise
                    j = rand(1:N)
                    z_rand = z_e[:, j] .+ xpu(randn(Float32, d) .* 0.01f0)
                    vq.embedding[:, k] .= z_rand
                    vq.ema_dw[:, k] .= z_rand
                    vq.ema_cluster_size[k] = 1f0
                    vq.dead_count[k] = 0
                end
            end
        end

        # ── Losses ────────────────────────────────────────────────────────
        # VQ loss: ‖sg[z_e] - e‖² (not needed with EMA, but useful as metric)
        vq_loss = Flux.mse(Flux.stop_gradient(z_e), z_q)
        # Commitment loss: ‖z_e - sg[e]‖²
        commit_loss = beta_commit * Flux.mse(z_e, Flux.stop_gradient(z_q))

        # ── Straight-through estimator ────────────────────────────────────
        z_q_st = z_e + Flux.stop_gradient(z_q - z_e)

        # ── Perplexity (codebook utilization metric) ──────────────────────
        avg_probs = mean(cpu(encodings); dims=2) |> vec  # (K,)
        avg_probs_safe = clamp.(avg_probs, 1f-10, 1f0)
        perplexity = exp(-sum(avg_probs_safe .* log.(avg_probs_safe)))

        # ── Entropy bonus (encourage uniform usage) ───────────────────────
        entropy_loss = sum(avg_probs_safe .* log.(avg_probs_safe))  # negative entropy

        return (; z_q=z_q_st, indices, encodings, vq_loss, commit_loss, perplexity, entropy_loss)
    end

    """
    Quantize without training updates (inference).
    """
    function quantize_inference(vq::VectorQuantizerEMA, z_e::AbstractMatrix{Float32})
        return vq(z_e; training=false)
    end
end

# ╔═╡ a0000008-0000-0000-0000-000000000001
md"## VQ-VAE Model"

# ╔═╡ a0000009-0000-0000-0000-000000000001
begin
    struct VQVAE{E,P,VQ,D}
        encoder::E         # Conv1DChain: waveform → flat features
        pre_vq::P          # Dense: flat → (d * T)
        quantizer::VQ      # VectorQuantizerEMA
        decoder::D         # Conv1DChain: d → waveform
        T::Int             # number of quantized vectors per waveform
        d::Int             # codebook embedding dimension
    end
    Flux.@layer VQVAE trainable = (encoder, pre_vq, decoder)

    """
    Forward pass through VQ-VAE.

    Arguments:
    - `x`: input waveform (nt, ntau, batch) or (nt, batch)
    - `training`: whether to update EMA codebook

    Returns named tuple with reconstruction `xhat`, losses, codebook indices, etc.
    """
    function (m::VQVAE)(x; beta_commit::Float32=0.25f0, training::Bool=true)
        x3 = add_dim3_reshape(x)
        n1, n2, n3 = size(x3)  # (nt, ntau, batch)

        # Encode: (nt, ntau, batch) → (flat, ntau, batch)
        feat = m.encoder(x3)  # (flat_len, ntau, batch)
        feat_flat = reshape(feat, size(feat, 1), :)  # (flat_len, ntau*batch)

        # Project to VQ space: (flat_len, N) → (d*T, N) → reshape to (d, T*N)
        z_pre = m.pre_vq(feat_flat)  # (d*T, N)
        N_total = size(z_pre, 2)  # ntau * batch
        z_e = reshape(z_pre, m.d, m.T * N_total)  # (d, T * N)

        # Quantize
        vq_result = m.quantizer(z_e; beta_commit, training)

        # Decode: (d, T*N) → reshape to (d*T, N) → decoder → (nt, N)
        z_q_for_dec = reshape(vq_result.z_q, m.d * m.T, N_total)  # (d*T, N)
        dec_input = reshape(z_q_for_dec, m.d * m.T, n2, n3)  # (d*T, ntau, batch)
        xhat = m.decoder(dec_input)  # (nt, ntau, batch)

        # Reshape indices: (T*N,) → (T, ntau, batch) for interpretability
        codebook_indices = reshape(cpu(vq_result.indices), m.T, n2, n3)

        return (;
            xhat,
            z_e = reshape(z_e, m.d, m.T, N_total),
            z_q = reshape(vq_result.z_q, m.d, m.T, N_total),
            codebook_indices,
            vq_loss = vq_result.vq_loss,
            commit_loss = vq_result.commit_loss,
            perplexity = vq_result.perplexity,
            entropy_loss = vq_result.entropy_loss,
        )
    end

    """
    Encode-only: get quantized codes and codebook indices for a batch of waveforms.
    No decoder pass.
    """
    function encode(m::VQVAE, x)
        x3 = add_dim3_reshape(x)
        n1, n2, n3 = size(x3)
        feat = m.encoder(x3)
        feat_flat = reshape(feat, size(feat, 1), :)
        z_pre = m.pre_vq(feat_flat)
        N_total = size(z_pre, 2)
        z_e = reshape(z_pre, m.d, m.T * N_total)
        vq_result = m.quantizer(z_e; training=false)
        codebook_indices = reshape(cpu(vq_result.indices), m.T, n2, n3)
        z_q = reshape(vq_result.z_q, m.d, m.T, n2, n3)
        return (; z_e=reshape(z_e, m.d, m.T, n2, n3), z_q, codebook_indices)
    end

    """
    Get codebook prototypes: returns (d, K) matrix.
    """
    get_codebook(m::VQVAE) = cpu(m.quantizer.embedding)

    """
    Get cluster assignment histogram for a batch.
    Returns (K,) vector of percentages.
    """
    function get_cluster_percentages(m::VQVAE, x)
        result = encode(m, x)
        indices = vec(result.codebook_indices)
        K = m.quantizer.K
        counts = zeros(Float32, K)
        for idx in indices
            counts[idx] += 1f0
        end
        return counts ./ sum(counts) .* 100f0
    end

    """
    Get waveforms assigned to cluster `k`.
    Returns `(data_in_cluster, indices_in_data)`.
    """
    function filter_cluster(m::VQVAE, x, k::Int)
        result = encode(m, x)
        if m.T == 1
            # Single-vector VQ: straightforward cluster assignment
            idx_flat = vec(result.codebook_indices)  # (ntau * batch,)
            selected = findall(==(k), idx_flat)
            x3 = add_dim3_reshape(x)
            n1, n2, n3 = size(x3)
            x_flat = reshape(x3, n1, n2 * n3)
            return x_flat[:, selected], selected
        else
            # Multi-vector VQ: assign based on majority vote across T vectors
            # codebook_indices: (T, ntau, batch)
            ci = result.codebook_indices
            n2, n3 = size(ci, 2), size(ci, 3)
            x3 = add_dim3_reshape(x)
            x_flat = reshape(x3, size(x3, 1), n2 * n3)
            ci_flat = reshape(ci, m.T, n2 * n3)
            selected = Int[]
            for j in 1:size(ci_flat, 2)
                # Majority vote
                counts = zeros(Int, m.quantizer.K)
                for t in 1:m.T
                    counts[ci_flat[t, j]] += 1
                end
                if argmax(counts) == k
                    push!(selected, j)
                end
            end
            return x_flat[:, selected], selected
        end
    end

    """
    Get cluster averages: mean waveform per cluster.
    Returns (nt, K) matrix.
    """
    function get_cluster_averages(m::VQVAE, x)
        result = encode(m, x)
        K = m.quantizer.K
        x3 = add_dim3_reshape(x)
        nt = size(x3, 1)
        n2, n3 = size(result.codebook_indices, 2), size(result.codebook_indices, 3)
        x_flat = cpu(reshape(x3, nt, n2 * n3))

        if m.T == 1
            indices = vec(result.codebook_indices)
        else
            # Majority vote assignment
            ci_flat = reshape(result.codebook_indices, m.T, n2 * n3)
            indices = [begin
                counts = zeros(Int, K)
                for t in 1:m.T; counts[ci_flat[t, j]] += 1; end
                argmax(counts)
            end for j in 1:size(ci_flat, 2)]
        end

        avgs = zeros(Float32, nt, K)
        counts = zeros(Int, K)
        for (j, k) in enumerate(indices)
            avgs[:, k] .+= x_flat[:, j]
            counts[k] += 1
        end
        for k in 1:K
            if counts[k] > 0
                avgs[:, k] ./= counts[k]
            end
        end
        return avgs
    end
end

# ╔═╡ a0000010-0000-0000-0000-000000000001
md"## Model Factory"

# ╔═╡ a0000011-0000-0000-0000-000000000001
"""
    get_vqvae(para::VQVAE_Para)

Build a VQ-VAE model and loss history from parameters.

Returns `(model, loss_history)`.
"""
function get_vqvae(para::VQVAE_Para)
    if para.seed !== nothing
        Random.seed!(para.seed)
    end

    # Encoder
    encoder, flat_len = get_vq_conv_encoder(para.nt;
        kernels=para.enc_kernels, filters=para.enc_filters,
        strides=para.enc_strides, use_bn=para.use_bn)

    # Pre-VQ projection: flat encoder output → (d * T) latent vectors
    pre_vq = Dense(flat_len, para.d * para.T) |> xpu

    # Vector Quantizer with EMA
    quantizer = VectorQuantizerEMA(para.K, para.d;
        decay=para.ema_decay, epsilon=para.epsilon,
        dead_threshold=para.dead_threshold)

    # Decoder: (d * T) → nt waveform
    decoder, _, _ = get_vq_conv_decoder(para.nt, para.d * para.T;
        kernels=para.dec_kernels, filters=para.dec_filters,
        upstrides=para.dec_upstrides, use_bn=para.use_bn)

    model = VQVAE(xpu(encoder), xpu(pre_vq), quantizer, xpu(decoder), para.T, para.d)

    loss_history = (;
        train_recon = Float32[],
        test_recon = Float32[],
        train_commit = Float32[],
        test_commit = Float32[],
        train_total = Float32[],
        test_total = Float32[],
        train_perplexity = Float32[],
        test_perplexity = Float32[],
        train_sym = Float32[],
        test_sym = Float32[],
    )

    return model, loss_history
end

# ╔═╡ a0000012-0000-0000-0000-000000000001
md"## Loss Functions"

# ╔═╡ a0000013-0000-0000-0000-000000000001
"""
Reconstruction MSE loss for a single batch (no symmetry).
"""
function loss_recon(model::VQVAE, x; training::Bool=false)
    x3 = add_dim3_reshape(xpu(x))
    result = model(x3; beta_commit=0.25f0, training=false)
    return Flux.mse(result.xhat, x3)
end

# ╔═╡ a0000014-0000-0000-0000-000000000001
"""
Full VQ-VAE loss for paired causal/acausal training:

    L = L_recon_ac + L_recon_c + β_commit · L_commit + β_sym · ‖z_ac - z_c‖² + β_entropy · H

Arguments:
- `model`: VQVAE model
- `x_ac`: acausal waveforms
- `x_c`: causal waveforms (same waveform windows, opposite branch)
- `para`: VQVAE_Para with loss weights
- `training`: whether to update EMA codebook
"""
function loss_vqvae_paired(model::VQVAE, x_ac, x_c, para::VQVAE_Para; training::Bool=true)
    x_ac3 = add_dim3_reshape(xpu(x_ac))
    x_c3 = add_dim3_reshape(xpu(x_c))

    # Forward both branches
    res_ac = model(x_ac3; beta_commit=para.beta_commit, training)
    res_c = model(x_c3; beta_commit=para.beta_commit, training)

    # Reconstruction losses
    recon_ac = Flux.mse(res_ac.xhat, x_ac3)
    recon_c = Flux.mse(res_c.xhat, x_c3)
    recon_loss = recon_ac + recon_c

    # Commitment losses (averaged)
    commit_loss = (res_ac.commit_loss + res_c.commit_loss) / 2f0

    # Symmetry loss: force causal and acausal to have similar latent codes
    # z_e: (d, T, N) — compare along dim 1,2, average over N
    sym_loss = Flux.mse(res_ac.z_e, res_c.z_e)

    # Entropy regularization (encourage uniform codebook usage)
    entropy_loss = (res_ac.entropy_loss + res_c.entropy_loss) / 2f0

    # Total
    total = recon_loss + commit_loss + para.beta_sym * sym_loss + para.entropy_weight * entropy_loss

    # Metrics
    perplexity = (res_ac.perplexity + res_c.perplexity) / 2f0

    return (; total, recon_loss, commit_loss, sym_loss, entropy_loss, perplexity)
end

# ╔═╡ a0000015-0000-0000-0000-000000000001
md"## Data Iterator"

# ╔═╡ a0000016-0000-0000-0000-000000000001
"""
Paired data iterator: yields `(x_ac_batch, x_c_batch)` of shape `(nt, ntau)` each.
Samples `ntau` waveforms from the same temporal windows for both branches.
"""
struct PairedDataIterator
    D_ac::AbstractMatrix{Float32}   # (nt, nwaveforms) acausal on GPU
    D_c::AbstractMatrix{Float32}    # (nt, nwaveforms) causal on GPU
    ntau::Int
    nsteps::Int
end

function Base.iterate(it::PairedDataIterator, step=1)
    step > it.nsteps && return nothing
    nw = size(it.D_ac, 2)
    idx = sort(randperm(nw)[1:min(it.ntau, nw)])
    return (it.D_ac[:, idx], it.D_c[:, idx]), step + 1
end
Base.length(it::PairedDataIterator) = it.nsteps

# ╔═╡ a0000017-0000-0000-0000-000000000001
md"## Training Loop"

# ╔═╡ eafe181e-19e9-409e-ad1d-ce859cf0e672
Base.@kwdef struct VQVAE_Training_Para
    ntau::Int = 100                            # waveforms per batch
    nsteps::Int = 100                          # steps per epoch
    nepoch::Int = 30                           # epochs
    nprint::Int = 1                            # print frequency
    initial_learning_rate::Float64 = 0.001
    lr_decay::Float64 = 0.99                   # exponential LR decay per epoch
end

# ╔═╡ a0000018-0000-0000-0000-000000000001
using ParameterSchedulers

# ╔═╡ a0000019-0000-0000-0000-000000000001
"""
    update(model, loss_history, D_ac_train, D_c_train, D_ac_test, D_c_test,
           para, training_para)

Train the VQ-VAE with paired causal/acausal data.

Arguments:
- `model::VQVAE`: the VQ-VAE model
- `loss_history`: named tuple of loss vectors (mutated in-place)
- `D_ac_train`, `D_c_train`: GPU matrices (nt, nwaveforms_train)
- `D_ac_test`, `D_c_test`: GPU matrices (nt, nwaveforms_test)
- `para::VQVAE_Para`: architecture & loss parameters
- `training_para::VQVAE_Training_Para`: training hyperparameters
"""
function update(model::VQVAE, loss_history, D_ac_train, D_c_train,
                D_ac_test, D_c_test, para::VQVAE_Para,
                training_para::VQVAE_Training_Para=VQVAE_Training_Para())

    lr_s = Exp(start=training_para.initial_learning_rate, decay=training_para.lr_decay)
    opt_state = Optimisers.setup(Optimisers.AdamW(eta=Float64(training_para.initial_learning_rate)), model)

    ntau = min(training_para.ntau, size(D_ac_train, 2))
    ntau_test = min(training_para.ntau, size(D_ac_test, 2))

    @progress name = "VQ-VAE training" for epoch = 1:training_para.nepoch
        # ── Data iterators ────────────────────────────────────────────────
        Xtrain = PairedDataIterator(D_ac_train, D_c_train, ntau, training_para.nsteps)
        Xtest = PairedDataIterator(D_ac_test, D_c_test, ntau_test, training_para.nsteps)

        # ── Evaluate on first batch ───────────────────────────────────────
        x_ac_tr, x_c_tr = first(Xtrain)
        x_ac_te, x_c_te = first(Xtest)

        train_metrics = loss_vqvae_paired(model, x_ac_tr, x_c_tr, para; training=false)
        test_metrics = loss_vqvae_paired(model, x_ac_te, x_c_te, para; training=false)

        push!(loss_history.train_recon, train_metrics.recon_loss)
        push!(loss_history.test_recon, test_metrics.recon_loss)
        push!(loss_history.train_commit, train_metrics.commit_loss)
        push!(loss_history.test_commit, test_metrics.commit_loss)
        push!(loss_history.train_total, train_metrics.total)
        push!(loss_history.test_total, test_metrics.total)
        push!(loss_history.train_perplexity, train_metrics.perplexity)
        push!(loss_history.test_perplexity, test_metrics.perplexity)
        push!(loss_history.train_sym, train_metrics.sym_loss)
        push!(loss_history.test_sym, test_metrics.sym_loss)

        # ── LR schedule ──────────────────────────────────────────────────
        Optimisers.adjust!(opt_state, lr_s(epoch))

        # ── Gradient-based training (encoder, pre_vq, decoder) ────────────
        # EMA codebook update happens inside the forward pass
        function train_loss(model, x_ac, x_c)
            return loss_vqvae_paired(model, x_ac, x_c, para; training=true).total
        end

        for (x_ac, x_c) in Xtrain
            g = Flux.gradient(train_loss, model, x_ac, x_c)[1]
            Optimisers.update!(opt_state, model, g)
        end

        if mod(epoch, training_para.nprint) == 0
            @info "Epoch $epoch" recon=train_metrics.recon_loss sym=train_metrics.sym_loss commit=train_metrics.commit_loss perplexity=train_metrics.perplexity test_recon=test_metrics.recon_loss
        end
    end
    return nothing
end

# ╔═╡ a0000020-0000-0000-0000-000000000001
md"## Plotting"

# ╔═╡ a0000021-0000-0000-0000-000000000001
using PlutoPlotly

# ╔═╡ a0000022-0000-0000-0000-000000000001
"""
Plot VQ-VAE training loss history.
"""
function plot_loss_history(loss_history; title="VQ-VAE Training")
    epochs = 1:length(loss_history.train_recon)

    traces = [
        PlutoPlotly.scatter(x=collect(epochs), y=loss_history.train_recon, mode="lines",
            name="Train Recon", line=attr(color="#1f77b4")),
        PlutoPlotly.scatter(x=collect(epochs), y=loss_history.test_recon, mode="lines",
            name="Test Recon", line=attr(color="#1f77b4", dash="dash")),
        PlutoPlotly.scatter(x=collect(epochs), y=loss_history.train_sym, mode="lines",
            name="Train Symmetry", line=attr(color="#d62728")),
        PlutoPlotly.scatter(x=collect(epochs), y=loss_history.test_sym, mode="lines",
            name="Test Symmetry", line=attr(color="#d62728", dash="dash")),
        PlutoPlotly.scatter(x=collect(epochs), y=loss_history.train_perplexity, mode="lines",
            name="Train Perplexity", line=attr(color="green"), yaxis="y2"),
        PlutoPlotly.scatter(x=collect(epochs), y=loss_history.test_perplexity, mode="lines",
            name="Test Perplexity", line=attr(color="green", dash="dash"), yaxis="y2"),
    ]

    layout = Layout(
        title=attr(text=title, font=attr(size=20, family="Computer Modern, Latin Modern Math, serif")),
        height=500, width=900,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=attr(title="Epoch", showgrid=true, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=attr(title="Loss", showgrid=true, gridcolor="rgba(128,128,128,0.2)", type="log"),
        yaxis2=attr(title="Perplexity", overlaying="y", side="right",
                    showgrid=false, range=[0, nothing]),
        legend=attr(x=0.5, xanchor="center", y=-0.2, orientation="h",
                    font=attr(size=14, family="Computer Modern, Latin Modern Math, serif")),
    )
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ a0000023-0000-0000-0000-000000000001
"""
Plot cluster assignment histogram.
"""
function plot_cluster_histogram(pct_ac, pct_c; title="Cluster Usage")
    K = length(pct_ac)
    traces = [
        PlutoPlotly.bar(x=1:K, y=pct_ac, name="Acausal",
            marker=attr(color="rgba(31,119,180,0.7)")),
        PlutoPlotly.bar(x=1:K, y=pct_c, name="Causal",
            marker=attr(color="rgba(214,39,40,0.7)")),
    ]
    layout = Layout(
        title=attr(text=title, font=attr(size=20, family="Computer Modern, Latin Modern Math, serif")),
        barmode="group", height=400, width=600,
        xaxis=attr(title="Codebook Entry", dtick=1),
        yaxis=attr(title="Percentage (%)"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ a0000024-0000-0000-0000-000000000001
md"## Codebook Analysis"

# ╔═╡ a0000025-0000-0000-0000-000000000001
"""
Compute agreement rate: fraction of waveform windows where causal and acausal
map to the same codebook entry (for T=1; for T>1 uses majority vote).

High agreement = encoder successfully extracts direction-invariant features.
"""
function codebook_agreement(model::VQVAE, D_ac, D_c)
    res_ac = encode(model, D_ac)
    res_c = encode(model, D_c)
    if model.T == 1
        idx_ac = vec(res_ac.codebook_indices)
        idx_c = vec(res_c.codebook_indices)
    else
        # Majority vote per waveform
        function majority(ci, K)
            ci_flat = reshape(ci, model.T, :)
            [begin
                counts = zeros(Int, K)
                for t in 1:model.T; counts[ci_flat[t, j]] += 1; end
                argmax(counts)
            end for j in 1:size(ci_flat, 2)]
        end
        K = model.quantizer.K
        idx_ac = majority(res_ac.codebook_indices, K)
        idx_c = majority(res_c.codebook_indices, K)
    end
    return mean(idx_ac .== idx_c)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
md"""
---
## Cell Order (Pluto metadata)
"""

# ╔═╡ Cell order:
# ╟─461f0505-2230-4b84-b6c6-1a9730808437
# ╠═cc11647d-1c56-4ceb-9677-703aca03c9f4
# ╠═d73472ff-9e09-45b0-8811-b7dd8d820358
# ╠═76dbf599-a9b3-459f-992b-16ab2f7b74f1
# ╠═4a95997e-5c12-4658-9b8e-a5065328e1c1
# ╠═97ae4222-5a3e-4cbd-b4d1-aa028d3e4ca8
# ╠═26fb86d5-c844-469a-aef5-ed3c2a9ba949
# ╠═6affb3b3-9dc4-4bbc-a582-495fc1783a7a
# ╠═dc7e0a28-2739-44c2-9a44-c66079aaae17
# ╟─a0000001-0000-0000-0000-000000000001
# ╠═80f77b52-84e0-4664-8aa0-3d79fded40de
# ╠═6ba143e2-50df-441a-8f38-3ea8d9edd4d8
# ╟─a0000002-0000-0000-0000-000000000001
# ╠═91a25156-e121-4d53-a5a1-422f1230d235
# ╟─a0000003-0000-0000-0000-000000000001
# ╠═89599b3f-8c20-46c5-8f5c-ccbb71b26b36
# ╠═a0000004-0000-0000-0000-000000000001
# ╟─a0000005-0000-0000-0000-000000000001
# ╠═64430447-c267-4eec-8d38-63ccf91d82c4
# ╟─a0000006-0000-0000-0000-000000000001
# ╠═a0000007-0000-0000-0000-000000000001
# ╟─a0000008-0000-0000-0000-000000000001
# ╠═a0000009-0000-0000-0000-000000000001
# ╟─a0000010-0000-0000-0000-000000000001
# ╠═a0000011-0000-0000-0000-000000000001
# ╟─a0000012-0000-0000-0000-000000000001
# ╠═a0000013-0000-0000-0000-000000000001
# ╠═a0000014-0000-0000-0000-000000000001
# ╟─a0000015-0000-0000-0000-000000000001
# ╠═a0000016-0000-0000-0000-000000000001
# ╟─a0000017-0000-0000-0000-000000000001
# ╠═eafe181e-19e9-409e-ad1d-ce859cf0e672
# ╠═a0000018-0000-0000-0000-000000000001
# ╠═a0000019-0000-0000-0000-000000000001
# ╟─a0000020-0000-0000-0000-000000000001
# ╠═a0000021-0000-0000-0000-000000000001
# ╠═a0000022-0000-0000-0000-000000000001
# ╠═a0000023-0000-0000-0000-000000000001
# ╟─a0000024-0000-0000-0000-000000000001
# ╠═a0000025-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000001
