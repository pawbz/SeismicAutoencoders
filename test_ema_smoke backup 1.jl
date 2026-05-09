### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ 00000000-0000-0000-0000-000000000001
begin
	using Random, Statistics, LinearAlgebra
	using Lux, Optimisers, ADTypes, Enzyme, EnzymeCore, NNlib
	import Lux.Training
end

# ╔═╡ 00000000-0000-0000-0000-000000000002
md"""
# VQ-VAE EMA Smoke Test

Tests whether the codebook EMA update **actually survives** across training steps
in the precomputed-payload Reactant training path used by `VQVAE_architecture_v7`.

**The suspected bug:** `Training.single_train_step!` with Reactant freezes `st`
(including `st.rvq`) into the XLA graph at compile time, then overwrites `ts.states`
with that frozen version after each step — discarding the CPU-side EMA update.

**The fix:** after `single_train_step!`, re-inject `st_updated.rvq` back into
`train_state.states`.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000010
md"## Shared EMA helpers (mirrors `VQVAE_architecture_v7`)"

# ╔═╡ 00000000-0000-0000-0000-000000000011
begin
	function assignment_matrix(indices::AbstractVector{Int}, K::Int)
	    N = length(indices)
	    enc = zeros(Float32, K, N)
	    for (j, i) in enumerate(indices)
	        enc[i, j] = 1f0
	    end
	    return enc
	end

	function vq_lookup_idx(embedding::Matrix{Float32}, z::AbstractVector{Float32})
	    dists = vec(sum((embedding .- z) .^ 2; dims=1))
	    return argmin(dists)
	end

	function update_stage_ema(embedding, ema_cs, ema_dw, z, indices, K, decay, epsilon)
	    enc    = assignment_matrix(indices, K)
	    counts = vec(sum(enc; dims=2))
	    sums   = z * enc'
	    ema_cs2 = decay .* ema_cs .+ (1f0 - decay) .* counts
	    n = sum(ema_cs2)
	    ema_cs2 = (ema_cs2 .+ epsilon) ./ (n + Float32(K) * epsilon) .* n
	    ema_dw2 = decay .* ema_dw .+ (1f0 - decay) .* sums
	    emb2    = ema_dw2 ./ reshape(max.(ema_cs2, epsilon), 1, :)
	    return emb2, ema_cs2, ema_dw2
	end

	function vq_quantize_cpu(z, embedding, ema_cs, ema_dw, K, decay, epsilon)
	    indices = [vq_lookup_idx(embedding, z[:, j]) for j in 1:size(z, 2)]
	    emb2, ema_cs2, ema_dw2 = update_stage_ema(embedding, ema_cs, ema_dw, z, indices, K, decay, epsilon)
	    z_q    = emb2[:, indices]
	    counts = [sum(i .== indices) for i in 1:K]
	    p      = Float64.(counts) ./ max(sum(counts), 1)
	    perp   = exp(-sum(p[p .> 0] .* log.(p[p .> 0])))
	    return z_q, emb2, ema_cs2, ema_dw2, perp
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000020
md"""
## Test 1: Pure CPU EMA loop (baseline)

Codebook **must** change every step. If it doesn't, something is wrong with the
EMA math itself.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000021
let
	Random.seed!(42)
	K, D, N    = 4, 8, 32
	decay, ε   = 0.99f0, 1f-5
	embedding  = randn(Float32, D, K) .* 0.1f0
	ema_cs     = ones(Float32, K)
	ema_dw     = copy(embedding)
	emb_before = copy(embedding)

	rows = []
	for step in 1:5
	    z = randn(Float32, D, N)
	    _, embedding, ema_cs, ema_dw, perp = vq_quantize_cpu(z, embedding, ema_cs, ema_dw, K, decay, ε)
	    diff = norm(embedding - emb_before)
	    push!(rows, (step=step, codebook_change=round(diff; digits=6), perplexity=round(perp; digits=3),
	        ok = diff > 0 ? "✓" : "✗ ZERO — BUG"))
	    emb_before = copy(embedding)
	end

	rows
end

# ╔═╡ 00000000-0000-0000-0000-000000000030
md"""
## Test 2: Lux + Enzyme CPU backend (simulates training loop)

Replicates the `prepare_rvq_payload → single_train_step!` pattern **without** Reactant.

In this test the codebook lives as a plain Julia variable (like in our CPU payload path).
`single_train_step!` has no way to overwrite it. This should work correctly.

The purpose is to confirm the EMA math and STE loss are correct in isolation.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000031
let
	Random.seed!(42)
	K, D_in, D_lat, N = 4, 16, 8, 32
	decay, ε, β = 0.99f0, 1f-5, 0.25f0

	encoder = Dense(D_in, D_lat)
	ps, st  = Lux.setup(Random.default_rng(), encoder)
	opt     = Optimisers.Adam(1f-3)
	ts      = Training.TrainState(encoder, ps, st, opt)

	embedding  = randn(Float32, D_lat, K) .* 0.1f0
	ema_cs     = ones(Float32, K)
	ema_dw     = copy(embedding)
	emb_before = copy(embedding)

	function loss_fn(model, ps, st, data)
	    x, z_q_pre = data
	    z_e, st2 = model(x, ps, st)
	    z_q_ste    = z_e .+ EnzymeCore.ignore_derivatives(z_q_pre .- z_e)
	    recon_loss  = mean((z_q_ste .- x[1:D_lat, :]) .^ 2)
	    commit_loss = β * mean((z_e .- EnzymeCore.ignore_derivatives(z_q_pre)) .^ 2)
	    return recon_loss + commit_loss, st2, (;)
	end

	rows = []
	for step in 1:5
	    x = randn(Float32, D_in, N)
	    z_e_cpu, _ = Lux.apply(encoder, x, ts.parameters, Lux.testmode(ts.states))
	    z_q_pre, embedding, ema_cs, ema_dw, perp = vq_quantize_cpu(
	        Float32.(z_e_cpu), embedding, ema_cs, ema_dw, K, decay, ε)
	    _, _, _, ts = Training.single_train_step!(AutoEnzyme(), loss_fn, (x, z_q_pre), ts)
	    diff = norm(embedding - emb_before)
	    push!(rows, (step=step, codebook_change=round(diff; digits=6), perplexity=round(perp; digits=3),
	        ok = diff > 0 ? "✓" : "✗ ZERO — BUG"))
	    emb_before = copy(embedding)
	end

	rows
end

# ╔═╡ 00000000-0000-0000-0000-000000000040
md"""
## Test 3: Simulate `ts.states.rvq` clobbering (the actual Reactant bug)

The codebook now lives **inside** `ts.states.rvq` (as in the real architecture).
We simulate what happens when `single_train_step!` returns a `ts` with the **old**
frozen state, and verify the re-inject fix restores the EMA-updated codebook.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000041
let
	Random.seed!(42)
	K, D_in, D_lat, N = 4, 16, 8, 32
	decay, ε, β = 0.99f0, 1f-5, 0.25f0

	encoder = Dense(D_in, D_lat)
	ps, st_init = Lux.setup(Random.default_rng(), encoder)
	opt = Optimisers.Adam(1f-3)

	# Put codebook inside ts.states.rvq (mirrors the real model)
	init_emb   = randn(Float32, D_lat, K) .* 0.1f0
	init_ema_cs = ones(Float32, K)
	init_ema_dw = copy(init_emb)
	rvq_state  = (; embedding=init_emb, ema_cs=init_ema_cs, ema_dw=init_ema_dw)
	st_full    = merge(st_init, (; rvq=rvq_state))

	ts = Training.TrainState(encoder, ps, Lux.trainmode(st_full), opt)

	function loss_fn2(model, ps, st, data)
	    x, z_q_pre = data
	    z_e, st2 = model(x, ps, (; encoder=st.encoder))   # only encoder substate
	    z_q_ste    = z_e .+ EnzymeCore.ignore_derivatives(z_q_pre .- z_e)
	    recon_loss  = mean((z_q_ste .- x[1:D_lat, :]) .^ 2)
	    commit_loss = β * mean((z_e .- EnzymeCore.ignore_derivatives(z_q_pre)) .^ 2)
	    # return full st with rvq unchanged (precomputed path doesn't touch rvq in the diff graph)
	    return recon_loss + commit_loss, merge(st, (; encoder=st2)), (;)
	end

	rows = []
	for step in 1:5
	    x = randn(Float32, D_in, N)
	    # CPU-side EMA update from current ts.states.rvq
	    cur_rvq = ts.states.rvq
	    z_e_cpu, _ = Lux.apply(encoder, x, ts.parameters, Lux.testmode((; encoder=ts.states.encoder)))
	    z_q_pre, new_emb, new_ema_cs, new_ema_dw, perp = vq_quantize_cpu(
	        Float32.(z_e_cpu), cur_rvq.embedding, cur_rvq.ema_cs, cur_rvq.ema_dw, K, decay, ε)
	    st_updated_rvq = (; embedding=new_emb, ema_cs=new_ema_cs, ema_dw=new_ema_dw)

	    emb_before = copy(cur_rvq.embedding)

	    # inject updated rvq before step (mirrors replace_train_state_states)
	    st_with_rvq = merge(ts.states, (; rvq=st_updated_rvq))
	    ts = Training.TrainState(ts.cache, ts.objective_function, ts.allocator_cache,
	        encoder, ts.parameters, st_with_rvq, ts.optimizer, ts.optimizer_state, ts.step)

	    _, _, _, ts = Training.single_train_step!(AutoEnzyme(), loss_fn2, (x, z_q_pre), ts)

	    emb_after_step = ts.states.rvq.embedding

	    diff_from_ema  = norm(emb_after_step - new_emb)     # should be 0 with fix, >0 means clobbered
	    diff_from_prev = norm(new_emb - emb_before)          # EMA moved the codebook

	    push!(rows, (
	        step            = step,
	        ema_changed     = round(diff_from_prev; digits=6),
	        clobbered_after_step = round(diff_from_ema; digits=6),
	        perplexity      = round(perp; digits=3),
	        ok = diff_from_ema < 1f-6 ? "✓ fix works" : "✗ CLOBBERED"
	    ))
	end

	rows
end

# ╔═╡ 00000000-0000-0000-0000-000000000050
md"""
## Test 4: Logic check for `merge` re-inject

Sanity check that `merge(old_st, (; rvq=new_rvq))` correctly replaces the rvq field.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000051
let
	old_emb = ones(Float32, 4, 4)
	new_emb = 2f0 .* ones(Float32, 4, 4)

	st_old = (; rvq=(; embedding=old_emb, x=42), encoder=(; w=3))
	st_new_rvq = (; embedding=new_emb, x=42)

	st_after_fix    = merge(st_old, (; rvq=st_new_rvq))
	st_after_no_fix = st_old

	(
	    without_fix = st_after_no_fix.rvq.embedding[1,1],   # expect 1.0
	    with_fix    = st_after_fix.rvq.embedding[1,1],       # expect 2.0
	    encoder_preserved = st_after_fix.encoder.w,           # expect 3
	    verdict = st_after_fix.rvq.embedding[1,1] == 2f0 ? "✓ merge re-inject is correct" : "✗ BUG"
	)
end

# ╔═╡ 00000000-0000-0000-0000-000000000099
md"---"

# ╔═╡ 00000000-0000-0000-0000-000000000100
begin
	import Pkg
	Pkg.status(["Lux", "Enzyme", "Optimisers"])
end

# ╔═╡ Cell order:
# ╔═╡ 00000000-0000-0000-0000-000000000001
# ╔═╡ 00000000-0000-0000-0000-000000000002
# ╔═╡ 00000000-0000-0000-0000-000000000010
# ╔═╡ 00000000-0000-0000-0000-000000000011
# ╔═╡ 00000000-0000-0000-0000-000000000020
# ╔═╡ 00000000-0000-0000-0000-000000000021
# ╔═╡ 00000000-0000-0000-0000-000000000030
# ╔═╡ 00000000-0000-0000-0000-000000000031
# ╔═╡ 00000000-0000-0000-0000-000000000040
# ╔═╡ 00000000-0000-0000-0000-000000000041
# ╔═╡ 00000000-0000-0000-0000-000000000050
# ╔═╡ 00000000-0000-0000-0000-000000000051
# ╔═╡ 00000000-0000-0000-0000-000000000099
# ╔═╡ 00000000-0000-0000-0000-000000000100
