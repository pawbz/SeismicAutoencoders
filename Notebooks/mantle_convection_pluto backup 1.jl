### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto,
# the following mock version of @bind gives bound variables a default value.
macro bind(def, element)
    return quote
        local iv = try
            Base.loaded_modules[
                Base.PkgId(
                    Base.UUID("6e696c72-6542-2067-7265-42206c756150"),
                    "AbstractPlutoDingetjes",
                ),
            ].Bonds.initial_value
        catch
            b -> missing
        end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 2f0136f4-33d1-4b7a-86c7-a17c8f6efdda
begin
    using PlutoUI
    using Plots
    using Statistics
    using LinearAlgebra
    using Random
end

# ╔═╡ a7c3f1f6-910e-4233-a7d8-f86d4c85138b
md"""
# Interactive Mantle Convection (ThermalConvection2D-style)

This notebook is a Pluto-friendly, interactive port inspired by:

- ParallelStencil miniapp: `ThermalConvection2D.jl`

It solves a reduced 2D Boussinesq thermo-mechanical model (infinite-Pr approximation with streamfunction diagnostics), with temperature-dependent viscosity and tunable initial perturbations including a cold slab-like anomaly.
"""

# ╔═╡ 7dcdb6f8-f6dd-4bbf-8bf4-836cb9ba4304
md"""
## Thermo-mechanical convection PDEs

The reference miniapp solves (dimensionless form) a Stokes + heat system:

$$
\nabla \cdot \mathbf{v} = 0,
$$

$$
-\nabla p + \nabla \cdot \left(2\eta(T)\,\dot{\varepsilon}(\mathbf{v})\right) + \mathrm{Ra}\,T\,\hat{\mathbf{z}} = 0,
$$

$$
\frac{\partial T}{\partial t} + \mathbf{v}\cdot\nabla T = \kappa\nabla^2 T + H.
$$

Where:

- $T$ is temperature,
- $\mathbf{v}=(v_x,v_z)$ is velocity,
- $\eta(T)$ is viscosity (often strongly temperature dependent),
- $\mathrm{Ra}$ is the Rayleigh number controlling buoyancy vigor.

For interactive speed in Pluto, the notebook uses a reduced streamfunction-based velocity solve each step while preserving the same physical coupling ideas:

- buoyancy from lateral temperature gradients,
- advection-diffusion of temperature,
- diagnostic viscosity field $\eta(T)$.
"""

# ╔═╡ 9c6f769d-f068-4c58-b4f8-b03870322038
md"""
## Controls

Change initial condition, viscosity contrast, Rayleigh number, and run length. Click **Recompute simulation** after changing inputs.
"""

# ╔═╡ 6020b4d9-f9ef-4ff3-a385-d015f6474f2c
@bind icase Select([
    "small_plume" => "Small central plume",
    "subduction_slab" => "Cold subduction slab",
    "double_plume" => "Double plume",
    "random_thermal" => "Random thermal noise",
]; default="subduction_slab")

# ╔═╡ d5efb6db-7c6d-4a13-b09f-1941e5a7f77f
@bind recompute CounterButton("Recompute simulation")

# ╔═╡ 7ad886a2-3d35-4cac-8b52-9cf8f5079cf0
begin
    @bind nx Slider(65:16:193; default=97, show_value=true)
    @bind nz Slider(49:16:145; default=65, show_value=true)
end

# ╔═╡ e53c40db-3eef-433a-b4d4-4f1dcd6f8ef0
begin
    @bind nsteps Slider(40:20:260; default=120, show_value=true)
    @bind save_every Slider(1:2:15; default=3, show_value=true)
    @bind poisson_iters Slider(20:10:140; default=60, show_value=true)
end

# ╔═╡ bfb8533d-a26a-4f1c-ad3d-870579710978
begin
    @bind Ra Slider([1e4, 3e4, 1e5, 3e5, 1e6, 3e6]; default=3e5, show_value=true)
    @bind eta_contrast Slider([5.0, 10.0, 30.0, 100.0, 300.0, 1000.0]; default=100.0, show_value=true)
end

# ╔═╡ a809d7c8-2dfb-4d5b-88ba-a20ea90cd5de
begin
    @bind slab_angle_deg Slider(-75:5:-10; default=-40, show_value=true)
    @bind slab_width Slider(0.03:0.01:0.20; default=0.08, show_value=true)
    @bind slab_strength Slider(0.05:0.05:0.80; default=0.35, show_value=true)
end

# ╔═╡ d80f4a5f-31b6-42d9-bef1-636af8fda1dc
begin
    recompute

    function laplacian!(out, A, dx, dz)
        nx_, nz_ = size(A)
        invdx2 = 1.0 / (dx * dx)
        invdz2 = 1.0 / (dz * dz)
        @inbounds for i in 2:(nx_ - 1), k in 2:(nz_ - 1)
            out[i, k] =
                (A[i + 1, k] - 2A[i, k] + A[i - 1, k]) * invdx2 +
                (A[i, k + 1] - 2A[i, k] + A[i, k - 1]) * invdz2
        end
        return out
    end

    function gradients!(dAdx, dAdz, A, dx, dz)
        nx_, nz_ = size(A)
        inv2dx = 1.0 / (2dx)
        inv2dz = 1.0 / (2dz)
        @inbounds for i in 2:(nx_ - 1), k in 2:(nz_ - 1)
            dAdx[i, k] = (A[i + 1, k] - A[i - 1, k]) * inv2dx
            dAdz[i, k] = (A[i, k + 1] - A[i, k - 1]) * inv2dz
        end
        return dAdx, dAdz
    end

    function apply_boundaries!(T)
        # Left-right insulating sidewalls.
        T[1, :] .= T[2, :]
        T[end, :] .= T[end - 1, :]
        # Bottom hot, top cold.
        T[:, 1] .= 1.0
        T[:, end] .= 0.0
        return T
    end

    function initial_temperature(X, Z; case, slab_angle, slab_w, slab_amp)
        T0 = 1 .- Z
        nx_, nz_ = size(T0)

        if case == "small_plume"
            T0 .+= 0.25 .* exp.(-((X .- 0.5) .^ 2 .+ (Z .- 0.15) .^ 2) ./ 0.012)
        elseif case == "double_plume"
            T0 .+= 0.20 .* exp.(-((X .- 0.30) .^ 2 .+ (Z .- 0.14) .^ 2) ./ 0.008)
            T0 .+= 0.20 .* exp.(-((X .- 0.70) .^ 2 .+ (Z .- 0.14) .^ 2) ./ 0.008)
        elseif case == "random_thermal"
            rng = MersenneTwister(42)
            T0 .+= 0.06 .* randn(rng, nx_, nz_)
        elseif case == "subduction_slab"
            x0 = 0.30
            z0 = 0.95
            slope = tand(slab_angle)
            slab_center = x0 .+ slope .* (Z .- z0)
            slab = exp.(-((X .- slab_center) .^ 2) ./ (2slab_w^2)) .* exp.(-((Z .- z0) .^ 2) ./ 0.18)
            T0 .-= slab_amp .* slab
        end

        clamp!(T0, 0.0, 1.0)
        return T0
    end

    function jacobi_poisson(rhs, dx, dz, niter)
        nx_, nz_ = size(rhs)
        psi = zeros(nx_, nz_)
        psi_new = similar(psi)
        cx = 1.0 / (dx * dx)
        cz = 1.0 / (dz * dz)
        c0 = 1.0 / (2cx + 2cz)

        for _ in 1:niter
            @inbounds for i in 2:(nx_ - 1), k in 2:(nz_ - 1)
                psi_new[i, k] = c0 * (
                    cx * (psi[i + 1, k] + psi[i - 1, k]) +
                    cz * (psi[i, k + 1] + psi[i, k - 1]) - rhs[i, k]
                )
            end

            psi_new[1, :] .= 0.0
            psi_new[end, :] .= 0.0
            psi_new[:, 1] .= 0.0
            psi_new[:, end] .= 0.0

            psi, psi_new = psi_new, psi
        end
        return psi
    end

    function simulate_convection(; nx_, nz_, Ra_, eta_ratio_, case, slab_angle, slab_w, slab_amp, nsteps_, save_every_, poisson_iters_)
        lx, lz = 1.5, 1.0
        x = range(0.0, lx; length=nx_)
        z = range(0.0, lz; length=nz_)
        dx, dz = step(x), step(z)
        X = [xx for xx in x, _ in z]
        Z = [zz for _ in x, zz in z]
        Xn = X ./ lx
        Zn = Z ./ lz

        kappa = 1.0
        T = initial_temperature(Xn, Zn; case=case, slab_angle=slab_angle, slab_w=slab_w, slab_amp=slab_amp)
        apply_boundaries!(T)

        dTdx = zeros(nx_, nz_)
        dTdz = zeros(nx_, nz_)
        lapT = zeros(nx_, nz_)
        lapVx = zeros(nx_, nz_)
        lapVz = zeros(nx_, nz_)

        T_hist = Matrix{Float64}[]
        eta_hist = Matrix{Float64}[]
        speed_hist = Matrix{Float64}[]
        vort_hist = Matrix{Float64}[]

        dt_diff = 0.22 * min(dx, dz)^2 / kappa

        for it in 1:nsteps_
            # Temperature dependent viscosity, hot -> low viscosity.
            eta = exp.(log(eta_ratio_) .* (0.5 .- T))

            # Reduced streamfunction forcing by lateral thermal gradient.
            gradients!(dTdx, dTdz, T, dx, dz)
            rhs = -Ra_ .* dTdx ./ eta
            psi = jacobi_poisson(rhs, dx, dz, poisson_iters_)

            vx = zeros(nx_, nz_)
            vz = zeros(nx_, nz_)
            gradients!(dTdx, dTdz, psi, dx, dz)
            vx .= dTdz
            vz .= .-dTdx

            vmax = maximum(abs, vx)
            wmax = maximum(abs, vz)
            dt_adv = 0.35 * min(dx / (vmax + 1e-8), dz / (wmax + 1e-8))
            dt = min(dt_diff, dt_adv)

            laplacian!(lapT, T, dx, dz)
            gradients!(dTdx, dTdz, T, dx, dz)

            Tnew = copy(T)
            @inbounds for i in 2:(nx_ - 1), k in 2:(nz_ - 1)
                adv = vx[i, k] * dTdx[i, k] + vz[i, k] * dTdz[i, k]
                Tnew[i, k] = T[i, k] + dt * (-adv + kappa * lapT[i, k])
            end
            T = clamp.(Tnew, 0.0, 1.0)
            apply_boundaries!(T)

            # Vorticity diagnostic.
            gradients!(dTdx, dTdz, vx, dx, dz)
            gradients!(lapVx, lapVz, vz, dx, dz)
            vort = lapVx .- dTdz

            if it == 1 || mod(it, save_every_) == 0 || it == nsteps_
                push!(T_hist, copy(T))
                push!(eta_hist, copy(eta))
                push!(speed_hist, sqrt.(vx .^ 2 .+ vz .^ 2))
                push!(vort_hist, copy(vort))
            end
        end

        return (
            x=x,
            z=z,
            T=T_hist,
            eta=eta_hist,
            speed=speed_hist,
            vort=vort_hist,
            nframes=length(T_hist),
        )
    end

    sim = simulate_convection(
        nx_=nx,
        nz_=nz,
        Ra_=Ra,
        eta_ratio_=eta_contrast,
        case=icase,
        slab_angle=slab_angle_deg,
        slab_w=slab_width,
        slab_amp=slab_strength,
        nsteps_=nsteps,
        save_every_=save_every,
        poisson_iters_=poisson_iters,
    )
end

# ╔═╡ bc6e234f-e887-4f7f-b73f-026f38f4ad26
md"""
## Playback controls

`Clock` drives frame updates; all heatmaps below are linked to the exact same current frame.
"""

# ╔═╡ ef74bf14-76c1-4d9d-a1f7-39276db4d13a
@bind autoplay CheckBox(default=true)

# ╔═╡ 1d5b4d8f-c7d7-4354-ac71-66cb62d5f6f1
@bind tick Clock(0.25, max_value=sim.nframes, repeat=true)

# ╔═╡ 9f2ff0d6-4f2f-4d71-bf53-a7f809f810f4
@bind frame_manual Slider(1:sim.nframes; default=1, show_value=true)

# ╔═╡ 32a26b2f-5ad9-4b7f-a060-a7d933c8370e
frame_id = autoplay ? tick : frame_manual

# ╔═╡ b8bce76f-ab39-40c2-b534-a2fa597f4bee
md"""
Current frame: **$(frame_id) / $(sim.nframes)**
"""

# ╔═╡ 9eae52d2-f975-47f0-b079-edf902220ca0
begin
    Tframe = sim.T[frame_id]
    etaframe = sim.eta[frame_id]
    speedframe = sim.speed[frame_id]
    vortframe = sim.vort[frame_id]

    p1 = heatmap(
        sim.x,
        sim.z,
        Tframe',
        xlabel="x",
        ylabel="z",
        title="Temperature T",
        c=:inferno,
        aspect_ratio=1,
    )

    p2 = heatmap(
        sim.x,
        sim.z,
        log10.(etaframe)',
        xlabel="x",
        ylabel="z",
        title="log10(viscosity)",
        c=:viridis,
        aspect_ratio=1,
    )

    p3 = heatmap(
        sim.x,
        sim.z,
        speedframe',
        xlabel="x",
        ylabel="z",
        title="Speed |v|",
        c=:plasma,
        aspect_ratio=1,
    )

    p4 = heatmap(
        sim.x,
        sim.z,
        vortframe',
        xlabel="x",
        ylabel="z",
        title="Vorticity",
        c=:RdBu,
        aspect_ratio=1,
    )

    plot(p1, p2, p3, p4; layout=(2, 2), size=(1000, 760))
end

# ╔═╡ 8dde7f3c-2a69-49b5-8067-675a13a462db
md"""
## Suggested additional fields to visualize

In addition to $T$, these fields are very informative for geologic interpretation:

- viscosity $\eta(T)$ (already included)
- speed magnitude $|\mathbf{v}|$ (already included)
- vorticity (already included)
- vertical velocity $v_z$ (good for plume upwelling/downwelling)
- strain-rate invariant $\sqrt{\dot{\varepsilon}_{ij}\dot{\varepsilon}_{ij}}$
- dissipation proxy $2\eta\,\dot{\varepsilon}_{ij}\dot{\varepsilon}_{ij}$
- compositional tracer concentration (if you add passive/active tracers)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Plots = "1"
PlutoUI = "0.7"
"""

# ╔═╡ Cell order:
# ╟─a7c3f1f6-910e-4233-a7d8-f86d4c85138b
# ╠═2f0136f4-33d1-4b7a-86c7-a17c8f6efdda
# ╟─7dcdb6f8-f6dd-4bbf-8bf4-836cb9ba4304
# ╟─9c6f769d-f068-4c58-b4f8-b03870322038
# ╠═6020b4d9-f9ef-4ff3-a385-d015f6474f2c
# ╠═d5efb6db-7c6d-4a13-b09f-1941e5a7f77f
# ╠═7ad886a2-3d35-4cac-8b52-9cf8f5079cf0
# ╠═e53c40db-3eef-433a-b4d4-4f1dcd6f8ef0
# ╠═bfb8533d-a26a-4f1c-ad3d-870579710978
# ╠═a809d7c8-2dfb-4d5b-88ba-a20ea90cd5de
# ╠═d80f4a5f-31b6-42d9-bef1-636af8fda1dc
# ╟─bc6e234f-e887-4f7f-b73f-026f38f4ad26
# ╠═ef74bf14-76c1-4d9d-a1f7-39276db4d13a
# ╠═1d5b4d8f-c7d7-4354-ac71-66cb62d5f6f1
# ╠═9f2ff0d6-4f2f-4d71-bf53-a7f809f810f4
# ╠═32a26b2f-5ad9-4b7f-a060-a7d933c8370e
# ╟─b8bce76f-ab39-40c2-b534-a2fa597f4bee
# ╠═9eae52d2-f975-47f0-b079-edf902220ca0
# ╟─8dde7f3c-2a69-49b5-8067-675a13a462db
# ╠═00000000-0000-0000-0000-000000000001