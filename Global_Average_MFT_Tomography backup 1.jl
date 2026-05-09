### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto,
# the mock version below gives bound variables a default value.
macro bind(def, element)
    return quote
        local iv = try
            Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value
        catch
            b -> missing
        end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 8b17699b-e077-4b6e-9f4c-c6f72f6b7b9d
begin
    using Base.Threads
    using ColorSchemes
    using Colors
    using DSP
    using JLD2
    using LinearAlgebra
    using PlutoLinks
    using PlutoPlotly
    using PlutoUI
    using Printf
    using Statistics
end

# ╔═╡ 7da772a9-d8e9-4370-a2ed-f2a2d2e72f9a
mft = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/MFT.jl")

# ╔═╡ 19aac2c8-9df6-40a9-b9bb-bf8aa5545cd2
md"""
# Global-Average MFT Tomography Sandbox

This notebook skips VQ-VAE training/GPU work. It directly loads selected
receiver-pair CCF files, computes one global causal/acausal mean per pair in
parallel, runs MFT, extracts consensus candidates, and scores candidate mixes
using receiver geometry.
"""

# ╔═╡ 0c0901a5-ac20-41ab-b302-a20541ec4927
begin
    data_filepath = "/mnt/NAS2/Sanket_data/California_TO_with_latlong/"
    dt = 1.0
    period_min = 3.0
    period_max = 10.0
    mft_nperiods = 100
    velocity_range = (1.0, 8.0)
    bandwidth_factor = 0.15
    zero_pad_factor = 4
end

# ╔═╡ 6de24b0e-7794-4df4-b861-93c07ca5fc67
md"## Pair Selection"

# ╔═╡ 9406bc19-a197-4c68-a56b-7260b55e55de
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

# ╔═╡ 47673eb3-850e-42f2-a36a-b30dd603f411
available_pairs = list_station_pairs(data_filepath)

# ╔═╡ 81e56e48-6f7f-41b0-b1c7-634e516c8c78
begin
    pair_options = ["$(p[1])-$(p[2])" for p in available_pairs]
    default_pair_names = pair_options[1:min(end, 8)]
    @bind selected_pair_names confirm(MultiCheckBox(pair_options; default=default_pair_names))
end

# ╔═╡ a910fc18-a1f0-4b64-9d94-58f877b20162
selected_pairs = begin
    isempty(selected_pair_names) && Tuple{String,String}[]
    [begin
        parts = split(name, "-")
        (String(parts[1]), String(parts[2]))
    end for name in selected_pair_names]
end

# ╔═╡ 83b99182-09e8-4f79-8a29-744d02b79289
md"Selected pairs: **$(length(selected_pairs))**"

# ╔═╡ 1d084d85-0f08-4261-a29d-bacff7e32686
md"## Load Global Causal/Acausal Means"

# ╔═╡ a2f46dc1-c75e-467b-a7ad-93d23238d034
function _pair_file(pair::Tuple{String,String}, filepath::String)
    pair_name = join(pair, "_")
    matches = filter(path -> occursin(pair_name, basename(path)), readdir(filepath, join=true))
    isempty(matches) && error("No JLD2 file matching pair $(pair_name) found in $(filepath)")
    return matches[1]
end

# ╔═╡ d2990988-e956-4f19-94b7-b8fb8ad7d962
function _normalize_columns(X::AbstractMatrix)
    Y = Matrix{Float64}(X)
    for j in axes(Y, 2)
        col = view(Y, :, j)
        col .-= mean(col)
        nrm = norm(col)
        nrm > 0.0 && (col ./= nrm)
    end
    return Y
end

# ╔═╡ ffc2f1e5-2bd2-4453-8cec-ac4b5592b53d
function _split_causal_acausal(X::AbstractMatrix; zero_lag::Bool=true, max_lag=nothing)
    nt, ntr = size(X)
    isodd(nt) || error("Expected odd CCF length with zero lag at center; got nt=$(nt)")
    center = div(nt + 1, 2)
    half = div(nt - 1, 2)
    N = isnothing(max_lag) ? half : max(0, min(half, max_lag))
    X_acausal = reverse(X[center-N:center-1, :], dims=1)
    X_causal = X[center+1:center+N, :]
    if zero_lag
        return vcat(zeros(1, ntr), Array(X_acausal)), vcat(zeros(1, ntr), Array(X_causal))
    end
    return Array(X_acausal), Array(X_causal)
end

# ╔═╡ bb38b58c-2251-4141-a1c4-094d5bc3a6cb
function _taper(X::AbstractMatrix)
    w = reshape(tukey(size(X, 1), 0.1), :, 1)
    return w .* X
end

# ╔═╡ ed677c56-7257-4eb9-abda-dcc4d64bb8f4
function load_global_average_pair(pair::Tuple{String,String};
                                  filepath::String,
                                  dt::Real,
                                  period_min::Real,
                                  period_max::Real,
                                  apply_period_bandpass::Bool=true)
    path = _pair_file(pair, filepath)
    raw = load(path)
    correlations = haskey(raw, "correlations") ? raw["correlations"] : raw["D"][1]
    distance = haskey(raw, "dist") ? Float64(raw["dist"]) :
        (haskey(raw, "Distances") ? Float64(raw["Distances"][1]) : nothing)
    latitudes = haskey(raw, "latitudes") ? Float64.(raw["latitudes"]) : nothing
    longitudes = haskey(raw, "longitudes") ? Float64.(raw["longitudes"]) : nothing

    D = _normalize_columns(correlations)
    D_ac, D_c = _split_causal_acausal(D; zero_lag=true)

    if apply_period_bandpass
        responsetype = Bandpass(inv(period_max), inv(period_min))
        designmethod = Butterworth(2)
        digfilter = digitalfilter(responsetype, designmethod; fs=inv(dt))
        D_ac = filtfilt(digfilter, _taper(D_ac))
        D_c = filtfilt(digfilter, _taper(D_c))
        D_ac = D_ac[2:end, :]
        D_c = D_c[2:end, :]
    end

    D_ac = _normalize_columns(D_ac)
    D_c = _normalize_columns(D_c)

    return (;
        pair,
        path,
        acausal_mean=vec(mean(D_ac; dims=2)),
        causal_mean=vec(mean(D_c; dims=2)),
        distance,
        latitudes,
        longitudes,
        ntraces=size(correlations, 2),
    )
end

# ╔═╡ b2f727b8-e516-42b7-96fc-8662d7b95d31
global_pair_means = let
    results = Vector{Any}(undef, length(selected_pairs))
    Threads.@threads for i in eachindex(selected_pairs)
        results[i] = load_global_average_pair(
            selected_pairs[i];
            filepath=data_filepath,
            dt=dt,
            period_min=period_min,
            period_max=period_max,
            apply_period_bandpass=true,
        )
    end
    results
end

# ╔═╡ 51e0fe98-7147-475a-9cfd-90d178be0174
md"Loaded **$(length(global_pair_means))** pair global averages using **$(Threads.nthreads())** Julia threads."

# ╔═╡ 16400aef-fcd2-4fef-9d3a-9a6475cf48ff
md"## MFT Per Pair"

# ╔═╡ c06f563d-76b3-4199-9f9e-8e88c7d99a7e
mft_periods = exp10.(range(log10(Float64(period_min)), log10(Float64(period_max)); length=mft_nperiods))

# ╔═╡ 737929d2-1e56-4d6c-9a6b-08b6bfcfaac1
global_mft_analyses = let
    analyses = Vector{Any}(undef, length(global_pair_means))
    Threads.@threads for i in eachindex(global_pair_means)
        item = global_pair_means[i]
        analyses[i] = mft.analyze_causal_acausal_branches(
            [mft.SeismicTrace(data=item.acausal_mean, dt=dt, distance=item.distance)],
            [mft.SeismicTrace(data=item.causal_mean, dt=dt, distance=item.distance)],
            mft_periods;
            state_labels=["Global average"],
            max_modes=6,
            velocity_range=velocity_range,
            bandwidth_factor=bandwidth_factor,
            zero_pad_factor=zero_pad_factor,
        )
    end
    analyses
end

# ╔═╡ b7b08de2-3e57-43b0-bcf7-3e4b44d9be96
global_consensus = [mft.consensus_group_velocity_picks(
    analysis;
    correlation_threshold=0.0,
    velocity_tolerance_fraction=0.10,
    cluster_tolerance_fraction=nothing,
    max_candidates=5,
    selection_mode=:low_velocity,
    min_candidate_periods=3,
    max_smooth_jump_fraction=0.08,
    max_gap_periods=1,
) for analysis in global_mft_analyses]

# ╔═╡ da49315a-b75d-46cb-b86e-a5472dac9eea
md"Computed MFT and per-pair consensus candidates for **$(length(global_consensus))** receiver pairs."

# ╔═╡ 767db22f-f134-4a54-8d00-30fccf682194
md"## Geometry-Aware Tomography Candidate Mixes"

# ╔═╡ d192bb59-d3f6-461b-9ff7-fec09b22fae2
tomography_pair_inputs = begin
    inputs = mft.PairConsensusForTomography[]
    for (item, consensus) in zip(global_pair_means, global_consensus)
        if !isnothing(item.latitudes) && !isnothing(item.longitudes)
            push!(inputs, mft.tomography_pair_consensus(
                item.pair,
                consensus;
                latitudes=item.latitudes,
                longitudes=item.longitudes,
                distance=item.distance,
                label="$(item.pair[1])-$(item.pair[2])",
            ))
        end
    end
    inputs
end

# ╔═╡ 62e13733-67d3-47cf-9667-d1d03d38546e
tomography_candidate_mixes = mft.tomography_candidate_mixes(
    tomography_pair_inputs;
    max_mix_parts=3,
    min_candidate_periods=3,
    midpoint_radius_km=75.0,
    azimuth_tolerance_deg=25.0,
    distance_tolerance_fraction=0.35,
    velocity_tolerance_fraction=0.10,
)

# ╔═╡ 34f21dad-0635-47d5-875b-d58ad7cb7f88e
function candidate_mix_table(mixes; n::Int=20)
    isempty(mixes) && return md"No tomography candidate mixes available."
    rows = ["| Rank | Pair / mix | Periods | Mean conf | Neighbor agreement | Score |",
            "|---:|---|---:|---:|---:|---:|"]
    for (rank, mix) in enumerate(mixes[1:min(n, length(mixes))])
        push!(rows, @sprintf("| %d | %s | %d | %.3f | %.3f | %.3f |",
                             rank, mix.label, mix.coverage_count,
                             mix.mean_confidence, mix.neighbor_agreement,
                             mix.total_score))
    end
    return Markdown.parse(join(rows, "\n"))
end

# ╔═╡ 7c4230e0-3d0d-489b-93a2-e5ec26b28229
candidate_mix_table(tomography_candidate_mixes; n=25)

# ╔═╡ 9d62e303-722d-43c8-bd64-85a9703b52c3
md"## Quick Plots"

# ╔═╡ 3957df97-29d4-495a-9efe-c4286fd2ab25
begin
    pair_labels_for_plot = ["$(item.pair[1])-$(item.pair[2])" for item in global_pair_means]
    @bind selected_plot_pair Select(pair_labels_for_plot)
end

# ╔═╡ a6b713a4-1044-489b-8c89-bfb8dd5088e9
selected_plot_index = findfirst(==(selected_plot_pair), ["$(item.pair[1])-$(item.pair[2])" for item in global_pair_means])

# ╔═╡ 3062f062-3c5b-4c1b-b191-6d23a14c48bc
if isnothing(selected_plot_index)
    md""
else
    WideCell(mft.plot_consensus_groupvelocity_picks(
        global_mft_analyses[selected_plot_index],
        global_consensus[selected_plot_index];
        correlation_threshold=0.0,
        velocity_tolerance_fraction=0.10,
        title="Global-average consensus $(selected_plot_pair)",
    ))
end

# ╔═╡ d84594ea-cff7-49cf-bb01-e377f13b1f09
function plot_top_tomography_mixes(mixes, pairs; n::Int=25)
    traces = [PlutoPlotly.scatter()]
    isempty(mixes) && return PlutoPlotly.plot(PlutoPlotly.scatter(x=[0], y=[0], text=["No mixes"]))
    top = mixes[1:min(n, length(mixes))]
    all_vels = Float64[]
    colors = [Colors.hex(get(ColorSchemes.viridis, (i - 1) / max(1, length(top) - 1))) for i in 1:length(top)]

    for (i, mix) in enumerate(top)
        valid = findall(v -> isfinite(v) && v > 0.0, mix.group_velocities)
        isempty(valid) && continue
        append!(all_vels, mix.group_velocities[valid])
        pair = pairs[mix.pair_index]
        push!(traces, PlutoPlotly.scatter(
            x=pair.consensus.periods[valid],
            y=mix.group_velocities[valid],
            mode="lines+markers",
            name=mix.label,
            line=attr(color=colors[i], width=2),
            marker=attr(size=6, color=colors[i]),
            customdata=[[mix.total_score, mix.neighbor_agreement, mix.coverage_count] for _ in valid],
            hovertemplate="%{fullData.name}<br>Period: %{x:.2f} s<br>v_g: %{y:.3f} km/s<br>Score: %{customdata[0]:.3f}<br>Neighbor agree: %{customdata[1]:.3f}<br>Periods: %{customdata[2]}<extra></extra>",
        ))
    end

    isempty(all_vels) && return PlutoPlotly.plot(PlutoPlotly.scatter(x=[0], y=[0], text=["No valid velocities"]))
    return PlutoPlotly.plot(traces, PlutoPlotly.Layout(
        title="Top tomography candidate mixes",
        xaxis=attr(title="Period (s)", type="linear", range=[period_min, period_max]),
        yaxis=attr(title="Group velocity (km/s)", range=[0.9 * minimum(all_vels), 1.1 * maximum(all_vels)]),
        width=1200,
        height=720,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=attr(x=1.02, y=1.0),
        margin=attr(l=80, r=180, t=70, b=70),
    ))
end

# ╔═╡ f65040ef-fd82-433b-bb9c-5faed80c3616
WideCell(plot_top_tomography_mixes(tomography_candidate_mixes, tomography_pair_inputs; n=20))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
"""

# ╔═╡ Cell order:
# ╠═8b17699b-e077-4b6e-9f4c-c6f72f6b7b9d
# ╠═7da772a9-d8e9-4370-a2ed-f2a2d2e72f9a
# ╟─19aac2c8-9df6-40a9-b9bb-bf8aa5545cd2
# ╠═0c0901a5-ac20-41ab-b302-a20541ec4927
# ╟─6de24b0e-7794-4df4-b861-93c07ca5fc67
# ╠═9406bc19-a197-4c68-a56b-7260b55e55de
# ╠═47673eb3-850e-42f2-a36a-b30dd603f411
# ╠═81e56e48-6f7f-41b0-b1c7-634e516c8c78
# ╠═a910fc18-a1f0-4b64-9d94-58f877b20162
# ╟─83b99182-09e8-4f79-8a29-744d02b79289
# ╟─1d084d85-0f08-4261-a29d-bacff7e32686
# ╠═a2f46dc1-c75e-467b-a7ad-93d23238d034
# ╠═d2990988-e956-4f19-94b7-b8fb8ad7d962
# ╠═ffc2f1e5-2bd2-4453-8cec-ac4b5592b53d
# ╠═bb38b58c-2251-4141-a1c4-094d5bc3a6cb
# ╠═ed677c56-7257-4eb9-abda-dcc4d64bb8f4
# ╠═b2f727b8-e516-42b7-96fc-8662d7b95d31
# ╟─51e0fe98-7147-475a-9cfd-90d178be0174
# ╟─16400aef-fcd2-4fef-9d3a-9a6475cf48ff
# ╠═c06f563d-76b3-4199-9f9e-8e88c7d99a7e
# ╠═737929d2-1e56-4d6c-9a6b-08b6bfcfaac1
# ╠═b7b08de2-3e57-43b0-bcf7-3e4b44d9be96
# ╟─da49315a-b75d-46cb-b86e-a5472dac9eea
# ╟─767db22f-f134-4a54-8d00-30fccf682194
# ╠═d192bb59-d3f6-461b-9ff7-fec09b22fae2
# ╠═62e13733-67d3-47cf-9667-d1d03d38546e
# ╠═34f21dad-0635-47d5-875b-d58ad7cb7f88e
# ╠═7c4230e0-3d0d-489b-93a2-e5ec26b28229
# ╟─9d62e303-722d-43c8-bd64-85a9703b52c3
# ╠═3957df97-29d4-495a-9efe-c4286fd2ab25
# ╠═a6b713a4-1044-489b-8c89-bfb8dd5088e9
# ╠═3062f062-3c5b-4c1b-b191-6d23a14c48bc
# ╠═d84594ea-cff7-49cf-bb01-e377f13b1f09
# ╠═f65040ef-fd82-433b-bb9c-5faed80c3616
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
