### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

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

# ╔═╡ a0f8a2b4-8fb5-4f06-bda6-c362a61065a1
begin
    using Base.Threads
    using ColorSchemes
    using Colors
    using JLD2
    using LinearAlgebra
    using PlutoLinks
    using PlutoPlotly
    using PlutoUI
    using Printf
    using Statistics
end

# ╔═╡ c7d47e38-e24f-4b40-b3a4-bc894188a750
mft = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/MFT.jl")

# ╔═╡ b00fd94f-291e-46d8-84ff-48f8606c2a1e
md"""
# Trained VQ-VAE Source-State MFT Tomography

CPU-only notebook for saved v8 source-state artifacts. It reads
`source_state_averages.jld2` files written by the training notebook, runs MFT for
each receiver pair, and scores geometry-aware tomography candidate mixes. It does
not load model weights, raw CCF data, or use GPU/Reactant.
"""

# ╔═╡ dcbf026e-957a-4b9b-9757-bd0638a25b26
begin
    saved_root = "/mnt/NAS2/Sanket_data/California_TO_with_latlong/SavedModels/vqvae_v8"
    dt = 1.0
    period_min = 3.0
    period_max = 10.0
    mft_nperiods = 100
    velocity_range = (1.0, 8.0)
    bandwidth_factor = 0.15
    zero_pad_factor = 4
    use_latest_run_per_seed = true
end

# ╔═╡ d50d63be-d58b-4704-8211-ed7875e04857
md"## Saved Runs"

# ╔═╡ c7ec82e0-d5b6-4f31-8600-c8b1d276dc92
function _parse_seed_timestamp(run_dir::String)
    name = basename(run_dir)
    m = match(r"^seed([0-9]+)_(.+)$", name)
    m === nothing && return (; seed=missing, timestamp=name)
    return (; seed=parse(Int, m.captures[1]), timestamp=m.captures[2])
end

# ╔═╡ e63099d0-5fb5-43b0-967c-b7c468dc4f83
function discover_vqvae_runs(saved_root::String)
    isdir(saved_root) || return NamedTuple[]
    runs = NamedTuple[]
    for pair_dir in sort(filter(isdir, readdir(saved_root, join=true)))
        pair_label = basename(pair_dir)
        parts = split(pair_label, "_")
        length(parts) == 2 || continue
        for run_dir in sort(filter(isdir, readdir(pair_dir, join=true)))
            isfile(joinpath(run_dir, "source_state_averages.jld2")) || continue
            parsed = _parse_seed_timestamp(run_dir)
            push!(runs, (; pair=(String(parts[1]), String(parts[2])),
                         pair_label=replace(pair_label, "_" => "-"),
                         run_dir, seed=parsed.seed, timestamp=parsed.timestamp))
        end
    end
    return runs
end

# ╔═╡ ed704922-12f0-475b-b628-19a9c37bca7a
all_saved_runs = discover_vqvae_runs(saved_root)

# ╔═╡ e216a473-6433-4658-b2b7-a4eaa670cc5e
begin
    pair_options = sort(unique([run.pair_label for run in all_saved_runs]))
    default_pair_names = pair_options[1:min(end, 8)]
    @bind selected_pair_names confirm(MultiCheckBox(pair_options; default=default_pair_names))
end

# ╔═╡ b5a1cf7d-d464-4409-b43a-074c8aa22108
selected_runs = begin
    raw = [run for run in all_saved_runs if run.pair_label in selected_pair_names]
    if use_latest_run_per_seed
        keep = Dict{Tuple{String,Any},Any}()
        for run in raw
            key = (run.pair_label, run.seed)
            if !haskey(keep, key) || string(run.timestamp) > string(keep[key].timestamp)
                keep[key] = run
            end
        end
        sort(collect(values(keep)), by=run -> (run.pair_label, string(run.seed), run.timestamp))
    else
        sort(raw, by=run -> (run.pair_label, string(run.seed), run.timestamp))
    end
end

# ╔═╡ a4c73d31-cada-44dd-81c8-fbb0f5e84f6a
md"Selected **$(length(selected_runs))** trained runs across **$(length(unique([r.pair_label for r in selected_runs])))** receiver pairs."

# ╔═╡ e4499887-1a64-4eaa-a599-4ed4941a7b2d
md"## Load Saved Source-State Averages"

# ╔═╡ e4f7b3cf-7f26-4ea9-b9bb-95f3df9e7790
function _source_state_artifact_path(run)
    return joinpath(run.run_dir, "source_state_averages.jld2")
end

# ╔═╡ 02d134b5-7ce3-47a9-86ef-a43e6c52287a
function _load_saved_source_state_averages(run)
    path = _source_state_artifact_path(run)
    isfile(path) || error("Missing source-state artifact: $(path)")
    d = load(path)
    return (;
        acausal=d["acausal"],
        causal=d["causal"],
        counts_ac=d["counts_ac"],
        counts_c=d["counts_c"],
        combo_labels=String.(d["combo_labels"]),
        window_headers=haskey(d, "window_headers") ? String.(d["window_headers"]) : String[],
        window_time_labels=haskey(d, "window_time_labels") ? String.(d["window_time_labels"]) : String[],
        source_state_ac=haskey(d, "source_state_ac") ? Int.(d["source_state_ac"]) : Int[],
        source_state_c=haskey(d, "source_state_c") ? Int.(d["source_state_c"]) : Int[],
        stage_assignments_ac=haskey(d, "stage_assignments_ac") ? Int.(d["stage_assignments_ac"]) : zeros(Int, 0, 0),
        stage_assignments_c=haskey(d, "stage_assignments_c") ? Int.(d["stage_assignments_c"]) : zeros(Int, 0, 0),
        assignment_table=haskey(d, "assignment_table") ? String.(d["assignment_table"]) : Matrix{String}(undef, 0, 0),
        assignment_table_columns=haskey(d, "assignment_table_columns") ? String.(d["assignment_table_columns"]) : String[],
        assignment_table_ac=haskey(d, "assignment_table_ac") ? String.(d["assignment_table_ac"]) : Matrix{String}(undef, 0, 0),
        assignment_table_c=haskey(d, "assignment_table_c") ? String.(d["assignment_table_c"]) : Matrix{String}(undef, 0, 0),
        assignment_table_ac_columns=haskey(d, "assignment_table_ac_columns") ? String.(d["assignment_table_ac_columns"]) : String[],
        assignment_table_c_columns=haskey(d, "assignment_table_c_columns") ? String.(d["assignment_table_c_columns"]) : String[],
        distance=d["distance"],
        latitudes=d["latitudes"],
        longitudes=d["longitudes"],
        pair=run.pair,
        pair_label=run.pair_label,
        run_dir=run.run_dir,
        seed=run.seed,
    )
end

# ╔═╡ ddfd42a8-4ae7-408a-8b8f-2d335746798b
run_source_state_averages = let
    out = Vector{Any}(undef, length(selected_runs))
    Threads.@threads for i in eachindex(selected_runs)
        out[i] = _load_saved_source_state_averages(selected_runs[i])
    end
    out
end

# ╔═╡ b7c7a358-71c8-4797-a259-68bc75ab6e65
md"Loaded source-state artifacts for **$(length(run_source_state_averages))** trained runs using **$(Threads.nthreads())** Julia threads."

# ╔═╡ eec83733-193d-4f52-9a75-e6f1d03c7aa5
md"## MFT By Receiver Pair"

# ╔═╡ 067d2587-8eb1-41c3-95f8-9f785171f2ce
mft_periods = exp10.(range(log10(Float64(period_min)), log10(Float64(period_max)); length=mft_nperiods))

# ╔═╡ b01af348-c4b0-4ad1-81cd-116e9f2ed765
pair_labels = sort(unique([item.pair_label for item in run_source_state_averages]))

# ╔═╡ c61a6cfe-b14e-4aa9-a711-450e35a3a9bd
pair_mft_analyses = let
    analyses = Dict{String,Any}()
    for pair_label in pair_labels
        items = [item for item in run_source_state_averages if item.pair_label == pair_label]
        ac_traces = mft.SeismicTrace[]
        c_traces = mft.SeismicTrace[]
        labels = String[]
        for item in items
            nstates = size(item.acausal, 2)
            for i in 1:nstates
                push!(ac_traces, mft.SeismicTrace(data=vec(item.acausal[:, i]), dt=dt, distance=item.distance))
                push!(c_traces,  mft.SeismicTrace(data=vec(item.causal[:, i]),  dt=dt, distance=item.distance))
                label = i <= length(item.combo_labels) ? item.combo_labels[i] : string(i)
                push!(labels, "$(pair_label) seed $(item.seed) | $(label)")
            end
        end
        analyses[pair_label] = mft.analyze_causal_acausal_branches(
            ac_traces, c_traces, mft_periods;
            state_labels=labels,
            max_modes=6,
            velocity_range=velocity_range,
            bandwidth_factor=bandwidth_factor,
            zero_pad_factor=zero_pad_factor,
        )
    end
    analyses
end

# ╔═╡ a7c6d7af-cec7-4c4c-adb6-e2b370a49042
pair_consensus = Dict(pair_label => mft.consensus_group_velocity_picks(
    analysis;
    correlation_threshold=0.0,
    velocity_tolerance_fraction=0.10,
    cluster_tolerance_fraction=nothing,
    max_candidates=5,
    selection_mode=:low_velocity,
    min_candidate_periods=3,
    max_smooth_jump_fraction=0.08,
    max_gap_periods=1,
) for (pair_label, analysis) in pair_mft_analyses)

# ╔═╡ ec938992-7a8a-45e0-b38e-4ba40bc7dfdc
md"Computed MFT consensus candidates for **$(length(pair_consensus))** receiver pairs."

# ╔═╡ b350e7a5-ae7e-46ca-a246-d60b66a68e17
md"## Geometry-Aware Tomography Candidate Mixes"

# ╔═╡ b68a7252-510c-40c3-825a-d004a51a4cc4
tomography_pair_inputs = begin
    inputs = mft.PairConsensusForTomography[]
    for pair_label in sort(collect(keys(pair_consensus)))
        item = first([x for x in run_source_state_averages if x.pair_label == pair_label])
        if !isnothing(item.latitudes) && !isnothing(item.longitudes)
            push!(inputs, mft.tomography_pair_consensus(
                item.pair,
                pair_consensus[pair_label];
                latitudes=item.latitudes,
                longitudes=item.longitudes,
                distance=item.distance,
                label=pair_label,
            ))
        end
    end
    inputs
end

# ╔═╡ a38d32af-7c17-4f4e-ac6f-2bfcc5eb737e
tomography_candidate_mixes = mft.tomography_candidate_mixes(
    tomography_pair_inputs;
    max_mix_parts=3,
    min_candidate_periods=3,
    midpoint_radius_km=75.0,
    azimuth_tolerance_deg=25.0,
    distance_tolerance_fraction=0.35,
    velocity_tolerance_fraction=0.10,
)

# ╔═╡ cfcb4a13-2a2e-4f4d-a008-f864292d059e
function candidate_mix_table(mixes; n::Int=25)
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

# ╔═╡ d8609990-6852-40c4-81e5-59474f5ebd7c
candidate_mix_table(tomography_candidate_mixes; n=30)

# ╔═╡ e89d81cb-8596-4364-8241-01578fb81c6b
md"## Quick Plots"

# ╔═╡ d7a8effd-1bc2-4f21-a947-7e8bbf82349a
@bind selected_plot_pair Select(pair_labels)

# ╔═╡ e3767110-37f7-4e37-a01f-93f72dcda465
if isempty(pair_labels) || !(selected_plot_pair in keys(pair_mft_analyses))
    md""
else
    WideCell(mft.plot_consensus_groupvelocity_picks(
        pair_mft_analyses[selected_plot_pair],
        pair_consensus[selected_plot_pair];
        correlation_threshold=0.0,
        velocity_tolerance_fraction=0.10,
        title="Trained VQ-VAE source-state consensus $(selected_plot_pair)",
    ))
end

# ╔═╡ aa37837d-6ba8-4356-acfa-452e10b03a50
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
        title="Top trained-VQ-VAE tomography candidate mixes",
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

# ╔═╡ f06d6be1-7986-4d46-bb9a-dd634a6a44c5
WideCell(plot_top_tomography_mixes(tomography_candidate_mixes, tomography_pair_inputs; n=20))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-4ee838fcccc9c8e"
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
# ╠═a0f8a2b4-8fb5-4f06-bda6-c362a61065a1
# ╠═c7d47e38-e24f-4b40-b3a4-bc894188a750
# ╟─b00fd94f-291e-46d8-84ff-48f8606c2a1e
# ╠═dcbf026e-957a-4b9b-9757-bd0638a25b26
# ╟─d50d63be-d58b-4704-8211-ed7875e04857
# ╠═c7ec82e0-d5b6-4f31-8600-c8b1d276dc92
# ╠═e63099d0-5fb5-43b0-967c-b7c468dc4f83
# ╠═ed704922-12f0-475b-b628-19a9c37bca7a
# ╠═e216a473-6433-4658-b2b7-a4eaa670cc5e
# ╠═b5a1cf7d-d464-4409-b43a-074c8aa22108
# ╟─a4c73d31-cada-44dd-81c8-fbb0f5e84f6a
# ╟─e4499887-1a64-4eaa-a599-4ed4941a7b2d
# ╠═e4f7b3cf-7f26-4ea9-b9bb-95f3df9e7790
# ╠═02d134b5-7ce3-47a9-86ef-a43e6c52287a
# ╠═ddfd42a8-4ae7-408a-8b8f-2d335746798b
# ╟─b7c7a358-71c8-4797-a259-68bc75ab6e65
# ╟─eec83733-193d-4f52-9a75-e6f1d03c7aa5
# ╠═067d2587-8eb1-41c3-95f8-9f785171f2ce
# ╠═b01af348-c4b0-4ad1-81cd-116e9f2ed765
# ╠═c61a6cfe-b14e-4aa9-a711-450e35a3a9bd
# ╠═a7c6d7af-cec7-4c4c-adb6-e2b370a49042
# ╟─ec938992-7a8a-45e0-b38e-4ba40bc7dfdc
# ╟─b350e7a5-ae7e-46ca-a246-d60b66a68e17
# ╠═b68a7252-510c-40c3-825a-d004a51a4cc4
# ╠═a38d32af-7c17-4f4e-ac6f-2bfcc5eb737e
# ╠═cfcb4a13-2a2e-4f4d-a008-f864292d059e
# ╠═d8609990-6852-40c4-81e5-59474f5ebd7c
# ╟─e89d81cb-8596-4364-8241-01578fb81c6b
# ╠═d7a8effd-1bc2-4f21-a947-7e8bbf82349a
# ╠═e3767110-37f7-4e37-a01f-93f72dcda465
# ╠═aa37837d-6ba8-4356-acfa-452e10b03a50
# ╠═f06d6be1-7986-4d46-bb9a-dd634a6a44c5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
