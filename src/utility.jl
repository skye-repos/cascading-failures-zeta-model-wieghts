include("../lib/graph.jl")
include("../lib/paths.jl")
using Serialization
using Tables, CSV
using Plots

function θ_weights(g::Graph; θ=1.0)
    k = degrees(g)
    w = [(k[i] * k[j])^θ
         for i = 1:size(g), j = 1:size(g)]
    return w
end

function gaussian_weights(g::Graph; σ=1.0, μ=0.0)
    N = size(g)
    adj = g.adjacency_list
    x = collect(-σ:0.01:σ) .+ μ
    gauss = [exp(-(i - μ)^2 / (2σ^2)) / (σ * (sqrt(2π))) for i in x]

    w = fill(0.0, (N, N))
    for u ∈ adj, v ∈ neighbors(g, u)
        w[u, v] = rand(gauss)
    end

    return w
end

"""
Detect transition point in series data `sol` and `p_list` by identifying when
the change in the second derivative is the maximum & it changes sign
"""
function detect_critical_point(sol::Vector{Float64}, p_list::Vector{Float64})
    length(p_list) ≠ length(sol) ? error("Please supply equal length arrays for `sol` and `p_list`") : nothing

    h = [p_list[i+1] - p_list[i] for i ∈ eachindex(p_list)[1:end-1]]
    diffs = []
    critical_point = 1

    for n ∈ eachindex(sol)[1:end-2]
        local val = (sol[n+2] - 2 * sol[n+1] + sol[n]) / (h[n])^2
        push!(diffs, val)
    end

    glob_max = 0

    for n ∈ eachindex(diffs)[1:end-1]
        local max = 0
        (diffs[n+1] > 0 && diffs[n] < 0) || (diffs[n+1] < 0 && diffs[n] > 0) ? max = abs(diffs[n+1] - diffs[n]) : nothing

        if max > glob_max
            glob_max = max
            critical_point = n + 1
        end
    end

    return critical_point, p_list[critical_point]
end

"""
Convert a vector of vectors to a matrix
"""
function vecvec_to_matrix(results::Vector{Vector{Float64}})
    sol = vcat([], results[1])

    for n ∈ eachindex(results)[2:end]
        sol = hcat(sol, results[n])
    end

    return sol
end

"""
Write a matrix `result` to file with name `file_name` and header `header`
"""
function write_to_file(result::AbstractMatrix; file_name::AbstractString, header::Vector{Float64})
    table = Tables.table(result)
    header = string.(header)
    CSV.write(file_name, table, header=header)

    return nothing
end

"""
Plot the results for either GC or Cascade time against θ values
"""
function results_plotter(results::AbstractMatrix, ylabel::String; θ_list=θ_list, p_list=p_list)
	labels = permutedims(["θ = $(θ)" for θ ∈ θ_list]);

    res = [results[i, :] for i ∈ eachindex(θ_list)]
    results_plot = plot(p_list, res;
                        xlabel="p", ylabel=ylabel,
                        title="L=$(L), α=$(α), ζ = $(ζ), k=$(k)",
                        label=labels,
                        palette=:twelvebitrainbow,
                        linestyle=:dashdot, linewidth=1.5,
                        marker=:circle, markersize=2)

    return results_plot
end

