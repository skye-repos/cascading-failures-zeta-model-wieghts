using Pkg
Pkg.activate(".")
using Distributed
addprocs(10)
@everywhere begin
    include("simulate.jl")
    include("utility.jl")
    using SharedArrays
    using Printf
end

"""
    simulate_cascade(
    graph::Graph;
    α::Number=α,
    p_min::Float64=p_min,
    n_pts::Int=n_pts,
    θ_list::AbstractVector)

Simulate the cascading-failure for a given `graph` and `θ` value.
"""
function simulate_cascade(
    graph::Graph;
    α::Number=α,
    p_min::Float64=p_min,
    n_pts::Int=n_pts,
    θ_list::AbstractVector)

    ## initial stuff
    l_θ = length(θ_list)
    rem_list::AbstractVector, chunk::Int = seed_failure(graph, p_min, n_pts)

    ## Shared Arrays for distributed processing
    S_results = SharedArray{Float64}(n_pts, l_θ)
    t_results = SharedArray{Float64}(n_pts, l_θ)
    p_results = SharedArray{Float64}(n_pts, l_θ, round(Int, 8 * k))

    @sync @distributed for i ∈ 1:l_θ
        θ = θ_list[i]
        weight_function(g) = θ_weights(g, θ=θ)

        capacity = (1 + α) * betweenness_centrality(
            graph, weight_function(graph);
            normalize=false)

        for n ∈ 1:n_pts
            h = copy(graph)
            removed = rem_list[1:n*chunk]

            for (u, v) ∈ removed
                remove_link!(h, u, v)
            end

            h, t = cascade!(h, capacity, weight_function)
            S = length(giant_component(h)) / nv(h)
            S_results[(n_pts-n+1), i] = S
            t_results[(n_pts-n+1), i] = t

            kₑ, pₑ = degree_distribution(h)
            for j ∈ eachindex(kₑ)
                p_results[(n_pts-n+1), i, j] = pₑ[j]
            end

            @info(@sprintf("%2d/%2d p's, θ = %.2f",
                n, n_pts, θ))
        end
    end

    k_max = 0
    for i ∈ eachindex(θ_list), n ∈ 1:n_pts
        km = findlast(!iszero, p_results[n, i, :])
        if km > k_max
            k_max = km
        end
    end

    p_results = p_results[:, :, 1:k_max]

    return S_results, t_results, p_results
end

const L::Int = 50
const ζ::Float64 = 15
const k::Float64 = 5
const α::Float64 = 2.0

const g::Graph = ζ_taurus(L, ζ, k);
const θ_list = append!(
    collect(-0.25:0.05:-0.05),
    collect(0.05:0.05:1.00))

const n_pts = 50

if ζ / L < 0.01
    const p_min = 0.85
else
    const p_min::Float64 = 0.5
end

const p_list = collect(range(p_min, 1.00, n_pts))

@time S_matrix, t_matrix, p_matrix = simulate_cascade(g, θ_list=θ_list)

CSV_dir = "results/CSV/L = $(L)/ζ = $(ζ), k = $(k), α = $(α)/"
plots_dir = "results/plots/L = $(L)/ζ = $(ζ), k = $(k), α = $(α)/"

if !ispath(CSV_dir)
    mkpath(CSV_dir)
end

if !ispath(plots_dir)
    mkpath(plots_dir)
end

id_gt = "0 - Gt Component " * join(string.(θ_list), ", ")
id_ct = "1 - Cascade Time " * join(string.(θ_list), ", ")

write_to_file(S_matrix,
    file_name=CSV_dir * id_gt * ".csv",
    header=p_list)

write_to_file(t_matrix,
    file_name=CSV_dir * id_ct * ".csv",
    header=p_list)

S_plot = results_plotter(S_matrix, ylabel="S", title="Giant component L = $(L), ζ = $(ζ)");
savefig(plots_dir * id_gt * ".png")
t_plot = results_plotter(t_matrix, ylabel="Cascade Time", title="Cascade Time L = $(L), ζ = $(ζ)");
savefig(plots_dir * id_ct * ".png")

colorscheme = [cgrad(:viridis)[z] for z ∈ range(0, 1, length=n_pts)]

hmp_S = heatmap(p_list, θ_list, S_matrix',
    c=colorscheme, size=(800, 500),
    title="GC heatmap: L = $(L) ζ = $(ζ)",
    xlabel="p", ylabel="θ")
savefig(hmp_S, plots_dir * "0 - Gt Component heatmap ζ = $(ζ).png")

for (i, θ) ∈ pairs(θ_list)
    ## Degree vector
    kₘ = size(p_matrix)[3]
    kᵥ = collect(1:kₘ)

    id_p = @sprintf("degree-distribution θ=%.2f, ζ=%.1f",
        θ, ζ)

    write_to_file(p_matrix[:, i, :],
        file_name=CSV_dir * id_p * ".csv",
        header=p_list)

    ## Heatmaps
    hmp = heatmap(kᵥ, p_list, p_matrix[:, i, :],
        c=:viridis, size=(800, 500),
        title="θ=$(θ), Heatmap of final pₖ",
        xlabel="degree", ylabel="p")
    savefig(hmp, plots_dir * "heatmap-final-dists-θ = $(θ), ζ = $(ζ).png")

    ## Individual Plots
    plt = plot(title="θ=$(θ), Comparing final pₖ",
        size=(800, 500), legend_columns=2,
        legendfontsize=6, legend=:topright)

    for (n, p) ∈ pairs(p_list)
        local pₖ = p_matrix[n, i, :]
        plt = plot!(plt, kᵥ, pₖ,
            color=colorscheme[n],
            label=@sprintf("p=%.2f", p))
    end
    savefig(plt, plots_dir * "comparison-final-dists-θ = $(θ).png")
end
