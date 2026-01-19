using Distributed
addprocs(4)
@everywhere begin
    include("simulate.jl")
    include("utility.jl")
    using SharedArrays
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

    l_θ = length(θ_list)
    rem_list::AbstractVector, chunk::Int = seed_failure(graph, p_min, n_pts)
    
    S_results = SharedArray{Float64}(n_pts, l_θ)
    t_results = SharedArray{Float64}(n_pts, l_θ)
    @sync @distributed for n ∈ 1:l_θ
        θ = θ_list[n]
        weight_function(g) = θ_weights(g, θ=θ)

        # capacity = (1 + α) * betweenness_centrality_sample(
        #     graph, weight_function(graph);
        #     k = 200)
        capacity = (1 + α) * betweenness_centrality(
            graph, weight_function(graph);
            normalize=false)

        for i ∈ 1:n_pts
            h = copy(graph)
            removed = rem_list[1:i*chunk]

            for (u, v) ∈ removed
                remove_link!(h, u, v)
            end

            h, t = cascade!(h, capacity, weight_function)
            S = length(giant_component(h)) / nv(h)
            S_results[i, n] = S
            t_results[i, n] = t
            @info "Finished $(i)/$(n_pts) p values for θ = $(θ)"
        end
    end

    return S_results, t_results
end

const L::Int = 15
const ζ::Float64 = 1
const k::Float64 = 5
const α::Float64 = 2.0

const g::Graph = ζ_sparse_taurus(L, ζ, k);
const θ_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

const n_pts = 20

if ζ / L < 0.01
    const p_min = 0.85
else
    const p_min::Float64 = 0.5
end

const p_list = collect(range(p_min, 1.00, n_pts))

@time S_matrix, t_matrix = simulate_cascade(g, θ_list=θ_list)

CSV_dir = "results/CSV/L = $(L)/ζ = $(ζ), k = $(k), α = $(α)/"
plots_dir = "results/plots/L = $(L)/ζ = $(ζ), k = $(k), α = $(α)/"

if !ispath(CSV_dir)
    mkpath(CSV_dir)
end

if !ispath(plots_dir)
    mkpath(plots_dir)
end

id_gt = "Gt Component " * join(string.(θ_list), ", ")
id_ct = "Cascade Time " * join(string.(θ_list), ", ")

write_to_file(S_matrix',
              file_name = CSV_dir * id_gt * ".csv",
              header = p_list)

write_to_file(t_matrix',
              file_name = CSV_dir * id_ct * ".csv",
              header = p_list)

S_plot = results_plotter(S_matrix, ylabel = "S", title = "Giant component L = $(L), ζ = $(ζ)")
savefig(plots_dir * id_gt * ".png")
t_plot = results_plotter(t_matrix, ylabel = "Cascade Time", title = "Cascade Time L = $(L), ζ = $(ζ)")
savefig(plots_dir * id_ct * ".png")
