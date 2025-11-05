using Distributed
addprocs(4)
@everywhere include("graph_builder.jl")
@everywhere include("cascading.jl")

const L::Int64 = 50
const k::Float64 = 4.0
const ζ::Float64 = 15
const α::Float64 = 2.0

g::Graph = ζ_2d(L, ζ, k);
# θ_list = [-0.5, -0.25, 0.25, 0.75, 1.0, 1.5];
θ_list = [0.1, 0.25, 0.75, 1.00, 1.25];
# θ_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
labels = permutedims(["θ = $(θ)" for θ ∈ θ_list])

if ζ < 1
    p_list::Vector{Float64} = collect(0.90:0.005:0.99);
else
    p_list::Vector{Float64} = collect(0.30:0.025:0.99);
end

graph_list, ordered_p_list = remove_random_links_p_list!(g, p_list)

all_results = pmap(graph_list) do graph
    sol = Vector{Float64}(undef, length(θ_list))
    t_list = Vector{Float64}(undef, length(θ_list))
    for (i, θ) ∈ enumerate(θ_list)
        weight_func(g) = θ_weights(g, θ=θ)
        local capacity = (1+α) * loads(g, weights=weight_func(g))
        local h, _, t = cascade!(copy(graph), capacity, weight_func)
        local S = length(giant_component(h)) / size(h)
        sol[i] = S
        t_list[i] = t
    end
    return (sol, t_list)
end

results = [r[1] for r ∈ all_results]

sol_list = []
sol_list = vcat(sol_list, results[1])

for n = 2:length(results)
    global sol_list = hcat(sol_list, results[n])
end

sol = [sol_list[i, :] for i = 1:length(θ_list)]

using Plots

plot_sol = plot(ordered_p_list, sol;
                xlabel="p", ylabel="S",
                title="L=$(L), α=$(α), ζ = $(ζ), k=$(k)",
                label=labels,
                palette=:dracula,
                linestyle=:dashdot, linewidth=1.5,
                marker=:circle, markersize=2)

plot_dir = "../plots/L=$(L), ζ=$(ζ), k=$(k), α=$(α)"

if !ispath(plot_dir)
    mkpath(plot_dir)
end

file_name = "θ comparison " * join(string.(θ_list), ", ") * ".png"

savefig(joinpath(plot_dir, file_name))
