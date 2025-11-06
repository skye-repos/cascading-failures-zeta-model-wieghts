using Distributed
addprocs(4)
@everywhere include("graph_builder.jl") ## Contains the code to generate original graph and remove random links
@everywhere include("cascading.jl") ## Contains the code to do the cascading

## First we initialize the constants needed to generate the graph. L is the grid
## size, k is the average degree of the graph, and ζ controls the characteristic
## lengths of the links on the grid. α will determine the node capacity
const L::Int64 = 50
const k::Float64 = 4.0
const ζ::Float64 = 10
const α::Float64 = 2.0

## Generate the graph, will take time for large L and ~ 30 Gb RAM for L =
## 200. Will be stored in ~ 5 Gb of memory
g::Graph = ζ_2d(L, ζ, k);

## List of θ values to run the simulation for.
# θ_list = [-0.5, -0.25, 0.25, 0.75, 1.0, 1.5];
# θ_list = [0.1, 0.25, 0.75, 1.00, 1.25];
θ_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
labels = permutedims(["θ = $(θ)" for θ ∈ θ_list]);

## Setting an appropriate plot range for a given ζ scale
if ζ < 1
    p_min, interval, p_max = 0.90, 0.005, 1.00
else
    p_min, interval, p_max = 0.50, 0.010, 1.00
end

## Make directories for storing results
graph_dir = "../graph/L=$(L), ζ=$(ζ), k=$(k)/Δp = $(interval)/";
plots_dir = "../plots/L=$(L), ζ=$(ζ), k=$(k), α=$(α)/";
CSV_dir = "../CSV/L=$(L), ζ=$(ζ), k=$(k), α=$(α)/";

if !ispath(plots_dir)
    mkpath(plots_dir)
end

if !ispath(graph_dir)
    mkpath(graph_dir)
end

if !ispath(CSV_dir)
    mkpath(CSV_dir)
end

## Creating a list of graphs with a seeded failure of (1 - p) amount of link
## removals for p ∈ p_list. We then save the graphs for later inspection.
graphs, p_list = remove_links_p_list!(g, p_min=p_min, p_max=p_max, interval=interval);

for (graph, p) ∈ zip(graphs, p_list)
    local name = joinpath(graph_dir, "p = $(p).bin")
    serialize(name, graph)
end

## We now run the actual simulations parallelized on the CPU
all_results = pmap(graphs) do graph
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

results = [r[1] for r ∈ all_results];
t_results = [r[2] for r ∈ all_results];

## Now we write the results
sol_list = [];
t_list = [];
sol_list = vcat(sol_list, results[1]);
t_list = vcat(t_list, t_results[1]);
for n = 2:length(results)
    global sol_list = hcat(sol_list, results[n])
    global t_list = hcat(t_list, t_results[n])
end

using Tables, CSV
sol_table = Tables.table(sol_list);
t_table = Tables.table(t_list);
csv_sol_file_name = "GC size " * join(string.(θ_list), ", ") * ".csv";
csv_t_file_name = "Cascade Time " * join(string.(θ_list), ", ") * ".csv";
CSV.write(joinpath(CSV_dir, csv_sol_file_name), sol_table, header=string.(p_list));
CSV.write(joinpath(CSV_dir, csv_t_file_name), t_table, header=string.(p_list));

## Plot the results
using Plots
sol = [sol_list[i, :] for i = 1:length(θ_list)];
plot_sol = plot(p_list, sol;
                xlabel="p", ylabel="S",
                title="L=$(L), α=$(α), ζ = $(ζ), k=$(k)",
                label=labels,
                palette=:dracula,
                linestyle=:dashdot, linewidth=1.5,
                marker=:circle, markersize=2)

S_plot_file_name = "GC size θ comparison " * join(string.(θ_list), ", ") * ".png";
savefig(joinpath(plots_dir, S_plot_file_name));


t = [t_list[i, :] for i = 1:length(θ_list)];
plot_sol = plot(p_list, t;
                xlabel="p", ylabel="Cascading time",
                title="L=$(L), α=$(α), ζ = $(ζ), k=$(k)",
                label=labels,
                palette=:dracula,
                linestyle=:dashdot, linewidth=1.5,
                marker=:circle, markersize=2)
t_plot_file_name = "Cascade time θ comparison " * join(string.(θ_list), ", ") * ".png";
savefig(joinpath(plots_dir, t_plot_file_name));

