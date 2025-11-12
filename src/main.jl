using Distributed
addprocs(4)
@everywhere include("utility.jl") ## Contains helper functions and miscellaneous items
@everywhere include("graph_builder.jl") ## Contains the code to generate original graph and remove random links
@everywhere include("cascading.jl") ## Contains the code to do the cascading

## First we initialize the constants needed to generate the graph. L is the grid
## size, k is the average degree of the graph, and ζ controls the characteristic
## lengths of the links on the grid. α will determine the node capacity
const L::Int64 = 50
const k::Float64 = 4.0
const ζ::Float64 = 15
const α::Float64 = 2.0

## Generate the graph, will take time for large L and ~ 30 Gb RAM for L =
## 200. Will be stored in ~ 5 Gb of memory
g::Graph = ζ_2d(L, ζ, k);

## List of θ values to run the simulation for.
# θ_list = [-0.5, -0.25, 0.25, 0.75, 1.0, 1.5];
# θ_list = [0.1, 0.25, 0.75, 1.00, 1.25];
θ_list = [0.25, 0.30, 0.35, 0.40, 0.65];


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

if !ispath(graph_dir)
    mkpath(graph_dir)
end
if !ispath(plots_dir)
    mkpath(plots_dir)
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
    for (i, θ) ∈ pairs(θ_list)
        weight_func(g) = θ_weights(g, θ=θ)
        local capacity = (1 + α) * loads(g, weights=weight_func(g))
        local h, _, t = cascade!(copy(graph), capacity, weight_func)
        local S = length(giant_component(h)) / size(h)
        sol[i] = S
        t_list[i] = t
    end
    return (sol, t_list)
end

S_results = [r[1] for r ∈ all_results];
t_results = [r[2] for r ∈ all_results];


## Here we store the results in a more usable matrix format. A[i, :] will give the results for the i'th value in the θ_list
S_matrix = vecvec_to_matrix(S_results)
t_matrix = vecvec_to_matrix(t_results)

## Now we write those results to disk
S_CSV_file = CSV_dir * "Gt Component " * join(string.(θ_list), ", ") * ".csv";
write_to_file(S_matrix, file_name=S_CSV_file, header=p_list)

t_CSV_file = CSV_dir * "Cascade Time " * join(string.(θ_list), ", ") * ".csv";
write_to_file(t_matrix, file_name=t_CSV_file, header=p_list)

## Plot the results
S_plot = results_plotter(S_matrix, "S")
S_plot_file = plots_dir * "Gt Component θ comparison " * join(string.(θ_list), ", ") * ".png";
savefig(S_plot_file);

t_plot = results_plotter(t_matrix, "Cascade time")
t_plot_file = plots_dir * "Cascade time θ comparison " * join(string.(θ_list), ", ") * ".png";
savefig(t_plot_file);
