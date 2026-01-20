include("lib/network-generator.jl")
include("lib/paths.jl")
using StatsBase
using SparseArrays
   
"""
    θ_weights(g::Graph; θ=1.0)

Compute degree based weights for sparse network `g` with `θ`.
"""
function θ_weights(g::Graph; θ=1.0)
    k = degrees(g)
    el = edges(g)
    N = nv(g)
    
    # Pre-allocate vectors for the sparse matrix components
    # We use 2 entries per edge (i->j and j->i)
    I = Int[]
    J = Int[]
    V = Float64[]
    
    for (i, j) in el
        val = (k[i] * k[j])^θ
        # Add entry for i,j
        push!(I, i); push!(J, j); push!(V, val)
        # Add entry for j,i
        push!(I, j); push!(J, i); push!(V, val)
    end
    
    # Create the sparse matrix directly without the dense intermediate
    return sparse(I, J, V, N, N)
end

"""
    seed_failure(graph::Graph, p_min::Float64, n_pts::Int)

Given `p_min` (assuming p_max = 1.0) and a full `graph` return a list of edges to remove & the number of edges to be removed at a time so that `n_pts` graphs can be made
"""
function seed_failure(graph::Graph, p_min::Float64, n_pts::Int)
    
    n_e = ne(graph)
    max_rem::Int = ceil((1-p_min) * n_e)
    
    rem_list = sample(edges(graph), max_rem;
                      replace=false)
    Δremoved::Int = floor(length(rem_list) / n_pts)

    return rem_list, Δremoved
end

"""
    cascade!(graph::Graph, capacity::AbstractVector, weight_function)

Compute the failure propagation for ζ model `graph` in the Motter-Lai formalism
with tolerance parameter `α` and weights specified by `weight_function`
"""
function cascade!(graph::Graph, capacity::AbstractVector, weight_function)
    node_list = nodes(graph)
    t_step = 1
    
    while true
        local failed_nodes = Int[]
        # local load = betweenness_centrality_sample(
        #     graph, weight_function(graph);
        #     k = k_sample)

        local load = betweenness_centrality(
            graph, weight_function(graph);
            normalize=false)
    
        for node ∈ node_list
            load[node] > capacity[node] ? push!(failed_nodes, node) : nothing
        end

        if length(failed_nodes) == 0
            break
        end

        for node ∈ failed_nodes, nbr ∈ neighbors(graph, node)
            remove_link!(graph, node, nbr)
        end

        t_step += 1
    end

    return graph, t_step
end
