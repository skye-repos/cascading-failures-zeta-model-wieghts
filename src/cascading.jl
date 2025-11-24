include("utility.jl")

"""
Calculate the load on each node of the network `g` using its
betweenness-centrality given `weights`
"""
function loads(g::Graph; weights::AbstractMatrix)
    loads, _ = betweenness_centrality(g, weights; normalize=false)
    return loads
end


"""
Returns a modified network `g` (changes in place), and nodes that failed per
iteration step

Cascade on a network `g` with max_loads `capacities` and link weights `weights`
until no more nodes fail
"""
function cascade!(
    g::Graph,
    capacities::Vector{Float64},
    weight_function=gaussian_weights,
)

    N = size(g)
    t_step = 1
    failed_nodes_per_step::Vector{Vector{Int64}} = []

    while true
        local failed_nodes::Vector{Int64} = []
        local loads0, _ = betweenness_centrality(g, weight_function(g), normalize=false)

        for node = 1:N
            loads0[node] > capacities[node] ? push!(failed_nodes, node) : nothing
        end

        length(failed_nodes) == 0 ? break : push!(failed_nodes_per_step, failed_nodes)

        for node ∈ failed_nodes, neighbor ∈ neighbors(g, node)
            remove_link!(g, node, neighbor)
        end

        t_step += 1
    end

    return g, failed_nodes_per_step, t_step
end
