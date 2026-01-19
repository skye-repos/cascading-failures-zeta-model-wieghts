abstract type AbstractGraph end

"""
An un-directed graph for a simple network specified by an adjacency list (no multi-links)
"""
mutable struct Graph <: AbstractGraph
    adjacency_list::Vector{Vector{Int64}}
end

"""
Gets the number of nodes in the network `g`
"""
function nv(g::Graph)
    return length(g.adjacency_list)
end

"""
Gets a vector of node degrees for a network `g`
"""
function degrees(g::Graph)
    N = nv(g)
    d = zeros(Int64, N)
	for (i, list) in enumerate(g.adjacency_list)
        d[i] = length(list)
    end

    return d
end

"""
Create a copy of a mutable graph
"""
function copy(g::Graph)
    a = Base.deepcopy(g.adjacency_list)
    return Graph(a)
end

"""
Returns a vector of neighbors for `node` in network `g`
"""
function neighbors(g::Graph, node::Int64)
    return g.adjacency_list[node]
end

"""
Average degree of a network `g`
"""
function k_average(g::Graph)
    return sum(degrees(g)) / nv(g)
end

"""
Compute the degree distribution of a network `g`.
Returns two objects - the first is the list of degrees
and the second is the distribution
"""
function degree_distribution(g::Graph)
    k_max = maximum(degrees(g))
    k_min = minimum(degrees(g))
    p = zeros(k_max)
    for k ∈ degrees(g)
        if k ≠ 0
            p[k] += 1 / nv(g)
        end
    end
    
    k_min == 0 ? k_min = 1 : nothing
    
    return collect(k_min:k_max), p[k_min:k_max]
end

"""
Modify a graph `g` in-place to remove a link between (`n1`, `n2`)
"""
function remove_link!(g::Graph, n1::Int64, n2::Int64)
    g.adjacency_list[n1] = filter(e -> e ≠ n2, g.adjacency_list[n1])
    g.adjacency_list[n2] = filter(e -> e ≠ n1, g.adjacency_list[n2])
    return nothing
end

"""
Modify a graph `g` in-place to add a link between (`n1`, `n2`)
"""
function add_link!(g::Graph, n1::Int64, n2::Int64)
    if n1 > nv(g) || n2 > nv(g)
        resize!(g.adjacency_list, max(n1, n2))
        g.adjacency_list[n1] = Vector{Int}()
        g.adjacency_list[n2] = Vector{Int}()
        push!(g.adjacency_list[n1], n2)
        push!(g.adjacency_list[n2], n1)
    else
        push!(g.adjacency_list[n1], n2)
        push!(g.adjacency_list[n2], n1)
    end
    return nothing
end

"""
Get the edge-list from the adjacency list of a graph `g`
"""
function edges(g::Graph; directed=false)
    el = Set{Tuple}()

    for node ∈ nodes(g), nbr ∈ neighbors(g, node)
        directed ? push!(el, (node, nbr)) : push!(el, minmax(node, nbr))
    end

    return collect(el)
end

"""
Converts an edge-list to a graph
"""
function graph_from_edges(el::Vector{Tuple})
    N = max(maximum(first.(el)), maximum(last.(el)))
    adj = [Vector{Int}() for _ = 1:N]
    g = Graph(adj)
    for (i, j) in el
        add_link!(g, Int(i), Int(j))
    end
    return g
end

"""
Returns a list of nodes that have links in a network `g`
"""
function nodes(g::Graph; directed = false)
    r = Set{Int}()
    adj = g.adjacency_list

    for i ∈ eachindex(adj)
        if adj[i] ≠ []
            push!(r, i)
        end

        if directed
            for j ∈ eachindex(adj)
                if i ∈ adj[j]
                    push!(r, i)
                end
            end
        end
    end
        
    return r
end

"""
Count the number of edges in the graph `g`
"""
function ne(g::Graph)
    return length(edges(g))
end
