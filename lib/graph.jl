abstract type AbstractGraph end

"""
An un-directed Graph for a simple network specified by an adjacency list (no multi-links)
"""
mutable struct Graph <: AbstractGraph
    adjacency_list::Vector{Vector{Int64}}
end

"""
    Graph(N = 1)

Construct a network with one node and no links by default. Optionally specify `N'
as the number of nodes.
"""
function Graph(N = 1)
	return Graph([Int64[] for _ ∈ 1:N])
end

"""
    nv(g::Graph)

Number of nodes in the network `g`
"""
function nv(g::Graph)
    return length(g.adjacency_list)
end

"""
    degrees(g::Graph)

Vector of degrees for a network `g`
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
    copy(g::Graph)

Create a copy of a mutable Graph
"""
function copy(g::Graph)
    a = Base.deepcopy(g.adjacency_list)
    return Graph(a)
end

"""
    neighbors(g::Graph, node::Int64)

Returns a vector of neighbors for `node` in network `g`
"""
function neighbors(g::Graph, node::Int64)
    return g.adjacency_list[node]
end

"""
    degree_average(g::Graph)

Average degree of a network `g`
"""
function degree_average(g::Graph)
    return sum(degrees(g)) / nv(g)
end

"""
    degree_distribution(g::Graph)

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
    remove_link!(g::Graph, n1::Int64, n2::Int64)

Modify a network `g` in-place to remove a link between (`n1`, `n2`)
"""
function remove_link!(g::Graph, n1::Int64, n2::Int64)
    idx1 = findfirst(v -> v == n2, g.adjacency_list[n1])
    if idx1 ≠ nothing
        deleteat!(g.adjacency_list[n1], idx1)
    end

    idx2 = findfirst(v -> v == n1, g.adjacency_list[n2])
    if idx2 ≠ nothing
        deleteat!(g.adjacency_list[n2], idx2)
    end
    
    return nothing
end

"""
    add_link!(g::Graph, n1::Int64, n2::Int64)

Modify a network `g` in-place to add a link between (`n1`, `n2`)
"""
function add_link!(g::Graph, n1::Int64, n2::Int64)
    max_n = max(n1, n2)

    if nv(g) < max_n
        old_nv = nv(g)
        resize!(g.adjacency_list, max_n)

        for i ∈ (old_nv + 1):max_n
            g.adjacency_list[i] = Int64[]
        end
    end
    
    push!(g.adjacency_list[n1], n2)
    push!(g.adjacency_list[n2], n1)

    return nothing
end

"""
    edges(g::Graph; directed=false)

Get the edge-list from the adjacency list of a network `g`
"""
function edges(g::Graph; directed=false)
    el = Set{Tuple}()

    for node ∈ nodes(g), nbr ∈ neighbors(g, node)
        directed ? push!(el, (node, nbr)) : push!(el, minmax(node, nbr))
    end

    return collect(el)
end

"""
    graph_from_edges(el::Vector{Tuple})

Converts an edge-list to a Graph
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
    nodes(g::Graph; directed = false)

Set of nodes that have links in a network `g`. Use collect for vector/array.
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
    ne(g::Graph)

Number of edges in the network `g`
"""
function ne(g::Graph)
    return length(edges(g))
end
