include("graph.jl")

"""
    connected_graph(N0::Int)

Computes the edge list for a connected graph with N0 nodes
"""
function connected_graph(N0::Int)
    adjacency_list::Vector{Vector{Int}} = []
    nodes = collect(1:N0)
    for i = 1:N0
        push!(adjacency_list, filter(e->e≠i, nodes))
    end
    return Graph(adjacency_list)
end

"""
    ba_step!(g::Graph, m::Int64)

Computes the next barabasi-albert iteration of a network `g` where a new node
brings with it `m` half links.
"""
function ba_step!(g::Graph, m::Int64)
    # Create a pool of nodes weighted by their degree
    pool = vcat([fill(i, degrees(g)[i]) for i in 1:nv(g)]...)
    targets = Int[]
    while length(targets) < m
        v = rand(pool)
        v ∉ targets && push!(targets, v)
    end

    for v in targets
        push!(g.adjacency_list[v], nv(g)+1)
    end

    push!(g.adjacency_list, targets)

    return g
end

"""
    barabasi_albert!(g::Graph, m::Int64, n_steps::Int64)

Given an initial network `g`, computes the Barabasi-Albert graph after `n_steps`
nodes are added each of which brings with it `m` half-links
"""
function barabasi_albert!(g::Graph, m::Int64, n_steps::Int64)
    for _ = 1:n_steps
        ba_step!(g, m)
    end
    return g
end

"""
    erdos_renyi(N, p)

Generate Erdos-Renyi random network G(N, p)
""" 
function erdos_renyi(N, p)
    adjacency_list = [Vector{Int64}() for _ = 1:N]
    g = Graph(adjacency_list)

    n_edges = Int(ceil(N*(N-1)*p/2))

    if p < 0.1
        for _ = 1:n_edges
            i = Int.(rand(1:N))
            j = Int.(rand(1:N))
            while j ∈ adjacency_list[i]
                j = Int.(rand(1:N))
            end
            add_link!(g, Int(i), Int(j))
        end
    else
        for i = 1:N
            for j = 1:N
                u = rand()
                if u < p
                    add_link!(g, Int(i), Int(j))
                end
            end
        end
    end
    return g
end

"""
    ζ_taurus(L::Integer, ζ::Number, k_avg::Number)

Generate a ζ-model network of side length `L`, characteristic link-length of
`ζ`, and average degree `k`
"""
function ζ_taurus(L::Integer, ζ::Number, k_avg::Number)
    local N = L^2

    coords = CartesianIndices((1:L, 1:L))

    @inline function distance(
        a::CartesianIndex{2},
        b::CartesianIndex{2})
        du = abs(a[1] - b[1])
        dv = abs(a[2] - b[2])
        dx = min(du, L - du)
        dy = min(dv, L - dv)

        return sqrt(dx^2 + dy^2)
    end

    s = 0.0
    for i ∈ 2:N
        s += exp(-distance(coords[1], coords[i]) / ζ)	    
    end
    coeff = k_avg / s

    g = Graph([Vector{Int64}() for _ ∈ 1:N])

    @inbounds for j ∈ 1:N, i ∈ 1:j-1
        prob = coeff * exp(-distance(coords[i], coords[j]) / ζ)
        rand() < prob && add_link!(g, Int(i), Int(j))
    end

    return g
end

"""
    ζ_sparse_taurus(L::Integer, ζ::Number, k_avg::Number)

Given a network with L^2 nodes on a taurus, that is sparse with k_avg links per
node & ζ as characteristic link length, generate a graph. Since the network is sparse, populate all N * k_avg / 2 links directly.
"""
function ζ_sparse_taurus(L::Integer, ζ::Number, k_avg::Number)
    local N = L^2

    coords = CartesianIndices((1:L, 1:L))

    @inline function distance(
        a::CartesianIndex{2},
        b::CartesianIndex{2})
        du = abs(a[1] - b[1])
        dv = abs(a[2] - b[2])
        dx = min(du, L - du)
        dy = min(dv, L - dv)

        return sqrt(dx^2 + dy^2)
    end
    
    adj = [Vector{Int64}() for _ ∈ 1:N]

    ne = ceil(N * k_avg / 2)
    ec = 0
    while ec < ne
        i::Int = rand(1:N)
        j::Int = rand(1:N)

        if i == j || j ∈ adj[i]
            continue
        end

        prob = exp(-distance(coords[i], coords[j]) / ζ)

        if rand() < prob
            push!(adj[i], j)
            push!(adj[j], i)
            ec += 1
        end
    end

    return Graph(adj)
end
