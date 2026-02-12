include("graph.jl")

"""
    connected_graph(N0::Int)

Computes the edge list for a connected graph with N0 nodes
"""
function connected_graph(N0::Int)
    g = Graph(N0)
    for i ∈ 1:N0, j ∈ i+1:N0
        add_link!(g, i, j)
    end
    return g
end

"""
    barabasi_albert(m::Integer, N::Integer)

Generate a scale-free barabasi-albert graph with `N' nodes and `m' links per new node
"""
function barabasi_albert(m::Integer, N::Integer)
    g = connected_graph(m+1)   
    pool = vcat([fill(i, degrees(g)[i]) for i in 1:nv(g)]...)

    for source ∈ m+2:N
        targets = Set{Int64}()

        while length(targets) < m
            push!(targets, rand(pool))
        end

        for t ∈ targets
            add_link!(g, source, t)
            push!(pool, t)
            push!(pool, source)
        end
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

    n_edges = Int(ceil(N * (N - 1) * p / 2))

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
