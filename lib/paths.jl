using Base: between, normalize_typevars
include("graph.jl")
using DataStructures

struct DijkstraState{T<:Number,U<:Integer}
    parents::Vector{U}
    dists::Vector{T}
    predecessors::Vector{Vector{U}}
    pathcounts::Vector{Float64}
    closest_vertices::Vector{U}
end

function dijkstra(
    g::Graph,
    srcs::Vector{U},
    weights::AbstractMatrix{T};
    all_paths=false,
    track_vertices=false,
    maxdist=typemax(T),
) where {T<:Number} where {U<:Integer}
    N = size(g)
    dists = fill(maxdist, size(g))
    parents = zeros(U, N)
    visited = zeros(Bool, N)

    pathcounts = zeros(N)
    preds = fill(Vector{U}(), N)
    H = PriorityQueue{U,T}()

    for src in srcs
        dists[src] = zero(T)
        visited[src] = true
        pathcounts[src] = one(Float64)
        H[src] = zero(T)
    end

    closest_vertices = Vector{U}()
    sizehint!(closest_vertices, N)

    while !isempty(H)
        u = dequeue!(H)

        if track_vertices
            push!(closest_vertices, u)
        end

        d = dists[u]

        for v in neighbors(g, u)
            alt = d + weights[u, v]

            alt > maxdist && continue

            if !visited[v]
                visited[v] = true
                dists[v] = alt
                parents[v] = u

                pathcounts[v] += pathcounts[u]

                if all_paths
                    preds[v] = [u;]
                end

                H[v] = alt

            elseif alt < dists[v]
                dists[v] = alt
                parents[v] = u
                pathcounts[v] = pathcounts[u]

                if all_paths
                    resize!(preds[v], 1)
                    preds[v][1] = u
                end

                H[v] = alt

            elseif alt == dists[v]
                pathcounts[v] += pathcounts[u]

                if all_paths
                    push!(preds[v], u)
                end
            end
        end
    end

    if track_vertices
        for i = 1:N
            if !visited[i]
                push!(closest_vertices, i)
            end
        end
    end

    for src in srcs
        pathcounts[src] = one(Float64)
        parents[src] = 0
        empty!(preds[src])
    end

    return DijkstraState{T,U}(parents, dists, preds, pathcounts, closest_vertices)
end


function dijkstra(
    g::Graph,
    src::Integer,
    weights::AbstractMatrix=weights(g);
    all_paths=false,
    track_vertices=false,
    maxdist=typemax(eltype(weights)),
)
    return dijkstra(
        g, [src;], weights; all_paths=all_paths, track_vertices=track_vertices, maxdist=maxdist
    )
end

function betweenness_centrality(
    g::Graph,
    weights::AbstractMatrix;
    nodes::Vector{Int64}=collect(1:size(g)),
    normalize=true,
    end_points=false,
)

    N = size(g)
    k = length(nodes)

    betweenness = zeros(N)
    preds = [Vector{Vector{Int}}() for _ ∈ 1:N]

    for src in nodes
        if degrees(g)[src] > 0
             state = dijkstra(g, src, weights, all_paths=true, track_vertices=true)
            preds[src] = state.predecessors
            if end_points
                _accumulate_endpoints!(betweenness, state, g, src)
            else
                _accumulate_basic!(betweenness, state, g, src)
            end
        end
    end

    _rescale!(betweenness, N, normalize, false, k)

    return betweenness, preds
end

function _accumulate_basic!(
    betweenness::Vector{Float64},
    state::DijkstraState,
    g::Graph,
    si::Integer
)

    n_v = size(g)
    δ = zeros(n_v)
    σ = state.pathcounts
    P = state.predecessors

    P[si] = []

    S = reverse(state.closest_vertices)
    for w in S
        coeff = (1.0 + δ[w]) / σ[w]
        for v in P[w]
            if v > 0
                δ[v] += (σ[v] * coeff)
            end
        end

        if w != si
            betweenness[w] += δ[w]
        end
    end
    return nothing
end

function _accumulate_endpoints!(
    betweenness::Vector{Float64},
    state::DijkstraState,
    g::Graph,
    si::Integer
)

    n_v = size(g)
    δ = zeros(n_v)
    σ = state.pathcounts
    P = state.predecessors

    S = reverse(state.closest_vertices)
    betweenness[si] += length(S) - 1

    for w in S
        coeff = (1.0 + δ[w]) / σ[w]
        for v in P[w]
            δ[v] += σ[v] * coeff
        end
        if w != si
            betweenness[w] += (δ[w] + 1)
        end
    end

    return nothing
end

function _rescale!(
    betweenness::Vector{Float64},
    n::Integer,
    normalize::Bool,
    directed::Bool,
    k::Integer
)

    scale = nothing
    if normalize
        if 2 < n
            scale = 1.0 / ((n - 1) * (n - 2))
        end
    elseif !directed
        scale = 0.5
    end

    if !isnothing(scale)
        if k > 0
            scale = scale * n / k
        end
        betweenness = scale .* betweenness
    end
    return nothing
end

"""
update the betweenness centrality incrementally. Call only after network has been updated!!
"""
function update_betweenness!(
    g::Graph,
    weights::AbstractMatrix,
    betweenness::AbstractVector,
    preds::AbstractVector;
    work_rev::Vector{Vector{Int}},
    work_aff::Vector{Int},
    work_vis::Vector{Bool},
    deleted_list::Vector{Int64},
    normalize=true,
    end_points=false,
)
    N = size(g)

    fill!(work_vis, false)
    empty!(work_aff)
    @inbounds for v ∈ 1:N
        empty!(work_rev[v])
    end

    for v ∈ 1:N, u_list ∈ preds[v], u ∈ u_list
        push!(work_rev[v], u)
    end

    Q = Queue{Int}()
    
    for d ∈ deleted_list
        work_vis[d] = true
        enqueue!(Q, d)
    end

    while !isempty(Q)
        u = dequeue!(Q)
        push!(work_aff, u)

        for v ∈ work_rev[u]
            if !work_vis[v]
                work_vis[v] = true
                enqueue!(Q, v)
            end
        end
    end

    new_betw, new_pred = betweenness_centrality(g, weights,
        nodes=work_aff,
        normalize=normalize,
        end_points=end_points)

    for node ∈ work_aff
        betweenness[node] = new_betw[node]
        preds[node] = new_pred[node]
    end

    return nothing
end

function components(label::Vector{T}) where {T<:Integer}
    d = Dict{T,T}()
    c = Vector{Vector{T}}()
    i = one(T)

    for (v, l) in enumerate(label)
        index = get!(d, l, i)
        if length(c) ≥ index
            push!(c[index], v)
        else
            push!(c, [v])
            i += 1
        end
    end

    return c, d
end

function connected_componenents(g::Graph)
    label = zeros(Int, size(g))

    for u ∈ collect(1:size(g))
        label[u] != 0 && continue
        label[u] = u
        Q = Vector{Int}()
        push!(Q, u)

        while !isempty(Q)
            src = popfirst!(Q)
            for v in neighbors(g, src)
                if label[v] == 0
                    push!(Q, v)
                    label[v] = u
                end
            end
        end
    end

    c, _ = components(label)

    return c
end

function giant_component(g::Graph)
    components = connected_componenents(g)
    _, i = findmax(length.(components))

    return components[i]
end
