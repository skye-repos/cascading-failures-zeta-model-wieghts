include("utility.jl")

"""
Builds a lattice network based on the ζ model on a grid of side length `L`,
scale parameter of `ζ`, and avg degree of `k_avg`
"""
function ζ_2d(L::Int64, ζ::Number, k_avg::Number)
    local N = L^2

    coordinates::Vector{Tuple{Int,Int}} = []

    g = Graph([Vector{Int64}() for _ = 1:N])

    for i = 1:L
        for j = 1:L
            push!(coordinates, Int.((i, j)))
        end
    end

    function dist(a, b)
        x = min(abs(a[1] - b[1]), L - abs(a[1] - b[1]))
        y = min(abs(a[2] - b[2]), L - abs(a[2] - b[2]))
        return sqrt(x^2 + y^2)
    end

    w = Matrix{Float64}(undef, N, N)

    for i = 1:N
        for j = i+1:N
            pos1 = coordinates[i]
            pos2 = coordinates[j]
            local r = dist(pos1, pos2)
            w[i, j] = exp(-r / ζ)
        end
    end

    w = w * (N * k_avg / 2) / sum(w)

    for i = 1:N
        for j = i+1:N
            if rand() < w[i, j]
                add_link!(g, Int(i), Int(j))
            end
        end
    end

    return g
end

"""
Remove a portion of links controlled by `p_list` from network `g` while making
sure when more links are removed that the new links are a super set of the
previously removed links
"""
function remove_links_p_list!(g::Graph; p_min::Float64=0.5, p_max::Float64=1.00, interval::Float64=0.01)

    num_edges = edge_count(g)
    el = edges(g)
    h = copy(g)

    Δremoved = ceil(interval * num_edges)
    min_removed = ceil((1 - p_max) * num_edges)
    max_removed = ceil((1 - p_min) * num_edges)
    removed_list = []

    graph_list::Vector{Graph} = []
    p_list = [p_min]

    for _ = 1:min_removed
        link = rand(el)
        while link ∈ removed_list
            link = rand(el)
        end
        push!(removed_list, link)
    end

    for link ∈ removed_list
        (i, j) = Int.(link)
        remove_link!(g, i, j)
    end

    pushfirst!(graph_list, copy(h))

    while length(removed_list) ≤ max_removed
        local additional_removed_list = []

        for _ = 1:Δremoved
            link = rand(el)
            while link ∈ removed_list
                link = rand(el)
            end
            push!(removed_list, link)
            push!(additional_removed_list, link)
        end

        for link ∈ additional_removed_list
            (i, j) = Int.(link)
            remove_link!(h, i, j)
        end

        local p = p_list[end] + interval
        pushfirst!(graph_list, copy(h))
        push!(p_list, p)
    end

    return graph_list, p_list
end
