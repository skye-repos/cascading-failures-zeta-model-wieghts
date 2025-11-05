using Distributed
include("utility.jl")

"""
Builds a lattice network based on the ζ model on a grid of side length `L`, scale parameter of `ζ`, and avg degree of `k_avg`
"""
function ζ_2d(L::Int64, ζ::Number, k_avg::Number)
    local N = L^2

    coordinates::Vector{Tuple{Int, Int}} = []

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
            w[i, j] = exp(- r / ζ)
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
Remove a random portion (1 - `p`) of the existing links in network `g`. If some links have already been removed but not specified, remove a superset of those links.
"""
function remove_random_links!(g::Graph, p::Float64, removed_list::Vector{Tuple}=[])
    el = edges(g)
    num_edges = length(el)

    if num_edges == 0
        return g
    end

    full_remove_count = ceil((1-p) * num_edges)
    remove_count = full_remove_count - length(removed_list)
    
    for _ = 1:remove_count
        link = rand(el)
        while link ∈ removed_list
            link = rand(el)
        end
        push!(removed_list, link)
    end
    
    for link in removed_list
        (i, j) = Int.(link)
        remove_link!(g, i, j)
    end
    return g, removed_list
end

"""
Remove a portion of links controlled by `p_list` from network `g` while making sure when more links are removed that the new links are a super set of the previously removed links
"""
function remove_random_links_p_list!(g::Graph, p_list::Vector{Float64})
    graph_copies = [copy(g) for _ = 1:length(p_list)]
    graph_list = Vector{Graph}(undef, length(p_list))
    ordered_p_list = sort(p_list, rev=true)
    removed_list::Vector{Tuple} = []
    
    removed_per_p = Vector{Vector{Tuple}}(undef, length(p_list))
    # removed_add_per_p = Vector{Vector{Tuple}}(undef, length(p_list))

    # graph = copy(g)
    
    for (i, p) ∈ enumerate(p_list)
        # graph, removed_list = remove_random_links!(graph, ordered_p_list[i], removed_list)
        # graph_list[i] = copy(graph)
        
        # local h, removed_list_local, add_removed_list = remove_random_links!(graph_copies[i], p, removed_list)
        local h, removed_list_local = remove_random_links!(graph_copies[i], p, removed_list)
        graph_list[i] = h
        
        removed_per_p[i] = removed_list_local
        # removed_add_per_p[i] = add_removed_list
        removed_list = removed_list_local
    end

    return graph_list, ordered_p_list
end
