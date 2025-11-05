include("../lib/graph.jl")
include("../lib/paths.jl")
using Serialization

function θ_weights(g::Graph; θ=1.0)
    k = degrees(g)
    w = [(k[i]*k[j])^θ
         for i=1:size(g),j=1:size(g)]
    return w    
end

function gaussian_weights(g::Graph; σ=1.0, μ=0.0)
    N = size(g)
    adj = g.adjacency_list
    x = collect(-σ:0.01:σ) .+ μ
    gauss = [exp(-(i - μ)^2/(2σ^2))/(σ*(sqrt(2π))) for i in x]

    w = fill(0.0, (N, N))
    for u ∈ adj, v ∈ neighbors(g, u)
        w[u, v] = rand(gauss)
    end

    return w
end
