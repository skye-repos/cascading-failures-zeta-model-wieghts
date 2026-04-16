using Pkg
Pkg.activate(".")
using DataFrames, CSV
using LsqFit

df_indices = CSV.read("results/CSV/L = 100/indices.csv", DataFrame)
ζ_list = df_indices[:, 1]
θ_list = parse.(Float64, names(df_indices)[2:end])

function get_S_and_θ(θ, ζ)
    file = "data/ζ = $(ζ).csv"
    df = CSV.read(file, DataFrame)

    ζ_index = findfirst(x->x==ζ, ζ_list)
    θ_index = findfirst(x->x==θ, θ_list)

    pci = Integer(df_indices[ζ_index, θ_index + 1])
    df = df[θ_index, (pci+1):end]
    p_list = parse.(Float64, names(df))

    return p_list, collect(df)
end

function linear(x, p)
    return p[1] .+ p[2].*x
end

function quadratic(x, p)
    return p[1] .+ p[2].*x .+ p[3].*x.*x
end
