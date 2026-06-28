using Pkg
Pkg.activate(".")
using DataFrames
using CSV, Tables

files = readdir("simplified/", join=true)
ζ_list = collect(5.0:2.5:50)

θ_list = append!(
    collect(-1.0:0.05:-0.05),
    collect(0.05:0.05:1.00)
)

header = names(CSV.read(files[1], DataFrame))[2:end]
names = append!(["ζ"], header)

if !ispath("fixed-thetas/")
    mkpath("fixed-thetas/")
end

function fixed_theta_csv(i)
    df_out = DataFrame([name => [] for name ∈ names])
    for ζ ∈ ζ_list
        file = "simplified/ζ = $(ζ).csv"
        df = CSV.read(file, DataFrame)
        vals = collect(df[i, 2:end])
        vals_out = append!([ζ], vals)
        push!(df_out, vals_out)
    end

    CSV.write("fixed-thetas/θ = $(θ_list[i]).csv", df_out)
    return nothing
end

for i ∈ eachindex(θ_list)
    fixed_theta_csv(i)
end
