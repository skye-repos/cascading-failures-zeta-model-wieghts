using Pkg
Pkg.activate(".")
using DataFrames, CSV
using Plots, LaTeXStrings

ζ_list = collect(5.0:2.5:50.0)

function β_row(ζ)
    file = "results/fits/L = 100/β, per-ζ/ζ = $(ζ).csv"
    df = CSV.read(file, DataFrame)

    header = string.(df[:, 1])
    names = append!(["ζ"], header)
    β_list = df[:, 4]
    vals = append!([ζ], β_list)
    df_loc = DataFrame([names[i] => vals[i] for i ∈ eachindex(vals)])

    return df_loc    
end

df = β_row(ζ_list[1])
for ζ ∈ ζ_list[2:end]
    df_loc = β_row(ζ)
    append!(df, df_loc)
end

θ_list = parse.(Float64, names(df)[2:end])

out_dir = "results/CSV/L = 100/"

if !ispath(out_dir)
    mkpath(out_dir)
end

out_csv = out_dir * "β(θ,ζ).csv"

CSV.write(out_csv, df)

out_plt_dir = "results/plots/L = 100/"
if !ispath(out_plt_dir)
    mkpath(out_plt_dir)
end

#Positive θ
β_matrix_pos = Matrix(df[:, 22:end])
pos_plt = heatmap(θ_list[21:end], ζ_list, β_matrix_pos,
              c=:viridis,
              xlabel="θ values",
              ylabel="ζ values",
              title=L"Heatmap of $β$ for θ > 0")
savefig(pos_plt, out_plt_dir * "β for θ > 0.png")

#Negative θ
β_matrix_neg = Matrix(df[:, 2:21])
neg_plt = heatmap(θ_list[1:20], ζ_list, β_matrix_neg,
              c=:viridis,
              xlabel="θ values",
              ylabel="ζ values",
              title=L"Heatmap of $β$ for θ < 0")
savefig(neg_plt, out_plt_dir * "β for θ < 0.png")
