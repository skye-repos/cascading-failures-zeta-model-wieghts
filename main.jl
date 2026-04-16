using Pkg
Pkg.activate(".")
using DataFrames, CSV
using LaTeXStrings
using Plots

ζ_list = [2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 50.0]

function find_critical(row)
    θ = row[1]
    vals = row[2:end]
    n = length(vals)

    p = parse.(Float64, names(vals))
    h = sum(p[2:end] .- p[1:end - 1]) / (n - 1)

    vals = collect(vals)

    critical_point = 1
    diffs = zeros(Float64, n)
    for i ∈ eachindex(vals)[1:end-2]
        d2 = (vals[i+2] + vals[i] - 2*vals[i+1]) / h^2
        diffs[i] = d2
    end

    glob_max = 0

    for j ∈ eachindex(diffs)[1:end-1]
        local max = 0
        if ((diffs[j+1] > 0 && diffs[j] < 0) ||
            (diffs[j+1] < 0 && diffs[j] > 0))
            max = abs(diffs[j+1] - diffs[j])
        end

        if max > glob_max
            glob_max = max
            critical_point = j + 1
        end
    end

    return θ, critical_point, p[critical_point]
end

function pc_row(ζ)
    file = "data/ζ = $(ζ).csv"
    df_z = CSV.read(file, DataFrame)

    header = string.(df_z[:, 1])
    names = append!(["ζ"], header)
    
    pc_list = []
    for row ∈ eachrow(df_z)
        θ, _, pc = find_critical(row)
        push!(pc_list, pc)
    end

    vals = append!([ζ], pc_list)
    df_loc = DataFrame([names[i] => vals[i] for i ∈ eachindex(vals)])

    return df_loc
end

df = pc_row(ζ_list[1])
for ζ ∈ ζ_list[2:end]
    df_loc = pc_row(ζ)
    append!(df, df_loc)
end
θ_list = parse.(Float64, names(df)[2:end])

out_csv = "results/CSV/L = 100 - values.csv"
out_plt = "results/plots/L = 100 - values - positive.png"
CSV.write(out_csv, df)

pc_matrix = Matrix(df[3:end, 22:end])
plt = heatmap(θ_list[21:end], ζ_list[3:end], pc_matrix,
              c=:viridis,
              xlabel="θ values",
              ylabel="ζ values",
              title=L"Heatmap of $p_c$")
savefig(plt, out_plt)
