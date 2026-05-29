using Pkg
Pkg.activate(".")
using DataFrames, CSV
using LaTeXStrings
using Plots

ζ_list = collect(5.0:2.5:50)

function gaussian_kernel(x, w::Int = 3)
    n = length(x)
    out = zeros(n - w)
    σ = 5
    γ = - 1 / (2σ^2)
    for i ∈ 1:n-w
        g = 0
        for j ∈ 1:w-1
            g += exp(γ * (x[i] - x[j])^2)
        end
        out[i] = g / w
    end

    return out
end

function find_critical(row)
    θ = row[1]
    vals = row[2:end]
    n = length(vals)

    p = parse.(Float64, names(vals))
    h = sum(p[2:end] .- p[1:end - 1]) / (n - 1)

    vals = collect(vals)
    vals = gaussian_kernel(vals, 3)

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

    critical_point += 1

    return θ, critical_point, p[critical_point]
end

function pc_row(ζ)
    file = "data/ζ = $(ζ).csv"
    df_z = CSV.read(file, DataFrame)

    header = string.(df_z[:, 1])
    names = append!(["ζ"], header)
    
    pc_list = []
    pci_list = []
    for row ∈ eachrow(df_z)
        θ, pci, pc = find_critical(row)
        push!(pc_list, pc)
        push!(pci_list, pci)
    end

    vals = append!([ζ], pc_list)
    valsi = append!([ζ], pci_list)
    df_loc = DataFrame([names[i] => vals[i] for i ∈ eachindex(vals)])
    dfi_loc = DataFrame([names[i] => valsi[i] for i ∈ eachindex(valsi)])

    return df_loc, dfi_loc
end

df, dfi = pc_row(ζ_list[1])
for ζ ∈ ζ_list[2:end]
    df_loc, dfi_loc = pc_row(ζ)
    append!(df, df_loc)
    append!(dfi, dfi_loc)
end

θ_list = parse.(Float64, names(df)[2:end])

out_dir = "results/CSV/L = 100/"

if !ispath(out_dir)
    mkpath(out_dir)
end

out_csv = out_dir * "p_c(θ, ζ).csv"
outi_csv = out_dir * "indices.csv"
CSV.write(out_csv, df)
CSV.write(outi_csv, dfi)

out_plt = "results/plots/L = 100/"
if !ispath(out_plt)
    mkpath(out_plt)
end

plt_pos = out_plt * "p_c(θ, ζ), θ > 0.png"
pc_pos = Matrix(df[:, 22:end])
plt = heatmap(θ_list[21:end], ζ_list, pc_pos,
              c=:viridis,
              xlabel="θ values",
              ylabel="ζ values",
              title=L"Heatmap of $p_c$ in Region A")
savefig(plt, plt_pos)

plt_neg = out_plt * "p_c(θ, ζ), θ < 0.png"
pc_neg = Matrix(df[:, 2:21])
plt = heatmap(θ_list[1:20], ζ_list, pc_neg,
              c=:viridis,
              xlabel="θ values",
              ylabel="ζ values",
              title=L"Heatmap of $p_c$ in Region B")
savefig(plt, plt_neg)

plt_all = out_plt * "p_c(θ, ζ), full.png"
pc_all = Matrix(df[:, 2:end])
plt = heatmap(θ_list, ζ_list, pc_all,
              c=:viridis,
              xlabel="θ values",
              ylabel="ζ values",
              title=L"Heatmap of $p_c$ in Region A and B")
savefig(plt, plt_all)
