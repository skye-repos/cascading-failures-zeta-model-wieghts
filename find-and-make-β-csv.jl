using Pkg
Pkg.activate(".")
using DataFrames, CSV
using LsqFit
using Plots, LaTeXStrings

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

function critical(x, p)
    return p[1] .* x .^ p[2]
end

function r2(y, fit)
    res = residuals(fit)
    ss1 = sum(res.*res)
    y_avg = sum(y) / length(y)
    a = y .- y_avg
    ss2 = sum(a.*a)

    return 1 - (ss1/ss2)
end

function bic(y, fit, k)
    n = length(y)
    rss = sum(abs2, residuals(fit))
    return n * log(rss / n) + k * log(n)
end

dir = "results/fits/L = 100/β, per-ζ/"

if !ispath(dir)
    mkpath(dir)
end

df_names = ["θ", "p_c + 0.01", "k (proportionality)", "β (exponent)", "R2", "BIC"]
p0 = [1.0, 0.5]

bic_rows = []
n = length(ζ_list) * length(θ_list)
sizehint!(bic_rows, n)

for ζ ∈ ζ_list
    out_file = dir * "ζ = $(ζ).csv"
    df = DataFrame([name => [] for name ∈ df_names])
    for θ ∈ θ_list
        x, y = get_S_and_θ(θ, ζ)
        y_diff = y .- y[1]
        x_diff = x .- x[1]
        fit = curve_fit(critical, x_diff, y_diff, p0)
        r_sq = r2(y, fit)
        b_i = bic(y, fit, 2)
        params = coef(fit)
        vals = [θ, x[1], params[2], params[1], r_sq, b_i]
        local df_loc = DataFrame([df_names[i] => vals[i] for i ∈ eachindex(vals)])
        append!(df, df_loc)

        push!(bic_rows, (ζ, θ, length(y), b_i, r_sq))
    end
    CSV.write(out_file, df)
end

bic_file = "BIC_scores.csv"
bic_header = !isfile(bic_file)
open(bic_file, "a") do f
    if bic_header
        write(f, "source,model,region,n,k,BIC,R2\n")
    end
    for (ζ, θ, ny, bi, rs) in bic_rows
        region = θ > 0 ? "θ > 0" : "θ < 0"
        write(f, "power_law_β,y = k*x^β,$region,$ny,2,$bi,$rs\n")
    end
end

function β_row(ζ)
    file = dir * "ζ = $(ζ).csv"
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
