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

function r2(y, fit)
    res = residuals(fit)
    ss1 = sum(res.*res)
    y_avg = sum(y) / length(y)
    a = y .- y_avg
    ss2 = sum(a.*a)

    return 1 - (ss1/ss2)
end

dir = "results/fits/L = 100/slopes/"

df_names = ["θ", "p_c + 0.05", "m (slope)", "b (intercept)", "R2"]
p0 = [1.0, 0.5]

for ζ ∈ ζ_list
    out_file = dir * "linear fits ζ = $(ζ).csv"
    df = DataFrame([name => [] for name ∈ df_names])
    for θ ∈ θ_list
        x, y = get_S_and_θ(θ, ζ)
        fit = curve_fit(linear, x, y, p0)
        r_sq = r2(y, fit)
        params = coef(fit)
        vals = [0, x[1], params[2], params[1], r_sq]
        local df_loc = DataFrame([df_names[i] => vals[i] for i ∈ eachindex(vals)])
        append!(df, df_loc)
    end
    CSV.write(out_file, df)
end
