using Pkg
Pkg.activate(".")
using LsqFit, LinearAlgebra
using DataFrames, CSV

df = CSV.read("results/CSV/L = 100/values.csv", DataFrame)

ζ_list = df[:, 1]
df_pos = df[:, 22:end]
df_neg = df[:, 2:21]
θ_pos = parse.(Float64, names(df_pos))
θ_neg = parse.(Float64, names(df_neg))

pc_pos_vec = vec(Matrix(df_pos))
pc_neg_vec = vec(Matrix(df_neg))

x_pos_data = hcat(repeat(ζ_list, length(θ_pos)),
    repeat(θ_pos, inner=length(ζ_list)))
x_neg_data = hcat(repeat(ζ_list, length(θ_neg)),
    repeat(θ_neg, inner=length(ζ_list)))


function pos(x, p)
    ζ = x[:, 1]
    θ = x[:, 2]

    return exp.(p[1] .+ p[2] ./ ζ .+ p[3] .* θ .+ p[4] .* θ .^ 2)
end

function pos_ln(x, p)
    ζ = x[:, 1]
    θ = x[:, 2]

    return exp.(p[1] .+ p[2] ./ log.(ζ) .+ p[3] .* θ .+ p[4] .* θ .^ 2)
end

function neg(x, p)
    ζ = x[:, 1]
    θ = x[:, 2]

    return p[1] .+ p[2] ./ ζ .+ p[3] .* θ + p[4] .* θ .^ 2
end

function neg_ln(x, p)
    ζ = x[:, 1]
    θ = x[:, 2]

    return p[1] .+ p[2] ./ log.(ζ) .+ p[3] .* θ + p[4] .* θ .^ 2
end

function r2(y, fit)
    res = residuals(fit)
    ss1 = sum(res .* res)
    y_avg = sum(y) / length(y)
    a = y .- y_avg
    ss2 = sum(a .* a)

    return 1 - (ss1 / ss2)
end

dir = "results/fits/L = 100/p_critical-ho/"

function fit_and_write(model, x, y, file, eqn)
    p0 = [1.0, 1.0, 1.0, 1.0]
    fit = curve_fit(model, x, y, p0)
    params = coef(fit)
    cov = vcov(fit)
    ci = confidence_interval(fit)
    r_sq = r2(y, fit)

    str = "$(eqn)\n" *
          "a = $(params[1])
b = $(params[2])
c = $(params[3])
d = $(params[4])

R^2 = $(r_sq)
det(σ^2) = $(det(cov))

Covariance Matrix:
$(repr("text/plain", cov))

Confidence_Interval:
$(repr("text/plain", ci))"

    write(dir * file, str)
    return nothing
end

fit_and_write(pos,
    x_pos_data, pc_pos_vec,
    "θ > 0 v1.txt",
    "p_c(θ, ζ) = exp(a + b/ζ + c*θ + d*θ^2)")

fit_and_write(pos_ln,
    x_pos_data, pc_pos_vec,
    "θ > 0 v2.txt",
    "p_c(θ, ζ) = exp(a + b/ln(ζ) + c*θ + d*θ^2)")

fit_and_write(neg,
    x_neg_data, pc_neg_vec,
    "θ < 0 v1.txt",
    "p_c(θ, ζ) = a + b/ζ + c*θ + d*θ^2")

fit_and_write(neg_ln,
    x_neg_data, pc_neg_vec,
    "θ < 0 v2.txt",
    "p_c(θ, ζ) = a + b/ln(ζ) + c*θ + d*θ^2")
