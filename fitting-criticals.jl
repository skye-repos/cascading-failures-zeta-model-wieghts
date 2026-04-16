using Pkg
Pkg.activate(".")
using LsqFit
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

    return p[1] .* exp.(p[2] ./ ζ .+ p[3] ./ θ)
end

function pos_alt(x, p)
    ζ = x[:, 1]
    θ = x[:, 2]

    return p[1] .* exp.(p[2] ./ ζ .+ p[3] .* θ)
end

function pos_rat(x, p)
    ζ = x[:, 1]
    θ = x[:, 2]

    return p[1] .* exp.(p[2] .* θ ./ ζ) .+ p[3]
end

function neg(x, p)
    ζ = x[:, 1]
    θ = x[:, 2]

    return p[1] .+ p[2] .* ζ .+ p[3] .* θ
end

function neg_alt(x, p)
    ζ = x[:, 1]
    θ = x[:, 2]

    return p[1] .+ p[2] ./ ζ .+ p[3] .* θ
end

function r2(y, fit)
    res = residuals(fit)
    ss1 = sum(res.*res)
    y_avg = sum(y) / length(y)
    a = y .- y_avg
    ss2 = sum(a.*a)

    return 1 - (ss1/ss2)
end

dir = "results/fits/L = 100/p_critical/"

function fit_and_write(model, x, y, file, eqn)
    p0 = [1.0, 1.0, 1.0]
    fit = curve_fit(model, x, y, p0)
    params = coef(fit)
    cov = vcov(fit)
    ci = confidence_interval(fit)
    r_sq = r2(y, fit)

    str = "$(eqn)\n"*
        "a = $(params[1])
b = $(params[2])
c = $(params[3])

R^2 = $(r_sq)

Covariance Matrix:
$(repr("text/plain", cov))

Confidence_Interval:
$(repr("text/plain", ci))"

    write(dir*file, str)
    return nothing
end

fit_and_write(pos,
              x_pos_data, pc_pos_vec,
              "θ > 0 v1.txt",
              "p_c(θ, ζ) = a * exp(b/ζ + c/θ)")

fit_and_write(pos_alt,
              x_pos_data, pc_pos_vec,
              "θ > 0 v2.txt",
              "p_c(θ, ζ) = a * exp(b/ζ + c*θ)")

fit_and_write(pos_rat,
              x_pos_data, pc_pos_vec,
              "θ > 0 v3.txt",
              "p_c(θ, ζ) = a * exp(b*θ/ζ) + c")

fit_and_write(neg,
              x_neg_data, pc_neg_vec,
              "θ < 0 v1.txt",
              "p_c(θ, ζ) = a + b*ζ + c*θ")

fit_and_write(neg_alt,
              x_neg_data, pc_neg_vec,
              "θ < 0 v2.txt",
              "p_c(θ, ζ) = a + b/ζ + c*θ")
