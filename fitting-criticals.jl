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

# x_data = hcat(repeat(ζ_list, length(θ_pos)), 
#               repeat(θ_pos, inner=length(ζ_list)))
x_data = hcat(repeat(ζ_list, length(θ_neg)), 
              repeat(θ_neg, inner=length(ζ_list)))

function model(x, p)
    ζ = x[:, 1]
    θ = x[:, 2]

    # return p[1] .* exp.(p[2] ./ ζ .+ p[3] ./ θ)
    # return p[1] .* exp.(p[2] ./ ζ .+ p[3] .* θ)
    # return p[1] .* exp.(p[2] .* θ ./ ζ) .+ p[3]
    # return p[1] .+ p[2] .* ζ .+ p[3] .* θ
    return p[1] .+ p[2] ./ ζ .+ p[3] .* θ
end

p0 = [1.0, 1.0, 1.0]
# fit = curve_fit(model, x_data, pc_pos_vec, p0)
fit = curve_fit(model, x_data, pc_neg_vec, p0)
params = coef(fit)
σ = stderror(fit)
cov = vcov(fit)
res = residuals(fit)
ci = confidence_interval(fit)

ss1 = sum(res.*res)
# pc_avg = sum(pc_pos_vec) / length(pc_pos_vec)
# a = pc_pos_vec .- pc_avg
pc_avg = sum(pc_neg_vec) / length(pc_neg_vec)
a = pc_neg_vec .- pc_avg
ss2 = sum(a.*a)

r2 = 1 - (ss1 / ss2)

# pos_fit = "p_c(θ, ζ) = a*e^(b/ζ + c/θ)
# a = $(params[1])
# b = $(params[2])
# c = $(params[3])

# R^2 = $(r2)

# Covariance Matrix:
# $(repr("text/plain", cov))

# Confidence_Interval:
# $(repr("text/plain", ci))"

# write("positive_theta.txt", pos_fit)

# pos_fit_alt = "p_c(θ, ζ) = a*e^(b/ζ + c*θ)
# a = $(params[1])
# b = $(params[2])
# c = $(params[3])

# R^2 = $(r2)

# Covariance Matrix:
# $(repr("text/plain", cov))

# Confidence_Interval:
# $(repr("text/plain", ci))"

# write("positive_theta_alt.txt", pos_fit_alt)

# pos_fit_rat = "p_c(θ, ζ) = a*e^(b*θ/ζ) + c
# a = $(params[1])
# b = $(params[2])
# c = $(params[3])

# R^2 = $(r2)

# Covariance Matrix:
# $(repr("text/plain", cov))

# Confidence_Interval:
# $(repr("text/plain", ci))"

# write("positive_theta_rat.txt", pos_fit_rat)

# neg_fit = "p_c(θ, ζ) = a + b*ζ + c*θ
# a = $(params[1])
# b = $(params[2])
# c = $(params[3])

# R^2 = $(r2)

# Covariance Matrix:
# $(repr("text/plain", cov))

# Confidence_Interval:
# $(repr("text/plain", ci))"

# write("negative_theta.txt", neg_fit)

neg_fit_alt = "p_c(θ, ζ) = a + b/ζ + c*θ
a = $(params[1])
b = $(params[2])
c = $(params[3])

R^2 = $(r2)

Covariance Matrix:
$(repr("text/plain", cov))

Confidence_Interval:
$(repr("text/plain", ci))"

write("negative_theta_alt.txt", neg_fit_alt)
