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

x_data = hcat(repeat(ζ_list, length(θ_neg)), 
              repeat(θ_neg, inner=length(ζ_list)))


function model(x, p)
    ζ = x[:, 1]
    θ = x[:, 2]

    return p[1] .* ζ .+ p[2] .* θ .+ p[3]
end

p0 = [1.0, 1.0, 1.0]
fit = curve_fit(model, x_data, pc_neg_vec, p0)
params = coef(fit)
σ = stderror(fit)
cov = vcov(fit)
res = residuals(fit)
ci = confidence_interval(fit)


# pos_fit = "
# p_c(θ, ζ) = a*e^(b/ζ + c/θ)
# a = $(params[1])
# b = $(params[2])
# c = $(params[3])

# Covariance Matrix:
# $(repr("text/plain", cov))

# Confidence_Interval:
# $(repr("text/plain", ci))"

# write("positive_theta.txt", pos_fit)

neg_fit = "
p_c(θ, ζ) = a*ζ + b*θ + c
a = $(params[1])
b = $(params[2])
c = $(params[3])

Covariance Matrix:
$(repr("text/plain", cov))

Confidence_Interval:
$(repr("text/plain", ci))"

write("negative_theta.txt", neg_fit)

