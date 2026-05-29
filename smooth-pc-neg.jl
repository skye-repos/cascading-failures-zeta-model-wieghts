using Pkg
Pkg.activate(".")
using DataFrames, CSV

# ===== Load raw p_c data =====
df = CSV.read("results/CSV/L = 100/p_c(θ, ζ).csv", DataFrame)
z  = df[:, 1]                                    # ζ values (19)
t  = parse.(Float64, names(df)[2:end])           # all 40 θ values
t_neg = t[1:20]                                  # θ < 0
t_pos = t[21:end]                                # θ > 0
pc_raw_neg = Matrix(df[:, 2:21])                 # 19 × 20, θ < 0
pc_raw_pos = Matrix(df[:, 22:end])               # 19 × 20, θ > 0

nz = length(z)

# ===== Gaussian kernel smoothing (2D) — only θ < 0 =====
function gaussian_smooth(mat, t_vals, z_vals, σt, σz)
    out = similar(mat)
    for iz in 1:size(mat, 1), it in 1:size(mat, 2)
        total, wsum = 0.0, 0.0
        for jz in 1:size(mat, 1), jt in 1:size(mat, 2)
            w = exp(-0.5 * (((t_vals[it] - t_vals[jt]) / σt)^2 +
                            ((z_vals[iz] - z_vals[jz]) / σz)^2))
            total += w * mat[jz, jt]
            wsum  += w
        end
        out[iz, it] = total / wsum
    end
    return out
end

σt = 0.10
σz = 5.0

pc_smoothed_neg = gaussian_smooth(pc_raw_neg, t_neg, z, σt, σz)

# ===== Combine smoothed negative + raw positive =====
pc_combined = hcat(pc_smoothed_neg, pc_raw_pos)

col_names = ["ζ"; string.(t)]
rows = [DataFrame([z[i] pc_combined[i, :]'], col_names) for i in 1:nz]
df_out = reduce(vcat, rows)

out_dir = "results/CSV/L = 100/"
out_path = out_dir * "p_c(θ, ζ)_smoothed_neg.csv"
CSV.write(out_path, df_out)

println("Output written to: $out_path")
println("  Columns 2–21 : smoothed (θ < 0,  σ_θ=$σt, σ_ζ=$σz)")
println("  Columns 22–41: raw      (θ > 0,  unsmoothed)")
println("  Dimensions: $nz ζ × $(length(t)) θ")
