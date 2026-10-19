module FSS

using CSV
using DataFrames
using LinearAlgebra
using Statistics

export run_fss

function fit_beta(p, S, p_c, p_index, window, min_points)
    baseline = S[p_index]
    lo = p_index
    hi = searchsortedlast(p, p_c + window)
    hi = min(hi, length(p))

    x = Float64[]
    y = Float64[]
    excluded = 0
    for j in (lo + 1):hi
        dy = S[j] - baseline
        dx = p[j] - p_c
        if dy > 0.0 && dx > 0.0
            push!(x, dx)
            push!(y, dy)
        else
            excluded += 1
        end
    end

    if length(x) < min_points || length(unique(x)) < 2
        return nothing
    end

    lx = log.(x)
    ly = log.(y)
    X = hcat(ones(length(lx)), lx)
    coef = X \ ly
    fitted = X * coef
    ss_res = sum((ly .- fitted) .^ 2)
    ss_tot = sum((ly .- mean(ly)) .^ 2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN

    return (
        k = exp(coef[1]),
        beta = coef[2],
        R2 = r2,
        n_points = length(x),
        n_excluded = excluded,
        window_low = p[lo + 1],
        window_high = p[hi],
    )
end

function group_sem(values)
    n = length(values)
    n <= 1 && return 0.0
    return std(values; corrected = true) / sqrt(n)
end

function saturation_report(rows, label::AbstractString,
                           value_col::Symbol, se_col::Symbol)
    lines = String[]
    for topology in sort(unique(rows.topology))
        df_topo = filter(:topology => ==(topology), rows)
        for regime in sort(unique(df_topo.regime))
            df = filter(:regime => ==(regime), df_topo)
            Ls = sort(unique(df.L))
            push!(lines, "topology = $topology, regime = $regime")
            if length(Ls) < 2
                push!(lines, "  insufficient L values for a $label saturation test")
                continue
            end

            vals = Float64[]
            ses = Float64[]
            for L in Ls
                sub = filter(:L => ==(L), df)
                push!(vals, mean(getproperty(sub, value_col)))
                push!(ses, mean(getproperty(sub, se_col)))
            end

            weights = [se > 0 ? 1.0 / se^2 : 1.0 for se in ses]
            value_bar = sum(weights .* vals) / sum(weights)
            se_bar = sqrt(1.0 / sum(weights))
            dof = length(vals) - 1
            chi2 = sum(weights .* (vals .- value_bar) .^ 2)
            chi2red = dof > 0 ? chi2 / dof : NaN

            push!(lines, "  $label(L) = " *
                  join(["$L:$(round(vals[i], digits=4))±$(round(ses[i], digits=4))"
                        for (i, L) in enumerate(Ls)], ", "))
            push!(lines, "  weighted mean $label = $(round(value_bar, digits=4)) ± " *
                  "$(round(se_bar, digits=4)), reduced chi2 = " *
                  "$(round(chi2red, digits=3))")

            if isfinite(chi2red) && chi2red <= 2.0
                push!(lines, "  decision: $label is constant within errors; " *
                      "report the universal value $(round(value_bar, digits=4)) ± " *
                      "$(round(se_bar, digits=4))")
            else
                X = hcat(ones(length(Ls)), 1.0 ./ Ls)
                W = Diagonal(weights)
                coef = (X' * W * X) \ (X' * W * vals)
                cov = inv(X' * W * X)
                se_slope = sqrt(max(cov[2, 2], 0.0))
                if abs(coef[2]) > 2.0 * se_slope
                    push!(lines, "  decision: $label drifts with L without saturation; " *
                          "report $label as unresolved")
                else
                    push!(lines, "  decision: $label drift is not significant; " *
                          "report $label as unresolved")
                end
            end
        end
    end
    return lines
end

function run_fss(; root, out, window = 0.1, min_points = 3,
                 ratios = Float64[], thetas = Float64[])
    summary_path = joinpath(root, "p_c_summary.csv")
    s_path = joinpath(root, "S_mean.csv")
    isfile(summary_path) || error("missing $summary_path")
    isfile(s_path) || error("missing $s_path")

    summary = CSV.read(summary_path, DataFrame)
    S_long = CSV.read(s_path, DataFrame)
    mkpath(out)

    if !isempty(ratios)
        keep = [any(isapprox(row.zeta / row.L, r; atol = 1e-9)
                    for r in ratios) for row in eachrow(summary)]
        summary = summary[keep, :]
    end

    if !isempty(thetas)
        keep = [any(isapprox(row.theta, t; atol = 1e-6)
                    for t in thetas) for row in eachrow(summary)]
        summary = summary[keep, :]
    end

    isempty(summary) && error("no summary rows match the requested ratios/thetas")

    summary[!, :regime] = ifelse.(summary.theta .< 0, "negative", "positive")

    beta_rows = NamedTuple[]

    for row in eachrow(summary)
        mask = (S_long.L .== row.L) .&
               (S_long.zeta .== row.zeta) .&
               (S_long.topology .== row.topology) .&
               (S_long.theta .== row.theta)
        df = S_long[mask, :]
        nrow(df) >= 3 || continue
        sort!(df, :p)

        p = Float64.(df.p)
        S = Float64.(df.S_mean)
        fit = fit_beta(p, S, row.p_c, row.p_index, window, min_points)
        fit === nothing && continue

        push!(beta_rows, (
            L = row.L,
            zeta = row.zeta,
            topology = row.topology,
            regime = row.regime,
            theta = row.theta,
            p_c = row.p_c,
            window = window,
            window_low = fit.window_low,
            window_high = fit.window_high,
            n_points = fit.n_points,
            n_excluded = fit.n_excluded,
            k = fit.k,
            beta = fit.beta,
            R2 = fit.R2,
        ))
    end

    isempty(beta_rows) && error("no beta fits were produced")
    beta_df = DataFrame(beta_rows)
    CSV.write(joinpath(out, "beta_fits.csv"), beta_df)

    beta_vs_L = NamedTuple[]
    for topology in sort(unique(beta_df.topology))
        df_topo = filter(:topology => ==(topology), beta_df)
        for regime in sort(unique(df_topo.regime))
            df_reg = filter(:regime => ==(regime), df_topo)
            for L in sort(unique(df_reg.L))
                vals = filter(:L => ==(L), df_reg).beta
                push!(beta_vs_L, (
                    L = L,
                    topology = topology,
                    regime = regime,
                    n = length(vals),
                    beta_mean = mean(vals),
                    beta_se = group_sem(vals),
                ))
            end
        end
    end
    beta_vs_L_df = DataFrame(beta_vs_L)
    CSV.write(joinpath(out, "beta_vs_L.csv"), beta_vs_L_df)
    for regime in ("negative", "positive")
        sub = filter(:regime => ==(regime), beta_df)
        isempty(sub) || CSV.write(joinpath(out, "beta_fits_$regime.csv"), sub)
        sub = filter(:regime => ==(regime), beta_vs_L_df)
        isempty(sub) || CSV.write(joinpath(out, "beta_vs_L_$regime.csv"), sub)
    end

    delta_rows = NamedTuple[]
    for topology in sort(unique(summary.topology))
        df_topo = filter(:topology => ==(topology), summary)
        for regime in sort(unique(df_topo.regime))
            df_reg = filter(:regime => ==(regime), df_topo)
            for L in sort(unique(df_reg.L))
                vals = filter(:L => ==(L), df_reg).delta_S
                push!(delta_rows, (
                    L = L,
                    topology = topology,
                    regime = regime,
                    n = length(vals),
                    delta_S_mean = mean(vals),
                    delta_S_se = group_sem(vals),
                ))
            end
        end
    end
    delta_df = DataFrame(delta_rows)
    CSV.write(joinpath(out, "delta_S_vs_L.csv"), delta_df)
    for regime in ("negative", "positive")
        sub = filter(:regime => ==(regime), delta_df)
        isempty(sub) || CSV.write(joinpath(out, "delta_S_vs_L_$regime.csv"), sub)
    end

    report = vcat(
        saturation_report(beta_vs_L_df, "beta", :beta_mean, :beta_se),
        ["", "delta_S saturation"],
        saturation_report(delta_df, "delta_S", :delta_S_mean, :delta_S_se),
    )
    write(joinpath(out, "fss_report.txt"), join(report, "\n") * "\n")

    println("Wrote beta/FSS results to $(out)")
end

end
