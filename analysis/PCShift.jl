module PCShift

using CSV
using DataFrames
using Statistics
using LinearAlgebra

using ..PCUtils
using ..FSS: fit_beta

ENV["GKSwstype"] = "100"
using Plots

export run_pc_shift, run_pc_shift_realization, run_pc_realization_plots,
       run_ratio_plots

function sem_or_zero(values)
    n = length(values)
    n > 1 ? std(values; corrected = true) / sqrt(n) : 0.0
end

function average_over_zeta(pc)
    df = filter(:topology => ==("waxman"), pc)
    long = combine(groupby(df, [:theta, :L]),
                   :p_c => mean => :p_c,
                   :p_c => sem_or_zero => :p_c_sem,
                   nrow => :n_zeta)
    sort!(long, [:theta, :L])
    return long
end

function write_wide_pc(long, dest; filename = "pc_L_vs_theta.csv")
    Ls = sort(unique(long.L))
    thetas = sort(unique(long.theta))
    lookup = Dict((row.theta, row.L) => row.p_c for row in eachrow(long))

    wide = DataFrame(theta = thetas)
    for L in Ls
        wide[!, string(L)] = [get(lookup, (theta, L), NaN) for theta in thetas]
    end
    CSV.write(joinpath(dest, filename), wide)
end

function linear_fit(x, y)
    n = length(x)
    X = hcat(ones(n), x)
    coef = X \ y
    pred = X * coef
    ss_res = sum((y .- pred) .^ 2)
    ss_tot = sum((y .- mean(y)) .^ 2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN

    se_slope = NaN
    if n > 2
        s2 = ss_res / (n - 2)
        cov = s2 * inv(X' * X)
        se_slope = sqrt(max(cov[2, 2], 0.0))
    end

    return (intercept = coef[1], slope = coef[2], r2 = r2,
            se_slope = se_slope)
end

function shift_fit(L, pc, pcinf, log_floor; mask)
    Lf = L[mask]
    pf = pc[mask]
    raw = abs.(pf .- pcinf)
    n_zero = count(==(0.0), raw)
    d = [v == 0.0 ? log_floor : v for v in raw]
    x = log10.(Lf)
    y = log10.(d)
    fit = linear_fit(x, y)
    nu = abs(fit.slope) > 0 ? -1.0 / fit.slope : NaN
    nu_se = (abs(fit.slope) > 0 && isfinite(fit.se_slope)) ?
            abs(fit.se_slope / fit.slope^2) : NaN
    return (intercept = fit.intercept, slope = fit.slope,
            r2 = fit.r2, se_slope = fit.se_slope,
            nu = nu, nu_se = nu_se, n_points = length(x),
            n_zero_replaced = n_zero)
end

function plot_shift(L, pc, fit1, pcinf1, fit2, pcinf2, theta,
                    log_floor, outpath; title_prefix = "p_c shift")
    plt = plot()

    mask1 = L .< maximum(L)
    x1 = log10.(L[mask1])
    d1 = abs.(pc[mask1] .- pcinf1)
    d1 = [v == 0.0 ? log_floor : v for v in d1]
    y1 = log10.(d1)
    scatter!(plt, x1, y1, color = :dodgerblue, marker = :circle,
             label = "Largest-L: ν=$(round(fit1.nu, digits = 3))")
    xf1 = range(minimum(x1), maximum(x1); length = 50)
    plot!(plt, xf1, fit1.intercept .+ fit1.slope .* xf1,
          color = :dodgerblue, label = "")

    x2 = log10.(L)
    d2 = abs.(pc .- pcinf2)
    d2 = [v == 0.0 ? log_floor : v for v in d2]
    y2 = log10.(d2)
    scatter!(plt, x2, y2, color = :firebrick, marker = :diamond,
             label = "1/L fit: ν=$(round(fit2.nu, digits = 3))")
    xf2 = range(minimum(x2), maximum(x2); length = 50)
    plot!(plt, xf2, fit2.intercept .+ fit2.slope .* xf2,
          color = :firebrick, label = "")

    xlabel!(plt, "log₁₀ L")
    ylabel!(plt, "log₁₀ |p_c(L) - p_c(∞)|")
    title!(plt, "$(title_prefix), θ=$(round(theta, digits = 2))")
    savefig(plt, outpath)
end

function collect_realization_fits(raw_root)
    runs = discover_realizations(raw_root)
    pc_rows = NamedTuple[]
    beta_rows = NamedTuple[]
    delta_rows = NamedTuple[]
    n_fail = 0

    for run in runs
        path = find_file(run.path, "Gt Component")
        path === nothing && continue
        θ, p, S = read_s_matrix(path)
        m = match(r"realization = (\d+)", basename(run.path))
        rid = m === nothing ? -1 : parse(Int, m.captures[1])

        for i in eachindex(θ)
            values = S[i, :]
            pc = find_critical(p, values; sigma_pts = 2.0)
            push!(pc_rows, (L = run.L, zeta = run.zeta,
                            topology = run.topology, realization = rid,
                            theta = θ[i], p_c = pc))

            p_index = argmin(abs.(p .- pc))
            klo = max(p_index - 1, 1)
            khi = min(p_index + 1, length(p))
            delta_S = values[khi] - values[klo]
            push!(delta_rows, (L = run.L, zeta = run.zeta,
                               topology = run.topology, realization = rid,
                               theta = θ[i],
                               regime = θ[i] < 0 ? "negative" : "positive",
                               delta_S = delta_S))

            fit = fit_beta(p, values, pc, p_index, 0.1, 3)
            if fit === nothing
                n_fail += 1
                continue
            end
            push!(beta_rows, (L = run.L, zeta = run.zeta,
                              topology = run.topology, realization = rid,
                              theta = θ[i],
                              regime = θ[i] < 0 ? "negative" : "positive",
                              beta = fit.beta))
        end
    end

    return DataFrame(pc_rows), DataFrame(beta_rows), DataFrame(delta_rows),
           n_fail
end

function aggregate_pc_realization(pc)
    df = filter(:topology => ==("waxman"), pc)
    long = combine(groupby(df, [:theta, :L]),
                   :p_c => mean => :p_c,
                   :p_c => sem_or_zero => :p_c_sem,
                   :realization => (x -> length(unique(x))) => :n_realizations,
                   :zeta => (x -> length(unique(x))) => :n_zeta,
                   nrow => :n_values)
    sort!(long, [:theta, :L])
    return long
end

function aggregate_beta_realization(beta)
    long = combine(groupby(beta, [:L, :topology, :regime]),
                   :beta => mean => :beta_mean,
                   :beta => sem_or_zero => :beta_se,
                   nrow => :n)
    sort!(long, [:topology, :regime, :L])
    return long
end

function aggregate_beta_by_theta(beta)
    long = combine(groupby(beta, [:L, :topology, :regime, :theta]),
                   :beta => mean => :beta_mean,
                   :beta => sem_or_zero => :beta_se,
                   nrow => :n)
    sort!(long, [:topology, :regime, :L, :theta])
    return long
end

function aggregate_delta_realization(delta)
    long = combine(groupby(delta, [:L, :topology, :regime]),
                   :delta_S => mean => :delta_S_mean,
                   :delta_S => sem_or_zero => :delta_S_se,
                   nrow => :n)
    sort!(long, [:topology, :regime, :L])
    return long
end

function aggregate_delta_by_theta(delta)
    long = combine(groupby(delta, [:L, :topology, :regime, :theta]),
                   :delta_S => mean => :delta_S_mean,
                   :delta_S => sem_or_zero => :delta_S_se,
                   nrow => :n)
    sort!(long, [:topology, :regime, :L, :theta])
    return long
end

function realization_shift_tables(long, log_floor)
    rows = NamedTuple[]
    for theta in sort(unique(long.theta))
        sub = filter(:theta => ==(theta), long)
        sort!(sub, :L)
        L = Float64.(sub.L)
        pcs = Float64.(sub.p_c)

        pcinf1 = pcs[argmax(L)]
        fit1 = shift_fit(L, pcs, pcinf1, log_floor;
                         mask = L .< maximum(L))

        inv_fit = linear_fit(1.0 ./ L, pcs)
        pcinf2 = inv_fit.intercept
        fit2 = shift_fit(L, pcs, pcinf2, log_floor;
                         mask = trues(length(L)))

        push!(rows, (theta = theta, method = "largest_L",
                     pc_inf = pcinf1, slope = fit1.slope,
                     nu = fit1.nu, slope_se = fit1.se_slope,
                     nu_se = fit1.nu_se, r2 = fit1.r2,
                     n_points = fit1.n_points,
                     n_zero_replaced = fit1.n_zero_replaced))
        push!(rows, (theta = theta, method = "invL_fit",
                     pc_inf = pcinf2, slope = fit2.slope,
                     nu = fit2.nu, slope_se = fit2.se_slope,
                     nu_se = fit2.nu_se, r2 = fit2.r2,
                     n_points = fit2.n_points,
                     n_zero_replaced = fit2.n_zero_replaced))
    end

    summary = DataFrame(rows)
    wide_rows = NamedTuple[]
    for theta in sort(unique(summary.theta))
        sub = filter(:theta => ==(theta), summary)
        nu1 = sub[sub.method .== "largest_L", :nu][1]
        nu2 = sub[sub.method .== "invL_fit", :nu][1]
        push!(wide_rows, (theta = theta, nu_largest_L = nu1,
                          nu_invL_fit = nu2))
    end
    return summary, DataFrame(wide_rows)
end

function nu_source_stats(wide, source)
    rows = NamedTuple[]
    for (method, col) in (("largest_L", :nu_largest_L),
                          ("invL_fit", :nu_invL_fit))
        v = Float64.(wide[!, col])
        push!(rows, (source = source, method = method, n = length(v),
                     median = median(v), mean = mean(v),
                     sd = std(v; corrected = true),
                     min = minimum(v), max = maximum(v)))
    end
    return rows
end

function write_nu_comparison(dest, real_wide)
    summary_path = joinpath(dest, "pc_shift_nu_crosscheck.csv")
    isfile(summary_path) || return
    summary_wide = CSV.read(summary_path, DataFrame)

    lookup = Dict(row.theta => row for row in eachrow(real_wide))
    rows = NamedTuple[]
    for row in eachrow(summary_wide)
        r = get(lookup, row.theta, nothing)
        push!(rows, (theta = row.theta,
                     nu_summary_largest_L = row.nu_largest_L,
                     nu_realization_largest_L = r === nothing ? NaN : r.nu_largest_L,
                     nu_summary_invL_fit = row.nu_invL_fit,
                     nu_realization_invL_fit = r === nothing ? NaN : r.nu_invL_fit))
    end
    CSV.write(joinpath(dest, "nu_comparison_by_theta.csv"), DataFrame(rows))

    stats = vcat(nu_source_stats(summary_wide, "summary"),
                 nu_source_stats(real_wide, "realization"))
    CSV.write(joinpath(dest, "nu_source_summary.csv"), DataFrame(stats))
end

function write_beta_comparison(analysis_dir, dest, beta_real)
    summary_path = joinpath(analysis_dir, "beta_vs_L.csv")
    isfile(summary_path) || return
    summary = CSV.read(summary_path, DataFrame)
    lookup = Dict((row.L, row.topology, row.regime) => row
                  for row in eachrow(summary))

    rows = NamedTuple[]
    for row in eachrow(beta_real)
        s = get(lookup, (row.L, row.topology, row.regime), nothing)
        push!(rows, (L = row.L, topology = row.topology,
                     regime = row.regime,
                     beta_summary = s === nothing ? NaN : s.beta_mean,
                     beta_summary_se = s === nothing ? NaN : s.beta_se,
                     n_summary = s === nothing ? 0 : s.n,
                     beta_realization = row.beta_mean,
                     beta_realization_se = row.beta_se,
                     n_realization = row.n))
    end
    CSV.write(joinpath(dest, "beta_source_comparison.csv"),
              DataFrame(rows))
end

function write_delta_comparison(analysis_dir, dest, delta_real)
    summary_path = joinpath(analysis_dir, "delta_S_vs_L.csv")
    isfile(summary_path) || return
    summary = CSV.read(summary_path, DataFrame)
    lookup = Dict((row.L, row.topology, row.regime) => row
                  for row in eachrow(summary))

    rows = NamedTuple[]
    for row in eachrow(delta_real)
        s = get(lookup, (row.L, row.topology, row.regime), nothing)
        push!(rows, (L = row.L, topology = row.topology,
                     regime = row.regime,
                     delta_S_summary = s === nothing ? NaN : s.delta_S_mean,
                     delta_S_summary_se = s === nothing ? NaN : s.delta_S_se,
                     n_summary = s === nothing ? 0 : s.n,
                     delta_S_realization = row.delta_S_mean,
                     delta_S_realization_se = row.delta_S_se,
                     n_realization = row.n))
    end
    CSV.write(joinpath(dest, "deltaS_source_comparison.csv"),
              DataFrame(rows))
end

function plot_pc_comparison(summary_long, real_long, theta, outpath)
    s = filter(:theta => ==(theta), summary_long)
    r = filter(:theta => ==(theta), real_long)
    sort!(s, :L)
    sort!(r, :L)

    plt = plot()
    plot!(plt, s.L, s.p_c, yerror = s.p_c_sem, marker = :circle,
          color = :dodgerblue, label = "summary p_c")
    plot!(plt, r.L, r.p_c, yerror = r.p_c_sem, marker = :diamond,
          color = :firebrick, label = "per-realization p_c")
    xlabel!(plt, "L")
    ylabel!(plt, "p_c")
    title!(plt, "p_c(L), θ=$(round(theta, digits = 2))")
    savefig(plt, outpath)
end

function plot_nu_comparison(cmp, outpath)
    plt = plot(layout = (1, 2), size = (1100, 490), margin = 5Plots.mm)
    specs = [(1, :nu_summary_largest_L, :nu_realization_largest_L,
              "Largest-L"),
             (2, :nu_summary_invL_fit, :nu_realization_invL_fit,
              "1/L fit")]
    for (i, summary_col, real_col, label) in specs
        plot!(plt, cmp.theta, cmp[!, summary_col], subplot = i,
              marker = :circle, color = :dodgerblue, label = "summary")
        plot!(plt, cmp.theta, cmp[!, real_col], subplot = i,
              marker = :diamond, color = :firebrick,
              label = "per realization")
        title!(plt, label, subplot = i)
        xlabel!(plt, "θ", subplot = i)
        ylabel!(plt, "ν", subplot = i)
    end
    savefig(plt, outpath)
end

function plot_beta_comparison(cmp, outpath)
    df = filter(:topology => ==("waxman"), cmp)
    plt = plot(layout = (1, 2), size = (1100, 490), margin = 5Plots.mm)
    for (i, regime) in enumerate(("negative", "positive"))
        sub = filter(:regime => ==(regime), df)
        sort!(sub, :L)
        plot!(plt, sub.L, sub.beta_summary,
              yerror = sub.beta_summary_se, subplot = i,
              marker = :circle, color = :dodgerblue,
              label = "mean-S fit")
        plot!(plt, sub.L, sub.beta_realization,
              yerror = sub.beta_realization_se, subplot = i,
              marker = :diamond, color = :firebrick,
              label = "per realization")
        title!(plt, regime == "negative" ? "Regime B (θ<0)" :
                     "Regime A (θ>0)", subplot = i)
        xlabel!(plt, "L", subplot = i)
        ylabel!(plt, "β", subplot = i)
    end
    savefig(plt, outpath)
end

function plot_delta_comparison(cmp, outpath)
    df = filter(:topology => ==("waxman"), cmp)
    plt = plot(layout = (1, 2), size = (1100, 490), margin = 5Plots.mm)
    for (i, regime) in enumerate(("negative", "positive"))
        sub = filter(:regime => ==(regime), df)
        sort!(sub, :L)
        plot!(plt, sub.L, sub.delta_S_summary,
              yerror = sub.delta_S_summary_se, subplot = i,
              marker = :circle, color = :dodgerblue,
              label = "mean-S fit")
        plot!(plt, sub.L, sub.delta_S_realization,
              yerror = sub.delta_S_realization_se, subplot = i,
              marker = :diamond, color = :firebrick,
              label = "per realization")
        title!(plt, regime == "negative" ? "Regime B (θ<0)" :
                     "Regime A (θ>0)", subplot = i)
        xlabel!(plt, "L", subplot = i)
        ylabel!(plt, "ΔS", subplot = i)
    end
    savefig(plt, outpath)
end

function plot_metric_vs_L_realization(df, value_col, se_col, ylabel,
                                      base, outdir)
    dfw = filter(:topology => ==("waxman"), df)

    plt = plot()
    for regime in ("negative", "positive")
        sub = filter(:regime => ==(regime), dfw)
        sort!(sub, :L)
        plot!(plt, sub.L, sub[!, value_col], yerror = sub[!, se_col],
              marker = :circle,
              label = regime == "negative" ? "Regime B (θ<0)" :
                      "Regime A (θ>0)")
    end
    xlabel!(plt, "L")
    ylabel!(plt, ylabel)
    title!(plt, "$(ylabel)(L), per realization")
    savefig(plt, joinpath(outdir, base * ".png"))

    for regime in ("negative", "positive")
        sub = filter(:regime => ==(regime), dfw)
        sort!(sub, :L)
        title_regime = regime == "negative" ? "Regime B (θ<0)" :
                       "Regime A (θ>0)"
        p = plot(sub.L, sub[!, value_col], yerror = sub[!, se_col],
                 marker = :circle, color = :firebrick,
                 xlabel = "L", ylabel = ylabel,
                 title = "$(ylabel)(L), $(title_regime), per realization")
        savefig(p, joinpath(outdir, base * "_$(regime).png"))
    end
end

function run_pc_realization_plots(; analysis_dir, dest, plot_dir)
    pc_real_path = joinpath(dest, "pc_realization_L_vs_theta_long.csv")
    nu_path = joinpath(dest, "nu_realization.csv")
    isfile(pc_real_path) ||
        error("missing $pc_real_path; run pc-realization first")
    isfile(nu_path) ||
        error("missing $nu_path; run pc-realization first")

    pc_long = CSV.read(pc_real_path, DataFrame)
    nu_tab = CSV.read(nu_path, DataFrame)

    s_path = joinpath(analysis_dir, "S_mean.csv")
    pvals = isfile(s_path) ?
            sort(unique(CSV.read(s_path, DataFrame).p)) : Float64[]
    dp = length(pvals) > 1 ? minimum(diff(pvals)) : 0.5 / 49.0
    log_floor = dp / 2.0

    summary_dir = joinpath(plot_dir, "summary")
    mkpath(summary_dir)

    for theta in sort(unique(pc_long.theta))
        sub = filter(:theta => ==(theta), pc_long)
        sort!(sub, :L)
        L = Float64.(sub.L)
        pcs = Float64.(sub.p_c)

        nsub = filter(:theta => ==(theta), nu_tab)
        row1 = filter(:method => ==("largest_L"), nsub)[1, :]
        row2 = filter(:method => ==("invL_fit"), nsub)[1, :]

        fit1 = shift_fit(L, pcs, row1.pc_inf, log_floor;
                         mask = L .< maximum(L))
        fit2 = shift_fit(L, pcs, row2.pc_inf, log_floor;
                         mask = trues(length(L)))

        fname = "pc_shift_realization_theta=$(round(theta, digits = 2)).png"
        plot_shift(L, pcs, fit1, row1.pc_inf, fit2, row2.pc_inf,
                   theta, log_floor, joinpath(summary_dir, fname);
                   title_prefix = "p_c shift (per realization)")
    end

    summary_long_path = joinpath(dest, "pc_L_vs_theta_long.csv")
    if isfile(summary_long_path)
        summary_long = CSV.read(summary_long_path, DataFrame)
        for theta in sort(unique(pc_long.theta))
            fname = "pc_L_summary_vs_realization_theta=$(round(theta, digits = 2)).png"
            plot_pc_comparison(summary_long, pc_long, theta,
                               joinpath(summary_dir, fname))
        end
    end

    nu_cmp_path = joinpath(dest, "nu_comparison_by_theta.csv")
    if isfile(nu_cmp_path)
        plot_nu_comparison(CSV.read(nu_cmp_path, DataFrame),
                           joinpath(summary_dir,
                                    "pc_shift_nu_comparison.png"))
    end

    beta_cmp_path = joinpath(dest, "beta_source_comparison.csv")
    if isfile(beta_cmp_path)
        plot_beta_comparison(CSV.read(beta_cmp_path, DataFrame),
                             joinpath(summary_dir,
                                      "beta_vs_L_summary_vs_realization.png"))
    end

    delta_cmp_path = joinpath(dest, "deltaS_source_comparison.csv")
    if isfile(delta_cmp_path)
        plot_delta_comparison(CSV.read(delta_cmp_path, DataFrame),
                              joinpath(summary_dir,
                                       "deltaS_vs_L_summary_vs_realization.png"))
    end

    beta_real_path = joinpath(dest, "beta_realization_vs_L.csv")
    if isfile(beta_real_path)
        plot_metric_vs_L_realization(CSV.read(beta_real_path, DataFrame),
                                     :beta_mean, :beta_se, "β",
                                     "beta_vs_L_per_realization", summary_dir)
    end

    delta_real_path = joinpath(dest, "deltaS_realization_vs_L.csv")
    if isfile(delta_real_path)
        plot_metric_vs_L_realization(CSV.read(delta_real_path, DataFrame),
                                     :delta_S_mean, :delta_S_se, "ΔS",
                                     "deltaS_vs_L_per_realization", summary_dir)
    end

    println("Wrote per-realization plots to $summary_dir")
end

function aggregate_ratio_metric(df, value_col, mean_name, se_name, ratio)
    sub = df[abs.(df.zeta ./ df.L .- ratio) .< 1e-9, :]
    sub = filter(:topology => ==("waxman"), sub)
    long = combine(groupby(sub, [:L, :topology, :regime]),
                   value_col => mean => mean_name,
                   value_col => sem_or_zero => se_name,
                   nrow => :n)
    sort!(long, [:regime, :L])
    return long
end

function plot_ratio_metric(df, mean_col, se_col, ylabel, ratio, outpath)
    plt = plot()
    for regime in ("negative", "positive")
        sub = filter(:regime => ==(regime), df)
        sort!(sub, :L)
        plot!(plt, sub.L, sub[!, mean_col], yerror = sub[!, se_col],
              marker = :circle,
              label = regime == "negative" ? "Regime B (θ<0)" :
                      "Regime A (θ>0)")
    end
    xlabel!(plt, "L")
    ylabel!(plt, ylabel)
    title!(plt, "$(ylabel)(L), ζ/L = $(ratio), per realization")
    savefig(plt, outpath)
end

function run_ratio_plots(; raw_root, analysis_dir, dest, plot_dir)
    pc_raw, beta_raw, delta_raw, n_fail = collect_realization_fits(raw_root)

    summary_dir = joinpath(plot_dir, "summary")
    mkpath(summary_dir)

    for ratio in (0.05, 0.5)
        beta = aggregate_ratio_metric(beta_raw, :beta, :beta_mean, :beta_se,
                                      ratio)
        delta = aggregate_ratio_metric(delta_raw, :delta_S, :delta_S_mean,
                                       :delta_S_se, ratio)

        CSV.write(joinpath(dest, "beta_vs_L_ratio=$(ratio).csv"), beta)
        CSV.write(joinpath(dest, "deltaS_vs_L_ratio=$(ratio).csv"), delta)

        plot_ratio_metric(beta, :beta_mean, :beta_se, "β", ratio,
                          joinpath(summary_dir,
                                   "beta_vs_L_ratio=$(ratio).png"))
        plot_ratio_metric(delta, :delta_S_mean, :delta_S_se, "ΔS", ratio,
                          joinpath(summary_dir,
                                   "deltaS_vs_L_ratio=$(ratio).png"))
    end

    pc_long = aggregate_pc_realization(pc_raw)

    s_path = joinpath(analysis_dir, "S_mean.csv")
    pvals = isfile(s_path) ?
            sort(unique(CSV.read(s_path, DataFrame).p)) : Float64[]
    dp = length(pvals) > 1 ? minimum(diff(pvals)) : 0.5 / 49.0
    log_floor = dp / 2.0

    _, nu = realization_shift_tables(pc_long, log_floor)
    CSV.write(joinpath(dest, "nu_vs_theta.csv"), nu)

    plt = plot()
    plot!(plt, nu.theta, nu.nu_largest_L, marker = :circle,
          color = :dodgerblue, label = "Largest-L")
    plot!(plt, nu.theta, nu.nu_invL_fit, marker = :diamond,
          color = :firebrick, label = "1/L fit")
    xlabel!(plt, "θ")
    ylabel!(plt, "ν")
    title!(plt, "ν(θ), per realization")
    savefig(plt, joinpath(summary_dir, "nu_vs_theta.png"))

    println("Wrote ratio tables to $dest")
    println("Wrote ratio/ν plots to $summary_dir")
    println("  β fit failures: $n_fail")
end

function run_pc_shift_realization(; raw_root, analysis_dir, dest)
    mkpath(dest)

    pc_raw, beta_raw, delta_raw, n_fail = collect_realization_fits(raw_root)
    CSV.write(joinpath(dest, "pc_realization_raw.csv"), pc_raw)
    CSV.write(joinpath(dest, "beta_realization_raw.csv"), beta_raw)
    CSV.write(joinpath(dest, "deltaS_realization_raw.csv"), delta_raw)

    pc_long = aggregate_pc_realization(pc_raw)
    write_wide_pc(pc_long, dest;
                  filename = "pc_realization_L_vs_theta.csv")
    CSV.write(joinpath(dest, "pc_realization_L_vs_theta_long.csv"), pc_long)

    beta_vs_L = aggregate_beta_realization(beta_raw)
    CSV.write(joinpath(dest, "beta_realization_vs_L.csv"), beta_vs_L)
    CSV.write(joinpath(dest, "beta_vs_L.csv"), beta_vs_L)

    delta_vs_L = aggregate_delta_realization(delta_raw)
    CSV.write(joinpath(dest, "deltaS_realization_vs_L.csv"), delta_vs_L)
    CSV.write(joinpath(dest, "deltaS_vs_L.csv"), delta_vs_L)

    for regime in ("negative", "positive")
        beta_reg = filter(:regime => ==(regime), beta_vs_L)
        delta_reg = filter(:regime => ==(regime), delta_vs_L)
        CSV.write(joinpath(dest, "beta_vs_L_$(regime).csv"), beta_reg)
        CSV.write(joinpath(dest, "deltaS_vs_L_$(regime).csv"), delta_reg)
    end

    beta_by_theta = aggregate_beta_by_theta(beta_raw)
    CSV.write(joinpath(dest, "beta_realization_by_theta.csv"), beta_by_theta)

    delta_by_theta = aggregate_delta_by_theta(delta_raw)
    CSV.write(joinpath(dest, "deltaS_realization_by_theta.csv"),
              delta_by_theta)

    s_path = joinpath(analysis_dir, "S_mean.csv")
    pvals = isfile(s_path) ?
            sort(unique(CSV.read(s_path, DataFrame).p)) : Float64[]
    dp = length(pvals) > 1 ? minimum(diff(pvals)) : 0.5 / 49.0
    log_floor = dp / 2.0

    nu_real, nu_real_wide = realization_shift_tables(pc_long, log_floor)
    CSV.write(joinpath(dest, "nu_realization.csv"), nu_real)
    CSV.write(joinpath(dest, "nu_realization_crosscheck.csv"), nu_real_wide)

    write_nu_comparison(dest, nu_real_wide)
    write_beta_comparison(analysis_dir, dest, beta_vs_L)
    write_delta_comparison(analysis_dir, dest, delta_vs_L)

    println("Wrote per-realization p_c/β/ΔS tables to $dest")
    println("  β fit failures: $n_fail")
end

function run_pc_shift(; analysis_dir, dest, plot_dir)
    mkpath(dest)
    pc_path = joinpath(analysis_dir, "p_c_summary.csv")
    isfile(pc_path) || error("missing $pc_path; run aggregate first")

    pc = CSV.read(pc_path, DataFrame)
    long = average_over_zeta(pc)
    write_wide_pc(long, dest)
    CSV.write(joinpath(dest, "pc_L_vs_theta_long.csv"), long)

    s_path = joinpath(analysis_dir, "S_mean.csv")
    pvals = isfile(s_path) ?
            sort(unique(CSV.read(s_path, DataFrame).p)) : Float64[]
    dp = length(pvals) > 1 ? minimum(diff(pvals)) : 0.5 / 49.0
    log_floor = dp / 2.0

    summary_dir = joinpath(plot_dir, "summary")
    mkpath(summary_dir)

    rows = NamedTuple[]
    for theta in sort(unique(long.theta))
        sub = filter(:theta => ==(theta), long)
        sort!(sub, :L)
        L = Float64.(sub.L)
        pcs = Float64.(sub.p_c)

        pcinf1 = pcs[argmax(L)]
        fit1 = shift_fit(L, pcs, pcinf1, log_floor; mask = L .< maximum(L))

        inv_fit = linear_fit(1.0 ./ L, pcs)
        pcinf2 = inv_fit.intercept
        fit2 = shift_fit(L, pcs, pcinf2, log_floor;
                        mask = trues(length(L)))

        push!(rows, (theta = theta, method = "largest_L",
                     pc_inf = pcinf1, slope = fit1.slope,
                     nu = fit1.nu, slope_se = fit1.se_slope,
                     nu_se = fit1.nu_se, r2 = fit1.r2,
                     n_points = fit1.n_points,
                     n_zero_replaced = fit1.n_zero_replaced))
        push!(rows, (theta = theta, method = "invL_fit",
                     pc_inf = pcinf2, slope = fit2.slope,
                     nu = fit2.nu, slope_se = fit2.se_slope,
                     nu_se = fit2.nu_se, r2 = fit2.r2,
                     n_points = fit2.n_points,
                     n_zero_replaced = fit2.n_zero_replaced))

        fname = "pc_shift_theta=$(round(theta, digits = 2)).png"
        plot_shift(L, pcs, fit1, pcinf1, fit2, pcinf2, theta,
                   log_floor, joinpath(summary_dir, fname))
    end

    CSV.write(joinpath(dest, "pc_shift_nu.csv"), DataFrame(rows))

    wide_rows = NamedTuple[]
    summary = DataFrame(rows)
    for theta in sort(unique(summary.theta))
        sub = filter(:theta => ==(theta), summary)
        nu1 = sub[sub.method .== "largest_L", :nu][1]
        nu2 = sub[sub.method .== "invL_fit", :nu][1]
        push!(wide_rows, (theta = theta, nu_largest_L = nu1,
                          nu_invL_fit = nu2))
    end
    CSV.write(joinpath(dest, "pc_shift_nu_crosscheck.csv"),
              DataFrame(wide_rows))

    summary = DataFrame(rows)
    plt = plot()
    for (method, label, color) in (("largest_L", "Largest-L", :dodgerblue),
                                   ("invL_fit", "1/L fit", :firebrick))
        sub = filter(:method => ==(method), summary)
        sort!(sub, :theta)
        plot!(plt, sub.theta, sub.nu,
              marker = :circle, color = color, label = label)
    end
    xlabel!(plt, "θ")
    ylabel!(plt, "ν")
    title!(plt, "p_c shift exponent ν(θ)")
    savefig(plt, joinpath(summary_dir, "pc_shift_nu_vs_theta.png"))

    println("Wrote pc-shift tables to $dest")
    println("Wrote pc-shift plots to $summary_dir")
end

end
