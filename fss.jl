using CSV
using DataFrames
using LinearAlgebra
using Statistics

const DEFAULT_ROOT = joinpath("results", "analysis")

function parse_float_list(text)
    isempty(strip(text)) && return Float64[]
    return parse.(Float64, split(text, ","))
end

function parse_args(args)
    root = DEFAULT_ROOT
    out = root
    window = 0.1
    min_points = 3
    ratios = Float64[]
    thetas = Float64[]

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--root" && i < length(args)
            root = args[i + 1]
            out = root
            i += 2
        elseif arg == "--out" && i < length(args)
            out = args[i + 1]
            i += 2
        elseif arg == "--window" && i < length(args)
            window = parse(Float64, args[i + 1])
            i += 2
        elseif arg == "--min-points" && i < length(args)
            min_points = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--ratios" && i < length(args)
            ratios = parse_float_list(args[i + 1])
            i += 2
        elseif arg == "--thetas" && i < length(args)
            thetas = parse_float_list(args[i + 1])
            i += 2
        elseif arg == "--help" || arg == "-h"
            println("Usage: julia fss.jl [--root DIR] [--out DIR] " *
                    "[--window 0.1] [--min-points 3] " *
                    "[--ratios 0.1,0.25,0.5] [--thetas -0.4,-0.2,0.2,0.4]")
            exit(0)
        else
            error("unknown argument: $arg")
        end
    end
    return (; root, out, window, min_points, ratios, thetas)
end

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
        df = filter(:topology => ==(topology), rows)
        Ls = sort(unique(df.L))
        push!(lines, "topology = $topology")
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
    return lines
end

function main()
    opts = parse_args(ARGS)
    summary_path = joinpath(opts.root, "p_c_summary.csv")
    s_path = joinpath(opts.root, "S_mean.csv")
    isfile(summary_path) || error("missing $summary_path")
    isfile(s_path) || error("missing $s_path")

    summary = CSV.read(summary_path, DataFrame)
    S_long = CSV.read(s_path, DataFrame)
    mkpath(opts.out)

    if !isempty(opts.ratios)
        keep = [any(isapprox(row.zeta / row.L, r; atol = 1e-9)
                    for r in opts.ratios) for row in eachrow(summary)]
        summary = summary[keep, :]
    end

    if !isempty(opts.thetas)
        keep = [any(isapprox(row.theta, t; atol = 1e-6)
                    for t in opts.thetas) for row in eachrow(summary)]
        summary = summary[keep, :]
    end

    isempty(summary) && error("no summary rows match the requested ratios/thetas")

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
        fit = fit_beta(p, S, row.p_c, row.p_index, opts.window, opts.min_points)
        fit === nothing && continue

        push!(beta_rows, (
            L = row.L,
            zeta = row.zeta,
            topology = row.topology,
            theta = row.theta,
            p_c = row.p_c,
            window = opts.window,
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
    CSV.write(joinpath(opts.out, "beta_fits.csv"), beta_df)

    beta_vs_L = NamedTuple[]
    for topology in sort(unique(beta_df.topology))
        df_topo = filter(:topology => ==(topology), beta_df)
        for L in sort(unique(df_topo.L))
            vals = filter(:L => ==(L), df_topo).beta
            push!(beta_vs_L, (
                L = L,
                topology = topology,
                n = length(vals),
                beta_mean = mean(vals),
                beta_se = group_sem(vals),
            ))
        end
    end
    beta_vs_L_df = DataFrame(beta_vs_L)
    CSV.write(joinpath(opts.out, "beta_vs_L.csv"), beta_vs_L_df)

    delta_rows = NamedTuple[]
    for topology in sort(unique(summary.topology))
        df_topo = filter(:topology => ==(topology), summary)
        for L in sort(unique(df_topo.L))
            vals = filter(:L => ==(L), df_topo).delta_S
            push!(delta_rows, (
                L = L,
                topology = topology,
                n = length(vals),
                delta_S_mean = mean(vals),
                delta_S_se = group_sem(vals),
            ))
        end
    end
    delta_df = DataFrame(delta_rows)
    CSV.write(joinpath(opts.out, "delta_S_vs_L.csv"), delta_df)

    report = vcat(
        saturation_report(beta_vs_L_df, "beta", :beta_mean, :beta_se),
        ["", "delta_S saturation"],
        saturation_report(delta_df, "delta_S", :delta_S_mean, :delta_S_se),
    )
    write(joinpath(opts.out, "fss_report.txt"), join(report, "\n") * "\n")

    println("Wrote beta/FSS results to $(opts.out)")
end

main()
