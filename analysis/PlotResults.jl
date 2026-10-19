module PlotResults

using CSV
using DataFrames
using LinearAlgebra
using Statistics

ENV["GKSwstype"] = "100"
using Plots

using ..PCUtils

export run_plots, run_beta_zeta

function parse_results(results_dir)
    data = Dict{String,Tuple{Vector{Float64},Vector{Float64},Matrix{Float64},Vector{String}}}()
    for name in sort(readdir(results_dir))
        endswith(name, ".csv") || continue
        (occursin(" mean", name) || occursin(" sem", name)) && continue
        (occursin("Gt Component", name) ||
         occursin("Cascade Time", name) ||
         occursin("degree-distribution", name)) || continue

        path = joinpath(results_dir, name)
        df = CSV.read(path, DataFrame; header = 1)
        header = string.(names(df))
        p = parse.(Float64, header[2:end])
        θ = Float64.(df[!, 1])
        mat = Matrix{Float64}(df[:, 2:end])
        data[name] = (θ, p, mat, header)
    end
    return data
end

function plot_comparison(mat, θ_list, p_list, ylabel, title, outpath)
    plt = plot()
    for (i, θ) in enumerate(θ_list)
        plot!(plt, p_list, mat[i, :], lw = 0.75,
              label = "θ=$(round(θ, digits = 2))")
    end
    xlabel!(plt, "p")
    ylabel!(plt, ylabel)
    title!(plt, title)
    savefig(plt, outpath)
end

function plot_heatmap(mat, θ_list, p_list, title, outpath)
    plt = heatmap(p_list, θ_list, mat,
                  xlabel = "p", ylabel = "θ", title = title)
    savefig(plt, outpath)
end

function plot_degree_heatmaps(data, p_list, plots_dir)
    for (name, (θ_col, _, mat, _)) in data
        occursin("degree-distribution", name) || continue
        θ = θ_col[1]
        k_vals = collect(1:size(mat, 1))

        plt = heatmap(p_list, k_vals, mat,
                      xlabel = "degree k", ylabel = "p",
                      title = "θ=$(round(θ, digits = 2)), heatmap of final pₖ")
        outpath = joinpath(plots_dir, "degree-heatmap_θ=$(round(θ, digits = 2)).png")
        savefig(plt, outpath)
    end
end

function process_one_dir(csv_dir, plots_dir)
    mkpath(plots_dir)

    data = parse_results(csv_dir)
    isempty(data) && (println("  No CSV files found in $csv_dir"); return)

    s_keys = [k for k in keys(data) if occursin("Gt Component", k)]
    t_keys = [k for k in keys(data) if occursin("Cascade Time", k)]
    isempty(s_keys) && (println("  No giant component CSV in $csv_dir"); return)

    θ_S, p_list, S_mat, _ = data[s_keys[1]]
    θ_list = θ_S

    plot_comparison(S_mat, θ_list, p_list,
                    "S", "Giant component",
                    joinpath(plots_dir, "comparison_giant_component.png"))
    plot_heatmap(S_mat, θ_list, p_list, "GC heatmap",
                 joinpath(plots_dir, "heatmap_giant_component.png"))

    if !isempty(t_keys)
        _, t_p, t_mat, _ = data[t_keys[1]]
        if length(t_p) == length(p_list)
            plot_comparison(t_mat, θ_list, t_p,
                            "Cascade Time", "Cascade Time",
                            joinpath(plots_dir, "comparison_cascade_time.png"))
        end
    end

    plot_degree_heatmaps(data, p_list, plots_dir)
    println("  Plots written to $plots_dir")
end

function select_first_paths(runs)
    selected = Dict{Tuple{Int,Float64,String},String}()
    for run in runs
        key = (run.L, run.zeta, run.topology)
        if !haskey(selected, key) || basename(run.path) == "realization = 0"
            selected[key] = run.path
        end
    end
    return sort(collect(values(selected)))
end

function plot_pc_heatmaps(pc::DataFrame, out_dir)
    for topology in sort(unique(pc.topology))
        df_topo = filter(:topology => ==(topology), pc)
        for L in sort(unique(df_topo.L))
            sub = filter(:L => ==(L), df_topo)
            θs = sort(unique(sub.theta))
            ζs = sort(unique(sub.zeta))
            mat = fill(NaN, length(ζs), length(θs))
            for row in eachrow(sub)
                iz = findfirst(==(row.zeta), ζs)
                it = findfirst(==(row.theta), θs)
                mat[iz, it] = row.p_c
            end

            suffix = topology == "waxman" ? "" : "_$(topology)"
            fname = "p_c_heatmap_L=$(L)$(suffix).png"
            plt = heatmap(θs, ζs, mat,
                          xlabel = "θ", ylabel = "ζ",
                          title = "p_c, L=$(L) ($(topology))")
            savefig(plt, joinpath(out_dir, fname))
        end
    end
end

function regime_label(regime)
    regime == "negative" && return "θ < 0"
    regime == "positive" && return "θ > 0"
    return String(regime)
end

function plot_scaling(df::DataFrame, value_col::Symbol, se_col::Symbol,
                      label, filename, out_dir; regime_filter = nothing)
    if !("regime" in names(df))
        df = copy(df)
        df[!, :regime] = fill("all", nrow(df))
    end

    if regime_filter !== nothing
        df = filter(:regime => ==(regime_filter), df)
        isempty(df) && return
    end

    plt = plot()
    if regime_filter === nothing
        for topology in sort(unique(df.topology))
            df_topo = filter(:topology => ==(topology), df)
            for regime in sort(unique(df_topo.regime))
                sub = filter(:regime => ==(regime), df_topo)
                sort!(sub, :L)
                label_str = topology == "waxman" ?
                    regime_label(regime) : "$(topology), $(regime_label(regime))"
                plot!(plt, sub.L, getproperty(sub, value_col),
                      yerror = getproperty(sub, se_col),
                      marker = :circle, label = label_str)
            end
        end
    else
        for topology in sort(unique(df.topology))
            sub = filter(:topology => ==(topology), df)
            sort!(sub, :L)
            plot!(plt, sub.L, getproperty(sub, value_col),
                  yerror = getproperty(sub, se_col),
                  marker = :circle, label = topology)
        end
    end

    xlabel!(plt, "L")
    ylabel!(plt, label)
    title!(plt, regime_filter === nothing ?
        "$(label)(L)" : "$(label)(L), $(regime_label(regime_filter))")
    savefig(plt, joinpath(out_dir, filename))
end

function plot_s_mean_ribbons(S_long::DataFrame, out_dir, ratios, thetas)
    if isempty(ratios)
        ratios = [0.1, 0.25, 0.5]
    end
    if isempty(thetas)
        thetas = [-0.4, -0.2, 0.2, 0.4]
    end

    if length(ratios) * length(thetas) > 24
        println("  Too many S(p) ribbon combinations; using representative subset")
        representative_ratios = [0.1, 0.25, 0.5]
        representative_thetas = [-0.4, -0.2, 0.2, 0.4]
        ratios = [r for r in representative_ratios
                  if any(isapprox(r, x; atol = 1e-9) for x in ratios)]
        thetas = [t for t in representative_thetas
                  if any(isapprox(t, x; atol = 1e-6) for x in thetas)]
        if isempty(ratios) || isempty(thetas)
            println("  Skipping S(p) ribbons: representative subset not present")
            return
        end
    end

    for ratio in ratios
        for theta in thetas
            plt = plot()
            added = false
            for L in sort(unique(S_long.L))
                mask = (S_long.L .== L) .&
                       (abs.(S_long.theta .- theta) .< 1e-6) .&
                       (abs.(S_long.zeta ./ S_long.L .- ratio) .< 1e-9)
                sub = S_long[mask, :]
                nrow(sub) == 0 && continue
                sort!(sub, :p)
                plot!(plt, sub.p, sub.S_mean, ribbon = sub.S_sem,
                      fillalpha = 0.2, label = "L=$(L)")
                added = true
            end
            added || continue
            xlabel!(plt, "p")
            ylabel!(plt, "S")
            title!(plt, "ratio=$(ratio), θ=$(theta)")
            fname = "S_mean_ratio=$(ratio)_theta=$(theta).png"
            savefig(plt, joinpath(out_dir, fname))
        end
    end
end

function filter_fss_rows(df::DataFrame, ratios, thetas)
    out = df
    if !isempty(ratios)
        keep = [any(isapprox(row.zeta / row.L, r; atol = 1e-9)
                    for r in ratios) for row in eachrow(out)]
        out = out[keep, :]
    end
    if !isempty(thetas)
        keep = [any(isapprox(row.theta, t; atol = 1e-6)
                    for t in thetas) for row in eachrow(out)]
        out = out[keep, :]
    end
    return out
end

function sem_or_zero(x)
    return length(x) > 1 ? std(x; corrected = true) / sqrt(length(x)) : 0.0
end

function plot_delta_S_over_sqrtL(pc::DataFrame, out, summary_dir, ratios, thetas)
    df = filter_fss_rows(pc, ratios, thetas)
    isempty(df) && return

    df = copy(df)
    df[!, :regime] = ifelse.(df.theta .< 0, "negative", "positive")
    df[!, :ratio] = df.zeta ./ df.L
    df[!, :L_over_zeta] = df.L ./ df.zeta

    g = combine(groupby(df, [:L, :topology, :regime, :ratio, :L_over_zeta]),
                :delta_S => mean => :delta_S_mean,
                :delta_S => sem_or_zero => :delta_S_se,
                nrow => :n)
    g[!, :delta_S_over_sqrtL] = g.delta_S_mean ./ sqrt.(g.L)
    g[!, :delta_S_over_sqrtL_se] = g.delta_S_se ./ sqrt.(g.L)
    sort!(g, [:regime, :topology, :L_over_zeta, :L])

    multi_topology = length(unique(g.topology)) > 1

    plt = plot()
    for topology in sort(unique(g.topology))
        for regime in sort(unique(g.regime))
            for lz in sort(unique(g.L_over_zeta))
                sub = filter(row -> row.topology == topology &&
                                    row.regime == regime &&
                                    row.L_over_zeta == lz, g)
                nrow(sub) < 2 && continue
                sort!(sub, :L)
                prefix = multi_topology ? "$(topology), " : ""
                plot!(plt, sub.L, sub.delta_S_over_sqrtL,
                      yerror = sub.delta_S_over_sqrtL_se,
                      marker = :circle,
                      label = "$(prefix)$(regime_label(regime)), L/ζ=$(round(lz, digits=2))")
            end
        end
    end
    xlabel!(plt, "L")
    ylabel!(plt, "ΔS / √L")
    title!(plt, "ΔS/√L at fixed L/ζ")
    savefig(plt, joinpath(summary_dir, "delta_S_over_sqrtL.png"))

    for regime in sort(unique(g.regime))
        plt = plot()
        for topology in sort(unique(g.topology))
            for lz in sort(unique(g.L_over_zeta))
                sub = filter(row -> row.topology == topology &&
                                    row.regime == regime &&
                                    row.L_over_zeta == lz, g)
                nrow(sub) < 2 && continue
                sort!(sub, :L)
                prefix = multi_topology ? "$(topology), " : ""
                plot!(plt, sub.L, sub.delta_S_over_sqrtL,
                      yerror = sub.delta_S_over_sqrtL_se,
                      marker = :circle,
                      label = "$(prefix)L/ζ=$(round(lz, digits=2))")
            end
        end
        xlabel!(plt, "L")
        ylabel!(plt, "ΔS / √L")
        title!(plt, "ΔS/√L, $(regime_label(regime))")
        savefig(plt, joinpath(summary_dir, "delta_S_over_sqrtL_$(regime).png"))
    end

    plt = plot()
    for topology in sort(unique(g.topology))
        for regime in sort(unique(g.regime))
            for L in sort(unique(g.L))
                sub = filter(row -> row.topology == topology &&
                                    row.regime == regime &&
                                    row.L == L, g)
                nrow(sub) == 0 && continue
                sort!(sub, :L_over_zeta)
                prefix = multi_topology ? "$(topology) " : ""
                plot!(plt, sub.L_over_zeta, sub.delta_S_over_sqrtL,
                      yerror = sub.delta_S_over_sqrtL_se,
                      seriestype = :scatter, marker = :circle,
                      label = "$(prefix)$(regime_label(regime)), L=$(L)")
            end
        end
    end
    xlabel!(plt, "L/ζ")
    ylabel!(plt, "ΔS / √L")
    title!(plt, "ΔS/√L vs L/ζ")
    savefig(plt, joinpath(summary_dir, "delta_S_over_sqrtL_vs_Lzeta.png"))

    CSV.write(joinpath(out, "delta_S_over_sqrtL.csv"), g)
    println("  Wrote ΔS/√L scaling to $(joinpath(out, "delta_S_over_sqrtL.csv"))")
end

function fit_scaling_alpha(df::DataFrame)
    best_rss = Inf
    best_alpha = 0.0
    ratios = sort(unique(df.ratio))
    for alpha in range(-1.0, 2.0; length = 6001)
        rss = 0.0
        for r in ratios
            s = filter(:ratio => ==(r), df)
            z = log.(s.y) .- alpha .* log.(s.L)
            rss += sum((z .- mean(z)) .^ 2)
        end
        if rss < best_rss
            best_rss = rss
            best_alpha = alpha
        end
    end
    return best_alpha, best_rss
end

function plot_delta_S_scaling(pc::DataFrame, out, summary_dir, ratios, thetas)
    df = filter_fss_rows(pc, ratios, thetas)
    isempty(df) && return

    df = copy(df)
    df[!, :regime] = ifelse.(df.theta .< 0, "negative", "positive")
    df[!, :ratio] = df.zeta ./ df.L
    df = df[df.delta_S .> 0, :]

    g = combine(groupby(df, [:L, :topology, :regime, :ratio]),
                :delta_S => mean => :y,
                :delta_S => sem_or_zero => :se,
                nrow => :n)
    g = g[g.y .> 0, :]

    multi_topology = length(unique(g.topology)) > 1
    fit_rows = NamedTuple[]
    data_rows = NamedTuple[]

    for topology in sort(unique(g.topology))
        for regime in sort(unique(g.regime))
            sub = filter(row -> row.topology == topology && row.regime == regime, g)
            nrow(sub) >= 4 || continue

            alpha, rss = fit_scaling_alpha(sub)
            sub = copy(sub)
            sub[!, :x] = 1.0 ./ sub.ratio
            sub[!, :scaled] = sub.y ./ (sub.L .^ alpha)
            sub[!, :scaled_se] = sub.se ./ (sub.L .^ alpha)

            rs = sort(unique(sub.ratio), by = r -> 1.0 / r)
            xs = 1.0 ./ rs
            fvals = [exp(mean(log.(filter(:ratio => ==(r), sub).scaled))) for r in rs]

            for row in eachrow(sub)
                fidx = findfirst(==(row.ratio), rs)
                push!(data_rows, (
                    L = row.L,
                    topology = topology,
                    regime = regime,
                    ratio = row.ratio,
                    L_over_zeta = row.x,
                    delta_S_mean = row.y,
                    delta_S_se = row.se,
                    alpha = alpha,
                    f_of_x = fvals[fidx],
                    scaled_delta_S = row.scaled,
                    scaled_delta_S_se = row.scaled_se,
                ))
            end

            log_rms = sqrt(rss / nrow(sub))
            push!(fit_rows, (
                topology = topology,
                regime = regime,
                alpha = alpha,
                log_rms = log_rms,
                pct_scatter = 100.0 * (exp(log_rms) - 1.0),
                n_points = nrow(sub),
            ))

            plt = plot()
            for L in sort(unique(sub.L))
                s = filter(:L => ==(L), sub)
                sort!(s, :x)
                plot!(plt, s.x, s.scaled, yerror = s.scaled_se,
                      seriestype = :scatter, marker = :circle,
                      label = "L=$(L)")
            end
            plot!(plt, xs, fvals, marker = :diamond, lw = 2,
                  label = "f(L/ζ)")
            plot!(plt, xscale = :log10)
            xlabel!(plt, "L/ζ")
            ylabel!(plt, "ΔS / L^α")
            title!(plt, "ΔS scaling, $(regime_label(regime)) " *
                        "(α=$(round(alpha, digits = 3)))")
            suffix = multi_topology ? "_$(topology)" : ""
            savefig(plt, joinpath(summary_dir,
                    "delta_S_scaling_$(regime)$(suffix).png"))
        end
    end

    isempty(fit_rows) && return
    CSV.write(joinpath(out, "delta_S_scaling_fit.csv"), DataFrame(fit_rows))
    CSV.write(joinpath(out, "delta_S_scaling.csv"), DataFrame(data_rows))
    println("  Wrote ΔS scaling fit to $(joinpath(out, "delta_S_scaling_fit.csv"))")
end

function plot_delta_S_scaling_zeta(pc::DataFrame, out, summary_dir, ratios, thetas)
    df = filter_fss_rows(pc, ratios, thetas)
    isempty(df) && return

    df = copy(df)
    df[!, :regime] = ifelse.(df.theta .< 0, "negative", "positive")
    df[!, :ratio] = df.zeta ./ df.L
    df = df[df.delta_S .> 0, :]

    g = combine(groupby(df, [:L, :topology, :regime, :ratio]),
                :delta_S => mean => :y,
                :delta_S => sem_or_zero => :se,
                nrow => :n)
    g = g[g.y .> 0, :]
    g[!, :x] = 1.0 ./ g.ratio

    fit_rows = NamedTuple[]
    data_rows = NamedTuple[]
    multi_topology = length(unique(g.topology)) > 1

    for topology in sort(unique(g.topology))
        for regime in sort(unique(g.regime))
            sub = filter(row -> row.topology == topology && row.regime == regime, g)
            nrow(sub) >= 4 || continue

            X = hcat(ones(nrow(sub)), log.(sub.L), log.(sub.x))
            coef = X \ log.(sub.y)
            fitted = X * coef
            r2 = 1.0 - sum((log.(sub.y) .- fitted) .^ 2) /
                       sum((log.(sub.y) .- mean(log.(sub.y))) .^ 2)
            rmse = sqrt(mean((log.(sub.y) .- fitted) .^ 2))
            alpha = coef[2]
            beta = coef[3]
            A = exp(coef[1])

            sub = copy(sub)
            sub[!, :alpha] = fill(alpha, nrow(sub))
            sub[!, :beta] = fill(beta, nrow(sub))
            sub[!, :A] = fill(A, nrow(sub))
            sub[!, :scaled] = sub.y ./ (sub.L .^ alpha)
            sub[!, :scaled_se] = sub.se ./ (sub.L .^ alpha)

            for row in eachrow(sub)
                push!(data_rows, (
                    L = row.L,
                    topology = topology,
                    regime = regime,
                    ratio = row.ratio,
                    L_over_zeta = row.x,
                    delta_S_mean = row.y,
                    delta_S_se = row.se,
                    alpha = alpha,
                    beta = beta,
                    A = A,
                    scaled_delta_S = row.scaled,
                    scaled_delta_S_se = row.scaled_se,
                    fitted_scaled = A * row.x^beta,
                ))
            end

            push!(fit_rows, (
                topology = topology,
                regime = regime,
                alpha = alpha,
                beta = beta,
                A = A,
                R2 = r2,
                log_rmse = rmse,
                n_points = nrow(sub),
            ))

            plt = plot()
            for L in sort(unique(sub.L))
                s = filter(:L => ==(L), sub)
                sort!(s, :x)
                plot!(plt, s.x, s.scaled, yerror = s.scaled_se,
                      seriestype = :scatter, marker = :circle,
                      label = "L=$(L)")
            end
            xs_grid = exp.(range(log(minimum(sub.x)), log(maximum(sub.x));
                                length = 100))
            plot!(plt, xs_grid, A .* xs_grid .^ beta, lw = 2,
                  label = "A (L/ζ)^β")
            plot!(plt, xscale = :log10)
            xlabel!(plt, "L/ζ")
            ylabel!(plt, "ΔS / L^α")
            title!(plt, "ζ-dependent ΔS scaling, $(regime_label(regime)) " *
                        "(α=$(round(alpha, digits = 3)), " *
                        "β=$(round(beta, digits = 3)))")
            suffix = multi_topology ? "_$(topology)" : ""
            savefig(plt, joinpath(summary_dir,
                    "delta_S_scaling_zeta_$(regime)$(suffix).png"))
        end
    end

    isempty(fit_rows) && return
    CSV.write(joinpath(out, "delta_S_scaling_zeta_fit.csv"), DataFrame(fit_rows))
    CSV.write(joinpath(out, "delta_S_scaling_zeta.csv"), DataFrame(data_rows))
    println("  Wrote ζ-dependent ΔS scaling fit to " *
            "$(joinpath(out, "delta_S_scaling_zeta_fit.csv"))")
end

function run_beta_zeta(; fit_file, out, plot_dir)
    isfile(fit_file) || error("missing $fit_file")

    d = CSV.read(fit_file, DataFrame)
    d[!, :regime] = ifelse.(d.theta .< 0, "negative", "positive")

    g = combine(groupby(d, [:L, :topology, :regime, :zeta]),
                :beta => mean => :beta_mean,
                :beta => sem_or_zero => :beta_se,
                nrow => :n)
    sort!(g, [:regime, :topology, :L, :zeta])

    test_rows = NamedTuple[]
    for topology in sort(unique(g.topology))
        for regime in sort(unique(g.regime))
            for L in sort(unique(g.L))
                sub = filter(row -> row.topology == topology &&
                                    row.regime == regime &&
                                    row.L == L, g)
                n = nrow(sub)
                n >= 2 || continue

                w = 1.0 ./ max.(sub.beta_se, 1e-6) .^ 2
                wmean = sum(w .* sub.beta_mean) / sum(w)
                chi2 = sum(w .* (sub.beta_mean .- wmean) .^ 2)
                dof = n - 1
                chi2red = dof > 0 ? chi2 / dof : NaN

                slope = NaN
                slope_se = NaN
                slope_sigma = NaN
                if n >= 3
                    X = hcat(ones(n), sub.zeta)
                    W = Diagonal(w)
                    Xt = transpose(X)
                    coef = (Xt * W * X) \ (Xt * W * sub.beta_mean)
                    cov = inv(Xt * W * X)
                    slope = coef[2]
                    slope_se = sqrt(max(cov[2, 2], 0.0))
                    slope_sigma = slope_se > 0 ? slope / slope_se : NaN
                end

                detectable = (isfinite(chi2red) && chi2red > 2.0) ||
                             (isfinite(slope_sigma) && abs(slope_sigma) > 2.0)

                push!(test_rows, (
                    L = L,
                    topology = topology,
                    regime = regime,
                    n_zeta = n,
                    weighted_mean_beta = wmean,
                    chi2 = chi2,
                    dof = dof,
                    chi2red = chi2red,
                    slope = slope,
                    slope_se = slope_se,
                    slope_sigma = slope_sigma,
                    detectable_zeta_dependence = detectable,
                ))
            end
        end
    end

    g[!, :ratio] = g.zeta ./ g.L
    g[!, :L_over_zeta] = g.L ./ g.zeta

    CSV.write(joinpath(out, "beta_vs_zeta.csv"), g)
    CSV.write(joinpath(out, "beta_vs_ratio.csv"), g)
    CSV.write(joinpath(out, "beta_zeta_test.csv"), DataFrame(test_rows))

    summary_dir = joinpath(plot_dir, "summary")
    mkpath(summary_dir)
    multi_topology = length(unique(g.topology)) > 1

    for regime in sort(unique(g.regime))
        plt = plot()
        for L in sort(unique(g.L))
            sub = filter(row -> row.regime == regime && row.L == L, g)
            for topology in sort(unique(sub.topology))
                s = filter(:topology => ==(topology), sub)
                sort!(s, :zeta)
                prefix = multi_topology ? "$topology, " : ""
                plot!(plt, s.zeta, s.beta_mean, yerror = s.beta_se,
                      marker = :circle, label = "$(prefix)L=$(L)")
            end
        end
        xlabel!(plt, "ζ")
        ylabel!(plt, "β")
        title!(plt, "β vs ζ, $(regime_label(regime))")
        savefig(plt, joinpath(summary_dir, "beta_vs_zeta_$(regime).png"))

        plt = plot()
        for L in sort(unique(g.L))
            sub = filter(row -> row.regime == regime && row.L == L, g)
            for topology in sort(unique(sub.topology))
                s = filter(:topology => ==(topology), sub)
                sort!(s, :ratio)
                prefix = multi_topology ? "$topology, " : ""
                plot!(plt, s.ratio, s.beta_mean, yerror = s.beta_se,
                      marker = :circle, label = "$(prefix)L=$(L)")
            end
        end
        xlabel!(plt, "ζ/L")
        ylabel!(plt, "β")
        title!(plt, "β vs ζ/L, $(regime_label(regime))")
        savefig(plt, joinpath(summary_dir,
                "beta_vs_zeta_over_L_$(regime).png"))

        plt = plot()
        for L in sort(unique(g.L))
            sub = filter(row -> row.regime == regime && row.L == L, g)
            for topology in sort(unique(sub.topology))
                s = filter(:topology => ==(topology), sub)
                sort!(s, :L_over_zeta)
                prefix = multi_topology ? "$topology, " : ""
                plot!(plt, s.L_over_zeta, s.beta_mean, yerror = s.beta_se,
                      marker = :circle, label = "$(prefix)L=$(L)")
            end
        end
        xlabel!(plt, "L/ζ")
        ylabel!(plt, "β")
        title!(plt, "β vs L/ζ, $(regime_label(regime))")
        savefig(plt, joinpath(summary_dir,
                "beta_vs_L_over_zeta_$(regime).png"))
    end

    sub_neg = filter(:regime => ==("negative"), g)
    for z in sort(unique(sub_neg.zeta))
        s = filter(:zeta => ==(z), sub_neg)
        plt = plot()
        for topology in sort(unique(s.topology))
            st = filter(:topology => ==(topology), s)
            sort!(st, :L_over_zeta)
            label = length(unique(s.topology)) > 1 ? topology : "β"
            plot!(plt, st.L_over_zeta, st.beta_mean, yerror = st.beta_se,
                  marker = :circle, label = label)
        end
        xlabel!(plt, "L/ζ")
        ylabel!(plt, "β")
        nL = length(unique(s.L))
        suffix = nL == 1 ? " (single L=$(first(s.L)))" : ""
        title!(plt, "β vs L/ζ, ζ=$(z), θ < 0$(suffix)")
        savefig(plt, joinpath(summary_dir,
                "beta_scaling_zeta=$(z)_negative.png"))
    end

    println("Wrote β vs ζ diagnostics to " *
            "$(joinpath(out, "beta_zeta_test.csv"))")
end

function run_summary_plots(out, plot_dir, ratios, thetas)
    summary_dir = joinpath(plot_dir, "summary")
    mkpath(summary_dir)

    pc_path = joinpath(out, "p_c_summary.csv")
    beta_path = joinpath(out, "beta_vs_L.csv")
    delta_path = joinpath(out, "delta_S_vs_L.csv")
    s_path = joinpath(out, "S_mean.csv")
    for path in (pc_path, beta_path, delta_path, s_path)
        isfile(path) || error("missing $path; run aggregate and fss first")
    end

    pc = CSV.read(pc_path, DataFrame)
    beta = CSV.read(beta_path, DataFrame)
    delta = CSV.read(delta_path, DataFrame)
    S_long = CSV.read(s_path, DataFrame)

    plot_pc_heatmaps(pc, summary_dir)
    plot_delta_S_over_sqrtL(pc, out, summary_dir, ratios, thetas)
    plot_delta_S_scaling(pc, out, summary_dir, ratios, thetas)
    plot_delta_S_scaling_zeta(pc, out, summary_dir, ratios, thetas)
    plot_scaling(beta, :beta_mean, :beta_se, "β", "beta_vs_L.png", summary_dir)
    plot_scaling(delta, :delta_S_mean, :delta_S_se, "ΔS",
                 "delta_S_vs_L.png", summary_dir)
    for regime in ("negative", "positive")
        plot_scaling(beta, :beta_mean, :beta_se, "β",
                     "beta_vs_L_$regime.png", summary_dir;
                     regime_filter = regime)
        plot_scaling(delta, :delta_S_mean, :delta_S_se, "ΔS",
                     "delta_S_vs_L_$regime.png", summary_dir;
                     regime_filter = regime)
    end
    plot_s_mean_ribbons(S_long, summary_dir, ratios, thetas)
    println("  Summary plots written to $summary_dir")
end

function run_plots(; root, out, plot_dir, mode = :summary,
                   ratios = Float64[], thetas = Float64[],
                   L_values = Int[])
    mode = Symbol(mode)
    mode in (:summary, :first, :all, :none) ||
        error("unknown plot mode: $mode")

    if mode in (:summary, :first, :all)
        run_summary_plots(out, plot_dir, ratios, thetas)
    end

    if mode in (:first, :all)
        runs = sort(discover_realizations(root);
                    by = r -> (r.L, r.topology, r.zeta, r.path))
        if !isempty(L_values)
            runs = [run for run in runs if run.L in L_values]
        end
        isempty(runs) && error("no realization directories found under $root " *
                               "for L values $(L_values)")
        paths = mode == :first ? select_first_paths(runs) : [r.path for r in runs]
        for path in paths
            dest = joinpath(plot_dir, relpath(path, root))
            process_one_dir(path, dest)
        end
    end

    mode == :none || println("Plots written under $plot_dir")
end

end
