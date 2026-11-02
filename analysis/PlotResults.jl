module PlotResults

using CSV
using DataFrames
using LinearAlgebra
using Statistics

ENV["GKSwstype"] = "100"
using Plots

using ..PCUtils

export run_plots, run_beta_zeta, run_aggregate_mean_plots, run_beta_heatmaps,
       run_S_ratio_heatmap, run_all_S_ratio_heatmaps, run_S_theta_heatmap

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

    df = filter(:topology => ==("waxman"), df)

    if regime_filter !== nothing
        df = filter(:regime => ==(regime_filter), df)
        isempty(df) && return
    end

    topo_order = sort(String.(unique(df.topology)); by = t -> (t == "ER", t))
    plt = plot()
    if regime_filter === nothing
        for topology in topo_order
            df_topo = filter(:topology => ==(topology), df)
            for regime in sort(unique(df_topo.regime))
                sub = filter(:regime => ==(regime), df_topo)
                sort!(sub, :L)
                label_str = topology == "waxman" ?
                    regime_label(regime) : "$(topology), $(regime_label(regime))"
                plot!(plt, sub.L, getproperty(sub, value_col),
                      yerror = getproperty(sub, se_col),
                      seriestype = topology == "ER" ? :scatter : :path,
                      marker = topology == "ER" ? :diamond : :circle,
                      markersize = topology == "ER" ? 8 : 4,
                      label = label_str)
            end
        end
    else
        for topology in topo_order
            sub = filter(:topology => ==(topology), df)
            sort!(sub, :L)
            plot!(plt, sub.L, getproperty(sub, value_col),
                  yerror = getproperty(sub, se_col),
                  seriestype = topology == "ER" ? :scatter : :path,
                  marker = topology == "ER" ? :diamond : :circle,
                  markersize = topology == "ER" ? 8 : 4,
                  label = topology)
        end
    end

    xlabel!(plt, "L")
    ylabel!(plt, label)
    title!(plt, regime_filter === nothing ?
        "$(label)(L)" : "$(label)(L), $(regime_label(regime_filter))")
    savefig(plt, joinpath(out_dir, filename))
end

function plot_variance_scaling(df::DataFrame, value_col::Symbol, se_col::Symbol,
                               label, base, out_dir)
    df = copy(df)
    df = filter(:topology => ==("waxman"), df)
    df[!, :variance] = getproperty(df, se_col) .^ 2

    function draw(sub, filename, title)
        isempty(sub) && return
        topo_order = sort(String.(unique(sub.topology)); by = t -> (t == "ER", t))
        plt = plot()
        for topology in topo_order
            for regime in sort(unique(sub.regime))
                s = filter(row -> row.topology == topology &&
                                  row.regime == regime, sub)
                nrow(s) == 0 && continue
                sort!(s, :L)
                prefix = length(topo_order) > 1 ? "$(topology), " : ""
                plot!(plt, s.L, s.variance,
                      seriestype = topology == "ER" ? :scatter : :path,
                      marker = topology == "ER" ? :diamond : :circle,
                      markersize = topology == "ER" ? 8 : 4,
                      label = "$(prefix)$(regime_label(regime))")
            end
        end
        xlabel!(plt, "L")
        ylabel!(plt, "Var($(label)) = SE²")
        title!(plt, title)
        savefig(plt, joinpath(out_dir, filename))
    end

    draw(df, "$(base).png", "Var($(label))(L)")
    for regime in sort(unique(df.regime))
        draw(filter(:regime => ==(regime), df),
             "$(base)_$(regime).png",
             "Var($(label))(L), $(regime_label(regime))")
    end
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

function sem_or_zero(x)
    return length(x) > 1 ? std(x; corrected = true) / sqrt(length(x)) : 0.0
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

function plot_beta_heatmap(fits, L, topology, regime, outpath)
    sub = filter(row -> row.L == L &&
                         row.topology == topology &&
                         row.regime == regime, fits)
    isempty(sub) && return

    ζs = sort(unique(sub.zeta))
    θs = sort(unique(sub.theta))
    mat = fill(NaN, length(θs), length(ζs))
    for row in eachrow(sub)
        i = findfirst(x -> isapprox(x, row.theta; atol = 1e-9), θs)
        j = findfirst(x -> isapprox(x, row.zeta; atol = 1e-9), ζs)
        (i === nothing || j === nothing) && continue
        mat[i, j] = row.beta
    end

    regime_name = regime == "negative" ? "B (θ < 0)" : "A (θ > 0)"
    topology_name = topology == "ER" ? "ER" : "Waxman"
    plt = heatmap(ζs, θs, mat,
                  xlabel = "ζ", ylabel = "θ",
                  color = :viridis, colorbar_title = "β",
                  title = "β(ζ, θ), L=$(L), regime $(regime_name), $(topology_name)")
    savefig(plt, outpath)
end

function run_beta_heatmaps(; fit_file, plot_dir, L_values = Int[],
                           topology = "waxman")
    isfile(fit_file) || error("missing $fit_file; run fss first")
    fits = CSV.read(fit_file, DataFrame)
    isempty(L_values) && (L_values = [maximum(fits.L)])

    summary_dir = joinpath(plot_dir, "summary")
    mkpath(summary_dir)
    for L in L_values
        for regime in ("negative", "positive")
            fname = "beta_heatmap_L=$(L)_$(regime).png"
            plot_beta_heatmap(fits, L, topology, regime,
                              joinpath(summary_dir, fname))
        end
    end
    println("  β heatmaps written to $summary_dir")
end

function run_S_ratio_heatmap(; out, plot_dir, theta, L = 100,
                             topology = "waxman")
    s_path = joinpath(out, "S_mean.csv")
    isfile(s_path) || error("missing $s_path; run aggregate first")

    S_long = CSV.read(s_path, DataFrame)
    sub = filter(row -> row.L == L &&
                         row.topology == topology &&
                         isapprox(row.theta, theta; atol = 1e-9), S_long)
    isempty(sub) && error("no S(p) rows for L=$L, θ=$theta, topology=$topology")

    sub[!, :ratio] = sub.zeta ./ L
    ratios = sort(unique(sub.ratio))
    pvals = sort(unique(sub.p))
    mat = fill(NaN, length(ratios), length(pvals))
    for row in eachrow(sub)
        i = findfirst(x -> isapprox(x, row.ratio; atol = 1e-9), ratios)
        j = findfirst(x -> isapprox(x, row.p; atol = 1e-12), pvals)
        (i === nothing || j === nothing) && continue
        mat[i, j] = row.S_mean
    end

    θ_label = string(round(theta, digits = 2))
    mkpath(out)
    csv_path = joinpath(out,
                        "S_heatmap_L=$(L)_theta=$(θ_label)_ratio_vs_p.csv")
    heatmap_df = DataFrame()
    heatmap_df[!, "ζ/L"] = ratios
    for (j, pv) in enumerate(pvals)
        heatmap_df[!, string(pv)] = mat[:, j]
    end
    CSV.write(csv_path, heatmap_df)

    summary_dir = joinpath(plot_dir, "summary")
    mkpath(summary_dir)
    plt = heatmap(pvals, ratios, mat,
                  xlabel = "p", ylabel = "ζ/L",
                  color = :viridis, clims = (0.0, 1.0),
                  colorbar_title = "S", size = (720, 460),
                  title = "S(p), L=$(L), θ=$(θ_label), 10-realization mean")
    png_path = joinpath(summary_dir,
                        "S_heatmap_L=$(L)_theta=$(θ_label)_ratio_vs_p.png")
    savefig(plt, png_path)

    println("Wrote $csv_path")
    println("Wrote $png_path")
end

function run_all_S_ratio_heatmaps(; out, plot_dir, L = 100,
                                  topology = "waxman")
    s_path = joinpath(out, "S_mean.csv")
    isfile(s_path) || error("missing $s_path; run aggregate first")

    S_long = CSV.read(s_path, DataFrame)
    sub = filter(row -> row.L == L && row.topology == topology, S_long)
    isempty(sub) && error("no S(p) rows for L=$L, topology=$topology")

    for theta in sort(unique(sub.theta))
        run_S_ratio_heatmap(; out = out, plot_dir = plot_dir,
                            theta = theta, L = L, topology = topology)
    end
end

function plot_S_theta_regime(sub, pvals, L, ζ_label, regime, out, plot_dir)
    rsub = filter(row -> regime == "negative" ? row.theta < 0 : row.theta > 0,
                  sub)
    isempty(rsub) && return

    θs = sort(unique(rsub.theta))
    mat = fill(NaN, length(θs), length(pvals))
    for row in eachrow(rsub)
        i = findfirst(x -> isapprox(x, row.theta; atol = 1e-9), θs)
        j = findfirst(x -> isapprox(x, row.p; atol = 1e-12), pvals)
        (i === nothing || j === nothing) && continue
        mat[i, j] = row.S_mean
    end

    base = "S_heatmap_L=$(L)_zeta=$(ζ_label)_$(regime)_theta_vs_p"
    heatmap_df = DataFrame()
    heatmap_df[!, "theta"] = θs
    for (j, pv) in enumerate(pvals)
        heatmap_df[!, string(pv)] = mat[:, j]
    end
    CSV.write(joinpath(out, base * ".csv"), heatmap_df)

    regime_name = regime == "negative" ? "B (θ < 0)" : "A (θ > 0)"
    plt = heatmap(pvals, θs, mat,
                  xlabel = "p", ylabel = "θ",
                  color = :viridis, clims = (0.0, 1.0),
                  colorbar_title = "S", size = (760, 460),
                  title = "S(p, θ) mean, L=$(L), ζ=$(ζ_label), " *
                          "regime $(regime_name)")
    png_path = joinpath(plot_dir, "summary", base * ".png")
    savefig(plt, png_path)
    println("Wrote $(joinpath(out, base * ".csv"))")
    println("Wrote $png_path")
end

function run_S_theta_heatmap(; out, plot_dir, zeta, L = 100,
                             topology = "waxman")
    s_path = joinpath(out, "S_mean.csv")
    isfile(s_path) || error("missing $s_path; run aggregate first")

    S_long = CSV.read(s_path, DataFrame)
    sub = filter(row -> row.L == L &&
                         row.topology == topology &&
                         isapprox(row.zeta, zeta; atol = 1e-9), S_long)
    isempty(sub) && error("no S(p) rows for L=$L, ζ=$zeta, topology=$topology")

    ζ_label = string(round(zeta, digits = 2))
    pvals = sort(unique(sub.p))
    mkpath(out)
    summary_dir = joinpath(plot_dir, "summary")
    mkpath(summary_dir)

    for regime in ("negative", "positive")
        plot_S_theta_regime(sub, pvals, L, ζ_label, regime, out, plot_dir)
    end
end

function group_dirname(plot_dir, L, ζ, topology)
    suffix = topology == "ER" ? ", topology = ER" : ""
    return joinpath(plot_dir, "L = $L", "ζ = $ζ" * suffix)
end

function group_title(L, ζ, topology)
    return topology == "ER" ? "L=$(L), ER" : "L=$(L), ζ=$(ζ)"
end

function plot_s_theta(s, pc_row, L, ζ, θ, topology, outpath)
    sort!(s, :p)
    plt = plot(s.p, s.S_mean, ribbon = max.(s.S_sem, 0.0),
               fillalpha = 0.25, linewidth = 1.5, color = :dodgerblue,
               label = "mean ± SEM")
    if pc_row !== nothing && isfinite(pc_row.p_c)
        vline!(plt, [pc_row.p_c], linestyle = :dash, color = :firebrick,
               label = "p_c=$(round(pc_row.p_c, digits = 3))")
    end
    xlabel!(plt, "p")
    ylabel!(plt, "S")
    n = pc_row === nothing ? 10 : pc_row.n_realizations
    title!(plt, "S(p), $(group_title(L, ζ, topology)), θ=$(round(θ, digits = 2)), n=$(n)")
    savefig(plt, outpath)
end

function plot_s_comparison(sub, L, ζ, topology, outpath)
    θs = sort(unique(sub.theta))
    colors = Plots.palette(:viridis, length(θs))
    plt = plot()
    for (i, θ) in enumerate(θs)
        s = filter(:theta => x -> isapprox(x, θ; atol = 1e-9), sub)
        sort!(s, :p)
        plot!(plt, s.p, s.S_mean, ribbon = max.(s.S_sem, 0.0),
              fillalpha = 0.08, linewidth = 1.2, color = colors[i],
              label = "θ=$(round(θ, digits = 2))")
    end
    xlabel!(plt, "p")
    ylabel!(plt, "S")
    title!(plt, "S(p) mean ± SEM, $(group_title(L, ζ, topology))")
    savefig(plt, outpath)
end

function plot_s_heatmap(sub, L, ζ, topology, outpath)
    θs = sort(unique(sub.theta))
    pvals = sort(unique(sub.p))
    mat = fill(NaN, length(θs), length(pvals))
    for row in eachrow(sub)
        i = findfirst(x -> isapprox(x, row.theta; atol = 1e-9), θs)
        j = findfirst(x -> isapprox(x, row.p; atol = 1e-12), pvals)
        (i === nothing || j === nothing) && continue
        mat[i, j] = row.S_mean
    end
    plt = heatmap(pvals, θs, mat, xlabel = "p", ylabel = "θ",
                  color = :viridis,
                  title = "S(p) mean, $(group_title(L, ζ, topology))")
    savefig(plt, outpath)
end

function plot_pc_vs_theta(pc_sub, L, ζ, topology, outpath)
    sort!(pc_sub, :theta)
    plt = plot(pc_sub.theta, pc_sub.p_c, yerror = pc_sub.p_c_se,
               marker = :circle, markersize = 3, linewidth = 1.2,
               color = :firebrick, label = "p_c ± SE")
    xlabel!(plt, "θ")
    ylabel!(plt, "p_c")
    title!(plt, "p_c(θ), $(group_title(L, ζ, topology))")
    savefig(plt, outpath)
end

function plot_cascade_comparison(group_dir, L, ζ, topology, outpath)
    isdir(group_dir) || return
    mean_path = find_file(group_dir, "Cascade Time mean")
    mean_path === nothing && return
    θs, p, T = read_s_matrix(mean_path)
    sem_path = find_file(group_dir, "Cascade Time sem")
    T_sem = sem_path === nothing ? zeros(size(T)) :
            read_s_matrix(sem_path)[3]

    colors = Plots.palette(:viridis, length(θs))
    plt = plot()
    for (i, θ) in enumerate(θs)
        plot!(plt, p, T[i, :], ribbon = max.(T_sem[i, :], 0.0),
              fillalpha = 0.08, linewidth = 1.2, color = colors[i],
              label = "θ=$(round(θ, digits = 2))")
    end
    xlabel!(plt, "p")
    ylabel!(plt, "Cascade Time")
    title!(plt, "Cascade time mean ± SEM, $(group_title(L, ζ, topology))")
    savefig(plt, outpath)
end

function plot_heatmap_matrix(df, L, topology, value, label, base, out_dir)
    sub = filter(row -> row.L == L && row.topology == topology, df)
    isempty(sub) && return
    ζs = sort(unique(sub.zeta))
    θs = sort(unique(sub.theta))
    mat = fill(NaN, length(ζs), length(θs))
    for row in eachrow(sub)
        i = findfirst(x -> isapprox(x, row.zeta; atol = 1e-9), ζs)
        j = findfirst(x -> isapprox(x, row.theta; atol = 1e-9), θs)
        (i === nothing || j === nothing) && continue
        mat[i, j] = getproperty(row, value)
    end
    plt = heatmap(ζs, θs, permutedims(mat), xlabel = "ζ", ylabel = "θ",
                  color = :viridis,
                  title = "$(label), L=$L")
    suffix = topology == "ER" ? "_ER" : ""
    savefig(plt, joinpath(out_dir, "$(base)$(suffix).png"))
end

function run_aggregate_mean_plots(; out, plot_dir, ratios = Float64[],
                                  thetas = Float64[], L_values = Int[])
    s_path = joinpath(out, "S_mean.csv")
    pc_path = joinpath(out, "p_c_summary.csv")
    isfile(s_path) || error("missing $s_path; run aggregate first")

    S_long = CSV.read(s_path, DataFrame)
    pc = isfile(pc_path) ? CSV.read(pc_path, DataFrame) : DataFrame()

    if !isempty(L_values)
        S_long = filter(:L => x -> x in L_values, S_long)
        isempty(pc) || (pc = filter(:L => x -> x in L_values, pc))
    end
    if !isempty(thetas)
        S_long = filter(:theta => x -> any(t -> isapprox(x, t; atol = 1e-9), thetas),
                        S_long)
        isempty(pc) || (pc = filter(:theta => x -> any(t -> isapprox(x, t; atol = 1e-9), thetas),
                                    pc))
    end
    if !isempty(ratios)
        S_long = filter(row -> any(r -> isapprox(row.zeta / row.L, r; atol = 1e-9), ratios),
                        S_long)
        isempty(pc) || (pc = filter(row -> any(r -> isapprox(row.zeta / row.L, r; atol = 1e-9), ratios),
                                    pc))
    end
    isempty(S_long) && error("no aggregate S(p) rows selected")

    means_dir = joinpath(plot_dir, "means")
    keys_present = unique([(row.L, row.zeta, row.topology) for row in eachrow(S_long)])
    sort!(keys_present; by = k -> (k[1], k[3], k[2]))

    for (L, ζ, topology) in keys_present
        sub = filter(row -> row.L == L &&
                             isapprox(row.zeta, ζ; atol = 1e-9) &&
                             row.topology == topology, S_long)
        pc_sub = isempty(pc) ? DataFrame() :
                 filter(row -> row.L == L &&
                               isapprox(row.zeta, ζ; atol = 1e-9) &&
                               row.topology == topology, pc)
        dir = group_dirname(means_dir, L, ζ, topology)
        mkpath(dir)

        plot_s_comparison(sub, L, ζ, topology,
                          joinpath(dir, "S_mean_comparison_theta.png"))
        plot_s_heatmap(sub, L, ζ, topology,
                       joinpath(dir, "S_mean_heatmap_theta.png"))
        if !isempty(pc_sub)
            plot_pc_vs_theta(pc_sub, L, ζ, topology,
                             joinpath(dir, "p_c_vs_theta.png"))
        end
        analysis_dir = group_dirname(out, L, ζ, topology)
        plot_cascade_comparison(analysis_dir, L, ζ, topology,
                                joinpath(dir, "Cascade_Time_mean_comparison_theta.png"))

        for θ in sort(unique(sub.theta))
            s = filter(:theta => x -> isapprox(x, θ; atol = 1e-9), sub)
            pc_row = isempty(pc_sub) ? nothing :
                     findfirst(row -> isapprox(row.theta, θ; atol = 1e-9),
                               eachrow(pc_sub))
            pc_row = pc_row === nothing ? nothing : pc_sub[pc_row, :]
            fname = "S_mean_theta=$(round(θ, digits = 2)).png"
            plot_s_theta(s, pc_row, L, ζ, θ, topology, joinpath(dir, fname))
        end
    end

    if !isempty(pc)
        for topology in sort(unique(pc.topology))
            for L in sort(unique(pc.L))
                L_dir = joinpath(means_dir, "L = $L")
                mkpath(L_dir)
                plot_heatmap_matrix(pc, L, topology, :p_c, "p_c", "p_c_heatmap", L_dir)
                plot_heatmap_matrix(pc, L, topology, :delta_S, "ΔS",
                                    "delta_S_heatmap", L_dir)
            end
        end
    end

    println("  Mean plots written to $means_dir")
end

function run_summary_plots(out, plot_dir, ratios, thetas)
    summary_dir = joinpath(plot_dir, "summary")
    mkpath(summary_dir)

    beta_path = joinpath(out, "beta_vs_L.csv")
    delta_path = joinpath(out, "delta_S_vs_L.csv")
    s_path = joinpath(out, "S_mean.csv")
    for path in (beta_path, delta_path, s_path)
        isfile(path) || error("missing $path; run aggregate and fss first")
    end

    beta = CSV.read(beta_path, DataFrame)
    delta = CSV.read(delta_path, DataFrame)
    S_long = CSV.read(s_path, DataFrame)

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
    beta_var = copy(filter(:topology => ==("waxman"), beta))
    beta_var[!, :beta_variance] = beta_var.beta_se .^ 2
    CSV.write(joinpath(out, "beta_variance_vs_L.csv"), beta_var)
    plot_variance_scaling(beta, :beta_mean, :beta_se, "β",
                          "beta_variance_vs_L", summary_dir)

    delta_var = copy(filter(:topology => ==("waxman"), delta))
    delta_var[!, :delta_S_variance] = delta_var.delta_S_se .^ 2
    CSV.write(joinpath(out, "delta_S_variance_vs_L.csv"), delta_var)
    plot_variance_scaling(delta, :delta_S_mean, :delta_S_se, "ΔS",
                          "delta_S_variance_vs_L", summary_dir)

    plot_s_mean_ribbons(S_long, summary_dir, ratios, thetas)
    println("  Summary plots written to $summary_dir")
end

function run_plots(; root, out, plot_dir, mode = :summary,
                   ratios = Float64[], thetas = Float64[],
                   L_values = Int[])
    mode = Symbol(mode)
    mode in (:summary, :means, :first, :all, :none) ||
        error("unknown plot mode: $mode")

    if mode in (:summary, :means, :all)
        run_aggregate_mean_plots(; out = out, plot_dir = plot_dir,
                                 ratios = ratios, thetas = thetas,
                                 L_values = L_values)
    end

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
