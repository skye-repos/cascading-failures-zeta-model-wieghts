module PlotResults

using CSV
using DataFrames

ENV["GKSwstype"] = "100"
using Plots

using ..PCUtils

export run_plots

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
    if isempty(ratios) || isempty(thetas)
        println("  Skipping S(p) ribbons: ratios/thetas filters disabled")
        return
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
                   ratios = Float64[], thetas = Float64[])
    mode = Symbol(mode)
    mode in (:summary, :first, :all, :none) ||
        error("unknown plot mode: $mode")

    if mode in (:summary, :first, :all)
        run_summary_plots(out, plot_dir, ratios, thetas)
    end

    if mode in (:first, :all)
        runs = sort(discover_realizations(root);
                    by = r -> (r.L, r.topology, r.zeta, r.path))
        isempty(runs) && error("no realization directories found under $root")
        paths = mode == :first ? select_first_paths(runs) : [r.path for r in runs]
        for path in paths
            dest = joinpath(plot_dir, relpath(path, root))
            process_one_dir(path, dest)
        end
    end

    mode == :none || println("Plots written under $plot_dir")
end

end
