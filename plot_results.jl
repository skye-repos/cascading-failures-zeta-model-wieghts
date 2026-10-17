using CSV
using DataFrames

ENV["GKSwstype"] = "100"
using Plots

include("pc_utils.jl")
using .PCUtils

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

function process_one_dir(csv_dir)
    plots_dir = occursin("/CSV/", csv_dir) ?
        replace(csv_dir, "/CSV/" => "/plots/") :
        joinpath(csv_dir, "plots")
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

function main()
    if isempty(ARGS)
        root = joinpath("results", "CSV")
        runs = discover_realizations(root)
        if isempty(runs)
            println("No realization results found under $root")
            return
        end
        for run in runs
            process_one_dir(run.path)
        end
    else
        target = ARGS[1]
        isdir(target) || error("directory not found: $target")
        runs = discover_realizations(target)
        if !isempty(runs)
            for run in runs
                process_one_dir(run.path)
            end
        else
            process_one_dir(target)
        end
    end
end

main()
