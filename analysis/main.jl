include("PCUtils.jl")
include("Aggregate.jl")
include("FSS.jl")
include("PlotResults.jl")
include("test_pc.jl")

using .PCUtils
using .Aggregate
using .FSS
using .PlotResults
using .PCTest

const ANALYSIS_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(ANALYSIS_DIR, ".."))

function parse_float_list(text)
    s = lowercase(strip(text))
    (isempty(s) || s == "all") && return Float64[]
    return parse.(Float64, split(text, ","))
end

function usage()
    println("""
Usage:
  julia --project=analysis analysis/main.jl <command> [options]

Commands:
  all         aggregate + fss + plots (default plot mode: summary)
  aggregate   S(p) mean/SEM, p_c, ΔS, degree summaries
  fss         β(L), ΔS(L), saturation report
  plots       summary and/or per-realization plots
  test        run p_c unit tests

Options:
  --root DIR          raw CSV root (default <repo>/results/CSV)
  --out DIR           analysis output (default <repo>/results/analysis)
  --plot-dir DIR      plot output (default <repo>/results/plots)
  --sigma-pts N       p_c smoothing width in grid points (default 2)
  --bootstrap N       bootstrap samples for p_c (default 1000)
  --seed N            bootstrap RNG seed (default 20261017)
  --window Δp         FSS fit window above p_c (default 0.1)
  --min-points N      minimum points per β fit (default 3)
  --ratios LIST|all   FSS ratios (default 0.1,0.25,0.5)
  --thetas LIST|all   FSS θ values (default -0.4,-0.2,0.2,0.4)
  --plots MODE        summary, first, all, or none (default summary)
  -h, --help          show this message

Examples:
  julia --project=analysis analysis/main.jl all
  julia --project=analysis analysis/main.jl aggregate
  julia --project=analysis analysis/main.jl fss
  julia --project=analysis analysis/main.jl plots --plots all
  julia --project=analysis analysis/main.jl test
""")
end

function parse_args(args)
    command = "all"
    root = joinpath(REPO_ROOT, "results", "CSV")
    out = joinpath(REPO_ROOT, "results", "analysis")
    plot_dir = joinpath(REPO_ROOT, "results", "plots")
    sigma_pts = 2.0
    bootstrap = 1000
    seed = 20261017
    window = 0.1
    min_points = 3
    ratios = [0.1, 0.25, 0.5]
    thetas = [-0.4, -0.2, 0.2, 0.4]
    plot_mode = "summary"

    i = 1
    if !isempty(args) && !startswith(args[1], "-")
        command = args[1]
        i = 2
    end

    while i <= length(args)
        arg = args[i]
        if arg == "--root" && i < length(args)
            root = args[i + 1]
            i += 2
        elseif arg == "--out" && i < length(args)
            out = args[i + 1]
            i += 2
        elseif arg == "--plot-dir" && i < length(args)
            plot_dir = args[i + 1]
            i += 2
        elseif arg == "--sigma-pts" && i < length(args)
            sigma_pts = parse(Float64, args[i + 1])
            i += 2
        elseif arg == "--bootstrap" && i < length(args)
            bootstrap = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--seed" && i < length(args)
            seed = parse(Int, args[i + 1])
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
        elseif arg == "--plots" && i < length(args)
            plot_mode = lowercase(args[i + 1])
            i += 2
        elseif arg == "-h" || arg == "--help"
            usage()
            exit(0)
        else
            error("unknown argument: $arg")
        end
    end

    command in ("all", "aggregate", "fss", "plots", "test") ||
        error("unknown command: $command")
    plot_mode in ("summary", "first", "all", "none") ||
        error("unknown plot mode: $plot_mode")

    return (; command, root, out, plot_dir, sigma_pts, bootstrap, seed,
            window, min_points, ratios, thetas, plot_mode)
end

function ensure_aggregate(opts)
    pc = joinpath(opts.out, "p_c_summary.csv")
    s = joinpath(opts.out, "S_mean.csv")
    if !isfile(pc) || !isfile(s)
        println("[aggregate] outputs missing; running aggregate")
        run_aggregate(; root = opts.root, out = opts.out,
                      sigma_pts = opts.sigma_pts,
                      bootstrap = opts.bootstrap, seed = opts.seed)
    end
end

function ensure_fss(opts)
    ensure_aggregate(opts)
    beta = joinpath(opts.out, "beta_vs_L.csv")
    delta = joinpath(opts.out, "delta_S_vs_L.csv")
    if !isfile(beta) || !isfile(delta)
        println("[fss] outputs missing; running fss")
        run_fss(; root = opts.out, out = opts.out, window = opts.window,
                min_points = opts.min_points,
                ratios = opts.ratios, thetas = opts.thetas)
    end
end

function main()
    opts = try
        parse_args(ARGS)
    catch e
        println(stderr, e)
        usage()
        exit(1)
    end

    if opts.command == "test"
        PCTest.run_tests()
        return
    end

    if opts.command == "aggregate"
        run_aggregate(; root = opts.root, out = opts.out,
                      sigma_pts = opts.sigma_pts,
                      bootstrap = opts.bootstrap, seed = opts.seed)
    elseif opts.command == "fss"
        ensure_aggregate(opts)
        run_fss(; root = opts.out, out = opts.out, window = opts.window,
                min_points = opts.min_points,
                ratios = opts.ratios, thetas = opts.thetas)
    elseif opts.command == "plots"
        if opts.plot_mode != "none"
            ensure_fss(opts)
        end
        run_plots(; root = opts.root, out = opts.out,
                  plot_dir = opts.plot_dir, mode = Symbol(opts.plot_mode),
                  ratios = opts.ratios, thetas = opts.thetas)
    elseif opts.command == "all"
        run_aggregate(; root = opts.root, out = opts.out,
                      sigma_pts = opts.sigma_pts,
                      bootstrap = opts.bootstrap, seed = opts.seed)
        run_fss(; root = opts.out, out = opts.out, window = opts.window,
                min_points = opts.min_points,
                ratios = opts.ratios, thetas = opts.thetas)
        if opts.plot_mode != "none"
            run_plots(; root = opts.root, out = opts.out,
                      plot_dir = opts.plot_dir, mode = Symbol(opts.plot_mode),
                      ratios = opts.ratios, thetas = opts.thetas)
        end
    end
end

main()
