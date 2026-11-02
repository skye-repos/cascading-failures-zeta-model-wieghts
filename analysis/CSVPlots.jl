module CSVPlots

using CSV
using DataFrames
using Statistics

export run_csv_plots

function sem_or_zero(values)
    n = length(values)
    n > 1 ? std(values; corrected = true) / sqrt(n) : 0.0
end

function write_wide_S(source, dest)
    df = CSV.read(source, DataFrame)
    meta = [:L, :zeta, :topology, :theta]
    sort!(df, [:L, :topology, :zeta, :theta, :p])

    keys = unique(select(df, meta))
    sort!(keys, [:L, :topology, :zeta, :theta])
    pvals = sort(unique(df.p))

    key_index = Dict{Tuple{Int,Float64,String,Float64},Int}()
    for (i, row) in enumerate(eachrow(keys))
        key_index[(row.L, row.zeta, row.topology, row.theta)] = i
    end
    p_index = Dict(pv => j for (j, pv) in enumerate(pvals))

    S_mean = fill(NaN, nrow(keys), length(pvals))
    S_sem = fill(NaN, nrow(keys), length(pvals))
    for row in eachrow(df)
        i = get(key_index, (row.L, row.zeta, row.topology, row.theta), 0)
        j = get(p_index, row.p, 0)
        (i == 0 || j == 0) && continue
        S_mean[i, j] = row.S_mean
        S_sem[i, j] = row.S_sem
    end

    out_mean = copy(keys)
    out_sem = copy(keys)
    for (j, pv) in enumerate(pvals)
        out_mean[!, string(pv)] = S_mean[:, j]
        out_sem[!, string(pv)] = S_sem[:, j]
    end

    CSV.write(joinpath(dest, "S_mean.csv"), out_mean)
    CSV.write(joinpath(dest, "S_sem.csv"), out_sem)
end

function aggregate_metric(source, value_col, mean_name, se_name;
                          ratio = nothing)
    df = CSV.read(source, DataFrame)
    if ratio !== nothing
        df = df[abs.(df.zeta ./ df.L .- ratio) .< 1e-9, :]
    end
    df[!, :regime] = ifelse.(df.theta .< 0, "negative", "positive")

    keys = unique(select(df, [:L, :topology, :regime]))
    sort!(keys, [:topology, :regime, :L])

    rows = NamedTuple[]
    for key in eachrow(keys)
        mask = (df.L .== key.L) .&
               (df.topology .== key.topology) .&
               (df.regime .== key.regime)
        values = df[mask, value_col]
        push!(rows, (
            L = key.L,
            topology = key.topology,
            regime = key.regime,
            n = length(values),
            value_mean = mean(values),
            value_se = sem_or_zero(values),
        ))
    end

    result = DataFrame(rows)
    rename!(result, :value_mean => mean_name, :value_se => se_name)
    return result
end

function run_csv_plots(; analysis_dir, dest)
    mkpath(dest)

    s_path = joinpath(analysis_dir, "S_mean.csv")
    pc_path = joinpath(analysis_dir, "p_c_summary.csv")
    beta_path = joinpath(analysis_dir, "beta_fits.csv")
    isfile(s_path) || error("missing $s_path; run aggregate first")
    isfile(pc_path) || error("missing $pc_path; run aggregate first")
    isfile(beta_path) || error("missing $beta_path; run fss first")

    write_wide_S(s_path, dest)

    for (label, ratio) in (("ratio=0.05", 0.05), ("ratio=0.5", 0.5))
        delta = aggregate_metric(pc_path, :delta_S, :delta_S_mean,
                                 :delta_S_se; ratio = ratio)
        CSV.write(joinpath(dest, "deltaS_vs_L_$(label).csv"), delta)

        beta = aggregate_metric(beta_path, :beta, :beta_mean, :beta_se;
                                ratio = ratio)
        CSV.write(joinpath(dest, "beta_vs_L_$(label).csv"), beta)
    end

    delta = aggregate_metric(pc_path, :delta_S, :delta_S_mean, :delta_S_se)
    CSV.write(joinpath(dest, "deltaS_vs_L_all_zeta.csv"), delta)

    beta = aggregate_metric(beta_path, :beta, :beta_mean, :beta_se)
    CSV.write(joinpath(dest, "beta_vs_L_all_zeta.csv"), beta)

    println("Wrote CSV plotting tables to $dest")
end

end
