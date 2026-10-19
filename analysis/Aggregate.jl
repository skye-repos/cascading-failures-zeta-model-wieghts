module Aggregate

using CSV
using DataFrames
using Random
using Statistics

using ..PCUtils

export run_aggregate

function bootstrap_pc(p, S_realizations, sigma_pts, B, rng)
    R = size(S_realizations, 1)
    pcs = Vector{Float64}(undef, B)
    for b in 1:B
        idx = rand(rng, 1:R, R)
        S_b = vec(mean(S_realizations[idx, :], dims = 1))
        pcs[b] = find_critical(p, S_b; sigma_pts = sigma_pts)
    end
    return mean(pcs), (R > 1 ? std(pcs; corrected = true) : 0.0)
end

function write_matrix_csv(path, θ, p, M)
    df = DataFrame()
    df[!, "θ"] = θ
    for (j, pv) in enumerate(p)
        df[!, string(pv)] = M[:, j]
    end
    CSV.write(path, df)
end

function manifest_stats(path)
    text = read(path, String)
    getfloat(key) = begin
        m = match(Regex("\"$key\"\\s*:\\s*([-+0-9.eE]+)"), text)
        m === nothing ? NaN : parse(Float64, m.captures[1])
    end
    getint(key) = begin
        m = match(Regex("\"$key\"\\s*:\\s*([-+0-9]+)"), text)
        m === nothing ? -1 : parse(Int, m.captures[1])
    end
    topos = match(r"\"topology\"\s*:\s*\"([^\"]+)\"", text)
    topology = topos === nothing ? "waxman" : topos.captures[1]
    return (
        L = getint("L"),
        zeta = getfloat("zeta"),
        topology = topology,
        realization_id = getint("realization_id"),
        n_vertices = getint("n_vertices"),
        n_edges = getint("n_edges"),
        k_mean = getfloat("k_mean"),
        k_variance = getfloat("k_variance"),
        k_second_moment = getfloat("k_second_moment"),
    )
end

function run_aggregate(; root, out, sigma_pts = 2.0, bootstrap = 1000,
                       seed = 20261017, L_values = Int[])
    runs = discover_realizations(root)
    if !isempty(L_values)
        runs = [run for run in runs if run.L in L_values]
        found = sort(unique(run.L for run in runs))
        missing = setdiff(L_values, found)
        if !isempty(missing)
            println("  warning: no results for L = $(join(missing, ", "))")
        end
    end
    isempty(runs) && error("no realization directories found under $root " *
                           "for L values $(L_values)")

    mkpath(out)

    groups = Dict{Tuple{Int,Float64,String},Vector{NamedTuple}}()
    for run in runs
        key = (run.L, run.zeta, run.topology)
        push!(get!(groups, key, NamedTuple[]), run)
    end

    summary_rows = NamedTuple[]
    s_long = NamedTuple[]
    degree_rows = NamedTuple[]

    rng = MersenneTwister(seed)

    for key in sort(collect(keys(groups)); by = k -> (k[1], k[3], k[2]))
        L, ζ, topology = key
        group_runs = sort(groups[key]; by = r -> r.path)

        θ_ref = nothing
        p_ref = nothing
        S_all = nothing

        for (r, run) in enumerate(group_runs)
            s_path = find_file(run.path, "Gt Component")
            s_path === nothing && continue
            θ, p, S = read_s_matrix(s_path)

            if θ_ref === nothing
                θ_ref = θ
                p_ref = p
                S_all = Array{Float64}(undef, length(group_runs), length(θ), length(p))
            end

            length(θ) == length(θ_ref) ||
                error("θ grid mismatch in $(run.path)")
            all(isapprox.(θ, θ_ref; atol = 1e-9)) ||
                error("θ values mismatch in $(run.path)")

            if length(p) == length(p_ref) && all(isapprox.(p, p_ref; atol = 1e-12))
                S_all[r, :, :] .= S
            else
                for (i, row) in enumerate(eachrow(S))
                    S_all[r, i, :] .= interpolate_series(p_ref, p, row)
                end
            end

            manifest_path = joinpath(run.path, "run_manifest.json")
            if isfile(manifest_path)
                st = manifest_stats(manifest_path)
                push!(degree_rows, (
                    L = st.L,
                    zeta = st.zeta,
                    topology = st.topology,
                    realization_id = st.realization_id,
                    n_vertices = st.n_vertices,
                    n_edges = st.n_edges,
                    k_mean = st.k_mean,
                    k_variance = st.k_variance,
                    k_second_moment = st.k_second_moment,
                ))
            end
        end

        S_all === nothing && continue

        θ = θ_ref
        p = p_ref
        R = size(S_all, 1)
        np = length(p)
        nθ = length(θ)

        group_dir = joinpath(out, "L = $L",
                             "ζ = $ζ" * (topology == "ER" ? ", topology = ER" : ""))
        mkpath(group_dir)

        S_mean = Array{Float64}(undef, nθ, np)
        S_sem = Array{Float64}(undef, nθ, np)

        for i in 1:nθ
            S_reals = S_all[:, i, :]
            S_mean[i, :] .= vec(mean(S_reals; dims = 1))
            S_sem[i, :] .= R > 1 ? vec(std(S_reals; dims = 1, corrected = true)) ./ sqrt(R) :
                                  zeros(np)

            pc_real = [find_critical(p, S_reals[r, :]; sigma_pts = sigma_pts)
                       for r in 1:R]
            pc_central = find_critical(p, S_mean[i, :]; sigma_pts = sigma_pts)
            pc_se = R > 1 ? std(pc_real; corrected = true) / sqrt(R) : 0.0
            pc_boot_mean, pc_boot_se = bootstrap_pc(
                p, S_reals, sigma_pts, bootstrap, rng)

            k = argmin(abs.(p .- pc_central))
            klo = max(k - 1, 1)
            khi = min(k + 1, np)
            ΔS_real = [S_reals[r, khi] - S_reals[r, klo] for r in 1:R]
            ΔS = S_mean[i, khi] - S_mean[i, klo]
            ΔS_se = R > 1 ? std(ΔS_real; corrected = true) / sqrt(R) : 0.0

            push!(summary_rows, (
                L = L,
                zeta = ζ,
                topology = topology,
                theta = θ[i],
                p_c = pc_central,
                p_index = k,
                p_c_se = pc_se,
                p_c_bootstrap_mean = pc_boot_mean,
                p_c_bootstrap_se = pc_boot_se,
                delta_S = ΔS,
                delta_S_se = ΔS_se,
                n_realizations = R,
            ))

            for j in 1:np
                push!(s_long, (
                    L = L,
                    zeta = ζ,
                    topology = topology,
                    theta = θ[i],
                    p = p[j],
                    S_mean = S_mean[i, j],
                    S_sem = S_sem[i, j],
                ))
            end
        end

        write_matrix_csv(joinpath(group_dir, " Gt Component mean.csv"),
                         θ, p, S_mean)
        write_matrix_csv(joinpath(group_dir, " Gt Component sem.csv"),
                         θ, p, S_sem)

        t_path = find_file(first(group_runs).path, "Cascade Time")
        if t_path !== nothing
            t_all = Array{Float64}(undef, R, nθ, np)
            fill!(t_all, NaN)
            for (r, run) in enumerate(group_runs)
                path = find_file(run.path, "Cascade Time")
                path === nothing && continue
                θt, pt, T = read_s_matrix(path)
                if length(pt) == length(p) && all(isapprox.(pt, p; atol = 1e-12))
                    t_all[r, :, :] .= T
                else
                    for (i, row) in enumerate(eachrow(T))
                        t_all[r, i, :] .= interpolate_series(p, pt, row)
                    end
                end
            end
            T_mean = Array{Float64}(undef, nθ, np)
            T_sem = Array{Float64}(undef, nθ, np)
            for i in 1:nθ
                T_reals = t_all[:, i, :]
                T_mean[i, :] .= vec(mean(T_reals; dims = 1))
                T_sem[i, :] .= R > 1 ?
                    vec(std(T_reals; dims = 1, corrected = true)) ./ sqrt(R) :
                    zeros(np)
            end
            write_matrix_csv(joinpath(group_dir, " Cascade Time mean.csv"),
                             θ, p, T_mean)
            write_matrix_csv(joinpath(group_dir, " Cascade Time sem.csv"),
                             θ, p, T_sem)
        end
    end

    summary_df = DataFrame(summary_rows)
    CSV.write(joinpath(out, "p_c_summary.csv"), summary_df)
    CSV.write(joinpath(out, "S_mean.csv"), DataFrame(s_long))

    if !isempty(degree_rows)
        CSV.write(joinpath(out, "degree_summary.csv"), DataFrame(degree_rows))
    end

    for topology in unique(summary_df.topology)
        df_topo = filter(:topology => ==(topology), summary_df)
        for L in sort(unique(df_topo.L))
            df_L = filter(:L => ==(L), df_topo)
            θs = sort(unique(df_L.theta))
            ζs = sort(unique(df_L.zeta))

            pc_mat = fill(NaN, length(ζs), length(θs))
            idx_mat = fill(-1, length(ζs), length(θs))

            for row in eachrow(df_L)
                iz = findfirst(==(row.zeta), ζs)
                it = findfirst(==(row.theta), θs)
                pc_mat[iz, it] = row.p_c
                idx_mat[iz, it] = row.p_index
            end

            pc_df = DataFrame()
            pc_df[!, "ζ"] = ζs
            idx_df = DataFrame()
            idx_df[!, "ζ"] = ζs
            for (it, th) in enumerate(θs)
                pc_df[!, string(th)] = pc_mat[:, it]
                idx_df[!, string(th)] = idx_mat[:, it]
            end

            suffix = topology == "ER" ? ", ER" : ""
            out_dir = joinpath(out, "L = $L")
            mkpath(out_dir)
            CSV.write(joinpath(out_dir, "p_c(θ, ζ)$suffix.csv"), pc_df)
            CSV.write(joinpath(out_dir, "indices$suffix.csv"), idx_df)
        end
    end

    println("Wrote aggregate results to $(out)")
end

end
