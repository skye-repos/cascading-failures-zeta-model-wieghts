module PCUtils

export smooth_along_p, find_critical, find_critical_full
export read_s_matrix, find_file, discover_realizations, interpolate_series

using CSV
using DataFrames
using Statistics

function smooth_along_p(p::AbstractVector{<:Real}, S::AbstractVector{<:Real};
                        sigma_pts::Real = 2.0)
    length(p) == length(S) ||
        throw(DimensionMismatch("p and S must have the same length"))
    n = length(p)
    n == 0 && return Float64[]
    n == 1 && return Float64.(S)

    Δp = (p[end] - p[1]) / (n - 1)
    σ = abs(sigma_pts * Δp)
    S̃ = Vector{Float64}(undef, n)
    if σ == 0
        @. S̃ = S
        return S̃
    end

    for i in 1:n
        num = 0.0
        den = 0.0
        for j in 1:n
            z = (p[i] - p[j]) / σ
            w = exp(-0.5 * z * z)
            num += w * S[j]
            den += w
        end
        S̃[i] = num / den
    end
    return S̃
end

function find_critical_full(p::AbstractVector{<:Real}, S::AbstractVector{<:Real};
                            sigma_pts::Real = 2.0)
    n = length(p)
    n >= 4 || throw(ArgumentError("find_critical needs at least 4 p values"))
    length(S) == n ||
        throw(DimensionMismatch("p and S must have the same length"))

    S̃ = smooth_along_p(p, S; sigma_pts = sigma_pts)
    Δp = (p[end] - p[1]) / (n - 1)

    d2 = [(S̃[i + 2] + S̃[i] - 2.0 * S̃[i + 1]) / (Δp^2) for i in 1:(n - 2)]

    best_jump = -Inf
    best_idx = 0
    for k in 1:(length(d2) - 1)
        if d2[k] * d2[k + 1] < 0
            jump = abs(d2[k + 1] - d2[k])
            if jump > best_jump
                best_jump = jump
                best_idx = k + 1
            end
        end
    end

    found = best_idx != 0
    if !found
        best_idx = argmax(abs.(d2))
        best_jump = abs(d2[best_idx])
    end

    pc = p[best_idx + 1]
    return (pc = pc, jump = best_jump, sign_change = found)
end

find_critical(p::AbstractVector{<:Real}, S::AbstractVector{<:Real};
              sigma_pts::Real = 2.0) =
    find_critical_full(p, S; sigma_pts = sigma_pts).pc

function read_s_matrix(path::AbstractString)
    df = CSV.read(path, DataFrame; header = 1)
    p = parse.(Float64, names(df)[2:end])
    θ = Float64.(df[!, 1])
    S = Matrix{Float64}(df[:, 2:end])
    return θ, p, S
end

function find_file(dir::AbstractString, needle::AbstractString)
    for name in readdir(dir)
        occursin(needle, name) && return joinpath(dir, name)
    end
    return nothing
end

function discover_realizations(root::AbstractString)
    runs = NamedTuple[]
    isdir(root) || return runs
    for (dir, _, files) in walkdir(root)
        occursin(r"^realization = \d+$", basename(dir)) || continue
        any(f -> occursin("Gt Component", f), files) || continue

        L = nothing
        ζ = NaN
        topology = "waxman"
        for part in splitpath(dir)
            m = match(r"^L = (\d+)$", part)
            if m !== nothing
                L = parse(Int, m.captures[1])
            end
            m = match(r"^ζ = ([0-9.eE+\-]+)", part)
            if m !== nothing
                ζ = parse(Float64, m.captures[1])
            end
            if occursin("topology = ER", part)
                topology = "ER"
            end
        end
        L === nothing && continue
        push!(runs, (path = dir, L = L, zeta = ζ, topology = topology))
    end
    return runs
end

function interpolate_series(x_new::AbstractVector{<:Real},
                            x::AbstractVector{<:Real},
                            y::AbstractVector{<:Real})
    x_new == x && return Float64.(y)
    y_new = Vector{Float64}(undef, length(x_new))
    for (i, xv) in enumerate(x_new)
        if xv <= x[1]
            y_new[i] = y[1]
        elseif xv >= x[end]
            y_new[i] = y[end]
        else
            j = searchsortedlast(x, xv)
            t = (xv - x[j]) / (x[j + 1] - x[j])
            y_new[i] = (1.0 - t) * y[j] + t * y[j + 1]
        end
    end
    return y_new
end

end
