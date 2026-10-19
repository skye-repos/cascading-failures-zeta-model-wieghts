if !isdefined(Main, :PCUtils)
    include("PCUtils.jl")
end

module PCTest

using Test
using Random
using ..PCUtils

export run_tests

function run_tests()
    p = collect(range(0.5, 1.0; length = 50))
    Δp = p[2] - p[1]

    @testset "p_c on a known sigmoid" begin
        S = @. 1.0 / (1.0 + exp(-(p - 0.75) / 0.02))
        result = find_critical_full(p, S; sigma_pts = 2.0)
        @test result.sign_change
        @test abs(result.pc - 0.75) <= 2.0 * Δp

        rng = MersenneTwister(20261017)
        S_noisy = S .+ 0.005 .* randn(rng, length(S))
        pc_noisy = find_critical(p, S_noisy; sigma_pts = 2.0)
        @test abs(pc_noisy - 0.75) <= 3.0 * Δp

        pc_raw = find_critical(p, S; sigma_pts = 0.0)
        @test abs(pc_raw - 0.75) <= 2.0 * Δp
    end

    @testset "smoothing preserves a constant curve" begin
        S = fill(0.5, length(p))
        S̃ = smooth_along_p(p, S; sigma_pts = 2.0)
        @test all(isapprox.(S̃, 0.5; atol = 1e-14))
    end

    println("p_c tests passed")
end

end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    PCTest.run_tests()
end
