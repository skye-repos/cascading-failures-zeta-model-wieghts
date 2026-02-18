using Plots: legendfont
using Plots
using CSV, Tables
gr()

"""
    write_to_file(result::AbstractMatrix; file_name::AbstractString, header::Vector{Float64})

Write a matrix `result` to file with name `file_name` and header `header`
"""
function write_to_file(result::AbstractMatrix; file_name::AbstractString, header::Vector{Float64})
    result = result'
    table = Tables.table(result)
    header = string.(header)
    CSV.write(file_name, table, header=header)

    return nothing
end

"""
    results_plotter(sol_matrix::AbstractMatrix, θ_list=θ_list; ylabel, title, p_min=p_min)

Plot the results for either GC or Cascade time against θ values
"""
function results_plotter(sol_matrix::AbstractMatrix, θ_list=θ_list; ylabel, title, p_min=p_min)
    n_pts, l_θ = Base.size(sol_matrix)

    p_list = collect(range(p_min, 1.00, n_pts))

    plt = plot(palette=:viridis,
        xlabel="p", ylabel=ylabel,
        title=title, linewidth=0.75,
        legend=:bottomright, legend_columns=2,
        legendfontsize=8)

    for i ∈ 1:l_θ
        plot!(plt, p_list, sol_matrix[:, i];
            label="θ = $(θ_list[i])")
    end

    return plt
end
