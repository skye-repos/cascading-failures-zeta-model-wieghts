using Pkg
Pkg.activate(".")
using DataFrames
using CSV, Tables
using Plots
gr()

files = readdir("fixed-thetas/", join=true)

ζ_list = collect(5.0:2.5:50.0)
labels = string.(ζ_list)
header = names(CSV.read(files[1], DataFrame))[2:end]
p_list = parse.(Float64, header)

function plot_results(file)
    df = CSV.read(file, DataFrame)
    θ = split(split(file, " ")[end], ".csv")[1]
    θ = parse(Float64, θ)
    plt = plot(
        title = "Giant Component for various ζ, θ = $(θ)",
        xlabel = "p", ylabel = "S",
        palette=:dracula)

    for row ∈ eachrow(df)
        row = collect(row)
        plot!(plt,
              p_list, row[2:end], label="ζ = $(row[1])")
    end

    savefig(plt, "plots/θ = $(θ).png")
    return nothing
end

function heatmap_results(file)
    df = CSV.read(file, DataFrame)
    θ = split(split(file, " ")[end], ".csv")[1]
    θ = parse(Float64, θ)
    df = Matrix(df[:, 2:end])

    hmp = heatmap(p_list, ζ_list, df, c=:viridis,
                  xlabel = "p", ylabel = "ζ", colorbar_title = "S")
    savefig(hmp, "heatmaps/θ = $(θ).png")

    return nothing
end

if !ispath("plots/") || !ispath("heatmaps/")
    mkpath("plots/")
    mkpath("heatmaps/")
end

for file ∈ files
    plot_results(file)
    heatmap_results(file)
end
