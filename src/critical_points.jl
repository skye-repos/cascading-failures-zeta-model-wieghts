include("main.jl")

## Here we calculate the critical points for each graph generated for each θ
critical_points_list = zeros(length(θ_list))
for i ∈ eachindex(θ_list)
    local _, pc = detect_critical_point(float.(S_matrix[i, :]), collect(p_list))
    critical_points_list[i] = pc
end

## Plot the stuff
plot_crit = scatter(θ_list, critical_points_list;
    xlabel="θ", ylabel="Critical p",
    title="L=$(L), α=$(α), ζ = $(ζ), k=$(k)",
    label=["θ vs p-critical"],
    # palette = :dracula,
    linestyle=:dashdot, linewidth=1.5,
    marker=:circle, markersize=3)
crit_plot_file_name = "Critical p value vs θ comparison " * join(string.(θ_list), ", ") * ".png";
savefig(joinpath(plots_dir, crit_plot_file_name));
