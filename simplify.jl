using Pkg
Pkg.activate(".")
using DataFrames
using CSV, Tables

dirs = readdir("L = 100/", join=true)

t_out = append!(
    collect(-1.0:0.05:-0.05),
    collect(0.05:0.05:1.00)
)

function create_single_csv(dir)
    files = readdir(dir, join=true)

    dfs = Vector{DataFrame}(undef, 2)
    for file ∈ files
        df = CSV.read(file, DataFrame)
        if size(df)[1] == 25
            df = df[6:25, :]
            dfs[2] = df
        else
            dfs[1] = df
        end        
    end
    df_out = append!(dfs[1], dfs[2])
    insertcols!(df_out, 1, :θ=>t_out)
    CSV.write(dir * ".csv", df_out)
    return nothing
end

for dir ∈ dirs
    create_single_csv(dir)
end
