using Plots, Distributions, Random
using Optim
using NLSolversBase, ForwardDiff, FiniteDifferences
using HypothesisTests
using Printf
using LinearAlgebra
using CSV
using DataFrames
using TypedTables

# Load dataset
df = CSV.read("psychtoday.csv", DataFrame; header=false)

# Rename columns
rename!(df, ["n_affairs", "constant", "age", "yrs_married", "religious", "occupation", "marriage_rating"])

# Define X amd y
X = Matrix(select(df, Not(:n_affairs)))  #RHS variables
y = df.n_affairs   #dependent variable

# Define log-likelihood function
loglike(β, X, y) = sum(-exp.(X * β) .+ y .* (X * β) .- log.(factorial.(y)))

# Define gradient of the negative log likelihood function
function gradient!(g, β; X = X, y = y)
    g .= X' * (y .- exp.(X * β))
end

# Initialize beta
β0 = zeros(size(X, 2))


function opti_multi_methods(method_name; f = (β -> -loglike(β, X, y)), gradient = nothing, initial_val = β0)
    # Measure time and run optimization
    time_elapsed = @elapsed begin
        if isnothing(gradient)
            result = optimize(f, initial_val, method_name)
        else
            result = optimize(f, gradient, initial_val, method_name)
        end
    end

    push!(results, (
        estimated_β=Optim.minimizer(result)
        iterations=Optim.iterations(result),
        fevals = Optim.f_calls(result),
        time = time,
    ))

    return results
end

# Optimize log-likelihood
time = @elapsed result = optimize(β -> -loglike(β, X, y), β0, BFGS())


# Extract estimated coefficients
β = Optim.minimizer(opt)

println("[Quasi-Newton with BFGS and a numerical derivative] Estimated β:", β)

opt = optimize(β -> -loglike(β, X, y), β0, NelderMead())

# Extract estimated coefficients
β = Optim.minimizer(opt)


# Optimize log-likelihood
opt = optimize(β -> -loglike(β, X, y), gradient!, β0, BFGS())

# Extract estimated coefficients
β = Optim.minimizer(opt)


# Code the BHHH algorithm
function bhhh_optimize(X, y, β0; max_iter=1000, tol=1E-7)
    n, k = size(X) 
    β = β0  # initialize β
    score_matrix(β, X, y) = - X .* (y .- exp.(X * β)) # define the score matrix for the negative log likelihood function
    fevals = 0  # Count function evaluations

    for iter in 1:max_iter
        fevals += 1
        S = score_matrix(β, X, y)  # Compute score matrix (n × k)

        # Compute the search step according to the notes
        d = -inv(S' * S) * S' * ones(n)  

        # Update β
        β_new = β + d

        # Check convergence
        if norm(β_new - β) < tol
            println("Converged after $iter iterations.")
            return (β=β_new, iter=iter, fevals = fevals)
        end

        β = β_new  # Update β
    end

    println("Maximum iterations reached.")
    return (; β, iter, fevals)
end

time_elapsed = @elapsed begin
    result = bhhh_optimize(X, y, β0)
end






