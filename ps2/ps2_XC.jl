using Optim
using NLSolversBase, ForwardDiff, FiniteDifferences, NLLSsolver
using Printf
using LinearAlgebra
using CSV
using DataFrames

# Load dataset
df = CSV.read("psychtoday.csv", DataFrame; header=false)

# Rename columns
rename!(df, ["n_affairs", "constant", "age", "yrs_married", "religious", "occupation", "marriage_rating"])

# Define X and y
X = Matrix(select(df, Not(:n_affairs)))  #RHS variables
y = df.n_affairs   #dependent variable

# Define log-likelihood function
loglike(β, X, y) = sum(-exp.(X * β) .+ y .* (X * β) .- log.(factorial.(y)))

# Define gradient of the negative log likelihood function
function gradient!(g, β; X = X, y = y)
    g .= -X' * (y .- exp.(X * β))
end

# Initialize beta
β0 = zeros(size(X, 2))

# Define function to optimize using multiple methods
function opti_multi_methods(method_name; f=(β -> -loglike(β, X, y)), gradient=nothing, initial_val=β0)
    local result, time_elapsed

    # Measure time and run optimization
    time_elapsed = @elapsed begin
        if isnothing(gradient)
            result = optimize(f, initial_val, method_name)
        else
            result = optimize(f, gradient, initial_val, method_name)
        end
    end

    # return estimated βs, number of iterations, number of function evaluations, and time
    return (
        estimated_β=Optim.minimizer(result),
        iterations=Optim.iterations(result),
        fevals=Optim.f_calls(result),
        time=time_elapsed
    )
end

# initialize a dataframe that stores results from different optimization methods
results = DataFrame(estimated_β=Vector[], iterations=Int[], fevals=Int[], time=Float64[])

# These are the 3 optimization methods that we want to run
methods = [
    (BFGS(), nothing),            # BFGS with numerical gradient
    (BFGS(), gradient!),          # BFGS with analytical gradient
    (NelderMead(), nothing)       # Nelder-Mead
]

# Run all three methods and push results
append!(results, [opti_multi_methods(method; gradient=grad) for (method, grad) in methods])

# Code a function for the BHHH algorithm
function bhhh_optimize(X, y, β0; max_iter=1000, tol=1E-7)
    n, k = size(X) 
    β = β0  # initialize β
    score_matrix(β, X, y) = - X .* (y .- exp.(X * β)) # define the score matrix for the negative log likelihood function
    fevals = 0  # Count function evaluations
    eigen_initial = eigvals(score_matrix(β, X, y)' * score_matrix(β, X, y))
    for iter in 1:max_iter
        fevals += 1     #This is probably wrong. CHECK.
        S = score_matrix(β, X, y)  # Compute score matrix (n × k)

        # Compute the search step according to the notes
        d = -(S' * S) \ (S' * ones(n)) 

        # Update β
        β_new = β + d

        # Check convergence
        if norm(β_new - β) < tol
            println("Converged after $iter iterations.")
            eigen_final = eigvals(score_matrix(β_new, X, y)' * score_matrix(β_new, X, y))
            return (β=β_new, iter=iter, fevals = fevals, eigen_final = eigen_final, eigen_initial = eigen_initial) # Why do we need these equal signs??
        end

        β = β_new  # Update β
    end

    println("Maximum iterations reached.")
    return (; β, iter, fevals, eigen_initial, eigen_final)
end

# Measure time for BHHH
time_elapsed = @elapsed result = bhhh_optimize(X, y, β0)

# Report the eigen values for the initial and final Hessian approximations
println("The initial eigenvalue is $(result.eigen_initial). The final eigenvalue is $(result.eigen_final)")

# Push results to the dataframe
push!(results, (result.β, result.iter, result.fevals, time_elapsed)) 


function NLLS(f, J, X, y, θ0; tol=1e-7, max_iter=1000)
    θ = θ0
    fevals = 0
    for iter in 1:max_iter
        fevals += 1 # Again, this is probably not right...

        # Compute residuals
        r = f(θ, X, y)  

        # Compute Jacobian matrix
        Jθ = J(θ, X)  

        # Approximate Hessian
        H_approx = Jθ' * Jθ  

        # Compute search direction
        d = -H_approx \ (Jθ' * r)  

        # Update parameters
        θ_new = θ + d

        # Check for convergence
        if norm(d) < tol
            println("Converged after $iter iterations.")
            return θ_new, iter, fevals
        end

        θ = θ_new
    end

    println("Max iterations reached. Returning last estimate.")
    return (; θ, iter, fevals)
end

# Define the function and Jacobian 
function residuals(θ, X, y)
    return y .- exp.(X * θ)  
end

function jacobian(θ, X)
    return -X .* exp.(X * θ) 
end

time_elapsed = @elapsed θ, iter, fevals = NLLS(residuals, jacobian, X, y, β0)

# Push results to the dataframe
push!(results, (θ, iter, fevals, time_elapsed)) 

# add a column that describes what the methods are
# (Probably not the best way to do this?)
insertcols!(results, 1, :method => ["BFGS (numerical)", "BFGS (analytical)", "Nelder-Mead", "BHHH", "NLLS"])

# Display results
println("\nOptimization Results:")
display(results)

CSV.write("opti_results.csv", results)








