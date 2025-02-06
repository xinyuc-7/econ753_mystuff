using Plots, Printf, LinearAlgebra, NonlinearSolve

# Define the function f! and its initial guess
function f!(F, x)
    F[1] = x[1]^2 + x[2]^2 - 1
    F[2] = x[1] - x[2]
end

# Define the jacobian for this specific problem
function jaco(x)
    return [2*x[1] 2*x[2]
            1 -1]
end


# Broyden's Method 
function broyden(f!, jaco, x0; tol=1e-8, max_iter=100)
    n = length(x0)              # Number of variables
    x = copy(x0)                # Current guess
    F = zeros(n)                # Residual vector
    println("first F is $F")
    f!(F, x)                    # Initial function evaluation

    # Initialize the Jacobian at the numerical value of x0
    B = jaco(x0)

    for iter in 1:max_iter
        # Solve B * dx = -F for dx
        dx = -B \ F

        # Update x
        x .= x .+ dx

        if norm(dx, Inf) < tol
            println("Converged in $iter iterations.")
            return x
        end

        # Compute new residual F
        F_new = zeros(n)
        f!(F_new, x)
        println("F_new is $F_new")

        # Update Jacobian approximation using Broyden's formula
        delta_F = F_new - F
        delta_x = dx
        B .= B .+ (delta_F - B * delta_x) * (delta_x') / (delta_x' * delta_x)

        # Update F for the next iteration
        F .= F_new
    end

    error("Broyden's method did not converge within $max_iter iterations.")
end

# Initial guess
x0 = [2.0, 1.0]

# Solve the system
solution = broyden(f!, jaco, x0)
println("Solution: $solution")


function betterbroyden(f!, jaco, x_initial; tolerance = 1E-7, maxiter = 1000)
    x_old = x_initial
    jaco_old = jaco(x_initial)
    error = Inf
    iter = 1

    while error > tolerance && iter <= maxiter
        dx = -f!(x_initial) \ jaco_old

        x_new = x_old .+ dx

        df = f!(x_new) - f!(x_old)

        jaco_new = jaco_old .+ 


    end
    
end