"""
    AbstractMFCQDirectionMethod

Base type for methods implemented for checking the direction condition of the MFCQ constraint qualification
"""
abstract type AbstractMFCQDirectionMethod end

"""
    MFCQDirectionPrimal <: AbstractMFCQDirectionMethod

Composite type for the method based on using a Max linear program to check the direction condition, 
Contains the following parameters:
-`silent`: bool setting the optimization solver used to silent (true) or not (false)
-`tol`: tolerance of the method
-`linear_program_solver`: solver used for the linear subproblem solved in the method
-`solver_options`: additional JuMP options to be given to the solver
-`K`: Upper bound on the solution to avoid having the optimal value be infinite if a solution exists
"""
struct MFCQDirectionPrimal <: AbstractMFCQDirectionMethod
    silent::Bool
    tol::Float64
    linear_program_solver::DataType
    solver_options::Tuple{Vararg{Pair}}
    K::Float64
end

"""
    MFCQDirectionPrimal(solver; kwargs...)

Create an MFCQDirectionPrimal where all fields can be specified as keyword arguments, a linear program solver must be provided.
The following default values are set
-`silent`: true
-`tol`: 1e-8
-`solver_options`: () i.e. none
-`K`: 1
"""
MFCQDirectionPrimal(linear_program_solver::DataType;silent = true, tol = 1e-8, solver_options = (), K = 1) = MFCQDirectionPrimal(silent, tol, linear_program_solver, solver_options, K)

"""
    MFCQDirectionDual <: AbstractMFCQDirectionMethod

Composite type for the method based on using a Max L1 problem in a Big-M MILP formulation to solve the dual formulation of the direction condition, 
Contains the following parameters:
-`silent`: bool setting the optimization solver used to silent (true) or not (false)
-`tol`: tolerance of the method
-`solver`: solver used for the Big-M MILP subproblem solved in the method
-`solver_options`: additional JuMP options to be given to the solver
-`K`: Upper bound on the solution to avoid having the optimal value be infinite if a non zero solution exist, 2*K is used as the Big-M value
"""
struct MFCQDirectionDual <: AbstractMFCQDirectionMethod
    silent::Bool
    tol::Float64
    solver::DataType
    solver_options::Tuple{Vararg{Pair}}
    K::Float64
end

"""
    MFCQDirectionDual(solver; kwargs...)

Create an MFCQDirectionDual where all fields can be specified as keyword arguments, a MILP solver must be provided.
The following default values are set
-`silent`: true
-`tol`: 1e-8
-`solver_options`: () i.e. none
-`K`: 1
"""
MFCQDirectionDual(solver::DataType;silent = true, tol = 1e-8, solver_options = (), K = 1) = MFCQDirectionDual(silent, tol, solver, solver_options, K)


"""
    check_MFCQ_direction(Jac, n_eq, method::MFCQDirectionPrimal)

Solves the problem
```
max_{t,d}   t
subject to  Jac_eq * d == 0     Jac_eq the jacobian of equality constraints
            Jac_ineq * d <= t   Jac_ineq the jacobian of inequality constraints
            -K <= d <= K    

```
if t > 0 the condition is verified

Return
the optimal solution of the above problem in the form of a named tuple with 2 fields:
-`direction`: d
-`slack`: t
"""
function check_MFCQ_direction(Jac, n_eq, method::MFCQDirectionPrimal)

    if isempty(Jac)
        return (direction = [], slack = [])
    end

    n_jac, n = size(Jac)
    n_ineq = n_jac - n_eq

    K = method.K

    if K <= 0
        throw(DomainError(K, " K must be positive"))
    end

    model = Model(optimizer_with_attributes(method.linear_program_solver,  method.solver_options...)) 

    if method.silent
        set_silent(model)
    end 

    @variable(model, -K <= d[1:n] <= K)

    if n_eq != 0
        @constraint(model, Jac[1:n_eq,:] * d == 0)
    end

    if n_ineq != 0
        @variable(model, t[1:n_ineq])
        @constraint(model, Jac[1+n_eq : n_jac,:] * d <= -t)

        @expression(model, obj_term, sum(t))
    else
        @warn "M"
        @expression(model, obj_term, 0.)
    end

    @objective(model, Max, obj_term)

    optimize!(model)

    if termination_status(model) != OPTIMAL 
        error("Linear subproblem failed to converge : $(termination_status(model))")
    end

    if n_ineq != 0
        t_sol = value.(t)
    else
        t_sol = []
    end

    d_sol = value.(d)

    return (direction = d_sol, slack = t_sol)   # TODO Smarter more compact return ?
end


"""
    check_MFCQ_direction(Jac, n_eq, method::MFCQDirectionDual)

Solves the problem
```
max_λ       ||λ||_1
subject to  Jac_t * λ == 0
            λ_i >= 0    Where i takes the indices of inequality constraints
            -K <= λ <= K    

```
With a Big-M MILP reformulation

Return
The optimal λ found for the above problem 
"""
function check_MFCQ_direction(Jac, n_eq, method::MFCQDirectionDual)

    if isempty(Jac)
        return []
    end

    n_jac, n = size(Jac)
    n_ineq = n_jac - n_eq
    Jac_t = transpose(Jac)

    K = method.K

    if K<= 0
        throw(DomainError(K, " K must be positive"))
    end

    model = Model(optimizer_with_attributes(method.solver,  method.solver_options...)) 

    if method.silent
        set_silent(model)
    end 

    @variable(model, -K <= λ[1:n_jac] <= K)
    @constraint(model, Jac_t * λ == 0)

    if n_eq != 0
        @variable(model, u[1:n_eq] >= 0)
        @variable(model, v[1:n_eq] >= 0)
        @variable(model, ξ[1:n_eq], Bin)

        @constraint(model, λ[1:n_eq] == u - v)

        @constraint(model, u <= 2*K*ξ)
        @constraint(model, v <= 2*K*(1 .- ξ))

        @expression(model, eqNorm, sum(u) + sum(v))
    else
        @expression(model, eqNorm, 0.)
    end

    if n_ineq != 0
        set_lower_bound.(λ[1+n_eq : n_jac], 0.)
    end


    @objective(model, Max, eqNorm + sum(λ[1+n_eq : n_jac]))

    optimize!(model)

    if termination_status(model) != OPTIMAL 
        error("MILP subproblem failed to converge : $(termination_status(model))")
    end

    λ_sol = value.(λ)
    return λ_sol
end

