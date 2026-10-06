"""
    AbstractActiveSetMethod

Base type for the different active set methods implemented
"""
abstract type AbstractActiveSetMethod end

"""
    BasicActiveSet <: AbstractActiveSetMethod

Composite type for the basic method consisting of running the test c(x) = 0,
contains one parameter
    `tol`: tolerance of the method
"""
struct BasicActiveSet <: AbstractActiveSetMethod
    tol::Float64
end

"""
    BasicActiveSet(; kwargs...)

Creates a BasicActiveSet with `tol` = 1e-8.

"""
BasicActiveSet(;tol = 1e-8) = BasicActiveSet(tol)

"""
    PrimalActiveSetLP <: AbstractActiveSetMethod

Composite type for the primal trust region method presented in [OberlinandWright-2006](@cite). We note that this method cannot detect weakly active constraints.
Contains the following fields:
-`silent`: bool setting the optimization solver used to silent (true) or not (false)
-`tol`: tolerance of the method
-`linear_program_solver`: linear solver used for the LP subproblem solved in the method
-`solver_options`: additional JuMP options to be given to the solver
-`ν`: penalty parameter, see [OberlinandWright-2006](@cite)
-`Δ`: trust region parameter, see [OberlinandWright-2006](@cite)

# Reference
[OberlinandWright-2006] Oberlin and Wright - 2006 - Active Set Identification in Nonlinear Programming

"""
struct PrimalActiveSetLP <: AbstractActiveSetMethod
    silent::Bool
    tol::Float64
    linear_program_solver::DataType
    solver_options::Tuple{Vararg{Pair}}
    ν::Float64
    Δ::Float64
end

"""
    PrimalActiveSetLP(linear_program_solver; kwargs...)

Create a PrimalActiveSetLP where all fields can be specified as keyword arguments, a linear program solver and the trust region parameter must be provided.
The following default values are set
-`silent`: true
-`tol`: 1e-8
-`solver_options`: () i.e. none
-`ν`: NaN which is replaced by 2*max(norm(λ, Inf), 1) when the associated find_active() is used, with λ the multipliers of the problem
"""
PrimalActiveSetLP(linear_program_solver::DataType, Δ;silent = true, tol = 1e-8, solver_options = (), ν = NaN) = PrimalActiveSetLP(silent, tol, linear_program_solver, solver_options, ν,Δ)


"""
    PrimalDualActiveSetLPEC <: AbstractActiveSetMethod

Composite type for the method based on the primal-dual estimate presented in [OberlinandWright-2006](@cite)
Contains the following fields:
-`silent`: bool setting the optimization solver used to silent (true) or not (false)
-`tol`: tolerance of the method
-`solver`: solver used for the MILP subproblem solved in the method
-`solver_options`: additional JuMP options to be given to the solver
-`M`: big M constant, see [OberlinandWright-2006](@cite)
-`β`: positive test parameter, see [OberlinandWright-2006](@cite)
-`σ`: test parameter in (0,1), see [OberlinandWright-2006](@cite)

# Reference
[OberlinandWright-2006] Oberlin and Wright - 2006 - Active Set Identification in Nonlinear Programming

"""
struct PrimalDualActiveSetLPEC <: AbstractActiveSetMethod
    silent::Bool
    tol::Float64
    solver::DataType
    solver_options::Tuple{Vararg{Pair}}
    M::Float64
    β::Float64
    σ::Float64
end

"""
    PrimalDualActiveSetLPEC(solver; kwargs...)

Create a PrimalDualActiveSetLPEC where all fields can be specified as keyword arguments, a MILP optimization solver must be provided.
The following default values are set:
-`silent`: true
-`tol`: 1e-8
-`solver_options`: () i.e. none
-`M`: NaN which is replaced by 10*max(norm(λ_ineq, Inf), norm(c_ineq, Inf)) when the associated find_active() is used, with c_ineq is the inequality constraints and λ_ineq their associated multipliers
-`β`: NaN which is replaced by 1/(m+n) when the associated find_active() is used, with n the number of variables and m of constraints
-`σ`: 0.75
"""
PrimalDualActiveSetLPEC(solver::DataType ;silent = true, tol = 1e-8, solver_options = (), M = NaN, β = NaN, σ = 0.75) = PrimalDualActiveSetLPEC(silent, tol, solver, solver_options, M, β, σ)

"""
    ApproximatePrimalDualActiveSetLPEC <: AbstractActiveSetMethod

Composite type for the linear programming approximation to the method based on the primal-dual estimate presented in [OberlinandWright-2006](@cite)
Contains the following fields:
-`silent`: bool setting the optimization solver used to silent (true) or not (false)
-`tol`: tolerance of the method
-`linear_program_solver`: linear solver used for the LP subproblem solved in the method
-`solver_options`: additional JuMP options to be given to the solver
-`β`: positive test parameter, see [OberlinandWright-2006](@cite)
-`σ`: test parameter in (0,1), see [OberlinandWright-2006](@cite)

# Reference
[OberlinandWright-2006] Oberlin and Wright - 2006 - Active Set Identification in Nonlinear Programming

"""

struct ApproximatePrimalDualActiveSetLPEC <: AbstractActiveSetMethod
    silent::Bool
    tol::Float64
    linear_program_solver::DataType
    solver_options::Tuple{Vararg{Pair}}
    β::Float64
    σ::Float64
end

"""
    ApproximatePrimalDualActiveSetLPEC(linear_program_solver; kwargs...)

Create an ApproximatePrimalDualActiveSetLPEC where all fields can be specified as keyword arguments, linear_program_solver must be provided.
The following default values are set:
-`silent`: true
-`tol`: 1e-8
-`solver_options`: () i.e. none
-`β`: NaN which is replaced by 1/(m+n) when the associated find_active() is used, with n the number of variables and m of constraints
-`σ`: 0.90
"""
ApproximatePrimalDualActiveSetLPEC(linear_program_solver::DataType ;silent = true, tol = 1e-8, solver_options = (), β = NaN, σ = 0.90) = ApproximatePrimalDualActiveSetLPEC(silent, tol, linear_program_solver, solver_options, β, σ)


# Convert the inequality rows of `form` detected as active into constraint and bound indices
function _active_set(form::StandardForm, rows)
    split = split_cons_bounds(rows, form.idx_ineq, form.is_con_ineq)
    return (active = split.constraints, active_boundary = split.bounds)
end

"""
    find_active(nlp, results, method::BasicActiveSet)

Find the active inequality constraints ``c_i`` matching their lower-bound (``c_i(x) = lb_i``) or their upper-bound (``c_i(x) = ub_i``) at the current solution ``x`` store in `results.solution`

Returns named tuple with 2 attributes:
- `active` : active set of usual constraints found
- `active_boundary` : active set of boundary constraints found

"""
function find_active(nlp, results, method::BasicActiveSet)
    form = standard_form(nlp, results.solution)

    rows = findall(form.c_ineq .>= -method.tol)

    return _active_set(form, rows)
end

"""
    find_active(nlp, results, method::PrimalActiveSetLP)

Implement the primal trust region method presented in [OberlinandWright-2006](@cite). For finding the active set at a solution with results being a point near the solution.
We note that this method cannot detect weakly active constraints.

Returns named tuple with 2 attributes:
- `active` : active set of usual constraints found
- `active_boundary` : active set of boundary constraints found

# Reference
[OberlinandWright-2006] Oberlin and Wright - 2006 - Active Set Identification in Nonlinear Programming
"""
function find_active(nlp, results, method::PrimalActiveSetLP)
    tol = method.tol

    x = results.solution

    y = results.multipliers
    zl = results.multipliers_L
    zu = results.multipliers_U

    form = standard_form(nlp, x)
    n = form.nvar

    g = NLPModels.grad(nlp, x)

    c_eq, J_eq = form.c_eq, form.J_eq
    n_eq = length(c_eq)

    # The LP subproblem uses the raw form c_low ≤ c ≤ c_upp of the inequality constraints
    c_ineq = form.val_ineq
    c_low = form.low_ineq
    c_upp = form.upp_ineq
    J_ineq = form.J_ineq_raw
    n_ineq = length(c_ineq)

    # Parameter calculation
    if isnan(method.ν)
        ν = 2 * max(norm(y,Inf), norm(zl, Inf), norm(zu, Inf), 1)
    else
        ν = method.ν
    end
    Δ = method.Δ

    # Error testing
    if Δ <= 0
        throw(DomainError(Δ, "Δ must be positive"))
    end
    if ν <= 0
        throw(DomainError(ν, "ν must be positive"))
    end

    # Sub problem resolution
    model = Model(optimizer_with_attributes(method.linear_program_solver,  method.solver_options...))

    if method.silent
        set_silent(model)
    end

    @variable(model, -Δ <=  d[1:n] <= Δ)    # Trust region constraint

    if n_eq != 0
        @variable(model, u[1:n_eq] >= 0)
        @variable(model, v[1:n_eq] >= 0)

        @constraint(model, J_eq * d + c_eq == u - v)

        @expression(model, slackCost_eq, ν * (sum(u)+sum(v)))
    else
        @expression(model, slackCost_eq, 0.)
    end

    if n_ineq != 0
        @variable(model,  r[1:n_ineq] >= 0)

        for i in 1:n_ineq
            if isfinite(c_upp[i])
                @constraint(model, J_ineq[i,:]' * d + c_ineq[i] <= c_upp[i] + r[i])
            end
            if isfinite(c_low[i])
                @constraint(model, J_ineq[i,:]' * d + c_ineq[i] >= c_low[i] - r[i])
            end
        end

        @expression(model, slackCost_ineq, ν * sum(r))
    else
        @expression(model, slackCost_ineq, 0.)
    end

    @objective(model, Min, dot(g,d) + slackCost_eq + slackCost_ineq)

    optimize!(model)

    if termination_status(model) != OPTIMAL
        error("LP subproblem failed to converge : $(termination_status(model))")
    end

    d_sol = value.(d)

    # Active set test
    c_lin = c_ineq + J_ineq * d_sol
    rows = findall((c_lin .>= c_upp .- tol) .| (c_lin .<= c_low .+ tol))

    return _active_set(form, rows)
end

"""
    find_active(nlp, results, method::PrimalDualActiveSetLPEC)

Implements the method based on the primal-dual estimate presented in [OberlinandWright-2006](@cite). For finding the active set at a solution with results being a point near the solution.

Returns named tuple with 2 attributes:
- `active` : active set of usual constraints found
- `active_boundary` : active set of boundary constraints found

# Reference
[OberlinandWright-2006] Oberlin and Wright - 2006 - Active Set Identification in Nonlinear Programming
"""
function find_active(nlp, results, method::PrimalDualActiveSetLPEC)
    tol = method.tol

    x = results.solution

    y = results.multipliers
    zl = results.multipliers_L
    zu = results.multipliers_U

    form = standard_form(nlp, x)
    n = form.nvar
    m = form.ncon

    g = NLPModels.grad(nlp, x)

    c_eq = form.c_eq
    n_eq = length(c_eq)
    Jt_eq = transpose(form.J_eq)

    # For inequality constraints, we use the convention c(x) <= 0
    c_ineq = form.c_ineq
    n_ineq = length(c_ineq)
    Jt_ineq = transpose(form.J_ineq)

    # Parameter calculation
    if isnan(method.M)
        M = 10* max(norm(c_ineq, Inf), norm(y, Inf), norm(zu, Inf), norm(zl, Inf))
    else
        M = method.M
    end
    if isnan(method.β)
        β = 1/(m+n)
    else
        β = method.β
    end

    σ = method.σ

    # Error testing
    if M <= 0
        throw(DomainError(M, " M must be positive"))
    end
    if β <= 0
        throw(DomainError(β, " β must be positive"))
    end
    if σ <= 0 || σ >=1
        throw(DomainError(σ, "σ must be in (0,1)"))
    end


    # Sub problem resolution
    model = Model(optimizer_with_attributes(method.solver,  method.solver_options...))

    if method.silent
        set_silent(model)
    end

    @variable(model, u[1:n]>=0)
    @variable(model, v[1:n]>=0)

    if n_eq != 0
        @variable(model, λ_eq[1:n_eq])
        @expression(model, Jtprod_eq, Jt_eq * λ_eq)
    else
        @expression(model, Jtprod_eq, zeros(n))
    end

    if n_ineq != 0
        @variable(model,  λ_ineq[1:n_ineq] >= 0)
        @variable(model, s[1:n_ineq])
        @variable(model, y[1:n_ineq], Bin)
        set_lower_bound.(s, max.(c_ineq, zeros(n_ineq)))

        @constraint(model, -c_ineq - s .<= -c_ineq .* y)
        @constraint(model, λ_ineq - s .<= M*(1 .- y))


        @expression(model, Jtprod_ineq, Jt_ineq * λ_ineq)
        @expression(model, slackCost, sum(s))
    else
        @expression(model, Jtprod_ineq, zeros(n))
        @expression(model, slackCost, 0.)
    end

    @constraint(model, Jtprod_eq + Jtprod_ineq + g == u - v)

    @objective(model, Min, slackCost + sum(u) + sum(v))

    optimize!(model)

    if termination_status(model) != OPTIMAL
        error("LP subproblem failed to converge : $(termination_status(model))")
    end

    ω = objective_value(model) + norm(c_eq,1)

    if ω < -tol
        throw(DomainError(ω, "Optimal value of subproblem is negative"))
    end

    ω = clamp(ω, 0., Inf)      # TODO better way to deal with the case where ω is in [-tol, 0) ?

    # Active set test
    rows = findall(c_ineq .>= -(β*ω)^σ - tol)

    return _active_set(form, rows)
end


"""
    find_active(nlp, results, method::ApproximatePrimalDualActiveSetLPEC)

Implements the linear programming approximation of the method based on the primal-dual estimate presented in [OberlinandWright-2006](@cite). For finding the active set at a solution with results being a point near the solution.

Returns named tuple with 2 attributes:
- `active` : active set of usual constraints found
- `active_boundary` : active set of boundary constraints found

# Reference
[OberlinandWright-2006] Oberlin and Wright - 2006 - Active Set Identification in Nonlinear Programming
"""
function find_active(nlp, results, method::ApproximatePrimalDualActiveSetLPEC)
    tol = method.tol

    x = results.solution
    y = results.multipliers
    zl = results.multipliers_L
    zu = results.multipliers_U

    form = standard_form(nlp, x)
    n = form.nvar
    m = form.ncon

    g = NLPModels.grad(nlp, x)

    c_eq = form.c_eq
    n_eq = length(c_eq)
    Jt_eq = transpose(form.J_eq)

    # For inequality constraints, we use the convention c(x) <= 0
    c_ineq = form.c_ineq
    n_ineq = length(c_ineq)
    Jt_ineq = transpose(form.J_ineq)

    neg_idx = findall(x -> x<-tol, c_ineq)
    pos_idx = setdiff(1:n_ineq, neg_idx)

    # Parameter calculation
    K_1 = max(norm(c_ineq, Inf), norm(y, Inf), norm(zl, Inf), norm(zu, Inf))

    if isnan(method.β)
        β = 1/(m+n)
    else
        β = method.β
    end
    σ = method.σ

    # Error testing
    if β <= 0
        throw(DomainError(β, " β must be positive"))
    end
    if σ <= 0 || σ >=1
        throw(DomainError(σ, "σ must be in (0,1)"))
    end

    # Sub problem resolution
    model = Model(optimizer_with_attributes(method.linear_program_solver,  method.solver_options...))

    if method.silent
        set_silent(model)
    end


    @variable(model, u[1:n]>=0)
    @variable(model, v[1:n]>=0)

    if n_eq != 0
        @variable(model, λ_eq[1:n_eq])
        @expression(model, Jtprod_eq, Jt_eq * λ_eq)
    else
        @expression(model, Jtprod_eq, zeros(n))
    end

    if n_ineq != 0
        @variable(model,  K_1 >= λ_ineq[1:n_ineq] >= 0)

        @expression(model, Jtprod_ineq, Jt_ineq * λ_ineq)
        @expression(model, ineqCost, -dot(c_ineq[neg_idx], λ_ineq[neg_idx]))
    else
        @expression(model, Jtprod_ineq, zeros(n))
        @expression(model, ineqCost, 0.)
    end

    @constraint(model, Jtprod_eq + Jtprod_ineq + g == u - v)

    @objective(model, Min, ineqCost + sum(u) + sum(v))

    optimize!(model)

    if termination_status(model) != OPTIMAL
        error("LP subproblem failed to converge : $(termination_status(model))")
    end

    if n_eq != 0
        λ_eq_sol = value.(λ_eq)
    else
        λ_eq_sol = Float64[]
    end

    if n_ineq != 0
        λ_ineq_sol = value.(λ_ineq)
    else
        λ_ineq_sol = Float64[]
    end

    #TODO find a better solution than clamp() to ensure ρ_sup is positive (under tol)
    ρ_sup = sum(clamp(-c_ineq[i] * λ_ineq_sol[i], 0., Inf)^(1/2) for i in neg_idx; init=0.) +
            sum(c_ineq[pos_idx]) +
            norm(c_eq, 1) +
            norm(Jt_ineq * λ_ineq_sol + Jt_eq * λ_eq_sol + g, 1
    )

    if ρ_sup < -tol
        throw(DomainError(ρ_sup, "test bound ρ_sup is negative"))
    end
    ρ_sup = clamp(ρ_sup, 0., Inf) #TODO same question here

    # Active set test
    rows = findall(c_ineq .>= -(β*ρ_sup)^σ - tol)

    return _active_set(form, rows)
end
