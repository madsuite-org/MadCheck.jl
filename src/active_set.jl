"""
    AbstractActiveSetMethod

Base type for the different active set methods implemented
"""
abstract type AbstractActiveSetMethod end

"""
    BasicActiveSet <: AbstractActiveSetMethod

Composite type for the basic method consisting of runing the test c(x) = 0, 
contains one parameter
    `tol`: tolerance of the method
"""
struct BasicActiveSet <: AbstractActiveSetMethod
    tol::Float64
end

"""
    BasicActiveSet(; kwargs...)

Creates an ActiveMethodeSimple with `tol` = 1e-8.

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

Create an PrimalActiveSetLP where all fields can be specified as keyword arguments, a linear program solver and the trust region parameter must be provided.
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

Create an PrimalDualActiveSetLPEC where all fields can be specified as keyword arguments, a MILP optimization solver must be provided.
The following default values are set:
-`silent`: true
-`tol`: 1e-8
-`solver_options`: () i.e. none
-`M`: NaN wich is replaced by 10*max(norm(λ_ineq, Inf), norm(c_ineq, Inf)) when the associated find_active() is used, with c_ineq is the inequality constraints and λ_ineq their associated multipliers
-`β`: NaN wich is replaced by 1/(m+n) when the associated find_active() is used, with n the number of variables and m of constraints
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
-`β`: NaN wich is replaced by 1/(m+n) when the associated find_active() is used, with n the number of variables and m of constraints
-`σ`: 0.90
"""

ApproximatePrimalDualActiveSetLPEC(linear_program_solver::DataType ;silent = true, tol = 1e-8, solver_options = (), β = NaN, σ = 0.90) = ApproximatePrimalDualActiveSetLPEC(silent, tol, linear_program_solver, solver_options, β, σ)


"""
    find_active(nlp, results, method::BasicActiveSet)
0
Find the active inequality constraints ``c_i`` matching their lower-bound (``c_i(x) = lb_i``) or their upper-bound (``c_i(x) = ub_i``) at the current solution ``x`` store in `results.solution`

Returns named tuple with 2 attributes:
- `active` : active set of usual constraints found
- `active_boundary` : active set of boundary constraints found

"""
function find_active(nlp, results, method::BasicActiveSet)
    tol = method.tol

    x = results.solution

    constraints = NLPModels.cons(nlp, x)

    lvar = nlp.meta.lvar
    uvar = nlp.meta.uvar

    lcon = nlp.meta.lcon
    ucon = nlp.meta.ucon

    jlow = nlp.meta.jlow
    jupp = nlp.meta.jupp
    jrng = nlp.meta.jrng

    ilow = nlp.meta.ilow
    iupp = nlp.meta.iupp
    irng = nlp.meta.irng


    var_ineq_idx = vcat(irng, iupp, ilow)
    con_ineq_idx = vcat(jrng, jlow, jupp)

    active = Int64[]
    active_boundary = Int64[]

    # Test run
    for i in var_ineq_idx
        if x[i]  - lvar[i] <= tol || uvar[i] - x[i] <= tol
            push!(active_boundary, i)
        end
    end

    for i in con_ineq_idx
        if constraints[i]  - lcon[i] <= tol || ucon[i] - constraints[i] <= tol
            push!(active, i)
        end
    end

    return (active = active, active_boundary = active_boundary)
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
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    tol = method.tol

    jfix = nlp.meta.jfix
    jlow = nlp.meta.jlow
    jupp = nlp.meta.jupp
    jrng = nlp.meta.jrng

    ifix = nlp.meta.ifix
    ilow = nlp.meta.ilow
    iupp = nlp.meta.iupp
    irng = nlp.meta.irng

    lvar = nlp.meta.lvar
    uvar = nlp.meta.uvar

    lcon = nlp.meta.lcon
    ucon = nlp.meta.ucon

    x = results.solution

    y = results.multipliers
    zl = results.multipliers_L
    zu = results.multipliers_U

    Ji, Jj = NLPModels.jac_structure(nlp)
    Jx = NLPModels.jac_coord(nlp, results.solution)
    Jac = sparse(Ji, Jj, Jx, m, n)

    constraints = NLPModels.cons(nlp, x)

    g = NLPModels.grad(nlp, x) 

    # Creating equality and inequality constraints and gradients in the right format
    c_eq = vcat(constraints[jfix] - ucon[jfix], x[ifix] - uvar[ifix])
    n_eq = length(c_eq)
    J_eq = vcat(Jac[jfix,:], spdiagm(ones(n))[ifix,:])

    c_ineq = vcat(constraints[jrng],
                  constraints[jupp],
                  constraints[jlow],
                  x[irng],
                  x[iupp],
                  x[ilow],
    )

    u = vcat(ucon[jrng],
             ucon[jupp],
             ucon[jlow],
             uvar[irng],
             uvar[iupp],
             uvar[ilow],
    )

    l = vcat(lcon[jrng],
             lcon[jupp],
             lcon[jlow],
             lvar[irng],
             lvar[iupp],
             lvar[ilow],
    )

    J_ineq = vcat(Jac[jrng, :],
                  Jac[jupp, :],
                  Jac[jlow, :],
                  spdiagm(ones(n))[irng,:],
                  spdiagm(ones(n))[iupp,:],
                  spdiagm(ones(n))[ilow,:],
    )

    indices_to_constraints = vcat(jrng, jupp, jlow, irng, iupp, ilow)
    n_ineq = length(c_ineq)
    n_ineq_con = length(jrng) + length(jupp) + length(jlow)

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
            if isfinite(u[i])
                @constraint(model, J_ineq[i,:]' * d + c_ineq[i] <= u[i] + r[i])
            end
            if isfinite(l[i])
                @constraint(model, J_ineq[i,:]' * d + c_ineq[i] >= l[i] + r[i])
            end
        end

        @expression(model, slackCost_ineq, ν * sum(r))
    else
        @expression(model, slackCost_eq, 0.)
    end
    
    @objective(model, Min, dot(g,d) + slackCost_eq + slackCost_ineq)

    optimize!(model)

    if termination_status(model) != OPTIMAL 
        error("LP subproblem failed to converge : $(termination_status(model))")
    end

    d_sol = value.(d)

    # Active set test

    active = Int64[]
    active_boundary = Int64[]

    for i in 1:n_ineq
        if dot(J_ineq[i, :], d_sol) + c_ineq[i] >= u[i] - tol || dot(J_ineq[i, :], d_sol) + c_ineq[i] <= l[i] + tol 
            if i <= n_ineq_con
                push!(active, indices_to_constraints[i])
            else
                push!(active_boundary, indices_to_constraints[i])
            end
        end
    end


    return (active = active, active_boundary = active_boundary)
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
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    tol = method.tol

    jfix = nlp.meta.jfix
    jlow = nlp.meta.jlow
    jupp = nlp.meta.jupp
    jrng = nlp.meta.jrng

    ifix = nlp.meta.ifix
    ilow = nlp.meta.ilow
    iupp = nlp.meta.iupp
    irng = nlp.meta.irng

    allupp_con = vcat(jrng, jupp)
    alllow_con = vcat(jrng, jlow)

    allupp_var = vcat(irng, iupp)
    alllow_var = vcat(irng, ilow)

    lvar = nlp.meta.lvar
    uvar = nlp.meta.uvar

    lcon = nlp.meta.lcon
    ucon = nlp.meta.ucon

    x = results.solution

    y = results.multipliers
    zl = results.multipliers_L    
    zu = results.multipliers_U

    Ji, Jj = NLPModels.jac_structure(nlp)
    Jx = NLPModels.jac_coord(nlp, x)
    Jac = sparse(Ji, Jj, Jx, m, n)

    constraints = NLPModels.cons(nlp, x)
    g = NLPModels.grad(nlp, x)

    # Creating equality and inequality constraints and gradients in the right format
    c_eq = vcat(constraints[jfix] - ucon[jfix], x[ifix] - uvar[ifix])
    n_eq = length(c_eq)
    J_eq = vcat(Jac[jfix,:], spdiagm(ones(n))[ifix,:])
    Jt_eq = transpose(J_eq)

    # For inequality contraints, we use the convention c(x) <= 0

    c_ineq = vcat(min.(constraints[jrng] - ucon[jrng], lcon[jrng] - constraints[jrng]),
                  constraints[jupp] -  ucon[jupp],
                  lcon[jlow] - constraints[jlow],
                  min.(x[irng] - uvar[irng], lvar[irng] - x[irng]),
                  x[iupp] - uvar[iupp],
                  lvar[ilow] - x[ilow],
    )

    sign_con = [(ucon[i] - constraints[i]) > (constraints[i] - lcon[i]) ? -1 : 1 for i in jrng]
    sign_var = [(uvar[i] - x[i]) > (x[i] - lvar[i]) ? -1 : 1 for i in jrng]

    J_ineq = vcat(Jac[jrng, :] .* sign_con,
                  Jac[jupp, :],
                  -Jac[jlow, :],
                  spdiagm(ones(n))[irng,:] .* sign_var,
                  spdiagm(ones(n))[iupp,:],
                  -spdiagm(ones(n))[ilow,:],
    )

    indices_to_constraints = vcat(jrng, jupp, jlow, irng, iupp, ilow)
    n_ineq = length(c_ineq)
    n_ineq_con = length(jrng) + length(jupp) + length(jlow)
    Jt_ineq = transpose(J_ineq)

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
        trow(DomainError(ω, "Optimal value of subproblem is negative"))
    end

    clamp(ω, 0., Inf)      # TODO better way to deal with the case where ω is in [-tol, 0) ?

    # Active set test

    active = Int64[]
    active_boundary = Int64[]

    for i in 1:n_ineq
        if c_ineq[i] >= -(β*ω)^σ - tol
            if i <= n_ineq_con
                push!(active, indices_to_constraints[i])
            else
                push!(active_boundary, indices_to_constraints[i])
            end
        end
    end

    return (active = active, active_boundary = active_boundary)
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
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    tol = method.tol

    jfix = nlp.meta.jfix
    jlow = nlp.meta.jlow
    jupp = nlp.meta.jupp
    jrng = nlp.meta.jrng

    ifix = nlp.meta.ifix
    ilow = nlp.meta.ilow
    iupp = nlp.meta.iupp
    irng = nlp.meta.irng

    allupp_con = vcat(jrng, jupp)
    alllow_con = vcat(jrng, jlow)

    allupp_var = vcat(irng, iupp)
    alllow_var = vcat(irng, ilow)

    lvar = nlp.meta.lvar
    uvar = nlp.meta.uvar

    lcon = nlp.meta.lcon
    ucon = nlp.meta.ucon


    x = results.solution
    y = results.multipliers
    zl = results.multipliers_L    
    zu = results.multipliers_U

    Ji, Jj = NLPModels.jac_structure(nlp)
    Jx = NLPModels.jac_coord(nlp, results.solution)
    Jac = sparse(Ji, Jj, Jx, m, n)
    constraints = NLPModels.cons(nlp, x)

    g = NLPModels.grad(nlp, x)

    # Creating equality and inequality constraints and gradients in the right format
    c_eq = vcat(constraints[jfix] - ucon[jfix], x[ifix] - uvar[ifix])
    n_eq = length(c_eq)
    J_eq = vcat(Jac[jfix,:], spdiagm(ones(n))[ifix,:])
    Jt_eq = transpose(J_eq)

    # For inequality contraints, we use the convention c(x) <= 0

    c_ineq = vcat(min.(constraints[jrng] - ucon[jrng], lcon[jrng] - constraints[jrng]),
                  constraints[jupp] -  ucon[jupp],
                  lcon[jlow] - constraints[jlow],
                  min.(x[irng] - uvar[irng], lvar[irng] - x[irng]),
                  x[iupp] - uvar[iupp],
                  lvar[ilow] - x[ilow],
    )

    sign_con = [(ucon[i] - constraints[i]) > (constraints[i] - lcon[i]) ? -1 : 1 for i in jrng]
    sign_var = [(uvar[i] - x[i]) > (x[i] - lvar[i]) ? -1 : 1 for i in jrng]

    J_ineq = vcat(Jac[jrng, :] .* sign_con,
                  Jac[jupp, :],
                  -Jac[jlow, :],
                  spdiagm(ones(n))[irng,:] .* sign_var,
                  spdiagm(ones(n))[iupp,:],
                  -spdiagm(ones(n))[ilow,:],
    )

    indices_to_constraints = vcat(jrng, jupp, jlow, irng, iupp, ilow)
    n_ineq = length(c_ineq)
    n_ineq_con = length(jrng) + length(jupp) + length(jlow)
    Jt_ineq = transpose(J_ineq)

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
        trow(DomainError(ρ_sup, "test bound ρ_sup is negative"))
    end
    ρ_sup = clamp(ρ_sup, 0., Inf) #TODO same question here

    # Active set test

    active = Int64[]
    active_boundary = Int64[]
    for i in 1:n_ineq
        if c_ineq[i] >= -(β*ρ_sup)^σ - tol
            if i <= n_ineq_con
                push!(active, indices_to_constraints[i])
            else
                push!(active_boundary, indices_to_constraints[i])
            end
        end
    end

    return (active = active, active_boundary = active_boundary)
end