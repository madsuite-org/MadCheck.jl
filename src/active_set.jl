"""
    AbstractActiveSetMethod

Base type for the different active set methods implemented
"""

abstract type AbstractActiveSetMethod end

"""
    ActiveMethodSimple <: AbstractActiveSetMethod

composite type for the naive method consisting of runing the test c(x) = 0, 
contains one parameter
    `tol`: tolerance of the method
"""

struct ActiveMethodSimple <: AbstractActiveSetMethod
    tol::Float64
end

"""
    ActiveMethodSimple(; kwargs...)

Creates an ActiveMethodeSimple with `tol` = 1e-8.

"""

ActiveMethodSimple(;tol = 1e-8) = ActiveMethodSimple(tol)

"""
    ActiveMethodLP_P <: AbstractActiveSetMethod

composite type for the primal trust region method presented in [OberlinandWright-2006](@cite)
Contains the following fields:
-`silent`: bool setting the optimization solver used to silent (true) or not (false)
-`tol`: tolerance of the method
-`linear_program_solver`: linear solver used for the LP subproblem solved in the method
-`solver_attributes`: additional JuMP options to be given to the solver
-`ν`: penalty parameter, see [OberlinandWright-2006](@cite)
-`Δ`: trust region parameter, see [OberlinandWright-2006](@cite)

# Reference
[OberlinandWright-2006] Oberlin and Wright - 2006 - Active Set Identification in Nonlinear Programming

"""

struct ActiveMethodLP_P <: AbstractActiveSetMethod
    silent::Bool
    tol::Float64
    linear_program_solver::DataType
    solver_attributes::Tuple{Vararg{Pair}}
    ν::Float64
    Δ::Float64
end

"""
    ActiveMethodeMethodLP_P(linear_program_solver; kwargs...)

Create an ActiveMethodeMethodLPEC where all fields can be specified as keyword arguments, a linear program solver must be provided.
The following default values are set
-`silent`: true
-`tol`: 1e-8
-`solver_attributes`: () i.e. none
-`ν`: NaN which is replaced by 2*max(norm(λ, Inf), 1) when the associated find_active() is used, with λ the multipliers of the problem
-`Δ`:-NaN which is replaced by 4/(m+n) when the associated find_active() is used, with n the number of variables and m of constraints
"""

ActiveMethodLP_P(linear_program_solver::DataType ;silent = true, tol = 1e-8, solver_attributes = (), ν = NaN, Δ = NaN) = ActiveMethodLP_P(silent, tol, linear_program_solver, solver_attributes, ν,Δ)


"""
    ActiveMethodLPEC <: AbstractActiveSetMethod

composite type for the method based on the primal-dual estimate presented in [OberlinandWright-2006](@cite)
Contains the following fields:
-`silent`: bool setting the optimization solver used to silent (true) or not (false)
-`tol`: tolerance of the method
-`solver`: solver used for the MILP subproblem solved in the method
-`solver_attributes`: additional JuMP options to be given to the solver
-`M`: big M constant, see [OberlinandWright-2006](@cite)
-`β`: positive test parameter, see [OberlinandWright-2006](@cite)
-`σ`: test parameter in (0,1), see [OberlinandWright-2006](@cite)

# Reference
[OberlinandWright-2006] Oberlin and Wright - 2006 - Active Set Identification in Nonlinear Programming

"""

struct ActiveMethodLPEC <: AbstractActiveSetMethod
    silent::Bool
    tol::Float64
    solver::DataType
    solver_attributes::Tuple{Vararg{Pair}}
    M::Float64
    β::Float64
    σ::Float64
end

"""
    ActiveMethodeMethodLPEC(solver; kwargs...)

Create an ActiveMethodeMethodLPEC where all fields can be specified as keyword arguments, a solver must be provided.
The following default values are set:
-`silent`: true
-`tol`: 1e-8
-`solver_attributes`: () i.e. none
-`M`: NaN wich is replaced by 10*max(norm(λ_ineq, Inf), norm(c_ineq, Inf)) when the associated find_active() is used, with c_ineq is the inequality constraints and λ_ineq their associated multipliers
-`β`: NaN wich is replaced by 1/(m+n) when the associated find_active() is used, with n the number of variables and m of constraints
-`σ`: 0.75
"""

ActiveMethodLPEC(solver::DataType ;silent = true, tol = 1e-8, solver_attributes = (), M = NaN, β = NaN, σ = 0.75) = ActiveMethodLPEC(silent, tol, solver, solver_attributes, M, β, σ)

"""
    ActiveMethodLPEC_A <: AbstractActiveSetMethod

composite type for the linear programming approximation to the method based on the primal-dual estimate presented in [OberlinandWright-2006](@cite)
Contains the following fields:
-`silent`: bool setting the optimization solver used to silent (true) or not (false)
-`tol`: tolerance of the method
-`linear_program_solver`: linear solver used for the LP subproblem solved in the method
-`solver_attributes`: additional JuMP options to be given to the solver
-`β`: positive test parameter, see [OberlinandWright-2006](@cite)
-`σ`: test parameter in (0,1), see [OberlinandWright-2006](@cite)

# Reference
[OberlinandWright-2006] Oberlin and Wright - 2006 - Active Set Identification in Nonlinear Programming

"""

struct ActiveMethodLPEC_A <: AbstractActiveSetMethod
    silent::Bool
    tol::Float64
    linear_program_solver::DataType
    solver_attributes::Tuple{Vararg{Pair}}
    β::Float64
    σ::Float64
end

"""
ActiveMethodeMethodLPEC_A(linear_program_solver; kwargs...)

Create an ActiveMethodeMethodLPEC_A where all fields can be specified as keyword arguments, linear_program_solver must be provided.
The following default values are set:
-`silent`: true
-`tol`: 1e-8
-`solver_attributes`: () i.e. none
-`β`: NaN wich is replaced by 1/(m+n) when the associated find_active() is used, with n the number of variables and m of constraints
-`σ`: 0.90
"""

ActiveMethodLPEC_A(linear_program_solver::DataType ;silent = true, tol = 1e-8, solver_attributes = (), β = NaN, σ = 0.90) = ActiveMethodLPEC_A(silent, tol, linear_program_solver, solver_attributes, β, σ)


"""
    find_active(nlp, results, method::ActiveMethodeNaive)

Finds the active set at results using the simple test c(x) = 0

Returns named tuple with 2 attributes:
- `active` : active set of usual constraints found
- `active_boundary` : active set of boundary constraints found

"""

function find_active(nlp, results, method::ActiveMethodSimple)
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
    find_active(nlp, results, method::ActiveMethodeMethodLP_P)

Implement the primal trust region method presented in [OberlinandWright-2006](@cite). For finding the active set at a solution with results being a point near the solution.

Returns named tuple with 2 attributes:
- `active` : active set of usual constraints found
- `active_boundary` : active set of boundary constraints found

# Reference
[OberlinandWright-2006] Oberlin and Wright - 2006 - Active Set Identification in Nonlinear Programming
"""


function find_active(nlp, results, method::ActiveMethodLP_P)
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

    # we double the range constraints and use the convention c(x) <= 0
    c_ineq = vcat(constraints[allupp_con] - ucon[allupp_con], 
             lcon[alllow_con] - constraints[alllow_con], 
             x[allupp_var] - uvar[allupp_var],
             lvar[alllow_var] - x[alllow_var]
    )
    indices_to_constraints = vcat(allupp_con, alllow_con, allupp_var, alllow_var)
    n_ineq = length(c_ineq)
    J_ineq = vcat(Jac[allupp_con,:], -Jac[alllow_con,:], spdiagm(ones(n))[allupp_var,:], -spdiagm(ones(n))[alllow_var,:])


    # Parameter calculation
    if isnan(method.ν)
        ν = 2 * max(norm(y,Inf), norm(zl, Inf), norm(zu, Inf), 1)
    else
        ν = method.ν
    end
    if isnan(method.Δ)
        Δ = 4/n
    else
        Δ = method.Δ
    end

    # Error testing
    if Δ <= 0
        throw(DomainError(Δ, "Δ must be positive"))
    end
    if ν <= 0
        throw(DomainError(ν, "ν must be positive"))
    end

    # Sub problem resolution
    model = Model(optimizer_with_attributes(method.linear_program_solver,  method.solver_attributes...))

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

        @constraint(model, J_ineq * d + c_ineq .<= r)

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
        if dot(J_ineq[i, :], d_sol) + c_ineq[i] >= -tol
            if i <= length(allupp_con) + length(alllow_con)
                !(i in active) && push!(active, indices_to_constraints[i])
            else
                !(i in active_boundary) && push!(active_boundary, indices_to_constraints[i])
            end
        end
    end


    return (active = active, active_boundary = active_boundary)
end

"""
    find_active(nlp, results, method::ActiveMethodeMethodLPEC)


Implements the method based on the primal-dual estimate presented in [OberlinandWright-2006](@cite). For finding the active set at a solution with results being a point near the solution.

Returns named tuple with 2 attributes:
- `active` : active set of usual constraints found
- `active_boundary` : active set of boundary constraints found

# Reference
[OberlinandWright-2006] Oberlin and Wright - 2006 - Active Set Identification in Nonlinear Programming
"""

function find_active(nlp, results, method::ActiveMethodLPEC)
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

    # we double the range constraints and use the convention c(x) <= 0
    c_ineq = vcat(constraints[allupp_con] - ucon[allupp_con], 
             lcon[alllow_con] - constraints[alllow_con], 
             x[allupp_var] - uvar[allupp_var],
             lvar[alllow_var] - x[alllow_var]
    )
    indices_to_constraints = vcat(allupp_con, alllow_con, allupp_var, alllow_var)
    n_ineq = length(c_ineq)
    J_ineq = vcat(Jac[allupp_con,:], -Jac[alllow_con,:], spdiagm(ones(n))[allupp_var,:], -spdiagm(ones(n))[alllow_var,:])
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
    model = Model(optimizer_with_attributes(method.solver,  method.solver_attributes...))

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
            if i <= length(allupp_con) + length(alllow_con)
                !(i in active) && push!(active, indices_to_constraints[i])
            else
                !(i in active_boundary) && push!(active_boundary, indices_to_constraints[i])
            end
        end
    end

    return (active = active, active_boundary = active_boundary)
end


"""
    find_active(nlp, results, method::ActiveMethodeMethodLPEC_A)

Implements the linear programming approximation of the method based on the primal-dual estimate presented in [OberlinandWright-2006](@cite). For finding the active set at a solution with results being a point near the solution.

Returns named tuple with 2 attributes:
- `active` : active set of usual constraints found
- `active_boundary` : active set of boundary constraints found

# Reference
[OberlinandWright-2006] Oberlin and Wright - 2006 - Active Set Identification in Nonlinear Programming
"""

function find_active(nlp, results, method::ActiveMethodLPEC_A)
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

    # we double the range constraints and use the convention c(x) <= 0
    c_ineq = vcat(constraints[allupp_con] - ucon[allupp_con], 
             lcon[alllow_con] - constraints[alllow_con], 
             x[allupp_var] - uvar[allupp_var],
             lvar[alllow_var] - x[alllow_var]
    )
    indices_to_constraints = vcat(allupp_con, alllow_con, allupp_var, alllow_var)
    n_ineq = length(c_ineq)
    neg_idx = findall(x -> x<-tol, c_ineq)
    pos_idx = setdiff(1:n_ineq, neg_idx)
    J_ineq = vcat(Jac[allupp_con,:], -Jac[alllow_con,:], spdiagm(ones(n))[allupp_var,:], -spdiagm(ones(n))[alllow_var,:])
    Jt_ineq = transpose(J_ineq)

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
    model = Model(optimizer_with_attributes(method.linear_program_solver,  method.solver_attributes...))

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
            norm(Jt_ineq * λ_ineq_sol + Jt_eq * λ_eq_sol + g, 1)


    if ρ_sup < -tol
        trow(DomainError(ρ_sup, "test bound ρ_sup is negative"))
    end
    ρ_sup = clamp(ρ_sup, 0., Inf) #TODO same question here

    # Active set test

    active = Int64[]
    active_boundary = Int64[]
    for i in 1:n_ineq
        if c_ineq[i] >= -(β*ρ_sup)^σ - tol
            if i <= length(allupp_con) + length(alllow_con)
                !(i in active) && push!(active, indices_to_constraints[i])
            else
                !(i in active_boundary) && push!(active_boundary, indices_to_constraints[i])
            end
        end
    end

    return (active = active, active_boundary = active_boundary)
end