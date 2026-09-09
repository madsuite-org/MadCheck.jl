"""
    build_work_jacobian(nlp, results, active, active_boundary)

Create the jacobian the method is working on based on what active set wants to be used (i.e. active = [] and active_boundary = [] for MFCQ). This jacobian can then be given to methods in jacobian_degeneracy and MFCQ_direction_condition for analysis.
We set the sign convention c(x) <= 0 for use in MFCQ.

Return
-`J_work`: Constructed jacobian
-`indices_to_constraints`: vector mapping the indices of the columns of `J_work` to the constraints indices, the vector used the order jfix, ifix, active, active_boundary which can be used to separate usual constraints and boundary constraints.
"""
function build_work_jacobian(nlp, results, active, active_boundary)
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)

    jfix = nlp.meta.jfix
    ifix = nlp.meta.ifix

    ucon = nlp.meta.ucon
    lcon = nlp.meta.lcon
    uvar = nlp.meta.lvar
    lvar = nlp.meta.uvar

    x = results.solution
    Ji, Jj = NLPModels.jac_structure(nlp)
    Jx = NLPModels.jac_coord(nlp, x)
    J = sparse(Ji, Jj, Jx, m, n)

    constraints = NLPModels.cons(nlp, x)

    sign = [(ucon[i] - constraints[i]) > (constraints[i] - lcon[i]) ? -1 : 1 for i in active]
    sign_boundary = [(uvar[i] - x[i]) > (x[i] - lvar[i]) ? -1 : 1 for i in active_boundary]

    J_work = vcat(
        J[jfix, :],
        spdiagm(ones(n))[ifix, :],
        J[active, :] .* sign,
        spdiagm(ones(n))[active_boundary, :] .* sign_boundary,
    )

    indices_to_constraints = vcat(jfix, ifix, active, active_boundary)

    return J_work, indices_to_constraints
end

"""
    check_LICQ(nlp, results, active_method::AbstractActiveSetMethod, degen_method::AbstractDegenJacMethod)

Check the LICQ condition at a given point by looking for dependent constraints in the set of active and equality constraints.
Takes the following arguments:
-`nlp`: non linear programming problem studied
-`results`: Point studied
-`active_method`: method used for finding the active set
-`degen_method`: method used for finding degenerate constraints in the set of equality and active constraints.

Returns a list of named tuples with 2 fields: `constraints` and `bounds` corresponding to a set of dependent constraints/bounds.
"""
function check_LICQ(
    nlp,
    results,
    active_method::AbstractActiveSetMethod,
    degen_method::AbstractDegenJacMethod,
)

    active, active_boundary = find_active(nlp, results, active_method)

    Jac, indices_to_constraints = build_work_jacobian(nlp, results, active, active_boundary)

    list_degen_cons = find_degenerate(Jac, degen_method)

    # Return
    rep = []

    n_jfix = length(nlp.meta.jfix)
    n_ifix = length(nlp.meta.ifix)
    n_a = length(active)

    for degen_cons in list_degen_cons
        cons = []
        bounds = []
        for i in degen_cons
            if i <= n_jfix || n_jfix + n_ifix + 1 <= i <= n_jfix + n_ifix + n_a
                push!(cons, indices_to_constraints[i])
            else
                push!(bounds, indices_to_constraints[i])
            end
        end

        push!(rep, (constraints = cons, bounds = bounds))
    end

    return rep
end


"""
    check_MFCQ(nlp, results, active_method::AbstractActiveSetMethod, degen_method::AbstractDegenJacMethod, direction_method::AbstractMFCQDirectionMethod)

Check the MFCQ condition at a given point by looking for dependent constraints in the set of equality constraints and by using direction_method to test the direction condition.
Takes the following arguments:
-`nlp`: non linear programming problem studied
-`results`: Point studied
-`active_method`: method used for finding the active set
-`degen_method`: method used for finding degenerate constraints in the set of equality constraints.
-`direction_method`: method used to test the direction condition


Returns a named tuples with 2 fields: 
-`direction_solution`: The return of the direction method employed
-`dependent_constraints`: list of named tuples with 2 fields: `constraints` and `bounds` corresponding to a set of dependent constraints/bounds.
"""
function check_MFCQ(
    nlp,
    results,
    active_method::AbstractActiveSetMethod,
    degen_method::AbstractDegenJacMethod,
    direction_method::AbstractMFCQDirectionMethod
)

    active, active_boundary = find_active(nlp, results, active_method)

    n_jfix = length(nlp.meta.jfix)
    n_ifix = length(nlp.meta.ifix)
    n_a = length(active)
    n_ab = length(active_boundary)

    if n_a + n_ab == 0
        @warn "No active constraints found; consider using check_LICQ instead"
    end
    
    Jac, indices_to_constraints_dir = build_work_jacobian(nlp, results, active, active_boundary)

    direction_return = check_MFCQ_direction(Jac, n_jfix + n_ifix, direction_method)    # TODO Smarter more general return ?
    
    Jac, indices_to_constraints_jac = build_work_jacobian(nlp, results, [], [])
    
    list_degen_cons = find_degenerate(Jac, degen_method)

    # Return
    rep = (direction_solution = direction_return, dependent_constraints = []) # TODO find a smarter way to transmit the direction information

    for degen_cons in list_degen_cons
        cons = []
        bounds = []
        for i in degen_cons
            if i <= n_jfix || n_jfix + n_ifix + 1 <= i <= n_jfix + n_ifix + n_a 
                push!(cons, indices_to_constraints_jac[i])
            else
                push!(bounds, indices_to_constraints_jac[i])
            end
        end

        push!(rep.dependent_constraints, (constraints = cons, bounds = bounds))
    end

    return rep
end


"""
    check_SCS(nlp, results, tol)

Check the SCS condition at a given point for bound and generic constraints.

Takes the following arguments:
- `nlp::AbstractNLPModel`: non linear programming problem studied
- `results`: Point studied
- `tol::Float64`

Returns a list of named tuples with 2 fields: `constraints` and `bounds` corresponding to a set of dependent constraints/bounds.
"""
function check_SCS(nlp::NLPModels.AbstractNLPModel, results, tol::Float64)
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    x = results.solution
    c = results.constraints
    y = results.multipliers
    zl = results.multipliers_L
    zu = results.multipliers_U
    xl, xu = NLPModels.get_lvar(nlp), NLPModels.get_uvar(nlp)
    cl, cu = NLPModels.get_lcon(nlp), NLPModels.get_ucon(nlp)

    # Check bound constraints
    index_bounds = Int[]
    for i = 1:n
        if min(x[i] - xl[i], zl[i]) >= tol || min(xu[i] - x[i], zu[i]) >= tol
            push!(index_bounds, i)
        end
    end

    # Check generic constraints
    index_constraints = Int[]
    for i = 1:m
        if (cl[i] < cu[i]) &&
           (min(c[i] - cl[i], -y[i]) >= tol || min(cu[i] - c[i], y[i]) >= tol)
            push!(index_constraints, i)
        end
    end

    return (constraints = index_constraints, bounds = index_bounds)
end

"""
    check_dulmage_mendelsohn(nlp, results, active_method)

Computes the equality and active jacobian at results, then creates the factor graph between constraints and variables by using the jacobian to determine if a given constraint depends on a given variable. Then applies the Dulmage-Mendelsohn decomposition [DulmageandMendelsohn-1958](@cite) to the factor graph, this method is based on the paper [Dulmage-Mendelsohn_method-2023](@cite). The given tolerance is used to decide if a value in the jacobian is zero.

Return
A named tuple with three fields `variables`, `constraints` and `bounds`,
Each are a named tuple with the folowing fields:
-`oc`: Overconstrained elements
-`uc`: Underconstrained elements
-`sq`: Well defined elements (square set)

References:
[DulmageandMendelsohn-1958] Dulage and Mendelsohn - 1958 - Coverings of Bipartite Graphs

[Dulmage-Mendelsohn_method-2023] Parker, Nicholson, Siirola, Biegler - 2023 - Applications of the Dulmage–Mendelsohn decomposition for debugging
nonlinear optimization problems
"""
function check_dulmage_mendelsohn(
    nlp,
    results,
    active_method::AbstractActiveSetMethod,
    tol::Float64,
)
    active, active_boundary = find_active(nlp, results, active_method)

    Jac, indices_to_constraints = build_work_jacobian(nlp, results, active, active_boundary)
    n_jac, n = size(Jac)
    A = collect(1:n)
    B = collect(1:n_jac)
    E = Tuple{Int64,Int64}[]

    I, J , V = findnz(Jac)
    for k in 1:length(V)
        if abs(V[k]) > tol
            push!(E, (J[k], I[k]))
        end
    end

    n_jfix = length(nlp.meta.jfix)
    n_ifix = length(nlp.meta.ifix)
    n_a = length(active)

    dm_var, dm_con = dulmage_mendelsohn(A, B, E)

    iscon(x) = x <= n_jfix || n_jfix + n_ifix + 1 <= x <= n_jfix + n_ifix + n_a

    dm_con_rep = (
        oc = indices_to_constraints[filter(iscon, dm_con.oc)],
        uc = indices_to_constraints[filter(iscon, dm_con.uc)],
        sq = indices_to_constraints[filter(iscon, dm_con.sq)],
    )

    dm_bound_rep = (
        oc = indices_to_constraints[filter(!iscon, dm_con.oc)],
        uc = indices_to_constraints[filter(!iscon, dm_con.uc)],
        sq = indices_to_constraints[filter(!iscon, dm_con.sq)],
    )

    return (variables = dm_var, constraints = dm_con_rep, bounds = dm_bound_rep)
end
