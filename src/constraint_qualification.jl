"""
    build_work_jacobian(form::StandardForm, active, active_boundary)
    build_work_jacobian(nlp, results, active, active_boundary)

Create the jacobian the method is working on based on what active set wants to be used (i.e. active = [] and active_boundary = [] for MFCQ). This jacobian can then be given to methods in jacobian_degeneracy and MFCQ_direction_condition for analysis.
We set the sign convention c(x) <= 0 for use in MFCQ.

Return
-`J_work`: Constructed jacobian
-`indices_to_constraints`: vector mapping the indices of the rows of `J_work` to the constraints indices, the vector uses the order jfix, ifix, active, active_boundary.
-`is_con`: vector telling whether each row of `J_work` is a constraint (true) or a bound (false).
"""
function build_work_jacobian(form::StandardForm, active, active_boundary)
    n = form.nvar

    J_work = vcat(
        form.jac[form.jfix, :],
        _selection_rows(form.ifix, n),
        form.jac[active, :] .* form.sign_con[active],
        _selection_rows(active_boundary, n) .* form.sign_var[active_boundary],
    )

    indices_to_constraints = vcat(form.jfix, form.ifix, active, active_boundary)
    is_con = vcat(
        trues(length(form.jfix)),
        falses(length(form.ifix)),
        trues(length(active)),
        falses(length(active_boundary)),
    )

    return J_work, indices_to_constraints, is_con
end

function build_work_jacobian(nlp, results, active, active_boundary)
    return build_work_jacobian(standard_form(nlp, results.solution), active, active_boundary)
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

    Jac, indices_to_constraints, is_con = build_work_jacobian(nlp, results, active, active_boundary)

    list_degen_cons = find_degenerate(Jac, degen_method)

    return [split_cons_bounds(degen_cons, indices_to_constraints, is_con) for degen_cons in list_degen_cons]
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

    if isempty(active) && isempty(active_boundary)
        @warn "No active constraints found; consider using check_LICQ instead"
    end

    form = standard_form(nlp, results.solution)
    n_eq = length(form.jfix) + length(form.ifix)

    Jac, _ = build_work_jacobian(form, active, active_boundary)

    direction_return = check_MFCQ_direction(Jac, n_eq, direction_method)    # TODO Smarter more general return ?

    Jac_eq, indices_to_constraints, is_con = build_work_jacobian(form, Int[], Int[])

    list_degen_cons = find_degenerate(Jac_eq, degen_method)

    dependent_constraints = [split_cons_bounds(degen_cons, indices_to_constraints, is_con) for degen_cons in list_degen_cons]

    return (direction_solution = direction_return, dependent_constraints = dependent_constraints) # TODO find a smarter way to transmit the direction information
end


"""
    check_SCS(nlp, results, tol)

Check the SCS condition at a given point for bound and generic constraints.

Takes the following arguments:
- `nlp::AbstractNLPModel`: non linear programming problem studied
- `results`: Point studied
- `tol::Float64`

Returns a named tuple with 2 fields: `constraints` and `bounds` containing the indices of the constraints/bounds violating SCS.
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
    check_dulmage_mendelsohn(nlp, results, active_method, tol)

Computes the equality and active jacobian at results, then creates the factor graph between constraints and variables by using the jacobian to determine if a given constraint depends on a given variable. Then applies the Dulmage-Mendelsohn decomposition [DulmageandMendelsohn-1958](@cite) to the factor graph, this method is based on the paper [Dulmage-Mendelsohn_method-2023](@cite). The given tolerance is used to decide if a value in the jacobian is zero.

Return
A named tuple with three fields `variables`, `constraints` and `bounds`,
Each are a named tuple with the following fields:
-`oc`: Overconstrained elements
-`uc`: Underconstrained elements
-`sq`: Well defined elements (square set)

References:
[DulmageandMendelsohn-1958] Dulmage and Mendelsohn - 1958 - Coverings of Bipartite Graphs

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

    Jac, indices_to_constraints, is_con = build_work_jacobian(nlp, results, active, active_boundary)
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

    dm_var, dm_con = dulmage_mendelsohn(A, B, E)

    dm_split = map(rows -> split_cons_bounds(rows, indices_to_constraints, is_con), dm_con)
    dm_con_rep = map(s -> s.constraints, dm_split)
    dm_bound_rep = map(s -> s.bounds, dm_split)

    return (variables = dm_var, constraints = dm_con_rep, bounds = dm_bound_rep)
end
