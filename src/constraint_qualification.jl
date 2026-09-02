# TODO add MFCQ
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
    test_LICQ(nlp, results, active_method::AbstractActiveSetMethod, degen_method::AbstractDegenJacMethod)

Tests the LICQ condition at a given point by looking for dependent constraints in the set of active and equality constraints.
Takes the following arguments:
-`nlp`: non linear programming problem studied
-`results`: Point studied
-`active_method`: method used for finding the active set
-`degen_method`: method used for finding degenerate constraints in the set of equality and active constraints.

Returns a list of named tuples with 2 fields: `constraints` and `bounds` corresponding to a set of dependent constraints/bounds.
"""
function test_LICQ(
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
