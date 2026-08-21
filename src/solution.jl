
"""
    check_kkt_residuals(nlp, results)

Check if the KKT conditions are satisfied in the solution stored inside `results`.

Return a named-tuple with four attributes:
- `inf_du`: dual infeasibility
- `inf_pr`: primal infeasibility
- `inf_cc_var`: violation in the bounds' complementarity conditions
- `inf_cc_con`: violation in the generic inequality constraints' complementarity conditions

"""
function check_kkt_residuals(nlp::NLPModels.AbstractNLPModel, results)
    x = results.solution
    y = results.multipliers
    zl = results.multipliers_L
    zu = results.multipliers_U

    g = NLPModels.grad(nlp, x)
    jtv = NLPModels.jtprod(nlp, x, y)

    # Dual infeasibility.
    inf_du = norm(g .+ jtv .- zl .+ zu)

    # Primal infeasibility.
    c = NLPModels.cons(nlp, x)
    cl = NLPModels.get_lcon(nlp)
    cu = NLPModels.get_ucon(nlp)
    inf_pr = norm(c .- clamp.(c, cl, cu), Inf)

    # Complementarity conditions.
    xl = NLPModels.get_lvar(nlp)
    xu = NLPModels.get_uvar(nlp)
    inf_cc_var = norm(max.(min.(xu .- x, zu), min.(x .- xl, zl)), Inf)

    ind_ineq = findall(cl .< cu)
    h = c[ind_ineq]
    hl = cl[ind_ineq]
    hu = cu[ind_ineq]
    vl = min.(0.0, .-y[ind_ineq])
    vu = min.(0.0,   y[ind_ineq])
    inf_cc_con = norm(max.(min.(hu .- h, vu), min.(h .- hl, vl)), Inf)

    return (
        inf_pr=inf_pr,
        inf_du=inf_du,
        inf_cc_var=inf_cc_var,
        inf_cc_con=inf_cc_con,
    )
end

