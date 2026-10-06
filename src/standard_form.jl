"""
    StandardForm

Constraints of an `NLPModel` evaluated at a point `x`, rewritten in a common
form shared by the active set methods and the constraint qualification checks.

Equality constraints (`jfix` then `ifix`) are stored as `c_eq = 0` with Jacobian `J_eq`.

Inequality constraints are stored in the order `jrng, jupp, jlow, irng, iupp, ilow`, in two forms:
- raw form: `low_ineq ≤ val_ineq ≤ upp_ineq`, with Jacobian `J_ineq_raw`;
- with the sign convention c(x) ≤ 0: `c_ineq ≤ 0`, with Jacobian `J_ineq`. Range constraints are oriented towards their closest bound.

Contains the following fields:
-`nvar`, `ncon`: number of variables and constraints of the problem
-`jfix`, `ifix`: indices of the equality constraints and of the fixed variables
-`jac`: Jacobian of the constraints at `x`
-`sign_con`, `sign_var`: orientation of each constraint/variable, `1` if closer to its upper bound, `-1` if closer to its lower bound
-`c_eq`, `J_eq`: equality constraints and their Jacobian
-`idx_ineq`: maps each inequality row to its constraint or variable index
-`is_con_ineq`: `true` if the inequality row is a constraint, `false` if it is a bound
-`val_ineq`, `low_ineq`, `upp_ineq`, `J_ineq_raw`: raw form of the inequality constraints
-`c_ineq`, `J_ineq`: inequality constraints and their Jacobian with the convention c(x) ≤ 0
"""
struct StandardForm{T}
    nvar::Int
    ncon::Int
    jfix::Vector{Int}
    ifix::Vector{Int}
    jac::SparseMatrixCSC{T,Int}
    sign_con::Vector{Int}
    sign_var::Vector{Int}
    c_eq::Vector{T}
    J_eq::SparseMatrixCSC{T,Int}
    idx_ineq::Vector{Int}
    is_con_ineq::BitVector
    val_ineq::Vector{T}
    low_ineq::Vector{T}
    upp_ineq::Vector{T}
    J_ineq_raw::SparseMatrixCSC{T,Int}
    c_ineq::Vector{T}
    J_ineq::SparseMatrixCSC{T,Int}
end

# Rows of the n×n identity matrix selected by idx
_selection_rows(idx, n, T = Float64) = sparse(1:length(idx), idx, ones(T, length(idx)), length(idx), n)

# -1 if val is closer to its lower bound, 1 otherwise
_orientation(val, low, upp) = (upp - val) > (val - low) ? -1 : 1

"""
    standard_form(nlp, x)

Evaluate the constraints of `nlp` and their Jacobian at `x` and store them in a [`StandardForm`](@ref).
"""
function standard_form(nlp::NLPModels.AbstractNLPModel{T}, x::AbstractVector) where {T}
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    meta = nlp.meta

    lvar, uvar = meta.lvar, meta.uvar
    lcon, ucon = meta.lcon, meta.ucon

    Ji, Jj = NLPModels.jac_structure(nlp)
    Jx = NLPModels.jac_coord(nlp, x)
    jac = sparse(Ji, Jj, Jx, m, n)

    cons = NLPModels.cons(nlp, x)

    sign_con = _orientation.(cons, lcon, ucon)
    sign_var = _orientation.(x, lvar, uvar)

    # Equality constraints
    c_eq = vcat(cons[meta.jfix] .- ucon[meta.jfix], x[meta.ifix] .- uvar[meta.ifix])
    J_eq = vcat(jac[meta.jfix, :], _selection_rows(meta.ifix, n, T))

    # Inequality constraints
    jcon = vcat(meta.jrng, meta.jupp, meta.jlow)
    ivar = vcat(meta.irng, meta.iupp, meta.ilow)

    idx_ineq = vcat(jcon, ivar)
    is_con_ineq = vcat(trues(length(jcon)), falses(length(ivar)))

    val_ineq = vcat(cons[jcon], x[ivar])
    low_ineq = vcat(lcon[jcon], lvar[ivar])
    upp_ineq = vcat(ucon[jcon], uvar[ivar])
    J_ineq_raw = vcat(jac[jcon, :], _selection_rows(ivar, n, T))

    sign_ineq = vcat(sign_con[jcon], sign_var[ivar])
    c_ineq = T[s > 0 ? v - u : l - v for (s, v, l, u) in zip(sign_ineq, val_ineq, low_ineq, upp_ineq)]
    J_ineq = J_ineq_raw .* sign_ineq

    return StandardForm{T}(
        n, m, meta.jfix, meta.ifix, jac, sign_con, sign_var,
        c_eq, J_eq,
        idx_ineq, is_con_ineq, val_ineq, low_ineq, upp_ineq, J_ineq_raw,
        c_ineq, J_ineq,
    )
end

"""
    split_cons_bounds(rows, indices, is_con)

Map the row numbers `rows` to the problem indices `indices[rows]`, separating constraints from bounds using `is_con`.

Return
A named tuple with 2 fields `constraints` and `bounds`.
"""
function split_cons_bounds(rows, indices, is_con)
    cons = Int[indices[i] for i in rows if is_con[i]]
    bounds = Int[indices[i] for i in rows if !is_con[i]]
    return (constraints = cons, bounds = bounds)
end
