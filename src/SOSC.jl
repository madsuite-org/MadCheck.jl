"""
    check_SOSC(nlp, results, active_method, tol)

Compute the reduced hessian of the KKT system to check if it is positive definite and thus check if SOSC is verified.
    
Return
If the reduced hessian is empty, returns `Inf`, else returns the value of the smallest eigenvalue of the reduced hessian.
"""
function check_SOSC(nlp, results, active_method::AbstractActiveSetMethod, tol::Float64)
    # Jacobian
    active, active_boundary = find_active(nlp, results, active_method)

    Jac, indices_to_constraints = build_work_jacobian(nlp, results, active, active_boundary)
    n_jac, n = size(Jac)

    # Checks
    SCS = check_SCS(nlp, results, tol)
    if length(SCS.bounds) + length(SCS.constraints) != 0
        @warn "Strict complementarity is not verified"
    end

    QR = MadCheck.find_degenerate(Jac, DegenJacQR(tol))
    if length(QR)!=0 && length(QR[1]) != 0
        @warn "Jacobian is not full row rank"
    end

    # Hessian
    x = results.solution
    y = results.multipliers
    Wi, Wj = NLPModels.hess_structure(nlp)
    Wx = NLPModels.hess_coord(nlp, x, y)
    Hess = sparse(Wi, Wj, Wx, n, n)

    Z = LinearAlgebra.nullspace(Array(Jac))

    # Reduced Hessian
    H = Symmetric(Z' * Symmetric(Hess, :L) * Z)    # TODO clarify this step and make it more efficient

    if isempty(H)
        return Inf
    end

    if size(H)[1] == 1  # Arpack.eigs needs H to be at least 2x2
        return H[1,1]
    end

    return Arpack.eigs(H; nev = 1, which = :SR)[1][1]
end
