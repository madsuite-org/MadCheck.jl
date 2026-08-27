# TODO implement Degen Hunter, Qr and maybe dulmage-mendelsohn

"""
    AbstractDegenJacMethod

Base type for the different methods implemented for determining sets of degenerate columns in a given Jacobian
"""
abstract type AbstractDegenJacMethod end

"""
    DegenJacSVD <: AbstractDegenJacMethod

Composite type for the method based on the singular value decomposition, 
contains one parameter
    `tol`: tolerance of the method
"""
struct DegenJacSVD <: AbstractDegenJacMethod
    tol::Float64
end


"""
    DegenJacSVD(; kwargs...)

Creates an DegenJacSVD with `tol` = 1e-8.

"""
DegenJacSVD(; tol = 1e-8) = DegenJacSVD(tol)

"""
    find_degenerate(Jac, method::DegenJacSVD)

Finds the dependent columns in the matrice Jac using it's singular value decomposition.

Return
list of vectors of indices corresponding to dependent sets of columns of Jac
"""
function find_degenerate(Jac, method::DegenJacSVD)
    n_jac, n = size(Jac)
    tol = method.tol

    if n_jac > n        # If n_cons_eq > n we need to use the full svd to have access to all the columns of U to then run the test on all of them
        J_svd = svd(Array(Jac), full = true)
    else
        J_svd = svd(Array(Jac))
    end
    U = J_svd.U
    S = J_svd.S

    r = min(n_jac, n)
    # Rank
    while r >= 1 && S[r] <= tol
        r-=1
    end

    list_degen_colums = []

    for u in eachcol(U[:, (r+1):n_jac])
        degen_columns = findall(x -> abs(x)>tol, u)
        if !(Set(degen_columns) in Set.(list_degen_colums))
            push!(list_degen_colums, degen_columns)
        end
    end

    return list_degen_colums
end
