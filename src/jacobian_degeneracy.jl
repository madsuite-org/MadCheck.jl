"""
    AbstractDegenJacMethod

Base type for the different methods implemented for determining sets of degenerate constraints in a given Jacobian
"""
abstract type AbstractDegenJacMethod end

"""
    DegenJacSVD <: AbstractDegenJacMethod

Composite type for the method based on the singular value decomposition,
contains one parameter
-`tol`: tolerance of the method
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
    DegenJacQR <: AbstractDegenJacMethod

Composite type for the method based on the rank revealing QR decomposition,
contains one parameter:
-`tol`: tolerance of the method
"""
struct DegenJacQR <: AbstractDegenJacMethod
    tol::Float64
end

"""
    DegenJacQR(; kwargs...)

Creates an DegenJacQR with `tol` = 1e-8.
"""
DegenJacQR(; tol = 1e-8) = DegenJacQR(tol)

"""
    DegenHunterJac <: AbstractDegenJacMethod

Composite type for the Degen Hunter method presented in [DowlingandBiegler-2015](@cite) that solves Big-M MILP problems to find irreducible degeneracy sets.
Contains the following parameters:
-`silent`: bool setting the optimization solver used to silent (true) or not (false)
-`tol`: tolerance of the method
-`solver`: solver used for the MILP subproblems solved in the method
-`solver_options`: additional JuMP options to be given to the solver)
-`M`: big M-value used for the MILP (see [DowlingandBiegler-2015](@cite))

Reference:
[DowlingandBiegler-2015] Dowling and Giegler - 2015 - Degeneracy Hunter: An Algorithm for Determining Irreducible Sets of Degenerate Constraints in Mathematical Programs
"""
struct DegenHunterJac <: AbstractDegenJacMethod
    silent::Bool
    tol::Float64
    solver::DataType
    solver_options::Tuple{Vararg{Pair}}
    M::Float64
end

"""
    DegenHunterJac(solver, M; kwargs...)

Create an DegenHunterJac where all fields can be specified as keyword arguments, a MILP solver and the big-M parameter must be provided.
The following default values are set
-`silent`: true
-`tol`: 1e-8
-`solver_options`: () i.e. none
"""
DegenHunterJac(solver::DataType, M;silent = true, tol = 1e-8, solver_options = ()) = DegenHunterJac(silent, tol, solver, solver_options, M)


"""
    DegenJacDulmageMendelsohn <: AbstractDegenJacMethod

Composite type for the method based on Dulmage-Mendelsohn decomposition as presented in [Dulmage-Mendelsohn_method-2023](@cite),
contains one parameter:
-`tol`: tolerance of the method

Reference:
[Dulmage-Mendelsohn_method-2023] Parker, Nicholson, Siirola, Biegler - 2023 - Applications of the Dulmage–Mendelsohn decomposition for debugging
nonlinear optimization problems
"""
struct DegenJacDulmageMendelsohn <: AbstractDegenJacMethod
    tol::Float64
end

"""
    DegenJacDulmageMendelsohn(; kwargs...)

Creates an DegenJacDulmageMendelsohn with `tol` = 1e-8.
"""
DegenJacDulmageMendelsohn(; tol = 1e-8) = DegenJacDulmageMendelsohn(tol)


"""
    find_degenerate(Jac, method::DegenJacSVD)

Finds the dependent rows in the matrice Jac using it's singular value decomposition.

Return
list of vectors of indices corresponding to dependent sets of rows of Jac
"""
function find_degenerate(Jac, method::DegenJacSVD)
    if iszero(Jac)
        return []
    end

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

    list_degen_rows = []

    for u in eachcol(U[:, (r+1):n_jac])
        degen_rows = findall(x -> abs(x)>tol, u)
        if !(Set(degen_rows) in Set.(list_degen_rows))
            push!(list_degen_rows, degen_rows)
        end
    end

    return list_degen_rows
end

"""
    find_degenerate(Jac, method::DegenJacQR)

finds m-r rows of that Jac that can be expressed as a linear combination of the others  (were r is the rank of Jac and m)

Return
list containing one vector: a set of row indices that are dependent on the other rows
"""
function find_degenerate(Jac, method::DegenJacQR)
    if iszero(Jac)
        return []
    end

    n_jac, n = size(Jac)
    tol = method.tol

    Jac_t = transpose(Jac)

    QR_Jac_t = qr(Jac_t)
    R = QR_Jac_t.R
    perm = QR_Jac_t.pcol
    s = size(QR_Jac_t.R)[1]

    zeros_of_R = findall(x->x<=tol, [norm(R[i,i]) for i in 1:s])

    r = s - length(zeros_of_R)  # rank

    degenerate_idx  = perm[vcat(zeros_of_R, [i for i in n+1:n_jac])]

    return [degenerate_idx]
end


"""
    find_degenerate(Jac, method::DegenHunterJac)


Implements the method presented in [DowlingandBiegler-2015](@cite) that solves Big-M MILP problems to find irreducible degeneracy sets.

Return:
list of vectors of indices where each vector corresponds to an irreducible degeneracy set.

Reference:
[DowlingandBiegler-2015] Dowling and Giegler - 2015 - Degeneracy Hunter: An Algorithm for Determining Irreducible Sets of Degenerate Constraints in Mathematical Programs
"""
function find_degenerate(Jac, method::DegenHunterJac)
    if iszero(Jac)
        return []
    end

    n_jac, n = size(Jac)
    Jac_t = transpose(Jac)

    degenerate_idx = find_degenerate(Jac, DegenJacQR(tol = method.tol))[1]

    M = method.M
    if M <= 0
        throw(DomainError(M, " M must be positive"))
    end

    irreducible_sets = []
    for j in degenerate_idx

        model = Model(optimizer_with_attributes(method.solver,  method.solver_options...))

        if method.silent
            set_silent(model)
        end

        @variable(model, y[1:n_jac], Bin)
        @variable(model, λ[1:n_jac])

        @objective(model, Min, sum(y))

        @constraint(model, λ[j] == 1)
        @constraint(model, [i = 1:n_jac], -y[i] * M<= λ[i])
        @constraint(model, [i = 1:n_jac], y[i] * M >= λ[i])
        @constraint(model, [i = 1:n], (Jac_t*λ)[i] == 0)

        optimize!(model)


        # TODO Find a way to give the user the real constraint, make this a warning?
        if termination_status(model) != OPTIMAL
            error("For line $j of the work Jacobian, MILP failed to converge : $(termination_status(model))")
        end

        degen_columns = findall(!iszero, value.(y))

        if !(Set(degen_columns) in Set.(irreducible_sets))
            push!(irreducible_sets, degen_columns)
        end
    end

    return irreducible_sets
end


"""
    find_degenerate(Jac, method::DegenJacDulmageMendelsohn)

Creates the factor graph between constraints and variables by using the jacobian to determine if a given constraint depends on a given variable. Then applies the Dulmage-Mendelsohn decomposition [DulmageandMendelsohn-1958](@cite) to the factor graph, this method is based on the paper [Dulmage-Mendelsohn_method-2023](@cite).

Return
The constraints in the overconstrained set (see [Dulmage-Mendelsohn_method-2023](@cite))

References:
[DulmageandMendelsohn-1958] Dulage and Mendelsohn - 1958 - Coverings of Bipartite Graphs

[Dulmage-Mendelsohn_method-2023] Parker, Nicholson, Siirola, Biegler - 2023 - Applications of the Dulmage–Mendelsohn decomposition for debugging
nonlinear optimization problems
"""
function find_degenerate(Jac, method::DegenJacDulmageMendelsohn)
    n_jac, n = size(Jac)
    tol = method.tol

    A = collect(1:n)
    B = collect(1:n_jac)
    E = Tuple{Int64, Int64}[]
    
    I, J , V = findnz(Jac)
    for k in 1:length(V)
        if abs(V[k]) > tol
            push!(E, (J[k], I[k]))
        end
    end
    
    dm_var, dm_con = dulmage_mendelsohn(A, B, E)

    return [dm_con.oc]
end
