"""
    AbstractFeasibilityMethod

Base type for the different feasibilty methods implemented
"""
abstract type AbstractFeasibilityMethod end

"""
    L2FeasibilyMethod <: AbstractFeasibilityMethod

Composite type for the L2 feasibility method, which writes the feasibility problem in an NLS form then solves it
Contains two fields:
-`solver`: solver used for the NLS problem (That follows the JSOSolvers interface)
-`solver_options`: additional options to be passed to the solver
"""
struct L2FeasibilyMethod <: AbstractFeasibilityMethod
    solver
    solver_options::Tuple{Vararg{Pair}}
end

"""
    L2FeasibilyMethod(linear_program_solver; kwargs...)

Create an L2FeasibilyMethod where the solver must be provided and solver_options can given specified as a keyword argument, the default value is `solver_options`= () i.e. none
"""
L2FeasibilyMethod(solver; solver_options = ()) = L2FeasibilyMethod(solver, solver_options)

"""
    L1FeasibilyMethod <: AbstractFeasibilityMethod

Composite type for the L1 feasibility method, which writes the feasibility problem with an elastic L1 formulation, then solves it
Contains two fields:
-`solver`: Solver used for the NLP problem (That follows the JSOSolvers interface)
-`solver_options`: additional options to be passed to the solver
"""
struct L1FeasibilyMethod <: AbstractFeasibilityMethod
    solver
    solver_options::Tuple{Vararg{Pair}}
end

"""
    L1FeasibilyMethod(linear_program_solver; kwargs...)

Create an L1FeasibilyMethod where the solver must be provided and solver_options can given specified as a keyword argument, the default value is `solver_options`= () i.e. none
"""
L1FeasibilyMethod(solver; solver_options = ()) = L1FeasibilyMethod(solver, solver_options)


"""
    check_feasibility(nlp, method::L2FeasibilyMethod)

Write the L2 feasibility problem of nlp in an NLS form using FeasibilityResidual from NLPModelsModifiers, then solves the problem using the solver provided in `method`.

Return
what the solver returns (i.e. the associated execution stats)
"""
function check_feasibility(nlp, method::L2FeasibilyMethod)
    nls = FeasibilityResidual(nlp)

    stats = method.solver(nls; method.solver_options...)

    return stats
end


"""
    check_feasibility(nlp, method::L1FeasibilyMethod)

Write the elastic L1 feasibility problem of nlp using FeasL1Model.jl , then solves the problem using the solver provided in `method`.

Return
what the solver returns (i.e. the associated execution stats)
"""
function check_feasibility(nlp, method::L1FeasibilyMethod)
    feas_l1_problem = FeasL1Model(nlp)

    stats = method.solver(feas_l1_problem; method.solver_options...)

    return stats
end