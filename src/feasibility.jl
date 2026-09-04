"""
    AbstractFeasibilityMethod

Base type for the different feasibilty methods implemented
"""
abstract type AbstractFeasibilityMethod end

"""
    L2FeasibilityMethod <: AbstractFeasibilityMethod

Composite type for the L2 feasibility method, which writes the feasibility problem in an NLS form then solves it
Contains two fields:
-`solver`: solver used for the NLS problem (That follows the JSOSolvers interface)
-`solver_options`: additional options to be passed to the solver
"""
struct L2FeasibilityMethod <: AbstractFeasibilityMethod
    solver
    solver_options::NamedTuple
end

"""
    L2FeasibilityMethod(linear_program_solver; kwargs...)

Create an L2FeasibilityMethod where the solver must be provided and solver_options can given specified as a keyword argument, the default value is `solver_options`= () i.e. none
"""
L2FeasibilityMethod(solver; solver_options = ()) = L2FeasibilityMethod(solver, solver_options)

"""
    L1FeasibilityMethod <: AbstractFeasibilityMethod

Composite type for the L1 feasibility method, which writes the feasibility problem with an elastic L1 formulation, then solves it
Contains two fields:
-`solver`: Solver used for the NLP problem (That follows the JSOSolvers interface)
-`solver_options`: additional options to be passed to the solver
"""
struct L1FeasibilityMethod <: AbstractFeasibilityMethod
    solver
    solver_options::NamedTuple
end

"""
    L1FeasibilityMethod(linear_program_solver; kwargs...)

Create an L1FeasibilityMethod where the solver must be provided and solver_options can given specified as a keyword argument, the default value is `solver_options`= () i.e. none
"""
L1FeasibilityMethod(solver; solver_options = ()) = L1FeasibilityMethod(solver, solver_options)


"""
    check_feasibility(nlp, method::L2FeasibilityMethod)

Write the L2 feasibility problem of nlp in an NLS form using FeasibilityResidual from NLPModelsModifiers, then solves the problem using the solver provided in `method`.

Return
what the solver returns (i.e. the associated execution stats)
"""
function check_feasibility(nlp, method::L2FeasibilityMethod)
    nls = NLPModelsModifiers.FeasibilityResidual(nlp)

    stats = method.solver(nls; method.solver_options...)

    return stats
end


"""
    check_feasibility(nlp, method::L1FeasibilityMethod)

Write the elastic L1 feasibility problem of nlp using FeasL1Model.jl , then solves the problem using the solver provided in `method`.

Return
what the solver returns (i.e. the associated execution stats)
"""
function check_feasibility(nlp, method::L1FeasibilityMethod)
    feas_l1_problem = FeasL1Model(nlp)

    stats = method.solver(feas_l1_problem; method.solver_options...)

    return stats
end