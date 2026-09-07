module MadCheck

using LinearAlgebra
using SparseArrays
using Graphs
using BipartiteMatching
import NLPModels
using JuMP

include("solution.jl")
include("active_set.jl")
include("dulmage_mendelsohn_algorithm.jl")
include("jacobian_degeneracy.jl")
include("constraint_qualification.jl")
include("feasibility.jl")

end # module MadCheck
