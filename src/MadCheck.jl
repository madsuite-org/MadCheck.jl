module MadCheck

using LinearAlgebra
using SparseArrays
import NLPModels
using JuMP

include("solution.jl")
include("constraint_qualification.jl")
include("active_set.jl")
include("feasibility.jl")

end # module MadCheck
