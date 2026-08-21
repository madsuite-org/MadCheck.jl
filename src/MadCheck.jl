module MadCheck

using LinearAlgebra
using SparseArrays
import NLPModels

include("solution.jl")
include("constraint_qualification.jl")
include("active_set.jl")
include("feasibility.jl")

end # module MadCheck
