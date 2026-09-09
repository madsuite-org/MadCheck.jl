module MadCheck

using LinearAlgebra
using SparseArrays
import NLPModels
using JuMP
using NLPModelsModifiers
using Arpack

include("solution.jl")
include("active_set.jl")
include("jacobian_degeneracy.jl")
include("MFCQ_direction_condition.jl")
include("constraint_qualification.jl")
include("SOSC.jl")
include("Models/feas_l1_model.jl")
include("feasibility.jl")

end # module MadCheck
