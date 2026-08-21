
using JuMP
using NLPModelsJuMP

function hs15_model()
    model = Model()
    x0 = [-2, 1]
    uvar = [0.5, Inf]
    @variable(model, x[i = 1:2] <= uvar[i], start = x0[i])
    @objective(model, Min, 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2)
    @constraint(model, x[1] * x[2] - 1 ≥ 0)
    @constraint(model, x[1] + x[2]^2 ≥ 0)
    return MathOptNLPModel(model)
end
