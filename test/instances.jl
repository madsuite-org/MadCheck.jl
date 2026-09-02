
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

function degenerate_bt4_model()
    x0 = [
        4.0382,
        -2.9470,
        -0.09115,
    ]

    model = JuMP.Model()
    @variable(model, x[i=1:3], start=x0[i])
    @objective(model, Min, x[1] - x[2] + x[2]^3)
    @expression(model, circ, -25 + x[1]^2 + x[2]^2 + x[3]^2)
    @constraint(model, circ == 0)
    @constraint(model, x[1] + x[2] + x[3] == 1)
    # Degenerate constraint
    @constraint(model, circ - circ^2 == 0)
    return MathOptNLPModel(model)
end

function degen_30303_model()
    lb = [0.0, 0.0, -Inf]
    model = Model()
    @variable(model,  x[i=1:3] >= lb[i])
    @objective(model, Min, 0.1*x[1] + 0.1*x[2] + 0.8*x[3])
    @constraint(model, -x[1] - x[2] - x[3] + 1 >= 0)
    @constraint(model, -x[1] - x[2] + x[3] + 1 >= 0)
    @constraint(model, -x[1]*x[2] - (x[1] + x[2] + x[3] - 1)*(x[1] + x[2] - x[3] - 1) >= 0)
    return MathOptNLPModel(model)
end

function degen_scs_1_model()
    model = Model()
    @variable(model, x >= 0)
    @objective(model, Min, x^2)
    return MathOptNLPModel(model)
end

function degen_scs_2_model()
    model = Model()
    @variable(model, x)
    @objective(model, Min, x^2)
    @constraint(model, -x <= 0.0)
    return MathOptNLPModel(model)
end
