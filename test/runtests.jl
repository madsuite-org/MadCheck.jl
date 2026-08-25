
using Test
using MadNLP
using NLPModels
using MadCheck
using HiGHS

include("instances.jl")

@testset "Test KKT residual checker" begin
    nlp = hs15_model()
    results = madnlp(nlp; print_level=MadNLP.ERROR)

    chk = MadCheck.check_kkt_residuals(nlp, results)
    @test chk.inf_du <= 1e-8
    @test chk.inf_pr <= 1e-8
    @test chk.inf_cc_var <= 1e-8
    @test chk.inf_cc_con <= 1e-8
end

@testset "Test active set detection" begin
    nlp = hs15_model()
    results = madnlp(nlp; print_level=MadNLP.ERROR)
    
    n = NLPModels.get_nvar(nlp)

    new_point = results
    max_dist = 0
    for i in 1:n
        noise = 2*rand() - 1
        new_point.solution[i] += (max_dist*noise)/n
    end

    active_solution = MadCheck.find_active(nlp, results,MadCheck.ActiveMethodSimple())

    active_LP_P = MadCheck.find_active(nlp, new_point, MadCheck.ActiveMethodLP_P(HiGHS.Optimizer; Δ = 4 * 1e-8/n))
    active_LPEC = MadCheck.find_active(nlp, new_point, MadCheck.ActiveMethodLPEC(HiGHS.Optimizer))
    active_LPEC_A = MadCheck.find_active(nlp, new_point, MadCheck.ActiveMethodLPEC_A(HiGHS.Optimizer))

    @test Set(active_LP_P.active) == Set(active_solution.active) && Set(active_LP_P.active_boundary) == Set(active_solution.active_boundary)
    @test Set(active_LPEC.active) == Set(active_solution.active) && Set(active_LPEC.active_boundary) == Set(active_solution.active_boundary)
    @test Set(active_LPEC_A.active) == Set(active_solution.active) && Set(active_LPEC_A.active_boundary) == Set(active_solution.active_boundary)
end