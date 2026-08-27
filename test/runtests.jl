
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

@testset "Test LICQ verification" begin
    nlp = degenerate_bt4_model()
    results = madnlp(nlp; print_level=MadNLP.ERROR)
    degen_cons_bt4 = MadCheck.test_LICQ(nlp, results, MadCheck.ApproximatePrimalDualActiveSetLPEC(HiGHS.Optimizer), MadCheck.DegenJacSVD())

    nlp = degen_30303_model()
    results = madnlp(nlp; print_level=MadNLP.ERROR)
    degen_cons_30303 = MadCheck.test_LICQ(nlp, results, MadCheck.ApproximatePrimalDualActiveSetLPEC(HiGHS.Optimizer), MadCheck.DegenJacSVD())

    @test degen_cons_bt4 == [(constraints = [2, 3], bounds = [])]
    @test degen_cons_30303 == [(constraints = [2, 3], bounds = [])]
end

@testset "Test active set detection" begin
    nlp = hs15_model()  #TODO Find a better test to verify the methods are working properly - as they are inexact by nature, more over, the primal method can not detect weakly active constraints
    results = madnlp(nlp; print_level=MadNLP.ERROR)
    
    n = NLPModels.get_nvar(nlp)

    new_point = results
    max_dist = 0
    for i in 1:n
        noise = 2*rand() - 1
        new_point.solution[i] += (max_dist*noise)/n
    end

    active_solution = MadCheck.find_active(nlp, results,MadCheck.BasicActiveSet())

    active_LP_P = MadCheck.find_active(nlp, new_point, MadCheck.PrimalActiveSetLP(HiGHS.Optimizer; Δ = 4 * 1e-8/n))
    active_LPEC = MadCheck.find_active(nlp, new_point, MadCheck.PrimalDualActiveSetLPEC(HiGHS.Optimizer))
    active_LPEC_A = MadCheck.find_active(nlp, new_point, MadCheck.ApproximatePrimalDualActiveSetLPEC(HiGHS.Optimizer))

    @test Set(active_LP_P.active) == Set(active_solution.active) && Set(active_LP_P.active_boundary) == Set(active_solution.active_boundary)
    @test Set(active_LPEC.active) == Set(active_solution.active) && Set(active_LPEC.active_boundary) == Set(active_solution.active_boundary)
    @test Set(active_LPEC_A.active) == Set(active_solution.active) && Set(active_LPEC_A.active_boundary) == Set(active_solution.active_boundary)
end