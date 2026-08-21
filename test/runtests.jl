
using Test
using MadNLP
using NLPModels
using MadCheck

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

