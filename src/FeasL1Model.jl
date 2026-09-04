"""
    FeasL1Model

Reformulate the nonlinear program
```
min_x        f(x)
subject to   c♭ ≤ c(x) ≤ c♯
             x ≥ 0

```
into an elastic ℓ1-exact penalty formulation :
```
min_{x,u}    1ᵀ u
subject to   c♭ - u ≤ c(x) ≤ c♯ + u
             x ≥ 0

```
The relaxed constraints are transformed to
```
    c♭ ≤ c(x) + u
         c(x) - u ≤ c♯

```
"""
struct FeasL1Model{T, VT, M, I, V} <:
    NLPModels.AbstractNLPModel{T, VT} where {M <: NLPModels.AbstractNLPModel{T, VT}, I <: Integer, V}
    nlp::M
    nx::I
    nr::I
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
    low::V
    upp::V
    rng::V
    nzj_rng::V
    tmp_ncon::VT  # for jtprod
end

# constructor
function FeasL1Model(nlp::NLPModels.AbstractNLPModel{T, S}) where {T, S}
    if (nlp.meta.ncon == 0)
        @warn "input problem $(nlp.meta.name) is unconstrained, not generating the feasibily problem model"
        return nlp
    end    

    # construct meta
    nx = NLPModels.get_nvar(nlp)
    low = findall((nlp.meta.lcon .> -Inf) .&& (nlp.meta.ucon .== Inf))
    upp = findall((nlp.meta.lcon .== -Inf) .&& (nlp.meta.ucon .< Inf))
    rng = findall((nlp.meta.lcon .> -Inf) .&& (nlp.meta.ucon .< Inf))
    nlow = length(low)
    nupp = length(upp)
    nrng = length(rng)

    # evaluate Jacobian sparity pattern to find the number of nonzeros contributed by range constraints
    rows, cols = NLPModels.jac_structure(nlp)
    nzj_rng = findall(i -> i ∈ rng, rows)
    nnzj_rng = length(nzj_rng)

    # there is one elastic variable per constraint, even if it is two-sided
    nr = nlow + nupp + nrng

    # bound constraints are
    # 1. bounds on x that were already there
    # 2. vL ≥ 0 and vU ≥ 0
    lvar = vcat(nlp.meta.lvar, fill!(similar(nlp.meta.x0, nr), -Inf))
    for i ∈ low
        lvar[nx + i] = 0
    end
    for i ∈ upp
        lvar[nx + i] = 0
    end
    for i ∈ rng
        lvar[nx + i] = 0
    end

    ncon = nlow + nupp + 2 * nrng  # range constraints become two constraints
    lcon = vcat(nlp.meta.lcon, fill!(similar(nlp.meta.lcon, nrng), -Inf))
    ucon = fill!(similar(nlp.meta.lcon, ncon), Inf)
    ucon[upp] .= nlp.meta.ucon[upp]
    for (k, i) ∈ enumerate(rng)
        ucon[nlow + nupp + nrng + k] = nlp.meta.ucon[i]
    end
    
    meta = NLPModels.NLPModelMeta{T, S}(
        nx + nr;
        lvar = lvar,
        uvar = vcat(nlp.meta.uvar, fill!(similar(nlp.meta.x0, nr), Inf)),
        x0 = vcat(nlp.meta.x0, fill!(similar(nlp.meta.x0, nr), 1.0)),
        y0 = vcat(nlp.meta.y0, fill!(similar(nlp.meta.y0, nrng), 1.0)),
        name = "FeasL1-" * nlp.meta.name,
        nnzj = nlp.meta.nnzj + nnzj_rng + ncon,
        nnzh = nlp.meta.nnzh,
        ncon = ncon,
        lcon = lcon,
        ucon = ucon,
        minimize = true
        # TODO: define nln, etc.
    )

    return FeasL1Model{T, S, typeof(nlp), typeof(nr), typeof(low)}(
        nlp,
        nx,
        nr,
        meta,
        NLPModels.Counters(),
        low,
        upp,
        rng,
        nzj_rng,
        S(undef, NLPModels.get_ncon(nlp)),
    )
end

function get_nelastics(
    fmodel::FeasL1Model{T, S, M, I, V},
) where {T, S, M <: NLPModels.AbstractNLPModel{T, S}, I <: Integer, V}
    return fmodel.nr
end

# ϕ(x, v) = ∑ v
function NLPModels.obj(
    fmodel::FeasL1Model{T, S, M, I, V},
    xv::S,
) where {T, S, M <: NLPModels.AbstractNLPModel{T, S}, I <: Integer, V}
    NLPModels.increment!(fmodel, :neval_obj)
    nx = NLPModels.get_nvar(fmodel.nlp)
    nr = get_nelastics(fmodel)
    v = view(xv, (nx + 1):(nx + nr))
    return sum(v)
end

# ∇ϕ(x, v) = [ 0 ]  (nx)
#            [ 1 ]. (nr)
# The variables v are in the same order as the constraints of fmodel.nlp.
function NLPModels.grad!(
     fmodel::FeasL1Model{T, S, M, I, V},
     xv::S,
     g::S,
) where {T, S, M <: NLPModels.AbstractNLPModel{T, S}, I <: Integer, V}
     NLPModels.increment!(fmodel, :neval_grad)
     nx = NLPModels.get_nvar(fmodel.nlp)
     nr = get_nelastics(fmodel)

     g[1:nx] .= zero(T)
     g[nx+1:nx+nr] .= one(T)

     return g
end

# The elastics appear linearly in the objective and constraints.
# They do not contribute nonzeros to the Hessian of the Lagrangian
# We also remove the Hessian of the objective function
# Thus it has the form
#   [ H-∇²f 0 ]
#   [   0   0 ].
function NLPModels.hess_structure!(
    fmodel::FeasL1Model{T, S, M, I, V},
    hrows::AbstractVector{<:Integer},
    hcols::AbstractVector{<:Integer},
) where {T, S, M <: NLPModels.AbstractNLPModel{T, S}, I <: Integer, V}
    NLPModels.increment!(fmodel, :neval_hess)
    return NLPModels.hess_structure!(fmodel.nlp, hrows, hcols)
end

function NLPModels.hess_coord!(
    fmodel::FeasL1Model{T, S, M, I, V},
    xv::AbstractVector,
    y::AbstractVector,
    hvals::AbstractVector;
    obj_weight::T = one(T),
) where {T, S, M <: NLPModels.AbstractNLPModel{T, S}, I <: Integer, V}
    NLPModels.increment!(fmodel, :neval_hess)
    nc = get_nelastics(fmodel)
    nrng = length(fmodel.rng)
    fill!(hvals, zero(T))
    x = view(xv, 1:NLPModels.get_nvar(fmodel.nlp))
    fmodel.tmp_ncon[fmodel.low] .= y[fmodel.low]
    fmodel.tmp_ncon[fmodel.upp] .= y[fmodel.upp]
    fmodel.tmp_ncon[fmodel.rng] .= y[fmodel.rng] .+ y[(nc + 1):(nc + nrng)]
    return NLPModels.hess_coord!(fmodel.nlp, x, fmodel.tmp_ncon, hvals; obj_weight = 0.0)
end

function NLPModels.hprod!(
    fmodel::FeasL1Model{T, S, M, I, V},
    xv::AbstractVector,
    u::AbstractVector,
    hu::AbstractVector;
    obj_weight::T = one(T),
) where {T, S, M <: NLPModels.AbstractNLPModel{T, S}, I <: Integer, V}
    NLPModels.increment!(fmodel, :neval_hprod)
    nx = NLPModels.get_nvar(fmodel.nlp)
    nr = get_nelastics(fmodel)
    x = view(xv, 1:nx)
    NLPModels.hprod!(fmodel.nlp, x, view(u, 1:nx), hu[1:nx], obj_weight = 0.0)
    hu[(nx + 1):(nx + nr)] .= 0
    return hu
end

function NLPModels.hprod!(
    fmodel::FeasL1Model{T, S, M, I, V},
    xv::AbstractVector,
    y::AbstractVector,
    u::AbstractVector,
    hu::AbstractVector;
    obj_weight::T = one(T),
) where {T, S, M <: NLPModels.AbstractNLPModel{T, S}, I <: Integer, V}
    NLPModels.increment!(fmodel, :neval_hprod)
    nx = NLPModels.get_nvar(fmodel.nlp)
    nr = get_nelastics(fmodel)
    nc = get_nelastics(fmodel)
    nrng = length(fmodel.rng)

    x = view(xv, 1:nx)
    fmodel.tmp_ncon[fmodel.low] .= y[fmodel.low]
    fmodel.tmp_ncon[fmodel.upp] .= y[fmodel.upp]
    fmodel.tmp_ncon[fmodel.rng] .= y[fmodel.rng] .+ y[(nc + 1):(nc + nrng)]
    NLPModels.hprod!(fmodel.nlp, x, fmodel.tmp_ncon, view(u, 1:nx), hu[1:nx], obj_weight = 0.0)
    hu[(nx + 1):(nx + nr)] .= 0
    return hu
end

# Range constraints
#   cL ≤ c(x) ≤ cU
# are transformed to
#   cL - v ≤ c(x) ≤ cU + v,
# i.e.,
#   cL ≤ c(x) + v
#        c(x) - v ≤ cU.
function NLPModels.cons!(
    fmodel::FeasL1Model{T, S, M, I, V},
    xv::AbstractVector,
    c::AbstractVector,
) where {T, S, M <: NLPModels.AbstractNLPModel{T, S}, I <: Integer, V}
    NLPModels.increment!(fmodel, :neval_cons)
    nx = NLPModels.get_nvar(fmodel.nlp)
    nr = get_nelastics(fmodel)  # = nlow + nupp + nrng
    x = view(xv, 1:nx)
    v = view(xv, (nx + 1):(nx + nr))
    NLPModels.cons!(fmodel.nlp, x, view(c, 1:nr))
    nrng = length(fmodel.rng)
    @views c[(nr + 1):(nr + nrng)] .= c[fmodel.rng]
    @views c[fmodel.low] .+= v[fmodel.low]
    @views c[fmodel.upp] .-= v[fmodel.upp]
    @views c[fmodel.rng] .+= v[fmodel.rng]
    for (k, i) ∈ enumerate(fmodel.rng)
        c[nr + k] -= v[i]
    end
    return c
end

function NLPModels.jac_structure!(
    fmodel::FeasL1Model{T, S, M, I, V},
    jrows::AbstractVector{<:Integer},
    jcols::AbstractVector{<:Integer},
) where {T, S, M <: NLPModels.AbstractNLPModel{T, S}, I <: Integer, V}
    NLPModels.increment!(fmodel, :neval_jac)
    nx = NLPModels.get_nvar(fmodel.nlp)
    nc = get_nelastics(fmodel)
    nrng = length(fmodel.rng)
    orig_nnzj = fmodel.nlp.meta.nnzj
    nzj_rng = fmodel.nzj_rng
    nnzj_rng = length(nzj_rng)
    # Jacobian of the original problem
    NLPModels.jac_structure!(fmodel.nlp, view(jrows, 1:orig_nnzj), view(jcols, 1:orig_nnzj))
    begin_nnzj = orig_nnzj + 1
    end_nnzj = orig_nnzj + nnzj_rng
    corr = sparsevec(fmodel.rng, 1:nrng)
    @views jrows[begin_nnzj:end_nnzj] .= [corr[i] + nc for i in jrows[nzj_rng]]
    @views jcols[begin_nnzj:end_nnzj] .= jcols[nzj_rng]
    # Identity blocks associated with elastics
    begin_nnzj = end_nnzj + 1
    end_nnzj = end_nnzj + nc
    @views jrows[begin_nnzj:end_nnzj] .= 1:nc
    @views jcols[begin_nnzj:end_nnzj] .= (nx + 1):(nx + nc)
    if nrng > 0
        begin_nnzj = end_nnzj + 1
        end_nnzj = end_nnzj + nrng
        @views jrows[begin_nnzj:end_nnzj] .= (nc + 1):(nc + nrng)
        @views jcols[begin_nnzj:end_nnzj] .= (nx .+ fmodel.rng)
    end
    return jrows, jcols
end

function NLPModels.jac_coord!(
    fmodel::FeasL1Model{T, S, M, I, V},
    xv::AbstractVector,
    jvals::AbstractVector,
) where {T, S, M <: NLPModels.AbstractNLPModel{T, S}, I <: Integer, V}
    NLPModels.increment!(fmodel, :neval_jac)
    nx = NLPModels.get_nvar(fmodel.nlp)
    nc = get_nelastics(fmodel)
    nrng = length(fmodel.rng)
    orig_nnzj = fmodel.nlp.meta.nnzj
    nzj_rng = fmodel.nzj_rng
    nnzj_rng = length(nzj_rng)
    x = view(xv, 1:nx)
    v = view(xv, (nx + 1):(nx + nc))
    # Jacobian of the original problem
    NLPModels.jac_coord!(fmodel.nlp, x, view(jvals, 1:orig_nnzj))
    begin_nnzj = orig_nnzj + 1
    end_nnzj = orig_nnzj + nnzj_rng
    @views jvals[begin_nnzj:end_nnzj] .= jvals[nzj_rng]
    # Identity blocks associated with elastics
    for i ∈ fmodel.low
        jvals[end_nnzj + i] = 1
    end
    for i ∈ fmodel.upp
        jvals[end_nnzj + i] = -1
    end
    for i ∈ fmodel.rng
        jvals[end_nnzj + i] = 1
    end
    for i ∈ 1:nrng
        jvals[end_nnzj + nc + i] = -1
    end
    return jvals
end

function NLPModels.jprod!(
    fmodel::FeasL1Model{T, S, M, I, V},
    xv::AbstractVector,
    u::AbstractVector,
    Ju::AbstractVector,
) where {T, S, M <: NLPModels.AbstractNLPModel{T, S}, I <: Integer, V}
    NLPModels.increment!(fmodel, :neval_jprod)
    nx = NLPModels.get_nvar(fmodel.nlp)
    nc = get_nelastics(fmodel)
    nrng = length(fmodel.rng)
    x = view(xv, 1:nx)
    ux = view(u, 1:nx)
    NLPModels.jprod!(fmodel.nlp, x, ux, view(Ju, 1:NLPModels.get_ncon(fmodel.nlp)))
    uv = view(u, (nx + 1):(nx + nc))
    @views Ju[fmodel.low] .+= uv[fmodel.low]
    @views Ju[fmodel.upp] .-= uv[fmodel.upp]
    k = 1
    for i ∈ fmodel.rng
        Ju[nc + k] = Ju[i]
        k += 1
    end
    @views Ju[fmodel.rng] .+= uv[fmodel.rng]
    @views Ju[(nc + 1):(nc + nrng)] .-= uv[fmodel.rng]
    return Ju
end

function NLPModels.jtprod!(
    fmodel::FeasL1Model{T, S, M, I, V},
    xv::AbstractVector,
    u::AbstractVector,
    Jtu::AbstractVector,
) where {T, S, M <: NLPModels.AbstractNLPModel{T, S}, I <: Integer, V}
    NLPModels.increment!(fmodel, :neval_jtprod)
    nx = NLPModels.get_nvar(fmodel.nlp)
    nc = get_nelastics(fmodel)
    nrng = length(fmodel.rng)
    x = view(xv, 1:nx)
    fmodel.tmp_ncon[fmodel.low] .= u[fmodel.low]
    fmodel.tmp_ncon[fmodel.upp] .= u[fmodel.upp]
    fmodel.tmp_ncon[fmodel.rng] .= u[fmodel.rng] .+ u[(nc + 1):(nc + nrng)]
    NLPModels.jtprod!(fmodel.nlp, x, fmodel.tmp_ncon, view(Jtu, 1:nx))
    for i ∈ fmodel.low
        Jtu[nx + i] = u[i]
    end
    for i ∈ fmodel.upp
        Jtu[nx + i] = -u[i]
    end
    for (k, i) ∈ enumerate(fmodel.rng)
        Jtu[nx + i] = u[i] - u[nc + k]
    end
    return Jtu
end

