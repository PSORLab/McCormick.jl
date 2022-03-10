# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A McCormick operator library in Julia
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# src/implicit_routines/implicit.jl
# Subroutines for generating relaxations of implicit functions.
#############################################################################

"""
$(SIGNATURES)

An operator that cuts the `x` object using the `y bounds` in a differentiable
or nonsmooth fashion to achieve a composite relaxation within `y`.
"""
function final_cut(x::MC{N,NS}, y::MC{N,NS}) where N
    Intv = x.Intv ∩ y.Intv
    if x.cc < y.cc
        cc = x.cc
        cc_grad::SVector{N,Float64} = x.cc_grad
    else
        cc = y.cc
        cc_grad = y.cc_grad
    end
    if x.cv > y.cv
        cv = x.cv
        cv_grad::SVector{N,Float64} = x.cv_grad
    else
        cv = y.cv
        cv_grad = y.cv_grad
    end
    x_out::MC{N,NS} = MC{N,NS}(cv, cc, Intv, cv_grad, cc_grad, x.cnst && y.cnst)
    return x_out
end
function final_cut(x::MC{N,Diff}, y::MC{N,Diff}) where N
    x_out = intersect(x, y)
    return x_out
end

"""
$(TYPEDEF)

An abstract type for each manner of contractor using in the implicit function relaxation algorithms.
"""
abstract type AbstractContractorMC end

"""
$(TYPEDEF)

The Gauss-Seidel implementation of the Newton contractor used in the implicit relaxation scheme.
"""
struct NewtonGS <: AbstractContractorMC end

"""
$(TYPEDEF)

The componentwise implementation of the Krawczyk contractor used in the implicit relaxation scheme.
"""
struct KrawczykCW <: AbstractContractorMC end


"""
$(TYPEDEF)

An abstract type for each manner of preconditioner used in the implicit function relaxation algorithms.
"""
abstract type AbstractPreconditionerMC end

"""
$(SIGNATURES)

Creates storage corresponding to `x::AbstractPreconditionerMC` and `t::T where T<:RelaxTag`.
"""
function preconditioner_storage(x::AbstractPreconditionerMC, t::T) where T <: RelaxTag
    error("Must define function that generates appropriate storage type for preconditioner")
end

"""
$(TYPEDEF)

An abstract type for each manner of callback functions used in the implicit function relaxation algorithms.
"""
abstract type AbstractMCCallback end

"""
$(TYPEDEF)

A structure used to compute implicit relaxations.

$(TYPEDFIELDS)
"""
struct MCCallback{FH <: Function, FJ <: Function, C <: AbstractContractorMC,
                  PRE <: AbstractPreconditionerMC, N, T <: RelaxTag,
                  AMAT <: AbstractMatrix} <:  AbstractMCCallback
    "Function h(x,p) = 0 defined in place by h!(out,x,p)"
    h!::FH
    "Jacobian of h(x,p) w.r.t x"
    hj!::FJ
    ""
    H::Vector{MC{N,T}}
    ""
    J::AMAT
    J0::Vector{AMAT}
    xz0::Vector{AMAT}
    xmid::Vector{Float64}
    X::Vector{Interval{Float64}}
    P::Vector{Interval{Float64}}
    "State space dimension"
    nx::Int
    "Decision space dimension"
    np::Int
    "Convex combination parameter"
    lambda::Float64
    "Tolerance for interval equality"
    eps::Float64
    "Number of contractor steps to take"
    kmax::Int
    pref_mc::Vector{MC{N,T}}
    p_mc::Vector{MC{N,T}}
    x0_mc::Vector{MC{N,T}}
    x_mc::Vector{MC{N,T}}
    xa_mc::Vector{MC{N,T}}
    xA_mc::Vector{MC{N,T}}
    aff_mc::Vector{MC{N,T}}
    z_mc::Vector{MC{N,T}}
    "Type of contractor used in implicit relaxation routine."
    contractor::C
    "Preconditioner used in the implicit relaxation routine."
    preconditioner::PRE
    "Boolean indicating that the preconditioner should be applied"
    apply_precond::Bool
    param::Vector{Vector{MC{N,T}}}
    use_apriori::Bool
end
function MCCallback(h!::FH, hj!::FJ, nx::Int, np::Int,
                    contractor::S = NewtonGS(),
                    preconditioner::T = DenseMidInv(zeros(Float64,1,1), zeros(Interval{Float64},1), 1, 1),
                    relax_tag::TAG = NS()) where {FH <: Function,
                                                  FJ <: Function,
                                                  S <: AbstractContractorMC,
                                                  T,
                                                  TAG <: RelaxTag}
    H = zeros(MC{np,TAG}, (nx,))
    xmid = zeros(Float64, (nx,))
    P = zeros(Interval{Float64}, (np,))
    X = zeros(Interval{Float64}, (nx,))
    lambda = 0.5
    eps = 0.0
    kmax = 2
    p_ref = zeros(MC{np,TAG}, (np,))
    p_mc = zeros(MC{np,TAG}, (np,))
    x0_mc = zeros(MC{np,TAG}, (nx,))
    x_mc = zeros(MC{np,TAG}, (nx,))
    xa_mc = zeros(MC{np,TAG}, (nx,))
    xA_mc = zeros(MC{np,TAG}, (nx,))
    aff_mc = zeros(MC{np,TAG}, (nx,))
    z_mc = zeros(MC{np,TAG}, (nx,))
    contractor = NewtonGS()
    preconditioner = preconditioner(h!, hj!, nx, np)
    J = preconditioner_storage(preconditioner, relax_tag)
    J0 = Matrix{MC{np, TAG}}[]
    for i = 1:kmax
        push!(J0, preconditioner_storage(preconditioner, relax_tag))
    end
    xz0 = Matrix{MC{np, TAG}}[]
    for i = 1:kmax
        push!(J0, preconditioner_storage(preconditioner, relax_tag))
    end
    apply_precond = true
    param = fill(zeros(MC{np,TAG}, (nx, )), (kmax,))
    use_apriori = false
    return MCCallback{FH, FJ, NewtonGS, DenseMidInv, np, TAG, typeof(J)}(h!, hj!, H, J, J0, xz0, xmid, X, P, nx, np,
                                                                         lambda, eps, kmax, p_ref, p_mc, x0_mc, x_mc,
                                                                         xa_mc, xA_mc, aff_mc, z_mc,
                                                                         contractor, preconditioner,
                                                                         apply_precond, param, use_apriori)
end
function (d::MCCallback)()
    d.h!(d.H, d.z_mc, d.p_mc)
    d.hj!(d.J, d.aff_mc, d.p_mc)
    return
end

include("preconditioner/dense.jl")

"""
$(SIGNATURES)

Computates the affine relaxations of the state variable.
"""
function affine_exp!(x::S, p::Vector{MC{N,T}}, d::MCCallback) where {S, N, T<:RelaxTag}

    S1 = zero(MC{N,T})
    S2 = zero(MC{N,T})
    S3 = zero(MC{N,T})
    @inbounds for i = 1:d.nx
        S1 = zero(MC{N,T})
        S2 = zero(MC{N,T})
        S3 = zero(MC{N,T})
        @inbounds for j = 1:N
            S1 += (p[j]-d.pref_mc[j])*x[i].cv_grad[j]
            S2 += (p[j]-d.pref_mc[j])*x[i].cc_grad[j]
            S3 += (d.lambda*x[i].cv_grad[j]+(1.0-d.lambda)*x[i].cc_grad[j])*(p[j]-d.pref_mc[j])
        end
        temp1 = x[i].cv + S1
        temp2 = x[i].cc + S2
        temp3 = d.lambda*x[i].cv+(1.0-d.lambda)*x[i].cc+S3
        d.xa_mc[i] = MC{N,T}(temp1.cv, temp1.cc, temp1.Intv, x[i].cv_grad, x[i].cv_grad, S1.cnst)
        d.xA_mc[i] = MC{N,T}(temp2.cv, temp2.cc, temp2.Intv, x[i].cc_grad, x[i].cc_grad, S2.cnst)
        d.z_mc[i] = MC{N,T}(temp3.cv, temp3.cc, temp3.Intv,
                            d.lambda*x[i].cv_grad+(1.0-d.lambda)*x[i].cc_grad,
                            d.lambda*x[i].cv_grad+(1.0-d.lambda)*x[i].cc_grad, S3.cnst)
    end
    return
end

"""
$(SIGNATURES)

Corrects the relaxation of the state variable `x_mc` if the affine relaxation,
"""
function correct_exp!(d::MCCallback{FH,FJ,C,PRE,N,T}) where {FH <: Function,
                                                             FJ <: Function,
                                                             C, PRE, N,
                                                             T<:RelaxTag}
    zero_grad = zeros(SVector{N,Float64})
    @inbounds for i = 1:d.nx
        if (d.z_mc[i].Intv.lo - d.eps < d.X[i].lo) && (d.z_mc[i].Intv.hi + d.eps > d.X[i].hi)
            d.x_mc[i] = MC{N,T}(d.X[i])
        elseif d.z_mc[i].Intv.lo - d.eps < d.X[i].lo
            d.x_mc[i] = MC{N,T}(d.X[i].lo, d.x_mc[i].cc, Interval(d.X[i].lo, d.x_mc[i].Intv.hi),
                                zero_grad, d.x_mc[i].cc_grad, d.x_mc[i].cnst)
        else
            if d.z_mc[i].Intv.hi + d.eps > d.X[i].hi
                d.x_mc[i] = MC{N,T}(d.x_mc[i].cv, d.X[i].hi, Interval(d.x_mc[i].Intv.lo, d.X[i].hi),
                                d.x_mc[i].cv_grad, zero_grad, d.x_mc[i].cnst)
            end
        end
    end
    return
end

include("contract.jl")

"""
$(SIGNATURES)
"""
function precond_and_contract!(callback!::MCCallback{FH,FJ,C,PRE,N,T}, k::Int, b::Bool) where {FH <: Function,
                                                                                       FJ <: Function,
                                                                                       C <: AbstractContractorMC,
                                                                                       PRE <: AbstractPreconditionerMC,
                                                                                       N, T<:RelaxTag}
    @. callback!.aff_mc = MC{N,T}(cv(callback!.xa_mc), cc(callback!.xA_mc))
    callback!()
    if callback!.apply_precond
        precondition!(callback!.preconditioner, callback!.H, callback!.J)
    end
    contract!(callback!.contractor, callback!, k, b)
    return
end

"""
$(SIGNATURES)

Populates `x_mc`, `xa_mc`, `xA_mc`, and `z_mc` with affine bounds.
"""
function populate_affine!(d::MCCallback{FH,FJ,C,PRE,N,T}, interval_bnds::Bool) where {FH <: Function,
                                                                                      FJ <: Function,
                                                                                      C <: AbstractContractorMC,
                                                                                      PRE <: AbstractPreconditionerMC,
                                                                                      N, T<:RelaxTag}
    if interval_bnds
        @inbounds for i in 1:d.nx
            d.x_mc[i] = MC{N,T}(d.X[i].lo, d.X[i].hi)
            d.xa_mc[i] = MC{N,T}(d.X[i].lo, d.X[i].lo)
            d.xA_mc[i] = MC{N,T}(d.X[i].hi, d.X[i].hi)
            d.z_mc[i] = d.lambda*d.xa_mc[i] + (1.0 - d.lambda)*d.xA_mc[i]
         end
    end
    return
end

"""
$(SIGNATURES)

Constructs parameters need to compute relaxations of `h`.
"""
function gen_expansion_params!(d::MCCallback, interval_bnds::Bool = true) where {N, T<:RelaxTag}
    populate_affine!(d, interval_bnds)
    @. d.param[1] = d.x_mc
    for k = 2:d.kmax
        precond_and_contract!(d, k, true)
        affine_exp!(d.x_mc, d.pref_mc, d)
        correct_exp!(d)
        @. d.param[k] = d.x_mc
    end
    return
end

"""
$(SIGNATURES)

Compute relaxations of `x(p)` defined by `h(x,p) = 0` where `h` is specifed as `h(out, x, p)`.
"""
function implicit_relax_h!(d::MCCallback, interval_bnds::Bool = true) where {N, T<:RelaxTag}
    populate_affine!(d, interval_bnds)
    for k = 1:d.kmax
        affine_exp!(d.param[k], d.p_mc, d)
        precond_and_contract!(d, k, false)
    end
    return
end
