# Copyright (c) 2018 Matthew Wilhelm, Robert Gottlieb, Dimitri Alston, 
# Matthew Stuber, and the University of Connecticut (UConn)
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A forward McCormick operator library
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
    Ibnd = intersect_interval(x.Intv, y.Intv)
    if x.cc < y.cc
        cc = x.cc
        cc_grad = x.cc_grad
    else
        cc = y.cc
        cc_grad = y.cc_grad
    end
    if x.cv > y.cv
        cv = x.cv
        cv_grad = x.cv_grad
    else
        cv = y.cv
        cv_grad = y.cv_grad
    end
    MC{N,NS}(cv, cc, Ibnd, cv_grad, cc_grad, x.cnst && y.cnst)
end
final_cut(x::MC{N,Diff}, y::MC{N,Diff}) where N = intersect(x, y)


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
Base.@kwdef mutable struct MCCallback{FH, FJ, C<:AbstractContractorMC, PRE<:AbstractPreconditionerMC, N, T<:RelaxTag, AMAT<:AbstractMatrix} <: AbstractMCCallback
    "Function h(x,p) = 0 defined in place by h!(out,x,p)"
    h!::FH
    "Jacobian of h(x,p) w.r.t x"
    hj!::FJ
    "Intermediate inplace storage for output of h!"
    H::Vector{MC{N,T}}
    "Intermediate inplace storage for output of hj!"
    J::AMAT
    J0::Vector{AMAT}
    xz0::Vector{AMAT}
    xmid::Vector{Float64}
    "State space `x` interval bounds"
    X::Vector{Interval{Float64}}
    "Decision space `p` interval bounds"
    P::Vector{Interval{Float64}}
    "State space dimension"
    nx::Int
    "Decision space dimension"
    np::Int
    "Convex combination parameter"
    λ::Float64 = 0.5
    "Tolerance for interval equality"
    eps::Float64 = 0.0
    "Number of contractor steps to take"
    kmax::Int = 2
    "Reference decision point at which affine relaxations are calculated (and used in subsequent calculations)."
    pref_mc::Vector{MC{N,T}}
    "Decision point at which relaxation is evaluated."
    p_mc::Vector{MC{N,T}}
    "Vector used to temporarily store p in gen_expansion_params! routine."
    p_temp_mc::Vector{MC{N,T}}
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
    apply_precond::Bool = true
    "Vector of relaxations of `x` at each iteration used to generated affine relaxations used in intermediate calculation."
    param::Vector{Vector{MC{N,T}}}
    "Indicates that subgradient-based apriori relaxations of multiplication should be used."
    use_apriori::Bool = false
end
function MCCallback(h!::FH, hj!::FJ, nx::Int, np::Int, contractor::S = NewtonGS(),
                    preconditioner::T = DenseMidInv(zeros(Float64,nx,nx), zeros(Interval{Float64},1), nx, np),
                    tag::TAG = NS(), kmax::Int = 2) where {FH, FJ, S <: AbstractContractorMC, T, TAG <: RelaxTag}
    J = preconditioner_storage(preconditioner, tag)
    return MCCallback{FH,FJ,NewtonGS,DenseMidInv,np,TAG,typeof(J)}(h! = h!,  hj! = hj!, 
                H = zeros(MC{np,TAG}, (nx,)), 
                J = J, kmax = kmax,
                J0 = create_vec_of_mutable(Matrix{MC{np, TAG}}, () -> preconditioner_storage(preconditioner, tag), kmax), 
                xz0 = create_vec_of_mutable(Matrix{MC{np, TAG}}, () -> preconditioner_storage(preconditioner, tag), kmax), 
                xmid = zeros(Float64, (nx,)), 
                X = zeros(Interval{Float64}, (nx,)), 
                P = zeros(Interval{Float64}, (np,)), 
                nx = nx, np = np,
                pref_mc = zeros(MC{np,TAG}, (np,)), 
                p_mc = zeros(MC{np,TAG}, (np,)),
                p_temp_mc = zeros(MC{np,TAG}, (np,)),
                x0_mc = zeros(MC{np,TAG}, (nx,)), 
                x_mc = zeros(MC{np,TAG}, (nx,)),
                xa_mc = zeros(MC{np,TAG}, (nx,)), 
                xA_mc = zeros(MC{np,TAG}, (nx,)), 
                aff_mc = zeros(MC{np,TAG}, (nx,)), 
                z_mc = zeros(MC{np,TAG}, (nx,)),
                contractor = NewtonGS(),
                preconditioner = preconditioner(h!, hj!, nx, np),
                param = create_vec_of_mutable(Vector{MC{np, TAG}}, () -> zeros(MC{np,TAG}, (nx, )), kmax),)
end

function (d::MCCallback)()
    @unpack h!, hj!, H, J, aff_mc, p_mc, z_mc = d

    h!(H, z_mc, p_mc)
    hj!(J, aff_mc, p_mc)
    return
end

include("preconditioner/dense.jl")

"""
$(SIGNATURES)

Computates the affine relaxations of the state variable.
"""
function affine_exp!(x::S, p::Vector{MC{N,T}}, d::MCCallback) where {S, N, T<:RelaxTag}
    @unpack λ, np, nx, pref_mc, xa_mc, xA_mc, z_mc = d

    S1 = zero(MC{N,T})
    S2 = zero(MC{N,T})
    S3 = zero(MC{N,T})
    @inbounds for i = 1:nx
        S1 = zero(MC{N,T})
        S2 = zero(MC{N,T})
        S3 = zero(MC{N,T})
        @inbounds for j = 1:np
            δp = p[j] - pref_mc[j]
            S1 += δp*x[i].cv_grad[j]
            S2 += δp*x[i].cc_grad[j]
            S3 += (λ*x[i].cv_grad[j] + (1.0 - λ)*x[i].cc_grad[j])*δp
        end
        t1 = x[i].cv + S1
        t2 = x[i].cc + S2
        t3 = λ*x[i].cv + (1.0 - λ)*x[i].cc + S3
        xa_mc[i] = MC{N,T}(t1.cv, t1.cc, t1.Intv, x[i].cv_grad, x[i].cv_grad, S1.cnst)
        xA_mc[i] = MC{N,T}(t2.cv, t2.cc, t2.Intv, x[i].cc_grad, x[i].cc_grad, S2.cnst)
        z_mc[i]  = MC{N,T}(t3.cv, t3.cc, t3.Intv, λ*x[i].cv_grad + (1.0 - λ)*x[i].cc_grad, λ*x[i].cv_grad + (1.0 - λ)*x[i].cc_grad, S3.cnst)
    end
    return
end

"""
$(SIGNATURES)

Corrects the relaxation of the state variable `x_mc` if the affine relaxation,
"""
function correct_exp!(d::MCCallback{FH,FJ,C,PRE,N,T}) where {FH, FJ, C, PRE, N, T<:RelaxTag}
    @unpack eps, nx, X, x_mc, z_mc = d

    zero_grad = zeros(SVector{N,Float64})
    @inbounds for i = 1:nx
        if (z_mc[i].Intv.bareinterval.lo - eps < X[i].bareinterval.lo) && (z_mc[i].Intv.bareinterval.hi + eps > X[i].bareinterval.hi)
            x_mc[i] = MC{N,T}(X[i])
        elseif z_mc[i].Intv.bareinterval.lo - eps < X[i].bareinterval.lo
            x_mc[i] = MC{N,T}(X[i].bareinterval.lo, x_mc[i].cc, interval(X[i].bareinterval.lo, x_mc[i].Intv.bareinterval.hi), zero_grad, x_mc[i].cc_grad, x_mc[i].cnst)
        else
            if z_mc[i].Intv.bareinterval.hi + eps > X[i].bareinterval.hi
                x_mc[i] = MC{N,T}(x_mc[i].cv, X[i].bareinterval.hi, interval(x_mc[i].Intv.bareinterval.lo, X[i].bareinterval.hi), x_mc[i].cv_grad, zero_grad, x_mc[i].cnst)
            end
        end
    end
    return
end

include("contract.jl")

"""
$(SIGNATURES)
"""
function precond_and_contract!(d!::MCCallback{FH,FJ,C,PRE,N,T}, k::Int, b::Bool) where {FH, FJ, C <: AbstractContractorMC, PRE <: AbstractPreconditionerMC, N, T<:RelaxTag}
    @unpack apply_precond, aff_mc, contractor, H, J, xa_mc, xA_mc, preconditioner = d!

    @. aff_mc = MC{N,T}(cv(xa_mc), cc(xA_mc))
    d!()
    if apply_precond
        precondition!(preconditioner, H, J)
    end
    contract!(contractor, d!, k, b)
    return
end

"""
$(SIGNATURES)

Populates `x_mc`, `xa_mc`, `xA_mc`, and `z_mc` with affine bounds.
"""
function populate_affine!(d::MCCallback{FH,FJ,C,PRE,N,T}, interval_bnds::Bool) where {FH, FJ, C <: AbstractContractorMC, PRE <: AbstractPreconditionerMC, N, T<:RelaxTag}
    @unpack λ, nx, X, x_mc, xa_mc, xA_mc, z_mc = d

    if interval_bnds
        @inbounds for i in 1:nx
            xL = X[i].bareinterval.lo
            xU = X[i].bareinterval.hi
            x_mc[i]  = MC{N,T}(xL, xU)
            xa_mc[i] = MC{N,T}(xL, xL)
            xA_mc[i] = MC{N,T}(xU, xU)
            z_mc[i]  = λ*xa_mc[i] + (1.0 - λ)*xA_mc[i]
         end
    end
    return
end

"""
$(SIGNATURES)

Constructs parameters need to compute relaxations of `h`.
"""
function gen_expansion_params!(d::MCCallback, interval_bnds::Bool = true)
    @unpack kmax, param, p_mc, p_temp_mc, pref_mc, x_mc = d

    populate_affine!(d, interval_bnds)
    @. p_temp_mc = p_mc 
    @. p_mc = pref_mc
    @. param[1] = x_mc
    for k = 2:kmax
        precond_and_contract!(d, k, true)
        affine_exp!(x_mc, pref_mc, d)
        correct_exp!(d)
        @. param[k] = x_mc
    end
    @. p_mc = p_temp_mc 
    return
end

"""
$(SIGNATURES)

Compute relaxations of `x(p)` defined by `h(x,p) = 0` where `h` is specifed as `h(out, x, p)`.
"""
function implicit_relax_h!(d::MCCallback, interval_bnds::Bool = true)
    @unpack kmax, param, p_mc = d
    
    populate_affine!(d, interval_bnds)
    for k = 1:kmax
        affine_exp!(param[k], p_mc, d)
        precond_and_contract!(d, k, false)
    end
    return
end
