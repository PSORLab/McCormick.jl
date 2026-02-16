# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A McCormick operator library in Julia
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# src/implicit_routines/preconditioner/dense.jl
# Definitions of dense preconditioners used with implicit relaxations.
#############################################################################

"""
$(TYPEDEF)

A dense LU preconditioner for implicit McCormick relaxation.
"""
struct DenseMidInv{S <: VecOrMat{Float64}} <: AbstractPreconditionerMC
    "Storage for the midpoint matrix and inverse"
    Y::S
    "Vector of length(1) used for intermediate calculation"
    YInterval::Vector{Interval{Float64}}
    "Number of state space variables"
    nx::Int
    "Number of decision space variables"
    np::Int
end
DenseMidInv(nx::Int, np::Int) = DenseMidInv(zeros(Float64,nx,nx), zeros(Interval{Float64},1), nx, np)
function precondition!(d::DenseMidInv{S}, H::Vector{MC{N,T}}, J::Array{MC{N,T},2}) where {N, T <: RelaxTag, S <: VecOrMat{Float64}}
    for i in eachindex(J)
        @inbounds d.YInterval[1] = J[i].Intv
        @inbounds d.Y[i] = 0.5*(d.YInterval[1].bareinterval.lo + d.YInterval[1].bareinterval.hi)
    end
    F = lu!(d.Y)
    H .= F\H
    J .= F\J
    return
end
function (d::DenseMidInv)(h!::FH, hj!::FJ, nx::Int, np::Int) where {FH <: Function, FJ <: Function}
    return DenseMidInv{Array{Float64,2}}(zeros(nx,nx), zeros(Interval{Float64},1), nx, np)
end
function preconditioner_storage(d::DenseMidInv, tag::T) where T <: RelaxTag
    zeros(MC{d.np, T}, (d.nx, d.nx))
end
