# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
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
    "Vector for intermediate calculation "
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
        @inbounds d.Y[i] = 0.5*(d.YInterval[1].lo + d.YInterval[1].hi)
    end
    F = lu!(d.Y)
    H .= F\H
    J .= F\J
    return
end
function (d::DenseMidInv)(h!::FH, hj!::FJ, nx::Int, np::Int) where {FH <: Function, FJ <: Function}
    S = nx == 1 ? Vector{Float64} : Array{Float64,2}
    return DenseMidInv{S}(zeros(nx,nx), zeros(Interval{Float64},1), nx, np)
end
function preconditioner_storage(d::DenseMidInv, tag::T) where T <: RelaxTag
    zeros(MC{d.np, T}, (d.nx, d.nx))
end
