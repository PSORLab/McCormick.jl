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
# src/forward_operators/concave_increasing.jl
# Contains definitions of functions used to enforce bounds on intermediate
# terms in the computation of relaxations:
# positive, negative, lower_bnd, upper_bnd, bnd
#############################################################################

function cut_kernel(x::MC{N,T}, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    zL = z.lo
    zU = z.hi
    cvv, ccv, cv_grad, cc_grad = cut(zL, zU, cvv, ccv, cv_grad, cc_grad)
    return MC{N, T}(cvv, ccv, z, cv_grad, cc_grad, x.cnst)
end

for (f, ii) in ((:positive, :(Interval(MC_DOMAIN_TOL, Inf))),
          (:negative, :(Interval(-Inf, -MC_DOMAIN_TOL))),
          (:lower_bnd, :(Interval(lb, Inf))),
          (:upper_bnd, :(Interval(-Inf, ub))),
          (:bnd, :(Interval(lb, ub))))
    @eval ($f)(x) = x
    @eval function ($f)(x::MC{N,T}) where {N, T<:Union{NS,MV}}
        cut_kernel(x, x.Intv ∩ $ii)
    end
end
