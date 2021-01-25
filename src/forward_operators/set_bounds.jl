# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This code is licensed under MIT license (see LICENSE.md for full details)
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
    cvv, ccv, cv_grad, cc_grad = cut(zL, zU, x.cv, x.cc, x.cv_grad, x.cc_grad)
    return MC{N, T}(cvv, ccv, z, cv_grad, cc_grad, x.cnst)
end



"""
positive(x::MC)

Sets the lower interval bound and the convex relaxation of `x` to a value of
at least `McCormick.MC_DOMAIN_TOL`. (Sub)gradients are adjusted appropriately.
"""
positive(x) = x
function positive(x::Interval)
     x ∩ Interval(MC_DOMAIN_TOL, Inf)
end
function positive_kernel(x::MC{N,T}, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    cut_kernel(x, z)
end
function positive(x::MC{N,T}) where {N, T<:Union{NS,MV}}
    positive_kernel(x, positive(x.Intv))
end



negative(x) = x
function negative(x::Interval)
     x ∩ Interval(-Inf, -MC_DOMAIN_TOL)
end
function negative_kernel(x::MC{N,T}, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    cut_kernel(x, z)
end

"""
negative(x::MC)

Sets the upper interval bound and the concave relaxation of `x` to a value of
at most `-McCormick.MC_DOMAIN_TOL`. (Sub)gradients are adjusted appropriately.
"""
function negative(x::MC{N,T}) where {N, T<:Union{NS,MV}}
    negative_kernel(x, negative(x.Intv))
end



lower_bnd(x, lb) = x
function lower_bnd(x::Interval{Float64}, lb::Float64)
     x ∩ Interval(lb, Inf)
end
function lower_bnd_kernel(x::MC{N,T}, lb::Float64, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    cut_kernel(x, z)
end

"""
lower_bnd(x::MC, lb::Float64)

Sets the lower interval bound and the convex relaxation of `x` to a value of
at least `lb`. (Sub)gradients are adjusted appropriately.
"""
function lower_bnd(x::MC{N,T}, lb::Float64) where {N, T<:Union{NS,MV}}
    lower_bnd_kernel(x, lb, lower_bnd(x.Intv, lb))
end


upper_bnd(x, lb) = x
function upper_bnd(x::Interval{Float64}, ub::Float64)
     x ∩ Interval(-Inf, ub)
end
function upper_bnd_kernel(x::MC{N,T}, ub::Float64, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    cut_kernel(x, z)
end

"""
upper_bnd(x::MC, ub)

Sets the upper interval bound and the concave relaxation of `x` to a value of
at most `ub`. (Sub)gradients are adjusted appropriately.
"""
function upper_bnd(x::MC{N,T}, lb::Float64) where {N, T<:Union{NS,MV}}
    upper_bnd_kernel(x, lb, upper_bnd(x.Intv, lb))
end

bnd(x, lb, ub) = x
function bnd(x::Interval{Float64}, lb::Float64, ub::Float64)
     x ∩ Interval(lb, ub)
end
function bnd_kernel(x::MC{N,T}, lb::Float64, ub::Float64, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    cut_kernel(x, z)
end

"""
bnd(x::MC, lb, ub)

Sets the lower interval bound and the convex relaxation of `x` to a value of
at least `lb`. Sets the upper interval bound and the concave relaxation of `x`
to a value of at most `ub`. (Sub)gradients are adjusted appropriately.
"""
function bnd(x::MC{N,T}, lb::Float64, ub::Float64) where {N, T<:Union{NS,MV}}
    bnd_kernel(x, lb, ub, bnd(x.Intv, lb, ub))
end


function d_lower_bnd_grad(g, x::T, y::T) where T<:Number
    g[1] = one(T)
    g[2] = zero(T)
    return
end
function d_upper_bnd_grad(g, x::T, y::T) where T<:Number
    g[1] = one(T)
    g[2] = zero(T)
    return
end
function d_bnd_grad(g, x::T, y::T, z::T) where T<:Number
    g[1] = one(T)
    g[2] = zero(T)
    g[3] = zero(T)
    return
end
