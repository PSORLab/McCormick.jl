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
# Contains definitions of commonly used activation functions:
# relu, parametric relu, leaky relu, XXX.
#############################################################################

# RELU DEFINITION
function relu_kernel(x::MC{N,T}, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    max_kernel(x, 0.0, z)
end
function relu(x::MC{N,T}) where {N, T<:Union{NS,MV}}
    relu_kernel(x, max(x.Intv, 0.0))
end
relu(x) = max(x, 0.0)

# PARAMETRIC RELU DEFINITION
function param_relu(x::Interval{Float64}, α::Float64)
    xL = x.lo
    xU = x.hi
    (xL < 0.0) && (xL *= α)
    (xU < 0.0) && (xU *= α)
    return Interval{Float64}(xL, xU)
end
param_relu(x, α) = x > 0.0 ? x : α*x
param_relu(x::Float64, α::Float64) = x > 0.0 ? x : α*x
param_relu_deriv(x::Float64, α::Float64) = x > 0.0 ? 1.0 : α
function param_relu_kernel(x::MC{N,T}, α::Float64, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    xLc = z.lo
    xUc = z.hi
    xL = x.Intv.lo
    xU = x.Intv.hi
    midcv = mid3v(x.cv, x.cc, xL)
    midcc = mid3v(x.cv, x.cc, xU)
    dcc = (xUc > xLc) ? (xUc - xLc)/(xU - xL) : 0.0
    convex = param_relu(midcv, α)
    concave = dcc*(midcc - xL) + xLc
    concave_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    convex_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*param_relu_deriv(midcv, α)
    convex, concave, convex_grad, concave_grad = cut(xLc, xUc, convex, concave, convex_grad, concave_grad)
    return MC{N, T}(convex, concave, z, convex_grad, concave_grad, x.cnst
end
@inline param_relu(x::MC, α::Float64) = param_relu_kernel(x, α, param_relu(x.Intv, α))

# LEAKY RELU DEFINITION
@inline leaky_relu_kernel(x::MC, z::Interval{Float64}) = param_relu_kernel(x, 0.01)
@inline leaky_relu(x::MC) = leaky_relu_kernel(x, param_relu(x.Intv, 0.01))
@inline leaky_relu(x) = leaky_relu(x, 0.01)

# DEFINE MAXSIG
@inline maxsig(x) = max(x, 1.0/(1.0 + exp(-x)))
@inline maxsig(x::Float64) = max(x, 1.0/(1.0 + exp(-x)))
@inline maxsig(x::Interval{Float64}) = max(x, 1.0/(1.0 + exp(-x)))
@inline function maxsig_deriv(x::Float64)
    if x > 1.0/(exp(-x) + 1.0)
        return 1.0
    end
    return exp(x)/(exp(x) + 1.0)^2
end
function maxsig_kernel(x::MC{N,T}, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    xLc = z.lo
    xUc = z.hi
    xL = x.Intv.lo
    xU = x.Intv.hi
    midcv = mid3v(x.cv, x.cc, xL)
    midcc = mid3v(x.cv, x.cc, xU)
    dcc = (xUc > xLc) ? (xUc - xLc)/(xU - xL) : 0.0
    convex = maxsig(midcv)
    concave = dcc*(midcc - xL) + xLc
    concave_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    convex_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*maxsig_deriv(midcv, α)
    convex, concave, convex_grad, concave_grad = cut(xLc, xUc, convex, concave, convex_grad, concave_grad)
    return MC{N, T}(convex, concave, z, convex_grad, concave_grad, x.cnst
end
maxsig(x::MC{N,T}) where {N, T<:Union{NS,MV}} = maxsig_kernel(x, maxsig(x.Intv))

# DEFINE MAXTANH
@inline maxtanh(x) = max(x, tanh(x))
@inline maxtanh(x::Float64) = max(x, tanh(x))
@inline function maxtanh(x::Interval{Float64})
    xL = x.lo
    xU = x.hi
    Interval(max(xL, tanh(xL)), max(xU, tanh(xU)))
end
@inline function maxtanh_deriv(x::Float64)
    if x > tanh(x)
        return 1.0
    end
    return sech(x)^2
end
function maxtanh_kernel(x::MC{N,T}, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    xLc = z.lo
    xUc = z.hi
    xL = x.Intv.lo
    xU = x.Intv.hi
    midcv = mid3v(x.cv, x.cc, xL)
    midcc = mid3v(x.cv, x.cc, xU)
    dcc = (xUc > xLc) ? (xUc - xLc)/(xU - xL) : 0.0
    convex = maxtanh(midcv)
    concave = dcc*(midcc - xL) + xLc
    concave_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    convex_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*maxtanh_deriv(midcv, α)
    convex, concave, convex_grad, concave_grad = cut(xLc, xUc, convex, concave, convex_grad, concave_grad)
    return MC{N, T}(convex, concave, z, convex_grad, concave_grad, x.cnst
end
maxtanh(x::MC{N,T}) where {N, T<:Union{NS,MV}} = maxtanh_kernel(x, maxtanh(x.Intv))

# DEFINE SOFTPLUS
@inline softplus(x) = log(1.0 + exp(x))
@inline softplus(x::Float64) = log(1.0 + exp(x))
@inline softplus(x::Interval{Float64}) = log(1.0 + exp(x))
@inline softplus_deriv(x::Float64) = exp(x)/(exp(x) + 1.0)
function softplus_kernel(x::MC{N,T}, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    xLc = z.lo
    xUc = z.hi
    xL = x.Intv.lo
    xU = x.Intv.hi
    midcv = mid3v(x.cv, x.cc, xL)
    midcc = mid3v(x.cv, x.cc, xU)
    dcc = (xUc > xLc) ? (xUc - xLc)/(xU - xL) : 0.0
    convex = softplus(midcv)
    concave = dcc*(midcc - xL) + xLc
    concave_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    convex_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*softplus_deriv(midcv, α)
    convex, concave, convex_grad, concave_grad = cut(xLc, xUc, convex, concave, convex_grad, concave_grad)
    return MC{N, T}(convex, concave, z, convex_grad, concave_grad, x.cnst
end
softplus(x::MC{N,T}) where {N, T<:Union{NS,MV}} = softplus_kernel(x, softplus(x.Intv))

# convexoconcave
pentanh(x) = x > 0 ? tanh(x) : tanh(0.25*x)
sigmoid(x) = 1.0/(1.0 + exp(-x))
bisigmoid(x) = 1.0 - exp(-x)/(1 + exp(-x))
Softsign(x) = x/(1.0 + abs(x))








swish(b, x) = x/(1.0 + exp(-b*x))
swish_deriv(b, x) = (exp(-b*x)*(b*x + 1.0) + 1.0)/(1.0 + exp(-b*x))^2

swish1(x) = swish(1.0, x)
swish1_deriv(x) = dswish(1.0, x)

gaussian(x) = exp(-x^2)
gaussian_deriv(x) = -2.0*x*exp(-x^2)

gaussian(b, x) = exp(-b*x^2)
gaussian_deriv(b, x) =  -2.0*b*x*exp(-b*x^2)

bent(x) = (sqrt(x^2 + 1.0) - 1.0)/2.0
bent_deriv(x) = x/(2.0*sqrt(x^2 + 1.0)) + 1.0

function sqnl(x)
    (x > 2.0) && return 1.0
    (2.0 >= x >= 0.0) && return x - (x^2)/4.0
    (0.0 > x >= -2.0) && return x + (x^2)/4.0
    return -1.0
end
function sqrbf(x)
    (abs(x) <= 1.0) && return 1.0 - (x^2)/2.0
    (abs(x) >= 2.0) && return 0.0
    return ((2.0 - abs(x))^2)/2.0
end

# concavoconvex ? convex...
gelu(x) = x*(1.0 + erf(x/sqrt(2)))/2.0

# linear-convex regions or concave
elu(α, x) = x > 0.0 ? x : α*(exp(x) - 1.0)

# nonconvex
cosid(x) = cos(x) - x
minsin(x) = min(x, sin(x))
arctid(x) = max(tan(x)^2) - x


selu(α, λ, x) = λ*elu(α, x)

#multiquad(α, x) = sqrt(α^2 + x^2)
#tpspline(x) = (x^2)*log(x)
# ((c/pi)^(1/2))*exp(-cx^2)
