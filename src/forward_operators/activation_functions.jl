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
    @assert α >= 0.0
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
    xLc = maxtanh(Interval(x.lo))
    xUc = maxtanh(Interval(x.hi))
    Interval(xLc.lo, xUc.hi)
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

# DEFINE PENTANH
@inline pentanh(x) = x > 0.0 ? tanh(x) : tanh(0.25*x)
@inline pentanh(x::Float64) = x > 0.0 ? tanh(x) : tanh(0.25*x)
function pentanh(x::Interval{Float64})
    (x.lo >= 0.0) && return tanh(x)
    (x.hi <= 0.0) && return tanh(0.25*x)
    lo_part = tanh(0.25*x)
    hi_part = tanh(x)
    Interval(lo_part.lo, hi_part.hi)
end
function pentanh_deriv(x::Float64)
    if x > 0.0
        return 1.0 - tanh(x)^2
    end
    0.25 - 0.25*tanh(0.25*x)^2
end
@inline function pentanh_env(x::Float64, y::Float64, z::Float64)
    (x - y) - (pentanh(x) - pentanh(y))/pentanh_deriv(x)
end
@inline function cv_pentanh(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(pentanh, pentanh_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return pentanh(x), pentanh_deriv(x), p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, pentanh_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, pentanh_env, xU, 0.0))
    end
    (x <= p) && (return pentanh(x), pentanh_deriv(x), p)
    return dline_seg(pentanh, pentanh_deriv, x, p, xU)..., p
end
@inline function cc_pentanh(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return pentanh(x), pentanh_deriv(x), p)
    (xU <= 0.0) && (return dline_seg(pentanh, pentanh_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, pentanh_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, pentanh_env, xL, 0.0))
    end
    (x <= p) && (return dline_seg(pentanh, pentanh_deriv, x, xL, p)..., p)
    return pentanh(x), pentanh_deriv(x), p
end

@inline sigmoid(x) = 1.0/(1.0 + exp(-x))
@inline sigmoid(x::Float64) = 1.0/(1.0 + exp(-x))
@inline sigmoid(x::Interval{Float64}) = 1.0/(1.0 + exp(-x))
@inline sigmoid_deriv(x::Float64) = sigmoid(x)*(1.0 - sigmoid(x))
@inline function sigmoid_env(x::Float64, y::Float64, z::Float64)
    (x - y) - (sigmoid(x) - sigmoid(y))/sigmoid_deriv(x)
end
@inline function cv_sigmoid(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(sigmoid, sigmoid_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return sigmoid(x), sigmoid_deriv(x), p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, sigmoid_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, sigmoid_env, xU, 0.0))
    end
    (x <= p) && (return sigmoid(x), sigmoid_deriv(x), p)
    return dline_seg(sigmoid, sigmoid_deriv, x, p, xU)..., p
end
@inline function cc_sigmoid(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return sigmoid(x), sigmoid_deriv(x), p)
    (xU <= 0.0) && (return dline_seg(sigmoid, sigmoid_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, sigmoid_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, sigmoid_env, xL, 0.0))
    end
    (x <= p) && (return dline_seg(sigmoid, sigmoid_deriv, x, xL, p)..., p)
    return sigmoid(x), sigmoid_deriv(x), p
end

@inline bisigmoid(x) = 1.0 - exp(-x)/(1 + exp(-x))
@inline bisigmoid(x::Float64) = 1.0 - exp(-x)/(1 + exp(-x))
@inline function bisigmoid(x::Interval{Float64})
    xLc = bisigmoid(Interval(x.lo))
    xUc = bisigmoid(Interval(x.hi))
    return Interval(xLc.hi, xUc.hi)
end
@inline bisigmoid_deriv(x::Float64) =  0.5*(1.0 + bisigmoid(x))*(1.0 - bisigmoid(x))
@inline function bisigmoid_env(x::Float64, y::Float64, z::Float64)
    (x - y) - (bisigmoid(x) - bisigmoid(y))/bisigmoid_deriv(x)
end
@inline function cv_bisigmoid(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(bisigmoid, bisigmoid_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return bisigmoid(x), bisigmoid_deriv(x), p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, bisigmoid_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, bisigmoid_env, xU, 0.0))
    end
    (x <= p) && (return bisigmoid(x), bisigmoid_deriv(x), p)
    return dline_seg(bisigmoid, bisigmoid_deriv, x, p, xU)..., p
end
@inline function cc_bisigmoid(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return bisigmoid(x), bisigmoid_deriv(x), p)
    (xU <= 0.0) && (return dline_seg(bisigmoid, bisigmoid_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, bisigmoid_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, bisigmoid_env, xL, 0.0))
    end
    (x <= p) && (return dline_seg(bisigmoid, bisigmoid_deriv, x, xL, p)..., p)
    return bisigmoid(x), bisigmoid_deriv(x), p
end

@inline softsign(x) = x/(1.0 + abs(x))
@inline softsign(x::Float64) = x/(1.0 + abs(x))
@inline function softsign(x::Interval{Float64})
    xLc = softsign(Interval(x.lo))
    xUc = softsign(Interval(x.hi))
    return Interval(xLc.hi, xUc.hi)
end
@inline softsign_deriv(x::Float64) = 1.0/(1.0 + abs(x))^2
@inline function softsign_env(x::Float64, y::Float64, z::Float64)
    (x - y) - (softsign(x) - softsign(y))/softsign_deriv(x)
end
@inline function cv_softsign(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(softsign, softsign_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return softsign(x), softsign_deriv(x), p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, softsign_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, softsign_env, xU, 0.0))
    end
    (x <= p) && (return softsign(x), softsign_deriv(x), p)
    return dline_seg(softsign, softsign_deriv, x, p, xU)..., p
end
@inline function cc_softsign(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return softsign(x), softsign_deriv(x), p)
    (xU <= 0.0) && (return dline_seg(softsign, softsign_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, softsign_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, softsign_env, xL, 0.0))
    end
    (x <= p) && (return dline_seg(softsign, softsign_deriv, x, xL, p)..., p)
    return softsign(x), softsign_deriv(x), p
end

gelu(x) = x*(1.0 + erf(x/sqrt(2)))/2.0

# define kernel and operator for sigmoid, bisigmoid, softsign, gelu
for expri in (:sigmoid, :bisigmoid, :softsign) #gelu
    expri_cv = Symbol("cv_"*String(expri))
    expri_cc = Symbol("cc_"*String(expri))
    expri_kernel = Symbol(String(expri)*"_kernel")
    eps_min = :xL
    eps_max = :xU
    @eval @inline function ($expri_kernel)(x::MC{N, T}, y::Interval{Float64},
                            cv_p::Float64, cc_p::Float64) where {N,T<:Union{NS,MV}}
        xL = x.Intv.lo
        xU = x.Intv.hi
        midcv, cv_id = mid3(x.cc, x.cv, $eps_min)
        midcc, cc_id = mid3(x.cc, x.cv, $eps_max)
        cv, dcv, cv_p = $(expri_cv)(midcv, xL, xU, cv_p)
        cc, dcc, cc_p = $(expri_cc)(midcc, xL, xU, cc_p)
        cv_grad = mid_grad(x.cv_grad, x.cc_grad, cv_id)*dcv
        cc_grad = mid_grad(x.cv_grad, x.cc_grad, cc_id)*dcc
        cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
        return MC{N, T}(cv, cc, y, cv_grad, cc_grad, x.cnst), cv_p, cc_p
    end
    @eval @inline function ($expri)(x::MC{N,T}) where {N, T<:RelaxTag}
        z, tp1, tp2 = ($expri_kernel)(x, ($expri)(x.Intv), Inf, Inf)
        return z
    end
end

swish(b, x) = x/(1.0 + exp(-b*x))
swish_deriv(b, x) = (exp(-b*x)*(b*x + 1.0) + 1.0)/(1.0 + exp(-b*x))^2

swish1(x) = swish(1.0, x)
swish1_deriv(x) = dswish(1.0, x)

# linear-convex regions or concave
elu(α, x) = x > 0.0 ? x : α*(exp(x) - 1.0)
selu(α, λ, x) = λ*elu(α, x)
