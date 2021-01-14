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
# relu, parametric relu, leaky relu, maxsig, elu, selu, gelu, sigmoid,
# bisigmoid, swish1, maxtanh, pentanh, softsign, softplus.
#############################################################################

# RELU DEFINITION
"""
relu

The Rectified Linear Unit (ReLU) activation function `relu(x) = max(x, 0.0)`.
"""
relu(x) = max(x, 0.0)
relu_deriv(x) = x > 0.0 ? 1.0 : 0.0
relu_deriv2(x) =  0.0
function relu_kernel(x::MC{N,T}, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    max_kernel(x, 0.0, z)
end
function relu(x::MC{N,T}) where {N, T<:Union{NS,MV}}
    relu_kernel(x, max(x.Intv, 0.0))
end

# PARAMETRIC RELU DEFINITION
"""
param_relu

The parametric Rectified Linear Unit activation function `param_relu(x, α) = (max(x, αx)` with α in [0,1].
"""
param_relu(x, α) = x > 0.0 ? x : α*x
function param_relu(x::Interval{Float64}, α::Float64)
    xL = x.lo
    xU = x.hi
    (xL < 0.0) && (xL *= α)
    (xU < 0.0) && (xU *= α)
    return Interval{Float64}(xL, xU)
end
param_relu(x::Float64, α::Float64) = x > 0.0 ? x : α*x
param_relu_deriv(x::Float64, α::Float64) = x > 0.0 ? 1.0 : α

function param_relu_grad(g, x::Float64, α::Float64)
    if x > 0.0
        g[1] = 1.0
        g[2] = 0.0
    end
    g[1] = α
    g[2] = x
    nothing
end
function param_relu_kernel(x::MC{N,T}, α::Float64, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    @assert α >= 0.0
    xLc = z.lo
    xUc = z.hi
    xL = x.Intv.lo
    xU = x.Intv.hi
    midcv, cv_id = mid3(x.cc, x.cv, xL)
    midcc, cc_id = mid3(x.cc, x.cv, xU)
    dcc = (xUc > xLc) ? (xUc - xLc)/(xU - xL) : 0.0
    convex = param_relu(midcv, α)
    concave = dcc*(midcc - xL) + xLc
    concave_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    convex_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*param_relu_deriv(midcv, α)
    convex, concave, convex_grad, concave_grad = cut(xLc, xUc, convex, concave, convex_grad, concave_grad)
    return MC{N, T}(convex, concave, z, convex_grad, concave_grad, x.cnst)
end
@inline param_relu(x::MC, α::Float64) = param_relu_kernel(x, α, param_relu(x.Intv, α))

# LEAKY RELU DEFINITION
"""
leaky_relu

The leaky Rectified Linear Unit activation function `leaky_relu(x) = max(x, 0.01x)`.
"""
@inline leaky_relu(x) = param_relu(x, 0.01)
@inline leaky_relu_kernel(x::MC, z::Interval{Float64}) = param_relu_kernel(x, 0.01, z)
@inline leaky_relu(x::MC) = leaky_relu_kernel(x, param_relu(x.Intv, 0.01))
@inline leaky_relu_deriv(x) = x > 0.0 ? 1.0 : 0.01
@inline leaky_relu_deriv2(x) = 0.0

# DEFINE MAXSIG
"""
maxsig

The `maxsig` activation function  `maxsig(x) = max(x, 1.0/(1.0 + exp(-x)))`.
"""
@inline maxsig(x) = max(x, 1.0/(1.0 + exp(-x)))
@inline maxsig(x::Float64) = max(x, 1.0/(1.0 + exp(-x)))
@inline maxsig(x::Interval{Float64}) = max(x, 1.0/(1.0 + exp(-x)))
@inline function maxsig_deriv(x::Float64)
    if x > 1.0/(exp(-x) + 1.0)
        return 1.0
    end
    return exp(-x)/(exp(-x) + 1.0)^2
end
@inline function maxsig_deriv2(x::Float64)
    if x > 1.0/(exp(-x) + 1.0)
        return 0.0
    end
    return exp(x)*(1.0 - exp(x))/(exp(x) + 1.0)^3 #
end
function maxsig_kernel(x::MC{N,T}, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    xLc = z.lo
    xUc = z.hi
    xL = x.Intv.lo
    xU = x.Intv.hi
    midcv, cv_id = mid3(x.cc, x.cv, xL)
    midcc, cc_id = mid3(x.cc, x.cv, xU)
    dcc = (xUc > xLc) ? (xUc - xLc)/(xU - xL) : 0.0
    convex = maxsig(midcv)
    concave = dcc*(midcc - xL) + xLc
    concave_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    convex_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*maxsig_deriv(midcv)
    convex, concave, convex_grad, concave_grad = cut(xLc, xUc, convex, concave, convex_grad, concave_grad)
    return MC{N, T}(convex, concave, z, convex_grad, concave_grad, x.cnst)
end
maxsig(x::MC{N,T}) where {N, T<:Union{NS,MV}} = maxsig_kernel(x, maxsig(x.Intv))

# DEFINE ELU
"""
elu

The Exponential Linear Unit (ELU) activation function  `elu(x, α) = x > 0 ? x : α*(exp(x) - 1.0)`.
"""
@inline elu(x, α) = x > 0 ? x : α*(exp(x) - 1.0)
@inline elu(x::Float64, α::Float64) = x > 0 ? x : α*(exp(x) - 1.0)
function elu(x::Interval{Float64}, α::Float64)
    xL = x.lo
    xU = x.hi
    if xU < 0.0
        return α*(exp(x) - 1.0)
    elseif xL > 0.0
        return x
    end
    xLIntv = α*(exp(x) - 1.0)
    Interval(xLIntv.lo, x.hi)
end
@inline elu_deriv(x::Float64, α::Float64) = x > 0.0 ? 1.0 : α*exp(x)
function elu_grad(g, x::Float64, α::Float64)
    g[1] = elu_deriv(x, α)
    g[2] = x > 0.0 ? 0.0 : exp(x)
    nothing
end
function elu_kernel(x::MC{N,T}, α::Float64, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    xLc = z.lo
    xUc = z.hi
    xL = x.Intv.lo
    xU = x.Intv.hi
    midcv, cv_id = mid3(x.cc, x.cv, xL)
    midcc, cc_id = mid3(x.cc, x.cv, xU)
    dcc = (xUc > xLc) ? (xUc - xLc)/(xU - xL) : 0.0
    convex = elu(midcv, α)
    concave = dcc*(midcc - xL) + xLc
    concave_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    convex_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*elu_deriv(midcv, α)
    convex, concave, convex_grad, concave_grad = cut(xLc, xUc, convex, concave, convex_grad, concave_grad)
    return MC{N,T}(convex, concave, z, convex_grad, concave_grad, x.cnst)
end
elu(x::MC{N,T}, α::Float64) where {N, T<:Union{NS,MV}} = elu_kernel(x, α, elu(x.Intv, α))


"""
selu

The Scaled Exponential Linear Unit (SELU) activation function  `selu(x, α, λ) = λ*elu(x, α)`.
"""
selu(x, α, λ) = λ*elu(x, α)
function selu_grad(g, x::Float64, α::Float64, λ::Float64)
    g[1] = λ*elu_deriv(x, α)
    g[2] = λ*(x > 0.0 ? 0.0 : exp(x))
    g[3] = elu(x, α)
    nothing
end

"""
maxtanh

The `maxtanh` activation function  `maxtanh(x) = max(x, tanh(x))`.
"""
@inline maxtanh(x) = max(x, tanh(x))
@inline maxtanh(x::Float64) = max(x, tanh(x))
@inline function maxtanh(x::Interval{Float64})
    xLintv = Interval(x.lo)
    xUintv = Interval(x.hi)
    xLc = max(xLintv, tanh(xLintv))
    xUc = max(xUintv, tanh(xUintv))
    Interval(xLc.lo, xUc.hi)
end
@inline function maxtanh_deriv(x::Float64)
    if x > tanh(x)
        return 1.0
    end
    return sech(x)^2
end
@inline function maxtanh_deriv2(x::Float64)
    if x > tanh(x)
        return 0.0
    end
    return -2.0*tanh(x)*sech(x)^2
end
function maxtanh_kernel(x::MC{N,T}, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    xLc = z.lo
    xUc = z.hi
    xL = x.Intv.lo
    xU = x.Intv.hi
    midcv, cv_id = mid3(x.cc, x.cv, xL)
    midcc, cc_id = mid3(x.cc, x.cv, xU)
    dcc = (xUc > xLc) ? (xUc - xLc)/(xU - xL) : 0.0
    convex = maxtanh(midcv)
    concave = dcc*(midcc - xL) + xLc
    concave_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    convex_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*maxtanh_deriv(midcv)
    convex, concave, convex_grad, concave_grad = cut(xLc, xUc, convex, concave, convex_grad, concave_grad)
    return MC{N, T}(convex, concave, z, convex_grad, concave_grad, x.cnst)
end
maxtanh(x::MC{N,T}) where {N, T<:Union{NS,MV}} = maxtanh_kernel(x, maxtanh(x.Intv))

"""
softplus

The `softplus` activation function  `softplus(x) = log(1.0 + exp(x))`.
"""
@inline softplus(x) = log(1.0 + exp(x))
@inline softplus(x::Float64) = log(1.0 + exp(x))
@inline softplus(x::Interval{Float64}) = log(1.0 + exp(x))
@inline softplus_deriv(x::Float64) = 1.0/(exp(-x) + 1.0)
@inline softplus_deriv2(x::Float64) = exp(-x)/(exp(-x) + 1.0)^2
function softplus_kernel(x::MC{N,T}, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    xLc = z.lo
    xUc = z.hi
    xL = x.Intv.lo
    xU = x.Intv.hi
    midcv, cv_id = mid3(x.cc, x.cv, xL)
    midcc, cc_id = mid3(x.cc, x.cv, xU)
    dcc = (xUc > xLc) ? (xUc - xLc)/(xU - xL) : 0.0
    convex = softplus(midcv)
    concave = dcc*(midcc - xL) + xLc
    concave_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    convex_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*softplus_deriv(midcv)
    convex, concave, convex_grad, concave_grad = cut(xLc, xUc, convex, concave, convex_grad, concave_grad)
    return MC{N, T}(convex, concave, z, convex_grad, concave_grad, x.cnst)
end
softplus(x::MC{N,T}) where {N, T<:Union{NS,MV}} = softplus_kernel(x, softplus(x.Intv))

"""
pentanh

The `pentanh` activation function `pentanh(x) = x > 0.0 ? tanh(x) : tanh(0.25*x)`.
"""
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
function pentanh_deriv2(x::Float64)
    if x > 0.0
        return -2.0*tanh(x)*sech(2)^2
    end
    -0.125*tanh(0.25*x)*sech(0.25*x)^2
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

"""
sigmoid

The `sigmoid` activation function `sigmoid(x) = 1.0/(1.0 + exp(-x))`.
"""
@inline sigmoid(x) = 1.0/(1.0 + exp(-x))
@inline sigmoid(x::Float64) = 1.0/(1.0 + exp(-x))
@inline sigmoid(x::Interval{Float64}) = 1.0/(1.0 + exp(-x))
@inline sigmoid_deriv(x::Float64) = sigmoid(x)*(1.0 - sigmoid(x))
@inline sigmoid_deriv2(x::Float64) = 2.0*exp(-2.0*x)/sigmoid(x)^3 - exp(-x)/sigmoid(x)^2
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

"""
bisigmoid

The `bisigmoid` activation function `bisigmoid(x) = (1.0 - exp(-x))/(1.0 + exp(-x))`.
"""
@inline bisigmoid(x) = (1.0 - exp(-x))/(1.0 + exp(-x))
@inline bisigmoid(x::Float64) = (1.0 - exp(-x))/(1.0 + exp(-x))
@inline function bisigmoid(x::Interval{Float64})
    xLintv = Interval(x.lo)
    xUintv = Interval(x.hi)
    xLc = (1.0 - exp(-xLintv))/(1.0 + exp(-xLintv))
    xUc = (1.0 - exp(-xUintv))/(1.0 + exp(-xUintv))
    return Interval(xLc.hi, xUc.hi)
end
@inline bisigmoid_deriv(x::Float64) = 2.0*exp(x)/(exp(x) + 1.0)^2
@inline function bisigmoid_deriv2(x::Float64)
    term1 = exp(-x)/(exp(-x) + 1.0)
    term2 = 2.0*exp(-2.0*x)/(exp(-x) + 1.0)^2
    term3 = (1.0 - exp(-x))*(term2 - term1)/(exp(-x) + 1.0)
    return term3 - term1 + term2
end
@inline function bisigmoid_env(x::Float64, y::Float64, z::Float64)
    bisigmoid(y) - bisigmoid(x) - bisigmoid_deriv(x)*(y - x)
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

"""
softsign

The `softsign` activation function `softsign(x) = x/(1.0 + abs(x))`.
"""
@inline softsign(x) = x/(1.0 + abs(x))
@inline softsign(x::Float64) = x/(1.0 + abs(x))
@inline function softsign(x::Interval{Float64})
    xLintv = Interval(x.lo)
    xUintv = Interval(x.hi)
    xLc = xLintv/(1.0 + abs(xLintv))
    xUc = xUintv/(1.0 + abs(xUintv))
    return Interval(xLc.hi, xUc.hi)
end
@inline softsign_deriv(x::Float64) = 1.0/(1.0 + abs(x))^2
@inline function softsign_deriv2(x::Float64)
    if x >= 0.0
        xp1 = 1.0 + x
        return 2.0*(x*xp1^(-3) - xp1^(-2))
    end
    xm1 = 1.0 - x
    return 2.0*(x*xm1^(-3) + xm1^(-2))
end
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

const GELU_MIN = -0.751791524693564457457904946779522
const GELU_2D_ROOT1 = -sqrt(2)
const GELU_2D_ROOT2 = sqrt(2)

const SWISH1_MIN = -1.27846454276107379510935873902298015543947748
const SWISH1_2D_ROOT1 = -2.399357280515467667832739697282283888523
const SWISH1_2D_ROOT2 = 2.399357280515467667832739697282283888523

"""
gelu

The Gaussian Error Linear Unit `gelu` activation function `gelu(x) = x/(1.0 + abs(x))`.
"""
@inline gelu(x) = x*(1.0 + erf(x/sqrt(2)))/2.0
@inline gelu(x::Float64) = x*(1.0 + erf(x/sqrt(2)))/2.0
@inline function gelu(x::Interval{Float64})
    xLintv = Interval(x.lo)
    xUintv = Interval(x.hi)
    xLc = xLintv*(1.0 + erf(xLintv/sqrt(2)))/2.0
    xUc = xUintv*(1.0 + erf(xUintv/sqrt(2)))/2.0
    if x.hi < GELU_MIN
        xLcv = xUc.lo
        xUcv = xLc.hi
    elseif GELU_MIN < x.lo
        xLcv = xLc.lo
        xUcv = xUc.hi
    else
        xLcv = GELU_MIN
        xUcv = max(xLc.hi, xUc.hi)
    end
    return Interval(xLcv, xUcv)
end
function gelu_deriv(x::Float64)
    0.5*(1.0 + erf(x/sqrt(2))) + (x/sqrt(2*pi))*exp((-x^2)/2.0)
end
function gelu_deriv2(x::Float64)
    (0.797885 - 0.398942*x^2)*exp((-x^2)/2.0)
end
function gelu_env(x::Float64, y::Float64, z::Float64)
    (y - x)*gelu_deriv(x) + gelu(x) - gelu(y)
end
function gelu_denv(x::Float64, y::Float64, z::Float64)
    (y - x)*gelu_deriv2(x)
end

@inline function gelu_envm(x::Float64, y::Float64, z::Float64)
    gelu_deriv(y)*(x - y) - (gelu(x) - gelu(y))
end

@inline function gelu_rt1(x::Float64, y::Float64, z::Float64)
    gelu_deriv(y)*(x - y) + gelu(y) -gelu(x)
end

@inline function gelu_rt1_deriv(x::Float64, y::Float64, z::Float64)
    gelu_deriv(y) - gelu_deriv(x)
end

@inline function cc_gelu(x::Float64, xL::Float64, xU::Float64, p1::Float64, p2::Float64)
    # Single convexity regions
    if xL >= GELU_2D_ROOT2
        return gelu(x), gelu_deriv(x), p1, p2
    elseif xU <= GELU_2D_ROOT1
        return gelu(x), gelu_deriv(x), p1, p2
    elseif (GELU_2D_ROOT1 <= xL) && (xU <= GELU_2D_ROOT2)
        return dline_seg(gelu, gelu_deriv, x, xL, xU)..., p1, p2
    end

    if xL < GELU_2D_ROOT1
        p2, flag = newton(0.0, GELU_MIN, 0.0, gelu_rt1, gelu_rt1_deriv, xL, 0.0)
        flag && (p2 = golden_section(GELU_2D_ROOT1, 0.0, gelu_rt1, xL, 0.0))
        if p2 > xU
            if p1 === Inf
                p1, flag = secant(xL, GELU_2D_ROOT1, xL, GELU_2D_ROOT1, gelu_env, xU, 0.0)
                flag && (p1 = golden_section(xL, GELU_2D_ROOT1, gelu_env, xU, 0.0))
            end
            (x >= p1) && (return dline_seg(gelu, gelu_deriv, x, p1, xU)..., p1, p2)
            return swish1(x), swish1_deriv(x), p1, p2
        else
            return dline_seg(gelu, gelu_deriv, x, xL, xU)..., p1, p2
        end
    end

    if xL > GELU_2D_ROOT1
        p2, flag = newton(0.5*(GELU_MIN + GELU_2D_ROOT2), GELU_MIN, GELU_2D_ROOT2, gelu_rt1, gelu_rt1_deriv, xU, 0.0)
        flag && (p2 = golden_section(GELU_MIN, GELU_2D_ROOT2, gelu_rt1, xU, 0.0))
        if p2 < xL
            if p1 === Inf
                p1, flag = secant(GELU_2D_ROOT2, xU, GELU_2D_ROOT2, xU, gelu_env, xL, 0.0)
                flag && (p1 = golden_section(GELU_2D_ROOT2, xU, gelu_env, xL, 0.0))
            end
            (x >= p1) && (return dline_seg(gelu, gelu_deriv, x, p1, xU)..., p1, p2)
            return gelu(x), gelu_deriv(x), p1, p2
        else
            return dline_seg(gelu, gelu_deriv, x, xL, xU)..., p1, p2
        end
    end
end
@inline function cv_gelu(x::Float64, xL::Float64, xU::Float64, p1::Float64, p2::Float64)

    # Single convexity regions
    if xL >= GELU_2D_ROOT2
        return dline_seg(gelu, gelu_deriv, x, xL, xU)..., p1, p2
    elseif xU <= GELU_2D_ROOT1
        return dline_seg(gelu, gelu_deriv, x, xL, xU)..., p1, p2
    elseif (GELU_2D_ROOT1 <= xL) && (xU <= GELU_2D_ROOT2)
        return gelu(x), gelu_deriv(x), p1, p2
    end

    if xL < GELU_2D_ROOT1
        if p1 === Inf
            p1, flag = newton(0.5*(GELU_2D_ROOT1 + GELU_MIN), GELU_2D_ROOT1, GELU_MIN, gelu_env, gelu_denv, xL, 0.0)
            flag && (p1 = golden_section(GELU_2D_ROOT1, GELU_MIN, gelu_env, xL, 0.0))
        end
    else
        p1 = -Inf
    end

    if xU > GELU_2D_ROOT2
        if p2 === Inf
            p2, flag = newton(0.5*(GELU_MIN + GELU_2D_ROOT2), GELU_MIN, GELU_2D_ROOT2, gelu_env, gelu_denv, xU, 0.0)
            flag && (p2 = golden_section(GELU_MIN, GELU_2D_ROOT2, gelu_env, xU, 0.0))
        end
    else
        p2 = Inf
    end

    if x < p1
        return dline_seg(gelu, gelu_deriv, x, xL, p1)..., p1, p2
    elseif x > p2
        return dline_seg(gelu, gelu_deriv, x, p2, xU)..., p1, p2
    end
    return gelu(x), gelu_deriv(x), p1, p2
end

"""
swish1

The Swish-1 activation function `swish1(x) = x/(1.0 + exp(-x))`.
"""
@inline swish1(x) = x/(1.0 + exp(-x))
@inline swish1(x::Float64) = x/(1.0 + exp(-x))
@inline function swish1(x::Interval{Float64})
    xLintv = Interval(x.lo)
    xUintv = Interval(x.hi)
    xLc = xLintv/(1.0 + exp(-xLintv))
    xUc = xUintv/(1.0 + exp(-xUintv))
    if x.hi < SWISH1_MIN
        xLcv = xUc.lo
        xUcv = xLc.hi
    elseif SWISH1_MIN < x.lo
        xLcv = xLc.lo
        xUcv = xUc.hi
    else
        xLcv = SWISH1_MIN
        xUcv = max(xLc.hi, xUc.hi)
    end
    return Interval(xLcv, xUcv)
end
@inline function swish1_deriv(x::Float64)
    sigmoid(x) + x*sigmoid_deriv(x)
end
@inline function swish1_deriv2(x::Float64)
    frac1 = 2.0*exp(-2.0*x)/(exp(-x) + 1.0)^3
    frac2 = exp(-x)/(exp(-x) + 1.0)^2
    2.0*exp(-x)/(exp(-x) + 1.0)^2 + (frac1 - frac2)*x
end
@inline function swish1_env(x::Float64, y::Float64, z::Float64)
    swish1_deriv(x)*(y - x) + swish1(x) - swish1(y)
end
@inline function swish1_denv(x::Float64, y::Float64, z::Float64)
    swish1_deriv2(x)*(y - x)
end

@inline function swish1_envm(x::Float64, y::Float64, z::Float64)
    swish1_deriv(y)*(x - y) - (swish1(x) - swish1(y))
end

@inline function swish_rt1(x::Float64, y::Float64, z::Float64)
    swish1_deriv(y)*(x - y) + swish1(y) - swish1(x)
end
@inline function swish_rt1_deriv(x::Float64, y::Float64, z::Float64)
    swish1_deriv(y) - swish1_deriv(x)
end

@inline function cc_swish1(x::Float64, xL::Float64, xU::Float64, p1::Float64, p2::Float64)
    # Single convexity regions
    if xL >= SWISH1_2D_ROOT2
        return swish1(x), swish1_deriv(x), p1, p2
    elseif xU <= SWISH1_2D_ROOT1
        return swish1(x), swish1_deriv(x), p1, p2
    elseif (SWISH1_2D_ROOT1 <= xL) && (xU <= SWISH1_2D_ROOT2)
        return dline_seg(swish1, swish1_deriv, x, xL, xU)..., p1, p2
    end

    if xL < SWISH1_2D_ROOT1
        p2, flag = newton(0.0, SWISH1_MIN, 0.0, swish_rt1, swish_rt1_deriv, xL, 0.0)
        flag && (p2 = golden_section(SWISH1_2D_ROOT1, 0.0, swish_rt1, xL, 0.0))
        if p2 > xU
            if p1 === Inf
                p1, flag = secant(xL, SWISH1_2D_ROOT1, xL, SWISH1_2D_ROOT1, swish1_env, xU, 0.0)
                flag && (p1 = golden_section(xL, SWISH1_2D_ROOT1, swish1_env, xU, 0.0))
            end
            (x >= p1) && (return dline_seg(swish1, swish1_deriv, x, p1, xU)..., p1, p2)
            return swish1(x), swish1_deriv(x), p1, p2
        else
            return dline_seg(swish1, swish1_deriv, x, xL, xU)..., p1, p2
        end
    end

    if xL > SWISH1_2D_ROOT1
        p2, flag = newton(0.5*(SWISH1_MIN + SWISH1_2D_ROOT2), SWISH1_MIN, SWISH1_2D_ROOT2, swish_rt1, swish_rt1_deriv, xU, 0.0)
        flag && (p2 = golden_section(SWISH1_MIN, SWISH1_2D_ROOT2, swish_rt1, xU, 0.0))
        if p2 < xL
            if p1 === Inf
                p1, flag = secant(SWISH1_2D_ROOT2, xU, SWISH1_2D_ROOT2, xU, swish1_env, xL, 0.0)
                flag && (p1 = golden_section(SWISH1_2D_ROOT2, xU, swish1_env, xL, 0.0))
            end
            (x >= p1) && (return dline_seg(swish1, swish1_deriv, x, p1, xU)..., p1, p2)
            return swish1(x), swish1_deriv(x), p1, p2
        else
            return dline_seg(swish1, swish1_deriv, x, xL, xU)..., p1, p2
        end
    end
end
@inline function cv_swish1(x::Float64, xL::Float64, xU::Float64, p1::Float64, p2::Float64)

    # Single convexity regions
    if xL >= SWISH1_2D_ROOT2
        return dline_seg(swish1, swish1_deriv, x, xL, xU)..., p1, p2
    elseif xU <= SWISH1_2D_ROOT1
        return dline_seg(swish1, swish1_deriv, x, xL, xU)..., p1, p2
    elseif (SWISH1_2D_ROOT1 <= xL) && (xU <= SWISH1_2D_ROOT2)
        return swish1(x), swish1_deriv(x), p1, p2
    end

    if xL < SWISH1_2D_ROOT1
        if p1 === Inf
            p1, flag = newton(0.5*(SWISH1_2D_ROOT1 + SWISH1_MIN), SWISH1_2D_ROOT1, SWISH1_MIN, swish1_env, swish1_denv, xL, 0.0)
            flag && (p1 = golden_section(SWISH1_2D_ROOT1, SWISH1_MIN, swish1_env, xL, 0.0))
        end
    else
        p1 = -Inf
    end

    if xU > SWISH1_2D_ROOT2
        if p2 === Inf
            p2, flag = newton(0.5*(SWISH1_MIN + SWISH1_2D_ROOT2), SWISH1_MIN, SWISH1_2D_ROOT2, swish1_env, swish1_denv, xU, 0.0)
            flag && (p2 = golden_section(SWISH1_MIN, SWISH1_2D_ROOT2, swish1_env, xU, 0.0))
        end
    else
        p2 = Inf
    end

    if x < p1
        return dline_seg(swish1, swish1_deriv, x, xL, p1)..., p1, p2
    elseif x > p2
        return dline_seg(swish1, swish1_deriv, x, p2, xU)..., p1, p2
    end
    return swish1(x), swish1_deriv(x), p1, p2
end

# define kernel and operator for sigmoid, bisigmoid, softsign, gelu
for expri in (:pentanh, :sigmoid, :bisigmoid, :softsign)
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

for expri in (:swish1, :gelu)
    expri_cv = Symbol("cv_"*String(expri))
    expri_cc = Symbol("cc_"*String(expri))
    expri_kernel = Symbol(String(expri)*"_kernel")
    if expri == swish1
        eps_min = :(SWISH1_MIN < xL ? xL : (SWISH1_MIN > xU ? xU : SWISH1_MIN))
        eps_max = :(swish1(xL) < swish1(xU) ? xU : xL)
    else
        eps_min = :(GELU_MIN > xU ? xU : (GELU_MIN < xL ? xL : GELU_MIN))
        eps_max = :(gelu(xL) < gelu(xU) ? xU : xL)
    end
    @eval @inline function ($expri_kernel)(x::MC{N, T}, y::Interval{Float64},
                            cv_p::Float64, cc_p::Float64, cv_p2::Float64,
                            cc_p2::Float64) where {N,T<:Union{NS,MV}}
        xL = x.Intv.lo
        xU = x.Intv.hi
        midcv, cv_id = mid3(x.cc, x.cv, $eps_min)
        midcc, cc_id = mid3(x.cc, x.cv, $eps_max)
        cv, dcv, cv_p, cv_p2 = $(expri_cv)(midcv, xL, xU, cv_p, cv_p2)
        cc, dcc, cc_p, cc_p2 = $(expri_cc)(midcc, xL, xU, cc_p, cc_p2)
        cv_grad = mid_grad(x.cv_grad, x.cc_grad, cv_id)*dcv
        cc_grad = mid_grad(x.cv_grad, x.cc_grad, cc_id)*dcc
        cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
        return MC{N, T}(cv, cc, y, cv_grad, cc_grad, x.cnst), cv_p, cc_p, cv_p2, cc_p2
    end
    @eval @inline function ($expri)(x::MC{N,T}) where {N, T<:RelaxTag}
        z, tp1, tp2, tp3, tp4 = ($expri_kernel)(x, ($expri)(x.Intv), Inf, Inf, Inf, Inf)
        return z
    end
end
