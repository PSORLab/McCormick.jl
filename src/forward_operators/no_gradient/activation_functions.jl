# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This code is licensed under MIT license (see LICENSE.md for full details)
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
relu_kernel(t::ANYRELAX, x::MCNoGrad, z::Interval{Float64}) = max_kernel(t, x, 0.0, z)
function relu(t::ANYRELAX, x::MCNoGrad)
    relu_Intv = max(x.Intv, 0.0)
    z, cvi, cci, dcv, dcc = relu_kernel(t, x, relu_Intv)
    return z
end

function maxtanh_kernel(t::ANYRELAX, x::MCNoGrad, z::Interval{Float64})
    xLc = z.lo
    xUc = z.hi
    xL = x.Intv.lo
    xU = x.Intv.hi
    midcv, cvi = mid3(x.cc, x.cv, xL)
    midcc, cci = mid3(x.cc, x.cv, xU)
    dcv = maxtanh_deriv(midcv)
    dcc = (xUc > xLc) ? (xUc - xLc)/(xU - xL) : 0.0
    u = maxtanh(midcv)
    o = dcc*(midcc - xL) + xLc
    return  MCNoGrad(u, o, z, x.cnst), cvi, cci, dcv, dcc
end
function maxtanh(t::ANYRELAX, x::MCNoGrad) 
    maxtanh_Intv = maxtanh(x.Intv)
    z, cvi, cci, dcv, dcc = maxtanh_kernel(t, x, maxtanh_Intv)
    return z
end

function softplus_kernel(t::ANYRELAX, x::MCNoGrad, z::Interval{Float64})
    xLc = z.lo
    xUc = z.hi
    xL = x.Intv.lo
    xU = x.Intv.hi
    midcv, cv_id = mid3(x.cc, x.cv, xL)
    midcc, cc_id = mid3(x.cc, x.cv, xU)
    dcv = softplus_deriv(midcv)
    dcc = (xUc > xLc) ? (xUc - xLc)/(xU - xL) : 0.0
    u = softplus(midcv)
    o = dcc*(midcc - xL) + xLc
    return MCNoGrad(u, o, z, x.cnst), cvi, cci, dcv, dcc
end
function softplus(t::ANYRELAX, x::MCNoGrad) 
    maxtanh_Intv = softplus(x.Intv)
    z, cvi, cci, dcv, dcc = softplus_kernel(t, x, maxtanh_Intv)
    return z
end

#=
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
@inline param_relu(x::MCNoGrad, α::Float64) = param_relu_kernel(x, α, param_relu(x.Intv, α))


@inline leaky_relu_kernel(x::MC, z::Interval{Float64}) = param_relu_kernel(x, 0.01, z)
@inline leaky_relu(x::MC) = leaky_relu_kernel(x, param_relu(x.Intv, 0.01))

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

@inline function logcosh_kernel(x::MC{N, T}, y::Interval{Float64}) where {N,T<:Union{NS,MV}}
    xL = x.Intv.lo
    xU = x.Intv.hi
    eps_min = in(0.0, x) ? 0.0 : (xU <= 0.0 ? xU : xL)
    eps_max = (abs(xL) < abs(xU)) ? xU : xL
    midcv, cv_id = mid3(x.cc, x.cv, eps_min)
    midcc, cc_id = mid3(x.cc, x.cv, eps_max)
    cv, dcv = cv_logcosh(midcv, xL, xU)
    cc, dcc = cc_logcosh(midcc, xL, xU)
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
    return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function logcosh_kernel(x::MC{N, T}, y::Interval{Float64}) where {N,T<:Diff}
    xL = x.Intv.lo
    xU = x.Intv.hi
    eps_min = in(0.0, x) ? 0.0 : (xU <= 0.0 ? xU : xL)
    eps_max = (abs(xL) < abs(xU)) ? xU : xL
    midcv, cv_id = mid3(x.cc, x.cv, eps_min)
    midcc, cc_id = mid3(x.cc, x.cv, eps_max)
    cv, dcv = cv_logcosh(midcv, xL, xU)
    cc, dcc = cc_logcosh(midcc, xL, xU)
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
function logcosh(x::MC{N,T}) where {N, T <: RelaxTag}
	logcosh_kernel(x, logcosh(x.Intv))
end
=#