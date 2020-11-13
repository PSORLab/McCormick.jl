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
# src/forward_operators/mixed_convexity.jl
# Defines the operators: cos, sin, sinh, atanh, tanh, atan, asin, tan, acos,
#                        asinh, cosh, rad2deg, deg2rad, sind, cosd..., sec,
#                        csc..., sech, csch, ...asech...
#############################################################################

# convex relaxation (envelope) of cos function
@inline function cv_cos(x::Float64, xL::Float64, xU::Float64, tp1::Float64, tp2::Float64)
     r = 0.0
     kL = Base.ceil(-0.5 - xL/(2.0*pi))
     if x <= (pi - 2.0*pi*kL)
         xL1 = xL + 2.0*pi*kL
         if xL1 >= pi/2.0
             return cos(x), -sin(x), tp1, tp2
         end
         xU1 = min(xU+2.0*pi*kL,pi)
         if (xL1 >= -pi/2) && (xU1 <= pi/2)
             r = (abs(xL - xU) < MC_ENV_TOL) ? 0.0 : (cos(xU) - cos(xL))/(xU - xL)
             return cos(xL) + r*(x - xL), r, tp1, tp2
         end
         val, dval, tp1 = cv_cosin(x + (2.0*pi)*kL, xL1, xU1, tp1)
         return val, dval, tp1, tp2
     end
     kU = Base.floor(0.5 - xU/(2.0*pi))
     if (x >= -pi - 2.0*pi*kU)
         xU2 = xU + 2.0*pi*kU
         (xU2 <= -pi/2.0) && (return cos(x), -sin(x), tp1, tp2)
         val, dval, tp2 = cv_cosin(x + 2.0*pi*kU, max(xL + 2.0*pi*kU, -pi), xU2, tp2)
         return val, dval, tp1, tp2
     end
     return -1.0, 0.0, tp1, tp2
end
# function for computing convex relaxation over nonconvex and nonconcave regions
@inline function cv_cosin(x::Float64, xL::Float64, xU::Float64, xj::Float64)
    if abs(xL) <= abs(xU)
        left = false
        x0 = xU
        xm = xL
    else
        left = true
        x0 = xL
        xm = xU
    end
    if xj === Inf
        xj, flag = newton(x0, xL, xU, cv_cosenv, dcv_cosenv, xm, 0.0)
        flag && (xj = golden_section(xL, xU, cv_cosenv, xm, 0.0))
    end
    if (left && x <= xj) || (~left && x >= xj)
        return cos(x), -sin(x), xj
    else
        r = abs(xm - xj) < MC_ENV_TOL ? 0.0 : (cos(xm) - cos(xj))/(xm - xj)
    end
    return cos(xm) + r*(x - xm), r, xj
end
@inline cv_cosenv(x::Float64, y::Float64, z::Float64) = (x - y)*sin(x) + cos(x) - cos(y)
@inline dcv_cosenv(x::Float64, y::Float64, z::Float64) = (x - y)*cos(x)
# concave relaxation (envelope) of cos function
@inline function cc_cos(x::Float64, xL::Float64, xU::Float64, tp1::Float64, tp2::Float64)
    temp = cv_cos(x - pi, xL - pi, xU - pi, tp1, tp2)
    return -temp[1], -temp[2], temp[3], temp[4]
end
@inline function cos_arg(xL::Float64, xU::Float64)
    kL = Base.ceil(-0.5 - xL/(2.0*pi))
    xL1 = xL + 2.0*pi*kL
    xU1 = xU + 2.0*pi*kL
    ~((xL1 >= -pi) && (xL1 <= pi)) && (return NaN, NaN)
    if xL1 <= 0.0
        if xU1 <= 0.0
            arg1 = xL
            arg2 = xU
        elseif xU1 >= pi
            arg1 = pi-2.0*pi*kL
            arg2 = -2.0*pi*kL
        else
            arg1 = (cos(xL1) <= cos(xU1)) ? xL : xU
            arg2 = -2.0*pi*kL
        end
        return arg1, arg2
    end
    if xU1 <= pi
        arg1 = xU
        arg2 = xL
    elseif xU1 >= 2.0*pi
        arg1 = pi - 2.0*pi*kL
        arg2 = 2.0*pi - 2.0*pi*kL
    else
        arg1 = pi - 2.0*pi*kL
        arg2 = (cos(xL1) >= cos(xU1)) ? xL : xU
    end
    return arg1, arg2
end
@inline function cos_kernel(x::MC{N, Diff}, y::Interval{Float64}, cv_tp1::Float64,
                              cv_tp2::Float64, cc_tp1::Float64, cc_tp2::Float64) where N
    xL = x.Intv.lo
    xU = x.Intv.hi
    xLc = y.lo
    xUc = y.hi
    eps_min, eps_max = cos_arg(x.Intv.lo, x.Intv.hi)
    midcc = mid3v(x.cv, x.cc, eps_max)
    midcv = mid3v(x.cv, x.cc, eps_min)
    cc, dcc, cc_tp1, cc_tp2 = cc_cos(midcc, x.Intv.lo, x.Intv.hi, cc_tp1, cc_tp2)
    cv, dcv, cv_tp1, cv_tp2 = cv_cos(midcv, x.Intv.lo, x.Intv.hi, cv_tp1, cv_tp2)
    gcc1, gdcc1, cc_tp1, cc_tp2 = cc_cos(x.cv, x.Intv.lo, x.Intv.hi, cc_tp1, cc_tp2)
    gcv1, gdcv1, cv_tp1, cv_tp2 = cv_cos(x.cv, x.Intv.lo, x.Intv.hi, cv_tp1, cv_tp2)
    gcc2, gdcc2, cc_tp1, cc_tp2 = cc_cos(x.cc, x.Intv.lo, x.Intv.hi, cc_tp1, cc_tp2)
    gcv2, gdcv2, cv_tp1, cv_tp2 = cv_cos(x.cc, x.Intv.lo, x.Intv.hi, cv_tp1, cv_tp2)
    cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
    cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
    return MC{N, Diff}(cv, cc, y, cv_grad, cc_grad, x.cnst), cv_tp1, cv_tp2, cc_tp1, cc_tp2
end
@inline function cos_kernel(x::MC{N, T}, y::Interval{Float64}, cv_tp1::Float64,
                              cv_tp2::Float64, cc_tp1::Float64, cc_tp2::Float64) where {N,T<:Union{NS,MV}}
    xL = x.Intv.lo
    xU = x.Intv.hi
    xLc = y.lo
    xUc = y.hi
    eps_min, eps_max = cos_arg(x.Intv.lo, x.Intv.hi)
    midcc, cc_id = mid3(x.cc, x.cv, eps_max)
    midcv, cv_id = mid3(x.cc, x.cv, eps_min)
    cc, dcc, cc_tp1, cc_tp2 = cc_cos(midcc, x.Intv.lo, x.Intv.hi, cc_tp1, cc_tp2)
    cv, dcv, cv_tp1, cv_tp2 = cv_cos(midcv, x.Intv.lo, x.Intv.hi, cv_tp1, cv_tp2)
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
    cv,cc,cv_grad,cc_grad = cut(xLc,xUc,cv,cc,cv_grad,cc_grad)
    return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst), cv_tp1, cv_tp2, cc_tp1, cc_tp2
end
@inline function cos(x::MC)
    y, tp1, tp2, tp3, tp4 = cos_kernel(x, cos(x.Intv), Inf, Inf, Inf, Inf)
    return y
end
@inline function sin_kernel(x::MC, y::Interval{Float64}, cv_tp1::Float64,
                            cv_tp2::Float64, cc_tp1::Float64, cc_tp2::Float64)
    cos_kernel(x-pi/2.0, y, cv_tp1, cv_tp2, cc_tp1, cc_tp2)
end
@inline function sin(x::MC)
    y, tp1, tp2, tp3, tp4 = sin_kernel(x, sin(x.Intv), Inf, Inf, Inf, Inf)
    return y
end

@inline sinh_deriv(x::Float64) = cosh(x)
@inline sinh_env(x::Float64, y::Float64, z::Float64) = (x-y)*cosh(x)-(sinh(x)-sinh(y))
@inline function cv_sinh(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return sinh(x), sinh_deriv(x), p)
    (xU <= 0.0) && (return dline_seg(sinh, sinh_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, sinh_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, sinh_env, xU, 0.0))
    end
    (x <= p) && (return dline_seg(sinh, sinh_deriv, x, xL, p)..., p)
    return sinh(x), sinh_deriv(x), p
end
@inline function cc_sinh(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(sinh, sinh_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return sinh(x), sinh_deriv(x), p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, sinh_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, sinh_env, xL, 0.0))
    end
    (x <= p) && (return sinh(x), sinh_deriv(x), p)
    return dline_seg(sinh, sinh_deriv, x, p, xU)..., p
end

@inline atanh_deriv(x::Float64) = 1.0/(1.0 - x^2)
@inline atanh_env(x::Float64, y::Float64, z::Float64) = (NaNMath.atanh(x) - NaNMath.atanh(y))*(1.0 - x^2) - x + y
@inline function cv_atanh(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL <= -1.0) && (return NaN, NaN, NaN)
    (xU >= 1.0) && (return NaN, NaN, NaN)
    (xL >= 0.0) && (return NaNMath.atanh(x), atanh_deriv(x), p)
    (xU <= 0.0) && (return dline_seg(NaNMath.atanh, atanh_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, atanh_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, atanh_env, xU, 0.0))
    end
    (x <= p) && (return dline_seg(NaNMath.atanh, atanh_deriv, x, xL, p)..., p)
    return NaNMath.atanh(x), atanh_deriv(x), p
end
@inline function cc_atanh(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL <= -1.0) && (return NaN, NaN, NaN)
    (xU >= 1.0) && (return NaN, NaN, NaN)
    (xL >= 0.0) && (return dline_seg(NaNMath.atanh, atanh_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return NaNMath.atanh(x), atanh_deriv(x), p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, atanh_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, atanh_env, xL, 0.0))
    end
    (x <= p) && (return NaNMath.atanh(x), atanh_deriv(x), p)
    return dline_seg(NaNMath.atanh, atanh_deriv, x, p, xU)..., p
end

@inline tanh_deriv(x::Float64, y::Float64, z::Float64) = sech(x)^2
@inline tanh_env(x::Float64, y::Float64, z::Float64) = (x - y) - (tanh(x) - tanh(y))/(1.0 - tanh(x)^2)
@inline tanh_envd(x::Float64, y::Float64, z::Float64)= 2.0*tanh(x)/(1.0 - tanh(x)^2)*(tanh(x) - tanh(y))
@inline function cv_tanh(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(tanh, tanh_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return tanh(x), sech(x)^2, p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, tanh_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, tanh_env, xU, 0.0))
    end
    (x <= p) && (return tanh(x), sech(x)^2, p)
    return dline_seg(tanh, tanh_deriv, x, p, xU)..., p
end
@inline function cc_tanh(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return tanh(x), sech(x)^2, p)
    (xU <= 0.0) && (return dline_seg(tanh, tanh_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, tanh_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, tanh_env, xL, 0.0))
    end
    (x <= p) && (return dline_seg(tanh, tanh_deriv, x, xL, p)..., p)
    return tanh(x), sech(x)^2, p
end

@inline atan_deriv(x::Float64, y::Float64, z::Float64) = 1.0/(1.0 + x^2)
@inline atan_env(x::Float64, y::Float64, z::Float64) = (x - y) - (1.0 + sqr(x))*(atan(x) - atan(y))
@inline function cv_atan(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(atan, atan_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return atan(x), 1.0/(1.0+x^2), p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, atan_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, atan_env, xU, 0.0))
    end
    (x <= p) && (return atan(x), 1.0/(1.0+x^2), p)
    return dline_seg(atan, atan_deriv, x, p, xU)..., p
end
@inline function cc_atan(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return atan(x), 1.0/(1.0+x^2), p)
    (xU <= 0.0) && (return dline_seg(atan, atan_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, atan_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, atan_env, xL, 0.0))
    end
    (x <= p) && (return dline_seg(atan, atan_deriv, x, xL, p)..., p)
    return atan(x), 1.0/(1.0+x^2), p
end

@inline asin_deriv(x::Float64, y::Float64, z::Float64) = 1.0/NaNMath.sqrt(1.0 - x^2)
@inline function asin_env(x::Float64, y::Float64, z::Float64)
    return (NaNMath.asin(x) - NaNMath.asin(y))*NaNMath.sqrt(1.0-x^2) - x + y
end
@inline function cv_asin(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return NaNMath.asin(x), 1.0/NaNMath.sqrt(1.0-x^2), p)
    (xU <= 0.0) && (return dline_seg(NaNMath.asin, asin_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, asin_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, asin_env, xL, 0.0))
  end
  (x <= p) && (return dline_seg(NaNMath.asin, asin_deriv, x, xL, p)..., p)
  return NaNMath.asin(x), 1.0/NaNMath.sqrt(1.0-x^2, p)
end
@inline function cc_asin(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(NaNMath.asin, asin_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return NaNMath.asin(x), 1.0/NaNMath.sqrt(1.0-x^2), p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, asin_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, asin_env, xU, 0.0))
    end
    (x <= p) && (return NaNMath.asin(x), 1.0/NaNMath.sqrt(1.0-x^2), p)
    return dline_seg(NaNMath.asin, asin_deriv, x, p, xU)..., p
end

@inline tan_deriv(x::Float64, y::Float64, z::Float64) = sec(x)^2
@inline function tan_env(x::Float64, y::Float64, z::Float64)
    return (x - y) - (NaNMath.tan(x) - NaNMath.tan(y))/(1.0 + NaNMath.tan(x)^2)
end
@inline function tan_envd(x::Float64, y::Float64, z::Float64)
    return 2.0*NaNMath.tan(x)/(1.0 + NaNMath.tan(x)^2)*(NaNMath.tan(x) - NaNMath.tan(y))
end
@inline function cv_tan(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return NaNMath.tan(x), sec(x)^2, p)
    (xU <= 0.0) && (return dline_seg(NaNMath.tan, tan_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, tan_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, tan_env, xL, 0.0))
    end
    (x <= p) && (return dline_seg(NaNMath.tan, tan_deriv, x, xL, p)..., p)
    return NaNMath.tan(x), sec(x)^2, p
end
@inline function cc_tan(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(NaNMath.tan, tan_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return NaNMath.tan(x), sec(x)^2, p)
    if p === Inf
        p, flag = secant(0.0, xL, xL, 0.0, tan_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, tan_env, xU, 0.0))
    end
    (x <= p) && (return NaNMath.tan(x), sec(x)^2, p)
    return dline_seg(NaNMath.tan, tan_deriv, x, p, xU)..., p
end

@inline acos_deriv(x::Float64, y::Float64, z::Float64) = -1.0/NaNMath.sqrt(1.0-x^2)
@inline acos_env(x::Float64, y::Float64, z::Float64) = -(NaNMath.acos(x) - NaNMath.acos(y))*NaNMath.sqrt(1-x^2) - x + y
@inline function cc_acos(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(NaNMath.acos, acos_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return NaNMath.acos(x), -1.0/NaNMath.sqrt(1.0-x^2), p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, acos_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, acos_env, xL, 0.0))
    end
    (x <= p) && (return dline_seg(NaNMath.acos, acos_deriv, x, xL, p)..., p)
    return NaNMath.acos(x), -1.0/NaNMath.sqrt(1.0-x^2), p
end
@inline function cv_acos(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return NaNMath.acos(x), -1.0/NaNMath.sqrt(1.0-x^2), p)
    (xU <= 0.0) && (return dline_seg(NaNMath.acos, acos_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, acos_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, acos_env, xU, 0.0))
    end
    (x <= p) && (return NaNMath.acos(x), -1.0/NaNMath.sqrt(1.0-x^2), p)
    return dline_seg(acos, acos_deriv, x, p, xU)..., p
end

@inline asinh_deriv(x::Float64, y::Float64, z::Float64) = 1.0/sqrt(1.0 + x^2)
@inline asinh_env(x::Float64, y::Float64, z::Float64) = (x - y) - sqrt(1.0 + sqr(x))*(asinh(x) - asinh(y))
@inline function cv_asinh(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(asinh, asinh_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return asinh(x), 1.0/sqrt(1.0 + x^2), p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, asinh_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, asinh_env, xU, 0.0))
    end
    (x <= p) && (return asinh(x), 1.0/sqrt(1.0+x^2), p)
    return dline_seg(asinh, asinh_deriv, x, p, xU)..., p
end
@inline function cc_asinh(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return asinh(x), 1.0/sqrt(1.0 + x^2), p)
    (xU <= 0.0) && (return dline_seg(asinh, asinh_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, asinh_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, asinh_env, xL, 0.0))
    end
    (x <= p) && (return dline_seg(asinh, asinh_deriv, x, xL, p)..., p)
    return asinh(x), 1.0/sqrt(1.0+x^2), p
end

@inline erf_deriv(x::Float64) = (2.0/sqrt(pi))*exp(-x^2)
@inline erf_deriv2(x::Float64) = 4.0*x*exp(-x^2)/sqrt(pi)
@inline function erf_env(x::Float64, y::Float64, z::Float64)
    (x - y) - (erf(x) - erf(y))/erf_deriv(x)
end
@inline function cv_erf(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(erf, erf_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return erf(x), erf_deriv(x), p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, erf_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, erf_env, xU, 0.0))
    end
    (x <= p) && (return erf(x), erf_deriv(x), p)
    return dline_seg(erf, erf_deriv, x, p, xU)..., p
end
@inline function cc_erf(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return erf(x), erf_deriv(x), p)
    (xU <= 0.0) && (return dline_seg(erf, erf_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, erf_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, erf_env, xL, 0.0))
    end
    (x <= p) && (return dline_seg(erf, erf_deriv, x, xL, p)..., p)
    return erf(x), erf_deriv(x), p
end

@inline erfinv_deriv(x::Float64) = (sqrt(pi)/2.0)*exp(erfinv(x)^2)
@inline erfinv_deriv2(x::Float64) = (sqrt(pi)/2.0)*exp(2.0*erfinv(x)^2)*erfinv(x)
@inline function erfinv_env(x::Float64, y::Float64, z::Float64)
    (x - y) - (erfinv(x) - erfinv(y))/erfinv_deriv(x)
end
@inline function cv_erfinv(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return erfinv(x), erfinv_deriv(x), p)
    (xU <= 0.0) && (return dline_seg(erfinv, erfinv_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, erfinv_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, erfinv_env, xL, 0.0))
    end
    (x <= p) && (return dline_seg(erfinv, erfinv_deriv, x, xL, p)..., p)
    return erfinv(x), erfinv_deriv(x), p
end
@inline function cc_erfinv(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(erfinv, erfinv_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return erfinv(x), erf_deriv(x), p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, erfinv_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, erfinv_env, xU, 0.0))
    end
    (x >= p) && (return dline_seg(erfinv, erfinv_deriv, x, p, xU)..., p)
    return erfinv(x), erfinv_deriv(x), p
end

@inline erfc_deriv(x::Float64) = (-2.0/sqrt(pi))*exp(-x^2)
@inline erfc_deriv2(x::Float64) = (4.0/sqrt(pi))*exp(-x^2)*x
@inline function erfc_env(x::Float64, y::Float64, z::Float64)
    (x - y) - (erfc(x) - erfc(y))/erfc_deriv(x)
end
@inline function cv_erfc(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return erfc(x), erfc_deriv(x), p)
    (xU <= 0.0) && (return dline_seg(erfc, erfc_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, erfc_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, erfc_env, xL, 0.0))
    end
    (x <= p) && (return dline_seg(erfc, erfc_deriv, x, xL, p)..., p)
    return erfc(x), erfc_deriv(x), p
end
@inline function cc_erfc(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(erfc, erfc_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return erfc(x), erfc_deriv(x), p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, erfc_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, erfc_env, xU, 0.0))
    end
    (x >= p) && (return dline_seg(erfc, erfc_deriv, x, p, xU)..., p)
    return erfc(x), erfc_deriv(x), p
end

@inline cbrt_deriv(x::Float64) = 1.0/(3.0*cbrt(x)^2)
@inline cbrt_deriv2(x::Float64) = -2.0/(9.0*cbrt(x)^5)
@inline function cbrt_env(x::Float64, y::Float64, z::Float64)
    (x - y) - (cbrt(x) - cbrt(y))/cbrt_deriv(x)
end
@inline function cv_cbrt(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return dline_seg(cbrt, cbrt_deriv, x, xL, xU)..., p)
    (xU <= 0.0) && (return cbrt(x), cbrt_deriv(x), p)
    if p === Inf
        p, flag = secant(xL, 0.0, xL, 0.0, cbrt_env, xU, 0.0)
        flag && (p = golden_section(xL, 0.0, cbrt_env, xU, 0.0))
    end
    (x <= p) && (return cbrt(x), cbrt_deriv(x), p)
    return dline_seg(cbrt, cbrt_deriv, x, p, xU)..., p
end
@inline function cc_cbrt(x::Float64, xL::Float64, xU::Float64, p::Float64)
    (xL >= 0.0) && (return cbrt(x), cbrt_deriv(x), p)
    (xU <= 0.0) && (return dline_seg(cbrt, cbrt_deriv, x, xL, xU)..., p)
    if p === Inf
        p, flag = secant(0.0, xU, 0.0, xU, cbrt_env, xL, 0.0)
        flag && (p = golden_section(0.0, xU, cbrt_env, xL, 0.0))
    end
    (x <= p) && (return dline_seg(cbrt, cbrt_deriv, x, xL, p)..., p)
    return cbrt(x), cbrt_deriv(x), p
end

# Defines interval version of cbrt if necessary
# Copy of recent version from IntervalArithmetic.jl
if VERSION < v"1.2-"
    function cbrt(x::BigFloat, r::RoundingMode)
            setrounding(BigFloat, r) do
                cbrt(x)
            end
        end
    cbrt(a::Interval{Float64}) where T = atomic(Interval{T}, cbrt(big53(a)))
    function cbrt(a::Interval{BigFloat})
        isempty(a) && return a
        @round(cbrt(a.lo), cbrt(a.hi))
    end
end

# basic method overloading operator (sinh, tanh, atanh, asinh), convexoconcave or concavoconvex
eps_min_dict = Dict{Symbol,Symbol}(:sinh => :xL, :tanh => :xL, :asinh => :xL,
                                 :atanh => :xL, :tan => :xL, :acos => :xU,
                                 :asin => :xL, :atan => :xL, :erf => :xL,
                                 :cbrt => :xL, :erfinv => :xL, :erfc => :xU)
eps_max_dict = Dict{Symbol,Symbol}(:sinh => :xU, :tanh => :xU, :asinh => :xU,
                                 :atanh => :xU, :tan => :xU, :acos => :xL,
                                 :asin => :xU, :atan => :xU, :erf => :xU,
                                 :cbrt => :xU, :erfinv => :xU, :erfc => :xL)

for expri in (:sinh, :tanh, :asinh, :atanh, :tan, :acos, :asin, :atan,
              (:(SpecialFunctions.erf), :erf), :cbrt,
              (:(SpecialFunctions.erfinv), :erfinv),
              (:(SpecialFunctions.erfc), :erfc))
    if expri isa Symbol
        expri_name = expri
        expri_sym = expri
    else
        expri_name = expri[1]
        expri_sym = expri[2]
    end
    expri_kernel = Symbol(String(expri_sym)*"_kernel")
    expri_cv = Symbol("cv_"*String(expri_sym))
    expri_cc = Symbol("cc_"*String(expri_sym))
    eps_min = eps_min_dict[expri_sym]
    eps_max = eps_max_dict[expri_sym]
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
    @eval @inline function ($expri_kernel)(x::MC{N, Diff}, y::Interval{Float64},
                            cv_p::Float64, cc_p::Float64) where N
        xL = x.Intv.lo
        xU = x.Intv.hi
        midcv, cv_id = mid3(x.cv, x.cc, $eps_min)
        midcc, cc_id = mid3(x.cv, x.cc, $eps_max)
        cv, dcv, cv_p = $(expri_cv)(midcv, xL, xU, cv_p)
        cc, dcc, cc_p = $(expri_cc)(midcc, xL, xU, cc_p)
        gcv1, gdcv1, cv_p = $(expri_cv)(x.cv, xL, xU, cv_p)
        gcc1, gdcc1, cc_p = $(expri_cc)(x.cv, xL, xU, cc_p)
        gcv2, gdcv2, cv_p = $(expri_cv)(x.cc, xL, xU, cv_p)
        gcc2, gdcc2, cc_p = $(expri_cc)(x.cc, xL, xU, cc_p)
        cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
        cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
        return MC{N,Diff}(cv, cc, y, cv_grad, cc_grad, x.cnst), cv_p, cc_p
    end
    @eval @inline function ($expri_name)(x::MC{N,T}) where {N, T<:RelaxTag}
        z, tp1, tp2 = ($expri_kernel)(x, ($expri_name)(x.Intv), Inf, Inf)
        return z
    end
end

# cosh convex
@inline cv_cosh(x::Float64, xL::Float64, xU::Float64) = cosh(x), sinh(x)
@inline cc_cosh(x::Float64, xL::Float64, xU::Float64) = dline_seg(cosh, sinh, x, xL, xU)
@inline function cosh_kernel(x::MC{N, T}, y::Interval{Float64}) where {N,T<:Union{NS,MV}}
    xL = x.Intv.lo
    xU = x.Intv.hi
    eps_max = abs(xU) > abs(xL) ?  xU : xL
    eps_min = in(0.0, x) ? 0.0 : (abs(xU) > abs(xL) ?  xL : xU)
    midcc, cc_id = mid3(x.cc, x.cv, eps_max)
    midcv, cv_id = mid3(x.cc, x.cv, eps_min)
    cc, dcc = cc_cosh(midcc, xL, xU)
    cv, dcv = cv_cosh(midcv, xL, xU)
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
    cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
    return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function cosh_kernel(x::MC{N,Diff}, y::Interval{Float64}) where N
    xL = x.Intv.lo
    xU = x.Intv.hi
    eps_max = abs(xU) > abs(xL) ?  xU : xL
    eps_min = in(0.0, x) ? 0.0 : (abs(xU) > abs(xL) ?  xL : xU)
    midcc = mid3v(x.cv, x.cc, eps_max)
    midcv = mid3v(x.cv, x.cc, eps_min)
    cc, dcc = cc_cosh(midcc, xL, xU)
    cv, dcv = cv_cosh(midcv, xL, xU)
    gcc1, gdcc1 = cc_cosh(x.cv, xL, xU)
    gcv1, gdcv1 = cv_cosh(x.cv, xL, xU)
    gcc2, gdcc2 = cc_cosh(x.cc, xL, xU)
    gcv2, gdcv2 = cv_cosh(x.cc, xL, xU)
    cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
    cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
    return MC{N,Diff}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline cosh(x::MC) = cosh_kernel(x, cosh(x.Intv))

@inline deg2rad(x::MC, y::Interval{Float64}) = mult_kernel(x, pi/180.0, y)
@inline rad2deg(x::MC, y::Interval{Float64}) = mult_kernel(x, 180.0/pi, y)
@inline deg2rad(x::MC) = deg2rad(x, (pi/180.0)*x.Intv)
@inline rad2deg(x::MC) = rad2deg(x, (180.0/pi)*x.Intv)

# TODO: ADD efficient kernels for below (if applicable)
@inline sec(x::MC)= inv(cos(x))
@inline csc(x::MC)= inv(sin(x))
@inline cot(x::MC)= inv(tan(x))

@inline asec(x::MC) = acos(inv(x))
@inline acsc(x::MC) = asin(inv(x))
@inline acot(x::MC) = atan(inv(x))

@inline sech(x::MC) = inv(cosh(x))
@inline csch(x::MC) = inv(sinh(x))
@inline coth(x::MC) = inv(tanh(x))

@inline acsch(x::MC) = log(sqrt(1.0 + inv(sqr(x))) + inv(x))
@inline acoth(x::MC) = 0.5*(log(1.0 + inv(x)) - log(1.0 - inv(x)))

@inline sind(x::MC) = sin(deg2rad(x))
@inline cosd(x::MC) = cos(deg2rad(x))
@inline tand(x::MC) = tan(deg2rad(x))
@inline secd(x::MC) = inv(cosd(x))
@inline cscd(x::MC) = inv(sind(x))
@inline cotd(x::MC) = inv(tand(x))

@inline asind(x::MC) = rad2deg(asin(x))
@inline acosd(x::MC) = rad2deg(acos(x))
@inline atand(x::MC) = rad2deg(atan(x))
@inline asecd(x::MC) = rad2deg(asec(x))
@inline acscd(x::MC) = rad2deg(acsc(x))
@inline acotd(x::MC) = rad2deg(acot(x))

@inline deg2rad_kernel(x::MC, y::Interval{Float64}) = deg2rad(x,y)
@inline rad2deg_kernel(x::MC, y::Interval{Float64}) = rad2deg(x,y)

@inline sinpi(x::MC) = sin(pi*x)
@inline cospi(x::MC) = cos(pi*x)

@inline erfcinv(x::MC) = erfinv(1.0 - x)

for expri in (:sec, :csc, :cot, :asec, :acsc, :acot, :sech, :csch, :coth,
              :acsch, :acoth, :sind, :cosd, :tand, :secd, :cscd, :cotd,
              :asind, :acosd, :atand, :asecd, :acscd, :acotd, :sinpi, :cospi)

     expri_kernel = Symbol(String(expri)*"_kernel")
     @eval @inline ($expri)(x::MC, y::Interval{Float64}) = ($expri)(x)
 end
 @eval @inline erfcinv_kernel(x::MC, y::Interval{Float64}) = erfcinv(x)

erf(x) = SpecialFunctions.erf(x)
erfc(x) = SpecialFunctions.erfc(x)
erfinv(x) = SpecialFunctions.erfinv(x)
erfcinv(x) = SpecialFunctions.erfcinv(x)
