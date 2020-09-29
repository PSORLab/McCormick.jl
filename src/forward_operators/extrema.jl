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
# src/forward_operators/extrema.jl
# Defines univariate and bivariate max and min operators.
#############################################################################

# Univariate min/max operators
@inline deriv_max(x::Float64, c::Float64) = (x < c) ? 0.0 : 1.0
@inline function cv_max(x::Float64, xL::Float64, xU::Float64, a::Float64)
    xU <= a && (return a, 0.0)
    a <= xL && (return x, 1.0)
    term = max(0.0, (x - a)/(xU - a))
    val = a + (xU - a)*term^MC_DIFF_MU1
    dval = MC_DIFF_MU1T*term^MC_DIFF_MU
    return val, dval
end
@inline function cv_max_ns(x::Float64, xL::Float64, xU::Float64, a::Float64)
    (x <= a) && (return a, 0.0)
    return x, 1.0
end

@inline cc_max(x::Float64, xL::Float64, xU::Float64, a::Float64) = dline_seg(max, deriv_max, x, xL, xU, a)
@inline function max_kernel(x::MC{N, Diff}, c::Float64, z::Interval{Float64}) where N
    midcv, cv_id = mid3(x.cc, x.cv, x.Intv.lo)
    midcc, cc_id = mid3(x.cc, x.cv, x.Intv.hi)
    cv, dcv = cv_max(midcv, x.Intv.lo, x.Intv.hi, c)
    cc, dcc = cc_max(midcc, x.Intv.lo, x.Intv.hi, c)
    gcc1,gdcc1 = cc_max(x.cv, x.Intv.lo, x.Intv.hi, c)
    gcv1,gdcv1 = cv_max(x.cv, x.Intv.lo, x.Intv.hi, c)
    gcc2,gdcc2 = cc_max(x.cc, x.Intv.lo, x.Intv.hi, c)
    gcv2,gdcv2 = cv_max(x.cc, x.Intv.lo, x.Intv.hi, c)
    cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
    cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
    return MC{N, Diff}(cv, cc, z, cv_grad, cc_grad, x.cnst)
end
@inline function max_kernel(x::MC{N, T}, c::Float64, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    midcv, cv_id = mid3(x.cc, x.cv, x.Intv.lo)
    midcc, cc_id = mid3(x.cc, x.cv, x.Intv.hi)
    cv, dcv = cv_max_ns(midcv, x.Intv.lo, x.Intv.hi, c)
    cc, dcc = cc_max(midcc, x.Intv.lo, x.Intv.hi, c)
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
    cv, cc, cv_grad, cc_grad = cut(z.lo, z.hi, cv, cc, cv_grad, cc_grad)
    return MC{N,T}(cv, cc, z, cv_grad, cc_grad, x.cnst)
end
@inline max(x::MC, c::Float64) = max_kernel(x, c, max(x.Intv, c))
@inline min_kernel(x::MC, c::Float64, z::Interval{Float64}) = -max(-x,-c)
@inline min(x::MC, c::Float64) = -max(-x,-c)

# Multivariate min/max operators
@inline function psil_max(x::Float64, y::Float64, lambda::Interval{Float64}, nu::Interval{Float64}, f1::MC{N, Diff}, f2::MC{N, Diff}) where N
    if nu.hi <= lambda.lo
        val = x
    elseif lambda.hi <= nu.lo
        val = y
    elseif nu.lo <= lambda.lo && lambda.lo < nu.hi
        val = x + (nu.hi - lambda.lo)*max(0.0, (y - x)/(nu.hi - lambda.lo))^MC_DIFF_MU1
    else
        val = y + (lambda.hi - nu.lo)*max(0.0, (x - y)/(lambda.hi - nu.lo))^MC_DIFF_MU1
    end
    if nu.hi <= lambda.lo
        grad_val = f1.cv_grad
    elseif lambda.hi <= nu.lo
        grad_val = f1.cv_grad
    else
        grad_val = max(0.0, psil_max_dx(x, y, lambda, nu))*f1.cv_grad +
                   min(0.0, psil_max_dx(x, y, lambda, nu))*f1.cc_grad +
                   max(0.0, psil_max_dy(x, y, lambda, nu))*f2.cv_grad +
                   min(0.0, psil_max_dy(x, y, lambda, nu))*f2.cc_grad
    end
    return val, grad_val
end

@inline function psil_max_dx(x::Float64, y::Float64, lambda::Interval{Float64}, nu::Interval{Float64})
    if in(lambda.lo, nu)
        return 1.0 - MC_DIFF_MU1*max(0.0, (y - x)/(nu.hi - lambda.lo))^MC_DIFF_MU
    end
    return MC_DIFF_MU1*max(0.0, (x - y)/(lambda.hi - nu.lo))^MC_DIFF_MU
end
@inline function psil_max_dy(x::Float64, y::Float64, lambda::Interval{Float64}, nu::Interval{Float64})
    if in(lambda.lo, nu)
        return MC_DIFF_MU1*max(0.0, (y - x)/(nu.hi - lambda.lo))^MC_DIFF_MU
    end
    return 1.0 - MC_DIFF_MU1*max(0.0, (x - y)/(lambda.hi - nu.lo))^MC_DIFF_MU
end
@inline function psir_max(x::Float64, y::Float64, xgrad::SVector{N,Float64}, ygrad::SVector{N,Float64},
                          lambda::Interval{Float64}, nu::Interval{Float64}) where N

    (nu.hi <= lambda.lo) && (return (x, xgrad))
    (lambda.hi <= nu.lo) && (return (y, ygrad))

    maxUU = max(lambda.hi, nu.hi)
    maxLU = max(lambda.lo, nu.hi)
    maxUL = max(lambda.hi, nu.lo)
    maxLL = max(lambda.lo, nu.lo)

    diamx = diam(lambda)
    diamy = diam(nu)
    delta = maxLL + maxUU - maxLU - maxUL
    theta_arg = max(0.0, (lambda.hi - x)diamx - (y - nu.lo)/diamy)
    thetar = delta*theta_arg^MC_DIFF_MU1

    coeff1 = maxUU - maxLU
    coeff2 = maxUU - maxUL
    val = maxUU - coeff1*(lambda.hi-x)/diamx - coeff2*(nu.hi - y)/diamy

    shared_term = -MC_DIFF_MU1T*delta*theta_arg^MC_DIFF_MU
    grad_val = ((coeff1 + shared_term)/diamx)*xgrad + ((coeff2 + shared_term)/diamy)*ygrad
    return val, grad_val
end
@inline function max_kernel(x::MC{N, Diff}, y::MC{N, Diff}, z::Interval{Float64}) where N

    if (y.Intv.hi <= x.Intv.lo) || (x.Intv.hi <= y.Intv.lo)
        cv, cv_grad = psil_max(x.cv, y.cv, x.Intv, y.Intv, x, y)
    elseif (y.Intv.lo <= x.Intv.lo) & (x.Intv.lo < y.Intv.hi)
        temp = mid3v(x.cv, x.cc, y.cv - (y.Intv.hi - x.Intv.lo)*MC_DIFF_DIV)
        cv, cv_grad = psil_max(temp, y.cv, x.Intv, y.Intv, x, y)
    else
        temp = mid3v(y.cv, y.cc, x.cv - (x.Intv.hi - y.Intv.lo)*MC_DIFF_DIV)
        cv, cv_grad = psil_max(x.cv, temp, x.Intv, y.Intv, x, y)
    end

    cc, cc_grad = psir_max(x.cc, y.cc, x.cc_grad, y.cv_grad, x.Intv, y.Intv)
    return MC{N, Diff}(cv, cc, z, cv_grad, cc_grad, (x.cnst && y.cnst))
end

@inline function max_kernel(x::MC{N, MV}, y::MC{N, MV}, z::Interval{Float64}) where N
    if x.Intv.hi <= y.Intv.lo
        cc = y.cc
        cc_grad = y.cnst ? zero(SVector{N,Float64}) : y.cc_grad
    elseif x.Intv.lo >= y.Intv.hi
        cc = x.cc
        cc_grad = x.cnst ? zero(SVector{N,Float64}) : x.cc_grad
    else
        maxLU = max(x.Intv.lo, y.Intv.hi)
        maxUL = max(x.Intv.hi, y.Intv.lo)

        m1a = isthin(x) ? 0.0 : (x.cc - lo(x))/diam(x)
        m1b = isthin(y) ? 0.0 : (y.cc - lo(y))/diam(y)
        maxLL = max(x.Intv.lo, y.Intv.lo)
        g1cc = maxLL + m1a*(maxUL - maxLL) + m1b*(maxLU - maxLL)

        m2a = isthin(x) ? 0.0 : (x.cc - hi(x))/diam(x)
        m2b = isthin(y) ? 0.0 : (y.cc - hi(y))/diam(y)
        maxUU = max(x.Intv.hi, y.Intv.hi)
        g2cc = maxUU + m2a*(maxUU - maxLU) + m2b*(maxUU - maxUL)

        if g1cc < g2cc
            cc = g1cc
            cc_grad = (maxUL - maxLL)*x.cc_grad + (maxLU - maxLL)*y.cc_grad
        else
            cc = g2cc
            cc_grad = (maxUU - maxLU)*x.cc_grad + (maxUU - maxUL)*y.cc_grad
        end
    end
    cv = max(x.cv, y.cv)
    cv_grad = (x.cv > y.cv) ? (x.cnst ? zero(SVector{N,Float64}) : x.cv_grad) :
                              (y.cnst ? zero(SVector{N,Float64}) : y.cv_grad)
    cv, cc, cv_grad, cc_grad = cut(z.lo, z.hi, cv, cc, cv_grad, cc_grad)
    return MC{N, MV}(cv, cc, z, cv_grad, cc_grad, y.cnst ? x.cnst : (x.cnst ? y.cnst : (x.cnst || y.cnst)))
end

@inline function max_kernel(x::MC{N, NS}, y::MC{N, NS}, z::Interval{Float64}) where N
    if x.Intv.hi <= y.Intv.lo
        cc = y.cc
        cc_grad = y.cnst ? zero(SVector{N,Float64}) : y.cc_grad
    elseif x.Intv.lo >= y.Intv.hi
        cc = x.cc
        cc_grad = x.cnst ? zero(SVector{N,Float64}) : x.cc_grad
    else
        ccMC = 0.5*(x + y + abs(x - y))
        cc = ccMC.cc
        cc_grad = ccMC.cc_grad
    end
    cv = max(x.cv, y.cv)
    cv_grad = (x.cv > y.cv) ? (x.cnst ? zero(SVector{N,Float64}) : x.cv_grad) :
                              (y.cnst ? zero(SVector{N,Float64}) : y.cv_grad)
    cv, cc, cv_grad, cc_grad = cut(z.lo, z.hi, cv, cc, cv_grad, cc_grad)
    return MC{N, NS}(cv, cc, z, cv_grad, cc_grad, y.cnst ? x.cnst : (x.cnst ? y.cnst : (x.cnst || y.cnst)))
end

@inline function min_kernel(x::MC{N, NS}, y::MC{N, NS}, z::Interval{Float64}) where N
    if x.Intv.hi <= y.Intv.lo
        cv = x.cv
        cv_grad = x.cnst ? zero(SVector{N,Float64}) : x.cv_grad
    elseif x.Intv.lo >= y.Intv.hi
        cv = y.cv
        cv_grad = y.cnst ? zero(SVector{N,Float64}) : y.cv_grad
    else
        cvMC = 0.5*(x + y - abs(x - y))
        cv = cvMC.cv
        cv_grad = cvMC.cv_grad
    end
    cc = min(x.cc, y.cc)
    cc_grad = (x.cv > y.cv) ? (x.cnst ? zero(SVector{N,Float64}) : x.cv_grad) :
                              (y.cnst ? zero(SVector{N,Float64}) : y.cv_grad)
    cv, cc, cv_grad, cc_grad = cut(z.lo, z.hi, cv, cc, cv_grad, cc_grad)
    return MC{N, NS}(cv, cc, z, cv_grad, cc_grad, y.cnst ? x.cnst : (x.cnst ? y.cnst : (x.cnst || y.cnst)))
end

@inline max(x::MC, y::MC) = max_kernel(x, y, max(x.Intv, y.Intv))
@inline min_kernel(x::MC{N,T}, y::MC{N,T}, z::Interval{Float64}) where {N, T<:Union{MV, Diff}} = -max(-x, -y)
@inline min(x::MC, y::MC) = min_kernel(x, y, min(x.Intv, y.Intv))
