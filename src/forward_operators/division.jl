# Copyright (c) 2018 Matthew Wilhelm, Robert Gottlieb, Dimitri Alston, 
# Matthew Stuber, and the University of Connecticut (UConn)
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A forward McCormick operator library
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# src/forward_operators/division.jl
# Contains definitions of division.
#############################################################################

@inline function div_alphaxy(es::T, nu::T, x::Interval{Float64}, y::Interval{Float64}) where T <: Real
    return (es/y.bareinterval.hi) + (x.bareinterval.lo/(y.bareinterval.lo*y.bareinterval.hi))*(y.bareinterval.hi - nu)
end
@inline function div_gammay(omega::T, y::Interval{Float64}) where T <: Real
    return (y.bareinterval.lo*(max(0.0, omega))^2)/(y.bareinterval.hi - omega*(y.bareinterval.hi - y.bareinterval.lo))
end
@inline function div_deltaxy(omega::T, x::Interval{Float64}, y::Interval{Float64}) where T <: Real
    return (1.0/(y.bareinterval.hi*y.bareinterval.lo))*(x.bareinterval.hi - x.bareinterval.lo)*(y.bareinterval.hi - y.bareinterval.lo)*div_gammay(omega, y)
end
@inline function div_psixy(es::T, nu::T, x::Interval{Float64}, y::Interval{Float64}) where T <: Real
    return div_alphaxy(es, nu, x, y) + div_deltaxy(((es - x.bareinterval.lo)/(x.bareinterval.hi - x.bareinterval.lo))-((nu - y.bareinterval.lo)/(y.bareinterval.hi - y.bareinterval.lo)), x, y)
end
@inline function div_omegaxy(x::Interval{Float64}, y::Interval{Float64})
    return (y.bareinterval.hi/(y.bareinterval.hi - y.bareinterval.lo))*(1.0 - sqrt((y.bareinterval.lo*(x.bareinterval.hi - x.bareinterval.lo))/((-x.bareinterval.lo)*(y.bareinterval.hi - y.bareinterval.lo) + (y.bareinterval.lo)*(x.bareinterval.hi - x.bareinterval.lo))))
end
@inline function div_lambdaxy(es::T, nu::T, x::Interval{Float64}) where T <: Real
    return (((es + sqrt(x.bareinterval.lo*x.bareinterval.hi))/(sqrt(x.bareinterval.lo) + sqrt(x.bareinterval.hi)))^2)/nu
end
@inline function div_nuline(x::Interval{Float64}, y::Interval{Float64}, z::T) where T <: Real
    return y.bareinterval.lo + (y.bareinterval.hi - y.bareinterval.lo)*(z - x.bareinterval.lo)/(x.bareinterval.hi - x.bareinterval.lo)
end

@inline function mid3v(x::T, y::T, z::T) where T <: Number
    z <= x && (return x)
    y <= z && (return y)
    return z
end

@inline function div_diffcv(x::MC{N,T}, y::MC{N,T}) where {N, T<:RelaxTag}
    dualx_cv = Dual{Nothing}(x.cv, Partials{N,Float64}(NTuple{N,Float64}(x.cv_grad)))
    dualy_cv = Dual{Nothing}(y.cv, Partials{N,Float64}(NTuple{N,Float64}(x.cv_grad)))
    dualy_cc = Dual{Nothing}(y.cc, Partials{N,Float64}(NTuple{N,Float64}(y.cc_grad)))
    nu_bar = div_nuline(x.Intv, y.Intv, dualx_cv)
    if (0.0 <= x.Intv.bareinterval.lo)
        dual_div = div_lambdaxy(dualx_cv, dualy_cc, x.Intv)
    elseif (x.Intv.bareinterval.lo < 0.0) && (nu_bar <= dualy_cv)
        dual_div = div_alphaxy(dualx_cv, dualy_cv, x.Intv, y.Intv)
    else
        mid3c = mid3v(dualy_cv, dualy_cc, nu_bar - diam(y)*div_omegaxy(x.Intv, y.Intv))
        dual_div = div_psixy(dualx_cv, mid3c, x.Intv, y.Intv)
    end
    val = dual_div.value
    grad = SVector{N,Float64}(dual_div.partials)
    return val, grad
end

@inline function div_MV(x::MC{N,Diff}, y::MC{N,Diff}, z::Interval{Float64}) where N
    if (0.0 < y.Intv.bareinterval.lo)
        cv, cv_grad = div_diffcv(x, y)
        cc, cc_grad = div_diffcv(-x, y)
        return MC{N,Diff}(cv, -cc, z, cv_grad, -cc_grad, x.cnst && y.cnst)
    end
    cv, cv_grad = div_diffcv(-x, -y)
    cc, cc_grad = div_diffcv(-x, y)
    return MC{N,Diff}(cv, -cc, z, cv_grad, -cc_grad, x.cnst && y.cnst)
end

@inline function div_kernel(x::MC{N,T}, y::MC{N,T}, z::Interval{Float64}) where {N, T <: Union{NS,MV}}
    if (x === y)
        zMC = one(MC{N,T})
    else
        zMC = mult_kernel(x, inv(y), z)
    end
    return zMC
end

@inline function div_kernel(x::MC{N,Diff}, y::MC{N,Diff}, z::Interval{Float64}) where {N}
    degen1 = (x.Intv.bareinterval.hi - x.Intv.bareinterval.lo == 0.0)
    degen2 = (y.Intv.bareinterval.hi - y.Intv.bareinterval.lo == 0.0)
    if x === y
        zMC = one(MC{N,Diff})
    elseif  !degen1 || !degen2
        zMC = div_MV(x, y, z)
    else
        zMC = mult_kernel(x, inv(y), z)
    end
    return zMC
end

@inline function /(x::MC{N,T}, y::MC{N,T}) where {N, T<:RelaxTag}
    0.0 ∉ y && (return div_kernel(x, y, x.Intv/y.Intv))
    return MC{N,T}(NaN, NaN, x.Intv/y.Intv, fill(0, SVector{N,Float64}), fill(0, SVector{N,Float64}), true)
end
