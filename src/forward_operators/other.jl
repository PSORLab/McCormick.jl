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
# src/forward_operators/other.jl
# Defines isempty, empty, isnan, step, sign, abs, intersect, in.
#############################################################################

@inline function empty(x::MC{N,T}) where {N, T <: RelaxTag}
	MC{N,T}(Inf, -Inf, Interval{Float64}(Inf,-Inf), zeros(SVector{N,Float64}),
            zeros(SVector{N,Float64}), false)
end

interval_MC(x::MC{S,T}) where {S, T<:RelaxTag} = MC{S,T}(x.Intv)

@inline isnan(x::MC) = isnan(x.cc) || isnan(x.cv)

########### Defines differentiable step relaxations
@inline function cv_step(x::Float64, xL::Float64, xU::Float64)
	 xU <= 0.0 && (return 0.0, 0.0)
	 xL >= 0.0 && (return 1.0, 0.0)
	 x >= 0.0 ? ((x/xU)^2, 2.0*x/xU^2) : (0.0, 0.0)
end
@inline function cc_step(x::Float64, xL::Float64, xU::Float64)
	 xU <= 0.0 && (return 0.0, 0.0)
	 xL >= 0.0 && (return 1.0, 0.0)
	 x >= 0.0 ? (1.0, 0.0) : (1.0-(x/xL)^2, -2.0*x/xL^2)
end
@inline function cv_step_NS(x::Float64, xL::Float64, xU::Float64)
	 xU <= 0.0 && (return 0.0, 0.0)
	 xL >= 0.0 && (return 1.0, 0.0)
	 x > 0.0 ? (x/xU, 1.0/xU) : (0.0, 0.0)
end
@inline function cc_step_NS(x::Float64, xL::Float64, xU::Float64)
	 xU <= 0.0 && (return 0.0, 0.0)
	 xL >= 0.0 && (return 1.0, 0.0)
	 x >= 0.0 ? (1.0, 0.0) : ((1.0 - (x/xL)), (-x/xL))
end
@inline function step_kernel(x::MC{N, T}, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
	 xL = x.Intv.lo
	 xU = x.Intv.hi
	 midcc, cc_id = mid3(x.cc, x.cv, xU)
	 midcv, cv_id = mid3(x.cc, x.cv, xL)
	 cc, dcc = cc_step_NS(midcc, xL, xU)
	 cv, dcv = cv_step_NS(midcv, xL, xU)
	 cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
	 cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
	 cv, cc, cv_grad, cc_grad = cut(z.lo, z.hi, cv, cc, cv_grad, cc_grad)
	 return MC{N,T}(cv, cc, z, cv_grad, cc_grad, x.cnst)
end
@inline function step_kernel(x::MC{N, Diff}, z::Interval{Float64}) where N
	 xL = x.Intv.lo
	 xU = x.Intv.hi
	 midcc = mid3v(x.cc, x.cv, xU)
	 midcv = mid3v(x.cc, x.cv, xL)
	 cc, dcc = cc_step(midcc, xL, xU)
	 cv, dcv = cv_step(midcv, xL, xU)
	 cc1, gdcc1 = cc_step(x.cv, xL, xU)
	 cv1, gdcv1 = cv_step(x.cv, xL, xU)
	 cc2, gdcc2 = cc_step(x.cc, xL, xU)
	 cv2, gdcv2 = cv_step(x.cc, xL, xU)
	 cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
	 cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
	 return MC{N,Diff}(cv, cc, z, cv_grad, cc_grad, x.cnst)
end
@inline step(x::MC) = step_kernel(x, step(x.Intv))

########### Defines sign
@inline function sign_kernel(x::MC{N, T}, z::Interval{Float64}) where {N, T<:RelaxTag}
	zMC = -step(-x) + step(x)
	return MC{N,T}(zMC.cv, zMC.cc, z, zMC.cv_grad, zMC.cc_grad, zMC.cnst)
end
@inline sign(x::MC) = sign_kernel(x, sign(x.Intv))

@inline function cv_abs(x::Float64, xL::Float64, xU::Float64)
	xL >= 0.0 && (return x, 1.0)
	xU <= 0.0 && (return -x, -1.0)
	if x >= 0.0
		return xU*(x/xU)^(MC_DIFF_MU+1), (MC_DIFF_MU+1)*(x/xU)^MC_DIFF_MU
	end
	return -xL*(x/xL)^(MC_DIFF_MU+1), -(MC_DIFF_MU+1)*(x/xL)^MC_DIFF_MU
end
@inline cc_abs(x::Float64,xL::Float64,xU::Float64) = dline_seg(abs, sign, x, xL, xU)
@inline cv_abs_NS(x::Float64,xL::Float64,xU::Float64) = abs(x), sign(x)

@inline function abs_kernel(x::MC{N, T}, z::Interval{Float64}) where {N, T <: Union{NS,MV}}

     xL = x.Intv.lo
	 xU = x.Intv.hi
	 eps_min = mid3v(xL, x.Intv.hi, 0.0)
	 eps_max = (abs(xU) >= abs(xL)) ? xU : xL
	 midcc, cc_id = mid3(x.cc, x.cv, eps_max)
	 midcv, cv_id = mid3(x.cc, x.cv, eps_min)

	 cc, dcc = cc_abs(midcc, xL, xU)
	 cv, dcv = cv_abs_NS(midcv, xL, xU)
	 cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
	 cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
	 cv, cc, cv_grad, cc_grad = cut(z.lo, z.hi, cv, cc, cv_grad, cc_grad)
	 return MC{N,T}(cv, cc, z, cv_grad, cc_grad, x.cnst)
end
@inline function abs_kernel(x::MC{N, Diff}, z::Interval{Float64}) where N

    xL = x.Intv.lo
    xU = x.Intv.hi
    eps_min, blank = mid3(xL, xU, 0.0)
    eps_max = (abs(xU) >= abs(xL)) ? xU : xL
    midcc = mid3v(x.cc, x.cv, eps_max)
    midcv = mid3v(x.cc, x.cv, eps_min)

    cc, dcc = cc_abs(midcc, xL, xU)
    cv, dcv = cv_abs(midcv, xL, xU)
    cc1, gdcc1 = cc_abs(x.cv, xL, xU)
    cv1, gdcv1 = cv_abs(x.cv, xL, xU)
    cc2, gdcc2 = cc_abs(x.cc, xL, xU)
    cv2, gdcv2 = cv_abs(x.cc, xL, xU)

    cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
    cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
    return MC{N, Diff}(cv, cc, z, cv_grad, cc_grad, x.cnst)
end
@inline abs(x::MC) = abs_kernel(x, abs(x.Intv))

@inline abs2_kernel(x::MC, z::Interval{Float64}) = sqr_kernel(x, z)
@inline abs2(x::MC) = abs2_kernel(x, pow(x.Intv,2))

@inline function correct_intersect(x::MC{N,T}, cv::Float64, cc::Float64, Intv::Interval{Float64}, cv_grad::SVector{N,Float64},
	                               cc_grad::SVector{N,Float64}, cnst::Bool) where {N, T <: RelaxTag}

    if cv - cc < MC_INTERSECT_TOL
		if diam(Intv) > 3.0*MC_INTERSECT_TOL
			cv = max(cv - MC_INTERSECT_TOL, Intv.lo)
			cc = min(cc + MC_INTERSECT_TOL, Intv.hi)
			if cv === Intv.lo
				cv_grad  = zero(SVector{N,Float64})
			end
			if cc === Intv.hi
				cc_grad = zero(SVector{N,Float64})
			end
			return MC{N,T}(cv, cc, Intv, cv_grad, cc_grad, cnst)
		else
			MC_INTERSECT_NOOP_FALLBACK && (return x)
		end
	end
	return MC{N,T}(NaN, NaN, Intv, cv_grad, cc_grad, false)
end

"""
$(TYPEDSIGNATURES)

Intersects two McCormick objects by computing setting `cv = max(cv1, cv2)`
and `cc = min(cc1, cc2)` with appropriate subgradients. Interval are intersected in the
usual fashion. Note that in a typical reverse propagation scheme this may result in
`cv > cc` due to rounding error. This is addressed in the following manner:
- if `cv - cc < MC_INTERSECT_TOL` and `diam(x.Intv ∩ y.Intv) > 3*MC_INTERSECT_TOL`,
then `cv = max(cv - MC_INTERSECT_TOL, Intv.lo)` and `cc = min(cc - MC_INTERSECT_TOL, Intv.hi)`.
Subgradients set to zero if `cv == Intv.lo` or `cc == Intv.hi`.
- if `cv - cc < MC_INTERSECT_TOL` and `diam(x.Intv ∩ y.Intv) < 3*MC_INTERSECT_TOL` and
`MC_INTERSECT_NOOP_FALLBACK == true` then return `x`.
- else return `MC{N,T}(NaN, NaN, x.Intv ∩ y.Intv, ...)`
"""
@inline function intersect(x::MC{N,T}, y::MC{N,T}) where {N, T<:Union{NS, MV}}

     Intv = x.Intv ∩ y.Intv
	 isempty(Intv) && (return empty(x))

	 cnst1 = (x.cc < y.cc) ? x.cnst : y.cnst
	 cnst2 = (x.cv > y.cv) ? x.cnst : y.cnst

     cc = (x.cc < y.cc) ? x.cc : y.cc
     cc_grad = (x.cc < y.cc) ? x.cc_grad : y.cc_grad

     cv = (x.cv > y.cv) ? x.cv : y.cv
     cv_grad = (x.cv > y.cv) ? x.cv_grad : y.cv_grad

	 if cv <= cc
		 return MC{N,T}(cv, cc, Intv, cv_grad, cc_grad, cnst1 && cnst2)
	 end

	 return correct_intersect(x, cv, cc, Intv, cv_grad, cc_grad, cnst1 && cnst2)
end

"""
$(TYPEDSIGNATURES)

Intersects two `MC{N, Diff}` in a manner than preserves differentiability. Interval are intersected in the
usual fashion. Note that in a typical reverse propagation scheme this may result in
`cv > cc` due to rounding error. This is addressed in the following manner:
- if `cv - cc < MC_INTERSECT_TOL` and `diam(x.Intv ∩ y.Intv) > 3*MC_INTERSECT_TOL`,
then `cv = max(cv - MC_INTERSECT_TOL, Intv.lo)` and `cc = min(cc - MC_INTERSECT_TOL, Intv.hi)`.
Subgradients set to zero if `cv == Intv.lo` or `cc == Intv.hi`.
- if `cv - cc < MC_INTERSECT_TOL` and `diam(x.Intv ∩ y.Intv) < 3*MC_INTERSECT_TOL` and
`MC_INTERSECT_NOOP_FALLBACK == true` then return `x`.
- else return `MC{N,T}(NaN, NaN, x.Intv ∩ y.Intv, ...)`
"""
@inline function intersect(x::MC{N, Diff}, y::MC{N, Diff}) where N

	Intv = intersect(x.Intv, y.Intv)
	isempty(Intv) && (return empty(x))

    max_MC = x - max(x - y, 0.0)
    min_MC = y - max(y - x, 0.0)

	if max_MC.cv <= min_MC.cc
    	return MC{N, Diff}(max_MC.cv, min_MC.cc, Intv, max_MC.cv_grad, min_MC.cc_grad, x.cnst && y.cnst)
	end

	return correct_intersect(x, max_MC.cv, min_MC.cc, Intv, max_MC.cv_grad, min_MC.cc_grad, x.cnst && y.cnst)
end

@inline function intersect(x::MC{N, T}, y::Interval{Float64}) where {N, T<:Union{NS,MV}}
	cnst1 = x.cnst
	cnst2 = x.cnst
	if x.cv >= y.lo
  		cv = x.cv
  		cv_grad = x.cv_grad
	else
		cnst1 = true
  		cv = y.lo
  		cv_grad = zero(SVector{N,Float64})
	end
	if x.cc <= y.hi
  		cc = x.cc
  		cc_grad = x.cc_grad
	else
		cnst2 = true
  		cc = y.hi
  		cc_grad = zero(SVector{N,Float64})
	end
	if cv <= cc
		return MC{N, T}(cv, cc, intersect(x.Intv, y), cv_grad, cc_grad, cnst1 && cnst2)
	end
	return correct_intersect(x, cv, cc, intersect(x.Intv, y), cv_grad, cc_grad, cnst1 && cnst2)
end
@inline function intersect(x::MC{N, Diff}, y::Interval{Float64}) where N
     max_MC = x - max(x - y, 0.0)
     min_MC = y - max(y - x, 0.0)
	 if max_MC.cv <= min_MC.cc
		 MC{N, Diff}(max_MC.cv, min_MC.cc, intersect(x.Intv, y), max_MC.cv_grad, min_MC.cc_grad, x.cnst)
	 end
     return correct_intersect(x, max_MC.cv, min_MC.cc, intersect(x.Intv, y), max_MC.cv_grad, min_MC.cc_grad, x.cnst && y.cnst)
end

@inline function intersect(c::Float64, x::MC{N,T}) where {N, T<:RelaxTag}
	isempty(x) && (return empty(x))
	isnan(x) && (return nan(x))

	intv = intersect(x.Intv, Interval{Float64}(c))
	isempty(intv) && (return empty(x))

	cv = max(c, x.cv)
	cc = min(c, x.cc)

	cv_grad = (cv == c) ? zero(SVector{N,Float64}) : x.cv_grad
	cc_grad = (cc == c) ? zero(SVector{N,Float64}) : x.cc_grad
	if cv <= c <= cc
		return MC{N, T}(c, c, intv, cv_grad, cc_grad, true)
	end
	correct_intersect(x, cv, cc, intv, cv_grad, cc_grad, cnst1 && cnst2)
end
@inline intersect(x::MC{N,T}, c::Float64) where {N, T<:RelaxTag} = intersect(c, x)

@inline in(a::Int, x::MC) = in(a, x.Intv)
@inline in(a::T, x::MC) where T<:AbstractFloat = in(a, x.Intv)
@inline isempty(x::MC) = isempty(x.Intv)

"""
xlogx

The function `xlogx` is defined as `xlogx(x) = x*log(x)`.
"""
xlogx(x::Float64) = x*log(x)
xlogx_deriv(x::Float64) = log(x) + 1.0
xlogx_deriv2(x::Float64) = 1.0/x
cc_xlogx(x::Float64, xL::Float64, xU::Float64) = dline_seg(xlogx, xlogx_deriv, x, xL, xU)
cv_xlogx(x::Float64, xL::Float64, xU::Float64) = xlogx(x), xlogx_deriv(x)
function xlogx(x::Interval{Float64})
	min_pnt = one(Interval{Float64})/exp(one(Interval{Float64}))
	xlogx_xL = Interval(x.lo)*log(Interval(x.lo))
	xlogx_xU = Interval(x.hi)*log(Interval(x.hi))
	xlogx_min_bool = isdisjoint(min_pnt, x)
	if isdisjoint(min_pnt, x)
		min_val = min(xlogx_xL.lo, xlogx_xU.hi)
	else
		min_val = -min_pnt.hi
	end
	Interval{Float64}(min_val, max(xlogx_xL.lo, xlogx_xU.hi))
end
@inline function xlogx_kernel(x::MC{N, T}, y::Interval{Float64}) where {N,T<:Union{NS,MV}}
    xL = x.Intv.lo
    xU = x.Intv.hi
    eps_max = xlogx(xU) > xlogx(xL) ?  xU : xL
    eps_min = in(-1.0/exp(1.0), x) ? -1.0/exp(1.0) : (xlogx(xU) > xlogx(xL) ?  xL : xU)
    midcc, cc_id = mid3(x.cc, x.cv, eps_max)
    midcv, cv_id = mid3(x.cc, x.cv, eps_min)
    cc, dcc = cc_xlogx(midcc, xL, xU)
    cv, dcv = cv_xlogx(midcv, xL, xU)
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
    cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
    return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function xlogx_kernel(x::MC{N,Diff}, y::Interval{Float64}) where N
    xL = x.Intv.lo
    xU = x.Intv.hi
    eps_max = xlogx(xU) > xlogx(xL) ?  xU : xL
    eps_min = in(-1.0/exp(1.0), x) ? -1.0/exp(1.0) : (xlogx(xU) > xlogx(xL) ?  xL : xU)
    midcc = mid3v(x.cv, x.cc, eps_max)
    midcv = mid3v(x.cv, x.cc, eps_min)
    cc, dcc = cc_xlogx(midcc, xL, xU)
    cv, dcv = cv_xlogx(midcv, xL, xU)
    gcc1, gdcc1 = cc_xlogx(x.cv, xL, xU)
    gcv1, gdcv1 = cv_xlogx(x.cv, xL, xU)
    gcc2, gdcc2 = cc_xlogx(x.cc, xL, xU)
    gcv2, gdcv2 = cv_xlogx(x.cc, xL, xU)
    cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
    cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
    return MC{N,Diff}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
function xlogx(x::MC{N,T}) where {N, T <: RelaxTag}
	xlogx_kernel(x, xlogx(x.Intv))
end

"""
arh

The arrhenius function `arh` is defined as `arh(x) = exp(-k/x)`.
"""
arh(x, k) = exp(-k/x)
arh(x::Float64, k::MC) = arh_kernel(x, k, x.Intv)
arh_kernel(x::Float64, k::MC, z::Interval{Float64}) = exp(-k/x)
arh(x::Float64, k::Float64) = exp(-k/x)
arh_deriv(x::Float64, k::Float64) = k*exp(-k/x)/x^2
arh_grad(x::Float64, k::Float64) = (k*exp(-k/x)/x^2, -exp(-k/x)/x)
@inline function arh_env(x::Float64, y::Float64, k::Float64)
    (x^2)*(exp((-k/y) + (k/x)) - 1.0) - k*(y + x)
end
@inline function arh_envm(x::Float64, y::Float64, k::Float64)
    (x^2)*(exp((-k/y) + (k/x)) - 1.0) + k*(y + x)
end
function cv_arh(x::Float64, xL::Float64, xU::Float64, k::Float64, p::Float64)
	if k > 0.0
	    (xL >= k/2.0) && (return dline_seg(arh, arh_deriv, x, xL, xU, k)..., p)
	    (xU <= k/2.0) && (return arh(x, k), arh_deriv(x, k), p)
	    if p === Inf
	        p, flag = secant(0.0, k/2.0, 0.0, k/2.0, arh_env, xU, k)
	        flag && (p = golden_section(0.0, k/2.0, arh_env, xU, k))
	    end
	    (x <= p) && (return arh(x, k), arh_deriv(x, k), p)
	    return dline_seg(arh, arh_deriv, x, p, xU, k)..., p
	end
	return arh(x, k), arh_deriv(x, k), 0.0
end
function cc_arh(x::Float64, xL::Float64, xU::Float64, k::Float64, p::Float64)
	if k > 0.0
		(xL >= k/2.0) && (return arh(x,k), arh_deriv(x,k), p)
		(xU <= k/2.0) && (return dline_seg(arh, arh_deriv, x, xL, xU, k)..., p)
		if p === Inf
			p, flag = secant(k/2.0, xU, k/2.0, xU, arh_envm, xL, k)
			flag && (p = golden_section(k/2.0, xU, arh_envm, xL, k))
		end
		(x >= p) && (return arh(x,k), arh_deriv(x,k), p)
		return dline_seg(arh, arh_deriv, x, xL, p, k)..., p
	end
	return arh(x,k), arh_deriv(x,k), 0.0
end
function arh_kernel(x::MC{N,T}, k::Float64, z::Interval{Float64},
	                cv_p::Float64, cc_p::Float64) where {N,T<:Union{NS,MV}}
	(k == 0.0) && return one(MC{N,T})
	in(0.0, x) && throw(DomainError(0.0))
	xL = x.Intv.lo
    xU = x.Intv.hi
	eps_min = xL
	eps_max = xU
    midcv, cv_id = mid3(x.cc, x.cv, eps_min)
    midcc, cc_id = mid3(x.cc, x.cv, eps_max)
    cv, dcv, cv_p = cv_arh(midcv, xL, xU, k, cv_p)
    cc, dcc, cc_p = cc_arh(midcc, xL, xU, k, cc_p)
    cv_grad = mid_grad(x.cv_grad, x.cc_grad, cv_id)*dcv
    cc_grad = mid_grad(x.cv_grad, x.cc_grad, cc_id)*dcc
    cv, cc, cv_grad, cc_grad = cut(z.lo, z.hi, cv, cc, cv_grad, cc_grad)
    return MC{N, T}(cv, cc, z, cv_grad, cc_grad, x.cnst), cv_p, cc_p
end
function arh_kernel(x::MC{N,Diff}, k::Float64, z::Interval{Float64},
	                cv_p::Float64, cc_p::Float64) where N
	(k == 0.0) && return one(MC{N,Diff})
	in(0.0, x) && throw(DomainError(0.0))
	xL = x.Intv.lo
	xU = x.Intv.hi
	eps_min = xL
	eps_max = xU
	midcv, cv_id = mid3(x.cv, x.cc, eps_min)
	midcc, cc_id = mid3(x.cv, x.cc, eps_max)
	cv, dcv, cv_p = cv_arh(midcv, xL, xU, k, cv_p)
	cc, dcc, cc_p = cc_arh(midcc, xL, xU, k, cc_p)
	gcv1, gdcv1, cv_p = cv_arh(x.cv, xL, xU, k, cv_p)
	gcc1, gdcc1, cc_p = cc_arh(x.cv, xL, xU, k, cc_p)
	gcv2, gdcv2, cv_p = cv_arh(x.cc, xL, xU, k, cv_p)
	gcc2, gdcc2, cc_p = cc_arh(x.cc, xL, xU, k, cc_p)
	cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
	cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
	return MC{N,Diff}(cv, cc, z, cv_grad, cc_grad, x.cnst), cv_p, cc_p
end
function arh_kernel(x::Float64, k::MC{N,T}, z::Interval{Float64},
	                cv_p::Float64, cc_p::Float64) where {N,T<:Union{NS,MV}}
	exp(-k/x), 0.0, 0.0
end
function arh_kernel(x::Float64, k::MC{N,Diff}, z::Interval{Float64},
	                cv_p::Float64, cc_p::Float64) where N
	exp(-k/x), 0.0, 0.0
end
function arh(x::MC{N,T}, k::Float64) where {N, T <: RelaxTag}
	yMC, tp1, tp2 = arh_kernel(x, k, exp(-k/x.Intv), Inf, Inf)
	return yMC
end

"""
expax

The `expax` function is defined as `expax(x, a) = x*exp(a*x)`.

Form defined in Najman, Jaromił, Dominik Bongartz, and Alexander Mitsos.
"Relaxations of thermodynamic property and costing models in process engineering."
Computers & Chemical Engineering 130 (2019): 106571.
"""
xexpax(x, a) = x*exp(a*x)
xexpax(x::Float64, a::Float64) = x*exp(a*x)
function xexpax(x::Interval{Float64}, a::Float64)
	(a == 0.0) && return x
	xL = x.lo
	xU = x.hi
	fxL = xexpax(Interval(xL), Interval(a)).lo
	fxU = xexpax(Interval(xU), Interval(a)).hi
	zpnt = -1.0/a
	if a > 0.0
		yL = (xL <= zpnt <= xU) ? xexpax(zpnt, a) : ((fxL <= fxU) ? fxL : fxU)
		yU = (fxL >= fxU) ? fxL : fxU
	else
		yL = (fxL <= fxU) ? fxL : fxU
		yU = (xL <= zpnt <= xU) ? xexpax(zpnt, a) : ((fxL >= fxU) ? fxL : fxU)
	end
	Interval(yL, yU)
end
xexpax_deriv(x::Float64, a::Float64) = exp(a*x)*(a*x + 1.0)
xexpax_grad(x::Float64, a::Float64) = (exp(a*x)*(a*x + 1.0), exp(a*x)*x^2)
@inline function xexpax_env(x::Float64, y::Float64, a::Float64)
    (y - x)*xexpax_deriv(x, a) - (xexpax(y, a) - xexpax(x, a))
end
@inline function xexpax_envm(x::Float64, y::Float64, a::Float64)
    (x - y) - (xexpax(x, a) - xexpax(y, a))/xexpax_deriv(x, a)
end
function cv_xexpax(x::Float64, xL::Float64, xU::Float64, a::Float64, p::Float64)
	if a > 0.0
	    (xU <= -2.0/a) && (return dline_seg(xexpax, xexpax_deriv, x, xL, xU, a)..., p)
	    (xL >= -2.0/a) && (return xexpax(x, a), xexpax_deriv(x, a), p)
	    if p === Inf
	        p, flag = secant(-2.0/a, xU, -2.0/a, xU, xexpax_env, xL, a)
	        flag && (p = golden_section(-2.0/a, xU, xexpax_env, xL, a))
	    end
	    (x <= p) && (return dline_seg(xexpax, xexpax_deriv, x, p, xL, a)..., p)
	    return xexpax(x, a), xexpax_deriv(x, a), p
	end
	@show "arc 1 cv"
	(xL >= -2.0/a) && (return xexpax(x, a), xexpax_deriv(x, a), p)
	@show "arc 2 cv"
	(xU <= -2.0/a) && (return dline_seg(xexpax, xexpax_deriv, x, xL, xU, a)..., p)
	if p === Inf
		p, flag = secant(-2.0/a, xU, -2.0/a, xU, xexpax_env, xL, a)
		flag && (p = golden_section(-2.0/a, xU, xexpax_env, xL, a))
	end
	@show "arc 3 cv"
	(x >= p) && (return xexpax(x, a), xexpax_deriv(x, a), p)
	@show "arc 4 cv"
	return dline_seg(xexpax, xexpax_deriv, x, xL, p, a)..., p
end
function cc_xexpax(x::Float64, xL::Float64, xU::Float64, a::Float64, p::Float64)
	if a > 0.0
		(xL >= -2.0/a) && (return dline_seg(xexpax, xexpax_deriv, x, xL, xU, a)..., p)
		(xU <= -2.0/a) && (return xexpax(x, a), xexpax_deriv(x, a), p)
		if p === Inf
			p, flag = secant(xL, -2.0/a, xL, -2.0/a, xexpax_envm, xU, a)
			flag && (p = golden_section(xL, -2.0/a, xexpax_envm, xU, a))
		end
		(x >= p) && (return xexpax(x,a), xexpax_deriv(x,a), p)
		return dline_seg(xexpax, xexpax_deriv, x, xL, p, a)..., p
	end
	@show "arc 1 cc"
	(xL >= -2.0/a) && (return dline_seg(xexpax, xexpax_deriv, x, xL, xU, a)..., p)
	@show "arc 2 cc"
	(xU <= -2.0/a) && (return xexpax(x,a), xexpax_deriv(x,a), p)
	if p === Inf
		p, flag = secant(xL, -2.0/a, xL, -2.0/a, xexpax_env, xU, a)
		flag && (p = golden_section(xL, a/2.0, xexpax_env, xU, a))
	end
	@show "arc 3 cc = $p"
	(x <= p) && (return xexpax(x,a), xexpax_deriv(x,a), p)
	@show "arc 3 cc"
	return dline_seg(xexpax, xexpax_deriv, x, p, xU, a)..., p
end
function xexpax_kernel(x::Float64, a::MC{N,T}, z::Interval{Float64},
	                   cv_p::Float64, cc_p::Float64) where {N,T<:Union{NS,MV}}
	x*exp(a*x), 0.0, 0.0
end
function xexpax_kernel(x::Float64, a::MC{N,Diff}, z::Interval{Float64},
	                   cv_p::Float64, cc_p::Float64) where N
	x*exp(a*x), 0.0, 0.0
end
function xexpax_kernel(x::MC{N,T}, a::Float64, z::Interval{Float64},
	                   cv_p::Float64, cc_p::Float64) where {N,T<:Union{NS,MV}}
	(a == 0.0) && return one(MC{N,T})
	xL = x.Intv.lo
    xU = x.Intv.hi
	zpnt = -1.0/a
	fxL = xexpax(xL, a)
	fxU = xexpax(xU, a)
	if a > 0.0
		eps_min = (xL <= zpnt <= xU) ? zpnt : ((fxL <= fxU) ? xL : xU)
		eps_max = (fxL >= fxU) ? xL : xU
	else
		eps_min = (fxL <= fxU) ? xL : xU
		eps_max = (xL <= zpnt <= xU) ? zpnt : ((fxL >= fxU) ? xL : xU)
	end
    midcv, cv_id = mid3(x.cc, x.cv, eps_min)
    midcc, cc_id = mid3(x.cc, x.cv, eps_max)
    cv, dcv, cv_p = cv_xexpax(midcv, xL, xU, a, cv_p)
    cc, dcc, cc_p = cc_xexpax(midcc, xL, xU, a, cc_p)
    cv_grad = mid_grad(x.cv_grad, x.cc_grad, cv_id)*dcv
    cc_grad = mid_grad(x.cv_grad, x.cc_grad, cc_id)*dcc
    cv, cc, cv_grad, cc_grad = cut(z.lo, z.hi, cv, cc, cv_grad, cc_grad)
    return MC{N, T}(cv, cc, z, cv_grad, cc_grad, x.cnst), cv_p, cc_p
end
function xexpax_kernel(x::MC{N,Diff}, a::Float64, z::Interval{Float64},
	                   cv_p::Float64, cc_p::Float64) where N
	(a == 0.0) && return one(MC{N,Diff})
	xL = x.Intv.lo
	xU = x.Intv.hi
	zpnt = -1.0/a
	fxL = xexpax(xL, a)
	fxU = xexpax(xU, a)
	if a > 0.0
		eps_min = (xL <= zpnt <= xU) ? zpnt : ((fxL <= fxU) ? xL : xU)
		eps_max = (fxL >= fxU) ? xL : xU
	else
		eps_min = (fxL <= fxU) ? xL : xU
		eps_max = (xL <= zpnt <= xU) ? zpnt : ((fxL >= fxU) ? xL : xU)
	end
	midcv, cv_id = mid3(x.cv, x.cc, eps_min)
	midcc, cc_id = mid3(x.cv, x.cc, eps_max)
	cv, dcv, cv_p = cv_xexpax(midcv, xL, xU, a, cv_p)
	cc, dcc, cc_p = cc_xexpax(midcc, xL, xU, a, cc_p)
	gcv1, gdcv1, cv_p = cv_xexpax(x.cv, xL, xU, a, cv_p)
	gcc1, gdcc1, cc_p = cc_xexpax(x.cv, xL, xU, a, cc_p)
	gcv2, gdcv2, cv_p = cv_xexpax(x.cc, xL, xU, a, cv_p)
	gcc2, gdcc2, cc_p = cc_xexpax(x.cc, xL, xU, a, cc_p)
	cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
	cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
	return MC{N,Diff}(cv, cc, z, cv_grad, cc_grad, x.cnst), cv_p, cc_p
end
xexpax(x::Float64, a::MC) = x*exp(a*x)
function xexpax(x::MC{N,T}, a::Float64) where {N, T <: RelaxTag}
	intv_xexpax = xexpax(x.Intv, a)
	yMC, tp1, tp2 = xexpax_kernel(x, a, intv_xexpax, Inf, Inf)
	return yMC
end
