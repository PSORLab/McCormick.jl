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

	intv = intersect(x.Intv, c)
	isempty(intv) && (return empty(x))

	cv = max(c, x.cv)
	cc = min(c, x.cc)

	cv_grad = (cv == c) ? zero(SVector{N,Float64}) : x.cv_grad
	cc_grad = (cc == c) ? zero(SVector{N,Float64}) : x.cc_grad
	if cv <= c <= cc
		return MC{N, T}(c, c, intv(x.Intv, c), cv_grad, cc_grad, true)
	end
	correct_intersect(x, cv, cc, intv, cv_grad, cc_grad, cnst1 && cnst2)
end
@inline intersect(x::MC{N,T}, c::Float64) where {N, T<:RelaxTag} = intersect(c, x)

@inline in(a::Int, x::MC) = in(a, x.Intv)
@inline in(a::T, x::MC) where T<:AbstractFloat = in(a, x.Intv)
@inline isempty(x::MC) = isempty(x.Intv)

#=

Relaxation given in Najman J, Bongartz D, Mitsos A. "Relaxations of
thermodynamic property and costing models in process engineering.
Computers & Chemical Engineering. 2019 Nov 2;130:106571." for
f(x) = x*exp(a*x)
=#
#=
@inline function cc_xexp(x::Float64, xL::Float64, xU::Float64, a::Float64)
   xstar = -2.0*a
   if a > 0.0
	   xstar
	   return dline_seg(abs, sign, x, xL, xU)
   else
	   return dline_seg(abs, sign, x, xL, xU)
   end
end
@inline cv_xexp(x::Float64,xL::Float64,xU::Float64) = abs(x), sign(x)
function xexp(a::Float64, x::MC{N,T}) where {N, T<:RelaxTag}
end
xexp(x::MC{N,T}) where {N, T <: RelaxTag} = xexp(1.0, x)
=#

xlog(x::Float64) where T = x*log(x)
dxlog(x::Float64) where T = log(x) + 1
cc_xlog(x::Float64, xL::Float64, xU::Float64) = dline_seg(xlog, dxlog, x, xL, xU)
cv_xlog(x::Float64, xL::Float64, xU::Float64) = xlog(x), dxlog(x)
function xlog(x::MC{N,T}) where {N, T <: Union{NS,MV}}
end
@inline function xlog_kernel(x::MC{N, T}, y::Interval{Float64}) where {N,T<:Union{NS,MV}}
    xL = x.Intv.lo
    xU = x.Intv.hi
    eps_max = xlog(xU) > xlog(xL) ?  xU : xL
    eps_min = in(-1.0/exp(1.0), x) ? -1.0/exp(1.0) : (xlog(xU) > xlog(xL) ?  xL : xU)
    midcc, cc_id = mid3(x.cc, x.cv, eps_max)
    midcv, cv_id = mid3(x.cc, x.cv, eps_min)
    cc, dcc = cc_xlog(midcc, xL, xU)
    cv, dcv = cv_xlog(midcv, xL, xU)
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
    cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
    return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function xlog_kernel(x::MC{N,Diff}, y::Interval{Float64}) where N
    xL = x.Intv.lo
    xU = x.Intv.hi
    eps_max = xlog(xU) > xlog(xL) ?  xU : xL
    eps_min = in(-1.0/exp(1.0), x) ? -1.0/exp(1.0) : (xlog(xU) > xlog(xL) ?  xL : xU)
    midcc = mid3v(x.cv, x.cc, eps_max)
    midcv = mid3v(x.cv, x.cc, eps_min)
    cc, dcc = cc_xlog(midcc, xL, xU)
    cv, dcv = cv_xlog(midcv, xL, xU)
    gcc1, gdcc1 = cc_xlog(x.cv, xL, xU)
    gcv1, gdcv1 = cv_xlog(x.cv, xL, xU)
    gcc2, gdcc2 = cc_xlog(x.cc, xL, xU)
    gcv2, gdcv2 = cv_xlog(x.cc, xL, xU)
    cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
    cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
    return MC{N,Diff}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline xlog(x::MC) = xlog_kernel(x, xlog(x.Intv))

# Log-mean
