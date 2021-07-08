# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A McCormick operator library in Julia
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# src/forward_operators/other.jl
# Defines isempty, empty, isnan, step, sign, abs, intersect, in.
#############################################################################


empty(x::MCNoGrad) = MCNoGrad(Inf, -Inf, Interval{Float64}(Inf,-Inf), false)
interval_MC(x::MCNoGrad) = MCNoGrad(x.Intv)

isnan(x::MCNoGrad) = isnan(x.cc) || isnan(x.cv)
isfinite(x::MCNoGrad) = isfinite(x.cc) && isfinite(x.cv) && isfinite(x.Intv)

@inline function step_kernel(t::Union{NS,MV}, x::MCNoGrad, z::Interval{Float64})
	xL = x.Intv.lo
	xU = x.Intv.hi
	midcc, cc_id = mid3(x.cc, x.cv, xU)
	midcv, cv_id = mid3(x.cc, x.cv, xL)
	cc, dcc = cc_step_NS(midcc, xL, xU)
	cv, dcv = cv_step_NS(midcv, xL, xU)
	return MCNoGrad(cv, cc, z, x.cnst), cv_id, cc_id, dcv, dcc
end
@inline step(t::Union{NS,MV}, x::MCNoGrad) = step_kernel(t, x, step(x.Intv))

@inline function abs_kernel(t::Union{NS,MV}, x::MC{N, T}, z::Interval{Float64})
	xL = x.Intv.lo
	xU = x.Intv.hi
	eps_min = mid3v(xL, x.Intv.hi, 0.0)
	eps_max = (abs(xU) >= abs(xL)) ? xU : xL
	midcc, cc_id = mid3(x.cc, x.cv, eps_max)
	midcv, cv_id = mid3(x.cc, x.cv, eps_min)
	cc, dcc = cc_abs(midcc, xL, xU)
	cv, dcv = cv_abs_NS(midcv, xL, xU)
	return MCNoGrad(cv, cc, z, x.cnst), dcv, dcc
end
@inline abs(t::Union{NS,MV}, x::MCNoGrad) = abs_kernel(t, x, abs(x.Intv))

@inline abs2_kernel(t::Union{NS,MV}, x::MCNoGrad, z::Interval{Float64}) = sqr_kernel(t, x, z)
@inline abs2(t::Union{NS,MV}, x::MCNoGrad) = abs2_kernel(t, x, pow(x.Intv,2))

@inline function xlogx_kernel(t::Union{NS,MV}, x::MCNoGrad, y::Interval{Float64})
    xL = x.Intv.lo
    xU = x.Intv.hi
    eps_max = xlogx(xU) > xlogx(xL) ?  xU : xL
    eps_min = in(1.0/exp(1.0), x) ? 1.0/exp(1.0) : (xlogx(xU) > xlogx(xL) ?  xL : xU)
    midcc, cc_id = mid3(x.cc, x.cv, eps_max)
    midcv, cv_id = mid3(x.cc, x.cv, eps_min)
    cc, dcc = cc_xlogx(midcc, xL, xU)
    cv, dcv = cv_xlogx(midcv, xL, xU)
    return MCNoGrad(cv, cc, y, x.cnst), cv_id, cc_id, dcv, dcc
end
xlogx(t::Union{NS,MV}, x::MCNoGrad) = xlogx_kernel(t, x, xlogx(x.Intv))

function arh_kernel(t::Union{NS,MV}, x::MCNoGrad, k::Float64, z::Interval{Float64}, cv_p::Float64, cc_p::Float64)
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
    return MCNoGrad(cv, cc, z, x.cnst), cv_id, cc_id, cv_p, cc_p
end
function arh_kernel(x::Float64, k::MC{N,T}, z::Interval{Float64}, cv_p::Float64, cc_p::Float64) where {N,T<:Union{NS,MV}}
	zMC, cv_id, cc_id = exp(-k/x)
	zMC, cv_id, cc_id, 0.0, 0.0
end
function arh(x::MC{N,T}, k::Float64) where {N, T <: RelaxTag}
	yMC, _, _, _, _ = arh_kernel(x, k, exp(-k/x.Intv), Inf, Inf)
	return yMC
end

#=
########### Defines sign
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
@inline intersect(c::Interval{Float64}, x::MC{N,T}) where {N, T<:RelaxTag} = intersect(c, x)

@inline in(a::Int, x::MC) = in(a, x.Intv)
@inline in(a::T, x::MC) where T<:AbstractFloat = in(a, x.Intv)
@inline isempty(x::MC) = isempty(x.Intv)

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
function xexpax_grad(g, x::Float64, a::Float64)
	g[1] = exp(a*x)*(a*x + 1.0)
	g[2] = exp(a*x)*x^2
	nothing
end
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
	(xL >= -2.0/a) && (return xexpax(x, a), xexpax_deriv(x, a), p)
	(xU <= -2.0/a) && (return dline_seg(xexpax, xexpax_deriv, x, xL, xU, a)..., p)
	if p === Inf
		p, flag = secant(-2.0/a, xU, -2.0/a, xU, xexpax_env, xL, a)
		flag && (p = golden_section(-2.0/a, xU, xexpax_env, xL, a))
	end
	(x >= p) && (return xexpax(x, a), xexpax_deriv(x, a), p)
	return dline_seg(xexpax, xexpax_deriv, x, xL, p, a)..., p
end
function cc_xexpax(x::Float64, xL::Float64, xU::Float64, a::Float64, p::Float64)
	if a > 0.0
		(xL >= -2.0/a) && (return dline_seg(xexpax, xexpax_deriv, x, xL, xU, a)..., p)
		(xU <= -2.0/a) && (return xexpax(x, a), xexpax_deriv(x, a), p)
		if p === Inf
			p, flag = secant(xL, -2.0/a, xL, -2.0/a, xexpax_env, xU, a)
			flag && (p = golden_section(xL, -2.0/a, xexpax_env, xU, a))
		end
		(x <= p) && (return xexpax(x,a), xexpax_deriv(x,a), p)
		return dline_seg(xexpax, xexpax_deriv, x, p, xU, a)..., p
	end
	(xL >= -2.0/a) && (return dline_seg(xexpax, xexpax_deriv, x, xL, xU, a)..., p)
	(xU <= -2.0/a) && (return xexpax(x,a), xexpax_deriv(x,a), p)
	if p === Inf
		p, flag = secant(xL, -2.0/a, xL, -2.0/a, xexpax_envm, xU, a)
		flag && (p = golden_section(xL, a/2.0, xexpax_envm, xU, a))
	end
	(x <= p) && (return xexpax(x, a), xexpax_deriv(x, a), p)
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

"""
xabsx

The function `xabsx` is defined as `xabsx(x) = x*abs(x)`.
"""
xabsx(x::Float64) = x*abs(x)
xabsx_deriv(x::Float64) = 2*abs(x)
xabsx_deriv2(x::Float64) = 2*abs(x)/x
@inline function cc_xabsx(x::Float64, xL::Float64, xU::Float64)
	(xU <= 0.0) && (return xabsx(x), xabsx_deriv(x))
	(0.0 <= xL) && (return dline_seg(xabsx, xabsx_deriv, x, xL, xU))

	root_f(x, xL, xU) = (xabsx(xU) - xabsx(x)) - 2*(xU - x)*abs(x)
	root_df(x, xL, xU) = xabsx_deriv(x) - xU*xabsx_deriv2(x)
	inflection, flag = newton(xL, xL, 0.0, root_f, root_df, xL, xU)
	flag && (inflection = golden_section(xL, xU, root_f, xL, xU))
	if x <= inflection
		return xabsx(x), xabsx_deriv(x)
	else
		return dline_seg(xabsx, xabsx_deriv, x, inflection, xU)
	end
end
@inline function cv_xabsx(x::Float64, xL::Float64, xU::Float64)
	(xU <= 0.0) && (return dline_seg(xabsx, xabsx_deriv, x, xL, xU))
	(0.0 <= xL) && (return xabsx(x), xabsx_deriv(x))

	root_f(x, xL, xU) = (xabsx(x) - xabsx(xL)) - 2*(x - xL)*abs(x)
	root_df(x, xL, xU) = -xabsx_deriv(x) + xL*xabsx_deriv2(x)
	inflection, flag = newton(xU, 0.0, xU, root_f, root_df, xL, xU)
	flag && (inflection = golden_section(xL, xU, root_f, xL, xU))
	if x <= inflection
		return dline_seg(xabsx, xabsx_deriv, x, xL, inflection)
	else
		return xabsx(x), xabsx_deriv(x)
	end
end
function xabsx(x::Interval{Float64})
	xabsx_xL = Interval(x.lo)*abs(Interval(x.lo))
	xabsx_xU = Interval(x.hi)*abs(Interval(x.hi))
	Interval{Float64}(xabsx_xL.lo, xabsx_xU.hi)
end
@inline function xabsx_kernel(x::MC{N, T}, y::Interval{Float64}) where {N,T<:Union{NS,MV}}
	xL = x.Intv.lo
	xU = x.Intv.hi
	eps_max = xU
	eps_min = xL
	midcc, cc_id = mid3(x.cc, x.cv, eps_max)
	midcv, cv_id = mid3(x.cc, x.cv, eps_min)
	cc, dcc = cc_xabsx(midcc, xL, xU)
	cv, dcv = cv_xabsx(midcv, xL, xU)
	cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
	cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
	cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
	return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function xabsx_kernel(x::MC{N, T}, y::Interval{Float64}) where {N,T<:Diff}
	xL = x.Intv.lo
	xU = x.Intv.hi
	eps_min = xL
	eps_max = xU
	midcc, cc_id = mid3(x.cc, x.cv, eps_max)
	midcv, cv_id = mid3(x.cc, x.cv, eps_min)
	cc, dcc = cc_xabsx(midcc, xL, xU)
	cv, dcv = cv_xabsx(midcv, xL, xU)
	cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
	cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
	return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
function xabsx(x::MC{N,T}) where {N, T <: RelaxTag}
	xabsx_kernel(x, xabsx(x.Intv))
end
=#