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
	 (xU <= 0.0) && (return 0.0, 0.0)
	 (xL >= 0.0) && (return 1.0, 0.0)
	 (x >= 0.0) ? ((x/xU)^2, 2.0*x/xU^2) : (0.0, 0.0)
end
@inline function cc_step(x::Float64, xL::Float64, xU::Float64)
	 (xU <= 0.0) && (return 0.0, 0.0)
	 (xL >= 0.0) && (return 1.0, 0.0)
	 (x >= 0.0) ? (1.0, 0.0) : (1.0-(x/xL)^2, -2.0*x/xL^2)
end
@inline function cv_step_NS(x::Float64, xL::Float64, xU::Float64)
	 (xU <= 0.0) && (return 0.0, 0.0)
	 (xL >= 0.0) && (return 1.0, 0.0)
	 (x > 0.0) ? (x/xU, 1.0/xU) : (0.0, 0.0)
end
@inline function cc_step_NS(x::Float64, xL::Float64, xU::Float64)
	 (xU <= 0.0) && (return 0.0, 0.0)
	 (xL >= 0.0) && (return 1.0, 0.0)
	 (x >= 0.0) ? (1.0, 0.0) : ((1.0 - (x/xL)), (-x/xL))
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
	(xL >= 0.0) && (return x, 1.0)
	(xU <= 0.0) && (return -x, -1.0)
	if (x >= 0.0)
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

@inline function intersect(x_mc::MC{N,T}, x_mc_int::MC{N,T}) where {N, T<:Union{NS,MV}}
     Intv = x_mc.Intv ∩ x_mc_int.Intv
	 cnst1 = false
	 cnst2 = false
     if x_mc.cc < x_mc_int.cc
		 cnst1 = x_mc.cnst
         cc = x_mc.cc
         cc_grad = x_mc.cc_grad
     else
		 cnst1 = x_mc_int.cnst
         cc = x_mc_int.cc
         cc_grad = x_mc_int.cc_grad
     end
     if x_mc.cv > x_mc_int.cv
		 cnst2 = x_mc.cnst
         cv = x_mc.cv
         cv_grad = x_mc.cv_grad
     else
		 cnst2 = x_mc_int.cnst
         cv = x_mc_int.cv
         cv_grad = x_mc_int.cv_grad
     end
     MC{N,T}(cv, cc, (x_mc.Intv ∩ x_mc_int.Intv), cv_grad, cc_grad, cnst1 && cnst2)
end

@inline function intersect(x::MC{N, Diff}, y::MC{N, Diff}) where N
    max_MC = x - max(x - y, 0.0)
    min_MC = y - max(y - x, 0.0)
    return MC{N, Diff}(max_MC.cv, min_MC.cc, intersect(x.Intv, y.Intv),
	                   max_MC.cv_grad, min_MC.cc_grad, (x.cnst && y.cnst))
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
	(cc < y.lo) && (return nan(MC{N,T}))
	(y.hi < cv) && (return nan(MC{N,T}))
	return MC{N, T}(cv, cc, intersect(x.Intv, y), cv_grad, cc_grad, cnst1 && cnst2)
end
@inline function intersect(x::MC{N, Diff}, y::Interval{Float64}) where N
     max_MC = x - max(x - y, 0.0)
     min_MC = y - max(y - x, 0.0)
     return MC{N, Diff}(max_MC.cv, min_MC.cc, intersect(x.Intv, y),
	                    max_MC.cv_grad, min_MC.cc_grad, x.cnst)
end

@inline in(a::Int, x::MC) = in(a, x.Intv)
@inline in(a::T, x::MC) where T<:AbstractFloat = in(a, x.Intv)
@inline isempty(x::MC) = isempty(x.Intv)
