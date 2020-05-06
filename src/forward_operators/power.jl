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
# src/forward_operators/power.jl
# Contains definitions of inv and ^.
#############################################################################

# defines square operator
@inline sqr(x::Float64) = x*x
@inline cv_sqr_NS(x::Float64, xL::Float64, xU::Float64) = x*x
@inline dcv_sqr_NS(x::Float64, xL::Float64, xU::Float64) = 2.0*x
@inline function cc_sqr(x::Float64, xL::Float64, xU::Float64)
	if (xU > xL)
		cc = xL^2 + (xL + xU)*(x - xL)
	else
		cc = xU^2
	end
	return cc
end
@inline dcc_sqr(x::Float64, xL::Float64, xU::Float64) = (xU > xL) ? (xL + xU) : 0.0
@inline function cv_sqr(x::Float64, xL::Float64, xU::Float64)
    (0.0 <= xL || xU <= 0.0) && return x^2
	((xL < 0.0) && (0.0 <= xU) && (0.0 <= x)) && return (x^3)/xU
	return (x^3)/xL
end
@inline function dcv_sqr(x::Float64, xL::Float64, xU::Float64)
    (0.0 <= xL || xU <= 0.0) && return 2.0*x
	((xL < 0.0) && (0.0 <= xU) && (0.0 <= x)) && (3.0*x^2)/xU
	return (3.0*x^2)/xL
end
@inline function sqr_kernel(x::MC{N,T}, y::Interval{Float64}) where {N,T<:Union{NS,MV}}
	if (x.Intv.hi < 0.0)
       eps_min = x.Intv.hi
       eps_max = x.Intv.lo
    elseif (x.Intv.lo > 0.0)
       eps_min = x.Intv.lo
       eps_max = x.Intv.hi
    else
       eps_min = 0.0
       eps_max = (abs(x.Intv.lo) >= abs(x.Intv.hi)) ? x.Intv.lo : x.Intv.hi
    end
	midcc, cc_id = mid3(x.cc, x.cv, eps_max)
	midcv, cv_id = mid3(x.cc, x.cv, eps_min)
	cc = cc_sqr(midcc, x.Intv.lo, x.Intv.hi)
	dcc = dcc_sqr(midcc, x.Intv.lo, x.Intv.hi)
	cv = cv_sqr_NS(midcv, x.Intv.lo, x.Intv.hi)
	dcv = dcv_sqr_NS(midcv, x.Intv.lo, x.Intv.hi)
	cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
	cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
	cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
	return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function sqr_kernel(x::MC{N,Diff}, y::Interval{Float64}) where N
	if (x.Intv.hi < 0.0)
       eps_min = x.Intv.hi
       eps_max = x.Intv.lo
    elseif (x.Intv.lo > 0.0)
       eps_min = x.Intv.lo
       eps_max = x.Intv.hi
    else
       eps_min = 0.0
       eps_max = (abs(x.Intv.lo) >= abs(x.Intv.hi)) ? x.Intv.lo : x.Intv.hi
    end
	midcc, cc_id = mid3(x.cc, x.cv, eps_max)
	midcv, cv_id = mid3(x.cc, x.cv, eps_min)
	cc = cc_sqr(midcc, x.Intv.lo, x.Intv.hi)
	dcc = dcc_sqr(midcc, x.Intv.lo, x.Intv.hi)
    cv = cv_sqr(midcv, x.Intv.lo, x.Intv.hi)
    dcv = dcv_sqr(midcv, x.Intv.lo, x.Intv.hi)
    gdcc1 = dcc_sqr(x.cv, x.Intv.lo, x.Intv.hi)
    gdcv1 = dcv_sqr(x.cv, x.Intv.lo, x.Intv.hi)
    gdcc2 = dcc_sqr(x.cc, x.Intv.lo, x.Intv.hi)
    gdcv2 = dcv_sqr(x.cc, x.Intv.lo, x.Intv.hi)
    cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
    cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
    return MC{N,Diff}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline sqr(x::MC) = sqr_kernel(x, pow(x.Intv,2))

# convex/concave relaxation (Khan 3.1-3.2) of integer powers of 1/x for positive reals
@inline pow_deriv(x::Float64, n::Z) where {Z <: Integer} = n*x^(n-1)
@inline function cv_npp_or_pow4(x::Float64, xL::Float64, xU::Float64, n::Z) where {Z <: Integer}
	temp = x^(n-1)
	x*temp, n*temp
end
@inline function cc_npp_or_pow4(x::Float64, xL::Float64, xU::Float64, n::Z) where {Z <: Integer}
	dline_seg(^, pow_deriv, x, xL, xU, n)
end
# convex/concave relaxation of integer powers of 1/x for negative reals
@inline function cv_negpowneg(x::Float64, xL::Float64, xU::Float64, n::Z) where {Z <: Integer}
    isodd(n) && (return dline_seg(^, pow_deriv, x, xL, xU, n))
    return x^n, n*x^(n-1)
end
@inline function cc_negpowneg(x::Float64, xL::Float64, xU::Float64, n::Z) where {Z <: Integer}
    isodd(n) && (return x^n,n*x^(n-1))
    return dline_seg(^, pow_deriv, x, xL, xU, n)
end
# convex/concave relaxation of odd powers
@inline function cv_powodd(x::Float64, xL::Float64, xU::Float64, n::Z) where {Z <: Integer}
    (xU <= 0.0) && (return dline_seg(^, pow_deriv, x, xL, xU, n))
    (0.0 <= xL) && (return x^n, n*x^(n - 1))
    val = (xL^n)*(xU - x)/(xU - xL) + (max(0.0, x))^n
    dval = -(xL^n)/(xU - xL) + n*(max(0.0, x))^(n-1)
    return val, dval
end
@inline function cc_powodd(x::Float64, xL::Float64, xU::Float64, n::Z) where {Z <: Integer}
    (xU <= 0.0) && (return x^n, n*x^(n - 1))
    (0.0 <= xL) && (return dline_seg(^, pow_deriv, x, xL, xU, n))
    val = (xU^n)*(x - xL)/(xU - xL) + (min(0.0, x))^n
    dval = (xU^n)/(xU - xL) + n*(min(0.0, x))^(n-1)
    return val, dval
end

@inline function npp_or_pow4(x::MC{N,T}, c::Z, y::Interval{Float64}) where {N, Z<:Integer, T<:Union{NS,MV}}
    if (x.Intv.hi < 0.0)
        eps_min = x.Intv.hi
        eps_max = x.Intv.lo
    elseif (x.Intv.lo > 0.0)
        eps_min = x.Intv.lo
        eps_max = x.Intv.hi
    else
        eps_min = 0.0
        eps_max = (abs(x.Intv.lo) >= abs(x.Intv.hi)) ? x.Intv.lo : x.Intv.hi
  	end
  	midcc, cc_id = mid3(x.cc, x.cv, eps_max)
  	midcv, cv_id = mid3(x.cc, x.cv, eps_min)
  	cc, dcc = cc_npp_or_pow4(midcc, x.Intv.lo, x.Intv.hi, c)
  	cv, dcv = cv_npp_or_pow4(midcv, x.Intv.lo, x.Intv.hi, c)
  	cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
  	cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
  	cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
  	return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function npp_or_pow4(x::MC{N,Diff}, c::Z, y::Interval{Float64}) where {N, Z<:Integer}
  	if (x.Intv.hi < 0.0)
    	eps_min = x.Intv.hi
    	eps_max = x.Intv.lo
  	elseif (x.Intv.lo > 0.0)
    	eps_min = x.Intv.lo
    	eps_max = x.Intv.hi
  	else
    	eps_min = 0.0
    	eps_max = (abs(x.Intv.lo) >= abs(x.Intv.hi)) ? x.Intv.lo : x.Intv.hi
  	end
  	midcc, cc_id = mid3(x.cc, x.cv, eps_max)
  	midcv, cv_id = mid3(x.cc, x.cv, eps_min)
  	cc, dcc = cc_npp_or_pow4(midcc, x.Intv.lo, x.Intv.hi, c)
  	cv, dcv = cv_npp_or_pow4(midcv, x.Intv.lo, x.Intv.hi, c)
  	gcc1, gdcc1 = cc_npp_or_pow4(x.cv, x.Intv.lo, x.Intv.hi, c)
  	gcv1, gdcv1 = cv_npp_or_pow4(x.cv, x.Intv.lo, x.Intv.hi, c)
  	gcc2, gdcc2 = cc_npp_or_pow4(x.cc, x.Intv.lo, x.Intv.hi, c)
  	gcv2, gdcv2 = cv_npp_or_pow4(x.cc, x.Intv.lo, x.Intv.hi, c)
  	cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
  	cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
  	return MC{N,Diff}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end

@inline function pos_odd(x::MC{N,T}, c::Z, y::Interval{Float64}) where {N, Z<:Integer, T<:Union{NS,MV}}
    midcc, cc_id = mid3(x.cc, x.cv, x.Intv.hi)
    midcv, cv_id = mid3(x.cc, x.cv, x.Intv.lo)
    cc, dcc = cc_powodd(midcc, x.Intv.lo, x.Intv.hi, c)
    cv, dcv = cv_powodd(midcv, x.Intv.lo, x.Intv.hi, c)
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
	cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
    return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function pos_odd(x::MC{N,Diff}, c::Z, y::Interval{Float64}) where {N, Z<:Integer}
    midcc, cc_id = mid3(x.cv, x.cc, x.Intv.hi)
    midcv, cv_id = mid3(x.cv, x.cc, x.Intv.lo)
    cc, dcc = cc_powodd(midcc, x.Intv.lo, x.Intv.hi, c)
    cv, dcv = cv_powodd(midcv, x.Intv.lo, x.Intv.hi, c)
    gcc1, gdcc1 = cc_powodd(x.cv, x.Intv.lo, x.Intv.hi, c)
    gcv1, gdcv1 = cv_powodd(x.cv, x.Intv.lo, x.Intv.hi, c)
    gcc2, gdcc2 = cc_powodd(x.cc, x.Intv.lo, x.Intv.hi, c)
    gcv2, gdcv2 = cv_powodd(x.cc, x.Intv.lo, x.Intv.hi, c)
    cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
    cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
    return MC{N,Diff}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end

# neg_powneg_odd computes the McComrick relaxation of x^c where x < 0.0 and c is odd
@inline function cv_neg_powneg_odd(x::Float64, xL::Float64, xU::Float64, c::Z) where Z<:Integer
	if (xU == xL)
		dcv = 0.0
        cv = x^c
    else
		xUc = xU^c
        dcv = (xUc- xL^c)/(xU - xL)
        cv = xUc + dcv*(x - xL)
    end
	cv, dcv
end
@inline function cc_neg_powneg_odd(x::Float64, xL::Float64, xU::Float64, c::Z) where Z<:Integer
	x^c, c*x^(c-1)
end
@inline function neg_powneg_odd(x::MC{N,T}, c::Z, y::Interval{Float64}) where {N, Z<:Integer, T<:Union{NS,MV}}
  	xL = x.Intv.lo
  	xU = x.Intv.hi
  	midcc, cc_id = mid3(x.cc, x.cv, xU)
  	midcv, cv_id = mid3(x.cc, x.cv, xL)
	cc, dcc = cc_neg_powneg_odd(midcc, xL, xU, c)
	cv, dcv = cv_neg_powneg_odd(midcv, xL, xU, c)
  	cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
  	cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
	cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
	return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function neg_powneg_odd(x::MC{N,Diff}, c::Z, y::Interval{Float64}) where {N, Z<:Integer}
  	xL = x.Intv.lo
  	xU = x.Intv.hi
  	midcc = mid3v(x.cv, x.cc, xU)
  	midcv = mid3v(x.cv, x.cc, xL)
  	cc, dcc = cc_negpowneg(midcc, xL, xU, c)
  	cv, dcv = cv_negpowneg(midcv, xL, xU, c)
  	gcc1, gdcc1 = cc_negpowneg(x.cv, xL, xU, c)
  	gcv1, gdcv1 = cv_negpowneg(x.cv, xL, xU, c)
  	gcc2, gdcc2 = cc_negpowneg(x.cc, xL, xU, c)
  	gcv2, gdcv2 = cv_negpowneg(x.cc, xL, xU, c)
  	cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
  	cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
	return MC{N,Diff}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end

# neg_powneg_odd computes the McComrick relaxation of x^c where x < 0.0 and c is even
@inline function neg_powneg_even(x::MC{N,T}, c::Z, y::Interval{Float64}) where {N, Z<:Integer, T<:Union{NS,MV}}
	xL = x.Intv.lo
	xU = x.Intv.hi
  	midcc, cc_id = mid3(x.cc, x.cv, xU)
  	midcv, cv_id = mid3(x.cc, x.cv, xL)
  	cc, dcc = cc_negpowneg(midcc, xL, xU, c)
  	cv, dcv = cv_negpowneg(midcv, xL, xU, c)
  	cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
  	cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
	cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
  	return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function neg_powneg_even(x::MC{N,Diff}, c::Z, y::Interval{Float64}) where {N, Z<:Integer}
	xL = x.Intv.lo
	xU = x.Intv.hi
  	midcc = mid3v(x.cv, x.cc, xU)
  	midcv = mid3v(x.cv, x.cc, xL)
  	cc, dcc = cc_negpowneg(midcc, xL, xU, c)
  	cv, dcv = cv_negpowneg(midcv, xL, xU, c)
  	gcc1, gdcc1 = cc_negpowneg(x.cv, xL, xU, c)
  	gcv1, gdcv1 = cv_negpowneg(x.cv, xL, xU, c)
  	gcc2, gdcc2 = cc_negpowneg(x.cc, xL, xU, c)
  	gcv2, gdcv2 = cv_negpowneg(x.cc, xL, xU, c)
  	cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
  	cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
  	return MC{N,Diff}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end

@inline function pow_kernel(x::MC{N,T}, c::Z, y::Interval{Float64}) where {Z<:Integer, N, T<:RelaxTag}
	isnan(x) && (return nan(MC{N,T}))
	c == 0 && (return one(MC{N,T}))
	c == 1 && (return x)
	if c > 0
        c == 2 && (return sqr_kernel(x, y))
		isodd(c) && (return pos_odd(x, c, y))
		return npp_or_pow4(x, c, y)
    else
        if lo(x) < 0.0
        	isodd(c) && (return neg_powneg_odd(x, c, y))
        	return neg_powneg_even(x, c, y)
		end
    end
	return npp_or_pow4(x, c, y)
end
@inline function pow(x::MC{N,T}, c::Z) where {Z<:Integer, N, T<:RelaxTag}
	if (x.Intv.lo <= 0.0 <= x.Intv.hi) && (c < 0)
		return nan(MC{N,T})
	end
	(c < 0) && pow_kernel(x, c, inv(pow(x.Intv,-c)))
	return pow_kernel(x, c, pow(x.Intv,c))
end
@inline (^)(x::MC, c::Z) where {Z <: Integer} = pow(x,c)

# Power of MC to float
@inline cv_flt_pow_1(x::Float64, xL::Float64, xU::Float64, n::Float64) = dline_seg(^, pow_deriv, x, xL, xU, n)
@inline cc_flt_pow_1(x::Float64, xL::Float64, xU::Float64, n::Float64) = x^n, n*x^(n-1)
@inline function flt_pow_1(x::MC{N,T}, c::Float64, y::Interval{Float64}) where {N, T<:Union{NS,MV}}
	midcc, cc_id = mid3(x.cc, x.cv, x.Intv.hi)
	midcv, cv_id = mid3(x.cc, x.cv, x.Intv.lo)
	cc, dcc = cc_flt_pow_1(midcc, x.Intv.lo, x.Intv.hi, c)
	cv, dcv = cv_flt_pow_1(midcv, x.Intv.lo, x.Intv.hi, c)
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
    cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
    return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function flt_pow_1(x::MC{N,Diff}, c::Float64, y::Interval{Float64}) where N
	midcc, cc_id = mid3(x.cc, x.cv, x.Intv.hi)
	midcv, cv_id = mid3(x.cc, x.cv, x.Intv.lo)
	cc, dcc = cc_flt_pow_1(midcc, x.Intv.lo, x.Intv.hi, c)
	cv, dcv = cv_flt_pow_1(midcv, x.Intv.lo, x.Intv.hi, c)
    gcc1, gdcc1 = cc_flt_pow_1(x.cv ,x.Intv.lo, x.Intv.hi, c)
    gcv1, gdcv1 = cv_flt_pow_1(x.cv, x.Intv.lo, x.Intv.hi, c)
    gcc2, gdcc2 = cc_flt_pow_1(x.cc, x.Intv.lo, x.Intv.hi, c)
    gcv2, gdcv2 = cv_flt_pow_1(x.cc, x.Intv.lo, x.Intv.hi, c)
    cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
    cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
    return MC{N,Diff}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end

@inline function (^)(x::MC{N,T}, c::Float64, y::Interval{Float64}) where {N, T<:Union{NS,MV}}
	if (x.Intv.lo <= 0.0 <= x.Intv.hi) && (c < 0)
		return nan(MC{N,T})
	end
    isinteger(c) && (return pow_kernel(x, Int(c), y))
    ((x.Intv.lo >= 0) && (0.0 < c < 1.0)) && (return flt_pow_1(x, c, y))
	z = exp(c*log(x))
    return MC{N,T}(z.cv, z.cc, y, z.cv_grad, z.cc_grad, x.cnst)
end
@inline function (^)(x::MC{N,Diff}, c::Float64, y::Interval{Float64}) where N
	if (x.Intv.lo <= 0.0 <= x.Intv.hi) && (c < 0)
		return nan(MC{N, Diff})
	end
    isinteger(c) && (return pow_kernel(x, Int(c), y))
    ((x.Intv.lo >= 0) && (0.0 < c < 1.0)) && (return flt_pow_1(x, c, y))
	z = exp(c*log(x))
    return MC{N,Diff}(z.cv, z.cc, y, z.cv_grad, z.cc_grad, x.cnst)
end

@inline (^)(x::MC, c::Float32, y::Interval{Float64}) = (^)(x, Float64(c), y)
@inline (^)(x::MC, c::Float16, y::Interval{Float64}) = (^)(x, Float64(c), y)
@inline (^)(x::MC, c::Float64) = (^)(x, c, x.Intv^c)
@inline (^)(x::MC, c::Float32) = x^Float64(c)
@inline (^)(x::MC, c::Float16) = x^Float64(c)
@inline (^)(x::MC, c::MC) = exp(c*log(x))
@inline pow(x::MC, c::F) where {F <: AbstractFloat} = x^c

# Define powers to MC of floating point number
@inline function pow(b::Float64, x::MC{N,T}) where {N,T<:RelaxTag}
	(b <= 0.0) && (return nan(MC{N,T}))
	exp(x*log(b))
end
@inline ^(b::Float64, x::MC) = pow(b, x) # DONE (no kernel)

########### Defines inverse
@inline function cc_inv1(x::Float64, xL::Float64, xU::Float64)
	t = (xL*xU)
	cc = (xU + xL - x)/t
	dcc = -1.0/t
	return cc, dcc
end
@inline function cv_inv1(x::Float64, xL::Float64, xU::Float64)
	cv = 1.0/x
	dcv = -1.0/(x*x)
	return cv, dcv
end
@inline function inv1(x::MC{N,T}, y::Interval{Float64}) where {N,T<:Union{NS,MV}}
	eps_min = x.Intv.hi
	eps_max = x.Intv.lo
  	midcc, cc_id = mid3(x.cc, x.cv, eps_max)
  	midcv, cv_id = mid3(x.cc, x.cv, eps_min)
  	cc, dcc = cc_inv1(midcc, x.Intv.lo, x.Intv.hi)
  	cv, dcv = cv_inv1(midcv, x.Intv.lo, x.Intv.hi)
  	cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
  	cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
  	cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
  	return MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function inv1(x::MC{N,Diff}, y::Interval{Float64}) where N
	xL = x.Intv.lo
	xU = x.Intv.hi
	midcc = mid3v(x.cv, x.cc, xL)
	midcv = mid3v(x.cv, x.cc, xU)
	cc, dcc = cc_inv1(midcc, xL, xU )
	cv, dcv = cv_inv1(midcv, xL, xU)
	gcc1, gdcc1 = cc_inv1(x.cv, xL, xU)
	gcv1, gdcv1 = cv_inv1(x.cv, xL, xU)
	gcc2, gdcc2 = cc_inv1(x.cc, xL, xU)
	gcv2, gdcv2 = cv_inv1(x.cc, xL, xU)
	cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
	cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
  	return MC{N,Diff}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function inv_kernel(x::MC{N,T}, y::Interval{Float64}) where {N,T<:RelaxTag}
	if x.Intv.lo <= 0.0 <= x.Intv.hi
		return nan(MC{N,T})
	end
	if x.Intv.hi < 0.0
		x = pos_odd(x, -1, y)
  	else x.Intv.lo > 0.0
		x = inv1(x, y)
	end
	return x
end
@inline inv(x::MC{N,T}) where {N,T<:RelaxTag} = inv_kernel(x, inv(x.Intv))
