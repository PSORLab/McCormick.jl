# Copyright (c) 2018 Matthew Wilhelm, Robert Gottlieb, Dimitri Alston, 
# Matthew Stuber, and the University of Connecticut (UConn)
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A forward McCormick operator library
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# src/forward_operators/multiplication.jl
# Defines multiplication operator.
#############################################################################

# Nonsmooth multiplication kernel definition
@inline function mul1_u1pos_u2pos(t::NS, x1::MCNoGrad, x2::MCNoGrad, z::Interval{Float64}, cnst::Bool)
    xLc = z.bareinterval.lo
    xUc = z.bareinterval.hi
  	cv1 = x2.Intv.bareinterval.hi*x1.cv + x1.Intv.bareinterval.hi*x2.cv - x1.Intv.bareinterval.hi*x2.Intv.bareinterval.hi
  	cv2 = x2.Intv.bareinterval.lo*x1.cv + x1.Intv.bareinterval.lo*x2.cv - x1.Intv.bareinterval.lo*x2.Intv.bareinterval.lo
  	if cv1 > cv2
    	cv = cv1
    	cv_grad = x2.Intv.bareinterval.hi*x1.cv_grad
  	else
    	cv = cv2
    	cv_grad = x2.Intv.bareinterval.lo*x1.cv_grad
  	end

  	cc1 = x2.Intv.bareinterval.lo*x1.cc + x1.Intv.bareinterval.hi*x2.cc - x1.Intv.bareinterval.hi*x2.Intv.bareinterval.lo
  	cc2 = x2.Intv.bareinterval.hi*x1.cc + x1.Intv.bareinterval.lo*x2.cc - x1.Intv.bareinterval.lo*x2.Intv.bareinterval.hi
  	if cc1 < cc2
    	cc = cc1
    	cc_grad = x2.Intv.bareinterval.lo*x1.cc_grad
  	else
    	cc = cc2
    	cc_grad = x2.Intv.bareinterval.hi*x1.cc_grad
  	end
  	return MCNoGrad(cv, cc, z, cnst)
end
@inline function mul1_u1pos_u2mix(t::NS, x1::MCNoGrad, x2::MCNoGrad, z::Interval{Float64}, cnst::Bool)
    xLc = z.bareinterval.lo
    xUc = z.bareinterval.hi
    cv1 = x2.Intv.bareinterval.hi*x1.cv + x1.Intv.bareinterval.hi*x2.cv - x1.Intv.bareinterval.hi*x2.Intv.bareinterval.hi
    cv2 = x2.Intv.bareinterval.lo*x1.cc + x1.Intv.bareinterval.lo*x2.cv - x1.Intv.bareinterval.lo*x2.Intv.bareinterval.lo
    if cv1 > cv2
        cv = cv1
        cv_grad = x2.Intv.bareinterval.hi*x1.cv_grad
    else
        cv = cv2
        cv_grad = x2.Intv.bareinterval.lo*x1.cc_grad
    end

    cc1 = x2.Intv.bareinterval.lo*x1.cv + x1.Intv.bareinterval.hi*x2.cc - x1.Intv.bareinterval.hi*x2.Intv.bareinterval.lo
    cc2 = x2.Intv.bareinterval.hi*x1.cc + x1.Intv.bareinterval.lo*x2.cc - x1.Intv.bareinterval.lo*x2.Intv.bareinterval.hi
    if cc1 < cc2
        cc = cc1
        cc_grad = x2.Intv.bareinterval.lo*x1.cv_grad
    else
        cc = cc2
        cc_grad = x2.Intv.bareinterval.hi*x1.cc_grad
    end
    return MCNoGrad(cv, cc, z, cnst)
end
@inline function mul1_u1mix_u2mix(t::NS, x1::MCNoGrad, x2::MCNoGrad, z::Interval{Float64}, cnst::Bool)
    xLc = z.bareinterval.lo
    xUc = z.bareinterval.hi
    cv1 = x2.Intv.bareinterval.hi*x1.cv + x1.Intv.bareinterval.hi*x2.cv - x1.Intv.bareinterval.hi*x2.Intv.bareinterval.hi
    cv2 = x2.Intv.bareinterval.lo*x1.cc + x1.Intv.bareinterval.lo*x2.cc - x1.Intv.bareinterval.lo*x2.Intv.bareinterval.lo
    cv = cv1
    if cv1 > cv2
        cv = cv1
        cv_grad = x2.Intv.bareinterval.hi*x1.cv_grad
    else
        cv = cv2
        cv_grad = x2.Intv.bareinterval.lo*x1.cc_grad
    end
    cc1 = x2.Intv.bareinterval.lo*x1.cv + x1.Intv.bareinterval.hi*x2.cc - x1.Intv.bareinterval.hi*x2.Intv.bareinterval.lo
    cc2 = x2.Intv.bareinterval.hi*x1.cc + x1.Intv.bareinterval.lo*x2.cv - x1.Intv.bareinterval.lo*x2.Intv.bareinterval.hi
    cc = cc1
    if cc1 < cc2
        cc = cc1
        cc_grad = x2.Intv.bareinterval.lo*x1.cv_grad
    else
        cc = cc2
        cc_grad = x2.Intv.bareinterval.hi*x1.cc_grad
    end
    return MCNoGrad(cv, cc, z, cnst)
end
@inline function mul2_u1pos_u2pos(t::NS, x1::MCNoGrad, x2::MCNoGrad, z::Interval{Float64}) 
	xLc = z.bareinterval.lo
	xUc = z.bareinterval.hi
	cnst = x2.cnst ? x1.cnst : (x1.cnst ? x2.cnst : x1.cnst || x2.cnst)
	cv1 = x2.Intv.bareinterval.hi*x1.cv + x1.Intv.bareinterval.hi*x2.cv - x1.Intv.bareinterval.hi*x2.Intv.bareinterval.hi
	cv2 = x2.Intv.bareinterval.lo*x1.cv + x1.Intv.bareinterval.lo*x2.cv - x1.Intv.bareinterval.lo*x2.Intv.bareinterval.lo
	if cv1 > cv2
		cv = cv1
		cv_grad = x2.Intv.bareinterval.hi*x1.cv_grad + x1.Intv.bareinterval.hi*x2.cv_grad
	else
		cv = cv2
		cv_grad = x2.Intv.bareinterval.lo*x1.cv_grad + x1.Intv.bareinterval.lo*x2.cv_grad
	end
	cc1 = x2.Intv.bareinterval.lo*x1.cc + x1.Intv.bareinterval.hi*x2.cc - x1.Intv.bareinterval.hi*x2.Intv.bareinterval.lo
	cc2 = x2.Intv.bareinterval.hi*x1.cc + x1.Intv.bareinterval.lo*x2.cc - x1.Intv.bareinterval.lo*x2.Intv.bareinterval.hi
	if cc1 < cc2
		cc = cc1
		cc_grad = x2.Intv.bareinterval.lo*x1.cc_grad + x1.Intv.bareinterval.hi*x2.cc_grad
	else
		cc = cc2
		cc_grad = x2.Intv.bareinterval.hi*x1.cc_grad + x1.Intv.bareinterval.lo*x2.cc_grad
	end
	return MCNoGrad(cv, cc, z, cnst)
end
@inline function mul2_u1pos_u2mix(t::NS, x1::MCNoGrad, x2::MCNoGrad, z::Interval{Float64}, cnst::Bool)
    xLc = z.bareinterval.lo
    xUc = z.bareinterval.hi
    cv1 = x2.Intv.bareinterval.hi*x1.cv + x1.Intv.bareinterval.hi*x2.cv - x1.Intv.bareinterval.hi*x2.Intv.bareinterval.hi
    cv2 = x2.Intv.bareinterval.lo*x1.cc + x1.Intv.bareinterval.lo*x2.cv - x1.Intv.bareinterval.lo*x2.Intv.bareinterval.lo
    if cv1 > cv2
        cv = cv1
		cvt2 = x1.Intv.bareinterval.hi
		cv2_is_cvg = 1
    else
        cv = cv2
		cvt2 = x1.Intv.bareinterval.lo
		cv2_is_cvg = 0
    end
	cvt1 = 0.0
	cv1_is_cvg = 2

    cc1 = x2.Intv.bareinterval.lo*x1.cv + x1.Intv.bareinterval.hi*x2.cc - x1.Intv.bareinterval.hi*x2.Intv.bareinterval.lo
    cc2 = x2.Intv.bareinterval.hi*x1.cc + x1.Intv.bareinterval.lo*x2.cc - x1.Intv.bareinterval.lo*x2.Intv.bareinterval.hi
    if cc1 < cc2
        cc = cc1
		cct2 = x1.Intv.bareinterval.hi
    else
        cc = cc2
		cct2 = x1.Intv.bareinterval.lo
    end
	cct1 = 0.0
	cc1_is_ccg = 2
	cc2_is_ccg = 1

	return MCNoGrad(cv, cc, z, cnst), cvt1, cvt2, cct1, cct2, cv1_is_cvg, cv2_is_cvg, cc1_is_ccg, cc2_is_ccg
end
@inline function mul2_u1mix_u2mix(t::NS, x1::MCNoGrad, x2::MCNoGrad, z::Interval{Float64})
	xLc = z.bareinterval.lo
	xUc = z.bareinterval.hi
  	cnst = x2.cnst ? x1.cnst : (x1.cnst ? x2.cnst : x1.cnst || x2.cnst)
	cv1 = x2.Intv.bareinterval.hi*x1.cv + x1.Intv.bareinterval.hi*x2.cv - x1.Intv.bareinterval.hi*x2.Intv.bareinterval.hi
	cv2 = x2.Intv.bareinterval.lo*x1.cc + x1.Intv.bareinterval.lo*x2.cc - x1.Intv.bareinterval.lo*x2.Intv.bareinterval.lo
	if cv1 > cv2
		cv = cv1
		cvt1 = x2.Intv.bareinterval.hi
		cvt2 = x1.Intv.bareinterval.hi
		cv1_is_cvg = 1
		cv2_is_cvg = 1
	else
		cv = cv2
		cvt1 = x2.Intv.bareinterval.lo
		cvt2 = x1.Intv.bareinterval.lo
		cv1_is_cvg = 0
		cv2_is_cvg = 0
	end

	cc1 = x2.Intv.bareinterval.lo*x1.cv + x1.Intv.bareinterval.hi*x2.cc - x1.Intv.bareinterval.hi*x2.Intv.bareinterval.lo
	cc2 = x2.Intv.bareinterval.hi*x1.cc + x1.Intv.bareinterval.lo*x2.cv - x1.Intv.bareinterval.lo*x2.Intv.bareinterval.hi
	if cc1 < cc2
		cc = cc1
		cct1 = x2.Intv.bareinterval.lo
		cct2 = x1.Intv.bareinterval.hi
		cc1_is_ccg = 0
		cc2_is_ccg = 1
	else
		cc = cc2
		cct1 = x2.Intv.bareinterval.hi
		cct2 = x1.Intv.bareinterval.lo
		cc1_is_ccg = 1
		cc2_is_ccg = 0
	end

	return MCNoGrad(cv, cc, z, cnst), cvt1, cvt2, cct1, cct2, cv1_is_cvg, cv2_is_cvg, cc1_is_ccg, cc2_is_ccg
end
@inline function mul3_u1pos_u2mix(t::NS, x1::MCNoGrad, x2::MCNoGrad, z::Interval{Float64})
	xLc = z.bareinterval.lo
	xUc = z.bareinterval.hi
    cnst = x2.cnst ? x1.cnst : (x1.cnst ? x2.cnst : x1.cnst || x2.cnst)
	cv1 = x2.Intv.bareinterval.hi*x1.cv + x1.Intv.bareinterval.hi*x2.cv - x1.Intv.bareinterval.hi*x2.Intv.bareinterval.hi
	cv2 = x2.Intv.bareinterval.lo*x1.cc + x1.Intv.bareinterval.lo*x2.cv - x1.Intv.bareinterval.lo*x2.Intv.bareinterval.lo
	if cv1 > cv2
		cv = cv1
		cvt1 = x2.Intv.bareinterval.hi
		cvt2 = x1.Intv.bareinterval.hi
		cv1_is_cvg = 1
	else
		cv = cv2
		cvt1 = x2.Intv.bareinterval.lo
		cvt2 = x1.Intv.bareinterval.lo
		cv1_is_cvg = 0
	end
	cv2_is_cvg = 1

	cc1 = x2.Intv.bareinterval.lo*x1.cv + x1.Intv.bareinterval.hi*x2.cc - x1.Intv.bareinterval.hi*x2.Intv.bareinterval.lo
	cc2 = x2.Intv.bareinterval.hi*x1.cc + x1.Intv.bareinterval.lo*x2.cc - x1.Intv.bareinterval.lo*x2.Intv.bareinterval.hi
	if cc1 < cc2
		cc = cc1
		cct1 = x2.Intv.bareinterval.lo
		cct2 = x1.Intv.bareinterval.hi
		cc1_is_ccg = 0
	else
		cc = cc2
		cct1 = x2.Intv.bareinterval.hi
		cct2 = x1.Intv.bareinterval.lo
		cc1_is_ccg = 1
	end
	cc2_is_ccg = 1
	return MCNoGrad(cv, cc, z, cnst), cvt1, cvt2, cct1, cct2, cv1_is_cvg, cv2_is_cvg, cc1_is_ccg, cc2_is_ccg
end

@inline function multiply_STD_NS(t::NS, x1::MCNoGrad, x2::MCNoGrad, y::Interval{Float64})
	if x2.Intv.bareinterval.lo >= 0.0
    	x2.cnst && (return mul1_u1pos_u2pos(t, x1, x2, y, x1.cnst))
    	x1.cnst && (return mul1_u1pos_u2pos(t, x2, x1, y, x2.cnst))
  		return mul2_u1pos_u2pos(t, x1, x2, y)
	elseif x2.Intv.bareinterval.hi <= 0.0
		return -mult_kernel(t, x1, -x2, -y)
	else
    	x2.cnst && (return mul1_u1pos_u2mix(t, x1, x2, y, x1.cnst))
    	x1.cnst && (return mul2_u1pos_u2mix(t, x1, x2, y, x2.cnst))
	end
	mul3_u1pos_u2mix(t, x1, x2, y)
end

@inline function mult_kernel(t::NS, x1::MCNoGrad, x2::MCNoGrad, y::Interval{Float64})
	isone(x1) && (return x2)
	isone(x2) && (return x1)
	if x1.Intv.bareinterval.lo >= 0.0
		return multiply_STD_NS(t, x1, x2, y)
	elseif x1.Intv.bareinterval.hi <= 0.0
		(x2.Intv.bareinterval.lo >= 0.0) && (return -mult_kernel(t, -x1, x2, -y))
	    (x2.Intv.bareinterval.hi <= 0.0) && (return mult_kernel(t, -x1, -x2, y))
		return -mult_kernel(t, x2, -x1, -y)
	elseif x2.Intv.bareinterval.lo >= 0.0
		return mult_kernel(t, x2, x1, y)
	elseif x2.Intv.bareinterval.hi <= 0.0
		return -mult_kernel(t, -x2, x1, -y)
	else
    	x2.cnst && (return mul1_u1mix_u2mix(t, x1, x2, y, x1.cnst))
    	x1.cnst && (return mul1_u1mix_u2mix(t, x2, x1, y, x2.cnst))
	end
	return mul2_u1mix_u2mix(t, x1, x2, y)
end

@inline function *(t::NS, x1::MCNoGrad, x2::MCNoGrad)
	mult_kernel(t, x1, x2, x1.Intv*x2.Intv)
end