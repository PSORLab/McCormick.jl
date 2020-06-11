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
# src/forward_operators/multiplication.jl
# Defines multiplication operator.
#############################################################################

# Differentiable multiplication kernel definition
@inline function sigu(y::Float64)
	x = y/MC_DIFF_MU1T
	(0.0 <= x) ? x^(1.0/MC_DIFF_MU) : -abs(x)^(1.0/MC_DIFF_MU)
end
@inline abs_pu(x::Float64) = abs(x)^MC_DIFF_MU1T
@inline psi_pow(x::Float64) = MC_DIFF_MU1T*x*abs(x)^MC_DIFF_MU1N

@inline function gCxAcv(alpha::Interval{Float64}, beta::Interval{Float64},
	            lambda::Interval{Float64}, nu::Interval{Float64}, x1::MC, x2::MC)

	# gCxA pre-terms
	alplo = alpha.lo
	alphi = alpha.hi
	betlo = beta.lo
	bethi = beta.hi
	LmdDel = lambda.hi - lambda.lo
	NuDel = nu.hi - nu.lo
	LmdSum = lambda.lo + lambda.hi
	NuSum = nu.lo + nu.hi

	xslo = lambda.lo + LmdDel*(((nu.hi - betlo)/NuDel) - sigu(NuSum/NuDel))
	xshi = lambda.lo + LmdDel*(((nu.hi - bethi)/NuDel) - sigu(NuSum/NuDel))
	yslo = nu.lo + NuDel*(((lambda.hi - alplo)/LmdDel) - sigu(LmdSum/LmdDel))
	yshi = nu.lo + NuDel*(((lambda.hi - alphi)/LmdDel) - sigu(LmdSum/LmdDel))

	term1, _ = mid3(alplo, alphi, xslo)
	term2, _ = mid3(alplo, alphi, xshi)
	term3, _ = mid3(betlo, bethi, yslo)
	term4, _ = mid3(betlo, bethi, yshi)

	# calculates the convex relaxation
	tempGxA1a = LmdDel*NuDel*abs_pu((betlo - nu.lo)/NuDel - (lambda.hi - term1)/LmdDel)
	tempGxA2a = LmdDel*NuDel*abs_pu((bethi - nu.lo)/NuDel - (lambda.hi - term2)/LmdDel)
	tempGxA3a = LmdDel*NuDel*abs_pu((term3 - nu.lo)/NuDel - (lambda.hi - alplo)/LmdDel)
	tempGxA4a = LmdDel*NuDel*abs_pu((term4 - nu.lo)/NuDel - (lambda.hi - alphi)/LmdDel)

	NuDotLmd = lambda.lo*nu.lo + lambda.hi*nu.hi
	tempGxA1 = 0.5*(term1*NuSum + betlo*LmdSum - NuDotLmd + tempGxA1a)
	tempGxA2 = 0.5*(term2*NuSum + bethi*LmdSum - NuDotLmd + tempGxA2a)
	tempGxA3 = 0.5*(alplo*NuSum + term3*LmdSum - NuDotLmd + tempGxA3a)
	tempGxA4 = 0.5*(alphi*NuSum + term4*LmdSum - NuDotLmd + tempGxA4a)

	# gets minima which is cv/cc/Intv depending on arguments
	cv = min(tempGxA1, tempGxA2, tempGxA3, tempGxA4)

	if (cv == tempGxA1)
		psi_eval = psi_pow((betlo - nu.lo)/NuDel - (lambda.hi - term1)/LmdDel)
		psi_mx1 = 0.5*(NuSum + NuDel*psi_eval)
		psi_my1 = 0.5*(LmdSum + LmdDel*psi_eval)
	elseif (cv == tempGxA2)
		psi_eval = psi_pow((bethi - nu.lo)/NuDel - (lambda.hi - term2)/LmdDel)
		psi_mx1 = 0.5*(NuSum + NuDel*psi_eval)
		psi_my1 = 0.5*(LmdSum + LmdDel*psi_eval)
	elseif (cv == tempGxA3)
		psi_eval = psi_pow((term3 - nu.lo)/NuDel - (lambda.hi - alplo)/LmdDel)
		psi_mx1 = 0.5*(NuSum + NuDel*psi_eval)
		psi_my1 = 0.5*(LmdSum + LmdDel*psi_eval)
	else
		psi_eval = psi_pow((term4 - nu.lo)/NuDel - (lambda.hi - alphi)/LmdDel)
		psi_mx1 = 0.5*(NuSum + NuDel*psi_eval)
		psi_my1 = 0.5*(LmdSum + LmdDel*psi_eval)
	end

    cv_grad = max(0.0, psi_mx1)*x1.cv_grad + min(0.0, psi_mx1)*x1.cc_grad +
		      max(0.0, psi_my1)*x2.cv_grad + min(0.0, psi_my1)*x2.cc_grad

    return cv, cv_grad
end

@inline function gCxAcc(alpha::Interval{Float64},beta::Interval{Float64},lambda::Interval{Float64},
	            nu::Interval{Float64},x1::MC,x2::MC)

	# gCxA pre-terms
	alplo = alpha.lo
	alphi = alpha.hi
	betlo = beta.lo
	bethi = beta.hi
	LmdDel = lambda.hi - lambda.lo
	NuDel = nu.hi - nu.lo
	LmdSum = lambda.lo + lambda.hi
	NuSum = nu.lo + nu.hi

	xslo = lambda.lo + LmdDel*(((nu.hi - betlo)/NuDel) - sigu(NuSum/NuDel))
	xshi = lambda.lo + LmdDel*(((nu.hi - bethi)/NuDel) - sigu(NuSum/NuDel))
	yslo = nu.lo + NuDel*(((lambda.hi - alplo)/LmdDel) - sigu(LmdSum/LmdDel))
	yshi = nu.lo + NuDel*(((lambda.hi - alphi)/LmdDel) - sigu(LmdSum/LmdDel))

	# calculates term 1
	term1, _ = mid3(alplo, alphi, xslo)
	term2, _ = mid3(alplo, alphi, xshi)
	term3, _ = mid3(betlo, bethi, yslo)
	term4, _ = mid3(betlo, bethi, yshi)

	# calculates the convex relaxation
	tempGxA1a = LmdDel*NuDel*abs_pu((betlo - nu.lo)/NuDel - (lambda.hi - term1)/LmdDel)
	tempGxA2a = LmdDel*NuDel*abs_pu((bethi - nu.lo)/NuDel - (lambda.hi - term2)/LmdDel)
	tempGxA3a = LmdDel*NuDel*abs_pu((term3 - nu.lo)/NuDel - (lambda.hi - alplo)/LmdDel)
	tempGxA4a = LmdDel*NuDel*abs_pu((term4 - nu.lo)/NuDel - (lambda.hi - alphi)/LmdDel)

	NuDotLmd = lambda.lo*nu.lo+lambda.hi*nu.hi
	tempGxA1 = 0.5*(term1*NuSum + betlo*LmdSum - NuDotLmd + tempGxA1a)
	tempGxA2 = 0.5*(term2*NuSum + bethi*LmdSum - NuDotLmd + tempGxA2a)
	tempGxA3 = 0.5*(alplo*NuSum + term3*LmdSum - NuDotLmd + tempGxA3a)
	tempGxA4 = 0.5*(alphi*NuSum + term4*LmdSum - NuDotLmd + tempGxA4a)

	# gets minima which is cv/cc/Intv depending on arguments
	ncc = min(tempGxA1, tempGxA2, tempGxA3, tempGxA4)

   if (ncc == tempGxA1)
	   psi_eval = psi_pow((betlo - nu.lo)/NuDel - (lambda.hi - term1)/LmdDel)
	   psi_mx1 = 0.5*(NuSum + NuDel*psi_eval)
	   psi_my1 = 0.5*(LmdSum + LmdDel*psi_eval)
	elseif (ncc == tempGxA2)
		psi_eval = psi_pow((bethi - nu.lo)/NuDel - (lambda.hi - term2)/LmdDel)
		psi_mx1 = 0.5*(NuSum + NuDel*psi_eval)
		psi_my1 = 0.5*(LmdSum + LmdDel*psi_eval)
	elseif (ncc == tempGxA3)
		psi_eval = psi_pow((term3 - nu.lo)/NuDel - (lambda.hi - alplo)/LmdDel)
		psi_mx1 = 0.5*(NuSum + NuDel*psi_eval)
		psi_my1 = 0.5*(LmdSum + LmdDel*psi_eval)
	else
		psi_eval = psi_pow((term4 - nu.lo)/NuDel - (lambda.hi - alphi)/LmdDel)
		psi_mx1 = 0.5*(NuSum + NuDel*psi_eval)
		psi_my1 = 0.5*(LmdSum + LmdDel*psi_eval)
	end
	ncc_grad = max(0.0,psi_mx1)*x1.cc_grad + min(0.0,psi_mx1)*x1.cv_grad -
		       max(0.0,psi_my1)*x2.cc_grad - min(0.0,psi_my1)*x2.cv_grad

    return ncc, ncc_grad
end

@inline function gCxAIntv(alpha::Interval{Float64},beta::Interval{Float64},lambda::Interval{Float64},
	              nu::Interval{Float64},x1::MC,x2::MC)

	# gCxA pre-terms
	alplo = alpha.lo
	alphi = alpha.hi
	betlo = beta.lo
	bethi = beta.hi
	LmdDel = lambda.hi-lambda.lo
	NuDel = nu.hi-nu.lo
	LmdSum = lambda.lo+lambda.hi
	NuSum = nu.lo+nu.hi

	xslo = lambda.lo + LmdDel*(((nu.hi - betlo)/NuDel) - sigu(NuSum/NuDel))
	xshi = lambda.lo + LmdDel*(((nu.hi - bethi)/NuDel) - sigu(NuSum/NuDel))
	yslo = nu.lo + NuDel*(((lambda.hi - alplo)/LmdDel) - sigu(LmdSum/LmdDel))
	yshi = nu.lo + NuDel*(((lambda.hi - alphi)/LmdDel) - sigu(LmdSum/LmdDel))

	# calculates terms
	term1, _ = mid3(alplo, alphi, xslo)
	term2, _ = mid3(alplo, alphi, xshi)
	term3, _ = mid3(betlo, bethi, yslo)
	term4, _ = mid3(betlo, bethi, yshi)

	# calculates the convex relaxation
	tempGxA1a = LmdDel*NuDel*abs_pu((betlo - nu.lo)/NuDel - (lambda.hi - term1)/LmdDel)
	tempGxA2a = LmdDel*NuDel*abs_pu((bethi - nu.lo)/NuDel - (lambda.hi - term2)/LmdDel)
	tempGxA3a = LmdDel*NuDel*abs_pu((term3 - nu.lo)/NuDel - (lambda.hi - alplo)/LmdDel)
	tempGxA4a = LmdDel*NuDel*abs_pu((term4 - nu.lo)/NuDel - (lambda.hi - alphi)/LmdDel)

	NuLmd = lambda.lo*nu.lo+lambda.hi*nu.hi
	tempGxA1 = 0.5*(term1*NuSum + betlo*LmdSum - NuLmd + tempGxA1a)
	tempGxA2 = 0.5*(term2*NuSum + bethi*LmdSum - NuLmd + tempGxA2a)
	tempGxA3 = 0.5*(alplo*NuSum + term3*LmdSum - NuLmd + tempGxA3a)
	tempGxA4 = 0.5*(alphi*NuSum + term4*LmdSum - NuLmd + tempGxA4a)

	# gets minima which is cv/cc/Intv depending on arguments
	return min(tempGxA1, tempGxA2, tempGxA3, tempGxA4)
end

@inline function multiply_MV(x1::MC{N,Diff}, x2::MC{N,Diff}, z::Interval{Float64}) where {N}

	x1cv = x1.cv
	x2cv = x2.cv
	x1cc = x1.cc
	x2cc = x2.cc
	x1lo = x1.Intv.lo
	x2lo = x2.Intv.lo
	x1hi = x1.Intv.hi
	x2hi = x2.Intv.hi
	x1h_l = x1hi - x1lo
	x1h_v = x1hi - x1cv
	x2h_l = x2hi - x2lo
	x2v_l = x2cv - x2lo
	cv = 0.0
	cc = 0.0

	alpha0 = Interval{Float64}(x1cv, x1cc)
	beta0 = Interval{Float64}(x2cv, x2cc)

	if (0.0 <= x1lo) && (0.0 <= x2lo)

		btemp1 = max(0.0, x2v_l/x2h_l - x1h_v/x1h_l)
		btemp2 = x2lo + MC_DIFF_MU1T*x2h_l*btemp1^MC_DIFF_MU
		btemp3 = x1lo + MC_DIFF_MU1T*x1h_l*btemp1^MC_DIFF_MU

		cv = x1cv*x2lo + x1lo*x2v_l + x1h_l*x2h_l*btemp1^MC_DIFF_MU1T
		cv_grad = btemp2*x1.cv_grad + btemp3*x2.cv_grad


	elseif (x1hi <= 0.0) && (x2hi <= 0.0)

		btemp1 = max(0.0,(x2hi - x2cc)/x2h_l - (x1cc-x1lo)/x1h_l)
		btemp2 = -x2hi + MC_DIFF_MU1T*(x2h_l)*max(0.0, btemp1)^MC_DIFF_MU
		btemp3 = -x1hi + MC_DIFF_MU1T*(x1h_l)*max(0.0, btemp1)^MC_DIFF_MU

		cv = x1cc*x2hi + x1hi*(x2cc - x2hi) + (x1h_l)*(x2h_l)*btemp1^MC_DIFF_MU1T
		cv_grad = (-btemp2)*x1.cc_grad - btemp3*x2.cc_grad

	else
		cv,cv_grad = gCxAcv(alpha0, beta0, x1.Intv, x2.Intv, x1, x2)
	end

	if (x1hi <= 0.0) && (0.0 <= x2lo)

		btemp1 = max(0.0, x2v_l/x2h_l - (x1cc-x1lo)/x1h_l)
		btemp2 = x2lo + MC_DIFF_MU1T*x2h_l*btemp1^MC_DIFF_MU
		btemp3 = -x1hi + MC_DIFF_MU1T*x1h_l*btemp1^MC_DIFF_MU
		cc = x1cc*x2lo - x1hi*(x2lo - x2cv) - x1h_l*x2h_l*btemp1^MC_DIFF_MU1T
		cc_grad = btemp2*x1.cc_grad - btemp3*x2.cv_grad

	elseif (0.0 <= x1.Intv.lo) && (x2.Intv.hi <= 0.0)

		btemp1 = max(0.0, (x2hi-x2cc)/x2h_l - x1h_v/x1h_l)
		btemp2 = -x2hi + MC_DIFF_MU1T*(x2h_l)*btemp1^MC_DIFF_MU
		btemp3 = x1lo + MC_DIFF_MU1T*x1h_l*btemp1^MC_DIFF_MU
		cc = x1cv*x2hi + x1lo*(x2cc - x2hi) - x1h_l*x2h_l*btemp1^MC_DIFF_MU1T
		cc_grad = -btemp2*x1.cv_grad + btemp3*x2.cc_grad

	else
		cct::Float64, cc_gradt = gCxAcc(-alpha0, beta0, -x1.Intv, x2.Intv, x1, x2)
		cc = -cct
		cc_grad = cc_gradt
	end

	cnst = x2.cnst ? x1.cnst : (x1.cnst ? x2.cnst : x1.cnst || x2.cnst)

	return MC{N,Diff}(cv, cc, z, cv_grad, cc_grad, cnst)
end

@inline function mult_kernel(x1::MC{N,Diff}, x2::MC{N,Diff}, y::Interval{Float64}) where N
	degen1 = ((x1.Intv.hi - x1.Intv.lo) <= MC_DEGEN_TOL)
	degen2 = ((x2.Intv.hi - x2.Intv.lo) <= MC_DEGEN_TOL)
	(degen1 || degen2) && (return nan(MC{N,Diff}))
	return multiply_MV(x1, x2, y)
end

# Nonsmooth multiplication kernel definition
@inline function mul1_u1pos_u2pos(x1::MC{N,T}, x2::MC{N,T}, z::Interval{Float64}, cnst::Bool) where {N,T<:RelaxTag}
    xLc = z.lo
    xUc = z.hi
  	cv1 = x2.Intv.hi*x1.cv + x1.Intv.hi*x2.cv - x1.Intv.hi*x2.Intv.hi
  	cv2 = x2.Intv.lo*x1.cv + x1.Intv.lo*x2.cv - x1.Intv.lo*x2.Intv.lo
  	if (cv1 > cv2)
    	cv = cv1
    	cv_grad = x2.Intv.hi*x1.cv_grad
  	else
    	cv = cv2
    	cv_grad = x2.Intv.lo*x1.cv_grad
  	end

  	cc1 = x2.Intv.lo*x1.cc + x1.Intv.hi*x2.cc - x1.Intv.hi*x2.Intv.lo
  	cc2 = x2.Intv.hi*x1.cc + x1.Intv.lo*x2.cc - x1.Intv.lo*x2.Intv.hi
  	if (cc1 < cc2)
    	cc = cc1
    	cc_grad = x2.Intv.lo*x1.cc_grad
  	else
    	cc = cc2
    	cc_grad = x2.Intv.hi*x1.cc_grad
  	end
  	cv, cc, cv_grad, cc_grad = cut(xLc, xUc, cv, cc, cv_grad, cc_grad)
  	return MC{N,T}(cv, cc, z, cv_grad, cc_grad, cnst)
end
@inline function mul1_u1pos_u2mix(x1::MC{N,T}, x2::MC{N,T}, z::Interval{Float64}, cnst::Bool) where {N,T<:RelaxTag}
    xLc = z.lo
    xUc = z.hi
    cv1 = x2.Intv.hi*x1.cv + x1.Intv.hi*x2.cv - x1.Intv.hi*x2.Intv.hi
    cv2 = x2.Intv.lo*x1.cc + x1.Intv.lo*x2.cv - x1.Intv.lo*x2.Intv.lo
    if (cv1 > cv2)
        cv = cv1
        cv_grad = x2.Intv.hi*x1.cv_grad
    else
        cv = cv2
        cv_grad = x2.Intv.lo*x1.cc_grad
    end

    cc1 = x2.Intv.lo*x1.cv + x1.Intv.hi*x2.cc - x1.Intv.hi*x2.Intv.lo
    cc2 = x2.Intv.hi*x1.cc + x1.Intv.lo*x2.cc - x1.Intv.lo*x2.Intv.hi
    if (cc1 < cc2)
        cc = cc1
        cc_grad = x2.Intv.lo*x1.cv_grad
    else
        cc = cc2
        cc_grad = x2.Intv.hi*x1.cc_grad
    end
    cv,cc,cv_grad,cc_grad = cut(xLc,xUc,cv,cc,cv_grad,cc_grad)
    return MC{N,T}(cv, cc, z, cv_grad, cc_grad, cnst)
end
@inline function mul1_u1mix_u2mix(x1::MC{N,T}, x2::MC{N,T}, z::Interval{Float64}, cnst::Bool) where {N,T<:RelaxTag}
    xLc = z.lo
    xUc = z.hi
    cv1 = x2.Intv.hi*x1.cv + x1.Intv.hi*x2.cv - x1.Intv.hi*x2.Intv.hi
    cv2 = x2.Intv.lo*x1.cc + x1.Intv.lo*x2.cc - x1.Intv.lo*x2.Intv.lo
    cv = cv1
    if (cv1 > cv2)
        cv = cv1
        cv_grad = x2.Intv.hi*x1.cv_grad
    else
        cv = cv2
        cv_grad = x2.Intv.lo*x1.cc_grad
    end
    cc1 = x2.Intv.lo*x1.cv + x1.Intv.hi*x2.cc - x1.Intv.hi*x2.Intv.lo
    cc2 = x2.Intv.hi*x1.cc + x1.Intv.lo*x2.cv - x1.Intv.lo*x2.Intv.hi
    cc = cc1
    if (cc1 < cc2)
        cc = cc1
        cc_grad = x2.Intv.lo*x1.cv_grad
    else
        cc = cc2
        cc_grad = x2.Intv.hi*x1.cc_grad
    end
    cv,cc,cv_grad,cc_grad = cut(xLc,xUc,cv,cc,cv_grad,cc_grad)
    return MC{N,T}(cv, cc, z, cv_grad, cc_grad, cnst)
end
@inline function mul2_u1pos_u2pos(x1::MC{N,T}, x2::MC{N,T}, z::Interval{Float64}) where {N,T<:RelaxTag}
	xLc = z.lo
	xUc = z.hi
	cnst = x2.cnst ? x1.cnst : (x1.cnst ? x2.cnst : x1.cnst || x2.cnst)
	cv1 = x2.Intv.hi*x1.cv + x1.Intv.hi*x2.cv - x1.Intv.hi*x2.Intv.hi
	cv2 = x2.Intv.lo*x1.cv + x1.Intv.lo*x2.cv - x1.Intv.lo*x2.Intv.lo
	if (cv1 > cv2)
		cv = cv1
		cv_grad = x2.Intv.hi*x1.cv_grad + x1.Intv.hi*x2.cv_grad
	else
		cv = cv2
		cv_grad = x2.Intv.lo*x1.cv_grad + x1.Intv.lo*x2.cv_grad
	end
	cc1 = x2.Intv.lo*x1.cc + x1.Intv.hi*x2.cc - x1.Intv.hi*x2.Intv.lo
	cc2 = x2.Intv.hi*x1.cc + x1.Intv.lo*x2.cc - x1.Intv.lo*x2.Intv.hi
	if (cc1 < cc2)
		cc = cc1
		cc_grad = x2.Intv.lo*x1.cc_grad + x1.Intv.hi*x2.cc_grad
	else
		cc = cc2
		cc_grad = x2.Intv.hi*x1.cc_grad + x1.Intv.lo*x2.cc_grad
	end
	cv, cc, cv_grad, cc_grad = cut(xLc, xUc, cv, cc, cv_grad, cc_grad)
	return MC{N,T}(cv, cc, z, cv_grad, cc_grad, cnst)
end
@inline function mul2_u1pos_u2mix(x1::MC{N,T}, x2::MC{N,T}, z::Interval{Float64}, cnst::Bool) where {N,T<:RelaxTag}
    xLc = z.lo
    xUc = z.hi
    cv1 = x2.Intv.hi*x1.cv + x1.Intv.hi*x2.cv - x1.Intv.hi*x2.Intv.hi
    cv2 = x2.Intv.lo*x1.cc + x1.Intv.lo*x2.cv - x1.Intv.lo*x2.Intv.lo
    if (cv1 > cv2)
        cv = cv1
        cv_grad = x1.Intv.hi*x2.cv_grad
    else
        cv = cv2
        cv_grad = x1.Intv.lo*x2.cc_grad
    end
    cc1 = x2.Intv.lo*x1.cv + x1.Intv.hi*x2.cc - x1.Intv.hi*x2.Intv.lo
    cc2 = x2.Intv.hi*x1.cc + x1.Intv.lo*x2.cc - x1.Intv.lo*x2.Intv.hi
    if (cc1 < cc2)
        cc = cc1
        cc_grad = x1.Intv.hi*x2.cc_grad
    else
        cc = cc2
        cc_grad = x1.Intv.lo*x2.cc_grad
    end
    cv, cc, cv_grad, cc_grad = cut(xLc, xUc, cv, cc, cv_grad, cc_grad)
    return MC{N,T}(cv, cc, z, cv_grad, cc_grad, cnst)
end
@inline function mul2_u1mix_u2mix(x1::MC{N,T}, x2::MC{N,T}, z::Interval{Float64}) where {N,T<:RelaxTag}
	xLc = z.lo
	xUc = z.hi
  	cnst = x2.cnst ? x1.cnst : (x1.cnst ? x2.cnst : x1.cnst || x2.cnst)
	cv1 = x2.Intv.hi*x1.cv + x1.Intv.hi*x2.cv - x1.Intv.hi*x2.Intv.hi
	cv2 = x2.Intv.lo*x1.cc + x1.Intv.lo*x2.cc - x1.Intv.lo*x2.Intv.lo
	if (cv1 > cv2)
		cv = cv1
		cv_grad = x2.Intv.hi*x1.cv_grad + x1.Intv.hi*x2.cv_grad
	else
		cv = cv2
		cv_grad = x2.Intv.lo*x1.cc_grad + x1.Intv.lo*x2.cc_grad
	end
	cc1 = x2.Intv.lo*x1.cv + x1.Intv.hi*x2.cc - x1.Intv.hi*x2.Intv.lo
	cc2 = x2.Intv.hi*x1.cc + x1.Intv.lo*x2.cv - x1.Intv.lo*x2.Intv.hi
	if (cc1 < cc2)
		cc = cc1
		cc_grad = x2.Intv.lo*x1.cv_grad + x1.Intv.hi*x2.cc_grad
	else
		cc = cc2
		cc_grad = x2.Intv.hi*x1.cc_grad + x1.Intv.lo*x2.cv_grad
	end
	cv, cc, cv_grad, cc_grad = cut(xLc, xUc, cv, cc, cv_grad, cc_grad)
	return MC{N,T}(cv, cc, z, cv_grad, cc_grad, cnst)
end
@inline function mul3_u1pos_u2mix(x1::MC{N,T}, x2::MC{N,T}, z::Interval{Float64}) where {N,T<:RelaxTag}
	xLc = z.lo
	xUc = z.hi
    cnst = x2.cnst ? x1.cnst : (x1.cnst ? x2.cnst : x1.cnst || x2.cnst)
	cv1 = x2.Intv.hi*x1.cv + x1.Intv.hi*x2.cv - x1.Intv.hi*x2.Intv.hi
	cv2 = x2.Intv.lo*x1.cc + x1.Intv.lo*x2.cv - x1.Intv.lo*x2.Intv.lo
	if cv1 > cv2
		cv = cv1
		cv_grad = x2.Intv.hi*x1.cv_grad + x1.Intv.hi*x2.cv_grad
	else
		cv = cv2
		cv_grad = x2.Intv.lo*x1.cc_grad + x1.Intv.lo*x2.cv_grad
	end
	cc1 = x2.Intv.lo*x1.cv + x1.Intv.hi*x2.cc - x1.Intv.hi*x2.Intv.lo
	cc2 = x2.Intv.hi*x1.cc + x1.Intv.lo*x2.cc - x1.Intv.lo*x2.Intv.hi
	if cc1 < cc2
		cc = cc1
		cc_grad = x2.Intv.lo*x1.cv_grad + x1.Intv.hi*x2.cc_grad
	else
		cc = cc2
		cc_grad = x2.Intv.hi*x1.cc_grad + x1.Intv.lo*x2.cc_grad
	end
	cv, cc, cv_grad, cc_grad = cut(xLc, xUc, cv, cc, cv_grad, cc_grad)
	return MC{N,T}(cv, cc, z, cv_grad, cc_grad, cnst)
end

@inline function multiply_STD_NS(x1::MC, x2::MC, y::Interval{Float64})
	if x2.Intv.lo >= 0.0
    	x2.cnst && (return mul1_u1pos_u2pos(x1, x2, y, x1.cnst))
    	x1.cnst && (return mul1_u1pos_u2pos(x2, x1, y, x2.cnst))
  		return mul2_u1pos_u2pos(x1, x2, y)
	elseif x2.Intv.hi <= 0.0
		return -mult_kernel(x1, -x2, -y)
	else
    	x2.cnst && (return mul1_u1pos_u2mix(x1, x2, y, x1.cnst))
    	x1.cnst && (return mul2_u1pos_u2mix(x1, x2, y, x2.cnst))
	end
	mul3_u1pos_u2mix(x1, x2, y)
end

@inline function mult_kernel(x1::MC{N,NS}, x2::MC{N,NS}, y::Interval{Float64}) where N
	isone(x1) && (return x2)
	isone(x2) && (return x1)
	if x1.Intv.lo >= 0.0
		return multiply_STD_NS(x1, x2, y)
	elseif x1.Intv.hi <= 0.0
		(x2.Intv.lo >= 0.0) && (return -mult_kernel(-x1, x2, -y))
	    (x2.Intv.hi <= 0.0) && (return mult_kernel(-x1, -x2, y))
		return -mult_kernel(x2, -x1, -y)
	elseif x2.Intv.lo >= 0.0
		return mult_kernel(x2, x1, y)
	elseif x2.Intv.hi <= 0.0
		return -mult_kernel(-x2, x1, -y)
	else
    	x2.cnst && (return mul1_u1mix_u2mix(x1, x2, y, x1.cnst))
    	x1.cnst && (return mul1_u1mix_u2mix(x2, x1, y, x2.cnst))
	end
	return mul2_u1mix_u2mix(x1, x2, y)
end

# Nonsmooth multivariate multiplication kernel definition
@inline mul_MV_ns1cv(x1::Float64, x2::Float64, MC1::MC, MC2::MC) = MC2.Intv.hi*x1 + MC1.Intv.hi*x2 - MC2.Intv.hi*MC1.Intv.hi
@inline mul_MV_ns2cv(x1::Float64, x2::Float64, MC1::MC, MC2::MC) = MC2.Intv.lo*x1 + MC1.Intv.lo*x2 - MC2.Intv.lo*MC1.Intv.lo
@inline mul_MV_ns3cv(x1::Float64, x2::Float64, MC1::MC, MC2::MC) = max(mul_MV_ns1cv(x1,x2,MC1,MC2), mul_MV_ns2cv(x1,x2,MC1,MC2))

@inline mul_MV_ns1cc(x1::Float64, x2::Float64, MC1::MC, MC2::MC) = MC2.Intv.lo*x1 + MC1.Intv.hi*x2 - MC2.Intv.lo*MC1.Intv.hi
@inline mul_MV_ns2cc(x1::Float64, x2::Float64, MC1::MC, MC2::MC) = MC2.Intv.hi*x1 + MC1.Intv.lo*x2 - MC2.Intv.hi*MC1.Intv.lo
@inline mul_MV_ns3cc(x1::Float64, x2::Float64, MC1::MC, MC2::MC) = min(mul_MV_ns1cc(x1,x2,MC1,MC2), mul_MV_ns2cc(x1,x2,MC1,MC2))

@inline tol_MC(x::Float64, y::Float64) = abs(x-y) <= (MC_MV_TOL + MC_MV_TOL*0.5*abs(x+y))

@inline function multiply_MV_NS(x1::MC{N,MV}, x2::MC{N,MV}, zIntv::Interval{Float64}, cnst::Bool) where N

	flag = true

	# convex calculation
    k = -diam(x2)/diam(x1)
    z = (x1.Intv.hi*x2.Intv.hi - x1.Intv.lo*x2.Intv.lo)/diam(x1)

 	x1vta = mid3v(x1.cv, x1.cc, (x2.cv - z)/k)
 	x1vtb = mid3v(x1.cv, x1.cc, (x2.cc - z)/k)
 	x2vta = mid3v(x2.cv, x2.cc, k*x1.cv + z)
 	x2vtb = mid3v(x2.cv ,x2.cc, k*x1.cc + z)

    x1vt = (x1.cv, x1.cc, x1vta, x1vtb, x1.cv, x1.cc)
    x2vt = (x2vta, x2vtb, x2.cv, x2.cc, x2.cv, x2.cc)

	vt = mul_MV_ns3cv.(x1vt, x2vt, x1, x2)
    cv, cvi = findmin(vt)

 	if tol_MC(mul_MV_ns1cv(x1vt[cvi], x2vt[cvi], x1, x2), mul_MV_ns2cv(x1vt[cvi], x2vt[cvi], x1, x2))

		alph1 = 0.0
		alph2 = 1.0

		MC1thin = tol_MC(x1.cv, x1.cc)
 		if ~MC1thin && (x1vt[cvi] > x1.cv) && ~tol_MC(x1vt[cvi], x1.cv)
 			alph2 = min(alph2, -x2.Intv.lo/diam(x2))
 		end

 		if ~MC1thin && (x1vt[cvi] < x1.cc) && ~tol_MC(x1vt[cvi], x1.cc)
			alph1 = max(alph1, -x2.Intv.lo/diam(x2))
		end

		MC2thin = tol_MC(x2.cv, x2.cc)
		if ~MC2thin && (x2vt[cvi] > x2.cv) && ~tol_MC(x2vt[cvi], x2.cv)
			alph2 = min(alph2, -x1.Intv.lo/diam(x1))
		end
		if ~MC2thin && (x2vt[cvi] < x2.cc) && ~tol_MC(x2vt[cvi], x2.cc)
			alph1 = max(alph1, -x1.Intv.lo/diam(x1))
		end

		alphthin = tol_MC(alph1, alph2)
		(~alphthin && (alph1 > alph2)) && (return false, x1)
		myalph = (alph1 + alph2)/2.0
	elseif mul_MV_ns1cv(x1vt[cvi], x2vt[cvi], x1,x2) > mul_MV_ns2cv(x1vt[cvi], x2vt[cvi], x1, x2)
		myalph = 1.0
	else
		myalph = 0.0
	end

	sigma1 = x2.Intv.lo + myalph*diam(x2)
	sigma2 = x1.Intv.lo + myalph*diam(x1)
	cv_grad1 = (sigma1 > 0.0) ? x1.cv_grad : x1.cc_grad
	cv_grad2 = (sigma2 > 0.0) ? x2.cv_grad : x2.cc_grad

	x1.cnst && (cv_grad = sigma2*cv_grad2)
	x2.cnst && (cv_grad = sigma1*cv_grad1)
	(~x1.cnst && ~x1.cnst) && (cv_grad = sigma1*cv_grad1 + sigma2*cv_grad2)
	(x1.cnst && x1.cnst) && (cv_grad = zero(SVector{N,Float64}))

	 # concave calculation
	 k = diam(x2)/diam(x1)
	 z = (x1.Intv.hi*x2.Intv.lo - x1.Intv.lo*x2.Intv.hi)/diam(x1)

	 x1vta = mid3v(x1.cv, x1.cc, (x2.cv - z)/k)
	 x1vtb = mid3v(x1.cv, x1.cc, (x2.cc - z)/k)
	 x2vta = mid3v(x2.cv, x2.cc, k*x1.cv + z)
	 x2vtb = mid3v(x2.cv, x2.cc, k*x1.cc + z)

	 x1vt = (x1.cv, x1.cc, x1vta, x1vtb, x1.cv, x1.cc)
	 x2vt = (x2vta, x2vtb, x2.cv, x2.cc, x2.cc, x2.cv)
	 vt = mul_MV_ns3cc.(x1vt, x2vt, x1, x2)
	 cc, cci = findmax(vt)

	 if tol_MC(mul_MV_ns1cc(x1vt[cci], x2vt[cci], x1, x2), mul_MV_ns2cc(x1vt[cci], x2vt[cci], x1, x2))

		alph1 = 0.0
	 	alph2 = 1.0

		 MC1thin = tol_MC(x1.cv, x1.cc)
		 if ~MC1thin && (x1vt[cci] > x1.cv) && ~tol_MC(x1vt[cci], x1.cv)
		 	alph1 = max(alph1, -x2.Intv.lo/diam(x2))
		 end
		 if ~MC1thin && (x1vt[cci] < x1.cc) && ~tol_MC(x1vt[cci], x1.cc)
		 	alph2 = min(alph2, -x2.Intv.lo/diam(x2))
		 end

		 MC2thin = tol_MC(x2.cv, x2.cc)
		 if ~MC2thin && (x2vt[cci] > x2.cv) && ~tol_MC(x2vt[cci], x2.cv)
		 	alph2 = min(alph2, x1.Intv.hi/diam(x1))
		 end
		 if ~MC2thin && (x2vt[cci] < x2.cc) && ~tol_MC(x2vt[cci], x2.cc)
		 	alph1 = max(alph1, x1.Intv.hi/diam(x1))
		 end

        alphthin = tol_MC(alph1, alph2)
		(~alphthin && (alph1 > alph2)) && (return false, x1)
	 	myalph = (alph1 + alph2)/2.0
	elseif mul_MV_ns1cc(x1vt[cci], x2vt[cci], x1, x2) > mul_MV_ns2cc(x1vt[cci], x2vt[cci], x1, x2)
	 	myalph = 1.0
	else
		myalph = 0.0
	end

	sigma1 = x2.Intv.lo + myalph*diam(x2)
	sigma2 = x1.Intv.hi - myalph*diam(x1)
	cc_grad1 = (sigma1 > 0.0) ? x1.cc_grad :  x1.cv_grad
	cc_grad2 = (sigma2 > 0.0) ? x2.cc_grad :  x2.cv_grad

	x1.cnst && (cc_grad = sigma2*cc_grad2)
	x2.cnst && (cc_grad = sigma1*cc_grad1)
	(~x1.cnst && ~x1.cnst) && (cc_grad = sigma1*cc_grad1 + sigma2*cc_grad2)
	(x1.cnst && x1.cnst) && (cc_grad = zero(SVector{N,Float64}))

    return flag, MC{N,MV}(cv, cc, zIntv, cv_grad, cc_grad, cnst)
end

@inline function mult_kernel(x1::MC{N,MV}, x2::MC{N,MV}, y::Interval{Float64}) where N
	degen1 = (x1.Intv.hi - x1.Intv.lo) <= MC_DEGEN_TOL
	degen2 = (x2.Intv.hi - x2.Intv.lo) <= MC_DEGEN_TOL
	if degen1 || degen2
		multiply_STD_NS(x1, x2, y)
	end
	flag, x3 = multiply_MV_NS(x1, x2, y, x1.cnst && x2.cnst)
	~flag && (return nan(MC{N,MV}))
	return x3
end

#
@inline function *(x1::MC{N,T}, x2::MC{N,T}) where {N, T<:Union{NS,MV}}
	mult_kernel(x1, x2, x1.Intv*x2.Intv)
end
@inline function *(x1::MC{N,Diff}, x2::MC{N,Diff}) where N
	degen1 = ((x1.Intv.hi - x1.Intv.lo) == 0.0)
	degen2 = ((x2.Intv.hi - x2.Intv.lo) == 0.0)
	if ~(degen1 || degen2)
		if (min(x1.Intv.lo, x2.Intv.lo) < 0.0 < max(x1.Intv.hi, x2.Intv.hi))
			lo_Intv_calc::Float64 = gCxAIntv(x1.Intv, x2.Intv, x1.Intv, x2.Intv, x1, x2)
			hi_Intv_calc::Float64 = -gCxAIntv(-x1.Intv, x2.Intv, -x1.Intv, x2.Intv, x1, x2)
			z = Interval{Float64}(lo_Intv_calc, hi_Intv_calc)
		else
			z = x1.Intv*x2.Intv
		end
	else
		z = x1.Intv*x2.Intv
	end
	return mult_kernel(x1, x2, z)
end
