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
# src/forward_operators/trilinear.jl
# Defines trilinear product-based on the Meyer-Floudas envelope.
#############################################################################

"""
Computes the maximum of six values and returns the position of their occurance.
"""
function max6(a::Float64, b::Float64, c::Float64, d::Float64, e::Float64, f::Float64)
    if a >= b && a >= c && a >= d && a >= e && a >= f
        return a, 1
    elseif b >= a && b >= c && b >= d && b >= e && b >= f
        return b, 2
    elseif c >= a && c >= b && c >= d && c >= e && c >= f
        return c, 3
    elseif d >= a && d >= b && d >= c && d >= e && d >= f
        return d, 4
    elseif e >= a && e >= b && e >= c && e >= d && e >= f
        return e, 5
    end
    return f, 6
end

"""
Computes the maximum of six values and returns the position of their occurance.
"""
function min6(a::Float64, b::Float64, c::Float64, d::Float64, e::Float64, f::Float64)
    if a <= b && a <= c && a <= d && a <= e && a <= f
        return a, 1
    elseif b <= a && b <= c && b <= d && b <= e && b <= f
        return b, 2
    elseif c <= a && c <= b && c <= d && c <= e && c <= f
        return c, 3
    elseif d <= a && d <= b && d <= c && d <= e && d <= f
        return d, 4
    elseif e <= a && e <= b && e <= c && e <= d && e <= f
        return e, 5
    end
    return f, 6
end

"""
Picks the `i + 1` argument provided of six elements.
"""
function coeff6(i::Int64, a::Float64, b::Float64, c::Float64, d::Float64, e::Float64, f::Float64)
    (i === 1) && (return a)
    (i === 2) && (return b)
    (i === 3) && (return c)
    (i === 4) && (return d)
    (i === 5) && (return e)
    return f
end

macro is_neg(x)
    new_sym::Symbol = Symbol(String(x)*"U")
    esc(quote $new_sym <= 0.0 end)
end
macro is_pos(x)
    new_sym::Symbol = Symbol(String(x)*"L")
    esc(quote $new_sym >= 0.0 end)
end
macro is_mix(x)
    new_symU::Symbol = Symbol(String(x)*"U")
    new_symL::Symbol = Symbol(String(x)*"L")
    esc(quote $new_symU >= 0.0 >= $new_symL end)
end


macro unpack_trilinear_bnd()
    esc(quote
            xL = x.Intv.lo;     xU = x.Intv.hi
            yL = y.Intv.lo;     yU = y.Intv.hi
            zL = z.Intv.lo;     zU = z.Intv.hi
            xyLL = xL*yL;       xyLU = xL*yU;
            xyUL = xU*yL;       xyUU = xU*yU
            xzLL = xL*zL;       xzLU = xL*zU;
            xzUL = xU*zL;       xzUU = xU*zU
            yzLL = yL*zL;       yzLU = yL*zU;
            yzUL = yU*zL;       yzUU = yU*zU
            xyzLLL = xyLL*zL;   xyzLLU = xyLL*zU
            xyzLUL = xyLU*zL;   xyzULL = xyUL*zL
            xyzLUU = xyLU*zU;   xyzULU = xyUL*zU
            xyzUUL = xyUU*zL;   xyzUUU = xyUU*zU
        end)
end

macro unpack_trilinear_end()
    esc(quote
        cv, cvind = max6(cv1, cv2, cv3, cv4, cv5, cv6)
        cvsubax = coeff6(cvind, cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6)
        cvsubay = coeff6(cvind, cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6)
        cvsubaz = coeff6(cvind, cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6)
        cv_grad = cvsubax*ifelse(cvsubax > 0.0, x.cv_grad, x.cc_grad) +
                  cvsubay*ifelse(cvsubay > 0.0, y.cv_grad, y.cc_grad) +
                  cvsubaz*ifelse(cvsubaz > 0.0, z.cv_grad, z.cc_grad)

        cc, ccind = min6(cc1, cc2, cc3, cc4, cc5, cc6)
        ccsubax = coeff6(ccind, cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6)
        ccsubay = coeff6(ccind, cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6)
        ccsubaz = coeff6(ccind, cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6)
        cc_grad = ccsubax*ifelse(ccsubax > 0.0, x.cc_grad, x.cv_grad) +
                  ccsubay*ifelse(ccsubay > 0.0, y.cc_grad, y.cv_grad) +
                  ccsubaz*ifelse(ccsubaz > 0.0, z.cc_grad, z.cv_grad)

        MC{N,T}(cv, cc, q, cv_grad, cc_grad, x.cnst && y.cnst && z.cnst)
    end)
end

"""
trilinear_case_1

Case 3.1 + Case 4.1 of Meyer-Floudas 2004
"""
function trilinear_case_1(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()

    delX = xU - xL
    θcv1 = xyzUUL - xyzLUU - xyzULL + xyzULU
    θcv2 = xyzLLU - xyzULL - xyzLUU + xyzLUL

    # define cv and coefficients
    cv_b1 = -2.0*xyzLLL
    cv_b2 = -2.0*xyzUUU
    cv_b3 = -(xyzLLU + xyzULU)
    cv_b4 = -(xyzUUL + xyzLUL)
    cv_b5 = -θcv1*xL/delX - xyzUUL - xyzULU + xyzLUU
    cv_b6 = θcv2*xU/delX - xyzLLU - xyzLUL + xyzULL

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzLL, yzUU, yzLU, yzUL, θcv1/delX, -θcv2/delX
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzLL, xzUU, xzLU, xzUL, xzUL, xzLU
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyLL, xyUU, xyUL, xyLU, xyUL, xyLU

    cv1 = cv_ax1*x.cv + cv_ay1*y.cv + cv_az1*z.cv + cv_b1
    cv2 = cv_ax2*x.cv + cv_ay2*y.cv + cv_az2*z.cv + cv_b2
    cv3 = cv_ax3*x.cv + cv_ay3*y.cv + cv_az3*z.cv + cv_b3
    cv4 = cv_ax4*x.cv + cv_ay4*y.cv + cv_az4*z.cv + cv_b4
    cv5 = cv_ax5*ifelse(cv_ax5 > 0.0, x.cv, -x.cc) + cv_ay5*y.cv + cv_az5*z.cv + cv_b5
    cv6 = cv_ax6*ifelse(cv_ax6 > 0.0, x.cv, -x.cc) + cv_ay6*y.cv + cv_az6*z.cv + cv_b6

    # define cc and coefficients
    cc_b1 = -(xyzUUL + xyzULL)
    cc_b2 = -(xyzUUL + xyzLUL)
    cc_b3 = -(xyzULU + xyzULL)
    cc_b4 = -(xyzLUU + xyzLUL)
    cc_b5 = -(xyzULU + xyzLLU)
    cc_b6 = -(xyzLUU + xyzLLU)

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzLL, yzUL, yzLL, yzUU, yzLU, yzUU
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzUL, xzLL, xzUU, xzLL, xzUU, xzLU
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyUU, xyUU, xyUL, xyLU, xyLL, xyLL

    cc1 = cc_ax1*x.cc + cc_ay1*y.cc + cc_az1*z.cc + cc_b1
    cc2 = cc_ax2*x.cc + cc_ay2*y.cc + cc_az2*z.cc + cc_b2
    cc3 = cc_ax3*x.cc + cc_ay3*y.cc + cc_az3*z.cc + cc_b3
    cc4 = cc_ax4*x.cc + cc_ay4*y.cc + cc_az4*z.cc + cc_b4
    cc5 = cc_ax5*x.cc + cc_ay5*y.cc + cc_az5*z.cc + cc_b5
    cc6 = cc_ax6*x.cc + cc_ay6*y.cc + cc_az6*z.cc + cc_b6


    @unpack_trilinear_end()
end

"""
trilinear_case_2

Case 3.2 + Case 4.2 of Meyer-Floudas 2004
"""
function trilinear_case_2(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()

    delZ = zU - zL
    θcv = xyzLUU - xyzUUL - xyzLLU + xyzULU
    θcc = xyzULL - xyzUUU - xyzLLL + xyzLUL

    # define cv and coefficients
    cv_b1 = -2.0*xyzUUU
    cv_b2 = -(xyzLUL + xyzLUU)
    cv_b3 = -(xyzLUL + xyzLLL)
    cv_b4 = -(xyzULU + xyzULL)
    cv_b5 = -(xyzULL + xyzLLL)
    cv_b6 = -θcv*zL/delZ - xyzLUU - xyzULU + xyzUUL

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzUU, yzUL, yzUL, yzLU, yzLL, yzLU
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzUU, xzLU, xzLL, xzUL, xzUL, xzLU
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyUU, xyLU, xyLL, xyUL, xyLL, θcv/delZ

    # define cc and coefficients
    cc_b1 = -2.0*xyzUUL
    cc_b2 = -(xyzULU + xyzULL)
    cc_b3 = -(xyzLUU + xyzLLU)
    cc_b4 = -(xyzLUU + xyzLUL)
    cc_b5 = -(xyzULU + xyzLLU)
    cc_b6 = θcc*zU/delZ - xyzULL - xyzLUL + xyzUUU

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzUL, yzLL, yzUU, yzUU, yzLU, yzLL
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzUL, xzUU, xzLU, xzLL, xzUU, xzLL
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyUU, xyUL, xyLL, xyLU, xyLL, -θcc/delZ

    cv1 = cv_ax1*x.cv + cv_ay1*y.cv + cv_az1*z.cv + cv_b1
    cv2 = cv_ax2*x.cc + cv_ay2*y.cv + cv_az2*z.cc + cv_b2
    cv3 = cv_ax3*x.cc + cv_ay3*y.cc + cv_az3*z.cc + cv_b3
    cv4 = cv_ax4*x.cv + cv_ay4*y.cc + cv_az4*z.cv + cv_b4
    cv5 = cv_ax5*x.cc + cv_ay5*y.cc + cv_az5*z.cc + cv_b5
    cv6 = cv_ax6*x.cv + cv_ay6*y.cv + cv_az6*ifelse(cv_az6 > 0.0, z.cv, -z.cc) + cv_b6

    cc1 = cc_ax1*x.cv + cc_ay1*y.cv + cc_az1*z.cc + cc_b1
    cc2 = cc_ax2*x.cv + cc_ay2*y.cc + cc_az2*z.cv + cc_b2
    cc3 = cc_ax3*x.cc + cc_ay3*y.cc + cc_az3*z.cv + cc_b3
    cc4 = cc_ax4*x.cc + cc_ay4*y.cv + cc_az4*z.cc + cc_b4
    cc5 = cc_ax5*x.cc + cc_ay5*y.cc + cc_az5*z.cv + cc_b5
    cc6 = cc_ax6*x.cv + cc_ay6*y.cv + cc_az6*ifelse(cc_az6 > 0.0, z.cc, -z.cv) + cc_b6

    @unpack_trilinear_end()
end

function trilinear_case3_chk_cv(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()
    xyzUUL + xyzLLU <= xyzLUL + xyzULU
end

function trilinear_case3_chk_cc(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()
    xyzLLL + xyzUUU >= xyzULL + xyzLUU
end

function trilinear_case3_cv(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    delY = yU - yL
    delZ = zU - zL

    θcv1 = xyzLLL - xyzUUL - xyzLLU + xyzULU
    θcv2 = xyzULU - xyzUUL - xyzLLU + xyzLUU

    cv_b1 = -2.0*xyzUUU
    cv_b2 = -2.0*xyzULL
    cv_b3 = -(xyzLUL + xyzLUU)
    cv_b4 = -(xyzLUL + xyzLLL)
    cv_b5 = θcv1*yU/delY - xyzLLL - xyzULU + xyzUUL
    cv_b6 = -θcv2*zL/delZ - xyzULU - xyzLUU + xyzUUL

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzUU, yzLL, yzUL, yzUL, yzLU, yzLU
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzUU, xzUL, xzLU, xzLL, -θcv1/delY, xzLU
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyUU, xyUL, xyLU, xyLL, xyLL, θcv2/delZ

    cv1 = cv_ax1*x.cv + cv_ay1*y.cv + cv_az1*z.cv + cv_b1
    cv2 = cv_ax2*x.cv + cv_ay2*y.cc + cv_az2*z.cc + cv_b2
    cv3 = cv_ax3*x.cc + cv_ay3*y.cv + cv_az3*z.cv + cv_b3
    cv4 = cv_ax4*x.cc + cv_ay4*y.cc + cv_az4*z.cc + cv_b4
    cv5 = cv_ax5*x.cc + cv_ay5*ifelse(cv_ay5 > 0.0, y.cv, -y.cc) + cv_az5*z.cc + cv_b5
    cv6 = cv_ax6*x.cc + cv_ay6*y.cv + cv_az6*ifelse(cv_az6 > 0.0, z.cv, -z.cc) + cv_b6

    cv, cvind = max6(cv1, cv2, cv3, cv4, cv5, cv6)
    cvsubax = coeff6(cvind, cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6)
    cvsubay = coeff6(cvind, cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6)
    cvsubaz = coeff6(cvind, cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6)
    cv_grad = cvsubax*ifelse(cvsubax > 0.0, x.cv_grad, x.cc_grad) +
              cvsubay*ifelse(cvsubay > 0.0, y.cv_grad, y.cc_grad) +
              cvsubaz*ifelse(cvsubaz > 0.0, z.cv_grad, z.cc_grad)

    return cv, cv_grad
end

function trilinear_case3_cc1(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    delY = yU - yL
    delZ = zU - zL

    θcc1 = xyzULL - xyzUUU - xyzLLL + xyzLUL
    θcc2 = xyzULL - xyzUUU - xyzLLL + xyzLLU

    cc_b1 = -2.0*xyzULU
    cc_b2 = -2.0*xyzUUL
    cc_b3 = -(xyzLUU + xyzLLU)
    cc_b4 = -(xyzLUU + xyzLUL)
    cc_b5 = θcc1*zU/delZ - xyzULL - xyzLUL + xyzUUU
    cc_b6 = θcc2*yU/delY - xyzULL - xyzLLU + xyzUUU

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzLU, yzUL, yzUU, yzUU, yzLL, yzLL
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzUU, xzUL, xzLU, xzLL, xzLL, -θcc2/delY
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyUL, xyUU, xyLL, xyLU, -θcc1/delZ, xyLL

    cc1 = cc_ax1*x.cv + cc_ay1*y.cc + cc_az1*z.cv + cc_b1
    cc2 = cc_ax2*x.cv + cc_ay2*y.cv + cc_az2*z.cc + cc_b2
    cc3 = cc_ax3*x.cc + cc_ay3*y.cc + cc_az3*z.cv + cc_b3
    cc4 = cc_ax4*x.cc + cc_ay4*y.cv + cc_az4*z.cc + cc_b4
    cc5 = cc_ax5*x.cc + cc_ay5*y.cv + cc_az5*ifelse(cc_az5 > 0.0, z.cc, -z.cv) + cc_b5
    cc6 = cc_ax6*x.cc + cc_ay6*ifelse(cc_ay6 > 0.0, y.cc, -z.cv) + cc_az6*z.cv + cc_b6

    cc, ccind = min6(cc1, cc2, cc3, cc4, cc5, cc6)
    ccsubax = coeff6(ccind, cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6)
    ccsubay = coeff6(ccind, cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6)
    ccsubaz = coeff6(ccind, cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6)
    cc_grad = ccsubax*ifelse(ccsubax > 0.0, x.cc_grad, x.cv_grad) +
              ccsubay*ifelse(ccsubay > 0.0, y.cc_grad, y.cv_grad) +
              ccsubaz*ifelse(ccsubaz > 0.0, z.cc_grad, z.cv_grad)

    return cc, cc_grad
end

function trilinear_case3_cc2(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    delY = yU - yL
    delZ = zU - zL

    θcc1 = xyzUUU - xyzULL - xyzLUU + xyzLLU
    θcc2 = xyzUUU - xyzULL - xyzLUU + xyzLUL

    cc_b1 = -2.0*xyzULU
    cc_b2 = -2.0*xyzUUL
    cc_b3 = -(xyzLLL + xyzLUL)
    cc_b4 = -(xyzLLL + xyzLLU)
    cc_b5 = -θcc1*zL/delZ - xyzUUU - xyzLLU + xyzULL
    cc_b6 = -θcc2*yL/delY - xyzUUU - xyzLUL + xyzULL

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzLU, yzUL, yzLL, yzLL, yzUU, yzUU
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzUU, xzUL, xzLL, xzLU, xzLU, θcc1/delZ
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyUL, xyUU, xyLU, xyLL, θcc2/delY, xyLU

    cc1 = cc_ax1*x.cv + cc_ay1*y.cc + cc_az1*z.cv + cc_b1
    cc2 = cc_ax2*x.cv + cc_ay2*y.cv + cc_az2*z.cc + cc_b2
    cc3 = cc_ax3*x.cc + cc_ay3*y.cv + cc_az3*z.cc + cc_b3
    cc4 = cc_ax4*x.cc + cc_ay4*y.cc + cc_az4*z.cv + cc_b4
    cc5 = cc_ax5*x.cc + cc_ay5*y.cc + cc_az5*ifelse(cc_az5 > 0.0, z.cc, -z.cv) + cc_b5
    cc6 = cc_ax6*x.cc + cc_ay6*ifelse(cc_ay6 > 0.0, z.cc, -z.cv) + cc_az6*z.cc + cc_b6

    cc, ccind = min6(cc1, cc2, cc3, cc4, cc5, cc6)
    ccsubax = coeff6(ccind, cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6)
    ccsubay = coeff6(ccind, cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6)
    ccsubaz = coeff6(ccind, cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6)
    cc_grad = ccsubax*ifelse(ccsubax > 0.0, x.cc_grad, x.cv_grad) +
              ccsubay*ifelse(ccsubay > 0.0, y.cc_grad, y.cv_grad) +
              ccsubaz*ifelse(ccsubaz > 0.0, z.cc_grad, z.cv_grad)

    return cc, cc_grad
end

"""
trilinear_case_3

Case 3.3 + Case 4.3 of Meyer-Floudas 2004
"""
function trilinear_case_3(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    xL = x.Intv.lo;     xU = x.Intv.hi
    yL = y.Intv.lo;     yU = y.Intv.hi
    zL = z.Intv.lo;     zU = z.Intv.hi

    if @is_pos(x) && @is_mix(y) && @is_mix(z)
        if trilinear_case3_chk_cv(x,y,z)
            cv, cv_grad = trilinear_case3_cv(x,y,z)
        else
            cv, cv_grad = trilinear_case3_cv(x,z,y)
        end
        if trilinear_case3_chk_cc(x,y,z)
            cc, cc_grad = trilinear_case3_cc1(x,y,z)
        elseif trilinear_case3_chk_cc(x,z,y)
            cc, cc_grad = trilinear_case3_cc1(x,z,y)
        else
            cc, cc_grad = trilinear_case3_cc2(x,y,z)
        end
        return MC{N,T}(cv, cc, q, cv_grad, cc_grad, x.cnst && y.cnst && z.cnst)
    elseif @is_mix(x) && @is_pos(y) && @is_mix(z)
        if trilinear_case3_chk_cv(y,x,z)
            cv, cv_grad = trilinear_case3_cv(y,x,z)
        else
            cv, cv_grad = trilinear_case3_cv(y,z,x)
        end
        if trilinear_case3_chk_cc(y,x,z)
            cc, cc_grad = trilinear_case3_cc1(y,x,z)
        elseif trilinear_case3_chk_cc(y,z,x)
            cc, cc_grad = trilinear_case3_cc1(y,z,x)
        else
            cc, cc_grad = trilinear_case3_cc2(y,x,z)
        end
        return MC{N,T}(cv, cc, q, cv_grad, cc_grad, x.cnst && y.cnst && z.cnst)
    end

    if trilinear_case3_chk_cv(z,x,y)
        cv, cv_grad = trilinear_case3_cv(z,x,y)
    else
        cv, cv_grad = trilinear_case3_cv(z,y,x)
    end
    if trilinear_case3_chk_cc(z,x,y)
        cc, cc_grad = trilinear_case3_cc1(z,x,y)
    elseif trilinear_case3_chk_cc(z,y,x)
        cc, cc_grad = trilinear_case3_cc1(z,y,x)
    else
        cc, cc_grad = trilinear_case3_cc2(z,x,y)
    end
    MC{N,T}(cv, cc, q, cv_grad, cc_grad, x.cnst && y.cnst && z.cnst)
end

function trilinear_case4_chk_cv1(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    (xyzUUL + xyzULU + xyzLUU <= xyzLLL + 2*xyzUUU) &&
    (xyzLLL + xyzUUL + xyzULU <= xyzLUU + 2*xyzULL) &&
    (xyzLLL + xyzUUL + xyzLUU <= xyzULU + 2*xyzLUL) &&
    (xyzLLL + xyzULU + xyzLUU <= xyzUUL + 2*xyzLLU)
end
function trilinear_case4_chk_cv2(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    xyzUUL + xyzULU + xyzLUU >= xyzLLL + 2*xyzUUU
end
function trilinear_case4_chk_cv3(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    xyzLLL + xyzUUL + xyzULU >= xyzLUU + 2.0*xyzULL
end

function trilinear_case4_cv1(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    delX = xU - xL
    delY = yU - yL
    delZ = zU - zL

    θcv1 = -0.5*(xyzLUU + xyzLLL - xyzUUL - xyzULU)/delX
    θcv2 = -0.5*(xyzULU + xyzLLL - xyzUUL - xyzLUU)/delY
    θcv3 = -0.5*(xyzUUL + xyzLLL - xyzULU - xyzLUU)/delX
    θcv4 = xyzLLL - θcv1*xL - θcv2*yL - θcv3*zL

    cv_b1 = -2.0*xyzLUL
    cv_b2 = -2.0*xyzLLU
    cv_b3 = -2.0*xyzUUU
    cv_b4 = -2.0*xyzULL
    cv_b5 = θcv4
    cv_b6 = Inf #TODO? eq 6 not in either paper

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzUL, yzLU, yzUU, yzLL, θcv1, Inf  # TODO? eq 6 not in paper
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzLL, xzLU, xzUU, xzUL, θcv2, Inf  # TODO? eq 6 not in paper
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyLU, xyLL, xyUU, xyUL, θcv3, Inf  # TODO? eq 6 not in paper

    cv1 = cv_ax1*x.cc + cv_ay1*y.cv + cv_az1*z.cc + cv_b1
    cv2 = cv_ax2*x.cc + cv_ay2*y.cc + cv_az2*z.cv + cv_b2
    cv3 = cv_ax3*x.cv + cv_ay3*y.cv + cv_az3*z.cv + cv_b3
    cv4 = cv_ax4*x.cv + cv_ay4*y.cc + cv_az4*z.cc + cv_b4
    cv5 = cv_ax5*ifelse(cv_ax5 > 0.0, x.cv, -x.cc) +
          cv_ay5*ifelse(cv_ay5 > 0.0, y.cv, -y.cc) +
          cv_az5*ifelse(cv_az5 > 0.0, z.cv, -z.cc) + cv_b5
    cv6 = -Inf # TODO? eq 6 not in either paper

    cv, cvind = max6(cv1, cv2, cv3, cv4, cv5, cv6)
    cvsubax = coeff6(cvind, cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6)
    cvsubay = coeff6(cvind, cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6)
    cvsubaz = coeff6(cvind, cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6)
    cv_grad = cvsubax*ifelse(cvsubax > 0.0, x.cv_grad, x.cc_grad) +
              cvsubay*ifelse(cvsubay > 0.0, y.cv_grad, y.cc_grad) +
              cvsubaz*ifelse(cvsubaz > 0.0, z.cv_grad, z.cc_grad)

    return cv, cv_grad
end
function trilinear_case4_cv2(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    delX = xU - xL
    delY = yU - yL
    delZ = zU - zL

    θcv1 = xyzUUL - xyzLLL - xyzUUU + xyzULU
    θcv2 = xyzULU - xyzLLL - xyzUUU + xyzLUU
    θcv3 = xyzUUL - xyzLLL - xyzUUU + xyzLUU

    cv_b1 = -2.0*xyzLUL
    cv_b2 = -2.0*xyzLLU
    cv_b3 = -2.0*xyzULL
    cv_b4 = -θcv1*xL/delX - xyzUUL - xyzULU + xyzLLL
    cv_b5 = -θcv2*zL/delZ - xyzULU - xyzLUU + xyzLLL
    cv_b6 = -θcv3*yL/delY - xyzUUL - xyzLUU + xyzLLL

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzUL, yzLU, yzLL, θcv1/delX, yzUU, yzUU
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzLL, xzLU, xzUL, xzUU, xzUU, θcv3/delY
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyLU, xyLL, xyUL, xyUU, θcv2/delZ, xyUU

    cv1 = cv_ax1*x.cc + cv_ay1*y.cv + cv_az1*z.cc + cv_b1
    cv2 = cv_ax2*x.cc + cv_ay2*y.cc + cv_az2*z.cv + cv_b2
    cv3 = cv_ax3*x.cv + cv_ay3*y.cc + cv_az3*z.cc + cv_b3
    cv4 = cv_ax4*ifelse(cv_ax4 > 0.0, x.cv, -x.cc) + cv_ay4*y.cv + cv_az4*z.cv + cv_b4
    cv5 = cv_ax5*x.cv + cv_ay5*ifelse(cv_ay5 > 0.0, y.cv, -y.cc) + cv_az5*z.cv + cv_b5
    cv6 = cv_ax6*x.cv + cv_ay6*y.cv + cv_az6*ifelse(cv_az6 > 0.0, z.cv, -z.cc) + cv_b6

    cv, cvind = max6(cv1, cv2, cv3, cv4, cv5, cv6)
    cvsubax = coeff6(cvind, cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6)
    cvsubay = coeff6(cvind, cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6)
    cvsubaz = coeff6(cvind, cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6)
    cv_grad = cvsubax*ifelse(cvsubax > 0.0, x.cv_grad, x.cc_grad) +
              cvsubay*ifelse(cvsubay > 0.0, y.cv_grad, y.cc_grad) +
              cvsubaz*ifelse(cvsubaz > 0.0, z.cv_grad, z.cc_grad)

    return cv, cv_grad
end
function trilinear_case4_cv3(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    delX = xU - xL
    delY = yU - yL
    delZ = zU - zL

    θcv1 = xyzLLL - xyzLUU - xyzULL + xyzULU
    θcv2 = xyzUUL - xyzLUU - xyzULL + xyzULU
    θcv3 = xyzLLL - xyzLUU - xyzULL + xyzUUL

    cv_b1 = -2.0*xyzLUL
    cv_b2 = -2.0*xyzLLU
    cv_b3 = -2.0*xyzUUU
    cv_b4 = θcv1*yU/delY - xyzLLL - xyzULU + xyzLUU
    cv_b5 = -θcv2*xL/delX - xyzUUL - xyzULU + xyzLUU
    cv_b6 = θcv3*zU/delZ - xyzLLL - xyzUUL + xyzLUU

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzUL, yzLU, yzUU, yzLL, θcv2/delX, yzLL
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzLL, xzLU, xzUU, -θcv1/delY, xzUL, xzUL
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyLU, xyLL, xyUU, xyUL, xyUL, -θcv3/delZ

    cv1 = cv_ax1*x.cc + cv_ay1*y.cv + cv_az1*z.cc + cv_b1
    cv2 = cv_ax2*x.cc + cv_ay2*y.cc + cv_az2*z.cv + cv_b2
    cv3 = cv_ax3*x.cv + cv_ay3*y.cv + cv_az3*z.cv + cv_b3
    cv4 = cv_ax4*x.cv + cv_ay4*ifelse(cv_ay4 > 0.0, y.cv, -y.cc) + cv_az4*z.cc + cv_b4
    cv5 = cv_ax5*ifelse(cv_ax5 > 0.0, x.cv, -x.cc) + cv_ay5*y.cc + cv_az5*z.cc + cv_b5
    cv6 = cv_ax6*x.cv + cv_ay6*y.cc + cv_az6*ifelse(cv_az6 > 0.0, z.cv, -z.cc) + cv_b6

    cv, cvind = max6(cv1, cv2, cv3, cv4, cv5, cv6)
    cvsubax = coeff6(cvind, cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6)
    cvsubay = coeff6(cvind, cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6)
    cvsubaz = coeff6(cvind, cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6)
    cv_grad = cvsubax*ifelse(cvsubax > 0.0, x.cv_grad, x.cc_grad) +
              cvsubay*ifelse(cvsubay > 0.0, y.cv_grad, y.cc_grad) +
              cvsubaz*ifelse(cvsubaz > 0.0, z.cv_grad, z.cc_grad)

    return cv, cv_grad
end

function trilinear_case4_chk_cc1(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    (xyzULL + xyzLUL + xyzLLU) >= (xyzUUU + 2.0*xyzLLL) &&
    (xyzLUL + xyzLLU + xyzUUU) >= (xyzULL + 2.0*xyzLUU) &&
    (xyzULL + xyzLLU + xyzUUU) >= (xyzLUL + 2.0*xyzULU) &&
    (xyzULL + xyzLUL + xyzUUU) >= (xyzLLU + 2.0*xyzUUL)
end
function trilinear_case4_chk_cc2(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    (xyzULL + xyzLUL + xyzLLU) <= (xyzUUU + 2.0*xyzLLL)
end
function trilinear_case4_chk_cc3(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    xyzLUL + xyzLLU + xyzUUU <= xyzULL + 2.0*xyzLUU
end

function trilinear_case4_cc1(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    delX = xU - xL
    delY = yU - yL
    delZ = zU - zL

    θcc1 = 0.5*(xyzULL + xyzUUU - xyzLLU - xyzLUL)/delX
    θcc2 = 0.5*(xyzLUL + xyzUUU - xyzLLU - xyzULL)/delY
    θcc3 = 0.5*(xyzLLU + xyzUUU - xyzLUL - xyzULL)/delZ
    θcc4 = xyzUUU - θcc1*xU - θcc2*yU - θcc3*zU

    cc_b1 = -2.0*xyzLLL
    cc_b2 = -2.0*xyzUUL
    cc_b3 = -2.0*xyzULU
    cc_b4 = -2.0*xyzLUU
    cc_b5 = θcc4
    cc_b6 = Inf #TODO? eq 6 not in paper

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzLL, yzUL, yzLU, yzUU, θcc1, Inf
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzLL, xzUL, xzUU, xzLU, θcc2, Inf
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyLL, xyUU, xyUL, xyLU, θcc3, Inf

    cc1 = cc_ax1*x.cc + cc_ay1*y.cc + cc_az1*z.cc + cc_b1
    cc2 = cc_ax2*x.cv + cc_ay2*y.cv + cc_az2*z.cc + cc_b2
    cc3 = cc_ax3*x.cv + cc_ay3*y.cc + cc_az3*z.cv + cc_b3
    cc4 = cc_ax4*x.cc + cc_ay4*y.cv + cc_az4*z.cv + cc_b4
    cc5 = cc_ax5*ifelse(cc_ax5 > 0.0, x.cc, -x.cv) +
          cc_ay5*ifelse(cc_ay5 > 0.0, y.cc, -y.cv) +
          cc_az5*ifelse(cc_az5 > 0.0, z.cc, -z.cv) + cc_b5
    cc6 = Inf # TODO? eq 6 not in paper

    cc, ccind = min6(cc1, cc2, cc3, cc4, cc5, cc6)
    ccsubax = coeff6(ccind, cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6)
    ccsubay = coeff6(ccind, cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6)
    ccsubaz = coeff6(ccind, cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6)
    cc_grad = ccsubax*ifelse(ccsubax > 0.0, x.cc_grad, x.cv_grad) +
              ccsubay*ifelse(ccsubay > 0.0, y.cc_grad, y.cv_grad) +
              ccsubaz*ifelse(ccsubaz > 0.0, z.cc_grad, z.cv_grad)

    return cc, cc_grad
end
function trilinear_case4_cc2(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    delX = xU - xL
    delY = yU - yL
    delZ = zU - zL

    θcc1 = xyzULL - xyzUUU - xyzLLL + xyzLUL
    θcc2 = xyzLLU - xyzUUU - xyzLLL + xyzLUL
    θcc3 = xyzLLU - xyzUUU - xyzLLL + xyzULL

    cc_b1 = -2.0*xyzUUL
    cc_b2 = -2.0*xyzULU
    cc_b3 = -2.0*xyzLUU
    cc_b4 = θcc1*zU/delZ - xyzULL - xyzLUL + xyzUUU
    cc_b5 = θcc2*xU/delX - xyzLLU - xyzLUL + xyzUUU
    cc_b6 = θcc3*yU/delY - xyzLLU - xyzULL + xyzUUU

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzUL, yzLU, yzUU, yzLL, -θcc2/delX, yzLL
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzUL, xzUU, xzLU, xzLL, xzLL, -θcc3/delY
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyUU, xyUL, xyLU, -θcc1/delZ, xyLL, xyLL

    cc1 = cc_ax1*x.cv + cc_ay1*y.cv + cc_az1*z.cc + cc_b1
    cc2 = cc_ax2*x.cv + cc_ay2*y.cc + cc_az2*z.cv + cc_b2
    cc3 = cc_ax3*x.cc + cc_ay3*y.cv + cc_az3*z.cv + cc_b3
    cc4 = cc_ax4*x.cc + cc_ay4*y.cc + cc_az4*ifelse(cc_az4 > 0.0, z.cc, -z.cv) + cc_b4
    cc5 = cc_ax5*ifelse(cc_ax5 > 0.0, x.cc, -x.cv) + cc_ay5*y.cc + cc_az5*z.cc + cc_b5
    cc6 = cc_ax6*x.cc + cc_ay6*ifelse(cc_ay6 > 0.0, y.cc, -y.cv) + cc_az6*z.cc + cc_b6

    cc, ccind = min6(cc1, cc2, cc3, cc4, cc5, cc6)
    ccsubax = coeff6(ccind, cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6)
    ccsubay = coeff6(ccind, cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6)
    ccsubaz = coeff6(ccind, cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6)
    cc_grad = ccsubax*ifelse(ccsubax > 0.0, x.cc_grad, x.cv_grad) +
              ccsubay*ifelse(ccsubay > 0.0, y.cc_grad, y.cv_grad) +
              ccsubaz*ifelse(ccsubaz > 0.0, z.cc_grad, z.cv_grad)

    return cc, cc_grad
end
function trilinear_case4_cc3(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    delX = xU - xL
    delY = yU - yL
    delZ = zU - zL

    θcc1 = xyzLLU - xyzULL - xyzLUU + xyzLUL
    θcc2 = xyzUUU - xyzULL - xyzLUU + xyzLUL
    θcc3 = xyzLLU - xyzULL - xyzLUU + xyzUUU

    cc_b1 = -2.0*xyzLLL
    cc_b2 = -2.0*xyzUUL
    cc_b3 = -2.0*xyzULU
    cc_b4 = θcc1*xU/delX - xyzLLU - xyzLUL + xyzULL
    cc_b5 = -θcc2*yL/delY - xyzUUU - xyzLUL + xyzULL
    cc_b6 = -θcc3*zL/delZ - xyzLLU - xyzUUU + xyzULL

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzLL, yzUL, yzLU, -θcc1/delX, yzUU, yzUU
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzLL, xzUL, xzUU, xzLU, θcc2/delY, xzLU
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyLL, xyUU, xyUL, xyLU, xyLU, θcc3/delZ

    cc1 = cc_ax1*x.cc + cc_ay1*y.cc + cc_az1*z.cc + cc_b1
    cc2 = cc_ax2*x.cv + cc_ay2*y.cv + cc_az2*z.cc + cc_b2
    cc3 = cc_ax3*x.cv + cc_ay3*y.cc + cc_az3*z.cv + cc_b3
    cc4 = cc_ax4*ifelse(cc_ax4 > 0.0, x.cc, -x.cv) + cc_ay4*y.cv + cc_az4*z.cv + cc_b4
    cc5 = cc_ax5*x.cc + cc_ay5*ifelse(cc_ay5 > 0.0, y.cc, -y.cv) + cc_az5*z.cv + cc_b5
    cc6 = cc_ax6*x.cc + cc_ay6*y.cv + cc_az6*ifelse(cc_az6 > 0.0, z.cc, -z.cv) + cc_b6

    cc, ccind = min6(cc1, cc2, cc3, cc4, cc5, cc6)
    ccsubax = coeff6(ccind, cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6)
    ccsubay = coeff6(ccind, cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6)
    ccsubaz = coeff6(ccind, cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6)
    cc_grad = ccsubax*ifelse(ccsubax > 0.0, x.cc_grad, x.cv_grad) +
              ccsubay*ifelse(ccsubay > 0.0, y.cc_grad, y.cv_grad) +
              ccsubaz*ifelse(ccsubaz > 0.0, z.cc_grad, z.cv_grad)

    return cc, cc_grad
end

"""
trilinear_case_4

Case 3.4 + Case 4.4 of Meyer-Floudas 2004
"""
function trilinear_case_4(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    if trilinear_case4_chk_cv1(x,y,z)
        cv, cv_grad = trilinear_case4_cv1(x,y,z)
    elseif trilinear_case4_chk_cv1(x,z,y)
        cv, cv_grad = trilinear_case4_cv1(x,z,y)
    elseif trilinear_case4_chk_cv1(y,x,z)
        cv, cv_grad = trilinear_case4_cv1(y,x,z)
    elseif trilinear_case4_chk_cv1(y,z,x)
        cv, cv_grad = trilinear_case4_cv1(y,z,x)
    elseif trilinear_case4_chk_cv1(z,x,y)
        cv, cv_grad = trilinear_case4_cv1(z,x,y)
    elseif trilinear_case4_chk_cv1(z,y,x)
        cv, cv_grad = trilinear_case4_cv1(z,y,x)
    elseif trilinear_case4_chk_cv2(x,y,z)
        cv, cv_grad = trilinear_case4_cv2(x,y,z)
    elseif trilinear_case4_chk_cv2(x,z,y)
        cv, cv_grad = trilinear_case4_cv2(x,z,y)
    elseif trilinear_case4_chk_cv2(y,x,z)
        cv, cv_grad = trilinear_case4_cv2(y,x,z)
    elseif trilinear_case4_chk_cv2(y,z,x)
        cv, cv_grad = trilinear_case4_cv2(y,z,x)
    elseif trilinear_case4_chk_cv2(z,x,y)
        cv, cv_grad = trilinear_case4_cv2(z,x,y)
    elseif trilinear_case4_chk_cv2(z,y,x)
        cv, cv_grad = trilinear_case4_cv2(z,y,x)
    elseif trilinear_case4_chk_cv3(x,y,z)
        cv, cv_grad = trilinear_case4_cv3(x,y,z)
    elseif trilinear_case4_chk_cv3(x,z,y)
        cv, cv_grad = trilinear_case4_cv3(x,z,y)
    elseif trilinear_case4_chk_cv3(y,x,z)
        cv, cv_grad = trilinear_case4_cv3(y,x,z)
    elseif trilinear_case4_chk_cv3(y,z,x)
        cv, cv_grad = trilinear_case4_cv3(y,z,x)
    elseif trilinear_case4_chk_cv3(z,x,y)
        cv, cv_grad = trilinear_case4_cv3(z,x,y)
    else
        cv, cv_grad = trilinear_case4_cv3(z,y,x)
    end

    if trilinear_case4_chk_cc1(x,y,z)
        cc, cc_grad = trilinear_case4_cc1(x,y,z)
    elseif trilinear_case4_chk_cc1(x,z,y)
        cc, cc_grad = trilinear_case4_cc1(x,z,y)
    elseif trilinear_case4_chk_cc1(y,x,z)
        cc, cc_grad = trilinear_case4_cc1(y,x,z)
    elseif trilinear_case4_chk_cc1(y,z,x)
        cc, cc_grad = trilinear_case4_cc1(y,z,x)
    elseif trilinear_case4_chk_cc1(z,x,y)
        cc, cc_grad = trilinear_case4_cc1(z,x,y)
    elseif trilinear_case4_chk_cc1(z,y,x)
        cc, cc_grad = trilinear_case4_cc1(z,y,x)
    elseif trilinear_case4_chk_cc2(x,y,z)
        cc, cc_grad = trilinear_case4_cc2(x,y,z)
    elseif trilinear_case4_chk_cc2(x,z,y)
        cc, cc_grad = trilinear_case4_cc2(x,z,y)
    elseif trilinear_case4_chk_cc2(y,x,z)
        cc, cc_grad = trilinear_case4_cc2(y,x,z)
    elseif trilinear_case4_chk_cc2(y,z,x)
        cc, cc_grad = trilinear_case4_cc2(y,z,x)
    elseif trilinear_case4_chk_cc2(z,x,y)
        cc, cc_grad = trilinear_case4_cc2(z,x,y)
    elseif trilinear_case4_chk_cc2(z,y,x)
        cc, cc_grad = trilinear_case4_cc2(z,y,x)
    elseif trilinear_case4_chk_cc3(x,y,z)
        cc, cc_grad = trilinear_case4_cc3(x,y,z)
    elseif trilinear_case4_chk_cc3(x,z,y)
        cc, cc_grad = trilinear_case4_cc3(x,z,y)
    elseif trilinear_case4_chk_cc3(y,x,z)
        cc, cc_grad = trilinear_case4_cc3(y,x,z)
    elseif trilinear_case4_chk_cc3(y,z,x)
        cc, cc_grad = trilinear_case4_cc3(y,z,x)
    elseif trilinear_case4_chk_cc3(z,x,y)
        cc, cc_grad = trilinear_case4_cc3(z,x,y)
    else
        cc, cc_grad = trilinear_case4_cc3(z,y,x)
    end

    MC{N,T}(cv, cc, q, cv_grad, cc_grad, x.cnst && y.cnst && z.cnst)
end

macro trilinear_case5_cv()
    esc(quote
        # define cv and coefficients
        cv_b1 = -(xyzLUL + xyzLLL)
        cv_b2 = -(xyzLUL + xyzLUU)
        cv_b3 = -(xyzULU + xyzULL)
        cv_b4 = -(xyzULU + xyzUUU)
        cv_b5 = -(xyzULL + xyzLLL)
        cv_b6 = -(xyzUUU + xyzLUU)

        cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzUL, yzUL, yzLU, yzLU, yzLL, yzUU
        cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzLL, xzLU, xzUL, xzUU, xzUL, xzLU
        cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyLL, xyLU, xyUL, xyUU, xyLL, xyUU

        cv1 = cv_ax1*x.cc + cv_ay1*y.cc + cv_az1*z.cv + cv_b1
        cv2 = cv_ax2*x.cc + cv_ay2*y.cc + cv_az2*z.cv + cv_b2
        cv3 = cv_ax3*x.cc + cv_ay3*y.cc + cv_az3*z.cv + cv_b3
        cv4 = cv_ax4*x.cc + cv_ay4*y.cc + cv_az4*z.cv + cv_b4
        cv5 = cv_ax5*x.cc + cv_ay5*y.cc + cv_az5*z.cv + cv_b5
        cv6 = cv_ax6*x.cc + cv_ay6*y.cc + cv_az6*z.cv + cv_b6
    end)
end

"""
trilinear_case_5a

Case 3.5 + Case 4.5 of Meyer-Floudas 2004
"""
function trilinear_case_5a(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()
    @trilinear_case5_cv()

    delZ = zU - zL
    θcc1 = xyzULL - xyzUUU - xyzLLL + xyzLUL
    θcc2 = xyzULU - xyzLLL - xyzUUU + xyzLUU

    cc_b1 = -2.0*xyzLLU
    cc_b2 = -2.0*xyzUUL
    cc_b3 = -(xyzULU + xyzULL)
    cc_b4 = -(xyzLUU + xyzLUL)
    cc_b5 = θcc1*zU/delZ - xyzULL - xyzLUL + xyzUUU
    cc_b6 = -θcc2*zL/delZ - xyzULU - xyzLUU + xyzLLL

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzLU, yzUL, yzLL, yzUU, yzLL, yzUU
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzLU, xzUL, xzUU, xzLL, xzLL, xzUU
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyLL, xyUU, xyUL, xyLU, -θcc1/delZ, θcc2/delZ

    cc1 = cc_ax1*x.cv + cc_ay1*y.cv + cc_az1*z.cc + cc_b1
    cc2 = cc_ax2*x.cv + cc_ay2*y.cv + cc_az2*z.cc + cc_b2
    cc3 = cc_ax3*x.cv + cc_ay3*y.cv + cc_az3*z.cc + cc_b3
    cc4 = cc_ax4*x.cv + cc_ay4*y.cv + cc_az4*z.cc + cc_b4
    cc5 = cc_ax5*x.cv + cc_ay5*y.cv + cc_az5*ifelse(cc_az5 > 0.0, z.cc, -z.cv) + cc_b5
    cc6 = cc_ax6*x.cv + cc_ay6*y.cv + cc_az6*ifelse(cc_az6 > 0.0, z.cc, -z.cv) + cc_b6

    @unpack_trilinear_end()
end

function trilinear_case_5b(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()
    @trilinear_case5_cv()

    delY = yU - yL
    θcc1 = xyzLLL - xyzLUU - xyzULL + xyzULU
    θcc2 = xyzLUL - xyzULL - xyzLUU + xyzUUU

    cc_b1 = -2.0*xyzLLU
    cc_b2 = -2.0*xyzUUL
    cc_b3 = -(xyzLLL + xyzLUL)
    cc_b4 = -(xyzULU + xyzUUU)
    cc_b5 = θcc1*yU/delY - xyzLLL - xyzULU + xyzLUU
    cc_b6 = -θcc2*yL/delY - xyzLUL - xyzUUU + xyzULL

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzLU, yzUL, yzLL, yzUU, yzLL, yzUU
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzLU, xzUL, xzLL, xzUU, -θcc1/delY, θcc2/delY
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyLL, xyUU, xyLU, xyUL, xyUL, xyLU

    cc1 = cc_ax1*x.cv + cc_ay1*y.cv + cc_az1*z.cc + cc_b1
    cc2 = cc_ax2*x.cv + cc_ay2*y.cv + cc_az2*z.cv + cc_b2
    cc3 = cc_ax3*x.cv + cc_ay3*y.cv + cc_az3*z.cv + cc_b3
    cc4 = cc_ax4*x.cv + cc_ay4*y.cv + cc_az4*z.cc + cc_b4
    cc5 = cc_ax5*x.cv + cc_ay5*y.cv + cc_az5*ifelse(cc_az5 > 0.0, z.cc, -z.cv) + cc_b5
    cc6 = cc_ax6*x.cv + cc_ay6*y.cv + cc_az6*ifelse(cc_az6 > 0.0, z.cc, -z.cv) + cc_b6

    @unpack_trilinear_end()
end

"""
trilinear_case_6

Note: Case 3.6/Case 4.6 of thesis yielded incorrect result. So reformulated...
"""
function trilinear_case_6(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    -trilinear_case_9(y, -x, z, q)
end

#=
function trilinear_case7_chk_cv(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()
    xyzLLL + xyzUUU <= xyzUUL + xyzLLU
end
function trilinear_case7_chk_cc(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()
    xyzULL + xyzLUU >= xyzLUL + xyzULU
end

function trilinear_case7_cv1(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    delX = xU - xL
    delY = yU - yL

    θcv1 = xyzUUL - xyzLLL - xyzUUU + xyzULU
    θcv2 = xyzUUL - xyzLLL - xyzUUU + xyzLUU

    cv_b1 = -2.0*xyzLUL
    cv_b2 = -2.0*xyzULL
    cv_b3 = -(xyzLUU + xyzLLU)
    cv_b4 = -(xyzULU + xyzLLU)
    cv_b5 = -θcv1*xL/delX - xyzUUL - xyzULU + xyzLLL
    cv_b6 = -θcv2*yL/delY - xyzUUL - xyzLUU + xyzLLL

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzUL, yzLL, yzUU, yzLU, θcv1/delX, yzUU
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzLL, xzUL, xzLU, xzUU, xzUU, θcv2/delY
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyLU, xyUL, xyLL, xyLL, xyUU, xyUU

    cv1 = cv_ax1*x.cc + cv_ay1*y.cv + cv_az1*z.cc + cv_b1
    cv2 = cv_ax2*x.cv + cv_ay2*y.cc + cv_az2*z.cc + cv_b2
    cv3 = cv_ax3*x.cc + cv_ay3*y.cv + cv_az3*z.cv + cv_b3
    cv4 = cv_ax4*x.cv + cv_ay4*y.cc + cv_az4*z.cv + cv_b4
    cv5 = cv_ax5*ifelse(cv_ax5 >= 0.0, x.cv, -x.cc) + cv_ay5*y.cc + cv_az5*z.cv + cv_b5
    cv6 = cv_ax6*x.cc + cv_ay6*ifelse(cv_ay6 >= 0.0, y.cv, -y.cc) + cv_az6*z.cv + cv_b6

    cv, cvind = max6(cv1, cv2, cv3, cv4, cv5, cv6)
    cvsubax = coeff6(cvind, cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6)
    cvsubay = coeff6(cvind, cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6)
    cvsubaz = coeff6(cvind, cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6)
    cv_grad = cvsubax*ifelse(cvsubax > 0.0, x.cv_grad, x.cc_grad) +
              cvsubay*ifelse(cvsubay > 0.0, y.cv_grad, y.cc_grad) +
              cvsubaz*ifelse(cvsubaz > 0.0, z.cv_grad, z.cc_grad)

    return cv, cv_grad
end
function trilinear_case7_cv2(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    delX = xU - xL
    delY = yU - yL

    θcv1 = xyzLLL - xyzUUL - xyzLLU + xyzLUU
    θcv2 = xyzLLL - xyzUUL - xyzLLU + xyzULU

    cv_b1 = -2.0*xyzLUL
    cv_b2 = -2.0*xyzULL
    cv_b3 = -(xyzULU + xyzUUU)
    cv_b4 = -(xyzLUU + xyzUUU)
    cv_b5 = θcv1*xU/delX - xyzLLL - xyzLUU + xyzUUL
    cv_b6 = θcv2*yU/delY - xyzLLL - xyzULU + xyzUUL

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzUL, yzLL, yzLU, yzUU, -θcv1/delX, yzLU
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzLL, xzUL, xzUU, xzLU, xzLU, -θcv2/delY
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyLU, xyUL, xyUU, xyUU, xyLL, xyLL

    cv1 = cv_ax1*x.cc + cv_ay1*y.cv + cv_az1*z.cc + cv_b1
    cv2 = cv_ax2*x.cv + cv_ay2*y.cc + cv_az2*z.cc + cv_b2
    cv3 = cv_ax3*x.cv + cv_ay3*y.cc + cv_az3*z.cv + cv_b3
    cv4 = cv_ax4*x.cc + cv_ay4*y.cv + cv_az4*z.cv + cv_b4
    cv5 = cv_ax5*ifelse(cv_ax6 > 0.0, x.cv, -x.cc) + cv_ay5*y.cv + cv_az5*z.cv + cv_b5
    cv6 = cv_ax6*x.cv + cv_ay6*ifelse(cv_ay5 > 0.0, y.cv, -y.cc) + cv_az6*z.cv + cv_b6

    cv, cvind = max6(cv1, cv2, cv3, cv4, cv5, cv6)
    cvsubax = coeff6(cvind, cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6)
    cvsubay = coeff6(cvind, cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6)
    cvsubaz = coeff6(cvind, cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6)
    cv_grad = cvsubax*ifelse(cvsubax > 0.0, x.cv_grad, x.cc_grad) +
              cvsubay*ifelse(cvsubay > 0.0, y.cv_grad, y.cc_grad) +
              cvsubaz*ifelse(cvsubaz > 0.0, z.cv_grad, z.cc_grad)

    return cv, cv_grad
end
function trilinear_case7_cc(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}

    @unpack_trilinear_bnd()

    delX = xU - xL
    delY = yU - yL

    θcc1 = xyzLUL - xyzULL - xyzLUU + xyzLLU
    θcc2 = xyzUUU - xyzULL - xyzLUU + xyzLUL

    # define cc and coefficients
    cc_b1 = -2.0*xyzLLL
    cc_b2 = -2.0*xyzUUL
    cc_b3 = -(xyzLLU + xyzULU)
    cc_b4 = -(xyzUUU + xyzULU)
    cc_b5 = θcc1*xU/delX - xyzLUL - xyzLLU + xyzULL
    cc_b6 = -θcc2*yL/delY - xyzUUU - xyzLUL + xyzULL

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzLL, yzUL, yzLU, yzUU, -θcc1/delX, yzUU
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzLL, xzUL, xzLU, xzUU, xzLU, θcc2/delY
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyLL, xyUU, xyUL, xyUL, xyLU, xyLU

    cc1 = cc_ax1*x.cc + cc_ay1*y.cc + cc_az1*z.cc + cc_b1
    cc2 = cc_ax2*x.cv + cc_ay2*y.cv + cc_az2*z.cc + cc_b2
    cc3 = cc_ax3*x.cc + cc_ay3*y.cc + cc_az3*z.cv + cc_b3
    cc4 = cc_ax4*x.cv + cc_ay4*y.cv + cc_az4*z.cv + cc_b4
    cc5 = cc_ax5*ifelse(cc_ax5 > 0.0, x.cc, -x.cv) + cc_ay5*y.cc + cc_az5*z.cv + cc_b5
    cc6 = cc_ax6*x.cv + cc_ay6*ifelse(cc_ay6 > 0.0, y.cc, -y.cv) + cc_az6*z.cv + cc_b6

    cc, ccind = min6(cc1, cc2, cc3, cc4, cc5, cc6)
    ccsubax = coeff6(ccind, cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6)
    ccsubay = coeff6(ccind, cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6)
    ccsubaz = coeff6(ccind, cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6)
    cc_grad = ccsubax*ifelse(ccsubax > 0.0, x.cc_grad, x.cv_grad) +
              ccsubay*ifelse(ccsubay > 0.0, y.cc_grad, y.cv_grad) +
              ccsubaz*ifelse(ccsubaz > 0.0, z.cc_grad, z.cv_grad)

    return cc, cc_grad
end
=#

macro trilinear_case8_cc()
    esc(quote
        # define cc and coefficients
        cc_b1 = -(xyzLLU + xyzLLL)
        cc_b2 = -(xyzLUL + xyzLLL)
        cc_b3 = -(xyzLLU + xyzULU)
        cc_b4 = -(xyzUUU + xyzULU)
        cc_b5 = -(xyzUUU + xyzUUL)
        cc_b6 = -(xyzLUL + xyzUUL)

        cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzLL, yzLL, yzLU, yzUU, yzUU, yzUL
        cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzLU, xzLL, xzLU, xzUU, xzUL, xzUL
        cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyLL, xyLU, xyUL, xyUL, xyUU, xyLU

        cc1 = cc_ax1*x.cc + cc_ay1*y.cv + cc_az1*z.cv + cc_b1
        cc2 = cc_ax2*x.cc + cc_ay2*y.cv + cc_az2*z.cv + cc_b2
        cc3 = cc_ax3*x.cc + cc_ay3*y.cv + cc_az3*z.cv + cc_b3
        cc4 = cc_ax4*x.cc + cc_ay4*y.cv + cc_az4*z.cv + cc_b4
        cc5 = cc_ax5*x.cc + cc_ay5*y.cv + cc_az5*z.cv + cc_b5
        cc6 = cc_ax6*x.cc + cc_ay6*y.cv + cc_az6*z.cv + cc_b6
    end)
end

"""
trilinear_case_8

Case 3.8 + Case 4.8 of Meyer-Floudas 2004
"""
function trilinear_case_8a(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()

    delX = xU - xL
    θcv1 = xyzLLU - xyzUUU - xyzLLL + xyzLUL
    θcv2 = xyzULU - xyzLLL - xyzUUU + xyzUUL

    cv_b1 = -2.0*xyzLUU
    cv_b2 = -2.0*xyzULL
    cv_b3 = -(xyzULU + xyzLLU)
    cv_b4 = -(xyzUUL + xyzLUL)
    cv_b5 = θcv1*xU/delX - xyzLLU - xyzLUL + xyzUUU
    cv_b6 = -θcv2*xL/delX - xyzULU - xyzUUL + xyzLLL

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzUU, yzLL, yzLU, yzUL, -θcv1/delX, θcv2/delX
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzLU, xzUL, xzUU, xzLL, xzLL, xzUU
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyLU, xyUL, xyLL, xyUU, xyLL, xyUU

    cv1 = cv_ax1*x.cv + cv_ay1*y.cc + cv_az1*z.cc + cv_b1
    cv2 = cv_ax2*x.cv + cv_ay2*y.cc + cv_az2*z.cc + cv_b2
    cv3 = cv_ax3*x.cv + cv_ay3*y.cc + cv_az3*z.cc + cv_b3
    cv4 = cv_ax4*x.cv + cv_ay4*y.cc + cv_az4*z.cc + cv_b4
    cv5 = cv_ax5*ifelse(cv_ax5 > 0.0, x.cv, -x.cc) + cv_ay5*y.cc + cv_az5*z.cc + cv_b5
    cv6 = cv_ax6*ifelse(cv_ax6 > 0.0, x.cv, -x.cc) + cv_ay6*y.cc + cv_az6*z.cc + cv_b6

    @trilinear_case8_cc()
    @unpack_trilinear_end()
end

function trilinear_case_8b(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()

    delY = yU - yL
    θcv1 = xyzUUU - xyzLLU - xyzUUL + xyzLUL
    θcv2 = xyzLLL - xyzUUL - xyzLLU + xyzULU

    cv_b1 = -2.0*xyzLUU
    cv_b2 = -2.0*xyzULL
    cv_b3 = -(xyzULU + xyzUUU)
    cv_b4 = -(xyzLLL + xyzLUL)
    cv_b5 = -θcv1*yL/delY - xyzUUU - xyzLUL + xyzLLU
    cv_b6 = θcv2*yU/delY - xyzLLL - xyzULU + xyzUUL

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzUU, yzLL, yzLU, yzUL, yzUL, yzLU
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzLU, xzUL, xzUU, xzLL, θcv1/delY, -θcv2/delY
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyLU, xyUL, xyUU, xyLL, xyUU, xyLL

    cv1 = cv_ax1*x.cv + cv_ay1*y.cc + cv_az1*z.cc + cv_b1
    cv2 = cv_ax2*x.cv + cv_ay2*y.cc + cv_az2*z.cc + cv_b2
    cv3 = cv_ax3*x.cv + cv_ay3*y.cc + cv_az3*z.cc + cv_b3
    cv4 = cv_ax4*x.cv + cv_ay4*y.cc + cv_az4*z.cc + cv_b4
    cv5 = cv_ax5*x.cv + cv_ay5*ifelse(cv_ay5 > 0.0, y.cv, -y.cc) + cv_az5*z.cc + cv_b5
    cv6 = cv_ax6*x.cv + cv_ay6*ifelse(cv_az6 > 0.0, y.cv, -y.cc) + cv_az6*z.cc + cv_b6

    @trilinear_case8_cc()
    @unpack_trilinear_end()
end

"""
trilinear_case_9

Case 3.9 + Case 4.9 of Meyer-Floudas 2004
"""
function trilinear_case_9(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()

    delX = xU - xL
    θcv = xyzUUL - xyzLLL - xyzUUU + xyzULU
    θcc = xyzLUL - xyzULL - xyzLUU + xyzLLU

    # define cv and coefficients
    cv_b1 = -2.0*xyzULL
    cv_b2 = -(xyzLUU + xyzLUL)
    cv_b3 = -(xyzLUU + xyzLLU)
    cv_b4 = -(xyzULU + xyzLLU)
    cv_b5 = -(xyzLUL + xyzUUL)
    cv_b6 = -θcv*xL/delX - xyzUUL - xyzULU + xyzLLL

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzLL, yzUU, yzUU, yzLU, yzUL, θcv/delX
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzUL, xzLL, xzLU, xzUU, xzLL, xzUU
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyUL, xyLU, xyLL, xyLL, xyUU, xyUU

    # define cc and coefficients
    cc_b1 = -2.0*xyzLLL
    cc_b2 = -(xyzLLU + xyzULU)
    cc_b3 = -(xyzUUU + xyzULU)
    cc_b4 = -(xyzUUU + xyzUUL)
    cc_b5 = -(xyzLUL + xyzUUL)
    cc_b6 = θcc*xU/delX - xyzLUL - xyzLLU + xyzULL

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzLL, yzLU, yzUU, yzUU, yzUL, -θcc/delX
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzLL, xzLU, xzUU, xzUL, xzUL, xzLU
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyLL, xyUL, xyUL, xyUU, xyLU, xyLU

    cv1 = cv_ax1*x.cv + cv_ay1*y.cc + cv_az1*z.cc + cv_b1
    cv2 = cv_ax2*x.cv + cv_ay2*y.cv + cv_az2*z.cv + cv_b2
    cv3 = cv_ax3*x.cv + cv_ay3*y.cv + cv_az3*z.cv + cv_b3
    cv4 = cv_ax4*x.cv + cv_ay4*y.cc + cv_az4*z.cv + cv_b4
    cv5 = cv_ax5*x.cv + cv_ay5*y.cv + cv_az5*z.cc + cv_b5
    cv6 = cv_ax6*ifelse(cv_ax6 > 0.0, x.cv, -x.cc) + cv_ay6*y.cc + cv_az6*z.cc + cv_b6

    cc1 = cc_ax1*x.cc + cc_ay1*y.cc + cc_az1*z.cc + cc_b1
    cc2 = cc_ax2*x.cc + cc_ay2*y.cc + cc_az2*z.cv + cc_b2
    cc3 = cc_ax3*x.cc + cc_ay3*y.cv + cc_az3*z.cv + cc_b3
    cc4 = cc_ax4*x.cc + cc_ay4*y.cv + cc_az4*z.cv + cc_b4
    cc5 = cc_ax5*x.cc + cc_ay5*y.cv + cc_az5*z.cc + cc_b5
    cc6 = cc_ax6*ifelse(cc_ax6 > 0.0, x.cc, -x.cv) + cc_ay6*y.cc + cc_az6*z.cc + cc_b6

    @unpack_trilinear_end()
end

#=
"""
trilinear_case_10

Case 3.10 + Case 4.10 of Meyer-Floudas 2004
"""
function trilinear_case_10(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()

    delX = xU - xL

    θcc1 = xyzLUL - xyzULL - xyzLUU + xyzLLU
    θcc2 = xyzUUL - xyzLUU - xyzULL + xyzULU

    # define cv and coefficients
    cv_b1 = -(xyzUUL + xyzULL)
    cv_b2 = -(xyzLUU + xyzLUL)
    cv_b3 = -(xyzUUL + xyzLUL)
    cv_b4 = -(xyzLUU + xyzLLU)
    cv_b5 = -(xyzULU + xyzLLU)
    cv_b6 = -(xyzULU + xyzULL)

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzLL, yzUU, yzUL, yzUU, yzLU, yzLL
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzUL, xzLL, xzLL, xzLU, xzUU, xzUU
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyUU, xyLU, xyUU, xyLL, xyLL, xyUL

    # define cc and coefficients
    cc_b1 = -2.0*xyzLLL
    cc_b2 = -2.0*xyzUUU
    cc_b3 = -(xyzLLU + xyzULU)
    cc_b4 = -(xyzLUL + xyzUUL)
    cc_b5 = -θcc1*xU/delX - xyzLUL - xyzLLU + xyzULL
    cc_b6 = θcc2*xL/delX - xyzUUL - xyzULU + xyzLUU

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzLL, yzUU, yzLU, yzUL, -θcc1*xU/delX, θcc2*xL/delX
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzLL, xzUU, xzLU, xzUL, xzLU, xzUL
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyLL, xyUU, xyUL, xyLU, xyLU, xyUL

    cv1 = cv_ax1*x.cv + cv_ay1*y.cv + cv_az1*z.cv + cv_b1
    cv2 = cv_ax2*x.cv + cv_ay2*y.cv + cv_az2*z.cv + cv_b2
    cv3 = cv_ax3*x.cv + cv_ay3*y.cv + cv_az3*z.cv + cv_b3
    cv4 = cv_ax4*x.cv + cv_ay4*y.cv + cv_az4*z.cv + cv_b4
    cv5 = cv_ax5*x.cv + cv_ay5*y.cv + cv_az5*z.cv + cv_b5
    cv6 = cv_ax6*x.cv + cv_ay6*y.cv + cv_az6*z.cv + cv_b6

    cc1 = cc_ax1*x.cc + cc_ay1*y.cc + cc_az1*z.cc + cc_b1
    cc2 = cc_ax2*x.cc + cc_ay2*y.cc + cc_az2*z.cc + cc_b2
    cc3 = cc_ax3*x.cc + cc_ay3*y.cc + cc_az3*z.cc + cc_b3
    cc4 = cc_ax4*x.cc + cc_ay4*y.cc + cc_az4*z.cc + cc_b4
    cc5 = cc_ax5*x.cc + cc_ay5*y.cc + cc_az5*z.cc + cc_b5
    cc6 = cc_ay6*x.cc + cc_ay6*y.cc + cc_az6*z.cc + cc_b6

    @unpack_trilinear_end()
end
=#

x_mul_y2(x, y) = x*y^2

function trilinear_case1_map_chk(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N, T<:Union{NS,MV}}
    @unpack_trilinear_bnd()
    (xyzULL + xyzLUU <= xyzLUL + xyzULU) && (xyzULL + xyzLUU <= xyzUUL + xyzLLU)
end

function trilinear_case5_map_chk1(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N, T<:Union{NS,MV}}
    @unpack_trilinear_bnd()
    (xyzLLL + xyzUUU >= xyzULL + xyzLUU) &&
    (xyzLLL + xyzUUU >= xyzLUL + xyzULU)
end

function trilinear_case5_map_chk2(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N, T<:Union{NS,MV}}
    @unpack_trilinear_bnd()
    (xyzULL + xyzLUU >= xyzLUL + xyzULU)
end

function trilinear_case8_map_chk(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N, T<:Union{NS,MV}}
    @unpack_trilinear_bnd()
    (xyzLLL + xyzUUU <= xyzLLL + xyzUUU) && (xyzLLL + xyzUUU <= xyzLUL + xyzULU)
end

function mult_kernel(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N, T<:Union{NS,MV}}
	if x == y
		if x == z
			return x^3
		else
			return x_mul_y2(z, x)
		end
	elseif y == z
		return x_mul_y2(x, y)
	end

    @unpack_trilinear_bnd()
	if @is_pos(x) && @is_pos(y) && @is_pos(z)
        if trilinear_case1_map_chk(x, y, z)
            return trilinear_case_1(x, y, z, q)
        elseif trilinear_case1_map_chk(x, z, y)
            return trilinear_case_1(x, z, y, q)
        elseif trilinear_case1_map_chk(y, x, z)
            return trilinear_case_1(y, x, z, q)
        elseif trilinear_case1_map_chk(y, z, x)
            return trilinear_case_1(y, z, x, q)
        elseif trilinear_case1_map_chk(z, x, y)
            return trilinear_case_1(z, x, y, q)
        else
            return trilinear_case_1(z, y, x, q)
        end
    end

    if @is_pos(x) && @is_pos(y) && @is_mix(z)
        return trilinear_case_2(x, y, z, q)
    elseif @is_pos(x) && @is_mix(y) && @is_pos(z)
        return trilinear_case_2(x, z, y, q)
    elseif @is_mix(x) && @is_pos(y) && @is_pos(z)
        return trilinear_case_2(y, z, x, q)
    elseif @is_mix(x) && @is_mix(y) && @is_mix(z)
        return trilinear_case_4(x, y, z, q)
    end

    if (@is_pos(x) && @is_pos(y) && @is_neg(z))
        if trilinear_case5_map_chk1(x, y, z)
            return trilinear_case_5a(x, y, z, q)
        elseif trilinear_case5_map_chk1(y, x, z)
            return trilinear_case_5a(y, x, z, q)
        elseif trilinear_case5_map_chk2(x, y, z)
            return trilinear_case_5b(x, y, z, q)
        elseif trilinear_case5_map_chk2(y, x, z)
            return trilinear_case_5b(y, x, z, q)
        end
    elseif (@is_pos(x) && @is_neg(y) && @is_pos(z))
        if trilinear_case5_map_chk1(x, z, y)
            return trilinear_case_5a(x, z, y, q)
        elseif trilinear_case5_map_chk1(z, x, y)
            return trilinear_case_5a(z, x, y, q)
        elseif trilinear_case5_map_chk2(x, z, y)
            return trilinear_case_5b(x, z, y, q)
        elseif trilinear_case5_map_chk2(z, x, y)
            return trilinear_case_5b(z, x, y, q)
        end
    elseif (@is_neg(x) && @is_pos(y) && @is_pos(z))
        if trilinear_case5_map_chk1(y, z, x)
            return trilinear_case_5a(y, z, x, q)
        elseif trilinear_case5_map_chk1(z, y, x)
            return trilinear_case_5a(z, y, x, q)
        elseif trilinear_case5_map_chk2(y, z, x)
            return trilinear_case_5b(y, z, x, q)
        elseif trilinear_case5_map_chk2(z, y, x)
            return trilinear_case_5b(z, y, x, q)
        end
    end

    if @is_pos(x) && @is_mix(y) && @is_neg(z)
        return trilinear_case_6(x, y, z, q)
    elseif @is_pos(x) && @is_neg(y) && @is_mix(z)
        return trilinear_case_6(x, z, y, q)
    elseif @is_neg(x) && @is_mix(y) && @is_pos(z)
        return trilinear_case_6(z, y, x, q)
    elseif @is_neg(x) && @is_pos(y) && @is_mix(z)
        return trilinear_case_6(y, z, x, q)
    elseif @is_mix(x) && @is_pos(y) && @is_neg(z)
        return trilinear_case_6(y, x, z, q)
    elseif @is_mix(x) && @is_neg(y) && @is_pos(z)
        return trilinear_case_6(z, x, y, q)
    end

    if (@is_mix(x) && @is_mix(y) && @is_neg(z))
        return -trilinear_case_3(-z, x, y, -q)
    elseif (@is_mix(x) && @is_neg(y) && @is_mix(z))
        return -trilinear_case_3(-y, x, z, -q)
    elseif (@is_neg(x) && @is_mix(y) && @is_mix(z))
        return -trilinear_case_3(-x, y, z, -q)
    end

    if @is_pos(x) && @is_neg(y) && @is_neg(z)
        if trilinear_case8_map_chk(x, y, z)
            return trilinear_case_8a(x, y, z, q)
        else
            return trilinear_case_8b(x, z, y, q)
        end
    elseif @is_neg(x) && @is_pos(y) && @is_neg(z)
        if trilinear_case8_map_chk(y, x, z)
            return trilinear_case_8a(y, x, z, q)
        else
            return trilinear_case_8b(y, z, x, q)
        end
    elseif @is_neg(x) && @is_neg(y) && @is_pos(z)
        if trilinear_case8_map_chk(z, x, y)
            return trilinear_case_8a(z, x, y, q)
        else
            return trilinear_case_8b(z, y, x, q)
        end
    elseif @is_mix(x) && @is_neg(y) && @is_neg(z)
        return trilinear_case_9(x, y, z, q)
    elseif @is_neg(x) && @is_mix(y) && @is_neg(z)
        return trilinear_case_9(y, x, z, q)
    elseif @is_neg(x) && @is_neg(y) && @is_mix(z)
        return trilinear_case_9(z, x, y, q)
    end

    if (@is_neg(x) && @is_neg(y) && @is_neg(z))
        return -mult_kernel(-x,y,z,-q)
    end

    return trilinear_case_3(x,y,z,q)
end

@inline function trilinear(x1::MC{N,T}, x2::MC{N,T}, x3::MC{N,T}) where {N, T<:Union{NS,MV}}
	#if x1 == x2
	#	if x1 == x3
	#		z = x1.Intv^3
	#	else
	#		z = x_mul_y2(x3.Intv, x1.Intv)
	#	end
	#elseif x2 == x3
	#	z = x_mul_y2(x1.Intv, x2.Intv)
	#else
	#	z = x1.Intv*x2.Intv*x3.Intv
	#end
    z = x1.Intv*x2.Intv*x3.Intv
	return mult_kernel(x1, x2, x3, z)
end

#=
*(a::MC{N,T}, b::MC{N,T}, c::MC{N,T}, d::MC{N,T}) where {N, T<:Union{NS,MV}} = *(*(a,b,c),d)

*(a::MC{N,T}, b::MC{N,T}, c::MC{N,T}, d::Float64) where {N, T<:Union{NS,MV}} = *(*(a,b,c),d)
*(a::MC{N,T}, b::MC{N,T}, c::Float64, d::MC{N,T}) where {N, T<:Union{NS,MV}} = *(*(a,b,d),c)
*(a::MC{N,T}, b::Float64, c::MC{N,T}, d::MC{N,T}) where {N, T<:Union{NS,MV}} = *(*(a,c,d),b)
*(a::Float64, b::MC{N,T}, c::MC{N,T}, d::MC{N,T}) where {N, T<:Union{NS,MV}} = *(*(b,c,d),a)

*(a::MC{N,T}, b::MC{N,T}, c::MC{N,T}, d::MC{N,T}, e::MC{N,T}) where {N, T<:Union{NS,MV}} = *(*(a,b,c),d,e)

*(a::MC{N,T}, b::MC{N,T}, c::MC{N,T}, d::MC{N,T}, e::Float64) where {N, T<:Union{NS,MV}} = *(a,b,c,d)*e
*(a::MC{N,T}, b::MC{N,T}, c::MC{N,T}, d::Float64, e::MC{N,T}) where {N, T<:Union{NS,MV}} = *(a,b,c,e)*d
*(a::MC{N,T}, b::MC{N,T}, c::Float64, d::MC{N,T}, e::MC{N,T}) where {N, T<:Union{NS,MV}} = *(a,b,d,e)*c
*(a::MC{N,T}, b::Float64, c::MC{N,T}, d::MC{N,T}, e::MC{N,T}) where {N, T<:Union{NS,MV}} = *(a,c,d,e)*b
*(a::Float64, b::MC{N,T}, c::MC{N,T}, d::MC{N,T}, e::MC{N,T}) where {N, T<:Union{NS,MV}} = *(b,c,d,e)*a

*(a::Float64, b::Float64, c::MC{N,T}, d::MC{N,T}, e::MC{N,T}) where {N, T<:Union{NS,MV}} = *(c,d,e)*a*b
*(a::Float64, b::MC{N,T}, c::Float64, d::MC{N,T}, e::MC{N,T}) where {N, T<:Union{NS,MV}} = *(b,d,e)*a*c
*(a::Float64, b::MC{N,T}, c::MC{N,T}, d::Float64, e::MC{N,T}) where {N, T<:Union{NS,MV}} = *(b,c,e)*a*d
*(a::Float64, b::MC{N,T}, c::MC{N,T}, d::MC{N,T}, e::Float64) where {N, T<:Union{NS,MV}} = *(b,c,d)*a*e

*(a::MC{N,T}, b::Float64, c::Float64, d::MC{N,T}, e::MC{N,T}) where {N, T<:Union{NS,MV}} = *(a,d,e)*b*c
*(a::MC{N,T}, b::Float64, c::MC{N,T}, d::Float64, e::MC{N,T}) where {N, T<:Union{NS,MV}} = *(a,c,e)*b*d
*(a::MC{N,T}, b::Float64, c::MC{N,T}, d::MC{N,T}, e::Float64) where {N, T<:Union{NS,MV}} = *(a,c,d)*b*e

*(a::MC{N,T}, b::MC{N,T}, c::Float64, d::Float64, e::MC{N,T}) where {N, T<:Union{NS,MV}} = *(a,b,e)*c*d
*(a::MC{N,T}, b::MC{N,T}, c::Float64, d::MC{N,T}, e::Float64) where {N, T<:Union{NS,MV}} = *(a,b,d)*c*e
*(a::MC{N,T}, b::MC{N,T}, c::MC{N,T}, d::Float64, e::Float64) where {N, T<:Union{NS,MV}} = *(a,b,c)*d*e

function *(a::MC{N,T}, b::MC{N,T}, c::MC{N,T}, d::MC{N,T}, e::MC{N,T}, f::MC{N,T}) where {N, T<:Union{NS,MV}}
    *(*(a,b,c),*(d,e,f))
end
=#

#=
function *(a,b,c,d,e,f,gs...)
    y = *(a,b,c,d,e,f)
    n = length(js)
    t1 = a
    t2 = b
    t3 = c
    for k = 1:n
        y = @inbounds x[k]
        !is_last
        if mod(k)
        end
    end
    for x in js;
        y = op(y,x);
    end
    y
end
=#

#=
function prod(A::Array{MC{N,T},Q}; dims) where {N, Q, T<:Union{NS,MV}}
end
=#
