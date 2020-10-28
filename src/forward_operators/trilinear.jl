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
    (return f
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
            xyzLUL = xyLU*zL;   xzyULL = xyUL*zL
            xyzLUU = xyLU*zU;   xyzULU = xyUL*zU
            xyzUUL = xyUU*zL;   xyzUUU = xyUU*zU
        end)
end

macro unpack_trilinear_end()
    esc(quote
        cv, cvind = max6(cv1, cv2, cv3, cv4, cv5, cv6)
        cv_grad = coeff6(cvind, cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6)*x.cv_grad +
                  coeff6(cvind, cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6)*y.cv_grad +
                  coeff6(cvind, cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6)*z.cv_grad

        cc, ccind = min6(cc1, cc2, cc3, cc4, cc5, cc6)
        cc_grad = coeff6(cvind, $(cc_ax[1]), $(cc_ax[2]), $(cc_ax[3]), $(cc_ax[4]), $(cc_ax[5]), $(cc_ax[6]))*x.cc_grad +
                  coeff6(cvind, $(cc_ay[1]), $(cc_ay[2]), $(cc_ay[3]), $(cc_ay[4]), $(cc_ay[5]), $(cc_ay[6]))*y.cc_grad +
                  coeff6(cvind, $(cc_az[1]), $(cc_az[2]), $(cc_az[3]), $(cc_az[4]), $(cc_az[5]), $(cc_az[6]))*z.cc_grad

        MC{N,T}(cv, cc, z, cv_grad, cc_grad, x.cnst && y.cnst && z.cnst)
    end
end

function is_tri_case_1(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}
    (lo(x) >= 0.0) && (lo(y) >= 0.0) && (lo(z) >= 0.0)
end
function is_tri_case_2(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}
    (lo(x) >= 0.0) && (lo(y) >= 0.0) && (lo(z) <= 0.0) && (hi(z) >= 0.0)
end
function is_tri_case_3(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}
    (lo(x) >= 0.0) && (lo(y) <= 0.0) && (lo(z) <= 0.0) && (hi(y) >= 0.0) && (hi(z) >= 0.0)
end
function is_tri_case_5(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}
    (lo(x) >= 0.0) && (lo(y) >= 0.0) && (hi(z) <= 0.0)
end
function is_tri_case_6(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}
    (lo(x) >= 0.0) && (lo(y) <= 0.0) && (hi(z) <= 0.0) && (hi(y) >= 0.0)
end
function is_tri_case_9(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}
    (lo(x) <= 0.0) && (hi(x) >= 0.0) && (hi(y) <= 0.0) && (hi(z) <= 0.0)
end
function is_tri_case_10(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}) where {N,T<:RelaxTag}
    (hi(x) <= 0.0) && (hi(y) <= 0.0) && (hi(z) <= 0.0)
end


"""
trilinear_case_1

Case 3.1 + Case 4.1 of Meyer-Floudas 2004
"""
function trilinear_case_1(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
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
    cv_b6 = -θcv*zL/delZ - xyzLUU - xyzULL + xyzUUL

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzUU, yzUL, yzUL, yzLU, yzLL, yzLU
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzUU, xzLU, xzLL, xzUL, xzLL, xzLU
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyUU, xyLU, xyLL, xyUL, xyLL, θcv/delZ

    # define cc and coefficients
    cc_b1 = -2.0*xyzUUL
    cc_b2 = -(xyzULU + xyzULL)
    cc_b3 = -(xyzLUU + xyzLLU)
    cc_b4 = -(xyzLUU + xyzLUL)
    cc_b5 = -(xyzULU + xyzLLU)
    cc_b6 = θcc*zU/delZ - xyzULL - xyzLUL + xyzUUU

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzUL, yzLL, yzUU, yzUU, yzLU, yzLU
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzUL, xzUU, xzLU, xzLL, xzUU, xzLU
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyUU, xyUL, xyLL, xyLU, xyLL, θcc/delZ

    cv1 = cv_ax1*x.cv + cv_ay1*y.cv + cv_az1*z.cv + cv_b1
    cv2 = cv_ax2*x.cv + cv_ay2*y.cv - cv_az2*z.cc + cv_b2
    cv3 = cv_ax3*x.cv - cv_ay3*y.cc - cv_az3*z.cc + cv_b3
    cv4 = cv_ax4*x.cv - cv_ay4*y.cc + cv_az4*z.cv + cv_b4
    cv5 = cv_ax5*x.cv - cv_ay5*y.cc - cv_az5*z.cc + cv_b5
    cv6 = cv_ax6*x.cv + cv_ay6*y.cv + cv_az6*ifelse(cv_az6 > 0.0, z.cv, -z.cc) + cv_b6

    cc1 = cc_ax1*x.cc + cc_ay1*y.cc + cc_az1*z.cc + cc_b1
    cc2 = cc_ax2*x.cc + cc_ay2*y.cc - cc_az2*z.cv + cc_b2
    cc3 = cc_ax3*x.cc + cc_ay3*y.cc - cc_az3*z.cv + cc_b3
    cc4 = cc_ax4*x.cc + cc_ay4*y.cc + cc_az4*z.cc + cc_b4
    cc5 = cc_ax5*x.cc + cc_ay5*y.cc - cc_az5*z.cv + cc_b5
    cc6 = cc_ax6*x.cc + cc_ay6*y.cc + cc_az6*ifelse(cc_az6 > 0.0, z.cc, -z.cv) + cc_b6

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
    cv2 = -cv_ax2*x.cc + cv_ay2*y.cv + cv_az2*z.cc + cv_b2
    cv3 = -cv_ax3*x.cc - cv_ay3*y.cc + cv_az3*z.cc + cv_b3
    cv4 = cv_ax4*x.cv - cv_ay4*y.cc + cv_az4*z.cv + cv_b4
    cv5 = -cv_ax5*x.cc - cv_ay5*y.cc + cv_az5*z.cc + cv_b5
    cv6 = cv_ax6*x.cv + cv_ay6*y.cv + cv_az6*ifelse(cv_az6 > 0.0, z.cv, -z.cc) + cv_b6

    cc1 = -cc_ax1*x.cv - cc_ay1*y.cv + cc_az1*z.cc + cc_b1
    cc2 = -cc_ax2*x.cv + cc_ay2*y.cc + cc_az2*z.cv + cc_b2
    cc3 = cc_ax3*x.cc + cc_ay3*y.cc + cc_az3*z.cv + cc_b3
    cc4 = cc_ax4*x.cc - cc_ay4*y.cv + cc_az4*z.cc + cc_b4
    cc5 = cc_ax5*x.cc + cc_ay5*y.cc + cc_az5*z.cv + cc_b5
    cc6 = -cc_ax6*x.cv - cc_ay6*y.cv + cc_az6*ifelse(cc_az6 > 0.0, z.cc, -z.cv) + cc_b6

    @unpack_trilinear_end()
end

"""
trilinear_case_3

Case 3.3 + Case 4.3 of Meyer-Floudas 2004
"""
function trilinear_case_3(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()

    delY = yU - yL
    delZ = zU - zL
    θcv1 = xyzLUU - xyzUUL - xyzLLU + xyzULU
    θcv2 = xyzLUU - xyzUUL - xyzLLU + xyzULU
    #θcc = xyzULL - xyzUUU - xyzLLL + xyzLUL

    # define cv and coefficients
    cv_b1 = -2.0*xyzUUU
    cv_b2 = -2.0*xyzULL
    cv_b3 = -(xyzLUL + xyzLUU)
    cv_b4 = -(xyzLUL + xyzLLL)
    cv_b5 = θcv1*yU/delY - xyzLLL - xyzULU + xyzUUL
    cv_b6 = -θcv2*zL/delZ - xyzULU - xyzLUU + xyzUUL

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzUU, yzLL, yzUL, yzUL, yzLU, yzLU
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzUU, xzUL, xzLU, xzLL, -θcv1/delY, xzLU
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyUU, xyUL, xyLU, xyLL, xyLL, θcv2/delZ

    # define cc and coefficients
    #cc_b1 = -2.0*xyzUUL
    #cc_b2 = -(xyzULU + xyzULL)
    #cc_b3 = -(xyzLUU + xyzLLU)
    #cc_b4 = -(xyzLUU + xyzLUL)
    #cc_b5 = -(xyzULU + xyzLLU)
    #cc_b6 = θcc*zU/delZ - xyzULL - xyzLUL + xyzUUU

    #cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzUL, yzLL, yzUU, yzUU, yzLU, yzLL
    #cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzUL, xzUU, xzLU, xzLL, xzUU, xzLL
    #cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyUU, xyUL, xyLL, xyLU, xyLL, -θcc/delZ

    cv1 = cv_ax1*x.cv + cv_ay1*y.cv + cv_az1*z.cv + cv_b1
    cv2 = cv_ax2*x.cv - cv_ay2*y.cc - cv_az2*z.cc + cv_b2
    cv3 = -cv_ax3*x.cc + cv_ay3*y.cv + cv_az3*z.cv + cv_b3
    cv4 = -cv_ax4*x.cc - cv_ay4*y.cc - cv_az4*z.cc + cv_b4
    cv5 = -cv_ax5*x.cc + cv_ay5*ifelse(cv_ay5 > 0.0, y.cv, -y.cc) - cv_az5*z.cc + cv_b5
    cv6 = -cv_ax6*x.cc + cv_ay6*y.cv + cv_az6*ifelse(cv_az6 > 0.0, z.cv, -z.cc) + cv_b6

    #cc1 = -cc_ax1*x.cv - cc_ay1*y.cv + cc_az1*z.cc + cc_b1
    #cc2 = -cc_ax2*x.cv + cc_ay2*y.cc + cc_az2*z.cv + cc_b2
    #cc3 = cc_ax3*x.cc + cc_ay3*y.cc + cc_az3*z.cv + cc_b3
    #cc4 = cc_ax4*x.cc - cc_ay4*y.cv + cc_az4*z.cc + cc_b4
    #cc5 = cc_ax5*x.cc + cc_ay5*y.cc + cc_az5*z.cv + cc_b5
    #cc6 = -cc_ax6*x.cv - cc_ay6*y.cv + cc_az6*ifelse(cc_az6 > 0.0, z.cc, -z.cv) + cc_b6

    @unpack_trilinear_end()
end

"""
trilinear_case_5

Case 3.5 + Case 4.5 of Meyer-Floudas 2004
"""
function trilinear_case_5(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()

    #delZ = zU - zL
    #θcv = xyzLUU - xyzUUL - xyzLLU + xyzULU
    #θcc = xyzULL - xyzUUU - xyzLLL + xyzLUL

    # define cv and coefficients
    #cv_b1 = -2.0*xyzUUU
    #cv_b2 = -(xyzLUL + xyzLUU)
    #cv_b3 = -(xyzLUL + xyzLLL)
    #cv_b4 = -(xyzULU + xyzULL)
    #cv_b5 = -(xyzULL + xyzLLL)
    #cv_b6 = -θcv*zL/delZ - xyzLUU - xyzULU + xyzUUL

    #cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzUU, yzUL, yzUL, yzLU, yzLL, yzLU
    #cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzUU, xzLU, xzLL, xzUL, xzUL, xzLU
    #cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyUU, xyLU, xyLL, xyUL, xyLL, θcv/delZ

    # define cc and coefficients
    #cc_b1 = -2.0*xyzUUL
    #cc_b2 = -(xyzULU + xyzULL)
    #cc_b3 = -(xyzLUU + xyzLLU)
    #cc_b4 = -(xyzLUU + xyzLUL)
    #cc_b5 = -(xyzULU + xyzLLU)
    #cc_b6 = θcc*zU/delZ - xyzULL - xyzLUL + xyzUUU

    #cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzUL, yzLL, yzUU, yzUU, yzLU, yzLL
    #cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzUL, xzUU, xzLU, xzLL, xzUU, xzLL
    #cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyUU, xyUL, xyLL, xyLU, xyLL, -θcc/delZ

    #cv1 = cv_ax1*x.cv + cv_ay1*y.cv + cv_az1*z.cv + cv_b1
    #cv2 = -cv_ax2*x.cc + cv_ay2*y.cv + cv_az2*z.cc + cv_b2
    #cv3 = -cv_ax3*x.cc - cv_ay3*y.cc + cv_az3*z.cc + cv_b3
    #cv4 = cv_ax4*x.cv - cv_ay4*y.cc + cv_az4*z.cv + cv_b4
    #cv5 = -cv_ax5*x.cc - cv_ay5*y.cc + cv_az5*z.cc + cv_b5
    #cv6 = cv_ax6*x.cv + cv_ay6*y.cv + cv_az6*ifelse(cv_az6 > 0.0, z.cv, -z.cc) + cv_b6

    #cc1 = -cc_ax1*x.cv - cc_ay1*y.cv + cc_az1*z.cc + cc_b1
    #cc2 = -cc_ax2*x.cv + cc_ay2*y.cc + cc_az2*z.cv + cc_b2
    #cc3 = cc_ax3*x.cc + cc_ay3*y.cc + cc_az3*z.cv + cc_b3
    #cc4 = cc_ax4*x.cc - cc_ay4*y.cv + cc_az4*z.cc + cc_b4
    #cc5 = cc_ax5*x.cc + cc_ay5*y.cc + cc_az5*z.cv + cc_b5
    #cc6 = -cc_ax6*x.cv - cc_ay6*y.cv + cc_az6*ifelse(cc_az6 > 0.0, z.cc, -z.cv) + cc_b6

    @unpack_trilinear_end()
end

"""
trilinear_case_6

Case 3.6 + Case 4.6 of Meyer-Floudas 2004
"""
function trilinear_case_6(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()

    delY = yU - yL
    θcv = xyzULU - xyzUUL - xyzLLU + xyzLLL
    θcc = xyzLUL - xyzULL - xyzLUU + xyzUUU

    # define cv and coefficients
    cv_b1 = -2.0*xyzULL
    cv_b2 = -(xyzLUL + xyzLUU)
    cv_b3 = -(xyzLUL + xyzLLL)
    cv_b4 = -(xyzULU + xyzUUU)
    cv_b5 = -(xyzLUU + xyzUUU)
    cv_b6 = θcv*yU/delY - xyzULU - xyzLLL + xyzUUL

    cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6 = yzLL, yzUL, yzUL, yzLU, yzUU, yzLU
    cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6 = xzUL, xzLU, xzLL, xzUU, xzLU, -θcv/delY
    cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6 = xyUL, xyLU, xyLL, xyUU, xyUU, xyLL

    # define cc and coefficients
    cc_b1 = -2.0*xyzUUL
    cc_b2 = -(xyzUUU + xyzULU)
    cc_b3 = -(xyzLUL + xyzLLL)
    cc_b4 = -(xyzLLU + xyzLLL)
    cc_b5 = -(xyzLLU + xyzULU)
    cc_b6 = -θcc*yL/delY - xyzLUL - xyzUUU + xyzULL

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzUL, yzUU, yzLL, yzLL, yzLU, yzUU
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzUL, xzUU, xzLL, xzLU, xzLU, θcc/delY
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyUU, xyUL, xyLU, xyLL, xyUL, xyLU

    cv1 = cv_ax1*x.cv - cv_ay1*y.cc + cv_az1*z.cc + cv_b1
    cv2 = -cv_ax2*x.cc - cv_ay2*y.cc - cv_az2*z.cv + cv_b2
    cv3 = -cv_ax3*x.cc - cv_ay3*y.cc - cv_az3*z.cc + cv_b3
    cv4 = cv_ax4*x.cv - cv_ay4*y.cc + cv_az4*z.cv + cv_b4
    cv5 = -cv_ax5*x.cc - cv_ay5*y.cc + cv_az5*z.cv + cv_b5
    cv6 = cv_ax6*x.cv + cv_ay6*ifelse(cv_ay6 > 0.0, y.cv, -y.cc) - cv_az6*z.cc + cv_b6

    cc1 = -cc_ax1*x.cv - cc_ay1*y.cv + cc_az1*z.cc + cc_b1
    cc2 = -cc_ax2*x.cv - cc_ay2*y.cv - cc_az2*z.cv + cc_b2
    cc3 = cc_ax3*x.cc - cc_ay3*y.cv + cc_az3*z.cc + cc_b3
    cc4 = cc_ax4*x.cc - cc_ay4*y.cv - cc_az4*z.cv + cc_b4
    cc5 = cc_ax5*x.cc - cc_ay5*y.cv - cc_az5*z.cv + cc_b5
    cc6 = -cc_ax6*x.cv + cc_ay6*ifelse(cc_ay6 > 0.0, y.cc, -y.cv) + cc_az6*z.cv + cc_b6

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
    cv_b3 = -(xyzLUL + xyzLLI)
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
    cc_b6 = θcc*xU/delX - xyzULL - xyzLUU + xyzLLU

    cc_ax1, cc_ax2, cc_ax3, cc_ax4, cc_ax5, cc_ax6 = yzLL, yzLU, yzUU, yzUU, yzUL, -θcc/delX
    cc_ay1, cc_ay2, cc_ay3, cc_ay4, cc_ay5, cc_ay6 = xzLL, xzLU, xzUU, xzUL, xzUL, xzLU
    cc_az1, cc_az2, cc_az3, cc_az4, cc_az5, cc_az6 = xyLL, xyUL, xyUL, xyUU, xyLU, xyLU

    cv1 = cv_ax1*x.cv - cv_ay1*y.cc - cv_az1*z.cc + cv_b1
    cv2 = cv_ax2*x.cv + cv_ay2*y.cv + cv_az2*z.cv + cv_b2
    cv3 = cv_ax3*x.cv + cv_ay3*y.cv + cv_az3*z.cv + cv_b3
    cv4 = cv_ax4*x.cv - cv_ay4*y.cc + cv_az4*z.cv + cv_b4
    cv5 = cv_ax5*x.cv + cv_ay5*y.cv - cv_az5*z.cc + cv_b5
    cv6 = cv_ax6*ifelse(cv_ax6 > 0.0, x.cv, -x.cc) - cv_ay6*y.cc - cv_az6*z.cc + cv_b6

    cc1 = cc_ax1*x.cc + cc_ay1*y.cc + cc_az1*z.cc + cc_b1
    cc2 = cc_ax2*x.cc + cc_ay2*y.cc - cc_az2*z.cv + cc_b2
    cc3 = cc_ax3*x.cc - cc_ay3*y.cv - cc_az3*z.cv + cc_b3
    cc4 = cc_ax4*x.cc - cc_ay4*y.cv - cc_az4*z.cv + cc_b4
    cc5 = cc_ax5*x.cc - cc_ay5*y.cv + cc_az5*z.cc + cc_b5
    cc6 = cc_ax6*ifelse(cc_ax6 > 0.0, x.cc, -x.cv) + cc_ay6*y.cc + cc_az6*z.cc + cc_b6

    @unpack_trilinear_end()
end

"""
trilinear_case_10

Case 3.10 + Case 4.10 of Meyer-Floudas 2004
"""
function trilinear_case_10(x::MC{N,T}, y::MC{N,T}, z::MC{N,T}, q::Interval{Float64}) where {N,T<:RelaxTag}
    @unpack_trilinear_bnd()

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
    cc_b1 = -2.0*xyLLL
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

x_mul_y2(x, y) = x*y^2

function mult_kernel(x1::MC{N,T}, x2::MC{N,T}, x3::MC{N,T}, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
	if x1 == x2
		if x1 == x3
			return x1^3
		else
			return x_mul_y2(x3, x1)
		end
	elseif x2 == x3
		return x_mul_y2(x1, x2)
	end
	is_tri_case_1(x1, x2, x3) && return trilinear_case_1(x1, x2, x3, z)
    is_tri_case_2(x1, x2, x3) && return x1*x2*x3
	return x1*x2*x3
end

@inline function *(x1::MC{N,T}, x2::MC{N,T}, x3::MC{N,T}) where {N, T<:Union{NS,MV}}
	if x1 == x2
		if x1 == x3
			z = x1.Intv^3
		else
			z = x_mul_y2(x3.Intv, x1.Intv)
		end
	elseif x2 == x3
		z = x_mul_y2(x1.Intv, x2.Intv)
	else
		z = x1.Intv*x2.Intv*x3.Intv
	end
	return mult_kernel(x1, x2, x3, z)
end
