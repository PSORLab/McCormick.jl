@inline function cos_kernel(::Union{NS,MV}, x::MCNoGrad, z::Interval{Float64},
                            cv_tp1::Float64, cv_tp2::Float64, cc_tp1::Float64, cc_tp2::Float64)
    xL = x.Intv.lo;    xU = x.Intv.hi;    zL = z.lo;    zU = z.hi
    eps_min, eps_max = cos_arg(xL, xU)
    mcc, cci = mid3(x.cc, x.cv, eps_max)
    mcv, cvi = mid3(x.cc, x.cv, eps_min)
    cc, dcc, cc_tp1, cc_tp2 = cc_cos(mcc, xL, xU, cc_tp1, cc_tp2)
    cv, dcv, cv_tp1, cv_tp2 = cv_cos(mcv, xL, xU, cv_tp1, cv_tp2)
    if zL > zcv
        cvi = 3
        dcv = 0.0
    end
    if zU < zcc
        cci = 3
        dcc = 0.0
    end
    return MCNoGrad(cv, cc, y, x.cnst), cvi, cci, dcv, dcc, cv_tp1, cc_tp1, cv_tp1, cc_tp2
end
@inline function cos(::ANYRELAX, x::MCNoGrad)
    y, _, _, _, _, _, _, _, _, = cos_kernel(x, cos(x.Intv), Inf, Inf, Inf, Inf)
    return y
end
function sin_kernel(::ANYRELAX, x::MCNoGrad, y::Interval{Float64},
                    cv_tp1::Float64, cv_tp2::Float64,
                    cc_tp1::Float64, cc_tp2::Float64)
    cos_kernel(x-pi/2.0, y, cv_tp1, cv_tp2, cc_tp1, cc_tp2)
end
function sin(::ANYRELAX, x::MCNoGrad)
    y, _, _, _, _ = sin_kernel(x, sin(x.Intv), Inf, Inf, Inf, Inf)
    return y
end
