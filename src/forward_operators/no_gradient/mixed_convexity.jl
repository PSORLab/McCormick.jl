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


# basic method overloading operator (sinh, tanh, atanh, asinh), convexoconcave or concavoconvex
eps_min_dict = Dict{Symbol,Symbol}(:sinh => :xL, :tanh => :xL, :asinh => :xL,
                                 :atanh => :xL, :tan => :xL, :acos => :xU,
                                 :asin => :xL, :atan => :xL, :erf => :xL,
                                 :cbrt => :xL, :erfinv => :xL, :erfc => :xU)
eps_max_dict = Dict{Symbol,Symbol}(:sinh => :xU, :tanh => :xU, :asinh => :xU,
                                 :atanh => :xU, :tan => :xU, :acos => :xL,
                                 :asin => :xU, :atan => :xU, :erf => :xU,
                                 :cbrt => :xU, :erfinv => :xU, :erfc => :xL)

for expri in (:sinh, :tanh, :asinh, :atanh, :tan, :acos, :asin, :atan,
              (:(SpecialFunctions.erf), :erf), :cbrt,
              (:(SpecialFunctions.erfinv), :erfinv),
              (:(SpecialFunctions.erfc), :erfc))
    if expri isa Symbol
        expri_name = expri
        expri_sym = expri
    else
        expri_name = expri[1]
        expri_sym = expri[2]
    end
    expri_kernel = Symbol(String(expri_sym)*"_kernel")
    expri_cv = Symbol("cv_"*String(expri_sym))
    expri_cc = Symbol("cc_"*String(expri_sym))
    eps_min = eps_min_dict[expri_sym]
    eps_max = eps_max_dict[expri_sym]
    @eval @inline function ($expri_kernel)(t::Union{NS,MV}, x::MCNoGrad, y::Interval{Float64}, cv_p::Float64, cc_p::Float64)
        xL = x.Intv.lo
        xU = x.Intv.hi
        midcv, cv_id = mid3(x.cc, x.cv, $eps_min)
        midcc, cc_id = mid3(x.cc, x.cv, $eps_max)
        cv, dcv, cv_p = $(expri_cv)(midcv, xL, xU, cv_p)
        cc, dcc, cc_p = $(expri_cc)(midcc, xL, xU, cc_p)
        return MCNoGrad(cv, cc, y, x.cnst), dcv, dcc, cv_p, cc_p
    end
    @eval @inline function ($expri_kernel)(t::Diff, x::MCNoGrad, y::Interval{Float64}, cv_p::Float64, cc_p::Float64)
        xL = x.Intv.lo
        xU = x.Intv.hi
        midcv, cv_id = mid3(x.cv, x.cc, $eps_min)
        midcc, cc_id = mid3(x.cv, x.cc, $eps_max)
        cv, dcv, cv_p = $(expri_cv)(midcv, xL, xU, cv_p)
        cc, dcc, cc_p = $(expri_cc)(midcc, xL, xU, cc_p)
        gcv1, gdcv1, cv_p = $(expri_cv)(x.cv, xL, xU, cv_p)
        gcc1, gdcc1, cc_p = $(expri_cc)(x.cv, xL, xU, cc_p)
        gcv2, gdcv2, cv_p = $(expri_cv)(x.cc, xL, xU, cv_p)
        gcc2, gdcc2, cc_p = $(expri_cc)(x.cc, xL, xU, cc_p)
        c_cv1 = max(0.0, gdcv1)
        c_cv2 = min(0.0, gdcv2)
        c_cc1 = min(0.0, gdcc1)
        c_cc2 = max(0.0, gdcc2)
        return MCNoGrad(cv, cc, y, cv_grad, cc_grad, x.cnst), c_cv1, c_cv2, c_cc1, c_cc2
    end
    @eval @inline function ($expri_name)(t::ANYRELAX, x::MCNoGrad)
        z, tp1, tp2 = ($expri_kernel)(t, x, ($expri_name)(x.Intv), Inf, Inf)
        return z
    end
end