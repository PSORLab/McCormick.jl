# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A McCormick operator library in Julia
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# src/McCormick.jl
# Main file for McCormick.jl. Contains constructors and helper functions.
#############################################################################

__precompile__()

module McCormick

using DocStringExtensions, LinearAlgebra
using DiffRules: diffrule
using StaticArrays: @SVector, SVector, zeros, ones
using ForwardDiff: Dual, Partials
using NaNMath, SpecialFunctions

import Base: +, -, *, /, convert, in, isempty, one, zero, real, eps, max, min,
             abs, inv, exp, exp2, exp10, expm1, log, log2, log10, log1p, acosh,
             sech, csch, coth, acsch, acoth, asech, rad2deg, deg2rad, sqrt, sin,
             cos, tan, min, max, sec, csc, cot, ^, step, sign, intersect,
             promote_rule, asinh, atanh, tanh, atan, asin, cosh, acos,
             sind, cosd, tand, asind, acosd, atand,
             secd, cscd, cotd, asecd, acscd, acotd, isone, isnan, isfinite, empty,
             <, <=, ==, fma, cbrt, sinpi, cospi, union

using IntervalArithmetic
using IntervalArithmetic: @round
if isdefined(IntervalArithmetic, :big53)
    big_val(x) = IntervalArithmetic.big53(x)
else
    big_val(x) = IntervalArithmetic.bigequiv(x)
end

using IntervalRootFinding
import IntervalArithmetic: dist, mid, pow, +, -, *, /, convert, in, isempty,
                           one, zero, real, eps, max, min, abs, exp,
                           expm1, log, log2, log10, log1p, sqrt, ^,
                           sin, cos, tan, min, max, sec, csc, cot, step, sech,
                           csch, coth, acsch, acoth, asech,
                           sign, dist, mid, pow, Interval, interval, sinh, cosh,
                           ∩, IntervalBox, bisect, isdisjoint, length,
                           atan, asin, acos, AbstractInterval, atomic,
                           sind, cosd, tand, asind, acosd, atand,
                           secd, cscd, cotd, asecd, acscd, acotd, half_pi,
                           setrounding, diam, isthin, abs2

if ~(VERSION < v"1.1-")
    import SpecialFunctions: erf, erfc, erfinv, erfcinv
    export erf, erfinv, erfc, erfcinv, erf_kernel,
           erfinv_kernel, erfc_kernel, erfcinv_kernel
end

import Base.MathConstants.golden

# Export forward operators
export MC, MCNoGrad, cc, cv, Intv, lo, hi,  cc_grad, cv_grad, cnst, +, -, *, /, convert,
       one, zero, dist, real, eps, mid, exp, exp2, exp10, expm1, log, log2,
       log10, log1p, acosh, sqrt, sin, cos, tan, min, max, sec, csc, cot, ^,
       abs, step, sign, pow, in, isempty, intersect, length, mid3,
       acos, asin, atan, sinh, cosh, tanh, asinh, atanh, inv, sqr, sech,
       csch, coth, acsch, acoth, asech, rad2deg, deg2rad, diam,
       sind, cosd, tand, asind, acosd, atand, nan,
       sinhd, coshd, tanhd, asinhd, acoshd, atanhd,
       secd, cscd, cotd, asecd, acscd, acotd,
       secdh, cschd, cothd, asechd, acschd, acothd, isone, isfinite, isnan, interval_MC,
       relu, param_relu, leaky_relu, maxsig, maxtanh, softplus, pentanh,
       sigmoid, bisigmoid, softsign, gelu, elu, selu, swish1,
       positive, negative, lower_bnd, upper_bnd, bnd, xlogx,
       <, <=, ==, fma, cbrt, abs2, sinpi, cospi, arh, xexpax, trilinear,
       xabsx, logcosh

# Export kernel operators
export plus_kernel, minus_kernel, mult_kernel, div_kernel, max_kernel,
       min_kernel, log_kernel, log2_kernel, log10_kernel, log1p_kernel,
       acosh_kernel, sqrt_kernel, exp_kernel, exp2_kernel, exp10_kernel,
       expm1_kernel, div_kernel, cos_kernel, sin_kernel, sinh_kernel,
       tanh_kernel, asinh_kernel, atanh_kernel, tan_kernel, acos_kernel,
       asin_kernel, atan_kernel, cosh_kernel, deg2rad_kernel, rad2deg_kernel,
       sec_kernel, csc_kernel, cot_kernel, asec_kernel, acsc_kernel,
       acot_kernel, sech_kernel, csch_kernel, coth_kernel, acsch_kernel,
       acoth_kernel, sind_kernel, cosd_kernel, tand_kernel, secd_kernel,
       cscd_kernel, cotd_kernel, asind_kernel, acosd_kernel,
       atand_kernel, asecd_kernel, acscd_kernel, acotd_kernel, relu_kernel,
       param_relu_kernel, leaky_relu_kernel, maxsig_kernel,
       maxtanh_kernel, softplus_kernel, pentanh_kernel, sigmoid_kernel,
       bisigmoid_kernel, softsign_kernel, gelu_kernel, elu_kernel, selu_kernel,
       swish1_kernel, positive_kernel, negative_kernel, lower_bnd_kernel,
       upper_bnd_kernel, bnd_kernel, xlogx_kernel, fma_kernel, cbrt_kernel,
       abs2_kernel, sinpi_kernel, cospi_kernel, arh_kernel, xexpax_kernel,
       trilinear_kernel, xabsx_kernel, logcosh_kernel

export seed_gradient, RelaxTag, NS, MV, Diff

export MCCallback, gen_expansion_params!, implicit_relax_h!, DenseMidInv,
       NewtonGS, KrawczykCW

"""
$(TYPEDEF)

An `abstract type` the subtypes of which define the manner of relaxation that will
be performed for each operator applied to the MC object. Currently, the `struct NS`
which specifies that standard (Mitsos 2009) are to be used is fully supported. Limited
support is provided for differentiable McCormick relaxations specified by `struct Diff`
(Khan 2017) and struct MV `struct MV` (Tsoukalas 2011.) A rounding-safe implementation of
the standard McCormick relaxations is specified by the struct `NSSafe` which is work in
progress.
"""
abstract type RelaxTag end

struct NS <: RelaxTag end
struct MV <: RelaxTag end
struct Diff <: RelaxTag end
const ANYRELAX = Union{NS, MV, Diff}


const MC_ENV_MAX_INT = 100
const MC_ENV_TOL = 1E-10

const MC_INTERSECT_NOOP_FALLBACK = true
const MC_INTERSECT_TOL = 1E-13


const MC_DIFF_MU = 1
const MC_DIFF_MUT = convert(Float64, MC_DIFF_MU)
const MC_DIFF_MU1 = MC_DIFF_MU + 1
const MC_DIFF_MU1T = convert(Float64, MC_DIFF_MU1)
const MC_DIFF_MU1N = MC_DIFF_MU - 1
const MC_DIFF_DIV = MC_DIFF_MU1^(-1/MC_DIFF_MU)

const MC_MV_TOL = 1E-8
const MC_DEGEN_TOL = 1E-14
const MC_DOMAIN_TOL = 1E-10

const IntervalConstr = interval
const Half64 = Float64(0.5)
const Two64 = Float64(2.0)
const Three64 = Float64(3.0)
const EqualityTolerance = Float64(1E-12)
const DegToRadIntv = atomic(Interval{Float64}, pi)/Interval(180.0)
const one_intv = one(Interval{Float64})
const half_intv = Interval{Float64}(0.5)
const two_intv = Interval{Float64}(2.0)
const log2_intv = log(Interval{Float64}(2.0))
const log10_intv = log(Interval{Float64}(10.0))

const NumberNotRelax = Union{Bool, Float16, Float32, Signed, Unsigned, BigFloat,
                             Int8, Int64, Int32, Int16, Int128}

########### number list
int_list = [Int8,UInt8,Int16,UInt16,
            Int32,UInt32,Int64,UInt64,Int128,UInt128]
float_list = [Float16,Float32,Float64]

########### differentiable functions unitary functions
CVList = Symbol[:cosh,:exp,:exp2,:exp10]        ### function is convex
CCList = Symbol[:acosh,:log,:log2,:log10,:sqrt] ### function is concave
CCtoCVList = Symbol[:asin,:sinh,:atanh,:tan]    ### function is concave then convex
CVtoCCList = Symbol[:atan,:acos,:tanh,:asinh]   ### function is convex then concave
Template_List = union(CVList,CCList,CCtoCVList,CVtoCCList)

########### non differentiable and non-unitary functions
OtherList = Symbol[:sin,:cos,:min,:max,:abs,:step, :sign, :inv, :*, :+, :-, :/,
:promote_rule, :convert, :one, :zero, :real, :dist, :eps, :fma, :^]


NaNsqrt(x) = x < 0.0 ? NaN : Base.sqrt(x)

"""
$(TYPEDSIGNATURES)

Creates a `x::SVector{N,Float64}` object that is one at `x[j]` and zero everywhere else.
"""
function seed_gradient(j::Int64, x::Val{N}) where N
  return SVector{N,Float64}(ntuple(i -> i == j ? 1.0 : 0.0, Val{N}()))
end

"""
$(TYPEDSIGNATURES)

Calculates the middle of three numbers returning the value and the index where `x >= y`.
"""
function mid3(x::Float64, y::Float64, z::Float64)
  (((x >= y) && (y >= z)) || ((z >= y) && (y >= x))) && (return y, 2)
  (((y >= x) && (x >= z)) || ((z >= x) && (x >= y))) && (return x, 1)
  return z, 3
end

"""
$(TYPEDSIGNATURES)

Calculates the middle of three numbers (x,y,z) returning the value where `x <= y`.
"""
function mid3v(x::Float64, y::Float64, z::Float64)
    z <= x && (return x)
    y <= z && (return y)
    return z
end

"""
$(TYPEDSIGNATURES)

Takes the concave relaxation gradient 'cc_grad', the convex relaxation gradient
'cv_grad', and the index of the midpoint returned 'id' and outputs the appropriate
gradient according to McCormick relaxation rules.
"""
function mid_grad(cc_grad::SVector{N,Float64}, cv_grad::SVector{N,Float64}, id::Int64) where N
    if (id == 1)
        return cc_grad
    elseif (id == 2)
        return cv_grad
    end
    return zero(SVector{N,Float64})
end

"""
$(TYPEDSIGNATURES)

Calculates the value of the slope line segment between `(xL, f(xL))` and `(xU, f(xU))`
defaults to evaluating the derivative of the function if the interval is tight.
"""
@inline function dline_seg(f::Function, df::Function, x::Float64, xL::Float64, xU::Float64)
    delta = xU - xL
    if delta == 0.0
        return f(x), df(x)
    else
        yL = f(xL)
        yU = f(xU)
        return (yL*(xU - x) + yU*(x - xL))/delta, (yU - yL)/delta
    end
end
@inline function dline_seg(f::Function, df::Function, x::Float64, xL::Float64, xU::Float64, n::Int64)
    delta = xU - xL
    if delta == 0.0
        return f(x, n), df(x, n)
    else
        yL = f(xL, n)
        yU = f(xU, n)
        return (yL*(xU - x) + yU*(x - xL))/delta, (yU - yL)/delta
    end
end
@inline function dline_seg(f::Function, df::Function, x::Float64, xL::Float64, xU::Float64, c::Float64)
    delta = xU - xL
    if delta == 0.0
        return f(x, c), df(x, c)
    else
        yL = f(xL, c)
        yU = f(xU, c)
        return (yL*(xU - x) + yU*(x - xL))/delta, (yU - yL)/delta
    end
end

"""
$(TYPEDSIGNATURES)

Refines convex/concave relaxations `cv` and `cc` with associated subgradients
`cv_grad` and `cc_grad` by intersecting them with the interval boudns `xL`
and `xU`.
"""
function cut(xL::Float64, xU::Float64, cv::Float64, cc::Float64,
             cv_grad::SVector{N,Float64}, cc_grad::SVector{N,Float64}) where N

    if cc > xU
        cco = xU
        cc_grado = zero(SVector{N,Float64})
    else
        cco = cc
        cc_grado = cc_grad
    end
    if cv < xL
        cvo = xL
        cv_grado = zero(SVector{N,Float64})
    else
        cvo = cv
        cv_grado = cv_grad
    end
    return cvo, cco, cv_grado, cc_grado
end

lo(x::Interval{Float64}) = x.lo
hi(x::Interval{Float64}) = x.hi

function step(x::Interval{Float64})
     isempty(x) && return emptyinterval(x)
     xmin::Float64 = ((x.lo) < 0.0) ? 0.0 : 1.0
     xmax::Float64 = ((x.hi) >= 0.0) ? 1.0 : 0.0
     return Interval{Float64}(xmin,xmax)
end


#abstract type AbstractMC <: Real end
"""
$(TYPEDEF)

`MC{N, T <: RelaxTag} <: Real` is the McCormick (w/ (sub)gradient) structure which is used to overload
standard calculations. The fields are:
$(TYPEDFIELDS)
"""
struct MC{N, T <: RelaxTag} <: Real
    "Convex relaxation"
    cv::Float64
    "Concave relaxation"
    cc::Float64
    "Interval bounds"
    Intv::Interval{Float64}
    "(Sub)gradient of convex relaxation"
    cv_grad::SVector{N,Float64}
    "(Sub)gradient of concave relaxation"
    cc_grad::SVector{N,Float64}
    "Boolean indicating whether the relaxations are constant over the domain. True if bounding an interval/constant.
     False, otherwise. This may change over the course of a calculation `cnst` for `zero(x)` is `true` even if `x.cnst`
     is `false`."
    cnst::Bool
    function MC{N,T}(cv1::Float64, cc1::Float64, Intv1::Interval{Float64},
                     cv_grad1::SVector{N,Float64}, cc_grad1::SVector{N,Float64},
                     cnst1::Bool) where {N, T <: RelaxTag}
        new(cv1, cc1, Intv1, cv_grad1, cc_grad1, cnst1)
    end
end

"""
MC{N,T}(y::Interval{Float64})

Constructs McCormick relaxation with convex relaxation equal to `y.lo` and
concave relaxation equal to `y.hi`.
"""
function MC{N,T}(y::Interval{Float64}) where {N, T <: RelaxTag}
    MC{N,T}(y.lo, y.hi, y, zero(SVector{N,Float64}),
                           zero(SVector{N,Float64}), true)
end

"""
MC{N,T}(y::Float64)

Constructs McCormick relaxation with convex relaxation equal to `y` and
concave relaxation equal to `y`.
"""
MC{N,T}(y::Float64) where {N, T <: RelaxTag} = MC{N,T}(Interval{Float64}(y))
function MC{N,T}(y::Y) where {N, T <: RelaxTag, Y <: AbstractIrrational}
    MC{N,T}(Interval{Float64}(y))
end
MC{N,T}(y::Q) where {N, T <: RelaxTag, Q <: NumberNotRelax} = MC{N,T}(Interval{Float64}(y))

"""
MC{N,T}(cv::Float64, cc::Float64)

Constructs McCormick relaxation with convex relaxation equal to `cv` and
concave relaxation equal to `cc`.
"""
function MC{N,T}(cv::Float64, cc::Float64) where {N, T <: RelaxTag}
    MC{N,T}(cv, cc, Interval{Float64}(cv, cc), zero(SVector{N,Float64}),
            zero(SVector{N,Float64}), true)
end

"""
MC{N,T}(val::Float64, Intv::Interval{Float64}, i::Int64)

Constructs McCormick relaxation with convex relaxation equal to `val`,
concave relaxation equal to `val`, interval bounds of `Intv`, and a unit subgradient
with nonzero's ith dimension of length N.
"""
function MC{N,T}(val::Float64, Intv::Interval{Float64}, i::Int64) where {N, T <: RelaxTag}
    MC{N,T}(val, val, Intv, seed_gradient(i, Val{N}()), seed_gradient(i, Val{N}()), false)
end
function MC{N,T}(x::MC{N,T}) where {N, T <: RelaxTag}
    MC{N,T}(x.cv, x.cc, x.Intv, x.cv_grad, x.cc_grad, x.cnst)
end

Intv(x::MC) = x.Intv
lo(x::MC) = x.Intv.lo
hi(x::MC) = x.Intv.hi
cc(x::MC) = x.cc
cv(x::MC) = x.cv
cc_grad(x::MC) = x.cc_grad
cv_grad(x::MC) = x.cv_grad
cnst(x::MC) = x.cnst
length(x::MC) = length(x.cc_grad)

diam(x::MC) = diam(x.Intv)
isthin(x::MC) = isthin(x.Intv)

function isone(x::MC)
    flag = true
    flag &= (x.Intv.lo == 1.0)
    flag &= (x.Intv.hi == 1.0)
    flag &= x.cnst
    return flag
end

"""
$(TYPEDEF)

`MCNoGrad <: Real` is a McCormick structure without RelaxType Tag or subgradients.
This structure is used for source-code transformation approaches to constructing
McCormick relaxations. Methods definitions and calls should specify the
relaxation type used (i.e.) `+(::NS, x::MCNoGrad, y::MCNoGrad)...`. Moreover,
the kernel associated with this returns all intermediate calculations necessary
to compute subgradient information whereas the overloading calculation simply
returns the `MCNoGrad` object. For univariate calculations without
tiepoints such as we `log2(::NS, x::MCNoGrad)::MCNoGrad` whereas
`log2_kernel(::NS, x::MCNoGrad, ::Bool) = (::MCNoGrad, cv_id::Int, cc_id::Int, dcv, dcc)`.
Univariate NS functions follow convention (MCNoGrad, cv_id, cc_id, dcv, dcc,
tp1cv, tp1cc, .... tpncv, tpncc) where cv_id is the subgradient selected
(1 = cv, 2 = cc, 3 = 0), dcv and dcc are derivatives (or elements of subdifferential)
of the outside function evaluated per theorem at the point being evaluated and
tpicv, tpicc are the ith tiepoints associated with computing the envelope
of the outside function.
.
$(TYPEDFIELDS)
"""
struct MCNoGrad <: Real
    "Convex relaxation"
    cv::Float64
    "Concave relaxation"
    cc::Float64
    "Interval bounds"
    Intv::Interval{Float64}
    "Boolean indicating whether the relaxations are constant over the domain. True if bounding an interval/constant.
     False, otherwise. This may change over the course of a calculation `cnst` for `zero(x)` is `true` even if `x.cnst`
     is `false`."
    cnst::Bool
    function MCNoGrad(u::Float64, o::Float64, X::Interval{Float64}, b::Bool)
        new(u, o, X, b)
    end
end

"""
MCNoGrad(y::Interval{Float64})

Constructs McCormick relaxation with convex relaxation equal to `y.lo` and
concave relaxation equal to `y.hi`.
"""
function MCNoGrad(y::Interval{Float64})
    MCNoGrad(y.lo, y.hi, y, true)
end

"""
MCNoGrad(y::Float64)

Constructs McCormick relaxation with convex relaxation equal to `y` and
concave relaxation equal to `y`.
"""
MCNoGrad(y::Float64) = MCNoGrad(Interval{Float64}(y))
function MCNoGrad(y::Y) where Y <: AbstractIrrational
    MCNoGrad(Interval{Float64}(y))
end
MCNoGrad(y::Q) where Q <: NumberNotRelax = MCNoGrad(Interval{Float64}(y))

"""
MCNoGrad(cv::Float64, cc::Float64)

Constructs McCormick relaxation with convex relaxation equal to `cv` and
concave relaxation equal to `cc`.
"""
function MCNoGrad(cv::Float64, cc::Float64)
    MC{N,T}(cv, cc, Interval{Float64}(cv, cc), true)
end

Intv(x::MCNoGrad) = x.Intv
lo(x::MCNoGrad) = x.Intv.lo
hi(x::MCNoGrad) = x.Intv.hi
cc(x::MCNoGrad) = x.cc
cv(x::MCNoGrad) = x.cv
cnst(x::MCNoGrad) = x.cnst

diam(x::MCNoGrad) = diam(x.Intv)
isthin(x::MCNoGrad) = isthin(x.Intv)

function isone(x::MCNoGrad)
    flag = true
    flag &= (x.Intv.lo == 1.0)
    flag &= (x.Intv.hi == 1.0)
    flag &= x.cnst
    return flag
end

"""
$(TYPEDSIGNATURES)

Defines a local 1D newton method to solve for the root of `f` between the bounds
`xL` and `xU` using `x0` as a starting point. The derivative of `f` is `df`. The
inputs `envp1` and `envp2` are the envelope calculation parameters.
"""
function newton(x0::Float64, xL::Float64, xU::Float64, f::Function, df::Function,
               envp1::Float64, envp2::Float64)

    dfk = 0.0
    xk = max(xL, min(x0, xU))
    fk::Float64 = f(xk, envp1, envp2)

    for i = 1:MC_ENV_MAX_INT
        dfk = df(xk, envp1, envp2)
        if (abs(fk) < MC_ENV_TOL)
            return (xk, false)
        end
        (dfk == 0.0) && return (0.0, true)
        if (xk == xL && fk/dfk > 0.0)
            return (xk, false)
        elseif (xk == xU && fk/dfk < 0.0)
            return (xk, false)
        end
        xk = max(xL, min(xU, xk - fk/dfk))
        fk = f(xk, envp1, envp2)
    end

    (0.0, true)
end


"""
$(TYPEDSIGNATURES)

Defines a local 1D secant method to solve for the root of `f` between
the bounds `xL` and `xU` using `x0` and `x1` as a starting points. The inputs
`envp1` and `envp2` are the envelope calculation parameters.
"""
function secant(x0::Float64, x1::Float64, xL::Float64, xU::Float64, f::Function,
                envp1::Float64, envp2::Float64)

    xkm = max(xL, min(xU, x0))
    xk = max(xL, min(xU, x1))
    fkm::Float64 = f(xkm, envp1, envp2)

    for i = 1:MC_ENV_MAX_INT
        fk::Float64  = f(xk, envp1, envp2)
        Bk = (fk - fkm)/(xk - xkm)
        if (abs(fk) < MC_ENV_TOL)
            return (xk, false)
        end
        (Bk == 0.0) && return (0.0, true)
        if (xk == xL) && (fk/Bk > 0.0)
            return (xk, false)
        elseif (xk == xU) && (fk/Bk < 0.0)
            return (xk, false)
        end
        xkm = xk
        fkm = fk
        xk = max(xL, min(xU, xk - fk/Bk))
    end

    (0.0,true)
end


"""
$(TYPEDSIGNATURES)

Defines a local 1D golden section method to solve for the root of `f` between
the bounds `xL` and `xU` using `x0` as a starting point. Define iteration used
in golden section method. The inputs `envp1` and `envp2` are the envelope
calculation parameters.
"""
function golden_section(xL::Float64, xU::Float64, f::Function, envp1::Float64,
                        envp2::Float64)
  fL::Float64 = f(xL, envp1, envp2)
  fU::Float64 = f(xU, envp1, envp2)

  fL*fU > 0.0 && (return NaN)
  xm = xU - (2.0 - golden)*(xU - xL)
  fm::Float64 = f(xm,envp1,envp2)
  return golden_section_it(1, xL, fL, xm, fm, xU, fU, f, envp1, envp2)
end

"""
$(TYPEDSIGNATURES)

Define iteration used in golden section method. The inputs `fa`,`fb`, and `fc`,
are the function `f` evaluated at `a`,`b`, and `c` respectively. The inputs
`envp1` and `envp2` are the envelope calculation parameters. The value `init` is
the iteration number of the golden section method.
"""
function golden_section_it(init::Int, a::Float64, fa::Float64, b::Float64,
                           fb::Float64, c::Float64, fc::Float64, f::Function,
                           envp1::Float64, envp2::Float64)
    flag = (c - b > b - a)
    x = flag ? b + (2.0 - golden)*(c - b) : b - (2.0 - golden)*(b - a)
    itr = init
    if abs(c-a) < MC_ENV_TOL*(abs(b) + abs(x)) || (itr > MC_ENV_MAX_INT)
        return (c + a)/2.0
    end
    itr += 1
    fx::Float64 = f(x, envp1, envp2)
    if flag
        if fa*fx < 0.0
            out = golden_section_it(itr, a, fa, b, fb, x, fx, f, envp1, envp2)
        else
            out = golden_section_it(itr, b, fb, x, fx, c, fc, f, envp1, envp2)
        end
    else
        if fa*fb < 0.0
            out = golden_section_it(itr, a, fa, x, fx, b, fb, f, envp1, envp2)
        else
            out = golden_section_it(itr, x, fx, b, fb, c, fc, f, envp1, envp2)
        end
    end
    return out
end

function check_relaxation_error!(x::MC{N,T}) where {N, T<:RelaxTag}
    if x.cc < x.cv
        @error "After performing calculations, a convex relaxation greater
                than the associated concave relaxation was encountered. Please
                open an issue at https://github.com/PSORLab/McCormick.jl/issues/new
                so we may address this error in the future."
    end
    return nothing
end

include("forward_operators/forward.jl")
include("implicit_routines/implicit.jl")

using Reexport
@reexport using IntervalArithmetic
@reexport using StaticArrays
@reexport using SpecialFunctions

end
