#!/usr/bin/env julia

using Test, McCormick
using ForwardDiff: Dual, Partials

# Create functions for comparing MC object to reference object and detail failures
function chk_ref_kernel(y::MC{N,T}, yref::MC{N,T}, mctol::Float64) where {N, T<:RelaxTag}
    pass_flag::Bool = true
    descr = "Failing Components: ("
    ~isapprox(y.cv, yref.cv; atol = mctol) && (descr = descr*" cv = $(y.cv)"; pass_flag = false)
    ~isapprox(y.cc, yref.cc; atol = mctol) && (descr = descr*" cc = $(y.cc) "; pass_flag = false)
    ~isapprox(y.Intv.bareinterval.lo, yref.Intv.bareinterval.lo; atol = mctol) && (descr = descr*" Intv.bareinterval.lo = $(y.Intv.bareinterval.lo) "; pass_flag = false)
    ~isapprox(y.Intv.bareinterval.hi, yref.Intv.bareinterval.hi; atol = mctol) && (descr = descr*" Intv.bareinterval.hi = $(y.Intv.bareinterval.hi) "; pass_flag = false)
    ~isapprox(y.cv_grad[1], yref.cv_grad[1]; atol = mctol) && (descr = descr*" cv_grad[1] = $(y.cv_grad[1]) "; pass_flag = false)
    ~isapprox(y.cv_grad[2], yref.cv_grad[2]; atol = mctol) && (descr = descr*" cv_grad[2] = $(y.cv_grad[2]) "; pass_flag = false)
    ~isapprox(y.cc_grad[1], yref.cc_grad[1]; atol = mctol) && (descr = descr*" cc_grad[1] = $(y.cc_grad[1]) "; pass_flag = false)
    ~isapprox(y.cc_grad[2], yref.cc_grad[2]; atol = mctol) && (descr = descr*" cc_grad[2] = $(y.cc_grad[2]) "; pass_flag = false)
    (descr !== "Failing Components: (") && println(descr*")")
    pass_flag
end
check_vs_ref1(f::Function, x::MC, yref::MC, mctol::Float64) = chk_ref_kernel(f(x), yref, mctol)
check_vs_ref2(f::Function, x::MC, y::MC, yref::MC, mctol::Float64) = chk_ref_kernel(f(x,y), yref, mctol)
check_vs_refv(f::Function, x::MC, c::Float64, yref::MC, mctol::Float64) = chk_ref_kernel(f(x, c), yref, mctol)
check_vs_refv(f::Function, x::MC, c::Int, yref::MC, mctol::Float64) = chk_ref_kernel(f(x, c), yref, mctol)
function check_vs_ref2(f::Function, x::MC, c::Float64, yref1::MC, yref2::MC, mctol::Float64)
    pass_flag = check_vs_ref_kernel(f(x, c), yref1, mctol)
    check_vs_ref_kernel(f(c, x), yref2, mctol) && pass_flag
end

include("forward_mccormick.jl")
include("implicit_mccormick.jl")
