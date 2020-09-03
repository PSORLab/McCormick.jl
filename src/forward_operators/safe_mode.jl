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
# src/forward_operators/safe_mode.jl
# UNDER DEVELOPMENT. CURRENTLY NOT USED. Implements a correctly-rounded
# version of standard McCormick arithmetic.
#############################################################################

# Need to add this to Project.toml... requires Julia 1.3+
using CRLibm
using RoundingEmulator

struct NSSafe <: RelaxTag end

# Safe addition
@inline function plus_kernel(x::MC{N,T}, y::MC{N,T}, z::Interval{Float64}) where {N, T <: Union{NSSafe}}
	cv = add_down(x.cv, y.cv)
	cc = add_up(x.cc, y.cc)
	MC{N,T}(cv, cc, z, x.cv_grad + y.cv_grad, x.cc_grad + y.cc_grad, (x.cnst && y.cnst))
end

# Safe scalar addition
@inline function plus_kernel(y::Float64, x::MC{N,T}, z::Interval{Float64}) where {N, T <: Union{NSSafe}}
	cv = add_down(x.cv, y)
	cc = add_up(x.cc, y)
	MC{N,T}(cv, cc, z, x.cv_grad, x.cc_grad, x.cnst)
end

# Safe subtraction
@inline function minus_kernel(x::MC{N,T}, y::MC{N,T}, z::Interval{Float64}) where {N, T <:  Union{NSSafe}}
	cv = sub_down(x.cv, y.cc)
	cc = sub_up(x.cc, y.cv)
	MC{N,T}(cv, cc, z, x.cv_grad - y.cc_grad, x.cc_grad - y.cv_grad, (x.cnst && y.cnst))
end

# Safe scalar subtraction
@inline function minus_kernel(x::MC{N,T}, c::Float64, z::Interval{Float64}) where {N, T <: Union{NSSafe}}
	cv = sub_down(x.cv, c)
	cc = sub_up(x.cc, c)
	MC{N,T}(cv, cc, z, x.cv_grad, x.cc_grad, x.cnst)
end
@inline function minus_kernel(c::Float64, x::MC{N,T}, z::Interval{Float64}) where {N, T <: Union{NSSafe}}
	cv = sub_down(c, x.cc)
	cc = sub_up(c, x.cv)
	MC{N,T}(cv, cc, z, -x.cc_grad, -x.cv_grad, x.cnst)
end

# Safe Scalar Multiplication
@inline function mult_kernel(x::MC{N,T}, c::Float64, z::Interval{Float64}) where {N, T <: Union{NSSafe}}
	if c >= 0.0
		cv = mul_down(c, x.cv)
		cc = mul_up(c, x.cc)
		zMC = MC{N,T}(cv, cc, z, c*x.cv_grad, c*x.cc_grad, x.cnst)
	else
		cv = mul_down(c, x.cc)
		cc = mul_up(c, x.cv)
		zMC = MC{N,T}(cc, cv, z, c*x.cc_grad, c*x.cv_grad, x.cnst)
	end
	return zMC
end

# Scalar Safe Kernel
@inline function div_kernel(x::MC{N,T}, c::Float64, z::Interval{Float64}) where {N, T <: Union{NSSafe}}
	if c >= 0.0
		cv = div_down(x.cv, c)
		cc = div_up(x.cc, c)
	else
		cc = div_down(x.cv, c)
		cv = div_up(x.cc, c)
	end
	return MC{N,T}(cc, cv, z, c*x.cc_grad, c*x.cv_grad, x.cnst)
end
