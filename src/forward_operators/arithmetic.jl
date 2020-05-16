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
# src/forward_operators/arithmetic.jl
# Contains definitions of +, -, /, *, promotions, conversion, one, zero.
#############################################################################

# Defines functions required for linear algebra packages
@inline nan(::Type{MC{N,T}}) where {N, T <: RelaxTag} = MC{N,T}(NaN, NaN, Interval{Float64}(NaN), fill(NaN, SVector{N,Float64}), fill(NaN, SVector{N,Float64}), true)
@inline nan(x::MC{N,T}) where {N, T <: RelaxTag} = MC{N,T}(NaN, NaN, Interval{Float64}(NaN), fill(NaN, SVector{N,Float64}), fill(NaN, SVector{N,Float64}), true)

@inline one(::Type{MC{N,T}}) where {N, T <: RelaxTag} = MC{N,T}(1.0, 1.0, one(Interval{Float64}), zero(SVector{N,Float64}), zero(SVector{N,Float64}), true)
@inline one(x::MC{N,T}) where {N, T <: RelaxTag} = MC{N,T}(1.0, 1.0, one(Interval{Float64}), zero(SVector{N,Float64}), zero(SVector{N,Float64}), true)

@inline zero(::Type{MC{N,T}}) where {N, T <: RelaxTag} = MC{N,T}(0.0, 0.0, zero(Interval{Float64}), zero(SVector{N,Float64}), zero(SVector{N,Float64}), true)
@inline zero(x::MC{N,T}) where {N, T <: RelaxTag} = zero(MC{N,T})

@inline real(x::MC) = x
@inline dist(x1::MC, x2::MC) = max(abs(x1.cc - x2.cc), abs(x1.cv - x2.cv))
@inline eps(x::MC) = max(eps(x.cc), eps(x.cv))
@inline mid(x::MC) = mid(x.Intv)

@inline function plus_kernel(x::MC{N,T}, y::MC{N,T}, z::Interval{Float64}) where {N, T <: RelaxTag}
	MC{N,T}(x.cv + y.cv, x.cc + y.cc, z, x.cv_grad + y.cv_grad, x.cc_grad + y.cc_grad, (x.cnst && y.cnst))
end
@inline +(x::MC, y::MC) = plus_kernel(x, y, x.Intv + y.Intv)
@inline plus_kernel(x::MC, y::Interval{Float64}) = x
@inline +(x::MC) = x

@inline minus_kernel(x::MC{N,T}, z::Interval{Float64}) where {N, T <: RelaxTag} = MC{N,T}(-x.cc, -x.cv, z, -x.cc_grad, -x.cv_grad, x.cnst)
@inline -(x::MC) = minus_kernel(x, -x.Intv)

@inline function minus_kernel(x::MC{N,T}, y::MC{N,T}, z::Interval{Float64}) where {N, T <: RelaxTag}
	MC{N,T}(x.cv - y.cc, x.cc - y.cv, z, x.cv_grad - y.cc_grad, x.cc_grad - y.cv_grad, (x.cnst && y.cnst))
end
@inline -(x::MC{N,T}, y::MC{N,T}) where {N, T <: RelaxTag} = minus_kernel(x, y, x.Intv - y.Intv)

################## CONVERT THROUGH BINARY DEFINITIONS #########################
# Addition
@inline function plus_kernel(x::MC{N,T}, y::Float64, z::Interval{Float64}) where {N, T <: RelaxTag}
	MC{N,T}(x.cv + y, x.cc + y, z, x.cv_grad, x.cc_grad, x.cnst)
end
@inline +(x::MC, y::Float64) = plus_kernel(x, y, x.Intv + y)
@inline +(y::Float64, x::MC) = plus_kernel(x, y, x.Intv + y)
@inline +(x::MC{N,T}, y::Interval{Float64}) where {N, T<:RelaxTag} = x + MC{N,T}(y)
@inline +(y::Interval{Float64}, x::MC{N,T}) where {N, T<:RelaxTag} = x + MC{N,T}(y)

@inline plus_kernel(x::MC, y::C, z::Interval{Float64}) where {C <: NumberNotRelax} = plus_kernel(x, convert(Float64, y), z)
@inline plus_kernel(x::C, y::MC, z::Interval{Float64}) where {C <: NumberNotRelax} = plus_kernel(y, convert(Float64, x), z)
@inline +(x::MC, y::C) where {C <: NumberNotRelax} = x + convert(Float64, y)
@inline +(y::C, x::MC) where {C <: NumberNotRelax} = x + convert(Float64, y)

# Subtraction
@inline function minus_kernel(x::MC{N,T}, c::Float64, z::Interval{Float64}) where {N, T <: RelaxTag}
	MC{N,T}(x.cv - c, x.cc - c, z, x.cv_grad, x.cc_grad, x.cnst)
end
@inline function minus_kernel(c::Float64, x::MC{N,T}, z::Interval{Float64}) where {N, T <: RelaxTag}
	MC{N,T}(c - x.cc, c - x.cv, z, -x.cc_grad, -x.cv_grad, x.cnst)
end
@inline -(x::MC, c::Float64) = minus_kernel(x, c, x.Intv - c)
@inline -(c::Float64, x::MC) = minus_kernel(c, x, c - x.Intv)
@inline -(x::MC{N,T}, y::Interval{Float64}) where {N, T<:RelaxTag} = x - MC{N,T}(y)
@inline -(y::Interval{Float64}, x::MC{N,T}) where {N, T<:RelaxTag} = MC{N,T}(y) - x

@inline minus_kernel(x::MC, y::C, z::Interval{Float64}) where {C <: NumberNotRelax} = minus_kernel(x, convert(Float64, y), z)
@inline minus_kernel(y::C, x::MC, z::Interval{Float64}) where {C <: NumberNotRelax} = minus_kernel(convert(Float64, y), x, z)
@inline -(x::MC, c::C) where {C <: NumberNotRelax} = x - convert(Float64,c)
@inline -(c::C, x::MC) where {C <: NumberNotRelax} = convert(Float64,c) - x

# Multiplication
@inline function mult_kernel(x::MC{N,T}, c::Float64, z::Interval{Float64}) where {N, T <: RelaxTag}
	if (c >= 0.0)
		zMC = MC{N,T}(c*x.cv, c*x.cc, z, c*x.cv_grad, c*x.cc_grad, x.cnst)
	else
		zMC = MC{N,T}(c*x.cc, c*x.cv, z, c*x.cc_grad, c*x.cv_grad, x.cnst)
	end
	return zMC
end
@inline *(x::MC, c::Float64) = mult_kernel(x, c, c*x.Intv)
@inline *(c::Float64, x::MC) = mult_kernel(x, c, c*x.Intv)
@inline *(x::MC{N,T}, y::Interval{Float64}) where {N, T<:RelaxTag} = x*MC{N,T}(y)
@inline *(y::Interval{Float64}, x::MC{N,T}) where {N, T<:RelaxTag} = MC{N,T}(y)*x

@inline mult_kernel(x::MC, c::C, z::Interval{Float64}) where {C <: NumberNotRelax} = mult_kernel(x, convert(Float64, c), z)
@inline mult_kernel(c::C, x::MC, z::Interval{Float64}) where {C <: NumberNotRelax} =  mult_kernel(x, convert(Float64, c), z)
@inline *(c::C, x::MC) where {C <: NumberNotRelax}  = x*Float64(c)
@inline *(x::MC, c::C) where {C <: NumberNotRelax}  = x*Float64(c)

# Division
@inline div_kernel(x::MC, y::Float64, z::Interval{Float64}) = mult_kernel(x, inv(y), z)
@inline div_kernel(x::Float64, y::MC, z::Interval{Float64}) = mult_kernel(inv(y), x, z)
@inline div_kernel(x::MC, y::C, z::Interval{Float64}) where {C <: NumberNotRelax}  = mult_kernel(x, inv(y), z)
@inline div_kernel(x::C, y::MC, z::Interval{Float64}) where {C <: NumberNotRelax}  = mult_kernel(inv(y), x, z)
@inline /(x::MC, y::Float64) = x*inv(y)
@inline /(x::Float64, y::MC) = x*inv(y)
@inline /(x::MC, y::C) where {C <: NumberNotRelax} = x*inv(convert(Float64,y))
@inline /(x::C, y::MC) where {C <: NumberNotRelax} = convert(Float64,x)*inv(y)
@inline /(x::MC{N,T}, y::Interval{Float64}) where {N, T<:RelaxTag} = x/MC{N,T}(y)
@inline /(y::Interval{Float64}, x::MC{N,T}) where {N, T<:RelaxTag} = MC{N,T}(y)/x

# Maximization
@inline max_kernel(c::Float64, x::MC, z::Interval{Float64}) = max_kernel(x, c, z)
@inline max_kernel(x::MC, c::C, z::Interval{Float64}) where {C <: NumberNotRelax} = max_kernel(x, convert(Float64, c), z)
@inline max_kernel(c::C, x::MC, z::Interval{Float64}) where {C <: NumberNotRelax} = max_kernel(x, convert(Float64, c), z)

@inline max(c::Float64, x::MC) = max_kernel(x, c, max(x.Intv, c))
@inline max(x::MC, c::C) where {C <: NumberNotRelax} = max_kernel(x, convert(Float64, c), max(x.Intv, c))
@inline max(c::C, x::MC) where {C <: NumberNotRelax} = max_kernel(x, convert(Float64, c), max(x.Intv, c))
@inline max(x::MC{N,T}, y::Interval{Float64}) where {N, T<:RelaxTag} = max(x, MC{N,T}(y))
@inline max(y::Interval{Float64}, x::MC{N,T}) where {N, T<:RelaxTag} = max(MC{N,T}(y), x)

# Minimization
@inline min_kernel(x::MC, c::C, z::Interval{Float64}) where {C <: NumberNotRelax} = min_kernel(x, convert(Float64, c), z)
@inline min_kernel(c::C, x::MC, z::Interval{Float64}) where {C <: NumberNotRelax} = min_kernel(x, convert(Float64, c), z)

@inline min(c::Float64, x::MC) = min_kernel(x, c, min(x.Intv, c))
@inline min(x::MC, c::C) where {C <: NumberNotRelax} = min_kernel(x, convert(Float64, c), min(x.Intv, c))
@inline min(c::C, x::MC) where {C <: NumberNotRelax} = min_kernel(x, convert(Float64, c), min(x.Intv, c))
@inline min(x::MC{N,T}, y::Interval{Float64}) where {N, T<:RelaxTag} = min(x, MC{N,T}(y))
@inline min(y::Interval{Float64}, x::MC{N,T}) where {N, T<:RelaxTag} = min(MC{N,T}(y), x)

# Promote and Convert
promote_rule(::Type{MC{N,T}}, ::Type{S}) where {S<:NumberNotRelax, N, T <: RelaxTag} = MC{N,T}
promote_rule(::Type{MC{N,T}}, ::Type{S}) where {S<:Real, N, T <: RelaxTag} = MC{N,T}

convert(::Type{MC{N,T}}, x::S) where {S<:NumberNotRelax, N, T <: RelaxTag} = MC{N,T}(Interval{Float64}(x))
convert(::Type{MC{N,T}}, x::S) where {S<:AbstractInterval, N, T <: RelaxTag} = MC{N,T}(Interval{Float64}(x.lo, x.hi))
