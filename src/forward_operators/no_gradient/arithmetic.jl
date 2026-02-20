# Copyright (c) 2018 Matthew Wilhelm, Robert Gottlieb, Dimitri Alston, 
# Matthew Stuber, and the University of Connecticut (UConn)
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A forward McCormick operator library
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# src/forward_operators/no_gradient/arithmetic.jl
# Contains definitions of +, -, /, *, promotions, conversion, one, zero.
#############################################################################

# Defines functions required for linear algebra packages
@inline nan(::ANYRELAX, ::Type{MCNoGrad}) = MCNoGrad(NaN, NaN, emptyinterval(), true)
@inline nan(::ANYRELAX, x::MCNoGrad)= MCNoGrad(NaN, NaN, emptyinterval(), true)

@inline one(::ANYRELAX, ::Type{MCNoGrad}) = MCNoGrad(1.0, 1.0, one(Interval), true)
@inline one(::ANYRELAX, x::MCNoGrad) = MCNoGrad(1.0, 1.0, one(Interval), true)

@inline zero(::ANYRELAX, ::Type{MCNoGrad}) = MCNoGrad(0.0, 0.0, zero(Interval), true)
@inline zero(::T, x::MCNoGrad) where T<:ANYRELAX = zero(T(), MCNoGrad)

@inline real(::ANYRELAX, x::MCNoGrad) = x
@inline dist(::ANYRELAX, x1::MCNoGrad, x2::MCNoGrad) = max(abs(x1.cc - x2.cc), abs(x1.cv - x2.cv))
@inline eps(::ANYRELAX, x::MCNoGrad) = max(eps(x.cc), eps(x.cv))
@inline mid(::ANYRELAX, x::MCNoGrad) = mid(x.Intv)

# Unsafe addition
@inline function plus_kernel(::ANYRELAX, x::MCNoGrad, y::MCNoGrad, z::Interval{Float64})
	MCNoGrad(x.cv + y.cv, x.cc + y.cc, z, x.cnst && y.cnst)
end
@inline +(::T, x::MCNoGrad, y::MCNoGrad) where T<:ANYRELAX = plus_kernel(T(), x, y, x.Intv + y.Intv)
@inline plus_kernel(::ANYRELAX, x::MCNoGrad, y::Interval{Float64}) = x
@inline +(::ANYRELAX, x::MCNoGrad) = x

@inline minus_kernel(::ANYRELAX, x::MCNoGrad, z::Interval{Float64}) = MCNoGrad(-x.cc, -x.cv, z, x.cnst)
@inline -(::ANYRELAX, x::MC) = minus_kernel(x, -x.Intv)
@inline -(::ANYRELAX, x::MCNoGrad, y::MCNoGrad) = minus_kernel(x, y, x.Intv - y.Intv)

# Unsafe subtraction
@inline function minus_kernel(::ANYRELAX, x::MCNoGrad, y::MCNoGrad, z::Interval{Float64})
	MCNoGrad(x.cv - y.cc, x.cc - y.cv, z, x.cnst && y.cnst)
end

################## CONVERT THROUGH BINARY DEFINITIONS #########################
# Unsafe scalar addition
@inline function plus_kernel(::ANYRELAX, x::MCNoGrad, y::Float64, z::Interval{Float64})
	MCNoGrad(x.cv + y, x.cc + y, z, x.cnst)
end
@inline +(::T, x::MCNoGrad, y::Float64) where T<:ANYRELAX = plus_kernel(T(), x, y, x.Intv + y)
@inline +(::T, y::Float64, x::MCNoGrad) where T<:ANYRELAX = plus_kernel(T(), x, y, x.Intv + y)
@inline +(::ANYRELAX, x::MCNoGrad, y::Interval{Float64}) = x + MCNoGrad(y)
@inline +(::ANYRELAX, y::Interval{Float64}, x::MCNoGrad) = x + MCNoGrad(y)

@inline plus_kernel(x::MCNoGrad, y::C, z::Interval{Float64}) where {C <: NumberNotRelax} = plus_kernel(x, convert(Float64, y), z)
@inline plus_kernel(x::C, y::MCNoGrad, z::Interval{Float64}) where {C <: NumberNotRelax} = plus_kernel(y, convert(Float64, x), z)
@inline +(::ANYRELAX, x::MCNoGrad, y::C) where {C <: NumberNotRelax} = x + convert(Float64, y)
@inline +(::ANYRELAX, y::C, x::MCNoGrad) where {C <: NumberNotRelax} = x + convert(Float64, y)

# Unsafe scalar subtraction
@inline function minus_kernel(::ANYRELAX, x::MCNoGrad, c::Float64, z::Interval{Float64})
	MCNoGrad(x.cv - c, x.cc - c, z, x.cnst)
end
@inline function minus_kernel(::ANYRELAX, c::Float64, x::MCNoGrad, z::Interval{Float64})
	MCNoGrad(c - x.cc, c - x.cv, z, x.cnst)
end
@inline -(::T, x::MCNoGrad, c::Float64) where T<:ANYRELAX = minus_kernel(T(), x, c, x.Intv - c)
@inline -(::T, c::Float64, x::MCNoGrad) where T<:ANYRELAX = minus_kernel(T(), c, x, c - x.Intv)
@inline -(::ANYRELAX, x::MCNoGrad, y::Interval{Float64}) = x - MCNoGrad(y)
@inline -(::ANYRELAX, y::Interval{Float64}, x::MCNoGrad) = MCNoGrad(y) - x

@inline minus_kernel(::T, x::MCNoGrad, y::C, z::Interval{Float64}) where {T<:ANYRELAX, C<:NumberNotRelax} = minus_kernel(T(), x, convert(Float64, y), z)
@inline minus_kernel(::T, y::C, x::MCNoGrad, z::Interval{Float64}) where {T<:ANYRELAX, C<:NumberNotRelax} = minus_kernel(T(), convert(Float64, y), x, z)
@inline -(::ANYRELAX, x::MCNoGrad, c::C) where {C <: NumberNotRelax} = x - convert(Float64,c)
@inline -(::ANYRELAX, c::C, x::MCNoGrad) where {C <: NumberNotRelax} = convert(Float64,c) - x

# Unsafe Scalar Multiplication
@inline function mult_kernel(::ANYRELAX, x::MCNoGrad, c::Float64, z::Interval{Float64})
	if c >= 0.0
		zMC = MCNoGrad(c*x.cv, c*x.cc, z, x.cnst)
	else
		zMC = MCNoGrad(c*x.cc, c*x.cv, z, x.cnst)
	end
	return zMC
end
@inline *(::T, x::MCNoGrad, c::Float64) where T <: ANYRELAX = mult_kernel(T(), x, c, c*x.Intv)
@inline *(::T, c::Float64, x::MCNoGrad) where T <: ANYRELAX = mult_kernel(T(), x, c, c*x.Intv)
@inline *(::ANYRELAX, x::MCNoGrad, y::Interval{Float64}) = x*MCNoGrad(y)
@inline *(::ANYRELAX, y::Interval{Float64}, x::MCNoGrad) = MCNoGrad(y)*x

@inline mult_kernel(x::MCNoGrad, c::C, z::Interval{Float64}) where {T<:ANYRELAX, C<:NumberNotRelax} = mult_kernel(T(), x, convert(Float64, c), z)
@inline mult_kernel(c::C, x::MCNoGrad, z::Interval{Float64}) where {T<:ANYRELAX, C<:NumberNotRelax} =  mult_kernel(T(), x, convert(Float64, c), z)
@inline *(::ANYRELAX, c::C, x::MCNoGrad) where {C <: NumberNotRelax} = x*Float64(c)
@inline *(::ANYRELAX, x::MCNoGrad, c::C) where {C <: NumberNotRelax} = x*Float64(c)

# Unsafe scalar division
@inline function div_kernel(::T, x::MCNoGrad, y::Float64, z::Interval{Float64}) where T<:ANYRELAX
	mult_kernel(T(), x, inv(y), z)
end
@inline function div_kernel(::T, x::Float64, y::MCNoGrad, z::Interval{Float64}) where T<:ANYRELAX
	mult_kernel(T(), inv(y), x, z)
end
@inline function div_kernel(::T, x::MCNoGrad, y::C, z::Interval{Float64}) where {C<:NumberNotRelax, T<:ANYRELAX}
	mult_kernel(T(), x, inv(y), z)
end
@inline function div_kernel(::T, x::C, y::MCNoGrad, z::Interval{Float64}) where {C<:NumberNotRelax, T<:ANYRELAX}
	mult_kernel(T(), inv(y), x, z)
end
@inline /(::ANYRELAX, x::MCNoGrad, y::Float64) = x*inv(y)
@inline /(::T, x::Float64, y::MCNoGrad) where T<:ANYRELAX = x*inv(T(),y)
@inline /(::ANYRELAX, x::MCNoGrad, y::C) = x*inv(convert(Float64,y))
@inline /(::T, x::C, y::MCNoGrad) where {C<:NumberNotRelax, T<:ANYRELAX} = convert(Float64,x)*inv(T(), y)
@inline /(::ANYRELAX, x::MCNoGrad, y::Interval{Float64}) = x/MCNoGrad(y)
@inline /(::T, y::Interval{Float64}, x::MCNoGrad) where T<:ANYRELAX = /(T(), MCNoGrad(y), x)

# Maximization
@inline function max_kernel(::T, c::Float64, x::MCNoGrad, z::Interval{Float64}) where T<:ANYRELAX
	max_kernel(T(), x, c, z)
end
@inline function max_kernel(::T, x::MCNoGrad, c::C, z::Interval{Float64}) where {C<:NumberNotRelax, T<:ANYRELAX}
	max_kernel(T(), x, convert(Float64, c), z)
end
@inline function max_kernel(::T, c::C, x::MCNoGrad, z::Interval{Float64}) where {C<:NumberNotRelax, T<:ANYRELAX}
	max_kernel(T(), x, convert(Float64, c), z)
end

@inline function max(::T, c::Float64, x::MCNoGrad) where T<:ANYRELAX
	max_kernel(T(), x, c, max(x.Intv, c))
end
@inline function max(::T, x::MCNoGrad, c::C) where {C<:NumberNotRelax, T<:ANYRELAX}
	max_kernel(T(), x, convert(Float64, c), max(x.Intv, c))
end
@inline function max(::T, c::C, x::MCNoGrad) where {C<:NumberNotRelax, T<:ANYRELAX}
	max_kernel(T(), x, convert(Float64, c), max(x.Intv, c))
end
@inline max(::T, x::MCNoGrad, y::Interval{Float64}) where T<:ANYRELAX = max(T(), x, MCNoGrad(y))
@inline max(::T, y::Interval{Float64}, x::MCNoGrad) where T<:ANYRELAX = max(T(), MCNoGrad(y), x)

# Minimization
@inline function min_kernel(::T, x::MCNoGrad, c::C, z::Interval{Float64}) where {C<:NumberNotRelax, T<:ANYRELAX}
	min_kernel(T(), x, convert(Float64, c), z)
end
@inline function min_kernel(::T, c::C, x::MCNoGrad, z::Interval{Float64}) where {C<:NumberNotRelax, T<:ANYRELAX}
	min_kernel(T(), x, convert(Float64, c), z)
end

@inline min(::T, c::Float64, x::MCNoGrad) where T<:ANYRELAX = min_kernel(T(), x, c, min(x.Intv, c))
@inline function min(::T, x::MCNoGrad, c::C) where {C<:NumberNotRelax, T<:ANYRELAX}
	min_kernel(T(), x, convert(Float64, c), min(x.Intv, c))
end
@inline function min(::T, c::C, x::MCNoGrad) where {C<:NumberNotRelax, T<:ANYRELAX}
	min_kernel(T(), x, convert(Float64, c), min(x.Intv, c))
end
@inline min(::T, x::MCNoGrad, y::Interval{Float64}) where T<:ANYRELAX = min(T(), x, MCNoGrad(y))
@inline min(::T, y::Interval{Float64}, x::MCNoGrad) where T<:ANYRELAX = min(T(), MCNoGrad(y), x)

# Add fma function
@inline fma(::ANYRELAX, x::MCNoGrad, y::Float64, z::Float64) = x*y + z
@inline fma(::ANYRELAX, x::MCNoGrad, y::MCNoGrad, z::Float64) = x*y + z
@inline fma(::ANYRELAX, x::MCNoGrad, y::Float64, z::MCNoGrad) = x*y + z
@inline fma(::ANYRELAX, x::MCNoGrad, y::MCNoGrad, z::MCNoGrad) = x*y + z
@inline fma(::ANYRELAX, x::Float64, y::MCNoGrad, z::Float64) = x*y + z
@inline fma(::ANYRELAX, x::Float64, y::MCNoGrad, z::MCNoGrad) = x*y + z
@inline fma(::ANYRELAX, x::Float64, y::Float64, z::MCNoGrad) = x*y + z

@inline fma(::ANYRELAX, x::MCNoGrad, y::Float64, z::Float64, q::Interval{Float64}) = x*y + z
@inline fma(::ANYRELAX, x::MCNoGrad, y::MCNoGrad, z::Float64, q::Interval{Float64}) = x*y + z
@inline fma(::ANYRELAX, x::MCNoGrad, y::Float64, z::MCNoGrad, q::Interval{Float64}) = x*y + z
@inline fma(::ANYRELAX, x::MCNoGrad, y::MCNoGrad, z::MCNoGrad, q::Interval{Float64}) = x*y + z
@inline fma(::ANYRELAX, x::Float64, y::MCNoGrad, z::Float64, q::Interval{Float64}) = x*y + z
@inline fma(::ANYRELAX, x::Float64, y::MCNoGrad, z::MCNoGrad, q::Interval{Float64}) = x*y + z
@inline fma(::ANYRELAX, x::Float64, y::Float64, z::MCNoGrad, q::Interval{Float64}) = x*y + z

# Promote and Convert
promote_rule(::Type{MCNoGrad}, ::Type{S}) where {S<:NumberNotRelax, N, T <: RelaxTag} = MCNoGrad
promote_rule(::Type{MCNoGrad}, ::Type{S}) where {S<:Real, N, T <: RelaxTag} = MCNoGrad

convert(::Type{MCNoGrad}, x::S) where {S<:NumberNotRelax, N, T <: RelaxTag} = MCNoGrad(interval(x))
convert(::Type{MCNoGrad}, x::S) where {S<:Interval, N, T <: RelaxTag} = MCNoGrad(interval(x.bareinterval.lo, x.bareinterval.hi))
interval(x::MCNoGrad) = x.Intv
