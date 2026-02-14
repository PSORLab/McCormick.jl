#=
Code taken from IntervalSpecialFunctions.jl which will be added as a dependency
once this has been tagged. Some modifications have been made.

The IntervalSpecialFunctions.jl package is licensed under the MIT "Expat" License:
Copyright (c) 2018: David Sanders.

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
=#
for f in (:erf, :erfc)

    @eval function($f)(x::BigFloat, r::RoundingMode)
        setrounding(BigFloat, r) do
            ($f)(x)
        end
    end

    @eval ($f)(a::Interval{Float64}) = convert(Interval{Float64}, ($f)(atomic(BigFloat, a)))
end

function erf(a::Interval{T}) where T
    isempty_interval(a) && return a
    convert(Interval{Float64}, interval(erf(a.bareinterval.lo), erf(a.bareinterval.hi)))
end

function erfc(a::Interval{T}) where T
    isempty_interval(a) && return a
    convert(Interval{Float64}, interval(erfc(a.bareinterval.hi), erfc(a.bareinterval.lo)))
end

erfinv(x, r::RoundingMode{:Down}) = _erfinv(x).bareinterval.lo
erfinv(x, r::RoundingMode{:Up}) = _erfinv(x).bareinterval.hi
erfcinv(x, r::RoundingMode{:Down}) = _erfcinv(x).bareinterval.hi
erfcinv(x, r::RoundingMode{:Up}) = _erfcinv(x).bareinterval.lo

function _erfinv(a::T) where T
    domain = convert(Interval{T}, interval(-1.0, 1.0))
    a ∉ domain && return DomainError("$a is not in [-1, 1]")
    f = x -> erf(x) - a
    fp = x->2/sqrt(interval(pi)) * exp(-x^2)
    rts = roots(f, fp, Interval{T}(-Inf, Inf))
    @assert length(rts) == 1 # && rts[1].status == :unique

    rts[1].interval
end

function erfinv(a::Interval{T}) where T
    domain = convert(Interval{T}, interval(-1.0, 1.0))
    a = intersect_interval(a, domain)

    isempty_interval(a) && return a
    convert(Interval{Float64}, interval(erfinv(a.bareinterval.lo), erfinv(a.bareinterval.hi)))
end

function _erfcinv(a::T) where T
    domain = convert(Interval{T}, interval(0.0, 2.0))
    a ∉ domain && return DomainError("$a is not in [0, 2]")
    f = x -> erfc(x) - a
    fp = x -> -2/sqrt(interval(pi)) * exp(-x^2)
    rts = roots(f, fp, interval(-Inf, Inf))
    @assert length(rts) == 1 # && rts[1].status == :unique

    rts[1].interval
end

function erfcinv(a::Interval{T}) where T
    domain = convert(Interval{T}, interval(0.0, 2.0))
    a = intersect_interval(a, domain)

    isempty_interval(a) && return a
    
    convert(Interval{Float64}, interval(erfcinv(a.bareinterval.hi), erfcinv(a.bareinterval.lo)))
end
