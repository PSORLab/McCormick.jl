using BenchmarkTools, IntervalArithmetic

# don't really want to measure interval performance, so we're using the
# fastest interval rounding mode
setrounding(Interval, :none)
using McCormick

println("Begin Allocation Check")

# Allocations are expected for the following operations:
# - MC(Float64, Interval, 1): Nonstatic definition of static type.
# - exp2, exp10, tanh, atanh, acosh: Underlying interval operation may allocate.
for T in (NS, Diff, MV)
    !iszero(allocs(@benchmark MC{$5,$T}(2.1) samples = 1)) && println("MC(Float64) Allocates")
    !iszero(allocs(@benchmark MC{$5,$T}(0.1,2.0) samples = 1)) && println("MC(Float64, Float64) Allocates")
    !iszero(allocs(@benchmark MC{$5,$T}((Interval{Float64}(0.1,2.0))) samples = 1)) && println("MC(Interval) Allocates")
    a = MC{5,T}(1.0,(Interval{Float64}(0.1,2.0)),2)
    for op in (+, -, exp, expm1, log, log2, log10, log1p,
               abs, one, zero, real, max, min, inv, cosh, sqrt,
               isone, isnan)
        !iszero(allocs(@benchmark ($op)($a) samples = 1)) && println("$op Allocates")
    end
           # domain violations asin, asind, acos, acosd, atan, atand, tan, tand, sec, secd,
           for op in (sin, sind, cos, cosd, step, sign, sinh)
               a = MC{5,T}(0.4,(Interval{Float64}(0.1,0.9)),2)
               !iszero(allocs(@benchmark ($op)($a) samples = 1)) && println("Positive $op Allocates")
               a = MC{5,T}(0.1,(Interval{Float64}(-0.4,0.3)),2)
               !iszero(allocs(@benchmark ($op)($a) samples = 1)) && println("0 in X $op Allocates")
               a = MC{5,T}(-0.5,(Interval{Float64}(-0.9,-0.1)),2)
               !iszero(allocs(@benchmark ($op)($a) samples = 1)) && println("Negative $op Allocates")
              end
           for op in (min, max, *, -, +, /)
               x = MC{5,T}(0.4,(Interval{Float64}(0.1,0.9)),2)
               y = MC{5,T}(0.5,(Interval{Float64}(0.3,0.9)),2)
               q = 1.34534
               !iszero(allocs(@benchmark ($op)($x, $q) samples = 1)) && println("$op(x,q) Allocates")
               !iszero(allocs(@benchmark ($op)($q, $x) samples = 1)) && println("$op(q,x) Allocates")
               !iszero(allocs(@benchmark ($op)($x, $y) samples = 1)) && println("$op(x,y) Allocates")
           end
           a = MC{5,T}(0.4,Interval{Float64}(0.1,0.9),2);    !iszero(allocs(@benchmark $a^2 samples = 1)) && println("a^2 Pos Allocates")
           a = MC{5,T}(0.1,Interval{Float64}(-0.4,0.3),2);   !iszero(allocs(@benchmark $a^2 samples = 1)) && println("a^2 Zero Allocates")
           a = MC{5,T}(-0.5,Interval{Float64}(-0.9,-0.1),2); !iszero(allocs(@benchmark $a^2 samples = 1)) && println("a^2 Neg Allocates")
           a = MC{5,T}(0.4,Interval{Float64}(0.1,0.9),2);    !iszero(allocs(@benchmark $a^3 samples = 1)) && println("a^3 Pos Allocates")
           a = MC{5,T}(0.1,Interval{Float64}(-0.4,0.3),2);   !iszero(allocs(@benchmark $a^3 samples = 1)) && println("a^3 Zero Allocates")
           a = MC{5,T}(-0.5,Interval{Float64}(-0.9,-0.1),2); !iszero(allocs(@benchmark $a^3 samples = 1)) && println("a^3 Neg Allocates")
           a = MC{5,T}(0.4,Interval{Float64}(0.1,0.9),2);    !iszero(allocs(@benchmark $a^4 samples = 1)) && println("a^4 Pos Allocates")
           a = MC{5,T}(0.1,Interval{Float64}(-0.4,0.3),2);   !iszero(allocs(@benchmark $a^4 samples = 1)) && println("a^4 Zero Allocates")
           a = MC{5,T}(-0.5,Interval{Float64}(-0.9,-0.1),2); !iszero(allocs(@benchmark $a^4 samples = 1)) && println("a^4 Neg Allocates")
           a = MC{5,T}(0.4,Interval{Float64}(0.1,0.9),2);    !iszero(allocs(@benchmark $a^(-1) samples = 1)) && println("a^-1 Pos Allocates")
           a = MC{5,T}(-0.5,Interval{Float64}(-0.9,-0.1),2); !iszero(allocs(@benchmark $a^(-1) samples = 1)) && println("a^-1 Neg Allocates")
           a = MC{5,T}(0.4,Interval{Float64}(0.1,0.9),2);    !iszero(allocs(@benchmark $a^(-2) samples = 1)) && println("a^-2 Pos Allocates")
           a = MC{5,T}(-0.5,Interval{Float64}(-0.9,-0.1),2); !iszero(allocs(@benchmark $a^(-2) samples = 1)) && println("a^-2 Neg Allocates")
           a = MC{5,T}(0.4,Interval{Float64}(0.1,0.9),2);    !iszero(allocs(@benchmark $a^(-3) samples = 1)) && println("a^-3 Pos Allocates")
           a = MC{5,T}(-0.5,Interval{Float64}(-0.9,-0.1),2); !iszero(allocs(@benchmark $a^(-3) samples = 1)) && println("a^-3 Neg Allocates")
   end
end

println("Allocation Check Complete!")
