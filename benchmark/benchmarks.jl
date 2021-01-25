# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A McCormick operator library in Julia
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# benchmark/benchmarks.jl
# Runs benchmarks on all supported McCormick operators.
# Allocations are expected for the following operations:
# - MC(Float64, Interval, 1): Nonstatic definition of static type.
# - exp2, exp10, tanh, atanh, acosh: Underlying interval operation may allocate.
#############################################################################

using BenchmarkTools, IntervalArithmetic

# don't really want to measure interval performance, so we're using the
# fastest interval rounding mode
setrounding(Interval, :none)
using McCormick

const SUITE = BenchmarkGroup()

function test_comp1(f, x, n)
    z = f(x)
    for i = 1:n
        z = f(z)
    end
    z
end

function test_comp2l(f, x, y, n)
    z = f(x, y)
    for i = 1:n
        z = f(z, y)
    end
    z
end

function test_comp2r(f, x, y, n)
    z = f(x, y)
    for i = 1:n
        z = f(x, z)
    end
    z
end

for T in (NS, ) #(NS, Diff, MV)
        begin
            S = SUITE["Constructors $(T)"] = BenchmarkGroup()
            S["MC(Float64)"] = @benchmarkable MC{5,$T}(2.1)
            S["MC(Float64,Float64)"] = @benchmarkable MC{5,$T}(0.1,2.0)
            S["MC(Interval{Float64})"] = @benchmarkable MC{5,$T}($(Interval{Float64}(0.1,2.0)))
            S["MC(Float64, Interval{Float64}, Int)"] = @benchmarkable MC{5,$T}(1.0,$(Interval{Float64}(0.1,2.0)),2)
            S = SUITE["OOP $(T) McCormick"] = BenchmarkGroup()
        end
    begin
        for op in (+, -, exp, expm1, log, log2, log10, log1p,
                   abs, one, zero, real, max, min, inv, cosh, sqrt,
                   isone, isnan)
            S[string(op)*"(x)"] = @benchmarkable $(test_comp1)($op, a, 1000) setup = (a = MC{5,$T}(1.0,$(Interval{Float64}(0.1,2.0)),2))
        end
        # domain violations asin, asind, acos, acosd, atan, atand, tan, tand, sec, secd,
        for op in (sin, sind, cos, cosd, step, sign, sinh)
            S[string(op)*"(x), x > 0"] = @benchmarkable $(test_comp1)($op, a, 1000) setup = (a = MC{5,$T}(0.4,$(Interval{Float64}(0.1,0.9)),2))
            S[string(op)*"(x), 0 ∈ x"] = @benchmarkable $(test_comp1)($op, a, 1000) setup = (a = MC{5,$T}(0.1,$(Interval{Float64}(-0.4,0.3)),2))
            S[string(op)*"(x), x < 0"] = @benchmarkable $(test_comp1)($op, a, 1000) setup = (a = MC{5,$T}(-0.5,$(Interval{Float64}(-0.9,-0.1)),2))
        end
        for op in (min, max, *, -, +, /)
            S[string(op)*"(x, Float64)"] =  @benchmarkable $(test_comp2l)($op, x, q, 10000) setup = (x = MC{5,$T}(0.4,$(Interval{Float64}(0.1,0.9)),2); q = 1.34534)
            S[string(op)*"(Float64, x)"] =  @benchmarkable $(test_comp2r)($op, q, x, 10000) setup = (x = MC{5,$T}(0.4,$(Interval{Float64}(0.1,0.9)),2); q = 1.34534)
            S[string(op)*"(x, y)"] =   @benchmarkable $(test_comp2l)($op, x, y, 1000) setup = (x = MC{5,$T}(0.4,$(Interval{Float64}(0.1,0.9)),2);
                                                                            y = MC{5,$T}(0.5,$(Interval{Float64}(0.3,0.9)),2);)
        end
        S["x^2, x > 0"] = @benchmarkable test_comp2l($^, a, 2, 1000) setup = (a = MC{5,$T}(0.4,$(Interval{Float64}(0.1,0.9)),2))
        S["x^2, 0 ∈ x"] = @benchmarkable test_comp2l($^, a, 2, 1000) setup = (a = MC{5,$T}(0.1,$(Interval{Float64}(-0.4,0.3)),2))
        S["x^2, x < 0"] = @benchmarkable test_comp2l($^, a, 2, 1000) setup = (a = MC{5,$T}(-0.5,$(Interval{Float64}(-0.9,-0.1)),2))
        S["x^3, x > 0"] = @benchmarkable test_comp2l($^, a, 3, 1000) setup = (a = MC{5,$T}(0.4,$(Interval{Float64}(0.1,0.9)),2))
        S["x^3, 0 ∈ x"] = @benchmarkable test_comp2l($^, a, 3, 1000) setup = (a = MC{5,$T}(0.1,$(Interval{Float64}(-0.4,0.3)),2))
        S["x^3, x < 0"] = @benchmarkable test_comp2l($^, a, 3, 1000) setup = (a = MC{5,$T}(-0.5,$(Interval{Float64}(-0.9,-0.1)),2))
        S["x^4, x > 0"] = @benchmarkable test_comp2l($^, a, 4, 1000) setup = (a = MC{5,$T}(0.4,$(Interval{Float64}(0.1,0.9)),2))
        S["x^4, 0 ∈ x"] = @benchmarkable test_comp2l($^, a, 4, 1000) setup = (a = MC{5,$T}(0.1,$(Interval{Float64}(-0.4,0.3)),2))
        S["x^4, x < 0"] = @benchmarkable test_comp2l($^, a, 4, 1000) setup = (a = MC{5,$T}(-0.5,$(Interval{Float64}(-0.9,-0.1)),2))
        S["x^-1, x > 0"] = @benchmarkable test_comp2l($^, a, -1, 1000) setup = (a = MC{5,$T}(0.4,$(Interval{Float64}(0.1,0.9)),2))
        S["x^-1, x < 0"] = @benchmarkable test_comp2l($^, a, -1, 1000) setup = (a = MC{5,$T}(-0.5,$(Interval{Float64}(-0.9,-0.1)),2))
        S["x^-2, x > 0"] = @benchmarkable test_comp2l($^, a, -2, 1000) setup = (a = MC{5,$T}(0.4,$(Interval{Float64}(0.1,0.9)),2))
        S["x^-2, x < 0"] = @benchmarkable test_comp2l($^, a, -2, 1000) setup = (a = MC{5,$T}(-0.5,$(Interval{Float64}(-0.9,-0.1)),2))
        S["x^-3, x > 0"] = @benchmarkable test_comp2l($^, a, -3, 1000) setup = (a = MC{5,$T}(0.4,$(Interval{Float64}(0.1,0.9)),2))
        S["x^-3, x < 0"] = @benchmarkable test_comp2l($^, a, -3, 1000) setup = (a = MC{5,$T}(-0.5,$(Interval{Float64}(-0.9,-0.1)),2))
    end
end
