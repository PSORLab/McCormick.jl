# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A McCormick operator library in Julia
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# src/forward_operators/comparison.jl
# Defines :<, :<=, :!=, :==.
#############################################################################

for f in (:<, :<=)
    @eval begin
        function ($f)(::ANYRELAX, x::MCNoGrad, y::MCNoGrad)
            ($f)(x.cv,y.cv) && ($f)(x.cc,y.cc) && ($f)(x.Intv, y.Intv)
        end
        function ($f)(::ANYRELAX, x::MCNoGrad, c::Float64)
            ($f)(x.cv, c) && ($f)(x.cc, c) && ($f)(x.Intv, c)
        end
        function ($f)(::ANYRELAX, c::Float64, y::MCNoGrad)
            ($f)(c, y.cv) && ($f)(c, y.cc) && ($f)(c, y.Intv)
        end
    end
end

function ==(::ANYRELAX, x::MCNoGrad, y::MCNoGrad)
    (x.cv == y.cv) && (x.cc == y.cc) && (x.Intv == y.Intv)
end
function ==(::ANYRELAX, x::MCNoGrad, c::Float64)
    (x.cv == c) && (x.cc == c) && (x.Intv == c)
end
==(::ANYRELAX, c::Float64, y::MCNoGrad) = (y == c)
