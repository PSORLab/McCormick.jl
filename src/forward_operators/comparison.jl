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
# src/forward_operators/comparison.jl
# Defines :<, :<=, :!=, :==.
#############################################################################

for f in (:<, :<=)
    @eval begin
        function ($f)(x::MC{N,T}, y::MC{N,T}) where {N, T<:RelaxTag}
            ($f)(x.cv,y.cv) && ($f)(x.cc,y.cc) && ($f)(x.Intv, y.Intv)
        end
        function ($f)(x::MC{N,T}, c::Float64) where {N, T<:RelaxTag}
            ($f)(x.cv, c) && ($f)(x.cc, c) && ($f)(x.Intv, c)
        end
        function ($f)(c::Float64, y::MC{N,T}) where {N, T<:RelaxTag}
            ($f)(c, y.cv) && ($f)(c, y.cc) && ($f)(c, y.Intv)
        end
    end
end

function ==(x::MC{N,T}, y::MC{N,T}) where {N, T<:RelaxTag}
    (x.cv == y.cv) && (x.cc == y.cc) && (x.Intv == y.Intv)
end
function ==(x::MC{N,T}, c::Float64) where {N, T<:RelaxTag}
    (x.cv == c) && (x.cc == c) && (x.Intv == c)
end
==(c::Float64, y::MC{N,T}) where {N, T<:RelaxTag} = (y == c)
