# Copyright (c) 2018 Matthew Wilhelm, Robert Gottlieb, Dimitri Alston, 
# Matthew Stuber, and the University of Connecticut (UConn)
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A forward McCormick operator library
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# src/forward_operators/comparison.jl
# Defines :<, :<=, :!=, :==.
#############################################################################

function <(x::MC{N,T}, y::MC{N,T}) where {N, T<:RelaxTag}
    (x.cv < y.cv) && (x.cc < y.cc) && (isstrictless(x.Intv, y.Intv))
end
function <(x::MC{N,T}, c::Float64) where {N, T<:RelaxTag}
    (x.cv < c) && (x.cc < c) && (isstrictless(x.Intv, interval(c)))
end
function <(c::Float64, y::MC{N,T}) where {N, T<:RelaxTag}
    (c < y.cv) && (c < y.cc) && (isstrictless(interval(c), y.Intv))
end

function <=(x::MC{N,T}, y::MC{N,T}) where {N, T<:RelaxTag}
    (x.cv <= y.cv) && (x.cc <= y.cc) && (isweakless(x.Intv, y.Intv))
end
function <=(x::MC{N,T}, c::Float64) where {N, T<:RelaxTag}
    (x.cv <= c) && (x.cc <= c) && (isweakless(x.Intv, interval(c)))
end
function <=(c::Float64, y::MC{N,T}) where {N, T<:RelaxTag}
    (c <= y.cv) && (c <= y.cc) && (isweakless(interval(c), y.Intv))
end

function ==(x::MC{N,T}, y::MC{N,T}) where {N, T<:RelaxTag}
    (x.cv == y.cv) && (x.cc == y.cc) && (isequal_interval(x.Intv, y.Intv))
end
function ==(x::MC{N,T}, c::Float64) where {N, T<:RelaxTag}
    (x.cv == c) && (x.cc == c) && (x.Intv == c)
end
==(c::Float64, y::MC{N,T}) where {N, T<:RelaxTag} = (y == c)
