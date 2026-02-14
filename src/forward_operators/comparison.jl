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
    (x.cv <= y.cv) && (x.cc <= y.cc) && (isstrictless(x.Intv, y.Intv) || isequal_interval(x.Intv, y.Intv))
end
function <=(x::MC{N,T}, c::Float64) where {N, T<:RelaxTag}
    (x.cv <= c) && (x.cc <= c) && (isstrictless(x.Intv, interval(c)) || isequal_interval(x.Intv, interval(c)))
end
function <=(c::Float64, y::MC{N,T}) where {N, T<:RelaxTag}
    (c <= y.cv) && (c <= y.cc) && (isstrictless(interval(c), y.Intv) || isequal_interval(interval(c), y.Intv))
end

function ==(x::MC{N,T}, y::MC{N,T}) where {N, T<:RelaxTag}
    (x.cv == y.cv) && (x.cc == y.cc) && (isequal_interval(x.Intv, y.Intv))
end
function ==(x::MC{N,T}, c::Float64) where {N, T<:RelaxTag}
    (x.cv == c) && (x.cc == c) && (x.Intv == c)
end
==(c::Float64, y::MC{N,T}) where {N, T<:RelaxTag} = (y == c)
