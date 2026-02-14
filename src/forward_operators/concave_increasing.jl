# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A McCormick operator library in Julia
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# src/forward_operators/concave_increasing.jl
# Contains definitions of log, log2, log10, log1p, acosh, sqrt.
#############################################################################

for opMC in (:log, :log2, :log10, :log1p)
   opMC_kernel = Symbol(String(opMC)*"_kernel")
   dop = diffrule(:Base, opMC, :midcc)
   MCexp = quote
              xLc = z.bareinterval.lo
              xUc = z.bareinterval.hi
              xL = x.Intv.bareinterval.lo
              xU = x.Intv.bareinterval.hi
              midcc, cc_id = mid3(x.cc, x.cv, xU)
              midcv, cv_id = mid3(x.cc, x.cv, xL)
              dcv = (xUc > xLc) ? (xUc - xLc)/(xU - xL) : 0.0
              convex = dcv*(midcv - xL) + xLc
              concave = (NaNMath.$opMC)(midcc)
              concave_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*$dop
              convex_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
              convex, concave, convex_grad, concave_grad = cut(xLc, xUc, convex, concave, convex_grad, concave_grad)
              return MC{N, T}(convex, concave, z, convex_grad, concave_grad, x.cnst)
        end
    dop = diffrule(:Base, opMC, :(x.cv))
    dMCexp = quote
               xLc = z.bareinterval.lo
               xUc = z.bareinterval.hi
               xL = x.Intv.bareinterval.lo
               xU = x.Intv.bareinterval.hi
               midcc = mid3v(x.cv, x.cc, xU)
               midcv = mid3v(x.cv, x.cc, xL)
               deltaX = (xU - xL)
               slope = (xUc - xLc)/deltaX
               dcv = (xUc > xLc) ? slope : 0.0
               if deltaX == 0.0
                  convex = xLc
               else
                  convex = (xLc*(xU - midcv) + xUc*(midcv - xL))/deltaX
               end
               concave = (NaNMath.$opMC)(midcc)
               convex_grad = slope*x.cv_grad
               concave_grad = ($dop)*x.cc_grad
               return MC{N, Diff}(convex, concave, z, convex_grad, concave_grad, x.cnst)
         end

     @eval @inline ($opMC_kernel)(x::MC{N, T}, z::Interval{Float64}) where {N,T<:Union{NS,MV}} = $MCexp
     @eval @inline ($opMC_kernel)(x::MC{N, Diff}, z::Interval{Float64}) where {N} = $dMCexp
     @eval @inline ($opMC)(x::MC) = ($opMC_kernel)(x, ($opMC)(x.Intv))
end

@inline function acosh_kernel(x::MC{N, T}, z::Interval{Float64}) where {N, T<:Union{NS, MV}}
     isempty(x) && (return empty(x))
     (x.Intv.bareinterval.lo < 1.0 || x.Intv.bareinterval.hi < 1.0) && (return nan(MC{N,T}))
     xLc = z.bareinterval.lo
     xUc = z.bareinterval.hi
     xL = x.Intv.bareinterval.lo
     xU = x.Intv.bareinterval.hi
     midcc, cc_id = mid3(x.cc, x.cv, xU)
     midcv, cv_id = mid3(x.cc, x.cv, xL)
     dcv = (xUc > xLc) ? (xUc - xLc)/(xU - xL) : 0.0
     convex = dcv*(midcv - xL) + xLc
     concave = NaNMath.acosh(midcc)
     concave_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*inv(NaNMath.sqrt(x.cv - 1.0)*NaNMath.sqrt(x.cv + 1.0))
     convex_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
     convex, concave, convex_grad, concave_grad = cut(xLc, xUc, convex, concave, convex_grad, concave_grad)
     return MC{N, T}(convex, concave, z, convex_grad, concave_grad, x.cnst)
end
@inline function acosh_kernel(x::MC{N, Diff}, z::Interval{Float64}) where N
     isempty(x) && (return empty(x))
     (x.Intv.bareinterval.lo < 1.0 || x.Intv.bareinterval.hi < 1.0) && (return nan(MC{N,Diff}))
     xLc = z.bareinterval.lo
     xUc = z.bareinterval.hi
     xL = x.Intv.bareinterval.lo
     xU = x.Intv.bareinterval.hi
     midcc = mid3v(x.cv, x.cc, xU)
     midcv = mid3v(x.cv, x.cc, xL)
     deltaX = (xU - xL)
     slope = (xUc - xLc)/deltaX
     dcv = (xUc > xLc) ? slope : 0.0
     if deltaX == 0.0
        convex = xLc
     else
        convex = (xLc*(xU - midcv) + xUc*(midcv - xL))/deltaX
     end
     concave = NaNMath.acosh(midcc)
     convex_grad = slope*x.cv_grad
     concave_grad = inv(NaNMath.sqrt(x.cv - 1.0)*NaNMath.sqrt(x.cv + 1.0))*x.cc_grad
     return MC{N, Diff}(convex, concave, z, convex_grad, concave_grad, x.cnst)
end
@inline acosh(x::MC) = acosh_kernel(x, acosh(x.Intv))

@inline function sqrt_kernel(x::MC{N, T}, z::Interval{Float64}) where {N, T<:Union{NS, MV}}
     isempty(x) && (return empty(x))
     (x.Intv.bareinterval.lo < 0.0 || x.Intv.bareinterval.hi < 0.0) && (return MC{N,T}(NaN, NaN, z, fill(0, SVector{N,Float64}), fill(0, SVector{N,Float64}), x.cnst))
     xLc = z.bareinterval.lo
     xUc = z.bareinterval.hi
     xL = x.Intv.bareinterval.lo
     xU = x.Intv.bareinterval.hi
     midcc, cc_id = mid3(x.cc, x.cv, xU)
     midcv, cv_id = mid3(x.cc, x.cv, xL)
     dcv = (xUc > xLc && !isinf(xUc)) ? (xUc - xLc)/(xU - xL) : 0.0
     convex = dcv*(midcv - xL) + xLc
     concave = NaNMath.sqrt(midcc)
     concave_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*(0.5/NaNMath.sqrt(x.cv))
     convex_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
     convex, concave, convex_grad, concave_grad = cut(xLc, xUc, convex, concave, convex_grad, concave_grad)
     return MC{N, T}(convex, concave, z, convex_grad, concave_grad, x.cnst)
end
@inline function sqrt_kernel(x::MC{N, Diff}, z::Interval{Float64}) where N
     isempty(x) && (return empty(x))
     (x.Intv.bareinterval.lo < 0.0 || x.Intv.bareinterval.hi < 0.0) && (return nan(MC{N,Diff}))
     xLc = z.bareinterval.lo
     xUc = z.bareinterval.hi
     xL = x.Intv.bareinterval.lo
     xU = x.Intv.bareinterval.hi
     midcc = mid3v(x.cv, x.cc, xU)
     midcv = mid3v(x.cv, x.cc, xL)
     deltaX = (xU - xL)
     slope = (xUc - xLc)/deltaX
     dcv = (xUc > xLc) ? slope : 0.0
     if deltaX == 0.0
        convex = xLc
     else
        convex = (xLc*(xU - midcv) + xUc*(midcv - xL))/deltaX
     end
     concave = NaNMath.sqrt(midcc)
     convex_grad = slope*x.cv_grad
     concave_grad = (0.5/NaNMath.sqrt(x.cv))*x.cc_grad
     return MC{N, Diff}(convex, concave, z, convex_grad, concave_grad, x.cnst)
end
@inline sqrt(x::MC) = sqrt_kernel(x, sqrt(x.Intv))
