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
# src/forward_operators/concave_increasing.jl
# Contains definitions of log, log2, log10, log1p, acosh, sqrt.
#############################################################################

for opMC in (:log, :log2, :log10, :log1p, :acosh, :sqrt)
   opMC_kernel = Symbol(String(opMC)*"_kernel")
   dop = diffrule(:Base, opMC, :midcc)
   MCexp = quote
              xLc = z.lo
              xUc = z.hi
              (isnan(xLc) || isinf(xLc)) && (return nan(MC{N,T}))
              (isnan(xUc) || isinf(xUc)) && (return nan(MC{N,T}))
              xL = x.Intv.lo
              xU = x.Intv.hi
              midcc, cc_id = mid3(x.cc, x.cv, xU)
              midcv, cv_id = mid3(x.cc, x.cv, xL)
              dcv = (xUc > xLc) ? (xUc - xLc)/(xU - xL) : 0.0
              convex = dcv*(midcv - xL) + xLc
              concave = ($opMC)(midcc)
              concave_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*$dop
              convex_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
              convex, concave, convex_grad, concave_grad = cut(xLc, xUc, convex, concave, convex_grad, concave_grad)
              return MC{N, T}(convex, concave, z, convex_grad, concave_grad, x.cnst)
        end
    dop = diffrule(:Base, opMC, :(x.cv))
    dMCexp = quote
               xLc = z.lo
               xUc = z.hi
               (isnan(xLc) || isinf(xLc)) && (return nan(MC{N, Diff}))
               (isnan(xUc) || isinf(xUc)) && (return nan(MC{N, Diff}))
               xL = x.Intv.lo
               xU = x.Intv.hi
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
               concave = ($opMC)(midcc)
               convex_grad = slope*x.cv_grad
               concave_grad = ($dop)*x.cc_grad
               return MC{N, Diff}(convex, concave, z, convex_grad, concave_grad, x.cnst)
         end

     @eval @inline ($opMC_kernel)(x::MC{N, T}, z::Interval{Float64}) where {N,T<:Union{NS,MV}} = $MCexp
     @eval @inline ($opMC_kernel)(x::MC{N, Diff}, z::Interval{Float64}) where {N} = $dMCexp
     @eval @inline ($opMC)(x::MC) = ($opMC_kernel)(x, ($opMC)(x.Intv))
end
