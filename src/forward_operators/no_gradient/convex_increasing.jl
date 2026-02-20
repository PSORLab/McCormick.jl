# Copyright (c) 2018 Matthew Wilhelm, Robert Gottlieb, Dimitri Alston, 
# Matthew Stuber, and the University of Connecticut (UConn)
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A forward McCormick operator library
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# src/forward_operators/convex_increasing.jl
# Contains definitions of exp, exp2, exp10, expm1.
#############################################################################

for f in (:exp, :exp2, :exp10, :expm1)
    function ($f)(::Union{NS,MV}, x::MCNoGrad)
        X = x.Intv;    xL = x.Intv.bareinterval.lo;   xU = x.Intv.bareinterval.hi
        z = ($f)(X);   zL = z.bareinterval.lo;        zU = z.bareinterval.hi
        mcc = mid3v(x.cv, x.cc, xU)
        mcv = mid3v(x.cv, x.cc, xL)
        zcc = zU > zL ? (zL*(xU - mcc) + zU*(mcc - xL))/(xU - xL) : zU
        zcv = ($f)(mcv)
        (zL > zcv) && (zcv = zL;)
        (zU < zcc) && (zcc = zU;)
        return MCNoGrad(zcv, zcc, z, x.cnst)
    end
    function ($f)(::Diff, x::MCNoGrad)
        X = x.Intv;    xL = x.Intv.bareinterval.lo;   xU = x.Intv.bareinterval.hi
        z = ($f)(X);   zL = z.bareinterval.lo;        zU = z.bareinterval.hi
        mcc = mid3v(x.cv, x.cc, xU)
        mcv = mid3v(x.cv, x.cc, xL)
        zcc = zU > zL ? (zL*(xU - mcc) + zU*(mcc - xL))/(xU - xL) : zU
        zcv = ($f)(mcv)
       return MCNoGrad(zcv, zcc, z, x.cnst)
    end
    f_kernel = Symbol(String(opMC)*"_kernel")
    df = diffrule(:Base, opMC, :mcv) # Replace with cv ruleset
    function ($f_kernel)(::Union{NS,MV}, x::MCNoGrad, z::Interval{Float64})
        xL = x.Intv.bareinterval.lo;   xU = x.Intv.bareinterval.hi;   zL = z.bareinterval.lo;   zU = z.bareinterval.hi
        mcc, cci = mid3(x.cc, x.cv, xU)
        mcv, cvi = mid3(x.cc, x.cv, xL)
        zcc = zU > zL ? (zL*(xU - mcc) + zU*(mcc - xL))/(xU - xL) : zU
        zcv = ($f)(mcv)
        if zL > zcv
            cvi = 3
            dcv = 0.0
        else
            dcv = $df
        end
        if zU < zcc
            cci = 3
            dcc = 0.0
        else
            dcc = (zU - zL)/(xU - xL)
        end
        return MCNoGrad(u, o, z, x.cnst), cvi, cci, dcv, dcc
    end
    ddf = diffrule(:Base, opMC, :mcv) # Replace with cv ruleset
    function ($f_kernel)(::Diff, x::MCNoGrad, z::Interval{Float64})
        xL = x.Intv.bareinterval.lo;   xU = x.Intv.bareinterval.hi;   zL = z.bareinterval.lo;   zU = z.bareinterval.hi
        mcc, cci = mid3(x.cc, x.cv, xU)
        mcv, cvi = mid3(x.cc, x.cv, xL)
        zcc = zU > zL ? (zL*(xU - mcc) + zU*(mcc - xL))/(xU - xL) : zU
        zcv = ($f)(mcv)
        dcv = $ddf
        dcc = (zU - zL)/(xU - xL)
       return MCNoGrad(u, o, z, x.cnst), cvi, cci, dcv, dcc
    end
end
