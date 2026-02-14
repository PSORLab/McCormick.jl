# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A McCormick operator library in Julia
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# src/forward_operators/division.jl
# Contains definitions of division.
#############################################################################

#=
@inline function div_MV(::Diff, x::MCNoGrad, y::MCNoGrad, z::Interval{Float64})
    if (0.0 < y.Intv.bareinterval.lo)
        cv, cv_grad = div_diffcv(x, y)
        cc, cc_grad = div_diffcv(-x, y)
        return MC{N,Diff}(cv, -cc, z, cv_grad, -cc_grad, x.cnst && y.cnst)
    end
    cv, cv_grad = div_diffcv(-x, -y)
    cc, cc_grad = div_diffcv(-x, y)
    return MC{N,Diff}(cv, -cc, z, cv_grad, -cc_grad, x.cnst && y.cnst)
end

@inline function div_kernel(t::Union{NS,MV}, x::MCNoGrad, y::MCNoGrad, z::Interval{Float64})
    if (x === y)
        zMC = one(MCNoGrad)
    else
        zMC = mult_kernel(t, x, inv(y), z)
    end
    return zMC
end

@inline function div_kernel(t::ANYRELAX, x::MCNoGrad, y::MCNoGrad, z::Interval{Float64})
    degen1 = (x.Intv.bareinterval.hi - x.Intv.bareinterval.lo == 0.0)
    degen2 = (y.Intv.bareinterval.hi - y.Intv.bareinterval.lo == 0.0)
    if x === y
        zMC = one(MCNoGrad)
    elseif  !degen1 || !degen2
        zMC = div_MV(t, x, y, z)
    else
        zMC = mult_kernel(t, x, inv(y), z)
    end
    return zMC
end
=#

@inline function /(t::ANYRELAX, x::MCNoGrad, y::MCNoGrad)
    if 0.0 ∉ y
        z, _ =  div_kernel(t, x, y, x.Intv/y.Intv)
        return z
    end
    return nan(MCNoGrad)
end
