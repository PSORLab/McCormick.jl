# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A McCormick operator library in Julia
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# src/implicit_routines/contact.jl
# Definitions for contract functions used in implicit relaxations.
#############################################################################

cv_del(x::MC{N,T}) where {N, T<:RelaxTag} = x.Intv - x.cv
cc_del(x::MC{N,T}) where {N, T<:RelaxTag} = x.Intv - x.cc
cv_diff(x::MC{N,T}, y::MC{N,T}) where {N, T<:RelaxTag} = x.cv - y.cv
cc_diff(x::MC{N,T}, y::MC{N,T}) where {N, T<:RelaxTag} = x.cc - y.cc

function affine_expand_del(f, x::Vector{MC{N,T}}, fx0::Float64, ∇fx0::SVector{N,Float64}) where {N, T<:RelaxTag}
    v = fx0
    for i=1:N
        v += ∇fx0[i]*f(x[i])
    end
    return v
end

function affine_expand_del(f, x::Vector{MC{N,T}}, xr::Vector{MC{N,T}}, fx0::Float64, ∇fx0::SVector{N,Float64}) where {N, T<:RelaxTag}
    v = fx0
    for i=1:N
        v += ∇fx0[i]*f(x[i], xr[i])
    end
    return v
end

function estimator_extrema(x::MC{N,T}, y::MC{N,T}, p::Vector{MC{N,T}}) where {N, T<:RelaxTag}
    xcv = x.cv;   xcvg = x.cv_grad
    ycv = y.cv;   ycvg = y.cv_grad
    xccn = -x.cc; xccgn = -x.cc_grad
    yccn = -y.cc; yccgn = -y.cc_grad
    t3 = affine_expand_del(cv_del, p, xcv,  xcvg)
    t4 = affine_expand_del(cv_del, p, ycv,  ycvg)
    s3 = affine_expand_del(cc_del, p, xccn, xccgn)
    s4 = affine_expand_del(cc_del, p, yccn, yccgn)
    return t3, t4, s3, s4
end

function estimator_under(x::MC{N,T}, y::MC{N,T}, p::Vector{MC{N,T}}, p0::Vector{MC{N,T}}) where {N, T<:RelaxTag}
    xcv = x.cv;   xcvg = x.cv_grad
    ycv = y.cv;   ycvg = y.cv_grad
    u1cv = affine_expand_del(cv_diff, p, p0,  xcv, xcvg)
    u2cv = affine_expand_del(cv_diff, p, p0, ycv, ycvg)
    return u1cv, u2cv, xcvg, ycvg
end

function estimator_over(x::MC{N,T}, y::MC{N,T}, p::Vector{MC{N,T}}, p0::Vector{MC{N,T}}) where {N, T<:RelaxTag}
    xccn = -x.cc; xccgn = -x.cc_grad
    yccn = -y.cc; yccgn = -y.cc_grad
    v1ccn = affine_expand_del(cc_diff, p, p0, xccn, xccgn)
    v2ccn = affine_expand_del(cc_diff, p, p0, yccn, yccgn)
    return v1ccn, v2ccn, xccgn, yccgn
end

"""
$(SIGNATURES)

Performs a single step of the parametric method associated with `t` assumes that
the inputs have been preconditioned.
"""
function contract! end

"""
$(TYPEDSIGNATURES)

Applies the Gauss-Siedel variant of the Newton type contractor.
"""
function contract!(t::NewtonGS, d::MCCallback{FH,FJ,C,PRE,N,T}, k::Int, b::Bool) where {FH, FJ, C, PRE, N, T<:RelaxTag}
    @unpack H, J0, J, p_mc, pref_mc, nx, x0_mc, x_mc, xz0, z_mc, use_apriori = d

    S = zero(MC{N,T})
    @. x0_mc = x_mc
    for i = 1:nx
        S = zero(MC{N,T})
        for j = 1:nx
            if i != j
                xv = @inbounds J[i,j]
                yv = @inbounds x_mc[j] - z_mc[j]
                zv = xv*yv
                if use_apriori
                    if b
                        J0[k][i,j] = xv
                        xz0[k][i,j] = yv
                    end
                    xr = J0[k][i,j]
                    yr = xz0[k][i,j]
                    u1max, u2max, v1nmax, v2nmax = estimator_extrema(xr, yr, p_ref)
                    wIntv = zv.Intv
                    if (u1max < xv.Intv.hi) || (u2max < yv.Intv.hi)
                        u1cv, u2cv, u1cvg, u2cvg = estimator_under(xr, yr, p_mc, p_ref)
                        za_l = mult_apriori_kernel(xv, yv, wIntv, u1cv, u2cv, u1max, u2max, u1cvg, u2cvg)
                        zv = zv ∩ za_l
                    end
                    if (v1nmax > -xv.Intv.lo) || (v2nmax > -yv.Intv.lo)
                        v1ccn, v2ccn, v1ccgn, v2ccgn = estimator_over(xr, yr, p_mc, p_ref)
                        za_u = mult_apriori_kernel(-xv, -yv, wIntv, v1ccn, v2ccn, v1nmax, v2nmax, v1ccgn, v2ccgn)
                        zv = zv ∩ za_u
                    end
                end
                S += zv
            end
        end
        @inbounds x_mc[i] = z_mc[i] - (H[i] + S)*McCormick.inv1(J[i,i], 1.0/J[i,i].Intv)
        @inbounds x_mc[i] = final_cut(x_mc[i], x0_mc[i])
    end
    return
end

"""
$(TYPEDSIGNATURES)

Applies the componentwise variant of the Krawczyk type contractor.
"""
function contract!(t::KrawczykCW, d::MCCallback{FH,FJ,C,PRE,N,T}, k::Int, b::Bool) where {FH, FJ, C, PRE, N, T<:RelaxTag}
    @unpack H, J0, J, p_mc, pref_mc, nx, x0_mc, x_mc, xz0, z_mc, use_apriori = d

    S = zero(MC{N,T})
    @. x0_mc = x_mc
    for i = 1:nx
        S = zero(MC{N,T})
        for j = 1:d.nx
            xv = (i != j) ? -J[i,j] : (one(MC{N,T}) - J[i,j])
            yv = @inbounds x_mc[j] - z_mc[j]
            zv = xv*yv
            if use_apriori
                if b
                    J0[k][i,j] = xv
                    xz0[k][i,j] = yv
                end
                xr = J0[k][i,j]
                yr = xz0[k][i,j]
                u1max, u2max, v1nmax, v2nmax = estimator_extrema(xr, yr, p_ref)
                wIntv = zv.Intv
                if (u1max < xv.Intv.hi) || (u2max < yv.Intv.hi)
                    u1cv, u2cv, u1cvg, u2cvg = estimator_under(xr, yr, p_mc, p_ref)
                    za_l = mult_apriori_kernel(xv, yv, wIntv, u1cv, u2cv, u1max, u2max, u1cvg, u2cvg)
                    zv = zv ∩ za_l
                end
                if (v1nmax > -xv.Intv.lo) || (v2nmax > -yv.Intv.lo)
                    v1ccn, v2ccn, v1ccgn, v2ccgn = estimator_over(xr, yr, p_mc, p_ref)
                    za_u = mult_apriori_kernel(-xv, -yv, wIntv, v1ccn, v2ccn, v1nmax, v2nmax, v1ccgn, v2ccgn)
                    zv = zv ∩ za_u
                end
            end
            S += zv
        end
        @inbounds x_mc[i] = z_mc[i] - H[i] + S
        @inbounds x_mc[i] = final_cut(x_mc[i], x0_mc[i])
    end
    return
end
