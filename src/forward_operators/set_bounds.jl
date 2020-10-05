
function cut_kernel(x::MC{N,T}, z::Interval{Float64}) where {N, T<:Union{NS,MV}}
    zL = z.lo
    zU = z.hi
    cvv, ccv, cv_grad, cc_grad = cut(zL, zU, cvv, ccv, cv_grad, cc_grad)
    return MC{N, T}(cvv, ccv, z, cv_grad, cc_grad, x.cnst)
end

for (f, ii) in ((:positive, :(Interval(MC_DOMAIN_TOL, Inf)))
          (:negative, :(Interval(-Inf, -MC_DOMAIN_TOL)))
          (:lower_bnd, :(Interval(lb, Inf)))
          (:upper_bnd, :(Interval(-Inf, ub)))
          (:bnd, :(Interval(lb, ub))))
    @eval ($f)(x) = x
    @eval function ($f)(x::MC{N,T}) where {N, T<:Union{NS,MV}}
        cut_kernel(x, x.Intv ∩ $ii)
    end
end
