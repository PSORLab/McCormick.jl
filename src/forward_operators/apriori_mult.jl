
#=
@macro temp_grad_min(c1, x1, y1, c2, x2, y2)
    sym_grad11 = Symbol(String(x1)*"_grad")
    sym_grad12 = Symbol(String(y1)*"_grad")
    sym_grad21 = Symbol(String(x2)*"_grad")
    sym_grad22 = Symbol(String(y2)*"_grad")
    esc(begin
        grad_temp = c1*x1 < c1*y1 ? c1*$sym_grad11 : c1*$sym_grad12
        grad_temp += c2*x2 < c2*y2 ? c2*$sym_grad21 : c2*$sym_grad22
    end)
end

@macro temp_grad_max(c1, x1, y1, c2, x2, y2)
    sym_grad11 = Symbol(String(x1)*"_grad")
    sym_grad12 = Symbol(String(y1)*"_grad")
    sym_grad21 = Symbol(String(x2)*"_grad")
    sym_grad22 = Symbol(String(y2)*"_grad")
    esc(begin
        grad_temp = c1*x1 > c1*y1 ? c1*$sym_grad11 : c1*$sym_grad12
        grad_temp += c2*x2 > c2*y2 ? c2*$sym_grad21 : c2*$sym_grad22
    end)
end

@inline function mult_kernel_plus(x1::MC{N,T}, x2::MC{N,T}, z::Interval{Float64},
                                  u1::Float64, u2::Float64, a1::Float64, a2::Float64,
                                  v1::Float64, v2::Float64, b1::Float64, b2::Float64,
                                  u1grad::SVector{N,Float64}, u2grad::SVector{N,Float64},
                                  v1cvgrad::SVector{N,Float64}, v2cvgrad::SVector{N,Float64}) where {N, T<:Union{NS, MV}

    x1L = lo(x1); x1U = hi(x1); x2L = lo(x2); x2U = hi(x2)
    x1cv = x1.cv; x1cc = x1.cc; x2cv = x2.cv; x2cc = x.cc

    w = x1*x2
    w0cv = w.cv
    w0cc = w.cc

    # additional constraints from underestimators (convex)
    w1cv = (x2U - a2)*u1 + (x1U - a1)*u2 + max(a2*x1cv, a2*x1cc) + max(a1*x2cv, a1*x2cc) + a1*a2 + -a1*x2U - a2*x1U
    w2cv = (x2U - x2L)*u1 + max(x2L*x1cv, x2L*x1cc) + max(a1*x2cv, a1*x2cc) - a1*x2U
    w3cv = (x1U - x1L)*u2 + max(a2*x1cv, a2*x1cc) + max(x1L*x2cv, x1L*x2cc) - x1U*a2
    w4cv = (a2 - x2L)*u1 + (a1 - x1L)*u2 + max(x2L*x1cv, x2L*x1cc) + max(x1L*x2cv, x1L*x2cc) - a1*a2

    # additional constraints from underestimators (concave)
    w1cc = (x2L - a2)*u1 + (a1 - x1U)*u2 + min(a2*x1cv, a2*x1cc) + min(x1U*x2cv, x1U*x2cc) - a1*x2L
    w2cc = (x2L - x2U)*u1 + min(a1*x2cv, a1*x2cc) + min(x2U*x1cv, x2U*x1cc) - x2L*a1
    w3cc = (x1L - x1U)*u2 + min(a2*x1cv, a2*x1cc) + min(x1U*x2cv, x1U*x2cc) - x1L*a2
    w4cc = (a2 - x2U)*u1 + (x1L - a1)*u2 + min(x2U*x1cv, x2U*x1cc) + min(a1*x2cv, a1*x2cc) - x1L*a2

    # additional constraints from overestimators (convex)
    w5cv = (xL2 - b2)*v1 + (xL1 - b1)*v2 + max(b2*x1cv, b2*x1cc) + max(x1L*x2cv, x1L*x2cc) + b1*b2 - b1*x2L - x1*b2
    w6cv = (x2L - x2U)*v1 + max(x2U*x1cv, x2U*x1cc) + max(b1*x2cv, b1*x2cc) - b1*x2L
    w7cv = (x1L - x1U)*v2 + max(b2*x1cv, b2*x1cc) + max(x1U*x2cv, x1U*x2cc) - b2*x1L
    w8cv = (b2 - x2U)*v1 + (b1 - x1U)*v2 + max(x2U*x1cv, x2U*x1cc) + max(x1U*x2cv, x1U*x2cc) - b1*b2

    # additional constraints from overestimators (concave)
    w5cc = (x2U - b2)*v1 + (b1 - x1L)*v2 + min(b2*x1cv, b2*x1cc) + min(x1L*x2cv, x1L*x2cc) - b1*x2U
    w6cc = (x2U - x2L)*v1 + min(b1*x2cv, b1*x2cc) + min(x2L*x1cv, x2L*x1cc) - x1U*b1
    w7cc = (x1U - x1L)*v2 + min(b2*x1cv, b2*x1cc) + min(x1L*x2cv, x1L*x2cc) - x1U*b2
    w8cc = (b2 - x2l)*v1 + (x1U - b1)*v2 + min(x2L*x1cv, x2L*x1cc) + min(b1*x2cv, b1*x2cc) - x1U*b2

    cv, cvind = max9(w0cv, w1cv, w2cv, w3cv, w4cv, w5cv, w6cv, w7cv, w8cv)
    if cvind == 1
        cv_grad = w.cv_grad
    elseif cvind == 2
        @temp_grad_max(:a2, :x1cv, :x1cc, :a1, :x2cv, :x2cc)
        cv_grad = (x2U - a2)*u2grad + (x1U - a1)*u2grad + grad_temp
    elseif cvind == 3
        @temp_grad_max(:x2L, :x1cv, :x1cc, :a1, :x2cv, :x2cc)
        cv_grad = (x2U - x2L)*u2grad + grad_temp
    elseif cvind == 4
        @temp_grad_max(:a2, :x1cv, :x1cc, :x1L, :x2cv, :x2cc)
        cv_grad = (x1U - x1L)*u2grad + grad_temp
    elseif cvind == 5
        @temp_grad_max(:x2L, :x1cv, :x1cc, :x1L, :x2cv, :x2cc)
        cv_grad = (a2 - x2L)*u1grad + (a1 - x1L)*u2grad + grad_temp
    elseif cvind == 6
        @temp_grad_max(:b2, :x1cv, :x1cc, :x1L, :x2cv, :x2cc)
        cv_grad = (xL2 - b2)*v1grad + (xL1 - b1)*v2grad + grad_temp
    elseif cvind == 7
        @temp_grad_max(:x2U, :x1cv, :x1cc, :b1, :x2cv, :x2cc)
        cv_grad = (x2L - x2U)*v1grad + grad_temp
    elseif cvind == 8
        @temp_grad_max(:b2, :x1cv, :x1cc, :b1, :x2cv, :x2cc)
        cv_grad = (x1L - x1U)*v2grad + grad_temp
    elseif cvind == 9
        @temp_grad_max(:x2U, :x1cv, :x1cc, :x1U, :x2cv, :x2cc)
        cv_grad = (b2 - x2U)*v1grad + (b1 - x1U)*v2grad + grad_temp
    end

    cc, ccind = min9(w0cc, w1cc, w2cc, w3cc, w4cc, w5cc, w6cc, w7cc, w8cc)
    if ccind == 1
        cc_grad = w.cc_grad
    elseif ccind == 2
        @temp_grad_min(:a2, :x1cv, :x1cc, :x1U, :x2cv, :x2cc)
        cc_grad = (x2L - a2)*u1grad + (a1 - x1U)*u2grad + grad_temp
    elseif ccind == 3
        @temp_grad_min(:a1, :x1cv, :x1cc, :x2U, :x1cv, :x1cc)
        cc_grad = (x2L - x2U)*u1grad + grad_temp
    elseif ccind == 4
        @temp_grad_min(:a2, :x1cv, :x1cc, :x1U, :x2cv, :x2cc)
        cc_grad = (x1L - x1U)*u2grad + grad_temp
    elseif ccind == 5
        @temp_grad_min(:x2U, :x1cv, :x1cc, :a1, :x2cv, :x2cc)
        cc_grad = (a2 - x2U)*u1grad + (x1L - a1)*u2grad + grad_temp
    elseif ccind == 6
        @temp_grad_min(:b2, :x1cv, :x1cc, :x1L, :x2cv, :x2cc)
        cc_grad = (x2U - b2)*v1grad + (b1 - x1L)*v2grad + grad_temp
    elseif ccind == 7
        @temp_grad_min(:b2, :x2cv, :x2cc, :x2L, :x1cv, :x1cc)
        cc_grad = (x2U - x2L)*v1grad + grad_temp
    elseif ccind == 8
        @temp_grad_min(:b2, :x1cv, :x1cc, :x1L, :x2cv, :x2cc)
        cc_grad = (x1U - x1L)*v2grad + grad_temp
    elseif ccind == 9
        @temp_grad_min(:x2L, :x1cv, :x1cc, :b1, :x2cv, :x2cc)
        cc_grad = (b2 - x2l)*v1grad + (x1U - b1)*v2grad + grad_temp
    end

    cv, cc, cv_grad, cc_grad = cut(z.lo, z.hi, cv, cc, cv_grad, cc_grad)
    MC{N,T}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
=#
