macro temp_grad_min(c1, x1, y1, c2, x2, y2)
    sym_grad11 = Symbol(String(x1)*"_grad")
    sym_grad12 = Symbol(String(y1)*"_grad")
    sym_grad21 = Symbol(String(x2)*"_grad")
    sym_grad22 = Symbol(String(y2)*"_grad")
    esc(quote
        grad_temp = $c1*$x1 < $c1*$y1 ? $c1*$sym_grad11 : $c1*$sym_grad12
        grad_temp += $c2*$x2 < $c2*$y2 ? $c2*$sym_grad21 : $c2*$sym_grad22
        end)
end

macro temp_grad_max(c1, x1, y1, c2, x2, y2)
    sym_grad11 = Symbol(String(x1)*"_grad")
    sym_grad12 = Symbol(String(y1)*"_grad")
    sym_grad21 = Symbol(String(x2)*"_grad")
    sym_grad22 = Symbol(String(y2)*"_grad")
    esc(quote
        grad_temp = $c1*$x1 > $c1*$y1 ? $c1*$sym_grad11 : $c1*$sym_grad12
        grad_temp += $c2*$x2 > $c2*$y2 ? $c2*$sym_grad21 : $c2*$sym_grad22
        end)
end

min4(args...) = findmin(args)
max4(args...) = findmax(args)

const APRIORI_DELTA = 1E-6

function mult_apriori_kernel(x1::MC{N,T}, x2::MC{N,T}, z::Interval{Float64},
                             u1::Float64, u2::Float64, a1::Float64, a2::Float64,
                             u1grad::SVector{N,Float64}, u2grad::SVector{N,Float64}) where {N, T<:Union{NS, MV}}

    x1L = lo(x1);  x1U = hi(x1);  x2L = lo(x2);  x2U = hi(x2)
    x1Li = Interval(lo(x1));  x1Ui = Interval(hi(x1));  x2Li = Interval(lo(x2));  x2Ui = Interval(hi(x2))
    u1i = Interval(u1); u2i = Interval(u2)
    a1i = Interval(a1); a2i = Interval(a2)
    x1cv = x1.cv
    x1cc = x1.cc
    x2cv = x2.cv 
    x2cc = x2.cc
    x1i = Interval(x1.cv, x1.cc)
    x2i = Interval(x2.cv, x2.cc)
    x1cv_grad = x1.cv_grad;  x1cc_grad = x1.cc_grad;
    x2cv_grad = x2.cv_grad;  x2cc_grad = x2.cc_grad

    a2x1i = a2i*x1i
    a1x2i = a1i*x2i
    x1Ux2i = x1Ui*x2i
    x2Ux1i = x2Ui*x1i
    x1Lx2i = x1Li*x2i
    x2Lx1i = x2Li*x1i

    # additional constraints from underestimators (convex)
    w1cvi = (x2Ui - a2i)*u1i + (x1Ui - a1i)*u2i + lo(a2x1i) + lo(a1x2i) + a1i*a2i - a1i*x2Ui - a2i*x1Ui - APRIORI_DELTA
    w2cvi = (x2Ui - x2Li)*u1i + lo(x2Lx1i) + lo(a1x2i) - a1i*x2Ui - APRIORI_DELTA
    w3cvi = (x1Ui - x1Li)*u2i + lo(a2x1i) + lo(x1Lx2i) - x1Ui*a2i - APRIORI_DELTA
    w4cvi = (a2i - x2Li)*u1i + (a1i - x1Li)*u2i + lo(x2Lx1i) + lo(x1Lx2i) - a1i*a2i - APRIORI_DELTA

    w1cv = lo(w1cvi)
    w2cv = lo(w2cvi)
    w3cv = lo(w3cvi)
    w4cv = lo(w4cvi)

    # additional constraints from underestimators (concave)
    w1cci = (x2Li - a2i)*u1i + (a1i - x1Ui)*u2i + hi(a2x1i) + hi(x1Ux2i) - a1i*x2Li + APRIORI_DELTA
    w2cci = (x2Li - x2Ui)*u1i + hi(a1x2i) + hi(x2Ux1i) - x2Li*a1i + APRIORI_DELTA
    w3cci = (x1Li - x1Ui)*u2i + hi(a2x1i) + hi(x1Ux2i) - x1Li*a2i + APRIORI_DELTA
    w4cci = (a2i - x2Ui)*u1i + (x1Li - a1i)*u2i + hi(x2Ux1i) + hi(a1x2i) - x1Li*a2i + APRIORI_DELTA

    w1cc = hi(w1cci)
    w2cc = hi(w2cci)
    w3cc = hi(w3cci)
    w4cc = hi(w4cci)

    cv, cvind = max4(w1cv, w2cv, w3cv, w4cv)
    if cvind == 1
        @temp_grad_min(a2, x1cv, x1cc, a1, x2cv, x2cc)
        cv_grad = (x2U - a2)*u1grad + (x1U - a1)*u2grad + grad_temp
    elseif cvind == 2
        @temp_grad_min(x2L, x1cv, x1cc, a1, x2cv, x2cc)
        cv_grad = (x2U - x2L)*u2grad + grad_temp
    elseif cvind == 3
        @temp_grad_min(a2, x1cv, x1cc, x1L, x2cv, x2cc)
        cv_grad = (x1U - x1L)*u2grad + grad_temp
    elseif cvind == 4
        @temp_grad_min(x2L, x1cv, x1cc, x1L, x2cv, x2cc)
        cv_grad = (a2 - x2L)*u1grad + (a1 - x1L)*u2grad + grad_temp
    end

    cc, ccind = min4(w1cc, w2cc, w3cc, w4cc)
    if ccind == 1
        @temp_grad_max(a2, x1cv, x1cc, x1U, x2cv, x2cc)
        cc_grad = (x2L - a2)*u1grad + (a1 - x1U)*u2grad + grad_temp
    elseif ccind == 2
        @temp_grad_max(a1, x1cv, x1cc, x2U, x1cv, x1cc)
        cc_grad = (x2L - x2U)*u1grad + grad_temp
    elseif ccind == 3
        @temp_grad_max(a2, x1cv, x1cc, x1U, x2cv, x2cc)
        cc_grad = (x1L - x1U)*u2grad + grad_temp
    elseif ccind == 4
        @temp_grad_max(x2U, x1cv, x1cc, a1, x2cv, x2cc)
        cc_grad = (a2 - x2U)*u1grad + (x1L - a1)*u2grad + grad_temp
    end

    out = MC{N,T}(cv, cc, z, cv_grad, cc_grad, x1.cnst && x2.cnst)
    return out
end
