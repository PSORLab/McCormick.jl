
#=
@inline function mult_kernel_plus(x1::MC{N,NS}, x2::MC{N,NS}, z::Interval{Float64},
                                  u1cv::Float64, u2cv::Float64, a1cv::Float64, a2cv::Float64,
                                  u1cvgrad::SVector{N,Float64}, u2cvgrad::SVector{N,Float64}) where N

    x1L = lo(x1); x1U = hi(x1); x2L = lo(x2); x2U = hi(x2)
    x1cv = x1.cv; x1cc = x1.cc; x2cv = x2.cv; x2cc = x.cc

    w = x1*x2
    w0cv = w.cv
    w0cc = w.cc

    # additional constraints from underestimators (convex)
    w1cv = (x2U - a2cv)*u1cv + (x1U - a1cv)*u2cv + max(a2cv*x1cv, a2cv*x1cc) + a1cv*x2cv + a1cv*a2cv + -a1cv*x2U - a2cv*x1U
    w2cv = (x2U - x2L)*u1cv + max(x2L*x1cv, x2L*x1cc) + max(a1*u2cv, a1*u2cc) - a1cv*x2U
    w3cv = (x1U - x1L)*u2cv + max(a2*x1cv, a2*x1cc) + max(x1L*x2cv, x1L*x2cc) - x1U*a2cv
    w4cv = (a2cv - x2L)*u1cv + (a1cv - x1L)*u2cv + max(x2L*x1cv, x2L*x1cc) + max(x1L*x2cv, x1L*x2cc) - a1cv*a2cv

    # additional constraints from underestimators (concave)
    w1cc = (x2L - a2cv)*u1cv + ()*u2cv
    w2cc = ()*u1cc
    w3cc = ()*u2cc
    w4cc = (a2cc - x2U)*u1cv + (x1L - a1cc)*u2cv + min(x2U*x1cc, x2U*x1cv) + min(a1cc*x2cc, a1cc*x2cv) - x1L*a2cc

    # additional constraints from overestimators (convex)
    w5cv =
    w6cv =
    w7cv =
    w8cv =

    # additional constraints from overestimators (concave)
    w5cc =
    w6cc =
    w7cc =
    w8cc =

    cv, cvind = max9(w0cv, w1cv, w2cv, w3cv, w4cv, w5cv, w6cv, w7cv, w8cv)
    cc, ccind = min9(w0cc, w1cc, w2cc, w3cc, w4cc, w5cc, w6cc, w7cc, w8cc)


    #=
    cvsubax = coeff6(cvind, cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6)
    cvsubay = coeff6(cvind, cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6)
    cvsubaz = coeff6(cvind, cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6)
    cv_grad = cvsubax*ifelse(cvsubax > 0.0, x.cv_grad, x.cc_grad) +
              cvsubay*ifelse(cvsubay > 0.0, y.cv_grad, y.cc_grad) +
              cvsubaz*ifelse(cvsubaz > 0.0, z.cv_grad, z.cc_grad)

    cvsubax = coeff6(cvind, cv_ax1, cv_ax2, cv_ax3, cv_ax4, cv_ax5, cv_ax6)
    cvsubay = coeff6(cvind, cv_ay1, cv_ay2, cv_ay3, cv_ay4, cv_ay5, cv_ay6)
    cvsubaz = coeff6(cvind, cv_az1, cv_az2, cv_az3, cv_az4, cv_az5, cv_az6)
    cv_grad = cvsubax*ifelse(cvsubax > 0.0, x.cv_grad, x.cc_grad) +
              cvsubay*ifelse(cvsubay > 0.0, y.cv_grad, y.cc_grad) +
              cvsubaz*ifelse(cvsubaz > 0.0, z.cv_grad, z.cc_grad)
    =#

    cv, cc, cv_grad, cc_grad = cut(z.lo, z.hi, cv, cc, cv_grad, cc_grad)
end
=#
