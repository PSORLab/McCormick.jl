@testset "Access Functions, Constructors, Utilities" begin
   x_exp = MC{2,NS}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   @test McCormick.Intv(x_exp) == x_exp.Intv
   @test McCormick.lo(x_exp) == x_exp.Intv.lo
   @test McCormick.hi(x_exp) == x_exp.Intv.hi
   @test McCormick.cc(x_exp) == x_exp.cc
   @test McCormick.cv(x_exp) == x_exp.cv
   @test McCormick.cc_grad(x_exp) == x_exp.cc_grad
   @test McCormick.cv_grad(x_exp) == x_exp.cv_grad
   @test McCormick.cnst(x_exp) == x_exp.cnst
   @test McCormick.length(x_exp) == length(x_exp.cc_grad)

   xcnst = MC{2,Diff}(1)
   @test xcnst.cv == 1.0
   @test xcnst.cc == 1.0

   nanMC = nan(x_exp)
   @test isnan(nanMC.cv) && isnan(nanMC.cc)

   @test MC{2,NS}(1.0) == MC{2,NS}(Interval{Float64}(1.0))
   @test MC{2,NS}(pi) == MC{2,NS}(Interval{Float64}(pi))

   xintv = Interval{Float64}(1.0,4.0)
   @test McCormick.lo(xintv) == xintv.lo
   @test McCormick.hi(xintv) == xintv.hi

   @test McCormick.mid_grad(zeros(SVector{2,Float64}), zeros(SVector{2,Float64}), 3) == zeros(SVector{2,Float64})

   f1(x) = 1.0
   df1(x) = 2.0
   @test (1.0,2.0) == McCormick.dline_seg(f1, df1, 0.0, 0.0, 0.0)

   f2(x,c) = 1.1
   df2(x,c) = 2.1
   @test (1.1,2.1) == McCormick.dline_seg(f2, df2, 0.0, 0.0, 0.0, 2)

   f3(x,c) = 1.2
   df3(x,c) = 2.2
   @test (1.2,2.2) == McCormick.dline_seg(f3, df3, 0.0, 0.0, 0.0, 2.1)

   @test empty(MC{2,NS}(Interval{Float64}(1.0))) == MC{2,NS}(Inf, -Inf, Interval{Float64}(Inf,-Inf),
                                                     SVector{2,Float64}(zeros(Float64,2)),
                                                     SVector{2,Float64}(zeros(Float64,2)), false)

   @test promote_rule(MC{2,NS}, Float16) == MC{2,NS}
   @test promote_rule(MC{2,NS}, Int32) == MC{2,NS}
   @test promote_rule(MC{2,NS}, Irrational{:π}) == MC{2,NS}
end

@testset "Rootfinding Routine" begin
   xL = 0.0
   xU = 2.0
   f(x,xlo,xup) = x^2 - 1.0
   out = McCormick.golden_section(xL, xU, f, 0.0, 0.0)
   @test out == 1.0
end

if ~(VERSION < v"1.1-")
   @testset "Test Error Functions" begin
      x_erf_p = MC{2,Diff}(0.3, 0.3, Interval{Float64}(0.2, 0.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
      x_erf_n = MC{2,Diff}(-0.3, -0.3, Interval{Float64}(-0.5, -0.2), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
      x_erf_z1 = MC{2,Diff}(-0.3, -0.3, Interval{Float64}(-0.5, 0.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
      x_erf_z2 = MC{2,Diff}(0.3, 0.3, Interval{Float64}(-0.5, 0.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

      yerfp = erf(x_erf_p)
      yerfn = erf(x_erf_n)
      yerfz1 = erf(x_erf_z1)
      yerfz2 = erf(x_erf_z2)

      @test isapprox(yerfp.cv, 0.3219683520780011, atol=1E-8)
      @test isapprox(yerfp.cc, 0.3286267594591274, atol=1E-8)
      @test isapprox(yerfn.cv, -0.3286267594591274, atol=1E-8)
      @test isapprox(yerfn.cc, -0.3219683520780011, atol=1E-8)
      @test isapprox(yerfz1.cv, -0.3286267594591274, atol=1E-8)
      @test isapprox(yerfz1.cc, -0.30801138150037066, atol=1E-8)
      @test isapprox(yerfz2.cv, 0.3080113815003707, atol=1E-8)
      @test isapprox(yerfz2.cc, 0.3286267594591274, atol=1E-8)

      yerfp = erfinv(x_erf_p)
      yerfn = erfinv(x_erf_n)
      yerfz1 = erfinv(x_erf_z1)
      yerfz2 = erfinv(x_erf_z2)

      @test isapprox(yerfp.cv, 0.2724627147267543, atol=1E-8)
      @test isapprox(yerfp.cc, 0.278407728482351, atol=1E-8)
      @test isapprox(yerfn.cv, -0.278407728482351, atol=1E-8)
      @test isapprox(yerfn.cc, -0.2724627147267543, atol=1E-8)
      @test isapprox(yerfz1.cv, -0.2896752187685166, atol=1E-8)
      @test isapprox(yerfz1.cc, -0.2724627147267543, atol=1E-8)
      @test isapprox(yerfz2.cv, 0.2724627147267543, atol=1E-8)
      @test isapprox(yerfz2.cc, 0.28967521876851665, atol=1E-8)

      yerfp = erfinv(x_erf_p)
      yerfn = erfinv(x_erf_n)
      yerfz1 = erfinv(x_erf_z1)
      yerfz2 = erfinv(x_erf_z2)

      @test isapprox(yerfp.cv, 0.2724627147267543, atol=1E-8)
      @test isapprox(yerfp.cc, 0.278407728482351, atol=1E-8)
      @test isapprox(yerfn.cv, -0.278407728482351, atol=1E-8)
      @test isapprox(yerfn.cc, -0.2724627147267543, atol=1E-8)
      @test isapprox(yerfz1.cv, -0.2896752187685166, atol=1E-8)
      @test isapprox(yerfz1.cc, -0.2724627147267543, atol=1E-8)
      @test isapprox(yerfz2.cv, 0.2724627147267543, atol=1E-8)
      @test isapprox(yerfz2.cc, 0.28967521876851665, atol=1E-8)

      yerfp = erfinv(x_erf_p)
      yerfn = erfinv(x_erf_n)
      yerfz1 = erfinv(x_erf_z1)
      yerfz2 = erfinv(x_erf_z2)

      @test isapprox(yerfp.cv, 0.2724627147267543, atol=1E-8)
      @test isapprox(yerfp.cc, 0.278407728482351, atol=1E-8)
      @test isapprox(yerfn.cv, -0.278407728482351, atol=1E-8)
      @test isapprox(yerfn.cc, -0.2724627147267543, atol=1E-8)
      @test isapprox(yerfz1.cv, -0.2896752187685166, atol=1E-8)
      @test isapprox(yerfz1.cc, -0.2724627147267543, atol=1E-8)
      @test isapprox(yerfz2.cv, 0.2724627147267543, atol=1E-8)
      @test isapprox(yerfz2.cc, 0.28967521876851665, atol=1E-8)

      x1 = MC{1,NS}(0.1, Interval(-1.5, 1.5), 1)
      x2 = MC{1,NS}(-0.5, Interval(-1.5, 1.5), 1)
      x3 = MC{1,NS}(0.7, Interval(0.5, 1.5), 1)

      x4 = MC{1,NS}(-6.5, Interval(-7.5, -5.5), 1)
      x5 = MC{1,NS}(6.5, Interval(5.5, 7.5), 1)
      x6 = MC{1,NS}(0.7, Interval(0.5, 1.5), 1)

      x7 = MC{1,NS}(3.1, Interval(-3.5, 3.5), 1)
      x8 = MC{1,NS}(1.1, Interval(-2.0, 2.5), 1)
      x9 =  MC{1,NS}(0.3, Interval(-2.0, 3.5), 1)

      x10 = MC{1,NS}(-1.1, Interval(-1.3, 1.3), 1)
      x11 = MC{1,NS}(0.9, Interval(-0.8, 1.1), 1)
      x12 =  MC{1,NS}(0.3, Interval(-0.8, 1.3), 1)

      y1 = gelu(x1)
      y2 = gelu(x2)
      y3 = gelu(x3)

      y4 = gelu(x4)
      y5 = gelu(x5)
      y6 = gelu(x6)

      y7 = gelu(x7)
      y8 = gelu(x8)
      y9 = gelu(x9)

      y10 = gelu(x10)
      y11 = gelu(x11)
      y12 = gelu(x12)

      @test isapprox(gelu(7), 6.999999999991041, atol=1E-8)
      @test isapprox(gelu(-0.75), -0.16997051428265114, atol=1E-8)

      @test isapprox(y1.cv, 0.053982783727702904, atol=1E-8)
      @test isapprox(y1.cc, 0.6997891980967129, atol=1E-8)
      @test isapprox(y2.cv, -0.15426876936299344, atol=1E-8)
      @test isapprox(y2.cc, 0.3997891980967129, atol=1E-8)
      @test isapprox(y3.cv, 0.5306254434438489, atol=1E-8)
      @test isapprox(y3.cc, 0.5565428241289478, atol=1E-8)

      @test isapprox(y4.cv, -5.222141651051171e-8, atol=1E-8)
      @test isapprox(y4.cc, -2.610399119085116e-10, atol=1E-8)
      @test isapprox(y5.cv, 6.4999999477785835, atol=1E-8)
      @test isapprox(y5.cc, 6.49999999973896, atol=1E-8)
      @test isapprox(y6.cv, 0.5306254434438489, atol=1E-8)
      @test isapprox(y6.cc, 0.5565428241289478, atol=1E-8)

      @test isapprox(y7.cv, 3.073809840032428, atol=1E-8)
      @test isapprox(y7.cc, 3.299185798223376, atol=1E-8)
      @test isapprox(y8.cv, 0.9505770065222403, atol=1E-8)
      @test isapprox(y8.cc, 1.697372160948963, atol=1E-8)
      @test isapprox(y9.cv, 0.18537342665668577, atol=1E-8)
      @test isapprox(y9.cc, 1.4368229984446212, atol=1E-8)

      @test isapprox(y10.cv, -0.14923266704102095, atol=1E-8)
      @test isapprox(y10.cc, -0.02584062996129348, atol=1E-8)
      @test isapprox(y11.cv, 0.7343458871879165, atol=1E-8)
      @test isapprox(y11.cc, 0.8328461064510111, atol=1E-8)
      @test isapprox(y12.cv, 0.18537342665668577, atol=1E-8)
      @test isapprox(y12.cc, 0.5343290419885047, atol=1E-8)
   end
end

@testset "Test Univariate" begin

   mctol = 1E-4

   ##### Exponent #####
   x_exp = MC{2,NS}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_exp2 = MC{2,NS}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_exp10 = MC{2,NS}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_expm1 = MC{2,NS}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_exp = MC{2,NS}(7.38906, 37.3049, Interval{Float64}(2.71828, 54.5982), SVector{2,Float64}([7.38906, 0.0]), SVector{2,Float64}([17.2933, 0.0]), false)
   yref_exp2 = MC{2,NS}(4.0, 11.33333333333333, Interval{Float64}(1.99999, 16.0001), SVector{2,Float64}([2.77259, 0.0]), SVector{2,Float64}([4.66667, 0.0]), false)
   yref_exp10 = MC{2,NS}(100.0, 6670.0000000000055, Interval{Float64}(9.999999999999999999, 10000.00000000001), SVector{2,Float64}([230.25850929940458, 0.0]), SVector{2,Float64}([3330.0, 0.0]), false)
   yref_expm1 = MC{2,NS}(6.38905609893065, 36.304860631582514, Interval{Float64}(1.71828, 53.5982), SVector{2,Float64}([7.38906, 0.0]), SVector{2,Float64}([17.2933, 0.0]), false)

   @test check_vs_ref1(exp, x_exp, yref_exp, mctol)
   @test check_vs_ref1(exp2, x_exp2, yref_exp2, mctol)
   @test check_vs_ref1(exp10, x_exp10, yref_exp10, mctol)
   @test check_vs_ref1(expm1, x_expm1, yref_expm1, mctol)

   ##### Logarithm #####
   x_log = MC{2,NS}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_log2 = MC{2,NS}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_log10 = MC{2,NS}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_log1p = MC{2,NS}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_log = MC{2,NS}(0.46209812037329695,1.09861, Interval{Float64}(0, 1.38629), SVector{2,Float64}([0.462098, 0.0]), SVector{2,Float64}([0.3333333, 0.0]), false)
   yref_log2 = MC{2,NS}(0.6666666666666666, 1.584962500721156, Interval{Float64}(0, 2), SVector{2,Float64}([0.666667, 0.0]), SVector{2,Float64}([0.4808983469629878, 0.0]), false)
   yref_log10 = MC{2,NS}(0.20068666377598746, 0.47712125471966244, Interval{Float64}(0, 0.60206), SVector{2,Float64}([0.200687, 0.0]), SVector{2,Float64}([0.14476482730108392, 0.0]), false)
   yref_log1p = MC{2,NS}(0.998577424517997, 1.3862943611198906, Interval{Float64}(0.693147, 1.60944), SVector{2,Float64}([0.30543, 0.0]), SVector{2,Float64}([0.25, 0.0]), false)

   @test check_vs_ref1(log, x_log, yref_log, mctol)
   @test check_vs_ref1(log2, x_log2, yref_log2, mctol)
   @test check_vs_ref1(log10, x_log10, yref_log10, mctol)
   @test check_vs_ref1(log1p, x_log1p, yref_log1p, mctol)

   #####  Square root #####
   x_sqrt_ns = MC{2,NS}(4.5, 4.5, Interval{Float64}(3.0,9.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_sqrt_d1 = MC{2,Diff}(4.0, 4.0, Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_sqrt_ns = MC{2,NS}(2.049038105676658, 2.1213203435596424, Interval{Float64}(1.73205, 3.0), SVector{2,Float64}([0.211325, 0.0]), SVector{2,Float64}([0.235702, 0.0]), false)
   yref_sqrt_d1 = MC{2,Diff}(1.9604759334428057, 2.0, Interval{Float64}(1.73205, 2.64576), SVector{2,Float64}([0.228425, 0.0]), SVector{2,Float64}([0.25, 0.0]), false)

   @test check_vs_ref1(sqrt, x_sqrt_ns, yref_sqrt_ns, mctol)
   @test check_vs_ref1(sqrt, x_sqrt_d1, yref_sqrt_d1, mctol)

   #####  Step #####
   x_step_p_ns = MC{2,NS}(4.0, 4.0, Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_step_n_ns = MC{2,NS}(-4.0, -4.0, -Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_step_z_ns = MC{2,NS}(-2.0, -2.0, Interval{Float64}(-3.0,1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_step_zp_ns = MC{2,NS}(0.5, 0.5, Interval{Float64}(-3.0,1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   x_step_d1_p = MC{2,Diff}(4.0, 4.0, Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_step_d1_n = MC{2,Diff}(-4.0, -4.0, -Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_step_d1_z = MC{2,Diff}(-2.0, -2.0, Interval{Float64}(-3.0,1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_step_p = MC{2,NS}(1.0, 1.0, Interval{Float64}(1.0, 1.0), @SVector[0.0, 0.0], @SVector[0.0, 0.0], false)
   yref_step_n = MC{2,NS}(0.0, 0.0, Interval{Float64}(0.0, 0.0), @SVector[0.0, 0.0], @SVector[0.0, 0.0], false)
   yref_step_ns_z = MC{2,NS}(0.0, 0.3333333333333333, Interval{Float64}(0.0, 1.0), @SVector[0.0, 0.0], @SVector[-0.6666666666666666, 0.0], false)
   yref_step_ns_zp = MC{2,NS}(0.5, 1.0, Interval{Float64}(0.0, 1.0), @SVector[1.0, 0.0], @SVector[0.0, 0.0], false)

   yref_step_d1_p = MC{2,Diff}(1.0, 1.0, Interval{Float64}(1.0, 1.0), @SVector[0.0, 0.0], @SVector[0.0, 0.0], false)
   yref_step_d1_n = MC{2,Diff}(0.0, 0.0, Interval{Float64}(0.0, 0.0), @SVector[0.0, 0.0], @SVector[0.0, 0.0], false)
   yref_step_d1_z = MC{2,Diff}(0.0, 0.5555555555555556, Interval{Float64}(0.0, 1.0), @SVector[0.0, 0.0], @SVector[0.444444, 0.0], false)

   @test check_vs_ref1(step, x_step_p_ns, yref_step_p, mctol)
   @test check_vs_ref1(step, x_step_n_ns, yref_step_n, mctol)
   @test check_vs_ref1(step, x_step_z_ns, yref_step_ns_z, mctol)
   @test check_vs_ref1(step, x_step_zp_ns, yref_step_ns_zp, mctol)

   @test check_vs_ref1(step, x_step_d1_p, yref_step_d1_p, mctol)
   @test check_vs_ref1(step, x_step_d1_n, yref_step_d1_n, mctol)
   @test check_vs_ref1(step, x_step_d1_z, yref_step_d1_z, mctol)

   #####  Absolute value #####
   x_abs_ns = MC{2,NS}(4.5, 4.5, Interval{Float64}(-3.0,8.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_abs_d1 = MC{2,Diff}(2.0, 2.0, Interval{Float64}(-5.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_abs_ns = MC{2,NS}(4.5, 6.409090909090908, Interval{Float64}(0.0, 8.0), @SVector[1.0, 0.0], @SVector[0.454545, 0.0], false)
   yref_abs_d1 = MC{2,Diff}(0.5714285714285714, 6.166666666666667, Interval{Float64}(0.0, 7.0), @SVector[0.5714285714285714, 0.0], @SVector[0.16666666666666666, 0.0], false)

   @test McCormick.cv_abs(-1.0, -2.0, 3.0) == (0.5,-1.0)
   @test check_vs_ref1(abs, x_abs_ns, yref_abs_ns, mctol)
   @test check_vs_ref1(abs, x_abs_d1, yref_abs_d1, mctol)

   #####  Sign #####
   x_sign_d1_p = MC{2,Diff}(4.0, 4.0, Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_sign_d1_n = MC{2,Diff}(-4.0, -4.0, Interval{Float64}(-7.0,-3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_sign_d1_z = MC{2,Diff}(-2.0, -2.0, Interval{Float64}(-3.0,1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_sign_d1_p = MC{2,Diff}(1.0, 1.0, Interval{Float64}(1.0, 1.0), @SVector[0.0, 0.0], @SVector[0.0, 0.0], false)
   yref_sign_d1_n = MC{2,Diff}(-1.0, -1.0, Interval{Float64}(-1.0, -1.0), @SVector[0.0, 0.0], @SVector[0.0, 0.0], false)
   yref_sign_d1_z = MC{2,Diff}(-1.0, 0.11111111111111116, Interval{Float64}(-1.0, 1.0), @SVector[0.0, 0.0], @SVector[0.888889, 0.0], false)

   @test check_vs_ref1(sign, x_sign_d1_p, yref_sign_d1_p, mctol)
   @test check_vs_ref1(sign, x_sign_d1_n, yref_sign_d1_n, mctol)
   @test check_vs_ref1(sign, x_sign_d1_z, yref_sign_d1_z, mctol)

   #####  Sine #####
   x_sin_p = MC{2,NS}(4.0, 4.0, Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_sin_n = MC{2,NS}(-4.0, -4.0, Interval{Float64}(-7.0,-3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_sin_z = MC{2,NS}(-2.0, -2.0, Interval{Float64}(-3.0,1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_sin_d1_z = MC{2,Diff}(-2.0, -2.0, Interval{Float64}(-3.0,1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_sin_ns_p = MC{2,NS}(-0.7568024953079283, 0.2700866557245978, Interval{Float64}(-1.0, 0.656987), @SVector[-0.653644, 0.0], @SVector[0.128967, 0.0], false)
   yref_sin_ns_n = MC{2,NS}(-0.2700866557245979, 0.7568024953079283, Interval{Float64}(-0.656987, 1.0), @SVector[0.128967, 0.0], @SVector[-0.653644, 0.0], false)
   yref_sin_ns_z = MC{2,NS}(-0.9092974268256817, 0.10452774015707458, Interval{Float64}(-1.0, 0.841471), @SVector[-0.416147, 0.0], @SVector[0.245648, 0.0], false)
   yref_sin_d1_z = MC{2,Diff}(-0.9092974268256817, 0.10452774015707458, Interval{Float64}(-1.0, 0.841471), @SVector[-0.416147, 0.0], @SVector[0.245648, 0.0], false)

   @test check_vs_ref1(sin, x_sin_p, yref_sin_ns_p, mctol)
   @test check_vs_ref1(sin, x_sin_n, yref_sin_ns_n, mctol)
   @test check_vs_ref1(sin, x_sin_z, yref_sin_ns_z, mctol)
   @test check_vs_ref1(sin, x_sin_d1_z, yref_sin_d1_z, mctol)

   ysinpi = sinpi(x_sin_z)
   @test isapprox(ysinpi.cv, -1.0, atol=1E-8)
   @test isapprox(ysinpi.cc, 0.682459570501031, atol=1E-8)

   #####  Cosine #####
   x_cos_p = MC{2,Diff}(4.0, 4.0, Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_cos_n = MC{2,Diff}(-4.0, -4.0, Interval{Float64}(-7.0,-3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_cos_z = MC{2,Diff}(-2.0, -2.0, Interval{Float64}(-3.0,1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_cos_z_ns = MC{2,NS}(-2.0, -2.0, Interval{Float64}(-3.0,1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_cos_d1_p = MC{2,Diff}(-0.703492113936536, -0.31034065427934965, Interval{Float64}(-1.0, 1.0), @SVector[0.485798, 0.0], @SVector[0.679652, 0.0], false)
   yref_cos_d1_n = MC{2,Diff}(-0.703492113936536, -0.31034065427934965, Interval{Float64}(-1.0, 1.0), @SVector[-0.485798, 0.0], @SVector[-0.679652, 0.0], false)
   yref_cos_d1_z = MC{2,Diff}(-0.6314158569813042, -0.222468094224762, Interval{Float64}(-0.989993, 1.0), @SVector[0.390573, 0.0], @SVector[0.76752, 0.0], false)
   yref_cos_ns_z = MC{2,NS}(-0.6314158569813042, -0.222468094224762, Interval{Float64}(-0.989993, 1.0), @SVector[0.390573, 0.0], @SVector[0.76752, 0.0], false)

   @test check_vs_ref1(cos, x_cos_p, yref_cos_d1_p, mctol)
   @test check_vs_ref1(cos, x_cos_n, yref_cos_d1_n, mctol)
   @test check_vs_ref1(cos, x_cos_z, yref_cos_d1_z, mctol)
   @test check_vs_ref1(cos, x_cos_z_ns, yref_cos_ns_z, mctol)

   ycospi = cospi(x_sin_z)
   @test isapprox(ycospi.cv, -1.0, atol=1E-8)
   @test isapprox(ycospi.cc, 1.0, atol=1E-8)

   ##### Tangent #####
   x_tan_p = MC{2,NS}(0.6, 0.6, Interval{Float64}(0.5,1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_tan_n = MC{2,NS}(-0.8, -0.8, Interval{Float64}(-1.0,-0.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_tan_z = MC{2,NS}(-0.3,-0.3, Interval{Float64}(-0.5,0.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_tan_d1_z = MC{2,Diff}(-0.3,-0.3, Interval{Float64}(-0.5,0.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_tan_err = MC{2,Diff}(0.6, 0.6, Interval{Float64}(-4.5, 5.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_tan_ns_p = MC{2,NS}(0.6841368083416923, 0.7485235368060128, Interval{Float64}(0.546302, 1.55741), @SVector[1.46804, 0.0], @SVector[2.02221, 0.0], false)
   yref_tan_ns_n = MC{2,NS}(-1.1529656307304577, -1.0296385570503641, Interval{Float64}(-1.55741, -0.546302), @SVector[2.02221, 0.0], @SVector[2.06016, 0.0], false)
   yref_tan_ns_z = MC{2,NS}(-0.332534, -0.30933624960962325, Interval{Float64}(-0.546303,0.546303), @SVector[1.06884, 0.0], @SVector[1.09569, 0.0], false)
   yref_tan_d1_z = MC{2,Diff}(-0.332534, -0.309336, Interval{Float64}(-0.546303,0.546303), @SVector[1.06884, 0.0], @SVector[1.09569, 0.0], false)

   @test check_vs_ref1(tan, x_tan_p, yref_tan_ns_p, mctol)
   @test check_vs_ref1(tan, x_tan_n, yref_tan_ns_n, mctol)
   @test check_vs_ref1(tan, x_tan_z, yref_tan_ns_z, mctol)
   @test check_vs_ref1(tan, x_tan_d1_z, yref_tan_d1_z, mctol)

   #####  Arcsine #####
   x_asin_p = MC{2,NS}(-0.7, -0.7, Interval{Float64}(-0.9,-0.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_asin_n = MC{2,NS}(0.7, 0.7, Interval{Float64}(0.5,0.9), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_asin_z = MC{2,NS}(-0.1, -0.1, Interval{Float64}(-0.5,0.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_asin_ns_z = MC{2,Diff}(-0.1, -0.1, Interval{Float64}(-0.5,0.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_asin_d1_p = MC{2,NS}(-0.8216841452984665, -0.775397496610753, Interval{Float64}(-1.11977, -0.523598), @SVector[1.49043, 0.0], @SVector[1.40028, 0.0], false)
   yref_asin_d1_n = MC{2,NS}(0.775397496610753, 0.8216841452984665, Interval{Float64}(0.523598, 1.11977), @SVector[1.40028, 0.0], @SVector[1.49043, 0.0], false)
   yref_asin_d1_z = MC{2,NS}(-0.10958805193420748, -0.0974173098978382, Interval{Float64}(-0.523599, 0.523599), @SVector[1.03503, 0.0], @SVector[1.03503, 0.0], false)
   yref_asin_ns_z = MC{2,Diff}(-0.10958805193420748, -0.0974173098978382, Interval{Float64}(-0.523599, 0.523599), @SVector[1.03503, 0.0], @SVector[1.03503, 0.0], false)

   @test check_vs_ref1(asin, x_asin_p, yref_asin_d1_p, mctol)
   @test check_vs_ref1(asin, x_asin_n, yref_asin_d1_n, mctol)
   @test check_vs_ref1(asin, x_asin_z, yref_asin_d1_z, mctol)
   @test check_vs_ref1(asin, x_asin_ns_z, yref_asin_ns_z, mctol)

   ##### Arctangent #####
   x_atan_p = MC{2,NS}(4.0, 4.0, Interval{Float64}(3.0, 7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   yref_atan_d1_p = MC{2,NS}(1.294009147346374, 1.3258176636680326, Interval{Float64}(1.24904, 1.4289), @SVector[0.04496337494811958, 0.0], @SVector[0.058823529411764705, 0.0], false)
   @test check_vs_ref1(atan, x_atan_p, yref_atan_d1_p, mctol)

   ##### Hyperbolic Sine #####
   x_sinh_p = MC{2,Diff}(4.0, 4.0, Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_sinh_n = MC{2,Diff}(-4.0, -4.0, Interval{Float64}(-7.0,-3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_sinh_z = MC{2,Diff}(-2.0, -2.0, Interval{Float64}(-3.0,1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_sinh_ns_p = MC{2,NS}(4.0, 4.0, Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_sinh_d1_p = MC{2,Diff}(27.28991719712775, 144.59243701386904, Interval{Float64}(10.0178, 548.3161232732466), @SVector[27.3082, 0.0], @SVector[134.57456208645914, 0.0], false)
   yref_sinh_d1_n = MC{2,Diff}(-144.59243701386904, -27.28991719712775, Interval{Float64}(-548.3161232732466, -10.0178), @SVector[134.57456208645914 , 0.0], @SVector[27.3082, 0.0], false)
   yref_sinh_d1_z = MC{2,Diff}(-6.212568527712605, -3.626860407847019, Interval{Float64}(-10.0179, 1.17521), @SVector[3.8053063996972973, 0.0], @SVector[3.7622, 0.0], false)
   yref_sinh_ns_p = MC{2,NS}(27.28991719712775, 144.59243701386904, Interval{Float64}(10.0178, 548.3161232732466), @SVector[27.3082, 0.0], @SVector[134.57456208645914, 0.0], false)

   @test check_vs_ref1(sinh, x_sinh_p, yref_sinh_d1_p, mctol)
   @test check_vs_ref1(sinh, x_sinh_n, yref_sinh_d1_n, mctol)
   @test check_vs_ref1(sinh, x_sinh_z, yref_sinh_d1_z, mctol)
   @test check_vs_ref1(sinh, x_sinh_ns_p, yref_sinh_ns_p, mctol)

   ##### Hyperbolic Cosine #####
   x_cosh_ns = MC{2,NS}(4.0, 4.0, Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_cosh_d1 = MC{2,Diff}(4.0, 4.0, Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   yref_cosh_ns = MC{2,NS}(27.308232836016487, 144.63000528563632, Interval{Float64}(10.0676, 548.317035), @SVector[27.28991719712775, 0.0], @SVector[134.56234328985857, 0.0], false)
   yref_cosh_d1 = MC{2,Diff}(27.308232836016487, 144.63000528563632, Interval{Float64}(10.0676, 548.317035), @SVector[27.28991719712775, 0.0], @SVector[134.56234328985857, 0.0], false)
   @test check_vs_ref1(cosh, x_cosh_ns, yref_cosh_ns, mctol)
   @test check_vs_ref1(cosh, x_cosh_d1, yref_cosh_d1, mctol)

   ##### Hyperbolic Tangent #####
   x_tanh_p = MC{2,Diff}(4.0, 4.0, Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_tanh_n = MC{2,Diff}(-4.0, -4.0, Interval{Float64}(-7.0,-3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_tanh_z1 = MC{2,Diff}(-2.0, -2.0, Interval{Float64}(-3.0,1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_tanh_z2 = MC{2,Diff}(2.0, 2.0, Interval{Float64}(-1.0,3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_tanh_z1_ns = MC{2,NS}(-2.0, -2.0, Interval{Float64}(-3.0,1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_tanh_d1_p = MC{2,Diff}(0.996290649501034, 0.999329299739067, Interval{Float64}(0.995054,0.999999), @SVector[0.0012359, 0.0], @SVector[0.00134095, 0.0], false)
   yref_tanh_d1_n = MC{2,Diff}(-0.999329299739067, -0.996290649501034, Interval{Float64}(-0.999999, -0.995054), @SVector[0.00134095, 0.0], @SVector[0.0012359, 0.0], false)
   yref_tanh_d1_z1 = MC{2,Diff}(-0.9640275800758169, -0.5558207301372651, Interval{Float64}(-0.995055, 0.761595), @SVector[0.0706508, 0.0], @SVector[0.439234, 0.0], false)
   yref_tanh_d1_z2 = MC{2,Diff}(0.5558207301372651, 0.9640275800758169, Interval{Float64}(-0.761595, 0.995055), @SVector[0.439234, 0.0], @SVector[0.0706508, 0.0], false)
   yref_tanh_ns = MC{2,NS}(-0.9640275800758169, -0.5558207301372651, Interval{Float64}(-0.995055, 0.761595), @SVector[0.0706508, 0.0], @SVector[0.439234, 0.0], false)

   @test check_vs_ref1(tanh, x_tanh_p, yref_tanh_d1_p, mctol)
   @test check_vs_ref1(tanh, x_tanh_n, yref_tanh_d1_n, mctol)
   @test check_vs_ref1(tanh, x_tanh_z1, yref_tanh_d1_z1, mctol)
   @test check_vs_ref1(tanh, x_tanh_z2, yref_tanh_d1_z2, mctol)
   @test check_vs_ref1(tanh, x_tanh_z1_ns, yref_tanh_ns, mctol)

   x_cbrt_p = MC{2,Diff}(0.3, 0.3, Interval{Float64}(0.2, 0.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_cbrt_n = MC{2,Diff}(-0.3, -0.3, Interval{Float64}(-0.5, -0.2), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_cbrt_z1 = MC{2,Diff}(-0.3, -0.3, Interval{Float64}(-0.5, 0.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_cbrt_z2 = MC{2,Diff}(0.3, 0.3, Interval{Float64}(-0.5, 0.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   ycbrtp = cbrt(x_cbrt_p)
   ycbrtn = cbrt(x_cbrt_n)
   ycbrtz1 = cbrt(x_cbrt_z1)
   ycbrtz2 = cbrt(x_cbrt_z2)

   @test isapprox(ycbrtp.cv, 0.6544358737564154, atol=1E-8)
   @test isapprox(ycbrtp.cc, 0.6694329500821695, atol=1E-8)
   @test isapprox(ycbrtn.cv, -0.6694329500821695, atol=1E-8)
   @test isapprox(ycbrtn.cc, -0.6544358737564154, atol=1E-8)
   @test isapprox(ycbrtz1.cv, -0.6694329500821695, atol=1E-8)
   @test isapprox(ycbrtz1.cc, -0.3703935787925799, atol=1E-8)
   @test isapprox(ycbrtz2.cv, 0.37039357879257984, atol=1E-8)
   @test isapprox(ycbrtz2.cc, 0.6694329500821695, atol=1E-8)

   x_arh_p1 = MC{2,Diff}(0.3, 0.3, Interval{Float64}(0.2, 0.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_arh_p2 = MC{2,Diff}(4.1, 4.1, Interval{Float64}(3.5, 4.2), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_arh_p3 = MC{2,Diff}(0.6, 0.6, Interval{Float64}(0.5, 3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_arh_p4 = MC{2,NS}(2.6, 2.6, Interval{Float64}(0.5, 3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_arh_n = MC{2,Diff}(-0.3, -0.3, Interval{Float64}(-0.5, -0.2), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)


   arhp1 = arh(x_arh_p1, 2.0)
   arhp2 = arh(x_arh_p2, 2.0)
   arhp3 = arh(x_arh_p3, 2.0)
   arhp4 = arh(x_arh_p4, 2.0)
   arhn = arh(x_arh_n, 2.0)

   @test isapprox(arhp1.cv, 0.0012726338013398079, atol=1E-8)
   @test isapprox(arhp1.cc, 0.006135479582753048, atol=1E-8)
   @test isapprox(arhp2.cv, 0.6130841525286383, atol=1E-8)
   @test isapprox(arhp2.cc, 0.6139726608994069, atol=1E-8)
   @test isapprox(arhp3.cv, 0.03850739431865814, atol=1E-8)
   @test isapprox(arhp3.cc, 0.03986037228553247, atol=1E-8)
   @test isapprox(arhp4.cv, 0.43426549824693633, atol=1E-8)
   @test isapprox(arhp4.cc, 0.46336936923117533, atol=1E-8)
   @test isapprox(arhn.cv, 785.7719942274175, atol=1E-8)
   @test isapprox(arhn.cc, 14702.509913215526, atol=1E-8)

   ##### Inverse Hyperbolic Sine #####
   x_asinh_p = MC{2,Diff}(0.3, 0.3, Interval{Float64}(0.1,0.7), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_asinh_n = MC{2,Diff}(-0.3, -0.3, Interval{Float64}(-0.7,-0.1), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_asinh_z1 = MC{2,Diff}(2.0, 2.0, Interval{Float64}(-3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_asinh_z2 = MC{2,Diff}(-2.0, -2.0, Interval{Float64}(-7.0,3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_asinh_z1_ns = MC{2,NS}(2.0, 2.0, Interval{Float64}(-3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_asinh_d1_p = MC{2,Diff}(0.2841115746269236, 0.29567304756342244, Interval{Float64}(0.099834,0.652667), @SVector[0.921387, 0.0], @SVector[0.957826, 0.0], false)
   yref_asinh_d1_n = MC{2,Diff}(-0.29567304756342244, -0.2841115746269236, Interval{Float64}(-0.652667,-0.099834), @SVector[0.957826, 0.0], @SVector[0.921387, 0.0], false)
   yref_asinh_d1_z1 = MC{2,Diff}(0.3730697449603356, 1.4436354751788103, Interval{Float64}(-1.81845, 2.64413), @SVector[0.45421, 0.0], @SVector[0.447214, 0.0], false)
   yref_asinh_d1_z2 = MC{2,Diff}(-1.4436354751788103, -0.3730697449603356, Interval{Float64}(-2.64413,1.81845), @SVector[0.447214, 0.0], @SVector[0.45421, 0.0], false)
   yref_asinh_ns = MC{2,NS}(0.37306974496033596, 1.4436354751788103, Interval{Float64}(-1.8184464592320668, 2.6441207610586295), @SVector[0.45421020321965866, 0.0], @SVector[0.44721359549, 0.0], false)

   @test check_vs_ref1(asinh, x_asinh_p, yref_asinh_d1_p, mctol)
   @test check_vs_ref1(asinh, x_asinh_n, yref_asinh_d1_n, mctol)
   @test check_vs_ref1(asinh, x_asinh_z1, yref_asinh_d1_z1, mctol)
   @test check_vs_ref1(asinh, x_asinh_z2, yref_asinh_d1_z2, mctol)
   @test check_vs_ref1(asinh, x_asinh_z1_ns, yref_asinh_ns, mctol)

   ##### Inverse Hyperbolic Cosine #####
   x_acosh = MC{2,NS}(4.0, 4.0, Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   yref_acosh = MC{2,NS}(1.9805393289917226, 2.0634370688955608, Interval{Float64}(1.76274,2.63392), @SVector[0.217792, 0.0], @SVector[0.258199, 0.0], false)
   @test check_vs_ref1(acosh, x_acosh, yref_acosh, mctol)

   x_acosh = MC{2,Diff}(2.1, 2.2, Interval{Float64}(2.0,3.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   yref_acosh = acosh(x_acosh)
   @test yref_acosh.cv == 1.3574838571457233
   @test yref_acosh.cc == 1.4254169430706127
   @test yref_acosh.Intv.lo == 1.3169578969248166
   @test yref_acosh.Intv.hi == 1.9248473002384139
   @test yref_acosh.cv_grad[1] == 0.4052596022090649
   @test yref_acosh.cc_grad[1] == 0.5415303610738823

   ##### Inverse Hyperbolic Tangent #####
   x_atanh_p = MC{2,Diff}(0.6, 0.6, Interval{Float64}(0.1,0.7), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_atanh_n = MC{2,Diff}(-0.6, -0.6, Interval{Float64}(-0.7,-0.1), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_atanh_z1 = MC{2,Diff}(0.6, 0.6, Interval{Float64}(-0.6,0.7), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_atanh_z2 = MC{2,Diff}(-0.5, -0.5, Interval{Float64}(-0.7,0.6), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_atanh_z1_ns = MC{2,NS}(0.6, 0.6, Interval{Float64}(-0.6,0.7), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_atanh_d1_p = MC{2,Diff}(0.6931471805599453, 0.7394729977002236, Interval{Float64}(0.100335,0.867301), @SVector[1.5625, 0.0], @SVector[1.27828, 0.0], false)
   yref_atanh_d1_n = MC{2,Diff}(-0.7394729977002236, -0.6931471805599453, Interval{Float64}(-0.867301,-0.100335), @SVector[1.27828, 0.0], @SVector[1.5625, 0.0], false)
   yref_atanh_d1_z1 = MC{2,Diff}(0.6931471805599453, 0.7263562565915014, Interval{Float64}(-0.6931471805599453,0.867301), @SVector[1.5625, 0.0], @SVector[1.4094427110255183, 0.0], false)
   yref_atanh_d1_z2 = MC{2,Diff}(-0.5854119854889636, -0.5493061443340549, Interval{Float64}(-0.867301,0.6931471805599453), @SVector[1.4094427110254484, 0.0], @SVector[1.3333333333333333, 0.0], false)
   yref_atanh_ns = MC{2,NS}(0.6931471805599453, 0.72635625, Interval{Float64}(-0.6931471805599453,0.867301), @SVector[1.5625, 0.0], @SVector[1.4094427110255183, 0.0], false)

   @test check_vs_ref1(atanh, x_atanh_p, yref_atanh_d1_p, mctol)
   @test check_vs_ref1(atanh, x_atanh_n, yref_atanh_d1_n, mctol)
   @test check_vs_ref1(atanh, x_atanh_z1, yref_atanh_d1_z1, mctol)
   @test check_vs_ref1(atanh, x_atanh_z2, yref_atanh_d1_z2, mctol)
   @test check_vs_ref1(atanh, x_atanh_z1_ns, yref_atanh_ns, mctol)

   # CONVERSION
   X = MC{2,NS}(4.5, 4.5, Interval{Float64}(-3.0,8.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   X1 = convert(MC{2,NS}, 1)
   X2 = convert(MC{2,NS}, 1.1)
   X3 = convert(MC{2,NS}, Interval{Float64}(2.1,4.3))
   @test X1.cc == 1.0
   @test X1.cv == 1.0
   @test X2.cc == 1.1
   @test X2.cv == 1.1
   @test X3.cc == 4.3
   @test X3.cv == 2.1

   @test +X == X

   xextras = MC{2,NS}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   @test acot(xextras) == atan(inv(xextras))
   @test sech(xextras) == inv(cosh(xextras))
   @test csch(xextras) == inv(sinh(xextras))
   @test coth(xextras) == inv(tanh(xextras))
   @test acsch(xextras) == log(sqrt(1.0+inv(sqr(xextras)))+inv(xextras))
   @test isapprox(acoth(xextras).cv, (0.5*(log(1.0+inv(xextras))-log(1.0-inv(xextras)))).cv, atol = 1E-8)
   @test isnan(acoth(xextras).cc) && isnan(0.5*(log(1.0+inv(xextras))-log(1.0-inv(xextras))))
   @test sind(xextras) == sin(deg2rad(xextras))
   @test cosd(xextras) == cos(deg2rad(xextras))
   @test tand(xextras) == tan(deg2rad(xextras))
   @test secd(xextras) == inv(cosd(xextras))
   @test cscd(xextras) == inv(sind(xextras))
   @test cotd(xextras) == inv(tand(xextras))
   @test atand(xextras) == rad2deg(atan(xextras))
   @test acotd(xextras) == rad2deg(acot(xextras))

   xextras1 = xextras = MC{2,NS}(0.1, 0.2, Interval{Float64}(0.05,0.21), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   @test csc(xextras) == inv(sin(xextras))
   @test sec(xextras) == inv(cos(xextras))
   @test cot(xextras) == inv(tan(xextras))
   @test asind(xextras) == rad2deg(asin(xextras))
   @test acosd(xextras) == rad2deg(acos(xextras))

   x_atan_p = MC{2,Diff}(0.6, 0.6, Interval{Float64}(0.1,0.7), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_atan_n = MC{2,Diff}(-0.6, -0.6, Interval{Float64}(-0.7,-0.1), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_atan_z1 = MC{2,Diff}(0.6, 0.6, Interval{Float64}(-0.6,0.7), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_atan_z2 = MC{2,Diff}(-0.5, -0.5, Interval{Float64}(-0.7,0.6), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x_atan_z1_ns = MC{2,NS}(0.6, 0.6, Interval{Float64}(-0.6,0.7), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_atan_d1_p = MC{2,Diff}(0.5255497457395342, 0.5404195002705842, Interval{Float64}(0.09966865249116202,0.6107259643892087), @SVector[0.8517621864967442, 0.0], @SVector[0.7352941176470589, 0.0], false)
   yref_atan_d1_n = MC{2,Diff}(-0.5404195002705842, -0.5255497457395342, Interval{Float64}(-0.6107259643892087,-0.09966865249116202), @SVector[0.7352941176470589, 0.0], @SVector[0.8517621864967442, 0.0], false)
   yref_atan_d1_z2 = MC{2,Diff}(-0.4636476090008061, -0.43024560156279196, Interval{Float64}(-0.6107259643892087,0.5404195002705842), @SVector[0.8, 0.0], @SVector[0.9024018141320833, 0.0], false)
   yref_atan_ns = MC{2,NS}(0.5204857829760002, 0.540419500270584, Interval{Float64}(-0.5404195002705842,0.6107259643892087), @SVector[0.9024018141320833, 0.0], @SVector[0.7352941176470589, 0.0], false)

   @test check_vs_ref1(atan, x_atan_p, yref_atan_d1_p, mctol)
   @test check_vs_ref1(atan, x_atan_n, yref_atan_d1_n, mctol)
   @test check_vs_ref1(atan, x_atan_z2, yref_atan_d1_z2, mctol)
   @test check_vs_ref1(atan, x_atan_z1_ns, yref_atan_ns, mctol)

   x = MC{2, NS}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   out1 = McCormick.log10_kernel(x, log10(x.Intv))
   @test isapprox(out1.cv, 0.20068666, atol=1E-5)

   y = MC{2, Diff}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   out2 = McCormick.log10_kernel(y, log10(y.Intv))
   @test isapprox(out2.cv, 0.20068666, atol=1E-5)

   z = MC{1, Diff}(1.1)
   out3 = McCormick.log10_kernel(z, Interval{Float64}(log10(1.1)))
   @test isapprox(out3.cv, 0.041392685, atol=1E-5)

   x = MC{2, NS}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   out4 = McCormick.exp_kernel(x, exp(x.Intv))
   @test isapprox(out4.cv, 7.38905609, atol=1E-5)

   y = MC{2, Diff}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   out5 = McCormick.exp_kernel(y, exp(y.Intv))
   @test isapprox(out5.cv, 7.38905609893065, atol=1E-5)

   x1dual = Dual{1,Float64,2}(2.1, Partials{2,Float64}(NTuple{2,Float64}([1.1; 2.9])))
   x2dual = Dual{1,Float64,2}(2.4, Partials{2,Float64}(NTuple{2,Float64}([1.1; 2.9])))
   x3dual = Dual{1,Float64,2}(2.7, Partials{2,Float64}(NTuple{2,Float64}([1.1; 2.9])))
   out6 = McCormick.mid3v(x1dual, x2dual, x3dual)
   out7 = McCormick.mid3v(x1dual, x3dual, x2dual)
   out8 = McCormick.mid3v(x2dual, x3dual, x1dual)
   @test out6.value == 2.4
   @test out7.value == 2.4
   @test out8.value == 2.4

   Y1 = MC{2,NS}(-4.0,-4.0,Interval{Float64}(-5.0,2.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)
   out1 = Y1^4
   @test out1.cv == 256.0
   @test out1.cc == 538.0

   Y1 = MC{2,Diff}(-4.0,-4.0,Interval{Float64}(-5.0,2.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)
   Y2 = MC{2,NS}(-4.0,-4.0,Interval{Float64}(-5.0,2.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)

   @test isnan(inv(Y1))
   @test isnan(inv(Y2))

   @test isnan(Y1^(-3))
   @test isnan(Y2^(-3))

   @test isnan(Y1^(-3.0))
   @test isnan(Y2^(-3.0))

   @test isapprox(McCormick.tanh_deriv(3.0, 2.0, 4.0), 0.009866, rtol=1E-3)
   @test isapprox(McCormick.tanh_envd(3.0, 2.0, 4.0), 6.258589, rtol=1E-3)
   @test isapprox(McCormick.atan_deriv(3.0, 2.0, 4.0), 0.100, rtol=1E-3)
   @test isapprox(McCormick.asin_deriv(0.5, 0.0, 0.9), 1.1547005, rtol=1E-3)
   @test isapprox(McCormick.tan_deriv(3.0, 2.0, 4.0), 1.0203195, rtol=1E-3)
   @test isapprox(McCormick.tan_envd(3.0, 2.0, 4.0), -0.570704, rtol=1E-3)
   @test isapprox(McCormick.acos_deriv(0.5, 0.0, 0.9), -1.1547005, rtol=1E-3)
   @test isapprox(McCormick.acos_env(0.5, 0.0, 0.9), -0.04655, rtol=1E-3)
   @test isapprox(McCormick.asinh_deriv(3.0, 2.0, 4.0), 0.31622, rtol=1E-3)

   xD = MC{2,Diff}(2.0,2.0,Interval{Float64}(1.0,4.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)
   xNS = MC{2,NS}(2.0,2.0,Interval{Float64}(1.0,4.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)

   @test xD^0 == one(xD)
   @test xNS^0 == one(xNS)
   @test ~isnan(xD)
   @test ~in(0.5,xD)
   @test in(2,xD)
   @test ~isempty(xD)

   @test isnan((-0.5)^xNS)

   xD = MC{2,Diff}(2.0,2.0,Interval{Float64}(1.0,4.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)
   xNS = MC{2,NS}(2.0,2.0,Interval{Float64}(1.0,4.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)

   mc2 = ^(0.5, xNS)
   mc3 = (^)(xD, 0.5, xNS.Intv^0.5)
   mc4 = (^)(xD, 2.3, xNS.Intv^2.3)
   mc5 = (^)(xNS, 0.5, xNS.Intv^0.5)
   mc6 = (^)(xNS, 2.3, xNS.Intv^2.3)
   mc7 = (^)(xNS, Float32(2.3), xNS.Intv^2.3)
   mc8 = (^)(xNS, Float16(2.3), xNS.Intv^2.3)
   mc9 = (^)(xNS, Float32(2.3))
   mc10 = (^)(xNS, Float16(2.3))

   @test  isapprox(mc10.cv, 2.895583, rtol=1E-4)
   @test  isapprox(mc10.cc, 12.63887, rtol=1E-4)
   @test  isapprox(mc10.Intv.lo, 1, rtol=1E-4)
   @test  isapprox(mc10.Intv.hi, 24.2778, rtol=1E-4)

   @test  isapprox(mc2.cv, 0.25, rtol=1E-4)
   @test  isapprox(mc2.cc, 0.354166, rtol=1E-4)
   @test  isapprox(mc2.Intv.lo, 0.0625, rtol=1E-4)
   @test  isapprox(mc2.Intv.hi, 0.500001, rtol=1E-4)

   @test  isapprox(mc3.cv, 1.33333, rtol=1E-4)
   @test  isapprox(mc3.cc, 1.414214, rtol=1E-4)
   @test  isapprox(mc3.Intv.lo, 1, rtol=1E-4)
   @test  isapprox(mc3.Intv.hi, 2, rtol=1E-4)

   @test  isapprox(mc4.cv, 2.8945384748807563, rtol=1E-4)
   @test  isapprox(mc4.cc, 12.62573, rtol=1E-4)
   @test  isapprox(mc4.Intv.lo, 1, rtol=1E-4)
   @test  isapprox(mc4.Intv.hi, 24.2515, rtol=1E-4)

   @test  isapprox(mc5.cv, 1.33333, rtol=1E-4)
   @test  isapprox(mc5.cc, 1.414214, rtol=1E-4)
   @test  isapprox(mc5.Intv.lo, 1, rtol=1E-4)
   @test  isapprox(mc5.Intv.hi, 2, rtol=1E-4)

   @test  isapprox(mc6.cv, 2.8945384, rtol=1E-4)
   @test  isapprox(mc6.cc, 12.625732, rtol=1E-4)
   @test  isapprox(mc6.Intv.lo, 1, rtol=1E-4)
   @test  isapprox(mc6.Intv.hi, 24.2515, rtol=1E-4)

   @test  isapprox(mc7.cv, 2.8945384, rtol=1E-4)
   @test  isapprox(mc7.cc, 12.625732, rtol=1E-4)
   @test  isapprox(mc7.Intv.lo, 1, rtol=1E-4)
   @test  isapprox(mc7.Intv.hi, 24.2515, rtol=1E-4)

   @test  isapprox(mc8.cv, 2.8955384, rtol=1E-4)
   @test  isapprox(mc8.cc, 12.63887, rtol=1E-4)
   @test  isapprox(mc8.Intv.lo, 1, rtol=1E-4)
   @test  isapprox(mc8.Intv.hi, 24.2515, rtol=1E-3)

   @test  isapprox(mc9.cv, 2.8945384, rtol=1E-4)
   @test  isapprox(mc9.cc, 12.625732, rtol=1E-4)
   @test  isapprox(mc9.Intv.lo, 1, rtol=1E-4)
   @test isapprox(mc9.Intv.hi, 24.2515, rtol=1E-3)

   a = MC{5,NS}(1.0,(Interval{Float64}(-5.1,5.9)),2)

   cv,dcv = McCormick.cv_neg_powneg_odd(3.0, 3.0, 3.0, 2)
   @test cv == 9.0
   @test dcv == 0.0

   @test McCormick.pow_kernel(xNS, 1, xNS.Intv) == xNS

   X = Interval(-0.9, 0.8)
   x = MC{1,Diff}(-0.75, X, 1)
   y = cos(x)*max(x^3, cos(x)*exp(x)/((x-4.0)*(x+2.0))-1.0)/(x+2.0)
   #@test y.cv == -0.5610120753581359
   #@test y.cc == 0.11281176084484947

   x = MC{1,Diff}(-0.25, X, 1)
   y = cos(x)*max(x^3, cos(x)*exp(x)/((x-4.0)*(x+2.0))-1.0)/(x+2.0)
   #@test y.cv == -0.4255132669583401
   #@test y.cc == 0.2664785395034585

   x = MC{1,Diff}(0.25, X, 1)
   y = cos(x)*max(x^3, cos(x)*exp(x)/((x-4.0)*(x+2.0))-1.0)/(x+2.0)
   #@test y.cv == -0.23783870584548902
   #@test y.cc == 0.22923736914602943

   x = MC{1,Diff}(0.7, X, 1)
   y = cos(x)*max(x^3, cos(x)*exp(x)/((x-4.0)*(x+2.0))-1.0)/(x+2.0)
   #@test y.cv == 0.02549714244205218
   #@test y.cc == 0.17002351564507798

   yD = xlogx(xD)
   yNS = xlogx(xNS)
   @test isapprox(yD.cv, 1.3862943611198906, atol=1E-8)
   @test isapprox(yD.cc, 1.8483924814931874, atol=1E-8)
   @test isapprox(yNS.cv, 1.3862943611198906, atol=1E-8)
   @test isapprox(yNS.cc, 1.8483924814931874, atol=1E-8)

   fma1 = fma(mc6, mc6, mc6)
   @test isapprox(fma1.cv, 7.683615424642269, atol = 1E-8)
   @test isapprox(fma1.cc, 307.1925114113242, atol = 1E-8)
end

@testset "Test Arithmetic w/Constant" begin

   mctol = 1E-4

   x0 = MC{2,NS}(4.5, 4.5, Interval{Float64}(2.0,8.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   fma1 = fma(x0, 2.0, x0)
   @test isapprox(fma1.cv, 13.5, atol = 1E-8)
   @test isapprox(fma1.cc, 13.5, atol = 1E-8)

   fma2 = fma(x0, x0, 2.0)
   @test isapprox(fma2.cv, 16.0, atol = 1E-8)
   @test isapprox(fma2.cc, 31.0, atol = 1E-8)

   fma3 = fma(2.0, x0, x0)
   @test isapprox(fma3.cv, 13.5, atol = 1E-8)
   @test isapprox(fma3.cc, 13.5, atol = 1E-8)

   fma4 = fma(x0, 2.0, 3.0)
   @test isapprox(fma4.cv, 12.0, atol = 1E-8)
   @test isapprox(fma4.cc, 12.0, atol = 1E-8)

   fma5 = fma(2.0, x0, 3.0)
   @test isapprox(fma5.cv, 12.0, atol = 1E-8)
   @test isapprox(fma5.cc, 12.0, atol = 1E-8)

   fma6 = fma(2.0, 3.0, x0)
   @test isapprox(fma6.cv, 10.5, atol = 1E-8)
   @test isapprox(fma6.cc, 10.5, atol = 1E-8)

   x0a = McCormick.div_kernel(x0, 2.0, x0.Intv/2.0)
   x0b = McCormick.div_kernel(2.0, x0, 2.0/x0.Intv)

   @test x0a.cv == 2.25
   @test x0a.cc == 2.25
   @test isapprox(x0b.cv, 0.4444444444444444, atol=1E-7)
   @test x0b.cc == 0.6875

   x = MC{2,NS}(4.5, 4.5, Interval{Float64}(-3.0,8.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   x1 = x + Interval(1.0,2.0); @test x1.cv == 5.5 ; @test x1.cc == 6.5
   x2 = Interval(1.0, 2.0) + x; @test x2.cv == 5.5 ; @test x2.cc == 6.5

   x1 = x - Interval(1.0, 2.0); @test x1.cv == 2.5 ; @test x1.cc == 3.5
   x2 = Interval(1.0, 2.0) - x; @test x2.cv == -3.5 ; @test x2.cc == -2.5

   x1 = min(x, Interval(1.0, 2.0)); @test isapprox(x1.cv, -0.4583333333333335, atol=1E-6) ; @test x1.cc == 2.0
   x2 = min(Interval(1.0, 2.0), x); @test isapprox(x2.cv, -0.4583333333333335, atol=1E-6) ; @test x2.cc == 2.0

   x1 = max(x, Interval(1.0, 2.0)); @test x1.cv == 4.5 ; @test isapprox(x1.cc, 6.458333333333334, atol=1E-6)
   x2 = max(Interval(1.0, 2.0), x); @test x2.cv == 4.5 ; @test isapprox(x2.cc, 6.458333333333334, atol=1E-6)

   x1 = x/Interval(1.0, 2.0); @test x1.cv == 0.75 ; @test x1.cc == 6.0
   x2 = Interval(1.0, 2.0)/(x+10.0)
   @test isapprox(x2.cv, 0.06896551724137931, atol=1E-6)
   @test isapprox(x2.cc, 0.16666666666666669, atol=1E-6)

   x1 = x*Interval(1.0, 2.0); @test x1.cv == 1.5 ; @test x1.cc == 12.0
   x2 = Interval(1.0, 2.0)*x; @test x2.cv == 1.5 ; @test x2.cc == 12.0

   x1 = x + 2.1; @test x1.cv == x1.cc == 6.6
   x2 = 2.3 + x; @test x2.cv == x2.cc == 6.8
   x3 = x + 2;   @test x3.cv == x3.cc == 6.5
   x4 = 3 + x;   @test x4.cv == x4.cc == 7.5

   x1 = x + Float16(2.1); @test x1.cv == x1.cc == 6.599609375
   x2 = Float16(2.3) + x; @test x2.cv == x2.cc == 6.80078125
   x3 = x + Int16(2);     @test x3.cv == x3.cc == 6.5
   x4 = Int16(3) + x;     @test x4.cv == x4.cc == 7.5

   x1 = x - 2.1; @test x1.cv == x1.cc == 2.4
   x2 = 2.3 - x; @test x2.cv == x2.cc == -2.2
   x3 = x - 2;   @test x3.cv == x3.cc == 2.5
   x4 = 3 - x;   @test x4.cv == x4.cc == -1.5

   x1 = x - Float16(2.1); @test x1.cv == x1.cc == 2.400390625
   x2 = Float16(2.3) - x; @test x2.cv == x2.cc == -2.19921875
   x3 = x - Int16(2);     @test x3.cv == x3.cc == 2.5
   x4 = Int16(3) - x;     @test x4.cv == x4.cc == -1.5

   x1 = x * 2.1; @test x1.cv == x1.cc == 9.450000000000001
   x2 = 2.3 * x; @test x2.cv == x2.cc == 10.35
   x3 = x * 2;   @test x3.cv == x3.cc == 9.0
   x4 = 3 * x;   @test x4.cv == x4.cc == 13.5

   x1 = x * Float16(2.1); @test x1.cv == x1.cc == 9.4482421875
   x2 = Float16(2.3) * x; @test x2.cv == x2.cc == 10.353515625
   x3 = x * Int16(2);     @test x3.cv == x3.cc == 9.0
   x4 =  Int16(3) * x;    @test x4.cv == x4.cc == 13.5

   x1 = x * (-2.1); @test x1.cv == x1.cc == -9.450000000000001
   x2 = (-2.3) * x; @test x2.cv == x2.cc == -10.35
   x3 = x * (-2);   @test x3.cv == x3.cc == -9.0
   x4 = (-3) * x;   @test x4.cv == x4.cc == -13.5

   x1 = x * Float16(-2.1); @test x1.cv == x1.cc == -9.4482421875
   x2 = Float16(-2.3) * x; @test x2.cv == x2.cc == -10.353515625
   x3 = x * Int16(-2);     @test x3.cv == x3.cc == -9.0
   x4 =  Int16(-3) * x;    @test x4.cv == x4.cc == -13.5

   x1 = 1.0/(x+4)
   @test isapprox(x1.cc, 0.37500000000000017, atol=mctol)
   @test isapprox(x1.cv, 0.11764705882352941, atol=mctol)
   @test isapprox(x1.cc_grad[1], -0.08333333333333334, atol=mctol)
   @test isapprox(x1.cc_grad[2], -0.0, atol=mctol)
   @test isapprox(x1.cv_grad[1], -0.01384083044982699, atol=mctol)
   @test isapprox(x1.cv_grad[2], -0.0, atol=mctol)

   x2 = 1/(x+4)
   @test isapprox(x2.cc, 0.37500000000000017, atol=1E-6)
   @test isapprox(x2.cv, 0.11764705882352941, atol=1E-6)
   @test isapprox(x2.cc_grad[1], -0.08333333333333334, atol=1E-6)
   @test isapprox(x2.cc_grad[2], -0.0, atol=1E-6)
   @test isapprox(x2.cv_grad[1], -0.01384083044982699, atol=1E-6)
   @test isapprox(x2.cv_grad[2], -0.0, atol=1E-6)

   x_min_ns = MC{2,NS}(1.0909090909090908, 3.0, Interval{Float64}(-3.0, 3.0), @SVector[0.545455, 0.0], @SVector[0.0, 0.0], false)
   x_max_ns = MC{2,NS}(1.0909090909090908, 3.0, Interval{Float64}(-3.0, 3.0), @SVector[0.545455, 0.0], @SVector[0.0, 0.0], false)
   x_min_d1 = MC{2,Diff}(1.0909090909090908, 3.0, Interval{Float64}(-3.0, 3.0), @SVector[0.545455, 0.0], @SVector[0.0, 0.0], false)
   x_max_d1 = MC{2,Diff}(1.0909090909090908, 3.0, Interval{Float64}(-3.0, 3.0), @SVector[0.545455, 0.0], @SVector[0.0, 0.0], false)

   yref1_min_ns = MC{2,NS}(1.0909090909090908, 3.0, Interval{Float64}(-3.0, 3.0), @SVector[0.545455, 0.0], @SVector[0.0, 0.0], false)
   yref2_min_ns = MC{2,NS}(1.0909090909090908, 3.0, Interval{Float64}(-3.0, 3.0), @SVector[0.545455, 0.0], @SVector[0.0, 0.0], false)
   yref1_max_ns = MC{2,NS}(3.0, 3.0, Interval{Float64}(3.0, 3.0), @SVector[0.0, 0.0], @SVector[0.0, 0.0], false)
   yref2_max_ns = MC{2,NS}(3.0, 3.0, Interval{Float64}(3.0, 3.0), @SVector[0.0, 0.0], @SVector[0.0, 0.0], false)
   yref1_min_d1 = MC{2,Diff}(1.0909090909090908, 3.0, Interval{Float64}(-3.0, 3.0), @SVector[0.545455, 0.0], @SVector[0.0, 0.0], false)
   yref2_min_d1 = MC{2,Diff}(1.0909090909090908, 3.0, Interval{Float64}(-3.0, 3.0), @SVector[0.545455, 0.0], @SVector[0.0, 0.0], false)
   yref1_max_d1 = MC{2,Diff}(3.0, 3.0, Interval{Float64}(3.0, 3.0), @SVector[0.0, 0.0], @SVector[0.0, 0.0], false)
   yref2_max_d1 = MC{2,Diff}(3.0, 3.0, Interval{Float64}(3.0, 3.0), @SVector[0.0, 0.0], @SVector[0.0, 0.0], false)

   @test check_vs_refv(min, x_min_ns, 3.0, yref1_min_ns, mctol)
   @test check_vs_refv(min, x_min_ns, 3.0, yref2_min_ns, mctol)
   @test check_vs_refv(max, x_max_ns, 3.0, yref1_max_ns, mctol)
   @test check_vs_refv(max, x_max_ns, 3.0, yref2_max_ns, mctol)

   @test check_vs_refv(min, x_min_d1, 3.0, yref1_min_d1, mctol)
   @test check_vs_refv(min, x_min_d1, 3.0, yref2_min_d1, mctol)
   @test check_vs_refv(max, x_max_d1, 3.0, yref1_max_d1, mctol)
   @test check_vs_refv(max, x_max_d1, 3.0, yref2_max_d1, mctol)

   x1 = MC{2,NS}(4.0, 4.0, Interval{Float64}(3.0, 7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x2 = MC{2,NS}(-4.5, -4.5, Interval{Float64}(-8.0, -3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x3 = MC{2,NS}(4.5, 4.5, Interval{Float64}(3.0, 8.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x4 = MC{2,NS}(-4.5, -4.5, Interval{Float64}(-8.0, -3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x5 = MC{2,NS}(-4.0, -4.0, Interval{Float64}(-7.0, -3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x6 = MC{2,NS}(-2.0, -2.0,Interval{Float64}(-3.0, 1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_ns_pow1 = MC{2,NS}(16.0, 19.0, Interval{Float64}(9.0, 49.0), @SVector[8.0, 0.0], @SVector[10.0, 0.0], false)
   yref_ns_pow2 = MC{2,NS}(0.0625, 0.08843537414965986, Interval{Float64}(0.0204081, 0.111112), @SVector[-0.03125, 0.0], @SVector[-0.0226757, 0.0], false)
   yref_ns_pow3 = MC{2,NS}(0.04938271604938271, 0.08246527777777776, Interval{Float64}(0.015625, 0.111112), @SVector[0.0219479, 0.0], @SVector[0.0190972, 0.0], false)
   yref_ns_pow3n = MC{2,NS}(-0.03703703703703704, -0.010973936899862827, Interval{Float64}(-0.03703703703703704, -0.001953125), @SVector[0.0, 0.0], @SVector[-0.0073159579332418845, 0.0], false)
   yref_ns_pow4 = MC{2,NS}(20.25, 25.5, Interval{Float64}(9.0, 64.0), @SVector[-9.0, 0.0], @SVector[-11.0, 0.0], false)
   yref_ns_pow5 = MC{2,NS}(-172.5, -91.125, Interval{Float64}(-512.0, -27.0), @SVector[97.0, 0.0], @SVector[60.75, 0.0], false)
   yref_ns_pow6 = MC{2,NS}(410.0625, 1285.5, Interval{Float64}(81.0, 4096.0), @SVector[-364.5, 0.0], @SVector[-803.0, 0.0], false)
   yref_ns_pow7 = MC{2,NS}(91.125, 172.5, Interval{Float64}(27.0, 512.0), @SVector[60.75, 0.0], @SVector[97.0, 0.0], false)
   yref_ns_pow8 = MC{2,NS}(410.0625, 1285.5, Interval{Float64}(81.0, 4096.0), @SVector[364.5, 0.0], @SVector[803.0, 0.0], false)
   yref_ns_pow9 = MC{2,NS}(410.0625, 1285.5, Interval{Float64}(81.0, 4096.0), @SVector[-364.5, 0.0], @SVector[-803.0, 0.0], false)
   @test check_vs_refv(^, x1, 2, yref_ns_pow1, mctol)
   @test check_vs_refv(^, x1, -2, yref_ns_pow2, mctol)
   @test check_vs_refv(^, x2, -2, yref_ns_pow3, mctol)
   @test check_vs_refv(^, x2, -3, yref_ns_pow3n, mctol)
   @test check_vs_refv(^, x2, 2, yref_ns_pow4, mctol)
   @test check_vs_refv(^, x2, 3, yref_ns_pow5, mctol)
   @test check_vs_refv(^, x2, 4, yref_ns_pow6, mctol)
   @test check_vs_refv(^, x3, 3, yref_ns_pow7, mctol)
   @test check_vs_refv(^, x3, 4, yref_ns_pow8, mctol)
   @test check_vs_refv(^, x4, 4, yref_ns_pow9, mctol)
   @test x2^(1) == x2

   y1_abs2 = abs2(x1)
   @test isapprox(y1_abs2.cv, 16.0, atol=1E-8)
   @test isapprox(y1_abs2.cc, 19.0, atol=1E-8)

   x1_d1 = MC{2,Diff}(4.0, 4.0, Interval{Float64}(3.0, 7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x2_d1 = MC{2,Diff}(-4.5, -4.5, Interval{Float64}(-8.0, -3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x3_d1 = MC{2,Diff}(4.5, 4.5, Interval{Float64}(3.0, 8.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x4_d1 = MC{2,Diff}(-4.5, -4.5, Interval{Float64}(-8.0, -3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x5_d1 = MC{2,Diff}(-4.0, -4.0, Interval{Float64}(-7.0, -3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x6_d1 = MC{2,Diff}(-2.0, -2.0,Interval{Float64}(-3.0, 1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   yref_d1_pow1 = MC{2,Diff}(0.25, 0.2857142857142857, Interval{Float64}(0.142857, 0.333334), @SVector[-0.0625, 0.0], @SVector[-0.047619, 0.0], false)
   yref_d1_pow2 = MC{2,Diff}(16.0, 19.0, Interval{Float64}(9.0, 49.0), @SVector[8.0, 0.0], @SVector[10.0, 0.0], false)
   yref_d1_pow3 = MC{2,Diff}(16.0, 19.0, Interval{Float64}(9.0, 49.0), @SVector[-8.0, 0.0], @SVector[-10.0, 0.0], false)
   yref_d1_pow4 = MC{2,Diff}(2.66666666666666, 7.0, Interval{Float64}(0.0, 9.0), @SVector[-4.0, 0.0], @SVector[-2.0, 0.0], false)
   yref_d1_pow5 = MC{2,Diff}(64.0, 106.0, Interval{Float64}(27.0, 343.0), @SVector[48.0, 0.0], @SVector[79.0, 0.0], false)
   yref_d1_pow6 = MC{2,Diff}(-106.0, -64.0, Interval{Float64}(-343.0, -27.0), @SVector[79.0, 0.0], @SVector[48.0, 0.0], false)
   yref_d1_pow7 = MC{2,Diff}(-20.25, -7.750, Interval{Float64}(-27.0, 1.0), @SVector[6.75, 0.0], @SVector[12.25, 0.0], false)
   yref_d1_pow8 = MC{2,Diff}(0.015625, 0.02850664075153871, Interval{Float64}(0.00291545, 0.0370371), @SVector[-0.0117188, 0.0], @SVector[-0.0085304, 0.0], false)
   yref_d1_pow8a = MC{2,Diff}(-0.026511863425925923, -0.010973936899862827, Interval{Float64}(-0.03703703703703704, -0.001953125), @SVector[-0.007016782407407407, 0.0], @SVector[-0.0073159579332418845, 0.0], false)
   yref_d1_pow9 = MC{2,Diff}(-0.02850664075153871, -0.015625, Interval{Float64}(-0.0370371, -0.00291545), @SVector[-0.0085304, 0.0], @SVector[-0.0117188, 0.0], false)
   yref_d1_pow10 = MC{2,Diff}(0.00390625, 0.009363382541225106, Interval{Float64}(0.000416493, 0.0123457), @SVector[-0.00390625, 0.0], @SVector[-0.0029823, 0.0], false)
   yref_d1_pow11 = MC{2,Diff}(0.00390625, 0.009363382541225106, Interval{Float64}(0.000416493, 0.0123457), @SVector[0.00390625, 0.0], @SVector[0.0029823, 0.0], false)
   yref_d1_pow12 = MC{2,Diff}(256.0, 661.0, Interval{Float64}(81.0, 2401.0), @SVector[256.0, 0.0], @SVector[580.0, 0.0], false)
   yref_d1_pow13 = MC{2,Diff}(256.0, 661.0, Interval{Float64}(81.0, 2401.0), @SVector[-256.0, 0.0], @SVector[-580.0, 0.0], false)
   yref_d1_pow14 = MC{2,Diff}(16.0, 61.0,  Interval{Float64}(0.0, 81.0), @SVector[-32.0, 0.0], @SVector[-20.0, 0.0], false)
   @test check_vs_refv(^, x1_d1, -1, yref_d1_pow1, mctol)
   @test check_vs_refv(^, x1_d1, 2, yref_d1_pow2, mctol)
   @test check_vs_refv(^, x5_d1, 2, yref_d1_pow3, mctol)
   @test check_vs_refv(^, x6_d1, 2, yref_d1_pow4, mctol)
   @test check_vs_refv(^, x1_d1, 3, yref_d1_pow5, mctol)
   @test check_vs_refv(^, x5_d1, 3, yref_d1_pow6, mctol)
   @test check_vs_refv(^, x6_d1, 3, yref_d1_pow7, mctol)
   @test check_vs_refv(^, x2_d1, -3, yref_d1_pow8a, mctol)
   @test check_vs_refv(^, x1_d1, -3, yref_d1_pow8, mctol)
   @test check_vs_refv(^, x5_d1, -3, yref_d1_pow9, mctol)
   @test check_vs_refv(^, x1_d1, -4, yref_d1_pow10, mctol)
   @test check_vs_refv(^, x5_d1, -4, yref_d1_pow11, mctol)
   @test check_vs_refv(^, x1_d1, 4, yref_d1_pow12, mctol)
   @test check_vs_refv(^, x5_d1, 4, yref_d1_pow13, mctol)
   @test check_vs_refv(^, x6_d1, 4, yref_d1_pow14, mctol)

   x = MC{2,NS}(2.1, 3.3, Interval(1.1,4.3), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   c = 1.2
   cInt = 2

   # c>0
   z01a = McCormick.plus_kernel(x, Float32(c), x.Intv + Float32(c))
   z02a = McCormick.plus_kernel(Float32(c), x, x.Intv + Float32(c))
   z01b = McCormick.plus_kernel(x, Float16(c), x.Intv + Float16(c))
   z02b = McCormick.plus_kernel(Float16(c), x, x.Intv + Float16(c))
   z03a = McCormick.plus_kernel(x, Int32(cInt), x.Intv + Int32(cInt))
   z04a = McCormick.plus_kernel(Int32(cInt), x, x.Intv + Int32(cInt))
   z03b = McCormick.plus_kernel(x, Int16(cInt), x.Intv + Int16(cInt))
   z04b = McCormick.plus_kernel(Int16(cInt), x, x.Intv + Int16(cInt))
   @test z03a.cv == z04a.cv == z03b.cv == z04b.cv == 4.1
   @test z03a.cc == z04a.cc == z03b.cc == z04b.cc == 5.3
   @test isapprox(z01a.cv, 3.3, atol=1E-6)
   @test isapprox(z02a.cv, 3.3, atol=1E-6)
   @test isapprox(z01b.cv, 3.3001953, atol=1E-6)
   @test isapprox(z02b.cv, 3.3001953, atol=1E-6)

   z05a = McCormick.mult_kernel(x, Float32(c), x.Intv*Float32(c))
   z06a = McCormick.mult_kernel(Float32(c), x, Float32(c)*x.Intv)
   z05b = McCormick.plus_kernel(x, Float16(c), x.Intv + Float16(c))
   z06b = McCormick.plus_kernel(Float16(c), x, x.Intv + Float16(c))
   z07a = McCormick.mult_kernel(x, Int32(cInt), x.Intv*Int32(cInt))
   z08a = McCormick.mult_kernel(Int32(cInt), x, Int32(cInt)*x.Intv)
   z07b = McCormick.mult_kernel(x, Int16(cInt), x.Intv*Int16(cInt))
   z08b = McCormick.mult_kernel(Int16(cInt), x, Int16(cInt)*x.Intv)
   @test z07a.cv == z08a.cv == z07b.cv == z08b.cv == 4.2
   @test z07a.cc == z08a.cc == z07b.cc == z08b.cc == 6.6
   @test isapprox(z05a.cv, 2.5200001, atol=1E-6)
   @test isapprox(z06a.cv, 2.5200001, atol=1E-6)
   @test isapprox(z05b.cc, 4.5001953125, atol=1E-6)
   @test isapprox(z06b.cc, 4.5001953125, atol=1E-6)

   z09a = McCormick.div_kernel(x, Float32(c), x.Intv/Float32(c))
   z10a = McCormick.div_kernel(Float32(c), x, Float32(c)/x.Intv)
   z11a = McCormick.div_kernel(x, Int32(cInt), x.Intv/Int32(cInt))
   z12a = McCormick.div_kernel(Int32(cInt), x, Int32(cInt)/x.Intv)
   z09b = McCormick.div_kernel(x, Float16(c), x.Intv/Float16(c))
   z10b = McCormick.div_kernel(Float16(c), x, Float16(c)/x.Intv)
   z11b = McCormick.div_kernel(x, Int16(cInt), x.Intv/Int16(cInt))
   z12b = McCormick.div_kernel(Int16(cInt), x, Int16(cInt)/x.Intv)
   @test isapprox(z09a.cv, 1.74999999, atol=1E-6)
   @test isapprox(z10a.cv, 0.36363637, atol=1E-6)
   @test isapprox(z11a.cv, 1.05, atol=1E-6)
   @test isapprox(z12a.cv, 0.6060606060606061, atol=1E-6)
   @test isapprox(z09b.cv, 1.74931640, atol=1E-6)
   @test isapprox(z10b.cv, 0.36369554, atol=1E-6)
   @test isapprox(z11b.cv, 1.05, atol=1E-6)
   @test isapprox(z12b.cv, 0.6060606060606061, atol=1E-6)

   # c<0
   z13a = McCormick.mult_kernel(x, Float32(-c), x.Intv*Float32(-c))
   z13b = McCormick.mult_kernel(x, Float16(-c), x.Intv*Float16(-c))
   @test isapprox(z13a.cv, -3.9600001, atol=1E-6)
   @test isapprox(z13b.cv, -3.96064453125, atol=1E-6)

   X = MC{2,Diff}(3.0, 3.0, Interval{Float64}(2.0,4.0), seed_gradient(1,Val(2)),seed_gradient(1,Val(2)),false)
   out1 = McCormick.flt_pow_1(X, 0.5, X.Intv^0.5)
   @test isapprox(out1.cv, 1.70710678, atol=1E-5)
   @test isapprox(out1.cc, 1.7320508, atol=1E-5)

   X = MC{2,NS}(3.0, 3.0, Interval{Float64}(2.0,4.0), seed_gradient(1,Val(2)),seed_gradient(1,Val(2)),false)
   out2 = McCormick.flt_pow_1(X, 0.5, X.Intv^0.5)
   @test isapprox(out2.cv, 1.70710678, atol=1E-5)
   @test isapprox(out2.cc, 1.7320508, atol=1E-5)

   a = MC{5,NS}(1.0,(Interval{Float64}(0.1,0.9)),2)
   b = MC{5,NS}(3.0,(Interval{Float64}(2.1,3.9)),2)
   m1 = b^a
   @test isapprox(m1.cv, 2.8948832853594535, atol=1E-6)
   @test isapprox(m1.cc, 3.1156729751564813, atol=1E-6)

   b = MC{5,NS}(3.0,(Interval{Float64}(2.1,3.9)),2)
   m1 = pow(b, Float32(2))
   @test m1.cv == 9.0
   @test isapprox(m1.cc, 9.809999999999999, atol=1E-6)

   b = MC{5,NS}(3.0,(Interval{Float64}(-2.1,3.9)),2)
   @test isnan(pow(b,-3))

   @test McCormick.pow_deriv(1.2, 2) == 2.4

   b = acos(MC{5,NS}(0.1,(Interval{Float64}(-0.9,0.8)),2))
   @test isapprox(b.cc, 1.5168930889582808, atol=1E-6)
   @test isapprox(b.cv, 1.4241657599950592, atol=1E-6)

   b = acos(MC{5,NS}(-0.1,(Interval{Float64}(-0.9,0.8)),2))
   @test isapprox(b.cc, 1.7516276395253312, atol=1E-6)
   @test isapprox(b.cv, 1.6472128031955662, atol=1E-6)

   b = acos(MC{5,NS}(-0.3,(Interval{Float64}(-0.9,-0.1)),2))
   @test isapprox(b.cc, 1.8754889808102941, atol=1E-6)
   @test isapprox(b.cv, 1.9258642714157252, atol=1E-6)

   b = acos(MC{5,NS}(0.3,(Interval{Float64}(0.1,0.9)),2))
   @test isapprox(b.cc, 1.2157283821740683, atol=1E-6)
   @test isapprox(b.cv, 1.2661036727794992, atol=1E-6)

   b = asec(MC{5,NS}(1.3,(Interval{Float64}(1.1,2.9)),2))
   @test isapprox(b.cc, 0.6252740702428741, atol=1E-6)
   @test isapprox(b.cv, 0.5616171714855654, atol=1E-6)

   b = acsc(MC{5,NS}(1.3,(Interval{Float64}(1.1,2.9)),2))
   @test isapprox(b.cc, 1.0534253760507535, atol=1E-6)
   @test isapprox(b.cv, 0.8776364192181427, atol=1E-6)

   b = asecd(MC{5,NS}(1.3,(Interval{Float64}(1.1,2.9)),2))
   @test isapprox(b.cc, 35.82556526388326, atol=1E-6)
   @test isapprox(b.cv, 32.1782936281979, atol=1E-6)

   b = acscd(MC{5,NS}(1.3,(Interval{Float64}(1.1,2.9)),2))
   @test isapprox(b.cc, 60.356828079689805, atol=1E-6)
   @test isapprox(b.cv, 50.28486276817379, atol=1E-6)
end

@testset "Multiplication Operator" begin

    mctol = 1E-4

    ##### Case 1 #####
    x1 = MC{2,NS}(0.0, 0.0, Interval{Float64}(-2.0,1.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
    y1 = MC{2,NS}(1.0, 1.0, Interval{Float64}(-1.0,2.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)), false)
    yref1 = MC{2,NS}(-1.0, 2.0, Interval{Float64}(-4.0,2.0), @SVector[2.0, 1.0], @SVector[2.0, -2.0], false)
    @test check_vs_ref2(*, x1, y1, yref1, mctol)

    ##### Case 2 #####
    x2 = MC{2,NS}(3.0, 3.0, Interval{Float64}(1.0,5.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
    y2 = MC{2,NS}(1.0, 1.0, Interval{Float64}(-1.0,2.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)), false)
    yref2 = MC{2,NS}(1.0, 5.0, Interval{Float64}(-5.0,10.0), @SVector[2.0, 5.0], @SVector[2.0, 1.0], false)
    @test check_vs_ref2(*, x2, y2, yref2, mctol)

    ##### Case 3 #####
    x3 = MC{2,NS}(-4.0, -4.0, Interval{Float64}(-6.0,-2.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
    y3 = MC{2,NS}(2.0, 2.0, Interval{Float64}(1.0,3.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)), false)
    yref3 = MC{2,NS}(-10.0, -6.0, Interval{Float64}(-18.000000000000004,-1.9999999999999998), @SVector[3.0, -2.0], @SVector[1.0, -2.0], false)
    @test check_vs_ref2(*, x3, y3, yref3, mctol)

    ##### Case 4 #####
    x4 = MC{2,NS}(-4.0, -4.0, Interval{Float64}(-6.0,-2.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
    y4 = MC{2,NS}(-5.0, -5.0, Interval{Float64}(-7.0,-3.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)), false)
    yref4 = MC{2,NS}(16.0, 24.0, Interval{Float64}(6.0,42.0), @SVector[-3.0, -2.0], @SVector[-7.0, -2.0], false)
    @test check_vs_ref2(*, x4, y4, yref4, mctol)

    ##### Case 5 #####
    x5 = MC{2,NS}(-4.0, -4.0, Interval{Float64}(-6.0,-2.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
    y5 = MC{2,NS}(-5.0, -5.0, Interval{Float64}(-7.0,4.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)), false)
    yref5 = MC{2,NS}(16.0, 24.0, Interval{Float64}(-24.000000000000004, 42.00000000000001), @SVector[-7.0, -6.0], @SVector[-7.0, -2.0], false)
    @test check_vs_ref2(*, x5, y5, yref5, mctol)

    ##### Case 6 #####
    x6 = MC{2,NS}(-2.0, -2.0, Interval{Float64}(-3.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
    y6 = MC{2,NS}(3.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)), false)
    yref6 = MC{2,NS}(-8.0, -5.0, Interval{Float64}(-12.0, 16.0), @SVector[1.0, -3.0], @SVector[4.0, -3.0], false)
    @test check_vs_ref2(*, x6, y6, yref6, mctol)

    ##### Case 7 #####
    x7 = MC{2,NS}(-2.0, -2.0, Interval{Float64}(-3.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
    y7 = MC{2,NS}(-4.0, -4.0, Interval{Float64}(-5.0,-3.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)), false)
    yref7 = MC{2,NS}(7.0, 9.0, Interval{Float64}(-20.0, 15.0), @SVector[-5.0, -3.0], @SVector[-3.0, -3.0], false)
    @test check_vs_ref2(*, x7, y7, yref7, mctol)

    ##### Testing for Smooth Standard Mult #####
    seed1 = seed_gradient(1,Val(2))
    seed2 = seed_gradient(2,Val(2))
    x1 = MC{2,Diff}(0.0,0.0,Interval{Float64}(-200.0,200.0),seed1,seed1,false)
    y1 = MC{2,Diff}(200.0,200.0,Interval{Float64}(0.0,400.0),seed2,seed2,false)
    z1 = x1*y1
    @test isapprox(z1.cc,40000,atol=1E-4)
    @test isapprox(z1.cv,-40000,atol=1E-4)

    x2 = MC{2,Diff}(170.0,170.0,Interval{Float64}(100.0,240.0),seed1,seed1,false)
    y2 = MC{2,Diff}(250.0,250.0,Interval{Float64}(100.0,400.0),seed2,seed2,false)
    z2 = x2*y2
    @test isapprox(z2.cc,53000,atol=1E-4)
    @test isapprox(z2.cv,32000,atol=1E-4)

    x3 = MC{2,Diff}(-200.0,-200.0,Interval{Float64}(-300.0,-100.0),seed1,seed1,false)
    y3 = MC{2,Diff}(-300.0,-300.0,Interval{Float64}(-400.0,-200.0),seed2,seed2,false)
    z3 = x3*y3
    @test isapprox(z3.cc,70000,atol=1E-4)
    @test isapprox(z3.cv,50000,atol=1E-4)

    # CHECK ME AGAIN???? -47187.5 new, -47460.9375 old
    x4 = MC{2,Diff}(150.0,150.0,Interval{Float64}(100.0,200.0),seed1,seed1,false)
    y4 = MC{2,Diff}(-250.0,-250.0,Interval{Float64}(-500.0,-100.0),seed2,seed2,false)
    z4 = x4*y4
    @test isapprox(z4.cc,-30000,atol=1E-3)
    @test isapprox(z4.cv,-47187.5,atol=1E-3)

    x5 = MC{2,Diff}(-150.0,-150.0,Interval{Float64}(-200.0,-100.0),seed1,seed1,false)
    y5 = MC{2,Diff}(300.0,300.0,Interval{Float64}(200.0,400.0),seed2,seed2,false)
    z5 = x5*y5
    @test isapprox(z5.cv,-50000,atol=1E-4)
    @test isapprox(z5.cc,-40000,atol=1E-4)

    x1a = MC{2,NS}(Interval(2.0, 4.0))
    x2b = MC{2,NS}(Interval(0.25, 0.5))
    x2c = MC{2,NS}(Interval(1.0, 2.0))
    z1 = Interval(2.0, 4.0)*Interval(0.25, 0.5)
    z2 =Interval(2.0, 4.0)*Interval(1.0, 2.0)

    out1 = McCormick.mul1_u1pos_u2mix(x1a, x2b, z1, false)
    @test out1.cv == 1.0
    @test out1.cc == 1.5
    @test isapprox(out1.Intv.lo, 0.5, atol =1E-6)
    @test isapprox(out1.Intv.hi, 2.0, atol =1E-6)

    out2 = McCormick.mul1_u1pos_u2mix(x1a, x2c, z2, false)
    @test out2.cv == 4.0
    @test out2.cc == 6.0
    @test isapprox(out2.Intv.lo, 2.0, atol =1E-6)
    @test isapprox(out2.Intv.hi, 8.0, atol =1E-6)

    out3 = McCormick.mul2_u1pos_u2mix(x1a, x2b, z1, false)
    @test out3.cv == 1.0
    @test out3.cc == 1.5
    @test isapprox(out3.Intv.lo, 0.5, atol =1E-6)
    @test isapprox(out3.Intv.hi, 2.0, atol =1E-6)

    out4 = McCormick.mul2_u1pos_u2mix(x1a, x2c, z2, false)
    @test out4.cv == 4.0
    @test out4.cc == 6.0
    @test isapprox(out4.Intv.lo, 2.0, atol =1E-6)
    @test isapprox(out4.Intv.hi, 8.0, atol =1E-6)

    x1a = MC{2,NS}(1.1, 2.3, Interval(0.1,3.3), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)
    x2a = MC{2,NS}(2.1, 3.3, Interval(1.1,4.3), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)
    mc1 = McCormick.mul1_u1mix_u2mix(x1a, x2a, x1a.Intv*x2a.Intv, false)
    mc2 = McCormick.mul1_u1mix_u2mix(x2a, x1a, x1a.Intv*x2a.Intv, false)
    @test mc1.cv == 2.75
    @test isapprox(mc1.cc, 8.46999999, atol=1E-6)
    @test isapprox(mc1.Intv.lo, 0.11, atol=1E-6)
    @test isapprox(mc1.Intv.hi, 14.19, atol=1E-6)
    @test mc2.cv == 2.75
    @test isapprox(mc2.cc, 8.46999999, atol=1E-6)

    flt1 = 1.34
    flt2 = 0.57
    @test isapprox(McCormick.mul_MV_ns1cv(flt1, flt2, x1a, x2a), -6.54699999, atol=1E-6)
    @test isapprox(McCormick.mul_MV_ns2cv(flt1, flt2, x1a, x2a), 1.421, atol=1E-6)
    @test isapprox(McCormick.mul_MV_ns3cv(flt1, flt2, x1a, x2a), 1.421, atol=1E-6)
    @test isapprox(McCormick.mul_MV_ns1cc(flt1, flt2, x1a, x2a), -0.27499999, atol=1E-6)
    @test isapprox(McCormick.mul_MV_ns2cc(flt1, flt2, x1a, x2a), 5.389, atol=1E-6)
    @test isapprox(McCormick.mul_MV_ns3cc(flt1, flt2, x1a, x2a), -0.27499999, atol=1E-6)
    @test ~McCormick.tol_MC(flt1, flt2)
end

@testset "Division" begin

   X1 = MC{2,Diff}(3.0,3.0,Interval{Float64}(2.0,4.0), seed_gradient(1,Val(2)),seed_gradient(1,Val(2)),false)
   Y1 = MC{2,Diff}(-4.0,-4.0,Interval{Float64}(-5.0,2.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)
   @test isnan(X1/Y1)

   X2 = MC{2,NS}(3.0,3.0,Interval{Float64}(2.0,4.0), seed_gradient(1,Val(2)),seed_gradient(1,Val(2)),false)
   Y2 = MC{2,NS}(-4.0,-4.0,Interval{Float64}(-5.0,2.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)
   @test isnan(X2/Y2)

   @test X1/X1 == one(X1)
   @test X2/X2 == one(X2)

    X = MC{2,NS}(3.0,3.0,Interval{Float64}(2.0,4.0), seed_gradient(1,Val(2)),seed_gradient(1,Val(2)),false)
    Y = MC{2,NS}(-4.0,-4.0,Interval{Float64}(-5.0,-3.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)
    out = X/Y
    @test isapprox(out.cc,-0.7000000000000002,atol=1E-6)
    @test isapprox(out.cv, -0.8666666666666665, atol=1E-6)
    @test isapprox(out.cc_grad[1],-0.19999999999999998, atol=1E-6)
    @test isapprox(out.cc_grad[2],-0.125,atol=1E-6)
    @test isapprox(out.cv_grad[1], -0.33333333333333337, atol=1E-6)
    @test isapprox(out.cv_grad[2], -0.1333333333333333, atol=1E-6)
    @test isapprox(out.Intv.lo,-1.33333333,atol=1E-6)
    @test isapprox(out.Intv.hi,-0.39999999999999997,atol=1E-6)

    x1a = MC{2,MV}(1.1, 2.3, Interval(0.1,3.3), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)
    x2a = MC{2,MV}(2.1, 3.3, Interval(1.1,4.3), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)

    div1 = x2a/Float32(1.1)
    div2 = x2a/Int32(2)
    @test isapprox(div1.cv, 1.9090908, atol=1E-6)
    @test isapprox(div1.cc, 2.9999999, atol=1E-6)
    @test isapprox(div2.cv, 1.05, atol=1E-6)
    @test isapprox(div2.cc, 1.65, atol=1E-6)

    x = Interval{Float64}(2.0, 5.0)
    y = Interval{Float64}(3.0, 7.0)
    z = 2.9
    es = 2.4
    nu = 3.5
    omega = 3.1
    @test isapprox( McCormick.div_alphaxy(es, nu, x, y), 0.676190476, atol = 1E-5)
    @test isapprox(McCormick.div_gammay(omega, y), -5.3388888888, atol = 1E-5)
    @test isapprox(McCormick.div_deltaxy(omega, x, y), -3.05079365, atol = 1E-5)
    @test isapprox(McCormick.div_psixy(es, nu, x, y), 0.67620756, atol = 1E-5)
    @test McCormick.div_omegaxy(x, y) == -3.5
    @test isapprox(McCormick.div_lambdaxy(es, nu, x), 0.66341388, atol=1E-5)
    @test McCormick.div_nuline(x, y, z)  == 4.2

    X = MC{2,Diff}(3.0,3.0,Interval{Float64}(2.0,4.0), seed_gradient(1,Val(2)),seed_gradient(1,Val(2)),false)
    Y = MC{2,Diff}(-4.0,-4.0,Interval{Float64}(-5.0,-3.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)

    out_mc1 = McCormick.div_diffcv(X, Y)
    @test isapprox(out_mc1[1], -0.72855339, atol=1E-5)
    @test isapprox(out_mc1[2][1], -0.25, atol=1E-5)
    @test isapprox(out_mc1[2][2], -0.18213834, atol=1E-5)

    out_mc2 = McCormick.div_MV(X, Y, X.Intv/Y.Intv)
    @test isapprox(out_mc2.cv, -0.86666666666, atol=1E-5)
    @test isapprox(out_mc2.cc, -0.73333333333, atol=1E-5)

    out_mc3 = McCormick.div_kernel(X, X, X.Intv/X.Intv)
    @test out_mc3.cv == 1.0

    X = MC{2,Diff}(3.0,3.0,Interval{Float64}(2.0,4.0), seed_gradient(1,Val(2)),seed_gradient(1,Val(2)),false)
    Y = MC{2,Diff}(-4.0,-4.0,Interval{Float64}(-5.0,-3.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)

    out_mc1 = McCormick.div_diffcv(X, -Y)
    @test isapprox(out_mc1[1], 0.72855339, atol=1E-5)
    @test isapprox(out_mc1[2][1], 0.25, atol=1E-5)
    @test isapprox(out_mc1[2][2], 0.18213834, atol=1E-5)

    out_mc2 = McCormick.div_MV(X, -Y, X.Intv/-Y.Intv)
    @test isapprox(out_mc2.cv, 0.72855339, atol=1E-5)
    @test isapprox(out_mc2.cc, 0.86666666666, atol=1E-5)

    out_mc3 = McCormick.div_kernel(X, -Y, X.Intv/-Y.Intv)
    @test isapprox(out_mc3.cv, 0.72855339, atol=1E-5)
    @test isapprox(out_mc3.cc, 0.86666666666, atol=1E-5)

    X = Interval{Float64}(-2,2)
    Y = Interval{Float64}(-2,2)
    xpnt1 = 2.0; ypnt1 = 1.0
    xpnt2 = 1.0; ypnt2 = 2.0
    xpnt3 = 0.0; ypnt3 = 0.0

    x1 = MC{1,MV}(xpnt1, X, 1)
    y1 = MC{1,MV}(ypnt1, Y, 2)
    out1 =  max((y1-1)^2,x1*y1)*min(y1^2,(x1+1)*y1)
    @test out1.cv == -29.7
    @test isapprox(out1.cc, 22.666666666666668, atol = 1E-6)

   x2 = MC{1,MV}(xpnt2, X, 1)
   y2 = MC{1,MV}(ypnt2, Y, 2)
   out2 = max((y2-1)^2,x2*y2)*min(y2^2,(x2+1)*y2)
   @test isapprox(out2.cv, -7.0, atol = 1E-6)
   @test isapprox(out2.cc, 16.0, atol = 1E-6)

   x3 = MC{1,MV}(xpnt3, X, 1)
   y3 = MC{1,MV}(ypnt3, Y, 2)
   out3 = max((y3-1)^2,x3*y3)*min(y3^2,(x3+1)*y3)
   @test isapprox(out3.cv, -40.666666666666664, atol = 1E-6)
   @test isapprox(out3.cc, 27.11111111111111, atol = 1E-6)

   b = MC{5,Diff}(3.0,(Interval{Float64}(2.1,3.9)),2)
   a = MC{5,Diff}(1.0,(Interval{Float64}(-5.1,5.9)),2)
   @test isnan(b/a)

   b = MC{5,Diff}(3.0,(Interval{Float64}(2.1,3.9)),2)
   a = MC{5,Diff}(5.5,(Interval{Float64}(5.1,5.9)),2)
   m1 = b/a
   @test isapprox(m1.cv, 0.5328925094773488, atol=1E-6)
   @test isapprox(m1.cc, 0.5603190428713859, atol=1E-6)

   b = MC{5,Diff}(3.0,Interval{Float64}(2.1,3.9), 2)
   m1 = McCormick.inv1(b, Interval{Float64}(2.1,3.9))
   @test isapprox(m1.cv, 0.3333333333333333, atol=1E-6)
   @test isapprox(m1.cc, 0.36630036630036633, atol=1E-6)
end

@testset "Min/Max" begin


   @test McCormick.deriv_max(2.0, 1.0) == 1.0
   c = 5.0
   z = Interval{Float64}(2.1,3.4)
   x = MC{2,NS}(z)
   x0 = MC{2,NS}(Interval{Float64}(1.1,5.4))

   @test max(c, x) == McCormick.max_kernel(x, c, max(x.Intv, c))
   @test max(x, Float32(c)) == McCormick.max_kernel(x, convert(Float64, Float32(c)), max(x.Intv, Float32(c)))
   @test max(Float32(c), x) == McCormick.max_kernel(x, convert(Float64, Float32(c)), max(x.Intv, Float32(c)))
   @test max(x, Int64(c)) == McCormick.max_kernel(x, convert(Float64, Int64(c)), max(x.Intv, Int64(c)))
   @test max(Int64(c), x) == McCormick.max_kernel(x, convert(Float64, Int64(c)), max(x.Intv, Int64(c)))

   @test min(c, x) == McCormick.min_kernel(x, c, min(x.Intv, c))
   @test min(x, Float32(c)) == McCormick.min_kernel(x, convert(Float64, Float32(c)), min(x.Intv, Float32(c)))
   @test min(Float32(c), x) == McCormick.min_kernel(x, convert(Float64, Float32(c)), min(x.Intv, Float32(c)))
   @test min(x, Int64(c)) == McCormick.min_kernel(x, convert(Float64, Int64(c)), min(x.Intv, Int64(c)))
   @test min(Int64(c), x) == McCormick.min_kernel(x, convert(Float64, Int64(c)), min(x.Intv, Int64(c)))

   @test McCormick.minus_kernel(x, Float32(c), z) == McCormick.minus_kernel(x, convert(Float64,Float32(c)), z)
   @test McCormick.minus_kernel(Float32(c), x, z) == McCormick.minus_kernel(convert(Float64,Float32(c)), x, z)
   @test McCormick.minus_kernel(x, Int64(c), z) == McCormick.minus_kernel(x, convert(Float64,Int64(c)), z)
   @test McCormick.minus_kernel(Int64(c), x, z) == McCormick.minus_kernel(convert(Float64,Int64(c)), x, z)

   @test McCormick.plus_kernel(x, z) == x

   @test zero(x) == zero(MC{2,NS})
   @test one(x) == one(MC{2,NS})
   @test isnan(nan(x).cv) && isnan(nan(x).cc)
   @test real(x) == x
   @test dist(x, x0) == max(abs(x.cc - x0.cc), abs(x.cv - x0.cv))
   @test eps(x) == max(eps(x.cc), eps(x.cv))
   @test mid(x) == mid(x.Intv)
   @test one(x) == MC{2,NS}(1.0, 1.0, one(Interval{Float64}), zero(SVector{2,Float64}), zero(SVector{2,Float64}), true)

   X = MC{2,NS}(3.0,3.0,Interval{Float64}(2.0,4.0), seed_gradient(1,Val(2)),seed_gradient(1,Val(2)),false)
   out1 = McCormick.max_kernel(3.0, X, max(3.0, X.Intv))
   out2 = McCormick.max_kernel(X, Float32(3.0), max(X.Intv, Float32(3.0)))
   out3 = McCormick.max_kernel(Float32(3.0), X, max(X.Intv, Float32(3.0)))
   out4 = McCormick.max_kernel(X, Int32(3), max(X.Intv, Int32(3.0)))
   out5 = McCormick.max_kernel(Int32(3), X, max(X.Intv, Int32(3.0)))
   out6 = McCormick.min_kernel(X, Float32(3.0), min(X.Intv, Float32(3.0)))
   out7 = McCormick.min_kernel(Float32(3.0), X, min(Float32(3.0), X.Intv))
   out8 = McCormick.min_kernel(X, Int32(3), min(X.Intv, Int32(3.0)))
   out9 = McCormick.min_kernel(Int32(3), X, min(Int32(3.0), X.Intv))

   @test out1.cv == 3.0
   @test out2.cv == 3.0
   @test out3.cv == 3.0
   @test out4.cv == 3.0
   @test out5.cv == 3.0
   @test out6.cv == 2.5
   @test out7.cv == 2.5
   @test out8.cv == 2.5
   @test out9.cv == 2.5

   @test out1.cc == 3.5
   @test out2.cc == 3.5
   @test out3.cc == 3.5
   @test out4.cc == 3.5
   @test out5.cc == 3.5
   @test out6.cc == 3.0
   @test out7.cc == 3.0
   @test out8.cc == 3.0
   @test out9.cc == 3.0

   X = MC{2,NS}(3.0,3.0,Interval{Float64}(2.0,4.0), seed_gradient(1,Val(2)),seed_gradient(1,Val(2)),false)
   Y = MC{2,NS}(-4.0,-4.0,Interval{Float64}(-5.0,-3.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)
   out1 = McCormick.max_kernel(X, Y, max(X.Intv,Y.Intv))
   @test out1.cv == 3.0
   @test out1.cc == 3.0

   X = MC{2,NS}(2.0,3.0,Interval{Float64}(2.0,4.0), seed_gradient(1,Val(2)),seed_gradient(1,Val(2)),false)
   Y = MC{2,NS}(4.0,4.0,Interval{Float64}(1.0,6.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)
   out2 = McCormick.max_kernel(X, Y, max(X.Intv,Y.Intv))
   @test out2.cv == 4.0
   @test isapprox(out2.cc, 5.3571428, atol=1E-5)

   X = MC{2,Diff}(3.0,3.0,Interval{Float64}(2.0,4.0), seed_gradient(1,Val(2)),seed_gradient(1,Val(2)),false)
   Y = MC{2,Diff}(-4.0,-4.0,Interval{Float64}(-5.0,-3.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)
   out3 = McCormick.max_kernel(X, Y, max(X.Intv,Y.Intv))
   @test out3.cv == 3.0
   @test out3.cc == 3.0

   X = MC{2,Diff}(2.0,3.0,Interval{Float64}(2.0,4.0), seed_gradient(1,Val(2)),seed_gradient(1,Val(2)),false)
   Y = MC{2,Diff}(4.0,4.0,Interval{Float64}(1.0,6.0), seed_gradient(2,Val(2)), seed_gradient(2,Val(2)),false)
   out4 = McCormick.max_kernel(X, Y, max(X.Intv,Y.Intv))
   @test out4.cv == 3.0
   @test out4.cc == 5.2

   x = MC{2, NS}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   x1 = MC{2, NS}(2.1, 3.1, Interval{Float64}(1.5,3.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   out1 = McCormick.interval_MC(x)
   @test out1.cv == 1.0
   @test out1.cc == 4.0

   y = MC{2, Diff}(2.0, 3.0, Interval{Float64}(1.0,4.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   z = MC{2, Diff}(2.1, 3.1, Interval{Float64}(1.5,3.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   out2 = intersect(y, z)
   @test isapprox(out2.cv, 0.3, atol=1E-5)
   @test out2.cc == 3.1

   out2a = intersect(x1, x)
   @test out2a.cv == 2.1
   @test out2a.cc == 3.0

   out2b = intersect(x, Interval{Float64}(-4.0,-1.0))
   @test isnan(out2b.cv)
   @test isnan(out2b.cc)

   out2c = intersect(x, Interval{Float64}(9.0,10.0))
   @test isnan(out2c.cv)
   @test isnan(out2c.cc)

   out2c1 = intersect(x, 2.0)
   @test out2c1.cv == 2.0
   @test out2c1.cc == 2.0

   out2c2 = intersect(2.0, x)
   @test out2c2.cv == 2.0
   @test out2c2.cc == 2.0

   X = Interval{Float64}(-2,2)
   Y = Interval{Float64}(-2,2)
   xpnt1 = 2.0; ypnt1 = 1.0
   xpnt2 = 1.0; ypnt2 = 2.0
   xpnt3 = 0.0; ypnt3 = 0.0

   x1 = MC{1,MV}(xpnt1, X, 1)
   y1 = MC{1,MV}(ypnt1, Y, 2)
   out1 = max((y1-1)^2, x1^2) + min(y1^2, (x1+1)^2)
   @test out1.cv == 5.0
   @test isapprox(out1.cc, 9.666666666666668, atol = 1E-6)

   x2 = MC{1,MV}(xpnt2, X, 1)
   y2 = MC{1,MV}(ypnt2, Y, 2)
   out2 = max((y2-1)^2, x2^2) + min(y2^2, (x2+1)^2)
   @test isapprox(out2.cv, 2.7777777777777777, atol = 1E-6)
   @test isapprox(out2.cc, 8.555555555555555, atol = 1E-6)

   x3 = MC{1,MV}(xpnt3, X, 1)
   y3 = MC{1,MV}(ypnt3, Y, 2)
   out3 = max((y3-1)^2, x3^2) + min(y3^2, (x3+1)^2)
   @test out3.cv == 1.0
   @test isapprox(out3.cc, 10.777777777777779, atol = 1E-6)

   x1 = MC{1,Diff}(xpnt1, X, 1)
   y1 = MC{1,Diff}(ypnt1, Y, 2)
   out1 = max((y1-1)^2, x1^2) + min(y1^2, (x1+1)^2)
   @test out1.cv == 3.0
   @test isapprox(out1.cc, 9.666666666666666, atol = 1E-6)

   x2 = MC{1,Diff}(xpnt2, X, 1)
   y2 = MC{1,Diff}(ypnt2, Y, 2)
   out2 = max((y2-1)^2, x2^2) + min(y2^2, (x2+1)^2)
   @test out2.cv == 1.0
   @test isapprox(out2.cc, 8.555555555555555, atol = 1E-6)

   x3 = MC{1,Diff}(xpnt3, X, 1)
   y3 = MC{1,Diff}(ypnt3, Y, 2)
   out3 = max((y3-1)^2, x3^2) + min(y3^2, (x3+1)^2)
   @test isapprox(out3.cv, 0.3333333333333333, atol = 1E-6)
   @test isapprox(out3.cc, 10.777777777777779, atol = 1E-6)

   a = MC{5,NS}(1.0,(Interval{Float64}(0.1,0.9)),2)
   b = MC{5,NS}(3.0,(Interval{Float64}(2.1,3.9)),2)
   m1 = max(a,b)
   @test m1.cv == 3.0
   @test m1.cc == 3.0

   a = MC{5,MV}(1.0,(Interval{Float64}(0.1,0.9)),2)
   b = MC{5,MV}(3.0,(Interval{Float64}(2.1,3.9)),2)
   m1 = max(a,b)
   @test m1.cv == 3.0
   @test m1.cc == 3.0

   a = MC{5,MV}(1.0,(Interval{Float64}(0.1,0.9)),2)
   b = MC{5,MV}(3.0,(Interval{Float64}(2.1,3.9)),2)
   m1 = max(b,a)
   @test m1.cv == 3.0
   @test m1.cc == 3.0

   b = MC{5,Diff}(3.0,(Interval{Float64}(2.1,3.9)),2)
   m1 = intersect(b, Interval(1.0,3.0))
   @test isapprox(m1.cv, 2.1, atol=1E-6)
   @test m1.cc == 3.0

   X = Interval{Float64}(-2,2)
   Y = Interval{Float64}(-2,2)
   xpnt1 = 2.0; ypnt1 = 1.0
   xpnt2 = 1.0; ypnt2 = 2.0
   xpnt3 = 0.0; ypnt3 = 0.0

   x1 = MC{1,Diff}(xpnt1, X, 1)
   y1 = MC{1,Diff}(ypnt1, Y, 2)
   out1 = max((y1-1)^2, x1*y1)*min(y1^2, (x1+1)*y1)
   @test isapprox(out1.cv, -33.79070216049382, atol=1E-6)
   @test isapprox(out1.cc, 33.160493827160494, atol=1E-6)

   x1 = MC{1,Diff}(xpnt2, X, 1)
   y1 = MC{1,Diff}(ypnt2, Y, 2)
   out1 = max((y1-1)^2, x1*y1)*min(y1^2, (x1+1)*y1)
   @test isapprox(out1.cv, -30.21738254458162, atol=1E-6)
   @test isapprox(out1.cc, 29.47050754458162, atol=1E-6)

   x1 = MC{1,Diff}(xpnt3, X, 1)
   y1 = MC{1,Diff}(ypnt3, Y, 2)
   out1 = max((y1-1)^2, x1*y1)*min(y1^2, (x1+1)*y1)
   @test isapprox(out1.cv, -46.30651577503429, atol=1E-6)
   @test isapprox(out1.cc, 35.478737997256516, atol=1E-6)

   x1 = MC{1,Diff}(xpnt3, X, 1)
   y1 = MC{1,Diff}(ypnt3, Y, 2)
   out1 = McCormick.final_cut(x1, y1)
   @test isapprox(out1.cv, -2.0, atol=1E-6)
   @test isapprox(out1.cc, 0.0, atol=1E-6)

   x1 = MC{1,NS}(xpnt1, X, 1)
   y1 = MC{1,NS}(ypnt2, Y, 2)
   out1 = intersect(x1, y1)
   @test isapprox(out1.cv, 2.0, atol=1E-6)
   @test isapprox(out1.cc, 2.0, atol=1E-6)

   out1 = intersect(y1, x1)
   @test isapprox(out1.cv, 2.0, atol=1E-6)
   @test isapprox(out1.cc, 2.0, atol=1E-6)
end

@testset "NaN Propagation" begin

   x = MC{2,NS}(-3.0, -3.0, Interval{Float64}(-5.0, 5.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   @test_nowarn log(x)
   @test_nowarn log2(x)
   @test_nowarn log10(x)
   @test_nowarn log1p(x)
   @test_nowarn asin(x)
   @test_nowarn acosh(x)
   @test_nowarn atanh(x)
   @test_nowarn acos(x)

end

@testset "Activation Functions" begin

   x1 = MC{1,NS}(0.1, Interval(-1.5, 1.5), 1)
   x2 = MC{1,NS}(-0.5, Interval(-1.5, 1.5), 1)
   x3 = MC{1,NS}(0.7, Interval(0.5, 1.5), 1)

   y1 = relu(x1)
   y2 = relu(x2)
   y3 = relu(x3)

   @test y1.cv == 0.1
   @test isapprox(y1.cc, 0.8000000000000002, atol=1E-8)
   @test y2.cv == 0.0
   @test y2.cc == 0.5
   @test y3.cv == 0.7
   @test y3.cc == 0.7

   @test relu(0.5) == 0.5
   @test McCormick.relu_deriv(0.5) == 1.0
   @test McCormick.relu_deriv2(0.5) == 0.0

   y1 = leaky_relu(x1)
   y2 = leaky_relu(x2)
   y3 = leaky_relu(x3)

   @test y1.cv == 0.1
   @test y1.cc == 0.793
   @test y2.cv == -0.005
   @test y2.cc == 0.49
   @test y3.cv == 0.7
   @test y3.cc == 0.7

   @test leaky_relu(0.5) == 0.5
   @test McCormick.leaky_relu_deriv(0.5) == 1.0
   @test McCormick.leaky_relu_deriv2(0.5) == 0.00

   y1 = param_relu(x1, 0.01)
   y2 = param_relu(x2, 0.01)
   y3 = param_relu(x3, 0.01)

   @test y1.cv == 0.1
   @test y1.cc == 0.793
   @test y2.cv == -0.005
   @test y2.cc == 0.49
   @test y3.cv == 0.7
   @test y3.cc == 0.7

   @test param_relu(0.5, 0.01) == 0.5
   @test param_relu(1, 1) == 1
   @test McCormick.param_relu_deriv(0.5, 0.01) == 1.0

   gout = zeros(2)
   McCormick.param_relu_grad(gout, 1.0, 0.5)
   @test gout[1] == 0.5
   @test gout[2] == 1.0

   McCormick.param_relu_grad(gout, -1.0, 0.5)
   @test gout[1] == 0.5
   @test gout[2] == -1.0

   y1 = maxsig(x1)
   y2 = maxsig(x2)
   y3 = maxsig(x3)

   @test isapprox(y1.cv, 0.52497918747894, atol=1E-8)
   @test isapprox(y1.cc, 0.885131911109633, atol=1E-8)
   @test isapprox(y2.cv, 0.3775406687981454, atol=1E-8)
   @test isapprox(y2.cc, 0.6216170158709042, atol=1E-8)
   @test y3.cv == 0.7
   @test isapprox(y3.cc, 0.7979674649614835, atol=1E-8)


   @test maxsig(1) == 1
   @test isapprox(maxsig(0.5), 0.6224593312018546, atol =1E-8)
   @test isapprox(McCormick.maxsig_deriv(0.5), 0.2350037122015945, atol =1E-8)
   @test isapprox(McCormick.maxsig_deriv2(5.5), 0.0, atol =1E-8)
   @test isapprox(McCormick.maxsig_deriv2(-3.5), 0.026784980676662237, atol =1E-8)

   y1 = elu(x1, 0.01)
   y2 = elu(x2, 0.01)
   y3 = elu(x3, 0.01)

   @test elu(0.5, 0.01) == 0.5
   @test elu(5, 1) == 5
   @test isapprox((elu(Interval(-5.0,-3.0), 0.5)).lo, -0.4966310265004573, atol=1E-8)

   gout = zeros(2)
   McCormick.elu_grad(gout, 1.0, 0.5)
   @test gout[1] == 1.0
   @test gout[2] == 0.0

   @test y1.cv == 0.1
   @test isapprox(y1.cc, 0.7963746074140261, atol=1E-8)
   @test isapprox(y2.cv, -0.003934693402873666, atol=1E-8)
   @test isapprox(y2.cc, 0.4948208677343229, atol=1E-8)
   @test y3.cv == 0.7
   @test y3.cc == 0.7

   y1 = selu(x1, 0.01, 0.5)
   y2 = selu(x2, 0.01, 0.5)
   y3 = selu(x3, 0.01, 0.5)

   @test selu(0.5, 0.01, 0.5) == 0.25

   gout = zeros(3)
   McCormick.selu_grad(gout, 0.5, 1.0, 0.5)
   @test gout[1] == 0.5
   @test gout[2] == 0.0
   @test gout[3] == 0.5


   @test y1.cv == 0.05
   @test isapprox(y1.cc, 0.39818730370701305, atol=1E-8)
   @test isapprox(y2.cv, -0.001967346701436833, atol=1E-8)
   @test isapprox(y2.cc, 0.24741043386716144, atol=1E-8)
   @test y3.cv == 0.35
   @test y3.cc == 0.35

   y1 = maxtanh(x1)
   y2 = maxtanh(x2)
   y3 = maxtanh(x3)

   @test maxtanh(0.75) == 0.75
   @test maxtanh(2) == 2
   @test McCormick.maxtanh_deriv2(0.75) == 0.0
   @test isapprox(McCormick.maxtanh_deriv2(-5.75), 8.103746506240892e-5, atol=1E-8)

   @test y1.cv == 0.1
   @test isapprox(y1.cc, 0.3775974816323957, atol=1E-8)
   @test isapprox(y2.cv, -0.46211715726000974, atol=1E-8)
   @test isapprox(y2.cc, -0.10343216909657771, atol=1E-8)
   @test y3.cv == 0.7
   @test y3.cc == 0.7

   y1 = softplus(x1)
   y2 = softplus(x2)
   y3 = softplus(x3)

   @test isapprox(softplus(7), 7.000911466453774, atol=1E-8)
   @test isapprox(softplus(0.75), 1.1368710061148999, atol=1E-8)
   @test isapprox(McCormick.softplus_deriv2(0.75), 0.217894993761814, atol=1E-8)

   @test isapprox(y1.cv, 0.744396660073571, atol=1E-8)
   @test isapprox(y1.cc, 1.0014132779827525, atol=1E-8)
   @test isapprox(y2.cv, 0.4740769841801067, atol=1E-8)
   @test isapprox(y2.cc, 0.7014132779827524, atol=1E-8)
   @test isapprox(y3.cv, 1.103186048885458, atol=1E-8)
   @test isapprox(y3.cc, 1.1195442429406357, atol=1E-8)

   y1 = pentanh(x1)
   y2 = pentanh(x2)
   y3 = pentanh(x3)

   @test isapprox(pentanh(7), 0.9999983369439447, atol=1E-8)
   @test isapprox(pentanh(0.75), 0.6351489523872873, atol=1E-8)
   @test isapprox(McCormick.pentanh_deriv2(0.75), -0.08974759478157028, atol=1E-8)
   @test isapprox(McCormick.pentanh_deriv2(-0.75), 0.022370912744158675, atol=1E-8)

   @test isapprox(y1.cv, 0.060343216909657764, atol=1E-8)
   @test isapprox(y1.cc, 0.35880541402007554, atol=1E-8)
   @test isapprox(y2.cv, -0.1243530017715962, atol=1E-8)
   @test isapprox(y2.cc, 0.08986935938100252, atol=1E-8)
   @test isapprox(y3.cv, 0.550723376536981, atol=1E-8)
   @test isapprox(y3.cc, 0.6043677771171636, atol=1E-8)

   y1 = sigmoid(x1)
   y2 = sigmoid(x2)
   y3 = sigmoid(x3)

   @test isapprox(sigmoid(7), 0.9990889488055994, atol=1E-8)
   @test isapprox(sigmoid(0.75), 0.679178699175393, atol=1E-8)
   @test isapprox(McCormick.sigmoid_deriv2(0.75), 0.4003874719875089, atol=1E-8)

   @test isapprox(y1.cv, 0.5084495314200309, atol=1E-8)
   @test isapprox(y1.cc, 0.5357111749761996, atol=1E-8)
   @test isapprox(y2.cv, 0.37596741223133967, atol=1E-8)
   @test isapprox(y2.cc, 0.4032290557875084, atol=1E-8)
   @test isapprox(y3.cv, 0.6614823602002124, atol=1E-8)
   @test isapprox(y3.cc, 0.6681877721681662, atol=1E-8)

   y1 = bisigmoid(x1)
   y2 = bisigmoid(x2)
   y3 = bisigmoid(x3)

   @test isapprox(bisigmoid(7), 0.9981778976111988, atol=1E-8)
   @test isapprox(bisigmoid(0.75), 0.35835739835078595, atol=1E-8)
   @test isapprox(McCormick.bisigmoid_deriv2(0.75), -0.1561685661562888, atol=1E-8)

   @test isapprox(y1.cv, 0.01689906284006173, atol=1E-8)
   @test isapprox(y1.cc, 0.07142234995239904, atol=1E-8)
   @test isapprox(y2.cv, -0.2480651755373207, atol=1E-8)
   @test isapprox(y2.cc, -0.19354188842498335, atol=1E-8)
   @test isapprox(y3.cv, 0.3229647204004248, atol=1E-8)
   @test isapprox(y3.cc, 0.3363755443363322, atol=1E-8)

   y1 = softsign(x1)
   y2 = softsign(x2)
   y3 = softsign(x3)

   @test softsign(7) == 0.875
   @test isapprox(softsign(0.75), 0.42857142857142855, atol=1E-8)
   @test isapprox(McCormick.softsign_deriv2(0.75), -0.3731778425655977, atol=1E-8)
   @test isapprox(McCormick.softsign_deriv2(-0.75), 0.3731778425655977, atol=1E-8)

   @test isapprox(y1.cv, -0.05362319120141322, atol=1E-8)
   @test isapprox(y1.cc, 0.14699793280161522, atol=1E-8)
   @test isapprox(y2.cv, -0.3333333333333333, atol=1E-8)
   @test isapprox(y2.cc, -0.1331262919989905, atol=1E-8)
   @test isapprox(y3.cv, 0.3866666666666666, atol=1E-8)
   @test isapprox(y3.cc, 0.4117647058823529, atol=1E-8)

   x4 = MC{1,NS}(-6.5, Interval(-7.5, -5.5), 1)
   x5 = MC{1,NS}(6.5, Interval(5.5, 7.5), 1)
   x6 = MC{1,NS}(0.7, Interval(0.5, 1.5), 1)

   x7 = MC{1,NS}(3.1, Interval(-3.5, 3.5), 1)
   x8 = MC{1,NS}(1.1, Interval(-2.0, 2.5), 1)
   x9 =  MC{1,NS}(0.3, Interval(-2.0, 3.5), 1)

   x10 = MC{1,NS}(-1.1, Interval(-1.3, 1.3), 1)
   x11 = MC{1,NS}(0.9, Interval(-0.8, 1.1), 1)
   x12 =  MC{1,NS}(0.3, Interval(-0.8, 1.3), 1)

   y1 = swish1(x1)
   y2 = swish1(x2)
   y3 = swish1(x3)

   y4 = swish1(x4)
   y5 = swish1(x5)
   y6 = swish1(x6)

   y7 = swish1(x7)
   y8 = swish1(x8)
   y9 = swish1(x9)

   y10 = swish1(x10)
   y11 = swish1(x11)
   y12 = swish1(x12)

   @test isapprox(swish1(7), 6.993622641639196, atol=1E-8)
   @test isapprox(swish1(-0.75), -0.24061597561845527, atol=1E-8)

   @test isapprox(y1.cv, 0.052497918747894, atol=1E-8)
   @test isapprox(y1.cc, 0.5263617142904655, atol=1E-8)
   @test isapprox(y2.cv, -0.1887703343990727, atol=1E-8)
   @test isapprox(y2.cc, 0.22636171429046548, atol=1E-8)
   @test isapprox(y3.cv, 0.46773144051771626, atol=1E-8)
   @test isapprox(y3.cc, 0.49425607533883487, atol=1E-8)

   @test isapprox(y4.cv, -0.013265798607177848, atol=1E-8)
   @test isapprox(y4.cc, -0.009757684668790446, atol=1E-8)
   @test isapprox(y5.cv, 6.486734201392823, atol=1E-8)
   @test isapprox(y5.cc, 6.490242315331209, atol=1E-8)
   @test isapprox(y6.cv, 0.46773144051771626, atol=1E-8)
   @test isapprox(y6.cc, 0.49425607533883487, atol=1E-8)

   @test isapprox(y7.cv, 2.961190227697252, atol=1E-8)
   @test isapprox(y7.cc, 3.1974071923702523, atol=1E-8)
   @test isapprox(y8.cv, 0.8252861161546294, atol=1E-8)
   @test isapprox(y8.cc, 1.5174068718163187, atol=1E-8)
   @test isapprox(y9.cv, 0.1723327550434977, atol=1E-8)
   @test isapprox(y9.cc, 1.2820250620927323, atol=1E-8)

   @test isapprox(y10.cv, -0.2747138838453706, atol=1E-8)
   @test isapprox(y10.cc, -0.17841452204467384, atol=1E-8)
   @test isapprox(y11.cv, 0.6398545523625035, atol=1E-8)
   @test isapprox(y11.cc, 0.7123064812859411, atol=1E-8)
   @test isapprox(y12.cv, 0.1723327550434977, atol=1E-8)
   @test isapprox(y12.cc, 0.4170112431680709, atol=1E-8)

   @test isapprox(McCormick.swish1_env(1.0, 2.0, 3.0), -0.1028650654542731, atol=1E-8)

   x4 = MC{1,NS}(-6.5, Interval(-7.5, -5.5), 1)
   x5 = MC{1,NS}(6.5, Interval(5.5, 7.5), 1)
   x6 = MC{1,NS}(0.7, Interval(0.5, 1.5), 1)

   x7 = MC{1,NS}(3.1, Interval(-3.5, 3.5), 1)
   x8 = MC{1,NS}(1.1, Interval(-2.0, 2.5), 1)
   x9 =  MC{1,NS}(0.3, Interval(-2.0, 3.5), 1)

   x10 = MC{1,NS}(-1.1, Interval(-1.3, 1.3), 1)
   x11 = MC{1,NS}(0.9, Interval(-0.8, 1.1), 1)
   x12 =  MC{1,NS}(0.3, Interval(-0.8, 1.3), 1)

   y1 = xexpax(x1,-0.5)
   y2 = xexpax(x2,-0.5)
   y3 = xexpax(x3,-0.5)

   y4 = xexpax(x4,-0.5)
   y5 = xexpax(x5,-0.5)
   y6 = xexpax(x6,-0.5)

   y7 = xexpax(x7,-0.5)
   y8 = xexpax(x8,-0.5)
   y9 = xexpax(x9,-0.5)

   y10 = xexpax(x10,-0.5)
   y11 = xexpax(x11,-0.5)
   y12 = xexpax(x12,-0.5)

   @test isapprox(y1.cv, -1.1040067694360605, atol=1E-8)
   @test isapprox(y1.cc, 0.09512294245007141, atol=1E-8)

   @test isapprox(y2.cv, -1.8808167402421674, atol=1E-8)
   @test isapprox(y2.cc, -0.6420127083438707, atol=1E-8)

   @test isapprox(y3.cv, 0.4532302790508663, atol=1E-8)
   @test isapprox(y3.cc, 0.4932816628030994, atol=1E-8)

   @test isapprox(y4.cv, -202.4712951817529, atol=1E-8)
   @test isapprox(y4.cc, -167.6372094617549, atol=1E-8)

   @test isapprox(y5.cv, 0.2520323509061931, atol=1E-8)
   @test isapprox(y5.cc, 0.26399316527848, atol=1E-8)

   @test isapprox(y6.cv, 0.4532302790508663, atol=1E-8)
   @test isapprox(y6.cc, 0.4932816628030994, atol=1E-8)

   @test isapprox(y7.cv, -0.5774665218146768, atol=1E-8)
   @test isapprox(y7.cc, 0.6579687188629034, atol=1E-8)

   @test isapprox(y8.cv, -1.1979504320041896, atol=1E-8)
   @test isapprox(y8.cc, 0.6346447914185354, atol=1E-8)

   @test isapprox(y9.cv, -2.9087497195203285, atol=1E-8)
   @test isapprox(y9.cc, 0.25821239292751735, atol=1E-8)

   @test isapprox(y10.cv, -2.2464444171405744, atol=1E-8)
   @test isapprox(y10.cc, -1.906578319654135, atol=1E-8)

   @test isapprox(y11.cv, 0.44221273357310886, atol=1E-8)
   @test isapprox(y11.cc, 0.5738653364595959, atol=1E-8)

   @test isapprox(y12.cv, -0.21282585587845876, atol=1E-8)
   @test isapprox(y12.cc, 0.25821239292751735, atol=1E-8)

   y1 = xexpax(x1,0.5)
   y2 = xexpax(x2,0.5)
   y3 = xexpax(x3,0.5)

   y4 = xexpax(x4,0.5)
   y5 = xexpax(x5,0.5)
   y6 = xexpax(x6,0.5)

   y7 = xexpax(x7,0.5)
   y8 = xexpax(x8,0.5)
   y9 = xexpax(x9,0.5)

   y10 = xexpax(x10,0.5)
   y11 = xexpax(x11,0.5)
   y12 = xexpax(x12,0.5)

   @test isapprox(y1.cv, 0.10512710963760241, atol=1E-8)
   @test isapprox(y1.cc, 1.3629434263714295, atol=1E-8)

   @test isapprox(y2.cv, -0.38940039153570244, atol=1E-8)
   @test isapprox(y2.cc, 0.5861334555653227, atol=1E-8)

   @test isapprox(y3.cv, 0.9933472840152799, atol=1E-8)
   @test isapprox(y3.cc, 1.148710171658899, atol=1E-8)

   @test isapprox(y4.cv, -0.26399316527848, atol=1E-8)
   @test isapprox(y4.cc, -0.2520323509061931, atol=1E-8)

   @test isapprox(y5.cv, 167.6372094617549, atol=1E-8)
   @test isapprox(y5.cc, 202.4712951817529, atol=1E-8)

   @test isapprox(y6.cv, 0.9933472840152799, atol=1E-8)
   @test isapprox(y6.cc, 1.148710171658899, atol=1E-8)

   @test isapprox(y7.cv, 14.605557566031301, atol=1E-8)
   @test isapprox(y7.cc, 18.955434042128815, atol=1E-8)

   @test isapprox(y8.cv, 1.906578319654135, atol=1E-8)
   @test isapprox(y8.cc, 5.782243441122052, atol=1E-8)

   @test isapprox(y9.cv, 0.3485502728184849, atol=1E-8)
   @test isapprox(y9.cc, 7.994567839699798, atol=1E-8)

   @test isapprox(y10.cv, -0.6346447914185354, atol=1E-8)
   @test isapprox(y10.cc, -0.43490084921182987, atol=1E-8)

   @test isapprox(y11.cv, 1.4114809669411519, atol=1E-8)
   @test isapprox(y11.cc, 1.6494378610770144, atol=1E-8)

   @test isapprox(y12.cv, 0.3485502728184849, atol=1E-8)
   @test isapprox(y12.cc, 1.0490320707911238, atol=1E-8)
end

@testset "Test bound enforcing functions" begin

   x = MC{2,NS}(-1.5, 1.5, Interval(-5.5, 7.5), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   y1 = positive(x)
   @test y1.cv == McCormick.MC_DOMAIN_TOL
   @test positive(1.0) == 1.0

   y2 = negative(x)
   @test y2.cc == -McCormick.MC_DOMAIN_TOL
   @test negative(1.0) == 1.0

   y3 = lower_bnd(x, -1.0)
   @test y3.cv == -1.0
   @test lower_bnd(1.0, -1.0) == 1.0

   y4 = upper_bnd(x, 1.0)
   @test y4.cc == 1.0
   @test upper_bnd(1.0, 1.0) == 1.0

   y5 = bnd(x, -1.0, 1.0)
   @test y5.cv == -1.0
   @test y5.cc == 1.0
   @test bnd(1.0, -1.0, 1.0) == 1.0

   gout = zeros(2)
   McCormick.d_lower_bnd_grad(gout, 0.0, 0.0)
   @test gout[1] == 1.0
   @test gout[2] == 0.0

   McCormick.d_upper_bnd_grad(gout, 0.0, 0.0)
   @test gout[1] == 1.0
   @test gout[2] == 0.0

   gout = zeros(3)
   McCormick.d_bnd_grad(gout, 0.0, 0.0, 0.0)
   @test gout[1] == 1.0
   @test gout[2] == 0.0
   @test gout[3] == 0.0
end

@testset "Comparisons" begin

   x = MC{2,NS}(4.0, 4.0, Interval{Float64}(3.0,7.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   y = MC{2,NS}(-4.0, -4.0, Interval{Float64}(-7.0,-3.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)
   z = MC{2,NS}(1.0, 4.0, Interval{Float64}(0.0,5.0), seed_gradient(1,Val(2)), seed_gradient(1,Val(2)), false)

   @test x >= y
   @test !(x <= z)
   @test y <= z
   @test x == x
   @test !(x == z)
   @test !(x != x)
   @test x != z
   @test x > y
   @test !(x < z)
   @test y < z
end

@testset "Trilinear" begin

   # improves and likely valid
   x = MC{3,NS}(2.0, Interval{Float64}(1.0, 3.0), 1)
   y = MC{3,NS}(2.4, Interval{Float64}(0.4, 3.0), 2)
   z = MC{3,NS}(1.4, Interval{Float64}(1.0, 3.5), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, 4.68, atol = 1E-3)
   @test isapprox(q.cc, 9.0, atol = 1E-3)

   x = MC{3,NS}(2.4, Interval{Float64}(0.4, 3.0), 2)
   y = MC{3,NS}(2.0, Interval{Float64}(1.0, 3.0), 1)
   z = MC{3,NS}(1.4, Interval{Float64}(1.0, 3.5), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, 4.68, atol = 1E-3)
   @test isapprox(q.cc, 9.0, atol = 1E-3)

   x = MC{3,NS}(1.4, Interval{Float64}(1.0, 3.5), 3)
   y = MC{3,NS}(2.0, Interval{Float64}(1.0, 3.0), 1)
   z = MC{3,NS}(2.4, Interval{Float64}(0.4, 3.0), 2)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, 4.68, atol = 1E-3)
   @test isapprox(q.cc, 9.0, atol = 1E-3)

   x = MC{3,NS}(1.4, Interval{Float64}(1.0, 3.5), 3)
   y = MC{3,NS}(2.4, Interval{Float64}(0.4, 3.0), 2)
   z = MC{3,NS}(2.0, Interval{Float64}(1.0, 3.0), 1)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, 4.68, atol = 1E-3)
   @test isapprox(q.cc, 9.0, atol = 1E-3)

   x = MC{3,NS}(2.4, Interval{Float64}(0.4, 3.0), 2)
   y = MC{3,NS}(1.4, Interval{Float64}(1.0, 3.5), 3)
   z = MC{3,NS}(2.0, Interval{Float64}(1.0, 3.0), 1)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, 4.68, atol = 1E-3)
   @test isapprox(q.cc, 9.0, atol = 1E-3)

   x = MC{3,NS}(2.0, Interval{Float64}(1.0, 3.0), 1)
   y = MC{3,NS}(1.4, Interval{Float64}(1.0, 3.5), 3)
   z = MC{3,NS}(2.4, Interval{Float64}(0.4, 3.0), 2)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, 4.68, atol = 1E-3)
   @test isapprox(q.cc, 9.0, atol = 1E-3)

   # improves and likely valid
   x = MC{3,NS}(2.0, Interval{Float64}(1.0, 3.0), 1)
   y = MC{3,NS}(2.4, Interval{Float64}(0.4, 3.0), 2)
   z = MC{3,NS}(1.4, Interval{Float64}(-1.0, 3.5), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, -0.6066666666666656, atol = 1E-3)
   @test isapprox(q.cc, 15.3, atol = 1E-3)

   x = MC{3,NS}(1.4, Interval{Float64}(-1.0, 3.5), 3)
   y = MC{3,NS}(2.0, Interval{Float64}(1.0, 3.0), 1)
   z = MC{3,NS}(2.4, Interval{Float64}(0.4, 3.0), 2)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, -0.6066666666666656, atol = 1E-3)
   @test isapprox(q.cc, 15.3, atol = 1E-3)

   x = MC{3,NS}(2.4, Interval{Float64}(0.4, 3.0), 2)
   y = MC{3,NS}(1.4, Interval{Float64}(-1.0, 3.5), 3)
   z = MC{3,NS}(2.0, Interval{Float64}(1.0, 3.0), 1)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, -0.6066666666666656, atol = 1E-3)
   @test isapprox(q.cc, 15.3, atol = 1E-3)

   x = MC{3,NS}(2.4, Interval{Float64}(0.4, 3.0), 2)
   y = MC{3,NS}(2.0, Interval{Float64}(1.0, 3.0), 1)
   z = MC{3,NS}(1.4, Interval{Float64}(-1.0, 3.5), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, -0.6066666666666656, atol = 1E-3)
   @test isapprox(q.cc, 15.3, atol = 1E-3)

   x = MC{3,NS}(1.4, Interval{Float64}(-1.0, 3.5), 3)
   y = MC{3,NS}(2.4, Interval{Float64}(0.4, 3.0), 2)
   z = MC{3,NS}(2.0, Interval{Float64}(1.0, 3.0), 1)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, -0.6066666666666656, atol = 1E-3)
   @test isapprox(q.cc, 15.3, atol = 1E-3)

   x = MC{3,NS}(2.0, Interval{Float64}(1.0, 3.0), 1)
   y = MC{3,NS}(1.4, Interval{Float64}(-1.0, 3.5), 3)
   z = MC{3,NS}(2.4, Interval{Float64}(0.4, 3.0), 2)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, -0.6066666666666656, atol = 1E-3)
   @test isapprox(q.cc, 15.3, atol = 1E-3)

   # improves and likely valid
   x = MC{3,NS}(1.5, Interval{Float64}(1.0, 2.0), 1)
   y = MC{3,NS}(1.4, Interval{Float64}(-1.0, 2.0), 2)
   z = MC{3,NS}(-0.5, Interval{Float64}(-1.0, 2.5), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv,-3.1500000000000004, atol = 1E-3)
   @test isapprox(q.cc, 0.20000000000000018, atol = 1E-3)
   println(" ")
   @show "Case 3"
   @show q
   @show qz

   # improves and likely valid
   x = MC{3,NS}(1.5, Interval{Float64}(1.0, 2.0), 1)
   y = MC{3,NS}(1.4, Interval{Float64}(-1.0, 2.0), 2)
   z = MC{3,NS}(-0.5, Interval{Float64}(-3.0, 1.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   #@test isapprox(q.cv, 1.8666666666666671, atol = 1E-3)
   @test isapprox(q.cc, 1.9833333333333334, atol = 1E-3)
   println(" ")
   @show "Case 4 (FAILS, CV)"
   @show q
   @show qz

   # improves and likely valid
   x = MC{3,NS}(-2.0, Interval{Float64}(-3.0, 2.0), 1)
   y = MC{3,NS}(0.5, Interval{Float64}(-1.0, 1.0), 2)
   z = MC{3,NS}(3.0, Interval{Float64}(-2.0, 4.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, -4.65, atol = 1E-3)
   @test isapprox(q.cc, 1.0, atol = 1E-3)
   println(" ")
   @show "Case 5"
   @show q
   @show qz

   # improves and likely valid
   x = MC{3,NS}(-2.0, Interval{Float64}(-3.0,2.0), 1)
   y = MC{3,NS}(-2.0, Interval{Float64}(-4.0,1.0), 2)
   z = MC{3,NS}(-1.0, Interval{Float64}(-2.0,1.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, -15.0, atol = 1E-3)
   @test isapprox(q.cc, 0.0, atol = 1E-3)
   println(" ")
   @show "Case 6"
   @show q
   @show qz

   # improves and likely valid
   x = MC{3,NS}(-2.0, Interval{Float64}(-3.0, 2.0), 1)
   y = MC{3,NS}(2.0, Interval{Float64}(-1.0, 3.0), 2)
   z = MC{3,NS}(2.0, Interval{Float64}(-1.0, 4.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   #@test isapprox(q.cv, -2.25, atol = 1E-3)
   #@test isapprox(q.cc, 6.0, atol = 1E-3)
   println(" ")
   @show "Case 7 (FAILS, cv)"
   @show q
   @show qz

   # improves and likely valid
   x = MC{3,NS}(1.5, Interval{Float64}(1.0,2.0), 1)
   y = MC{3,NS}(2.0, Interval{Float64}(1.0,3.0), 2)
   z = MC{3,NS}(-2.0, Interval{Float64}(-3.0,-1.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, -9.5, atol = 1E-3)
   @test isapprox(q.cc, -4.5, atol = 1E-3)
   println(" ")
   @show "Case 8"
   @show q
   @show qz

   x = MC{3,NS}(1.5, Interval{Float64}(1.0,4.0), 1)
   y = MC{3,NS}(2.0, Interval{Float64}(1.0,3.0), 2)
   z = MC{3,NS}(-2.0, Interval{Float64}(-3.0, -1.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz =(x*y)*z
   @test isapprox(q.cv, -9.5, atol = 1E-3)
   @test isapprox(q.cc, -4.1666666, atol = 1E-3)
   println(" ")
   @show "Case 9"
   @show q
   @show qz

   x = MC{3,NS}(4.0, Interval{Float64}(2.0,5.0), 1)
   y = MC{3,NS}(-2.0, Interval{Float64}(-3.0,3.0), 2)
   z = MC{3,NS}(-4.0, Interval{Float64}(-7.0,-3.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, 13.0, atol = 1E-3)
   @test isapprox(q.cc, 45.0, atol = 1E-3)
   println(" ")
   @show "Case 10"
   @show q
   @show qz

   # improves and likely valid
   x = MC{3,NS}(2.0, Interval{Float64}(-1.0, 5.0), 1)
   y = MC{3,NS}(-2.0, Interval{Float64}(-4.0, 1.0), 2)
   z = MC{3,NS}(-2.0, Interval{Float64}(-3.0, -1.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   #@test isapprox(q7a.cv, -1.0, atol = 1E-3)
   #@test isapprox(q7a.cc, 8.0, atol = 1E-3)
   qz = (x*y)*z
   println(" ")
   @show "Case 11 (FAILS, CV)"
   @show q
   @show qz

   # improves and likely valid (GOOD)
   x = MC{3,NS}(-2.0, Interval{Float64}(-3.0,2.0), 1)
   y = MC{3,NS}(-0.5, Interval{Float64}(-1.0,1.0), 2)
   z = MC{3,NS}(-4.0, Interval{Float64}(-8.0,-1.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
#   @test isapprox(q7b.cv, -11.5, atol = 1E-3)
#   @test isapprox(q7b.cc, 7.000000000000001, atol = 1E-3)
   qz = (x*y)*z
   println(" ")
   @show "Case 12"
   @show q
   @show qz

   # improves and likely valid (GOOD)
   x = MC{3,NS}(1.5, Interval{Float64}(1.0, 2.0), 1)
   y = MC{3,NS}(-3.4, Interval{Float64}(-4.0, -2.0), 2)
   z = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -1.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 9.799999999999997, atol = 1E-3)
   @test isapprox(q.cc, 15.0, atol = 1E-3)

   x = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -1.0), 3)
   y = MC{3,NS}(1.5, Interval{Float64}(1.0, 2.0), 1)
   z = MC{3,NS}(-3.4, Interval{Float64}(-4.0, -2.0), 2)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, 10.8, atol = 1E-3)
   @test isapprox(q.cc, 15.0, atol = 1E-3)
   @show "Case 13a (modified)"

   x = MC{3,NS}(-3.4, Interval{Float64}(-4.0, -2.0), 2)
   y = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -1.0), 3)
   z = MC{3,NS}(1.5, Interval{Float64}(1.0, 2.0), 1)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 9.799999999999997, atol = 1E-3)
   @test isapprox(q.cc, 15.0, atol = 1E-3)

   x = MC{3,NS}(-3.4, Interval{Float64}(-4.0, -2.0), 2)
   y = MC{3,NS}(1.5, Interval{Float64}(1.0, 2.0), 1)
   z = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -1.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 9.799999999999997, atol = 1E-3)
   @test isapprox(q.cc, 15.0, atol = 1E-3)

   x = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -1.0), 3)
   y = MC{3,NS}(-3.4, Interval{Float64}(-4.0, -2.0), 2)
   z = MC{3,NS}(1.5, Interval{Float64}(1.0, 2.0), 1)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, 10.8, atol = 1E-3)
   @test isapprox(q.cc, 15.0, atol = 1E-3)
   @show "Case 13d (modified)"

   x = MC{3,NS}(1.5, Interval{Float64}(1.0, 2.0), 1)
   y = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -1.0), 3)
   z = MC{3,NS}(-3.4, Interval{Float64}(-4.0, -2.0), 2)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   qz = (x*y)*z
   @test isapprox(q.cv, 10.8, atol = 1E-3)
   @test isapprox(q.cc, 15.0, atol = 1E-3)
   @show "Case 13e (modified)"

   # improves and likely valid (GOOD)
   x = MC{3,NS}(1.5, Interval{Float64}(1.0, 5.0), 1)
   y = MC{3,NS}(-3.4, Interval{Float64}(-4.0, -2.0), 2)
   z = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -1.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 9.299999999999997, atol = 1E-3)
   @test isapprox(q.cc, 15.0, atol = 1E-3)

   x = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -1.0), 3)
   y = MC{3,NS}(1.5, Interval{Float64}(1.0, 5.0), 1)
   z = MC{3,NS}(-3.4, Interval{Float64}(-4.0, -2.0), 2)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 9.299999999999997, atol = 1E-3)
   @test isapprox(q.cc, 15.0, atol = 1E-3)

   x = MC{3,NS}(-3.4, Interval{Float64}(-4.0, -2.0), 2)
   y = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -1.0), 3)
   z = MC{3,NS}(1.5, Interval{Float64}(1.0, 5.0), 1)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 9.299999999999997, atol = 1E-3)
   @test isapprox(q.cc, 15.0, atol = 1E-3)

   x = MC{3,NS}(-3.4, Interval{Float64}(-4.0, -2.0), 2)
   y = MC{3,NS}(1.5, Interval{Float64}(1.0, 5.0), 1)
   z = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -1.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 9.299999999999997, atol = 1E-3)
   @test isapprox(q.cc, 15.0, atol = 1E-3)

   x = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -1.0), 3)
   y = MC{3,NS}(-3.4, Interval{Float64}(-4.0, -2.0), 2)
   z = MC{3,NS}(1.5, Interval{Float64}(1.0, 5.0), 1)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 9.299999999999997, atol = 1E-3)
   @test isapprox(q.cc, 15.0, atol = 1E-3)

   x = MC{3,NS}(1.5, Interval{Float64}(1.0, 5.0), 1)
   y = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -1.0), 3)
   z = MC{3,NS}(-3.4, Interval{Float64}(-4.0, -2.0), 2)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 9.299999999999997, atol = 1E-3)
   @test isapprox(q.cc, 15.0, atol = 1E-3)

   # improves and likely valid
   x = MC{3,NS}(2.0, Interval{Float64}(-1.0, 3.0), 1)
   y = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -0.4), 2)
   z = MC{3,NS}(-1.4, Interval{Float64}(-3.5, -1.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 3.0, atol = 1E-3)
   @test isapprox(q.cc, 10.2, atol = 1E-3)

   x = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -0.4), 2)
   y = MC{3,NS}(2.0, Interval{Float64}(-1.0, 3.0), 1)
   z = MC{3,NS}(-1.4, Interval{Float64}(-3.5, -1.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 3.0, atol = 1E-3)
   @test isapprox(q.cc, 10.2, atol = 1E-3)

   x = MC{3,NS}(-1.4, Interval{Float64}(-3.5, -1.0), 3)
   y = MC{3,NS}(2.0, Interval{Float64}(-1.0, 3.0), 1)
   z = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -0.4), 2)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 3.0, atol = 1E-3)
   @test isapprox(q.cc, 10.2, atol = 1E-3)

   x = MC{3,NS}(-1.4, Interval{Float64}(-3.5, -1.0), 3)
   y = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -0.4), 2)
   z = MC{3,NS}(2.0, Interval{Float64}(-1.0, 3.0), 1)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 3.0, atol = 1E-3)
   @test isapprox(q.cc, 10.2, atol = 1E-3)

   x = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -0.4), 2)
   y = MC{3,NS}(-1.4, Interval{Float64}(-3.5, -1.0), 3)
   z = MC{3,NS}(2.0, Interval{Float64}(-1.0, 3.0), 1)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 3.0, atol = 1E-3)
   @test isapprox(q.cc, 10.2, atol = 1E-3)

   x = MC{3,NS}(2.0, Interval{Float64}(-1.0, 3.0), 1)
   y = MC{3,NS}(-1.4, Interval{Float64}(-3.5, -1.0), 3)
   z = MC{3,NS}(-2.4, Interval{Float64}(-3.0, -0.4), 2)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, 3.0, atol = 1E-3)
   @test isapprox(q.cc, 10.2, atol = 1E-3)

   # improves and likely valid
   x = MC{3,NS}(-4.0, Interval{Float64}(-8.0,-3.0), 1)
   y = MC{3,NS}(-4.0, Interval{Float64}(-7.0,-3.0), 2)
   z = MC{3,NS}(-4.0, Interval{Float64}(-7.0,-2.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, -106.0, atol = 1E-3)
   @test isapprox(q.cc, -48.0, atol = 1E-3)

   x = MC{3,NS}(-4.0, Interval{Float64}(-7.0,-2.0), 3)
   y = MC{3,NS}(-4.0, Interval{Float64}(-8.0,-3.0), 1)
   z = MC{3,NS}(-4.0, Interval{Float64}(-7.0,-3.0), 2)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, -106.0, atol = 1E-3)
   @test isapprox(q.cc, -48.0, atol = 1E-3)

   x = MC{3,NS}(-4.0, Interval{Float64}(-7.0,-3.0), 2)
   y = MC{3,NS}(-4.0, Interval{Float64}(-7.0,-2.0), 3)
   z = MC{3,NS}(-4.0, Interval{Float64}(-8.0,-3.0), 1)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, -106.0, atol = 1E-3)
   @test isapprox(q.cc, -50.0, atol = 1E-3)

   x = MC{3,NS}(-4.0, Interval{Float64}(-7.0,-3.0), 2)
   y = MC{3,NS}(-4.0, Interval{Float64}(-8.0,-3.0), 1)
   z = MC{3,NS}(-4.0, Interval{Float64}(-7.0,-2.0), 3)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, -106.0, atol = 1E-3)
   @test isapprox(q.cc, -48.0, atol = 1E-3)

   x = MC{3,NS}(-4.0, Interval{Float64}(-7.0,-2.0), 3)
   y = MC{3,NS}(-4.0, Interval{Float64}(-7.0,-3.0), 2)
   z = MC{3,NS}(-4.0, Interval{Float64}(-8.0,-3.0), 1)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, -106.0, atol = 1E-3)
   @test isapprox(q.cc, -48.0, atol = 1E-3)

   x = MC{3,NS}(-4.0, Interval{Float64}(-8.0,-3.0), 1)
   y = MC{3,NS}(-4.0, Interval{Float64}(-7.0,-2.0), 3)
   z = MC{3,NS}(-4.0, Interval{Float64}(-7.0,-3.0), 2)
   q = mult_kernel(x, y, z, x.Intv*y.Intv*z.Intv)
   @test isapprox(q.cv, -106.0, atol = 1E-3)
   @test isapprox(q.cc, -50.0, atol = 1E-3)
end
