@testset "Implicit" begin
   function h!(out::A,x::B,p::C) where {A,B,C}
       out[1] = x[1]^2 + x[1]*p[1] + 4.0
       return
   end
   function hj!(out::A,x::B,p::C) where {A,B,C}
       out[1,1] = 2.0*x[1]+p[1]
       return
   end
   nx = 1
   np = 1
   P = [Interval{Float64}(6.0,9.0)]
   X = [Interval{Float64}(-0.78,-0.4)]
   pmid = mid.(P)
   pref_mc = [MC{1,NS}(pmid[1],P[1],1)]
   p_mc = [MC{1,NS}(pmid[1],P[1],1)]

   mccall! = MCCallback(h!, hj!, nx, np)
   mccall!.P[:] = P
   mccall!.X[:] = X
   mccall!.p_mc[1] = MC{1,NS}(pmid[1],P[1],1)
   mccall!.pref_mc[1] = MC{1,NS}(pmid[1],P[1],1)

   gen_expansion_params!(mccall!)
   implicit_relax_h!(mccall!)

   @test isapprox(mccall!.x_mc[1].cv, -0.6307841, atol = 1E-5)
   @test isapprox(mccall!.x_mc[1].cc, -0.5209112, atol = 1E-5)
   @test isapprox(mccall!.x_mc[1].Intv.lo, -0.7720045045045048, atol = 1E-5)
   @test isapprox(mccall!.x_mc[1].cv_grad[1], 0.09414689079323225, atol = 1E-5)
   @test isapprox(mccall!.x_mc[1].cc_grad[1], 0.0455312, atol = 1E-5)

   dense = McCormick.DenseMidInv(nx, np)
   @test dense.Y == zeros(Float64,nx,nx)
   @test dense.YInterval == zeros(Interval{Float64},1)
   @test dense.nx == nx
   @test dense.np == np
end
