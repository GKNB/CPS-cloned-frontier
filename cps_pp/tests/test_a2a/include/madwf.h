#pragma once

CPS_START_NAMESPACE

#ifdef USE_GRID

template<typename GridA2Apolicies>
void testComputeLowModeMADWF(const A2AArg &a2a_args, const LancArg &lanc_arg,
			     typename GridA2Apolicies::FgridGFclass &lattice, const typename SIMDpolicyBase<4>::ParamType &simd_dims, 
			     const double tol){
  //If we use the same eigenvectors and the same Dirac operator we should get the same result
  //FIXME

  /* GridLanczosWrapper<GridA2Apolicies> evecs_rand; */
  /* evecs_rand.randomizeEvecs(lanc_arg, lattice); */
  /* EvecInterfaceGrid<GridA2Apolicies> eveci_rand(evecs_rand.evec, evecs_rand.eval); */

  /* A2AvectorV<GridA2Apolicies> V_orig(a2a_args,simd_dims); */
  /* A2AvectorW<GridA2Apolicies> W_orig(a2a_args,simd_dims); */

  /* A2AvectorV<GridA2Apolicies> V_test(a2a_args,simd_dims); */
  /* A2AvectorW<GridA2Apolicies> W_test(a2a_args,simd_dims); */

  /* int Ls = GJP.Snodes() * GJP.SnodeSites(); */
  /* double mob_b = lattice.get_mob_b(); */
  /* double mob_c = mob_b - 1.; */
  
  /* CGcontrols cg_con_orig; */

  /* CGcontrols cg_con_test; */
  /* cg_con_test.madwf_params.Ls_inner = Ls; */
  /* cg_con_test.madwf_params.b_plus_c_inner = mob_b + mob_c; */
  /* cg_con_test.madwf_params.use_ZMobius = false; */
  /* cg_con_test.madwf_params.precond = SchurOriginal; */
  
  /* computeVWlowStandard(V_orig, W_orig, lattice, eveci_rand, evecs_rand.mass, cg_con_orig); */
  /* computeVWlowMADWF(V_test, W_test, lattice, eveci_rand, evecs_rand.mass, cg_con_test); */

  /* int nl = a2a_args.nl; */
  /* for(int i=0;i<nl;i++){ */
  /*   if( ! V_orig.getVl(i).equals(  V_test.getVl(i), tol, true ) ){ std::cout << "FAIL" << std::endl; exit(1); } */
  /* } */
  /* if(!UniqueID()) printf("Passed Vl test\n"); */

  /* for(int i=0;i<nl;i++){ */
  /*   if( ! W_orig.getWl(i).equals(  W_test.getWl(i), tol, true ) ){ std::cout << "FAIL" << std::endl; exit(1); } */
  /* } */

}

#endif


#ifdef USE_GRID

//Test that the "MADWF" codepath is the same for the two different supported preconditionings
//Requires multiple hits
template<typename GridA2Apolicies>
void testMADWFprecon(const A2AArg &a2a_args, const LancArg &lanc_arg,
			     typename GridA2Apolicies::FgridGFclass &lattice, const typename SIMDpolicyBase<4>::ParamType &simd_dims, 
			     const double tol){

  //FIXME

  // if(!UniqueID()){ printf("Computing SchurOriginal evecs"); fflush(stdout); }
  // GridLanczosWrapper<GridA2Apolicies> evecs_orig;
  // evecs_orig.compute(lanc_arg, lattice, SchurOriginal);
  // evecs_orig.toSingle();

    
  // if(!UniqueID()){ printf("Computing SchurDiagTwo evecs"); fflush(stdout); }
  // GridLanczosWrapper<GridA2Apolicies> evecs_diagtwo;
  // evecs_diagtwo.compute(lanc_arg, lattice, SchurDiagTwo);
  // evecs_diagtwo.toSingle();

  // int Ls = GJP.Snodes() * GJP.SnodeSites();
  // double mob_b = lattice.get_mob_b();
  // double mob_c = mob_b - 1.;
  
  // CGcontrols cg_con_orig;  
  // cg_con_orig.CGalgorithm = AlgorithmMixedPrecisionMADWF;
  // cg_con_orig.CG_tolerance = 1e-8;
  // cg_con_orig.CG_max_iters = 10000;
  // cg_con_orig.mixedCG_init_inner_tolerance =1e-4;
  // cg_con_orig.madwf_params.Ls_inner = Ls;
  // cg_con_orig.madwf_params.b_plus_c_inner = mob_b + mob_c;
  // cg_con_orig.madwf_params.use_ZMobius = false;
  // cg_con_orig.madwf_params.precond = SchurOriginal;

  // CGcontrols cg_con_diagtwo(cg_con_orig);
  // cg_con_diagtwo.madwf_params.precond = SchurDiagTwo;


  // EvecInterfaceGridSinglePrec<GridA2Apolicies> eveci_orig(evecs_orig.evec_f, evecs_orig.eval, lattice, lanc_arg.mass, cg_con_orig);
  // EvecInterfaceGridSinglePrec<GridA2Apolicies> eveci_diagtwo(evecs_diagtwo.evec_f, evecs_diagtwo.eval, lattice, lanc_arg.mass, cg_con_diagtwo);

  // typename GridA2Apolicies::SourceFieldType::InputParamType simd_dims_3d;
  // setupFieldParams<typename GridA2Apolicies::SourceFieldType>(simd_dims_3d);

  // Grid::ComplexD sum_tr_orig(0), sum_tr_diagtwo(0);
  // fVector<Grid::ComplexD> sum_pion2pt_orig_srcavg, sum_pion2pt_diagtwo_srcavg;

  // int hits = 100;
  // for(int h=0;h<hits;h++){
  //   A2AvectorV<GridA2Apolicies> V_orig(a2a_args,simd_dims);
  //   A2AvectorW<GridA2Apolicies> W_orig(a2a_args,simd_dims);
  //   computeVW(V_orig, W_orig, lattice, eveci_orig, evecs_orig.mass, cg_con_orig);

  //   A2AvectorV<GridA2Apolicies> V_diagtwo(a2a_args,simd_dims);
  //   A2AvectorW<GridA2Apolicies> W_diagtwo(a2a_args,simd_dims);
  //   computeVW(V_diagtwo, W_diagtwo, lattice, eveci_diagtwo, evecs_diagtwo.mass, cg_con_diagtwo);

  //   //This one doesn't seem to care much about the low modes
  //   {
  //     typedef typename mult_vv_field<GridA2Apolicies, A2AvectorV, A2AvectorW>::PropagatorField PropagatorField;
  //     PropagatorField prop_orig(simd_dims), prop_diagtwo(simd_dims);
  //     mult(prop_orig, V_orig, W_orig, false, true);
  //     mult(prop_diagtwo, V_diagtwo, W_diagtwo, false, true);
      
  //     Grid::ComplexD tr_orig = globalSumReduce(Trace(prop_orig));
  //     Grid::ComplexD tr_diagtwo = globalSumReduce(Trace(prop_diagtwo));
      
  //     sum_tr_orig = sum_tr_orig + tr_orig;
  //     sum_tr_diagtwo = sum_tr_diagtwo + tr_diagtwo;
      
  //     Grid::ComplexD avg_tr_orig = sum_tr_orig/double(h+1);
  //     Grid::ComplexD avg_tr_diagtwo = sum_tr_diagtwo/double(h+1);
  //     double reldiff_r = 2. * (avg_tr_orig.real() - avg_tr_diagtwo.real())/(avg_tr_orig.real() + avg_tr_diagtwo.real());
  //     double reldiff_i = 2. * (avg_tr_orig.imag() - avg_tr_diagtwo.imag())/(avg_tr_orig.imag() + avg_tr_diagtwo.imag());

      
  //     if(!UniqueID()){ printf("Hits %d Tr(VW^dag) Orig (%g,%g) DiagTwo (%g,%g) Rel.Diff (%g,%g)\n", h+1,
  // 			      avg_tr_orig.real(), avg_tr_orig.imag(),
  // 			      avg_tr_diagtwo.real(), avg_tr_diagtwo.imag(),
  // 			      reldiff_r, reldiff_i); fflush(stdout); }
  //   }
  //   {
  //     //Do a pion two point function comparison
  //     assert(GridA2Apolicies::GPARITY == 1);
  //     ThreeMomentum p_quark_plus(0,0,0), p_quark_minus(0,0,0);
  //     for(int i=0;i<3;i++)
  // 	if(GJP.Bc(i) == BND_CND_GPARITY){ //sum to +2
  // 	  p_quark_plus(i) = 3;
  // 	  p_quark_minus(i) = -1;
  // 	}
  //     RequiredMomentum quark_mom;
  //     quark_mom.addPandMinusP({p_quark_plus,p_quark_minus});
      
  //     MesonFieldMomentumContainer<GridA2Apolicies> mf_orig, mf_diagtwo;
  //     typedef computeGparityLLmesonFields1sSumOnTheFly<GridA2Apolicies, RequiredMomentum, 15, sigma2> mfCompute;
  //     mfCompute::computeMesonFields(mf_orig, quark_mom, W_orig, V_orig, 1., lattice, simd_dims_3d);
  //     mfCompute::computeMesonFields(mf_diagtwo, quark_mom, W_diagtwo, V_diagtwo, 1., lattice, simd_dims_3d);

  //     fMatrix<Grid::ComplexD> pion2pt_orig, pion2pt_diagtwo;
  //     ComputePion<GridA2Apolicies>::compute(pion2pt_orig, mf_orig, quark_mom, 0);
  //     ComputePion<GridA2Apolicies>::compute(pion2pt_diagtwo, mf_diagtwo, quark_mom, 0);

  //     fVector<Grid::ComplexD> pion2pt_orig_srcavg = rowAverage(pion2pt_orig);
  //     fVector<Grid::ComplexD> pion2pt_diagtwo_srcavg = rowAverage(pion2pt_diagtwo);

  //     if(h==0){
  // 	sum_pion2pt_orig_srcavg = pion2pt_orig_srcavg;
  // 	sum_pion2pt_diagtwo_srcavg = pion2pt_diagtwo_srcavg;
  //     }else{
  // 	sum_pion2pt_orig_srcavg += pion2pt_orig_srcavg;
  // 	sum_pion2pt_diagtwo_srcavg += pion2pt_diagtwo_srcavg;
  //     }

  //     fVector<Grid::ComplexD> avg_pion2pt_orig_srcavg = sum_pion2pt_orig_srcavg;  
  //     avg_pion2pt_orig_srcavg /= double(h+1);
  //     fVector<Grid::ComplexD> avg_pion2pt_diagtwo_srcavg = sum_pion2pt_diagtwo_srcavg; 
  //     avg_pion2pt_diagtwo_srcavg /= double(h+1);

  //     if(!UniqueID()){
  // 	for(int t=0;t<pion2pt_orig_srcavg.size();t++){
  // 	  double reldiff_r = 2. * (avg_pion2pt_orig_srcavg(t).real() - avg_pion2pt_diagtwo_srcavg(t).real())/(avg_pion2pt_orig_srcavg(t).real() + avg_pion2pt_diagtwo_srcavg(t).real());
  // 	  double reldiff_i = 2. * (avg_pion2pt_orig_srcavg(t).imag() - avg_pion2pt_diagtwo_srcavg(t).imag())/(avg_pion2pt_orig_srcavg(t).imag() + avg_pion2pt_diagtwo_srcavg(t).imag());
	  
  // 	  printf("Hits %d pion2pt[%d] Orig (%g,%g) DiagTwo (%g,%g) Rel.Diff (%g,%g)\n", h+1, t,
  // 		 avg_pion2pt_orig_srcavg(t).real(), avg_pion2pt_orig_srcavg(t).imag(),
  // 		 avg_pion2pt_diagtwo_srcavg(t).real(), avg_pion2pt_diagtwo_srcavg(t).imag(),
  // 		 reldiff_r, reldiff_i); 
  // 	  fflush(stdout);
  // 	}
  //     }
      
  //   }


  //}


}

#endif //USE_GRID


CPS_END_NAMESPACE
