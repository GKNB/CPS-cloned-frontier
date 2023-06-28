#pragma once

CPS_START_NAMESPACE

template<typename ResultsTypeA, typename ResultsTypeB>
bool compareKtoPiPi(const ResultsTypeA &result_A, const std::string &Adescr,
		    const ResultsTypeB &result_B, const std::string &Bdescr,
		    const std::string &descr, double tol){
  if(result_A.nElementsTotal() != result_B.nElementsTotal()){
    std::cout << descr << " fail: size mismatch " << result_A.nElementsTotal() << " " << result_B.nElementsTotal() << std::endl;
    return false;
  }
    
  bool fail = false;
  for(int i=0;i<result_A.nElementsTotal();i++){
    std::complex<double> val_A = convertComplexD(result_A[i]);
    std::complex<double> val_B = convertComplexD(result_B[i]);
    
    double rdiff = fabs(val_A.real()-val_B.real());
    double idiff = fabs(val_A.imag()-val_B.imag());
    if(rdiff > tol|| idiff > tol){
      printf("!!!Fail: %s elem %d %s (%g,%g) %s (%g,%g) Diff (%g,%g)\n", descr.c_str(), i,
	     Adescr.c_str(), val_A.real(),val_A.imag(),
	     Bdescr.c_str(), val_B.real(),val_B.imag(),
	     val_A.real()-val_B.real(), val_A.imag()-val_B.imag());
      fail = true;
    }//else printf("Pass: Type1 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
  }
  return !fail;
}


template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testKtoPiPiContractionGridStd(A2AvectorV<A2Apolicies_std> &V_std, A2AvectorW<A2Apolicies_std> &W_std,
				   A2AvectorV<A2Apolicies_grid> &V_grid, A2AvectorW<A2Apolicies_grid> &W_grid,
				   typename A2Apolicies_grid::FgridGFclass &lattice,
				   typename SIMDpolicyBase<3>::ParamType simd_dims_3d,
				   double tol){
  std::cout << "Starting testKtoPiPiContractionGridStd" << std::endl;

#if 0
  ThreeMomentum p_plus( GJP.Bc(0)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(1)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(2)==BND_CND_GPARITY? 1 : 0 );
  ThreeMomentum p_minus = -p_plus;

  ThreeMomentum p_pi_plus = p_plus * 2;
  
  StandardPionMomentaPolicy momenta;
  MesonFieldMomentumContainer<A2Apolicies_std> mf_ll_con_std;
  MesonFieldMomentumContainer<A2Apolicies_grid> mf_ll_con_grid;

  std::cout << "testKtoPiPiContractionGridStd computing LL meson fields standard calc" << std::endl;
  computeGparityLLmesonFields1s<A2Apolicies_std,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_std,momenta,W_std,V_std,2.0,lattice);

  std::cout << "testKtoPiPiContractionGridStd computing LL meson fields Grid calc" << std::endl;
  computeGparityLLmesonFields1s<A2Apolicies_grid,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_grid,momenta,W_grid,V_grid,2.0,lattice,simd_dims_3d);

  std::cout << "testKtoPiPiContractionGridStd comparing LL MF momenta" << std::endl;
  std::vector<ThreeMomentum> mom_std; mf_ll_con_std.getMomenta(mom_std);
  std::vector<ThreeMomentum> mom_grid; mf_ll_con_grid.getMomenta(mom_grid);

  assert(mom_std.size() == mom_grid.size());
  for(int pp=0;pp<mom_std.size();pp++)
    assert(mom_std[pp] == mom_grid[pp]);

  std::cout << "testKtoPiPiContractionGridStd comparison of LL MF momenta passed" << std::endl;

  std::cout << "testKtoPiPiContractionGridStd comparing LL meson fields between standard and Grid implementations" << std::endl;
  for(int pp=0;pp<mom_std.size();pp++){
    const auto &MFvec_std = mf_ll_con_std.get(mom_std[pp]);
    const auto &MFvec_grid = mf_ll_con_grid.get(mom_grid[pp]);
    std::cout << "Comparing meson fields for momentum " << mom_std[pp] << std::endl;
    assert(compare(MFvec_std, MFvec_grid, tol));
  }
  std::cout << "testKtoPiPiContractionGridStd comparison of LL meson fields between standard and Grid implementations passed" << std::endl;
 
  StandardLSWWmomentaPolicy ww_mom;

  std::cout << "testKtoPiPiContractionGridStd computing WW fields standard calc" << std::endl;
  std::vector<A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww_std;
  ComputeKtoPiPiGparity<A2Apolicies_std>::generatelsWWmesonfields(mf_ls_ww_std,W_std,W_std, ww_mom, 2.0,lattice);
  
  std::cout << "testKtoPiPiContractionGridStd computing WW fields Grid calc" << std::endl;
  std::vector<A2AmesonField<A2Apolicies_grid,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww_grid;
  ComputeKtoPiPiGparity<A2Apolicies_grid>::generatelsWWmesonfields(mf_ls_ww_grid,W_grid,W_grid, ww_mom, 2.0,lattice, simd_dims_3d);

  assert(mf_ls_ww_std.size() == mf_ls_ww_grid.size());
  
  std::cout << "testKtoPiPiContractionGridStd comparing WW fields" << std::endl;
  assert(compare(mf_ls_ww_std, mf_ls_ww_grid, tol));
  
  std::cout << "testKtoPiPiContractionGridStd comparison of WW fields passed" << std::endl;
  
  mf_ll_con_grid.printMomenta(std::cout);
#else

  const int nsimd = A2Apolicies_grid::ComplexType::Nsimd();      
 
  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  std::vector<A2AmesonField<A2Apolicies_grid,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww_grid(Lt);
  std::vector<A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww_std(Lt);
  
  for(int t=0;t<Lt;t++){
    mf_ls_ww_grid[t].setup(W_grid,W_grid,t,t);
    mf_ls_ww_grid[t].testRandom();

    mf_ls_ww_std[t].setup(W_std,W_std,t,t);
  }
  copy(mf_ls_ww_std, mf_ls_ww_grid);
  assert(compare(mf_ls_ww_std, mf_ls_ww_grid, 1e-12));
  
  ThreeMomentum p_pi_plus(2,2,2);
  ThreeMomentum p_pi_minus = -p_pi_plus;

  MesonFieldMomentumContainer<A2Apolicies_grid> mf_ll_con_grid;
  std::vector<A2AmesonField<A2Apolicies_grid,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_grid_tmp(Lt);

  MesonFieldMomentumContainer<A2Apolicies_std> mf_ll_con_std;
  std::vector<A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_std_tmp(Lt);

  for(int t=0;t<Lt;t++){
    mf_pion_grid_tmp[t].setup(W_grid,V_grid,t,t);
    mf_pion_grid_tmp[t].testRandom();

    mf_pion_std_tmp[t].setup(W_std,V_std,t,t);
  }
  copy(mf_pion_std_tmp, mf_pion_grid_tmp);
  
  mf_ll_con_grid.copyAdd(p_pi_plus, mf_pion_grid_tmp);
  mf_ll_con_std.copyAdd(p_pi_plus, mf_pion_std_tmp);
  
  for(int t=0;t<Lt;t++){
    mf_pion_grid_tmp[t].testRandom();
  }
  copy(mf_pion_std_tmp, mf_pion_grid_tmp);
  
  mf_ll_con_grid.copyAdd(p_pi_minus, mf_pion_grid_tmp);
  mf_ll_con_std.copyAdd(p_pi_minus, mf_pion_std_tmp);

  assert(mf_ll_con_grid.get(p_pi_plus).size() ==  Lt);
  assert(mf_ll_con_grid.get(p_pi_minus).size() ==  Lt);
  assert(mf_ll_con_std.get(p_pi_plus).size() ==  Lt);
  assert(mf_ll_con_std.get(p_pi_minus).size() ==  Lt);
    
  assert(compare(mf_ll_con_grid.get(p_pi_plus), mf_ll_con_std.get(p_pi_plus),1e-12));
  assert(compare(mf_ll_con_grid.get(p_pi_minus), mf_ll_con_std.get(p_pi_minus),1e-12));
#endif
 
  if(1){
    std::cout << "testKtoPiPiContractionGridStd computing type1 standard calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type1_std;
    //const int tsep_k_pi, const int tsep_pion, const int tstep, const int xyzStep,
    ComputeKtoPiPiGparity<A2Apolicies_std>::type1(type1_std, 4, 2, 1, 1, p_pi_plus, mf_ls_ww_std, mf_ll_con_std, V_std, V_std, W_std, W_std);

#ifdef GPU_VEC
    //Test the SIMD CPU version agrees with the non-SIMD CPU version first
    std::cout << "testKtoPiPiContractionGridStd computing type1 standard calc with SIMD" << std::endl;    
    std::vector<int> tsep_k_pi(1,4);
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type1_grid_cpu;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type1_omp(&type1_grid_cpu, tsep_k_pi, 2, 1, 1, p_pi_plus, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);

    if(!compareKtoPiPi(type1_std, "non-SIMD", type1_grid_cpu, "SIMD",  "type1", tol))
      ERR.General("","testKtoPiPiContractionGridStd","non-SIMD vs Grid SIMD implementation type1 test failed\n");
    std::cout << "testKtoPiPiContractionGridStd non-SIMD vs Grid SIMD type1 comparison passed" << std::endl;
#endif
       
    std::cout << "testKtoPiPiContractionGridStd computing type1 Grid calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type1_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type1(type1_grid, 4, 2, 1, 1, p_pi_plus, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);
  
    std::cout << "testKtoPiPiContractionGridStd comparing type1" << std::endl;
    if(!compareKtoPiPi(type1_std, "CPS", type1_grid, "Grid",  "type1", tol))
      ERR.General("","testKtoPiPiContractionGridStd","Standard vs Grid implementation type1 test failed\n");
    std::cout << "testKtoPiPiContractionGridStd type1 comparison passed" << std::endl;
  }
  if(1){
    std::cout << "testKtoPiPiContractionGridStd computing type2 standard calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type2_std;
    ComputeKtoPiPiGparity<A2Apolicies_std>::type2(type2_std, 4, 2, 1, p_pi_plus, mf_ls_ww_std, mf_ll_con_std, V_std, V_std, W_std, W_std);

#ifdef GPU_VEC
    //Test the SIMD CPU version agrees with the non-SIMD CPU version first
    std::cout << "testKtoPiPiContractionGridStd computing type2 standard calc with SIMD" << std::endl;    
    std::vector<int> tsep_k_pi(1,4);
    std::vector<ThreeMomentum> p(1, p_pi_plus);
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type2_grid_cpu;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type2_omp_v2(&type2_grid_cpu, tsep_k_pi, 2, 1, p, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);

    if(!compareKtoPiPi(type2_std, "non-SIMD", type2_grid_cpu, "SIMD",  "type2", tol))
      ERR.General("","testKtoPiPiContractionGridStd","non-SIMD vs Grid SIMD implementation type2 test failed\n");
    std::cout << "testKtoPiPiContractionGridStd non-SIMD vs Grid SIMD type2 comparison passed" << std::endl;
#endif
    
    std::cout << "testKtoPiPiContractionGridStd computing type2 Grid calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type2_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type2(type2_grid, 4, 2, 1, p_pi_plus, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);

    std::cout << "testKtoPiPiContractionGridStd comparing type2" << std::endl;
    if(!compareKtoPiPi(type2_std, "CPS", type2_grid, "Grid",  "type2", tol))
      ERR.General("","testKtoPiPiContractionGridStd","Standard vs Grid implementation type2 test failed\n");

    std::cout << "testKtoPiPiContractionGridStd type2 comparison passed" << std::endl;
  }
  if(1){
    std::cout << "testKtoPiPiContractionGridStd computing type3 standard calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type3_std;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::MixDiagResultsContainerType type3_mix_std;
    ComputeKtoPiPiGparity<A2Apolicies_std>::type3(type3_std, type3_mix_std, 4, 2, 1, p_pi_plus, mf_ls_ww_std, mf_ll_con_std, V_std, V_std, W_std, W_std);

    std::cout << "testKtoPiPiContractionGridStd computing type3 Grid calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type3_grid;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::MixDiagResultsContainerType type3_mix_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type3(type3_grid, type3_mix_grid, 4, 2, 1, p_pi_plus, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);

    std::cout << "testKtoPiPiContractionGridStd comparing type3" << std::endl;
    if(!compareKtoPiPi(type3_std, "CPS", type3_grid, "Grid",  "type3", tol))
      ERR.General("","testKtoPiPiContractionGridStd","Standard vs Grid implementation type3 test failed\n");
    
    std::cout << "testKtoPiPiContractionGridStd type3 comparison passed" << std::endl;
  }
  if(1){
    std::cout << "testKtoPiPiContractionGridStd computing type4 Grid calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type4_grid;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::MixDiagResultsContainerType type4_mix_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type4(type4_grid, type4_mix_grid, 1, mf_ls_ww_grid, V_grid, V_grid, W_grid, W_grid);

    std::cout << "testKtoPiPiContractionGridStd computing type4 standard calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type4_std;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::MixDiagResultsContainerType type4_mix_std;
    ComputeKtoPiPiGparity<A2Apolicies_std>::type4(type4_std, type4_mix_std, 1, mf_ls_ww_std, V_std, V_std, W_std, W_std);

    std::cout << "testKtoPiPiContractionGridStd comparing type4" << std::endl;
    if(!compareKtoPiPi(type4_std, "CPS", type4_grid, "Grid",  "type4", tol))
      ERR.General("","testKtoPiPiContractionGridStd","Standard vs Grid implementation type4 test failed\n");
    std::cout << "testKtoPiPiContractionGridStd type4 comparison passed" << std::endl;
  }
  std::cout << "Passed testKtoPiPiContractionGridStd" << std::endl;
}


template<typename mf_Policies>
class ComputeKtoPiPiGparityTest: public ComputeKtoPiPiGparity<mf_Policies>{
public:
  typedef typename ComputeKtoPiPiGparity<mf_Policies>::ResultsContainerType ResultsContainerType;
  typedef typename ComputeKtoPiPiGparity<mf_Policies>::SCFmat SCFmat;
  typedef typename ComputeKtoPiPiGparity<mf_Policies>::SCFmatrixField SCFmatrixField;

  static void type4_contract_test(ResultsContainerType &result, const int t_K, const int t_dis, const int thread_id, 
				  const SCFmat &part1, const SCFmat &part2_L, const SCFmat &part2_H){
    ComputeKtoPiPiGparity<mf_Policies>::type4_contract(result, t_K, t_dis, thread_id, part1, part2_L, part2_H);
  }
  static void type4_contract_test(ResultsContainerType &result, const int t_K, 
				  const SCFmatrixField &part1, const SCFmatrixField &part2_L, const SCFmatrixField &part2_H){
    ComputeKtoPiPiGparity<mf_Policies>::type4_contract(result, t_K, part1, part2_L, part2_H);
  }
};

template<typename GridA2Apolicies>
void testKtoPiPiType4FieldContraction(const double tol){
  std::cout << "Starting testKtoPiPiType4FieldContraction" << std::endl;
  typedef typename GridA2Apolicies::ComplexType ComplexType;
  typedef typename GridA2Apolicies::ScalarComplexType ScalarComplexType;
  typedef CPSspinColorFlavorMatrix<ComplexType> VectorMatrixType;
  typedef CPSmatrixField<VectorMatrixType> PropagatorField;

  static const int nsimd = GridA2Apolicies::ComplexType::Nsimd();
  typename PropagatorField::InputParamType simd_dims;
  PropagatorField::SIMDdefaultLayout(simd_dims,nsimd,2);

  PropagatorField part1(simd_dims), part2_L(simd_dims), part2_H(simd_dims);
  {
    CPSautoView(part1_v,part1,HostWrite);
    CPSautoView(part2_L_v,part2_L,HostWrite);
    CPSautoView(part2_H_v,part2_H,HostWrite);
  
    for(size_t x4d=0; x4d< part1.size(); x4d++){
      for(int s1=0;s1<4;s1++){
	for(int c1=0;c1<3;c1++){
	  for(int f1=0;f1<2;f1++){
	    for(int s2=0;s2<4;s2++){
	      for(int c2=0;c2<3;c2++){
		for(int f2=0;f2<2;f2++){
		  {
		    ComplexType &v = (*part1_v.site_ptr(x4d))(s1,s2)(c1,c2)(f1,f2);
		    for(int s=0;s<nsimd;s++) v.putlane( ScalarComplexType( LRG.Urand(FOUR_D), LRG.Urand(FOUR_D) ), s );
		  }		

		  {
		    ComplexType &v = (*part2_L_v.site_ptr(x4d))(s1,s2)(c1,c2)(f1,f2);
		    for(int s=0;s<nsimd;s++) v.putlane( ScalarComplexType( LRG.Urand(FOUR_D), LRG.Urand(FOUR_D) ), s );
		  }		

		  {
		    ComplexType &v = (*part2_H_v.site_ptr(x4d))(s1,s2)(c1,c2)(f1,f2);
		    for(int s=0;s<nsimd;s++) v.putlane( ScalarComplexType( LRG.Urand(FOUR_D), LRG.Urand(FOUR_D) ), s );
		  }		

		}
	      }
	    }
	  }
	}
      }
    }
  }
  
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType;

  static const int n_contract = 10; //ten type4 diagrams
  static const int con_off = 23; //index of first contraction in set
  int nthread = omp_get_max_threads();

  ResultsContainerType expect_r(n_contract, nthread);
  ResultsContainerType got_r(n_contract);

  int t_K = 1;
  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  {
    CPSautoView(part1_v,part1,HostRead);
    CPSautoView(part2_L_v,part2_L,HostRead);
    CPSautoView(part2_H_v,part2_H,HostRead);
   
    for(int t_loc=0;t_loc<GJP.TnodeSites();t_loc++){
      int t_glob = t_loc + GJP.TnodeSites()*GJP.TnodeCoor();
      int t_dis =  ComputeKtoPiPiGparityBase::modLt(t_glob - t_K, Lt);
    
      size_t vol3d = part1.size()/GJP.TnodeSites();
#pragma omp parallel for
      for(size_t x3d=0;x3d<vol3d;x3d++){
	int me = omp_get_thread_num();
	ComputeKtoPiPiGparityTest<GridA2Apolicies>::type4_contract_test(expect_r, t_K, t_dis, me,
									*part1_v.site_ptr(part1.threeToFour(x3d,t_loc)),
									*part2_L_v.site_ptr(part1.threeToFour(x3d,t_loc)),
									*part2_H_v.site_ptr(part1.threeToFour(x3d,t_loc)));
      }
    }
  }

  ComputeKtoPiPiGparityTest<GridA2Apolicies>::type4_contract_test(got_r, t_K, part1, part2_L, part2_H);
  
  got_r.nodeSum();
  expect_r.threadSum();
  expect_r.nodeSum();
  
  bool fail = false;
  for(int tdis=0;tdis<Lt;tdis++){
    for(int cidx=0; cidx<n_contract; cidx++){
      for(int gcombidx=0;gcombidx<8;gcombidx++){
	std::cout << "tdis " << tdis << " C" << cidx+con_off << " gcombidx " << gcombidx << std::endl;
	ComplexD expect = convertComplexD(expect_r(t_K,tdis,cidx,gcombidx));
	ComplexD got = convertComplexD(got_r(t_K,tdis,cidx,gcombidx));

	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: KtoPiPi type4 contract (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoPiPi type4 contract failed\n");
    
  std::cout << "Passed testKtoPiPiType4FieldContraction" << std::endl;
}




template<typename GridA2Apolicies>
void testKtoPiPiType4FieldFull(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting testKtoPiPiType4FieldFull" << std::endl;;

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorW<GridA2Apolicies> Wgrid(a2a_args, simd_dims), Whgrid(a2a_args, simd_dims);
  A2AvectorV<GridA2Apolicies> Vgrid(a2a_args, simd_dims), Vhgrid(a2a_args, simd_dims);

  Wgrid.testRandom();
  Vgrid.testRandom();

  Whgrid.testRandom();
  Vhgrid.testRandom();

  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::mf_WW mf_WW;
  std::vector<mf_WW> mf_kaon(Lt);
  for(int t=0;t<Lt;t++){
    mf_kaon[t].setup(Wgrid,Whgrid,t,t);
    mf_kaon[t].testRandom();
  }
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType;
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::MixDiagResultsContainerType MixDiagResultsContainerType;

  ResultsContainerType expect_r;
  ResultsContainerType got_r;
  MixDiagResultsContainerType expect_mix_r;
  MixDiagResultsContainerType got_mix_r;

  int tstep = 2;
  ComputeKtoPiPiGparity<GridA2Apolicies>::type4_omp(expect_r, expect_mix_r, tstep, mf_kaon, Vgrid, Vhgrid, Wgrid, Whgrid);
  ComputeKtoPiPiGparity<GridA2Apolicies>::type4_field_SIMD(got_r, got_mix_r, tstep, mf_kaon, Vgrid, Vhgrid, Wgrid, Whgrid);  

  static const int n_contract = 10; //ten type4 diagrams
  static const int con_off = 23; //index of first contraction in set
  
  bool fail = false;
  for(int t_K=0;t_K<Lt;t_K++){
    for(int tdis=0;tdis<Lt;tdis++){
      for(int cidx=0; cidx<n_contract; cidx++){
	for(int gcombidx=0;gcombidx<8;gcombidx++){
	  std::cout << "tK " << t_K << " tdis " << tdis << " C" << cidx+con_off << " gcombidx " << gcombidx << std::endl;
	  ComplexD expect = convertComplexD(expect_r(t_K,tdis,cidx,gcombidx));
	  ComplexD got = convertComplexD(got_r(t_K,tdis,cidx,gcombidx));
	  
	  double rdiff = fabs(got.real()-expect.real());
	  double idiff = fabs(got.imag()-expect.imag());
	  if(rdiff > tol|| idiff > tol){
	    printf("Fail: KtoPiPi type4 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	    fail = true;
	  }else
	    printf("Pass: KtoPiPi type4 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoPiPi type4 contract full failed\n");
    
  for(int t_K=0;t_K<Lt;t_K++){
    for(int tdis=0;tdis<Lt;tdis++){
      for(int cidx=0; cidx<2; cidx++){
	std::cout << "tK " << t_K << " tdis " << tdis << " mix4(" << cidx << ")" << std::endl;
	ComplexD expect = convertComplexD(expect_mix_r(t_K,tdis,cidx));
	ComplexD got = convertComplexD(got_mix_r(t_K,tdis,cidx));
	  
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	    printf("Fail: KtoPiPi mix4 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	    fail = true;
	}else
	  printf("Pass: KtoPiPi mix4 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
      }
    }
  }

  if(fail) ERR.General("","","KtoPiPi mix4 contract full failed\n");

  std::cout << "testKtoPiPiType4FieldFull passed" << std::endl;
}


//Test the openmp Grid implementation vs the non-Grid implementation of type 1
template<typename StandardA2Apolicies, typename GridA2Apolicies>
void testKtoPiPiType1GridOmpStd(const A2AArg &a2a_args,
				const A2AvectorW<GridA2Apolicies> &Wgrid, A2AvectorV<GridA2Apolicies> &Vgrid,
				const A2AvectorW<GridA2Apolicies> &Whgrid, A2AvectorV<GridA2Apolicies> &Vhgrid,
				const A2AvectorW<StandardA2Apolicies> &Wstd, A2AvectorV<StandardA2Apolicies> &Vstd,
				const A2AvectorW<StandardA2Apolicies> &Whstd, A2AvectorV<StandardA2Apolicies> &Vhstd,
				const double tol){
  std::cout << "Starting testKtoPiPiType1GridOmpStd type1 full test\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      
 
  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::mf_WW mf_WW_grid;
  typedef typename ComputeKtoPiPiGparity<StandardA2Apolicies>::mf_WW mf_WW_std;
  std::vector<mf_WW_grid> mf_kaon_grid(Lt);
  std::vector<mf_WW_std> mf_kaon_std(Lt);
  
  for(int t=0;t<Lt;t++){
    mf_kaon_grid[t].setup(Wgrid,Whgrid,t,t);
    mf_kaon_grid[t].testRandom();

    mf_kaon_std[t].setup(Wstd,Whstd,t,t);
  }
  copy(mf_kaon_std, mf_kaon_grid);
  assert(compare(mf_kaon_std, mf_kaon_grid, 1e-12));

  
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType_grid;
  typedef typename ComputeKtoPiPiGparity<StandardA2Apolicies>::ResultsContainerType ResultsContainerType_std;
  
  std::vector<int> tsep_k_pi = {3,4};
  std::vector<ResultsContainerType_std> std_r(2);
  std::vector<ResultsContainerType_grid> grid_r(2);

  //int tstep = 2;
  int tstep = 1;
  int tsep_pion = 1;
  ThreeMomentum p_pi1(1,1,1);
  ThreeMomentum p_pi2 = -p_pi1;

  MesonFieldMomentumContainer<GridA2Apolicies> mf_pion_grid;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_grid_tmp(Lt);

  MesonFieldMomentumContainer<StandardA2Apolicies> mf_pion_std;
  std::vector<A2AmesonField<StandardA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_std_tmp(Lt);

  for(int t=0;t<Lt;t++){
    mf_pion_grid_tmp[t].setup(Wgrid,Vgrid,t,t);
    mf_pion_grid_tmp[t].testRandom();

    mf_pion_std_tmp[t].setup(Wstd,Vstd,t,t);
  }
  copy(mf_pion_std_tmp, mf_pion_grid_tmp);
  
  mf_pion_grid.copyAdd(p_pi1, mf_pion_grid_tmp);
  mf_pion_std.copyAdd(p_pi1, mf_pion_std_tmp);
  
  for(int t=0;t<Lt;t++){
    mf_pion_grid_tmp[t].testRandom();
  }
  copy(mf_pion_std_tmp, mf_pion_grid_tmp);
  
  mf_pion_grid.copyAdd(p_pi2, mf_pion_grid_tmp);
  mf_pion_std.copyAdd(p_pi2, mf_pion_std_tmp);

  assert(mf_pion_grid.get(p_pi1).size() ==  Lt);
  assert(mf_pion_grid.get(p_pi2).size() ==  Lt);
  assert(mf_pion_std.get(p_pi1).size() ==  Lt);
  assert(mf_pion_std.get(p_pi2).size() ==  Lt);
    
  assert(compare(mf_pion_grid.get(p_pi1), mf_pion_std.get(p_pi1),1e-12));
  assert(compare(mf_pion_grid.get(p_pi2), mf_pion_std.get(p_pi2),1e-12));

  for(int t=0;t<Lt;t++)
    std::cout << "TEST pi1 " << mf_pion_grid.get(p_pi1)[t].norm2() << " " << mf_pion_std.get(p_pi1)[t].norm2() << std::endl;
  
  for(int t=0;t<Lt;t++)
    std::cout << "TEST pi2 " << mf_pion_grid.get(p_pi2)[t].norm2() << " " << mf_pion_std.get(p_pi2)[t].norm2() << std::endl;

  
  std::cout << "testKtoPiPiType1GridOmpStd computing using SIMD implementation" << std::endl;
  ComputeKtoPiPiGparity<GridA2Apolicies>::type1_omp(grid_r.data(), tsep_k_pi, tsep_pion, tstep, 1,  p_pi1, mf_kaon_grid, mf_pion_grid, Vgrid, Vhgrid, Wgrid, Whgrid);

  std::cout << "testKtoPiPiType1GridOmpStd computing using non-SIMD implementation" << std::endl;
  ComputeKtoPiPiGparity<StandardA2Apolicies>::type1_omp(std_r.data(), tsep_k_pi, tsep_pion, tstep, 1,  p_pi1, mf_kaon_std, mf_pion_std, Vstd, Vhstd, Wstd, Whstd);

  for(int tsep_k_pi_idx=0; tsep_k_pi_idx<2; tsep_k_pi_idx++){
    std::cout << "testKtoPiPiType1GridOmpStd comparing results for tsep_k_pi idx " << tsep_k_pi_idx << std::endl;
      
    if(!compareKtoPiPi(std_r[tsep_k_pi_idx], "std",
  		       grid_r[tsep_k_pi_idx], "grid",
  		       "KtoPiPi type1", tol)){
      ERR.General("","testKtoPiPiType1GridOmpStd","KtoPiPi type1 contract full failed\n");
    }
  }	  
  std::cout << "testKtoPiPiType1GridOmpStd passed" << std::endl;
}




template<typename GridA2Apolicies>
void testKtoPiPiType1FieldFull(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting testKtoPiPiType1FieldFull" << std::endl;

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorW<GridA2Apolicies> Wgrid(a2a_args, simd_dims), Whgrid(a2a_args, simd_dims);
  A2AvectorV<GridA2Apolicies> Vgrid(a2a_args, simd_dims), Vhgrid(a2a_args, simd_dims);

  Wgrid.testRandom();
  Vgrid.testRandom();

  Whgrid.testRandom();
  Vhgrid.testRandom();

  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::mf_WW mf_WW;
  std::vector<mf_WW> mf_kaon(Lt);
  for(int t=0;t<Lt;t++){
    mf_kaon[t].setup(Wgrid,Whgrid,t,t);
    mf_kaon[t].testRandom();
  }
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType;
  
  std::vector<int> tsep_k_pi = {3,4};
  std::vector<ResultsContainerType> expect_r(2);
  std::vector<ResultsContainerType> got_r(2);

  int tstep = 2;
  int tsep_pion = 1;
  ThreeMomentum p_pi1(1,1,1);
  ThreeMomentum p_pi2 = -p_pi1;

  MesonFieldMomentumContainer<GridA2Apolicies> mf_pion;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_tmp(Lt);
  for(int t=0;t<Lt;t++){
    mf_pion_tmp[t].setup(Wgrid,Vgrid,t,t);
    mf_pion_tmp[t].testRandom();
  }
  mf_pion.copyAdd(p_pi1, mf_pion_tmp);
  for(int t=0;t<Lt;t++){
    mf_pion_tmp[t].testRandom();
  }
  mf_pion.copyAdd(p_pi2, mf_pion_tmp);

  ComputeKtoPiPiGparity<GridA2Apolicies>::type1_omp(expect_r.data(), tsep_k_pi, tsep_pion, tstep, 1,  p_pi1, mf_kaon, mf_pion, Vgrid, Vhgrid, Wgrid, Whgrid);  
  ComputeKtoPiPiGparity<GridA2Apolicies>::type1_field_SIMD(got_r.data(), tsep_k_pi, tsep_pion, tstep, p_pi1, mf_kaon, mf_pion, Vgrid, Vhgrid, Wgrid, Whgrid);

  static const int n_contract = 6; //ten type4 diagrams
  static const int con_off = 1; //index of first contraction in set
  
  bool fail = false;
  for(int tsep_k_pi_idx=0; tsep_k_pi_idx<2; tsep_k_pi_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<n_contract; cidx++){
	  for(int gcombidx=0;gcombidx<8;gcombidx++){
	    std::cout << "tsep_k_pi=" << tsep_k_pi[tsep_k_pi_idx] << " tK " << t_K << " tdis " << tdis << " C" << cidx+con_off << " gcombidx " << gcombidx << std::endl;
	    ComplexD expect = convertComplexD(expect_r[tsep_k_pi_idx](t_K,tdis,cidx,gcombidx));
	    ComplexD got = convertComplexD(got_r[tsep_k_pi_idx](t_K,tdis,cidx,gcombidx));
	    
	    double rdiff = fabs(got.real()-expect.real());
	    double idiff = fabs(got.imag()-expect.imag());
	    if(rdiff > tol|| idiff > tol){
	      printf("Fail rank %d: KtoPiPi type1 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",UniqueID(),got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      printf("Pass rank %d: KtoPiPi type1 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",UniqueID(),got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","testKtoPiPiType1FieldFull","KtoPiPi type1 contract full failed\n");
  std::cout << "testKtoPiPiType1FieldFull passed" << std::endl;
}



template<typename GridA2Apolicies>
void testKtoPiPiType2FieldFull(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting testKtoPiPiType2FieldFull" << std::endl;

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorW<GridA2Apolicies> Wgrid(a2a_args, simd_dims), Whgrid(a2a_args, simd_dims);
  A2AvectorV<GridA2Apolicies> Vgrid(a2a_args, simd_dims), Vhgrid(a2a_args, simd_dims);

  Wgrid.testRandom();
  Vgrid.testRandom();

  Whgrid.testRandom();
  Vhgrid.testRandom();

  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::mf_WW mf_WW;
  std::vector<mf_WW> mf_kaon(Lt);
  for(int t=0;t<Lt;t++){
    mf_kaon[t].setup(Wgrid,Whgrid,t,t);
    mf_kaon[t].testRandom();
  }
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType;
  
  std::vector<int> tsep_k_pi = {3,4};
  std::vector<ResultsContainerType> expect_r(2);
  std::vector<ResultsContainerType> got_r(2);

  int tstep = 2;
  int tsep_pion = 1;

  //For type2 we want >1 p_pi1 to test the average over momenta
  ThreeMomentum p_pi1(1,1,1);
  ThreeMomentum p_pi2 = -p_pi1;

  ThreeMomentum p_pi1_2(-1,1,1);
  ThreeMomentum p_pi2_2 = -p_pi1_2;


  MesonFieldMomentumContainer<GridA2Apolicies> mf_pion;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_tmp(Lt);
  for(int t=0;t<Lt;t++){
    mf_pion_tmp[t].setup(Wgrid,Vgrid,t,t);
    mf_pion_tmp[t].testRandom();
  }
  mf_pion.copyAdd(p_pi1, mf_pion_tmp);
  for(int t=0;t<Lt;t++)
    mf_pion_tmp[t].testRandom();
  mf_pion.copyAdd(p_pi2, mf_pion_tmp);

  for(int t=0;t<Lt;t++)
    mf_pion_tmp[t].testRandom();  
  mf_pion.copyAdd(p_pi1_2, mf_pion_tmp);

  for(int t=0;t<Lt;t++)
    mf_pion_tmp[t].testRandom();
  mf_pion.copyAdd(p_pi2_2, mf_pion_tmp);

  std::vector<ThreeMomentum> p_pi1_all(2);
  p_pi1_all[0] = p_pi1;
  p_pi1_all[1] = p_pi1_2;


  ComputeKtoPiPiGparity<GridA2Apolicies>::type2_omp_v2(expect_r.data(), tsep_k_pi, tsep_pion, tstep,  p_pi1_all, mf_kaon, mf_pion, Vgrid, Vhgrid, Wgrid, Whgrid);  
  ComputeKtoPiPiGparity<GridA2Apolicies>::type2_field_SIMD(got_r.data(), tsep_k_pi, tsep_pion, tstep, p_pi1_all, mf_kaon, mf_pion, Vgrid, Vhgrid, Wgrid, Whgrid);

  static const int n_contract = 6; //ten type4 diagrams
  static const int con_off = 7; //index of first contraction in set
  
  bool fail = false;
  for(int tsep_k_pi_idx=0; tsep_k_pi_idx<2; tsep_k_pi_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<n_contract; cidx++){
	  for(int gcombidx=0;gcombidx<8;gcombidx++){
	    std::cout << "tsep_k_pi=" << tsep_k_pi[tsep_k_pi_idx] << " tK " << t_K << " tdis " << tdis << " C" << cidx+con_off << " gcombidx " << gcombidx << std::endl;
	    ComplexD expect = convertComplexD(expect_r[tsep_k_pi_idx](t_K,tdis,cidx,gcombidx));
	    ComplexD got = convertComplexD(got_r[tsep_k_pi_idx](t_K,tdis,cidx,gcombidx));
	    
	    double rdiff = fabs(got.real()-expect.real());
	    double idiff = fabs(got.imag()-expect.imag());
	    if(rdiff > tol|| idiff > tol){
	      printf("Fail: KtoPiPi type2 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      printf("Pass: KtoPiPi type2 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoPiPi type2 contract full failed\n");

  std::cout << "testKtoPiPiType2FieldFull passed" << std::endl;
}




template<typename GridA2Apolicies>
void testKtoPiPiType3FieldFull(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting testKtoPiPiType3FieldFull" << std::endl;

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorW<GridA2Apolicies> Wgrid(a2a_args, simd_dims), Whgrid(a2a_args, simd_dims);
  A2AvectorV<GridA2Apolicies> Vgrid(a2a_args, simd_dims), Vhgrid(a2a_args, simd_dims);

  Wgrid.testRandom();
  Vgrid.testRandom();

  Whgrid.testRandom();
  Vhgrid.testRandom();

  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::mf_WW mf_WW;
  std::vector<mf_WW> mf_kaon(Lt);
  for(int t=0;t<Lt;t++){
    mf_kaon[t].setup(Wgrid,Whgrid,t,t);
    mf_kaon[t].testRandom();
  }
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType;
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::MixDiagResultsContainerType MixDiagResultsContainerType;  

  std::vector<int> tsep_k_pi = {3,4};
  std::vector<ResultsContainerType> expect_r(2);
  std::vector<ResultsContainerType> got_r(2);

  std::vector<MixDiagResultsContainerType> expect_mix_r(2);
  std::vector<MixDiagResultsContainerType> got_mix_r(2);

  int tstep = 2;
  int tsep_pion = 1;
  ThreeMomentum p_pi1(1,1,1);
  ThreeMomentum p_pi2 = -p_pi1;

  MesonFieldMomentumContainer<GridA2Apolicies> mf_pion;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_tmp(Lt);
  for(int t=0;t<Lt;t++){
    mf_pion_tmp[t].setup(Wgrid,Vgrid,t,t);
    mf_pion_tmp[t].testRandom();
  }
  mf_pion.copyAdd(p_pi1, mf_pion_tmp);
  for(int t=0;t<Lt;t++){
    mf_pion_tmp[t].testRandom();
  }
  mf_pion.copyAdd(p_pi2, mf_pion_tmp);

  std::vector<ThreeMomentum> p_pi1_all(1, p_pi1);

  ComputeKtoPiPiGparity<GridA2Apolicies>::type3_omp_v2(expect_r.data(), expect_mix_r.data(), tsep_k_pi, tsep_pion, tstep,  p_pi1_all, mf_kaon, mf_pion, Vgrid, Vhgrid, Wgrid, Whgrid);  
  ComputeKtoPiPiGparity<GridA2Apolicies>::type3_field_SIMD(got_r.data(), got_mix_r.data(), tsep_k_pi, tsep_pion, tstep, p_pi1_all, mf_kaon, mf_pion, Vgrid, Vhgrid, Wgrid, Whgrid);

  static const int n_contract = 10; //ten type4 diagrams
  static const int con_off = 13; //index of first contraction in set
  
  bool fail = false;
  for(int tsep_k_pi_idx=0; tsep_k_pi_idx<2; tsep_k_pi_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<n_contract; cidx++){
	  for(int gcombidx=0;gcombidx<8;gcombidx++){
	    std::cout << "tsep_k_pi=" << tsep_k_pi[tsep_k_pi_idx] << " tK " << t_K << " tdis " << tdis << " C" << cidx+con_off << " gcombidx " << gcombidx << std::endl;
	    ComplexD expect = convertComplexD(expect_r[tsep_k_pi_idx](t_K,tdis,cidx,gcombidx));
	    ComplexD got = convertComplexD(got_r[tsep_k_pi_idx](t_K,tdis,cidx,gcombidx));
	    
	    double rdiff = fabs(got.real()-expect.real());
	    double idiff = fabs(got.imag()-expect.imag());
	    if(rdiff > tol|| idiff > tol){
	      printf("Fail: KtoPiPi type3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      printf("Pass: KtoPiPi type3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoPiPi type3 contract full failed\n");

  for(int tsep_k_pi_idx=0; tsep_k_pi_idx<2; tsep_k_pi_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<2; cidx++){
	  std::cout << "tsep_k_pi=" << tsep_k_pi[tsep_k_pi_idx] << " tK " << t_K << " tdis " << tdis << " mix3(" << cidx << ")" << std::endl;
	  ComplexD expect = convertComplexD(expect_mix_r[tsep_k_pi_idx](t_K,tdis,cidx));
	  ComplexD got = convertComplexD(got_mix_r[tsep_k_pi_idx](t_K,tdis,cidx));
	  
	  double rdiff = fabs(got.real()-expect.real());
	  double idiff = fabs(got.imag()-expect.imag());
	  if(rdiff > tol|| idiff > tol){
	    printf("Fail: KtoPiPi mix3 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	    fail = true;
	  }else
	    printf("Pass: KtoPiPi mix3 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	}
      }
    }
  }

  if(fail) ERR.General("","","KtoPiPi mix3 contract full failed\n");

  std::cout << "testKtoPiPiType3FieldFull passed" << std::endl;
}



CPS_END_NAMESPACE
