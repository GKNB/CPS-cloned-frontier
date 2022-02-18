#pragma once

CPS_START_NAMESPACE

template<typename GridA2Apolicies>
void testKtoSigmaType12FieldFull(const A2AArg &a2a_args, const double tol){
  if(!UniqueID()) std::cout << "Starting K->sigma type1/2 full test\n";

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
  
  std::vector<int> tsep_k_sigma = {3,4};
  std::vector<ResultsContainerType> expect_r(2);
  std::vector<ResultsContainerType> got_r(2);

  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_sigma(Lt);
  for(int t=0;t<Lt;t++){
    mf_sigma[t].setup(Wgrid,Vgrid,t,t);
    mf_sigma[t].testRandom();
  }

  ComputeKtoSigma<GridA2Apolicies> compute(Vgrid, Wgrid, Vhgrid, Whgrid, mf_kaon, tsep_k_sigma);

  compute.type12_omp(expect_r, mf_sigma);
  compute.type12_field_SIMD(got_r, mf_sigma);

  static const int n_contract = 5;  
  
  bool fail = false;
  for(int tsep_k_sigma_idx=0; tsep_k_sigma_idx<2; tsep_k_sigma_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<n_contract; cidx++){
	  for(int gcombidx=0;gcombidx<8;gcombidx++){
	    if(!UniqueID()) std::cout << "tsep_k_sigma=" << tsep_k_sigma[tsep_k_sigma_idx] << " tK " << t_K << " tdis " << tdis << " C" << cidx << " gcombidx " << gcombidx << std::endl;
	    ComplexD expect = convertComplexD(expect_r[tsep_k_sigma_idx](t_K,tdis,cidx,gcombidx));
	    ComplexD got = convertComplexD(got_r[tsep_k_sigma_idx](t_K,tdis,cidx,gcombidx));
	    
	    double rdiff = fabs(got.real()-expect.real());
	    double idiff = fabs(got.imag()-expect.imag());
	    if(rdiff > tol|| idiff > tol){
	      if(!UniqueID()) printf("Fail: KtoSigma type1/2 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      if(!UniqueID()) printf("Pass: KtoSigma type1/2 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoSigma type1/2 contract full failed on node %d\n", UniqueID());

}



template<typename GridA2Apolicies>
void testKtoSigmaType3FieldFull(const A2AArg &a2a_args, const double tol){
  if(!UniqueID()) std::cout << "Starting K->sigma type3 full test\n";

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
  
  std::vector<int> tsep_k_sigma = {3,4};
  std::vector<ResultsContainerType> expect_r(2);
  std::vector<ResultsContainerType> got_r(2);
  std::vector<MixDiagResultsContainerType> expect_mix_r(2);
  std::vector<MixDiagResultsContainerType> got_mix_r(2);

  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_sigma(Lt);
  for(int t=0;t<Lt;t++){
    mf_sigma[t].setup(Wgrid,Vgrid,t,t);
    mf_sigma[t].testRandom();
  }

  ComputeKtoSigma<GridA2Apolicies> compute(Vgrid, Wgrid, Vhgrid, Whgrid, mf_kaon, tsep_k_sigma);

  compute.type3_omp(expect_r, expect_mix_r, mf_sigma);
  compute.type3_field_SIMD(got_r, got_mix_r, mf_sigma);

  static const int n_contract = 9;  
  
  bool fail = false;
  for(int tsep_k_sigma_idx=0; tsep_k_sigma_idx<2; tsep_k_sigma_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<n_contract; cidx++){
	  for(int gcombidx=0;gcombidx<8;gcombidx++){
	    if(!UniqueID()) std::cout << "tsep_k_sigma=" << tsep_k_sigma[tsep_k_sigma_idx] << " tK " << t_K << " tdis " << tdis << " C" << cidx << " gcombidx " << gcombidx << std::endl;
	    ComplexD expect = convertComplexD(expect_r[tsep_k_sigma_idx](t_K,tdis,cidx,gcombidx));
	    ComplexD got = convertComplexD(got_r[tsep_k_sigma_idx](t_K,tdis,cidx,gcombidx));
	    
	    double rdiff = fabs(got.real()-expect.real());
	    double idiff = fabs(got.imag()-expect.imag());
	    if(rdiff > tol|| idiff > tol){
	      if(!UniqueID()) printf("Fail: KtoSigma type3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      if(!UniqueID()) printf("Pass: KtoSigma type3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoSigma type3 contract full failed node %d\n", UniqueID());


  for(int tsep_k_sigma_idx=0; tsep_k_sigma_idx<2; tsep_k_sigma_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<2; cidx++){
	  if(!UniqueID()) std::cout << "tsep_k_sigma=" << tsep_k_sigma[tsep_k_sigma_idx] << " tK " << t_K << " tdis " << tdis << " mix3(" << cidx << ")" << std::endl;
	  ComplexD expect = convertComplexD(expect_mix_r[tsep_k_sigma_idx](t_K,tdis,cidx));
	  ComplexD got = convertComplexD(got_mix_r[tsep_k_sigma_idx](t_K,tdis,cidx));
	  
	  double rdiff = fabs(got.real()-expect.real());
	  double idiff = fabs(got.imag()-expect.imag());
	  if(rdiff > tol|| idiff > tol){
	    if(!UniqueID()) printf("Fail: KtoSigma mix3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	    fail = true;
	  }else
	    if(!UniqueID()) printf("Pass: KtoSigma mix3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	}
      }
    }
  }

  if(fail) ERR.General("","","KtoSigma mix3 contract full failed node %d\n", UniqueID());
}




template<typename GridA2Apolicies>
void testKtoSigmaType4FieldFull(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting testKtoSigmaType4FieldFull: K->sigma type4 full test\n";

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
  
  std::vector<int> tsep_k_sigma = {3,4};
  ResultsContainerType expect_r;
  ResultsContainerType got_r;
  MixDiagResultsContainerType expect_mix_r;
  MixDiagResultsContainerType got_mix_r;

  ComputeKtoSigma<GridA2Apolicies> compute(Vgrid, Wgrid, Vhgrid, Whgrid, mf_kaon, tsep_k_sigma);

  compute.type4_omp(expect_r, expect_mix_r);
  compute.type4_field_SIMD(got_r, got_mix_r);

  static const int n_contract = 9;  
  
  bool fail = false;
  for(int t_K=0;t_K<Lt;t_K++){
    for(int tdis=0;tdis<Lt;tdis++){
      for(int cidx=0; cidx<n_contract; cidx++){
	for(int gcombidx=0;gcombidx<8;gcombidx++){
	  if(!UniqueID()) std::cout << "tK " << t_K << " tdis " << tdis << " C" << cidx << " gcombidx " << gcombidx << std::endl;
	  ComplexD expect = convertComplexD(expect_r(t_K,tdis,cidx,gcombidx));
	  ComplexD got = convertComplexD(got_r(t_K,tdis,cidx,gcombidx));
	    
	  double rdiff = fabs(got.real()-expect.real());
	  double idiff = fabs(got.imag()-expect.imag());
	  if(rdiff > tol|| idiff > tol){
	    if(!UniqueID()) printf("Fail: KtoSigma type4 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	    fail = true;
	  }else
	    if(!UniqueID()) printf("Pass: KtoSigma type4 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	}
      }
    }
  }
  
  if(fail) ERR.General("","","KtoSigma type4 contract full failed node %d\n", UniqueID());

  
  for(int t_K=0;t_K<Lt;t_K++){
    for(int tdis=0;tdis<Lt;tdis++){
      for(int cidx=0; cidx<2; cidx++){
	if(!UniqueID()) std::cout << "tK " << t_K << " tdis " << tdis << " mix3(" << cidx << ")" << std::endl;
	ComplexD expect = convertComplexD(expect_mix_r(t_K,tdis,cidx));
	ComplexD got = convertComplexD(got_mix_r(t_K,tdis,cidx));
	  
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  if(!UniqueID()) printf("Fail: KtoSigma mix4 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}else
	  if(!UniqueID()) printf("Pass: KtoSigma mix4 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
      }
    }
  }

  if(fail) ERR.General("","","KtoSigma mix4 contract full failed node %d\n", UniqueID());
  std::cout << "testKtoSigmaType4FieldFull passed" << std::endl;
}


CPS_END_NAMESPACE
