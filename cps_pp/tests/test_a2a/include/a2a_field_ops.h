#pragma once

CPS_START_NAMESPACE

//Test the getFlavorDilutedVect methods
template<typename A2Apolicies_std>
void testA2AfieldGetFlavorDilutedVect(const A2AArg &a2a_args, double tol){
  assert(a2a_args.src_width == 1);
  std::cout << "Running testA2AfieldGetFlavorDilutedVect" << std::endl;
  assert(GJP.Gparity());
  A2AvectorWfftw<A2Apolicies_std> Wf_p(a2a_args);
  Wf_p.testRandom();
  typedef typename A2Apolicies_std::FermionFieldType::FieldSiteType FieldSiteType;
  typedef typename A2Apolicies_std::ComplexType ComplexType;

  //wFFT^(i)_{sc,f}(p,t) = \rho^(i_h,i_f,i_t,i_sc)_sc(p,i_t) \delta_{i_f,f}\delta_{i_t, t}
  //wFFTP^(i_h,i_sc)_{sc,f} (p,t) = \rho^(i_h,f,t,i_sc)_sc(p,t)
  
  //For high modes getFlavorDilutedVect returns   wFFTP^(j_h,j_sc)_{sc',f'}(p,t) \delta_{f',j_f}   [cf a2a_dilutions.h]
  //i is only used for low mode indices. If i>=nl  the appropriate data is picked using i_high_unmapped
  //t is the local time

  int p3d=0;

  TimePackedIndexDilution dil_tp(a2a_args);
  StandardIndexDilution dil_full(a2a_args);
  
  int nl = Wf_p.getNl();
  int nh = dil_tp.getNmodes() - nl;
    
  for(int tpmode=nl+1;tpmode<nl+nh;tpmode++){
    std::cout << "Checking timepacked mode " << tpmode << std::endl;
    //Unmap mode index
    modeIndexSet u; dil_tp.indexUnmap(tpmode-nl,u);

    //u should contain all indices but time
    assert(u.spin_color != -1 && u.flavor != -1 && u.hit != -1 && u.time == -1);

    for(int tlcl=0;tlcl<GJP.TnodeSites();tlcl++){
      int tglb = tlcl+GJP.TnodeCoor()*GJP.TnodeSites();
      std::cout << "Mode flavor is " << u.flavor << std::endl;
      //Check it is a delta function flavor
      //note the first arg, the mode index, is ignored for mode>=nl
      SCFvectorPtr<FieldSiteType> ptr = Wf_p.getFlavorDilutedVect(nl, u, p3d, tlcl);

      //Should be non-zero for the f = u.flavor
      ComplexType const *dfa = ptr.getPtr(u.flavor);
      ComplexType const *dfb = ptr.getPtr(!u.flavor);
      
      for(int sc=0;sc<12;sc++){
	
	std::cout << "For sc=" << sc << " t=" << tglb << "  f=" << u.flavor << " : (" << dfa->real() << "," << dfa->imag() << ") f=" << !u.flavor << " (" << dfb->real() << "," << dfb->imag() << ")" << std::endl; 
    
	assert(!ptr.isZero(u.flavor));
	assert(dfa->real() != 0.0 && dfa->imag() != 0.0);    
	assert(ptr.isZero(!u.flavor));
	assert(dfb->real() == 0.0 && dfb->imag() == 0.0);    
     
	//Get elem directly
	//We have  wFFTP^(j_h,j_sc)_{sc,f}(p,t) \delta_{f,j_f}
	//         = \rho^(j_h,f,t,j_sc)_sc(p,t) \delta_{f,j_f}
	//         = wFFT(j_h,f,t,j_sc)_{sc,f}(p,t) \delta_{f,j_f}
	//Full mode (j_h,j_f,t,j_sc):
	int full_mode = nl + dil_full.indexMap(u.hit, tglb, u.spin_color, u.flavor);

	FieldSiteType ea = Wf_p.elem(full_mode, p3d, tlcl, sc, u.flavor);

	std::cout << "Using elem function : (" << ea.real() << "," << ea.imag() << ")" << std::endl; 
	
	assert(*dfa == ea);
	++dfa;
	++dfb;
      }
    }
    
  }
  std::cout << "testA2AfieldGetFlavorDilutedVect passed" << std::endl;
}

void testA2AallocFree(const A2AArg &a2a_args,Lattice &lat){
#ifdef USE_GRID
  typedef A2ApoliciesSIMDdoubleManualAlloc A2Apolicies;
#else  
  typedef A2ApoliciesDoubleManualAlloc A2Apolicies;
#endif
  
  typedef typename A2Apolicies::FermionFieldType FermionFieldType;
  typedef typename A2Apolicies::SourcePolicies SourcePolicies;
  typedef typename A2Apolicies::ComplexType mf_Complex;
  
  typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);

  A2AvectorVfftw<A2Apolicies> Vfft(a2a_args,fp);
  double size =  A2AvectorVfftw<A2Apolicies>::Mbyte_size(a2a_args,fp);
  
  for(int i=0;i<100;i++){
    if(!UniqueID()) printf("Pre-init\n");
    printMem(); fflush(stdout);

    if(!UniqueID()) printf("Expected size %f MB\n",size);
    
    if(!UniqueID()) printf("Post-init\n");
    printMem(); fflush(stdout);

    Vfft.allocModes();

    for(int i=0;i<Vfft.getNmodes();i++){
      assert(&Vfft.getMode(i) != NULL);
      Vfft.getMode(i).zero();
    }
    if(!UniqueID()) printf("Post-alloc\n");
    printMem(); fflush(stdout);

    Vfft.freeModes();

    for(int i=0;i<Vfft.getNmodes();i++)
      assert(&Vfft.getMode(i) == NULL);
    
    if(!UniqueID()) printf("Post-free\n");
    printMem(); fflush(stdout);
  }
}



template<typename A2Apolicies>
void testA2AvectorIO(const A2AArg &a2a_args){
  if(!UniqueID()) printf("testA2AvectorIO called\n");
  typedef typename A2AvectorV<A2Apolicies>::FieldInputParamType FieldParams;
  
  setupFieldParams2<A2Apolicies, typename ComplexClassify<typename A2Apolicies::ComplexType>::type> p;

  CPSfield_checksumType cksumtype[2] = { checksumBasic, checksumCRC32 };
  FP_FORMAT fileformat[2] = { FP_IEEE64BIG, FP_IEEE64LITTLE };
  
  for(int i=0;i<2;i++){
    for(int j=0;j<2;j++){
  
      {
  	A2AvectorV<A2Apolicies> Va(a2a_args, p.params);
  	Va.testRandom();

  	Va.writeParallel("Vvector", fileformat[j], cksumtype[i]);

  	A2AArg def;
  	def.nl = 1; def.nhits = 1; def.rand_type = UONE; def.src_width = 1;

  	A2AvectorV<A2Apolicies> Vb(def, p.params);
  	Vb.readParallel("Vvector");

  	assert( Va.paramsEqual(Vb) );
  	assert( Va.getNmodes() == Vb.getNmodes() );
  
  	for(int i=0;i<Va.getNmodes();i++){
  	  assert( Va.getMode(i).equals(Vb.getMode(i)) );
  	}
      }

  
      {
  	A2AvectorW<A2Apolicies> Wa(a2a_args, p.params);
  	Wa.testRandom();

  	Wa.writeParallel("Wvector", fileformat[j], cksumtype[i]);

  	A2AArg def;
  	def.nl = 1; def.nhits = 1; def.rand_type = UONE; def.src_width = 1;

  	A2AvectorW<A2Apolicies> Wb(def, p.params);
  	Wb.readParallel("Wvector");

  	assert( Wa.paramsEqual(Wb) );
  	assert( Wa.getNmodes() == Wb.getNmodes() );
  
  	for(int i=0;i<Wa.getNl();i++){
  	  assert( Wa.getWl(i).equals(Wb.getWl(i)) );
  	}
  	for(int i=0;i<Wa.getNhits();i++){
  	  assert( Wa.getWh(i).equals(Wb.getWh(i)) );
  	}    
      }      
    }
  }




  //Test parallel read/write with separate metadata
  for(int i=0;i<2;i++){
    {//V  
      A2AvectorV<A2Apolicies> Va(a2a_args, p.params);
      Va.testRandom();
    
      Va.writeParallelSeparateMetadata("Vvector_split", fileformat[i]);
    
      A2AvectorV<A2Apolicies> Vb(a2a_args, p.params);

      Vb.readParallelSeparateMetadata("Vvector_split");
    
      assert( Va.paramsEqual(Vb) );
      assert( Va.getNmodes() == Vb.getNmodes() );
    
      for(int i=0;i<Va.getNmodes();i++){
	assert( Va.getMode(i).equals(Vb.getMode(i)) );
      }
    }//V
    {//W
      A2AvectorW<A2Apolicies> Wa(a2a_args, p.params);
      Wa.testRandom();
      
      Wa.writeParallelSeparateMetadata("Wvector_split", fileformat[i]);

      A2AvectorW<A2Apolicies> Wb(a2a_args, p.params);
      Wb.readParallelSeparateMetadata("Wvector_split");

      assert( Wa.paramsEqual(Wb) );
      assert( Wa.getNmodes() == Wb.getNmodes() );
      
      for(int i=0;i<Wa.getNl();i++){
	assert( Wa.getWl(i).equals(Wb.getWl(i)) );
      }
      for(int i=0;i<Wa.getNhits();i++){
	assert( Wa.getWh(i).equals(Wb.getWh(i)) );
      }    
    }//W
  }
 
}



template<typename A2Apolicies>
void testA2AvectorTimesliceExtraction(const A2AArg &a2a_args){
  typedef typename A2AvectorV<A2Apolicies>::FieldInputParamType FieldParams;  
  setupFieldParams2<A2Apolicies, typename ComplexClassify<typename A2Apolicies::ComplexType>::type> p;

  {
    A2AvectorV<A2Apolicies> V(a2a_args, p.params);
    V.testRandom();

    A2AvectorV<A2Apolicies> V2(a2a_args, p.params);
    V2.zero();

    typedef typename A2AvectorV<A2Apolicies>::FieldSiteType FieldSiteType;

    for(int i=0;i<V.getNmodes();i++){    
      for(int t=0;t<GJP.TnodeSites();t++){
	FieldSiteType const* from0, *from1;
	FieldSiteType const* to0, *to1; 
	size_t sz1, sz2;
	V.getModeTimesliceData(from0, from1, sz1, i, t);
	V2.getModeTimesliceData(to0, to1, sz2, i, t);
	assert(sz1 == sz2);
	
	memcpy( (void*)to0, (void const*)from0, sz1 * sizeof(FieldSiteType));
	if(GJP.Gparity()) memcpy( (void*)to1, (void const*)from1, sz1 * sizeof(FieldSiteType));
      }
    }
    for(int i=0;i<V.getNmodes();i++){    
      assert(V2.getMode(i).equals( V.getMode(i), 1e-13 ) );
    }
    std::cout << "Passed V timeslice extraction test" << std::endl;
  }


  {
    A2AvectorW<A2Apolicies> W(a2a_args, p.params);
    W.testRandom();

    A2AvectorW<A2Apolicies> W2(a2a_args, p.params);
    W2.zero();

    typedef typename A2AvectorW<A2Apolicies>::FieldSiteType FieldSiteType;

    for(int i=0;i<W.getNmodes();i++){    
      for(int t=0;t<GJP.TnodeSites();t++){
	FieldSiteType const* from0, *from1;
	FieldSiteType const* to0, *to1; 
	size_t sz1, sz2;
	W.getModeTimesliceData(from0, from1, sz1, i, t);
	W2.getModeTimesliceData(to0, to1, sz2, i, t);
	assert(sz1 == sz2);
	
	memcpy( (void*)to0, (void const*)from0, sz1 * sizeof(FieldSiteType));
	if(GJP.Gparity()) memcpy( (void*)to1, (void const*)from1, sz1 * sizeof(FieldSiteType));
      }
    }
    for(int i=0;i<W.getNl();i++){    
      assert(W2.getWl(i).equals( W.getWl(i), 1e-13 ) );
    }
    for(int i=0;i<W.getNhits();i++){    
      assert(W2.getWh(i).equals( W.getWh(i), 1e-13 ) );
    }

    std::cout << "Passed W timeslice extraction test" << std::endl;
  }



  {
    A2AvectorVfftw<A2Apolicies> V(a2a_args, p.params);
    V.testRandom();

    A2AvectorVfftw<A2Apolicies> V2(a2a_args, p.params);
    V2.zero();

    typedef typename A2AvectorVfftw<A2Apolicies>::FieldSiteType FieldSiteType;

    for(int i=0;i<V.getNmodes();i++){    
      for(int t=0;t<GJP.TnodeSites();t++){
	FieldSiteType const* from0, *from1;
	FieldSiteType const* to0, *to1; 
	size_t sz1, sz2;
	V.getModeTimesliceData(from0, from1, sz1, i, t);
	V2.getModeTimesliceData(to0, to1, sz2, i, t);
	assert(sz1 == sz2);
	
	memcpy( (void*)to0, (void const*)from0, sz1 * sizeof(FieldSiteType));
	if(GJP.Gparity()) memcpy( (void*)to1, (void const*)from1, sz1 * sizeof(FieldSiteType));
      }
    }
    for(int i=0;i<V.getNmodes();i++){    
      assert(V2.getMode(i).equals( V.getMode(i), 1e-13 ) );
    }
    std::cout << "Passed Vfftw timeslice extraction test" << std::endl;
  }


  {
    A2AvectorWfftw<A2Apolicies> W(a2a_args, p.params);
    W.testRandom();

    A2AvectorWfftw<A2Apolicies> W2(a2a_args, p.params);
    W2.zero();

    typedef typename A2AvectorWfftw<A2Apolicies>::FieldSiteType FieldSiteType;

    for(int i=0;i<W.getNmodes();i++){    
      for(int t=0;t<GJP.TnodeSites();t++){
	FieldSiteType const* from0, *from1;
	FieldSiteType const* to0, *to1; 
	size_t sz1, sz2;
	W.getModeTimesliceData(from0, from1, sz1, i, t);
	W2.getModeTimesliceData(to0, to1, sz2, i, t);
	assert(sz1 == sz2);
	
	memcpy( (void*)to0, (void const*)from0, sz1 * sizeof(FieldSiteType));
	if(GJP.Gparity()) memcpy( (void*)to1, (void const*)from1, sz1 * sizeof(FieldSiteType));
      }
    }
    for(int i=0;i<W.getNl();i++){    
      assert(W2.getWl(i).equals( W.getWl(i), 1e-13 ) );
    }
    for(int i=0;i<W.getNhits();i++){
      for(int sc=0;sc<12;sc++)
	assert(W2.getWh(i,sc).equals( W.getWh(i,sc), 1e-13 ) );
    }

    std::cout << "Passed Wfftw timeslice extraction test" << std::endl;
  }

   
}
    

template<typename A2Apolicies>
void testA2AvectorWnorm(const A2AArg &a2a_args){
  assert(GJP.Gparity());

  typedef typename A2AvectorW<A2Apolicies>::FieldInputParamType FieldParams;  
  setupFieldParams2<A2Apolicies, typename ComplexClassify<typename A2Apolicies::ComplexType>::type> p;

  A2AvectorW<A2Apolicies> W(a2a_args, p.params);
  A2AhighModeSourceOriginal<A2Apolicies> impl;
  impl.setHighModeSources(W);
  
  StandardIndexDilution stdidx(a2a_args);

  CPSfermion4D<typename A2Apolicies::ComplexTypeD,typename A2Apolicies::FermionFieldType::FieldMappingPolicy, 
    typename A2Apolicies::FermionFieldType::FieldAllocPolicy> tmp1(p.params);

  int Nh = stdidx.getNh();
  std::vector< CPSfermion4D<cps::ComplexD,FourDpolicy<DynamicFlavorPolicy>, StandardAllocPolicy> > eta(Nh);

  for(int i=0;i<stdidx.getNh();i++){
    W.getDilutedSource(tmp1, i);
    tmp1.exportField(eta[i]);
  }

  int Lt = GJP.Tnodes()*GJP.TnodeSites();

  //LAZY
  //assert(GJP.Tnodes() == 1); //lazy
  if(GJP.Tnodes()!=1){
    std::cout << "WARNING (testA2AvectorWnorm) : Test does not presently support >1 node in time direction, skipping test" <<std::endl;
    return;
  }        
  
  size_t x3d = 0;

  //Check that W is correctly normalized such that \sum_i \eta_i(x0,t1) \eta^\dag_i(x0,t2)  produces a unit matrix in spin-color,flavor and time for some arbitrary x0
  bool fail = false;
  for(int sc1=0;sc1<12;sc1++){
    for(int f1=0;f1<2;f1++){
      for(int t1=0;t1<Lt;t1++){
	  
	size_t x4d1 = eta[0].threeToFour(x3d, t1);
	  
	for(int sc2=0;sc2<12;sc2++){
	  for(int f2=0;f2<2;f2++){
	    for(int t2=0;t2<Lt;t2++){

	      size_t x4d2 = eta[0].threeToFour(x3d, t2);

	      cps::ComplexD sum = 0;
	      bool expect_zero = sc1 != sc2 || f1 != f2 || t1 != t2;
	      cps::ComplexD expect = expect_zero ? 0. : 1.;

	      for(int i=0;i<Nh;i++){		  
		cps::ComplexD v1 = *( eta[i].site_ptr(x4d1, f1) + sc1);
		cps::ComplexD v2 = *( eta[i].site_ptr(x4d2, f2) + sc2);
		sum += std::conj(v1)*v2;
	      }  
	      
	      cps::ComplexD diff = sum - expect;

	      bool tfail = fabs(diff.real()) > 1e-08 || fabs(diff.imag()) > 1e-08;

	      std::cout << sc1 << " " << f1 << " " << t1 << " " << sc2 << " " << f2 << " " << t2 << " " << " got " << sum  << " expect " << expect << " diff " << diff << " " << (tfail ? "FAIL" : "pass") << std::endl;

	      if(tfail) fail=true;

	    }
	  }
	}
      }
    }
  }
  if(fail) std::cout << "W norm test FAILED" << std::endl;
  else std::cout << "W norm test passed" << std::endl;

  //For same site should get exactly a unit matrix structure in flavor
  {
    cps::ComplexD fstruct[2][2];
    
    int t=0;
    int sc=0;

    size_t x4d1 = eta[0].threeToFour(x3d, t);
    size_t x4d2 = eta[0].threeToFour(x3d, t);
  
    for(int f1=0;f1<2;f1++){
      for(int f2=0;f2<2;f2++){

	cps::ComplexD sum = 0;
	for(int i=0;i<Nh;i++){		  
	  cps::ComplexD v1 = *( eta[i].site_ptr(x4d1, f1) + sc);
	  cps::ComplexD v2 = *( eta[i].site_ptr(x4d2, f2) + sc);
	  sum += std::conj(v1)*v2;
	}  
	fstruct[f1][f2] = sum;
      }
    }

    std::cout << "For x1 == x2, flavor structure:" << std::endl;
    for(int f1=0;f1<2;f1++){
      for(int f2=0;f2<2;f2++){
	std::cout << fstruct[f1][f2] << " ";
      }
      std::cout << std::endl;
    }
    if( fabs(fstruct[0][1].real())>1e-8 || fabs(fstruct[0][1].imag())>1e-8 ){ std::cout << "Expect elem 0,1 to be 0!" << std::endl; fail=true; }
    if( fabs(fstruct[1][0].real())>1e-8 || fabs(fstruct[1][0].imag())>1e-8 ){ std::cout << "Expect elem 1,0 to be 0!" << std::endl; fail=true; }
    if( fabs(fstruct[0][0].real()-1.0)>1e-8 || fabs(fstruct[0][0].imag())>1e-8 ){ std::cout << "Expect elem 0,0 to be 1!" << std::endl; fail=true; }
    if( fabs(fstruct[1][1].real()-1.0)>1e-8 || fabs(fstruct[1][1].imag())>1e-8 ){ std::cout << "Expect elem 1,1 to be 1!" << std::endl; fail=true; }
  }  
  if(fail) std::cout << "W flavor struct test same site FAILED" << std::endl;
  else std::cout << "W flavor struct test same site passed" << std::endl;


  //For some other site on the same timeslice we should get a random phase
  {
    cps::ComplexD fstruct[2][2];
    
    size_t x3d2 = 1;
    int t=0;
    int sc=0;

    size_t x4d1 = eta[0].threeToFour(x3d, t);
    size_t x4d2 = eta[0].threeToFour(x3d2, t);
  
    for(int f1=0;f1<2;f1++){
      for(int f2=0;f2<2;f2++){

	cps::ComplexD sum = 0;
	for(int i=0;i<Nh;i++){		  
	  cps::ComplexD v1 = *( eta[i].site_ptr(x4d1, f1) + sc);
	  cps::ComplexD v2 = *( eta[i].site_ptr(x4d2, f2) + sc);
	  sum += std::conj(v1)*v2;
	}  
	fstruct[f1][f2] = sum;
      }
    }

    std::cout << "For x1 != x2, flavor structure:" << std::endl;
    for(int f1=0;f1<2;f1++){
      for(int f2=0;f2<2;f2++){
	std::cout << fstruct[f1][f2] << " ";
      }
      std::cout << std::endl;
    }
    if( fabs(fstruct[0][1].real())>1e-8 || fabs(fstruct[0][1].imag())>1e-8 ){ std::cout << "Expect elem 0,1 to be 0!" << std::endl; fail=true; }
    if( fabs(fstruct[1][0].real())>1e-8 || fabs(fstruct[1][0].imag())>1e-8 ){ std::cout << "Expect elem 1,0 to be 0!" << std::endl; fail=true; }
    if( fabs(std::norm(fstruct[0][0]) - 1.0) > 1e-8 ){ std::cout << "Expect elem 0,0 to be U(1)!" << std::endl; fail=true; }
    if( fabs(std::norm(fstruct[1][1]) - 1.0) > 1e-8 ){ std::cout << "Expect elem 1,1 to be U(1)!" << std::endl; fail=true; }
  }  
  if(fail) std::cout << "W flavor struct test diff site FAILED" << std::endl;
  else std::cout << "W flavor struct test diff site passed" << std::endl;

}
    

template<typename A2Apolicies>
void testXconjWsrc(typename A2Apolicies::FgridGFclass &lat){
  std::cout << "Starting testXconjWsrc" << std::endl;
  assert(GJP.Gparity());

  A2AArg a2a_args;
  a2a_args.nl = 100;
  a2a_args.nhits = 1;
  a2a_args.rand_type = UONE;
  a2a_args.src_width = 1;

  typedef typename A2AvectorW<A2Apolicies>::FieldInputParamType FieldParams;  
  setupFieldParams2<A2Apolicies, typename ComplexClassify<typename A2Apolicies::ComplexType>::type> p;

  A2AhighModeSourceXconj<A2Apolicies> impl;
  A2AvectorW<A2Apolicies> W(a2a_args, p.params);
  impl.setHighModeSources(W);
  
  typedef typename A2Apolicies::GridFermionField GridField;
  GridField tmp(lat.getUGrid());
  
  //Check Kronecker deltas
  StandardIndexDilution dil(a2a_args);
  int Lt = GJP.TnodeSites()*GJP.Tnodes();
 
  int hit = 0;
  for(int tcol=0;tcol<GJP.TnodeSites()*GJP.Tnodes();tcol++){
    for(int scol=0;scol<4;scol++){
      for(int ccol=0;ccol<3;ccol++){
	for(int fcol=0;fcol<2;fcol++){	
	  std::cout << "fcol:" << fcol << " ccol:" << ccol << " scol:" << scol << " tcol:" << tcol << std::endl;

	  int id = dil.indexMap(hit,tcol,ccol+3*scol,fcol);
	  tmp = Grid::Zero();
	  impl.get4DinverseSource(tmp, id, W);
	
	  //Check temporal delta
	  //Grid::Coordinate site(
	  typedef typename GridField::scalar_object sobj;
	  std::vector<sobj> slice_sums;
	  Grid::sliceSum(tmp,slice_sums,3);

	  std::cout << "Check temporal delta" << std::endl;
	  assert(slice_sums.size() == Lt);
	  for(int trow=0;trow<Lt;trow++){
	    double n = norm2(slice_sums[trow]);
	    std::cout << trow << " " << n << " expect " << (trow == tcol ? "non-zero" : "zero") << std::endl; 
	    if(trow == tcol) assert(n > 1e-8);
	    else assert(n < 1e-8);
	  }
	
	  //Check color delta
	  std::cout << "Check color delta" << std::endl;
	  for(int crow=0;crow<3;crow++){
	    auto v = Grid::PeekIndex<ColourIndex>(tmp,crow);
	    double n = norm2(v);
	    std::cout << crow << " " << n << " expect " << (crow == ccol ? "non-zero" : "zero") << std::endl;
	    if(crow == ccol) assert(n > 1e-8);
	    else assert(n < 1e-8);
	  }

	  //Check source is X-conjugate
	  Grid::Gamma C = Grid::Gamma(Grid::Gamma::Algebra::MinusGammaY) * Grid::Gamma(Grid::Gamma::Algebra::GammaT);
	  Grid::Gamma g5 = Grid::Gamma(Grid::Gamma::Algebra::Gamma5);
	  Grid::Gamma X = C*g5;
	  
	  auto field_f0 = Grid::PeekIndex<GparityFlavourIndex>(tmp,0);
	  auto field_f1 = Grid::PeekIndex<GparityFlavourIndex>(tmp,1);
	  decltype(field_f0) tmp_grid(field_f0.Grid());
	  tmp_grid = -(X*conjugate(field_f0));
	  tmp_grid = field_f1 - tmp_grid;
	  double n = norm2(tmp_grid);
	  std::cout << "Check source is X-conjugate " << n << " expect 0" << std::endl;
	  assert(n < 1e-8);
	}
      }
    }
  }

  std::cout << "testXconjWsrc passed" << std::endl;
}



template<typename GridFermionFieldD>
class A2AhighModeComputeDoNothing: public A2AhighModeCompute<GridFermionFieldD>{
public:
  Grid::GridBase* grid;  

  A2AhighModeComputeDoNothing(Grid::GridBase* grid): grid(grid){}

  Grid::GridBase* get4Dgrid() const override{return grid; }

  void highModeContribution4D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in, 
			      const EvecInterface<GridFermionFieldD> &evecs, const int nl) const override{
    out.resize(in.size(),grid);
    for(int i=0;i<in.size();i++) out[i]=in[i];
  }
};

template<typename A2Apolicies>
void testXconjWsrcPostOp(typename A2Apolicies::FgridGFclass &lat){
  std::cout << "Starting testXconjWsrcPostOp" << std::endl;
  assert(GJP.Gparity());

  int nl = 10;
  A2AArg a2a_args;
  a2a_args.nl = nl;
  a2a_args.nhits = 1;
  a2a_args.rand_type = UONE;
  a2a_args.src_width = 1;

  typedef typename A2AvectorW<A2Apolicies>::FieldInputParamType FieldParams;  
  setupFieldParams2<A2Apolicies, typename ComplexClassify<typename A2Apolicies::ComplexType>::type> p;

  A2AhighModeSourceXconj<A2Apolicies> Wsrc_impl;
  A2AvectorW<A2Apolicies> W(a2a_args, p.params);
  Wsrc_impl.setHighModeSources(W);

  LancArg lanc_arg;
  lanc_arg.N_get=nl;
  lanc_arg.N_use=nl+4;
  lanc_arg.N_true_get=nl;
  lanc_arg.precon=1;

  GridLanczosDoubleConvSingle<A2Apolicies> evecs;
  evecs.randomizeEvecs(lanc_arg,lat);

  auto iface = evecs.createInterface();
  
  A2AhighModeComputeDoNothing<typename A2Apolicies::GridFermionField> highmode_donothin(lat.getUGrid());
  
  A2AvectorV<A2Apolicies> V(a2a_args, p.params);

  computeVWhigh(V,W,Wsrc_impl,*iface,highmode_donothin,1);

  //Because we didn't apply the inverse, the solutions should have the form
  //V =  (rho , 0)    H   H^dag   = (rho , 0)
  //     ( 0,  rho*)                ( 0 , rho*)  

  typedef typename A2Apolicies::GridFermionField GridField;
  GridField Vf(lat.getUGrid()), Wf(lat.getUGrid());
  typedef typename A2AvectorW<A2Apolicies>::FermionFieldType CPSFieldType;
  CPSFieldType Wf_cps(W.getFieldInputParams());

  typedef decltype( Grid::PeekIndex<GparityFlavourIndex>(Vf,0) ) GridFlavourField;
  GridFlavourField src_00(Vf.Grid()), src_10(Vf.Grid());

  //Check Kronecker deltas
  StandardIndexDilution dil(a2a_args);
  int Lt = GJP.TnodeSites()*GJP.Tnodes();

  int hit = 0;
  for(int tcol=0;tcol<GJP.TnodeSites()*GJP.Tnodes();tcol++){
    for(int scol=0;scol<4;scol++){
      for(int ccol=0;ccol<3;ccol++){
	for(int fcol=0;fcol<2;fcol++){	
	  std::cout << "fcol:" << fcol << " ccol:" << ccol << " scol:" << scol << " tcol:" << tcol << std::endl;

	  int id = dil.indexMap(hit,tcol,ccol+3*scol,fcol);
	  Vf = Grid::Zero();
	  V.getMode(nl + id).exportGridField(Vf);

	  W.getDilutedSource(Wf_cps,id);
	  Wf_cps.exportGridField(Wf);

	  //Check same as original diluted source
	  double n = norm2(GridField(Vf-Wf));
	  std::cout << "Check V=W : " << n << " expect 0" << std::endl;

	  double n2 = norm2(GridField(Vf-conjugate(Wf)));
	  std::cout << "Check V!=W* : " << n2 << " expect !0" << std::endl;

	  assert(n<1e-8);
	  assert(n2>1e-8);

	  typedef typename GridField::scalar_object sobj;
	  std::vector<sobj> slice_sums;
	  Grid::sliceSum(Vf,slice_sums,3);

	  std::cout << "Check temporal delta" << std::endl;
	  assert(slice_sums.size() == Lt);
	  for(int trow=0;trow<Lt;trow++){
	    double n = norm2(slice_sums[trow]);
	    std::cout << trow << " " << n << " expect " << (trow == tcol ? "non-zero" : "zero") << std::endl; 
	    if(trow == tcol) assert(n > 1e-8);
	    else assert(n < 1e-8);
	  }
	
	  //Check color delta
	  std::cout << "Check color delta" << std::endl;
	  for(int crow=0;crow<3;crow++){
	    auto v = Grid::PeekIndex<ColourIndex>(Vf,crow);
	    double n = norm2(v);
	    std::cout << crow << " " << n << " expect " << (crow == ccol ? "non-zero" : "zero") << std::endl;
	    if(crow == ccol) assert(n > 1e-8);
	    else assert(n < 1e-8);
	  }

	  //Check spin delta
	  std::cout << "Check spin delta" << std::endl;
	  for(int srow=0;srow<4;srow++){
	    auto v = Grid::PeekIndex<SpinIndex>(Vf,srow);
	    double n = norm2(v);
	    std::cout << srow << " " << n << " expect " << (srow == scol ? "non-zero" : "zero") << std::endl;
	    if(srow == scol) assert(n > 1e-8);
	    else assert(n < 1e-8);
	  }
	  
	  //Check flavor delta
	  std::cout << "Check flavor delta" << std::endl;
	  for(int frow=0;frow<2;frow++){
	    auto v = Grid::PeekIndex<GparityFlavourIndex>(Vf,frow);
	    double n = norm2(v);
	    std::cout << frow << " " << n << " expect " << (frow == fcol ? "non-zero" : "zero") << std::endl;
	    if(frow == fcol) assert(n > 1e-8);
	    else assert(n < 1e-8);

	    if(fcol == 0)
	      if(frow==0) src_00 = v;
	      else src_10 = v;
	  }
	  
	  //Check flavor structure
	  if(fcol == 1){	    
	    auto src_01 = Grid::PeekIndex<GparityFlavourIndex>(Vf,0);
	    auto src_11 = Grid::PeekIndex<GparityFlavourIndex>(Vf,1);

	    double n01 = norm2(src_01);
	    double n10 = norm2(src_10);

	    std::cout << "Check flavor structure: off diagonal " << n01 << " " << n10 << " expect 0 for both" << std::endl;
	    assert(n01<1e-8 && n10 < 1e-8);
	    
	    GridFlavourField diff = src_00 - conjugate(src_11);
	    double nd = norm2(diff);
	    std::cout << "Check flavor structure: diagonals are complex conjugate pair " << norm2(src_00) << " " << norm2(src_11) << " norm conj-diff " << nd << " expect 0" << std::endl;
	    assert(nd < 1e-8);
	  }

	}//fcol
      }//ccol
    }//scol
  }//tcol
  
  //Should get the same result if we apply the *nothing* operation to the unrotated source
  A2AhighModeSourceOriginal<A2Apolicies> Wsrc_impl_unrot;
  A2AvectorV<A2Apolicies> V_unrot(a2a_args, p.params);

  computeVWhigh(V_unrot,W,Wsrc_impl_unrot,*iface,highmode_donothin,1);
  //Compare the V vectors
  int N =  V_unrot.getNmodes();
  assert(V.getNmodes() == N);

  StandardIndexDilution stdidx(W.getArgs());  
  int dhit,dtblock,dspin_color,dflavor,dspin,dcolor;

  for(int i=0;i<N;i++){
    auto const & mode_rot = V.getMode(i);
    auto const & mode_unrot = V_unrot.getMode(i);
    
    if(i<a2a_args.nl){
      std::cout << "Testing low mode " << i << std::endl;
    }else{
      int h = i - a2a_args.nl;
      stdidx.indexUnmap(h,dhit,dtblock,dspin_color,dflavor);  
      dspin = dspin_color / 3; //c+3*s
      dcolor = dspin_color % 3;
      std::cout << "Testing high mode " << h << ": hit=" << dhit << " tblock=" << dtblock << " spin=" << dspin << " color=" << dcolor << " flavor=" << dflavor << std::endl;
    }
    assert( mode_rot.equals(mode_unrot,1e-5,true));
  }

  std::cout << "testXconjWsrcPostOp passed" << std::endl;
}


//Inverting upon the 2f 5D X-conj vectors should be achievable with either the Xconj or original Gparity Dirac operator
template<typename A2Apolicies>
void testXconjWsrcInverse(typename A2Apolicies::FgridGFclass &lattice){
  std::cout << "Starting testXconjWsrcInverse" << std::endl;
  assert(GJP.Gparity());

  typedef typename A2Apolicies::GridFermionField Field2f;
  typedef typename A2Apolicies::GridXconjFermionField Field1f;

  typedef typename A2Apolicies::GridFermionFieldF Field2fF;
  typedef typename A2Apolicies::GridXconjFermionFieldF Field1fF;

  typedef typename A2Apolicies::GridDirac Dirac2f;
  typedef typename A2Apolicies::GridDiracXconj Dirac1f;
  
  Grid::GridCartesian *UGrid = lattice.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();
  Grid::GridCartesian *FGrid = lattice.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();
  Grid::GridRedBlackCartesian *FrbGridF = lattice.getFrbGridF();
  Grid::LatticeGaugeFieldD *Umu = lattice.getUmu();

  double b = lattice.get_mob_b();
  double c = b - 1.;   //b-c = 1
  double M5 = GJP.DwfHeight();

  double mass = 0.01;

  typename Dirac2f::ImplParams params_gp;
  lattice.SetParams(params_gp);

  typename Dirac1f::ImplParams params_x;
  params_x.twists = params_gp.twists;
  params_x.boundary_phase = 1.0;

  Dirac1f actionX(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c, params_x);
  Dirac2f actionGP(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c, params_gp);

  A2ASchurOriginalOperatorImpl<Dirac2f> SchurOpGP(actionGP);
  A2ASchurOriginalOperatorImpl<Dirac1f> SchurOpX(actionX);
  A2Ainverter5dCG<Field2f> inv5d_GP(SchurOpGP.getLinOp(),1e-8,10000);
  A2Ainverter5dCG<Field1f> inv5d_X(SchurOpX.getLinOp(),1e-8,10000);
  A2Ainverter5dXconjWrapper<Field2f> inv5d_Xwrap(inv5d_X,true);

  typedef typename A2AvectorW<A2Apolicies>::FieldInputParamType FieldParams;  
  setupFieldParams2<A2Apolicies, typename ComplexClassify<typename A2Apolicies::ComplexType>::type> p;

  A2AArg a2a_args;
  a2a_args.nl = 10;
  a2a_args.nhits = 1;
  a2a_args.rand_type = UONE;
  a2a_args.src_width = 1;

  A2AvectorW<A2Apolicies> W(a2a_args,p.params);
  A2AvectorV<A2Apolicies> VX(a2a_args,p.params);
  A2AvectorV<A2Apolicies> VGP(a2a_args,p.params);
  
  //Use some random eigenvectors for the low-mode contribution
  LancArg lanc_arg;
  lanc_arg.mass = 0.01;
  lanc_arg.stop_rsd = 1e-08;
  lanc_arg.N_true_get = 10;
  GridXconjLanczosDoubleConvSingle<A2Apolicies> lancGP; //we need eigenvectors that maintain the complex-conjugate symmetry of the Dirac operator (real G-parity evecs would also satisfy)
  lancGP.randomizeEvecs(lanc_arg,lattice);
  auto eveci_low = lancGP.createInterface();

  //Do the high modes the same way, but do the inversions in one case with the 2f operator and in the other case the 1f operator
  A2AhighModeComputeSchurPreconditioned<Dirac2f> highmode_compute_2f(SchurOpGP,inv5d_GP);
  A2AhighModeComputeSchurPreconditioned<Dirac2f> highmode_compute_1fwrap(SchurOpGP,inv5d_Xwrap); //wrapped 5D inverter

  //Set the high mode sources to the correct form
  A2AhighModeSourceXconj<A2Apolicies> Wsrc_impl;
  Wsrc_impl.setHighModeSources(W);
  
  //Compute with X-conjugate inverter
  computeVWhigh(VX,W,Wsrc_impl,*eveci_low,highmode_compute_1fwrap,1);

  //Compute with G-parity inverter
  computeVWhigh(VGP,W,Wsrc_impl,*eveci_low,highmode_compute_2f,1);

  //Compare the V vectors
  int N =  VGP.getNmodes();
  assert(VX.getNmodes() == N);
  for(int i=0;i<N;i++){
    auto const & mode_X = VX.getMode(i);
    auto const & mode_GP = VGP.getMode(i);
    
    std::cout << "Testing mode " << i << std::endl;
    assert( mode_X.equals(mode_GP,1e-5,true));
  }

  std::cout << "testXconjWsrcInverse passed" << std::endl;
}




//Test the complete new source implementation by inverting on the unmodified source with the GP operator
//and on the rotated source with the Xconj operator, rotating the results
template<typename A2Apolicies>
void testXconjWsrcFull(typename A2Apolicies::FgridGFclass &lattice){
  std::cout << "Starting testXconjWsrcFull" << std::endl;
  assert(GJP.Gparity());

  typedef typename A2Apolicies::GridFermionField Field2f;
  typedef typename A2Apolicies::GridXconjFermionField Field1f;

  typedef typename A2Apolicies::GridFermionFieldF Field2fF;
  typedef typename A2Apolicies::GridXconjFermionFieldF Field1fF;

  typedef typename A2Apolicies::GridDirac Dirac2f;
  typedef typename A2Apolicies::GridDiracXconj Dirac1f;
  
  Grid::GridCartesian *UGrid = lattice.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();
  Grid::GridCartesian *FGrid = lattice.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();
  Grid::GridRedBlackCartesian *FrbGridF = lattice.getFrbGridF();
  Grid::LatticeGaugeFieldD *Umu = lattice.getUmu();

  double b = lattice.get_mob_b();
  double c = b - 1.;   //b-c = 1
  double M5 = GJP.DwfHeight();

  double mass = 0.01;

  typename Dirac2f::ImplParams params_gp;
  lattice.SetParams(params_gp);

  typename Dirac1f::ImplParams params_x;
  params_x.twists = params_gp.twists;
  params_x.boundary_phase = 1.0;

  Dirac1f actionX(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c, params_x);
  Dirac2f actionGP(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c, params_gp);

  A2ASchurOriginalOperatorImpl<Dirac2f> SchurOpGP(actionGP);
  A2ASchurOriginalOperatorImpl<Dirac1f> SchurOpX(actionX);

  A2Ainverter5dCG<Field2f> inv5d_GP(SchurOpGP.getLinOp(),1e-8,10000);
  A2Ainverter5dCG<Field1f> inv5d_X(SchurOpX.getLinOp(),1e-8,10000);
  A2Ainverter5dXconjWrapper<Field2f> inv5d_Xwrap(inv5d_X,true);

  typedef typename A2AvectorW<A2Apolicies>::FieldInputParamType FieldParams;  
  setupFieldParams2<A2Apolicies, typename ComplexClassify<typename A2Apolicies::ComplexType>::type> p;

  A2AArg a2a_args;
  a2a_args.nl = 10;
  a2a_args.nhits = 1;
  a2a_args.rand_type = UONE;
  a2a_args.src_width = 1;

  A2AvectorW<A2Apolicies> W(a2a_args,p.params);
  A2AvectorV<A2Apolicies> VX(a2a_args,p.params);
  A2AvectorV<A2Apolicies> VGP(a2a_args,p.params);
  
  //Use some random eigenvectors for the low-mode contribution
  LancArg lanc_arg;
  lanc_arg.mass = 0.01;
  lanc_arg.stop_rsd = 1e-08;
  lanc_arg.N_true_get = 10;
  GridXconjLanczosDoubleConvSingle<A2Apolicies> lancGP; //we need eigenvectors that maintain the complex-conjugate symmetry of the Dirac operator (real G-parity evecs would also satisfy)
  lancGP.randomizeEvecs(lanc_arg,lattice);
  auto eveci_low = lancGP.createInterface();

  //Do the high modes the same way, but do the inversions in one case with the 2f operator and in the other case the 1f operator
  A2AhighModeComputeSchurPreconditioned<Dirac2f> highmode_compute_2f(SchurOpGP,inv5d_GP);
  A2AhighModeComputeSchurPreconditioned<Dirac2f> highmode_compute_1fwrap(SchurOpGP,inv5d_Xwrap); //wrapped 5D inverter

  //Set the high mode sources to the correct form
  A2AhighModeSourceXconj<A2Apolicies> Wsrc_impl;
  Wsrc_impl.setHighModeSources(W); //do it now so it won't be called again internally
  
  //Compute with X-conjugate inverter
  computeVWhigh(VX,W,Wsrc_impl,*eveci_low,highmode_compute_1fwrap,1);

  //Compute with G-parity inverter
  A2AhighModeSourceOriginal<A2Apolicies> Wsrc_impl_GP; //no rotation, inversion on unmodified source
  computeVWhigh(VGP,W,Wsrc_impl_GP,*eveci_low,highmode_compute_2f,1);

  //Compare the V vectors
  int N =  VGP.getNmodes();
  assert(VX.getNmodes() == N);

  StandardIndexDilution stdidx(W.getArgs());  
  int dhit,dtblock,dspin_color,dflavor,dspin,dcolor;

  for(int i=0;i<N;i++){
    auto const & mode_X = VX.getMode(i);
    auto const & mode_GP = VGP.getMode(i);
    
    if(i<a2a_args.nl){
      std::cout << "Testing low mode " << i << std::endl;
    }else{
      int h = i - a2a_args.nl;
      stdidx.indexUnmap(h,dhit,dtblock,dspin_color,dflavor);  
      dspin = dspin_color / 3; //c+3*s
      dcolor = dspin_color % 3;
      std::cout << "Testing high mode " << h << ": hit=" << dhit << " tblock=" << dtblock << " spin=" << dspin << " color=" << dcolor << " flavor=" << dflavor << std::endl;
    }
    assert( mode_X.equals(mode_GP,1e-5,true));
  }

  std::cout << "testXconjWsrcFull passed" << std::endl;
}




//Demonstrate the A2A propagator with the new W source form satisfies the complex conjugate relationship
//even for 1 hit
template<typename A2Apolicies>
void testXconjWsrcCConjReln(typename A2Apolicies::FgridGFclass &lattice){
  std::cout << "Starting testXconjWsrcCConjReln" << std::endl;
  assert(GJP.Gparity());

  typedef typename A2Apolicies::GridFermionField Field2f;
  typedef typename A2Apolicies::GridXconjFermionField Field1f;

  typedef typename A2Apolicies::GridFermionFieldF Field2fF;
  typedef typename A2Apolicies::GridXconjFermionFieldF Field1fF;

  typedef typename A2Apolicies::GridDirac Dirac2f;
  typedef typename A2Apolicies::GridDiracXconj Dirac1f;
  
  Grid::GridCartesian *UGrid = lattice.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();
  Grid::GridCartesian *FGrid = lattice.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();
  Grid::GridRedBlackCartesian *FrbGridF = lattice.getFrbGridF();
  Grid::LatticeGaugeFieldD *Umu = lattice.getUmu();

  double b = lattice.get_mob_b();
  double c = b - 1.;   //b-c = 1
  double M5 = GJP.DwfHeight();

  double mass = 0.01;

  typename Dirac2f::ImplParams params_gp;
  lattice.SetParams(params_gp);

  typename Dirac1f::ImplParams params_x;
  params_x.twists = params_gp.twists;
  params_x.boundary_phase = 1.0;

  Dirac1f actionX(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c, params_x);
  Dirac2f actionGP(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c, params_gp);

  A2ASchurOriginalOperatorImpl<Dirac2f> SchurOpGP(actionGP);
  A2ASchurOriginalOperatorImpl<Dirac1f> SchurOpX(actionX);

  //We don't do the initial deflation as the high-mode compute implementation used here will set the 
  A2Ainverter5dCG<Field1f> inv5d_X(SchurOpX.getLinOp(),1e-8,10000);
  A2Ainverter5dXconjWrapper<Field2f> inv5d_Xwrap(inv5d_X,true);

  typedef typename A2AvectorW<A2Apolicies>::FieldInputParamType FieldParams;  
  setupFieldParams2<A2Apolicies, typename ComplexClassify<typename A2Apolicies::ComplexType>::type> p;

  A2AArg a2a_args;
  a2a_args.nl = 10;
  a2a_args.nhits = 1;
  a2a_args.rand_type = UONE;
  a2a_args.src_width = 1;

  A2AvectorW<A2Apolicies> W(a2a_args,p.params);
  A2AvectorV<A2Apolicies> V(a2a_args,p.params);
  
  //Use some random eigenvectors for the low-mode contribution
  LancArg lanc_arg;
  lanc_arg.mass = 0.01;
  lanc_arg.stop_rsd = 1e-08;
  lanc_arg.N_true_get = 10;
  GridXconjLanczosDoubleConvSingle<A2Apolicies> lancGP; //we need eigenvectors that maintain the complex-conjugate symmetry of the Dirac operator (real G-parity evecs would also satisfy)
  lancGP.randomizeEvecs(lanc_arg,lattice);
  auto eveci_low = lancGP.createInterface();

  A2AhighModeComputeSchurPreconditioned<Dirac2f> highmode_compute_1fwrap(SchurOpGP,inv5d_Xwrap); //wrapped 5D inverter
  A2AhighModeSourceXconj<A2Apolicies> Wsrc_impl;
  computeVWhigh(V,W,Wsrc_impl,*eveci_low,highmode_compute_1fwrap,1);
  
  typedef typename A2Apolicies::ComplexType VectorComplexType;
  typedef CPSspinColorFlavorMatrix<VectorComplexType> SCFmat;
  typedef CPSmatrixField<SCFmat> SCFmatrixField;

  typedef typename A2AvectorV<A2Apolicies>::FermionFieldType CPSfermionField;

  SCFmatrixField prop(p.params);
  StandardIndexDilution stdidx(a2a_args);

  for(int hit=0;hit<stdidx.getNhits();hit++){
    for(int tcol=0;tcol<stdidx.getNtBlocks();tcol++){
      
      for(int scol=0;scol<4;scol++){
	for(int ccol=0;ccol<3;ccol++){
	  for(int fcol=0;fcol<2;fcol++){
	    const CPSfermionField & vv = V.getMode(a2a_args.nl + stdidx.indexMap(hit,tcol,ccol+3*scol,fcol) );
	    
#pragma omp parallel for
	    for(size_t x=0;x<vv.nsites();x++){
	      SCFmat &into = *prop.site_ptr(x);
	      for(int frow=0;frow<2;frow++){
		VectorComplexType const* from_base = vv.site_ptr(x,frow);
		for(int srow=0;srow<4;srow++)
		  for(int crow=0;crow<3;crow++)
		    into(srow,scol)(crow,ccol)(frow,fcol) = *(from_base + crow + 3*srow);
	      }
	    }
	  }
	}
      }
      
      SCFmatrixField propconj = cconj(prop);
      //Xi = -i s2 X
      //X = C g5
      //C=-gY gT = gT gY
      //CConj rel:   M = Xi M* Xi
      
      gr(propconj,3);
      gr(propconj,1);
      gr(propconj,-5);
      pr(propconj,sigma2);
      
      gl(propconj,-5);
      gl(propconj,1);
      gl(propconj,3);
      pl(propconj,sigma2);

      timesMinusOne(propconj);
      
      SCFmatrixField diff = prop - propconj;

      std::cout << "Check propagator cconj reln hit=" << hit << " tcol=" << tcol << " result=" << CPSmatrixFieldNorm2(diff) << " (expect 0)" << std::endl;

    }//tcol
  }//hit

  std::cout << "testXconjWsrcCConjReln passed" << std::endl;
}



CPS_END_NAMESPACE
