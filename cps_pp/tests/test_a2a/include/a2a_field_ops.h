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


CPS_END_NAMESPACE
