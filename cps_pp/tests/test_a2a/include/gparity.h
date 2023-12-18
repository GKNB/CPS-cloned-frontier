#pragma once

CPS_START_NAMESPACE

template<typename A2Apolicies>
void testA2AvectorFFTrelnGparity(const A2AArg &a2a_args,Lattice &lat){
  assert(GJP.Gparity());

  if(lat.FixGaugeKind() == FIX_GAUGE_NONE){
    FixGaugeArg fix_gauge_arg;
    fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
    fix_gauge_arg.hyperplane_start = 0;
    fix_gauge_arg.hyperplane_step = 1;
    fix_gauge_arg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites();
    fix_gauge_arg.stop_cond = 1e-08;
    fix_gauge_arg.max_iter_num = 10000;

    CommonArg common_arg;
  
    AlgFixGauge fix_gauge(lat,&common_arg,&fix_gauge_arg);
    fix_gauge.run();
  }
  
  //Demonstrate relation between FFTW fields
  typedef typename A2Apolicies::ComplexType mf_Complex;
  typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);
  
  A2AvectorW<A2Apolicies> W(a2a_args,fp);
  W.testRandom();

  int p_p1[3];
  GparityBaseMomentum(p_p1,+1);

  int p_m1[3];
  GparityBaseMomentum(p_m1,-1);

  //Perform base FFTs
  //twist<typename A2Apolicies::FermionFieldType> twist_p1(p_p1);
  //twist<typename A2Apolicies::FermionFieldType> twist_m1(p_m1);

  gaugeFixAndTwist<typename A2Apolicies::FermionFieldType> twist_p1(p_p1,lat);
  gaugeFixAndTwist<typename A2Apolicies::FermionFieldType> twist_m1(p_m1,lat);
  
  A2AvectorWfftw<A2Apolicies> Wfftw_p1(a2a_args,fp);
  Wfftw_p1.fft(W,&twist_p1);

  A2AvectorWfftw<A2Apolicies> Wfftw_m1(a2a_args,fp);
  Wfftw_m1.fft(W,&twist_m1);


  int p[3];  
  A2AvectorWfftw<A2Apolicies> result(a2a_args,fp);
  A2AvectorWfftw<A2Apolicies> compare(a2a_args,fp);
  
  //Get twist for first excited momentum in p1 set
  {
    memcpy(p,p_p1,3*sizeof(int));
    p[0] = 5;
    //twist<typename A2Apolicies::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies::FermionFieldType> twist_p(p,lat);    
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in x direction\n",p[0],p[1],p[2]);
   
    for(int i=0;i<compare.getNmodes();i++){
      if(!UniqueID()) printf("Compare mode %d\n", i);
      printRow(result.getMode(i),0,  "Result ");
      printRow(compare.getMode(i),0, "Compare");
      assert( compare.getMode(i).equals( result.getMode(i), 1e-7, true ) );
    }
  }
  //Get twist for first negative excited momentum in p1 set
  if(GJP.Bc(1) == BND_CND_GPARITY){
    memcpy(p,p_p1,3*sizeof(int));
    p[1] = -3;
    //twist<typename A2Apolicies::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies::FermionFieldType> twist_p(p,lat);    
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in y direction\n",p[0],p[1],p[2]);
   
    for(int i=0;i<compare.getNmodes();i++){
      if(!UniqueID()) printf("Compare mode %d\n", i);
      printRow(result.getMode(i),0,  "Result ");
      printRow(compare.getMode(i),0, "Compare");
      assert( compare.getMode(i).equals( result.getMode(i), 1e-7, true ) );
    }
  }
  //Try two directions
  if(GJP.Bc(1) == BND_CND_GPARITY){
    memcpy(p,p_p1,3*sizeof(int));
    p[0] = -3;
    p[1] = -3;
    //twist<typename A2Apolicies::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies::FermionFieldType> twist_p(p,lat);    
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in y direction\n",p[0],p[1],p[2]);
   
    for(int i=0;i<compare.getNmodes();i++){
      if(!UniqueID()) printf("Compare mode %d\n", i);
      printRow(result.getMode(i),0,  "Result ");
      printRow(compare.getMode(i),0, "Compare");
      assert( compare.getMode(i).equals( result.getMode(i), 1e-7, true ) );
    }
  }
  //Try 3 directions
  if(GJP.Bc(1) == BND_CND_GPARITY && GJP.Bc(2) == BND_CND_GPARITY){
    memcpy(p,p_p1,3*sizeof(int));
    p[0] = -3;
    p[1] = -3;
    p[2] = -3;
    //twist<typename A2Apolicies::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies::FermionFieldType> twist_p(p,lat);    
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in y direction\n",p[0],p[1],p[2]);
  
    for(int i=0;i<compare.getNmodes();i++){
      if(!UniqueID()) printf("Compare mode %d\n", i);
      printRow(result.getMode(i),0,  "Result ");
      printRow(compare.getMode(i),0, "Compare");
      assert( compare.getMode(i).equals( result.getMode(i), 1e-7, true ) );
    }
  }
  //Get twist for first excited momentum in m1 set
  {
    memcpy(p,p_m1,3*sizeof(int));
    p[0] = 3;
    //twist<typename A2Apolicies::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies::FermionFieldType> twist_p(p,lat);   
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in x direction\n",p[0],p[1],p[2]);

    for(int i=0;i<compare.getNmodes();i++){
      if(!UniqueID()) printf("Compare mode %d\n", i);
      printRow(result.getMode(i),0,  "Result ");
      printRow(compare.getMode(i),0, "Compare");
      assert( compare.getMode(i).equals( result.getMode(i), 1e-7, true ) );
    }
  }
  
}


template<typename A2Apolicies>
void testMfFFTreln(const A2AArg &a2a_args,Lattice &lat){
  assert(GJP.Gparity());

  if(lat.FixGaugeKind() == FIX_GAUGE_NONE){
    FixGaugeArg fix_gauge_arg;
    fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
    fix_gauge_arg.hyperplane_start = 0;
    fix_gauge_arg.hyperplane_step = 1;
    fix_gauge_arg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites();
    fix_gauge_arg.stop_cond = 1e-08;
    fix_gauge_arg.max_iter_num = 10000;

    CommonArg common_arg;
  
    AlgFixGauge fix_gauge(lat,&common_arg,&fix_gauge_arg);
    fix_gauge.run();
  }
  typedef typename A2Apolicies::ComplexType mf_Complex;
  typedef typename A2Apolicies::SourcePolicies SourcePolicies;
  
  typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);

  typedef typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType SrcInputParamType;
  SrcInputParamType sp; defaultFieldParams<SrcInputParamType, mf_Complex>::get(sp);

  A2AvectorW<A2Apolicies> W(a2a_args,fp);
  A2AvectorV<A2Apolicies> V(a2a_args,fp);
  W.testRandom();
  V.testRandom();

  int pp[3]; GparityBaseMomentum(pp,+1); //(1,1,1)
  int pm[3]; GparityBaseMomentum(pm,-1); //(-1,-1,-1)

  //M_ij^{4a+k,4b+l} =  \sum_{n=0}^{L-1} \Omega^{\dagger,4a+k}_i(n) \Gamma \gamma(n) N^{4b+l}_j(n)     (1)
  //                    \sum_{n=0}^{L-1} \Omega^{\dagger,k}_i(n-a-b) \Gamma \gamma(n-b) N^l_j(n)         (2)
  
  //\Omega^{\dagger,k}_i(n) = [ \sum_{x=0}^{L-1} e^{-2\pi i nx/L} e^{- (-k) \pi ix/2L} W_i(x) ]^\dagger
  //N^l_j(n) = \sum_{x=0}^{L-1} e^{-2\pi ix/L} e^{-l \pi ix/2L} V_i(x)

  //Use a state with total momentum 0; k=1 l=-1 a=-1 b=1  so total momentum  -3 + 3  = 0

  int a = -1;
  int b = 1;
  int k = 1;
  int l = -1;

  assert(a+b == 0); //don't want to permute W V right now
  
  //For (1) 
  int p1w[3] = { -(4*a+k), pm[1],pm[1] };  //fix other momenta to first allowed
  int p1v[3] = { 4*b+l, pm[1],pm[1] };
  
  //For (2)
  int p2w[3] = {-k, pm[1],pm[1]};
  int p2v[3] = {l, pm[1],pm[1]};
  typedef A2AvectorV<A2Apolicies> Vtype;
  typedef A2AvectorW<A2Apolicies> Wtype;
  typedef A2AflavorProjectedExpSource<SourcePolicies> SrcType;
  typedef SCFspinflavorInnerProduct<0,mf_Complex,SrcType,true,false> InnerType; //unit matrix spin structure
  typedef GparityFlavorProjectedBasicSourceStorage<Vtype,Wtype, InnerType> StorageType;

  SrcType src1(2., pp, sp);
  SrcType src2(2., pp, sp);
  cyclicPermute( src2.getSource(), src2.getSource(), 0, 1, b);

  InnerType inner1(sigma0,src1);
  InnerType inner2(sigma0,src2);
  StorageType mf_store1(inner1);
  StorageType mf_store2(inner2);

  mf_store1.addCompute(0,0, ThreeMomentum(p1w), ThreeMomentum(p1v) );
  mf_store1.addCompute(0,0, ThreeMomentum(p2w), ThreeMomentum(p2v) );
  
  mf_store2.addCompute(0,0, ThreeMomentum(p2w), ThreeMomentum(p2v) );

  typename ComputeMesonFields<Vtype,Wtype,StorageType>::WspeciesVector Wspecies(1, &W);
  typename ComputeMesonFields<Vtype,Wtype,StorageType>::VspeciesVector Vspecies(1, &V);

  ComputeMesonFields<Vtype,Wtype,StorageType>::compute(mf_store1,Wspecies,Vspecies,lat);
  ComputeMesonFields<Vtype,Wtype,StorageType>::compute(mf_store2,Wspecies,Vspecies,lat);

  printf("Testing mf relation\n"); fflush(stdout);
  assert( mf_store1[0][0].equals( mf_store2[0][0], 1e-6, true) );
  printf("MF Relation proven\n");

  // StorageType mf_store3(inner1);
  // mf_store3.addCompute(0,0, ThreeMomentum(p1w), ThreeMomentum(p1v), true );
#if 1
  
  typedef GparitySourceShiftInnerProduct<mf_Complex,SrcType,flavorMatrixSpinColorContract<0,true,false> > ShiftInnerType;
  typedef GparityFlavorProjectedShiftSourceStorage<Vtype,Wtype, ShiftInnerType> ShiftStorageType;
  
  SrcType src3(2., pp, sp);
  ShiftInnerType shift_inner(sigma0,src3);
  ShiftStorageType mf_shift_store(shift_inner,src3);
  mf_shift_store.addCompute(0,0, ThreeMomentum(p1w), ThreeMomentum(p1v) );
  mf_shift_store.addCompute(0,0, ThreeMomentum(p2w), ThreeMomentum(p2v) );
  int nc = mf_shift_store.nCompute();
  printf("Number of optimized computations: %d\n",nc);

  ComputeMesonFields<Vtype,Wtype,ShiftStorageType>::compute(mf_shift_store,Wspecies,Vspecies,lat);

  assert( mf_shift_store[0][0].equals( mf_store1[0][0], 1e-6, true) );
  assert( mf_shift_store[1][0].equals( mf_store1[1][0], 1e-6, true) );
  printf("Passed test of shift storage for single source type\n");

  typedef Elem<SrcType, Elem<SrcType,ListEnd > > SrcList;
  typedef A2AmultiSource<SrcList> MultiSrcType;
  typedef GparitySourceShiftInnerProduct<mf_Complex,MultiSrcType,flavorMatrixSpinColorContract<0,true,false> > ShiftMultiSrcInnerType;
  typedef GparityFlavorProjectedShiftSourceStorage<Vtype,Wtype, ShiftMultiSrcInnerType> ShiftMultiSrcStorageType;

  MultiSrcType multisrc;
  multisrc.template getSource<0>().setup(3.,pp, sp);
  multisrc.template getSource<1>().setup(2.,pp, sp);
  ShiftMultiSrcInnerType shift_inner_multisrc(sigma0,multisrc);
  ShiftMultiSrcStorageType mf_shift_multisrc_store(shift_inner_multisrc, multisrc);
  mf_shift_multisrc_store.addCompute(0,0, ThreeMomentum(p1w), ThreeMomentum(p1v) );
  mf_shift_multisrc_store.addCompute(0,0, ThreeMomentum(p2w), ThreeMomentum(p2v) );
  
  ComputeMesonFields<Vtype,Wtype,ShiftMultiSrcStorageType>::compute(mf_shift_multisrc_store,Wspecies,Vspecies,lat);

  assert( mf_shift_multisrc_store(1,0)[0].equals( mf_store1[0][0], 1e-6, true) );
  assert( mf_shift_multisrc_store(1,1)[0].equals( mf_store1[1][0], 1e-6, true) );
  
  
#endif
}


CPS_END_NAMESPACE
