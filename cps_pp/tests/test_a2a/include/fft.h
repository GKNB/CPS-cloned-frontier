#pragma once

CPS_START_NAMESPACE

template<typename GridA2Apolicies>
void testGridGetTwistedFFT(const A2AArg &a2a_args, const int nthreads, const double tol){
#ifdef USE_GRID
  std::cout << "Starting testGridGetTwistedFFT\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorWfftw<GridA2Apolicies> W1(a2a_args, simd_dims);
  A2AvectorWfftw<GridA2Apolicies> W2(a2a_args, simd_dims);
  
  W1.testRandom();
  W2.testRandom();
  
  int pbase[3];
  for(int i=0;i<3;i++) pbase[i] = GJP.Bc(i) == BND_CND_GPARITY ? 1 : 0;

  ThreeMomentum pp(pbase);
  ThreeMomentum pm = -pp;

  A2AvectorWfftw<GridA2Apolicies> Wtmp(a2a_args, simd_dims);

  //Does copy work?
  std::cout << "Testing mode copy" << std::endl;
  for(int i=0;i<W1.getNmodes();i++)
    Wtmp.getMode(i) = W1.getMode(i);
      
  for(int i=0;i<W1.getNmodes();i++){
    assert(Wtmp.getMode(i).equals(W1.getMode(i),1e-12,true));
  }
 
  std::cout << "Testing copy" << std::endl;
  Wtmp = W1;
  for(int i=0;i<W1.getNmodes();i++){
    assert(Wtmp.getMode(i).equals(W1.getMode(i),1e-12,true));
  }

  std::cout << "Testing getTwistedFFT" << std::endl;
  
  Wtmp.getTwistedFFT(pp.ptr(), &W1, &W2);  //base_p, base_m

  //Should be same as W1 without a shift
  for(int i=0;i<W1.getNmodes();i++){
    assert(Wtmp.getMode(i).equals(W1.getMode(i),1e-12,true));
  }
  std::cout << "testGridGetTwistedFFT passed" << std::endl;
#endif
}


template<typename A2Apolicies>
void testFFTopt(){
  typedef typename A2Apolicies::FermionFieldType::FieldSiteType mf_Complex;
  typedef typename A2Apolicies::FermionFieldType::FieldMappingPolicy MappingPolicy;
  typedef typename A2Apolicies::FermionFieldType::FieldAllocPolicy AllocPolicy;

  typedef typename MappingPolicy::template Rebase<OneFlavorPolicy>::type OneFlavorMap;
  
  typedef CPSfield<mf_Complex,12,OneFlavorMap, AllocPolicy> FieldType;
  typedef typename FieldType::InputParamType FieldInputParamType;
  FieldInputParamType fp; setupFieldParams<FieldType>(fp);

  {
    bool do_dirs[4] = {1,1,0,0};
    
    FieldType in(fp);
    in.testRandom();

    FieldType out1(fp);
    if(!UniqueID()) printf("FFT orig\n");
    fft(out1,in,do_dirs);

    FieldType out2(fp);
    if(!UniqueID()) printf("FFT opt\n");
    fft_opt(out2,in,do_dirs);

    assert( out1.equals(out2, 1e-8, true ) );
    printf("Passed FFT test\n");

    //Test inverse
    FieldType inv(fp);
    if(!UniqueID()) printf("FFT opt inverse\n");
    fft_opt(inv,out2,do_dirs,true);

    assert( inv.equals(in, 1e-8, true ) );
    printf("Passed FFT inverse test\n");  
  }

  { //test it works a second time! (plans are persistent)
    bool do_dirs[4] = {0,1,1,0};
    
    FieldType in(fp);
    in.testRandom();

    FieldType out1(fp);
    if(!UniqueID()) printf("FFT orig (2)\n");
    fft(out1,in,do_dirs);

    FieldType out2(fp);
    if(!UniqueID()) printf("FFT opt (2)\n");
    fft_opt(out2,in,do_dirs);

    assert( out1.equals(out2, 1e-8, true ) );
    printf("Passed FFT test\n");

    //Test inverse
    FieldType inv(fp);
    if(!UniqueID()) printf("FFT opt inverse (2)\n");
    fft_opt(inv,out2,do_dirs,true);

    assert( inv.equals(in, 1e-8, true ) );
    printf("Passed FFT inverse test (2)\n");  
  }

}

template<typename A2Apolicies>
void demonstrateFFTreln(const A2AArg &a2a_args){
  //Demonstrate relation between FFTW fields
  A2AvectorW<A2Apolicies> W(a2a_args);
  A2AvectorV<A2Apolicies> V(a2a_args);
  W.testRandom();
  V.testRandom();

  int p1[3] = {1,1,1};
  int p5[3] = {5,1,1};

  twist<typename A2Apolicies::FermionFieldType> twist_p1(p1);
  twist<typename A2Apolicies::FermionFieldType> twist_p5(p5);
    
  A2AvectorVfftw<A2Apolicies> Vfftw_p1(a2a_args);
  Vfftw_p1.fft(V,&twist_p1);

  A2AvectorVfftw<A2Apolicies> Vfftw_p5(a2a_args);
  Vfftw_p5.fft(V,&twist_p5);

  //f5(n) = f1(n+1)
  for(int i=0;i<Vfftw_p1.getNmodes();i++)
    cyclicPermute(Vfftw_p1.getMode(i), Vfftw_p1.getMode(i), 0, -1, 1);
    
  printRow(Vfftw_p1.getMode(0),0, "T_-1 V(p1) T_-1");
  printRow(Vfftw_p5.getMode(0),0, "V(p5)          ");

  for(int i=0;i<Vfftw_p1.getNmodes();i++)
    assert( Vfftw_p1.getMode(i).equals( Vfftw_p5.getMode(i), 1e-7, true ) );

  A2AvectorWfftw<A2Apolicies> Wfftw_p1(a2a_args);
  Wfftw_p1.fft(W,&twist_p1);

  A2AvectorWfftw<A2Apolicies> Wfftw_p5(a2a_args);
  Wfftw_p5.fft(W,&twist_p5);

  for(int i=0;i<Wfftw_p1.getNmodes();i++)
    cyclicPermute(Wfftw_p1.getMode(i), Wfftw_p1.getMode(i), 0, -1, 1);

  printRow(Wfftw_p1.getMode(0),0, "T_-1 W(p1) T_-1");
  printRow(Wfftw_p5.getMode(0),0, "W(p5)          ");

  for(int i=0;i<Wfftw_p1.getNmodes();i++)
    assert( Wfftw_p1.getMode(i).equals( Wfftw_p5.getMode(i), 1e-7, true ) );

  if(!UniqueID()) printf("Passed FFT relation test\n");
}


template<typename A2Apolicies>
void testA2AFFTinv(const A2AArg &a2a_args,Lattice &lat){
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
  
  A2AvectorVfftw<A2Apolicies> Vfft(a2a_args,fp);
  Vfft.fft(V);

  A2AvectorV<A2Apolicies> Vrec(a2a_args,fp);
  Vfft.inversefft(Vrec);

  for(int i=0;i<V.getNmodes();i++){
    assert( Vrec.getMode(i).equals( V.getMode(i), 1e-08, true) ); 
  }
  if(!UniqueID()) printf("Passed V fft/inverse test\n");

  A2AvectorWfftw<A2Apolicies> Wfft(a2a_args,fp);
  Wfft.fft(W);

  A2AvectorW<A2Apolicies> Wrec(a2a_args,fp);
  Wfft.inversefft(Wrec);

  for(int i=0;i<W.getNl();i++){
    assert( Wrec.getWl(i).equals( W.getWl(i), 1e-08, true) ); 
  }
  if(!UniqueID()) printf("Passed Wl fft/inverse test\n"); 

  for(int i=0;i<W.getNhits();i++){
    assert( Wrec.getWh(i).equals( W.getWh(i), 1e-08, true) ); 
  }
  if(!UniqueID()) printf("Passed Wh fft/inverse test\n"); 
}


template<typename ManualAllocA2Apolicies>
void testDestructiveFFT(const A2AArg &a2a_args,Lattice &lat){
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
  typedef typename ManualAllocA2Apolicies::FermionFieldType FermionFieldType;
  typedef typename ManualAllocA2Apolicies::SourcePolicies SourcePolicies;
  typedef typename ManualAllocA2Apolicies::ComplexType mf_Complex;
  
  typedef typename A2AvectorWfftw<ManualAllocA2Apolicies>::FieldInputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);

  typedef typename ManualAllocA2Apolicies::SourcePolicies::MappingPolicy::ParamType SrcInputParamType;
  SrcInputParamType sp; defaultFieldParams<SrcInputParamType, mf_Complex>::get(sp);

  A2AvectorW<ManualAllocA2Apolicies> W(a2a_args,fp);
  A2AvectorV<ManualAllocA2Apolicies> V(a2a_args,fp);
  
  for(int i=0;i<V.getNmodes();i++) assert( &V.getMode(i) == NULL);
  V.allocModes();
  for(int i=0;i<V.getNmodes();i++) assert( &V.getMode(i) != NULL);
  
  V.testRandom();

  W.allocModes();
  for(int i=0;i<W.getNl();i++) assert( &W.getWl(i) != NULL);
  for(int i=0;i<W.getNhits();i++) assert( &W.getWh(i) != NULL);
  W.testRandom();

  
  A2AvectorV<ManualAllocA2Apolicies> Vcopy = V;
  A2AvectorW<ManualAllocA2Apolicies> Wcopy = W;
  
  int pp[3]; GparityBaseMomentum(pp,+1); //(1,1,1)
  int pm[3]; GparityBaseMomentum(pm,-1); //(-1,-1,-1)

  gaugeFixAndTwist<FermionFieldType> fft_op(pp,lat);  
  reverseGaugeFixAndTwist<FermionFieldType> invfft_op(pp,lat);
  
  A2AvectorVfftw<ManualAllocA2Apolicies> Vfft(a2a_args,fp); //no allocation yet performed
  Vfft.destructivefft(V, &fft_op);

  for(int i=0;i<V.getNmodes();i++) assert( &V.getMode(i) == NULL);
  for(int i=0;i<Vfft.getNmodes();i++) assert( &Vfft.getMode(i) != NULL);

  
  A2AvectorV<ManualAllocA2Apolicies> Vrec(a2a_args,fp);
  Vfft.destructiveInversefft(Vrec, &invfft_op);

  for(int i=0;i<Vrec.getNmodes();i++) assert( &Vrec.getMode(i) != NULL);
  for(int i=0;i<Vfft.getNmodes();i++) assert( &Vfft.getMode(i) == NULL); 

  for(int i=0;i<Vrec.getNmodes();i++) assert( Vrec.getMode(i).equals( Vcopy.getMode(i), 1e-08, true) );

  
  printf("Passed V destructive fft/inverse test\n");
   
  A2AvectorWfftw<ManualAllocA2Apolicies> Wfft(a2a_args,fp);
  Wfft.destructiveGaugeFixTwistFFT(W,pp,lat);

  for(int i=0;i<W.getNl();i++) assert( &W.getWl(i) == NULL);
  for(int i=0;i<W.getNhits();i++) assert( &W.getWh(i) == NULL);
  
  for(int i=0;i<Wfft.getNmodes();i++) assert( &Wfft.getMode(i) != NULL);
  
  A2AvectorW<ManualAllocA2Apolicies> Wrec(a2a_args,fp);
  Wfft.destructiveUnapplyGaugeFixTwistFFT(Wrec, pp,lat);
  
  for(int i=0;i<Wfft.getNmodes();i++) assert( &Wfft.getMode(i) == NULL);

  for(int i=0;i<Wrec.getNl();i++) assert( &Wrec.getWl(i) != NULL);
  for(int i=0;i<Wrec.getNhits();i++) assert( &Wrec.getWh(i) != NULL);
  
  for(int i=0;i<Wrec.getNl();i++){
    assert( Wrec.getWl(i).equals( Wcopy.getWl(i), 1e-08, true) ); 
  }
  if(!UniqueID()) printf("Passed Wl destructive fft/inverse test\n"); 

  for(int i=0;i<Wrec.getNhits();i++){
    assert( Wrec.getWh(i).equals( Wcopy.getWh(i), 1e-08, true) ); 
  }
  if(!UniqueID()) printf("Passed Wh destructive fft/inverse test\n"); 
  
  
}


CPS_END_NAMESPACE
