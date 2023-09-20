#pragma once

CPS_START_NAMESPACE

template<typename A2Apolicies, typename A2Apolicies_destructive>
void testWunitaryBasic(const A2AArg &a2a_arg, const typename SIMDpolicyBase<4>::ParamType &simd_dims, Lattice &lat){
  std::cout << "Starting testWunitaryBasic" << std::endl;

  typedef typename A2Apolicies::FermionFieldType FermionFieldType;
  typedef typename A2Apolicies::ComplexFieldType ComplexFieldType;

  typedef typename A2Apolicies::ScalarFermionFieldType ScalarFermionFieldType;
  typedef typename A2Apolicies::ScalarComplexFieldType ScalarComplexFieldType;

  FermionFieldType tmp_ferm(simd_dims);
  NullObject null_obj;
  ScalarFermionFieldType tmp_ferm_s(null_obj);

  int nl = a2a_arg.nl;
  int nhit = a2a_arg.nhits;
  int nf = GJP.Gparity()+1;

  A2AvectorW<A2Apolicies> W(a2a_arg,simd_dims);
  A2AvectorWunitary<A2Apolicies> Wu(a2a_arg,simd_dims);
  
  for(int i=0;i<nl;i++){
    tmp_ferm_s.testRandom();
    tmp_ferm.importField(tmp_ferm_s);
    
    W.importWl(tmp_ferm, i);
    Wu.importWl(tmp_ferm, i);
  }

  //Original W assumes diagonal flavor structure
  //We can mimic that with Wunitary by zeroing the off-diagonal
  std::vector<ScalarComplexFieldType> wh(nhit, null_obj);
  std::vector<ScalarComplexFieldType> whu(nhit*nf, null_obj);
  for(int i=0;i<whu.size();i++) whu[i].zero();

  for(int h=0;h<nhit;h++){
    wh[h].testRandom();
 
    //Equivalent fermion field type uses the same random numbers for all spin/color
    for(int f=0;f<nf;f++){
      CPSautoView(whu_v,whu[Wu.indexMap(h,f)],HostWrite);
      CPSautoView(wh_v,wh[h],HostRead);
   
#pragma omp parallel for 
      for(size_t x=0;x<wh_v.nsites();x++){
	auto const *from = wh_v.site_ptr(x,f);
	auto *to = whu_v.site_ptr(x,f);
	*to = *from;
      }
    }
  }

  W.setWh(wh);
  Wu.setWh(whu);

  //Test elem
  int vol3d_node = GJP.VolNodeSites()/GJP.TnodeSites();
  int Lt_node = GJP.TnodeSites();
  int Lt = Lt_node * GJP.Tnodes();

  bool fail = false;
  {
    CPSautoView(W_v,W,HostRead);
    CPSautoView(Wu_v,Wu,HostRead);
    for(int mode=0;mode<nl+12*nf*Lt*nhit;mode++){ //full mode!
      for(int x3d=0;x3d<vol3d_node;x3d++){
	for(int t=0;t<Lt_node;t++){
	  for(int sc=0;sc<12;sc++){
	    for(int f=0;f<nf;f++){
	      const auto &ew = W_v.elem(mode,x3d,t,sc,f);
	      const auto &ewsc = Wu_v.elem(mode,x3d,t,sc,f);
	      if(!vTypeEquals(ew,ewsc,1e-12,false)){
		std::cout << "FAIL " << mode << " "<< x3d << " " << t << " " << sc << " " << f << std::endl;
		fail=true;
	      }
	    }
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","testWscDilutedBasic","elem test failed");

  //Test getDilutedSource
  FermionFieldType dil_w(simd_dims), dil_wu(simd_dims), tmpferm(simd_dims);
  for(int h=0;h<12*nf*Lt*nhit;h++){
    W.getDilutedSource(dil_w,h);
    Wu.getDilutedSource(dil_wu,h);
    assert(dil_w.equals(dil_wu,1e-12,true));
  }

  //Test getSpinColorDilutedSource
  dil_w.zero(); dil_wu.zero(); tmpferm.zero();
  for(int h=0;h<nhit;h++){
    for(int sc=0;sc<12;sc++){
      W.getSpinColorDilutedSource(dil_w,h,sc);

      //To get the Wu corresponding field we can sum the two columns (they live on different flavor row indices)
      Wu.getSpinColorDilutedSource(tmpferm, Wu.indexMap(h,0) ,sc);
      Wu.getSpinColorDilutedSource(dil_wu, Wu.indexMap(h,1) ,sc);
      dil_wu = dil_wu + tmpferm;

      assert(dil_w.equals(dil_wu,1e-12,true));
    }
  }

  //Test FFT
  A2AvectorWfftw<A2Apolicies> W_fft(a2a_arg,simd_dims);
  A2AvectorWunitaryfftw<A2Apolicies> Wu_fft(a2a_arg,simd_dims);
  W_fft.fft(W);
  Wu_fft.fft(Wu);
  
  //Wfft is still flavor-packed, but because the columns are flavor-orthogonal in this test we can sum them
  for(int i=0;i<nl;i++){
    assert(W_fft.getMode(i).equals(Wu_fft.getMode(i),1e-8,true));
  }
  for(int i=0;i<W_fft.getNhighModes();i++){
    int hit, sc;
    W_fft.indexUnmap(i,hit,sc);
    tmpferm = Wu_fft.getHighMode( Wu_fft.indexMap(hit,sc,0) ) + Wu_fft.getHighMode( Wu_fft.indexMap(hit,sc,1) );
    assert(W_fft.getHighMode(i).equals(tmpferm,1e-8,true));
  }

  //Test inverse FFT
  A2AvectorW<A2Apolicies> W_inv(a2a_arg,simd_dims);
  A2AvectorWunitary<A2Apolicies> Wu_inv(a2a_arg,simd_dims);
  W_fft.inversefft(W_inv);
  Wu_fft.inversefft(Wu_inv);
  
  for(int i=0;i<nl;i++){
    assert(W_inv.getWl(i).equals(W.getWl(i),1e-8,true));
    assert(Wu_inv.getWl(i).equals(Wu.getWl(i),1e-8,true));
  }
  for(int i=0;i<nhit;i++){
    assert(W_inv.getWh(i).equals(W.getWh(i),1e-8,true));
    for(int f=0;f<nf;f++){
      int mode = Wu.indexMap(i,f);
      assert(Wu_inv.getWh(mode).equals(Wu.getWh(mode),1e-8,true));
    }
  }
  
  //Test gaugeFixTwistFFT
  int p[3] = {1,1,1};
  W_fft.zero(); Wu_fft.zero();

  W_fft.gaugeFixTwistFFT(W,p,lat);
  Wu_fft.gaugeFixTwistFFT(Wu,p,lat);
  for(int i=0;i<nl;i++){
    assert(W_fft.getMode(i).equals(Wu_fft.getMode(i),1e-8,true));
  }
  for(int i=0;i<W_fft.getNhighModes();i++){
    int hit, sc;
    W_fft.indexUnmap(i,hit,sc);
    tmpferm = Wu_fft.getHighMode( Wu_fft.indexMap(hit,sc,0) ) + Wu_fft.getHighMode( Wu_fft.indexMap(hit,sc,1) );
    assert(W_fft.getHighMode(i).equals(tmpferm,1e-8,true));
  }

  //Test unapplyGaugeFixTwistFFT
  W_inv.zero(); Wu_inv.zero();
  W_fft.unapplyGaugeFixTwistFFT(W_inv,p,lat);
  Wu_fft.unapplyGaugeFixTwistFFT(Wu_inv,p,lat);

  for(int i=0;i<nl;i++){
    assert(W_inv.getWl(i).equals(W.getWl(i),1e-8,true));
    assert(Wu_inv.getWl(i).equals(Wu.getWl(i),1e-8,true));
  }
  for(int i=0;i<nhit;i++){
    assert(W_inv.getWh(i).equals(W.getWh(i),1e-8,true));
    for(int f=0;f<nf;f++){
      int mode = Wu.indexMap(i,f);
      assert(Wu_inv.getWh(mode).equals(Wu.getWh(mode),1e-8,true));
    }
  }

  //Test destructiveGaugeFixTwistFFT
  A2AvectorW<A2Apolicies_destructive> W_dest(a2a_arg,simd_dims);
  A2AvectorWunitary<A2Apolicies_destructive> Wu_dest(a2a_arg,simd_dims);
  W_dest.allocModes();
  Wu_dest.allocModes();
  for(int i=0;i<nl;i++){
    W_dest.getWl(i) = W.getWl(i);
    Wu_dest.getWl(i) = Wu.getWl(i);
  }
  for(int h=0;h<nhit;h++) W_dest.getWh(h) = W.getWh(h);
  for(int h=0;h<Wu.getNhighModes();h++) Wu_dest.getWh(h) = Wu.getWh(h);
  
  A2AvectorWfftw<A2Apolicies_destructive> W_dest_fft(a2a_arg,simd_dims);
  A2AvectorWunitaryfftw<A2Apolicies_destructive> Wu_dest_fft(a2a_arg,simd_dims);
  
  W_dest_fft.destructiveGaugeFixTwistFFT(W_dest,p,lat);
  Wu_dest_fft.destructiveGaugeFixTwistFFT(Wu_dest,p,lat);

  for(int i=0;i<nl;i++){
    assert(W_dest_fft.getMode(i).equals(Wu_dest_fft.getMode(i),1e-8,true));
  }
  for(int i=0;i<W_dest_fft.getNhighModes();i++){
    int hit, sc;
    W_dest_fft.indexUnmap(i,hit,sc);
    tmpferm = Wu_dest_fft.getHighMode( Wu_dest_fft.indexMap(hit,sc,0) ) + Wu_dest_fft.getHighMode( Wu_dest_fft.indexMap(hit,sc,1) );
    assert(W_dest_fft.getHighMode(i).equals(tmpferm,1e-8,true));
  }


  //Test destructiveUnapplyGaugeFixTwistFFT
  A2AvectorW<A2Apolicies_destructive> W_dest_inv(a2a_arg,simd_dims);
  A2AvectorWunitary<A2Apolicies_destructive> Wu_dest_inv(a2a_arg,simd_dims);
  W_dest_fft.destructiveUnapplyGaugeFixTwistFFT(W_dest_inv,p,lat);
  Wu_dest_fft.destructiveUnapplyGaugeFixTwistFFT(Wu_dest_inv,p,lat);

  for(int i=0;i<nl;i++){
    assert(W_dest_inv.getWl(i).equals(W.getWl(i),1e-8,true)); //check against original W as W_dest has been deallocated!
    assert(Wu_dest_inv.getWl(i).equals(Wu.getWl(i),1e-8,true));
  }
  for(int i=0;i<nhit;i++){
    assert(W_dest_inv.getWh(i).equals(W.getWh(i),1e-8,true));
    for(int f=0;f<nf;f++){
      int mode = Wu.indexMap(i,f);
      assert(Wu_dest_inv.getWh(mode).equals(Wu.getWh(mode),1e-8,true));
    }
  }


  //Test mesonfield generation
  //Because the flavor is unpacked during the generation process, they should give the same results
  A2Aparams a2a_params(a2a_arg);
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf;
  A2AmesonField<A2Apolicies,A2AvectorWunitaryfftw,A2AvectorVfftw> mfu;
  typedef GparityNoSourceInnerProduct<typename A2Apolicies::ComplexType, flavorMatrixSpinColorContract<15,false,true> > InnerProduct;
  InnerProduct inner(sigma3);
  int t=0;

  W_fft.gaugeFixTwistFFT(W,p,lat);
  Wu_fft.gaugeFixTwistFFT(Wu,p,lat);
  A2AvectorVfftw<A2Apolicies> V_fft(a2a_arg,simd_dims);
  V_fft.testRandom();
  mf.compute(W_fft,inner,V_fft,t);
  mfu.compute(Wu_fft,inner,V_fft,t);
  
  assert(mf.size() == mfu.size());
  //The underlying mesonfield data should be the same, but the equivalence operator won't work because the types differ
  //circumvent with memcpy
  {
    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfu_conv;
    mfu_conv.setup(W_fft,V_fft,0,0);
    {
      CPSautoView(mfu_conv_v,mfu_conv,HostWrite);
      CPSautoView(mfu_v,mfu,HostRead);
      memcpy(mfu_conv_v.ptr(), mfu_v.ptr(), mfu_v.size()*sizeof(typename A2Apolicies::ScalarComplexType));
    }
    assert(mf.equals(mfu_conv,1e-8,true));
  }

  //Test vector*vector operation with WW and VW forms
  A2AvectorV<A2Apolicies> V(a2a_arg,simd_dims);
  V.testRandom();

  typedef typename getPropagatorFieldType<A2Apolicies>::type PropagatorField;
  PropagatorField pfield_w(simd_dims), pfield_wu(simd_dims);

  mult(pfield_w, W, W, false, true);
  mult(pfield_wu, Wu, Wu, false, true);

  CPSspinColorFlavorMatrix<typename A2Apolicies::ComplexType> WuWu_0, WW_0;
  {
    CPSautoView(Wu_v,Wu,HostRead);
    mult_slow(WuWu_0,Wu_v,Wu_v,0,0,false,true);

    CPSautoView(W_v,W,HostRead);
    mult_slow(WW_0,W_v,W_v,0,0,false,true);
  }

  std::cout << "MULT SLOW" << std::endl;
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  for(int f1=0;f1<2;f1++)
	    for(int f2=0;f2<2;f2++){
	      auto diff = WW_0(s1,s2)(c1,c2)(f1,f2) - WuWu_0(s1,s2)(c1,c2)(f1,f2);
		std::cout << s1 << " " << s2 << " " << c1 << " " << c2 << " " << f1 << " " << f2 << " " <<  WuWu_0(s1,s2)(c1,c2)(f1,f2) << " <----> " << WW_0(s1,s2)(c1,c2)(f1,f2) << " diff " << diff << std::endl;
	    }
  
  WuWu_0.zero(); WW_0.zero();
  {
    CPSautoView(Wu_v,Wu,HostRead);
    mult(WuWu_0,Wu_v,Wu_v,0,0,false,true);

    CPSautoView(W_v,W,HostRead);
    mult(WW_0,W_v,W_v,0,0,false,true);
  }

  std::cout << "MULT OPT" << std::endl;
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  for(int f1=0;f1<2;f1++)
	    for(int f2=0;f2<2;f2++){
	      auto diff = WW_0(s1,s2)(c1,c2)(f1,f2) - WuWu_0(s1,s2)(c1,c2)(f1,f2);	      
	      std::cout << s1 << " " << s2 << " " << c1 << " " << c2 << " " << f1 << " " << f2 << " " <<  WuWu_0(s1,s2)(c1,c2)(f1,f2) << " <----> " << WW_0(s1,s2)(c1,c2)(f1,f2) << " diff " << diff << std::endl;
	    }


  
  std::cout << "FIELD OPT SITE" << std::endl;
  {
    CPSautoView(pfield_w_v,pfield_w,HostRead);
    CPSautoView(pfield_wu_v,pfield_wu,HostRead);

    size_t s=0;
    auto WW_0 = *pfield_w_v.site_ptr(s);
    auto WuWu_0 = *pfield_wu_v.site_ptr(s);

    for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  for(int f1=0;f1<2;f1++)
	    for(int f2=0;f2<2;f2++){
	      auto diff = WW_0(s1,s2)(c1,c2)(f1,f2) - WuWu_0(s1,s2)(c1,c2)(f1,f2);	      
	      std::cout << s1 << " " << s2 << " " << c1 << " " << c2 << " " << f1 << " " << f2 << " " <<  WuWu_0(s1,s2)(c1,c2)(f1,f2) << " <----> " << WW_0(s1,s2)(c1,c2)(f1,f2) << " diff " << diff << std::endl;
	    }
  }


  auto pfield_w_up = linearUnpack(pfield_w);
  auto pfield_wu_up = linearUnpack(pfield_wu);
  assert(pfield_w_up.equals(pfield_wu_up,1e-8,true));
  
  mult(pfield_w, V, W, false, true);
  mult(pfield_wu, V, Wu, false, true);
  pfield_w_up = linearUnpack(pfield_w);
  pfield_wu_up = linearUnpack(pfield_wu);
  assert(pfield_w_up.equals(pfield_wu_up,1e-8,true));


  //Test vector*matrix*vector with W M W and W M V forms
  pfield_w.zero(); pfield_wu.zero();

  mult(pfield_w, W, mf, W, false, true);
  mult(pfield_wu, Wu, mfu, Wu, false, true);
  pfield_w_up = linearUnpack(pfield_w);
  pfield_wu_up = linearUnpack(pfield_wu);
  assert(pfield_w_up.equals(pfield_wu_up,1e-8,true));
  
  mult(pfield_w, W, mf, V, false, true);
  mult(pfield_wu, Wu, mfu, V, false, true);
  pfield_w_up = linearUnpack(pfield_w);
  pfield_wu_up = linearUnpack(pfield_wu);
  assert(pfield_w_up.equals(pfield_wu_up,1e-8,true));
 
  std::cout << "testWunitaryBasic passed" << std::endl;
}


template<typename A2Apolicies>
void testWunitaryUnitaryRandomSrc(A2AArg a2a_arg, const typename SIMDpolicyBase<4>::ParamType &simd_dims, Lattice &lat){
  std::cout << "Starting testWunitaryUnitaryRandomSrc" << std::endl;

  typedef typename A2Apolicies::FermionFieldType FermionFieldType;
  typedef typename A2Apolicies::ComplexFieldType ComplexFieldType;

  typedef typename A2Apolicies::ScalarFermionFieldType ScalarFermionFieldType;
  typedef typename A2Apolicies::ScalarComplexFieldType ScalarComplexFieldType;

  FermionFieldType tmp_ferm(simd_dims);
  NullObject null_obj;
  ScalarFermionFieldType tmp_ferm_s(null_obj);

  a2a_arg.nl = 0;

  int nf = GJP.Gparity()+1;

  A2AvectorWunitary<A2Apolicies> Wu(a2a_arg,simd_dims);
  
  A2AhighModeSourceFlavorUnitary<A2Apolicies> src_pol;
  src_pol.setHighModeSources(Wu);

  typedef typename getPropagatorFieldType<A2Apolicies>::type PropagatorField;

  {
    //Test unitary source by computing   W(x) W^dag(x)
    PropagatorField pfield(simd_dims);
    mult(pfield,Wu,Wu,false,true);
       
    PropagatorField pfield_expect(simd_dims);
    setUnit(pfield_expect);

    PropagatorField pfield_diff = pfield - pfield_expect;

    std::cout << "W unitary check: " << CPSmatrixFieldNorm2(pfield_diff) << std::endl;
  }
  {
    //Test W(x) W^dag(y)  ->   0  in large hit limit   
    A2AvectorWunitary<A2Apolicies> Wu_shift(a2a_arg,simd_dims);
    for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 1);

    PropagatorField pfield_sum(simd_dims);
    pfield_sum.zero();

    PropagatorField pfield(simd_dims);

    int iter=0;
    while(iter < 15000){
      mult(pfield,Wu,Wu_shift,false,true);
      pfield_sum = pfield_sum + pfield;

      pfield = cps::ComplexD(1./(iter+1)) * pfield_sum;

      std::cout << "TEST " << iter << " " << CPSmatrixFieldNorm2(pfield) << std::endl;

      src_pol.setHighModeSources(Wu);
      for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 1);
      ++iter;
    }
  }

  std::cout << "testWunitaryUnitaryRandomSrc passed" << std::endl;
}


template<typename A2Apolicies>
void testWunitaryRotYRandomSrc(A2AArg a2a_arg, const typename SIMDpolicyBase<4>::ParamType &simd_dims, Lattice &lat){
  std::cout << "Starting testWunitaryRotYRandomSrc" << std::endl;

  typedef typename A2Apolicies::FermionFieldType FermionFieldType;
  typedef typename A2Apolicies::ComplexFieldType ComplexFieldType;

  typedef typename A2Apolicies::ScalarFermionFieldType ScalarFermionFieldType;
  typedef typename A2Apolicies::ScalarComplexFieldType ScalarComplexFieldType;

  FermionFieldType tmp_ferm(simd_dims);
  NullObject null_obj;
  ScalarFermionFieldType tmp_ferm_s(null_obj);

  a2a_arg.nl = 0;

  int nf = GJP.Gparity()+1;

  A2AvectorWunitary<A2Apolicies> Wu(a2a_arg,simd_dims);
  
  A2AhighModeSourceFlavorRotY<A2Apolicies> src_pol;
  src_pol.setHighModeSources(Wu);

  typedef typename getPropagatorFieldType<A2Apolicies>::type PropagatorField;

  {
    //Test unitary source by computing   W(x) W^dag(x)
    PropagatorField pfield(simd_dims);
    mult(pfield,Wu,Wu,false,true);
       
    PropagatorField pfield_expect(simd_dims);
    setUnit(pfield_expect);

    PropagatorField pfield_diff = pfield - pfield_expect;

    std::cout << "W unitary check: " << CPSmatrixFieldNorm2(pfield_diff) << std::endl;
  }
  {
    //Show noise is only proportional to unit matrix and sigma2
    A2AvectorWunitary<A2Apolicies> Wu_shift(a2a_arg,simd_dims);
    for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 1);

    PropagatorField pfield(simd_dims);

    mult(pfield,Wu,Wu_shift,false,true);

    std::cout << "Testing noise structure" << std::endl;

    auto c0 = TraceIndex<2>(pfield);
    std::cout << "sigma0: " << CPSmatrixFieldNorm2(c0) << std::endl;
    
    PropagatorField tmp(simd_dims);

    tmp = pfield; pr(tmp,sigma1);
    auto c1 = TraceIndex<2>(tmp);
    std::cout << "sigma1: " << CPSmatrixFieldNorm2(c1) << std::endl;

    tmp = pfield; pr(tmp,sigma2);
    auto c2 = TraceIndex<2>(tmp);
    std::cout << "sigma2: " << CPSmatrixFieldNorm2(c2) << std::endl;

    tmp = pfield; pr(tmp,sigma3);
    auto c3 = TraceIndex<2>(tmp);
    std::cout << "sigma3: " << CPSmatrixFieldNorm2(c3) << std::endl;
  }

  if(0){
    //Test W(x) W^dag(y)  ->   0  in large hit limit   
    A2AvectorWunitary<A2Apolicies> Wu_shift(a2a_arg,simd_dims);
    for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 1);

    PropagatorField pfield_sum(simd_dims);
    pfield_sum.zero();

    PropagatorField pfield(simd_dims);

    int iter=0;
    while(iter < 15000){
      mult(pfield,Wu,Wu_shift,false,true);
      pfield_sum = pfield_sum + pfield;

      pfield = cps::ComplexD(1./(iter+1)) * pfield_sum;

      std::cout << "TEST " << iter << " " << CPSmatrixFieldNorm2(pfield) << std::endl;

      src_pol.setHighModeSources(Wu);
      for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 1);
      ++iter;
    }
  }

  std::cout << "testWunitaryRotYRandomSrc passed" << std::endl;
}


template<typename A2Apolicies>
void testSourceConvergence(A2AArg a2a_arg, const typename SIMDpolicyBase<4>::ParamType &simd_dims, const typename SIMDpolicyBase<3>::ParamType &simd_dims_3d, typename A2Apolicies::FgridGFclass &lat){
  typedef typename A2Apolicies::ScalarComplexType ScalarComplexType;
  typedef typename A2Apolicies::GridFermionField GridFermionFieldD;
  typedef typename A2Apolicies::GridFermionFieldF GridFermionFieldF;

  a2a_arg.nl = 0;

  CGcontrols cg_orig;
  cg_orig.CGalgorithm = AlgorithmMixedPrecisionReliableUpdateCG;
  cg_orig.CG_tolerance = 1e-8;
  cg_orig.CG_max_iters = 10000;
  cg_orig.reliable_update_delta = 0.1;
  cg_orig.reliable_update_transition_tol = 0;
  cg_orig.highmode_source = A2AhighModeSourceTypeOrig;
  cg_orig.multiCG_block_size = 1;

  CGcontrols cg_new = cg_orig;
  cg_new.highmode_source = A2AhighModeSourceTypeXconj;

  CGcontrols cg_u1x = cg_orig;
  cg_new.highmode_source = A2AhighModeSourceTypeU1X;

  typedef A2AvectorV<A2Apolicies> Vtype;
  typedef A2AvectorW<A2Apolicies> Wtype;
  typedef A2AvectorWtimePacked<A2Apolicies> WtimePackedType;
  ThreeMomentum p(2,2,2);
  EvecInterfaceMixedPrecNone<GridFermionFieldD, GridFermionFieldF> eveci(lat.getFrbGrid(), lat.getFrbGridF());
  StandardPionMomentaPolicy mompol;

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  fVector<ScalarComplexType> Vdis_sum_orig(Lt), Vdis_sum_new(Lt);

  for(int hit=0;hit<100;hit++){
    
    //Orig
    {
      Wtype W(a2a_arg,simd_dims);
      Vtype V(a2a_arg,simd_dims);      
      computeVW(V,W,lat,eveci,0.01,cg_orig);      
      MesonFieldMomentumContainer<getMesonFieldType<Wtype,Vtype> > mf_ll_con;
      computeGparityLLmesonFields1sSumOnTheFly<Vtype,Wtype,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con, mompol, W, V, 2, lat, simd_dims_3d);
      fVector<ScalarComplexType> Vdis;
      ComputePiPiGparity<Vtype,Wtype>::computeFigureVdis(Vdis, p, 3, mf_ll_con);
      assert(Vdis.size() == Lt);
      for(int t=0;t<Lt;t++) Vdis_sum_orig(t) += Vdis(t);
    }
#if 0
    //New
    {
      Wtype W(a2a_arg,simd_dims);
      Vtype V(a2a_arg,simd_dims);      
      computeVW(V,W,lat,eveci,0.01,cg_new);      
      MesonFieldMomentumContainer<getMesonFieldType<Wtype,Vtype> > mf_ll_con;
      computeGparityLLmesonFields1sSumOnTheFly<Vtype,Wtype,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con, mompol, W, V, 2, lat, simd_dims_3d);
      fVector<ScalarComplexType> Vdis;
      ComputePiPiGparity<Vtype,Wtype>::computeFigureVdis(Vdis, p, 3, mf_ll_con);
      assert(Vdis.size() == Lt);
      for(int t=0;t<Lt;t++) Vdis_sum_new(t) += Vdis(t);
    }
#endif
    {
      WtimePackedType W(a2a_arg,simd_dims);
      Vtype V(a2a_arg,simd_dims);      
      computeVW(V,W,lat,eveci,0.01,cg_u1x);      
      MesonFieldMomentumContainer<getMesonFieldType<WtimePackedType,Vtype> > mf_ll_con;
      computeGparityLLmesonFields1sSumOnTheFly<Vtype,WtimePackedType,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con, mompol, W, V, 2, lat, simd_dims_3d);
      fVector<ScalarComplexType> Vdis;
      ComputePiPiGparity<Vtype,WtimePackedType>::computeFigureVdis(Vdis, p, 3, mf_ll_con);
      assert(Vdis.size() == Lt);
      for(int t=0;t<Lt;t++) Vdis_sum_new(t) += Vdis(t);
    }

    for(int t=0;t<Lt;t++){
      std::cout << "TEST " << hit << " " << t << " " << Vdis_sum_orig(t) * (1./(hit+1)) << " " << Vdis_sum_new(t) * (1./(hit+1)) << std::endl;
    }
  }



  
}







template<typename A2Apolicies, typename A2Apolicies_destructive>
void testWtimePackedBasic(const A2AArg &a2a_arg, const typename SIMDpolicyBase<4>::ParamType &simd_dims, Lattice &lat){
  std::cout << "Starting testWtimePackedBasic" << std::endl;

  typedef typename A2Apolicies::FermionFieldType FermionFieldType;
  typedef typename A2Apolicies::ComplexFieldType ComplexFieldType;

  typedef typename A2Apolicies::ScalarFermionFieldType ScalarFermionFieldType;
  typedef typename A2Apolicies::ScalarComplexFieldType ScalarComplexFieldType;

  FermionFieldType tmp_ferm(simd_dims);
  NullObject null_obj;
  ScalarFermionFieldType tmp_ferm_s(null_obj);

  int nl = a2a_arg.nl;
  int nhit = a2a_arg.nhits;
  int nf = GJP.Gparity()+1;
  int nsc = 12;

  A2AvectorW<A2Apolicies> W(a2a_arg,simd_dims);
  A2AvectorWtimePacked<A2Apolicies> Wu(a2a_arg,simd_dims);
  
  for(int i=0;i<nl;i++){
    tmp_ferm_s.testRandom();
    tmp_ferm.importField(tmp_ferm_s);
    
    W.importWl(tmp_ferm, i);
    Wu.importWl(tmp_ferm, i);
  }

  //Original W assumes diagonal flavor structure
  //We can mimic that with Wunitary by zeroing the off-diagonal
  std::vector<ScalarComplexFieldType> wh(nhit, null_obj);
  std::vector<ScalarFermionFieldType> whu(nhit*nsc*nf, null_obj);
  for(int i=0;i<whu.size();i++) whu[i].zero();

  for(int h=0;h<nhit;h++){
    wh[h].testRandom();
 
    //Equivalent fermion field type uses the same random numbers for all spin/color diagonal elements
    for(int f=0;f<nf;f++){
      for(int sc=0;sc<nsc;sc++){
	CPSautoView(whu_v,whu[Wu.indexMap(h,sc,f)],HostWrite);
	CPSautoView(wh_v,wh[h],HostRead);
   
#pragma omp parallel for 
	for(size_t x=0;x<wh_v.nsites();x++){
	  auto const *from = wh_v.site_ptr(x,f);
	  auto *to = whu_v.site_ptr(x,f) + sc;
	  *to = *from;
	}
      }
    }
  }

  W.setWh(wh);
  Wu.setWh(whu);

  //Test elem
  int vol3d_node = GJP.VolNodeSites()/GJP.TnodeSites();
  int Lt_node = GJP.TnodeSites();
  int Lt = Lt_node * GJP.Tnodes();

  bool fail = false;
  {
    CPSautoView(W_v,W,HostRead);
    CPSautoView(Wu_v,Wu,HostRead);
    for(int mode=0;mode<nl+12*nf*Lt*nhit;mode++){ //full mode!
      for(int x3d=0;x3d<vol3d_node;x3d++){
	for(int t=0;t<Lt_node;t++){
	  for(int sc=0;sc<nsc;sc++){
	    for(int f=0;f<nf;f++){
	      const auto &ew = W_v.elem(mode,x3d,t,sc,f);
	      const auto &ewsc = Wu_v.elem(mode,x3d,t,sc,f);
	      if(!vTypeEquals(ew,ewsc,1e-12,false)){
		std::cout << "FAIL " << mode << " "<< x3d << " " << t << " " << sc << " " << f << std::endl;
		fail=true;
	      }
	    }
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","testWscDilutedBasic","elem test failed");



  //Test getDilutedSource
  FermionFieldType dil_w(simd_dims), dil_wu(simd_dims), tmpferm(simd_dims);
  for(int h=0;h<12*nf*Lt*nhit;h++){
    W.getDilutedSource(dil_w,h);
    Wu.getDilutedSource(dil_wu,h);
    assert(dil_w.equals(dil_wu,1e-12,true));
  }

  //Test getSpinColorDilutedSource
  dil_w.zero(); dil_wu.zero(); tmpferm.zero();
  for(int h=0;h<nhit;h++){
    for(int sc=0;sc<12;sc++){
      W.getSpinColorDilutedSource(dil_w,h,sc); 
      //this returns a fermion with the sc element populated and other elements 0. The value will be the same for all sc index
      //both flavor elements will be populated, but not necessarily with the same value

      //To get the Wu corresponding field we can sum the two flavor columns (they live on different flavor row indices)
      //Because of the way we set it up above, the value will be zero unless   mode_sc == sc, but we can test better by summing over all
      dil_wu.zero();
      for(int mode_sc=0;mode_sc<12;mode_sc++){
	Wu.getSpinColorDilutedSource(tmpferm, Wu.indexMap(h,mode_sc,0) ,sc);
	dil_wu = dil_wu + tmpferm;
	
	Wu.getSpinColorDilutedSource(tmpferm, Wu.indexMap(h,mode_sc,1) ,sc);
	dil_wu = dil_wu + tmpferm;
      }

      assert(dil_w.equals(dil_wu,1e-12,true));
    }
  }

  //Test FFT
  A2AvectorWfftw<A2Apolicies> W_fft(a2a_arg,simd_dims);
  A2AvectorWunitaryfftw<A2Apolicies> Wu_fft(a2a_arg,simd_dims);
  W_fft.fft(W);
  Wu_fft.fft(Wu);
  
  //Wfft is still flavor-packed, but because the columns are flavor-orthogonal in this test we can sum them
  for(int i=0;i<nl;i++){
    assert(W_fft.getMode(i).equals(Wu_fft.getMode(i),1e-8,true));
  }
  for(int i=0;i<W_fft.getNhighModes();i++){
    int hit, sc;
    W_fft.indexUnmap(i,hit,sc);
    tmpferm = Wu_fft.getHighMode( Wu_fft.indexMap(hit,sc,0) ) + Wu_fft.getHighMode( Wu_fft.indexMap(hit,sc,1) );
    assert(W_fft.getHighMode(i).equals(tmpferm,1e-8,true));
  }

  //Test inverse FFT
  A2AvectorW<A2Apolicies> W_inv(a2a_arg,simd_dims);
  A2AvectorWtimePacked<A2Apolicies> Wu_inv(a2a_arg,simd_dims);
  W_fft.inversefft(W_inv);
  Wu_fft.inversefft(Wu_inv);
  
  for(int i=0;i<nl;i++){
    assert(W_inv.getWl(i).equals(W.getWl(i),1e-8,true));
    assert(Wu_inv.getWl(i).equals(Wu.getWl(i),1e-8,true));
  }
  for(int i=0;i<nhit;i++){
    assert(W_inv.getWh(i).equals(W.getWh(i),1e-8,true));
    for(int f=0;f<nf;f++){
      int mode = Wu.indexMap(i,f);
      assert(Wu_inv.getWh(mode).equals(Wu.getWh(mode),1e-8,true));
    }
  }

  
  //Test gaugeFixTwistFFT
  int p[3] = {1,1,1};
  W_fft.zero(); Wu_fft.zero();

  W_fft.gaugeFixTwistFFT(W,p,lat);
  Wu_fft.gaugeFixTwistFFT(Wu,p,lat);
  for(int i=0;i<nl;i++){
    assert(W_fft.getMode(i).equals(Wu_fft.getMode(i),1e-8,true));
  }
  for(int i=0;i<W_fft.getNhighModes();i++){
    int hit, sc;
    W_fft.indexUnmap(i,hit,sc);
    tmpferm = Wu_fft.getHighMode( Wu_fft.indexMap(hit,sc,0) ) + Wu_fft.getHighMode( Wu_fft.indexMap(hit,sc,1) );
    assert(W_fft.getHighMode(i).equals(tmpferm,1e-8,true));
  }

  //Test unapplyGaugeFixTwistFFT
  W_inv.zero(); Wu_inv.zero();
  W_fft.unapplyGaugeFixTwistFFT(W_inv,p,lat);
  Wu_fft.unapplyGaugeFixTwistFFT(Wu_inv,p,lat);

  for(int i=0;i<nl;i++){
    assert(W_inv.getWl(i).equals(W.getWl(i),1e-8,true));
    assert(Wu_inv.getWl(i).equals(Wu.getWl(i),1e-8,true));
  }
  for(int i=0;i<nhit;i++){
    assert(W_inv.getWh(i).equals(W.getWh(i),1e-8,true));
    for(int f=0;f<nf;f++){
      int mode = Wu.indexMap(i,f);
      assert(Wu_inv.getWh(mode).equals(Wu.getWh(mode),1e-8,true));
    }
  }

  //Test destructiveGaugeFixTwistFFT
  A2AvectorW<A2Apolicies_destructive> W_dest(a2a_arg,simd_dims);
  A2AvectorWtimePacked<A2Apolicies_destructive> Wu_dest(a2a_arg,simd_dims);
  W_dest.allocModes();
  Wu_dest.allocModes();
  for(int i=0;i<nl;i++){
    W_dest.getWl(i) = W.getWl(i);
    Wu_dest.getWl(i) = Wu.getWl(i);
  }
  for(int h=0;h<nhit;h++) W_dest.getWh(h) = W.getWh(h);
  for(int h=0;h<Wu.getNhighModes();h++) Wu_dest.getWh(h) = Wu.getWh(h);
  
  A2AvectorWfftw<A2Apolicies_destructive> W_dest_fft(a2a_arg,simd_dims);
  A2AvectorWunitaryfftw<A2Apolicies_destructive> Wu_dest_fft(a2a_arg,simd_dims);
  
  W_dest_fft.destructiveGaugeFixTwistFFT(W_dest,p,lat);
  Wu_dest_fft.destructiveGaugeFixTwistFFT(Wu_dest,p,lat);

  for(int i=0;i<nl;i++){
    assert(W_dest_fft.getMode(i).equals(Wu_dest_fft.getMode(i),1e-8,true));
  }
  for(int i=0;i<W_dest_fft.getNhighModes();i++){
    int hit, sc;
    W_dest_fft.indexUnmap(i,hit,sc);
    tmpferm = Wu_dest_fft.getHighMode( Wu_dest_fft.indexMap(hit,sc,0) ) + Wu_dest_fft.getHighMode( Wu_dest_fft.indexMap(hit,sc,1) );
    assert(W_dest_fft.getHighMode(i).equals(tmpferm,1e-8,true));
  }


  //Test destructiveUnapplyGaugeFixTwistFFT
  A2AvectorW<A2Apolicies_destructive> W_dest_inv(a2a_arg,simd_dims);
  A2AvectorWtimePacked<A2Apolicies_destructive> Wu_dest_inv(a2a_arg,simd_dims);
  W_dest_fft.destructiveUnapplyGaugeFixTwistFFT(W_dest_inv,p,lat);
  Wu_dest_fft.destructiveUnapplyGaugeFixTwistFFT(Wu_dest_inv,p,lat);

  for(int i=0;i<nl;i++){
    assert(W_dest_inv.getWl(i).equals(W.getWl(i),1e-8,true)); //check against original W as W_dest has been deallocated!
    assert(Wu_dest_inv.getWl(i).equals(Wu.getWl(i),1e-8,true));
  }
  for(int i=0;i<nhit;i++){
    assert(W_dest_inv.getWh(i).equals(W.getWh(i),1e-8,true));
    for(int f=0;f<nf;f++){
      int mode = Wu.indexMap(i,f);
      assert(Wu_dest_inv.getWh(mode).equals(Wu.getWh(mode),1e-8,true));
    }
  }

  //Test mesonfield generation
  //Because the flavor is unpacked during the generation process, they should give the same results
  A2Aparams a2a_params(a2a_arg);
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf;
  A2AmesonField<A2Apolicies,A2AvectorWunitaryfftw,A2AvectorVfftw> mfu;
  typedef GparityNoSourceInnerProduct<typename A2Apolicies::ComplexType, flavorMatrixSpinColorContract<15,false,true> > InnerProduct;
  InnerProduct inner(sigma3);
  int t=0;

  W_fft.gaugeFixTwistFFT(W,p,lat);
  Wu_fft.gaugeFixTwistFFT(Wu,p,lat);
  A2AvectorVfftw<A2Apolicies> V_fft(a2a_arg,simd_dims);
  V_fft.testRandom();
  mf.compute(W_fft,inner,V_fft,t);
  mfu.compute(Wu_fft,inner,V_fft,t);
  
  assert(mf.size() == mfu.size());
  //The underlying mesonfield data should be the same, but the equivalence operator won't work because the types differ
  //circumvent with memcpy
  {
    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfu_conv;
    mfu_conv.setup(W_fft,V_fft,0,0);
    {
      CPSautoView(mfu_conv_v,mfu_conv,HostWrite);
      CPSautoView(mfu_v,mfu,HostRead);
      memcpy(mfu_conv_v.ptr(), mfu_v.ptr(), mfu_v.size()*sizeof(typename A2Apolicies::ScalarComplexType));
    }
    assert(mf.equals(mfu_conv,1e-8,true));
  }

  //Test vector*vector operation with WW and VW forms
  A2AvectorV<A2Apolicies> V(a2a_arg,simd_dims);
  V.testRandom();

  typedef typename getPropagatorFieldType<A2Apolicies>::type PropagatorField;
  PropagatorField pfield_w(simd_dims), pfield_wu(simd_dims);

  mult(pfield_w, W, W, false, true);
  mult(pfield_wu, Wu, Wu, false, true);

  CPSspinColorFlavorMatrix<typename A2Apolicies::ComplexType> WuWu_0, WW_0;
  {
    CPSautoView(Wu_v,Wu,HostRead);
    mult_slow(WuWu_0,Wu_v,Wu_v,0,0,false,true);

    CPSautoView(W_v,W,HostRead);
    mult_slow(WW_0,W_v,W_v,0,0,false,true);
  }

  std::cout << "MULT SLOW" << std::endl;
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  for(int f1=0;f1<2;f1++)
	    for(int f2=0;f2<2;f2++){
	      auto diff = WW_0(s1,s2)(c1,c2)(f1,f2) - WuWu_0(s1,s2)(c1,c2)(f1,f2);
		std::cout << s1 << " " << s2 << " " << c1 << " " << c2 << " " << f1 << " " << f2 << " " <<  WuWu_0(s1,s2)(c1,c2)(f1,f2) << " <----> " << WW_0(s1,s2)(c1,c2)(f1,f2) << " diff " << diff << std::endl;
	    }
  
  WuWu_0.zero(); WW_0.zero();
  {
    CPSautoView(Wu_v,Wu,HostRead);
    mult(WuWu_0,Wu_v,Wu_v,0,0,false,true);

    CPSautoView(W_v,W,HostRead);
    mult(WW_0,W_v,W_v,0,0,false,true);
  }

  std::cout << "MULT OPT" << std::endl;
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  for(int f1=0;f1<2;f1++)
	    for(int f2=0;f2<2;f2++){
	      auto diff = WW_0(s1,s2)(c1,c2)(f1,f2) - WuWu_0(s1,s2)(c1,c2)(f1,f2);	      
	      std::cout << s1 << " " << s2 << " " << c1 << " " << c2 << " " << f1 << " " << f2 << " " <<  WuWu_0(s1,s2)(c1,c2)(f1,f2) << " <----> " << WW_0(s1,s2)(c1,c2)(f1,f2) << " diff " << diff << std::endl;
	    }


  
  std::cout << "FIELD OPT SITE" << std::endl;
  {
    CPSautoView(pfield_w_v,pfield_w,HostRead);
    CPSautoView(pfield_wu_v,pfield_wu,HostRead);

    size_t s=0;
    auto WW_0 = *pfield_w_v.site_ptr(s);
    auto WuWu_0 = *pfield_wu_v.site_ptr(s);

    for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  for(int f1=0;f1<2;f1++)
	    for(int f2=0;f2<2;f2++){
	      auto diff = WW_0(s1,s2)(c1,c2)(f1,f2) - WuWu_0(s1,s2)(c1,c2)(f1,f2);	      
	      std::cout << s1 << " " << s2 << " " << c1 << " " << c2 << " " << f1 << " " << f2 << " " <<  WuWu_0(s1,s2)(c1,c2)(f1,f2) << " <----> " << WW_0(s1,s2)(c1,c2)(f1,f2) << " diff " << diff << std::endl;
	    }
  }


  auto pfield_w_up = linearUnpack(pfield_w);
  auto pfield_wu_up = linearUnpack(pfield_wu);
  assert(pfield_w_up.equals(pfield_wu_up,1e-8,true));
  
  mult(pfield_w, V, W, false, true);
  mult(pfield_wu, V, Wu, false, true);
  pfield_w_up = linearUnpack(pfield_w);
  pfield_wu_up = linearUnpack(pfield_wu);
  assert(pfield_w_up.equals(pfield_wu_up,1e-8,true));


  //Test vector*matrix*vector with W M W and W M V forms
  pfield_w.zero(); pfield_wu.zero();

  mult(pfield_w, W, mf, W, false, true);
  mult(pfield_wu, Wu, mfu, Wu, false, true);
  pfield_w_up = linearUnpack(pfield_w);
  pfield_wu_up = linearUnpack(pfield_wu);
  assert(pfield_w_up.equals(pfield_wu_up,1e-8,true));
  
  mult(pfield_w, W, mf, V, false, true);
  mult(pfield_wu, Wu, mfu, V, false, true);
  pfield_w_up = linearUnpack(pfield_w);
  pfield_wu_up = linearUnpack(pfield_wu);
  assert(pfield_w_up.equals(pfield_wu_up,1e-8,true));
 
  std::cout << "testWtimePackedBasic passed" << std::endl;
}


template<typename A2Apolicies>
void testWtimePackedU1Xsrc(A2AArg a2a_arg, const typename SIMDpolicyBase<4>::ParamType &simd_dims, Lattice &lat){
  std::cout << "Starting testWtimePackedU1Xsrc" << std::endl;

  typedef typename A2Apolicies::FermionFieldType FermionFieldType;
  typedef typename A2Apolicies::ComplexFieldType ComplexFieldType;

  typedef typename A2Apolicies::ScalarFermionFieldType ScalarFermionFieldType;
  typedef typename A2Apolicies::ScalarComplexFieldType ScalarComplexFieldType;

  FermionFieldType tmp_ferm(simd_dims);
  NullObject null_obj;
  ScalarFermionFieldType tmp_ferm_s(null_obj);

  a2a_arg.nl = 0;

  int nf = GJP.Gparity()+1;

  A2AvectorWtimePacked<A2Apolicies> Wu(a2a_arg,simd_dims);
  
  A2AhighModeSourceU1X<A2Apolicies> src_pol;
  src_pol.setHighModeSources(Wu);

  typedef typename getPropagatorFieldType<A2Apolicies>::type PropagatorField;

  {
    //Test unitary source by computing   W(x) W^dag(x)
    PropagatorField pfield(simd_dims);
    mult(pfield,Wu,Wu,false,true);
       
    PropagatorField pfield_expect(simd_dims);
    setUnit(pfield_expect);

    PropagatorField pfield_diff = pfield - pfield_expect;

    std::cout << "W time packed check: " << CPSmatrixFieldNorm2(pfield_diff) << std::endl;
  }
  {
    //Show noise is only proportional to unit flavor matrix
    A2AvectorWtimePacked<A2Apolicies> Wu_shift(a2a_arg,simd_dims);
    for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 1);

    PropagatorField pfield(simd_dims);

    mult(pfield,Wu,Wu_shift,false,true);

    std::cout << "Testing noise structure" << std::endl;

    auto c0 = TraceIndex<2>(pfield);
    std::cout << "sigma0: " << CPSmatrixFieldNorm2(c0) << std::endl;
    
    PropagatorField tmp(simd_dims);

    tmp = pfield; pr(tmp,sigma1);
    auto c1 = TraceIndex<2>(tmp);
    std::cout << "sigma1: " << CPSmatrixFieldNorm2(c1) << std::endl;

    tmp = pfield; pr(tmp,sigma2);
    auto c2 = TraceIndex<2>(tmp);
    std::cout << "sigma2: " << CPSmatrixFieldNorm2(c2) << std::endl;

    tmp = pfield; pr(tmp,sigma3);
    auto c3 = TraceIndex<2>(tmp);
    std::cout << "sigma3: " << CPSmatrixFieldNorm2(c3) << std::endl;
  }

  if(1){
    //Test W(x) W^dag(y)  ->   0  in large hit limit   
    A2AvectorWtimePacked<A2Apolicies> Wu_shift(a2a_arg,simd_dims);
    for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 1);

    PropagatorField pfield_sum(simd_dims);
    pfield_sum.zero();

    PropagatorField pfield(simd_dims);

    int iter=0;
    while(iter < 15000){
      mult(pfield,Wu,Wu_shift,false,true);
      pfield_sum = pfield_sum + pfield;

      pfield = cps::ComplexD(1./(iter+1)) * pfield_sum;

      std::cout << "TEST " << iter << " " << CPSmatrixFieldNorm2(pfield) << std::endl;

      src_pol.setHighModeSources(Wu);
      for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 1);
      ++iter;
    }
  }

  std::cout << "testWtimePackedU1Xsrc passed" << std::endl;
}


template<typename A2Apolicies>
void testWtimePackedU1g0src(A2AArg a2a_arg, const typename SIMDpolicyBase<4>::ParamType &simd_dims, Lattice &lat){
  std::cout << "Starting testWtimePackedU1g0src" << std::endl;

  typedef typename A2Apolicies::FermionFieldType FermionFieldType;
  typedef typename A2Apolicies::ComplexFieldType ComplexFieldType;

  typedef typename A2Apolicies::ScalarFermionFieldType ScalarFermionFieldType;
  typedef typename A2Apolicies::ScalarComplexFieldType ScalarComplexFieldType;

  FermionFieldType tmp_ferm(simd_dims);
  NullObject null_obj;
  ScalarFermionFieldType tmp_ferm_s(null_obj);

  a2a_arg.nl = 0;

  int nf = GJP.Gparity()+1;

  A2AvectorWtimePacked<A2Apolicies> Wu(a2a_arg,simd_dims);
  
  A2AhighModeSourceU1g0<A2Apolicies> src_pol;
  src_pol.setHighModeSources(Wu);

  typedef typename getPropagatorFieldType<A2Apolicies>::type PropagatorField;

  {
    //Test unitary source by computing   W(x) W^dag(x)
    PropagatorField pfield(simd_dims);
    mult(pfield,Wu,Wu,false,true);
       
    PropagatorField pfield_expect(simd_dims);
    setUnit(pfield_expect);

    PropagatorField pfield_diff = pfield - pfield_expect;

    std::cout << "W time packed check: " << CPSmatrixFieldNorm2(pfield_diff) << std::endl;
  }
  {
    //Show noise is only proportional to unit flavor matrix
    A2AvectorWtimePacked<A2Apolicies> Wu_shift(a2a_arg,simd_dims);
    for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 1);

    PropagatorField pfield(simd_dims);

    mult(pfield,Wu,Wu_shift,false,true);

    std::cout << "Testing noise structure" << std::endl;

    auto c0 = TraceIndex<2>(pfield);
    std::cout << "sigma0: " << CPSmatrixFieldNorm2(c0) << std::endl;
    
    PropagatorField tmp(simd_dims);

    tmp = pfield; pr(tmp,sigma1);
    auto c1 = TraceIndex<2>(tmp);
    std::cout << "sigma1: " << CPSmatrixFieldNorm2(c1) << std::endl;

    tmp = pfield; pr(tmp,sigma2);
    auto c2 = TraceIndex<2>(tmp);
    std::cout << "sigma2: " << CPSmatrixFieldNorm2(c2) << std::endl;

    tmp = pfield; pr(tmp,sigma3);
    auto c3 = TraceIndex<2>(tmp);
    std::cout << "sigma3: " << CPSmatrixFieldNorm2(c3) << std::endl;
  }

  if(1){
    //Test W(x) W^dag(y)  ->   0  in large hit limit   
    A2AvectorWtimePacked<A2Apolicies> Wu_shift(a2a_arg,simd_dims);
    for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 1);

    PropagatorField pfield_sum(simd_dims);
    pfield_sum.zero();

    PropagatorField pfield(simd_dims);

    int iter=0;
    while(iter < 15000){
      mult(pfield,Wu,Wu_shift,false,true);
      pfield_sum = pfield_sum + pfield;

      pfield = cps::ComplexD(1./(iter+1)) * pfield_sum;

      std::cout << "TEST " << iter << " " << CPSmatrixFieldNorm2(pfield) << std::endl;

      src_pol.setHighModeSources(Wu);
      for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 1);
      ++iter;
    }
  }

  std::cout << "testWtimePackedU1g0src passed" << std::endl;
}

template<typename ViewType>
cps::ComplexD matAccess(int srow, int scol, int crow, int ccol, int frow, int fcol, int x4d, int hit, ViewType &W){
  assert(W.getNlowModes() == 0);
  int x[4];
  FourDpolicy<> pp;
  pp.siteUnmap(x4d,x);
  
  int x4ds = W.getHighMode(0).siteMap(x);
  int lane = W.getHighMode(0).SIMDmap(x);

  int mode = W.indexMap(hit, ccol+3*scol, fcol);
  auto v =*( W.getHighMode(mode).site_ptr(x4ds,frow) + crow+3*srow );
  return v.getlane(lane);
}
double norm2(const CPSspinMatrix<cps::ComplexD> &m){
  double out = 0;
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      out += norm(m(i,j));
  return out;
}



template<typename A2Apolicies>
void testWtimePackedU1Hsrc(A2AArg a2a_arg, const typename SIMDpolicyBase<4>::ParamType &simd_dims, Lattice &lat){
  std::cout << "Starting testWtimePackedU1Hsrc" << std::endl;

  typedef typename A2Apolicies::FermionFieldType FermionFieldType;
  typedef typename A2Apolicies::ComplexFieldType ComplexFieldType;

  typedef typename A2Apolicies::ScalarFermionFieldType ScalarFermionFieldType;
  typedef typename A2Apolicies::ScalarComplexFieldType ScalarComplexFieldType;

  FermionFieldType tmp_ferm(simd_dims);
  NullObject null_obj;
  ScalarFermionFieldType tmp_ferm_s(null_obj);

  a2a_arg.nl = 0;

  int nf = GJP.Gparity()+1;

  A2AvectorWtimePacked<A2Apolicies> Wu(a2a_arg,simd_dims);
  
  A2AhighModeSourceU1H<A2Apolicies> src_pol;
  src_pol.setHighModeSources(Wu);

  typedef typename getPropagatorFieldType<A2Apolicies>::type PropagatorField;

  {
    //Test unitary source by computing   W(x) W^dag(x)
    PropagatorField pfield(simd_dims);
    mult(pfield,Wu,Wu,false,true);
       
    PropagatorField pfield_expect(simd_dims);
    setUnit(pfield_expect);

    PropagatorField pfield_diff = pfield - pfield_expect;

    std::cout << "W time packed check: " << CPSmatrixFieldNorm2(pfield_diff) << std::endl;
  }

  typedef CPSspinMatrix<cps::ComplexD> SpinMat;
  SpinMat C; C.unit(); C.gl(1).gl(3); //C=-gY gT = gT gY
  SpinMat X = C; X.gr(-5);
  SpinMat one; one.unit();
  cps::ComplexD _i(0,1);
  SpinMat Pplus = 0.5*(one + _i*X);
  SpinMat Pminus = 0.5*(one - _i*X);
  SpinMat mXPplus = -X*Pplus;
  SpinMat mXPminus = -X*Pminus;

  {
    //A2AvectorWtimePacked<A2Apolicies> Wu_tmp(a2a_arg,simd_dims);

    std::cout << "P_+ " << std::endl << Pplus << std::endl;
    std::cout << "P_- " << std::endl << Pminus << std::endl;

    std::cout << "P_+ + P- (expect 1) : " << std::endl << (Pplus+Pminus) << std::endl;

    SpinMat X2 = X*X;
    std::cout << "X^2 (expect -1): " << std::endl << X2 << std::endl << "norm2: " << norm2(X2) << std::endl;    

    SpinMat PpPm = Pplus * Pminus;
    std::cout << "P_+ P_- (expect 0): " << std::endl << PpPm << std::endl << "norm2: " << norm2(PpPm) << std::endl;

    SpinMat PmPp = Pminus * Pplus;
    std::cout << "P_- P_+ (expect 0): " << std::endl << PmPp << std::endl << "norm2: " << norm2(PmPp) << std::endl;

    SpinMat test = Dagger(Pminus) - Pminus;
    std::cout << "P_-^dag - P_- (expect 0): "<< std::endl << test << std::endl;

    test = Dagger(Pplus) - Pplus;
    std::cout << "P_+^dag - P_+ (expect 0): "<< std::endl << test << std::endl;


    test = cconj(Pminus) - Pplus;
    std::cout << "P_-^* - P_+ (expect 0): "<< std::endl << test << std::endl;

    test = cconj(Pplus) - Pminus;
    std::cout << "P_+^* - P_- (expect 0): "<< std::endl << test << std::endl;

    {
      //Hit with P_+. Even sites should be flavor diagonal and odd sites flavor non-diagonal
      CPSautoView(Wfrom_v, Wu, HostRead);
      
      SpinMat st_even[2][2], st_odd[2][2];
      for(int f1=0;f1<2;f1++){
	for(int f2=0;f2<2;f2++){	      
	  st_even[f1][f2].zero();	      
	  st_odd[f1][f2].zero();
	}
      }    

      int x4d_even=0, x4d_odd = 1;
      int c1=0,c2=0; //color diagonal
      for(int f1=0;f1<2;f1++){
	for(int f2=0;f2<2;f2++){	      
	  for(int s1=0; s1<4; s1++){
	    for(int s2=0; s2<4; s2++){
	      for(int s3=0;s3<4; s3++){
		st_even[f1][f2](s1,s2) += Pplus(s1,s3) * matAccess(s3, s2,  c1,c2,f1,f2,x4d_even,0,Wfrom_v);
		st_odd[f1][f2](s1,s2) += Pplus(s1,s3) * matAccess(s3, s2,  c1,c2,f1,f2,x4d_odd,0,Wfrom_v);
	      }
	    }
	  }
	}
      }
      std::cout << "Test P+ projection,  even site (expect flavor diag):\n";
      std::cout << norm2(st_even[0][0]) << " " << norm2(st_even[0][1]) << std::endl << norm2(st_even[1][0]) << " " << norm2(st_even[1][1]) << std::endl;
      std::cout << "Test P+ projection,  odd site (expect flavor off-diag):\n";
      std::cout << norm2(st_odd[0][0]) << " " << norm2(st_odd[0][1]) << std::endl << norm2(st_odd[1][0]) << " " << norm2(st_odd[1][1]) << std::endl;

      //Hit with P_-, both should go to zero
      SpinMat zeven[2][2], zodd[2][2];
      for(int f1=0;f1<2;f1++)
	for(int f2=0;f2<2;f2++){
	  zeven[f1][f2] = Pminus * st_even[f1][f2];
	  zodd[f1][f2] = Pminus * st_odd[f1][f2];
	}
      
      std::cout << "Test P-P+ projection,  even site (expect zero):\n";
      std::cout << norm2(zeven[0][0]) << " " << norm2(zeven[0][1]) << std::endl << norm2(zeven[1][0]) << " " << norm2(zeven[1][1]) << std::endl;
      std::cout << "Test P-P+ projection,  odd site (expect zero):\n";
      std::cout << norm2(zodd[0][0]) << " " << norm2(zodd[0][1]) << std::endl << norm2(zodd[1][0]) << " " << norm2(zodd[1][1]) << std::endl;

      //Check sites are unitary and that odd * even^dag = even * odd^dag = 0
      for(int f1=0;f1<2;f1++){
	for(int f2=0;f2<2;f2++){	      
	  for(int s1=0; s1<4; s1++){
	    for(int s2=0; s2<4; s2++){
	      st_even[f1][f2](s1,s2) = matAccess(s1, s2,  c1,c2,f1,f2,x4d_even,0,Wfrom_v);
	      st_odd[f1][f2](s1,s2) = matAccess(s1, s2,  c1,c2,f1,f2,x4d_odd,0,Wfrom_v);
	    }
	  }
	}
      }
      SpinMat zeven2[2][2], zodd2[2][2], zevenodddag[2][2], zoddevendag[2][2];
      for(int f1=0;f1<2;f1++){
	for(int f2=0;f2<2;f2++){
	  zeven2[f1][f2].zero(); 	  
	  zodd2[f1][f2].zero(); 
	  zevenodddag[f1][f2].zero(); 
	  zoddevendag[f1][f2].zero(); 

	  for(int f3=0;f3<2;f3++){
	    zeven2[f1][f2] = zeven2[f1][f2] + st_even[f1][f3] * Dagger(st_even[f2][f3]);
	    zodd2[f1][f2] = zodd2[f1][f2] + st_odd[f1][f3] * Dagger(st_odd[f2][f3]);
	    zevenodddag[f1][f2] = zevenodddag[f1][f2] + st_even[f1][f3] * Dagger(st_odd[f2][f3]);
	    zoddevendag[f1][f2] = zoddevendag[f1][f2] + st_odd[f1][f3] * Dagger(st_even[f2][f3]);
	  }
	}
      }
      std::cout << "Test |even site|^2 (expect 1): " << std::endl;
      std::cout << norm2(zeven2[0][0]) << " " << norm2(zeven2[0][1]) << std::endl << norm2(zeven2[1][0]) << " " << norm2(zeven2[1][1]) << std::endl;
      std::cout << "Test |odd site|^2 (expect 1): " << std::endl;
      std::cout << norm2(zodd2[0][0]) << " " << norm2(zodd2[0][1]) << std::endl << norm2(zodd2[1][0]) << " " << norm2(zodd2[1][1]) << std::endl;
      std::cout << "Test (even site)*(odd site)^dag (expect 0): " << std::endl;
      std::cout << norm2(zevenodddag[0][0]) << " " << norm2(zevenodddag[0][1]) << std::endl << norm2(zevenodddag[1][0]) << " " << norm2(zevenodddag[1][1]) << std::endl;
      std::cout << "Test (odd site)*(even site)^dag (expect 0): " << std::endl;
      std::cout << norm2(zoddevendag[0][0]) << " " << norm2(zoddevendag[0][1]) << std::endl << norm2(zoddevendag[1][0]) << " " << norm2(zoddevendag[1][1]) << std::endl;

      std::cout << "|even site|^2(0,0): " << std::endl << zeven2[0][0] << std::endl;
      
      SpinMat etest1, etest2, etest3;
      etest1 = Dagger(st_even[0][0]);
      std::cout << "even(0,0): " << std::endl << st_even[0][0] << std::endl;
      std::cout << "even^dagsc(0,0): " << std::endl << etest1 << std::endl;

      etest2 = Pplus * st_even[0][0] - st_even[0][0];
      etest3 = Pplus * etest1 - etest1;
      
      std::cout << "P_+ even(0,0) - even(0,0)  (expect 0): " << std::endl << etest2 << std::endl;
      std::cout << "P_+ even^dagsc(0,0) - even^dagsc(0,0)  (expect 0): " << std::endl << etest3 << std::endl;

      SpinMat etest4, etest5, etest6;
      etest4 = Dagger(st_even[0][1]);
      std::cout << "even(0,1): " << std::endl << st_even[0][1] << std::endl;
      std::cout << "even^dagsc(0,1): " << std::endl << etest4 << std::endl;

      etest5 = Pminus * st_even[0][1] - st_even[0][1];
      etest6 = Pminus * etest4 - etest4;
      
      std::cout << "P_- even(0,1) - even(0,1)  (expect 0): " << std::endl << etest5 << std::endl;
      std::cout << "P_- even^dagsc(0,1) - even^dagsc(0,1)  (expect 0): " << std::endl << etest6 << std::endl;

      SpinMat etest7 = st_even[0][0]*Dagger(st_even[0][0]);
      SpinMat etest8 = st_even[0][1]*Dagger(st_even[0][1]);
      SpinMat etest9 = etest7 + etest8;
      std::cout << "even(0,0)*even^dagsc(0,0) + even(0,1)*even^dagsc(0,1) = " << std::endl <<
	etest7 << std::endl << "+" << std::endl << etest8  << std::endl << "=" << std::endl << etest9 << std::endl;

    }	    

  }




    //Test projector structure of noise. Hitting it once with P_+ then again with P_- should give 0



  {
    //Show noise is only proportional to unit flavor matrix and exists only on even displacements >= 2
    A2AvectorWtimePacked<A2Apolicies> Wu_shift(a2a_arg,simd_dims);
    for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 1);

    PropagatorField pfield(simd_dims);

    //mult(pfield,Wu,Wu_shift,false,true); //This is \sum_x W_xi W^*_xj = \sum_x W^T_ix W^dag_jx    x is space-time,spin,color,flavor
    //Want  \sum_x W^*_xi W_xj = \sum_x W^dag_ix W_xj   -------> FIX FOR OTHER SRCS
    //IS THE FIELD COMPONENT DEFINITELY THE ROW INDEX? YES  M^-1_x W_xj W^dag_j y = M^-1_y

    //WHY IS THIS FAILING??
    //TEST USING PROJECTION MATRICES THAT THE STRUCTURE IS RIGHT
    
    mult(pfield,Wu,Wu_shift,true,false);

    std::cout << "Testing noise structure shift 1, expect zero" << std::endl;

    auto c0 = TraceIndex<2>(pfield);
    std::cout << "sigma0: " << CPSmatrixFieldNorm2(c0) << std::endl;
    
    PropagatorField tmp(simd_dims);

    tmp = pfield; pr(tmp,sigma1);
    auto c1 = TraceIndex<2>(tmp);
    std::cout << "sigma1: " << CPSmatrixFieldNorm2(c1) << std::endl;

    tmp = pfield; pr(tmp,sigma2);
    auto c2 = TraceIndex<2>(tmp);
    std::cout << "sigma2: " << CPSmatrixFieldNorm2(c2) << std::endl;

    tmp = pfield; pr(tmp,sigma3);
    auto c3 = TraceIndex<2>(tmp);
    std::cout << "sigma3: " << CPSmatrixFieldNorm2(c3) << std::endl;

    for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 2);

    //mult(pfield,Wu,Wu_shift,false,true);
    mult(pfield,Wu,Wu_shift,true,false);

    std::cout << "Testing noise structure shift 2, expect non-zero" << std::endl;

    c0 = TraceIndex<2>(pfield);
    std::cout << "sigma0: " << CPSmatrixFieldNorm2(c0) << std::endl;
    
    tmp = pfield; pr(tmp,sigma1);
    c1 = TraceIndex<2>(tmp);
    std::cout << "sigma1: " << CPSmatrixFieldNorm2(c1) << std::endl;

    tmp = pfield; pr(tmp,sigma2);
    c2 = TraceIndex<2>(tmp);
    std::cout << "sigma2: " << CPSmatrixFieldNorm2(c2) << std::endl;

    tmp = pfield; pr(tmp,sigma3);
    c3 = TraceIndex<2>(tmp);
    std::cout << "sigma3: " << CPSmatrixFieldNorm2(c3) << std::endl;
  }

  if(0){
    //Test W(x) W^dag(y)  ->   0  in large hit limit   
    A2AvectorWtimePacked<A2Apolicies> Wu_shift(a2a_arg,simd_dims);
    for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 1);

    PropagatorField pfield_sum(simd_dims);
    pfield_sum.zero();

    PropagatorField pfield(simd_dims);

    int iter=0;
    while(iter < 15000){
      //mult(pfield,Wu,Wu_shift,false,true);
      mult(pfield,Wu,Wu_shift,true,false);
      pfield_sum = pfield_sum + pfield;

      pfield = cps::ComplexD(1./(iter+1)) * pfield_sum;

      std::cout << "TEST " << iter << " " << CPSmatrixFieldNorm2(pfield) << std::endl;

      src_pol.setHighModeSources(Wu);
      for(int i=0;i<Wu.getNhighModes();i++) cyclicPermute( Wu_shift.getWh(i), Wu.getWh(i), 0, +1, 1);
      ++iter;
    }
  }

  std::cout << "testWtimePackedU1Hsrc passed" << std::endl;
}




CPS_END_NAMESPACE
