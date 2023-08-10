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

  //Test vector*vector operation with WW and VW forms
  A2AvectorV<A2Apolicies> V(a2a_arg,simd_dims);
  V.testRandom();

  typedef typename getPropagatorFieldType<A2Apolicies>::type PropagatorField;
  PropagatorField pfield_w(simd_dims), pfield_wu(simd_dims);

  mult(pfield_w, W, W, false, true);
  mult(pfield_wu, Wu, Wu, false, true);
  auto pfield_w_up = linearUnpack(pfield_w);
  auto pfield_wu_up = linearUnpack(pfield_wu);
  assert(pfield_w_up.equals(pfield_wu_up,1e-8,true));
  
  mult(pfield_w, V, W, false, true);
  mult(pfield_wu, V, Wu, false, true);
  pfield_w_up = linearUnpack(pfield_w);
  pfield_wu_up = linearUnpack(pfield_wu);
  assert(pfield_w_up.equals(pfield_wu_up,1e-8,true));

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




 
  /* //Test vector*matrix*vector with W M W and W M V forms */
  
  /* A2Aparams a2a_params(a2a_arg); */
  /* A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf; */
  /* mf.setup(a2a_params,a2a_params,0,0); */
  /* mf.testRandom(); */

  /* pfield_w.zero(); pfield_wsc.zero(); */

  /* mult(pfield_w, W, mf, W, false, true); */
  /* mult(pfield_wsc, Wsc, mf, Wsc, false, true); */
  /* pfield_w_up = linearUnpack(pfield_w); */
  /* pfield_wsc_up = linearUnpack(pfield_wsc); */
  /* assert(pfield_w_up.equals(pfield_wsc_up,1e-8,true)); */
  
  /* mult(pfield_w, W, mf, V, false, true); */
  /* mult(pfield_wsc, Wsc, mf, V, false, true); */
  /* pfield_w_up = linearUnpack(pfield_w); */
  /* pfield_wsc_up = linearUnpack(pfield_wsc); */
  /* assert(pfield_w_up.equals(pfield_wsc_up,1e-8,true)); */
 

  std::cout << "testWunitaryBasic passed" << std::endl;
}


CPS_END_NAMESPACE
