#pragma once

CPS_START_NAMESPACE

template<typename StandardA2Apolicies, typename GridA2Apolicies>
void testMesonFieldNormGridStd(const A2AArg &a2a_args, const double tol){
  std::cout << "Running testMesonFieldNorm" << std::endl;
  A2Aparams params(a2a_args);
  std::vector<A2AmesonField<StandardA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_std(1);
  mf_std[0].setup(params,params,0,0);
  mf_std[0].testRandom();

  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>  > mf_grid(1);
  mf_grid[0].setup(params,params,0,0);
  copy(mf_grid,mf_std);

  assert(compare(mf_grid,mf_std,tol));

  double n1 = mf_grid[0].norm2(), n2 = mf_std[0].norm2();
  if(fabs(n1-n2) > tol){
    std::cout << "testMesonFieldNorm failed: " << n1 << " " << n2 << " diff " << n1 - n2 << std::endl;
    exit(1);
  }
  

  std::cout << "testMesonFieldNorm passed" << std::endl;
}

template<typename ScalarA2Apolicies>
void testMesonFieldReadWrite(const A2AArg &a2a_args){
  std::cout << "Running testMesonFieldReadWrite" << std::endl;
  A2AvectorWfftw<ScalarA2Apolicies> W(a2a_args);
  A2AvectorVfftw<ScalarA2Apolicies> V(a2a_args);
  
  A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf(W,V);
  mf.testRandom();
  
  {
    mf.write("mesonfield.dat",FP_IEEE64BIG);
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfr;
    mfr.read("mesonfield.dat");
    assert( mfr.equals(mf,1e-18,true));
    if(!UniqueID()) printf("Passed mf single IO test\n");
  }
  {
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfa;
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfb;
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfc;
    mfa.setup(W,V,0,0);
    mfb.setup(W,V,1,1);
    mfc.setup(W,V,2,2);		
      
    mfa.testRandom();
    mfb.testRandom();
    mfc.testRandom();

    std::ofstream *fp = !UniqueID() ? new std::ofstream("mesonfield_many.dat") : NULL;

    mfa.write(fp,FP_IEEE64BIG);
    mfb.write(fp,FP_IEEE64LITTLE);
    mfc.write(fp,FP_IEEE64BIG);

    if(!UniqueID()) fp->close();

    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfra;
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfrb;
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfrc;

    std::ifstream *ifp = !UniqueID() ? new std::ifstream("mesonfield_many.dat") : NULL;

    mfra.read(ifp);
    mfrb.read(ifp);
    mfrc.read(ifp);

    if(!UniqueID()) ifp->close();

    assert( mfra.equals(mfa,1e-18,true) );
    assert( mfrb.equals(mfb,1e-18,true) );
    assert( mfrc.equals(mfc,1e-18,true) );
    if(!UniqueID()) printf("Passed mf multi IO test\n");
    if(fp) delete fp;
    if(ifp) delete ifp;
  }
  {
    std::vector< A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mfv(3);
    for(int i=0;i<3;i++){
      mfv[i].setup(W,V,i,i);
      mfv[i].testRandom();
    }
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::write("mesonfield_vec.dat", mfv, FP_IEEE64LITTLE);
	
    std::vector< A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mfrv;
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::read("mesonfield_vec.dat", mfrv);

    for(int i=0;i<3;i++)
      assert( mfrv[i].equals(mfv[i], 1e-18, true) );
    if(!UniqueID()) printf("Passed mf vector IO test\n");
  }
  std::cout << "Passed testMesonFieldReadWrite" << std::endl;
}	


template<typename A2Apolicies>
void testMesonFieldTraceSingle(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting testMesonFieldTraceSingle" << std::endl;
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_grid;
  mf_grid.setup(a2a_args,a2a_args,0,0);

  LRG.AssignGenerator(0); //always uses the RNG at coord 0 on node 0 - should always be the same one!
  mf_grid.testRandom();

  typedef typename A2Apolicies::ScalarComplexType mf_Complex;  
  mf_Complex fast = trace(mf_grid);
  mf_Complex slow = trace_slow(mf_grid);

  bool fail = false;
  if(!UniqueID()) printf("Trace Fast (%g,%g) Slow (%g,%g) Diff (%g,%g)\n",fast.real(),fast.imag(), slow.real(),slow.imag(), fast.real()-slow.real(), fast.imag()-slow.imag());
  double rdiff = fabs(fast.real()-slow.real());
  double idiff = fabs(fast.imag()-slow.imag());
  if(rdiff > tol|| idiff > tol){
    fail = true;
  }
  if(fail) ERR.General("","","MF single trace test failed\n");
  else if(!UniqueID()) printf("MF single trace pass\n");

  //Manually test node number independence of the node distributed trace
  std::vector<typename A2Apolicies::ScalarComplexType> into;
  std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > m(3);
  for(int i=0;i<m.size();i++){
    m[i].setup(a2a_args,a2a_args,0,0);
    LRG.AssignGenerator(0);
    m[i].testRandom();
  }
  trace(into,m);

  if(!UniqueID()){
    printf("Distributed traces:");
    for(int i=0;i<into.size();i++){
      printf(" (%g,%g)",into[i].real(),into[i].imag());
    }
    printf("\n");
  } 
  std::cout << "testMesonFieldTraceSingle passed" << std::endl;
}


template<typename A2Apolicies>
void testMesonFieldTraceSingleTblock(A2AArg a2a_args, const double tol){
  std::cout << "Starting testMesonFieldTraceSingleTblock" << std::endl;
  a2a_args.src_width = 2;
  
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_grid;
  mf_grid.setup(a2a_args,a2a_args,1,1); //block index doesn't match timeslice

  LRG.AssignGenerator(0); //always uses the RNG at coord 0 on node 0 - should always be the same one!
  mf_grid.testRandom();

  typedef typename A2Apolicies::ScalarComplexType mf_Complex;  
  mf_Complex fast = trace(mf_grid);
  mf_Complex slow = trace_slow(mf_grid);

  bool fail = false;
  if(!UniqueID()) printf("Trace Fast (%g,%g) Slow (%g,%g) Diff (%g,%g)\n",fast.real(),fast.imag(), slow.real(),slow.imag(), fast.real()-slow.real(), fast.imag()-slow.imag());
  double rdiff = fabs(fast.real()-slow.real());
  double idiff = fabs(fast.imag()-slow.imag());
  if(rdiff > tol|| idiff > tol){
    fail = true;
  }
  if(fail) ERR.General("","","MF single trace test failed\n");
  else if(!UniqueID()) printf("MF single trace pass\n");

  //Manually test node number independence of the node distributed trace
  std::vector<typename A2Apolicies::ScalarComplexType> into;
  std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > m(3);
  for(int i=0;i<m.size();i++){
    m[i].setup(a2a_args,a2a_args,0,0);
    LRG.AssignGenerator(0);
    m[i].testRandom();
  }
  trace(into,m);

  if(!UniqueID()){
    printf("Distributed traces:");
    for(int i=0;i<into.size();i++){
      printf(" (%g,%g)",into[i].real(),into[i].imag());
    }
    printf("\n");
  } 
  std::cout << "testMesonFieldTraceSingleTblock passed" << std::endl;
}






template<typename A2Apolicies>
void testMesonFieldTraceProduct(const A2AArg &a2a_args, const double tol){
  std::cout << "Testing mesonfield trace-product" << std::endl;
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf1,mf2;
  mf1.setup(a2a_args,a2a_args,0,0);
  mf2.setup(a2a_args,a2a_args,0,0);

  LRG.AssignGenerator(0); //always uses the RNG at coord 0 on node 0 - should always be the same one!
  mf1.testRandom();
  mf2.testRandom();

  typedef typename A2Apolicies::ScalarComplexType mf_Complex;  
  mf_Complex fast = trace_cpu(mf1,mf2); //cpu version is the same as the generic version
  mf_Complex slow = trace_slow(mf1,mf2);

  bool fail = false;
  if(!UniqueID()) printf("Trace Fast (%g,%g) Slow (%g,%g) Diff (%g,%g)\n",fast.real(),fast.imag(), slow.real(),slow.imag(), fast.real()-slow.real(), fast.imag()-slow.imag());
  double rdiff = fabs(fast.real()-slow.real());
  double idiff = fabs(fast.imag()-slow.imag());
  if(rdiff > tol|| idiff > tol){
    fail = true;
  }
  if(fail) ERR.General("","","MF trace-product test failed\n"); 
  else if(!UniqueID()) printf("MF trace-product pass\n");

#ifdef GPU_VEC
  //Test the GPU version
  fast = 0;
  fast = trace_gpu(mf1,mf2);

  fail = false;
  if(!UniqueID()) printf("GPU Trace Fast (%g,%g) Slow (%g,%g) Diff (%g,%g)\n",fast.real(),fast.imag(), slow.real(),slow.imag(), fast.real()-slow.real(), fast.imag()-slow.imag());
  rdiff = fabs(fast.real()-slow.real());
  idiff = fabs(fast.imag()-slow.imag());
  if(rdiff > tol|| idiff > tol){
    fail = true;
  }
  if(fail) ERR.General("","","MF GPU trace-product test failed\n"); 
  else if(!UniqueID()) printf("MF GPU trace-product pass\n");

  //Test the GPU version with precomputed views
  {
    CPSautoView(mf1_v, mf1, DeviceRead);
    CPSautoView(mf2_v, mf2, DeviceRead);
    
    fast = 0;
    fast = trace_gpu(mf1,mf2, &mf1_v, &mf2_v);

    //Do it twice to test to ensure the view is maintained
    fast = 0;
    fast = trace_gpu(mf1,mf2, &mf1_v, &mf2_v);
  }

  fail = false;
  if(!UniqueID()) printf("GPU Trace Fast (%g,%g) Slow (%g,%g) Diff (%g,%g)\n",fast.real(),fast.imag(), slow.real(),slow.imag(), fast.real()-slow.real(), fast.imag()-slow.imag());
  rdiff = fabs(fast.real()-slow.real());
  idiff = fabs(fast.imag()-slow.imag());
  if(rdiff > tol|| idiff > tol){
    fail = true;
  }
  if(fail) ERR.General("","","MF GPU view trace-product test failed\n"); 
  else if(!UniqueID()) printf("MF GPU view trace-product pass\n");
#endif
}


template<typename A2Apolicies>
void testMesonFieldTraceProductTblock(A2AArg a2a_args, const double tol){
  std::cout << "Testing mesonfield trace-product tblock" << std::endl;
  a2a_args.src_width = 2;

  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf1,mf2;
  mf1.setup(a2a_args,a2a_args,1,1); //ensure tblock != t
  mf2.setup(a2a_args,a2a_args,1,1);

  LRG.AssignGenerator(0); //always uses the RNG at coord 0 on node 0 - should always be the same one!
  mf1.testRandom();
  mf2.testRandom();

  typedef typename A2Apolicies::ScalarComplexType mf_Complex;  
  mf_Complex fast = trace_cpu(mf1,mf2); //cpu version is the same as the generic version
  mf_Complex slow = trace_slow(mf1,mf2);

  bool fail = false;
  if(!UniqueID()) printf("Trace Fast (%g,%g) Slow (%g,%g) Diff (%g,%g)\n",fast.real(),fast.imag(), slow.real(),slow.imag(), fast.real()-slow.real(), fast.imag()-slow.imag());
  double rdiff = fabs(fast.real()-slow.real());
  double idiff = fabs(fast.imag()-slow.imag());
  if(rdiff > tol|| idiff > tol){
    fail = true;
  }
  if(fail) ERR.General("","","MF trace-product test failed\n"); 
  else if(!UniqueID()) printf("MF trace-product pass\n");

#ifdef GPU_VEC
  //Test the GPU version
  fast = 0;
  fast = trace_gpu(mf1,mf2);

  fail = false;
  if(!UniqueID()) printf("GPU Trace Fast (%g,%g) Slow (%g,%g) Diff (%g,%g)\n",fast.real(),fast.imag(), slow.real(),slow.imag(), fast.real()-slow.real(), fast.imag()-slow.imag());
  rdiff = fabs(fast.real()-slow.real());
  idiff = fabs(fast.imag()-slow.imag());
  if(rdiff > tol|| idiff > tol){
    fail = true;
  }
  if(fail) ERR.General("","","MF GPU trace-product test failed\n"); 
  else if(!UniqueID()) printf("MF GPU trace-product pass\n");

  //Test the GPU version with precomputed views
  {
    CPSautoView(mf1_v, mf1, DeviceRead);
    CPSautoView(mf2_v, mf2, DeviceRead);
    
    fast = 0;
    fast = trace_gpu(mf1,mf2, &mf1_v, &mf2_v);

    //Do it twice to test to ensure the view is maintained
    fast = 0;
    fast = trace_gpu(mf1,mf2, &mf1_v, &mf2_v);
  }

  fail = false;
  if(!UniqueID()) printf("GPU Trace Fast (%g,%g) Slow (%g,%g) Diff (%g,%g)\n",fast.real(),fast.imag(), slow.real(),slow.imag(), fast.real()-slow.real(), fast.imag()-slow.imag());
  rdiff = fabs(fast.real()-slow.real());
  idiff = fabs(fast.imag()-slow.imag());
  if(rdiff > tol|| idiff > tol){
    fail = true;
  }
  if(fail) ERR.General("","","MF GPU view trace-product test failed\n"); 
  else if(!UniqueID()) printf("MF GPU view trace-product pass\n");
#endif
  std::cout << "testMesonFieldTraceProductTblock passed" << std::endl;
}





template<typename A2Apolicies>
void testMesonFieldTraceProductAllTimes(const A2AArg &a2a_args, const double tol){
  std::cout << "Testing mesonfield all-times trace-product" << std::endl;
  int Lt = GJP.Tnodes()*GJP.TnodeSites();

  LRG.AssignGenerator(0); //always uses the RNG at coord 0 on node 0 - should always be the same one!
  
  std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf1(Lt),mf2(Lt);
  for(int t=0;t<Lt;t++){
    mf1[t].setup(a2a_args,a2a_args,t,t);
    mf2[t].setup(a2a_args,a2a_args,t,t);
    mf1[t].testRandom();
    mf2[t].testRandom();  
  }

  typedef typename A2Apolicies::ScalarComplexType mf_Complex;
  fMatrix<mf_Complex> ref(Lt,Lt);
  for(int t1=0;t1<Lt;t1++)
    for(int t2=0;t2<Lt;t2++)
      ref(t1,t2) = trace_slow(mf1[t1],mf2[t2]);
    
  fMatrix<mf_Complex> got;
  trace(got, mf1,mf2);

  bool fail = false;
  for(int t1=0;t1<Lt;t1++)
    for(int t2=0;t2<Lt;t2++){
      const mf_Complex &r = ref(t1,t2);
      const mf_Complex &g = got(t1,t2);
      double rdiff = fabs(g.real()-r.real());
      double idiff = fabs(g.imag()-r.imag());
      std::cout << t1 << " " << t2 << " " << r << " " << g << " : " << rdiff << " " << idiff << std::endl;
      
      if(rdiff > tol|| idiff > tol){
	fail = true;
      }
    }
  if(fail) ERR.General("","","MF all-times trace-product test failed\n"); 
  else if(!UniqueID()) printf("MF all-times trace-product pass\n");
}


template<typename MFtype, typename ScalarComplexType>
void checkunpacked(const MFtype &mf, ScalarComplexType const* into, double tol, const std::string &descr){
  int rows_full = mf.getNrowsFull();
  int cols_full = mf.getNcolsFull();
   
  bool fail = false;
  CPSautoView(mf_v,mf,HostRead);
  for(int i=0;i<rows_full;i++){
    for(int j=0;j<cols_full;j++){
      Complex got = into[j+cols_full*i];
      Complex expect = mf_v.elem(i,j);
      
      double rdiff = fabs(got.real()-expect.real());
      double idiff = fabs(got.imag()-expect.imag());
      
      if(rdiff > tol|| idiff > tol){
	std::cout << i << " " << j << " " << got << " " << expect << " : " << rdiff << " " << idiff << std::endl;
	fail = true;
      }
    }
  }
  if(fail){    
    std::cout  << "MF unpack test " << descr << " failed" << std::endl;
    assert(0);
  }
  std::cout  << "MF unpack test " << descr << " passed" << std::endl;
}


template<typename MFtype>
void checkpacked(const MFtype &got, const MFtype &expect, double tol, const std::string &descr){
  if(!expect.equals(got,tol,true)){
    std::cout << "MF pack test " << descr << " failed" << std::endl;
    assert(0);
  }else{
    std::cout << "MF pack test " << descr << " pass" << std::endl;
  }
}


template<typename A2Apolicies>
void testMesonFieldUnpackPack(const A2AArg &a2a_args, const double tol){
  std::cout << "Testing mesonfield unpack, pack" << std::endl;

  LRG.AssignGenerator(0); //always uses the RNG at coord 0 on node 0 - should always be the same one!

  //Check a WV type
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf1;
  mf1.setup(a2a_args,a2a_args,0,0);
  mf1.testRandom();
  
  typedef typename A2Apolicies::ScalarComplexType Complex;

  int rows_full = mf1.getNrowsFull();
  int cols_full = mf1.getNcolsFull();

  size_t into_size = rows_full * cols_full * sizeof(Complex);
  Complex* into = (Complex*)malloc(into_size);
  Complex* device_into = (Complex*)device_alloc_check(into_size);
  
  mf1.unpack(into);
  checkunpacked(mf1, into, tol, "WV");
  
  mf1.unpack_device(device_into);
  memset(into,0,into_size);  
  copy_device_to_host(into, device_into, into_size);
  
  checkunpacked(mf1, into, tol, "WV device");

  //Do a test once with the view precreated
  {
    device_memset(device_into,0,into_size);
    CPSautoView(mf1_v,mf1,DeviceRead);

    mf1.unpack_device(device_into, &mf1_v);
    memset(into,0,into_size);  
    copy_device_to_host(into, device_into, into_size);
    
    checkunpacked(mf1, into, tol, "WV device with view");
  }

  
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf1_p;
  mf1_p.setup(a2a_args,a2a_args,0,0); 
  mf1_p.pack(into);
  checkpacked(mf1_p, mf1, tol, "WV");
  
  mf1_p.zero();
  mf1_p.pack_device(device_into);
  checkpacked(mf1_p, mf1, tol, "WV device");
  

  
  //Check a VV type
  memset(into,0,into_size);
  A2AmesonField<A2Apolicies,A2AvectorVfftw,A2AvectorVfftw> mf2;
  mf2.setup(a2a_args,a2a_args,0,0);
  mf2.testRandom();
  mf2.unpack(into);
  checkunpacked(mf2, into, tol, "VV");

  mf2.unpack_device(device_into);
  memset(into,0,into_size);  
  copy_device_to_host(into, device_into, into_size);
  checkunpacked(mf2, into, tol, "VV device");
  
  A2AmesonField<A2Apolicies,A2AvectorVfftw,A2AvectorVfftw> mf2_p;
  mf2_p.setup(a2a_args,a2a_args,0,0);  
  mf2_p.pack(into);
  checkpacked(mf2_p, mf2, tol, "VV");

  mf2_p.zero();
  mf2_p.pack_device(device_into);
  checkpacked(mf2_p, mf2, tol, "VV device");
  

  //Check a WW type
  memset(into,0,into_size);
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw> mf3;
  mf3.setup(a2a_args,a2a_args,0,0);
  mf3.testRandom();
  mf3.unpack(into);
  checkunpacked(mf3, into, tol, "WW");

  mf3.unpack_device(device_into);
  memset(into,0,into_size);  
  copy_device_to_host(into, device_into, into_size);
  checkunpacked(mf3, into, tol, "WW device");
  
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw> mf3_p;
  mf3_p.setup(a2a_args,a2a_args,0,0); 
  mf3_p.pack(into);
  checkpacked(mf3_p, mf3, tol, "WW");
  
  mf3_p.zero();
  mf3_p.pack_device(device_into);
  checkpacked(mf3_p, mf3, tol, "WW device");
  
  free(into);
}


template<typename A2Apolicies>
void testMesonFieldUnpackPackTblock(A2AArg a2a_args, const double tol){
  a2a_args.src_width = 2;
  std::cout << "Testing mesonfield unpack, pack tblock" << std::endl;

  LRG.AssignGenerator(0); //always uses the RNG at coord 0 on node 0 - should always be the same one!

  //Check a WV type
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf1;
  mf1.setup(a2a_args,a2a_args,0,0);
  mf1.testRandom();
  
  typedef typename A2Apolicies::ScalarComplexType Complex;

  int rows_full = mf1.getNrowsFull();
  int cols_full = mf1.getNcolsFull();

  size_t into_size = rows_full * cols_full * sizeof(Complex);
  Complex* into = (Complex*)malloc(into_size);
  Complex* device_into = (Complex*)device_alloc_check(into_size);
  
  mf1.unpack(into);
  checkunpacked(mf1, into, tol, "WV");
  
  mf1.unpack_device(device_into);
  memset(into,0,into_size);  
  copy_device_to_host(into, device_into, into_size);
  
  checkunpacked(mf1, into, tol, "WV device");

  //Do a test once with the view precreated
  {
    device_memset(device_into,0,into_size);
    CPSautoView(mf1_v,mf1,DeviceRead);

    mf1.unpack_device(device_into, &mf1_v);
    memset(into,0,into_size);  
    copy_device_to_host(into, device_into, into_size);
    
    checkunpacked(mf1, into, tol, "WV device with view");
  }

  
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf1_p;
  mf1_p.setup(a2a_args,a2a_args,0,0); 
  mf1_p.pack(into);
  checkpacked(mf1_p, mf1, tol, "WV");
  
  mf1_p.zero();
  mf1_p.pack_device(device_into);
  checkpacked(mf1_p, mf1, tol, "WV device");
  

  
  //Check a VV type
  memset(into,0,into_size);
  A2AmesonField<A2Apolicies,A2AvectorVfftw,A2AvectorVfftw> mf2;
  mf2.setup(a2a_args,a2a_args,0,0);
  mf2.testRandom();
  mf2.unpack(into);
  checkunpacked(mf2, into, tol, "VV");

  mf2.unpack_device(device_into);
  memset(into,0,into_size);  
  copy_device_to_host(into, device_into, into_size);
  checkunpacked(mf2, into, tol, "VV device");
  
  A2AmesonField<A2Apolicies,A2AvectorVfftw,A2AvectorVfftw> mf2_p;
  mf2_p.setup(a2a_args,a2a_args,0,0);  
  mf2_p.pack(into);
  checkpacked(mf2_p, mf2, tol, "VV");

  mf2_p.zero();
  mf2_p.pack_device(device_into);
  checkpacked(mf2_p, mf2, tol, "VV device");
  

  //Check a WW type
  memset(into,0,into_size);
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw> mf3;
  mf3.setup(a2a_args,a2a_args,0,0);
  mf3.testRandom();
  mf3.unpack(into);
  checkunpacked(mf3, into, tol, "WW");

  mf3.unpack_device(device_into);
  memset(into,0,into_size);  
  copy_device_to_host(into, device_into, into_size);
  checkunpacked(mf3, into, tol, "WW device");
  
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw> mf3_p;
  mf3_p.setup(a2a_args,a2a_args,0,0); 
  mf3_p.pack(into);
  checkpacked(mf3_p, mf3, tol, "WW");
  
  mf3_p.zero();
  mf3_p.pack_device(device_into);
  checkpacked(mf3_p, mf3, tol, "WW device");
  
  free(into);
}




void testMesonFieldNodeDistributeUnique(const A2AArg &a2a_args){
  //Generate a policy with the disk storage method so that we can test even for 1 rank
  A2APOLICIES_TEMPLATE(A2ApoliciesTmp, 1, BaseGridPoliciesGparity, SET_A2AVECTOR_AUTOMATIC_ALLOC, SET_MFSTORAGE_NODESCRATCH, UVMallocPolicy);

  int Lt = GJP.Tnodes()*GJP.TnodeSites();

  typedef A2AmesonField<A2ApoliciesTmp,A2AvectorWfftw,A2AvectorVfftw> MfType;
  std::vector<MfType> mf1(Lt);
  std::vector<MfType> mf2(Lt);

  for(int t=0;t<Lt;t++){
    mf1[t].setup(a2a_args,a2a_args,t,t);
    mf1[t].testRandom();

    mf2[t].setup(a2a_args,a2a_args,t,t);
    mf2[t].testRandom();
  }

  {
    //Distribute all of mf1
    std::vector< std::vector<MfType>* > to_distribute = {&mf1};
    std::vector< std::vector<MfType> const*> to_keep = {};
    nodeDistributeUnique(to_distribute,to_keep);

    for(int t=0;t<Lt;t++) assert(!mf1[t].isOnNode());
    nodeGetMany(1,&mf1);
  }
  {
    //Distribute all of mf1 not in mf2; this is all of them!
    std::vector< std::vector<MfType>* > to_distribute = {&mf1};
    std::vector< std::vector<MfType> const*> to_keep = {&mf2};
    nodeDistributeUnique(to_distribute,to_keep);

    for(int t=0;t<Lt;t++) assert(!mf1[t].isOnNode());
    nodeGetMany(1,&mf1);	
  }
  {
    //Keep all of mf1; should distribute nothing
    std::vector< std::vector<MfType>* > to_distribute = {&mf1};
    std::vector< std::vector<MfType> const*> to_keep = {&mf1};
    nodeDistributeUnique(to_distribute,to_keep);

    for(int t=0;t<Lt;t++) assert(mf1[t].isOnNode());
  }
  {
    //Keep all of mf1 but not mf2
    std::vector< std::vector<MfType>* > to_distribute = {&mf1,&mf2};
    std::vector< std::vector<MfType> const*> to_keep = {&mf1};
    nodeDistributeUnique(to_distribute,to_keep);

    for(int t=0;t<Lt;t++) assert(mf1[t].isOnNode());
    for(int t=0;t<Lt;t++) assert(!mf2[t].isOnNode());
    nodeGetMany(1,&mf2);	
  }
}

void testMesonFieldNodeDistributeOneSided(const A2AArg &a2a_args){
  //Test the one-sided storage method
  int nodes=1;
  for(int i=0;i<4;i++) nodes *= GJP.Nodes(i);

  //Require more than 1 node
  if(nodes > 1){
    A2APOLICIES_TEMPLATE(A2ApoliciesTmp, 1, BaseGridPoliciesGparity, SET_A2AVECTOR_AUTOMATIC_ALLOC, SET_MFSTORAGE_DISTRIBUTEDONESIDED, UVMallocPolicy);

    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    
    typedef A2AmesonField<A2ApoliciesTmp,A2AvectorWfftw,A2AvectorVfftw> MfType;
    std::vector<MfType> mf1(Lt);
    std::vector<MfType> mf1_cp(Lt);

    
    for(int t=0;t<Lt;t++){
      mf1[t].setup(a2a_args,a2a_args,t,t);
      mf1[t].testRandom();
      mf1_cp[t] = mf1[t];
    }

    //Check distribute
    {
      std::cout << "Checking distribute" << std::endl;
      for(int t=0;t<Lt;t++){	
	mf1[t].nodeDistribute();
	std::cout << "t=" << t << " master uid " << mf1[t].masterUID() << std::endl;	
	if(UniqueID() == mf1[t].masterUID()){
	  assert(mf1[t].isInitialized());
	}else{
	  assert(!mf1[t].isInitialized());
	}
      }
    }
    //Check gather
    {
      std::cout << "Checking gather" << std::endl;
      for(int t=0;t<Lt;t++){
	mf1[t].nodeGet();
	assert(mf1[t].isInitialized());
	assert(mf1[t].equals(mf1_cp[t],1e-12,true));	       
      }
    }    
    //Check gathering a second time does not require another communication
    {
      std::cout << "Checking consecutive gather" << std::endl;
      for(int t=0;t<Lt;t++){
	DistributedMemoryStorageOneSided::perf().reset();
	mf1[t].nodeGet();
	assert(DistributedMemoryStorageOneSided::perf().gather_calls == 0);
      }
    }
  }
}

template<typename A2Apolicies>
void testMesonFieldViews(const A2AArg &params){
  std::cout << "Starting testMesonFieldViews" << std::endl;
  typedef A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> MFtype;
  MFtype mf;
  mf.setup(params,params,0,0);

  //Test device read
  {
    CPSautoView(mf_v, mf, HostWrite);
    memset(mf_v.ptr(), 0x1A, sizeof(unsigned char));
  }
  {
    unsigned char* p = (unsigned char*)device_alloc_check(128, sizeof(unsigned char));    
    CPSautoView(mf_v, mf, DeviceRead);
    using namespace Grid;
    accelerator_for(i,1,1,
		       {
			 *p = * (  (unsigned char*)mf_v.ptr() );
		       });
    unsigned char hp;
    copy_device_to_host(&hp,p,sizeof(unsigned char));
    assert(hp == 0x1A);
  }
  //Test device write
  {
    CPSautoView(mf_v, mf, DeviceWrite);
    using namespace Grid;
    accelerator_for(i,1,1,
		       {
			 * (  (unsigned char*)mf_v.ptr() ) = 0x1F;
		       });
  }
  {
    CPSautoView(mf_v, mf, HostRead);
    assert( *((unsigned char*)mf_v.ptr()) == 0x1F );
  }
  //Test device read/write
  {
    CPSautoView(mf_v, mf, HostWrite);
    memset(mf_v.ptr(), 0x3, sizeof(unsigned char));
  }
  {
    CPSautoView(mf_v, mf, DeviceReadWrite);
    using namespace Grid;
    accelerator_for(i,1,1,
		       {
			 unsigned char v = * (  (unsigned char*)mf_v.ptr() );
			 v = v & 0x5;			 
			 * (  (unsigned char*)mf_v.ptr() ) = v;
		       });
  }
  {
    CPSautoView(mf_v, mf, HostRead);
    assert( *((unsigned char*)mf_v.ptr()) == 0x1 );
  }
  std::cout << "testMesonFieldViews passed" << std::endl;
};


template<typename A2Apolicies>
void testMesonFieldNodeSumPartialAsync(const A2AArg &params){
  std::cout << "Starting testMesonFieldNodeSumPartialAsync" << std::endl;
  typedef A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> MFtype;
#ifdef MF_ASYNCREDUCE_THREAD
  MPIallReduceQueued::globalInstance().setVerbose(true);
#endif
  
  MFtype mf_base;
  mf_base.setup(params,params,0,0);
  mf_base.testRandom();

  MFtype mf_cp1(mf_base), mf_cp2(mf_base);
  
  mf_cp1.nodeSum();

  int iblocksz = mf_base.getNrows()/10;
  int jblocksz = mf_base.getNcols()/10;
  
  std::vector<nodeSumPartialAsyncHandle<typename A2Apolicies::AllocPolicy> > handles;
  for(int i0=0; i0<mf_base.getNrows(); i0+=iblocksz){
    int iblocksz_true = std::min(iblocksz, mf_base.getNrows()-i0);
    for(int j0=0; j0<mf_base.getNcols(); j0+=jblocksz){
      int jblocksz_true = std::min(jblocksz, mf_base.getNcols()-j0);
  
      handles.push_back( mf_cp2.template nodeSumPartialAsync<typename A2Apolicies::AllocPolicy>(i0,iblocksz_true,j0,jblocksz_true) );
    }
  }
  for(int i=0;i<handles.size();i++) mf_cp2.nodeSumPartialAsyncComplete(handles[i]);
  if(! mf_cp1.equals(mf_cp2,1e-10,true)){
    std::cout << "testMesonFieldNodeSumPartialAsync FAILED" << std::endl;
    exit(1);
  }
  std::cout << "testMesonFieldNodeSumPartialAsync passed" << std::endl;
#ifdef MF_ASYNCREDUCE_THREAD
  MPIallReduceQueued::globalInstance().setVerbose(false);
#endif

}

template<typename A2Apolicies>
void testMesonFieldNodeSumRowBlockAsyncInPlace(const A2AArg &params){
  std::cout << "Starting testMesonFieldNodeSumRowBlockAsyncInPlace" << std::endl;
  typedef A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> MFtype;
#ifdef MF_ASYNCREDUCE_THREAD
  MPIallReduceQueued::globalInstance().setVerbose(true);
#endif
  
  MFtype mf_base;
  mf_base.setup(params,params,0,0);
  mf_base.testRandom();

  MFtype mf_cp1(mf_base), mf_cp2(mf_base);
  
  mf_cp1.nodeSum();

  int iblocksz = mf_base.getNrows()/10;
  
  std::vector<nodeSumRowBlockAsyncInPlaceHandle> handles;
  for(int i0=0; i0<mf_base.getNrows(); i0+=iblocksz){
    int iblocksz_true = std::min(iblocksz, mf_base.getNrows()-i0);
    handles.push_back( mf_cp2.nodeSumRowBlockAsyncInPlace(i0,iblocksz_true) );
  }
  for(int i=0;i<handles.size();i++) mf_cp2.nodeSumRowBlockAsyncInPlaceComplete(handles[i]);
  if(! mf_cp1.equals(mf_cp2,1e-10,true)){
    std::cout << "testMesonFieldNodeSumRowBlockAsyncInPlace FAILED" << std::endl;
    exit(1);
  }
  std::cout << "testMesonFieldNodeSumRowBlockAsyncInPlace passed" << std::endl;
#ifdef MF_ASYNCREDUCE_THREAD
  MPIallReduceQueued::globalInstance().setVerbose(false);
#endif

}


CPS_END_NAMESPACE
