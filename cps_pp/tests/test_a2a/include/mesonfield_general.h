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
  A2AvectorWfftw<ScalarA2Apolicies> W(a2a_args);
  A2AvectorVfftw<ScalarA2Apolicies> V(a2a_args);
  
  A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf;
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
}	


template<typename A2Apolicies>
void testTraceSingle(const A2AArg &a2a_args, const double tol){
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
}


CPS_END_NAMESPACE
