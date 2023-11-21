#pragma once

CPS_START_NAMESPACE

template<typename GridA2Apolicies>
void benchmarkMfTraceProd(const A2AArg &a2a_args, const int ntests){
  std::cout << "Starting MF trace-product benchmark\n";

  A2Aparams params(a2a_args);
  
  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf, mf2;
  mf.setup(params,params,0,0);     
  mf2.setup(params,params,0,0);     

  std::cout << "Using WV mesonfields of size " << mf.getNrows() << "x" << mf.getNcols() << " and memory size " << double(mf.byte_size())/1024./1024. << " MB" << std::endl;
  
  typedef typename GridA2Apolicies::ScalarComplexType grid_Complex;
  
  double time = 0;
  for(int test=0;test<ntests;test++){
    mf.testRandom();
    mf2.testRandom();

    time -= dclock();
    grid_Complex val = trace(mf,mf2);
    time += dclock();
  }
  time /= ntests;

  std::cout << "Cold trace time avg for " << ntests << " tests " << time << "s" << std::endl;
  
  time = 0;
  for(int test=0;test<ntests;test++){
    time -= dclock();
    grid_Complex val = trace(mf,mf2);
    time += dclock();
  }
  time /= ntests;

  std::cout << "Hot trace time avg for " << ntests << " tests " << time << "s" << std::endl;
}

template<typename GridA2Apolicies>
void benchmarkMfTraceProdGPU(const A2AArg &a2a_args, const int ntests){
#ifdef GPU_VEC
  std::cout << "Starting MF GPU trace-product benchmark\n";

  A2Aparams params(a2a_args);
  
  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf, mf2;
  mf.setup(params,params,0,0);     
  mf2.setup(params,params,0,0);     

  std::cout << "Using WV mesonfields of size " << mf.getNrows() << "x" << mf.getNcols() << " and memory size " << double(mf.byte_size())/1024./1024. << " MB" << std::endl;
  
  typedef typename GridA2Apolicies::ScalarComplexType grid_Complex;
  
  double time = 0;
  for(int test=0;test<ntests;test++){
    mf.testRandom();
    mf2.testRandom();

    time -= dclock();
    grid_Complex val = trace_gpu(mf,mf2);
    time += dclock();
  }
  time /= ntests;

  std::cout << "Cold trace time avg for " << ntests << " tests " << time << "s" << std::endl;

  mesonfield_trace_prod_gpu_timings::data().report();
  mesonfield_trace_prod_gpu_timings::data().reset();

  //"Hot" performance uses precomputed views
  {
    CPSautoView(mf_v,mf, DeviceRead);
    CPSautoView(mf2_v,mf2, DeviceRead);
    
    time = 0;
    for(int test=0;test<ntests;test++){
      time -= dclock();
      grid_Complex val = trace_gpu(mf,mf2,&mf_v,&mf2_v);
      time += dclock();
    }
    time /= ntests;
  }    

  std::cout << "Hot (pre-copied views) trace time avg for " << ntests << " tests " << time << "s" << std::endl;

  mesonfield_trace_prod_gpu_timings::data().report();
  mesonfield_trace_prod_gpu_timings::data().reset();
#endif
}





template<typename GridA2Apolicies>
void benchmarkMfVectorTraceProd(const A2AArg &a2a_args, const int ntests){
  std::cout << "Starting MF-vector trace-product benchmark\n";

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  A2Aparams params(a2a_args);

  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf,mf2;
  mf.setup(params,params,0,0);     
  mf2.setup(params,params,0,0);     
 
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_t(Lt), mf2_t(Lt);

  int nodes;
  MPI_Comm_size(MPI_COMM_WORLD, &nodes);
  
  std::cout << "Using WV mesonfields of size " << mf.getNrows() << "x" << mf.getNcols() << " and memory size " << double(mf.byte_size())/1024./1024. << " MB" << std::endl;
  std::cout << "Lt=" << Lt << " thus Lt*Lt=" << Lt*Lt << " traces distributed over " << nodes << " nodes" << std::endl;
  
  typedef typename GridA2Apolicies::ScalarComplexType grid_Complex;

  fMatrix<grid_Complex> mat(Lt,Lt);
  
  double time = 0;
  for(int test=0;test<ntests;test++){
    mf.testRandom();
    mf2.testRandom();
    for(int t=0;t<Lt;t++){
      mf_t[t] = mf; mf2_t[t] = mf2;
    }
    
    time -= dclock();
    trace(mat,mf_t,mf2_t);
    time += dclock();
  }
  time /= ntests;

  std::cout << "Cold trace time avg for " << ntests << " tests " << time << "s" << std::endl;

  time = 0;
  for(int test=0;test<ntests;test++){
    time -= dclock();
    trace(mat,mf_t,mf2_t);
    time += dclock();
  }
  time /= ntests;

  std::cout << "Hot trace time avg for " << ntests << " tests " << time << "s" << std::endl;
}
template<typename GridA2Apolicies>
void mf_plus_equals_gpu(A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> &into, const A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> &from){
  using namespace Grid;
  CPSautoView(into_v,into,DeviceReadWrite);
  CPSautoView(from_v,from,DeviceRead);
  accelerator_for(i, into_v.size(), 1, {
      into_v.ptr()[i] = into_v.ptr()[i] + from_v.ptr()[i];
    });  
}

template<typename GridA2Apolicies>
void benchmarkMfPlusEquals(const A2AArg &a2a_args, const int ntests){
  std::cout << "Starting benchmarkMfPlusEquals\n";

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  A2Aparams params(a2a_args);

  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf,mf2;
  mf.setup(params,params,0,0);     
  mf2.setup(params,params,0,0);

  mf.testRandom();
  mf2.testRandom();

  std::cout << "Benchmarking CPU hot plus_equals" << std::endl;
  mf.plus_equals(mf2);

  double time = 0;
  for(int test=0;test<ntests;test++){
    time -= dclock();
    mf.plus_equals(mf2);
    time += dclock();
  }
  std::cout << "Performed " << ntests << " operations in " << time << "s, avg " << time / ntests << "s" << std::endl;

  time = 0;
  std::cout << "Benchmarking GPU hot plus_equals" << std::endl;
  mf_plus_equals_gpu(mf,mf2);
  time = 0;
  for(int test=0;test<ntests;test++){
    time -= dclock();
    mf_plus_equals_gpu(mf,mf2);
    time += dclock();
  }
  std::cout << "Performed " << ntests << " operations in " << time << "s, avg " << time / ntests << "s" << std::endl;

  time = 0;
  std::cout << "Benchmarking GPU cold plus_equals" << std::endl;
  time = 0;
  double copy_back = 0;
  for(int test=0;test<ntests;test++){
    mf.testRandom(); //force host location
    mf2.testRandom();
  
    time -= dclock();
    mf_plus_equals_gpu(mf,mf2);
    time += dclock();

    copy_back -= dclock();
    {      
      CPSautoView(mf_v,mf,HostRead);
    }
    copy_back += dclock();
  }
  std::cout << "Performed " << ntests << " operations in " << time << "s, avg " << time / ntests << "s" << std::endl;
  std::cout << "Copy-back for output MF an additional " << copy_back / ntests << "s per iteration" << std::endl;
}

template<typename GridA2Apolicies>
void mf_times_equals_gpu(A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> &into, const typename GridA2Apolicies::ScalarComplexType nrm){
  using namespace Grid;
  CPSautoView(into_v,into,DeviceReadWrite);
  accelerator_for(i, into_v.size(), 1, {
      into_v.ptr()[i] = into_v.ptr()[i] * nrm;
    });  
}

template<typename GridA2Apolicies>
void benchmarkMfTimesEquals(const A2AArg &a2a_args, const int ntests){
  std::cout << "Starting benchmarkMfTimesEquals\n";

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  A2Aparams params(a2a_args);

  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf;
  mf.setup(params,params,0,0);     

  mf.testRandom();

  typename GridA2Apolicies::ScalarComplexType nrm(1.0001,1.0002);

  std::cout << "Benchmarking CPU hot times_equals" << std::endl;
  mf.times_equals(nrm);

  double time = 0;
  for(int test=0;test<ntests;test++){
    time -= dclock();
    mf.times_equals(nrm);
    time += dclock();
  }
  std::cout << "Performed " << ntests << " operations in " << time << "s, avg " << time / ntests << "s" << std::endl;

  time = 0;
  std::cout << "Benchmarking GPU hot times_equals" << std::endl;
  mf_times_equals_gpu(mf,nrm);
  time = 0;
  for(int test=0;test<ntests;test++){
    time -= dclock();
    mf_times_equals_gpu(mf,nrm);
    time += dclock();
  }
  std::cout << "Performed " << ntests << " operations in " << time << "s, avg " << time / ntests << "s" << std::endl;

  time = 0;
  std::cout << "Benchmarking GPU cold times_equals" << std::endl;
  time = 0;
  double copy_back = 0;
  for(int test=0;test<ntests;test++){
    mf.testRandom(); //force host location
  
    time -= dclock();
    mf_times_equals_gpu(mf,nrm);
    time += dclock();

    copy_back -= dclock();
    {      
      CPSautoView(mf_v,mf,HostRead);
    }
    copy_back += dclock();
  }
  std::cout << "Performed " << ntests << " operations in " << time << "s, avg " << time / ntests << "s" << std::endl;
  std::cout << "Copy-back for output MF an additional " << copy_back / ntests << "s per iteration" << std::endl;
}



CPS_END_NAMESPACE
