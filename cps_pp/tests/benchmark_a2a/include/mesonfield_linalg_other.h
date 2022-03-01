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

  mesonfield_trace_prod_timings::data().report();
  mesonfield_trace_prod_timings::data().reset();
  
  time = 0;
  for(int test=0;test<ntests;test++){
    time -= dclock();
    grid_Complex val = trace(mf,mf2);
    time += dclock();
  }
  time /= ntests;

  std::cout << "Hot trace time avg for " << ntests << " tests " << time << "s" << std::endl;

  mesonfield_trace_prod_timings::data().report();
  mesonfield_trace_prod_timings::data().reset();
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

  mesonfield_trace_prod_timings::data().report();
  mesonfield_trace_prod_timings::data().reset();
  
  time = 0;
  for(int test=0;test<ntests;test++){
    time -= dclock();
    trace(mat,mf_t,mf2_t);
    time += dclock();
  }
  time /= ntests;

  std::cout << "Hot trace time avg for " << ntests << " tests " << time << "s" << std::endl;

  mesonfield_trace_prod_timings::data().report();
  mesonfield_trace_prod_timings::data().reset();
}



CPS_END_NAMESPACE
