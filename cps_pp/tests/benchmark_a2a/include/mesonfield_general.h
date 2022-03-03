#pragma once

CPS_START_NAMESPACE

template<typename A2Apolicies>
void benchmarkMesonFieldUnpack(const A2AArg &a2a_args,const int ntest){
  std::cout << "Timing mesonfield unpack" << std::endl;

  LRG.AssignGenerator(0); //always uses the RNG at coord 0 on node 0 - should always be the same one!

  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf1;
  mf1.setup(a2a_args,a2a_args,0,0);

  typedef typename A2Apolicies::ScalarComplexType Complex;

  int rows_full = mf1.getNrowsFull();
  int cols_full = mf1.getNcolsFull();

  size_t into_size = rows_full * cols_full * sizeof(Complex);
  Complex* into = (Complex*)malloc(into_size);
  
  double time = 0;
  for(int i=0;i<ntest;i++){
    mf1.testRandom();
    time -= dclock();
    mf1.unpack(into);
    time += dclock();
  }

  std::cout << "Cold " << ntest << " iterations, avg time " << time / ntest << "s" << std::endl;

  time = 0;
  for(int i=0;i<ntest;i++){
    time -= dclock();
    mf1.unpack(into);
    time += dclock();
  }

  std::cout << "Hot " << ntest << " iterations, avg time " << time / ntest << "s" << std::endl;
  free(into);
}



template<typename A2Apolicies>
void benchmarkMesonFieldPack(const A2AArg &a2a_args,const int ntest){
  std::cout << "Timing mesonfield pack" << std::endl;

  LRG.AssignGenerator(0); //always uses the RNG at coord 0 on node 0 - should always be the same one!

  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf1;
  mf1.setup(a2a_args,a2a_args,0,0);

  typedef typename A2Apolicies::ScalarComplexType Complex;

  int rows_full = mf1.getNrowsFull();
  int cols_full = mf1.getNcolsFull();

  size_t into_size = rows_full * cols_full * sizeof(Complex);
  Complex* from = (Complex*)malloc(into_size);  
  
  double time = 0;
  for(int i=0;i<ntest;i++){
    memset(from, i, into_size);    
    time -= dclock();
    mf1.pack(from);
    time += dclock();
  }

  std::cout << "Cold " << ntest << " iterations, avg time " << time / ntest << "s" << std::endl;

  time = 0;
  for(int i=0;i<ntest;i++){
    time -= dclock();
    mf1.pack(from);
    time += dclock();
  }

  std::cout << "Hot " << ntest << " iterations, avg time " << time / ntest << "s" << std::endl;
  free(from);
}



CPS_END_NAMESPACE
