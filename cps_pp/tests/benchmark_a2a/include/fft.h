#pragma once

CPS_START_NAMESPACE

//non-SIMD data
template<typename A2Apolicies>
void benchmarkFFT(const int ntest){
  typedef typename A2Apolicies::FermionFieldType::FieldSiteType mf_Complex;
  typedef typename A2Apolicies::FermionFieldType::FieldMappingPolicy MappingPolicy;
  typedef typename A2Apolicies::FermionFieldType::FieldAllocPolicy AllocPolicy;

  typedef typename MappingPolicy::template Rebase<OneFlavorPolicy>::type OneFlavorMap;
  
  typedef CPSfield<mf_Complex,12,OneFlavorMap, AllocPolicy> FieldType;
  typedef typename FieldType::InputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);

  bool do_dirs[4] = {1,1,1,0}; //3D fft
  
  FieldType in(fp);
  in.testRandom();

  FieldType out1(fp);
  double t_orig = -dclock();
  for(int i=0;i<ntest;i++){
    if(!UniqueID()) printf("FFT orig %d\n",i);
    fft(out1,in,do_dirs);
  }
  t_orig += dclock();
  t_orig /= ntest;

  FieldType out2(fp);
  double t_opt = -dclock();
  for(int i=0;i<ntest;i++){  
    if(!UniqueID()) printf("FFT opt %d\n",i);
    fft_opt(out2,in,do_dirs);
  }
  t_opt += dclock();
  t_opt /= ntest;

  if(!UniqueID()){
    printf("3D FFT timings: orig %f s   opt %f s\n", t_orig, t_opt);    
    fft_opt_mu_timings::get().print();
  }
  
}


CPS_END_NAMESPACE
