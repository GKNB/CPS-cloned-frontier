#pragma once

CPS_START_NAMESPACE


template<typename A2Apolicies, int isGparity>
struct _benchmarkKtoSigmaOffload{};

template<typename A2Apolicies>
struct _benchmarkKtoSigmaOffload<A2Apolicies, 0>{
  static void run(const A2AArg &a2a_args,Lattice &lat){}
};
template<typename A2Apolicies>
struct _benchmarkKtoSigmaOffload<A2Apolicies, 1>{
  typedef typename A2Apolicies::ComplexType mf_Complex;
  typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;    
  typedef std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_WV;
  typedef std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw> > mf_WW;
  typedef typename ComputeKtoSigma<A2Apolicies>::ResultsContainerType ResultsContainerType;
  typedef typename ComputeKtoSigma<A2Apolicies>::MixDiagResultsContainerType MixDiagResultsContainerType;

  Lattice &lat;
  const A2AArg &a2a_args;
  A2AArg a2a_args_s;
  A2Aparams params;
  A2Aparams params_s;
  int Lt;

  FieldInputParamType fp;
  A2AvectorW<A2Apolicies> *W;
  A2AvectorV<A2Apolicies> *V;
  A2AvectorW<A2Apolicies> *Wh;
  A2AvectorV<A2Apolicies> *Vh;

  mf_WV tmp_WV;
  mf_WW tmp_WW;
    
  std::vector<int> tsep_k_s;

  ~_benchmarkKtoSigmaOffload(){
    delete W;
    delete V;
    delete Wh;
    delete Vh;
  }

  _benchmarkKtoSigmaOffload(const A2AArg &a2a_args,Lattice &lat, const std::vector<int> &tsep_k_s): a2a_args(a2a_args), lat(lat), params(a2a_args), Lt(GJP.Tnodes()*GJP.TnodeSites()), 
												    tmp_WV(Lt), tmp_WW(Lt), tsep_k_s(tsep_k_s){
    assert(GJP.Gparity());

    a2a_args_s = a2a_args;
    a2a_args_s.nl = 0;

    params_s = A2Aparams(a2a_args_s);
    
    defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);
  
    W = new A2AvectorW<A2Apolicies>(a2a_args,fp);
    V = new A2AvectorV<A2Apolicies>(a2a_args,fp);
    W->testRandom();
    V->testRandom();

    Wh = new A2AvectorW<A2Apolicies>(a2a_args_s,fp);
    Vh = new A2AvectorV<A2Apolicies>(a2a_args_s,fp);
    Wh->testRandom();
    Vh->testRandom();

    for(int t=0;t<Lt;t++){
      tmp_WV[t].setup(params,params,t,t);
      tmp_WV[t].testRandom();

      tmp_WW[t].setup(params,params_s,t,t); //W W_s
      tmp_WW[t].testRandom();
    }
  }

 
  void type12(){
    if(!UniqueID()){ printf("Timing K->sigma type 1/2 field version\n"); fflush(stdout); }
    ComputeKtoSigma<A2Apolicies> compute(*V, *W, *Vh, *Wh, tmp_WW, tsep_k_s);
    std::vector<ResultsContainerType> result(tsep_k_s.size());
    compute.type12(result, tmp_WV);
    if(!UniqueID()){ printf("End of timing of K->sigma type 1/2 field version\n"); fflush(stdout); }
  }

};


template<typename A2Apolicies>
void benchmarkKtoSigmaType12offload(const A2AArg &a2a_args,Lattice &lat, const std::vector<int> &tsep_k_s){
  _benchmarkKtoSigmaOffload<A2Apolicies, A2Apolicies::GPARITY> calc(a2a_args,lat,tsep_k_s);
  calc.type12();
}



CPS_END_NAMESPACE
