#pragma once

CPS_START_NAMESPACE


template<typename A2Apolicies, int isGparity>
struct _benchmarkKtoPiPiOffload{};

template<typename A2Apolicies>
struct _benchmarkKtoPiPiOffload<A2Apolicies, 0>{
  static void run(const A2AArg &a2a_args,Lattice &lat){}
};
template<typename A2Apolicies>
struct _benchmarkKtoPiPiOffload<A2Apolicies, 1>{
  typedef typename A2Apolicies::ComplexType mf_Complex;
  typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;    
  typedef std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_WV;
  typedef std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw> > mf_WW;
  typedef typename ComputeKtoPiPiGparity<A2Apolicies>::ResultsContainerType ResultsContainerType;
  typedef typename ComputeKtoPiPiGparity<A2Apolicies>::MixDiagResultsContainerType MixDiagResultsContainerType;

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

  MesonFieldMomentumContainer<A2Apolicies> mf_pions;

  mf_WV tmp_WV;
  mf_WW tmp_WW;
    
  ThreeMomentum pp;
  ThreeMomentum pm;

  int pipi_sep;
  int tstep;
  std::vector<int> tsep_k_pi;

  ~_benchmarkKtoPiPiOffload(){
    delete W;
    delete V;
    delete Wh;
    delete Vh;
  }

  _benchmarkKtoPiPiOffload(const A2AArg &a2a_args,Lattice &lat, const std::vector<int> &tsep_k_pi): a2a_args(a2a_args), lat(lat), params(a2a_args), Lt(GJP.Tnodes()*GJP.TnodeSites()), 
												    tmp_WV(Lt), tmp_WW(Lt), pipi_sep(2), tstep(1), tsep_k_pi(tsep_k_pi){
    assert(GJP.Gparity());
    a2a_args_s = a2a_args;
    a2a_args_s.nl = 0;

    params_s = A2Aparams(a2a_args_s);
    
    printMem("benchmarkKtoPiPiOffload start");
    
    defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);
  
    W = new A2AvectorW<A2Apolicies>(a2a_args,fp);
    V = new A2AvectorV<A2Apolicies>(a2a_args,fp);
    W->testRandom();
    V->testRandom();
    printMem("benchmarkKtoPiPiOffload W,V setup");

    Wh = new A2AvectorW<A2Apolicies>(a2a_args_s,fp);
    Vh = new A2AvectorV<A2Apolicies>(a2a_args_s,fp);
    Wh->testRandom();
    Vh->testRandom();
    printMem("benchmarkKtoPiPiOffload Wh,Vh setup");

    int p[3];
    GparityBaseMomentum(p,+1);
    pp = ThreeMomentum(p);

    GparityBaseMomentum(p,-1);
    pm = ThreeMomentum(p);

    for(int t=0;t<Lt;t++){
      tmp_WV[t].setup(params,params,t,t);
      tmp_WV[t].testRandom();

      tmp_WW[t].setup(params,params_s,t,t); //W W_s
      tmp_WW[t].testRandom();
    }
    mf_pions.copyAdd(pp,tmp_WV);
    mf_pions.copyAdd(pm,tmp_WV);
    printMem("benchmarkKtoPiPiOffload MF setup");
  }

  void type1(){
    if(!UniqueID()){ printf("Timing K->pipi type 1 field version\n"); fflush(stdout); }
    std::vector<ResultsContainerType> result(tsep_k_pi.size());
    ComputeKtoPiPiGparity<A2Apolicies>::type1_field(result.data(), tsep_k_pi, pipi_sep, tstep, pp, tmp_WW, mf_pions, *V, *Vh, *W, *Wh);
    if(!UniqueID()){ printf("End of timing of K->pipi type 1 field version\n"); fflush(stdout); }
  }

  void type4(){
    if(!UniqueID()){ printf("Timing K->pipi type 4 field version\n"); fflush(stdout); }
  
    ResultsContainerType result;
    MixDiagResultsContainerType mix;

    ComputeKtoPiPiGparity<A2Apolicies>::type4_field(result, mix, 1, tmp_WW, *V, *Vh, *W, *Wh);
    if(!UniqueID()){ printf("End of timing of K->pipi type 4 field version\n"); fflush(stdout); }
  }
};


template<typename A2Apolicies>
void benchmarkKtoPiPiType1offload(const A2AArg &a2a_args,Lattice &lat, const std::vector<int> &tsep_k_pi){
  _benchmarkKtoPiPiOffload<A2Apolicies, A2Apolicies::GPARITY> calc(a2a_args,lat,tsep_k_pi);
  calc.type1();
}

template<typename A2Apolicies>
void benchmarkKtoPiPiType4offload(const A2AArg &a2a_args,Lattice &lat, const std::vector<int> &tsep_k_pi){
  _benchmarkKtoPiPiOffload<A2Apolicies, A2Apolicies::GPARITY> calc(a2a_args,lat,tsep_k_pi);
  calc.type4();
}


CPS_END_NAMESPACE
