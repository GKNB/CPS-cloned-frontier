#ifndef _KTOPIPI_MAIN_A2A_PION_H_
#define _KTOPIPI_MAIN_A2A_PION_H_

template<typename Vtype, typename Wtype, typename PionMomentumPolicy>
void randomizeLLmesonFields(MesonFieldMomentumContainer<getMesonFieldType<Wtype,Vtype> > &mf_ll_con,
			    MesonFieldMomentumContainer<getMesonFieldType<Wtype,Vtype> > &mf_ll_con_2s,
			    Vtype &V, Wtype &W,
			    const PionMomentumPolicy &pion_mom){
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  typedef getMesonFieldType<Wtype,Vtype> MesonFieldType;
  std::vector<MesonFieldType> mf(Lt);
  std::vector<MesonFieldType> *ins;
  for(int t=0;t<Lt;t++) mf[t].setup(W,V,t,t);

  for(int p=0;p<pion_mom.nMom();p++){
    for(int t=0;t<Lt;t++) mf[t].testRandom();
    ins = &mf_ll_con.copyAdd(pion_mom.getTotalMomentum(p),mf);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(1,ins);
#endif
    if(GJP.Gparity()){
      for(int t=0;t<Lt;t++) mf[t].testRandom();
      ins = &mf_ll_con_2s.copyAdd(pion_mom.getTotalMomentum(p),mf);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
      nodeDistributeMany(1,ins);
#endif
    }
  }
}

template<typename Vtype, typename Wtype, typename PionMomentumPolicy>
void randomizeLLmesonFields(MesonFieldMomentumContainer<getMesonFieldType<Wtype,Vtype> > &mf_ll_con,
			    Vtype &V, Wtype &W,
			    const PionMomentumPolicy &pion_mom){
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  typedef getMesonFieldType<Wtype,Vtype> MesonFieldType;
  std::vector<MesonFieldType> mf(Lt);
  std::vector<MesonFieldType> *ins;
  for(int t=0;t<Lt;t++) mf[t].setup(W,V,t,t);

  for(int p=0;p<pion_mom.nMom();p++){
    for(int t=0;t<Lt;t++) mf[t].testRandom();
    ins = &mf_ll_con.copyAdd(pion_mom.getTotalMomentum(p),mf);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(1,ins);
#endif
  }
}



template<typename Vtype, typename Wtype, typename PionMomentumPolicy>
void computeLLmesonFields(MesonFieldMomentumContainer<getMesonFieldType<Wtype,Vtype> > &mf_ll_con, 
			  MesonFieldMomentumContainer<getMesonFieldType<Wtype,Vtype> > &mf_ll_con_2s,
			  Vtype &V, Wtype &W,
			  const PionMomentumPolicy &pion_mom,
			  const int conf, Lattice &lat, const Parameters &params, 
			  const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams, 
			  const bool randomize_mf,
			  const std::string &mf_write_postpend = ""){
  LOGA2A << "Computing light-light meson fields" << std::endl;
  double time = -dclock();
  if(randomize_mf){
    randomizeLLmesonFields(mf_ll_con, mf_ll_con_2s, V, W, pion_mom);
  }else{ 
    assert(GJP.Gparity());
    computeGparityLLmesonFields1s2s<Vtype,Wtype,PionMomentumPolicy>::computeMesonFields(mf_ll_con, mf_ll_con_2s, params.meas_arg.WorkDirectory,conf, pion_mom, W, V, params.jp.pion_rad, lat, field3dparams,mf_write_postpend);
  }

  time += dclock();
  a2a_print_time("main","Light-light meson fields",time);

  printMem("Memory after light-light meson field computation");
}

template<typename Vtype, typename Wtype, typename PionMomentumPolicy>
void computeLLmesonFields1s(MesonFieldMomentumContainer<getMesonFieldType<Wtype,Vtype> > &mf_ll_con,
			    Vtype &V, Wtype &W,
			    const PionMomentumPolicy &pion_mom,
			    Lattice &lat, const Parameters &params, 
			    const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams, 
			    const bool randomize_mf){
  LOGA2A << "Computing light-light 1s pion meson fields" << std::endl;
  double time = -dclock();
  if(randomize_mf){
    randomizeLLmesonFields(mf_ll_con, V, W, pion_mom);
  }else{ 
#define MF_1S_SUM_ON_THE_FLY
#ifdef MF_1S_SUM_ON_THE_FLY
    typedef computeGparityLLmesonFields1sSumOnTheFly<Vtype,Wtype,PionMomentumPolicy, 15, sigma3> computeType;
#else
    typedef computeGparityLLmesonFields1s<Vtype,Wtype,PionMomentumPolicy, 15, sigma3> computeType;
#endif

    assert(GJP.Gparity());
    typename computeType::Options opt;
#ifdef ARCH_BGQ
    opt.thr_internal = 32;
#endif

    computeType::computeMesonFields(mf_ll_con, pion_mom, W, V, params.jp.pion_rad, lat, field3dparams, opt);
  }

  time += dclock();
  a2a_print_time("main","Light-light 1s pion meson fields",time);

  printMem("Memory after light-light 1s pion meson field computation");
}


template<typename Vtype, typename Wtype, typename PionMomentumPolicy>
void computePion2pt(MesonFieldMomentumContainer<getMesonFieldType<Wtype,Vtype> > &mf_ll_con, 
		    const PionMomentumPolicy &pion_mom, const int conf, const Parameters &params, const std::string &postpend = ""){
  const int nmom = pion_mom.nMom();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  
  LOGA2A << "Computing pion 2pt function" << std::endl;
  double time = -dclock();
  for(int p=0;p<nmom;p+=2){ //note odd indices 1,3,5 etc have equal and opposite momenta to 0,2,4... 
    LOGA2A << "Starting pidx " << p << std::endl;
    fMatrix<typename A2Apolicies::ScalarComplexType> pion(Lt,Lt);
    ComputePion<Vtype,Wtype>::compute(pion, mf_ll_con, pion_mom, p);
    //Note it seems Daiqian's pion momenta are opposite what they should be for 'conventional' Fourier transform phase conventions:
    //f'(p) = \sum_{x,y}e^{ip(x-y)}f(x,y)  [conventional]
    //f'(p) = \sum_{x,y}e^{-ip(x-y)}f(x,y) [Daiqian]
    //This may have been a mistake as it only manifests in the difference between the labelling of the pion momenta and the sign of 
    //the individual quark momenta.
    //However it doesn't really make any difference. If you enable DAIQIAN_PION_PHASE_CONVENTION
    //the output files will be labelled in Daiqian's convention
#define DAIQIAN_PION_PHASE_CONVENTION

    std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_pioncorr_mom";
#ifndef DAIQIAN_PION_PHASE_CONVENTION
    os << pion_mom.getMesonMomentum(p).file_str(2);  //note the divisor of 2 is to put the momenta in units of pi/L and not pi/2L
#else
    os << (-pion_mom.getMesonMomentum(p)).file_str(2);
#endif
    os << postpend;
    pion.write(os.str());
#ifdef WRITE_HEX_OUTPUT
    os << ".hexfloat";
    pion.write(os.str(),true);
#endif
  }
  time += dclock();
  a2a_print_time("main","Pion 2pt function",time);

  printMem("Memory after pion 2pt function computation");
}


#endif
