#ifndef _KTOPIPI_MAIN_A2A_KTOPIPI_H_
#define _KTOPIPI_MAIN_A2A_KTOPIPI_H_

struct LSWWmesonFields{
  typedef std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw> > mfVector;
  mfVector mf_ls_ww;
  LSWWmesonFields(){}

  void move(mfVector &r){
    mf_ls_ww.resize(r.size());
    for(int i=0;i<r.size();i++)
      mf_ls_ww[i].move(r[i]);
  }
  void average(LSWWmesonFields &r){
    for(int i=0;i<mf_ls_ww.size();i++){
      bool redist_l = false, redist_r = false;
      if(!mf_ls_ww[i].isOnNode()){ mf_ls_ww[i].nodeGet(); redist_l = true; }
      if(!r.mf_ls_ww[i].isOnNode()){ r.mf_ls_ww[i].nodeGet(); redist_r = true; }
      
      mf_ls_ww[i].average(r.mf_ls_ww[i]);

      if(redist_l) mf_ls_ww[i].nodeDistribute();
      if(redist_r) r.mf_ls_ww[i].nodeDistribute();
    }      
  }
  void distribute(){
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(1,&mf_ls_ww);
#endif
  }
  void gather(){
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeGetMany(1,&mf_ls_ww);
#endif
  }
  void free_mem(){
    for(int i=0;i<mf_ls_ww.size();i++)
      mf_ls_ww[i].free_mem();
  }
  void write(const std::string &filename, const bool redistribute){
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeGetMany(1,&mf_ls_ww);
#endif

#ifndef MEMTEST_MODE
    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw>::write(filename,mf_ls_ww);
#endif

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    if(redistribute) nodeDistributeMany(1,&mf_ls_ww);
#endif
  }
  void read(const std::string &filename, const bool distribute){
#ifndef MEMTEST_MODE
    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw>::write(filename,mf_ls_ww);
#endif

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    if(distribute) nodeDistributeMany(1,&mf_ls_ww);
#endif
  }


};


  
  


//Type 4 doesn't depend on pipi src so if you are doing multiple sources you can set do_type4=false for subsequent runs
template<typename Type1pionMomentumPolicy, typename Type23pionMomentumPolicy>
void computeKtoPiPiContractions(const A2AvectorV<A2Apolicies> &V, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W,
				const A2AvectorV<A2Apolicies> &V_s, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W_s,
				const LSWWmesonFields &mf_ls_ww_con, MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con,
				const Type1pionMomentumPolicy &type1_pion_mom, const Type23pionMomentumPolicy &type23_pion_mom, const int conf, const Parameters &params, 
				const std::string &src_descr, const std::string &src_fappend, bool do_type4, const std::string &postpend = ""){
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  
  typedef ComputeKtoPiPiGparity<A2Apolicies>::ResultsContainerType ResultsContainerType;
  typedef ComputeKtoPiPiGparity<A2Apolicies>::MixDiagResultsContainerType MixDiagResultsContainerType;

  const std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw> > &mf_ls_ww = mf_ls_ww_con.mf_ls_ww;

  std::vector<int> k_pi_separation(params.jp.k_pi_separation.k_pi_separation_len);
  for(int i=0;i<params.jp.k_pi_separation.k_pi_separation_len;i++) k_pi_separation[i] = params.jp.k_pi_separation.k_pi_separation_val[i];

    
  //For type1 loop over momentum of pi1 (conventionally the pion closest to the kaon)
  int ngp = 0; for(int i=0;i<3;i++) if(GJP.Bc(i)==BND_CND_GPARITY) ngp++;


  if(!UniqueID()) printf("Starting type 1 contractions, nmom = %d\n",type1_pion_mom.nMom());
  double time = -dclock();

  for(int pidx=0; pidx < type1_pion_mom.nMom(); pidx++){     
    ThreeMomentum p_pi1 = type1_pion_mom.getMesonMomentum(pidx);

    if(!UniqueID()) printf("Starting type 1 contractions with p_pi1=%s and source %s\n",p_pi1.str().c_str(),src_descr.c_str());
    printMem("Memory status before type1 K->pipi");

    std::vector<ResultsContainerType> type1;
    ComputeKtoPiPiGparity<A2Apolicies>::type1(type1,
					      k_pi_separation, params.jp.pipi_separation, params.jp.tstep_type12, params.jp.xyzstep_type1, p_pi1,
					      mf_ls_ww, mf_ll_con,
					      V, V_s,
					      W, W_s);
    for(int kpi_idx=0;kpi_idx<k_pi_separation.size();kpi_idx++){
      std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_type1_deltat_" << k_pi_separation[kpi_idx] << src_fappend << "_sep_" << params.jp.pipi_separation;
#ifndef DAIQIAN_PION_PHASE_CONVENTION
      os << "_mom" << p_pi1.file_str(2);
#else
      os << "_mom" << (-p_pi1).file_str(2);
#endif
      os << postpend;
      type1[kpi_idx].write(os.str());
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      type1[kpi_idx].write(os.str(),true);
#endif
    }
    printMem("Memory status after type1 K->pipi");
  }
  print_time("main","K->pipi type 1",time+dclock());
  printMem("Memory after type1 K->pipi");

  //Type 2 and 3 are optimized by performing the sum over pipi momentum orientations within the contraction
  time = -dclock();
  {
    if(!UniqueID()) printf("Starting type 2 contractions with source %s\n", src_descr.c_str());
    std::vector<ResultsContainerType> type2;
    ComputeKtoPiPiGparity<A2Apolicies>::type2(type2,
					      k_pi_separation, params.jp.pipi_separation, params.jp.tstep_type12, type23_pion_mom,
					      mf_ls_ww, mf_ll_con,
					      V, V_s,
					      W, W_s);
    for(int kpi_idx=0;kpi_idx<k_pi_separation.size();kpi_idx++){
      std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_type2_deltat_" << k_pi_separation[kpi_idx] << src_fappend << "_sep_" << params.jp.pipi_separation;
      os << postpend;
      type2[kpi_idx].write(os.str());
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      type2[kpi_idx].write(os.str(),true);
#endif
    }
  }
  print_time("main","K->pipi type 2",time+dclock());
  printMem("Memory after type2 K->pipi");

  time = -dclock();
  {
    if(!UniqueID()) printf("Starting type 3 contractions with source %s\n", src_descr.c_str());
    std::vector<ResultsContainerType> type3;
    std::vector<MixDiagResultsContainerType> mix3;
    ComputeKtoPiPiGparity<A2Apolicies>::type3(type3,mix3,
					      k_pi_separation, params.jp.pipi_separation, 1, type23_pion_mom,
					      mf_ls_ww, mf_ll_con,
					      V, V_s,
					      W, W_s);
    for(int kpi_idx=0;kpi_idx<k_pi_separation.size();kpi_idx++){
      std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_type3_deltat_" << k_pi_separation[kpi_idx] << src_fappend << "_sep_" << params.jp.pipi_separation;
      os << postpend;
      write(os.str(),type3[kpi_idx],mix3[kpi_idx]);
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      write(os.str(),type3[kpi_idx],mix3[kpi_idx],true);
#endif
    }
  }
  print_time("main","K->pipi type 3",time+dclock());
  printMem("Memory after type3 K->pipi");
    

  if(do_type4){
    //Type 4 has no momentum loop as the pion disconnected part is computed as part of the pipi 2pt function calculation
    time = -dclock();
    if(!UniqueID()) printf("Starting type 4 contractions\n");
    ResultsContainerType type4;
    MixDiagResultsContainerType mix4;
      
    ComputeKtoPiPiGparity<A2Apolicies>::type4(type4, mix4,
					      1,
					      mf_ls_ww,
					      V, V_s,
					      W, W_s);
      
    {
      std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_type4";
      os << postpend;
      write(os.str(),type4,mix4);
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      write(os.str(),type4,mix4,true);
#endif
    }
    print_time("main","K->pipi type 4",time+dclock());
  }

  printMem("Memory at end of K->pipi contractions");
}

template<typename PionMomentumPolicy>
void computeKtoPiPiContractions(const A2AvectorV<A2Apolicies> &V, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W,
				const A2AvectorV<A2Apolicies> &V_s, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W_s,
				const LSWWmesonFields &mf_ls_ww_con, MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con,
				const PionMomentumPolicy &pion_mom, const int conf, const Parameters &params, 
				const std::string &src_descr, const std::string &src_fappend, bool do_type4, const std::string &postpend = ""){
#define TYPE1_DO_ASSUME_ROTINVAR_GP3  //For GPBC in 3 directions we can assume rotational invariance around the G-parity diagonal vector (1,1,1) and therefore calculate only one off-diagonal momentum
  
#ifdef TYPE1_DO_ASSUME_ROTINVAR_GP3 

  struct Type1momentumSubset{
    inline int nMom() const{ return 4; }
    inline ThreeMomentum getMesonMomentum(const int i) const{ 
      switch(i){
      case 0:
	return ThreeMomentum(-2,-2,-2);
      case 1:
	return ThreeMomentum(2,-2,-2);
      case 2:
	return ThreeMomentum(2,2,2);
      case 3:
	return ThreeMomentum(-2,2,2);
      default:
	assert(0);
      }
    }
  };
  Type1momentumSubset type1_mom;
  computeKtoPiPiContractions(V, W, V_s, W_s, mf_ls_ww_con, mf_ll_con, type1_mom, pion_mom, conf, params, src_descr, src_fappend, do_type4, postpend);

#else
  computeKtoPiPiContractions(V, W, V_s, W_s, mf_ls_ww_con, mf_ll_con, pion_mom, pion_mom, conf, params, src_descr, src_fappend, do_type4, postpend);
#endif
}



template<typename LSWWmomentumPolicy>		
void computeKtoPipiWWmesonFields(LSWWmesonFields &mf_ls_ww_con,
				 typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W,
				 typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W_s,
				 Lattice &lat, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
				 const LSWWmomentumPolicy &lsWW_mom, const Parameters &params, bool randomize_mf){
  if(!UniqueID()) printf("Computing WW light-heavy meson fields\n");
  double time = -dclock();

  if(randomize_mf){
    const int Lt=GJP.Tnodes()*GJP.TnodeSites();
    mf_ls_ww_con.mf_ls_ww.resize(Lt);
    for(int t=0;t<Lt;t++){ 
      mf_ls_ww_con.mf_ls_ww[t].setup(W,W_s,t,t); 
      mf_ls_ww_con.mf_ls_ww[t].testRandom();
    }
  }else{
    ComputeKtoPiPiGparity<A2Apolicies>::generatelsWWmesonfields(mf_ls_ww_con.mf_ls_ww,W,W_s,lsWW_mom,params.jp.kaon_rad,lat, field3dparams);
  }

  print_time("main","WW meson fields",time+dclock());
  printMem("Memory after WW meson fields");
}


template<typename PionMomentumPolicy>
void computeKtoPiPiContractions(const A2AvectorV<A2Apolicies> &V, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W,
				const A2AvectorV<A2Apolicies> &V_s, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W_s,
				const LSWWmesonFields &mf_ls_ww_con, MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con_2s,
				const PionMomentumPolicy &pion_mom, const int conf, const Parameters &params, const std::string &postpend = ""){
  for(int i=0;i<GJP.Gparity()+1;i++){
    computeKtoPiPiContractions(V,W,V_s,W_s,mf_ls_ww_con, 
			       i==0 ? mf_ll_con : mf_ll_con_2s,
			       pion_mom, conf, params,
			       i==0 ? "1s" : "2s",
			       i==0 ? "" : "_src2s",
			       i==0,
			       postpend);
   }
}

//Both 2s and 1s
template<typename PionMomentumPolicy, typename LSWWmomentumPolicy>
void computeKtoPiPi(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con_2s,
		    const A2AvectorV<A2Apolicies> &V, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W,
		    const A2AvectorV<A2Apolicies> &V_s, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W_s,
		    Lattice &lat, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
		    const PionMomentumPolicy &pion_mom, const LSWWmomentumPolicy &lsWW_mom, const int conf, const Parameters &params, bool randomize_mf,
		    LSWWmesonFields* mf_ls_ww_keep = NULL){

  
  //We first need to generate the light-strange W*W contraction
  LSWWmesonFields mf_ls_ww_con;
  computeKtoPipiWWmesonFields(mf_ls_ww_con,W,W_s,lat,field3dparams,lsWW_mom,params,randomize_mf);
  printMem("Memory after computing W*W meson fields");

  computeKtoPiPiContractions(V,W,V_s,W_s,mf_ls_ww_con,mf_ll_con,mf_ll_con_2s,pion_mom,conf,params);
  
  if(mf_ls_ww_keep != NULL) mf_ls_ww_keep->move(mf_ls_ww_con.mf_ls_ww);
}

//Just 1s
template<typename PionMomentumPolicy, typename LSWWmomentumPolicy>
void computeKtoPiPi(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con,
		    const A2AvectorV<A2Apolicies> &V, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W,
		    const A2AvectorV<A2Apolicies> &V_s, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W_s,
		    Lattice &lat, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
		    const PionMomentumPolicy &pion_mom, const LSWWmomentumPolicy &lsWW_mom, const int conf, const Parameters &params, bool randomize_mf,
		    LSWWmesonFields* mf_ls_ww_keep = NULL){

  
  //We first need to generate the light-strange W*W contraction
  LSWWmesonFields mf_ls_ww_con;
  computeKtoPipiWWmesonFields(mf_ls_ww_con,W,W_s,lat,field3dparams,lsWW_mom,params,randomize_mf);
  printMem("Memory after computing W*W meson fields");

  computeKtoPiPiContractions(V,W,V_s,W_s,mf_ls_ww_con,mf_ll_con,pion_mom,conf,params,"1s","",true,"");

  if(mf_ls_ww_keep != NULL) mf_ls_ww_keep->move(mf_ls_ww_con.mf_ls_ww);
}





//Versions of the above but where meson fields are temporarily stored to disk to save space

template<typename PionMomentumPolicy>
void computeKtoPiPiContractionsDumpRestore(const A2AvectorV<A2Apolicies> &V, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W,
					   const A2AvectorV<A2Apolicies> &V_s, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W_s,
					   const LSWWmesonFields &mf_ls_ww_con, MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con_2s,
					   const PionMomentumPolicy &pion_mom, const int conf, const Parameters &params, bool do_restore, const std::string &postpend = ""){
  bool redistribute = false; 
#ifdef NODE_DISTRIBUTE_MESONFIELDS
  redistribute = true;
#endif

  mf_ll_con_2s.dumpToDiskAndFree("dump_mf_ll_2s");

  computeKtoPiPiContractions(V,W,V_s,W_s,mf_ls_ww_con, 
			     mf_ll_con,
			     pion_mom, conf, params,
			     "1s","",true,postpend);
  mf_ll_con.dumpToDiskAndFree("dump_mf_ll_1s");
  mf_ll_con_2s.restoreFromDisk("dump_mf_ll_2s", false); //no point redistributing as we are about to use them!

  computeKtoPiPiContractions(V,W,V_s,W_s,mf_ls_ww_con, 
			     mf_ll_con_2s,
			     pion_mom, conf, params,
			     "2s","_src2s",false,postpend);
  
  if(do_restore) mf_ll_con.restoreFromDisk("dump_mf_ll_1s", redistribute);
}

template<typename PionMomentumPolicy, typename LSWWmomentumPolicy>
void computeKtoPiPiDumpRestore(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con_2s,
			       const A2AvectorV<A2Apolicies> &V, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W,
			       const A2AvectorV<A2Apolicies> &V_s, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W_s,
			       Lattice &lat, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
			       const PionMomentumPolicy &pion_mom, const LSWWmomentumPolicy &lsWW_mom, const int conf, const Parameters &params, bool randomize_mf, bool do_restore,
			       LSWWmesonFields* mf_ls_ww_keep = NULL){

  
  //We first need to generate the light-strange W*W contraction
  LSWWmesonFields mf_ls_ww_con;
  computeKtoPipiWWmesonFields(mf_ls_ww_con,W,W_s,lat,field3dparams,lsWW_mom,params,randomize_mf);
  printMem("Memory after computing W*W meson fields");

  computeKtoPiPiContractionsDumpRestore(V,W,V_s,W_s,mf_ls_ww_con,mf_ll_con,mf_ll_con_2s,pion_mom,conf,params,do_restore);
  
  if(mf_ls_ww_keep != NULL) mf_ls_ww_keep->move(mf_ls_ww_con.mf_ls_ww);
}

#endif
