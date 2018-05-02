#ifndef _KTOPIPI_MAIN_A2A_DO_CONTRACTIONS_H_
#define _KTOPIPI_MAIN_A2A_DO_CONTRACTIONS_H_

void doContractionsBasic(const int conf, Parameters &params, const CommandLineArgs &cmdline, Lattice& lat,
			 A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W,
			 A2AvectorV<A2Apolicies> &V_s, A2AvectorW<A2Apolicies> &W_s,
			 const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
#ifdef USE_SYMMETRIC_MOM_POLICIES
  typedef SymmetricKaonMomentaPolicy KaonMomentumPolicy;
  typedef StationarySigmaMomentaPolicy SigmaMomentumPolicy; //as we just save the meson fields this is unimportant (only applies to sigma because we do +/-p anyway)
  typedef SymmetricPionMomentaPolicy PionMomentumPolicy;
  typedef SymmetricLSWWmomentaPolicy LSWWmomentumPolicy;
#elif defined(USE_SYMMETRIC_PION_MOM_POLICIES) //pion only symmetrized
  typedef StationaryKaonMomentaPolicy KaonMomentumPolicy;
  typedef StationarySigmaMomentaPolicy SigmaMomentumPolicy;
  typedef SymmetricPionMomentaPolicy PionMomentumPolicy;
  typedef StandardLSWWmomentaPolicy LSWWmomentumPolicy;
#else
  typedef StationaryKaonMomentaPolicy KaonMomentumPolicy;
  typedef StationarySigmaMomentaPolicy SigmaMomentumPolicy;
  typedef StandardPionMomentaPolicy PionMomentumPolicy;
  typedef StandardLSWWmomentaPolicy LSWWmomentumPolicy;
#endif
  //-------------------Fix gauge----------------------------
  doGaugeFix(lat, cmdline.skip_gauge_fix, params);

  //-------------------------Compute the kaon two-point function---------------------------------

  KaonMomentumPolicy kaon_mom;
  if(cmdline.do_kaon2pt) computeKaon2pt(V,W,V_s,W_s,kaon_mom,conf,lat,params,field3dparams,cmdline.randomize_mf);

  //----------------------------Compute the sigma meson fields---------------------------------
  SigmaMomentumPolicy sigma_mom;
  if(cmdline.do_sigma) computeSigmaMesonFields(V,W,sigma_mom,conf,lat,params,field3dparams);

  //The pion two-point function and pipi/k->pipi all utilize the same meson fields. Generate those here
  //For convenience pointers to the meson fields are collected into a single object that is passed to the compute methods
  PionMomentumPolicy pion_mom; //these are the W and V momentum combinations

  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con; //stores light-light meson fields, accessible by momentum
  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con_2s; //Gparity only

  computeLLmesonFields(mf_ll_con, mf_ll_con_2s, V, W, pion_mom, conf, lat, params, field3dparams, cmdline.randomize_mf);

  //----------------------------Compute the pion two-point function---------------------------------
  if(cmdline.do_pion2pt) computePion2pt(mf_ll_con, pion_mom, conf, params);
    
  //------------------------------I=0 and I=2 PiPi two-point function---------------------------------
  if(cmdline.do_pipi) computePiPi2pt(mf_ll_con, pion_mom, conf, params);

  //--------------------------------------K->pipi contractions--------------------------------------------------------
  LSWWmomentumPolicy lsWW_mom;
  if(cmdline.do_ktopipi) computeKtoPiPi(mf_ll_con,mf_ll_con_2s,V,W,V_s,W_s,lat,field3dparams,pion_mom,lsWW_mom,conf,params, cmdline.randomize_mf);
}


//Same as the above but do both the standard and symmetric V/W^dag momentum assignments
void doContractionsStandardAndSymmetric(const int conf, Parameters &params, const CommandLineArgs &cmdline, Lattice& lat,
				      A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W,
				      A2AvectorV<A2Apolicies> &V_s, A2AvectorW<A2Apolicies> &W_s,
				      const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  //-------------------Fix gauge----------------------------
  doGaugeFix(lat, cmdline.skip_gauge_fix, params);

  //-------------------------Compute the kaon two-point function---------------------------------
  StationaryKaonMomentaPolicy kaon_mom_std;
  ReverseKaonMomentaPolicy kaon_mom_rev;
  if(cmdline.do_kaon2pt) computeKaon2ptStandardAndSymmetric(V,W,V_s,W_s,kaon_mom_std,kaon_mom_rev,conf,lat,params,field3dparams,cmdline.randomize_mf);

  //----------------------------Compute the sigma meson fields---------------------------------
  StationarySigmaMomentaPolicy sigma_mom_std;
  if(cmdline.do_sigma) computeSigmaMesonFields(V,W,sigma_mom_std,conf,lat,params,field3dparams);

  //The pion two-point function and pipi/k->pipi all utilize the same meson fields. Generate those here
  //For convenience pointers to the meson fields are collected into a single object that is passed to the compute methods
  StandardPionMomentaPolicy pion_mom_std; //these are the W and V momentum combinations
  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con_std; //stores light-light meson fields, accessible by momentum
  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con_2s_std; //Gparity only

  computeLLmesonFields(mf_ll_con_std, mf_ll_con_2s_std, V, W, pion_mom_std, conf, lat, params, field3dparams, cmdline.randomize_mf);

  //----------------------------Compute the pion two-point function---------------------------------
  if(cmdline.do_pion2pt) computePion2pt(mf_ll_con_std, pion_mom_std, conf, params);
    
  //------------------------------I=0 and I=2 PiPi two-point function---------------------------------
  if(cmdline.do_pipi) computePiPi2pt(mf_ll_con_std, pion_mom_std, conf, params);

  //--------------------------------------K->pipi contractions--------------------------------------------------------
  StandardLSWWmomentaPolicy lsWW_mom_std;
  LSWWmesonFields mf_ls_ww_con_std;
  if(cmdline.do_ktopipi){
#ifdef SYMM_DUMP_RESTORE
    computeKtoPiPiDumpRestore(mf_ll_con_std,mf_ll_con_2s_std,V,W,V_s,W_s,lat,field3dparams,pion_mom_std,lsWW_mom_std,conf,params, cmdline.randomize_mf, true,&mf_ls_ww_con_std);
#else
    computeKtoPiPi(mf_ll_con_std,mf_ll_con_2s_std,V,W,V_s,W_s,lat,field3dparams,pion_mom_std,lsWW_mom_std,conf,params, cmdline.randomize_mf, &mf_ls_ww_con_std);
#endif
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  //Do the reverse momentum assignment, average with standard and repeat measurements
  ////////////////////////////////////////////////////////////////////////////////////////////
  
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
  {
    if(!UniqueID()){
      std::ostringstream os; DistributedMemoryStorage::block_allocator().stats(os);
      printf("Trimming block allocator. Current stats: %s\n",os.str().c_str()); fflush(stdout);
    }
    DistributedMemoryStorage::block_allocator().trim();
    if(!UniqueID()){
      std::ostringstream os; DistributedMemoryStorage::block_allocator().stats(os);
      printf("Post-trim stats: %s\n",os.str().c_str()); fflush(stdout);
    }
  }
#endif

  ReversePionMomentaPolicy pion_mom_rev; //these are the W and V momentum combinations
  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con_symm; //stores light-light meson fields, accessible by momentum
  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con_2s_symm; //Gparity only
  computeLLmesonFields(mf_ll_con_symm, mf_ll_con_2s_symm, V, W, pion_mom_rev, conf, lat, params, field3dparams, cmdline.randomize_mf, "_rev");

  mf_ll_con_symm.average(mf_ll_con_std);
  mf_ll_con_2s_symm.average(mf_ll_con_2s_std);
  mf_ll_con_std.free_mem();
  mf_ll_con_2s_std.free_mem();

  //----------------------------Compute the pion two-point function---------------------------------
  if(cmdline.do_pion2pt) computePion2pt(mf_ll_con_symm, pion_mom_std, conf, params, "_symm");
    
  //------------------------------I=0 and I=2 PiPi two-point function---------------------------------
  if(cmdline.do_pipi) computePiPi2pt(mf_ll_con_symm, pion_mom_std, conf, params, "_symm");

  //--------------------------------------K->pipi contractions--------------------------------------------------------
  if(cmdline.do_ktopipi){
    LSWWmesonFields mf_ls_ww_con_symm;
    ReverseLSWWmomentaPolicy lsWW_mom_rev;
    computeKtoPipiWWmesonFields(mf_ls_ww_con_symm,W,W_s,lat,field3dparams,lsWW_mom_rev,params,cmdline.randomize_mf);
    mf_ls_ww_con_symm.average(mf_ls_ww_con_std);
    mf_ls_ww_con_std.free_mem();
    
    printMem("Memory after computing W*W meson fields");
#ifdef SYMM_DUMP_RESTORE
    computeKtoPiPiContractionsDumpRestore(V,W,V_s,W_s,mf_ls_ww_con_symm,mf_ll_con_symm,mf_ll_con_2s_symm,pion_mom_std,conf,params,false,"_symm");
#else
    computeKtoPiPiContractions(V,W,V_s,W_s,mf_ls_ww_con_symm,mf_ll_con_symm,mf_ll_con_2s_symm,pion_mom_std,conf,params,"_symm");
#endif
  }
}



//Same as the above but do both the standard and symmetric V/W^dag momentum assignments. In this version we do not symmetrize the kaon momenta
void doContractionsStandardAndSymmetricPion(const int conf, Parameters &params, const CommandLineArgs &cmdline, Lattice& lat,
				      A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W,
				      A2AvectorV<A2Apolicies> &V_s, A2AvectorW<A2Apolicies> &W_s,
				      const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  //-------------------Fix gauge----------------------------
  doGaugeFix(lat, cmdline.skip_gauge_fix, params);

  //-------------------------Compute the kaon two-point function---------------------------------
  StationaryKaonMomentaPolicy kaon_mom_std;
  if(cmdline.do_kaon2pt) computeKaon2pt(V,W,V_s,W_s,kaon_mom_std,conf,lat,params,field3dparams,cmdline.randomize_mf);

  //----------------------------Compute the sigma meson fields---------------------------------
  StationarySigmaMomentaPolicy sigma_mom_std;
  if(cmdline.do_sigma) computeSigmaMesonFields(V,W,sigma_mom_std,conf,lat,params,field3dparams);

  //The pion two-point function and pipi/k->pipi all utilize the same meson fields. Generate those here
  //For convenience pointers to the meson fields are collected into a single object that is passed to the compute methods
  StandardPionMomentaPolicy pion_mom_std; //these are the W and V momentum combinations
  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con_std; //stores light-light meson fields, accessible by momentum
  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con_2s_std; //Gparity only

  computeLLmesonFields(mf_ll_con_std, mf_ll_con_2s_std, V, W, pion_mom_std, conf, lat, params, field3dparams, cmdline.randomize_mf);

  //----------------------------Compute the pion two-point function---------------------------------
  if(cmdline.do_pion2pt) computePion2pt(mf_ll_con_std, pion_mom_std, conf, params);
    
  //------------------------------I=0 and I=2 PiPi two-point function---------------------------------
  if(cmdline.do_pipi) computePiPi2pt(mf_ll_con_std, pion_mom_std, conf, params);

  //--------------------------------------K->pipi contractions--------------------------------------------------------
  StandardLSWWmomentaPolicy lsWW_mom_std;
  LSWWmesonFields mf_ls_ww_con_std;
  if(cmdline.do_ktopipi){
    computeKtoPiPi(mf_ll_con_std,mf_ll_con_2s_std,V,W,V_s,W_s,lat,field3dparams,pion_mom_std,lsWW_mom_std,conf,params, cmdline.randomize_mf,&mf_ls_ww_con_std);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  //Do the reverse momentum assignment, average with standard and repeat measurements
  ////////////////////////////////////////////////////////////////////////////////////////////
  
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
  {
    if(!UniqueID()){
      std::ostringstream os; DistributedMemoryStorage::block_allocator().stats(os);
      printf("Trimming block allocator. Current stats: %s\n",os.str().c_str()); fflush(stdout);
    }
    DistributedMemoryStorage::block_allocator().trim();
    if(!UniqueID()){
      std::ostringstream os; DistributedMemoryStorage::block_allocator().stats(os);
      printf("Post-trim stats: %s\n",os.str().c_str()); fflush(stdout);
    }
  }
#endif

  ReversePionMomentaPolicy pion_mom_rev; //these are the W and V momentum combinations
  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con_symm; //stores light-light meson fields, accessible by momentum
  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con_2s_symm; //Gparity only
  computeLLmesonFields(mf_ll_con_symm, mf_ll_con_2s_symm, V, W, pion_mom_rev, conf, lat, params, field3dparams, cmdline.randomize_mf, "_rev");

  mf_ll_con_symm.average(mf_ll_con_std);
  mf_ll_con_2s_symm.average(mf_ll_con_2s_std);
  mf_ll_con_std.free_mem();
  mf_ll_con_2s_std.free_mem();

  //----------------------------Compute the pion two-point function---------------------------------
  if(cmdline.do_pion2pt) computePion2pt(mf_ll_con_symm, pion_mom_std, conf, params, "_symm");
    
  //------------------------------I=0 and I=2 PiPi two-point function---------------------------------
  if(cmdline.do_pipi) computePiPi2pt(mf_ll_con_symm, pion_mom_std, conf, params, "_symm");

  //--------------------------------------K->pipi contractions--------------------------------------------------------
  if(cmdline.do_ktopipi){  
    printMem("Memory after computing W*W meson fields");
    computeKtoPiPiContractions(V,W,V_s,W_s,mf_ls_ww_con_std,mf_ll_con_symm,mf_ll_con_2s_symm,pion_mom_std,conf,params,"_symmpi");

    //Do the reversed kaon too for testing
    // {
    //   ReverseLSWWmomentaPolicy lsWW_mom_rev;
    //   LSWWmesonFields mf_ls_ww_con_rev;
    //   computeKtoPipiWWmesonFields(mf_ls_ww_con_rev,W,W_s,lat,field3dparams,lsWW_mom_rev,params,cmdline.randomize_mf);
    //   computeKtoPiPiContractions(V,W,V_s,W_s,mf_ls_ww_con_rev,mf_ll_con_symm,mf_ll_con_2s_symm,pion_mom_std,conf,params,"_symmpi_revtest");
    // }
  }
}

struct AllPionMomenta{
  std::vector<ThreeMomentum> p;
  inline int nMom() const{ return p.size(); }

  template<typename From>
  inline void import(const From &from){
    for(int i=0;i<from.nMom();i++) p.push_back(from.getMesonMomentum(i));
  }
    
  inline ThreeMomentum getMesonMomentum(const int i) const{ return p[i]; }
};

//Extended calculation with additional pion momenta
void doContractionsExtendedCalcV1(const int conf, Parameters &params, const CommandLineArgs &cmdline, Lattice& lat,
				  A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W,
				  A2AvectorV<A2Apolicies> &V_s, A2AvectorW<A2Apolicies> &W_s,
				  const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  //-------------------Fix gauge----------------------------
  doGaugeFix(lat, cmdline.skip_gauge_fix, params);

  //-------------------------Compute the kaon two-point function---------------------------------
  StationaryKaonMomentaPolicy kaon_mom_std;
  if(cmdline.do_kaon2pt) computeKaon2pt(V,W,V_s,W_s,kaon_mom_std,conf,lat,params,field3dparams,cmdline.randomize_mf);

  //-------------------------Compute the LL meson fields ------------------------  
  //Compute the 1s meson fields with the original set of momenta
  StandardPionMomentaPolicy pion_mom_std;
  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con_std;
  computeLLmesonFields1s(mf_ll_con_std, V, W, pion_mom_std, lat, params, field3dparams, cmdline.randomize_mf);

  //Do 1s pion, compute K->pipi
  //then reverse and symmetrize, then recompute K->pipi
  //After this we no longer need the strange A2A vectors and we can free them for extra memory.
  StandardLSWWmomentaPolicy lsWW_mom_std;
  LSWWmesonFields mf_ls_ww_con_std;
  if(cmdline.do_ktopipi){
    computeKtoPiPi(mf_ll_con_std,V,W,V_s,W_s,lat,field3dparams,pion_mom_std,lsWW_mom_std,conf,params, cmdline.randomize_mf,&mf_ls_ww_con_std);
  }

  ReversePionMomentaPolicy pion_mom_rev; //these are the W and V momentum combinations
  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con_symm; //stores light-light meson fields, accessible by momentum
  computeLLmesonFields1s(mf_ll_con_symm, V, W, pion_mom_rev, lat, params, field3dparams, cmdline.randomize_mf);
  mf_ll_con_symm.average(mf_ll_con_std);
  mf_ll_con_std.free_mem();

  if(cmdline.do_ktopipi){  
    computeKtoPiPiContractions(V,W,V_s,W_s,mf_ls_ww_con_std,mf_ll_con_symm,pion_mom_std,conf,params,"1s","",false,"_symmpi");
  }

  //Free strange quark fields 
  V_s.free_mem();
  W_s.free_mem();
  mf_ls_ww_con_std.free_mem();
  
  //Compute the other 24 meson fields (symmetrized by default)
  ExtendedPionMomentaPolicy pion_mom_extended_symm;
  computeLLmesonFields1s(mf_ll_con_symm, V, W, pion_mom_extended_symm, lat, params, field3dparams, cmdline.randomize_mf);

  AllPionMomenta all_pimom; all_pimom.import(pion_mom_std); all_pimom.import(pion_mom_extended_symm);

   //----------------------------Compute the pion two-point function--------------------------------- */
  if(cmdline.do_pion2pt) computePion2pt(mf_ll_con_symm, all_pimom, conf, params, "_symm");

  if(cmdline.do_pipi) computePiPi2ptFromFile(mf_ll_con_symm, "pipi_correlators.in", all_pimom, conf, params, "_symm");
}








void doContractions(const int conf, Parameters &params, const CommandLineArgs &cmdline, Lattice& lat,
		    A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W,
		    A2AvectorV<A2Apolicies> &V_s, A2AvectorW<A2Apolicies> &W_s,
		    const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  if(cmdline.nthread_contractions != cmdline.nthreads && !UniqueID()) printf("Changing threads to %d for contractions\n", cmdline.nthread_contractions);
  GJP.SetNthreads(cmdline.nthread_contractions);

#ifdef USE_STANDARD_AND_SYMMETRIC_MOM_POLICIES
  doContractionsStandardAndSymmetric(conf,params,cmdline,lat,V,W,V_s,W_s,field3dparams);
#elif defined(USE_STANDARD_AND_SYMMETRIC_PION_MOM_POLICIES)
  doContractionsStandardAndSymmetricPion(conf,params,cmdline,lat,V,W,V_s,W_s,field3dparams);
#elif defined(DO_EXTENDED_CALC_V1)
  doContractionsExtendedCalcV1(conf,params,cmdline,lat,V,W,V_s,W_s,field3dparams);
#else
  doContractionsBasic(conf,params,cmdline,lat,V,W,V_s,W_s,field3dparams);
#endif
}

#endif
