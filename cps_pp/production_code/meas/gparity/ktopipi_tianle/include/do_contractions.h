#ifndef _KTOPIPI_MAIN_A2A_DO_CONTRACTIONS_H_
#define _KTOPIPI_MAIN_A2A_DO_CONTRACTIONS_H_

template<typename Vtype, typename Wtype>
void doContractionsBasic(const int conf, Parameters &params, const CommandLineArgs &cmdline, Lattice& lat,
			 Vtype &V, Wtype &W,
			 Vtype &V_s, Wtype &W_s,
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
  typedef getMesonFieldType<Wtype,Vtype> MesonFieldWVtype;

  //-------------------Fix gauge----------------------------
  doGaugeFix(lat, cmdline.skip_gauge_fix, params.fix_gauge_arg);

  //-------------------------Compute the kaon two-point function---------------------------------

  KaonMomentumPolicy kaon_mom;
  if(cmdline.do_kaon2pt) computeKaon2pt(V,W,V_s,W_s,kaon_mom,conf,lat,params,field3dparams,cmdline.randomize_mf);

  //----------------------------Compute the sigma meson fields---------------------------------
  SigmaMomentumPolicy sigma_mom;
  if(cmdline.do_sigma) computeSigmaMesonFields(V,W,sigma_mom,conf,lat,params,field3dparams);

  //The pion two-point function and pipi/k->pipi all utilize the same meson fields. Generate those here
 //For convenience pointers to the meson fields are collected into a single object that is passed to the compute methods
  PionMomentumPolicy pion_mom; //these are the W and V momentum combinations

  MesonFieldMomentumContainer<MesonFieldWVtype> mf_ll_con; //stores light-light meson fields, accessible by momentum
  MesonFieldMomentumContainer<MesonFieldWVtype> mf_ll_con_2s; //Gparity only

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
template<typename Vtype, typename Wtype>
void doContractionsStandardAndSymmetric(const int conf, Parameters &params, const CommandLineArgs &cmdline, Lattice& lat,
					Vtype &V, Wtype &W,
					Vtype &V_s, Wtype &W_s,
				      const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  typedef getMesonFieldType<Wtype,Vtype> MesonFieldWVtype;

  //-------------------Fix gauge----------------------------
  doGaugeFix(lat, cmdline.skip_gauge_fix, params.fix_gauge_arg);

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
  MesonFieldMomentumContainer<MesonFieldWVtype> mf_ll_con_std; //stores light-light meson fields, accessible by momentum
  MesonFieldMomentumContainer<MesonFieldWVtype> mf_ll_con_2s_std; //Gparity only

  computeLLmesonFields(mf_ll_con_std, mf_ll_con_2s_std, V, W, pion_mom_std, conf, lat, params, field3dparams, cmdline.randomize_mf);

  //----------------------------Compute the pion two-point function---------------------------------
  if(cmdline.do_pion2pt) computePion2pt(mf_ll_con_std, pion_mom_std, conf, params);
    
  //------------------------------I=0 and I=2 PiPi two-point function---------------------------------
  if(cmdline.do_pipi) computePiPi2pt(mf_ll_con_std, pion_mom_std, conf, params);

  //--------------------------------------K->pipi contractions--------------------------------------------------------
  StandardLSWWmomentaPolicy lsWW_mom_std;
  LSWWmesonFields<Wtype> mf_ls_ww_con_std;
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
    {
      std::ostringstream os; DistributedMemoryStorage::block_allocator().stats(os);
      LOGA2A << "Trimming block allocator. Current stats: " << os.str() << std::endl;
    }
    DistributedMemoryStorage::block_allocator().trim();
    {
      std::ostringstream os; DistributedMemoryStorage::block_allocator().stats(os);
      LOGA2A << "Post-trim stats: " << os.str() << std::endl;
    }
  }
#endif

  ReversePionMomentaPolicy pion_mom_rev; //these are the W and V momentum combinations
  MesonFieldMomentumContainer<MesonFieldWVtype> mf_ll_con_symm; //stores light-light meson fields, accessible by momentum
  MesonFieldMomentumContainer<MesonFieldWVtype> mf_ll_con_2s_symm; //Gparity only
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
    LSWWmesonFields<Wtype> mf_ls_ww_con_symm;
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
template<typename Vtype, typename Wtype>
void doContractionsStandardAndSymmetricPion(const int conf, Parameters &params, const CommandLineArgs &cmdline, Lattice& lat,
					    Vtype &V, Wtype &W,
					    Vtype &V_s, Wtype &W_s,
				      const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  typedef getMesonFieldType<Wtype,Vtype> MesonFieldWVtype;

  //-------------------Fix gauge----------------------------
  doGaugeFix(lat, cmdline.skip_gauge_fix, params.fix_gauge_arg);

  //-------------------------Compute the kaon two-point function---------------------------------
  StationaryKaonMomentaPolicy kaon_mom_std;
  if(cmdline.do_kaon2pt) computeKaon2pt(V,W,V_s,W_s,kaon_mom_std,conf,lat,params,field3dparams,cmdline.randomize_mf);

  //----------------------------Compute the sigma meson fields---------------------------------
  StationarySigmaMomentaPolicy sigma_mom_std;
  if(cmdline.do_sigma) computeSigmaMesonFields(V,W,sigma_mom_std,conf,lat,params,field3dparams);

  //The pion two-point function and pipi/k->pipi all utilize the same meson fields. Generate those here
  //For convenience pointers to the meson fields are collected into a single object that is passed to the compute methods
  StandardPionMomentaPolicy pion_mom_std; //these are the W and V momentum combinations
  MesonFieldMomentumContainer<MesonFieldWVtype> mf_ll_con_std; //stores light-light meson fields, accessible by momentum
  MesonFieldMomentumContainer<MesonFieldWVtype> mf_ll_con_2s_std; //Gparity only

  computeLLmesonFields(mf_ll_con_std, mf_ll_con_2s_std, V, W, pion_mom_std, conf, lat, params, field3dparams, cmdline.randomize_mf);

  //----------------------------Compute the pion two-point function---------------------------------
  if(cmdline.do_pion2pt) computePion2pt(mf_ll_con_std, pion_mom_std, conf, params);
    
  //------------------------------I=0 and I=2 PiPi two-point function---------------------------------
  if(cmdline.do_pipi) computePiPi2pt(mf_ll_con_std, pion_mom_std, conf, params);

  //--------------------------------------K->pipi contractions--------------------------------------------------------
  StandardLSWWmomentaPolicy lsWW_mom_std;
  LSWWmesonFields<Wtype> mf_ls_ww_con_std;
  if(cmdline.do_ktopipi){
    computeKtoPiPi(mf_ll_con_std,mf_ll_con_2s_std,V,W,V_s,W_s,lat,field3dparams,pion_mom_std,lsWW_mom_std,conf,params, cmdline.randomize_mf,&mf_ls_ww_con_std);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  //Do the reverse momentum assignment, average with standard and repeat measurements
  ////////////////////////////////////////////////////////////////////////////////////////////
  
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
  {
    {
      std::ostringstream os; DistributedMemoryStorage::block_allocator().stats(os);
      LOGA2A << "Trimming block allocator. Current stats: " << os.str() << std::endl;
    }
    DistributedMemoryStorage::block_allocator().trim();
    {
      std::ostringstream os; DistributedMemoryStorage::block_allocator().stats(os);
      LOGA2A << "Post-trim stats: " << os.str() << std::endl;
    }
  }
#endif

  ReversePionMomentaPolicy pion_mom_rev; //these are the W and V momentum combinations
  MesonFieldMomentumContainer<MesonFieldWVtype> mf_ll_con_symm; //stores light-light meson fields, accessible by momentum
  MesonFieldMomentumContainer<MesonFieldWVtype> mf_ll_con_2s_symm; //Gparity only
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

//Extended calculation with additional pion momenta and K->sigma
template<typename Vtype, typename Wtype>
void doContractionsExtendedCalcV1(const int conf, Parameters &params, const CommandLineArgs &cmdline, Lattice& lat,
				  Vtype &V, Wtype &W,
				  Vtype &V_s, Wtype &W_s,
				  const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  typedef getMesonFieldType<Wtype,Vtype> MesonFieldWVtype;

  if(cmdline.save_all_a2a_inputs){
    { std::ostringstream os; os << cmdline.save_all_a2a_inputs_dir << "/traj_" << conf << "_V"; V.writeParallelWithGrid(os.str()); }
    { std::ostringstream os; os << cmdline.save_all_a2a_inputs_dir << "/traj_" << conf << "_W"; W.writeParallelWithGrid(os.str()); }
    { std::ostringstream os; os << cmdline.save_all_a2a_inputs_dir << "/traj_" << conf << "_Vs"; V_s.writeParallelWithGrid(os.str()); }
    { std::ostringstream os; os << cmdline.save_all_a2a_inputs_dir << "/traj_" << conf << "_Ws"; W_s.writeParallelWithGrid(os.str()); }
  }

  //-------------------------Compute the kaon two-point function---------------------------------
  StationaryKaonMomentaPolicy kaon_mom_std;
  if(cmdline.do_kaon2pt) computeKaon2pt(V,W,V_s,W_s,kaon_mom_std,conf,lat,params,field3dparams,cmdline.randomize_mf);

  //-------------------------Compute the sigma meson fields--------------------------------------
  StationarySigmaMomentaPolicy sigma_mom;
  MesonFieldMomentumPairContainer<MesonFieldWVtype> mf_sigma;
  computeSigmaMesonFieldsExt(mf_sigma, V, W, sigma_mom, conf, lat, params, cmdline, field3dparams);

  //-------------------------Compute the K->sigma matrix elements -------------------------------
  //First compute kaon WW meson fields
  StandardLSWWmomentaPolicy lsWW_mom_std;
  LSWWmesonFields<Wtype> mf_ls_ww_con_std;
  //computeKtoPipiWWmesonFields<Vtype,Wtype>(mf_ls_ww_con_std, W, W_s, lat, field3dparams, lsWW_mom_std, params, cmdline.randomize_mf);
  ComputeKtoPiPiGparity<Vtype,Wtype>::generatelsWWmesonfields(mf_ls_ww_con_std.mf_ls_ww,W,W_s, lsWW_mom_std,params.jp.kaon_rad,lat, field3dparams);


  if(cmdline.save_all_a2a_inputs){
    std::ostringstream os; os << cmdline.save_all_a2a_inputs_dir << "/traj_" << conf << "_kaon_mfww.dat";
    mf_ls_ww_con_std.write(os.str(),false);
  }

  if(cmdline.do_ktosigma) computeKtoSigmaContractions(V, W, V_s, W_s, mf_ls_ww_con_std, mf_sigma, sigma_mom, conf, params, "1s", "", true, "");

  printMem("Memory prior to WW meson field distribute");
  mf_ls_ww_con_std.distribute();
  printMem("Memory prior to LL meson field compute");


  //----------------------------Compute the sigma 2pt function--------------------------------------
  std::vector< fVector<typename A2Apolicies::ScalarComplexType> > sigma_bub;
  if(cmdline.do_sigma2pt) computeSigma2pt<Vtype,Wtype>(sigma_bub, mf_sigma, sigma_mom, conf, params);

#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
  printMem("Memory prior to trim");
  if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
  LOGA2A << "Trimming block allocator" << std::endl;
  DistributedMemoryStorage::block_allocator().trim();
  if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
  printMem("Memory after trim");
#endif

  //-------------------------Compute the LL meson fields ------------------------  
  MesonFieldMomentumContainer<MesonFieldWVtype> mf_ll_con;

  SymmetricPionMomentaPolicy pion_mom_orig;   //Pion meson fields with original quark momentum selections (base+alt, symmetrized)
  AltExtendedPionMomentaPolicy pion_mom_extended; //The other 24 pion meson fields (base+alt, symmetrized)

  RequiredMomentum all_pimom;
  all_pimom.combineSameTotalMomentum(true);
  all_pimom.addAll(pion_mom_orig);
  all_pimom.addAll(pion_mom_extended);

  computeLLmesonFields1s(mf_ll_con, V, W, all_pimom, lat, params, field3dparams, cmdline.randomize_mf);  

  if(cmdline.save_all_a2a_inputs){
    std::ostringstream os; os << cmdline.save_all_a2a_inputs_dir << "/traj_" << conf << "_pion_mf_hyd1s_rad" << (int)params.jp.pion_rad;
    mf_ll_con.write(os.str(),true);
  }

  //----------------------------Compute pipi->sigma----------------------------------------------------
  if(cmdline.do_pipitosigma) computePiPiToSigma<Vtype,Wtype>(sigma_bub, mf_sigma,sigma_mom, mf_ll_con, all_pimom, conf, params);

  //Sigma meson fields no longer needed
  mf_sigma.free_mem();

#ifdef MESONFIELD_USE_DISTRIBUTED_STORAGE  

  /* //TEST!!! */
  /* //Deliberately cause an imbalance */
  /* { */
  /*   int nodes = 1; */
  /*   for(int i=0;i<5;i++) nodes *= GJP.Nodes(i); */

  /*   std::vector<DistributedMemoryStorage*> ptrs; */
  /*   typedef typename MesonFieldMomentumContainer<A2Apolicies>::iterator iterator; */

  /*   for(iterator it = mf_ll_con.begin(); it != mf_ll_con.end(); it++){ */
  /*     for(int t=0;t<it->second->size();t++) */
  /* 	ptrs.push_back( (DistributedMemoryStorage*)&it->second->operator[](t) ); */
  /*   } */
  /*   UniformRandomGenerator rng; */
  /*   rng.Reset(1234); */

  /*   int off = 0; */
  /*   int nrem = ptrs.size(); */
  /*   for(int n=0;n<nodes;n++){ */
  /*     int nown = (n == nodes-1 ? nrem : (int)rng.Rand(nrem,0)); */
  /*     for(int i=0;i<nown;i++) */
  /* 	ptrs[off++]->reassign(n);      */
  /*     nrem -= nown; */
  /*   } */
  /* } */
  /* //TEST */

  //Try to rebalance the number of meson fields over the nodes
  mf_ll_con.rebalance();
#endif

#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
  printMem("Memory prior to trim");
  if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
  LOGA2A << "Trimming block allocator" << std::endl;
  DistributedMemoryStorage::block_allocator().trim();
  if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
  printMem("Memory after trim");
#endif

  //--------------------------Compute the K->pipi contractions---------------------------------------
  if(cmdline.do_ktopipi){  
    mf_ls_ww_con_std.gather();

    //For K->pipi type1 with original momentum set we use the rotational symmetry to reduce the number of contractions
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
    Type1momentumSubset type1_subset;
    computeKtoPiPiContractions(V,W,V_s,W_s,mf_ls_ww_con_std,mf_ll_con, type1_subset, pion_mom_orig, conf, params,"1s","",true,"_symmpi");

    //For extended momenta do all of them (FOR NOW)
    computeKtoPiPiContractions(V,W,V_s,W_s,mf_ls_ww_con_std,mf_ll_con, pion_mom_extended, pion_mom_extended, conf, params,"1s","",false,"_symmpi_ext");
  }

  //Free strange quark fields 
  V_s.free_mem();
  W_s.free_mem();
  mf_ls_ww_con_std.free_mem();

  //----------------------------Compute the pion two-point function--------------------------------- */
  if(cmdline.do_pion2pt) computePion2pt<Vtype,Wtype>(mf_ll_con, all_pimom, conf, params, "_symm");

  //----------------------------Compute the pipi 2pt function ---------------------------------------
  if(cmdline.do_pipi) computePiPi2ptFromFile<Vtype,Wtype>(mf_ll_con, "pipi_correlators.in", all_pimom, conf, params, "_symm");

  LOGA2A << "Completed contractions" << std::endl;
}







template<typename Vtype, typename Wtype>
void doContractions(const int conf, Parameters &params, const CommandLineArgs &cmdline, Lattice& lat,
		    Vtype &V, Wtype &W,
		    Vtype &V_s, Wtype &W_s,
		    const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  if(cmdline.nthread_contractions != cmdline.nthreads) LOGA2A << "Changing threads to " << cmdline.nthread_contractions << " for contractions" << std::endl;
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