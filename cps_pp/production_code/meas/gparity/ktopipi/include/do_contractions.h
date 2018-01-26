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


void doContractions(const int conf, Parameters &params, const CommandLineArgs &cmdline, Lattice& lat,
		    A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W,
		    A2AvectorV<A2Apolicies> &V_s, A2AvectorW<A2Apolicies> &W_s,
		    const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
#ifdef USE_STANDARD_AND_SYMMETRIC_MOM_POLICIES
  doContractionsStandardAndSymmetric(conf,params,cmdline,lat,V,W,V_s,W_s,field3dparams);
#else
  doContractionsBasic(conf,params,cmdline,lat,V,W,V_s,W_s,field3dparams);
#endif
}

#endif
