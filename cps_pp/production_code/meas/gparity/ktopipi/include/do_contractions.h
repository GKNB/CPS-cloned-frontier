#ifndef _KTOPIPI_MAIN_A2A_DO_CONTRACTIONS_H_
#define _KTOPIPI_MAIN_A2A_DO_CONTRACTIONS_H_


void doContractions(const int conf, Parameters &params, const CommandLineArgs &cmdline, Lattice& lat,
		    A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W,
		    A2AvectorV<A2Apolicies> &V_s, A2AvectorW<A2Apolicies> &W_s,
		    const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  //-------------------Fix gauge----------------------------
  doGaugeFix(lat, cmdline.skip_gauge_fix, params);

  //-------------------------Compute the kaon two-point function---------------------------------
  StationaryKaonMomentaPolicy kaon_mom;
  if(cmdline.do_kaon2pt) computeKaon2pt(V,W,V_s,W_s,kaon_mom,conf,lat,params,field3dparams);

  //----------------------------Compute the sigma meson fields---------------------------------
  StationarySigmaMomentaPolicy sigma_mom;
  if(cmdline.do_sigma) computeSigmaMesonFields(V,W,sigma_mom,conf,lat,params,field3dparams);

  //The pion two-point function and pipi/k->pipi all utilize the same meson fields. Generate those here
  //For convenience pointers to the meson fields are collected into a single object that is passed to the compute methods
  StandardPionMomentaPolicy pion_mom; //these are the W and V momentum combinations

  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con; //stores light-light meson fields, accessible by momentum
  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con_2s; //Gparity only

  computeLLmesonFields(mf_ll_con, mf_ll_con_2s, V, W, pion_mom, conf, lat, params, field3dparams);

  //----------------------------Compute the pion two-point function---------------------------------
  if(cmdline.do_pion2pt) computePion2pt(mf_ll_con, pion_mom, conf, params);
    
  //------------------------------I=0 and I=2 PiPi two-point function---------------------------------
  if(cmdline.do_pipi) computePiPi2pt(mf_ll_con, pion_mom, conf, params);

  //--------------------------------------K->pipi contractions--------------------------------------------------------
  StandardLSWWmomentaPolicy lsWW_mom;
  if(cmdline.do_ktopipi) computeKtoPiPi(mf_ll_con,mf_ll_con_2s,V,W,V_s,W_s,lat,field3dparams,pion_mom,lsWW_mom,conf,params);
}


#endif
