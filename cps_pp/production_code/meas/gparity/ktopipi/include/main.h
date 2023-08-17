#ifndef _MAIN_H
#define _MAIN_H

//Header for main program
using namespace cps;

#include "a2a_policies.h"
#include "cmdline.h"
#include "pipi_momfile.h"
#include "args.h"
#include "misc.h"
#include "a2a_vectors.h"
#include "kaon.h"
#include "pion.h"
#include "sigma.h"
#include "pipitosigma.h"
#include "pipi.h"
#include "ktopipi.h"
#include "ktosigma.h"
#include "do_contractions.h"

typedef typename A2Apolicies::FgridGFclass LatticeType;

void doConfiguration(const int conf, Parameters &params, const CommandLineArgs &cmdline,
		     const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
		     const typename A2Apolicies::FermionFieldType::InputParamType &field4dparams){
  params.meas_arg.TrajCur = conf;

  std::string dir(params.meas_arg.WorkDirectory);

  //-------------------- Read gauge field --------------------//
  readGaugeRNG(params,cmdline);
  printMem("Memory after gauge and RNG read");

  runInitialGridBenchmarks(cmdline,params);

  typedef std::unique_ptr<EvecManager<typename A2Apolicies::GridFermionField,typename A2Apolicies::GridFermionFieldF> > LanczosPtrType;
  LanczosPtrType eig = A2ALanczosFactory<A2Apolicies>(params.jp.lanczos_controls);
  LanczosPtrType eig_s = A2ALanczosFactory<A2Apolicies>(params.jp.lanczos_controls);

  if(cmdline.tune_lanczos_light) computeEvecs(*eig, Light, params, cmdline.evec_opts_l);
  if(cmdline.tune_lanczos_heavy) computeEvecs(*eig_s, Heavy, params, cmdline.evec_opts_h);
  if(cmdline.tune_lanczos_light||cmdline.tune_lanczos_heavy) return; //tune and exit
  if(cmdline.tune_gauge_fix){
    Lattice* lat = (Lattice*)createFgridLattice<LatticeType>(params.jp);
    doGaugeFix(*lat, false, params.fix_gauge_arg);
    delete lat;
    return;
  }

  //-------------------- Light quark Lanczos ---------------------//
  if(!cmdline.randomize_vw || cmdline.force_evec_compute) computeEvecs(*eig, Light, params, cmdline.evec_opts_l);

  //-------------------- Light quark v and w --------------------//
  A2AvectorV<A2Apolicies> V(params.a2a_arg, field4dparams);
  A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
  computeVW(V, W, Light, params, *eig, cmdline.randomize_vw);

  LOGA2A << "Freeing light evecs" << std::endl;
  printMem("Memory before light evec free");
  eig->freeEvecs();
  printMem("Memory after light evec free");
    
  //-------------------- Strange quark Lanczos ---------------------//
  if(!cmdline.randomize_vw || cmdline.force_evec_compute) computeEvecs(*eig_s, Heavy, params, cmdline.evec_opts_h);

  //-------------------- Strange quark v and w --------------------//
  A2AvectorV<A2Apolicies> V_s(params.a2a_arg_s,field4dparams);
  A2AvectorW<A2Apolicies> W_s(params.a2a_arg_s,field4dparams);
  computeVW(V_s, W_s, Heavy, params, *eig_s, cmdline.randomize_vw);

  printMem("Memory before heavy evec free");
  eig_s->freeEvecs();
  printMem("Memory after heavy evec free");

  //The rest of the code passes the pointer to the lattice around rather than recreating on-the-fly
  Lattice* lat = (Lattice*)createFgridLattice<LatticeType>(params.jp);

  //-------------------Fix gauge----------------------------
  //This may be done with Grid so we will do it before we free up Grid's temporary memory
  doGaugeFix(*lat, cmdline.skip_gauge_fix, params.fix_gauge_arg);

  freeGridSharedMem();
  GridMemoryManagerFree();

  doContractions(conf,params,cmdline,*lat,V,W,V_s,W_s,field3dparams);
  delete lat;
}

void doConfigurationSplit(const int conf, Parameters &params, const CommandLineArgs &cmdline,
			  const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
			  const typename A2Apolicies::FermionFieldType::InputParamType &field4dparams){
  checkWriteable(cmdline.checkpoint_dir,conf);
  params.meas_arg.TrajCur = conf;

  std::string dir(params.meas_arg.WorkDirectory);

  //-------------------- Read gauge field --------------------//
  readGaugeRNG(params,cmdline);
    
  printMem("Memory after gauge and RNG read");

  runInitialGridBenchmarks(cmdline,params);
  
  if(cmdline.split_job_part == 0){
    //Do light Lanczos, strange Lanczos and strange CG, store results
    
    //-------------------- Light quark Lanczos ---------------------//
    {
      GridLanczosDoubleConvSingle<A2Apolicies> eig;
      computeEvecs(eig, Light, params, cmdline.evec_opts_l);
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.lanczos_l.cfg" << conf;
      if(!UniqueID()){ printf("Writing light Lanczos to %s\n",os.str().c_str()); fflush(stdout); }
      double time = -dclock();
      if(params.lanc_arg.N_true_get > 0) eig.writeParallel(os.str());
      time+=dclock();
      print_time("main","Light Lanczos write",time);
    }

    {//Do the light A2A vector random fields to ensure same ordering as unsplit job
      A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
#ifdef USE_DESTRUCTIVE_FFT
      W.allocModes();
#endif
      A2AhighModeSourceOriginal<A2Apolicies> Wsrc_impl;
      Wsrc_impl.setHighModeSources(W);
    }
    
    //-------------------- Strange quark Lanczos ---------------------//
    GridLanczosDoubleConvSingle<A2Apolicies> eig_s;
    computeEvecs(eig_s, Heavy, params, cmdline.evec_opts_h);

    //-------------------- Strange quark v and w --------------------//
    A2AvectorV<A2Apolicies> V_s(params.a2a_arg_s,field4dparams);
    A2AvectorW<A2Apolicies> W_s(params.a2a_arg_s,field4dparams);

    computeVW(V_s, W_s, Heavy, params, eig_s, cmdline.randomize_vw);
    
    {
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.V_s.cfg" << conf;
      if(!UniqueID()){ printf("Writing V_s to %s\n",os.str().c_str()); fflush(stdout); }
      double time = -dclock();
      V_s.writeParallel(os.str());
      time+=dclock();
      print_time("main","V_s write",time);
    }
    {
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.W_s.cfg" << conf;
      if(!UniqueID()){ printf("Writing W_s to %s\n",os.str().c_str()); fflush(stdout); }
      double time = -dclock();
      W_s.writeParallel(os.str());
      time+=dclock();
      print_time("main","W_s write",time);
    }
    
  }else if(cmdline.split_job_part == 1){
    //Do light CG and contractions

    GridLanczosDoubleConvSingle<A2Apolicies> eig;
    {
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.lanczos_l.cfg" << conf;
      if(!UniqueID()) printf("Reading light Lanczos from %s\n",os.str().c_str());
      double time = -dclock();
      eig.readParallel(os.str());      
      time+=dclock();
      print_time("main","Light Lanczos read",time);
    }

    //-------------------- Light quark v and w --------------------//
    A2AvectorV<A2Apolicies> V(params.a2a_arg, field4dparams);
    A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
    computeVW(V, W, Light, params, eig, cmdline.randomize_vw);
    
    eig.freeEvecs();
    printMem("Memory after light evec free");    
    
    //-------------------- Strange quark v and w read --------------------//
    A2AvectorV<A2Apolicies> V_s(params.a2a_arg_s,field4dparams);
    A2AvectorW<A2Apolicies> W_s(params.a2a_arg_s,field4dparams);
#ifdef USE_DESTRUCTIVE_FFT
    V_s.allocModes(); W_s.allocModes();
#endif
    {
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.V_s.cfg" << conf;
      if(!UniqueID()) printf("Reading V_s from %s\n",os.str().c_str());
      double time = -dclock();
      V_s.readParallel(os.str());
      time+=dclock();
      print_time("main","V_s read",time);
    }
    {
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.W_s.cfg" << conf;
      if(!UniqueID()) printf("Reading W_s from %s\n",os.str().c_str());
      double time = -dclock();
      W_s.readParallel(os.str());
      time+=dclock();
      print_time("main","W_s read",time);
    }

    typedef typename A2Apolicies::FgridGFclass LatticeType;
    Lattice* lat = (Lattice*)createFgridLattice<LatticeType>(params.jp);
    freeGridSharedMem();
    doContractions(conf,params,cmdline,*lat,V,W,V_s,W_s,field3dparams);
    delete lat;
  }else{ //part 1
    ERR.General("","doConfigurationSplit","Invalid part index %d\n", cmdline.split_job_part);
  }

    
}


//Just compute the light quark propagators, save to disk then exit
void doConfigurationLLprops(const int conf, Parameters &params, const CommandLineArgs &cmdline,
		     const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
		     const typename A2Apolicies::FermionFieldType::InputParamType &field4dparams){

  params.meas_arg.TrajCur = conf;

  std::string dir(params.meas_arg.WorkDirectory);

  //-------------------- Read gauge field --------------------//
  readGaugeRNG(params,cmdline);
    
  printMem("Memory after gauge and RNG read");
  
  //-------------------- Light quark Lanczos ---------------------//
  GridLanczosDoubleConvSingle<A2Apolicies> eig;
  if(!cmdline.randomize_vw || cmdline.force_evec_compute || cmdline.tune_lanczos_light) computeEvecs(eig, Light, params, cmdline.evec_opts_l);
  if(cmdline.tune_lanczos_light) return;
  
  //-------------------- Light quark v and w --------------------//
  A2AvectorV<A2Apolicies> V(params.a2a_arg, field4dparams);
  A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
  computeVW(V, W, Light, params, eig, cmdline.randomize_vw);
  
  size_t nodes = GJP.Xnodes()*GJP.Ynodes()*GJP.Znodes()*GJP.Tnodes();
  {
    double sz = A2AvectorV<A2Apolicies>::Mbyte_size(params.a2a_arg, field4dparams) * nodes;
    std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.V.cfg" << conf;
    if(!UniqueID()){ printf("Writing V of size %g MB to %s\n",sz,os.str().c_str()); fflush(stdout); }
    double time = -dclock();
    V.writeParallel(os.str());
    time+=dclock();
    print_time("main","V write",time);
  }
  {
    double sz = A2AvectorW<A2Apolicies>::Mbyte_size(params.a2a_arg, field4dparams) * nodes;
    std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.W.cfg" << conf;
    if(!UniqueID()){ printf("Writing W of size %g MB to %s\n",sz,os.str().c_str()); fflush(stdout); }
    double time = -dclock();
    W.writeParallel(os.str());
    time+=dclock();
    print_time("main","W write",time);
  }


}




void doConfigurationLLpropsSplit(const int conf, Parameters &params, const CommandLineArgs &cmdline,
		     const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
		     const typename A2Apolicies::FermionFieldType::InputParamType &field4dparams){

  params.meas_arg.TrajCur = conf;

  std::string dir(params.meas_arg.WorkDirectory);

  //-------------------- Read gauge field --------------------//
  readGaugeRNG(params,cmdline);
    
  printMem("Memory after gauge and RNG read");

  GridLanczosDoubleConvSingle<A2Apolicies> eig;
  if(cmdline.tune_lanczos_light){ computeEvecs(eig, Light, params, cmdline.evec_opts_l); return; }

  if(cmdline.split_job_part == 0){
    //-------------------- Light quark Lanczos ---------------------//
    
    if(!cmdline.randomize_vw || cmdline.force_evec_compute) computeEvecs(eig, Light, params, cmdline.evec_opts_l);

    //Write to disk
    {
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.lanczos_l.cfg" << conf;
      if(!UniqueID()){ printf("Writing light Lanczos to %s\n",os.str().c_str()); fflush(stdout); }
      double time = -dclock();
      if(params.lanc_arg.N_true_get > 0) eig.writeParallel(os.str());
      time+=dclock();
      print_time("main","Light Lanczos write",time);
    }
  }else if(cmdline.split_job_part == 1){
    {
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.lanczos_l.cfg" << conf;
      if(!UniqueID()) printf("Reading light Lanczos from %s\n",os.str().c_str());
      double time = -dclock();
      eig.readParallel(os.str());      
      time+=dclock();
      print_time("main","Light Lanczos read",time);
    }

    //-------------------- Light quark v and w --------------------//
    A2AvectorV<A2Apolicies> V(params.a2a_arg, field4dparams);
    A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
    computeVW(V, W, Light, params, eig, cmdline.randomize_vw);

    size_t nodes = GJP.Xnodes()*GJP.Ynodes()*GJP.Znodes()*GJP.Tnodes();
    {
      double sz = A2AvectorV<A2Apolicies>::Mbyte_size(params.a2a_arg, field4dparams) * nodes;
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.V.cfg" << conf;
      if(!UniqueID()){ printf("Writing V of size %g MB to %s\n",sz,os.str().c_str()); fflush(stdout); }
      double time = -dclock();
      V.writeParallel(os.str());
      time+=dclock();
      print_time("main","V write",time);
    }
    {
      double sz = A2AvectorW<A2Apolicies>::Mbyte_size(params.a2a_arg, field4dparams) * nodes;
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.W.cfg" << conf;
      if(!UniqueID()){ printf("Writing W of size %g MB to %s\n",sz,os.str().c_str()); fflush(stdout); }
      double time = -dclock();
      W.writeParallel(os.str());
      time+=dclock();
      print_time("main","W write",time);
    }
  }else{ //part 1
    ERR.General("","doConfigurationLLpropsSplit","Invalid part index %d\n", cmdline.split_job_part);
  }
}






#endif
