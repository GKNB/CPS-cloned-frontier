#ifndef _MAIN_H
#define _MAIN_H

//Header for main program
using namespace cps;

#include "a2a_policies.h"
#include "cmdline.h"
#include "args.h"
#include "misc.h"
#include "a2a_vectors.h"
#include "kaon.h"
#include "pion.h"
#include "sigma.h"
#include "pipi.h"
#include "ktopipi.h"
#include "do_contractions.h"

void doConfiguration(const int conf, Parameters &params, const CommandLineArgs &cmdline,
		     const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
		     const typename A2Apolicies::FermionFieldType::InputParamType &field4dparams, BFMGridSolverWrapper &solvers){

  params.meas_arg.TrajCur = conf;

  std::string dir(params.meas_arg.WorkDirectory);

  //-------------------- Read gauge field --------------------//
  readGaugeRNG(params,cmdline);
    
  printMem("Memory after gauge and RNG read");

  runInitialGridBenchmarks(cmdline,params);
  
  if(cmdline.tune_lanczos_light || cmdline.tune_lanczos_heavy) LanczosTune(cmdline.tune_lanczos_light, cmdline.tune_lanczos_heavy, params, solvers);    

  //-------------------- Light quark Lanczos ---------------------//
  BFMGridLanczosWrapper<A2Apolicies> eig(solvers, params.jp);
  if(!cmdline.randomize_vw || cmdline.force_evec_compute) computeEvecs(eig, Light, params, cmdline.randomize_evecs);

  //-------------------- Light quark v and w --------------------//
  A2AvectorV<A2Apolicies> V(params.a2a_arg, field4dparams);
  A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
  {
    BFMGridA2ALatticeWrapper<A2Apolicies> latwrp(solvers, params.jp); //lattice created temporarily
    computeVW(V, W, Light, params, eig, latwrp, cmdline.randomize_vw);
  }
  if(!UniqueID()){ printf("Freeing light evecs\n"); fflush(stdout); }
  eig.freeEvecs();
  printMem("Memory after light evec free");
    
  //-------------------- Strange quark Lanczos ---------------------//
  BFMGridLanczosWrapper<A2Apolicies> eig_s(solvers, params.jp);
  if(!cmdline.randomize_vw || cmdline.force_evec_compute) computeEvecs(eig_s, Heavy, params, cmdline.randomize_evecs);

  //-------------------- Strange quark v and w --------------------//
  A2AvectorV<A2Apolicies> V_s(params.a2a_arg_s,field4dparams);
  A2AvectorW<A2Apolicies> W_s(params.a2a_arg_s,field4dparams);
  BFMGridA2ALatticeWrapper<A2Apolicies> latwrp(solvers, params.jp);
  computeVW(V_s, W_s, Heavy, params, eig_s, latwrp, cmdline.randomize_vw);

  eig_s.freeEvecs();
  printMem("Memory after heavy evec free");

  //From now one we just need a generic lattice instance, so use a2a_lat
  Lattice& lat = (Lattice&)(*latwrp.a2a_lat);
  
  freeGridSharedMem();
  doContractions(conf,params,cmdline,lat,V,W,V_s,W_s,field3dparams);
}

void doConfigurationSplit(const int conf, Parameters &params, const CommandLineArgs &cmdline,
			  const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
			  const typename A2Apolicies::FermionFieldType::InputParamType &field4dparams, BFMGridSolverWrapper &solvers){
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
      BFMGridLanczosWrapper<A2Apolicies> eig(solvers, params.jp);
      computeEvecs(eig, Light, params, cmdline.randomize_evecs);
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
      W.setWhRandom();
    }
    
    //-------------------- Strange quark Lanczos ---------------------//
    BFMGridLanczosWrapper<A2Apolicies> eig_s(solvers, params.jp);
    computeEvecs(eig_s, Heavy, params, cmdline.randomize_evecs);

    //-------------------- Strange quark v and w --------------------//
    A2AvectorV<A2Apolicies> V_s(params.a2a_arg_s,field4dparams);
    A2AvectorW<A2Apolicies> W_s(params.a2a_arg_s,field4dparams);

    BFMGridA2ALatticeWrapper<A2Apolicies> latwrp(solvers, params.jp);
    computeVW(V_s, W_s, Heavy, params, eig_s, latwrp, cmdline.randomize_vw);
    
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

    BFMGridLanczosWrapper<A2Apolicies> eig(solvers, params.jp);
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
    BFMGridA2ALatticeWrapper<A2Apolicies> latwrp(solvers, params.jp);
    computeVW(V, W, Light, params, eig, latwrp, cmdline.randomize_vw);
    
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

    //From now one we just need a generic lattice instance, so use a2a_lat
    Lattice& lat = (Lattice&)(*latwrp.a2a_lat);

    freeGridSharedMem();
    doContractions(conf,params,cmdline,lat,V,W,V_s,W_s,field3dparams);
  }else{ //part 1
    ERR.General("","doConfigurationSplit","Invalid part index %d\n", cmdline.split_job_part);
  }

    
}







#endif
