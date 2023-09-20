#ifndef _KTOPIPI_MAIN_A2A_VECTORS_H_
#define _KTOPIPI_MAIN_A2A_VECTORS_H_

enum LightHeavy { Light, Heavy };

typedef EvecManager<typename A2Apolicies::GridFermionField,typename A2Apolicies::GridFermionFieldF> EvecManagerType;

//Compute the eigenvectors and convert to single precision
void computeEvecs(EvecManagerType &eig, const LancArg &lanc_arg, const JobParams &jp, const char* name, const computeEvecsOpts &opts = computeEvecsOpts()){
  LOGA2A << "Running " << name << " quark Lanczos" << std::endl;
  double time = -dclock();

  auto lanczos_lat = createFgridLattice<typename A2Apolicies::FgridGFclass>(jp);
  A2Apreconditioning precond = SchurOriginal;
  if(lanc_arg.precon && jp.cg_controls.CGalgorithm == AlgorithmMixedPrecisionMADWF) precond = jp.cg_controls.madwf_params.precond; //SchurDiagTwo often used for ZMADWF

  //TEST
  testXconjAction<A2Apolicies>(*lanczos_lat);
  //TEST

  if(opts.randomize_evecs) eig.randomizeEvecs(lanc_arg, *lanczos_lat);
  else if(!opts.load_evecs){
    eig.compute(lanc_arg, *lanczos_lat, precond);
    if(opts.save_evecs){
      LOGA2A << "Writing " << name << " eigenvectors" << std::endl;
      eig.writeParallel(opts.save_evecs_stub);
    }
  }else{
    LOGA2A << "Reading " << name << " eigenvectors" << std::endl;
    eig.readParallel(opts.load_evecs_stub);
  }

  delete lanczos_lat;
  time += dclock();

  std::ostringstream os; os << name << " quark Lanczos";
      
  a2a_print_time("main",os.str().c_str(),time);

  LOGA2A << "Memory after " << name << " quark Lanczos:" << std::endl;
  printMem();
}

void computeEvecs(EvecManagerType &eig, const LightHeavy lh, const Parameters &params, const computeEvecsOpts &opts = computeEvecsOpts()){
  const char* name = (lh ==  Light ? "light" : "heavy");
  const LancArg &lanc_arg = (lh == Light ? params.lanc_arg : params.lanc_arg_s);
  return computeEvecs(eig, lanc_arg, params.jp, name, opts);
}

template<typename Vtype, typename Wtype>
void computeVW(Vtype &V, Wtype &W, const EvecManagerType &eig, double mass, const CGcontrols &cg_controls, 
	       typename A2Apolicies::FgridGFclass *lat, const computeVWopts &opts = computeVWopts()){
#ifdef USE_DESTRUCTIVE_FFT
  LOGA2A << "Allocating V,W vectors" << std::endl;  
  V.allocModes(); W.allocModes();
  LOGA2A << "Allocated V (" << V.Mbyte_size(V.getArgs(),V.getFieldInputParams()) << " MB), and W (" << W.Mbyte_size(W.getArgs(),W.getFieldInputParams()) << " MB) vectors " << std::endl;  
  LOGA2A << "Initializing V,W" << std::endl;
  V.zero(); W.zero(); //force the memory to be assigned right now (TESTING)
#endif
  if(opts.skip_vw){
    LOGA2A << "Skipping generation of V,W vectors" << std::endl;
    //Should still set the W sources to ensure the random numbers are consistent even if V,W is skipped
    std::unique_ptr<A2AhighModeSource<A2Apolicies> > Wsrc_impl(highModeSourceFactory<A2Apolicies>(cg_controls.highmode_source));
    Wsrc_impl->setHighModeSources(W);
    W.free_mem(); V.free_mem();
  }else if(opts.randomize_vw){
    LOGA2A << "Creating random VW vectors" << std::endl;
    randomizeVW(V,W);
  }else if(opts.load_vw){
    LOGA2A << "Loading V,W vectors" << std::endl;
    W.readParallelWithGrid(opts.load_vw_stub + "_w");
    V.readParallelWithGrid(opts.load_vw_stub + "_v");
  }else{
    LOGA2A << "Creating interface and running VW calculation" << std::endl;
    auto ei = eig.createInterface();
    cps::computeVW(V,W,*lat,*ei,mass,cg_controls);
  }

  if(opts.save_vw){
    LOGA2A << "Saving V,W vectors" << std::endl;
    W.writeParallelWithGrid(opts.save_vw_stub + "_w");
    V.writeParallelWithGrid(opts.save_vw_stub + "_v");
  }
}
template<typename Vtype, typename Wtype>
void computeVW(Vtype &V, Wtype &W, const LightHeavy lh, const Parameters &params, const EvecManagerType &eig, const computeVWopts &opts = computeVWopts()){
  auto lat = createFgridLattice<typename A2Apolicies::FgridGFclass>(params.jp);

  const LancArg &lanc_arg = (lh == Light ? params.lanc_arg : params.lanc_arg_s);
  const char* name = (lh ==  Light ? "light" : "heavy");
 
  LOGA2A << "Computing " << name << " quark A2A vectors" << std::endl;
  double time = -dclock();

  computeVW(V,W,eig,lanc_arg.mass, params.jp.cg_controls,lat,opts);
  
  printMem(stringize("Memory after %s A2A vector computation", name));

  delete lat;
  time += dclock();
  std::ostringstream os; os << name << " quark A2A vectors";
  a2a_print_time("main",os.str().c_str(),time);
}

#endif
