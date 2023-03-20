#ifndef _KTOPIPI_MAIN_A2A_VECTORS_H_
#define _KTOPIPI_MAIN_A2A_VECTORS_H_



//Tune the Lanczos and exit
void LanczosTune(bool tune_lanczos_light, bool tune_lanczos_heavy, const Parameters &params){
  if(tune_lanczos_light){
    GridLanczosWrapper<A2Apolicies> eig;
    if(!UniqueID()) printf("Tuning lanczos light with mass %f\n", params.lanc_arg.mass);

    double time = -dclock();
    eig.compute(params.jp,params.lanc_arg);
    time += dclock();
    print_time("main","Lanczos light",time);
  }
  if(tune_lanczos_heavy){
    GridLanczosWrapper<A2Apolicies> eig;      
    if(!UniqueID()) printf("Tuning lanczos heavy with mass %f\n", params.lanc_arg_s.mass);

    double time = -dclock();
    eig.compute(params.jp,params.lanc_arg_s);
    time += dclock();
    print_time("main","Lanczos heavy",time);
  }
  exit(0);
}


enum LightHeavy { Light, Heavy };

//Compute the eigenvectors and convert to single precision
void computeEvecs(GridLanczosWrapper<A2Apolicies> &eig, const LancArg &lanc_arg, const JobParams &jp, const char* name, const bool randomize_evecs){
  if(!UniqueID()) printf("Running %s quark Lanczos\n",name);
  double time = -dclock();
  if(randomize_evecs) eig.randomizeEvecs(jp, lanc_arg);
  else eig.compute(jp, lanc_arg);
  time += dclock();

  std::ostringstream os; os << name << " quark Lanczos";
      
  print_time("main",os.str().c_str(),time);

  if(!UniqueID()) printf("Memory after %s quark Lanczos:\n",name);
  printMem();

#ifndef A2A_LANCZOS_SINGLE
  if(jp.convert_evecs_to_single_precision){
    eig.toSingle();
    if(!UniqueID()) printf("Memory after single-prec conversion of %s quark evecs:\n",name);
    printMem();
  }
#endif
}

void computeEvecs(GridLanczosWrapper<A2Apolicies> &eig, const LightHeavy lh, const Parameters &params, const bool randomize_evecs){
  const char* name = (lh ==  Light ? "light" : "heavy");
  const LancArg &lanc_arg = (lh == Light ? params.lanc_arg : params.lanc_arg_s);
  return computeEvecs(eig, lanc_arg, params.jp, name, randomize_evecs);
}

void computeVW(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, const GridLanczosWrapper<A2Apolicies> &eig, const CGcontrols &cg_controls, 
	       typename A2Apolicies::FgridGFclass *lat, bool randomize_vw){
#ifdef USE_DESTRUCTIVE_FFT
  V.allocModes(); W.allocModes();
#endif
  if(!randomize_vw){
    if(eig.singleprec_evecs){
      cps::computeVW(V, W, *lat, eig.evec_f, eig.eval, eig.mass, cg_controls);
    }else{
      cps::computeVW(V, W, *lat, eig.evec, eig.eval, eig.mass, cg_controls);
    }
  }else randomizeVW<A2Apolicies>(V,W);
}

void computeVW(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, const LightHeavy lh, const Parameters &params,
	       const GridLanczosWrapper<A2Apolicies> &eig, const bool randomize_vw){
  auto lat = createFgridLattice<typename A2Apolicies::FgridGFclass>(params.jp);

  const A2AArg &a2a_arg = lh == Light ? params.a2a_arg : params.a2a_arg_s;  
  const char* name = (lh ==  Light ? "light" : "heavy");
 
  if(!UniqueID()) printf("Computing %s quark A2A vectors\n",name);
  double time = -dclock();

  computeVW(V,W,eig,params.jp.cg_controls,lat,randomize_vw);
  
  printMem(stringize("Memory after %s A2A vector computation", name));

  delete lat;
  time += dclock();
  std::ostringstream os; os << name << " quark A2A vectors";
  print_time("main",os.str().c_str(),time);
}

#endif
