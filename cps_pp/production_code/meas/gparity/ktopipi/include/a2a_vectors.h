#ifndef _KTOPIPI_MAIN_A2A_VECTORS_H_
#define _KTOPIPI_MAIN_A2A_VECTORS_H_

enum LightHeavy { Light, Heavy };

typedef EvecManager<typename A2Apolicies::GridFermionField,typename A2Apolicies::GridFermionFieldF> EvecManagerType;

//Compute the eigenvectors and convert to single precision
void computeEvecs(EvecManagerType &eig, const LancArg &lanc_arg, const JobParams &jp, const char* name, const bool randomize_evecs){
  if(!UniqueID()) printf("Running %s quark Lanczos\n",name);
  double time = -dclock();

  auto lanczos_lat = createFgridLattice<typename A2Apolicies::FgridGFclass>(jp);
  A2Apreconditioning precond = SchurOriginal;
  if(lanc_arg.precon && jp.cg_controls.CGalgorithm == AlgorithmMixedPrecisionMADWF) precond = jp.cg_controls.madwf_params.precond; //SchurDiagTwo often used for ZMADWF

  //TEST
  testXconjAction<A2Apolicies>(*lanczos_lat);
  //TEST

  if(randomize_evecs) eig.randomizeEvecs(lanc_arg, *lanczos_lat);
  else eig.compute(lanc_arg, *lanczos_lat, precond);

  delete lanczos_lat;
  time += dclock();

  std::ostringstream os; os << name << " quark Lanczos";
      
  print_time("main",os.str().c_str(),time);

  if(!UniqueID()) printf("Memory after %s quark Lanczos:\n",name);
  printMem();
}

void computeEvecs(EvecManagerType &eig, const LightHeavy lh, const Parameters &params, const bool randomize_evecs){
  const char* name = (lh ==  Light ? "light" : "heavy");
  const LancArg &lanc_arg = (lh == Light ? params.lanc_arg : params.lanc_arg_s);
  return computeEvecs(eig, lanc_arg, params.jp, name, randomize_evecs);
}

void computeVW(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, const EvecManagerType &eig, double mass, const CGcontrols &cg_controls, 
	       typename A2Apolicies::FgridGFclass *lat, bool randomize_vw){
#ifdef USE_DESTRUCTIVE_FFT
  V.allocModes(); W.allocModes();
#endif
  if(!randomize_vw){
    auto ei = eig.createInterface();
    cps::computeVW(V,W,*lat,*ei,mass,cg_controls);
  }else randomizeVW<A2Apolicies>(V,W);
}

void computeVW(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, const LightHeavy lh, const Parameters &params, const EvecManagerType &eig, const bool randomize_vw){
  auto lat = createFgridLattice<typename A2Apolicies::FgridGFclass>(params.jp);

  const LancArg &lanc_arg = (lh == Light ? params.lanc_arg : params.lanc_arg_s);
  const char* name = (lh ==  Light ? "light" : "heavy");
 
  if(!UniqueID()) printf("Computing %s quark A2A vectors\n",name);
  double time = -dclock();

  computeVW(V,W,eig,lanc_arg.mass, params.jp.cg_controls,lat,randomize_vw);
  
  printMem(stringize("Memory after %s A2A vector computation", name));

  delete lat;
  time += dclock();
  std::ostringstream os; os << name << " quark A2A vectors";
  print_time("main",os.str().c_str(),time);
}

#endif
