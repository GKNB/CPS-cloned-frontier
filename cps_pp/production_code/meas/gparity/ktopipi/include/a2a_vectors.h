#ifndef _KTOPIPI_MAIN_A2A_VECTORS_H_
#define _KTOPIPI_MAIN_A2A_VECTORS_H_



//Tune the Lanczos and exit
void LanczosTune(bool tune_lanczos_light, bool tune_lanczos_heavy, const Parameters &params, BFMGridSolverWrapper &solvers){
  if(tune_lanczos_light){
    BFMGridLanczosWrapper<A2Apolicies> eig(solvers, params.jp);
    if(!UniqueID()) printf("Tuning lanczos light with mass %f\n", params.lanc_arg.mass);

    double time = -dclock();
    eig.compute(params.lanc_arg);
    time += dclock();
    print_time("main","Lanczos light",time);
  }
  if(tune_lanczos_heavy){
    BFMGridLanczosWrapper<A2Apolicies> eig(solvers, params.jp);
    if(!UniqueID()) printf("Tuning lanczos heavy with mass %f\n", params.lanc_arg_s.mass);

    double time = -dclock();
    eig.compute(params.lanc_arg_s);
    time += dclock();
    print_time("main","Lanczos heavy",time);
  }
  exit(0);
}


enum LightHeavy { Light, Heavy };

void computeEvecs(BFMGridLanczosWrapper<A2Apolicies> &eig, const LightHeavy lh, const Parameters &params, const bool randomize_evecs){
  const char* name = (lh ==  Light ? "light" : "heavy");
  const LancArg &lanc_arg = (lh == Light ? params.lanc_arg : params.lanc_arg_s);
  return computeEvecs(eig, lanc_arg, params.jp, name, randomize_evecs);
}

void computeVW(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, const LightHeavy lh, const Parameters &params,
	       const BFMGridLanczosWrapper<A2Apolicies> &eig, const BFMGridA2ALatticeWrapper<A2Apolicies> &a2a_lat,
	       const bool randomize_vw){
  const A2AArg &a2a_arg = lh == Light ? params.a2a_arg : params.a2a_arg_s;  
  const char* name = (lh ==  Light ? "light" : "heavy");

  //If using destructive FFTs we have not yet allocated V!
  /* typedef typename A2Apolicies::FermionFieldType::InputParamType Field4DparamType; */
  /* Field4DparamType field4dparams = V.getFieldInputParams(); */
  
  /* if(!UniqueID()){ printf("V vector requires %f MB, W vector %f MB of memory\n",  */
  /* 			  A2AvectorV<A2Apolicies>::Mbyte_size(a2a_arg,field4dparams), A2AvectorW<A2Apolicies>::Mbyte_size(a2a_arg,field4dparams) ); */
  /*   fflush(stdout); */
  /* } */
  
  if(!UniqueID()) printf("Computing %s quark A2A vectors\n",name);
  double time = -dclock();

  a2a_lat.computeVW(V,W,eig,params.jp.cg_controls,randomize_vw);
  
  printMem(stringize("Memory after %s A2A vector computation", name));

  time += dclock();
  std::ostringstream os; os << name << " quark A2A vectors";
  print_time("main",os.str().c_str(),time);
}

#endif
