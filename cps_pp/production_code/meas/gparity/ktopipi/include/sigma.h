#ifndef _KTOPIPI_MAIN_A2A_SIGMA_H_
#define _KTOPIPI_MAIN_A2A_SIGMA_H_

template<typename SigmaMomentumPolicy>
void computeSigmaMesonFields(typename ComputeSigma<A2Apolicies>::Vtype &V, typename ComputeSigma<A2Apolicies>::Wtype &W, const SigmaMomentumPolicy &sigma_mom,
			     const int conf, Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  double time = -dclock();
  if(!UniqueID()) printf("Computing sigma mesonfield computation\n");
  ComputeSigma<A2Apolicies>::computeAndWrite(params.meas_arg.WorkDirectory,conf,sigma_mom,W,V, params.jp.pion_rad, lat, field3dparams);
  time += dclock();
  print_time("main","Sigma meson fields ",time);
}

#endif
