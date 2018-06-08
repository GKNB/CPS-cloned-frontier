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

//Compute sigma 2pt function with file in Tianle's format
template<typename SigmaMomentumPolicy>
void computeSigma2pt(std::vector< fVector<typename A2Apolicies::ScalarComplexType> > &sigma_bub, //output bubble
			   MesonFieldMomentumPairContainer<A2Apolicies> &mf_sigma_con, const SigmaMomentumPolicy &sigma_mom, const int conf, const Parameters &params){
  const int nmom = sigma_mom.nMom();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();

  //All momentum combinations have total momentum 0 at source and sink
  if(!UniqueID()) printf("Computing sigma 2pt function\n");
  double time = -dclock();

  sigma_bub.resize(nmom);
  for(int pidx=0;pidx<nmom;pidx++){
    //Compute the disconnected bubble
    if(!UniqueID()) printf("Sigma disconnected bubble pidx=%d\n",pidx);
    fVector<typename A2Apolicies::ScalarComplexType> &into = sigma_bub[pidx]; into.resize(Lt);
    ComputeSigmaContractions<A2Apolicies>::computeDisconnectedBubble(into, mf_sigma_con, sigma_mom, pidx);

    std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf;
    os  << "_sigmaself_mom" << sigma_mom.getWmom(pidx).file_str(1) << "_v2"; //note Vmom == -WdagMom = Wmom for sigma as momentum 0
    into.write(os.str());
# ifdef WRITE_HEX_OUTPUT
    os << ".hexfloat";
    into.write(os.str(),true);
# endif
  }

  for(int psnkidx=0;psnkidx<nmom;psnkidx++){
    for(int psrcidx=0;psrcidx<nmom;psrcidx++){
      if(!UniqueID()) printf("Sigma connected psrcidx=%d psnkidx=%d\n",psrcidx,psnkidx);
      fMatrix<typename A2Apolicies::ScalarComplexType> into(Lt,Lt);
      ComputeSigmaContractions<A2Apolicies>::computeConnected(into, mf_sigma_con, sigma_mom, psrcidx, psnkidx);

      fMatrix<typename A2Apolicies::ScalarComplexType> disconn(Lt,Lt);
      ComputeSigmaContractions<A2Apolicies>::computeDisconnectedDiagram(disconn, sigma_bub[psnkidx], sigma_bub[psrcidx]);

      into += disconn;

      std::ostringstream os; //traj_0_sigmacorr_mompsrc_1_1_1psnk_1_1_1_v2
      os
	<< params.meas_arg.WorkDirectory << "/traj_" << conf << "_sigmacorr_mom"
	<< "psrc" << sigma_mom.getWmom(psrcidx).file_str() << "psnk" << sigma_mom.getWmom(psnkidx).file_str() << "_v2";

      into.write(os.str());
# ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      into.write(os.str(),true);
# endif
    }
  }
        
  time += dclock();
  print_time("main","Sigma 2pt function",time);

  printMem("Memory after Sigma 2pt function computation");
}


#endif
