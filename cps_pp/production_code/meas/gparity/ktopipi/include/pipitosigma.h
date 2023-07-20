#ifndef _KTOPIPI_MAIN_PIPI_TO_SIGMA_GPARITY_H_
#define _KTOPIPI_MAIN_PIPI_TO_SIGMA_GPARITY_H_

//Compute pipi->sigma with file in Tianle's format
template<typename PionMomentumPolicy, typename SigmaMomentumPolicy>
void computePiPiToSigma(const std::vector< fVector<typename A2Apolicies::ScalarComplexType> > &sigma_bub,
			MesonFieldMomentumPairContainer<A2Apolicies> &mf_sigma_con, const SigmaMomentumPolicy &sigma_mom, 
			MesonFieldMomentumContainer<A2Apolicies> &mf_pion_con, const PionMomentumPolicy &pion_mom,
			const int conf, const Parameters &params){
  const int nmom_sigma = sigma_mom.nMom();
  const int nmom_pi = pion_mom.nMom();
  
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();

  LOGA2A << "Computing Pipi->sigma" << std::endl;

  std::vector<fVector<typename A2Apolicies::ScalarComplexType> > pipi_bub(nmom_pi);
  for(int pidx=0;pidx<nmom_pi;pidx++){
    ComputePiPiGparity<A2Apolicies>::computeFigureVdis(pipi_bub[pidx], pion_mom.getMesonMomentum(pidx), params.jp.pipi_separation, mf_pion_con);
  }
  
  //All momentum combinations have total momentum 0 at source and sink
  double time = -dclock();
  for(int ppi1_idx=0;ppi1_idx<nmom_pi;ppi1_idx++){
    for(int psigma_idx=0;psigma_idx<nmom_sigma;psigma_idx++){
      a2a_printf("Pipi->sigma connected psigma_idx=%d ppi1_idx=%d\n",psigma_idx,ppi1_idx);
      fMatrix<typename A2Apolicies::ScalarComplexType> into(Lt,Lt);

      ComputePiPiToSigmaContractions<A2Apolicies>::computeConnected(into,mf_sigma_con,sigma_mom,psigma_idx,
								    mf_pion_con,pion_mom,ppi1_idx,
								    params.jp.pipi_separation, params.jp.tstep_pipi); //reuse same tstep currently

      //Tianle also computes the disconnected part
      fMatrix<typename A2Apolicies::ScalarComplexType> disconn(Lt,Lt);
      ComputePiPiToSigmaContractions<A2Apolicies>::computeDisconnectedDiagram(disconn, sigma_bub[psigma_idx], pipi_bub[ppi1_idx], params.jp.tstep_pipi);

      into += disconn;
      
      std::ostringstream os;
      os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_pipitosigma_sigmawdagmom";
      os << sigma_mom.getWmom(psigma_idx).file_str() << "_pionmom" << (-pion_mom.getMesonMomentum(ppi1_idx)).file_str() << "_v2"; //Note this is Daiqian's phase convention (which is - the standard one)
      
      LOGA2A << "Pipi->sigma writing to file " << os.str() << std::endl;

      into.write(os.str());
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      into.write(os.str(),true);
#endif
    }
  }
  time += dclock();
  a2a_print_time("main","Pipi->sigma function",time);

  printMem("Memory after Pipi->sigma function computation");
}


#endif
