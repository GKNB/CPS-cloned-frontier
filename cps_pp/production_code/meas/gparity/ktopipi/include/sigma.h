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

template<typename SigmaMomentumPolicy>
void randomizeSigmaMesonFields(MesonFieldMomentumPairContainer<A2Apolicies> &mf_sigma,
			       typename computeMesonFieldsBase<A2Apolicies>::Vtype &V, typename computeMesonFieldsBase<A2Apolicies>::Wtype &W,
			       const SigmaMomentumPolicy &sigma_mom){
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf(Lt);
  std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > *ins;
  for(int t=0;t<Lt;t++) mf[t].setup(W,V,t,t);

  for(int p=0;p<sigma_mom.nMom();p++){
    for(int t=0;t<Lt;t++) mf[t].testRandom();
    ins = &mf_sigma.copyAdd(sigma_mom.getWdagMom(p), sigma_mom.getVmom(p),mf);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(1,ins);
#endif 
  }
}


template<typename SigmaMomentumPolicy>
void computeSigmaMesonFieldsExt(MesonFieldMomentumPairContainer<A2Apolicies> &mf_sigma,
				typename ComputeSigma<A2Apolicies>::Vtype &V, typename ComputeSigma<A2Apolicies>::Wtype &W, const SigmaMomentumPolicy &sigma_mom,
				const int conf, Lattice &lat, const Parameters &params, const CommandLineArgs &cmdline, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  if(cmdline.randomize_mf){
    randomizeSigmaMesonFields(mf_sigma, V, W, sigma_mom);
  }else{ 
    if(cmdline.ktosigma_load_sigma_mf){
      if(!UniqueID()) printf("Reading sigma meson fields from disk\n");
      double time = dclock();
      computeSigmaMesonFields1s<A2Apolicies, StationarySigmaMomentaPolicy>::read(mf_sigma, sigma_mom, cmdline.ktosigma_sigma_mf_dir, conf, params.jp.pion_rad);
      print_time("main","Sigma meson field read", dclock()-time);
    }else{
      if(!UniqueID()) printf("Computing sigma meson fields\n");
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
      if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
#endif
      double time = dclock();      
      computeSigmaMesonFields1s<A2Apolicies, StationarySigmaMomentaPolicy>::Options opt;
#ifdef ARCH_BGQ
      opt.nshift_combine_max = 2;
      opt.thr_internal = 32;
#endif
      computeSigmaMesonFields1s<A2Apolicies, StationarySigmaMomentaPolicy>::computeMesonFields(mf_sigma, sigma_mom, W, V, params.jp.pion_rad, lat, field3dparams, opt);
      print_time("main","Sigma meson field compute", dclock()-time);
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
      if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
#endif
    }
  }

  if(cmdline.ktosigma_save_sigma_mf){
    if(!UniqueID()) printf("Writing sigma meson fields to disk\n");
    double time = dclock();
    computeSigmaMesonFields1s<A2Apolicies, StationarySigmaMomentaPolicy>::write(mf_sigma, sigma_mom, params.meas_arg.WorkDirectory, conf, params.jp.pion_rad);
    print_time("main","Sigma meson field write", dclock()-time);
  }
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
      ComputeSigmaContractions<A2Apolicies>::computeDisconnectedDiagram(disconn, sigma_bub[psrcidx], sigma_bub[psnkidx]);

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
