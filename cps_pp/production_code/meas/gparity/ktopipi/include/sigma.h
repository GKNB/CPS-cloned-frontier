#ifndef _KTOPIPI_MAIN_A2A_SIGMA_H_
#define _KTOPIPI_MAIN_A2A_SIGMA_H_

template<typename Vtype, typename Wtype, typename SigmaMomentumPolicy>
void computeSigmaMesonFields(Vtype &V, Wtype &W, const SigmaMomentumPolicy &sigma_mom,
			     const int conf, Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  double time = -dclock();
  LOGA2A << "Computing sigma mesonfield computation" << std::endl;
  ComputeSigma<Vtype,Wtype>::computeAndWrite(params.meas_arg.WorkDirectory,conf,sigma_mom,W,V, params.jp.pion_rad, lat, field3dparams);
  time += dclock();
  a2a_print_time("main","Sigma meson fields ",time);
}

template<typename Vtype, typename Wtype, typename SigmaMomentumPolicy>
void randomizeSigmaMesonFields(MesonFieldMomentumPairContainer<getMesonFieldType<Wtype,Vtype> > &mf_sigma,
			       Vtype &V, Wtype &W,
			       const SigmaMomentumPolicy &sigma_mom){
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  typedef getMesonFieldType<Wtype,Vtype> MesonFieldType;
  std::vector<MesonFieldType> mf(Lt);
  std::vector<MesonFieldType> *ins;
  for(int t=0;t<Lt;t++) mf[t].setup(W,V,t,t);

  for(int p=0;p<sigma_mom.nMom();p++){
    for(int t=0;t<Lt;t++) mf[t].testRandom();
    ins = &mf_sigma.copyAdd(sigma_mom.getWdagMom(p), sigma_mom.getVmom(p),mf);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(1,ins);
#endif 
  }
}


template<typename Vtype, typename Wtype, typename SigmaMomentumPolicy>
void computeSigmaMesonFieldsExt(MesonFieldMomentumPairContainer<getMesonFieldType<Wtype,Vtype> > &mf_sigma,
				Vtype &V, Wtype &W, const SigmaMomentumPolicy &sigma_mom,
				const int conf, Lattice &lat, const Parameters &params, const CommandLineArgs &cmdline, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  if(cmdline.randomize_mf){
    randomizeSigmaMesonFields(mf_sigma, V, W, sigma_mom);
  }else{ 
    if(cmdline.ktosigma_load_sigma_mf){
      LOGA2A << "Reading sigma meson fields from disk" << std::endl;
      double time = dclock();
      computeSigmaMesonFields1s<Vtype,Wtype,StationarySigmaMomentaPolicy>::read(mf_sigma, sigma_mom, cmdline.ktosigma_sigma_mf_dir, conf, params.jp.pion_rad);
      a2a_print_time("main","Sigma meson field read", dclock()-time);
    }else{
      LOGA2A << "Computing sigma meson fields" << std::endl;
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
      if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
#endif
      double time = dclock();      
      typename computeSigmaMesonFields1s<Vtype,Wtype,StationarySigmaMomentaPolicy>::Options opt;
#ifdef ARCH_BGQ
      opt.nshift_combine_max = 2;
      opt.thr_internal = 32;
#endif
      computeSigmaMesonFields1s<Vtype,Wtype,StationarySigmaMomentaPolicy>::computeMesonFields(mf_sigma, sigma_mom, W, V, params.jp.pion_rad, lat, field3dparams, opt);
      a2a_print_time("main","Sigma meson field compute", dclock()-time);
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
      if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
#endif
    }
  }

  if(cmdline.ktosigma_save_sigma_mf){
    LOGA2A << "Writing sigma meson fields to disk" << std::endl;
    double time = dclock();
    computeSigmaMesonFields1s<Vtype,Wtype,StationarySigmaMomentaPolicy>::write(mf_sigma, sigma_mom, params.meas_arg.WorkDirectory, conf, params.jp.pion_rad);
    a2a_print_time("main","Sigma meson field write", dclock()-time);
  }
}


//Compute sigma 2pt function with file in Tianle's format
template<typename Vtype, typename Wtype, typename SigmaMomentumPolicy>
void computeSigma2pt(std::vector< fVector<typename A2Apolicies::ScalarComplexType> > &sigma_bub, //output bubble
		     MesonFieldMomentumPairContainer<getMesonFieldType<Wtype,Vtype> > &mf_sigma_con, const SigmaMomentumPolicy &sigma_mom, const int conf, const Parameters &params){
  const int nmom = sigma_mom.nMom();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();

  //All momentum combinations have total momentum 0 at source and sink
  LOGA2A << "Computing sigma 2pt function" << std::endl;
  double time = -dclock();

  typedef typename A2Apolicies::ScalarComplexType ScalarComplexType;

  sigma_bub.resize(nmom);
  for(int pidx=0;pidx<nmom;pidx++){
    //Compute the disconnected bubble
    LOGA2A << "Sigma disconnected bubble pidx=" << pidx << std::endl;
    fVector<ScalarComplexType> &into = sigma_bub[pidx]; into.resize(Lt);
    ComputeSigmaContractions<Vtype,Wtype>::computeDisconnectedBubble(into, mf_sigma_con, sigma_mom, pidx);

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
      a2a_printf("Sigma connected psrcidx=%d psnkidx=%d\n",psrcidx,psnkidx);
      fMatrix<ScalarComplexType> into(Lt,Lt);
      ComputeSigmaContractions<Vtype,Wtype>::computeConnected(into, mf_sigma_con, sigma_mom, psrcidx, psnkidx);

      fMatrix<ScalarComplexType> disconn(Lt,Lt);
      ComputeSigmaContractions<Vtype,Wtype>::computeDisconnectedDiagram(disconn, sigma_bub[psrcidx], sigma_bub[psnkidx]);

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
  a2a_print_time("main","Sigma 2pt function",time);

  printMem("Memory after Sigma 2pt function computation");
}


#endif
