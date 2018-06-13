#ifndef _COMPUTE_KTOSIGMA_H_
#define _COMPUTE_KTOSIGMA_H_

template<typename SigmaMomentumPolicy>
void computeKtoSigmaContractions(const A2AvectorV<A2Apolicies> &V, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W,
				const A2AvectorV<A2Apolicies> &V_s, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W_s,
				const LSWWmesonFields &mf_ls_ww_con, MesonFieldMomentumPairContainer<A2Apolicies> &mf_sigma_con,
				const SigmaMomentumPolicy &sigma_mom, const int conf, const Parameters &params, 
				const std::string &src_descr, const std::string &src_fappend, bool do_type4, const std::string &postpend = ""){
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  
  typedef ComputeKtoPiPiGparity<A2Apolicies>::ResultsContainerType ResultsContainerType;
  typedef ComputeKtoPiPiGparity<A2Apolicies>::MixDiagResultsContainerType MixDiagResultsContainerType;

  const std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw> > &mf_ls_ww = mf_ls_ww_con.mf_ls_ww;

  std::vector<int> k_sigma_separation(params.jp.k_pi_separation.k_pi_separation_len);
  for(int i=0;i<params.jp.k_pi_separation.k_pi_separation_len;i++) k_sigma_separation[i] = params.jp.k_pi_separation.k_pi_separation_val[i];

  printMem("Memory at start of K->sigma contraction section");

  //Pre-average over sigma meson fields
  double time = dclock();
  std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_sigma(Lt);
  
  for(int s = 0; s< sigma_mom.nMom(); s++){
    ThreeMomentum pwdag = sigma_mom.getWdagMom(s);
    ThreeMomentum pv = sigma_mom.getVmom(s);

    std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > &mf_sigma_s = mf_sigma_con.get(pwdag,pv);

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeGetMany(1, &mf_sigma_s);
#endif
    
    if(s==0)
      for(int t=0;t<Lt;t++) mf_sigma[t] = mf_sigma_s[t];
    else
      for(int t=0;t<Lt;t++) mf_sigma[t].plus_equals(mf_sigma_s[t]);


#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(1,&mf_sigma_s);
#endif
  }

  for(int t=0;t<Lt;t++) mf_sigma[t].times_equals(1./sigma_mom.nMom());

  nodeDistributeMany(1,&mf_sigma);

  print_time("computeKtoSigmaContractions","Sigma meson field pre-average", dclock()-time);
	

#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
  if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
  if(!UniqueID()) printf("Trimming block allocator\n");
  DistributedMemoryStorage::block_allocator().trim();
  if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
#endif


  printMem("Memory at start of K->sigma contraction compute");

  //Start the computation
  ComputeKtoSigma<A2Apolicies> compute(V, W, V_s, W_s, mf_ls_ww, k_sigma_separation);

  //Type1/2
  {
    double time = -dclock();
    if(!UniqueID()) printf("Starting K->sigma type 1/2 contractions with source %s\n",src_descr.c_str());
    std::vector<ResultsContainerType> result;
    compute.type12(result, mf_sigma);
  
    for(int i=0;i<k_sigma_separation.size();i++){
      std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_ktosigma_type12_deltat_" << k_sigma_separation[i] << src_fappend << postpend;
      result[i].write(os.str());
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat"; result[i].write(os.str(),true);
#endif
    }
    print_time("main","K->sigma type 1/2",time+dclock());
  }
  //Type3
  {
    double time = -dclock();
    if(!UniqueID()) printf("Starting K->sigma type 3 contractions with source %s\n",src_descr.c_str());
    std::vector<ResultsContainerType> result;
    std::vector<MixDiagResultsContainerType> mix;
    compute.type3(result, mix, mf_sigma);
  
    for(int i=0;i<k_sigma_separation.size();i++){
      std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_ktosigma_type3_deltat_" << k_sigma_separation[i] << src_fappend << postpend;
      write(os.str(), result[i], mix[i]);
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat"; write(os.str(), result[i], mix[i], true);
#endif
    }
    print_time("main","K->sigma type 3",time+dclock());
  }  
  //Type4
  if(do_type4){
    double time = -dclock();
    if(!UniqueID()) printf("Starting K->sigma type 4 contractions\n");
    ResultsContainerType result;
    MixDiagResultsContainerType mix;
    compute.type4(result, mix);
  
    std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_ktosigma_type4" << postpend;
    write(os.str(), result, mix);
#ifdef WRITE_HEX_OUTPUT
    os << ".hexfloat"; write(os.str(), result, mix, true);
#endif
    print_time("main","K->sigma type 4",time+dclock());
  }

  printMem("Memory at end of K->sigma contractions");
}

#endif
