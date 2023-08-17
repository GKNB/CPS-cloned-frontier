#ifndef _COMPUTE_KTOSIGMA_H_
#define _COMPUTE_KTOSIGMA_H_

template<typename Vtype, typename Wtype, typename SigmaMomentumPolicy>
void computeKtoSigmaContractions(const Vtype &V, Wtype &W,
				 const Vtype &V_s, Wtype &W_s,
				 const LSWWmesonFields<Wtype> &mf_ls_ww_con, MesonFieldMomentumPairContainer<getMesonFieldType<Wtype,Vtype> > &mf_sigma_con,
				 const SigmaMomentumPolicy &sigma_mom, const int conf, const Parameters &params, 
				 const std::string &src_descr, const std::string &src_fappend, bool do_type4, const std::string &postpend = ""){
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  
  typedef ComputeKtoPiPiGparity<Vtype,Wtype> ComputeP;
  typedef typename ComputeP::ResultsContainerType ResultsContainerType;
  typedef typename ComputeP::MixDiagResultsContainerType MixDiagResultsContainerType;
  typedef typename LSWWmesonFields<Wtype>::MesonFieldType mf_WW;
  typedef getMesonFieldType<Wtype,Vtype> mf_WV;
  const std::vector<mf_WW> &mf_ls_ww = mf_ls_ww_con.mf_ls_ww;

  std::vector<int> k_sigma_separation(params.jp.k_pi_separation.k_pi_separation_len);
  for(int i=0;i<params.jp.k_pi_separation.k_pi_separation_len;i++) k_sigma_separation[i] = params.jp.k_pi_separation.k_pi_separation_val[i];

  printMem("Memory at start of K->sigma contraction section");

  //Pre-average over sigma meson fields
  double time = dclock();
  std::vector<mf_WV> mf_sigma(Lt);

#ifdef NODE_DISTRIBUTE_MESONFIELDS
  void* gather_buf;
  size_t gather_buf_size = NULL;
#endif  

  for(int s = 0; s< sigma_mom.nMom(); s++){
    ThreeMomentum pwdag = sigma_mom.getWdagMom(s);
    ThreeMomentum pv = sigma_mom.getVmom(s);
 
    std::vector<mf_WV> &mf_sigma_s = mf_sigma_con.get(pwdag,pv);

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    if(s==0){
      gather_buf_size = mf_sigma_s[0].byte_size();
      gather_buf = memalign_check(128, gather_buf_size); //setup the buffer memory
    }
#endif

    for(int t=0;t<Lt;t++){
#ifdef NODE_DISTRIBUTE_MESONFIELDS
#if 0
      mf_sigma_s[t].enableExternalBuffer(gather_buf, gather_buf_size, 128);
#endif
      mf_sigma_s[t].nodeGet(); //allocs of non-master node use external buf
#endif
    
      if(s==0){
	mf_sigma[t] = mf_sigma_s[t];
      }else{
#ifndef MEMTEST_MODE
	mf_sigma[t].plus_equals(mf_sigma_s[t]);
#endif
      }

#ifdef NODE_DISTRIBUTE_MESONFIELDS
      mf_sigma_s[t].nodeDistribute();
#if 0
      mf_sigma_s[t].disableExternalBuffer();
#endif
#endif
    }
  }

#ifndef MEMTEST_MODE
  for(int t=0;t<Lt;t++) mf_sigma[t].times_equals(1./sigma_mom.nMom());
#endif

#ifdef NODE_DISTRIBUTE_MESONFIELDS
  nodeDistributeMany(1,&mf_sigma);
  free(gather_buf);
#endif

  a2a_print_time("computeKtoSigmaContractions","Sigma meson field pre-average", dclock()-time);
	

#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
  if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
  LOGA2A << printf("Trimming block allocator" << std::endl;
  DistributedMemoryStorage::block_allocator().trim();
  if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
#endif


  printMem("Memory at start of K->sigma contraction compute");

  //Start the computation
  ComputeKtoSigma<Vtype,Wtype> compute(V, W, V_s, W_s, mf_ls_ww, k_sigma_separation);

  //Type1/2
  {    
    double time = -dclock();
    LOGA2A << "Starting K->sigma type 1/2 contractions with source " << src_descr << std::endl;
    printMem("Memory at start of K->sigma type 1/2 contraction");     
    
    std::vector<ResultsContainerType> result;
    compute.type12(result, mf_sigma);
  
    for(int i=0;i<k_sigma_separation.size();i++){
      std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_ktosigma_type12_deltat_" << k_sigma_separation[i] << src_fappend << postpend;
      result[i].write(os.str());
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat"; result[i].write(os.str(),true);
#endif
    }
    a2a_print_time("main","K->sigma type 1/2",time+dclock());
  }

  //Type3
  {
    double time = -dclock();
    LOGA2A << "Starting K->sigma type 3 contractions with source " << src_descr << std::endl;
    printMem("Memory at start of K->sigma type 3 contraction");     
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
    a2a_print_time("main","K->sigma type 3",time+dclock());
  }  

  //Type4
  if(do_type4){
    double time = -dclock();
    LOGA2A << "Starting K->sigma type 4 contractions" << std::endl;
    printMem("Memory at start of K->sigma type 4 contraction");     
    ResultsContainerType result;
    MixDiagResultsContainerType mix;
    compute.type4(result, mix);
  
    std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_ktosigma_type4" << postpend;
    write(os.str(), result, mix);
#ifdef WRITE_HEX_OUTPUT
    os << ".hexfloat"; write(os.str(), result, mix, true);
#endif
    a2a_print_time("main","K->sigma type 4",time+dclock());
  }

  printMem("Memory at end of K->sigma contractions");
}

#endif
