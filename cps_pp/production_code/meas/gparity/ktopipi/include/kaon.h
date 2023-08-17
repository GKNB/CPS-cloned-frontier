#ifndef _KTOPIPI_MAIN_A2A_KAON_H_
#define _KTOPIPI_MAIN_A2A_KAON_H_

struct KaonMesonFields{
  typedef std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mfVector;
  mfVector mf_ls;
  mfVector mf_sl;
  KaonMesonFields(){}

  inline void move(KaonMesonFields &r){
    int n = r.mf_ls.size();
    mf_ls.resize(n);
    mf_sl.resize(n);
    for(int i=0;i<n;i++){
      mf_ls[i].move(r.mf_ls[i]);
      mf_sl[i].move(r.mf_sl[i]);
    }
  }
  inline void free_mem(){
    for(int i=0;i<mf_ls.size();i++){
      mf_ls[i].free_mem();
      mf_sl[i].free_mem();
    }
  }
  inline void average(KaonMesonFields &with){
    for(int a=0;a<2;a++){
      mfVector &l = a==0 ? mf_ls : mf_sl;
      mfVector &r = a==0 ? with.mf_ls : with.mf_sl;
    
      for(int i=0;i<l.size();i++){
	bool redist_l = false, redist_r = false;
	if(!l[i].isOnNode()){ l[i].nodeGet(); redist_l = true; }
	if(!r[i].isOnNode()){ r[i].nodeGet(); redist_r = true; }
      
	l[i].average(r[i]);

	if(redist_l) l[i].nodeDistribute();
	if(redist_r) r[i].nodeDistribute();	
      }
    }
  }
};

void computeKaon2ptContraction(const int conf, const Parameters &params, const KaonMesonFields &mf, const std::string &postpend = ""){
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  fMatrix<typename A2Apolicies::ScalarComplexType> kaon(Lt,Lt);
  ComputeKaon<A2Apolicies>::compute(kaon, mf.mf_ls, mf.mf_sl);
  
  std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_kaoncorr" << postpend;
  kaon.write(os.str());
#ifdef WRITE_HEX_OUTPUT
  os << ".hexfloat";
  kaon.write(os.str(),true);
#endif
}

template<typename KaonMomentumPolicy>
void randomizeKaonMesonFields(KaonMesonFields &mf, typename ComputeKaon<A2Apolicies>::Vtype &V, typename ComputeKaon<A2Apolicies>::Wtype &W, 
			      typename ComputeKaon<A2Apolicies>::Vtype &V_s, typename ComputeKaon<A2Apolicies>::Wtype &W_s,
			      const KaonMomentumPolicy &kaon_mom){
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  mf.mf_ls.resize(Lt);
  mf.mf_sl.resize(Lt);
  for(int t=0;t<Lt;t++){ 
    mf.mf_ls[t].setup(W,V_s,t,t); 
    mf.mf_ls[t].testRandom();
    mf.mf_sl[t].setup(W_s,V,t,t); 
    mf.mf_sl[t].testRandom();
  }
}
			      

template<typename KaonMomentumPolicy>
void computeKaonMesonFields(KaonMesonFields &mf, typename ComputeKaon<A2Apolicies>::Vtype &V, typename ComputeKaon<A2Apolicies>::Wtype &W, 
			    typename ComputeKaon<A2Apolicies>::Vtype &V_s, typename ComputeKaon<A2Apolicies>::Wtype &W_s,
			    const KaonMomentumPolicy &kaon_mom,
			    Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
			    bool randomize_mf){
  if(randomize_mf){    
    randomizeKaonMesonFields(mf,V,W,V_s,W_s,kaon_mom);
  }else{
    ComputeKaon<A2Apolicies>::computeMesonFields(mf.mf_ls, mf.mf_sl,
						 W, V, W_s, V_s, kaon_mom,
						 params.jp.kaon_rad, lat, field3dparams);
  }
}


template<typename KaonMomentumPolicy>
void computeKaon2pt(typename ComputeKaon<A2Apolicies>::Vtype &V, typename ComputeKaon<A2Apolicies>::Wtype &W, 
		    typename ComputeKaon<A2Apolicies>::Vtype &V_s, typename ComputeKaon<A2Apolicies>::Wtype &W_s,
		    const KaonMomentumPolicy &kaon_mom,
		    const int conf, Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams, bool randomize_mf,
		    KaonMesonFields *keep_mesonfields = NULL){
  LOGA2A << "Computing kaon 2pt function" << std::endl;
  double time = -dclock();
  
  {  
    KaonMesonFields mf;
    computeKaonMesonFields(mf,V,W,V_s,W_s,kaon_mom,lat,params,field3dparams,randomize_mf);
    
    computeKaon2ptContraction(conf,params,mf);
    
    if(keep_mesonfields != NULL) keep_mesonfields->move(mf);
  }  
  a2a_print_time("main","Kaon 2pt function",time + dclock());
  printMem("Memory after kaon 2pt function computation");
  
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
  LOGA2A << "Trimming block storage" << std::endl;
  DistributedMemoryStorage::block_allocator().trim();
  if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
#endif
}


template<typename KaonMomentumPolicyStd, typename KaonMomentumPolicyReverse>
  void computeKaon2ptStandardAndSymmetric(typename ComputeKaon<A2Apolicies>::Vtype &V, typename ComputeKaon<A2Apolicies>::Wtype &W, 
					  typename ComputeKaon<A2Apolicies>::Vtype &V_s, typename ComputeKaon<A2Apolicies>::Wtype &W_s,
					  const KaonMomentumPolicyStd &kaon_mom_std, const KaonMomentumPolicyReverse &kaon_mom_rev,
					  const int conf, Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,bool randomize_mf){
   LOGA2A << "Computing kaon 2pt function standard and symmetric" << std::endl;
   double time = -dclock();
   const int Lt = GJP.Tnodes() * GJP.TnodeSites();
   
   KaonMesonFields mf_std;
   computeKaonMesonFields(mf_std,V,W,V_s,W_s,kaon_mom_std,lat,params,field3dparams,randomize_mf);

   computeKaon2ptContraction(conf,params,mf_std);
   
   KaonMesonFields mf_symm;
   computeKaonMesonFields(mf_symm,V,W,V_s,W_s,kaon_mom_rev,lat,params,field3dparams,randomize_mf);

   mf_symm.average(mf_std);
   mf_std.free_mem();
     
   computeKaon2ptContraction(conf,params,mf_symm,"_symm");
   
   a2a_print_time("main","Kaon 2pt function",time + dclock());
   printMem("Memory after kaon 2pt function computation");
}
 
#endif
