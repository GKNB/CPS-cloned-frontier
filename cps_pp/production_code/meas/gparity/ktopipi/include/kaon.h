#ifndef _KTOPIPI_MAIN_A2A_KAON_H_
#define _KTOPIPI_MAIN_A2A_KAON_H_

struct KaonMesonFields{
  typedef std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mfVector;
  mfVector mf_ls;
  mfVector mf_sl;
  KaonMesonFields(){}

  void move(KaonMesonFields &r){
    int n = r.mf_ls.size();
    for(int i=0;i<n;i++){
      mf_ls[i].move(r.mf_ls[i]);
      mf_sl[i].move(r.mf_sl[i]);
    }
  }
};

template<typename KaonMomentumPolicy>
void computeKaon2pt(typename ComputeKaon<A2Apolicies>::Vtype &V, typename ComputeKaon<A2Apolicies>::Wtype &W, 
		    typename ComputeKaon<A2Apolicies>::Vtype &V_s, typename ComputeKaon<A2Apolicies>::Wtype &W_s,
		    const KaonMomentumPolicy &kaon_mom,
		    const int conf, Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
		    KaonMesonFields *keep_mesonfields = NULL){
  if(!UniqueID()) printf("Computing kaon 2pt function\n");
  double time = -dclock();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();

  typedef std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mfVector;
  
  KaonMesonFields mf;
  ComputeKaon<A2Apolicies>::computeMesonFields(mf.mf_ls, mf.mf_sl,
					       W, V, W_s, V_s, kaon_mom,
					       params.jp.kaon_rad, lat, field3dparams);

  fMatrix<typename A2Apolicies::ScalarComplexType> kaon(Lt,Lt);
  ComputeKaon<A2Apolicies>::compute(kaon, mf.mf_ls, mf.mf_sl);
  
  std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_kaoncorr";
  kaon.write(os.str());
#ifdef WRITE_HEX_OUTPUT
  os << ".hexfloat";
  kaon.write(os.str(),true);
#endif
  time += dclock();
  print_time("main","Kaon 2pt function",time);

  if(keep_mesonfields != NULL) keep_mesonfields->move(mf);
  
  printMem("Memory after kaon 2pt function computation");
}

#endif
