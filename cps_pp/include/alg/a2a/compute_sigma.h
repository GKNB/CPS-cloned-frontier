#ifndef _COMPUTE_SIGMA_H
#define _COMPUTE_SIGMA_H

#include<alg/a2a/required_momenta.h>
#include<alg/a2a/mesonfield_computemany.h>

//Compute stationary sigma meson two-point function with and without GPBC
CPS_START_NAMESPACE

//Policy for stationary sigma with and without GPBC
class StationarySigmaMomentaPolicy{
public:
  void setupMomenta(const int &ngp){
    RequiredMomentum<StationarySigmaMomentaPolicy> *tt = static_cast<RequiredMomentum<StationarySigmaMomentaPolicy>*>(this);
    if(ngp == 0){
      tt->addP("(0,0,0) + (0,0,0)");
    }else if(ngp == 1){
      tt->addPandMinusP("(-1,0,0) + (1,0,0)");
      tt->addPandMinusP("(-3,0,0) + (3,0,0)");
    }else if(ngp == 2){
      tt->addPandMinusP("(-1,-1,0) + (1,1,0)");
      tt->addPandMinusP("(3,-1,0) + (-3,1,0)");
    }else if(ngp == 3){
      tt->addPandMinusP("(-1,-1,-1) + (1,1,1)");
      tt->addPandMinusP("(3,-1,-1) + (-3,1,1)");
    }else{
      ERR.General("StationarySigmaMomentaPolicy","setupMomenta","ngp cannot be >3\n");
    }
  }
};


template<typename mf_Policies>
class ComputeSigma{
 public:
  typedef typename A2Asource<typename mf_Policies::SourcePolicies::ComplexType, typename mf_Policies::SourcePolicies::DimensionPolicy, typename mf_Policies::SourcePolicies::AllocPolicy>::FieldType::InputParamType FieldParamType;
  
  //Computes sigma meson fields and saves to disk
  static void computeAndWrite(const std::string &work_dir, const int traj,
			      const A2AvectorW<mf_Policies> &W, const A2AvectorV<mf_Policies> &V, const Float &rad, Lattice &lattice,
			      const FieldParamType &src_setup_params = NullObject()){

    typedef typename mf_Policies::ComplexType ComplexType;
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    typedef typename mf_Policies::SourcePolicies SourcePolicies;
    typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> MesonFieldType;
    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    RequiredMomentum<StationarySigmaMomentaPolicy> momenta;

    std::vector< A2AvectorW<mf_Policies> const*> Wspecies(1,&W);
    std::vector< A2AvectorV<mf_Policies> const*> Vspecies(1,&V);
    
    if(GJP.Gparity()){
      typedef A2AflavorProjectedExpSource<SourcePolicies> ExpSrcType;
      typedef A2AflavorProjectedHydrogenSource<SourcePolicies> HydSrcType;
      
      typedef Elem<ExpSrcType, Elem<HydSrcType,ListEnd > > SrcList;
      typedef A2AmultiSource<SrcList> MultiSrcType;
      typedef SCFspinflavorInnerProduct<0,ComplexType,MultiSrcType,true,false> MultiInnerType; //unit matrix spin structure
      typedef GparityFlavorProjectedMultiSourceStorage<mf_Policies, MultiInnerType> StorageType;

      int pbase[3]; //we reset the momentum for each computation so we technically don't need this - however the code demands a valid momentum
      GparityBaseMomentum(pbase,+1);
      
      MultiSrcType src;
      src.template getSource<0>().setup(rad,pbase,src_setup_params); //1s
      src.template getSource<1>().setup(2,0,0,rad,pbase,src_setup_params); //2s

      int nSources = MultiSrcType::nSources;
      std::string src_names[2] = {"1s","2s"};
      
      MultiInnerType gunit_s0_inner(sigma0, src);
      StorageType mf_store(gunit_s0_inner);

      for(int pidx=0;pidx<momenta.nMom();pidx++){
	ThreeMomentum p_w = momenta.getWmom(pidx,false);
	ThreeMomentum p_v = momenta.getVmom(pidx,false);
	mf_store.addCompute(0,0, p_w,p_v);	
      }
      ComputeMesonFields<mf_Policies,StorageType>::compute(mf_store,Wspecies,Vspecies,lattice);

      for(int pidx=0;pidx<momenta.nMom();pidx++){
	ThreeMomentum p_wdag = -momenta.getWmom(pidx,false);
	ThreeMomentum p_v = momenta.getVmom(pidx,false);
	
	for(int s=0;s<nSources;s++){
	  std::ostringstream os; //momenta in units of pi/2L
	  os << work_dir << "/traj_" << traj << "_sigma_mfwv_mom" << p_wdag.file_str() << "_plus" << p_v.file_str() << "_hyd" << src_names[s] << "_rad" << rad << ".dat";
	  MesonFieldType::write(os.str(),mf_store(s,pidx));
	}
      } 
      
    }else{
      typedef A2AexpSource<SourcePolicies> SrcType;
      typedef SCspinInnerProduct<ComplexType,SrcType> InnerType;
      typedef BasicSourceStorage<mf_Policies,InnerType> StorageType;
      
      SrcType src(rad,src_setup_params);
      InnerType gunit_inner(0,src);

      StorageType mf_store(gunit_inner);

      for(int pidx=0;pidx<momenta.nMom();pidx++){
	ThreeMomentum p_w = momenta.getWmom(pidx,false);
	ThreeMomentum p_v = momenta.getVmom(pidx,false);
	mf_store.addCompute(0,0, p_w,p_v);	
      }
      ComputeMesonFields<mf_Policies,StorageType>::compute(mf_store,Wspecies,Vspecies,lattice);
      
      for(int pidx=0;pidx<momenta.nMom();pidx++){
	ThreeMomentum p_wdag = -momenta.getWmom(pidx,false);
	ThreeMomentum p_v = momenta.getVmom(pidx,false);
	
	std::ostringstream os; //momenta in units of pi/2L
	os << work_dir << "/traj_" << traj << "_sigma_mfwv_mom" << p_wdag.file_str() << "_plus" << p_v.file_str() << "_hyd1s_rad" << rad << ".dat";
	MesonFieldType::write(os.str(),mf_store[pidx]);
      } 
    }
  }

};

CPS_END_NAMESPACE

#endif

