#ifndef _COMPUTE_KAON_H
#define _COMPUTE_KAON_H

#include<alg/a2a/required_momenta.h>
#include<alg/a2a/mesonfield_computemany.h>
#include<alg/a2a/inner_product.h>

//Compute stationary kaon two-point function with and without GPBC
CPS_START_NAMESPACE

//Policy for stationary kaon with and without GPBC
class StationaryKaonMomentaPolicy{
public:
  RequiredMomentum LightHeavy; //source
  RequiredMomentum HeavyLight; //sink

  //In this original policy the light-heavy (source) meson field is assigned the specified V/W momenta and the heavy-light (sink) meson field the negatives of those
  StationaryKaonMomentaPolicy(){
    LightHeavy.combineSameTotalMomentum(true); //momentum pairs with same total momentum will be added to same entry and treated as 'alternates' which we average together below
    HeavyLight.combineSameTotalMomentum(true);
    const int ngp = LightHeavy.nGparityDirs();
    std::string p;
    if(ngp == 0){
      p = "(0,0,0) + (0,0,0)";      
    }else if(ngp == 1){
      p = "(-1,0,0) + (1,0,0)";
    }else if(ngp == 2){
      p = "(-1,-1,0) + (1,1,0)";
    }else if(ngp == 3){
      p = "(-1,-1,-1) + (1,1,1)";
    }else{
      ERR.General("StationaryKaonMomentaPolicy","constructor","ngp cannot be >3\n");
    }
    std::pair<ThreeMomentum,ThreeMomentum> p2 = ThreeMomentum::parse_str_two_mom(p);
    
    LightHeavy.addP(p2);  HeavyLight.addP(std::pair<ThreeMomentum,ThreeMomentum>(-p2.first,-p2.second)); 
  };
};


//Same as the above but where we reverse the momentum assignments of the W^dag and V
class ReverseKaonMomentaPolicy: public StationaryKaonMomentaPolicy{
public:
  ReverseKaonMomentaPolicy(): StationaryKaonMomentaPolicy() {
    this->LightHeavy.reverseABmomentumAssignments();
    this->HeavyLight.reverseABmomentumAssignments();
  }
};

//Add Wdag, V momenta and the reverse assignment to make a symmetric combination
class SymmetricKaonMomentaPolicy: public StationaryKaonMomentaPolicy{
public:
  SymmetricKaonMomentaPolicy(): StationaryKaonMomentaPolicy() {
    this->LightHeavy.symmetrizeABmomentumAssignments();
    this->HeavyLight.symmetrizeABmomentumAssignments();
  }
};



template<typename mf_Policies>
class ComputeKaon{
 public:
  typedef typename A2Asource<typename mf_Policies::SourcePolicies::ComplexType, typename mf_Policies::SourcePolicies::MappingPolicy, typename mf_Policies::SourcePolicies::AllocPolicy>::FieldType::InputParamType FieldParamType;
#ifdef USE_DESTRUCTIVE_FFT
  typedef A2AvectorW<mf_Policies> Wtype;
  typedef A2AvectorV<mf_Policies> Vtype;
#else
  typedef const A2AvectorW<mf_Policies> Wtype;
  typedef const A2AvectorV<mf_Policies> Vtype;
#endif

  //Compute the two-point function using a hydrogen-wavefunction source of radius 'rad'
  //result is indexed by (tsrc, tsep)  where tsep is the source-sink separation
  template<typename KaonMomentumPolicy>
  static void compute(fMatrix<typename mf_Policies::ScalarComplexType> &into,
		      Wtype &W, Vtype &V, 
		      Wtype &W_s, Vtype &V_s,
		      const KaonMomentumPolicy &kaon_momentum,
		      const Float &rad, Lattice &lattice,
		      const FieldParamType &src_setup_params = NullObject()){
    typedef typename mf_Policies::ComplexType ComplexType;
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    typedef typename mf_Policies::SourcePolicies SourcePolicies;

    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt,Lt);

    std::vector<Wtype*> Wspecies(2); Wspecies[0] = &W; Wspecies[1] = &W_s;
    std::vector<Vtype*> Vspecies(2); Vspecies[0] = &V; Vspecies[1] = &V_s;
    
    //Construct the meson fields
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > mf_ls(Lt);
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > mf_sl(Lt);

    assert(kaon_momentum.LightHeavy.nMom() == 1); assert(kaon_momentum.HeavyLight.nMom() == 1);
    if(!GJP.Gparity()){
      assert(kaon_momentum.LightHeavy.nAltMom(0) == 1); assert(kaon_momentum.HeavyLight.nAltMom(0) == 1);
    
      typedef A2AexpSource<SourcePolicies> SourceType;
      typedef SCspinInnerProduct<15,ComplexType, SourceType> InnerType;
      typedef BasicSourceStorage<mf_Policies,InnerType> StorageType;
      
      SourceType src(rad,src_setup_params);
      InnerType g5_inner(src);
      StorageType mf_store(g5_inner);
      
      mf_store.addCompute(0,1, kaon_momentum.LightHeavy.getWmom(0), kaon_momentum.LightHeavy.getVmom(0));
      mf_store.addCompute(1,0, kaon_momentum.HeavyLight.getWmom(0), kaon_momentum.HeavyLight.getVmom(0));

      ComputeMesonFields<mf_Policies,StorageType>::compute(mf_store,Wspecies,Vspecies,lattice);
      mf_ls = mf_store[0];
      mf_sl = mf_store[1];
      
    }else{ //For GPBC we need a different smearing function for source and sink because the flavor structure depends on the momentum of the V field, which is opposite between source and sink
      typedef A2AflavorProjectedExpSource<SourcePolicies> SourceType;
      typedef SCFspinflavorInnerProduct<15, ComplexType, SourceType > InnerType;
      typedef GparityFlavorProjectedBasicSourceStorage<mf_Policies,InnerType> StorageType;

      int pbase[3]; //we reset the momentum for each computation so we technically don't need this - however the code demands a valid momentum
      GparityBaseMomentum(pbase,+1);
      
      SourceType src(rad,pbase,src_setup_params);
      InnerType g5_s0_inner(sigma0,src);
      StorageType mf_store(g5_s0_inner,src);

      int nalt = kaon_momentum.LightHeavy.nAltMom(0);  assert(kaon_momentum.HeavyLight.nAltMom(0) == nalt);
      for(int a=0;a<nalt;a++){
	mf_store.addCompute(0,1, kaon_momentum.LightHeavy.getWmom(0,a), kaon_momentum.LightHeavy.getVmom(0,a));
	mf_store.addCompute(1,0, kaon_momentum.HeavyLight.getWmom(0,a), kaon_momentum.HeavyLight.getVmom(0,a));
      }
      
      ComputeMesonFields<mf_Policies,StorageType>::compute(mf_store,Wspecies,Vspecies,lattice
#ifdef NODE_DISTRIBUTE_MESONFIELDS
							   ,true
#endif
							   );
      struct indexer{
	const int base;
	const int stride;
	indexer(const int b, const int s): base(b), stride(s){}	
	inline int operator()(const int i) const{
	  return base+i*stride;
	}
      };
      
      stridedAverageFree(mf_store,indexer(0,2), nalt, true);
      stridedAverageFree(mf_store,indexer(1,2), nalt, true);

      for(int t=0;t<Lt;t++){
      	mf_ls[t].move(mf_store[0][t]);
	mf_sl[t].move(mf_store[1][t]);
      } 
    }//if G-parity

    //Compute the two-point function
    trace(into,mf_sl,mf_ls);
    into *= ScalarComplexType(0.5,0);
    rearrangeTsrcTsep(into); //rearrange temporal ordering
  }

};

CPS_END_NAMESPACE

#endif

