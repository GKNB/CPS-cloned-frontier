#ifndef _COMPUTE_LL_MESONFIELDS_H_
#define _COMPUTE_LL_MESONFIELDS_H_

//Compute the light-light meson fields used in the main job

#include<alg/a2a/required_momenta.h>
#include<alg/a2a/mesonfield_computemany.h>
#include<alg/a2a/inner_product.h>
#include<alg/a2a/mf_momcontainer.h>

CPS_START_NAMESPACE

template<typename mf_Policies>
struct computeMesonFieldsBase{
  typedef typename A2Asource<typename mf_Policies::SourcePolicies::ComplexType, typename mf_Policies::SourcePolicies::MappingPolicy, typename mf_Policies::SourcePolicies::AllocPolicy>::FieldType::InputParamType FieldParamType;

#ifdef USE_DESTRUCTIVE_FFT
  typedef A2AvectorW<mf_Policies> Wtype;
  typedef A2AvectorV<mf_Policies> Vtype;
#else
  typedef const A2AvectorW<mf_Policies> Wtype;
  typedef const A2AvectorV<mf_Policies> Vtype;
#endif

  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> MesonFieldType;
  typedef std::vector<MesonFieldType> MesonFieldVectorType;
  typedef typename mf_Policies::ComplexType ComplexType;
  typedef typename mf_Policies::SourcePolicies SourcePolicies;

#define INHERIT(TYPE,FROM) typedef typename FROM::TYPE TYPE

#define INHERIT_FROM_BASE \
  INHERIT(FieldParamType, computeMesonFieldsBase<mf_Policies>); 	\
  INHERIT(Wtype, computeMesonFieldsBase<mf_Policies>); 	\
  INHERIT(Vtype, computeMesonFieldsBase<mf_Policies>); 	\
  INHERIT(MesonFieldType, computeMesonFieldsBase<mf_Policies>); 	\
  INHERIT(MesonFieldVectorType, computeMesonFieldsBase<mf_Policies>); 	\
  INHERIT(ComplexType, computeMesonFieldsBase<mf_Policies>); 	\
  INHERIT(SourcePolicies, computeMesonFieldsBase<mf_Policies>)
};


//Non-Gparity 1s meson fields
template<typename mf_Policies, typename PionMomentumPolicy>
class computeNonGparityLLmesonFields1s{
public:
  INHERIT_FROM_BASE;
  
  static void computeMesonFields(MesonFieldMomentumContainer<mf_Policies> &mf_ll_con, //container for 1s pion output fields, accessible by ThreeMomentum of pion
				 const std::string &work_dir, const int traj,  //all meson fields stored to disk
				 const PionMomentumPolicy &pion_mom, //object that tells us what quark momenta to use
				 Wtype &W, Vtype &V,
				 const Float &rad, //exponential wavefunction radius
				 Lattice &lattice,			      
				 const FieldParamType &src_setup_params = NullObject()){
    assert(!GJP.Gparity());
    Float time = -dclock();    
    std::vector<Wtype*> Wspecies(1, &W);
    std::vector<Vtype*> Vspecies(1, &V);

    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    const int nmom = pion_mom.nMom();

    typedef A2AexpSource<SourcePolicies> SrcType;
    typedef SCspinInnerProduct<15,ComplexType,SrcType> InnerType;
    typedef BasicSourceStorage<mf_Policies,InnerType> StorageType;

    SrcType src(rad,src_setup_params);
    InnerType g5_inner(src);

    StorageType mf_store(g5_inner);

    for(int pidx=0;pidx<nmom;pidx++){
      ThreeMomentum p_w = pion_mom.getWmom(pidx,false);
      ThreeMomentum p_v = pion_mom.getVmom(pidx,false);
      mf_store.addCompute(0,0, p_w,p_v);	
    }
    ComputeMesonFields<mf_Policies,StorageType>::compute(mf_store,Wspecies,Vspecies,lattice);

    for(int pidx=0;pidx<nmom;pidx++){
      const ThreeMomentum pi_mom_pidx = pion_mom.getMesonMomentum(pidx);
      MesonFieldVectorType &stored = mf_ll_con.copyAdd(pi_mom_pidx, mf_store[pidx]);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
      if(!UniqueID()){ printf("Distributing mf_ll[%d]\n",pidx); fflush(stdout); }
      nodeDistributeMany(1,&stored);
#endif      
    }
    
    time += dclock();
    print_time("ComputePion::computeMesonFields","total",time);      
  }
};


//Methods for generating meson fields for 1s and 2s hydrogen wavefunction sources together
template<typename mf_Policies, typename PionMomentumPolicy>
class computeGparityLLmesonFields1s2s{
public:
  INHERIT_FROM_BASE;

  typedef A2AflavorProjectedExpSource<SourcePolicies> ExpSrcType;
  typedef A2AflavorProjectedHydrogenSource<SourcePolicies> HydSrcType;
  typedef Elem<ExpSrcType, Elem<HydSrcType,ListEnd > > SrcList;
  typedef A2AmultiSource<SrcList> MultiSrcType;
  //typedef SCFspinflavorInnerProduct<15,ComplexType,MultiSrcType,true,false> MultiInnerType;
  //typedef GparityFlavorProjectedMultiSourceStorage<mf_Policies, MultiInnerType> StorageType;
  
  //Allows for more memory efficient computation algorithm
  typedef GparitySourceShiftInnerProduct<ComplexType,MultiSrcType, flavorMatrixSpinColorContract<15,ComplexType,true,false> > MultiInnerType;
  typedef GparityFlavorProjectedShiftSourceStorage<mf_Policies, MultiInnerType> StorageType;
  
private:
  static StorageType* doCompute(const PionMomentumPolicy &pion_mom,
				const std::vector<Wtype*> &Wspecies, const std::vector<Vtype*> &Vspecies,
				const Float &rad, Lattice &lattice, const FieldParamType &src_setup_params){
    const int nmom = pion_mom.nMom();
    int pbase[3]; //we reset the momentum for each computation so we technically don't need this - however the code demands a valid momentum
    GparityBaseMomentum(pbase,+1);
      
    MultiSrcType src;
    src.template getSource<0>().setup(rad,pbase,src_setup_params); //1s
    src.template getSource<1>().setup(2,0,0,rad,pbase,src_setup_params); //2s
    
    MultiInnerType g5_s3_inner(sigma3, src);
    
    StorageType* mf_store = new StorageType(g5_s3_inner,src);

    int nalt;
    for(int pidx=0;pidx<nmom;pidx++)
      if(pidx == 0) nalt = pion_mom.nAltMom(pidx);
      else assert(pion_mom.nAltMom(pidx) == nalt);
    
    for(int alt=0;alt<nalt;alt++){      
      for(int pidx=0;pidx<nmom;pidx++){      
	ThreeMomentum p_w = pion_mom.getWmom(pidx,alt);
	ThreeMomentum p_v = pion_mom.getVmom(pidx,alt);
	mf_store->addCompute(0,0, p_w,p_v);
      }
    }

    ComputeMesonFields<mf_Policies,StorageType>::compute(*mf_store,Wspecies,Vspecies,lattice
#ifdef NODE_DISTRIBUTE_MESONFIELDS
										       ,true
#endif
										       );
    
    return mf_store;
  }

  //Replace MF with pidx < nmom  with avg( Mf[pidx],  Mf[pidx + nmom] ), i.e. combine with alternate momentum configuration
  static void averageAltMomenta(const PionMomentumPolicy &pion_mom, StorageType* mf_store){

    printMemNodeFile("averageAltMomenta");

    const int nmom = pion_mom.nMom();

    for(int pidx=0;pidx<nmom;pidx++){
      for(int src_idx=0; src_idx<2; src_idx++){
	struct indexer{
	  const int base;
	  const int stride;
	  const int src_idx;
	  indexer(const int b, const int s, const int src): base(b), stride(s), src_idx(src){}	  
	  inline std::pair<int,int> operator()(const int i) const{ return std::pair<int,int>(src_idx, base+i*stride); }
	};
	stridedAverageFree(*mf_store,indexer(pidx, nmom, src_idx), pion_mom.nAltMom(pidx));
      }
    }
  }

  //Writes meson fields with pidx < nMom   (i.e. after alt_mom have been averaged)
  static void writeMF(const std::string &work_dir, const int traj, const Float &rad,
		      const PionMomentumPolicy &pion_mom,
		      StorageType* mf_store, const std::string &postpend = ""){
    printMemNodeFile("writeMF");

    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    const int nmom = pion_mom.nMom();
    std::string src_names[2] = {"1s","2s"};

    for(int pidx=0;pidx<nmom;pidx++){
      for(int s=0;s<2;s++){
	const ThreeMomentum pi_mom_pidx = pion_mom.getMesonMomentum(pidx);
		
	MesonFieldVectorType &mf_base = (*mf_store)(s,pidx);	    
#ifdef NODE_DISTRIBUTE_MESONFIELDS
	nodeGetMany(1,&mf_base);
#endif	
	std::ostringstream os; //momenta in units of pi/2L
	os << work_dir << "/traj_" << traj << "_pion_mf_mom" << pi_mom_pidx.file_str() << "_hyd" << src_names[s] << "_rad" << rad << postpend << ".dat";

#ifndef MEMTEST_MODE	
	MesonFieldType::write(os.str(),mf_base);
#endif

#ifdef NODE_DISTRIBUTE_MESONFIELDS
	nodeDistributeMany(1,&mf_base);
#endif
	
      }
    }
  }

  //Put the meson field in the container
  static void store(MesonFieldMomentumContainer<mf_Policies> &mf_ll_con,
		    const PionMomentumPolicy &pion_mom,
		    const int src_idx,
		    StorageType* mf_store){

    printMemNodeFile("store src_idx="+anyToStr(src_idx));

    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    const int nmom = pion_mom.nMom();
			   
    for(int pidx=0;pidx<nmom;pidx++){
      const ThreeMomentum pi_mom_pidx = pion_mom.getMesonMomentum(pidx);
      MesonFieldVectorType & mf_pbase = (*mf_store)(src_idx,pidx);
      MesonFieldVectorType & stored = mf_ll_con.moveAdd(pi_mom_pidx, mf_pbase);
    }
  }
  
public:

  static void computeMesonFields(MesonFieldMomentumContainer<mf_Policies> &mf_ll_1s_con, //container for 1s pion output fields, accessible by ThreeMomentum of pion
				 MesonFieldMomentumContainer<mf_Policies> &mf_ll_2s_con, //same for 2s source
				 const std::string &work_dir, const int traj,  //all meson fields stored to disk
				 const PionMomentumPolicy &pion_mom, //object that tells us what quark momenta to use
				 Wtype &W, Vtype &V,
				 const Float &rad, //exponential wavefunction radius
				 Lattice &lattice,
				 const FieldParamType &src_setup_params = NullObject(), const std::string &mf_write_postpend = ""){
    assert(GJP.Gparity());
    Float time = -dclock();    
    std::vector<Wtype*> Wspecies(1, &W);
    std::vector<Vtype*> Vspecies(1, &V);

    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    const int nmom = pion_mom.nMom();

#ifdef ARCH_BGQ
    int init_thr = omp_get_max_threads();
    if(init_thr > 32) omp_set_num_threads(32);
#endif

    StorageType* mf_store = doCompute(pion_mom,Wspecies,Vspecies,rad,lattice,src_setup_params);

#ifdef ARCH_BGQ
    omp_set_num_threads(init_thr);
#endif

    averageAltMomenta(pion_mom,mf_store);
    writeMF(work_dir, traj, rad, pion_mom, mf_store, mf_write_postpend); //write all meson fields to disk
    
    store(mf_ll_1s_con, pion_mom, 0, mf_store);
    store(mf_ll_2s_con, pion_mom, 1, mf_store);    

    delete mf_store;
  }

};

#undef INHERIT
#undef INHERIT_FROM_BASE

CPS_END_NAMESPACE
#endif
