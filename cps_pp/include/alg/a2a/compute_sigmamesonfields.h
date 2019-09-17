#ifndef _COMPUTE_SIGMA_MESONFIELDS_H_
#define _COMPUTE_SIGMA_MESONFIELDS_H_

//Compute the meson fields for the sigma operator
#include<alg/a2a/required_momenta.h>
#include<alg/a2a/mesonfield_computemany.h>
#include<alg/a2a/inner_product.h>
#include<alg/a2a/mf_momcontainer.h>

CPS_START_NAMESPACE

//Policy for stationary sigma with and without GPBC
class StationarySigmaMomentaPolicy: public RequiredMomentum{
public:
  StationarySigmaMomentaPolicy(): RequiredMomentum() {
    const int ngp = this->nGparityDirs();
    if(ngp == 0){
      addP("(0,0,0) + (0,0,0)");
    }else if(ngp == 1){
      addPandMinusP("(-1,0,0) + (1,0,0)");
      addPandMinusP("(-3,0,0) + (3,0,0)");
    }else if(ngp == 2){
      addPandMinusP("(-1,-1,0) + (1,1,0)");
      addPandMinusP("(3,-1,0) + (-3,1,0)");
      addPandMinusP("(-1,3,0) + (1,-3,0)");
    }else if(ngp == 3){
      addPandMinusP("(-1,-1,-1) + (1,1,1)");
      addPandMinusP("(3,-1,-1) + (-3,1,1)");
      addPandMinusP("(-1,3,-1) + (1,-3,1)");
      addPandMinusP("(-1,-1,3) + (1,1,-3)");
    }else{
      ERR.General("StationarySigmaMomentaPolicy","constructor","ngp cannot be >3\n");
    }
  };
};


//Use structs to control how the meson fields are stored (either in memory or to disk)
template<typename mf_Policies>
struct WriteSigmaMesonFields{
  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> MesonFieldType;
  typedef std::vector<MesonFieldType> MesonFieldVectorType;

  const std::string work_dir;
  const std::string src;
  const int traj;
  const int rad;

  WriteSigmaMesonFields(const std::string &_work_dir, const std::string &_src, const int _traj, const int _rad): work_dir(_work_dir), src(_src), traj(_traj), rad(_rad){}
  
  void operator()(const ThreeMomentum &p_wdag, const ThreeMomentum &p_v, MesonFieldVectorType &mf_q) const{
    std::ostringstream os; //momenta in units of pi/2L
    os << work_dir << "/traj_" << traj << "_sigma_mfwv_mom" << p_wdag.file_str() << "_plus" << p_v.file_str() << "_" << src << "_rad" << rad << ".dat";

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeGetMany(1,&mf_q);
#endif

#ifndef MEMTEST_MODE
    MesonFieldType::write(os.str(),mf_q);
#endif
    for(int t=0;t<GJP.Tnodes()*GJP.TnodeSites();t++) mf_q[t].free_mem(); //no longer needed 
  }
};
template<typename mf_Policies>
struct MemStoreSigmaMesonFields{
  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> MesonFieldType;
  typedef std::vector<MesonFieldType> MesonFieldVectorType;

  MesonFieldMomentumPairContainer<mf_Policies> &storage;

  MemStoreSigmaMesonFields(MesonFieldMomentumPairContainer<mf_Policies> &_storage): storage(_storage){}
  
  void operator()(const ThreeMomentum &p_wdag, const ThreeMomentum &p_v, MesonFieldVectorType &mf_q) const{
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeGetMany(1,&mf_q);
#endif
    MesonFieldVectorType &stored = storage.moveAdd(p_wdag, p_v, mf_q);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(1,&stored);
#endif          
  }
};    




//This just computes and optionally writes the sigma meson fields		  
template<typename mf_Policies>
class ComputeSigma{
 public:
  typedef typename A2Asource<typename mf_Policies::SourcePolicies::ComplexType, typename mf_Policies::SourcePolicies::MappingPolicy, typename mf_Policies::SourcePolicies::AllocPolicy>::FieldType::InputParamType FieldParamType;

#ifdef USE_DESTRUCTIVE_FFT
  typedef A2AvectorW<mf_Policies> Wtype;
  typedef A2AvectorV<mf_Policies> Vtype;
#else
  typedef const A2AvectorW<mf_Policies> Wtype;
  typedef const A2AvectorV<mf_Policies> Vtype;
#endif

  typedef typename mf_Policies::ComplexType ComplexType;
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  typedef typename mf_Policies::SourcePolicies SourcePolicies;
  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> MesonFieldType;
  typedef std::vector<MesonFieldType> MesonFieldVectorType;

private:

  template<typename MomentumPolicy>
  static void GparitySeparateSourcesWrite(const std::string &work_dir, const int traj,
					  const MomentumPolicy &sigma_mom,
					  Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
					  const FieldParamType &src_setup_params = NullObject()){
    WriteSigmaMesonFields<mf_Policies> write_1s(work_dir, "hyd1s", traj, rad);
    WriteSigmaMesonFields<mf_Policies> write_2s(work_dir, "hyd2s", traj, rad);
    
    GparitySeparateSources(write_1s, write_2s, sigma_mom, W, V, rad, lattice, src_setup_params);
  }
  template<typename MomentumPolicy>
  static void GparitySeparateSourcesStore(MesonFieldMomentumPairContainer<mf_Policies> &store_1s,
					  MesonFieldMomentumPairContainer<mf_Policies> &store_2s,
					  const MomentumPolicy &sigma_mom,
					  Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
					  const FieldParamType &src_setup_params = NullObject()){

    MemStoreSigmaMesonFields<mf_Policies> storewrp1s(store_1s);
    MemStoreSigmaMesonFields<mf_Policies> storewrp2s(store_2s);
  
    GparitySeparateSources(storewrp1s, storewrp2s, sigma_mom, W, V, rad, lattice, src_setup_params);
  }
  

  //Multi-src multi-mom strategy consumes a lot of memory - too much for a 64-node job on Cori I. This version does the two sources separately, reducing the memory usage by a factor of 2 at the loss of computational efficiency
  template<typename MfStore, typename MomentumPolicy>
  static void GparitySeparateSources(MfStore &store_1s, MfStore &store_2s,
				     const MomentumPolicy &sigma_mom,
				     Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
				     const FieldParamType &src_setup_params = NullObject()){
#ifdef ARCH_BGQ
    int init_thr = omp_get_max_threads();
    if(init_thr > 32) omp_set_num_threads(32);
#endif
    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    std::vector<Wtype*> Wspecies(1,&W);
    std::vector<Vtype*> Vspecies(1,&V);
    
    typedef A2AflavorProjectedExpSource<SourcePolicies> ExpSrcType;
    typedef A2AflavorProjectedHydrogenSource<SourcePolicies> HydSrcType;

    int pbase[3]; //we reset the momentum for each computation so we technically don't need this - however the code demands a valid momentum
    GparityBaseMomentum(pbase,+1);

    typedef GparitySourceShiftInnerProduct<ComplexType,ExpSrcType, flavorMatrixSpinColorContract<0,true,false> > ExpInnerType;
    typedef GparityFlavorProjectedShiftSourceStorage<mf_Policies, ExpInnerType> ExpStorageType;

    typedef GparitySourceShiftInnerProduct<ComplexType,HydSrcType, flavorMatrixSpinColorContract<0,true,false> > HydInnerType;
    typedef GparityFlavorProjectedShiftSourceStorage<mf_Policies, HydInnerType> HydStorageType;

    ExpSrcType exp_src(rad,pbase,src_setup_params); //1s
    HydSrcType hyd_src(2,0,0,rad,pbase,src_setup_params); //2s

    int nsplit =  1; //further splitting over the momenta (must be a divisor of nMom which is 8 for G-parity in 3 dirs) 

#ifdef ARCH_BGQ
    if(sigma_mom.nMom() == 8) nsplit = 2;
#endif

    assert( sigma_mom.nMom() % nsplit == 0 );    
    const int pincr = sigma_mom.nMom() / nsplit;

    for(int split = 0; split < nsplit; split++){
      const int p_lo = split * pincr;
      const int p_hi = (split+1) * pincr;
      
      ExpInnerType exp_gunit_s0_inner(sigma0, exp_src);
      ExpStorageType exp_mf_store(exp_gunit_s0_inner,exp_src);

      HydInnerType hyd_gunit_s0_inner(sigma0, hyd_src);
      HydStorageType hyd_mf_store(hyd_gunit_s0_inner,hyd_src);

      for(int pidx=p_lo;pidx<p_hi;pidx++){
	ThreeMomentum p_w = sigma_mom.getWmom(pidx,false);
	ThreeMomentum p_v = sigma_mom.getVmom(pidx,false);
	exp_mf_store.addCompute(0,0, p_w,p_v);	
	hyd_mf_store.addCompute(0,0, p_w,p_v);	
      }
      if(!UniqueID()) printf("Computing sigma meson fields with 1s source for %d <= pidx < %d\n", p_lo,p_hi);
    
      ComputeMesonFields<mf_Policies,ExpStorageType>::compute(exp_mf_store,Wspecies,Vspecies,lattice
#  ifdef NODE_DISTRIBUTE_MESONFIELDS
							      ,true
#  endif
							      );

      for(int pidx=p_lo;pidx<p_hi;pidx++){
	ThreeMomentum p_wdag = -sigma_mom.getWmom(pidx,false);
	ThreeMomentum p_v = sigma_mom.getVmom(pidx,false);
	
	store_1s(p_wdag, p_v, exp_mf_store[pidx-p_lo]);
      }
  
      if(!UniqueID()) printf("Computing sigma meson fields with 2s source for %d <= pidx < %d\n", p_lo,p_hi);
    
      ComputeMesonFields<mf_Policies,HydStorageType>::compute(hyd_mf_store,Wspecies,Vspecies,lattice
#  ifdef NODE_DISTRIBUTE_MESONFIELDS
							      ,true
#  endif
							      );

      for(int pidx=p_lo;pidx<p_hi;pidx++){
	ThreeMomentum p_wdag = -sigma_mom.getWmom(pidx,false);
	ThreeMomentum p_v = sigma_mom.getVmom(pidx,false);
	
	store_2s(p_wdag, p_v, hyd_mf_store[pidx-p_lo]);
      }

      
    }//split
    
#ifdef ARCH_BGQ
    omp_set_num_threads(init_thr);
#endif
  }

  template<typename MomentumPolicy>
  static void GparityAllInOneWrite(const std::string &work_dir, const int traj,
				   const MomentumPolicy &sigma_mom,
			      Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
			      const FieldParamType &src_setup_params = NullObject()){
    WriteSigmaMesonFields<mf_Policies> write_1s(work_dir, "hyd1s", traj, rad);
    WriteSigmaMesonFields<mf_Policies> write_2s(work_dir, "hyd2s", traj, rad);
    
    GparityAllInOne(write_1s, write_2s, sigma_mom, W, V, rad, lattice, src_setup_params);
  }
  template<typename MomentumPolicy>
  static void GparityAllInOneStore(MesonFieldMomentumPairContainer<mf_Policies> &store_1s,
				   MesonFieldMomentumPairContainer<mf_Policies> &store_2s,
				   const MomentumPolicy &sigma_mom,
				   Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
				   const FieldParamType &src_setup_params = NullObject()){

    MemStoreSigmaMesonFields<mf_Policies> storewrp1s(store_1s);
    MemStoreSigmaMesonFields<mf_Policies> storewrp2s(store_2s);
  
    GparityAllInOne(storewrp1s, storewrp2s, sigma_mom, W, V, rad, lattice, src_setup_params);
  }
  
    
    

  template<typename MfStore,typename MomentumPolicy>
  static void GparityAllInOne(MfStore &store_1s, MfStore &store_2s,
			      const MomentumPolicy &sigma_mom,
			      Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
			      const FieldParamType &src_setup_params = NullObject()){

    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    std::vector<Wtype*> Wspecies(1,&W);
    std::vector<Vtype*> Vspecies(1,&V);
    
    typedef A2AflavorProjectedExpSource<SourcePolicies> ExpSrcType;
    typedef A2AflavorProjectedHydrogenSource<SourcePolicies> HydSrcType;

    int pbase[3]; //we reset the momentum for each computation so we technically don't need this - however the code demands a valid momentum
    GparityBaseMomentum(pbase,+1);
  
    typedef Elem<ExpSrcType, Elem<HydSrcType,ListEnd > > SrcList;
    typedef A2AmultiSource<SrcList> MultiSrcType;      
    //typedef SCFspinflavorInnerProduct<0,ComplexType,MultiSrcType,true,false> MultiInnerType; //unit matrix spin structure
    //typedef GparityFlavorProjectedMultiSourceStorage<mf_Policies, MultiInnerType> StorageType;

    //Allows for more memory efficient computation algorithm
    typedef GparitySourceShiftInnerProduct<ComplexType,MultiSrcType, flavorMatrixSpinColorContract<0,true,false> > MultiInnerType;
    typedef GparityFlavorProjectedShiftSourceStorage<mf_Policies, MultiInnerType> StorageType;
      
    MultiSrcType src;
    src.template getSource<0>().setup(rad,pbase,src_setup_params); //1s
    src.template getSource<1>().setup(2,0,0,rad,pbase,src_setup_params); //2s
      
    MultiInnerType gunit_s0_inner(sigma0, src);
    StorageType mf_store(gunit_s0_inner,src);

    for(int pidx=0;pidx<sigma_mom.nMom();pidx++){
      ThreeMomentum p_w = sigma_mom.getWmom(pidx,false);
      ThreeMomentum p_v = sigma_mom.getVmom(pidx,false);
      mf_store.addCompute(0,0, p_w,p_v);	
    }
    if(!UniqueID()) printf("Computing sigma meson fields with 1s/2s sources\n");

    ComputeMesonFields<mf_Policies,StorageType>::compute(mf_store,Wspecies,Vspecies,lattice
#  ifdef NODE_DISTRIBUTE_MESONFIELDS
							 ,true
#  endif
							 );
      
    std::string src_names[2] = {"1s","2s"};
    if(!UniqueID()) printf("Writing sigma meson fields to disk\n");
    for(int pidx=0;pidx<sigma_mom.nMom();pidx++){
      ThreeMomentum p_wdag = -sigma_mom.getWmom(pidx,false);
      ThreeMomentum p_v = sigma_mom.getVmom(pidx,false);
	
      for(int s=0;s<2;s++){
	MesonFieldVectorType &mf_q = mf_store(s,pidx);
	if(s == 0) store_1s(p_wdag, p_v, mf_q);
	else store_2s(p_wdag, p_v, mf_q);
      }
    } 
  }
  
  template<typename MomentumPolicy>
  static void noGparityWrite(const std::string &work_dir, const int traj,
			     const MomentumPolicy &sigma_mom,
			     Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
			     const FieldParamType &src_setup_params = NullObject()){
    WriteSigmaMesonFields<mf_Policies> write_1s(work_dir, "hyd1s", traj, rad);
    
    noGparity(write_1s, sigma_mom, W, V, rad, lattice, src_setup_params);
  }
  template<typename MomentumPolicy>
  static void noGparityStore(MesonFieldMomentumPairContainer<mf_Policies> &store_1s,
			     MesonFieldMomentumPairContainer<mf_Policies> &store_2s,
			     const MomentumPolicy &sigma_mom,
			     Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
			     const FieldParamType &src_setup_params = NullObject()){
    
    MemStoreSigmaMesonFields<mf_Policies> storewrp1s(store_1s);
  
    noGparity(storewrp1s, sigma_mom, W, V, rad, lattice, src_setup_params);
  }
  
  template<typename MfStore, typename MomentumPolicy>
  static void noGparity(MfStore &store_1s,
			const MomentumPolicy &sigma_mom,
			Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
			const FieldParamType &src_setup_params = NullObject()){

    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    std::vector<Wtype*> Wspecies(1,&W);
    std::vector<Vtype*> Vspecies(1,&V);

    typedef A2AexpSource<SourcePolicies> SrcType;
    typedef SCspinInnerProduct<0,ComplexType,SrcType> InnerType;
    typedef BasicSourceStorage<mf_Policies,InnerType> StorageType;
      
    SrcType src(rad,src_setup_params);
    InnerType gunit_inner(src);

    StorageType mf_store(gunit_inner);

    for(int pidx=0;pidx<sigma_mom.nMom();pidx++){
      ThreeMomentum p_w = sigma_mom.getWmom(pidx,false);
      ThreeMomentum p_v = sigma_mom.getVmom(pidx,false);
      mf_store.addCompute(0,0, p_w,p_v);	
    }
    ComputeMesonFields<mf_Policies,StorageType>::compute(mf_store,Wspecies,Vspecies,lattice);
      
    for(int pidx=0;pidx<sigma_mom.nMom();pidx++){
      ThreeMomentum p_wdag = -sigma_mom.getWmom(pidx,false);
      ThreeMomentum p_v = sigma_mom.getVmom(pidx,false);
      store_1s(p_wdag, p_v, mf_store[pidx]);
    }
  }
public:
  
  //Computes sigma meson fields and saves to disk
  template<typename MomentumPolicy>
  static void computeAndWrite(const std::string &work_dir, const int traj,
			      const MomentumPolicy &sigma_mom,
			      Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
			      const FieldParamType &src_setup_params = NullObject()){
    if(GJP.Gparity()){
#ifdef SIGMA_DO_SOURCES_SEPARATELY
      GparitySeparateSourcesWrite(work_dir,traj,sigma_mom,W,V,rad,lattice,src_setup_params);
#else
      GparityAllInOneWrite(work_dir,traj,sigma_mom,W,V,rad,lattice,src_setup_params);
#endif
    }else{
      noGparityWrite(work_dir,traj,sigma_mom,W,V,rad,lattice,src_setup_params);
    }
  }
  template<typename MomentumPolicy>
  static void computeGparityMesonFields(MesonFieldMomentumPairContainer<mf_Policies> &store_1s,
					MesonFieldMomentumPairContainer<mf_Policies> &store_2s,
					const MomentumPolicy &sigma_mom,
					Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
					const FieldParamType &src_setup_params = NullObject()){
    if(!GJP.Gparity()) ERR.General("ComputeSigma","computeGparityMesonFields","G-parity BCs not active!");
      
#ifdef SIGMA_DO_SOURCES_SEPARATELY
    GparitySeparateSourcesStore(store_1s,store_2s,sigma_mom,W,V,rad,lattice,src_setup_params);
#else
    GparityAllInOneStore(store_1s,store_2s,sigma_mom,W,V,rad,lattice,src_setup_params);
#endif
  }
  
};



template<typename mf_Policies>
struct computeSigmaMesonFieldsBase{
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
  INHERIT(FieldParamType, computeSigmaMesonFieldsBase<mf_Policies>); 	\
  INHERIT(Wtype, computeSigmaMesonFieldsBase<mf_Policies>); 	\
  INHERIT(Vtype, computeSigmaMesonFieldsBase<mf_Policies>); 	\
  INHERIT(MesonFieldType, computeSigmaMesonFieldsBase<mf_Policies>); 	\
  INHERIT(MesonFieldVectorType, computeSigmaMesonFieldsBase<mf_Policies>); 	\
  INHERIT(ComplexType, computeSigmaMesonFieldsBase<mf_Policies>); 	\
  INHERIT(SourcePolicies, computeSigmaMesonFieldsBase<mf_Policies>)

};

//Note this differs from the computation of the pion meson fields for 2 reasons: the spin/flavor structure and the fact that the resulting meason fields are placed in MesonFieldMomentumPairContainer that keys the meson fields on the combination of both quark momenta rather than the total momentum.
template<typename mf_Policies, typename SigmaMomentumPolicy>
class computeSigmaMesonFields1s{
public:
  INHERIT_FROM_BASE;

  typedef A2AflavorProjectedExpSource<SourcePolicies> ExpSrcType;
  typedef GparitySourceShiftInnerProduct<ComplexType,ExpSrcType, flavorMatrixSpinColorContract<0,true,false> > InnerType;
  typedef GparityFlavorProjectedShiftSourceStorage<mf_Policies, InnerType> StorageType;
   
public:

  struct Options{
    int thr_internal; //number of threads used in the computation
    int mom_block_size; //number of meson momenta fed into the computeMany algorithm
    int nshift_combine_max; //max number of source shifts shiftSourceStorage is allowed to combine
    Options(): thr_internal(-1), mom_block_size(-1), nshift_combine_max(-1){}
  };

  static void computeMesonFields(MesonFieldMomentumPairContainer<mf_Policies> &mf_con, 
				 const SigmaMomentumPolicy &sigma_mom, //object that tells us what quark momenta to use
				 Wtype &W, Vtype &V,
				 const Float rad_1s, //exponential wavefunction radius
				 Lattice &lattice,
				 const FieldParamType &src_setup_params = NullObject(),
				 const Options &opt = Options()){
    assert(GJP.Gparity());
    Float time = -dclock();    
    std::vector<Wtype*> Wspecies(1, &W);
    std::vector<Vtype*> Vspecies(1, &V);

    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    const int nmom = sigma_mom.nMom();

    int init_thr = omp_get_max_threads();
    if(opt.thr_internal != -1) omp_set_num_threads(opt.thr_internal);

    int pbase[3]; //we reset the momentum for each computation so we technically don't need this - however the code demands a valid momentum
    GparityBaseMomentum(pbase,+1);
        
    ExpSrcType src(rad_1s,pbase,src_setup_params);
    InnerType inner(sigma0, src);    
    StorageType mf_store(inner,src, opt.nshift_combine_max);

    int nmom_block = opt.mom_block_size != -1 ? opt.mom_block_size : nmom;

    if(!UniqueID()) printf("Computing %d momenta with block sizes of %d\n",nmom,nmom_block);

    for(int b=0; b<nmom; b+= nmom_block){
      int nmom_rem = nmom - b;
      int nmom_block_actual = nmom_rem < nmom_block ? nmom_rem : nmom_block;

      if(!UniqueID()) printf("Doing block %d->%d\n",b,b+nmom_rem);

    
      for(int pidx=b;pidx<b+nmom_block_actual;pidx++){      
	ThreeMomentum p_w = sigma_mom.getWmom(pidx);
	ThreeMomentum p_v = sigma_mom.getVmom(pidx);
	mf_store.addCompute(0,0, p_w,p_v);
      }

      ComputeMesonFields<mf_Policies,StorageType>::compute(mf_store,Wspecies,Vspecies,lattice
#ifdef NODE_DISTRIBUTE_MESONFIELDS
							   ,true
#endif
							   );
      for(int pidx=b;pidx<b+nmom_block_actual;pidx++)
	mf_con.moveAdd(sigma_mom.getWdagMom(pidx), sigma_mom.getVmom(pidx), mf_store[pidx-b]);
    }

    if(opt.thr_internal != -1) omp_set_num_threads(init_thr);
  }

  static void write(MesonFieldMomentumPairContainer<mf_Policies> &mf_con, const SigmaMomentumPolicy &sigma_mom, const std::string &work_dir, const int traj,const Float &rad){
    for(int i=0;i<sigma_mom.nMom();i++){
      ThreeMomentum p_wdag = sigma_mom.getWdagMom(i);
      ThreeMomentum p_v = sigma_mom.getVmom(i);

      std::ostringstream os; //momenta in units of pi/2L
      os << work_dir << "/traj_" << traj << "_sigma_mfwv_mom" << p_wdag.file_str() << "_plus" << p_v.file_str() << "_hyd1s_rad" << rad << ".dat";

      MesonFieldVectorType &mf_q = mf_con.get(p_wdag,p_v);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
      nodeGetMany(1,&mf_q);
#endif

#ifndef MEMTEST_MODE
      MesonFieldType::write(os.str(),mf_q);
#endif

#ifdef NODE_DISTRIBUTE_MESONFIELDS
      nodeDistributeMany(1,&mf_q);
#endif
    }
  }

  static void read(MesonFieldMomentumPairContainer<mf_Policies> &mf_con, const SigmaMomentumPolicy &sigma_mom, const std::string &work_dir, const int traj,const Float &rad){
    for(int i=0;i<sigma_mom.nMom();i++){
      ThreeMomentum p_wdag = sigma_mom.getWdagMom(i);
      ThreeMomentum p_v = sigma_mom.getVmom(i);

      std::ostringstream os; //momenta in units of pi/2L
      os << work_dir << "/traj_" << traj << "_sigma_mfwv_mom" << p_wdag.file_str() << "_plus" << p_v.file_str() << "_hyd1s_rad" << rad << ".dat";
      
      MesonFieldVectorType mf_q;
#ifndef MEMTEST_MODE
      MesonFieldType::read(os.str(),mf_q);
#endif
#ifdef NODE_DISTRIBUTE_MESONFIELDS
      nodeDistributeMany(1,&mf_q);
#endif
      mf_con.moveAdd(p_wdag, p_v, mf_q);
    }
  }

};

CPS_END_NAMESPACE

#endif
