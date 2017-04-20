#ifndef _COMPUTE_SIGMA_H
#define _COMPUTE_SIGMA_H

#include<alg/a2a/required_momenta.h>
#include<alg/a2a/mesonfield_computemany.h>
#include<alg/a2a/inner_product.h>
#include<alg/a2a/mf_momcontainer.h>

//Compute stationary sigma meson two-point function with and without GPBC
CPS_START_NAMESPACE

//Policy for stationary sigma with and without GPBC
class StationarySigmaMomentaPolicy: public RequiredMomentum{
public:
  StationarySigmaMomentaPolicy(): RequiredMomentum() {
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

  //Multi-src multi-mom strategy consumes a lot of memory - too much for a 64-node job on Cori I. This version does the two sources separately, reducing the memory usage by a factor of 2 at the loss of computational efficiency
  static void GparitySeparateSources(const std::string &work_dir, const int traj,
				     Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
				     const FieldParamType &src_setup_params = NullObject()){
#ifdef ARCH_BGQ
    int init_thr = omp_get_max_threads();
    if(init_thr > 32) omp_set_num_threads(32);
#endif
    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    StationarySigmaMomentaPolicy momenta;

    std::vector<Wtype*> Wspecies(1,&W);
    std::vector<Vtype*> Vspecies(1,&V);
    
    typedef A2AflavorProjectedExpSource<SourcePolicies> ExpSrcType;
    typedef A2AflavorProjectedHydrogenSource<SourcePolicies> HydSrcType;

    int pbase[3]; //we reset the momentum for each computation so we technically don't need this - however the code demands a valid momentum
    GparityBaseMomentum(pbase,+1);

    typedef GparitySourceShiftInnerProduct<ComplexType,ExpSrcType, flavorMatrixSpinColorContract<0,ComplexType,true,false> > ExpInnerType;
    typedef GparityFlavorProjectedShiftSourceStorage<mf_Policies, ExpInnerType> ExpStorageType;

    typedef GparitySourceShiftInnerProduct<ComplexType,HydSrcType, flavorMatrixSpinColorContract<0,ComplexType,true,false> > HydInnerType;
    typedef GparityFlavorProjectedShiftSourceStorage<mf_Policies, HydInnerType> HydStorageType;

    ExpSrcType exp_src(rad,pbase,src_setup_params); //1s
    HydSrcType hyd_src(2,0,0,rad,pbase,src_setup_params); //2s

    int nsplit =  1; //further splitting over the momenta (must be a divisor of nMom which is 8 for G-parity in 3 dirs) 

#ifdef ARCH_BGQ
    if(momenta.nMom() == 8) nsplit = 2;
#endif

    assert( momenta.nMom() % nsplit == 0 );    
    const int pincr = momenta.nMom() / nsplit;

    for(int split = 0; split < nsplit; split++){
      const int p_lo = split * pincr;
      const int p_hi = (split+1) * pincr;
      
      ExpInnerType exp_gunit_s0_inner(sigma0, exp_src);
      ExpStorageType exp_mf_store(exp_gunit_s0_inner,exp_src);

      HydInnerType hyd_gunit_s0_inner(sigma0, hyd_src);
      HydStorageType hyd_mf_store(hyd_gunit_s0_inner,hyd_src);

      for(int pidx=p_lo;pidx<p_hi;pidx++){
	ThreeMomentum p_w = momenta.getWmom(pidx,false);
	ThreeMomentum p_v = momenta.getVmom(pidx,false);
	exp_mf_store.addCompute(0,0, p_w,p_v);	
	hyd_mf_store.addCompute(0,0, p_w,p_v);	
      }
      if(!UniqueID()) printf("Computing sigma meson fields with 1s source for %d <= pidx < %d\n", p_lo,p_hi);
    
      ComputeMesonFields<mf_Policies,ExpStorageType>::compute(exp_mf_store,Wspecies,Vspecies,lattice
#  ifdef NODE_DISTRIBUTE_MESONFIELDS
							      ,true
#  endif
							      );

      if(!UniqueID()) printf("Writing 1s sigma meson fields to disk for %d <= pidx < %d\n", p_lo,p_hi);
      for(int pidx=p_lo;pidx<p_hi;pidx++){
	ThreeMomentum p_wdag = -momenta.getWmom(pidx,false);
	ThreeMomentum p_v = momenta.getVmom(pidx,false);
	
	std::ostringstream os; //momenta in units of pi/2L
	os << work_dir << "/traj_" << traj << "_sigma_mfwv_mom" << p_wdag.file_str() << "_plus" << p_v.file_str() << "_hyd1s_rad" << rad << ".dat";
	MesonFieldVectorType &mf_q = exp_mf_store[pidx-p_lo];

#ifdef NODE_DISTRIBUTE_MESONFIELDS
	nodeGetMany(1,&mf_q);
#endif

#ifndef MEMTEST_MODE
	MesonFieldType::write(os.str(),mf_q);
#endif
	for(int t=0;t<Lt;t++) mf_q[t].free_mem(); //no longer needed      
      }
  
      if(!UniqueID()) printf("Computing sigma meson fields with 2s source for %d <= pidx < %d\n", p_lo,p_hi);
    
      ComputeMesonFields<mf_Policies,HydStorageType>::compute(hyd_mf_store,Wspecies,Vspecies,lattice
#  ifdef NODE_DISTRIBUTE_MESONFIELDS
							      ,true
#  endif
							      );

      if(!UniqueID()) printf("Writing 2s sigma meson fields to disk for %d <= pidx < %d\n", p_lo,p_hi);
      for(int pidx=p_lo;pidx<p_hi;pidx++){
	ThreeMomentum p_wdag = -momenta.getWmom(pidx,false);
	ThreeMomentum p_v = momenta.getVmom(pidx,false);
	
	std::ostringstream os; //momenta in units of pi/2L
	os << work_dir << "/traj_" << traj << "_sigma_mfwv_mom" << p_wdag.file_str() << "_plus" << p_v.file_str() << "_hyd2s_rad" << rad << ".dat";
	MesonFieldVectorType &mf_q = hyd_mf_store[pidx-p_lo];

#ifdef NODE_DISTRIBUTE_MESONFIELDS
	nodeGetMany(1,&mf_q);
#endif

#ifndef MEMTEST_MODE
	MesonFieldType::write(os.str(),mf_q);
#endif
	for(int t=0;t<Lt;t++) mf_q[t].free_mem(); //no longer needed      
      }

      
    }//split
    
#ifdef ARCH_BGQ
    omp_set_num_threads(init_thr);
#endif
  }


  
  static void GparityAllInOne(const std::string &work_dir, const int traj,
			      Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
			      const FieldParamType &src_setup_params = NullObject()){

    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    StationarySigmaMomentaPolicy momenta;

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
    typedef GparitySourceShiftInnerProduct<ComplexType,MultiSrcType, flavorMatrixSpinColorContract<0,ComplexType,true,false> > MultiInnerType;
    typedef GparityFlavorProjectedShiftSourceStorage<mf_Policies, MultiInnerType> StorageType;
      
    MultiSrcType src;
    src.template getSource<0>().setup(rad,pbase,src_setup_params); //1s
    src.template getSource<1>().setup(2,0,0,rad,pbase,src_setup_params); //2s
      
    MultiInnerType gunit_s0_inner(sigma0, src);
    StorageType mf_store(gunit_s0_inner,src);

    for(int pidx=0;pidx<momenta.nMom();pidx++){
      ThreeMomentum p_w = momenta.getWmom(pidx,false);
      ThreeMomentum p_v = momenta.getVmom(pidx,false);
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
    for(int pidx=0;pidx<momenta.nMom();pidx++){
      ThreeMomentum p_wdag = -momenta.getWmom(pidx,false);
      ThreeMomentum p_v = momenta.getVmom(pidx,false);
	
      for(int s=0;s<2;s++){
	std::ostringstream os; //momenta in units of pi/2L
	os << work_dir << "/traj_" << traj << "_sigma_mfwv_mom" << p_wdag.file_str() << "_plus" << p_v.file_str() << "_hyd" << src_names[s] << "_rad" << rad << ".dat";
	MesonFieldVectorType &mf_q = mf_store(s,pidx);

#ifdef NODE_DISTRIBUTE_MESONFIELDS
	nodeGetMany(1,&mf_q);
#endif

#ifndef MEMTEST_MODE
	MesonFieldType::write(os.str(),mf_q);
#endif
	for(int t=0;t<Lt;t++) mf_q[t].free_mem(); //no longer needed
      }
    } 
  }


  static void noGparity(const std::string &work_dir, const int traj,
			Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
			const FieldParamType &src_setup_params = NullObject()){

    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    StationarySigmaMomentaPolicy momenta;

    std::vector<Wtype*> Wspecies(1,&W);
    std::vector<Vtype*> Vspecies(1,&V);

    typedef A2AexpSource<SourcePolicies> SrcType;
    typedef SCspinInnerProduct<0,ComplexType,SrcType> InnerType;
    typedef BasicSourceStorage<mf_Policies,InnerType> StorageType;
      
    SrcType src(rad,src_setup_params);
    InnerType gunit_inner(src);

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
public:
  
  //Computes sigma meson fields and saves to disk
  static void computeAndWrite(const std::string &work_dir, const int traj,
			      Wtype &W, Vtype &V, const Float &rad, Lattice &lattice,
			      const FieldParamType &src_setup_params = NullObject()){
    if(GJP.Gparity()){
#ifdef SIGMA_DO_SOURCES_SEPARATELY
      GparitySeparateSources(work_dir,traj,W,V,rad,lattice,src_setup_params);
#else
      GparityAllInOne(work_dir,traj,W,V,rad,lattice,src_setup_params);
#endif
    }else{
      noGparity(work_dir,traj,W,V,rad,lattice,src_setup_params);
    }
  }

};

CPS_END_NAMESPACE

#endif

