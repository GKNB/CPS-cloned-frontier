#ifndef _COMPUTE_PION_H
#define _COMPUTE_PION_H

#include<alg/a2a/required_momenta.h>
#include<alg/a2a/mesonfield_computemany.h>
#include<alg/a2a/inner_product.h>
#include<alg/a2a/mf_momcontainer.h>

CPS_START_NAMESPACE

//Policy for RequiredMomenta. This is the set of momenta that Daiqian used.
class StandardPionMomentaPolicy{
public:
  void setupMomenta(const int &ngp){
    RequiredMomentum<StandardPionMomentaPolicy> *tt = static_cast<RequiredMomentum<StandardPionMomentaPolicy>*>(this);
    if(ngp == 0){
      //p_pi = (0,0,0)
      tt->addP("(0,0,0) + (0,0,0)",false);
    }else if(ngp == 1){
      //p_pi = (-2,0,0)     (units of pi/2L)    
      tt->addPandMinusP("(-1,0,0) + (-1,0,0)",false); tt->addPandMinusP("(1,0,0) + (-3,0,0)",true); //alternative momentum   
      //(In case you're wondering why my first momentum has the opposite sign to Daiqian's, its because mine is for W^dagger, not W)
    }else if(ngp == 2){
      //Along G-parity direction:
      //p_pi = (-2,-2,0)     (units of pi/2L)  
      tt->addPandMinusP("(-1,-1,0) + (-1,-1,0)",false); tt->addPandMinusP("(1,1,0) + (-3,-3,0)",true);

      //Along off-diagonal direction:      
      //p_pi = (2,-2,0)
      tt->addPandMinusP("(-1,-1,0) + (3,-1,0)",false); tt->addPandMinusP("(1,1,0) + (1,-3,0)",true);
    }else if(ngp == 3){
      //p_pi = (-2,-2,-2)     (units of pi/2L)  
      tt->addPandMinusP("(-1,-1,-1) + (-1,-1,-1)",false); tt->addPandMinusP("(1,1,1) + (-3,-3,-3)",true);

      //p_pi = (2,-2,-2)
      tt->addPandMinusP("(-1,-1,-1) + (3,-1,-1)",false); tt->addPandMinusP("(1,1,1) + (1,-3,-3)",true);

      //p_pi = (-2,2,-2)
      tt->addPandMinusP("(-1,-1,-1) + (-1,3,-1)",false); tt->addPandMinusP("(1,1,1) + (-3,1,-3)",true);

      //p_pi = (-2,-2,2)
      tt->addPandMinusP("(-1,-1,-1) + (-1,-1,3)",false); tt->addPandMinusP("(1,1,1) + (-3,-3,1)",true);      
    }else{
      ERR.General("StandardPionMomentaPolicy","setupMomenta","ngp cannot be >3\n");
    }
  }
};

//This set of momenta does not include the second momentum combination with which we average to reduce the G-parity rotational symmetry breaking
class H4asymmetricMomentaPolicy{
public:
  void setupMomenta(const int &ngp){
    RequiredMomentum<H4asymmetricMomentaPolicy> *tt = static_cast<RequiredMomentum<H4asymmetricMomentaPolicy>*>(this);
    if(ngp == 0){
      //p_pi = (0,0,0)
      tt->addP("(0,0,0) + (0,0,0)",false);
    }else if(ngp == 1){
      //p_pi = (-2,0,0)     (units of pi/2L)    
      tt->addPandMinusP("(-1,0,0) + (-1,0,0)",false);
      //(In case you're wondering why my first momentum has the opposite sign to Daiqian's, its because mine is for W^dagger, not W)
    }else if(ngp == 2){
      //Along G-parity direction:
      //p_pi = (-2,-2,0)     (units of pi/2L)  
      tt->addPandMinusP("(-1,-1,0) + (-1,-1,0)",false);

      //Along off-diagonal direction:      
      //p_pi = (2,-2,0)
      tt->addPandMinusP("(-1,-1,0) + (3,-1,0)",false);
    }else if(ngp == 3){
      //p_pi = (-2,-2,-2)     (units of pi/2L)  
      tt->addPandMinusP("(-1,-1,-1) + (-1,-1,-1)",false);

      //p_pi = (2,-2,-2)
      tt->addPandMinusP("(-1,-1,-1) + (3,-1,-1)",false);

      //p_pi = (-2,2,-2)
      tt->addPandMinusP("(-1,-1,-1) + (-1,3,-1)",false);

      //p_pi = (-2,-2,2)
      tt->addPandMinusP("(-1,-1,-1) + (-1,-1,3)",false);
    }else{
      ERR.General("H4asymmetricMomentaPolicy","setupMomenta","ngp cannot be >3\n");
    }
  }
};









template<typename mf_Policies>
class ComputePion{
public:
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


private:

  struct GparityComputeTypes{
    typedef A2AflavorProjectedExpSource<SourcePolicies> ExpSrcType;
    typedef A2AflavorProjectedHydrogenSource<SourcePolicies> HydSrcType;
    typedef Elem<ExpSrcType, Elem<HydSrcType,ListEnd > > SrcList;
    typedef A2AmultiSource<SrcList> MultiSrcType;
    //typedef SCFspinflavorInnerProduct<15,ComplexType,MultiSrcType,true,false> MultiInnerType;
    //typedef GparityFlavorProjectedMultiSourceStorage<mf_Policies, MultiInnerType> StorageType;
    
    //Allows for more memory efficient computation algorithm
    typedef GparitySourceShiftInnerProduct<ComplexType,MultiSrcType, flavorMatrixSpinColorContract<15,ComplexType,true,false> > MultiInnerType;
    typedef GparityFlavorProjectedShiftSourceStorage<mf_Policies, MultiInnerType> StorageType;
  };

  template<typename PionMomentumPolicy>
  static typename GparityComputeTypes::StorageType* GparityDoCompute(const RequiredMomentum<PionMomentumPolicy> &pion_mom,
							      const std::vector<Wtype*> &Wspecies, const std::vector<Vtype*> &Vspecies,
							      const Float &rad, Lattice &lattice, const FieldParamType &src_setup_params){
    const int nmom = pion_mom.nMom();
    int pbase[3]; //we reset the momentum for each computation so we technically don't need this - however the code demands a valid momentum
    GparityBaseMomentum(pbase,+1);
      
    typename GparityComputeTypes::MultiSrcType src;
    src.template getSource<0>().setup(rad,pbase,src_setup_params); //1s
    src.template getSource<1>().setup(2,0,0,rad,pbase,src_setup_params); //2s
    
    typename GparityComputeTypes::MultiInnerType g5_s3_inner(sigma3, src);
    
    typename GparityComputeTypes::StorageType* mf_store = new typename GparityComputeTypes::StorageType(g5_s3_inner,src);
    //Base momenta
    for(int pidx=0;pidx<nmom;pidx++){
      ThreeMomentum p_w = pion_mom.getWmom(pidx,false);
      ThreeMomentum p_v = pion_mom.getVmom(pidx,false);
      mf_store->addCompute(0,0, p_w,p_v);	
    }
    //Alt momenta
    for(int pidx=0;pidx<nmom;pidx++){
      ThreeMomentum p_w = pion_mom.getWmom(pidx,true);
      ThreeMomentum p_v = pion_mom.getVmom(pidx,true);
      mf_store->addCompute(0,0, p_w,p_v);	
    }

    ComputeMesonFields<mf_Policies,typename GparityComputeTypes::StorageType>::compute(*mf_store,Wspecies,Vspecies,lattice
#ifdef NODE_DISTRIBUTE_MESONFIELDS
							 ,true
#endif
							 );
    
    return mf_store;
  }

  //Replace MF with pidx < nmom  with avg( Mf[pidx],  Mf[pidx + nmom] ), i.e. combine with alternate momentum configuration
  template<typename PionMomentumPolicy>
  static void GparityAverageAltMomenta(const RequiredMomentum<PionMomentumPolicy> &pion_mom,
				       typename GparityComputeTypes::StorageType* mf_store){

    printMemNodeFile("GparityAverageAltMomenta");

    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    const int nmom = pion_mom.nMom();

    for(int pidx=0;pidx<nmom;pidx++){
      for(int src_idx=0; src_idx<2; src_idx++){
	const ThreeMomentum pi_mom_pidx = pion_mom.getMesonMomentum(pidx);
	MesonFieldVectorType & mf_pbase = (*mf_store)(src_idx,pidx);
	MesonFieldVectorType & mf_palt = (*mf_store)(src_idx,pidx+nmom);

#ifdef NODE_DISTRIBUTE_MESONFIELDS
	nodeGetMany(1,&mf_pbase);
	nodeGetMany(1,&mf_palt);
#endif
	for(int t=0;t<Lt;t++){
	  mf_pbase[t].average(mf_palt[t]);
	  mf_palt[t].free_mem();
	}
      
#ifdef NODE_DISTRIBUTE_MESONFIELDS
	nodeDistributeMany(1,&mf_pbase);
#endif
      }
    }
  }

  //Writes meson fields with pidx < nMom   (i.e. after alt_mom have been averaged)
  template<typename PionMomentumPolicy>
  static void GparityWriteMF(const std::string &work_dir, const int traj, const Float &rad,
				    const RequiredMomentum<PionMomentumPolicy> &pion_mom,
				    typename GparityComputeTypes::StorageType* mf_store){
    printMemNodeFile("GparityWriteMF");

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
	os << work_dir << "/traj_" << traj << "_pion_mf_mom" << pi_mom_pidx.file_str() << "_hyd" << src_names[s] << "_rad" << rad << ".dat";

#ifndef MEMTEST_MODE	
	MesonFieldType::write(os.str(),mf_base);
#endif

#ifdef NODE_DISTRIBUTE_MESONFIELDS
	nodeDistributeMany(1,&mf_base);
#endif
	
      }
    }
  }

  template<typename PionMomentumPolicy>
  static void GparityStore(MesonFieldMomentumContainer<mf_Policies> &mf_ll_con,
			   const RequiredMomentum<PionMomentumPolicy> &pion_mom,
			   const int src_idx,
			   typename GparityComputeTypes::StorageType* mf_store){

    printMemNodeFile("GparityStore src_idx="+anyToStr(src_idx));

    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    const int nmom = pion_mom.nMom();
			   
    for(int pidx=0;pidx<nmom;pidx++){
      const ThreeMomentum pi_mom_pidx = pion_mom.getMesonMomentum(pidx);
      MesonFieldVectorType & mf_pbase = (*mf_store)(src_idx,pidx);
      MesonFieldVectorType & stored = mf_ll_con.moveAdd(pi_mom_pidx, mf_pbase);
    }
  }

public:
  
  //These meson fields are also used by the pi-pi and K->pipi calculations
  template<typename PionMomentumPolicy>
  static void computeMesonFields(MesonFieldMomentumContainer<mf_Policies> &mf_ll_con, //container for 1s pion output fields, accessible by ThreeMomentum of pion
				 const std::string &work_dir, const int traj,  //all meson fields stored to disk
				 const RequiredMomentum<PionMomentumPolicy> &pion_mom, //object that tells us what quark momenta to use
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
    if(pion_mom.nAltMom() > 0 && pion_mom.nAltMom() != nmom)
      ERR.General("ComputePion","computeMesonFields","If alternate momentum combinations are specified there must be one for each pion momentum!\n");
    

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

  //Same as above but gives output containers for both G-parity sources
  template<typename PionMomentumPolicy>
  static void computeGparityMesonFields(MesonFieldMomentumContainer<mf_Policies> &mf_ll_1s_con, //container for 1s pion output fields, accessible by ThreeMomentum of pion
					MesonFieldMomentumContainer<mf_Policies> &mf_ll_2s_con, //same for 2s source
					const std::string &work_dir, const int traj,  //all meson fields stored to disk
					const RequiredMomentum<PionMomentumPolicy> &pion_mom, //object that tells us what quark momenta to use
					Wtype &W, Vtype &V,
					const Float &rad, //exponential wavefunction radius
					Lattice &lattice,			      
					const FieldParamType &src_setup_params = NullObject()){
    assert(GJP.Gparity());
    Float time = -dclock();    
    std::vector<Wtype*> Wspecies(1, &W);
    std::vector<Vtype*> Vspecies(1, &V);

    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    const int nmom = pion_mom.nMom();
    if(pion_mom.nAltMom() > 0 && pion_mom.nAltMom() != nmom)
      ERR.General("ComputePion","computeGparityMesonFields","If alternate momentum combinations are specified there must be one for each pion momentum!\n");

#ifdef ARCH_BGQ
    int init_thr = omp_get_max_threads();
    if(init_thr > 32) omp_set_num_threads(32);
#endif

    typename GparityComputeTypes::StorageType* mf_store = GparityDoCompute(pion_mom,Wspecies,Vspecies,rad,lattice,src_setup_params);

#ifdef ARCH_BGQ
    omp_set_num_threads(init_thr);
#endif

    GparityAverageAltMomenta(pion_mom,mf_store);
    GparityWriteMF(work_dir, traj, rad, pion_mom, mf_store); //write all meson fields to disk
    
    GparityStore(mf_ll_1s_con, pion_mom, 0, mf_store);
    GparityStore(mf_ll_2s_con, pion_mom, 1, mf_store);    

    delete mf_store;
  }
    
  //Compute the two-point function using the pre-generated meson fields.
  //The pion is given the momentum associated with index 'pidx' in the RequiredMomentum
  //result is indexed by (tsrc, tsep)  where tsep is the source-sink separation
  template<typename PionMomentumPolicy>
  static void compute(fMatrix<typename mf_Policies::ScalarComplexType> &into, MesonFieldMomentumContainer<mf_Policies> &mf_ll_con, const RequiredMomentum<PionMomentumPolicy> &pion_mom, const int pidx){
    typedef typename mf_Policies::ComplexType ComplexType;
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt,Lt);

    ThreeMomentum p_pi_src = pion_mom.getMesonMomentum(pidx); //exp(-ipx)
    ThreeMomentum p_pi_snk = -p_pi_src; //exp(+ipx)

    assert(mf_ll_con.contains(p_pi_src));
    assert(mf_ll_con.contains(p_pi_snk));
	   
    //Construct the meson fields
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_ll_src = mf_ll_con.get(p_pi_src);
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_ll_snk = mf_ll_con.get(p_pi_snk);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    if(!UniqueID()){ printf("Gathering meson fields\n");  fflush(stdout); }
    nodeGetMany(2,&mf_ll_src,&mf_ll_snk);
    sync();
#endif

    //Compute the two-point function
    // \sum_{xsnk,xsrc} exp(ip xsnk)exp(-ipxsrc) tr( S G(xsnk, tsnk; xsrc, tsrc) S G(xsrc,tsrc; xsnk, tsnk) ) 
    //where S is the vertex spin/color/flavor structure

    //= tr( [[\sum_{xsnk} exp(ip xsnk) w^dag(xsnk,tsnk) S v(xsnk,tsnk)]] [[\sum_{xsrc} exp(-ip xsrc) w^dag(xsrc,tsrc) S v(xsrc,tsrc) ]] ) 
    if(!UniqueID()){ printf("Starting trace\n");  fflush(stdout); }
    trace(into,mf_ll_snk,mf_ll_src);
    into *= ScalarComplexType(0.5,0);
    rearrangeTsrcTsep(into); //rearrange temporal ordering
    
    sync();
    if(!UniqueID()){ printf("Finished trace\n");  fflush(stdout); }

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(2,&mf_ll_src,&mf_ll_snk);
#endif
  }

};



CPS_END_NAMESPACE

#endif
