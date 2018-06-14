#ifndef _COMPUTE_LL_MESONFIELDS_H_
#define _COMPUTE_LL_MESONFIELDS_H_

//Compute the light-light meson fields used in the main job

#include<alg/a2a/required_momenta.h>
#include<alg/a2a/mesonfield_computemany.h>
#include<alg/a2a/inner_product.h>
#include<alg/a2a/mf_momcontainer.h>

CPS_START_NAMESPACE


//Policy for RequiredMomenta. This is the set of momenta that Daiqian used.
class StandardPionMomentaPolicy: public RequiredMomentum{
public:
  StandardPionMomentaPolicy(): RequiredMomentum() {
    this->combineSameTotalMomentum(true); //momentum pairs with same total momentum will be added to same entry and treated as 'alternates' which we average together below

    const int ngp = this->nGparityDirs();
    if(ngp == 0){
      //p_pi = (0,0,0)
      addP("(0,0,0) + (0,0,0)");
      //one unit of momenta in units of 2pi/L
      addPandMinusP("(1,0,0) + (0,0,0)");
      addPandMinusP("(0,1,0) + (0,0,0)");
      addPandMinusP("(0,0,1) + (0,0,0)");
    }else if(ngp == 1){
      //p_pi = (-2,0,0)     (units of pi/2L)    
      addPandMinusP("(-1,0,0) + (-1,0,0)"); addPandMinusP("(1,0,0) + (-3,0,0)"); //alternative momentum   
      //(In case you're wondering why my first momentum has the opposite sign to Daiqian's, its because mine is for W^dagger, not W)
    }else if(ngp == 2){
      //Along G-parity direction:
      //p_pi = (-2,-2,0)     (units of pi/2L)  
      addPandMinusP("(-1,-1,0) + (-1,-1,0)"); addPandMinusP("(1,1,0) + (-3,-3,0)");

      //Along off-diagonal direction:      
      //p_pi = (2,-2,0)
      addPandMinusP("(-1,-1,0) + (3,-1,0)"); addPandMinusP("(1,1,0) + (1,-3,0)");
    }else if(ngp == 3){
      //p_pi = (-2,-2,-2)     (units of pi/2L)
      addPandMinusP("(-1,-1,-1) + (-1,-1,-1)"); addPandMinusP("(1,1,1) + (-3,-3,-3)");

      //p_pi = (2,-2,-2)
      addPandMinusP("(-1,-1,-1) + (3,-1,-1)"); addPandMinusP("(1,1,1) + (1,-3,-3)");

      //p_pi = (-2,2,-2)
      addPandMinusP("(-1,-1,-1) + (-1,3,-1)"); addPandMinusP("(1,1,1) + (-3,1,-3)");

      //p_pi = (-2,-2,2)
      addPandMinusP("(-1,-1,-1) + (-1,-1,3)"); addPandMinusP("(1,1,1) + (-3,-3,1)");

      assert(nMom() == 8);
      for(int i=0;i<8;i++) assert(nAltMom(i) == 2);
    }else{
      ERR.General("StandardPionMomentaPolicy","constructor","ngp cannot be >3\n");
    }
  };
};


//Same as the above but where we reverse the momentum assignments of the W^dag and V
class ReversePionMomentaPolicy: public StandardPionMomentaPolicy{
public:
  ReversePionMomentaPolicy(): StandardPionMomentaPolicy() {
    this->reverseABmomentumAssignments();
  }
};

//Add Wdag, V momenta and the reverse assignment to make a symmetric combination
class SymmetricPionMomentaPolicy: public StandardPionMomentaPolicy{
public:
  SymmetricPionMomentaPolicy(): StandardPionMomentaPolicy() {
    this->symmetrizeABmomentumAssignments();
  }
};


//This set of momenta does not include the second momentum combination with which we average to reduce the G-parity rotational symmetry breaking
class H4asymmetricMomentaPolicy: public RequiredMomentum{
public:
  void setupMomenta(){
    const int ngp = this->nGparityDirs();
    if(ngp == 0){
      //p_pi = (0,0,0)
      addP("(0,0,0) + (0,0,0)");
    }else if(ngp == 1){
      //p_pi = (-2,0,0)     (units of pi/2L)    
      addPandMinusP("(-1,0,0) + (-1,0,0)");
      //(In case you're wondering why my first momentum has the opposite sign to Daiqian's, its because mine is for W^dagger, not W)
    }else if(ngp == 2){
      //Along G-parity direction:
      //p_pi = (-2,-2,0)     (units of pi/2L)  
      addPandMinusP("(-1,-1,0) + (-1,-1,0)");

      //Along off-diagonal direction:      
      //p_pi = (2,-2,0)
      addPandMinusP("(-1,-1,0) + (3,-1,0)");
    }else if(ngp == 3){
      //p_pi = (-2,-2,-2)     (units of pi/2L)  
      addPandMinusP("(-1,-1,-1) + (-1,-1,-1)");

      //p_pi = (2,-2,-2)
      addPandMinusP("(-1,-1,-1) + (3,-1,-1)");

      //p_pi = (-2,2,-2)
      addPandMinusP("(-1,-1,-1) + (-1,3,-1)");

      //p_pi = (-2,-2,2)
      addPandMinusP("(-1,-1,-1) + (-1,-1,3)");
    }else{
      ERR.General("H4asymmetricMomentaPolicy","setupMomenta","ngp cannot be >3\n");
    }
  }
  H4asymmetricMomentaPolicy(): RequiredMomentum() { setupMomenta();};
};




//Additional pion momenta for the extended calculation. We symmetrize the quark momenta but don't include alt momenta. *THIS HAS ROTATIONAL SYMMETRY BREAKING*
class ExtendedPionMomentaPolicy: public RequiredMomentum{
public:
  ExtendedPionMomentaPolicy(): RequiredMomentum() {
    this->combineSameTotalMomentum(true); //momentum pairs with same total momentum will be added to same entry and treated as 'alternates' which we average together below
    const int ngp = this->nGparityDirs();
    assert(ngp == 3);

    //For the (+-6,+-2,+-2) define the 8 orientations of (-6.-2,-2) obtained by giving each component a minus sign respectively, and then cyclically permute to move the -6 around
    std::vector<std::pair<ThreeMomentum, ThreeMomentum> > base(4);

    //(-6, -2, -2) (-1, -1, -1)+(-5, -1, -1) 
    base[0] = ThreeMomentum::parse_str_two_mom("(-1, -1, -1)+(-5, -1, -1)");
    
    //(6, -2, -2) (-1, -1, -1)+(7, -1, -1) 
    base[1] = ThreeMomentum::parse_str_two_mom("(-1, -1, -1)+(7, -1, -1)");
    
    //(-6, 2, -2) (-1, -1, -1)+(-5, 3, -1) 
    base[2] = ThreeMomentum::parse_str_two_mom("(-1, -1, -1)+(-5, 3, -1)");
    
    //(-6, -2, 2) (-1, -1, -1)+(-5, -1, 3) 
    base[3] = ThreeMomentum::parse_str_two_mom("(-1, -1, -1)+(-5, -1, 3)");

    for(int perm=0;perm<3;perm++){
      for(int o=0;o<4;o++){ 
	addPandMinusP(base[o]);
	base[o].first.cyclicPermute();
	base[o].second.cyclicPermute();
      }
    }
    symmetrizeABmomentumAssignments();
	
    assert(nMom() == 24);
    for(int i=0;i<24;i++) assert(nAltMom(i) == 2);
  };
};


//Have a base + alt momentum, symmetrized. These satisfy the conditions p1+p2=p3+p4=ptot  and p1-p2 + p3-p4 = n*ptot  with n=-2. *THIS DOES |NOT| HAVE ROTATIONAL SYMMETRY BREAKING*
class AltExtendedPionMomentaPolicy: public RequiredMomentum{
public:
  AltExtendedPionMomentaPolicy(): RequiredMomentum() {
    this->combineSameTotalMomentum(true); //momentum pairs with same total momentum will be added to same entry and treated as 'alternates' which we average together below
    const int ngp = this->nGparityDirs();
    assert(ngp == 3);

    //For the (+-6,+-2,+-2) define the 8 orientations of (-6.-2,-2) obtained by giving each component a minus sign respectively, and then cyclically permute to move the -6 around
    std::vector<std::pair<ThreeMomentum, ThreeMomentum> > base(4);
    std::vector<std::pair<ThreeMomentum, ThreeMomentum> > alt(4);

    //(-6, -2, -2) (-1, -1, -1)+(-5, -1, -1) (1, 1, 1)+(-7, -3, -3)
    base[0] = ThreeMomentum::parse_str_two_mom("(-1, -1, -1)+(-5, -1, -1)");
    alt[0] = ThreeMomentum::parse_str_two_mom("(1, 1, 1)+(-7, -3, -3)");
    
    //(6, -2, -2) (-1, -1, -1)+(7, -1, -1) (1, 1, 1)+(5, -3, -3)
    base[1] = ThreeMomentum::parse_str_two_mom("(-1, -1, -1)+(7, -1, -1)");
    alt[1] = ThreeMomentum::parse_str_two_mom("(1, 1, 1)+(5, -3, -3)");
    
    //(-6, 2, -2) (-1, -1, -1)+(-5, 3, -1) (1, 1, 1)+(-7, 1, -3)
    base[2] = ThreeMomentum::parse_str_two_mom("(-1, -1, -1)+(-5, 3, -1)");
    alt[2] = ThreeMomentum::parse_str_two_mom("(1, 1, 1)+(-7, 1, -3)");
    
    //(-6, -2, 2) (-1, -1, -1)+(-5, -1, 3) (1, 1, 1)+(-7, -3, 1)
    base[3] = ThreeMomentum::parse_str_two_mom("(-1, -1, -1)+(-5, -1, 3)");
    alt[3] = ThreeMomentum::parse_str_two_mom("(1, 1, 1)+(-7, -3, 1)");    

    for(int perm=0;perm<3;perm++){
      for(int o=0;o<4;o++){ 
	addPandMinusP(base[o]);
	addPandMinusP(alt[o]);
	base[o].first.cyclicPermute();
	base[o].second.cyclicPermute();
	alt[o].first.cyclicPermute();
	alt[o].second.cyclicPermute();
      }
    }
    symmetrizeABmomentumAssignments();
	
    assert(nMom() == 24);
    for(int i=0;i<24;i++) assert(nAltMom(i) == 4);
  };
};







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


template<typename mf_Policies, typename PionMomentumPolicy>
class computeGparityLLmesonFields1s{
public:
  INHERIT_FROM_BASE;

  typedef A2AflavorProjectedExpSource<SourcePolicies> ExpSrcType;
  typedef GparitySourceShiftInnerProduct<ComplexType,ExpSrcType, flavorMatrixSpinColorContract<15,ComplexType,true,false> > InnerType;
  typedef GparityFlavorProjectedShiftSourceStorage<mf_Policies, InnerType> StorageType;
   
public:

  struct Options{
    int thr_internal; //number of threads used in the computation
    int mom_block_size; //number of meson momenta fed into the computeMany algorithm
    int nshift_combine_max; //max number of source shifts shiftSourceStorage is allowed to combine
    Options(): thr_internal(-1), mom_block_size(-1), nshift_combine_max(-1){}
  };


  static void computeMesonFields(MesonFieldMomentumContainer<mf_Policies> &mf_ll_1s_con, //container for 1s pion output fields, accessible by ThreeMomentum of pion
				 const PionMomentumPolicy &pion_mom, //object that tells us what quark momenta to use
				 Wtype &W, Vtype &V,
				 const Float rad_1s, //exponential wavefunction radius
				 Lattice &lattice,
				 const FieldParamType &src_setup_params = NullObject(), const Options &opt = Options() ){
    assert(GJP.Gparity());
    Float time = -dclock();    
    std::vector<Wtype*> Wspecies(1, &W);
    std::vector<Vtype*> Vspecies(1, &V);

    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    const int nmom = pion_mom.nMom();

    int init_thr = omp_get_max_threads();
    if(opt.thr_internal != -1) omp_set_num_threads(opt.thr_internal);

    int pbase[3]; //we reset the momentum for each computation so we technically don't need this - however the code demands a valid momentum
    GparityBaseMomentum(pbase,+1);
        
    ExpSrcType src(rad_1s,pbase,src_setup_params);
    InnerType g5_s3_inner(sigma3, src);    
    
    int nmom_block = opt.mom_block_size != -1 ? opt.mom_block_size : nmom;

    if(!UniqueID()) printf("Computing %d momenta with block sizes of %d\n",nmom,nmom_block);

    for(int b=0; b<nmom; b+= nmom_block){
      int nmom_rem = nmom - b;
      int nmom_block_actual = nmom_rem < nmom_block ? nmom_rem : nmom_block;

      if(!UniqueID()) printf("Doing block %d->%d\n",b,b+nmom_rem);

      StorageType mf_store(g5_s3_inner,src, opt.nshift_combine_max);
      std::vector< std::vector<int> > toavg(nmom_block_actual);

      int cidx = 0;
      for(int pidx=b;pidx<b+nmom_block_actual;pidx++){      
	for(int alt=0;alt<pion_mom.nAltMom(pidx);alt++){
	  ThreeMomentum p_w = pion_mom.getWmom(pidx,alt);
	  ThreeMomentum p_v = pion_mom.getVmom(pidx,alt);
	  mf_store.addCompute(0,0, p_w,p_v);
	  toavg[pidx-b].push_back(cidx++);
	}
      }

      ComputeMesonFields<mf_Policies,StorageType>::compute(mf_store,Wspecies,Vspecies,lattice
#ifdef NODE_DISTRIBUTE_MESONFIELDS
							   ,true
#endif
										       );
      struct indexer{
	const std::vector<int> &cidx;
	indexer(const std::vector<int> &cidx): cidx(cidx){}
	inline int operator()(const int i) const{ return cidx[i]; }
      };
      for(int q=0;q<nmom_block_actual;q++){
	indexer idxr(toavg[q]);
	stridedAverageFree(mf_store, idxr, toavg[q].size());
	mf_ll_1s_con.moveAdd(pion_mom.getMesonMomentum(b+q), mf_store[toavg[q][0]]);
      }
    } //block loop

    if(opt.thr_internal != -1) omp_set_num_threads(init_thr);
  }

};



template<typename mf_Policies, typename PionMomentumPolicy>
class computeGparityLLmesonFieldsPoint{
public:
  INHERIT_FROM_BASE;

  typedef A2AflavorProjectedPointSource<SourcePolicies> PointSrcType;
  typedef GparitySourceShiftInnerProduct<ComplexType,PointSrcType, flavorMatrixSpinColorContract<15,ComplexType,true,false> > InnerType;
  typedef GparityFlavorProjectedShiftSourceStorage<mf_Policies, InnerType> StorageType;
   
public:

  static void computeMesonFields(MesonFieldMomentumContainer<mf_Policies> &mf_con, //container for 1s pion output fields, accessible by ThreeMomentum of pion
				 const PionMomentumPolicy &pion_mom, //object that tells us what quark momenta to use
				 Wtype &W, Vtype &V,
				 Lattice &lattice,
				 const FieldParamType &src_setup_params = NullObject()){
    assert(GJP.Gparity());
    Float time = -dclock();    
    std::vector<Wtype*> Wspecies(1, &W);
    std::vector<Vtype*> Vspecies(1, &V);

    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    const int nmom = pion_mom.nMom();

/* #ifdef ARCH_BGQ */
/*     int init_thr = omp_get_max_threads(); */
/*     if(init_thr > 32) omp_set_num_threads(32); */
/* #endif */
      
    PointSrcType src(src_setup_params);
    InnerType g5_s3_inner(sigma3, src);    
    StorageType mf_store(g5_s3_inner,src);
    
    std::vector< std::vector<int> > toavg(nmom);

    int cidx = 0;
    for(int pidx=0;pidx<nmom;pidx++){      
      for(int alt=0;alt<pion_mom.nAltMom(pidx);alt++){
	ThreeMomentum p_w = pion_mom.getWmom(pidx,alt);
	ThreeMomentum p_v = pion_mom.getVmom(pidx,alt);
	mf_store.addCompute(0,0, p_w,p_v);
	toavg[pidx].push_back(cidx++);
      }
    }

    ComputeMesonFields<mf_Policies,StorageType>::compute(mf_store,Wspecies,Vspecies,lattice
#ifdef NODE_DISTRIBUTE_MESONFIELDS
										       ,true
#endif
										       );
/* #ifdef ARCH_BGQ */
/*     omp_set_num_threads(init_thr); */
/* #endif */
  
    struct indexer{
      const std::vector<int> &cidx;
      indexer(const std::vector<int> &cidx): cidx(cidx){}
      inline int operator()(const int i) const{ return cidx[i]; }
    };
    for(int pidx=0;pidx<nmom;pidx++){
      indexer idxr(toavg[pidx]);
      stridedAverageFree(mf_store, idxr, toavg[pidx].size());
      mf_con.moveAdd(pion_mom.getMesonMomentum(pidx), mf_store[toavg[pidx][0]]);
    }
  }

};





//Methods for generating meson fields for 1s and point sources together
template<typename mf_Policies, typename PionMomentumPolicy>
class computeGparityLLmesonFields1sPoint{
public:
  INHERIT_FROM_BASE;

  typedef A2AflavorProjectedExpSource<SourcePolicies> ExpSrcType;
  typedef A2AflavorProjectedPointSource<SourcePolicies> PointSrcType;
  typedef Elem<ExpSrcType, Elem<PointSrcType,ListEnd > > SrcList;
  typedef A2AmultiSource<SrcList> MultiSrcType;
  //typedef SCFspinflavorInnerProduct<15,ComplexType,MultiSrcType,true,false> MultiInnerType;
  //typedef GparityFlavorProjectedMultiSourceStorage<mf_Policies, MultiInnerType> StorageType;
  
  //Allows for more memory efficient computation algorithm
  typedef GparitySourceShiftInnerProduct<ComplexType,MultiSrcType, flavorMatrixSpinColorContract<15,ComplexType,true,false> > MultiInnerType;
  typedef GparityFlavorProjectedShiftSourceStorage<mf_Policies, MultiInnerType> StorageType;
   
public:

  static void computeMesonFields(MesonFieldMomentumContainer<mf_Policies> &mf_ll_1s_con, //container for 1s pion output fields, accessible by ThreeMomentum of pion
				 MesonFieldMomentumContainer<mf_Policies> &mf_ll_pt_con, //same for point source
				 const PionMomentumPolicy &pion_mom, //object that tells us what quark momenta to use
				 Wtype &W, Vtype &V,
				 const Float rad_1s, //exponential wavefunction radius
				 Lattice &lattice,
				 const FieldParamType &src_setup_params = NullObject()){
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

    int pbase[3]; //we reset the momentum for each computation so we technically don't need this - however the code demands a valid momentum
    GparityBaseMomentum(pbase,+1);
    
    MultiSrcType src;
    src.template getSource<0>().setup(rad_1s,pbase,src_setup_params); //1s
    src.template getSource<1>().setup(src_setup_params); //point
    
    MultiInnerType g5_s3_inner(sigma3, src);
    
    StorageType mf_store(g5_s3_inner,src);
  
    std::vector< std::vector<int> > toavg(nmom);

    int cidx = 0;
    for(int pidx=0;pidx<nmom;pidx++){      
      for(int alt=0;alt<pion_mom.nAltMom(pidx);alt++){
	ThreeMomentum p_w = pion_mom.getWmom(pidx,alt);
	ThreeMomentum p_v = pion_mom.getVmom(pidx,alt);
	mf_store->addCompute(0,0, p_w,p_v);
	toavg[pidx].push_back(cidx++);
      }
    }

    ComputeMesonFields<mf_Policies,StorageType>::compute(mf_store,Wspecies,Vspecies,lattice
#ifdef NODE_DISTRIBUTE_MESONFIELDS
							 ,true
#endif
							 );
#ifdef ARCH_BGQ
    omp_set_num_threads(init_thr);
#endif
    
    MesonFieldMomentumContainer<mf_Policies>* con[2] = { &mf_ll_1s_con, &mf_ll_pt_con };

    for(int src_idx=0;src_idx<2;src_idx){
      struct indexer{
	const std::vector<int> &cidx;
	const int src_idx;
	indexer(const std::vector<int> &cidx, const int src_idx): cidx(cidx), src_idx(src_idx){}	
	inline std::pair<int,int> operator()(const int i) const{ return std::pair<int,int>(src_idx, cidx[i]); }
      };
      for(int pidx=0;pidx<nmom;pidx++){
	indexer idxr(toavg[pidx],src_idx);
	stridedAverageFree(mf_store, idxr, toavg[pidx].size());
	con[src_idx]->moveAdd(pion_mom.getMesonMomentum(pidx), mf_store(src_idx,toavg[pidx][0]));
      }
    }
  }

};


#undef INHERIT
#undef INHERIT_FROM_BASE

CPS_END_NAMESPACE
#endif
