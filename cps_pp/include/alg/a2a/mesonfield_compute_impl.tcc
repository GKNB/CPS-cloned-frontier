//Meson field computation code

#ifndef _MESONFIELD_COMPUTE_IMPL
#define _MESONFIELD_COMPUTE_IMPL

#ifndef USE_GRID
#define __SSC_MARK(I)
#endif

//For all mode indices l_i and r_j, compute the meson field  \sum_p l_i^\dagger(p,t) M(p,t) r_j(p,t)
//It is assumed that A2AfieldL and A2AfieldR are Fourier transformed field containers
//M(p,t) is a completely general momentum-space spin/color/flavor matrix per temporal slice

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
template<typename InnerProduct>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::compute(const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, const int &t, bool do_setup){
  if(do_setup) setup(l,r,t,t); //both vectors have same timeslice
  else zero();
  
  if(!UniqueID()) printf("Starting A2AmesonField::compute timeslice %d with %d threads\n",t, omp_get_max_threads());

  double time = -dclock();

  //For W vectors we dilute out the flavor index in-place while performing this contraction
  const typename mf_Policies::FermionFieldType &mode0 = l.getMode(0);
  const int size_3d = mode0.nodeSites(0)*mode0.nodeSites(1)*mode0.nodeSites(2);
  if(mode0.nodeSites(3) != GJP.TnodeSites()) ERR.General("A2AmesonField","compute","Not implemented for fields where node time dimension != GJP.TnodeSites()\n");

  int nl_l = lindexdilution.getNl();
  int nl_r = rindexdilution.getNl();

  int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();
  if(t_lcl >= 0 && t_lcl < GJP.TnodeSites()){ //if timeslice is on-node

#pragma omp parallel for
    for(int i = 0; i < nmodes_l; i++){
      cps::ComplexD mf_accum;

      modeIndexSet i_high_unmapped; if(i>=nl_l) lindexdilution.indexUnmap(i-nl_l,i_high_unmapped);

      for(int j = 0; j < nmodes_r; j++) {
	modeIndexSet j_high_unmapped; if(j>=nl_r) rindexdilution.indexUnmap(j-nl_r,j_high_unmapped);

	mf_accum = 0.;

	for(int p_3d = 0; p_3d < size_3d; p_3d++) {
	  SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> lscf = l.getFlavorDilutedVect(i,i_high_unmapped,p_3d,t_lcl); //dilute flavor in-place if it hasn't been already
	  SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> rscf = r.getFlavorDilutedVect(j,j_high_unmapped,p_3d,t_lcl);

	  mf_accum += M(lscf,rscf,p_3d,t); //produces double precision output by spec
	}
	(*this)(i,j) = mf_accum; //downcast after accumulate      
      }
    }
  }
  sync();
  print_time("A2AmesonField","local compute",time + dclock());
  time = -dclock();

  //Sum over all nodes so all nodes have a copy
  nodeSum();
  print_time("A2AmesonField","nodeSum",time + dclock());
}




//Compute meson fields for all timeslices. This version is more efficient on multi-nodes
#ifdef AVX512
CPS_END_NAMESPACE
#include<simd/Intel512common.h>
CPS_START_NAMESPACE
#endif

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename mf_Element, typename mf_Element_Vector>
class MultKernel{
public:
#ifdef AVX512
  static void prefetchFvec(const char* ptr){
    //T0 hint
#define _VPREFETCH1(O,A) VPREFETCH1(O,A)
    //T1 hint
#define _VPREFETCH2(O,A) VPREFETCH2(O,A)
    
#define _PREF(O,A) VPREFETCH2(O,A)

    __asm__ ( 
    _PREF(0,%rdi) \
    _PREF(1,%rdi) \
    _PREF(2,%rdi) \
    _PREF(3,%rdi) \
    _PREF(4,%rdi) \
    _PREF(5,%rdi) \
    _PREF(6,%rdi) \
    _PREF(7,%rdi) \
    _PREF(8,%rdi) \
    _PREF(9,%rdi) \
    _PREF(10,%rdi) \
    _PREF(11,%rdi) 
	      );
  }
#endif


  inline static void prefetchAdvanceSite(SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> &lscf,
					 SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> &rscf,
					 const std::pair<int,int> &site_offset_i, const std::pair<int,int> &site_offset_j){
#ifdef AVX512
    lscf.incrementPointers(site_offset_i);
    prefetchFvec((const char*)lscf.getPtr(0));
    prefetchFvec((const char*)lscf.getPtr(1));
    rscf.incrementPointers(site_offset_j);
    prefetchFvec((const char*)rscf.getPtr(0));
    prefetchFvec((const char*)rscf.getPtr(1));
    lscf.incrementPointers(site_offset_i,-1);
    rscf.incrementPointers(site_offset_j,-1);
#endif
  }
  inline static void prefetchSite(const SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> &lscf,
				  const SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> &rscf){				  
#ifdef AVX512
    prefetchFvec((const char*)lscf.getPtr(0));
    prefetchFvec((const char*)lscf.getPtr(1));
    prefetchFvec((const char*)rscf.getPtr(0));
    prefetchFvec((const char*)rscf.getPtr(1));
#endif
  }

  //Lowest level of blocked matrix mult. Ideally this should fit in L1 cache.
  template<typename InnerProduct>
  static void mult_kernel(basicMatrix<mf_Element,Aligned128AllocPolicy> & mf_accum_m, const InnerProduct &M, const int t,
			  const int i0, const int iup, const int j0, const int jup, const int p0, const int pup,
			  const std::vector<SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> > &base_ptrs_i,
			  const std::vector<SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> > &base_ptrs_j,
			  const std::vector<std::pair<int,int> > &site_offsets_i,
			  const std::vector<std::pair<int,int> > &site_offsets_j){
    for(int i = i0; i < iup; i++){	      
      for(int j = j0; j < jup; j++) {		
	
	mf_Element &mf_accum = mf_accum_m(i,j);
	
	SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> lscf(base_ptrs_i[i], site_offsets_i[i], p0);
	SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> rscf(base_ptrs_j[j], site_offsets_j[j], p0);

	//prefetchSite(lscf,rscf);

	for(int p_3d = p0; p_3d < pup; p_3d++) {
	  //prefetchAdvanceSite(lscf,rscf,site_offsets_i[i],site_offsets_j[j]);

	  M(mf_accum,lscf,rscf,p_3d,t);	 
	  lscf.incrementPointers(site_offsets_i[i]);
	  rscf.incrementPointers(site_offsets_j[j]);		  
	}
      }
    }
  }
  //Do a second layer of blocked dgemm to try to fit in the L1 cache
  //note the i0, iup, etc are the low and high range limits from the outer blocking
  template<typename InnerProduct>
  static void inner_block_mult(basicMatrix<mf_Element,Aligned128AllocPolicy> &mf_accum_m, const InnerProduct &M, const int t,
			       const int i0, const int iup, const int j0, const int jup, const int p0, const int pup,
			       const std::vector<SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> > &base_ptrs_i,
			       const std::vector<SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> > &base_ptrs_j,
			       const std::vector<std::pair<int,int> > &site_offsets_i,
			       const std::vector<std::pair<int,int> > &site_offsets_j){
    const int bii = BlockedMesonFieldArgs::bii;
    const int bjj = BlockedMesonFieldArgs::bjj;
    const int bpp = BlockedMesonFieldArgs::bpp;
    
    for(int ii0=i0; ii0 < iup; ii0+=bii){
      int iiup = std::min(ii0+bii,iup);
      for(int jj0=j0; jj0 < jup; jj0+=bjj){
	int jjup = std::min(jj0+bjj,jup);
	for(int pp0=p0; pp0 < pup; pp0+=bpp){
	  int ppup = std::min(pp0+bpp,pup);

	  MultKernel<mf_Policies,A2AfieldL,A2AfieldR,mf_Element,mf_Element_Vector>::mult_kernel(mf_accum_m, M, t,
												ii0, iiup, jj0, jjup, pp0, ppup,
												base_ptrs_i, base_ptrs_j, site_offsets_i, site_offsets_j);
	}
      }
    }
  }
};

//Policies for single and multi-src outputs

//Single src
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename Allocator, typename InnerProduct>
struct SingleSrcVectorPolicies{
  typedef std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator > mfVectorType;
  typedef cps::ComplexD mf_Element;
  typedef typename AlignedVector<mf_Element>::type mf_Element_Vector;

  static inline void setupPolicy(const InnerProduct &M){ 
    if(!UniqueID()){ printf("Using SingleSrcVectorPolicies\n"); fflush(stdout); }
    assert(M.mfPerTimeSlice() == 1); 
  }
  static inline void initializeElement(mf_Element &e){ e = mf_Element(0.); }
  static void initializeMesonFields(mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const A2AfieldR<mf_Policies> &r, const int Lt, const bool do_setup){
    mf_t.resize(Lt);
    for(int t=0;t<Lt;t++) 
      if(do_setup) mf_t[t].setup(l,r,t,t); //both vectors have same timeslice (zeroes the starting matrix)
      else{
	assert(mf_t[t].ptr() != NULL);
	mf_t[t].zero();
      }
  }
  static inline void sumThreadedResults(mfVectorType &mf_t, const std::vector<basicMatrix<mf_Element,Aligned128AllocPolicy> > &mf_accum_thr, const int i, const int j, const int t, const int nthread){
    for(int thr=0;thr<nthread;thr++)
      mf_t[t](i,j) += mf_accum_thr[thr](i,j);
  }
  //Used to get information about rows and cols
  static inline const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> & getReferenceMf(const mfVectorType &mf_t, const int t){
    return mf_t[t];
  }
  static inline void nodeSum(mfVectorType &mf_t, const int Lt){
    for(int t=0; t<Lt; t++) mf_t[t].nodeSum();
  }
  static inline void printElement(const mf_Element &e){
    std::cout << "(" << e.real() << "," << e.imag() << ")";
  }
};

//Multisrc
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename Allocator, typename InnerProduct>
struct MultiSrcVectorPolicies{
  int mfPerTimeSlice;
  
  typedef std::vector< std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator >* > mfVectorType;  //indexed by [srcidx][t]
  typedef cps::ComplexD* mf_Element;
  typedef std::vector<mf_Element> mf_Element_Vector;

  cps::ComplexD* mem_pool;
  size_t mem_pool_size;
  std::vector<size_t> mem_pool_off; //accumulate next pointer offset per thread
  
  MultiSrcVectorPolicies(): mem_pool(NULL){}
  ~MultiSrcVectorPolicies(){ if(mem_pool) free(mem_pool); }

  inline void createMemPool(const int nmodes_l, const int nmodes_r){
    const int nthread = omp_get_max_threads();
    size_t size = nthread * nmodes_l * nmodes_r * mfPerTimeSlice;
    if(mem_pool == NULL || (mem_pool != NULL && mem_pool_size != size) ){
      if(mem_pool != NULL) free(mem_pool);

      if(!UniqueID()){
	printf("Initializing memory pool. Node 0 memory status: \n");
	printMem(0);
	fflush(stdout);
      }
      cps::sync();

      //mem_pool = (cps::ComplexD*)memalign(128, size * sizeof(cps::ComplexD));
      
      int r = posix_memalign( (void**)&mem_pool, 128, size * sizeof(cps::ComplexD));
      if(r){
	std::cout << "MultiSrcVectorPolicies::createMemPool Node " << UniqueID() << " unable to allocate memory of size " << byte_to_MB(size*sizeof(cps::ComplexD)) << "  MB" 
		  << ". Error code " << r << " (" << EINVAL << "=EINVAL, " << ENOMEM << "=ENOMEM)\n";
	std::cout << "Memory status: \n";	
	std::cout.flush();
	printMem(UniqueID());
	fflush(stdout);
	exit(-1);
      }

      mem_pool_size = size;
    }
    mem_pool_off.resize(nthread);
    for(int i=0;i<nthread;i++)
      mem_pool_off[i] = i*nmodes_l * nmodes_r * mfPerTimeSlice;
    
    memset(mem_pool, 0, size * sizeof(cps::ComplexD));
  } 
  
  inline void setupPolicy(const InnerProduct &M){
    mfPerTimeSlice = M.mfPerTimeSlice();
    if(!UniqueID()){ printf("Using MultiSrcVectorPolicies with #MF per timeslice %d\n",mfPerTimeSlice); fflush(stdout); }
  }
  
  inline void initializeElement(mf_Element &e){
    int me = omp_get_thread_num();
    e = mem_pool + mem_pool_off[me];
    mem_pool_off[me] += mfPerTimeSlice;    
  }
  
  void initializeMesonFields(mfVectorType &mf_st, const A2AfieldL<mf_Policies> &l, const A2AfieldR<mf_Policies> &r, const int Lt, const bool do_setup) const{
    if(mf_st.size() != mfPerTimeSlice) ERR.General("mf_Vector_policies <multi src>","initializeMesonFields","Expect output vector to be of size %d, got size %d\n",mfPerTimeSlice,mf_st.size());

    size_t mf_size = A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::byte_size(l,r);
    size_t total_size = mf_size * Lt * mfPerTimeSlice;
    if(!UniqueID()){
      typename A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::LeftDilutionType ll = l;
      typename A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::RightDilutionType rr = r;
      
      printf("Initializing meson fields of size %d * %d. Memory requirement is %g MB, memory status is:\n", ll.getNmodes(),rr.getNmodes(),byte_to_MB(total_size));
      printMem(0);
      fflush(stdout);
    }
    cps::sync();


    for(int s=0;s<mfPerTimeSlice;s++){
      mf_st[s]->resize(Lt);
      for(int t=0;t<Lt;t++) 
	if(do_setup) mf_st[s]->operator[](t).setup(l,r,t,t); //both vectors have same timeslice (zeroes the starting matrix)
	else{
	  assert(mf_st[s]->operator[](t).ptr() != NULL);
	  mf_st[s]->operator[](t).zero();
	}
    }
  }
  inline void sumThreadedResults(mfVectorType &mf_st, const std::vector<basicMatrix<mf_Element,Aligned128AllocPolicy> > &mf_accum_thr, const int i, const int j, const int t, const int nthread) const{
    for(int thr=0;thr<nthread;thr++)
      for(int s=0;s<mfPerTimeSlice;s++)
	mf_st[s]->operator[](t)(i,j) += mf_accum_thr[thr](i,j)[s];
  }

  //Used to get information about rows and cols
  inline const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> & getReferenceMf(const mfVectorType &mf_st, const int t){
    const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &mf_ref = mf_st[0]->operator[](t);
    createMemPool(mf_ref.getNrows(), mf_ref.getNcols());
    return mf_ref;
  }
  inline void nodeSum(mfVectorType &mf_st, const int Lt) const{
    for(int s=0;s<mfPerTimeSlice;s++)
      for(int t=0; t<Lt; t++) mf_st[s]->operator[](t).nodeSum();
  }
  inline void printElement(const mf_Element &e) const{
    for(int i=0;i<mfPerTimeSlice;i++) std::cout << i << ":(" << e[i].real() << "," << e[i].imag() << ") ";
  }
};


#ifdef USE_GRID
//Single src vectorized with delayed reduction
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename Allocator, typename InnerProduct>
struct SingleSrcVectorPoliciesSIMD{
  typedef std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator > mfVectorType;
  typedef Grid::vComplexD mf_Element;
  typedef typename AlignedVector<mf_Element>::type mf_Element_Vector;
  
  static inline void setupPolicy(const InnerProduct &M){ 
    if(!UniqueID()){ printf("Using SingleSrcVectorPoliciesSIMD\n"); fflush(stdout); }
    assert(M.mfPerTimeSlice() == 1); 
  }
  static inline void initializeElement(mf_Element &e){ zeroit(e); }
  static void initializeMesonFields(mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const A2AfieldR<mf_Policies> &r, const int Lt, const bool do_setup){
    mf_t.resize(Lt);
    for(int t=0;t<Lt;t++) 
      if(do_setup) mf_t[t].setup(l,r,t,t); //both vectors have same timeslice (zeroes the starting matrix)
      else{
	assert(mf_t[t].ptr() != NULL);
	mf_t[t].zero();
      }
  }
  static inline void sumThreadedResults(mfVectorType &mf_t, const std::vector<basicMatrix<mf_Element,Aligned128AllocPolicy> > &mf_accum_thr, const int i, const int j, const int t, const int nthread){
    mf_Element tmp = mf_accum_thr[0](i,j);
    for(int thr=1;thr<nthread;thr++) tmp += mf_accum_thr[thr](i,j);
    mf_t[t](i,j) += Reduce(tmp);
  }
  //Used to get information about rows and cols
  static inline const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> & getReferenceMf(const mfVectorType &mf_t, const int t){
    return mf_t[t];
  }
  static inline void nodeSum(mfVectorType &mf_t, const int Lt){
    for(int t=0; t<Lt; t++) mf_t[t].nodeSum();
  }
  static inline void printElement(const mf_Element &e){
  }
};


//Multisrc with delayed reduction
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename Allocator, typename InnerProduct>
struct MultiSrcVectorPoliciesSIMD{
  int mfPerTimeSlice;
  
  typedef std::vector< std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator >* > mfVectorType;  //indexed by [srcidx][t]
  typedef typename Grid::vComplexD* mf_Element;
  typedef std::vector<mf_Element> mf_Element_Vector;

  Grid::vComplexD* mem_pool;
  size_t mem_pool_size;
  std::vector<size_t> mem_pool_off;
  size_t mem_pool_reduce_off;
  
  MultiSrcVectorPoliciesSIMD(): mem_pool(NULL){}
  ~MultiSrcVectorPoliciesSIMD(){ if(mem_pool) free(mem_pool); }

  inline void createMemPool(const int nmodes_l, const int nmodes_r){
    const int nthread = omp_get_max_threads();
    size_t size = nthread * nmodes_l * nmodes_r * mfPerTimeSlice + nthread * mfPerTimeSlice; //extra space for final threaded accumulations
    if(mem_pool == NULL || (mem_pool != NULL && mem_pool_size != size) ){
      if(mem_pool != NULL) free(mem_pool);
      mem_pool = (Grid::vComplexD*)memalign(128, size * sizeof(Grid::vComplexD));
      mem_pool_size = size;
    }
    mem_pool_reduce_off = nthread * nmodes_l * nmodes_r * mfPerTimeSlice;
    
    mem_pool_off.resize(nthread);
    for(int i=0;i<nthread;i++)
      mem_pool_off[i] = i*nmodes_l * nmodes_r * mfPerTimeSlice;
    
    memset(mem_pool, 0, size * sizeof(Grid::vComplexD));
  } 

  inline void setupPolicy(const InnerProduct &M){
    mfPerTimeSlice = M.mfPerTimeSlice();
    if(!UniqueID()){ printf("Using MultiSrcVectorPoliciesSIMD with #MF per timeslice %d\n",mfPerTimeSlice); fflush(stdout); }
  }
  
  inline void initializeElement(mf_Element &e){
    int me = omp_get_thread_num();
    e = mem_pool + mem_pool_off[me];
    mem_pool_off[me] += mfPerTimeSlice;
  }
  void initializeMesonFields(mfVectorType &mf_st, const A2AfieldL<mf_Policies> &l, const A2AfieldR<mf_Policies> &r, const int Lt, const bool do_setup) const{
    if(mf_st.size() != mfPerTimeSlice) ERR.General("mf_Vector_policies <multi src>","initializeMesonFields","Expect output vector to be of size %d, got size %d\n",mfPerTimeSlice,mf_st.size());

    for(int s=0;s<mfPerTimeSlice;s++){
      mf_st[s]->resize(Lt);
      for(int t=0;t<Lt;t++) 
	if(do_setup) mf_st[s]->operator[](t).setup(l,r,t,t); //both vectors have same timeslice (zeroes the starting matrix)
	else{
	  assert(mf_st[s]->operator[](t).ptr() != NULL);
	  mf_st[s]->operator[](t).zero();
	}
    }
  }
  inline void sumThreadedResults(mfVectorType &mf_st, const std::vector<basicMatrix<mf_Element,Aligned128AllocPolicy> > &mf_accum_thr, const int i, const int j, const int t, const int nthread) const{
    Grid::vComplexD* tmp = mem_pool + mem_pool_reduce_off + omp_get_thread_num() * mfPerTimeSlice;
    
    for(int s=0;s<mfPerTimeSlice;s++) tmp[s] = mf_accum_thr[0](i,j)[s];

    for(int thr=1;thr<nthread;thr++)
      for(int s=0;s<mfPerTimeSlice;s++)
    	tmp[s] += mf_accum_thr[thr](i,j)[s];

    for(int s=0;s<mfPerTimeSlice;s++)
      mf_st[s]->operator[](t)(i,j) += Reduce(tmp[s]);    
  }

  //Used to get information about rows and cols
  inline const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> & getReferenceMf(const mfVectorType &mf_st, const int t){
    const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &mf_ref = mf_st[0]->operator[](t);
    createMemPool(mf_ref.getNrows(), mf_ref.getNcols()); //hacky but it does the job!
    return mf_ref;
  }
  inline void nodeSum(mfVectorType &mf_st, const int Lt) const{
    for(int s=0;s<mfPerTimeSlice;s++)
      for(int t=0; t<Lt; t++) mf_st[s]->operator[](t).nodeSum();
  }
  inline void printElement(const mf_Element &e) const{
    //for(int i=0;i<mfPerTimeSlice;i++) std::cout << i << ":(" << e[i].real() << "," << e[i].imag() << ") ";
  }
};






#endif



template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename InnerProduct, typename mfVectorPolicies>
struct mfComputeGeneral: public mfVectorPolicies{
  typedef typename mfVectorPolicies::mfVectorType mfVectorType;

  void compute(mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
    typedef typename mfVectorPolicies::mf_Element mf_Element;
    typedef typename mfVectorPolicies::mf_Element_Vector mf_Element_Vector;
    this->setupPolicy(M);
    
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    if(!UniqueID()) printf("Starting A2AmesonField::compute (blocked) for %d timeslices with %d threads\n",Lt, omp_get_max_threads());
#ifdef KNL_OPTIMIZATIONS
    if(!UniqueID()) printf("Using KNL optimizations\n");
#else
    if(!UniqueID()) printf("NOT using KNL optimizations\n");
#endif
    if(!UniqueID()){
      printf("Node 0 memory status:\n");
      printMem(0);
      fflush(stdout);
    }
    cps::sync();


    double time = -dclock();
    this->initializeMesonFields(mf_t,l,r,Lt,do_setup);
    print_time("A2AmesonField","setup",time + dclock());

    time = -dclock();
    //For W vectors we dilute out the flavor index in-place while performing this contraction
    const typename mf_Policies::FermionFieldType &mode0 = l.getMode(0);
    const int size_3d = mode0.nodeSites(0)*mode0.nodeSites(1)*mode0.nodeSites(2);
    if(mode0.nodeSites(3) != GJP.TnodeSites()) ERR.General("A2AmesonField","compute","Not implemented for fields where node time dimension != GJP.TnodeSites()\n");
  
    //Each node only works on its time block
    for(int t=GJP.TnodeCoor()*GJP.TnodeSites(); t<(GJP.TnodeCoor()+1)*GJP.TnodeSites(); t++){
      const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> & mf_ref = this->getReferenceMf(mf_t,t); //assumes all meson fields of the mf_Element type have the same mode parameters
      
      double ttime = -dclock();

      const int nl_l = mf_ref.getRowParams().getNl();
      const int nl_r = mf_ref.getColParams().getNl();
      const int nmodes_l = mf_ref.getNrows();
      const int nmodes_r = mf_ref.getNcols();
      
      int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();

      const int bi = BlockedMesonFieldArgs::bi;
      const int bj = BlockedMesonFieldArgs::bj;
      const int bp = BlockedMesonFieldArgs::bp;

      int nthread = omp_get_max_threads();
      std::vector<basicMatrix<mf_Element,Aligned128AllocPolicy> > mf_accum_thr(nthread);

#pragma omp parallel for
      for(int thr=0;thr<nthread;thr++){
	mf_accum_thr[thr].resize(nmodes_l,nmodes_r);
	mf_Element* ptr = mf_accum_thr[thr].ptr();
	for(int ij=0; ij< mf_accum_thr[thr].size(); ij++) this->initializeElement( *(ptr++) );
      }
	
      //Make a table of p base pointers and site offsets for each i,j
      std::vector<SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> > base_ptrs_i(nmodes_l);
      std::vector<SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> > base_ptrs_j(nmodes_r);
      std::vector<std::pair<int,int> > site_offsets_i(nmodes_l);
      std::vector<std::pair<int,int> > site_offsets_j(nmodes_r);
#ifndef MEMTEST_MODE
      __SSC_MARK(0x1);

#pragma omp parallel
      {
	int me = omp_get_thread_num();

	//Generate the tables
	int thr_tabwork, thr_taboff;
	thread_work(thr_tabwork, thr_taboff, nmodes_l, me, omp_get_num_threads());
	for(int i=thr_taboff; i<thr_taboff+thr_tabwork;i++){ //i table
	  modeIndexSet i_high_unmapped; if(i>=nl_l) mf_ref.getRowParams().indexUnmap(i-nl_l,i_high_unmapped);
	  base_ptrs_i[i] = l.getFlavorDilutedVect(i,i_high_unmapped,0,t_lcl);
	  site_offsets_i[i] = std::pair<int,int>( l.siteStride3D(i,i_high_unmapped,0), l.siteStride3D(i,i_high_unmapped,1) );
	}
	thread_work(thr_tabwork, thr_taboff, nmodes_r, me, omp_get_num_threads());
	for(int j=thr_taboff; j<thr_taboff+thr_tabwork;j++){ //j table
	  modeIndexSet j_high_unmapped; if(j>=nl_r) mf_ref.getColParams().indexUnmap(j-nl_r,j_high_unmapped);
	  base_ptrs_j[j] = r.getFlavorDilutedVect(j,j_high_unmapped,0,t_lcl);
	  site_offsets_j[j] = std::pair<int,int>( r.siteStride3D(j,j_high_unmapped,0), r.siteStride3D(j,j_high_unmapped,1) );
	}
#pragma omp barrier

	for(int i0 = 0; i0 < nmodes_l; i0+=bi){
	  int iup = std::min(i0+bi,nmodes_l);
	    
	  for(int j0 = 0; j0< nmodes_r; j0+=bj) {
	    int jup = std::min(j0+bj,nmodes_r);

	    for(int p0 = 0; p0 < size_3d; p0+=bp){
	      int pup = std::min(p0+bp,size_3d);
      
	      int thr_pwork, thr_poff;
	      thread_work(thr_pwork, thr_poff, pup-p0, me, omp_get_num_threads());

	      int thr_p0 = p0 + thr_poff;
#ifdef USE_INNER_BLOCKING
	      MultKernel<mf_Policies,A2AfieldL,A2AfieldR,mf_Element,mf_Element_Vector>::inner_block_mult(mf_accum_thr[me], M, t,
											  i0, iup, j0, jup, thr_p0, thr_p0+thr_pwork,
											  base_ptrs_i, base_ptrs_j, site_offsets_i, site_offsets_j);
#else
	      MultKernel<mf_Policies,A2AfieldL,A2AfieldR,mf_Element,mf_Element_Vector>::mult_kernel(mf_accum_thr[me], M, t,
										     i0, iup, j0, jup, thr_p0, thr_p0+thr_pwork,
										     base_ptrs_i, base_ptrs_j, site_offsets_i, site_offsets_j);
#endif

	    }
	  
	  }
	}
#pragma omp barrier

	const int nthread = omp_get_num_threads();
	const int ijwork = nmodes_l * nmodes_r;
	int thr_ijwork, thr_ijoff;
	thread_work(thr_ijwork, thr_ijoff, ijwork, me, nthread);
	for(int ij=thr_ijoff; ij<thr_ijoff + thr_ijwork; ij++){  //ij = j + mf_t[t].nmodes_r * i
	  int i=ij / nmodes_r;
	  int j=ij % nmodes_r;
	  this->sumThreadedResults(mf_t,mf_accum_thr,i,j,t,nthread);
	}		
      
      }//end of parallel region

      __SSC_MARK(0x2);
#endif //memtest mode
      std::ostringstream os; os << "timeslice " << t << " from range " << GJP.TnodeCoor()*GJP.TnodeSites() << " to " << (GJP.TnodeCoor()+1)*GJP.TnodeSites()-1 << " : " << nmodes_l << "*" <<  nmodes_r << " modes and inner p loop of size " <<  size_3d <<  " divided over " << omp_get_max_threads() << " threads";
      print_time("A2AmesonField",os.str().c_str(),ttime + dclock());
    }

    print_time("A2AmesonField","local compute",time + dclock());

    time = -dclock();
    sync();
    print_time("A2AmesonField","sync",time + dclock());

    //Accumulate
    time = -dclock();
#ifndef MEMTEST_MODE
    this->nodeSum(mf_t,Lt);
#endif
    print_time("A2AmesonField","nodeSum",time + dclock());
  }
};




template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename InnerProduct, typename Allocator, typename ComplexClass>
struct _choose_vector_policies{};

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename InnerProduct, typename Allocator>
struct _choose_vector_policies<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct,Allocator,complex_double_or_float_mark>{
  typedef SingleSrcVectorPolicies<mf_Policies, A2AfieldL, A2AfieldR, Allocator, InnerProduct> SingleSrcVectorPoliciesT;
  typedef MultiSrcVectorPolicies<mf_Policies, A2AfieldL, A2AfieldR, Allocator, InnerProduct> MultiSrcVectorPoliciesT;
};

#ifdef USE_GRID
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename InnerProduct, typename Allocator>
struct _choose_vector_policies<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct,Allocator,grid_vector_complex_mark>{
  // typedef SingleSrcVectorPoliciesSIMD<mf_Policies, A2AfieldL, A2AfieldR, Allocator, InnerProduct> SingleSrcVectorPoliciesT;
  // typedef MultiSrcVectorPoliciesSIMD<mf_Policies, A2AfieldL, A2AfieldR, Allocator, InnerProduct> MultiSrcVectorPoliciesT;
  typedef SingleSrcVectorPolicies<mf_Policies, A2AfieldL, A2AfieldR, Allocator, InnerProduct> SingleSrcVectorPoliciesT;
  typedef MultiSrcVectorPolicies<mf_Policies, A2AfieldL, A2AfieldR, Allocator, InnerProduct> MultiSrcVectorPoliciesT;
};
#endif



template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
template<typename InnerProduct, typename Allocator>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::compute(std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator > &mf_t,
							     const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
  typedef typename _choose_vector_policies<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct,Allocator, typename ComplexClassify<typename mf_Policies::ComplexType>::type>::SingleSrcVectorPoliciesT VectorPolicies;  
  mfComputeGeneral<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct, VectorPolicies> cg;
  cg.compute(mf_t,l,M,r,do_setup);
}

  //Version of the above for multi-src inner products (output vector indexed by [src idx][t]
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
template<typename InnerProduct, typename Allocator>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::compute(std::vector< std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator >* > &mf_st,
							     const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
  typedef typename _choose_vector_policies<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct,Allocator, typename ComplexClassify<typename mf_Policies::ComplexType>::type>::MultiSrcVectorPoliciesT VectorPolicies;  
  mfComputeGeneral<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct, VectorPolicies> cg;
  cg.compute(mf_st,l,M,r,do_setup);
}


#endif
