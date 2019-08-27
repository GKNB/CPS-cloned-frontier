#ifndef _MESONFIELD_COMPUTE_IMPL_OFFLOAD
#define _MESONFIELD_COMPUTE_IMPL_OFFLOAD


template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename Allocator, typename InnerProduct>
struct SingleSrcVectorPoliciesSIMDoffload{
  typedef std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator > mfVectorType;
  typedef Grid::vComplexD accumType;
  inline const int accumMultiplicity() const{ return 1;}

  enum { Nsimd = accumType::Nsimd() }; //should be 1 for scalar types

  static inline void setupPolicy(const mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r){ 
    if(!UniqueID()){ printf("Using SingleSrcVectorPoliciesSIMDoffload\n"); fflush(stdout); }
    assert(M.mfPerTimeSlice() == 1); 
  }

  static void initializeMesonFields(mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const A2AfieldR<mf_Policies> &r, const int Lt, const bool do_setup){
    mf_t.resize(Lt);
    for(int t=0;t<Lt;t++) 
      if(do_setup) mf_t[t].setup(l,r,t,t); //both vectors have same timeslice (zeroes the starting matrix)
      else{
	assert(mf_t[t].ptr() != NULL);
	mf_t[t].zero();
      }
  }
  //Used to get information about rows and cols
  static inline const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> & getReferenceMf(const mfVectorType &mf_t, const int t){
    return mf_t[t];
  }
  static inline void nodeSum(mfVectorType &mf_t, const int Lt){
    for(int t=0; t<Lt; t++) mf_t[t].nodeSum();
  }
};

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename InnerProduct, typename mfVectorPolicies>
struct mfComputeGeneralOffload: public mfVectorPolicies{
  typedef typename mfVectorPolicies::mfVectorType mfVectorType;
  typedef typename mfVectorPolicies::accumType accumType;
  enum { Nsimd = mfVectorPolicies::Nsimd };
  
  void compute(mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
    this->setupPolicy(mf_t,l,M,r);
    
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    if(!UniqueID()) printf("Starting A2AmesonField::compute (blocked) for %d timeslices with %d threads\n",Lt, omp_get_max_threads());
    if(!UniqueID()) printMem("mfComputeGeneralOffload node 0 memory status",0);

    cps::sync();

    double time = -dclock();
    this->initializeMesonFields(mf_t,l,r,Lt,do_setup);
    print_time("A2AmesonField","setup",time + dclock());

    time = -dclock();
    //For W vectors we dilute out the flavor index in-place while performing this contraction
    const typename mf_Policies::FermionFieldType &mode0 = l.getMode(0);
    const size_t size_3d = mode0.nodeSites(0)*mode0.nodeSites(1)*mode0.nodeSites(2);
    if(mode0.nodeSites(3) != GJP.TnodeSites()) ERR.General("A2AmesonField","compute","Not implemented for fields where node time dimension != GJP.TnodeSites()\n");

    for(int t=1;t<Lt;t++){
      assert(this->getReferenceMf(mf_t,t).getRowParams().paramsEqual(this->getReferenceMf(mf_t,0).getRowParams() ) );
      assert(this->getReferenceMf(mf_t,t).getColParams().paramsEqual(this->getReferenceMf(mf_t,0).getColParams() ) );
    }

    const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> & mf_ref = this->getReferenceMf(mf_t,0); //assumes all meson fields of the mf_Element type have the same mode parameters
    const size_t nl_l = mf_ref.getRowParams().getNl();
    const size_t nl_r = mf_ref.getColParams().getNl();
    const size_t nmodes_l = mf_ref.getNrows();
    const size_t nmodes_r = mf_ref.getNcols();

    const size_t bi = BlockedMesonFieldArgs::bi, bj = BlockedMesonFieldArgs::bj, bp = BlockedMesonFieldArgs::bp;

    //Make a table of p base pointers and site offsets (stride between 3d sites) for each i,j
    //These need to be in managed memory so they can be accessed on the device
    typedef SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> vPtr;
    Grid::alignedAllocator<vPtr> vPtr_alloc;
    vPtr *base_ptrs_i = vPtr_alloc.allocate(nmodes_l);
    vPtr *base_ptrs_j = vPtr_alloc.allocate(nmodes_r);

    typedef std::pair<int,int> offsetT;
    Grid::alignedAllocator<offsetT> offsetT_alloc;
    offsetT *site_offsets_i = offsetT_alloc.allocate(nmodes_l);
    offsetT *site_offsets_j = offsetT_alloc.allocate(nmodes_r);
 
    //Total number of work items is nmodes_l * nmodes_r * size_3d, and kernel is M
    //A reduction is performed over the 3d site
    //Access pattern should be blocked to ensure cache reuse of rows and columns

    //In Grid's model, for CUDA, the number of gpu threads is a global variable. The number of work items per block is fixed to gpu_threads * nsimd in a 2d array
    //and the number of blocks is scaled to the problem

    //If each work item writes to a separate memory location we need  nmodes_l * nmodes_r * size_3d temporary storage, which is far too big. We thus need to divide the
    //problem into smaller blocks of size   nl_block * nr_block * size3d,   where nl_block is tuned such that the temporaries fit in GPU memory
    //We will use BlockedMesonFieldArgs::bi and BlockedMesonFieldArgs::bj for this purpose

    //Note, block sizes will depend on the multiplicity of the accumulation type

    //Allocate work item temp memory
    size_t naccum = bi * bj * size_3d * this->accumMultiplicity();
    Grid::alignedAllocator<accumType> accum_alloc;
    accumType *accum = accum_alloc.allocate(naccum); 

    //Each node only works on its time block
    for(int t=GJP.TnodeCoor()*GJP.TnodeSites(); t<(GJP.TnodeCoor()+1)*GJP.TnodeSites(); t++){   
      double ttime = -dclock();
      const int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();

#ifndef MEMTEST_MODE

      //Generate the offset and pointer tables on the host
      thread_for(i, nmodes_l, 
		 {
		   modeIndexSet i_high_unmapped; if(i>=nl_l) mf_ref.getRowParams().indexUnmap(i-nl_l,i_high_unmapped);
		   base_ptrs_i[i] = l.getFlavorDilutedVect(i,i_high_unmapped,0,t_lcl);
		   site_offsets_i[i] = offsetT( l.siteStride3D(i,i_high_unmapped,0), l.siteStride3D(i,i_high_unmapped,1) );
		 });
      thread_for(j, nmodes_r, 
		 {
		   modeIndexSet j_high_unmapped; if(j>=nl_r) mf_ref.getColParams().indexUnmap(j-nl_r,j_high_unmapped);
		   base_ptrs_j[j] = r.getFlavorDilutedVect(j,j_high_unmapped,0,t_lcl);
		   site_offsets_j[j] = offsetT( r.siteStride3D(j,j_high_unmapped,0), r.siteStride3D(j,j_high_unmapped,1) );
		 });

      for(size_t i0 = 0; i0 < nmodes_l; i0+=bi){
	size_t iup = std::min(i0+bi,nmodes_l);
	
	for(size_t j0 = 0; j0< nmodes_r; j0+=bj) {
	  size_t jup = std::min(j0+bj,nmodes_r);
      
	  memset(accum, 0, naccum * sizeof(accumType));

	  //Number of work items is (iup - i0) * (jup - j0) * size_3d
	  size_t nib = iup - i0;
	  size_t njb = jup - j0;

	  size_t nwork = nib * njb * size_3d;
	  
	  accelerator_for(item, nwork, Nsimd, 
			  {
			    size_t rem = item;
			    size_t x = rem % size_3d; rem /= size_3d;
			    size_t jj = rem % njb; rem /= njb;
			    size_t ii = rem;
			    
			    size_t i = ii+i0;
			    size_t j = jj+j0;

			    accumType *into = accum + this->accumMultiplicity()*(x + size_3d*(jj + bj*ii));
			    vPtr lptr = base_ptrs_i[i]; lptr.incrementPointers(site_offsets_i[i], x);
			    vPtr rptr = base_ptrs_j[j]; rptr.incrementPointers(site_offsets_j[j], x);

			    M(*into,lptr,rptr,x,t);
			  });

	  //Reduce over size_3d
	  //(Do this on host for now) //GENERALIZE ME
	  assert(this->accumMultiplicity()==1);
	  for(int ii=0;ii<nib;ii++)
	    for(int jj=0;jj<njb;jj++){
	      size_t i = ii+i0;
	      size_t j = jj+j0;
	      accumType const* from_base = accum + size_3d*(jj + bj*ii);
	      for(int x=0;x<size_3d;x++)
		mf_t[t](i,j) += Reduce(from_base[x]);
	    }
	}
      }
 
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

    vPtr_alloc.deallocate(base_ptrs_i, nmodes_l);
    vPtr_alloc.deallocate(base_ptrs_j, nmodes_r);
    offsetT_alloc.deallocate(site_offsets_i, nmodes_l);
    offsetT_alloc.deallocate(site_offsets_j, nmodes_r);
    accum_alloc.deallocate(accum, naccum); 

    print_time("A2AmesonField","nodeSum",time + dclock());
  }
};


#endif
