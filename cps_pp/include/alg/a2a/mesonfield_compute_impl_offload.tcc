#ifndef _MESONFIELD_COMPUTE_IMPL_OFFLOAD
#define _MESONFIELD_COMPUTE_IMPL_OFFLOAD

#ifdef GRID_NVCC

template<typename ComplexType>
__global__ void reduceKernel(typename ComplexType::scalar_type* into, ComplexType const* from, const size_t nib, const size_t njb, const size_t bj, const size_t size_3d, const size_t multiplicity){
  constexpr int nsimd = ComplexType::Nsimd();
  __shared__ typename SIMT<ComplexType>::value_type thrbuf[nsimd];
  int ii = blockIdx.x;
  int jj = blockIdx.y;
  int m = blockIdx.z;
  int lane = threadIdx.x;

  //input off = m + multiplicity*(x + size_3d*(jj + bj*ii))
  //output off = m + multiplicity*(jj + bj*ii)
  
  ComplexType const* fb = from + m + multiplicity*size_3d*(jj + bj*ii); //x=0
  typename ComplexType::scalar_type* ib = into + m + multiplicity*(jj + njb*ii);

  //Each thread sums its lane into a temp shared buffer
  typename SIMT<ComplexType>::value_type &v = thrbuf[lane];
  v = SIMT<ComplexType>::read(fb[0],lane);
  
  for(size_t x=1; x<size_3d; x++){
    v += SIMT<ComplexType>::read(fb[multiplicity*x],lane);
  }
  __syncthreads();

  //Thread 0 sums over the temp buffer
  if(lane == 0){
    *ib = thrbuf[0];
    for(int i=1; i<nsimd; i++) *ib += thrbuf[i];
  }
}

template<typename ComplexType>
void blockReduce(typename ComplexType::scalar_type* into, ComplexType const* from, const size_t nib, const size_t njb, const size_t bj, const size_t size_3d, const size_t multiplicity){
  //We will work with 1 thread per block and blocks over a 3d grid   nij x njb x multiplicity
  //Each thread does thr reduction for a single element over the whole 3d grid
  dim3 blocks(nib, njb, multiplicity);
  constexpr int nsimd = ComplexType::Nsimd();
  
  reduceKernel<<< blocks, nsimd>>>(into, from, nib, njb, bj, size_3d, multiplicity);
  cudaDeviceSynchronize();

  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err ) {
    printf("blockReduce: Cuda error %s\n",cudaGetErrorString( err ));
    exit(0);
  }
}

#endif


template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename Allocator, typename InnerProduct>
struct SingleSrcVectorPoliciesSIMDoffload{
  typedef std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator > mfVectorType;
  typedef Grid::vComplexD accumType;
  typedef Grid::vComplexD& accessType; //used by the source
  accelerator_inline static accessType getAccessor(accumType *p){ return *p; }

  static accelerator_inline int accumMultiplicity(){ return 1;}

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

  //Sum over x and SIMD reduce
  static inline void reduce(mfVectorType &mf_t, accumType const* accum,
			    const size_t i0, const size_t j0, //block index
			    const size_t nib, const size_t njb, //true size of this block
			    const size_t bj, //size of block. If block size not an even divisor of the number of modes, the above will differ from this for the last block 
			    const int t, const size_t size_3d){

#ifdef GRID_NVCC
    double talloc_free = 0;
    double tkernel = 0;
    double tpoke = 0;
  
    const int multiplicity = 1;

    double time = dclock();
    Grid::ComplexD* tmp = (Grid::ComplexD*)managed_alloc_check(nib * njb * multiplicity * sizeof(Grid::ComplexD));
    talloc_free += dclock() - time;
    time = dclock();
    blockReduce(tmp, accum, nib, njb, bj, size_3d, multiplicity);
    tkernel += dclock() - time;

    time = dclock();
#pragma omp parallel for
    for(size_t z=0; z < nib*njb; z++){
      size_t jj = z % njb;
      size_t ii = z / njb;

      size_t i = ii+i0;
      size_t j = jj+j0;

      mf_t[t](i,j) += tmp[jj + njb*ii];
    }
    tpoke += dclock() - time;

    time = dclock();
    managed_free(tmp);
    talloc_free += dclock() - time;

    print_time("GPU reduce","alloc_free",talloc_free);
    print_time("GPU reduce","kernel",tkernel);
    print_time("GPU reduce","poke",tpoke);
#else
    //Reduce over size_3d
    //Paralllelize me!
    for(int ii=0;ii<nib;ii++)
      for(int jj=0;jj<njb;jj++){
	size_t i = ii+i0;
	size_t j = jj+j0;
	accumType const* from_base = accum + size_3d*(jj + bj*ii);
	for(int x=0;x<size_3d;x++)
	  mf_t[t](i,j) += Reduce(from_base[x]);
      }
#endif
  }
  
};


template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename Allocator, typename InnerProduct>
struct MultiSrcVectorPoliciesSIMDoffload{
  typedef std::vector<std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator >* > mfVectorType;
  typedef Grid::vComplexD accumType;
  typedef Grid::vComplexD* accessType; //used by the source
  accelerator_inline static accessType getAccessor(accumType *p){ return p; }
  
  int mfPerTimeSlice;
  
  accelerator_inline int accumMultiplicity() const{ return mfPerTimeSlice;}

  enum { Nsimd = accumType::Nsimd() }; //should be 1 for scalar types

  inline void setupPolicy(const mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r){
    mfPerTimeSlice = M.mfPerTimeSlice();
    if(!UniqueID()){ printf("Using MultiSrcVectorPoliciesSIMDoffload with $MF per timeslice %d\n", mfPerTimeSlice); fflush(stdout); }
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

  inline const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> & getReferenceMf(const mfVectorType &mf_st, const int t) const{
    return mf_st[0]->operator[](t);
  }

  inline void nodeSum(mfVectorType &mf_st, const int Lt) const{
    for(int s=0;s<mfPerTimeSlice;s++)
      for(int t=0; t<Lt; t++) mf_st[s]->operator[](t).nodeSum();
  }

  
  //Sum over x and SIMD reduce
  inline void reduce(mfVectorType &mf_st, accumType const* accum,
		     const size_t i0, const size_t j0, //block index
		     const size_t nib, const size_t njb, //true size of this block
		     const int bj, //size of block. If block size not an even divisor of the number of modes, the above will differ from this for the last block 
		     const int t, const size_t size_3d) const{
#ifdef GRID_NVCC
    double talloc_free = 0;
    double tkernel = 0;
    double tpoke = 0;
  
    const int multiplicity = mfPerTimeSlice;

    double time = dclock();
    Grid::ComplexD* tmp = (Grid::ComplexD*)managed_alloc_check(nib * njb * multiplicity * sizeof(Grid::ComplexD));
    talloc_free += dclock() - time;
    time = dclock();
    blockReduce(tmp, accum, nib, njb, bj, size_3d, multiplicity);
    tkernel += dclock() - time;

    time = dclock();
#pragma omp parallel for
    for(size_t z=0; z < nib*njb*multiplicity; z++){
      size_t rem = z;     
      size_t m = rem % multiplicity; rem /= multiplicity;
      size_t jj = rem % njb; rem /= njb;
      size_t ii = rem;
      
      size_t i = ii+i0;
      size_t j = jj+j0;

      mf_st[m]->operator[](t)(i,j) += tmp[m + multiplicity *(jj + njb*ii)];
    }
    tpoke += dclock() - time;

    time = dclock();
    managed_free(tmp);
    talloc_free += dclock() - time;

    print_time("GPU reduce","alloc_free",talloc_free);
    print_time("GPU reduce","kernel",tkernel);
    print_time("GPU reduce","poke",tpoke);
#else
    //Reduce over size_3d
    //(Do this on host for now) //GENERALIZE ME
    for(int ii=0;ii<nib;ii++)
      for(int jj=0;jj<njb;jj++){
	size_t i = ii+i0;
	size_t j = jj+j0;
	accumType const* from_base = accum + mfPerTimeSlice*size_3d*(jj + bj*ii);
	for(int x=0;x<size_3d;x++)
	  for(int s=0;s<mfPerTimeSlice;s++)
	    mf_st[s]->operator[](t)(i,j) += Reduce(from_base[s + mfPerTimeSlice*x]);    
      }
#endif
  }

  
  
};



template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename InnerProduct, typename mfVectorPolicies>
struct mfComputeGeneralOffload: public mfVectorPolicies{
  typedef typename mfVectorPolicies::mfVectorType mfVectorType;
  typedef typename mfVectorPolicies::accumType accumType;
  enum { Nsimd = mfVectorPolicies::Nsimd };

  template<typename T>
  inline T* Alloc(size_t n){
    T* p = (T*)managed_alloc_check(128, n*sizeof(T));
    return p;
  }
  template<typename T>
  inline void Free(T* p){
    managed_free((void*)p);
  }

    
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
    vPtr *base_ptrs_i = Alloc<vPtr>(nmodes_l);
    vPtr *base_ptrs_j = Alloc<vPtr>(nmodes_r);

    typedef std::pair<int,int> offsetT;
    offsetT *site_offsets_i = Alloc<offsetT>(nmodes_l);
    offsetT *site_offsets_j = Alloc<offsetT>(nmodes_r);
 
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
    const size_t multiplicity = this->accumMultiplicity();
    const size_t naccum = bi * bj * size_3d * multiplicity;
    accumType *accum = Alloc<accumType>(naccum);

    if(!UniqueID()){ printf("Using block sizes %d %d, temp memory requirement is %f MB\n", bi, bj, byte_to_MB(naccum * sizeof(accumType))); }
    
    double reduce_time = 0;
    
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
	    
	  {
	  accelerator_for(item, nwork, Nsimd, 
			  {
			    size_t rem = item;
			    size_t x = rem % size_3d; rem /= size_3d;
			    size_t jj = rem % njb; rem /= njb;
			    size_t ii = rem;
			    
			    size_t i = ii+i0;
			    size_t j = jj+j0;

			    accumType *into = accum + multiplicity*(x + size_3d*(jj + bj*ii));
			    vPtr lptr = base_ptrs_i[i]; lptr.incrementPointers(site_offsets_i[i], x);
			    vPtr rptr = base_ptrs_j[j]; rptr.incrementPointers(site_offsets_j[j], x);

			    typename mfVectorPolicies::accessType acc = this->getAccessor(into);
			    
			    M(acc,lptr,rptr,x,t);
			  });
	  }   

	  double treduce = -dclock();
	  this->reduce(mf_t, accum, i0, j0, nib, njb, bj, t, size_3d);
	  treduce += dclock();

	  reduce_time += treduce;
	}
      }
 
#endif //memtest mode
      std::ostringstream os; os << "timeslice " << t << " from range " << GJP.TnodeCoor()*GJP.TnodeSites() << " to " << (GJP.TnodeCoor()+1)*GJP.TnodeSites()-1 << " : " << nmodes_l << "*" <<  nmodes_r << " modes and inner p loop of size " <<  size_3d <<  " divided over " << omp_get_max_threads() << " threads";
      print_time("A2AmesonField",os.str().c_str(),ttime + dclock());
    }

    print_time("A2AmesonField","local compute",time + dclock());
    print_time("A2AmesonField","reduce time in local compute",reduce_time);
    
    time = -dclock();
    sync();
    print_time("A2AmesonField","sync",time + dclock());

    //Accumulate
    time = -dclock();
#ifndef MEMTEST_MODE
    this->nodeSum(mf_t,Lt);
#endif
    print_time("A2AmesonField","nodeSum",time + dclock());
    
    Free(base_ptrs_i);
    Free(base_ptrs_j);
    Free(site_offsets_i);
    Free(site_offsets_j);
    Free(accum);
  }
};


//Use offload only if A2A policies is Grid vectorized


template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename InnerProduct, typename Allocator, typename ComplexClass>
struct _choose_mf_mult_impl_offload;

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename InnerProduct, typename Allocator>
struct _choose_mf_mult_impl_offload<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct,Allocator,grid_vector_complex_mark>{
  typedef SingleSrcVectorPoliciesSIMDoffload<mf_Policies, A2AfieldL, A2AfieldR, Allocator, InnerProduct> VectorPoliciesSingle;
  typedef mfComputeGeneralOffload<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct, VectorPoliciesSingle> ComputeImplSingle;

  typedef MultiSrcVectorPoliciesSIMDoffload<mf_Policies, A2AfieldL, A2AfieldR, Allocator, InnerProduct> VectorPoliciesMulti;
  typedef mfComputeGeneralOffload<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct, VectorPoliciesMulti> ComputeImplMulti;
};

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename InnerProduct, typename Allocator>
struct _choose_mf_mult_impl_offload<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct,Allocator,complex_double_or_float_mark>{
  typedef SingleSrcVectorPolicies<mf_Policies, A2AfieldL, A2AfieldR, Allocator, InnerProduct> VectorPoliciesSingle;
  typedef mfComputeGeneral<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct, VectorPoliciesSingle> ComputeImplSingle;
  
  typedef MultiSrcVectorPolicies<mf_Policies, A2AfieldL, A2AfieldR, Allocator, InnerProduct> VectorPoliciesMulti;
  typedef mfComputeGeneral<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct, VectorPoliciesMulti> ComputeImplMulti;
};










#endif
