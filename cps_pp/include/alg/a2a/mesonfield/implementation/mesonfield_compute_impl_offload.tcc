#ifndef _MESONFIELD_COMPUTE_IMPL_OFFLOAD
#define _MESONFIELD_COMPUTE_IMPL_OFFLOAD

//Implementation of offloaded reduction kernel only CUDA and HIP currently

#ifdef GRID_HIP

template<typename ComplexType>
__global__ void reduceKernel(typename ComplexType::scalar_type* into, ComplexType const* from, const size_t bi_true, const size_t bj_true, const size_t bj, const size_t size_3d, const size_t multiplicity)
{
#ifdef GRID_SIMT
//FIXME: at this point we have to put the code inside GRID_SIMT condition in order to make sure that it is compiled on for GPU	
  constexpr int nsimd = ComplexType::Nsimd();
  __shared__ typename SIMT<ComplexType>::value_type thrbuf[nsimd];
  int ii = blockIdx.x;
  int jj = blockIdx.y;
  int m = blockIdx.z;
  int lane = threadIdx.x;

  //input off = m + multiplicity*(x + size_3d*(jj + bj*ii))
  //output off = m + multiplicity*(jj + bj*ii)
  
  ComplexType const* fb = from + m + multiplicity*size_3d*(jj + bj*ii); //x=0
  typename ComplexType::scalar_type* ib = into + m + multiplicity*(jj + bj_true*ii);

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

#else
#warning "The reduceKernel is compiled for host! We don't want that happen!"
#endif
}

template<typename ComplexType>
void blockReduce(typename ComplexType::scalar_type* into, ComplexType const* from, const size_t bi_true, const size_t bj_true, const size_t bj, const size_t size_3d, const size_t multiplicity){
  //We will work with 1 thread per block and blocks over a 3d grid   nij x bj_true x multiplicity
  //Each thread does thr reduction for a single element over the whole 3d grid
  dim3 blocks(bi_true, bj_true, multiplicity);
  constexpr int nsimd = ComplexType::Nsimd();
  
  reduceKernel<<< blocks, nsimd>>>(into, from, bi_true, bj_true, bj, size_3d, multiplicity);
  hipDeviceSynchronize();

  hipError_t err = hipGetLastError();
  if ( hipSuccess != err ) {
    printf("blockReduce: Hip error %s\n",hipGetErrorString( err ));
    exit(0);
  }
}

//end of GRID_HIP
#elif defined(GRID_CUDA)

CPS_END_NAMESPACE
#include <cuda_profiler_api.h>
//#include<int_fastdiv.h>
CPS_START_NAMESPACE

template<typename ComplexType>
__global__ void reduceKernel(typename ComplexType::scalar_type* into, ComplexType const* from, const size_t bi_true, const size_t bj_true, const size_t bj, const size_t size_3d, const size_t multiplicity){
  constexpr int nsimd = ComplexType::Nsimd();
  __shared__ typename SIMT<ComplexType>::value_type thrbuf[nsimd];
  int ii = blockIdx.x;
  int jj = blockIdx.y;
  int m = blockIdx.z;
  int lane = threadIdx.x;

  //input off = m + multiplicity*(x + size_3d*(jj + bj*ii))
  //output off = m + multiplicity*(jj + bj_true*ii)
  
  ComplexType const* fb = from + m + multiplicity*size_3d*(jj + bj*ii); //x=0
  typename ComplexType::scalar_type* ib = into + m + multiplicity*(jj + bj_true*ii);

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
void blockReduce(typename ComplexType::scalar_type* into, ComplexType const* from, const size_t bi_true, const size_t bj_true, const size_t bj, const size_t size_3d, const size_t multiplicity){
  //We will work with 1 thread per block and blocks over a 3d grid   nij x bj_true x multiplicity
  //Each thread does thr reduction for a single element over the whole 3d grid
  dim3 blocks(bi_true, bj_true, multiplicity);
  constexpr int nsimd = ComplexType::Nsimd();
  
  reduceKernel<<< blocks, nsimd>>>(into, from, bi_true, bj_true, bj, size_3d, multiplicity);
  cudaDeviceSynchronize();

  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err ) {
    printf("blockReduce: Cuda error %s\n",cudaGetErrorString( err ));
    exit(0);
  }
}

#else //end of GRID_CUDA

//NB: into's size is bi_true*bj_true*multiplicity
template<typename ComplexType>
void blockReduce(typename ComplexType::scalar_type* into, ComplexType const* from, const size_t bi_true, const size_t bj_true, const size_t bj, const size_t size_3d, const size_t multiplicity){
  using namespace Grid;
  accelerator_for2d(ii, bi_true, jj, bj_true, 1, {
      for(int m=0; m<multiplicity; m++){
	ComplexType const* fb = from + m + multiplicity*size_3d*(jj + bj*ii); //x=0
	typename ComplexType::scalar_type* ib = into + m + multiplicity*(jj + bj_true*ii);

	*ib = Reduce(fb[0]);
	for(size_t x=1; x<size_3d; x++){
	  *ib = *ib + Reduce(fb[multiplicity*x]);
	}
      }
    });
}

#endif


//acc(int m):  return a view to the meson field for multiplicity index m
//m = {0..multiplicity-1}
template<typename AllocPolicy, typename accumType, typename ViewType>
void mesonFieldComputeReduce(accumType const* accum,
			     const size_t i0, const size_t j0, //block index
			     const size_t bi_true, const size_t bj_true, //true size of this block
			     const size_t bj, //size of block. If block size not an even divisor of the number of modes, the above will differ from this for the last block 
			     const size_t size_3d,
			     const int multiplicity, ViewType *mf_views){ //mf_views expected to be of size=multiplicity
  double talloc_free = 0;
  double tkernel = 0;
  double tpoke = 0;
  
  talloc_free -= dclock();
  VectorWithAview<Grid::ComplexD, AllocPolicy> tmp(bi_true * bj_true * multiplicity, AllocLocationPref::Device);
  talloc_free += dclock();
  
  tkernel -= dclock();
  {
    CPSautoView(tmp_v,tmp,DeviceWrite);
    blockReduce(tmp_v.data(), accum, bi_true, bj_true, bj, size_3d, multiplicity);
  }
  tkernel += dclock();
  
  tpoke -= dclock();
  {
    CPSautoView(tmp_v,tmp,HostRead);

#pragma omp parallel for
    for(size_t z=0; z < bi_true*bj_true*multiplicity; z++){
      size_t rem = z;     
      size_t m = rem % multiplicity; rem /= multiplicity;
      size_t jj = rem % bj_true; rem /= bj_true;
      size_t ii = rem;
      
      size_t i = ii+i0;
      size_t j = jj+j0;
      
      mf_views[m](i,j) += tmp_v[m + multiplicity *(jj + bj_true*ii)];
    }
  }
  tpoke += dclock();
}


template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename Allocator, typename InnerProduct>
struct SingleSrcVectorPoliciesSIMDoffload{
  typedef A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> MesonFieldType;
  typedef std::vector<MesonFieldType, Allocator > mfVectorType;
  typedef Grid::vComplexD accumType;
  typedef Grid::vComplexD& accessType; //used by the source
  accelerator_inline static accessType getAccessor(accumType *p){ return *p; }

  static accelerator_inline int accumMultiplicity(){ return 1;}

  enum { Nsimd = accumType::Nsimd() }; //should be 1 for scalar types

  static inline void setupPolicy(const mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r){ 
    LOGA2A << "Using SingleSrcVectorPoliciesSIMDoffload" << std::endl;
    assert(M.mfPerTimeSlice() == 1); 
  }

  static void initializeMesonFields(mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const A2AfieldR<mf_Policies> &r, const int Lt, const bool do_setup){
    mf_t.resize(Lt);
    for(int t=0;t<Lt;t++) 
      if(do_setup) mf_t[t].setup(l,r,t,t); //both vectors have same timeslice (zeroes the starting matrix)
      else{
	{
	  CPSautoView(mf_t_v,mf_t[t],HostRead);
	  assert(mf_t_v.ptr() != NULL);
	}
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

  inline void nodeSumPartialAsyncStart(std::vector<MesonFieldType*> &hinto, std::vector<nodeSumPartialAsyncHandle> &handles,  mfVectorType &mf_st, 
				       const int Lt, const int istart, const int ni, const int jstart, const int nj) const{
    for(int t=0; t<Lt; t++){
      MesonFieldType &mf = mf_st[t];
      hinto.push_back(&mf); handles.push_back(mf.nodeSumPartialAsync(istart,ni,jstart,nj));
    }
  }
  inline void nodeSumPartialAsyncComplete(std::vector<MesonFieldType*> &hinto, std::vector<nodeSumPartialAsyncHandle> &handles) const{
    for(int i=0;i<handles.size();i++) hinto[i]->nodeSumPartialComplete(handles[i]);
  }

  //Sum over x and SIMD reduce
  static inline void reduce(mfVectorType &mf_t, accumType const* accum,
			    const size_t i0, const size_t j0, //block index
			    const size_t bi_true, const size_t bj_true, //true size of this block
			    const size_t bj, //size of block. If block size not an even divisor of the number of modes, the above will differ from this for the last block 
			    const int t, const size_t size_3d){
    CPSautoView(mf_t_v,mf_t[t],HostReadWrite);
    mesonFieldComputeReduce<typename mf_Policies::AllocPolicy>(accum, i0, j0, bi_true, bj_true, bj, size_3d, 1, &mf_t_v);
  }
  
};


template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename Allocator, typename InnerProduct>
struct MultiSrcVectorPoliciesSIMDoffload{
  typedef A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> MesonFieldType;
  typedef std::vector<MesonFieldType, Allocator > mfTimeVector;
  typedef std::vector<mfTimeVector* > mfVectorType;
  typedef Grid::vComplexD accumType;
  typedef Grid::vComplexD* accessType; //used by the source
  accelerator_inline static accessType getAccessor(accumType *p){ return p; }
  
  int mfPerTimeSlice;
  
  accelerator_inline int accumMultiplicity() const{ return mfPerTimeSlice;}

  enum { Nsimd = accumType::Nsimd() }; //should be 1 for scalar types

  inline void setupPolicy(const mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r){
    mfPerTimeSlice = M.mfPerTimeSlice();
    a2a_printf("Using MultiSrcVectorPoliciesSIMDoffload with $MF per timeslice %d\n", mfPerTimeSlice);
  }

  void initializeMesonFields(mfVectorType &mf_st, const A2AfieldL<mf_Policies> &l, const A2AfieldR<mf_Policies> &r, const int Lt, const bool do_setup) const{
    if(mf_st.size() != mfPerTimeSlice) ERR.General("mf_Vector_policies <multi src>","initializeMesonFields","Expect output vector to be of size %d, got size %d\n",mfPerTimeSlice,mf_st.size());

    for(int s=0;s<mfPerTimeSlice;s++){
      mf_st[s]->resize(Lt);
      for(int t=0;t<Lt;t++) 
	if(do_setup) mf_st[s]->operator[](t).setup(l,r,t,t); //both vectors have same timeslice (zeroes the starting matrix)
	else{
	  {
	    CPSautoView(mf_st_v, (*mf_st[t])[t], HostRead);
	    assert(mf_st_v.ptr() != NULL);
	  }
	  mf_st[s]->operator[](t).zero();
	}
    }
  }

  inline const MesonFieldType & getReferenceMf(const mfVectorType &mf_st, const int t) const{
    return mf_st[0]->operator[](t);
  }

  inline void nodeSum(mfVectorType &mf_st, const int Lt) const{
    for(int s=0;s<mfPerTimeSlice;s++)
      for(int t=0; t<Lt; t++) mf_st[s]->operator[](t).nodeSum();
  }
  
  inline void nodeSumPartialAsyncStart(std::vector<MesonFieldType*> &hinto, std::vector<nodeSumPartialAsyncHandle> &handles,  mfVectorType &mf_st, 
				       const int Lt, const int istart, const int ni, const int jstart, const int nj) const{
    for(int s=0;s<mfPerTimeSlice;s++)
      for(int t=0; t<Lt; t++){
	MesonFieldType &mf = mf_st[s]->operator[](t);
	hinto.push_back(&mf); handles.push_back(mf.nodeSumPartialAsync(istart,ni,jstart,nj));
      }
  }
  inline void nodeSumPartialAsyncComplete(std::vector<MesonFieldType*> &hinto, std::vector<nodeSumPartialAsyncHandle> &handles) const{
    for(int i=0;i<handles.size();i++) hinto[i]->nodeSumPartialComplete(handles[i]);
  }

  //Sum over x and SIMD reduce
  inline void reduce(mfVectorType &mf_st, accumType const* accum,
		     const size_t i0, const size_t j0, //block index
		     const size_t bi_true, const size_t bj_true, //true size of this block
		     const int bj, //size of block. If block size not an even divisor of the number of modes, the above will differ from this for the last block 
		     const int t, const size_t size_3d) const{
    typedef typename MesonFieldType::View ViewType;
    ViewType* views = (ViewType*)memalign_check(128, mfPerTimeSlice*sizeof(ViewType));
    for(int m=0;m<mfPerTimeSlice;m++) new (views+m) ViewType( (*mf_st[m])[t].view(HostReadWrite) );
    mesonFieldComputeReduce<typename mf_Policies::AllocPolicy>(accum, i0, j0, bi_true, bj_true, bj, size_3d, mfPerTimeSlice, views);
    for(int m=0;m<mfPerTimeSlice;m++) views[m].free();
    free(views);
  }
  
};

//Starting at 'init', sucessively reduce the value 'out' until it is an integer divisor of 'of'
inline size_t nearestDivisor(const size_t of, const size_t init){
  size_t out = init;
  while(of % out != 0) --out;
  return out;
}


template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR, typename InnerProduct, typename mfVectorPolicies>
struct mfComputeGeneralOffload: public mfVectorPolicies{
  typedef typename mfVectorPolicies::mfVectorType mfVectorType;
  typedef typename mfVectorPolicies::accumType accumType;
  typedef typename mfVectorPolicies::MesonFieldType MesonFieldType;
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

  void compute_v3(mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
    double total_time = -dclock();
    this->setupPolicy(mf_t,l,M,r);
    
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    a2a_printf("Starting A2AmesonField::compute (GPU, blocked) for %d timeslices with %d threads\n",Lt, omp_get_max_threads());
    if(!UniqueID()) printMem("mfComputeGeneralOffload node 0 memory status",0);

    cps::sync();

    LOGA2A << "Initializing meson fields" << std::endl;
    double time = -dclock();
    this->initializeMesonFields(mf_t,l,r,Lt,do_setup); //assumed to zero meson fields
    a2a_print_time("A2AmesonField","setup",time + dclock());
   
    time = -dclock();
    //For W vectors we dilute out the flavor index in-place while performing this contraction
    const typename mf_Policies::FermionFieldType &mode0 = l.getMode(0);
    const size_t size_3d = mode0.nodeSites(0)*mode0.nodeSites(1)*mode0.nodeSites(2);
    if(mode0.nodeSites(3) != GJP.TnodeSites()) ERR.General("A2AmesonField","compute","Not implemented for fields where node time dimension != GJP.TnodeSites()\n");

    for(int t=1;t<Lt;t++){
      assert(this->getReferenceMf(mf_t,t).getRowParams().paramsEqual(this->getReferenceMf(mf_t,0).getRowParams() ) );
      assert(this->getReferenceMf(mf_t,t).getColParams().paramsEqual(this->getReferenceMf(mf_t,0).getColParams() ) );
    }
    const MesonFieldType & mf_ref = this->getReferenceMf(mf_t,0);
    const size_t nl_l = mf_ref.getRowParams().getNl();
    const size_t nl_r = mf_ref.getColParams().getNl();
    const size_t nmodes_l = mf_ref.getNrows();
    const size_t nmodes_r = mf_ref.getNcols();

    size_t bi = BlockedMesonFieldArgs::bi, bj = BlockedMesonFieldArgs::bj, bx = BlockedMesonFieldArgs::bp;
    if(bi > nmodes_l || bi == 0) bi = nmodes_l;
    if(bj > nmodes_r || bj == 0) bj = nmodes_r;
    if(bx > size_3d || bx == 0) bx = size_3d; //optional disable of x blocking

#ifdef MF_OFFLOAD_INNER_BLOCKING
    //Note these will be shrunk if necessary to be an exact divisor of the true block size, which can be smaller for the last block if the block size is not a divisor
    size_t sbi = BlockedMesonFieldArgs::bii, sbj = BlockedMesonFieldArgs::bjj, sbx = BlockedMesonFieldArgs::bpp;
    if(sbi == 0 || sbi > bi) sbi = bi;
    if(sbj == 0 || sbj > bj) sbj = bj;
    if(sbx == 0 || sbx > bx) sbx = bx;
#endif

    //Handles for async nodesum
#ifdef MF_OFFLOAD_NODESUM_ASYNC
    std::vector<MesonFieldType*> nodesum_hinto;
    std::vector<nodeSumPartialAsyncHandle> nodesum_handles;
#endif

    //Types for offset and pointer tables
    typedef SCFvectorPtr<typename mf_Policies::FermionFieldType::FieldSiteType> vPtr;
    typedef std::pair<int,int> offsetT;
    
    //Total number of work items is nmodes_l * nmodes_r * size_3d, and kernel is M
    //A reduction is performed over the 3d site
    //Access pattern should be blocked to ensure cache reuse of rows and columns

    //In Grid's model, for CUDA, the number of gpu threads is a global variable. The number of work items per block is fixed to gpu_threads * nsimd in a 2d array
    //and the number of blocks is scaled to the problem

    //If each work item writes to a separate memory location we need  nmodes_l * nmodes_r * size_3d temporary storage, which is far too big. We thus need to divide the
    //problem into smaller blocks of size   nl_block * nr_block * np_block,   where nl_block is tuned such that the temporaries fit in GPU memory
    //We will use BlockedMesonFieldArgs::bi, BlockedMesonFieldArgs::bj and BlockedMesonFieldArgs::bp for this purpose

    //Note, block sizes will depend on the multiplicity of the accumulation type
    

    //Allocate work item temp memory
    const size_t multiplicity = this->accumMultiplicity();
    const size_t naccum = bi * bj * bx * multiplicity;

    VectorWithAview<accumType, typename mf_Policies::AllocPolicy> accum(naccum, AllocLocationPref::Device);   
    VectorWithAview<vPtr, typename mf_Policies::AllocPolicy> base_ptrs_i(bi), base_ptrs_j(bj);
    VectorWithAview<offsetT, typename mf_Policies::AllocPolicy> site_offsets_i(bi), site_offsets_j(bj);

    LOGA2A << "Using block sizes " << bi << " " << bj << " " << bx << ", temp memory requirement is " << byte_to_MB(naccum * sizeof(accumType)) << " MB" << std::endl;
    

#ifdef MF_OFFLOAD_INNER_BLOCKING
    LOGA2A << "Using inner block sizes " << sbi << " " << sbj << " " << sbx << std::endl;
#endif
    size_t nioblocks = (nmodes_l + bi-1)/bi,  njoblocks = (nmodes_r + bj-1)/bj,  nxoblocks = (size_3d + bx-1)/bx;
    LOGA2A << "Number of outer blocks " << nioblocks << " " << njoblocks << " " << nxoblocks << std::endl;

    std::vector<int> il_map(nmodes_l), jr_map(nmodes_r);
    thread_for(i, nmodes_l, 
	       {
		 if(i<nl_l) il_map[i] = i;
		 else{		   
		   modeIndexSet i_high_unmapped; mf_ref.getRowParams().indexUnmap(i-nl_l,i_high_unmapped);
		   il_map[i] = nl_l + l.indexMap(i_high_unmapped);
		 }
	       });
    thread_for(j, nmodes_r, 
	       {
		 if(j<nl_r) jr_map[j] = j;
		 else{		   
		   modeIndexSet j_high_unmapped; mf_ref.getColParams().indexUnmap(j-nl_r,j_high_unmapped);
		   jr_map[j] = nl_r + r.indexMap(j_high_unmapped);
		 }
	       });

    
    CPSautoView(M_v,M,DeviceRead);
    CPSautoView(accum_v,accum,DeviceWrite); //keep open throughout
    
    double reduce_time = 0;
    double ptr_setup_time = 0;
    double kernel_time = 0;
    double copy_prefetch_time = 0;
    double prefetch_wait_time = 0;
    double async_nodesum_init_time = 0;

#ifndef MEMTEST_MODE     
    for(size_t i0 = 0; i0 < nmodes_l; i0+=bi){
      LOGA2A << "i-block " << i0/bi << "/" << nioblocks << std::endl;
      
      size_t iup = std::min(i0+bi,nmodes_l);
      size_t bi_true = iup - i0;
#ifdef MF_OFFLOAD_INNER_BLOCKING
      int sbi_use = nearestDivisor(bi_true, sbi);
      int niblk = bi_true / sbi_use;
#endif

      //Open view to required l fields here, and only close after inner loop so it stays on the device
      copy_prefetch_time -= dclock();
      std::vector<bool> lmodes_used(l.getNmodes(),false);
      for(int i=i0;i<iup;i++) lmodes_used[il_map[i]] = true;
      auto l_v = l.view(DeviceRead, lmodes_used);
      copy_prefetch_time += dclock();
	
      for(size_t j0 = 0; j0< nmodes_r; j0+=bj) {
	LOGA2A << "j-block " << j0/bj << "/" << njoblocks << std::endl;
	size_t jup = std::min(j0+bj,nmodes_r);
	size_t bj_true = jup - j0;
#ifdef MF_OFFLOAD_INNER_BLOCKING
	int sbj_use = nearestDivisor(bj_true, sbj);
	int njblk = bj_true / sbj_use;
#endif
	  
	//Get view for r
	copy_prefetch_time -= dclock();
	std::vector<bool> rmodes_used(r.getNmodes(),false);
	for(int j=j0;j<jup;j++) rmodes_used[jr_map[j]] = true;
	auto r_v = r.view(DeviceRead, rmodes_used);

	//Prefetch next block(s)
	{ 
	  size_t j0_nxt = j0+bj;
	  size_t i0_nxt = i0+bi;
	  if(j0_nxt < nmodes_r){
	    size_t jup_nxt = std::min(j0_nxt+bj,nmodes_r);
	    std::vector<bool> rmodes_used_nxt(r.getNmodes(),false);
	    for(int j=j0_nxt;j<jup_nxt;j++) rmodes_used_nxt[jr_map[j]] = true;
	    r.enqueuePrefetch(DeviceRead,rmodes_used_nxt);	    
	    A2AfieldR<mf_Policies>::startPrefetches();
	  }else if(i0_nxt < nmodes_l){
	    j0_nxt = 0; //loops back round
	    size_t jup_nxt = std::min(j0_nxt+bj,nmodes_r);
	    std::vector<bool> rmodes_used_nxt(r.getNmodes(),false);
	    for(int j=j0_nxt;j<jup_nxt;j++) rmodes_used_nxt[jr_map[j]] = true;
	    r.enqueuePrefetch(DeviceRead,rmodes_used_nxt);	    
	    
	    //prefetch next i block also!
	    size_t iup_nxt = std::min(i0_nxt+bi,nmodes_l);
	    std::vector<bool> lmodes_used_nxt(l.getNmodes(),false);
	    for(int i=i0_nxt;i<iup_nxt;i++) lmodes_used_nxt[il_map[i]] = true;
	    l.enqueuePrefetch(DeviceRead,lmodes_used_nxt);
	    
	    A2AfieldR<mf_Policies>::startPrefetches();
	  }
	}	
	copy_prefetch_time += dclock();

	//Each node only works on its time block
	for(int t=GJP.TnodeCoor()*GJP.TnodeSites(); t<(GJP.TnodeCoor()+1)*GJP.TnodeSites(); t++){   
	  const int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();
	  LOGA2A << "local timeslice " << t_lcl << "/" << GJP.TnodeSites() << std::endl;
	  
	  //Generate device base pointer and offset tables
	  ptr_setup_time -= dclock();
	  {
	    CPSautoView(base_ptrs_i_v,base_ptrs_i,HostWrite);
	    CPSautoView(site_offsets_i_v,site_offsets_i,HostWrite);
	  
	    thread_for(ii, bi_true, 
		       {
			 size_t i = ii + i0;
			 modeIndexSet i_high_unmapped; if(i>=nl_l) mf_ref.getRowParams().indexUnmap(i-nl_l,i_high_unmapped);
			 base_ptrs_i_v[ii] = l_v.getFlavorDilutedVect(i,i_high_unmapped,0,t_lcl); //Use the view to ensure we get device pointers. Here we take advantage of the fact that the 3d timeslices are contiguous
			 site_offsets_i_v[ii] = offsetT( l.siteStride3D(i,i_high_unmapped,0), l.siteStride3D(i,i_high_unmapped,1) ); //for some modes one or the other flavor is zero due to delta function
		       });
	  }
	  {
	    CPSautoView(base_ptrs_j_v,base_ptrs_j,HostWrite);
	    CPSautoView(site_offsets_j_v,site_offsets_j,HostWrite);
	    	    
	    thread_for(jj, bj_true, 
		       {
			 size_t j = jj + j0;
			 modeIndexSet j_high_unmapped; if(j>=nl_r) mf_ref.getColParams().indexUnmap(j-nl_r,j_high_unmapped);
			 base_ptrs_j_v[jj] = r_v.getFlavorDilutedVect(j,j_high_unmapped,0,t_lcl);
			 site_offsets_j_v[jj] = offsetT( r.siteStride3D(j,j_high_unmapped,0), r.siteStride3D(j,j_high_unmapped,1) );
		       });
	  }

	  //Open views to tables
	  CPSautoView(base_ptrs_i_v,base_ptrs_i,DeviceRead);
	  CPSautoView(site_offsets_i_v,site_offsets_i,DeviceRead);
	  CPSautoView(base_ptrs_j_v,base_ptrs_j,DeviceRead);
	  CPSautoView(site_offsets_j_v,site_offsets_j,DeviceRead);

	  ptr_setup_time += dclock();

	  device_memset(accum_v.data(),0,naccum*sizeof(accumType));

	  //Chunk over x-blocks
	  for(int x0 = 0; x0<size_3d; x0+=bx){
	    int xup = std::min(x0+bx, size_3d);
	    int bx_true = xup - x0;
#ifdef MF_OFFLOAD_INNER_BLOCKING
	    int sbx_use = nearestDivisor(bx_true, sbx);
	    int nxblk = bx_true / sbx_use;
#endif

	    if(t_lcl == 0 && x0 == 0 && j0 == 0 && i0 == 0 && BlockedMesonFieldArgs::enable_profiling) device_profile_start();

	    size_t nwork = bi_true * bj_true * bx_true;	  
    
	    kernel_time -= dclock();
	    
	    using namespace Grid;
	    {
	      accelerator_for(elem, nwork, Nsimd, 
			      {
#ifdef MF_OFFLOAD_INNER_BLOCKING
				//item = xs + sbx_use*( js + sbj_use * ( is + sbi_use * ( xblk + nxblk * (jblk + njblk * iblk))))
				int rem = elem;
				int xs = rem % sbx_use; rem /= sbx_use;
				int js = rem % sbj_use; rem /= sbj_use;
				int is = rem % sbi_use; rem /= sbi_use;
				int xblk = rem % nxblk; rem /= nxblk;
				int jblk = rem % njblk; rem /= njblk;
				int iblk = rem;

				int ii = is + sbi_use*iblk;
				int jj = js + sbj_use*jblk;
				int xx = xs + sbx_use*xblk;
				
				int i = ii + i0;
				int j = jj + j0;
				int x = xx + x0;
								
#else
				int rem = elem;
				int xx = rem % bx_true; rem /= bx_true;
				int jj = rem % bj_true; rem /= bj_true;
				int ii = rem;			    
				
				int i = ii+i0;
				int j = jj+j0;
				int x = xx+x0;
#endif
				
				accumType *into = accum_v.data() + multiplicity*(xx + bx*(jj + bj*ii));			
				vPtr lptr = base_ptrs_i_v[ii]; lptr.incrementPointers(site_offsets_i_v[ii], x);
				vPtr rptr = base_ptrs_j_v[jj]; rptr.incrementPointers(site_offsets_j_v[jj], x);

				typename mfVectorPolicies::accessType acc = mfVectorPolicies::getAccessor(into);
				
				M_v(acc,lptr,rptr,x,t);
			      });
	    }
	    kernel_time += dclock();
	    
	    if(x0 == 0 && j0 == 0 && i0 == 0 && BlockedMesonFieldArgs::enable_profiling) device_profile_stop();
	  }//x0

	  reduce_time -= dclock();
	  this->reduce(mf_t, accum_v.data(), i0, j0, bi_true, bj_true, bj, t, bx);
	  reduce_time += dclock();

#ifdef MF_OFFLOAD_NODESUM_ASYNC
	  async_nodesum_init_time -= dclock();
	  this->nodeSumPartialAsyncStart(nodesum_hinto,nodesum_handles,mf_t,Lt,i0,bi_true,j0,bj_true);
	  async_nodesum_init_time += dclock();
#endif
	}//t

	LOGA2A << "Waiting for prefetch completion prior to starting next j-block" << std::endl;
	prefetch_wait_time -= dclock();
	A2AfieldR<mf_Policies>::waitPrefetches();
	prefetch_wait_time += dclock();
	
	r_v.free();
      }//j0
      l_v.free();
    }//i0
      
#endif //memtest mode

    a2a_print_time("A2AmesonField","local compute",time + dclock());
    a2a_print_time("A2AmesonField","kernel time in local compute",kernel_time);
    a2a_print_time("A2AmesonField","ptr setup time in local compute",ptr_setup_time);
    a2a_print_time("A2AmesonField","device copy/prefetch init time in local compute",copy_prefetch_time);
    a2a_print_time("A2AmesonField","device copy/prefetch wait time in local compute",prefetch_wait_time);
    a2a_print_time("A2AmesonField","reduce time in local compute",reduce_time);
    
    //Complete all outstanding node sums
#ifndef MEMTEST_MODE
#ifdef MF_OFFLOAD_NODESUM_ASYNC
    a2a_print_time("A2AmesonField","nodeSum async start",async_nodesum_init_time);
    time = -dclock();
    this->nodeSumPartialAsyncComplete(nodesum_hinto, nodesum_handles);
    a2a_print_time("A2AmesonField","nodeSum async complete",time + dclock());
#else
    time = -dclock();
    this->nodeSum(mf_t, Lt);
    a2a_print_time("A2AmesonField","nodeSum",time + dclock());
#endif
#endif

    a2a_print_time("A2AmesonField","total",total_time + dclock());
  }
 

  inline void compute(mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
    compute_v3(mf_t,l,M,r,do_setup);
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
