#ifndef _MESONFIELD_COMPUTE_IMPL_OFFLOAD
#define _MESONFIELD_COMPUTE_IMPL_OFFLOAD

//Implementation of offloaded reduction kernel only CUDA currently
#ifdef GRID_CUDA

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

#endif //GRID_CUDA


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
			    const size_t bi_true, const size_t bj_true, //true size of this block
			    const size_t bj, //size of block. If block size not an even divisor of the number of modes, the above will differ from this for the last block 
			    const int t, const size_t size_3d){
//CUDA only currently
#ifdef GRID_CUDA
    //std::cout << "CUDA GPU reduce (single src)" << std::endl; fflush(stdout);
    double talloc_free = 0;
    double tkernel = 0;
    double tpoke = 0;
  
    const int multiplicity = 1;

    double time = dclock();
    Grid::ComplexD* tmp = (Grid::ComplexD*)managed_alloc_check(bi_true * bj_true * multiplicity * sizeof(Grid::ComplexD));
    talloc_free += dclock() - time;
    time = dclock();
    blockReduce(tmp, accum, bi_true, bj_true, bj, size_3d, multiplicity);
    tkernel += dclock() - time;

    time = dclock();
#pragma omp parallel for
    for(size_t z=0; z < bi_true*bj_true; z++){
      size_t jj = z % bj_true;
      size_t ii = z / bj_true;

      size_t i = ii+i0;
      size_t j = jj+j0;

      mf_t[t](i,j) += tmp[jj + bj_true*ii];
    }
    tpoke += dclock() - time;

    time = dclock();
    managed_free(tmp);
    talloc_free += dclock() - time;

    //print_time("CUDA GPU reduce","alloc_free",talloc_free);
    //print_time("CUDA GPU reduce","kernel",tkernel);
    //print_time("CUDA GPU reduce","poke",tpoke);
#else
    assert(0); //NEEDS DEVICE -> HOST COPY OF ACCUM
    
    //Reduce over size_3d
    //Paralllelize me!
    for(int ii=0;ii<bi_true;ii++)
      for(int jj=0;jj<bj_true;jj++){
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
		     const size_t bi_true, const size_t bj_true, //true size of this block
		     const int bj, //size of block. If block size not an even divisor of the number of modes, the above will differ from this for the last block 
		     const int t, const size_t size_3d) const{

    //CUDA only currently
#ifdef GRID_CUDA
    //std::cout << "CUDA GPU reduce (multi src)" << std::endl;

    double talloc_free = 0;
    double tkernel = 0;
    double tpoke = 0;
  
    const int multiplicity = mfPerTimeSlice;

    double time = dclock();
    Grid::ComplexD* tmp = (Grid::ComplexD*)managed_alloc_check(bi_true * bj_true * multiplicity * sizeof(Grid::ComplexD));
    talloc_free += dclock() - time;
    time = dclock();
    blockReduce(tmp, accum, bi_true, bj_true, bj, size_3d, multiplicity);
    tkernel += dclock() - time;

    time = dclock();
#pragma omp parallel for
    for(size_t z=0; z < bi_true*bj_true*multiplicity; z++){
      size_t rem = z;     
      size_t m = rem % multiplicity; rem /= multiplicity;
      size_t jj = rem % bj_true; rem /= bj_true;
      size_t ii = rem;
      
      size_t i = ii+i0;
      size_t j = jj+j0;

      mf_st[m]->operator[](t)(i,j) += tmp[m + multiplicity *(jj + bj_true*ii)];
    }
    tpoke += dclock() - time;

    time = dclock();
    managed_free(tmp);
    talloc_free += dclock() - time;

    // print_time("CUDA GPU reduce","alloc_free",talloc_free);
    // print_time("CUDA GPU reduce","kernel",tkernel);
    // print_time("CUDA GPU reduce","poke",tpoke);
#else
    assert(0); //NEEDS DEVICE -> HOST COPY OF ACCUM
    //Reduce over size_3d
    //(Do this on host for now) //GENERALIZE ME
    for(int ii=0;ii<bi_true;ii++)
      for(int jj=0;jj<bj_true;jj++){
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

//Starting at 'init', sucessively reduce the value 'out' until it is an integer divisor of 'of'
inline int nearestDivisor(const int of, const int init){
  int out = init;
  while(of % out != 0) --out;
  return out;
}


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

    
  void compute_v1(mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
    this->setupPolicy(mf_t,l,M,r);
    
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    if(!UniqueID()) printf("Starting A2AmesonField::compute (blocked) for %d timeslices with %d threads\n",Lt, omp_get_max_threads());
    if(!UniqueID()) printMem("mfComputeGeneralOffload node 0 memory status",0);

    cps::sync();

    if(!UniqueID()){ printf("Initializing meson fields\n"); fflush(stdout); }
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

    size_t bi = BlockedMesonFieldArgs::bi, bj = BlockedMesonFieldArgs::bj, bx = BlockedMesonFieldArgs::bp;
    if(bi > nmodes_l || bi == 0) bi = nmodes_l;
    if(bj > nmodes_r || bj == 0) bj = nmodes_r;
    if(bx > size_3d || bx == 0) bx = size_3d; //optional disable of x blocking

#ifdef GRID_CUDA
    int device;
    cudaGetDevice(&device);
#endif
    
#define MF_OFFLOAD_INNER_BLOCKING
#ifdef MF_OFFLOAD_INNER_BLOCKING
    //Note these will be shrunk if necessary to be an exact divisor of the true block size, which can be smaller for the last block if the block size is not a divisor
    size_t sbi = BlockedMesonFieldArgs::bii, sbj = BlockedMesonFieldArgs::bjj, sbx = BlockedMesonFieldArgs::bpp;
    if(sbi == 0 || sbi > bi) sbi = bi;
    if(sbj == 0 || sbj > bj) sbj = bj;
    if(sbx == 0 || sbx > bx) sbx = bx;
#endif
   
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
    //problem into smaller blocks of size   nl_block * nr_block * np_block,   where nl_block is tuned such that the temporaries fit in GPU memory
    //We will use BlockedMesonFieldArgs::bi, BlockedMesonFieldArgs::bj and BlockedMesonFieldArgs::bp for this purpose

    //Note, block sizes will depend on the multiplicity of the accumulation type
    

    //Allocate work item temp memory
    const size_t multiplicity = this->accumMultiplicity();
    const size_t naccum = bi * bj * bx * multiplicity;
    accumType *accum = (accumType*)device_alloc_check(naccum*sizeof(accumType));

    std::cout << "Using block sizes " << bi << " " << bj << " " << bx << ", temp memory requirement is " << byte_to_MB(naccum * sizeof(accumType)) << " MB" << std::endl;
    

#ifdef MF_OFFLOAD_INNER_BLOCKING
    std::cout << "Using inner block sizes " << sbi << " " << sbj << " " << sbx << std::endl;
#endif
    size_t nioblocks = (nmodes_l + bi-1)/bi,  njoblocks = (nmodes_r + bj-1)/bj,  nxoblocks = (size_3d + bx-1)/bx;
    size_t kernel_execs = nioblocks * njoblocks * nxoblocks;
    
    std::cout << "Number of outer blocks " << nioblocks << " " << njoblocks << " " << nxoblocks << " for " << kernel_execs << " kernel executions per timeslice" << std::endl;

    
    double reduce_time = 0;
    double ptr_setup_time = 0;
    double kernel_time = 0;
    
    //Each node only works on its time block
    for(int t=GJP.TnodeCoor()*GJP.TnodeSites(); t<(GJP.TnodeCoor()+1)*GJP.TnodeSites(); t++){   
      double ttime = -dclock();
      const int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();

#ifndef MEMTEST_MODE

      //Generate the offset and pointer tables on the host
      thread_for(i, nmodes_l, 
		 {
		   modeIndexSet i_high_unmapped; if(i>=nl_l) mf_ref.getRowParams().indexUnmap(i-nl_l,i_high_unmapped);
		   base_ptrs_i[i] = l.getFlavorDilutedVect(i,i_high_unmapped,0,t_lcl); //here we take advantage of the fact that the 3d timeslices are contiguous
		   site_offsets_i[i] = offsetT( l.siteStride3D(i,i_high_unmapped,0), l.siteStride3D(i,i_high_unmapped,1) ); //for some modes one or the other flavor is zero due to delta function
		 });
      thread_for(j, nmodes_r, 
		 {
		   modeIndexSet j_high_unmapped; if(j>=nl_r) mf_ref.getColParams().indexUnmap(j-nl_r,j_high_unmapped);
		   base_ptrs_j[j] = r.getFlavorDilutedVect(j,j_high_unmapped,0,t_lcl);
		   site_offsets_j[j] = offsetT( r.siteStride3D(j,j_high_unmapped,0), r.siteStride3D(j,j_high_unmapped,1) );
		 });
            
      ptr_setup_time += dclock()+ttime;
      //if(!UniqueID()){ printf("Generated tables for t=%d\n",t); fflush(stdout); }

      size_t kernel_exec_it = 0;
      
      for(size_t i0 = 0; i0 < nmodes_l; i0+=bi){
	size_t iup = std::min(i0+bi,nmodes_l);
	size_t bi_true = iup - i0;
#ifdef MF_OFFLOAD_INNER_BLOCKING
	int sbi_use = nearestDivisor(bi_true, sbi);
	int niblk = bi_true / sbi_use;
	//int_fastdiv sbi_use_div(sbi_use);
#endif
	
	for(size_t j0 = 0; j0< nmodes_r; j0+=bj) {
	  size_t jup = std::min(j0+bj,nmodes_r);
	  size_t bj_true = jup - j0;
#ifdef MF_OFFLOAD_INNER_BLOCKING
	  int sbj_use = nearestDivisor(bj_true, sbj);
	  int njblk = bj_true / sbj_use;
	  //int_fastdiv sbj_use_div(sbj_use);
	  //int_fastdiv njblk_div(njblk);
#endif
	  
	  for(int x0 = 0; x0<size_3d; x0+=bx){
	    int xup = std::min(x0+bx, size_3d);
	    int bx_true = xup - x0;
#ifdef MF_OFFLOAD_INNER_BLOCKING
	    int sbx_use = nearestDivisor(bx_true, sbx);
	    int nxblk = bx_true / sbx_use;
	    //int_fastdiv sbx_use_div(sbx_use);
	    //int_fastdiv nxblk_div(nxblk);
	    //if(!UniqueID()){ printf("Kernel execute with true outer block sizes %d %d %d and inner %d %d %d. Number of blocks %d %d %d\n",
	    //			    bi_true, bj_true, bx_true, sbi_use, sbj_use, sbx_use, niblk, njblk, nxblk); fflush(stdout); }
#endif

#ifdef GRID_CUDA
	    if(t_lcl == 0 && x0 == 0 && j0 == 0 && i0 == 0 && BlockedMesonFieldArgs::enable_profiling) cudaProfilerStart();
#endif
	    size_t nwork = bi_true * bj_true * bx_true;	  

// #ifdef GRID_CUDA
// 	    for(int ii=0;ii<bi_true;ii++){
// 	      int i = ii+i0;
// 	      vPtr lptr = base_ptrs_i[i]; lptr.incrementPointers(site_offsets_i[i], x0);
// 	      if(!lptr.isZero(0)) cudaMemPrefetchAsync(lptr.getPtr(0), site_offsets_i[i].first*bx_true*sizeof(typename mf_Policies::FermionFieldType::FieldSiteType), device, NULL);
// 	      if(!lptr.isZero(1)) cudaMemPrefetchAsync(lptr.getPtr(1), site_offsets_i[i].second*bx_true*sizeof(typename mf_Policies::FermionFieldType::FieldSiteType), device, NULL);
// 	    }
// 	    for(int jj=0;jj<bj_true;jj++){
// 	      int j = jj+j0;
// 	      vPtr rptr = base_ptrs_j[j]; rptr.incrementPointers(site_offsets_j[j], x0);
// 	      if(!rptr.isZero(0)) cudaMemPrefetchAsync(rptr.getPtr(0), site_offsets_j[j].first*bx_true*sizeof(typename mf_Policies::FermionFieldType::FieldSiteType), device, NULL);
// 	      if(!rptr.isZero(1)) cudaMemPrefetchAsync(rptr.getPtr(1), site_offsets_j[j].second*bx_true*sizeof(typename mf_Policies::FermionFieldType::FieldSiteType), device, NULL);
// 	    }
// #endif
	    
	    kernel_time -= dclock();
	    CPSautoView(M_v,M); //auto M_v = M.view();
	    
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
				int rem = item;
				int xx = rem % bx_true; rem /= bx_true;
				int jj = rem % bj_true; rem /= bj_true;
				int ii = rem;			    
				
				int i = ii+i0;
				int j = jj+j0;
				int x = xx+x0;
#endif
				
				accumType *into = accum + multiplicity*(xx + bx*(jj + bj*ii));
				typename SIMT<accumType>::value_type zero; Grid::zeroit(zero);
				for(int m=0;m<multiplicity;m++)			      
				  SIMT<accumType>::write(into[m], zero);
				
				vPtr lptr = base_ptrs_i[i]; lptr.incrementPointers(site_offsets_i[i], x);
				vPtr rptr = base_ptrs_j[j]; rptr.incrementPointers(site_offsets_j[j], x);

				typename mfVectorPolicies::accessType acc = mfVectorPolicies::getAccessor(into);
				
				M_v(acc,lptr,rptr,x,t);
			      });
	    };
	    kernel_time += dclock();

	    ++kernel_exec_it;
	    if(kernel_exec_it % 50 == 0 && !UniqueID()){ printf("Kernel iteration %zu / %zu\n", kernel_exec_it, kernel_execs ); fflush(stdout); }	    
	    
	    reduce_time -= dclock();
	    this->reduce(mf_t, accum, i0, j0, bi_true, bj_true, bj, t, bx_true);
	    reduce_time += dclock();

#ifdef GRID_CUDA
	    if(x0 == 0 && j0 == 0 && i0 == 0 && BlockedMesonFieldArgs::enable_profiling) cudaProfilerStop();
#endif
	  }
	}
      } 
      
#endif //memtest mode
      //std::ostringstream os; os << "timeslice " << t << " from range " << GJP.TnodeCoor()*GJP.TnodeSites() << " to " << (GJP.TnodeCoor()+1)*GJP.TnodeSites()-1 << " : " << nmodes_l << "*" <<  nmodes_r << " modes and inner p loop of size " <<  size_3d << std::endl;
      //print_time("A2AmesonField",os.str().c_str(),ttime + dclock());
    }

    print_time("A2AmesonField","local compute",time + dclock());
    print_time("A2AmesonField","kernel time in local compute",kernel_time);
    print_time("A2AmesonField","ptr setup time in local compute",ptr_setup_time);
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
    device_free(accum);
  }


  //Gather chunks of l,r fields
  void gatherLRchunks(hostDeviceMirroredContainer<typename mf_Policies::FermionFieldType::FieldSiteType> &iblock_data,
		      hostDeviceMirroredContainer<typename mf_Policies::FermionFieldType::FieldSiteType> &jblock_data,
		      hostDeviceMirroredContainer<std::pair<bool,bool> > &iblock_flav_is_zero,
		      hostDeviceMirroredContainer<std::pair<bool,bool> > &jblock_flav_is_zero,
		      int t_lcl,
		      size_t i0, size_t j0, size_t x0,
		      size_t bi_true, size_t bj_true, size_t bx_true,
		      size_t bx, int nf, size_t nl_l, size_t nl_r,
		      const A2AfieldL<mf_Policies> &l, const A2AfieldR<mf_Policies> &r, const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> & mf_ref){
    typedef typename mf_Policies::FermionFieldType::FieldSiteType FieldSiteType;
    typedef SCFvectorPtr<FieldSiteType> vPtr;
    typedef std::pair<int,int> offsetT;
    
    FieldSiteType *iblock_data_host = iblock_data.getHostWritePtr(),  *jblock_data_host = jblock_data.getHostWritePtr();
    std::pair<bool,bool> *iblock_flav_is_zero_host = iblock_flav_is_zero.getHostWritePtr(), *jblock_flav_is_zero_host = jblock_flav_is_zero.getHostWritePtr();
    
    thread_for(ii, bi_true, {
	size_t i = i0 + ii;
	modeIndexSet i_high_unmapped; if(i>=nl_l) mf_ref.getRowParams().indexUnmap(i-nl_l,i_high_unmapped);
	vPtr in_base_ptr = l.getFlavorDilutedVect(i,i_high_unmapped,0,t_lcl); //here we take advantage of the fact that the 3d timeslices are contiguous
	offsetT in_site_offset( l.siteStride3D(i,i_high_unmapped,0), l.siteStride3D(i,i_high_unmapped,1) ); //for some modes one or the other flavor is zero due to delta function
	if( (in_site_offset.first != 0 && in_site_offset.first != 12) ||
	    (in_site_offset.second != 0 && in_site_offset.second != 12)) {
	  ERR.General("mfComputeGeneralOffload", "compute_v2", "Expect l site offsets of 12 or 0!");
	}
	in_base_ptr.incrementPointers(in_site_offset, x0);
	      
	//use mapping  scf + 12*( x3d_blk + bx*(f + nf*i))
	for(int f=0;f<nf;f++){	      
	  FieldSiteType *to_base = iblock_data_host + (0 + 12*(0 + bx*(f + nf*ii)));
		  if(!in_base_ptr.isZero(f)) memcpy(to_base, in_base_ptr.getPtr(f), 12*bx_true*sizeof(FieldSiteType));
	}
	
	iblock_flav_is_zero_host[ii] = std::pair<bool,bool>(in_base_ptr.isZero(0), in_base_ptr.isZero(1));			
      });

    thread_for(jj, bj_true, {
	size_t j = j0 + jj;
	modeIndexSet j_high_unmapped; if(j>=nl_r) mf_ref.getColParams().indexUnmap(j-nl_r,j_high_unmapped);
	vPtr in_base_ptr = r.getFlavorDilutedVect(j,j_high_unmapped,0,t_lcl);
	offsetT in_site_offset( r.siteStride3D(j,j_high_unmapped,0), r.siteStride3D(j,j_high_unmapped,1) );
	if( (in_site_offset.first != 0 && in_site_offset.first != 12) ||
	    (in_site_offset.second != 0 && in_site_offset.second != 12)) {
	  ERR.General("mfComputeGeneralOffload", "compute_v2", "Expect r site offsets of 12 or 0!");
	}
	in_base_ptr.incrementPointers(in_site_offset, x0);
	      
	//use mapping  scf + 12*( x3d_blk + bx*(f + nf*j))
	for(int f=0;f<nf;f++){	      
	  FieldSiteType *to_base = jblock_data_host + (0 + 12*(0 + bx*(f + nf*jj)));
	  if(!in_base_ptr.isZero(f)) memcpy(to_base, in_base_ptr.getPtr(f), 12*bx_true*sizeof(FieldSiteType));
	}
	
	jblock_flav_is_zero_host[jj] = std::pair<bool,bool>(in_base_ptr.isZero(0), in_base_ptr.isZero(1));
      });    
  }



  
  //This version does an explicit copy of chunks of the a2a fields to the device
  void compute_v2(mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
    this->setupPolicy(mf_t,l,M,r);
    
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    if(!UniqueID()) printf("Starting A2AmesonField::compute (blocked) for %d timeslices with %d threads\n",Lt, omp_get_max_threads());
    if(!UniqueID()) printMem("mfComputeGeneralOffload node 0 memory status",0);

    cps::sync();

    if(!UniqueID()){ printf("Initializing meson fields\n"); fflush(stdout); }
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

    size_t bi = BlockedMesonFieldArgs::bi, bj = BlockedMesonFieldArgs::bj, bx = BlockedMesonFieldArgs::bp;
    if(bi > nmodes_l || bi == 0) bi = nmodes_l;
    if(bj > nmodes_r || bj == 0) bj = nmodes_r;
    if(bx > size_3d || bx == 0) bx = size_3d; //optional disable of x blocking

#ifdef GRID_CUDA
    int device;
    cudaGetDevice(&device);
#endif
    
#define MF_OFFLOAD_INNER_BLOCKING
#ifdef MF_OFFLOAD_INNER_BLOCKING
    //Note these will be shrunk if necessary to be an exact divisor of the true block size, which can be smaller for the last block if the block size is not a divisor
    size_t sbi = BlockedMesonFieldArgs::bii, sbj = BlockedMesonFieldArgs::bjj, sbx = BlockedMesonFieldArgs::bpp;
    if(sbi == 0 || sbi > bi) sbi = bi;
    if(sbj == 0 || sbj > bj) sbj = bj;
    if(sbx == 0 || sbx > bx) sbx = bx;
#endif
   
    //Make a table of p base pointers and site offsets (stride between 3d sites) for each i,j
    //These need to be in managed memory so they can be accessed on the device
    typedef typename mf_Policies::FermionFieldType::FieldSiteType FieldSiteType;
    typedef SCFvectorPtr<FieldSiteType> vPtr;
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
    accumType *accum = (accumType*)device_alloc_check(naccum*sizeof(accumType));

    std::cout << "Using block sizes " << bi << " " << bj << " " << bx << ", temp memory requirement for block MF accumulation is " << byte_to_MB(naccum * sizeof(accumType)) << " MB" << std::endl;

    //Allocate temporary memory for chunks of a2a fields
    int nf = GJP.Gparity()+1;
    size_t iblock_data_size = bi * nf * bx * 12; //use mapping  scf + 12*( x3d_blk + bx*(f + nf*i))
    size_t jblock_data_size = bj * nf * bx * 12;
    size_t ijblock_data_flav_offset = 12 * bx;
    std::cout << "Require " << byte_to_MB(iblock_data_size*sizeof(FieldSiteType)) << "MB and " << byte_to_MB(jblock_data_size*sizeof(FieldSiteType)) << "MB for i and j chunk data, respectively" << std::endl;

    hostDeviceMirroredContainer<FieldSiteType> iblock_data(iblock_data_size), jblock_data(jblock_data_size);  
    hostDeviceMirroredContainer<std::pair<bool,bool> > iblock_flav_is_zero(bi), jblock_flav_is_zero(bj);

    //Swapping between two buffers allows us to overlap compute and gather
    hostDeviceMirroredContainer<FieldSiteType> iblock_data2(iblock_data_size), jblock_data2(jblock_data_size);  
    hostDeviceMirroredContainer<std::pair<bool,bool> > iblock_flav_is_zero2(bi), jblock_flav_is_zero2(bj);

    
#ifdef MF_OFFLOAD_INNER_BLOCKING
    std::cout << "Using inner block sizes " << sbi << " " << sbj << " " << sbx << std::endl;
#endif
    size_t nioblocks = (nmodes_l + bi-1)/bi,  njoblocks = (nmodes_r + bj-1)/bj,  nxoblocks = (size_3d + bx-1)/bx;
    size_t kernel_execs = nioblocks * njoblocks * nxoblocks;
    
    std::cout << "Number of outer blocks " << nioblocks << " " << njoblocks << " " << nxoblocks << " for " << kernel_execs << " kernel executions per timeslice" << std::endl;
    
    double reduce_time = 0;
    double kernel_time = 0;
    double gather_chunk_time = 0;
    double device_copy_chunk_time = 0;
    
    //Each node only works on its time block
    size_t iter = 0;
    
    for(int t=GJP.TnodeCoor()*GJP.TnodeSites(); t<(GJP.TnodeCoor()+1)*GJP.TnodeSites(); t++){   
      double ttime = -dclock();
      const int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();

#ifndef MEMTEST_MODE

      size_t kernel_exec_it = 0;
      
      for(size_t i0 = 0; i0 < nmodes_l; i0+=bi){
	size_t iup = std::min(i0+bi,nmodes_l);
	size_t bi_true = iup - i0;
#ifdef MF_OFFLOAD_INNER_BLOCKING
	int sbi_use = nearestDivisor(bi_true, sbi);
	int niblk = bi_true / sbi_use;
#endif
	
	for(size_t j0 = 0; j0< nmodes_r; j0+=bj) {
	  size_t jup = std::min(j0+bj,nmodes_r);
	  size_t bj_true = jup - j0;
#ifdef MF_OFFLOAD_INNER_BLOCKING
	  int sbj_use = nearestDivisor(bj_true, sbj);
	  int njblk = bj_true / sbj_use;
#endif
	  
	  for(int x0 = 0; x0<size_3d; x0+=bx){
	    int xup = std::min(x0+bx, size_3d);
	    int bx_true = xup - x0;
#ifdef MF_OFFLOAD_INNER_BLOCKING
	    int sbx_use = nearestDivisor(bx_true, sbx);
	    int nxblk = bx_true / sbx_use;
#endif

#ifdef GRID_CUDA
	    if(t_lcl == 0 && x0 == 0 && j0 == 0 && i0 == 0 && BlockedMesonFieldArgs::enable_profiling) cudaProfilerStart();
#endif
	    size_t swp = iter % 2;

	    //Get the data containers we read from this iteratiopn
	    hostDeviceMirroredContainer<FieldSiteType> &iblock_data_read = swp == 0 ? iblock_data : iblock_data2;
	    hostDeviceMirroredContainer<FieldSiteType> &jblock_data_read = swp == 0 ? jblock_data : jblock_data2;
	    hostDeviceMirroredContainer<std::pair<bool,bool> > &iblock_flav_is_zero_read = swp == 0 ? iblock_flav_is_zero : iblock_flav_is_zero2;
	    hostDeviceMirroredContainer<std::pair<bool,bool> > &jblock_flav_is_zero_read = swp == 0 ? jblock_flav_is_zero : jblock_flav_is_zero2;

	    //Get the offsets of the next iter
	    bool fetch_next_iter = true;
	    int x0_next = x0, t_lcl_next = t_lcl;
	    size_t j0_next = j0, i0_next = i0;	    
	    
	    x0_next += bx;
	    if(x0_next >= size_3d){
	      x0_next = 0; j0_next += bj;
	      if(j0_next >= nmodes_r){
		j0_next = 0; i0_next += bi;
		if(i0_next >= nmodes_l){
		  i0_next = 0; t_lcl_next++;
		  if(t_lcl_next >= GJP.TnodeSites() ){
		    fetch_next_iter = false;
		  }
		}
	      }
	    }
	    int bx_true_next = std::min(x0_next+bx, size_3d) - x0_next;
	    size_t bj_true_next = std::min(j0_next+bj,nmodes_r) - j0_next;
	    size_t bi_true_next = std::min(i0_next+bi,nmodes_l) - i0_next;
	    
	    //Get the data containers we are going to prefetch to
	    hostDeviceMirroredContainer<FieldSiteType> &iblock_data_write = swp == 0 ? iblock_data2 : iblock_data;
	    hostDeviceMirroredContainer<FieldSiteType> &jblock_data_write = swp == 0 ? jblock_data2 : jblock_data;
	    hostDeviceMirroredContainer<std::pair<bool,bool> > &iblock_flav_is_zero_write = swp == 0 ? iblock_flav_is_zero2 : iblock_flav_is_zero;
	    hostDeviceMirroredContainer<std::pair<bool,bool> > &jblock_flav_is_zero_write = swp == 0 ? jblock_flav_is_zero2 : jblock_flav_is_zero;

	    
	    //iter 0 doesn't have data prefetched, do blocking gather
	    if(iter == 0){
	      gather_chunk_time -= dclock();
	      gatherLRchunks(iblock_data_read, jblock_data_read, iblock_flav_is_zero_read, jblock_flav_is_zero_read, t_lcl, i0, j0, x0, bi_true, bj_true, bx_true, bx, nf, nl_l, nl_r, l, r, mf_ref);
	      gather_chunk_time += dclock();	    
	    }
	    
	    using namespace Grid;
	    {
	      //Copy data to device
	      device_copy_chunk_time -= dclock();
	      FieldSiteType const *iblock_data_device = iblock_data_read.getDeviceReadPtr(),  *jblock_data_device = jblock_data_read.getDeviceReadPtr();
	      std::pair<bool,bool> const *iblock_flav_is_zero_device = iblock_flav_is_zero_read.getDeviceReadPtr(), *jblock_flav_is_zero_device = jblock_flav_is_zero_read.getDeviceReadPtr();
	      device_copy_chunk_time += dclock();

	      //Launch kernel asynchronously
	      size_t nwork = bi_true * bj_true * bx_true;	  
	      
	      kernel_time -= dclock();
	      CPSautoView(M_v,M);
	      
	      accelerator_forNB(elem, nwork, Nsimd, 
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
				int rem = item;
				int xx = rem % bx_true; rem /= bx_true;
				int jj = rem % bj_true; rem /= bj_true;
				int ii = rem;			    
				
				int i = ii+i0;
				int j = jj+j0;
				int x = xx+x0;
#endif
      
				accumType *into = accum + multiplicity*(xx + bx*(jj + bj*ii));
				typename SIMT<accumType>::value_type zero; Grid::zeroit(zero);
				for(int m=0;m<multiplicity;m++)			      
				  SIMT<accumType>::write(into[m], zero);
				
				//use mapping  scf + 12*( x3d_blk + bx*(f + nf*i))
				FieldSiteType const* l_f0_ptr = iblock_data_device + 12*(xx + bx*nf*ii);
				FieldSiteType const* l_f1_ptr = l_f0_ptr + ijblock_data_flav_offset; //should still be fine for nf=1 as we simply won't access the pointer
				const std::pair<bool,bool> &l_zero = iblock_flav_is_zero_device[ii];
				
				vPtr lptr(l_f0_ptr, l_f1_ptr, l_zero.first, l_zero.second);

				FieldSiteType const* r_f0_ptr = jblock_data_device + 12*(xx + bx*nf*jj);
				FieldSiteType const* r_f1_ptr = r_f0_ptr + ijblock_data_flav_offset;
				const std::pair<bool,bool> &r_zero = jblock_flav_is_zero_device[jj];

				vPtr rptr(r_f0_ptr, r_f1_ptr, r_zero.first, r_zero.second);

				typename mfVectorPolicies::accessType acc = mfVectorPolicies::getAccessor(into);
				
				M_v(acc,lptr,rptr,x,t);
			      });
	    };

	    //Overlap gather and kernel	    
	    if(fetch_next_iter){
	      gatherLRchunks(iblock_data_write, jblock_data_write, iblock_flav_is_zero_write, jblock_flav_is_zero_write, t_lcl_next, i0_next, j0_next, x0_next, bi_true_next, bj_true_next, bx_true_next, bx, nf, nl_l, nl_r, l, r, mf_ref);
	    }

	    //Wait for kernel to finish
	    accelerator_barrier(dummy);	    
	    kernel_time += dclock();

	    ++kernel_exec_it;
	    if(kernel_exec_it % 50 == 0 && !UniqueID()){ printf("Kernel iteration %zu / %zu\n", kernel_exec_it, kernel_execs ); fflush(stdout); }	    
	    
	    reduce_time -= dclock();
	    this->reduce(mf_t, accum, i0, j0, bi_true, bj_true, bj, t, bx_true);
	    reduce_time += dclock();

#ifdef GRID_CUDA
	    if(x0 == 0 && j0 == 0 && i0 == 0 && BlockedMesonFieldArgs::enable_profiling) cudaProfilerStop();
#endif
	    ++iter;
	  }
	}
      } 
      
#endif //memtest mode
      //std::ostringstream os; os << "timeslice " << t << " from range " << GJP.TnodeCoor()*GJP.TnodeSites() << " to " << (GJP.TnodeCoor()+1)*GJP.TnodeSites()-1 << " : " << nmodes_l << "*" <<  nmodes_r << " modes and inner p loop of size " <<  size_3d << std::endl;
      //print_time("A2AmesonField",os.str().c_str(),ttime + dclock());
    }

    print_time("A2AmesonField","local compute",time + dclock());
    print_time("A2AmesonField","kernel time in local compute",kernel_time);
    print_time("A2AmesonField","chunk gather time in local compute",gather_chunk_time);
    print_time("A2AmesonField","chunk device copy time in local compute",device_copy_chunk_time);    
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
    
    device_free(accum);
  }

    

  inline void compute(mfVectorType &mf_t, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
    compute_v2(mf_t,l,M,r,do_setup);
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
