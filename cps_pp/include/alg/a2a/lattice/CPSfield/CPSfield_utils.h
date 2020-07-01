#ifndef CPS_FIELD_UTILS_H
#define CPS_FIELD_UTILS_H

#include "CPSfield.h"

#ifdef GRID_NVCC
//Make sure you link -lcufft
#include <cufft.h>
#endif

CPS_START_NAMESPACE
#include "implementation/CPSfield_utils_impl.tcc"
#include "implementation/CPSfield_utils_cyclicpermute.tcc"


//Compare 5d fermion fields
inline void compareFermion(const CPSfermion5D<ComplexD> &A, const CPSfermion5D<ComplexD> &B, const std::string &descr = "Ferms", const double tol = 1e-9){
  double fail = 0.;
  for(int i=0;i<GJP.VolNodeSites()*GJP.SnodeSites();i++){
    int x[5]; int rem = i;
    for(int ii=0;ii<5;ii++){ x[ii] = rem % GJP.NodeSites(ii); rem /= GJP.NodeSites(ii); }
    
    for(int f=0;f<GJP.Gparity()+1;f++){
      for(int sc=0;sc<24;sc++){
	double vbfm = *((double*)A.site_ptr(i,f) + sc);
	double vgrid = *((double*)B.site_ptr(i,f) + sc);
	    
	double diff_rat = fabs( 2.0 * ( vbfm - vgrid )/( vbfm + vgrid ) );
	double rat_grid_bfm = vbfm/vgrid;
	if(vbfm == 0.0 && vgrid == 0.0){ diff_rat = 0.;	 rat_grid_bfm = 1.; }
	if( (vbfm == 0.0 && fabs(vgrid) < 1e-50) || (vgrid == 0.0 && fabs(vbfm) < 1e-50) ){ diff_rat = 0.;	 rat_grid_bfm = 1.; }

	if(diff_rat > tol){
	  printf("Fail: (%d,%d,%d,%d,%d; %d; %d) A %g B %g rat_A_B %g fracdiff %g\n",x[0],x[1],x[2],x[3],x[4],f,sc,vbfm,vgrid,rat_grid_bfm,diff_rat);
	  fail = 1.0;
	}//else printf("Pass: (%d,%d,%d,%d,%d; %d; %d) A %g B %g rat_A_B %g fracdiff %g\n",x[0],x[1],x[2],x[3],x[4],f,sc,vbfm,vgrid,rat_grid_bfm,diff_rat);
      }
    }
  }
  glb_max(&fail);
  
  if(fail!=0.0){
    if(!UniqueID()){ printf("Failed %s check\n", descr.c_str()); fflush(stdout); } 
    exit(-1);
  }else{
    if(!UniqueID()){ printf("Passed %s check\n", descr.c_str()); fflush(stdout); }
  }
}

//Compare general CPSfield
template<typename FieldType, typename my_enable_if<_equal<typename ComplexClassify<typename FieldType::FieldSiteType>::type, complex_double_or_float_mark>::value,int>::type = 0>
inline void compareField(const FieldType &A, const FieldType &B, const std::string &descr = "Field", const double tol = 1e-9, bool print_all = false){
  typedef typename FieldType::FieldSiteType::value_type value_type;
  
  double fail = 0.;
  for(int xf=0;xf<A.nfsites();xf++){
    int f; int x[FieldType::FieldMappingPolicy::EuclideanDimension];
    A.fsiteUnmap(xf, x,f);

    for(int i=0;i<FieldType::FieldSiteSize;i++){
      value_type const* av = (value_type const*)(A.fsite_ptr(xf)+i);
      value_type const* bv = (value_type const*)(B.fsite_ptr(xf)+i);
      for(int reim=0;reim<2;reim++){
	value_type diff_rat = (av[reim] == 0.0 && bv[reim] == 0.0) ? 0.0 : fabs( 2.*(av[reim]-bv[reim])/(av[reim]+bv[reim]) );
	if(diff_rat > tol || print_all){
	  if(diff_rat > tol) std::cout << "!FAIL: ";
	  else std::cout << "Pass: ";
	  
	  std::cout << "coord=(";
	  for(int xx=0;xx<FieldType::FieldMappingPolicy::EuclideanDimension-1;xx++)
	    std::cout << x[xx] << ", ";
	  std::cout << x[FieldType::FieldMappingPolicy::EuclideanDimension-1];

	  std::cout << ") f=" << f << " i=" << i << " reim=" << reim << " A " << av[reim] << " B " << bv[reim] << " fracdiff " << diff_rat << std::endl;
	  if(diff_rat > tol) fail = 1.;
	}
      }
    }
  }
  glb_max(&fail);
  
  if(fail!=0.0){
    if(!UniqueID()){ printf("Failed %s check\n", descr.c_str()); fflush(stdout); } 
    exit(-1);
  }else{
    if(!UniqueID()){ printf("Passed %s check\n", descr.c_str()); fflush(stdout); }
  }
}



#ifdef USE_BFM
//Convert from a BFM field to a CPSfield
inline void exportBFMcb(CPSfermion5D<cps::ComplexD> &into, Fermion_t from, bfm_evo<double> &dwf, int cb, bool singleprec_evec = false){
  Fermion_t zero_a = dwf.allocFermion();
#pragma omp parallel
  {   
    dwf.set_zero(zero_a); 
  }
  Fermion_t etmp = dwf.allocFermion(); 
  Fermion_t tmp[2];
  tmp[!cb] = zero_a;
  if(singleprec_evec){
    const int len = 24 * dwf.node_cbvol * (1 + dwf.gparity) * dwf.cbLs;
#pragma omp parallel for
    for(int j = 0; j < len; j++) {
      ((double*)etmp)[j] = ((float*)(from))[j];
    }
    tmp[cb] = etmp;
  }else tmp[cb] = from;

  dwf.cps_impexFermion((double*)into.ptr(),tmp,0);
  dwf.freeFermion(zero_a);
  dwf.freeFermion(etmp);
}
#endif


#ifdef USE_GRID
//Export a checkerboarded Grid field to an un-checkerboarded CPS field
template<typename GridPolicies>
inline void exportGridcb(CPSfermion5D<cps::ComplexD> &into, typename GridPolicies::GridFermionField &from, typename GridPolicies::FgridFclass &latg){
  Grid::GridCartesian *FGrid = latg.getFGrid();
  typename GridPolicies::GridFermionField tmp_g(FGrid);
  tmp_g = Grid::Zero();

  setCheckerboard(tmp_g, from);
  latg.ImportFermion((Vector*)into.ptr(), tmp_g);
}
#endif



//Cyclic permutation of *4D* and *3D* CPSfield with std::complex type and FourDpolicy dimension policy
//Conventions are direction of *data flow*: For shift n in direction +1   f'(x) = f(x-\hat i)  so data is sent in the +x direction.
template< typename mf_Complex, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void cyclicPermute(CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> &to, const CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> &from,
		   const int dir, const int pm, const int n){
  if(n >= GJP.NodeSites(dir)){ //deal with n > node size
    CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> tmp1(from);
    CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> tmp2(from);

    CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy>* from_i = &tmp1;
    CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy>* to_i = &tmp2;
    int nn = n;
    while(nn >= GJP.NodeSites(dir)){
      cyclicPermuteImpl(*to_i, *from_i, dir, pm, GJP.NodeSites(dir)-1);
      nn -= (GJP.NodeSites(dir)-1);
      std::swap(from_i, to_i); //on last iteration the data will be in from_i after leaving the loop
    }
    cyclicPermuteImpl(to, *from_i, dir, pm, nn);
  }else{
    cyclicPermuteImpl(to, from, dir, pm, n);
  }
}


//+1 if of>0,  -1 otherwise
inline int getShiftSign(const int of){ return of > 0 ? +1 : -1; }


//Invoke multiple independent permutes to offset field by vector 'shift' assuming field is periodic
template<typename FieldType>
void shiftPeriodicField(FieldType &to, const FieldType &from, const std::vector<int> &shift){
  int nd = shift.size(); //assume ascending: x,y,z,t
  int nshift_dirs = 0;
  for(int i=0;i<nd;i++) if(shift[i]!=0) ++nshift_dirs;

  if(nshift_dirs == 0){
    if(&to != &from) to = from;
    return;
  }else if(nshift_dirs == 1){
    for(int d=0;d<nd;d++){
      if(shift[d] != 0){
	cyclicPermute(to,from,d,getShiftSign(shift[d]),abs(shift[d]) );
	return;
      }
    }    
  }else{
    FieldType tmp1 = from;
    FieldType tmp2 = from;
    FieldType * send = &tmp1;
    FieldType * recv = &tmp2;

    int shifts_done = 0;
    for(int d=0;d<nd;d++){
      if(shift[d] != 0){
	cyclicPermute(shifts_done < nshift_dirs-1 ? *recv : to,*send,d,getShiftSign(shift[d]),abs(shift[d]) );
	++shifts_done;
	if(shifts_done < nshift_dirs) std::swap(send,recv);
	else return;
      }
    }   
  }
}



//Perform an FFT of a CPSfield
//do_dirs is a vector of bool of dimension equal to the field dimension, and indicate which directions in which to perform the FFT
//inverse_transform specifies whether to do the inverse FFT
//Output and input fields can be the same
//This version is for non-SIMD complex type
template<typename CPSfieldType>
void fft(CPSfieldType &into, const CPSfieldType &from, const bool* do_dirs, const bool inverse_transform = false,
	 typename my_enable_if<_equal<typename ComplexClassify<typename CPSfieldType::FieldSiteType>::type, complex_double_or_float_mark>::value, const int>::type = 0
	 ){
  typedef typename LocalToGlobalInOneDirMap<typename CPSfieldType::FieldMappingPolicy>::type DimPolGlobalInOneDir;
  typedef CPSfieldGlobalInOneDir<typename CPSfieldType::FieldSiteType, CPSfieldType::FieldSiteSize, DimPolGlobalInOneDir, typename CPSfieldType::FieldAllocPolicy> CPSfieldTypeGlobalInOneDir;

  int dcount = 0;
  
  for(int mu=0;mu<CPSfieldType::FieldMappingPolicy::EuclideanDimension;mu++)
    if(do_dirs[mu]){
      CPSfieldTypeGlobalInOneDir tmp_dbl(mu);
      tmp_dbl.gather( dcount==0 ? from : into );
      tmp_dbl.fft(inverse_transform);
      tmp_dbl.scatter(into);
      dcount ++;
    }
}

//Version of the above for SIMD complex type
#ifdef USE_GRID
template<typename CPSfieldType>
void fft(CPSfieldType &into, const CPSfieldType &from, const bool* do_dirs, const bool inverse_transform = false,
	 typename my_enable_if<_equal<typename ComplexClassify<typename CPSfieldType::FieldSiteType>::type, grid_vector_complex_mark>::value, const int>::type = 0
	 ){
  typedef typename Grid::GridTypeMapper<typename CPSfieldType::FieldSiteType>::scalar_type ScalarType;
  typedef typename CPSfieldType::FieldMappingPolicy::EquivalentScalarPolicy ScalarMapPol;
  typedef CPSfield<ScalarType, CPSfieldType::FieldSiteSize, ScalarMapPol, StandardAllocPolicy> ScalarFieldType;

  NullObject null_obj;
  ScalarFieldType tmp_in(null_obj);
  ScalarFieldType tmp_out(null_obj);
  tmp_in.importField(from);
  fft(tmp_out, tmp_in, do_dirs, inverse_transform);
  tmp_out.exportField(into);
}
#endif
  
//in-place non-SIMD FFT of a field
//do_dirs is a vector of bool of dimension equal to the field dimension, and indicate which directions in which to perform the FFT
template<typename CPSfieldType>
void fft(CPSfieldType &fftme, const bool* do_dirs){
  fft(fftme,fftme,do_dirs);
}

//An optimized implementation of the FFT
//Requires MPI
//do_dirs is a vector of bool of dimension equal to the field dimension, and indicate which directions in which to perform the FFT
//inverse_transform specifies whether to do the inverse FFT
//Output and input fields can be the same
template<typename CPSfieldType>
void fft_opt(CPSfieldType &into, const CPSfieldType &from, const bool* do_dirs, const bool inverse_transform = false,
	     typename my_enable_if<_equal<typename ComplexClassify<typename CPSfieldType::FieldSiteType>::type, complex_double_or_float_mark>::value, const int>::type = 0
	     ){
#ifndef USE_MPI
  //if(!UniqueID()) printf("fft_opt reverting to fft because USE_MPI not enabled\n");
  fft(into,from,do_dirs,inverse_transform);
#else
  //if(!UniqueID()) printf("Using fft_opt\n");

  enum { Dimension = CPSfieldType::FieldMappingPolicy::EuclideanDimension };
  int ndirs_fft = 0; for(int i=0;i<Dimension;i++) if(do_dirs[i]) ++ndirs_fft;
  if(! ndirs_fft ) return;

  //Need info on the MPI node mapping
  assert(GJP.Snodes() == 1);
  std::vector<int> node_map;
  getMPIrankMap(node_map);

  CPSfieldType tmp(from.getDimPolParams());

  //we want the last fft to end up in 'into'. Intermediate FFTs cycle between into and tmp as temp storage. Thus for odd ndirs_fft, the first fft should output to 'into', for even it should output to 'tmp'
  CPSfieldType *tmp1, *tmp2;
  if(ndirs_fft % 2 == 1){
    tmp1 = &into; tmp2 = &tmp;
  }else{
    tmp1 = &tmp; tmp2 = &into;
  }
  
  CPSfieldType* src = tmp2;
  CPSfieldType* out = tmp1;

  int fft_count = 0;
  for(int mu=0; mu<Dimension; mu++){
    if(do_dirs[mu]){
      CPSfieldType const *msrc = fft_count == 0 ? &from : src;
      fft_opt_mu(*out, *msrc, mu, node_map, inverse_transform);
      ++fft_count;
      std::swap(src,out);      
    }
  }
#endif
}

//Optimized FFT in a single direction mu
//node_map is the mapping of CPS UniqueID to MPI rank, and can be obtained with   "getMPIrankMap(node_map)"
#ifdef USE_MPI


//Timers for fft_opt_mu
struct fft_opt_mu_timings{
  struct timers{
    size_t calls;
    double setup;
    double gather;
    double comm_gather;
    double fft;
    double comm_scatter;
    double scatter;

    timers(): calls(0), setup(0), gather(0), comm_gather(0), fft(0), comm_scatter(0), scatter(0){}

    void reset(){
      setup = gather = comm_gather = fft = comm_scatter = scatter = 0;
      calls = 0;
    }
    void average(){
      setup/=calls;
      gather/=calls;
      comm_gather/=calls;
      fft/=calls;
      comm_scatter/=calls;
      scatter/=calls;
    }
    void print(){
      average();
      printf("calls=%zu setup=%g gather=%g comm_gather=%g fft=%g comm_scatter=%g scatter=%g\n", calls, setup, gather, comm_gather, fft, comm_scatter, scatter);
    }
  };
  static timers & get(){ static timers t; return t; }
};



template<typename CPSfieldType>
void fft_opt_mu(CPSfieldType &into, const CPSfieldType &from, const int mu, const std::vector<int> &node_map, const bool inverse_transform,
	     typename my_enable_if<_equal<typename ComplexClassify<typename CPSfieldType::FieldSiteType>::type, complex_double_or_float_mark>::value, const int>::type = 0
	     ){
  fft_opt_mu_timings::get().calls++;
  fft_opt_mu_timings::get().setup -= dclock();
  
  enum {SiteSize = CPSfieldType::FieldSiteSize, Dimension = CPSfieldType::FieldMappingPolicy::EuclideanDimension };
  typedef typename CPSfieldType::FieldSiteType ComplexType;
  typedef typename ComplexType::value_type FloatType;
  typedef typename FFTWwrapper<FloatType>::complexType FFTComplex;
  const int nf = from.nflavors();
  const int foff = from.flav_offset();
  const int nthread = omp_get_max_threads();

  //Eg for fft in X-direction, divide up Y,Z,T work over nodes in X-direction doing linear FFTs.
  const int munodesites = GJP.NodeSites(mu);
  const int munodes = GJP.Nodes(mu);
  const int mutotalsites = munodesites*munodes;
  const int munodecoor = GJP.NodeCoor(mu);
  const int n_orthdirs = Dimension - 1;
  FloatType Lmu(mutotalsites);
  
  int orthdirs[n_orthdirs]; //map of orthogonal directions to mu
  size_t total_work_munodes = 1; //sites orthogonal to FFT direction
  int o=0;
  for(int i=0;i< Dimension;i++)
    if(i!=mu){
      total_work_munodes *= GJP.NodeSites(i);
      orthdirs[o++] = i;
    }

  //Divvy up work over othogonal directions
  size_t munodes_work[munodes];
  size_t munodes_off[munodes];
  for(int i=0;i<munodes;i++)
    thread_work(munodes_work[i],munodes_off[i], total_work_munodes, i, munodes); //use for node work instead :)

  //Get MPI ranks of nodes in mu direction
  int my_node_coor[4];
  for(int i=0;i<4;i++) my_node_coor[i] = GJP.NodeCoor(i);
  
  int munodes_mpiranks[munodes];
  for(int i=0;i<munodes;i++){
    int munode_coor[4]; memcpy(munode_coor,my_node_coor,4*sizeof(int));
    munode_coor[mu] = i;

    const int munode_lex = node_lex( munode_coor, 4 );
    munodes_mpiranks[i] = node_map[munode_lex];
  }
  
  fft_opt_mu_timings::get().setup += dclock();

  fft_opt_mu_timings::get().gather -= dclock();

  //Gather send data
  ComplexType* send_bufs[munodes];
  size_t send_buf_sizes[munodes];
  for(int i=0;i<munodes;i++){
    send_buf_sizes[i] = munodes_work[i] * munodesites * nf * SiteSize;
    send_bufs[i] = (ComplexType*)malloc_check( send_buf_sizes[i] * sizeof(ComplexType) );

#pragma omp parallel for
    for(size_t w = 0; w < munodes_work[i]; w++){ //index of orthogonal site within workload for i'th node in mu direction
      const int orthsite = munodes_off[i] + w;
      int coor_base[Dimension] = {0};
	  
      //Unmap orthsite into a base coordinate
      int rem = orthsite;
      for(int a=0;a<n_orthdirs;a++){
	const int dir_a = orthdirs[a];
	coor_base[dir_a] = rem % GJP.NodeSites(dir_a); rem /= GJP.NodeSites(dir_a);
      }

      for(int f=0;f<nf;f++){
	for(int xmu=0;xmu<munodesites;xmu++){
	  ComplexType* to = send_bufs[i] + SiteSize * (w + munodes_work[i]*( f + nf*xmu ) );  //with musite changing slowest
	  coor_base[mu] = xmu;
	  ComplexType const* frm = from.site_ptr(coor_base,f);

	  memcpy(to,frm,SiteSize*sizeof(ComplexType));
	}
      }
    }
  }

  fft_opt_mu_timings::get().gather += dclock();

  fft_opt_mu_timings::get().comm_gather -= dclock();

  MPI_Request send_req[munodes];
  MPI_Request recv_req[munodes];
  MPI_Status status[munodes];

  //Prepare recv buf
  const size_t bufsz = munodes_work[munodecoor] * mutotalsites * nf * SiteSize; //complete line in mu for each orthogonal coordinate
  ComplexType* recv_buf = (ComplexType*)malloc_check(bufsz * sizeof(ComplexType) );

  //Setup send/receive    
  for(int i=0;i<munodes;i++){ //works fine to send to all nodes, even if this involves a send to self.
    assert( MPI_Isend(send_bufs[i], send_buf_sizes[i]*sizeof(ComplexType), MPI_CHAR, munodes_mpiranks[i], 0, MPI_COMM_WORLD, &send_req[i]) == MPI_SUCCESS );
    assert( MPI_Irecv(recv_buf + i*munodes_work[munodecoor]*nf*SiteSize*munodesites, send_buf_sizes[i]*sizeof(ComplexType), MPI_CHAR, munodes_mpiranks[i], MPI_ANY_TAG, MPI_COMM_WORLD, &recv_req[i]) == MPI_SUCCESS );
  }      
  assert( MPI_Waitall(munodes,recv_req,status) == MPI_SUCCESS );
   
  fft_opt_mu_timings::get().comm_gather += dclock();
  
  fft_opt_mu_timings::get().fft -= dclock();

  //Do FFT
  const size_t howmany = munodes_work[munodecoor] * nf * SiteSize;

#ifdef GRID_NVCC
  //if(!UniqueID()) printf("Performing FFT using CUFFT\n");
  
  //------------------------------------------------------------------------
  //Perform FFT using CUFFT
  //------------------------------------------------------------------------
  static int plan_howmany[Dimension];
  static bool plan_init = false;
  static cufftHandle handle[Dimension];
  
  if(!plan_init || plan_howmany[mu] != howmany){
    if(!plan_init) for(int i=0;i<Dimension;i++) plan_howmany[i] = -1;

    //ComplexType* to = send_bufs[i] + SiteSize * (w + munodes_work[i]*( f + nf*xmu ) );  //with musite changing slowest

    // cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed,
    // 		  int istride, int idist, int *onembed, int ostride,
    // 		  int odist, cufftType type, int batch);

    int rank=1;
    int n[1] = {mutotalsites}; //size of each dimension
    //The only non-gibberish explanation of the imbed parameter I've yet seen can be found on page 9 of http://acarus.uson.mx/docs/cuda-5.5/CUFFT_Library.pdf
    int inembed[1] = {howmany}; //Pointer of size rank that indicates the storage dimen-sions of the input data in memory (up to istride).
    int istride = howmany; //distance between elements. We have ordered data such that the elements are the slowest index
    int idist = 1; //distance between first element of two consecutive batches
    int* onembed = inembed;
    int ostride=istride;
    int odist=idist;
    cufftType type = CUFFT_Z2Z; //double complex
    int batch = howmany; //how many FFTs are we doing?

    assert( cufftCreate(&handle[mu])== CUFFT_SUCCESS );
    assert( cufftPlanMany(&handle[mu], rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch) == CUFFT_SUCCESS );    
  }

  static_assert(sizeof(cufftDoubleComplex) == sizeof(ComplexType));
  
  cufftDoubleComplex* device_in = (cufftDoubleComplex*)device_alloc_check(bufsz * sizeof(cufftDoubleComplex));
  assert(cudaMemcpy(device_in, recv_buf, bufsz * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice) == cudaSuccess);
  
  int fft_phase = inverse_transform ? CUFFT_INVERSE : CUFFT_FORWARD;
  assert( cufftExecZ2Z(handle[mu], device_in,  device_in, fft_phase) == CUFFT_SUCCESS );
  
  assert(cudaMemcpy(recv_buf, device_in,  bufsz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost) == cudaSuccess);
  
  device_free(device_in);

#else //GRID_NVCC
  //if(!UniqueID()) printf("Performing FFT using FFTW (threaded)\n");
  //------------------------------------------------------------------------
  //Perform FFT using FFTW (threaded)
  //------------------------------------------------------------------------

  const size_t howmany_per_thread_base = howmany / nthread;
  //Divide work orthogonal to mu, 'howmany', over threads. Note, this may not divide howmany equally. The difference is made up by adding 1 unit of work to threads in ascending order until total work matches. Thus we need 2 plans: 1 for the base amount and one for the base+1

  //if(!UniqueID()) printf("FFT work per site %d, divided over %d threads with %d work each. Remaining work %d allocated to ascending threads\n", howmany, nthread, howmany_per_thread_base, howmany - howmany_per_thread_base*nthread);

  int fft_phase = inverse_transform ? FFTW_BACKWARD : FFTW_FORWARD;
  
  static FFTplanContainer<FloatType> plan_f_base[Dimension]; //destructors deallocate plans
  static FFTplanContainer<FloatType> plan_f_base_p1[Dimension];
      
  static int plan_howmany[Dimension];
  static bool plan_init = false;
  static int plan_fft_phase;
  
  if(!plan_init || plan_howmany[mu] != howmany || fft_phase != plan_fft_phase){
    if(!plan_init) for(int i=0;i<Dimension;i++) plan_howmany[i] = -1;

    typename FFTWwrapper<FloatType>::complexType *tmp_f; //I don't think it actually does anything with this

    plan_fft_phase = fft_phase;
    const int fft_work_per_musite = howmany_per_thread_base;
    const int musite_stride = howmany; //stride between musites
    
    plan_f_base[mu].setPlan(1, &mutotalsites, fft_work_per_musite, 
			    tmp_f, NULL, musite_stride, 1,
			    tmp_f, NULL, musite_stride, 1,
			    plan_fft_phase, FFTW_ESTIMATE);
    plan_f_base_p1[mu].setPlan(1, &mutotalsites, fft_work_per_musite+1, 
			       tmp_f, NULL, musite_stride, 1,
			       tmp_f, NULL, musite_stride, 1,
			       plan_fft_phase, FFTW_ESTIMATE);	
    plan_init = true; //other mu's will still init later
  }
  FFTComplex*fftw_mem = (FFTComplex*)recv_buf;

#pragma omp parallel
  {
    assert(nthread == omp_get_num_threads()); //plans will be messed up if not true
    const int me = omp_get_thread_num();
    size_t thr_work, thr_off;
    thread_work(thr_work, thr_off, howmany, me, nthread);

    const FFTplanContainer<FloatType>* thr_plan_ptr;
    
    if(thr_work == howmany_per_thread_base) thr_plan_ptr = &plan_f_base[mu];
    else if(thr_work == howmany_per_thread_base + 1) thr_plan_ptr = &plan_f_base_p1[mu];
    else assert(0); //catch if logic for thr_work changes

    FFTWwrapper<FloatType>::execute_dft(thr_plan_ptr->getPlan(), fftw_mem + thr_off, fftw_mem + thr_off); 
  }


#endif //GRID_NVCC
  assert(MPI_Waitall(munodes,send_req,status) == MPI_SUCCESS);
      
  fft_opt_mu_timings::get().fft += dclock();

  fft_opt_mu_timings::get().comm_scatter -= dclock();

  //Send back out. Reuse the old send buffers as receive buffers and vice versa
  for(int i=0;i<munodes;i++){ //works fine to send to all nodes, even if this involves a send to self
    assert( MPI_Isend(recv_buf + i*munodes_work[munodecoor]*nf*SiteSize*munodesites, send_buf_sizes[i]*sizeof(ComplexType), MPI_CHAR, munodes_mpiranks[i], 0, MPI_COMM_WORLD, &send_req[i]) == MPI_SUCCESS );
    assert( MPI_Irecv(send_bufs[i], send_buf_sizes[i]*sizeof(ComplexType), MPI_CHAR, munodes_mpiranks[i], MPI_ANY_TAG, MPI_COMM_WORLD, &recv_req[i]) == MPI_SUCCESS );
  }
  
  assert( MPI_Waitall(munodes,recv_req,status) == MPI_SUCCESS );

  fft_opt_mu_timings::get().comm_scatter += dclock();


  fft_opt_mu_timings::get().scatter -= dclock();

  //Poke into output
  for(int i=0;i<munodes;i++){
#pragma omp parallel for
    for(size_t w = 0; w < munodes_work[i]; w++){ //index of orthogonal site within workload for i'th node in mu direction
      const size_t orthsite = munodes_off[i] + w;
      int coor_base[Dimension] = {0};
	  
      //Unmap orthsite into a base coordinate
      size_t rem = orthsite;
      for(int a=0;a<n_orthdirs;a++){
	int dir_a = orthdirs[a];
	coor_base[dir_a] = rem % GJP.NodeSites(dir_a); rem /= GJP.NodeSites(dir_a);
      }

      for(int f=0;f<nf;f++){
	for(int xmu=0;xmu<munodesites;xmu++){	      
	  coor_base[mu] = xmu;
	  ComplexType* to = into.site_ptr(coor_base,f);
	  ComplexType const* frm = send_bufs[i] + SiteSize * (w + munodes_work[i]*( f + nf*xmu ) );
	  if(!inverse_transform) memcpy(to,frm,SiteSize*sizeof(ComplexType));
	  else for(int s=0;s<SiteSize;s++) to[s] = frm[s]/Lmu;
	}
      }
    }
  }

  assert( MPI_Waitall(munodes,send_req,status) == MPI_SUCCESS);
      
  fft_opt_mu_timings::get().scatter += dclock();

  fft_opt_mu_timings::get().setup -= dclock();

  free(recv_buf);
  for(int i=0;i<munodes;i++) free(send_bufs[i]);

  fft_opt_mu_timings::get().setup += dclock();
}
#endif


//Optimized FFT for SIMD data
//Unpacks data into non-SIMD format and invokes regular optimized FFT
#ifdef USE_GRID
template<typename CPSfieldType>
void fft_opt(CPSfieldType &into, const CPSfieldType &from, const bool* do_dirs, const bool inverse_transform = false,
	     typename my_enable_if<_equal<typename ComplexClassify<typename CPSfieldType::FieldSiteType>::type, grid_vector_complex_mark>::value, const int>::type = 0
	     ){ //we can avoid the copies below but with some effort - do at some point
# ifndef USE_MPI
  fft(into,from,do_dirs,inverse_transform);
# else
  //if(!UniqueID()) printf("fft_opt converting Grid SIMD field to scalar field\n");
  typedef typename Grid::GridTypeMapper<typename CPSfieldType::FieldSiteType>::scalar_type ScalarType;
  typedef typename CPSfieldType::FieldMappingPolicy::EquivalentScalarPolicy ScalarDimPol;
  typedef CPSfield<ScalarType, CPSfieldType::FieldSiteSize, ScalarDimPol, StandardAllocPolicy> ScalarFieldType;

  NullObject null_obj;
  ScalarFieldType tmp_in(null_obj);
  ScalarFieldType tmp_out(null_obj);
  tmp_in.importField(from);
  fft_opt(tmp_out, tmp_in, do_dirs, inverse_transform);
  tmp_out.exportField(into);
# endif
}
#endif


//Setup the input parameters for a CPSfield to default settings. This is specifically relevant to SIMD data
//NOTE: If you choose to choose a different SIMD layout, *ensure that the time direction is not SIMD-folded otherwise many A2A algorithms will not work*
template<typename FieldType>
void setupFieldParams(typename FieldType::InputParamType &p){
  _setupFieldParams<typename FieldType::FieldSiteType, typename FieldType::FieldMappingPolicy, typename FieldType::FieldMappingPolicy::ParamType>::doit(p);
}

//Get the underlying CPSfield type associated with a class derived from CPSfield
template<typename DerivedType>
struct baseCPSfieldType{
  typedef CPSfield<typename DerivedType::FieldSiteType, DerivedType::FieldSiteSize, typename DerivedType::FieldMappingPolicy, typename DerivedType::FieldAllocPolicy> type;
};

CPS_END_NAMESPACE
#endif
