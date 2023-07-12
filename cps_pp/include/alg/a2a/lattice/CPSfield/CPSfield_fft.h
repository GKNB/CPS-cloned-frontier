#ifndef CPS_FIELD_FFT_H
#define CPS_FIELD_FFT_H

#include "CPSfield.h"
#include "CPSfield_utils.h"

CPS_START_NAMESPACE

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
  typedef CPSfield<ScalarType, CPSfieldType::FieldSiteSize, ScalarMapPol> ScalarFieldType;

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
    std::string method;
    
    timers(): calls(0), setup(0), gather(0), comm_gather(0), fft(0), comm_scatter(0), scatter(0), method("unset"){}

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
      printf("calls=%zu method=%s setup=%g gather=%g comm_gather=%g fft=%g comm_scatter=%g scatter=%g\n", calls, method.c_str(), setup, gather, comm_gather, fft, comm_scatter, scatter);
    }
  };
  static timers & get(){ static timers t; return t; }
};

#include "implementation/CPSfield_dofft_fftw.tcc"
#include "implementation/CPSfield_dofft_cufft.tcc"
#include "implementation/CPSfield_dofft_rocfft.tcc"
#include "implementation/CPSfield_dofft_onemkl.tcc"

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
  CPSautoView(from_v,from,HostRead);
  
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
	  ComplexType const* frm = from_v.site_ptr(coor_base,f);

	  memcpy(to,frm,SiteSize*sizeof(ComplexType));
	}
      }
    }
  }

  fft_opt_mu_timings::get().gather += dclock();

  fft_opt_mu_timings::get().comm_gather -= dclock();

  MPI_Request send_req[munodes], recv_req[munodes];
  MPI_Status status[munodes];

  //Prepare recv buf
  const size_t bufsz = munodes_work[munodecoor] * mutotalsites * nf * SiteSize; //complete line in mu for each orthogonal coordinate
  ComplexType* recv_buf = (ComplexType*)malloc_check(bufsz * sizeof(ComplexType) );

  //Setup send/receive    
  for(int i=0;i<munodes;i++){ //works fine to send to all nodes, even if this involves a send to self.
    assert( MPI_Isend(send_bufs[i], send_buf_sizes[i]*sizeof(ComplexType), MPI_CHAR, 
		      munodes_mpiranks[i], 0, MPI_COMM_WORLD, &send_req[i]) == MPI_SUCCESS );

    assert( MPI_Irecv(recv_buf + i*munodes_work[munodecoor]*nf*SiteSize*munodesites, send_buf_sizes[i]*sizeof(ComplexType), MPI_CHAR, 
		      munodes_mpiranks[i], MPI_ANY_TAG, MPI_COMM_WORLD, &recv_req[i]) == MPI_SUCCESS );
  }      
  assert( MPI_Waitall(munodes,recv_req,status) == MPI_SUCCESS );
   
  fft_opt_mu_timings::get().comm_gather += dclock();
  
  fft_opt_mu_timings::get().fft -= dclock();

  //Do FFT
  const size_t howmany = munodes_work[munodecoor] * nf * SiteSize;

#ifdef GRID_CUDA
  fft_opt_mu_timings::method = "cufft";
  CPSfield_do_fft_cufft<FloatType,Dimension>(mutotalsites, howmany, inverse_transform, (FFTComplex*)recv_buf, bufsz);
#elif defined(GRID_HIP)
  fft_opt_mu_timings::method = "rocfft";
  CPSfield_do_fft_rocfft<FloatType,Dimension>(mutotalsites, howmany, inverse_transform, (FFTComplex*)recv_buf, bufsz);
#elif defined(GRID_SYCL)
  fft_opt_mu_timings::method = "onemkl";
  CPSfield_do_fft_onemkl<FloatType,Dimension>(mutotalsites, howmany, inverse_transform, (FFTComplex*)recv_buf, bufsz);
#else //GRID_CUDA
  fft_opt_mu_timings::method = "fftw";
  CPSfield_do_fft_fftw<FloatType>(mutotalsites, howmany, inverse_transform, (FFTComplex*)recv_buf);
#endif //!GRID_CUDA
  assert(MPI_Waitall(munodes,send_req,status) == MPI_SUCCESS);
      
  fft_opt_mu_timings::get().fft += dclock();

  //Send back out. Reuse the old send buffers as receive buffers and vice versa
  fft_opt_mu_timings::get().comm_scatter -= dclock();
  for(int i=0;i<munodes;i++){ //works fine to send to all nodes, even if this involves a send to self
    assert( MPI_Isend(recv_buf + i*munodes_work[munodecoor]*nf*SiteSize*munodesites, send_buf_sizes[i]*sizeof(ComplexType), MPI_CHAR, munodes_mpiranks[i], 0, MPI_COMM_WORLD, &send_req[i]) == MPI_SUCCESS );
    assert( MPI_Irecv(send_bufs[i], send_buf_sizes[i]*sizeof(ComplexType), MPI_CHAR, munodes_mpiranks[i], MPI_ANY_TAG, MPI_COMM_WORLD, &recv_req[i]) == MPI_SUCCESS );
  }
  
  assert( MPI_Waitall(munodes,recv_req,status) == MPI_SUCCESS );

  fft_opt_mu_timings::get().comm_scatter += dclock();


  fft_opt_mu_timings::get().scatter -= dclock();

  //Poke into output
  CPSautoView(into_v,into,HostWrite);
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
	  ComplexType* to = into_v.site_ptr(coor_base,f);
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
#error "Not using MPI"
  fft(into,from,do_dirs,inverse_transform);
# else
  //if(!UniqueID()) printf("fft_opt converting Grid SIMD field to scalar field\n");
  typedef typename Grid::GridTypeMapper<typename CPSfieldType::FieldSiteType>::scalar_type ScalarType;
  typedef typename CPSfieldType::FieldMappingPolicy::EquivalentScalarPolicy ScalarDimPol;
  typedef CPSfield<ScalarType, CPSfieldType::FieldSiteSize, ScalarDimPol> ScalarFieldType;

  NullObject null_obj;
  ScalarFieldType tmp_in(null_obj);
  ScalarFieldType tmp_out(null_obj);
  tmp_in.importField(from);
  fft_opt(tmp_out, tmp_in, do_dirs, inverse_transform);
  tmp_out.exportField(into);
# endif
}
#endif


CPS_END_NAMESPACE

#endif

