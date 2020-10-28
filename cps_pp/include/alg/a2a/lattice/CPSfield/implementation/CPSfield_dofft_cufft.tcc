#ifndef CPSFIELD_DOFFT_CUFFT_TCC_
#define CPSFIELD_DOFFT_CUFFT_TCC_

//------------------------------------------------------------------------
//Perform FFT using Cuda CUFFT
//------------------------------------------------------------------------
#ifdef GRID_CUDA

CPS_END_NAMESPACE
//Make sure you link -lcufft
#include <cufft.h>
CPS_START_NAMESPACE

template<typename FloatType, int Dimension,
	 typename std::enable_if<std::is_same<FloatType,double>, int>::type = 0 //only for double currently
	 >
void CPSfield_do_fft_cufft(const int mutotalsites, const size_t howmany,
			  const bool inverse_transform, 
			  typename FFTWwrapper<FloatType>::complexType *data){

  typedef typename FFTWwrapper<FloatType>::complexType ComplexType;
  static size_t plan_howmany[Dimension];
  static bool plan_init = false;
  static cufftHandle handle[Dimension];
  
  if(!plan_init || plan_howmany[mu] != howmany){
    if(!plan_init) for(int i=0;i<Dimension;i++) plan_howmany[i] = 0; //must be initialized at some point
    //For reference:
    //    ComplexType* to = send_bufs[i] + SiteSize * (w + munodes_work[i]*( f + nf*xmu ) );  //with musite changing slowest
    //
    //    cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed,
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

    if(plan_init && plan_howmany[mu] != 0){
      assert( cufftDestroy(handle[mu]) == CUFFT_SUCCESS );
    }

    assert( cufftCreate(&handle[mu])== CUFFT_SUCCESS );
    assert( cufftPlanMany(&handle[mu], rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch) == CUFFT_SUCCESS );    
    plan_howmany[mu] = howmany;
    plan_init = true; //other mu's will still init later	
  }

  static_assert(sizeof(cufftDoubleComplex) == sizeof(ComplexType));
  
  cufftDoubleComplex* device_in = (cufftDoubleComplex*)device_alloc_check(bufsz * sizeof(cufftDoubleComplex));
  assert(cudaMemcpy(device_in, data, bufsz * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice) == cudaSuccess);
  
  int fft_phase = inverse_transform ? CUFFT_INVERSE : CUFFT_FORWARD;
  assert( cufftExecZ2Z(handle[mu], device_in,  device_in, fft_phase) == CUFFT_SUCCESS );
  
  assert(cudaMemcpy(data, device_in,  bufsz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost) == cudaSuccess);
  
  device_free(device_in);
}

#endif //GRID_CUDA

#endif
