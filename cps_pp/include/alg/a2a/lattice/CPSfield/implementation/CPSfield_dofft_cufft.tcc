#ifndef CPSFIELD_DOFFT_CUFFT_TCC_
#define CPSFIELD_DOFFT_CUFFT_TCC_

//------------------------------------------------------------------------
//Perform FFT using Cuda CUFFT
//------------------------------------------------------------------------
#ifdef GRID_CUDA

CPS_END_NAMESPACE
//Make sure you link -lcufft
#include <cufft.h>
#include <map>
CPS_START_NAMESPACE


CPS_END_NAMESPACE
#include <map>
CPS_START_NAMESPACE

class CUFFTplanContainer{
  cufftHandle plan;
  bool active;
public:

  CUFFTplanContainer(): active(false){}

  ~CUFFTplanContainer(){
    if(active) assert( cufftDestroy(plan) == CUFFT_SUCCESS );
  }
  
  void createPlan(int mutotalsites, int howmany){
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
    
    if(active) assert( cufftDestroy(plan) == CUFFT_SUCCESS );

    assert( cufftCreate(&plan)== CUFFT_SUCCESS );
    assert( cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch) == CUFFT_SUCCESS );    
    active = true;
  }

  void destroyPlan(){
    if(active){
      assert( cufftDestroy(plan) == CUFFT_SUCCESS );
      active = false;
    }
  }

  cufftHandle &getPlan(){ return plan; }
};

struct CPSfieldFFTplanParamsCUFFT{
  int mutotalsites; //total sites in mu direction (global)
  int howmany; //how many FFTs

  CPSfieldFFTplanParamsCUFFT(int _mutotalsites, int _howmany):
    mutotalsites(_mutotalsites), howmany(_howmany){}

  CPSfieldFFTplanParamsCUFFT(){}

  bool operator<(const CPSfieldFFTplanParamsCUFFT &r) const{
    if(mutotalsites == r.mutotalsites){
      return howmany < r.howmany;
    }else return mutotalsites < r.mutotalsites;
  }

  bool operator==(const CPSfieldFFTplanParamsCUFFT &r) const{
    return mutotalsites == r.mutotalsites && howmany == r.howmany;
  }

  void createPlan(CUFFTplanContainer &plan) const{
    plan.createPlan(mutotalsites, howmany);
  }

};



//data_size = # of complex numbers on data array
template<typename FloatType, int Dimension,
	 typename std::enable_if<std::is_same<FloatType,double>::value, int>::type = 0 //only for double currently
	 >
void CPSfield_do_fft_cufft(const int mutotalsites, const size_t howmany,
			   const bool inverse_transform, 
			   typename FFTWwrapper<FloatType>::complexType *data,
			   const size_t data_size){

  typedef typename FFTWwrapper<FloatType>::complexType ComplexType;

  //Get the plan
  static std::map<CPSfieldFFTplanParamsCUFFT, CUFFTplanContainer> fft_plans;
  
  CPSfieldFFTplanParamsCUFFT params(mutotalsites, howmany);

  cufftHandle *plan_ptr;
  auto it = fft_plans.find(params);
  if(it == fft_plans.end()){
    CUFFTplanContainer &plan_con = fft_plans[params];
    params.createPlan(plan_con);
    plan_ptr = &plan_con.getPlan();
  }else plan_ptr = &it->second.getPlan();

  //Do the FFT
  static_assert(sizeof(cufftDoubleComplex) == sizeof(ComplexType)); 
  cufftDoubleComplex* device_in = (cufftDoubleComplex*)device_alloc_check(data_size * sizeof(cufftDoubleComplex));
  assert(cudaMemcpy(device_in, data, data_size * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice) == cudaSuccess);
  
  int fft_phase = inverse_transform ? CUFFT_INVERSE : CUFFT_FORWARD;
  assert( cufftExecZ2Z(*plan_ptr, device_in,  device_in, fft_phase) == CUFFT_SUCCESS );
  
  assert(cudaMemcpy(data, device_in,  data_size * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost) == cudaSuccess);
  
  device_free(device_in);
}

#endif //GRID_CUDA

#endif
