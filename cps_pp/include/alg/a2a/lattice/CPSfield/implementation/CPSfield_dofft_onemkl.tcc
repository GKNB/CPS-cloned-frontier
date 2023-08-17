#ifndef CPSFIELD_DOFFT_ONEMKL_TCC_
#define CPSFIELD_DOFFT_ONEMKL_TCC_

//------------------------------------------------------------------------
//Perform FFT using Intel oneMKL FFT
//------------------------------------------------------------------------
#ifdef GRID_SYCL

CPS_END_NAMESPACE
//Make sure you link-lcufft
#include <mkl.h>
#include <mkl_dfti.h>
#include <map>

CPS_START_NAMESPACE

class oneMKLfftPlanContainer{
  DFTI_DESCRIPTOR_HANDLE plan;
  bool active;
public:

  oneMKLfftPlanContainer(): active(false){}

  ~oneMKLfftPlanContainer(){
    if(active) assert( DftiFreeDescriptor(&plan) == 0 );
  }
  
  void createPlan(int mutotalsites, int howmany){
    std::cout << "oneMKL FFT wrapper creating plan with mutotalsites=" << mutotalsites << " and howmany=" << howmany << std::endl;

    if(active) assert( DftiFreeDescriptor(&plan) == 0 );
    //For reference:
    //    ComplexType* to = send_bufs[i] + SiteSize * (w + munodes_work[i]*( f + nf*xmu ) );  //with musite changing slowest
    //
    //    const size_t howmany = munodes_work[munodecoor] * nf * SiteSize;
    //
    //status = DftiCreateDescriptor(&desc_handle,precision,forward_domain,dimension,length);
    assert( DftiCreateDescriptor(&plan, DFTI_DOUBLE, DFTI_COMPLEX, 1, mutotalsites ) == 0 );
    assert( DftiSetValue( plan, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)howmany ) == 0 );
    //Different FFTs are separated by 1 and their elements are howmany apart
    assert( DftiSetValue( plan, DFTI_INPUT_DISTANCE,  1 ) == 0); //The distance between the first data elements of consecutive data sets

    //cf https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/fourier-transform-functions/fft-functions/configuration-settings/dfti-input-strides-dfti-output-strides.html
    MKL_LONG istrides[2] = {0, howmany};
    assert( DftiSetValue( plan, DFTI_INPUT_STRIDES, istrides ) == 0 );

    assert( DftiCommitDescriptor( plan ) == 0 );
    active = true;
  }

  void destroyPlan(){
    if(active){
      assert( DftiFreeDescriptor(&plan) == 0 );
      active = false;
    }
  }

  DFTI_DESCRIPTOR_HANDLE &getPlan(){ return plan; }
};

struct CPSfieldFFTplanParamsoneMKL{
  int mutotalsites; //total sites in mu direction (global)
  int howmany; //how many FFTs

  CPSfieldFFTplanParamsoneMKL(int _mutotalsites, int _howmany):
    mutotalsites(_mutotalsites), howmany(_howmany){}

  CPSfieldFFTplanParamsoneMKL(){}

  bool operator<(const CPSfieldFFTplanParamsoneMKL &r) const{
    if(mutotalsites == r.mutotalsites){
      return howmany < r.howmany;
    }else return mutotalsites < r.mutotalsites;
  }

  bool operator==(const CPSfieldFFTplanParamsoneMKL &r) const{
    return mutotalsites == r.mutotalsites && howmany == r.howmany;
  }

  void createPlan(oneMKLfftPlanContainer &plan) const{
    plan.createPlan(mutotalsites, howmany);
  }

};



//data_size = # of complex numbers on data array
template<typename FloatType, int Dimension,
	 typename std::enable_if<std::is_same<FloatType,double>::value, int>::type = 0 //only for double currently
	 >
void CPSfield_do_fft_onemkl(const int mutotalsites, const size_t howmany,
			   const bool inverse_transform, 
			   typename FFTWwrapper<FloatType>::complexType *data,
			   const size_t data_size){
  LOGA2A << "Doing FFT with oneMKL" << std::endl;
  typedef typename FFTWwrapper<FloatType>::complexType ComplexType;

  //Get the plan
  static std::map<CPSfieldFFTplanParamsoneMKL, oneMKLfftPlanContainer> fft_plans;
  
  CPSfieldFFTplanParamsoneMKL params(mutotalsites, howmany);

  DFTI_DESCRIPTOR_HANDLE *plan_ptr;
  auto it = fft_plans.find(params);
  if(it == fft_plans.end()){
    oneMKLfftPlanContainer &plan_con = fft_plans[params];
    params.createPlan(plan_con);
    plan_ptr = &plan_con.getPlan();
  }else plan_ptr = &it->second.getPlan();

  //Do the FFT
  double *ddata = (double*)data;
  if(inverse_transform){
    assert( DftiComputeBackward(*plan_ptr, ddata) == 0 );
  }else{
    assert( DftiComputeForward(*plan_ptr, ddata) == 0 );
  }
}

#endif //GRID_CUDA

#endif
