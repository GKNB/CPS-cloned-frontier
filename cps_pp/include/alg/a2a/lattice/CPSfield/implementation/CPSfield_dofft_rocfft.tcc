#ifndef CPSFIELD_DOFFT_ROCFFT_TCC_
#define CPSFIELD_DOFFT_ROCFFT_TCC_

//------------------------------------------------------------------------
//Perform FFT using HIP ROCFFT
//------------------------------------------------------------------------
#ifdef GRID_HIP

CPS_END_NAMESPACE
//Make sure you link -lrocfft
#include <rocfft.h>
#include <map>
CPS_START_NAMESPACE

class ROCFFTplanContainer{
  rocfft_plan_description gpu_description;
  rocfft_plan plan;
  bool active;
public:

  ROCFFTplanContainer(): active(false){}

  ~ROCFFTplanContainer(){
    if(active) 
    {
      assert( rocfft_plan_description_destroy(gpu_description) == rocfft_status_success && "Error: Failed to destroy plan description in ~ROCFFTplanContainer()\n");
      assert( rocfft_plan_destroy(plan) == rocfft_status_success && "Error: Failed to destroy plan in ~ROCFFTplanContainer()\n");
    }
  }
  
  void createPlan(int mutotalsites, int howmany, bool inverse_transform){

    //For reference:
    //    ComplexType* to = send_bufs[i] + SiteSize * (w + munodes_work[i]*( f + nf*xmu ) );  //with musite changing slowest
    //
    //FIXME:Tianle: make sure the below two variables are not used in rocfft!
    //The only non-gibberish explanation of the imbed parameter I've yet seen can be found on page 9 of http://acarus.uson.mx/docs/cuda-5.5/CUFFT_Library.pdf
    //int inembed[1] = {howmany}; //Pointer of size rank that indicates the storage dimen-sions of the input data in memory (up to istride).
    //int* onembed = inembed;

    if(active) 
    {
      assert( rocfft_plan_description_destroy(gpu_description) == rocfft_status_success && "Error: Failed to destroy plan description in createPlan\n");
      assert( rocfft_plan_destroy(plan) == rocfft_status_success && "Error: Failed to destroy plan in createPlan\n");
    }

    rocfft_status rc = rocfft_plan_description_create(&gpu_description);
    assert(rc == rocfft_status_success && "Error: Failed to create plan description\n");

    size_t istride[1] = {static_cast<size_t>(howmany)}; //distance between elements. We have ordered data such that the elements are the slowest index
    size_t *ostride = istride;
    int idist = 1; //distance between first element of two consecutive batches
    int odist=idist;
    rc = rocfft_plan_description_set_data_layout(gpu_description,
                                                 rocfft_array_type_complex_interleaved,
                                                 rocfft_array_type_complex_interleaved,
                                                 NULL,
                                                 NULL,
                                                 1, // input stride length
                                                 istride, // input stride data
                                                 idist, // input batch distance
                                                 1, // output stride length
                                                 ostride, // output stride data
                                                 odist); // ouptut batch distance
    assert(rc == rocfft_status_success && "Error: Failed to set data layout\n");

    const rocfft_transform_type direction = (inverse_transform ? rocfft_transform_type_complex_inverse : rocfft_transform_type_complex_forward);
    int rank=1;
    size_t n[1] = {static_cast<size_t>(mutotalsites)}; //size of each dimension
    rc = rocfft_plan_create(&plan,
		    	    rocfft_placement_inplace,
                            direction,
                            rocfft_precision_double,
                            rank, // Dimension
                            n, // lengths
                            howmany, // Number of transforms
                            gpu_description); // Description
    assert(rc == rocfft_status_success && "Error: Failed to create plan\n");

    active = true;
  }

  void destroyPlan(){
    if(active) 
    {
      assert( rocfft_plan_description_destroy(gpu_description) == rocfft_status_success && "Error: Failed to destroy plan description in destroyPlan\n");
      assert( rocfft_plan_destroy(plan) == rocfft_status_success && "Error: Failed to destroy plan in destroyPlan\n");
      active = false;
    }
  }

  rocfft_plan &getPlan(){ return plan; }
};

struct CPSfieldFFTplanParamsROCFFT{
  int mutotalsites; //total sites in mu direction (global)
  int howmany; //how many FFTs
  bool inverse_transform;

  CPSfieldFFTplanParamsROCFFT(int _mutotalsites, int _howmany, bool _inverse_transform):
    mutotalsites(_mutotalsites), howmany(_howmany), inverse_transform(_inverse_transform){}

  CPSfieldFFTplanParamsROCFFT(){}

  bool operator<(const CPSfieldFFTplanParamsROCFFT &r) const{
    if(mutotalsites == r.mutotalsites){
      return (2 * howmany + inverse_transform) < (2 * r.howmany + r.inverse_transform);
    }else return mutotalsites < r.mutotalsites;
  }

  bool operator==(const CPSfieldFFTplanParamsROCFFT &r) const{
    return mutotalsites == r.mutotalsites && howmany == r.howmany && inverse_transform == r.inverse_transform;
  }

  void createPlan(ROCFFTplanContainer &plan) const{
    plan.createPlan(mutotalsites, howmany, inverse_transform);
  }

};



//data_size = # of complex numbers on data array
template<typename FloatType, int Dimension,
	 typename std::enable_if<std::is_same<FloatType,double>::value, int>::type = 0 //only for double currently
	 >
void CPSfield_do_fft_rocfft(const int mutotalsites, const size_t howmany,
			    const bool inverse_transform, 
			    typename FFTWwrapper<FloatType>::complexType *data,
			    const size_t data_size){

//  rocfft_setup();	//FIXME: Tianle: Need to test 1). if setup can only be called once; 2). if 1) is true, if CPSfield_do_fft_rocfft is only called once
  typedef typename FFTWwrapper<FloatType>::complexType ComplexType;

  //Get the plan
  static std::map<CPSfieldFFTplanParamsROCFFT, ROCFFTplanContainer> fft_plans;
  
  CPSfieldFFTplanParamsROCFFT params(mutotalsites, howmany, inverse_transform);

  rocfft_plan *plan_ptr;
  auto it = fft_plans.find(params);
  if(it == fft_plans.end())
  {
    ROCFFTplanContainer &plan_con = fft_plans[params];
    params.createPlan(plan_con);
    plan_ptr = &plan_con.getPlan();
  }
  else 
   plan_ptr = &it->second.getPlan();

  rocfft_execution_info planinfo = NULL;
  assert(rocfft_execution_info_create(&planinfo) == rocfft_status_success && "Error: Failed to create execution info\n");

  //Do the FFT
  static_assert(sizeof(double2) == sizeof(ComplexType)); 
  double2* device_in = (double2*)device_alloc_check(data_size * sizeof(double2));
  assert(hipMemcpy(device_in, data, data_size * sizeof(double2), hipMemcpyHostToDevice) == hipSuccess);
 
  assert(rocfft_execute(*plan_ptr, (void**)&device_in, NULL, planinfo) == rocfft_status_success && "Error: Failed to execute\n");
  
  assert(hipMemcpy(data, device_in,  data_size * sizeof(double2), hipMemcpyDeviceToHost) == hipSuccess);

  rocfft_execution_info_destroy(planinfo);
  device_free(device_in);
//  rocfft_cleanup();	//FIXME: Tianle: Need to figure out where to put this!!!
}

#endif //GRID_HIP

#endif
