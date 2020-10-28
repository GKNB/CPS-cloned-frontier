#ifndef CPSFIELD_DOFFT_FFTW_TCC_
#define CPSFIELD_DOFFT_FFTW_TCC_

//------------------------------------------------------------------------
//Perform FFT using FFTW (threaded)
//------------------------------------------------------------------------

CPS_END_NAMESPACE
#include <map>
CPS_START_NAMESPACE

struct CPSfieldFFTplanParams{
  int mutotalsites; //total sites in mu direction (global)
  int fft_work_per_musite; //orthogonal sites * nf * siteSize divided over threads
  int musite_stride; //stride between elements of each fft
  int plan_fft_phase; //forwards or backwards

  CPSfieldFFTplanParams(int _mutotalsites, int _fft_work_per_musite, int _musite_stride, int _plan_fft_phase):
    mutotalsites(_mutotalsites), fft_work_per_musite(_fft_work_per_musite), 
    musite_stride(_musite_stride), plan_fft_phase(_plan_fft_phase){}

  CPSfieldFFTplanParams(){}

  bool operator<(const CPSfieldFFTplanParams &r) const{
    if(mutotalsites == r.mutotalsites){
      if(fft_work_per_musite == r.fft_work_per_musite){
	if(musite_stride == r.musite_stride){
	  return plan_fft_phase < r.plan_fft_phase;	  
	}else return musite_stride < r.musite_stride;
      }else return fft_work_per_musite < r.fft_work_per_musite;
    }else return mutotalsites < r.mutotalsites;
  }

  bool operator==(const CPSfieldFFTplanParams &r) const{
    return mutotalsites == r.mutotalsites &&
      fft_work_per_musite == r.fft_work_per_musite &&
      musite_stride == r.musite_stride && 
      plan_fft_phase == r.plan_fft_phase;
  }

  template<typename FloatType>
  void createPlan(FFTplanContainer<FloatType> &plan) const{
    typename FFTWwrapper<FloatType>::complexType *tmp_f = NULL; //not used
    plan.setPlan(1, &mutotalsites, fft_work_per_musite, 
		 tmp_f, NULL, musite_stride, 1,
		 tmp_f, NULL, musite_stride, 1,
		 plan_fft_phase, FFTW_ESTIMATE);
  }

};


template<typename FloatType>
void CPSfield_do_fft_fftw(const int mutotalsites, const size_t howmany,
			  const bool inverse_transform, 
			  typename FFTWwrapper<FloatType>::complexType *data){

  const int nthread = omp_get_max_threads();
  const size_t howmany_per_thread_base = howmany / nthread;

  //Divide work orthogonal to mu, 'howmany', over threads. Note, this may not divide howmany equally. The difference is made up by adding 1 unit of work to threads in ascending order until total work matches. Thus we need 2 plans: 1 for the base amount and one for the base+1
  int fft_phase = inverse_transform ? FFTW_BACKWARD : FFTW_FORWARD;
  int fft_work_per_musite = howmany_per_thread_base;
  int musite_stride = howmany; //stride between musites

  //Get the plan
  typedef std::map<CPSfieldFFTplanParams, FFTplanContainer<FloatType> > PlanMapType;
  typedef typename PlanMapType::iterator PlanMapIterator;

  static PlanMapType fft_plans;

  CPSfieldFFTplanParams params(mutotalsites, fft_work_per_musite, musite_stride, fft_phase);
  CPSfieldFFTplanParams params_p1(mutotalsites, fft_work_per_musite+1, musite_stride, fft_phase);

  PlanMapIterator it = fft_plans.find(params);
  if(it == fft_plans.end()) params.createPlan(fft_plans[params]);
  it = fft_plans.find(params_p1);
  if(it == fft_plans.end()) params_p1.createPlan(fft_plans[params_p1]);

  FFTplanContainer<FloatType> &plan = fft_plans[params];
  FFTplanContainer<FloatType> &plan_p1 = fft_plans[params_p1];

  //Perform the FFTs
#pragma omp parallel
  {
    assert(nthread == omp_get_num_threads()); //plans will be messed up if not true
    const int me = omp_get_thread_num();
    size_t thr_work, thr_off;
    thread_work(thr_work, thr_off, howmany, me, nthread);

    const FFTplanContainer<FloatType>* thr_plan_ptr;
    
    if(thr_work == howmany_per_thread_base) thr_plan_ptr = &plan;
    else if(thr_work == howmany_per_thread_base + 1) thr_plan_ptr = &plan_p1;
    else assert(0); //catch if logic for thr_work changes

    FFTWwrapper<FloatType>::execute_dft(thr_plan_ptr->getPlan(), data + thr_off, data + thr_off); 
  }
}

#endif
