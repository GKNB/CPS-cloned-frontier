#ifndef _A2A_FFT_H
#define _A2A_FFT_H

#include<alg/a2a/field_operation.h>
#include<alg/a2a/CPSfield_utils.h>

CPS_START_NAMESPACE

struct VFFTfieldPolicyBasic{
  template<typename T>
  static inline void actionOutputMode(T &v, const int i){}
  template<typename T>
  static inline void actionInputMode(T &v, const int i){}
};
struct VFFTfieldPolicyAllocFree{
  template<typename T>
  static inline void actionOutputMode(T &v, const int i){
    v.allocMode(i);
  }
  template<typename T>
  static inline void actionInputMode(T &v, const int i){
    v.freeMode(i);
  }
};


template<typename OutputType, typename InputType, typename FFTfieldPolicy>
struct _V_fft_impl{
  typedef typename InputType::FermionFieldType FermionFieldType;
  
  static inline void fft(OutputType &to, InputType &from, fieldOperation<FermionFieldType>* mode_preop){
    if(!UniqueID()){ printf("Doing V FFT\n"); fflush(stdout); }
    typedef typename FermionFieldType::InputParamType FieldParamType;
    FieldParamType field_setup = from.getMode(0).getDimPolParams();  
    FermionFieldType tmp(field_setup);
  
    Float preop_time = 0;
    Float fft_time = 0;

    const bool fft_dirs[4] = {true,true,true,false};
  
    for(int mode=0;mode<from.getNmodes();mode++){
      FermionFieldType const* init_gather_from = &from.getMode(mode);
      if(mode_preop != NULL){
	Float dtime = dclock();
	(*mode_preop)(from.getMode(mode),tmp);
	init_gather_from = &tmp;
	preop_time += dclock()-dtime;
      }
      Float dtime = dclock();
      
      FFTfieldPolicy::actionOutputMode(to, mode); //alloc
      
#ifndef MEMTEST_MODE
      cps::fft_opt(to.getMode(mode), *init_gather_from, fft_dirs);
#endif
      fft_time += dclock() - dtime;

      FFTfieldPolicy::actionInputMode(from, mode); //free
    }
    if(!UniqueID()){ printf("Finishing V FFT\n"); fflush(stdout); }
    print_time("A2AvectorVfftw::fft","Preop",preop_time);
    print_time("A2AvectorVfftw::fft","FFT",fft_time);
  }
};



template<typename OutputType, typename InputType, typename FFTfieldPolicy>
struct _V_invfft_impl{
  typedef typename InputType::FermionFieldType FermionFieldType;

  static inline void inversefft(OutputType &to, InputType &from, fieldOperation<FermionFieldType>* mode_postop){
    if(!UniqueID()){ printf("Doing V inverse FFT\n"); fflush(stdout); }
    typedef typename FermionFieldType::InputParamType FieldParamType;
    FieldParamType field_setup = from.getMode(0).getDimPolParams();  
    FermionFieldType tmp(field_setup);
  
    Float postop_time = 0;
    Float fft_time = 0;

    const bool fft_dirs[4] = {true,true,true,false};
    for(int mode=0;mode<from.getNmodes();mode++){
      //if(!UniqueID()) printf("Mode %d, memory before output alloc\n",mode);
      //printMem();
      
      FFTfieldPolicy::actionOutputMode(to, mode); //alloc

      //if(!UniqueID()) printf("Mode %d, memory after output alloc\n",mode);
      //printMem();
      
      FermionFieldType* out = mode_postop == NULL ? &to.getMode(mode) : &tmp;
    
      Float dtime = dclock();
#ifndef MEMTEST_MODE
      cps::fft_opt(*out, from.getMode(mode), fft_dirs, true);
#endif

      //if(!UniqueID()) printf("Mode %d, memory before input free\n",mode);
      //printMem();
      
      FFTfieldPolicy::actionInputMode(from, mode); //alloc

      //if(!UniqueID()) printf("Mode %d, memory after input free\n",mode);
      //printMem();
      
      if(mode_postop != NULL){
	Float dtime = dclock();
	(*mode_postop)(tmp,to.getMode(mode));
	postop_time += dclock()-dtime;
      }
      fft_time += dclock() - dtime;
      //printMem();
    }
    if(!UniqueID()){ printf("Finishing V invert FFT\n"); fflush(stdout); }
    print_time("A2AvectorVfftw::inversefft","FFT",fft_time);
    print_time("A2AvectorVfftw::inversefft","Postop",postop_time);
  }
};


struct WFFTfieldPolicyBasic{
  template<typename T>
  static inline void actionOutputLowMode(T &v, const int i){}
  template<typename T>
  static inline void actionOutputHighMode(T &v, const int i){}
  
  template<typename T>
  static inline void actionInputLowMode(T &v, const int i){}
  template<typename T>
  static inline void actionInputHighMode(T &v, const int i){}
};
struct WFFTfieldPolicyAllocFree{
  template<typename T>
  static inline void actionOutputLowMode(T &v, const int i){
    v.allocLowMode(i);
  }
  template<typename T>
  static inline void actionOutputHighMode(T &v, const int i){
    v.allocHighMode(i);
  }
  
  template<typename T>
  static inline void actionInputLowMode(T &v, const int i){
    v.freeLowMode(i);
  }
  template<typename T>
  static inline void actionInputHighMode(T &v, const int i){
    v.freeHighMode(i);
  }
};


template<typename OutputType, typename InputType, typename FFTfieldPolicy>
struct _W_fft_impl{
  typedef typename InputType::FermionFieldType FermionFieldType;

  inline static void fft(OutputType &to, InputType &from, fieldOperation<FermionFieldType>* mode_preop){
    if(!UniqueID()){ printf("Doing W FFT\n"); fflush(stdout); }
    typedef typename FermionFieldType::InputParamType FieldParamType;
    FieldParamType field_setup = from.getWh(0).getDimPolParams();  
    FermionFieldType tmp(field_setup), tmp2(field_setup);

    Float preop_time = 0;
    Float fft_time = 0;

    const bool fft_dirs[4] = {true,true,true,false};
  
    //Do wl
    for(int mode=0;mode<from.getNl();mode++){
      FermionFieldType const* init_gather_from = &from.getWl(mode);
      if(mode_preop != NULL){
	Float dtime = dclock();
	(*mode_preop)(from.getWl(mode),tmp);
	init_gather_from = &tmp;
	preop_time += dclock()-dtime;
      }
      FFTfieldPolicy::actionOutputLowMode(to, mode); //alloc
      Float dtime = dclock();
#ifndef MEMTEST_MODE
      cps::fft_opt(to.getWl(mode), *init_gather_from, fft_dirs);
#endif
      fft_time += dclock() - dtime;
      FFTfieldPolicy::actionInputLowMode(from, mode); //free
    }
    //Do wh. First we need to uncompact the spin/color index as this is acted upon by the operator
    for(int hit=0;hit<from.getNhits();hit++){
      for(int sc=0;sc<12;sc++){ //spin/color dilution index
	from.getSpinColorDilutedSource(tmp2,hit,sc);
	FermionFieldType* init_gather_from = &tmp2;
	if(mode_preop != NULL){
	  Float dtime = dclock();
	  (*mode_preop)(tmp2,tmp);
	  init_gather_from = &tmp;
	  preop_time += dclock()-dtime;
	}
	Float dtime = dclock();
	FFTfieldPolicy::actionOutputHighMode(to, sc+12*hit); //alloc
#ifndef MEMTEST_MODE
	cps::fft_opt(to.getWh(hit,sc), *init_gather_from, fft_dirs);
#endif
	fft_time += dclock()-dtime;
      }
      FFTfieldPolicy::actionInputHighMode(from, hit); //free
    }
    if(!UniqueID()){ printf("Finishing W FFT\n"); fflush(stdout); }
    print_time("A2AvectorWfftw::fft","Preop",preop_time);
    print_time("A2AvectorWfftw::fft","FFT",fft_time);
  }
};

template<typename OutputType, typename InputType, typename FFTfieldPolicy>
struct _W_invfft_impl{
  typedef typename InputType::FermionFieldType FermionFieldType;

  static inline void inversefft(OutputType &to, InputType &from, fieldOperation<FermionFieldType>* mode_postop){
    if(!UniqueID()){ printf("Doing W inverse FFT\n"); fflush(stdout); }
    typedef typename FermionFieldType::InputParamType FieldParamType;
    FieldParamType field_setup = from.getWh(0,0).getDimPolParams();  
    FermionFieldType tmp(field_setup), tmp2(field_setup);

    Float postop_time = 0;
    Float fft_time = 0;

    const bool fft_dirs[4] = {true,true,true,false};
  
    //Do wl
    for(int mode=0;mode<from.getNl();mode++){
      FFTfieldPolicy::actionOutputLowMode(to, mode); //alloc
      FermionFieldType * unfft_to = mode_postop == NULL ? &to.getWl(mode) : &tmp;

      Float dtime = dclock();
#ifndef MEMTEST_MODE
      cps::fft_opt(*unfft_to, from.getWl(mode), fft_dirs, true);
#endif
      fft_time += dclock() - dtime;

      if(mode_postop != NULL){
	Float dtime = dclock();
	(*mode_postop)(tmp,to.getWl(mode));
	postop_time += dclock()-dtime;
      }
      FFTfieldPolicy::actionInputLowMode(from, mode); //free
    }
    //Do wh. First we need to uncompact the spin/color index as this is acted upon by the operator
    for(int hit=0;hit<from.getNhits();hit++){
      FFTfieldPolicy::actionOutputHighMode(to, hit); //alloc
      typename InputType::ComplexFieldType & to_hit = to.getWh(hit);
    
      const int sc = 0;
      FermionFieldType * compress = mode_postop == NULL ? &tmp2 : &tmp;
      Float dtime = dclock();
#ifndef MEMTEST_MODE
      cps::fft_opt(tmp2, from.getWh(hit,sc), fft_dirs, true);
#endif
      fft_time += dclock()-dtime;

      if(mode_postop != NULL){
	Float dtime = dclock();
	(*mode_postop)(tmp2,tmp);
	postop_time += dclock()-dtime;
      }
      //Should give a multiple of the 12-component unit vector with 1 on index sc
#pragma omp parallel for
      for(int i=0;i<to_hit.nfsites();i++)
	*(to_hit.fsite_ptr(i)) = *(compress->fsite_ptr(i) + sc);

      for(int ssc=0;ssc<12;ssc++) FFTfieldPolicy::actionInputHighMode(from, ssc + 12*hit); //free for all sc
      
    }
    if(!UniqueID()){ printf("Finishing W inverse FFT\n"); fflush(stdout); }
    print_time("A2AvectorWfftw::fftinverse","FFT",fft_time);
    print_time("A2AvectorWfftw::fftinverse","Postop",postop_time);
  }
};
  


CPS_END_NAMESPACE

#endif
