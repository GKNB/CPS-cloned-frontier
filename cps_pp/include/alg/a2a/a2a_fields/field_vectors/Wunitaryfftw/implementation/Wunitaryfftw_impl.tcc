template< typename mf_Policies>
double A2AvectorWunitaryfftw<mf_Policies>::Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params){
  return VW_Mbyte_size<A2AvectorWunitaryfftw<mf_Policies> >(_args,field_setup_params);
}

template< typename mf_Policies>
void A2AvectorWunitaryfftw<mf_Policies>::initialize(const FieldInputParamType &field_setup_params){
  checkSIMDparams<FieldInputParamType>::check(field_setup_params);
  wl.resize(nl); this->allocInitializeLowModeFields(wl,field_setup_params);
  wh.resize(this->getNhighModes()); this->allocInitializeHighModeFields(wh,field_setup_params);
}

template< typename mf_Policies>
A2AvectorWunitaryfftw<mf_Policies>::A2AvectorWunitaryfftw(const A2AArg &_args): TimePackedIndexDilution(_args){ initialize(NullObject()); }

template< typename mf_Policies>
A2AvectorWunitaryfftw<mf_Policies>::A2AvectorWunitaryfftw(const A2AArg &_args, const FieldInputParamType &field_setup_params): TimePackedIndexDilution(_args){ initialize(field_setup_params); }

template< typename mf_Policies>
A2AvectorWunitaryfftw<mf_Policies>::A2AvectorWunitaryfftw(const A2Aparams &_args): TimePackedIndexDilution(_args){ initialize(NullObject()); }

template< typename mf_Policies>
A2AvectorWunitaryfftw<mf_Policies>::A2AvectorWunitaryfftw(const A2Aparams &_args, const FieldInputParamType &field_setup_params): TimePackedIndexDilution(_args){ initialize(field_setup_params); }

struct WunitaryFFTfieldPolicyBasic{
  template<typename T>
  static inline void actionOutputLowMode(T &v, const int i){}
  template<typename T>
  static inline void actionOutputHighMode(T &v, const int i){}
  
  template<typename T>
  static inline void actionInputLowMode(T &v, const int i){}
  template<typename T>
  static inline void actionInputHighMode(T &v, const int i){}
};
struct WunitaryFFTfieldPolicyAllocFree{
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
struct _Wunitary_fft_impl{
  typedef typename InputType::FermionFieldType FermionFieldType;

  inline static void fft(OutputType &to, InputType &from, fieldOperation<FermionFieldType>* mode_preop){
    fft_opt_mu_timings::get().reset();
    fft_opt_timings::get().reset();
    a2a_printf("Doing Wunitary FFT\n");
    typedef typename FermionFieldType::InputParamType FieldParamType;
    FieldParamType field_setup = from.getFieldInputParams();
    FermionFieldType tmp(field_setup), tmp2(field_setup);

    Float action_output_mode_time = 0;
    Float action_input_mode_time = 0;
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
      Float dtime = dclock();
      FFTfieldPolicy::actionOutputLowMode(to, mode); //alloc
      action_output_mode_time += dclock() - dtime;

      dtime = dclock();
#ifndef MEMTEST_MODE
      cps::fft_opt(to.getWl(mode), *init_gather_from, fft_dirs);
#endif
      fft_time += dclock() - dtime;

      dtime = dclock();
      FFTfieldPolicy::actionInputLowMode(from, mode); //free
      action_input_mode_time += dclock() - dtime;
    }
    //Do wh. First we need to uncompact the spin/color index as this is acted upon by the operator
    for(int hidx=0;hidx<from.getNhighModes();hidx++){
      int hit, flavor;
      from.indexUnmap(hidx,hit,flavor);

      for(int sc=0;sc<12;sc++){ //spin/color dilution index
	from.getSpinColorDilutedSource(tmp2,hidx,sc);
	FermionFieldType* init_gather_from = &tmp2;
	if(mode_preop != NULL){
	  Float dtime = dclock();
	  (*mode_preop)(tmp2,tmp);
	  init_gather_from = &tmp;
	  preop_time += dclock()-dtime;
	}
	Float dtime = dclock();
	FFTfieldPolicy::actionOutputHighMode(to, to.indexMap(hit,sc,flavor)); //alloc
	action_output_mode_time += dclock() - dtime;

	dtime = dclock();
#ifndef MEMTEST_MODE
	cps::fft_opt(to.getWh(hit,sc,flavor), *init_gather_from, fft_dirs);
#endif
	fft_time += dclock()-dtime;
      }

      Float dtime = dclock();
      FFTfieldPolicy::actionInputHighMode(from, hidx); //free
      action_input_mode_time += dclock() - dtime;
    }
    a2a_printf("Finishing Wunitary FFT\n");
    a2a_print_time("A2AvectorWunitaryfftw::fft","Preop",preop_time);
    a2a_print_time("A2AvectorWunitaryfftw::fft","FFT",fft_time);
    a2a_print_time("A2AvectorWunitaryfftw::fft","actionOutputMode",action_output_mode_time);
    a2a_print_time("A2AvectorWunitaryfftw::fft","actionInputMode",action_input_mode_time);
    fft_opt_timings::get().print();
    fft_opt_mu_timings::get().print();
  }
};

//Set this object to be the fast Fourier transform of the input field
//Can optionally supply an object mode_preop that performs a transformation on each mode prior to the FFT
template< typename mf_Policies>
void A2AvectorWunitaryfftw<mf_Policies>::fft(const A2AvectorWunitary<mf_Policies> &from, fieldOperation<FermionFieldType>* mode_preop){
  _Wunitary_fft_impl<A2AvectorWunitaryfftw<mf_Policies>, const A2AvectorWunitary<mf_Policies>, WunitaryFFTfieldPolicyBasic>::fft(*this,from,mode_preop);
}


template<typename OutputType, typename InputType, typename FFTfieldPolicy>
struct _Wunitary_invfft_impl{
  typedef typename InputType::FermionFieldType FermionFieldType;

  static inline void inversefft(OutputType &to, InputType &from, fieldOperation<FermionFieldType>* mode_postop){
    a2a_printf("Doing Wunitary inverse FFT\n");
    fft_opt_mu_timings::get().reset();
    fft_opt_timings::get().reset();
    typedef typename FermionFieldType::InputParamType FieldParamType;
    FieldParamType field_setup = from.getFieldInputParams();
    FermionFieldType tmp(field_setup), tmp2(field_setup);

    Float action_output_mode_time = 0;
    Float action_input_mode_time = 0;
    Float postop_time = 0;
    Float fft_time = 0;

    const bool fft_dirs[4] = {true,true,true,false};
  
    //Do wl
    for(int mode=0;mode<from.getNl();mode++){
      Float dtime = dclock();
      FFTfieldPolicy::actionOutputLowMode(to, mode); //alloc
      action_output_mode_time += dclock() - dtime;
      
      FermionFieldType * unfft_to = mode_postop == NULL ? &to.getWl(mode) : &tmp;

      dtime = dclock();
#ifndef MEMTEST_MODE
      cps::fft_opt(*unfft_to, from.getWl(mode), fft_dirs, true);
#endif
      fft_time += dclock() - dtime;

      if(mode_postop != NULL){
	dtime = dclock();
	(*mode_postop)(tmp,to.getWl(mode));
	postop_time += dclock()-dtime;
      }

      dtime = dclock();
      FFTfieldPolicy::actionInputLowMode(from, mode); //free
      action_input_mode_time += dclock() - dtime;
    }
    //Do wh. We need to recompact the spin/color index
    for(int hidx=0;hidx<to.getNhighModes();hidx++){
      int hit,flavor;
      to.indexUnmap(hidx,hit,flavor);

      Float dtime = dclock();
      FFTfieldPolicy::actionOutputHighMode(to, hidx); //alloc
      action_output_mode_time += dclock() - dtime;

      typename InputType::ComplexFieldType & to_mode = to.getWh(hidx);
    
      const int sc = 0;
      FermionFieldType * compress = mode_postop == NULL ? &tmp2 : &tmp;
      
      dtime = dclock();
#ifndef MEMTEST_MODE
      cps::fft_opt(tmp2, from.getWh(hit,sc,flavor), fft_dirs, true);
#endif
      fft_time += dclock()-dtime;

      if(mode_postop != NULL){
	dtime = dclock();
	(*mode_postop)(tmp2,tmp);
	postop_time += dclock()-dtime;
      }
      //Should give a multiple of the 12-component unit vector with 1 on index sc
      {
	CPSautoView(to_mode_v,to_mode,HostWrite);
	CPSautoView(compress_v,(*compress),HostRead);

#pragma omp parallel for
	for(int i=0;i<to_mode.nfsites();i++)
	  *(to_mode_v.fsite_ptr(i)) = *(compress_v.fsite_ptr(i) + sc);
      }

      dtime = dclock();
      for(int ssc=0;ssc<12;ssc++) FFTfieldPolicy::actionInputHighMode(from, from.indexMap(hit,ssc,flavor)); //free for all sc
      action_input_mode_time += dclock() - dtime;      
    }
    a2a_printf("Finishing W inverse FFT\n");
    a2a_print_time("A2AvectorWunitaryfftw::fftinverse","FFT",fft_time);
    a2a_print_time("A2AvectorWunitaryfftw::fftinverse","Postop",postop_time);
    a2a_print_time("A2AvectorWunitaryfftw::fftinverse","actionOutputMode",action_output_mode_time);
    a2a_print_time("A2AvectorWunitaryfftw::fftinverse","actionInputMode",action_input_mode_time);
    fft_opt_timings::get().print();
    fft_opt_mu_timings::get().print();
  }
};

template< typename mf_Policies>
void A2AvectorWunitaryfftw<mf_Policies>::inversefft(A2AvectorWunitary<mf_Policies> &to, fieldOperation<FermionFieldType>* mode_postop) const{
  _Wunitary_invfft_impl<A2AvectorWunitary<mf_Policies>, const A2AvectorWunitaryfftw<mf_Policies>, WunitaryFFTfieldPolicyBasic>::inversefft(to,*this,mode_postop);
}

//Use the relations between FFTs to obtain the FFT for a chosen quark momentum
//With G-parity BCs there are 2 disjoint sets of momenta hence there are 2 base FFTs
template< typename mf_Policies>
void A2AvectorWunitaryfftw<mf_Policies>::getTwistedFFT(const int p[3], A2AvectorWunitaryfftw<Policies> const *base_p, A2AvectorWunitaryfftw<Policies> const *base_m){
  Float time = -dclock();
  
  std::vector<int> shift(3);
  A2AvectorWunitaryfftw<mf_Policies> const* base = getBaseAndShift(&shift[0], p, base_p, base_m);
  if(base == NULL) ERR.General("A2AvectorWunitaryfftw","getTwistedFFT","Base pointer for twist momentum (%d,%d,%d) is NULL\n",p[0],p[1],p[2]);

  wl = base->wl;
  wh = base->wh;
  
  int nshift = 0;
  for(int i=0;i<3;i++) if(shift[i]) nshift++;

  if(nshift > 0){
    for(int i=0;i<this->getNmodes();i++)
      shiftPeriodicField( this->getMode(i), base->getMode(i), shift);
  }
  time += dclock();
  a2a_print_time("A2AvectorWunitaryfftw::getTwistedFFT","Twist",time);
}


template< typename mf_Policies>
void A2AvectorWunitaryfftw<mf_Policies>::shiftFieldsInPlace(const std::vector<int> &shift){
  Float time = -dclock();
  int nshift = 0;
  for(int i=0;i<3;i++) if(shift[i]) nshift++;
  if(nshift > 0){
    for(int i=0;i<this->getNmodes();i++)
      shiftPeriodicField( this->getMode(i), this->getMode(i), shift);
  }
  a2a_print_time("A2AvectorWunitaryfftw::shiftFieldsInPlace","Total",time + dclock());
}

//A version of the above that directly shifts the base Wfftw rather than outputting into a separate storage
//Returns the pointer to the Wfftw acted upon and the *shift required to restore the Wfftw to it's original form*
template< typename mf_Policies>
std::pair< A2AvectorWunitaryfftw<mf_Policies>*, std::vector<int> > A2AvectorWunitaryfftw<mf_Policies>::inPlaceTwistedFFT(const int p[3], A2AvectorWunitaryfftw<mf_Policies> *base_p, A2AvectorWunitaryfftw<mf_Policies> *base_m){
  Float time = -dclock();
  
  std::vector<int> shift(3);
  A2AvectorWunitaryfftw<mf_Policies>* base = const_cast<A2AvectorWunitaryfftw<mf_Policies>*>(getBaseAndShift(&shift[0], p, base_p, base_m));
  if(base == NULL) ERR.General("A2AvectorWunitaryfftw","getTwistedFFT","Base pointer for twist momentum (%d,%d,%d) is NULL\n",p[0],p[1],p[2]);

  base->shiftFieldsInPlace(shift);

  for(int i=0;i<3;i++) shift[i] = -shift[i];
  
  time += dclock();
  a2a_print_time("A2AvectorWunitaryfftw::inPlaceTwistedFFT","Twist",time);

  return std::pair< A2AvectorWunitaryfftw<mf_Policies>*, std::vector<int> >(base,shift);
}
