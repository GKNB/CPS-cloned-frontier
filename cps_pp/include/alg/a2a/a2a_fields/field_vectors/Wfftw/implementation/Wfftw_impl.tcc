template< typename mf_Policies>
double A2AvectorWfftw<mf_Policies>::Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params){
  return VW_Mbyte_size<A2AvectorWfftw<mf_Policies> >(_args,field_setup_params);
}

template< typename mf_Policies>
void A2AvectorWfftw<mf_Policies>::initialize(const FieldInputParamType &field_setup_params){
  checkSIMDparams<FieldInputParamType>::check(field_setup_params);
  wl.resize(nl); this->allocInitializeLowModeFields(wl,field_setup_params);
  wh.resize(12*nhits); this->allocInitializeHighModeFields(wh,field_setup_params);
}

template< typename mf_Policies>
A2AvectorWfftw<mf_Policies>::A2AvectorWfftw(const A2AArg &_args): TimeFlavorPackedIndexDilution(_args){ initialize(NullObject()); }

template< typename mf_Policies>
A2AvectorWfftw<mf_Policies>::A2AvectorWfftw(const A2AArg &_args, const FieldInputParamType &field_setup_params): TimeFlavorPackedIndexDilution(_args){ initialize(field_setup_params); }

template< typename mf_Policies>
A2AvectorWfftw<mf_Policies>::A2AvectorWfftw(const A2Aparams &_args): TimeFlavorPackedIndexDilution(_args){ initialize(NullObject()); }


template< typename mf_Policies>
A2AvectorWfftw<mf_Policies>::A2AvectorWfftw(const A2Aparams &_args, const FieldInputParamType &field_setup_params): TimeFlavorPackedIndexDilution(_args){ initialize(field_setup_params); }

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
	action_output_mode_time += dclock() - dtime;

	dtime = dclock();
#ifndef MEMTEST_MODE
	cps::fft_opt(to.getWh(hit,sc), *init_gather_from, fft_dirs);
#endif
	fft_time += dclock()-dtime;
      }

      Float dtime = dclock();
      FFTfieldPolicy::actionInputHighMode(from, hit); //free
      action_input_mode_time += dclock() - dtime;
    }
    if(!UniqueID()){ printf("Finishing W FFT\n"); fflush(stdout); }
    print_time("A2AvectorWfftw::fft","Preop",preop_time);
    print_time("A2AvectorWfftw::fft","FFT",fft_time);
    print_time("A2AvectorWfftw::fft","actionOutputMode",action_output_mode_time);
    print_time("A2AvectorWfftw::fft","actionInputMode",action_input_mode_time);
  }
};

//Set this object to be the fast Fourier transform of the input field
//Can optionally supply an object mode_preop that performs a transformation on each mode prior to the FFT
template< typename mf_Policies>
void A2AvectorWfftw<mf_Policies>::fft(const A2AvectorW<mf_Policies> &from, fieldOperation<FermionFieldType>* mode_preop){
  _W_fft_impl<A2AvectorWfftw<mf_Policies>, const A2AvectorW<mf_Policies>, WFFTfieldPolicyBasic>::fft(*this,from,mode_preop);
}


template<typename OutputType, typename InputType, typename FFTfieldPolicy>
struct _W_invfft_impl{
  typedef typename InputType::FermionFieldType FermionFieldType;

  static inline void inversefft(OutputType &to, InputType &from, fieldOperation<FermionFieldType>* mode_postop){
    if(!UniqueID()){ printf("Doing W inverse FFT\n"); fflush(stdout); }
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
    //Do wh. First we need to uncompact the spin/color index as this is acted upon by the operator
    for(int hit=0;hit<from.getNhits();hit++){
      Float dtime = dclock();
      FFTfieldPolicy::actionOutputHighMode(to, hit); //alloc
      action_output_mode_time += dclock() - dtime;

      typename InputType::ComplexFieldType & to_hit = to.getWh(hit);
    
      const int sc = 0;
      FermionFieldType * compress = mode_postop == NULL ? &tmp2 : &tmp;
      
      dtime = dclock();
#ifndef MEMTEST_MODE
      cps::fft_opt(tmp2, from.getWh(hit,sc), fft_dirs, true);
#endif
      fft_time += dclock()-dtime;

      if(mode_postop != NULL){
	dtime = dclock();
	(*mode_postop)(tmp2,tmp);
	postop_time += dclock()-dtime;
      }
      //Should give a multiple of the 12-component unit vector with 1 on index sc
      {
	CPSautoView(to_hit_v,to_hit,HostWrite);
	CPSautoView(compress_v,(*compress),HostRead);

#pragma omp parallel for
	for(int i=0;i<to_hit.nfsites();i++)
	  *(to_hit_v.fsite_ptr(i)) = *(compress_v.fsite_ptr(i) + sc);
      }

      dtime = dclock();
      for(int ssc=0;ssc<12;ssc++) FFTfieldPolicy::actionInputHighMode(from, ssc + 12*hit); //free for all sc
      action_input_mode_time += dclock() - dtime;      
    }
    if(!UniqueID()){ printf("Finishing W inverse FFT\n"); fflush(stdout); }
    print_time("A2AvectorWfftw::fftinverse","FFT",fft_time);
    print_time("A2AvectorWfftw::fftinverse","Postop",postop_time);
    print_time("A2AvectorWfftw::fftinverse","actionOutputMode",action_output_mode_time);
    print_time("A2AvectorWfftw::fftinverse","actionInputMode",action_input_mode_time);
  }
};

template< typename mf_Policies>
void A2AvectorWfftw<mf_Policies>::inversefft(A2AvectorW<mf_Policies> &to, fieldOperation<FermionFieldType>* mode_postop) const{
  _W_invfft_impl<A2AvectorW<mf_Policies>, const A2AvectorWfftw<mf_Policies>, WFFTfieldPolicyBasic>::inversefft(to,*this,mode_postop);
}

//Use the relations between FFTs to obtain the FFT for a chosen quark momentum
//With G-parity BCs there are 2 disjoint sets of momenta hence there are 2 base FFTs
template< typename mf_Policies>
void A2AvectorWfftw<mf_Policies>::getTwistedFFT(const int p[3], A2AvectorWfftw<Policies> const *base_p, A2AvectorWfftw<Policies> const *base_m){
  Float time = -dclock();
  
  std::vector<int> shift(3);
  A2AvectorWfftw<mf_Policies> const* base = getBaseAndShift(&shift[0], p, base_p, base_m);
  if(base == NULL) ERR.General("A2AvectorWfftw","getTwistedFFT","Base pointer for twist momentum (%d,%d,%d) is NULL\n",p[0],p[1],p[2]);

  wl = base->wl;
  wh = base->wh;
  
  int nshift = 0;
  for(int i=0;i<3;i++) if(shift[i]) nshift++;

  if(nshift > 0){
    for(int i=0;i<this->getNmodes();i++)
      shiftPeriodicField( this->getMode(i), base->getMode(i), shift);
  }
  time += dclock();
  print_time("A2AvectorWfftw::getTwistedFFT","Twist",time);
}


template< typename mf_Policies>
void A2AvectorWfftw<mf_Policies>::shiftFieldsInPlace(const std::vector<int> &shift){
  Float time = -dclock();
  int nshift = 0;
  for(int i=0;i<3;i++) if(shift[i]) nshift++;
  if(nshift > 0){
    for(int i=0;i<this->getNmodes();i++)
      shiftPeriodicField( this->getMode(i), this->getMode(i), shift);
  }
  print_time("A2AvectorWfftw::shiftFieldsInPlace","Total",time + dclock());
}

//A version of the above that directly shifts the base Wfftw rather than outputting into a separate storage
//Returns the pointer to the Wfftw acted upon and the *shift required to restore the Wfftw to it's original form*
template< typename mf_Policies>
std::pair< A2AvectorWfftw<mf_Policies>*, std::vector<int> > A2AvectorWfftw<mf_Policies>::inPlaceTwistedFFT(const int p[3], A2AvectorWfftw<mf_Policies> *base_p, A2AvectorWfftw<mf_Policies> *base_m){
  Float time = -dclock();
  
  std::vector<int> shift(3);
  A2AvectorWfftw<mf_Policies>* base = const_cast<A2AvectorWfftw<mf_Policies>*>(getBaseAndShift(&shift[0], p, base_p, base_m));
  if(base == NULL) ERR.General("A2AvectorWfftw","getTwistedFFT","Base pointer for twist momentum (%d,%d,%d) is NULL\n",p[0],p[1],p[2]);

  base->shiftFieldsInPlace(shift);

  for(int i=0;i<3;i++) shift[i] = -shift[i];
  
  time += dclock();
  print_time("A2AvectorWfftw::inPlaceTwistedFFT","Twist",time);

  return std::pair< A2AvectorWfftw<mf_Policies>*, std::vector<int> >(base,shift);
}
