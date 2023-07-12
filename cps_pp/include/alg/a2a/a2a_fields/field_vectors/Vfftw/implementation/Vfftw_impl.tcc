template< typename mf_Policies>
double A2AvectorVfftw<mf_Policies>::Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params){
  return VW_Mbyte_size<A2AvectorVfftw<mf_Policies> >(_args,field_setup_params);
}

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
    fft_opt_mu_timings::reset();
    typedef typename FermionFieldType::InputParamType FieldParamType;
    FieldParamType field_setup = from.getFieldInputParams();
    FermionFieldType tmp(field_setup);

    Float action_output_mode_time = 0;
    Float action_input_mode_time = 0;
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
      action_output_mode_time += dclock() - dtime;

      dtime = dclock();      
#ifndef MEMTEST_MODE
      cps::fft_opt(to.getMode(mode), *init_gather_from, fft_dirs);
#endif
      fft_time += dclock() - dtime;

      dtime = dclock();
      FFTfieldPolicy::actionInputMode(from, mode); //free
      action_input_mode_time += dclock() - dtime;
    }
    if(!UniqueID()){ printf("Finishing V FFT\n"); fflush(stdout); }
    print_time("A2AvectorVfftw::fft","Preop",preop_time);
    print_time("A2AvectorVfftw::fft","FFT",fft_time);
    print_time("A2AvectorVfftw::fft","actionOutputMode",action_output_mode_time);
    print_time("A2AvectorVfftw::fft","actionInputMode",action_input_mode_time);
    fft_opt_mu_timings::print();
  }
};


//Set this object to be the fast Fourier transform of the input field
//Can optionally supply an object mode_preop that performs a transformation on each mode prior to the FFT
template< typename mf_Policies>
void A2AvectorVfftw<mf_Policies>::fft(const A2AvectorV<mf_Policies> &from, fieldOperation<FermionFieldType>* mode_preop){
  _V_fft_impl<A2AvectorVfftw<mf_Policies>, const A2AvectorV<mf_Policies>, VFFTfieldPolicyBasic>::fft(*this,from,mode_preop);
}


template<typename OutputType, typename InputType, typename FFTfieldPolicy>
struct _V_invfft_impl{
  typedef typename InputType::FermionFieldType FermionFieldType;

  static inline void inversefft(OutputType &to, InputType &from, fieldOperation<FermionFieldType>* mode_postop){
    if(!UniqueID()){ printf("Doing V inverse FFT\n"); fflush(stdout); }
    fft_opt_mu_timings::reset();
    typedef typename FermionFieldType::InputParamType FieldParamType;
    FieldParamType field_setup = from.getFieldInputParams();
    FermionFieldType tmp(field_setup);

    Float action_output_mode_time = 0;
    Float action_input_mode_time = 0;  
    Float postop_time = 0;
    Float fft_time = 0;

    const bool fft_dirs[4] = {true,true,true,false};
    for(int mode=0;mode<from.getNmodes();mode++){
      Float dtime = dclock();      
      FFTfieldPolicy::actionOutputMode(to, mode); //alloc
      action_output_mode_time += dclock() - dtime;

      FermionFieldType* out = mode_postop == NULL ? &to.getMode(mode) : &tmp;
    
      dtime = dclock();
#ifndef MEMTEST_MODE
      cps::fft_opt(*out, from.getMode(mode), fft_dirs, true);
#endif
      fft_time += dclock() - dtime;
      
      dtime = dclock();
      FFTfieldPolicy::actionInputMode(from, mode); //alloc
      action_input_mode_time += dclock() - dtime;

      if(mode_postop != NULL){
	dtime = dclock();
	(*mode_postop)(tmp,to.getMode(mode));
	postop_time += dclock()-dtime;
      }

    }
    if(!UniqueID()){ printf("Finishing V invert FFT\n"); fflush(stdout); }
    print_time("A2AvectorVfftw::inversefft","FFT",fft_time);
    print_time("A2AvectorVfftw::inversefft","Postop",postop_time);
    print_time("A2AvectorVfftw::inversefft","actionOutputMode",action_output_mode_time);
    print_time("A2AvectorVfftw::inversefft","actionInputMode",action_input_mode_time);
    fft_opt_mu_timings::print();
  }
};

template< typename mf_Policies>
void A2AvectorVfftw<mf_Policies>::inversefft(A2AvectorV<Policies> &to, fieldOperation<FermionFieldType>* mode_postop) const{
  _V_invfft_impl<A2AvectorV<Policies>, const A2AvectorVfftw<mf_Policies>, VFFTfieldPolicyBasic>::inversefft(to,*this,mode_postop);
}

template< typename mf_Policies>
void A2AvectorVfftw<mf_Policies>::getTwistedFFT(const int p[3], A2AvectorVfftw<Policies> const *base_p, A2AvectorVfftw<Policies> const *base_m){
  Float time = -dclock();
  
  std::vector<int> shift(3);
  A2AvectorVfftw<mf_Policies> const* base = getBaseAndShift(&shift[0], p, base_p, base_m);
  if(base == NULL) ERR.General("A2AvectorVfftw","getTwistedFFT","Base pointer for twist momentum (%d,%d,%d) is NULL\n",p[0],p[1],p[2]);
  
  v = base->v;
  
  int nshift = 0;
  for(int i=0;i<3;i++) if(shift[i]) nshift++;

  if(nshift > 0){
    for(int i=0;i<this->getNmodes();i++)
      shiftPeriodicField( this->getMode(i), base->getMode(i), shift);
  }
  time += dclock();
  print_time("A2AvectorVfftw::getTwistedFFT","Twist",time);
}

template< typename mf_Policies>
void A2AvectorVfftw<mf_Policies>::shiftFieldsInPlace(const std::vector<int> &shift){
  Float time = -dclock();
  int nshift = 0;
  for(int i=0;i<3;i++) if(shift[i]) nshift++;
  if(nshift > 0){
    for(int i=0;i<this->getNmodes();i++)
      shiftPeriodicField( this->getMode(i), this->getMode(i), shift);
  }
  print_time("A2AvectorVfftw::shiftFieldsInPlace","Total",time + dclock());
}

//A version of the above that directly shifts the base Wfftw rather than outputting into a separate storage
//Returns the pointer to the Wfftw acted upon and the *shift required to restore the Wfftw to it's original form*
template< typename mf_Policies>
std::pair< A2AvectorVfftw<mf_Policies>*, std::vector<int> > A2AvectorVfftw<mf_Policies>::inPlaceTwistedFFT(const int p[3], A2AvectorVfftw<mf_Policies> *base_p, A2AvectorVfftw<mf_Policies> *base_m){
  Float time = -dclock();
  
  std::vector<int> shift(3);
  A2AvectorVfftw<mf_Policies>* base = const_cast<A2AvectorVfftw<mf_Policies>*>(getBaseAndShift(&shift[0], p, base_p, base_m));
  if(base == NULL) ERR.General("A2AvectorWfftw","getTwistedFFT","Base pointer for twist momentum (%d,%d,%d) is NULL\n",p[0],p[1],p[2]);

  base->shiftFieldsInPlace(shift);

  for(int i=0;i<3;i++) shift[i] = -shift[i];
  
  time += dclock();
  print_time("A2AvectorVfftw::inPlaceTwistedFFT","Twist",time);

  return std::pair< A2AvectorVfftw<mf_Policies>*, std::vector<int> >(base,shift);
}
