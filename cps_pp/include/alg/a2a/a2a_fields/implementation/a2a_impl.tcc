//Implementations of methods in a2a.h

template<typename VWtype>
inline double VW_Mbyte_size(const A2AArg &_args, const typename VWtype::FieldInputParamType &field_setup_params){
  typedef typename VWtype::DilutionType DilutionType;
  typedef typename VWtype::FermionFieldType FermionFieldType;
  DilutionType dil(_args); const int sz = dil.getNmodes();
  double field_size = double(FermionFieldType::byte_size(field_setup_params))/(1024.*1024.);
  return sz * field_size;
}


template< typename mf_Policies>
double A2AvectorV<mf_Policies>::Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params){
  return VW_Mbyte_size<A2AvectorV<mf_Policies> >(_args,field_setup_params);
}
template< typename mf_Policies>
double A2AvectorVfftw<mf_Policies>::Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params){
  return VW_Mbyte_size<A2AvectorVfftw<mf_Policies> >(_args,field_setup_params);
}
template< typename mf_Policies>
double A2AvectorW<mf_Policies>::Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params){
  FullyPackedIndexDilution dil(_args);
  double ffield_size = double(FermionFieldType::byte_size(field_setup_params))/(1024.*1024.);
  double cfield_size = double(ComplexFieldType::byte_size(field_setup_params))/(1024.*1024.);
  return dil.getNl() * ffield_size + dil.getNhits() * cfield_size;
}
template< typename mf_Policies>
double A2AvectorWfftw<mf_Policies>::Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params){
  return VW_Mbyte_size<A2AvectorWfftw<mf_Policies> >(_args,field_setup_params);
}




//Set this object to be the fast Fourier transform of the input field
//Can optionally supply an object mode_preop that performs a transformation on each mode prior to the FFT
template< typename mf_Policies>
void A2AvectorVfftw<mf_Policies>::fft(const A2AvectorV<mf_Policies> &from, fieldOperation<FermionFieldType>* mode_preop){
  _V_fft_impl<A2AvectorVfftw<mf_Policies>, const A2AvectorV<mf_Policies>, VFFTfieldPolicyBasic>::fft(*this,from,mode_preop);
}


template< typename mf_Policies>
void A2AvectorVfftw<mf_Policies>::inversefft(A2AvectorV<Policies> &to, fieldOperation<FermionFieldType>* mode_postop) const{
  _V_invfft_impl<A2AvectorV<Policies>, const A2AvectorVfftw<mf_Policies>, VFFTfieldPolicyBasic>::inversefft(to,*this,mode_postop);
}

template< typename mf_Policies>
void A2AvectorW<mf_Policies>::initialize(const FieldInputParamType &field_setup_params){
  checkSIMDparams<FieldInputParamType>::check(field_setup_params);
  wl.resize(nl); this->allocInitializeLowModeFields(wl,field_setup_params);
  wh.resize(nhits); this->allocInitializeHighModeFields(wh,field_setup_params);
}

template< typename mf_Policies>
A2AvectorW<mf_Policies>::A2AvectorW(const A2AArg &_args): FullyPackedIndexDilution(_args), wh_rand_performed(false){ initialize(NullObject()); }

template< typename mf_Policies>
A2AvectorW<mf_Policies>::A2AvectorW(const A2AArg &_args, const FieldInputParamType &field_setup_params): FullyPackedIndexDilution(_args), wh_rand_performed(false){
  initialize(field_setup_params); }

template< typename mf_Policies>
A2AvectorW<mf_Policies>::A2AvectorW(const A2Aparams &_args): FullyPackedIndexDilution(_args), wh_rand_performed(false){ initialize(NullObject()); }

template< typename mf_Policies>
A2AvectorW<mf_Policies>::A2AvectorW(const A2Aparams &_args, const FieldInputParamType &field_setup_params): FullyPackedIndexDilution(_args), wh_rand_performed(false){ initialize(field_setup_params); }



template< typename mf_Policies>
void A2AvectorWfftw<mf_Policies>::initialize(const FieldInputParamType &field_setup_params){
  checkSIMDparams<FieldInputParamType>::check(field_setup_params);
  wl.resize(nl); this->allocInitializeLowModeFields(wl,field_setup_params);
  wh.resize(12*nhits); this->allocInitializeHighModeFields(wh,field_setup_params);
  for(int i=0;i<12;i++) CPSsetZero(zerosc[i]);
}

template< typename mf_Policies>
A2AvectorWfftw<mf_Policies>::A2AvectorWfftw(const A2AArg &_args): TimeFlavorPackedIndexDilution(_args)
#ifdef ZEROSC_MANAGED
    , zerosc(12)
#endif
{ initialize(NullObject()); }

template< typename mf_Policies>
A2AvectorWfftw<mf_Policies>::A2AvectorWfftw(const A2AArg &_args, const FieldInputParamType &field_setup_params): TimeFlavorPackedIndexDilution(_args)
#ifdef ZEROSC_MANAGED
    , zerosc(12)
#endif
{ initialize(field_setup_params); }

template< typename mf_Policies>
A2AvectorWfftw<mf_Policies>::A2AvectorWfftw(const A2Aparams &_args): TimeFlavorPackedIndexDilution(_args)
#ifdef ZEROSC_MANAGED
    , zerosc(12)
#endif
    { initialize(NullObject()); }


template< typename mf_Policies>
A2AvectorWfftw<mf_Policies>::A2AvectorWfftw(const A2Aparams &_args, const FieldInputParamType &field_setup_params): TimeFlavorPackedIndexDilution(_args)
#ifdef ZEROSC_MANAGED
    , zerosc(12)
#endif
  { initialize(field_setup_params); }


//Set this object to be the fast Fourier transform of the input field
//Can optionally supply an object mode_preop that performs a transformation on each mode prior to the FFT
template< typename mf_Policies>
void A2AvectorWfftw<mf_Policies>::fft(const A2AvectorW<mf_Policies> &from, fieldOperation<FermionFieldType>* mode_preop){
  _W_fft_impl<A2AvectorWfftw<mf_Policies>, const A2AvectorW<mf_Policies>, WFFTfieldPolicyBasic>::fft(*this,from,mode_preop);
}

template< typename mf_Policies>
void A2AvectorWfftw<mf_Policies>::inversefft(A2AvectorW<mf_Policies> &to, fieldOperation<FermionFieldType>* mode_postop) const{
  _W_invfft_impl<A2AvectorW<mf_Policies>, const A2AvectorWfftw<mf_Policies>, WFFTfieldPolicyBasic>::inversefft(to,*this,mode_postop);
}

template< typename mf_Policies>
void A2AvectorW<mf_Policies>::setWh(const std::vector<ScalarComplexFieldType> &to){
  assert(to.size() == nhits);
  for(int i=0;i<nhits;i++) wh[i]->importField(to[i]);
  wh_rand_performed = true;
}


//Get the diluted source with index id.
//We allow for time dilution into Lt/src_width blocks of size src_width in the time direction
//Alongside the spin/color/flavor index upon which to place the random numbers, the index dil_id also contains the time block index
template< typename mf_Policies>
template<typename TargetFermionFieldType>
void A2AvectorW<mf_Policies>::getDilutedSource(TargetFermionFieldType &into, const int dil_id) const{
  typedef FieldSiteType mf_Complex;
  typedef typename TargetFermionFieldType::FieldSiteType TargetComplex;
  const char* fname = "getDilutedSource(...)";
  int hit, tblock, spin_color, flavor;
  StandardIndexDilution stdidx(getArgs());  
  stdidx.indexUnmap(dil_id,hit,tblock,spin_color,flavor);
  
  //Dimensions of 4d (possibly SIMD vectorized [spatial only]) complex field
  const int src_layout[4] = { wh[hit]->nodeSites(0), wh[hit]->nodeSites(1), wh[hit]->nodeSites(2), wh[hit]->nodeSites(3) };
  
  assert(src_layout[3] == GJP.TnodeSites()); //check no vectorization in t
  
  assert(GJP.Tnodes()*GJP.TnodeSites() % args.src_width == 0); //assumed an even number of time blocks fit into lattice

  VRB.Result("A2AvectorW", fname, "Generating random wall source %d = (%d, %d, %d, %d).\n    ", dil_id, hit, tblock, flavor, spin_color);
  const int tblock_origt = tblock * args.src_width; //origin of t block in global coordinates
  const int tblock_lessthant = tblock_origt + args.src_width; //where does it end?

  int tblock_origt_lcl = tblock_origt - GJP.TnodeCoor()*GJP.TnodeSites(); //same as above in local coords
  int tblock_lessthant_lcl = tblock_lessthant - GJP.TnodeCoor()*GJP.TnodeSites();
  
  into.zero();

  if(tblock_lessthant_lcl <= 0 || tblock_origt_lcl >= GJP.TnodeSites()){ //none of source is on this node
    VRB.Result("A2AvectorW", fname, "Not on node\n    ");
    return;
  }

  //Some of the source is on this node
  if(tblock_origt_lcl < 0) tblock_origt_lcl = 0; //beginning of source is before origin
  if(tblock_lessthant_lcl > GJP.TnodeSites()) tblock_lessthant_lcl = GJP.TnodeSites(); //end is after local time size

  const int lcl_src_twidth = tblock_lessthant_lcl - tblock_origt_lcl;
  
  const int src_size = src_layout[0]*src_layout[1]*src_layout[2]*lcl_src_twidth;  //size of source 3D*width slice in units of complex numbers  
  CPSautoView(into_v,into,HostWrite);
  CPSautoView(wh_v,(*wh[hit]),HostRead);
#pragma omp parallel for
  for(int i=0;i<src_size;i++){
    int x[4];
    int rem = i;
    x[0] = rem % src_layout[0]; rem /= src_layout[0];
    x[1] = rem % src_layout[1]; rem /= src_layout[1];
    x[2] = rem % src_layout[2]; rem /= src_layout[2];
    x[3] = tblock_origt_lcl + rem;

    TargetComplex *into_site = (TargetComplex*)(into_v.site_ptr(x,flavor) + spin_color);
    mf_Complex const* from_site = (mf_Complex*)wh_v.site_ptr(x,flavor); //note same random numbers for each spin/color!
    *into_site = *from_site;
  }
}

//When gauge fixing prior to taking the FFT it is necessary to uncompact the wh field in the spin-color index, as these indices are acted upon by the gauge fixing
//(I suppose technically only the color indices need uncompacting; this might be considered as a future improvement)
template< typename mf_Policies>
void A2AvectorW<mf_Policies>::getSpinColorDilutedSource(FermionFieldType &into, const int hit, const int sc_id) const{
  const char* fname = "getSpinColorDilutedSource(...)";
  
  into.zero();
  CPSautoView(into_v,into,HostWrite);
  CPSautoView(wh_v,(*wh[hit]),HostRead);
#pragma omp parallel for
  for(int i=0;i<wh[hit]->nfsites();i++){ //same mapping, different site_size
    FieldSiteType &into_site = *(into_v.fsite_ptr(i) + sc_id);
    const FieldSiteType &from_site = *(wh_v.fsite_ptr(i));
    into_site = from_site;
  }
}


template<typename mf_Policies,typename ComplexClass>
struct _randomizeVWimpl{};

template<typename mf_Policies>
struct _randomizeVWimpl<mf_Policies,complex_double_or_float_mark>{
  static inline void randomizeVW(A2AvectorV<mf_Policies> &V, A2AvectorW<mf_Policies> &W){
    typedef typename mf_Policies::FermionFieldType FermionFieldType;
    typedef typename mf_Policies::ComplexFieldType ComplexFieldType;
  
    int nl = V.getNl();
    int nh = V.getNh(); //number of fully diluted high-mode indices
    int nhit = V.getNhits();
    assert(nl == W.getNl());
    assert(nh == W.getNh());
    assert(nhit == W.getNhits());
  

    std::vector<FermionFieldType> wl(nl);
    for(int i=0;i<nl;i++) wl[i].setUniformRandom();
  
    std::vector<FermionFieldType> vl(nl);
    for(int i=0;i<nl;i++) vl[i].setUniformRandom();
  
    std::vector<ComplexFieldType> wh(nhit);
    for(int i=0;i<nhit;i++) wh[i].setUniformRandom();
  
    std::vector<FermionFieldType> vh(nh);
    for(int i=0;i<nh;i++) vh[i].setUniformRandom();
    
    for(int i=0;i<nl;i++){
      V.importVl(vl[i],i);
      W.importWl(wl[i],i);
    }

    for(int i=0;i<nh;i++)
      V.importVh(vh[i],i);
  
    for(int i=0;i<nhit;i++)
      W.importWh(wh[i],i);
  }
};

//Ensure this generates randoms in the same order as the scalar version
template<typename mf_Policies>
struct _randomizeVWimpl<mf_Policies,grid_vector_complex_mark>{
  static inline void randomizeVW(A2AvectorV<mf_Policies> &V, A2AvectorW<mf_Policies> &W){
    typedef typename mf_Policies::FermionFieldType::FieldMappingPolicy::EquivalentScalarPolicy ScalarMappingPolicy;
  
    typedef CPSfermion4D<typename mf_Policies::ScalarComplexType, ScalarMappingPolicy, typename mf_Policies::AllocPolicy> ScalarFermionFieldType;
    typedef CPScomplex4D<typename mf_Policies::ScalarComplexType, ScalarMappingPolicy, typename mf_Policies::AllocPolicy> ScalarComplexFieldType;
  
    int nl = V.getNl();
    int nh = V.getNh(); //number of fully diluted high-mode indices
    int nhit = V.getNhits();
    assert(nl == W.getNl());
    assert(nh == W.getNh());
    assert(nhit == W.getNhits());

    ScalarFermionFieldType tmp;
    ScalarComplexFieldType tmp_cmplx;
  
    for(int i=0;i<nl;i++){
      tmp.setUniformRandom();
      W.getWl(i).importField(tmp);
    }
    for(int i=0;i<nl;i++){
      tmp.setUniformRandom();
      V.getVl(i).importField(tmp);
    }
    for(int i=0;i<nhit;i++){
      tmp_cmplx.setUniformRandom();
      W.getWh(i).importField(tmp_cmplx);
    }
    for(int i=0;i<nh;i++){
      tmp.setUniformRandom();
      V.getVh(i).importField(tmp);
    }
  }
};

template<typename mf_Policies>
void randomizeVW(A2AvectorV<mf_Policies> &V, A2AvectorW<mf_Policies> &W){
#ifndef MEMTEST_MODE
  return _randomizeVWimpl<mf_Policies,typename ComplexClassify<typename mf_Policies::ComplexType>::type>::randomizeVW(V,W);
#endif
}


template< typename FieldType>
FieldType const * getBaseAndShift(int shift[3], const int p[3], FieldType const *base_p, FieldType const *base_m){
  //With G-parity base_p has momentum +1 in each G-parity direction, base_m has momentum -1 in each G-parity direction.
  //Non-Gparity directions are assumed to have momentum 0

  //Units of momentum are 2pi/L for periodic BCs, pi/L for antiperiodic and pi/2L for Gparity
  FieldType const * out = GJP.Gparity() ? NULL : base_p;
  for(int d=0;d<3;d++){
    if(GJP.Bc(d) == BND_CND_GPARITY){
      //Type 1 : f_{p=4b+1}(n) = f_+1(n+b)     // p \in {.. -7 , -3, 1, 5, 9 ..}
      //Type 2 : f_{p=4b-1}(n) = f_-1(n+b)     // p \n  {.. -5, -1, 3, 7 , 11 ..}
      if( (p[d]-1) % 4 == 0 ){
	//Type 1
	int b = (p[d]-1)/4;
	shift[d] = -b;  //shift f_+1 backwards by b
	if(out == NULL) out = base_p;
	else if(out != base_p) ERR.General("","getBaseAndShift","Momentum (%d,%d,%d) appears to be invalid because momenta in different G-parity directions do not reside in the same set\n",p[0],p[1],p[2]);
	
      }else if( (p[d]+1) % 4 == 0 ){
	//Type 2
	int b = (p[d]+1)/4;
	shift[d] = -b;  //shift f_-1 backwards by b
	if(out == NULL) out = base_m;
	else if(out != base_m) ERR.General("","getBaseAndShift","Momentum (%d,%d,%d) appears to be invalid because momenta in different G-parity directions do not reside in the same set\n",p[0],p[1],p[2]);
	
      }else ERR.General("","getBaseAndShift","Momentum (%d,%d,%d) appears to be invalid because one or more components in G-parity directions are not allowed\n",p[0],p[1],p[2]);
    }else{
      //f_b(n) = f_0(n+b)
      //Let the other directions decide on which base to use if some of them are G-parity dirs ; otherwise the pointer defaults to base_p above
      shift[d] = -p[d];
    }
  }
  if(!UniqueID()) printf("getBaseAndShift for p=(%d,%d,%d) determined shift=(%d,%d,%d) from ptr %c\n",p[0],p[1],p[2],shift[0],shift[1],shift[2],out == base_p ? 'p' : 'm');
  assert(out != NULL);
  
  return out;
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
