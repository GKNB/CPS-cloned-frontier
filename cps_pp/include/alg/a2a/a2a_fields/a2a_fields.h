#ifndef CK_A2A
#define CK_A2A

#include<util/lattice.h>
#include<util/time_cps.h>

#ifdef USE_GRID
#include<util/lattice/fgrid.h>
#endif

#include<alg/ktopipi_jobparams.h>
#include<alg/a2a/base/a2a_dilutions.h>
#include<alg/a2a/utils.h>
#include<alg/a2a/lattice.h>

#include "a2a_policies.h"
#include "a2a_fft.h"

CPS_START_NAMESPACE

//If using SIMD, we don't want to vectorize across the time direction
template<typename FieldInputParamType>
struct checkSIMDparams{
  inline static void check(const FieldInputParamType &p){}
};
#ifdef USE_GRID
template<int Dimension>
struct checkSIMDparams<SIMDdims<Dimension> >{
  inline static void check(const SIMDdims<Dimension> &p){
    assert(p[3] == 1);
  }
};
#endif

template< typename mf_Policies>
class A2AvectorVfftw;

template< typename mf_Policies>
class A2AvectorV: public StandardIndexDilution, public mf_Policies::A2AvectorVpolicies{
public:
  typedef mf_Policies Policies;
  typedef typename Policies::FermionFieldType FermionFieldType;
  typedef typename FermionFieldType::FieldSiteType FieldSiteType;
  typedef typename FermionFieldType::InputParamType FieldInputParamType;
private:
  ManagedVector<ManagedPtrWrapper<FermionFieldType> > v;

public:
  typedef StandardIndexDilution DilutionType;

  A2AvectorV(const A2AArg &_args): StandardIndexDilution(_args){    
    v.resize(nv);
    //When computing V and W we can re-use previous V solutions as guesses. Set default to zero here so we have zero guess when no 
    //previously computed solutions
    this->allocInitializeFields(v,NullObject());
  }
  
  A2AvectorV(const A2AArg &_args, const FieldInputParamType &field_setup_params): StandardIndexDilution(_args){
    checkSIMDparams<FieldInputParamType>::check(field_setup_params);
    v.resize(nv);
    this->allocInitializeFields(v,field_setup_params);
  }
  
  static double Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params);

  inline FieldInputParamType getFieldInputParams() const{
    if(v.size() == 0) ERR.General("A2AvectorV","getFieldInputParams","Vector size is zero\n");
    if(!v[0].assigned()) ERR.General("A2AvectorV","getFieldInputParams","Zeroth field is unassigned\n");
    return v[0]->getDimPolParams();
  }    
  
  accelerator_inline const FermionFieldType & getMode(const int i) const{ return *v[i]; }
  accelerator_inline FermionFieldType & getMode(const int i){ return *v[i]; }  

  //Get a mode from the low mode part
  accelerator_inline FermionFieldType & getVl(const int il){ return *v[il]; }
  accelerator_inline const FermionFieldType & getVl(const int il) const{ return *v[il]; }

  //Get a mode from the high-mode part
  accelerator_inline FermionFieldType & getVh(const int ih){ return *v[nl+ih]; }
  accelerator_inline const FermionFieldType & getVh(const int ih) const{ return *v[nl+ih]; }

  //Get a particular site/spin/color element of a given mode 
  accelerator_inline const FieldSiteType & elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
    int x4d = v[mode]->threeToFour(x3d,t);
    return  *(v[mode]->site_ptr(x4d,flavor) + spin_color);
  }
  //Get a particular site/spin/color element of a given *native* (packed) mode. For V this does the same as the above
  accelerator_inline const FieldSiteType & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
    return *(v[i]->site_ptr(site,flavor)+spin_color);
  }

  void importVl(const FermionFieldType &vv, const int il){
    *v[il] = vv;
  }
  void importVh(const FermionFieldType &vv, const int ih){
    *v[nl+ih] = vv;
  }

  //Set each float to a uniform random number in the specified range.
  //WARNING: Uses only the current RNG in LRG, and does not change this based on site. This is therefore only useful for testing*
  void testRandom(const Float &hi = 0.5, const Float &lo = -0.5){
    for(int i=0;i<nv;i++) v[i]->testRandom(hi,lo);
  }

  void writeParallel(const std::string &file_stub, FP_FORMAT fileformat = FP_AUTOMATIC, CPSfield_checksumType cksumtype = checksumCRC32) const; //node id will be appended
  void readParallel(const std::string &file_stub);

  //Read/write to binary files per node with separate metadata files. User provides path which is created internally
  void writeParallelSeparateMetadata(const std::string &path, FP_FORMAT fileformat = FP_AUTOMATIC) const;
  void readParallelSeparateMetadata(const std::string &path);

  inline void free_mem(){ v.free(); }
};


template< typename mf_Policies>
class A2AvectorVfftw: public StandardIndexDilution, public mf_Policies::A2AvectorVfftwPolicies{  
public:
  typedef mf_Policies Policies;
  typedef typename Policies::FermionFieldType FermionFieldType;
  typedef typename FermionFieldType::FieldSiteType FieldSiteType;
  typedef typename FermionFieldType::InputParamType FieldInputParamType;

  #define VFFTW_ENABLE_IF_MANUAL_ALLOC(P) typename my_enable_if<  _equal<typename P::A2AvectorVfftwPolicies::FieldAllocStrategy,ManualAllocStrategy>::value , void>::type
private:
  ManagedVector<ManagedPtrWrapper<FermionFieldType> > v;

public:
  typedef StandardIndexDilution DilutionType;

  A2AvectorVfftw(const A2AArg &_args): StandardIndexDilution(_args){
    v.resize(nv);
    this->allocInitializeFields(v,NullObject());
  }
  A2AvectorVfftw(const A2AArg &_args, const FieldInputParamType &field_setup_params): StandardIndexDilution(_args){
    checkSIMDparams<FieldInputParamType>::check(field_setup_params);
    v.resize(nv);
    this->allocInitializeFields(v,field_setup_params);
  }
  
  static double Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params);

  inline FieldInputParamType getFieldInputParams() const{
    if(v.size() == 0) ERR.General("A2AvectorVfftw","getFieldInputParams","Vector size is zero\n");
    if(!v[0].assigned()) ERR.General("A2AvectorVfftw","getFieldInputParams","Zeroth field is unassigned\n");
    return v[0]->getDimPolParams();
  }   
  
  accelerator_inline const FermionFieldType & getMode(const int i) const{ return *v[i]; }
  accelerator_inline const FermionFieldType & getMode(const int i, const modeIndexSet &i_high_unmapped) const{ return getMode(i); }

  accelerator_inline FermionFieldType & getMode(const int i){ return *v[i]; }
  
  //Set this object to be the threaded fast Fourier transform of the input field
  //Can optionally supply an object that performs a transformation on each mode prior to the FFT. 
  //We can use this to avoid intermediate storage for the gauge fixing and momentum phase application steps
  void fft(const A2AvectorV<Policies> &from, fieldOperation<FermionFieldType>* mode_preop = NULL);
  
  void inversefft(A2AvectorV<Policies> &to, fieldOperation<FermionFieldType>* mode_postop = NULL) const;
  
  //For each mode, gauge fix, apply the momentum factor, then perform the FFT and store the result in this object
  void gaugeFixTwistFFT(const A2AvectorV<Policies> &from, const int _p[3], Lattice &_lat){
    gaugeFixAndTwist<FermionFieldType> op(_p,_lat); fft(from, &op);
  }

  //Unapply the phase and gauge fixing to give back a V vector
  void unapplyGaugeFixTwistFFT(A2AvectorV<Policies> &to, const int _p[3], Lattice &_lat) const{
    reverseGaugeFixAndTwist<FermionFieldType> op(_p,_lat); inversefft(to, &op);
  }
  
  //Use the relations between FFTs to obtain the FFT for a chosen quark momentum
  //With G-parity BCs there are 2 disjoint sets of momenta hence there are 2 base FFTs
  void getTwistedFFT(const int p[3], A2AvectorVfftw<Policies> const *base_p, A2AvectorVfftw<Policies> const *base_m = NULL);

  void shiftFieldsInPlace(const std::vector<int> &shift);

  //A version of the above that directly shifts the base Wfftw rather than outputting into a separate storage
  //Returns the pointer to the Wfftw acted upon and the *shift required to restore the Wfftw to it's original form* (use shiftFieldsInPlace to restore)
  
  static std::pair< A2AvectorVfftw<mf_Policies>*, std::vector<int> > inPlaceTwistedFFT(const int p[3], A2AvectorVfftw<mf_Policies> *base_p, A2AvectorVfftw<mf_Policies> *base_m = NULL);
  
  accelerator_inline const FieldSiteType & elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
    int site = v[mode]->threeToFour(x3d,t);
    return *(v[mode]->site_ptr(site,flavor) + spin_color);
  }
  //Get a particular site/spin/color element of a given 'native' (packed) mode. For V this does the same thing as the above
  accelerator_inline const FieldSiteType & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
    return *(v[i]->site_ptr(site,flavor)+spin_color);
  }

  //i_high_unmapped is the index i unmapped to its high mode sub-indices (if it is a high mode of course!)
  inline SCFvectorPtr<FieldSiteType> getFlavorDilutedVect(const int i, const modeIndexSet &i_high_unmapped, const int p3d, const int t) const{
    const FermionFieldType &field = getMode(i);
    const int x4d = field.threeToFour(p3d,t);
    FieldSiteType const *f0 = field.site_ptr(x4d,0);
    return SCFvectorPtr<FieldSiteType>(f0,f0+field.flav_offset());
  }
  //Return the pointer stride for between 3d coordinates for a given mode index and flavor. Relies on the dimension policy implementing dimpol_site_stride_3d
  inline int siteStride3D(const int i, const modeIndexSet &i_high_unmapped, const int f) const{
    const FermionFieldType &field = getMode(i);
    return field.dimpol_site_stride_3d()*field.siteSize();
  }
  
  //Replace this vector with the average of this another vector, 'with'
  void average(const A2AvectorVfftw<Policies> &with, const bool &parallel = true){
    if( !paramsEqual(with) ) ERR.General("A2AvectorVfftw","average","Second field must share the same underlying parameters\n");
    for(int i=0;i<nv;i++) v[i]->average(with.v[i]);
  }
  //Set each float to a uniform random number in the specified range.
  //WARNING: Uses only the current RNG in LRG, and does not change this based on site. This is therefore only useful for testing*
  void testRandom(const Float &hi = 0.5, const Float &lo = -0.5){
    for(int i=0;i<nv;i++) v[i]->testRandom(hi,lo);
  }

  template<typename extPolicies>
  void importFields(const A2AvectorVfftw<extPolicies> &r){
    if( !paramsEqual(r) ) ERR.General("A2AvectorVfftw","importFields","External field-vector must share the same underlying parameters\n");
    for(int i=0;i<nv;i++) v[i]->importField(r.getMode(i));
  }  

  inline void free_mem(){ v.free(); }
};


template< typename mf_Policies>
class A2AvectorW: public FullyPackedIndexDilution, public mf_Policies::A2AvectorWpolicies{
public:
  typedef mf_Policies Policies;
  typedef typename Policies::FermionFieldType FermionFieldType;
  typedef typename Policies::ComplexFieldType ComplexFieldType;

  typedef typename Policies::ScalarComplexType ScalarComplexType;
  typedef typename Policies::ScalarComplexFieldType ScalarComplexFieldType;

  typedef typename my_enable_if< _equal<typename FermionFieldType::FieldSiteType, typename ComplexFieldType::FieldSiteType>::value,  typename FermionFieldType::FieldSiteType>::type FieldSiteType;
  typedef typename my_enable_if< _equal<typename FermionFieldType::InputParamType, typename ComplexFieldType::InputParamType>::value,  typename FermionFieldType::InputParamType>::type FieldInputParamType;
private:
  ManagedVector<ManagedPtrWrapper<FermionFieldType> > wl; //The low mode part of the W field, comprised of nl fermion fields
  ManagedVector<ManagedPtrWrapper<ComplexFieldType> > wh; //The high mode random part of the W field, comprised of nhits complex scalar fields. Note: the dilution is performed later

  bool wh_rand_performed; //store if the wh random numbers have been set
public:
  typedef FullyPackedIndexDilution DilutionType;

  A2AvectorW(const A2AArg &_args): FullyPackedIndexDilution(_args), wh_rand_performed(false){
    wl.resize(nl); this->allocInitializeLowModeFields(wl,NullObject());
    wh.resize(nhits); this->allocInitializeHighModeFields(wh,NullObject());
  }
  A2AvectorW(const A2AArg &_args, const FieldInputParamType &field_setup_params): FullyPackedIndexDilution(_args), wh_rand_performed(false){
    checkSIMDparams<FieldInputParamType>::check(field_setup_params);
    wl.resize(nl); this->allocInitializeLowModeFields(wl,field_setup_params);
    wh.resize(nhits); this->allocInitializeHighModeFields(wh,field_setup_params);
  }

  //Generate the wh field. We store in a compact notation that knows nothing about any dilution we apply when generating V from this
  //For reproducibility we want to generate the wh field in the same order that Daiqian did originally. Here nhit random numbers are generated for each site/flavor
  //Note - this does not have to be called manually; it will be called by computeVWhigh if not previously called
  void setWhRandom();
  
  //Manually set the Wh random fields. Expects a vector of size nhits
  void setWhRandom(const std::vector<ScalarComplexFieldType> &to);

  static double Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params);

  inline FieldInputParamType getFieldInputParams() const{
    if(wl.size() == 0 && wh.size() == 0) ERR.General("A2AvectorW","getFieldInputParams","Wl and Wh sizes are both zero\n");
    if(wl.size() != 0 && wl[0].assigned()) return wl[0]->getDimPolParams();
    if(wh.size() != 0 && wh[0].assigned()) return wh[0]->getDimPolParams();
    
    ERR.General("A2AvectorW","getFieldInputParams","Neither of the zeroth fields are assigned\n");
  }   
  
  accelerator_inline const FermionFieldType & getWl(const int i) const{ return *wl[i]; }
  accelerator_inline const ComplexFieldType & getWh(const int hit) const{ return *wh[hit]; }

  accelerator_inline FermionFieldType & getWl(const int i){ return *wl[i]; }
  accelerator_inline ComplexFieldType & getWh(const int hit){ return *wh[hit]; }
  
  void importWl(const FermionFieldType &wlin, const int i){
    *wl[i] = wlin;
  }
  void importWh(const ComplexFieldType &whin, const int hit){
    *wh[hit] = whin;
  }

  //Get the diluted source with StandardIndex high-mode index dil_id.
  //We use the same set of random numbers for each spin and dilution as we do not need to rely on stochastic cancellation to separate them
  //For legacy reasons we use different random numbers for the two G-parity flavors, although this is not strictly necessary
  //Here dil_id is the combined spin-color/flavor/hit/tblock index
  template<typename TargetFermionFieldType>
  void getDilutedSource(TargetFermionFieldType &into, const int dil_id) const;

  //When gauge fixing prior to taking the FFT it is necessary to uncompact the wh field in the spin-color index, as these indices are acted upon by the gauge fixing
  //(I suppose technically only the color indices need uncompacting; this might be considered as a future improvement)
  void getSpinColorDilutedSource(FermionFieldType &into, const int hit, const int sc_id) const;

  //The spincolor, flavor and timeslice dilutions are packed so we must treat them differently
  //Mode is a full 'StandardIndex', (unpacked mode index)
  accelerator_inline const FieldSiteType & elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
    static FieldSiteType zero(0.0);
    if(mode < nl){
      int site = getWl(mode).threeToFour(x3d,t);
      return *(getWl(mode).site_ptr(site,flavor) + spin_color);
    }else{
      int mode_hit, mode_tblock, mode_spin_color,mode_flavor;
      const StandardIndexDilution &dilfull = static_cast<StandardIndexDilution const&>(*this);
      dilfull.indexUnmap(mode-nl,mode_hit,mode_tblock,mode_spin_color,mode_flavor);
      //flavor and time block indices match those of the mode, the result is zero
      int tblock = (t+GJP.TnodeSites()*GJP.TnodeCoor())/args.src_width;
      if(spin_color != mode_spin_color || flavor != mode_flavor || tblock != mode_tblock) return zero;
      int site = getWh(mode_hit).threeToFour(x3d,t);
      return *(getWh(mode_hit).site_ptr(site,flavor)); //we use different random fields for each time and flavor, although we didn't have to
    }
  }
  //Get a particular site/spin/color element of a given *native* (packed) mode 
  accelerator_inline const FieldSiteType & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
    return i < nl ? 
      *(wl[i]->site_ptr(site,flavor)+spin_color) :
      *(wh[i-nl]->site_ptr(site,flavor)); //we use different random fields for each time and flavor, although we didn't have to
  }

  //Set each float to a uniform random number in the specified range.
  //WARNING: Uses only the current RNG in LRG, and does not change this based on site. This is therefore only useful for testing*
  void testRandom(const Float &hi = 0.5, const Float &lo = -0.5){
    for(int i=0;i<nl;i++) wl[i]->testRandom(hi,lo);
    for(int i=0;i<nhits;i++) wh[i]->testRandom(hi,lo);
  }

  void writeParallel(const std::string &file_stub, FP_FORMAT fileformat = FP_AUTOMATIC, CPSfield_checksumType cksumtype = checksumCRC32) const; //node id will be appended
  void readParallel(const std::string &file_stub);

  //Write V/W fields to a format with metadata and binary data separate. User provides a unique directory path. Directory is created if doesn't already exist
  void writeParallelSeparateMetadata(const std::string &path, FP_FORMAT fileformat = FP_AUTOMATIC) const;
  void readParallelSeparateMetadata(const std::string &path);

  inline void free_mem(){
    wl.free();    wh.free();
  }
};


template< typename mf_Policies>
class A2AvectorWfftw: public TimeFlavorPackedIndexDilution, public mf_Policies::A2AvectorWfftwPolicies{
public:
  typedef mf_Policies Policies;
  typedef typename Policies::FermionFieldType FermionFieldType;
  typedef typename Policies::ComplexFieldType ComplexFieldType;
  typedef typename FermionFieldType::FieldSiteType FieldSiteType;
  typedef typename my_enable_if< _equal<typename FermionFieldType::InputParamType, typename ComplexFieldType::InputParamType>::value,  typename FermionFieldType::InputParamType>::type FieldInputParamType;

#define WFFTW_ENABLE_IF_MANUAL_ALLOC(P) typename my_enable_if<  _equal<typename P::A2AvectorWfftwPolicies::FieldAllocStrategy,ManualAllocStrategy>::value , int>::type
private:

  ManagedVector<ManagedPtrWrapper<FermionFieldType> > wl;
  ManagedVector<ManagedPtrWrapper<FermionFieldType> > wh; //these have been diluted in spin/color but not the other indices, hence there are nhit * 12 fields here (spin/color index changes fastest in mapping)

  //#define ZEROSC_MANAGED
  
#ifdef ZEROSC_MANAGED
  ManagedVector<FieldSiteType> zerosc;
#else
  FieldSiteType zerosc[12];
#endif
public:
  typedef TimeFlavorPackedIndexDilution DilutionType;

 A2AvectorWfftw(const A2AArg &_args): TimeFlavorPackedIndexDilution(_args)
#ifdef ZEROSC_MANAGED
    , zerosc(12)
#endif
    {
    wl.resize(nl); this->allocInitializeLowModeFields(wl,NullObject());
    wh.resize(12*nhits); this->allocInitializeHighModeFields(wh,NullObject());
    for(int i=0;i<12;i++) CPSsetZero(zerosc[i]);
  }
 A2AvectorWfftw(const A2AArg &_args, const FieldInputParamType &field_setup_params): TimeFlavorPackedIndexDilution(_args)
#ifdef ZEROSC_MANAGED
    , zerosc(12)
#endif
  {
    checkSIMDparams<FieldInputParamType>::check(field_setup_params);
    wl.resize(nl); this->allocInitializeLowModeFields(wl,field_setup_params);
    wh.resize(12*nhits); this->allocInitializeHighModeFields(wh,field_setup_params);
    for(int i=0;i<12;i++) CPSsetZero(zerosc[i]);
  }

  static double Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params);

  inline FieldInputParamType getFieldInputParams() const{
    if(wl.size() == 0 && wh.size() == 0) ERR.General("A2AvectorWfft","getFieldInputParams","Wl and Wh sizes are both zero\n");
    if(wl.size() != 0 && wl[0].assigned()) return wl[0]->getDimPolParams();
    if(wh.size() != 0 && wh[0].assigned()) return wh[0]->getDimPolParams();
    
    ERR.General("A2AvectorWfftw","getFieldInputParams","Neither of the zeroth fields are assigned\n");
  }   
  
  accelerator_inline const FermionFieldType & getWl(const int i) const{ return *wl[i]; }
  accelerator_inline const FermionFieldType & getWh(const int hit, const int spin_color) const{ return *wh[spin_color + 12*hit]; }

  accelerator_inline const FermionFieldType & getMode(const int i) const{ return i < nl ? *wl[i] : *wh[i-nl]; }

  accelerator_inline FermionFieldType & getWl(const int i){ return *wl[i]; }
  accelerator_inline FermionFieldType & getWh(const int hit, const int spin_color){ return *wh[spin_color + 12*hit]; }

  accelerator_inline FermionFieldType & getMode(const int i){ return i < nl ? *wl[i] : *wh[i-nl]; }

  //This version allows for the possibility of a different high mode mapping for the index i by passing the unmapped indices: for i>=nl the modeIndexSet is used to obtain the appropriate mode 
  inline const FermionFieldType & getMode(const int i, const modeIndexSet &i_high_unmapped) const{ return i >= nl ? getWh(i_high_unmapped.hit, i_high_unmapped.spin_color): getWl(i); }
  
  //Set this object to be the threaded fast Fourier transform of the input field
  //Can optionally supply an object that performs a transformation on each mode prior to the FFT. 
  //We can use this to avoid intermediate storage for the gauge fixing and momentum phase application steps
  void fft(const A2AvectorW<Policies> &from, fieldOperation<FermionFieldType>* mode_preop = NULL);
  
  void inversefft(A2AvectorW<Policies> &to, fieldOperation<FermionFieldType>* mode_postop = NULL) const;
  
  //For each mode, gauge fix, apply the momentum factor, then perform the FFT and store the result in this object
  void gaugeFixTwistFFT(const A2AvectorW<Policies> &from, const int _p[3], Lattice &_lat){
    gaugeFixAndTwist<FermionFieldType> op(_p,_lat); fft(from, &op);
  }
  
  //Unapply the phase and gauge fixing to give back a V vector
  void unapplyGaugeFixTwistFFT(A2AvectorW<Policies> &to, const int _p[3], Lattice &_lat) const{
    reverseGaugeFixAndTwist<FermionFieldType> op(_p,_lat); inversefft(to, &op);
  }

  //Use the relations between FFTs to obtain the FFT for a chosen quark momentum
  //With G-parity BCs there are 2 disjoint sets of momenta hence there are 2 base FFTs
  void getTwistedFFT(const int p[3], A2AvectorWfftw<Policies> const *base_p, A2AvectorWfftw<Policies> const *base_m = NULL);

  void shiftFieldsInPlace(const std::vector<int> &shift);

  //A version of the above that directly shifts the base Wfftw rather than outputting into a separate storage
  //Returns the pointer to the Wfftw acted upon and the *shift required to restore the Wfftw to it's original form* (use shiftFieldsInPlace to restore)
  
  static std::pair< A2AvectorWfftw<mf_Policies>*, std::vector<int> > inPlaceTwistedFFT(const int p[3], A2AvectorWfftw<mf_Policies> *base_p, A2AvectorWfftw<mf_Policies> *base_m = NULL);
  
  //The flavor and timeslice dilutions are still packed so we must treat them differently
  //Mode is a full 'StandardIndex', (unpacked mode index)
  inline const FieldSiteType & elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
    static FieldSiteType zero(0.0);
    if(mode < nl){
      int site = getWl(mode).threeToFour(x3d,t);
      return *(getWl(mode).site_ptr(site,flavor) + spin_color);
    }else{
      int mode_hit, mode_tblock, mode_spin_color,mode_flavor;
      const StandardIndexDilution &dilfull = static_cast<StandardIndexDilution const&>(*this);
      dilfull.indexUnmap(mode-nl,mode_hit,mode_tblock,mode_spin_color,mode_flavor);
      //flavor and time block indices match those of the mode, the result is zero
      int tblock = (t+GJP.TnodeSites()*GJP.TnodeCoor())/args.src_width;
      if(flavor != mode_flavor || tblock != mode_tblock) return zero;

      int site = getWh(mode_hit,mode_spin_color).threeToFour(x3d,t);
      return *(getWh(mode_hit,mode_spin_color).site_ptr(site,flavor) +spin_color); //because we multiplied by an SU(3) matrix, the field is not just a delta function in spin/color
    }
  }
  //Get a particular site/spin/color element of a given *native* (packed) mode 
  accelerator_inline const FieldSiteType & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
    return i < nl ? 
      *(wl[i]->site_ptr(site,flavor)+spin_color) :
      *(wh[i-nl]->site_ptr(site,flavor)+spin_color); //spin_color index diluted out.
  }

  //Replace this vector with the average of this another vector, 'with'
  void average(const A2AvectorWfftw<Policies> &with, const bool &parallel = true){
    if( !paramsEqual(with) ) ERR.General("A2AvectorWfftw","average","Second field must share the same underlying parameters\n");
    for(int i=0;i<nl;i++) wl[i]->average(*with.wl[i]);
    for(int i=0;i<12*nhits;i++) wh[i]->average(*with.wh[i]);
  }

  //Set each float to a uniform random number in the specified range.
  //WARNING: Uses only the current RNG in LRG, and does not change this based on site. This is therefore only useful for testing*
  void testRandom(const Float &hi = 0.5, const Float &lo = -0.5){
    for(int i=0;i<nl;i++) wl[i]->testRandom(hi,lo);
    for(int i=0;i<12*nhits;i++) wh[i]->testRandom(hi,lo);
  }

  //BELOW are for use by the meson field
  //Meson field W-type indices are described in terms of the timePacked dilution index , where flavor has been diluted out in the process of computing the meson fields
  //This method performs the flavor dilution 'in-place' (i.e. without actually unpacking into another object). 
  //'site' is a local canonical-ordered, packed four-vector
  //i_high_unmapped is the index i unmapped to its high mode sub-indices (if it is a high mode of course!)

  inline SCFvectorPtr<FieldSiteType> getFlavorDilutedVect(const int i, const modeIndexSet &i_high_unmapped, const int p3d, const int t) const{
    const FermionFieldType &field = i >= nl ? getWh(i_high_unmapped.hit, i_high_unmapped.spin_color): getWl(i);
    bool zero_hint[2] = {false,false};
    if(i >= nl) zero_hint[ !i_high_unmapped.flavor ] = true;

    const int x4d = field.threeToFour(p3d,t);
    return SCFvectorPtr<FieldSiteType>(zero_hint[0] ? &zerosc[0] : field.site_ptr(x4d,0), zero_hint[1] ? &zerosc[0] : field.site_ptr(x4d,1), zero_hint[0], zero_hint[1]);
  }
  //Return the pointer stride for between 3d coordinates for a given mode index and flavor. Relies on the dimension policy implementing dimpol_site_stride_3d
  inline int siteStride3D(const int i, const modeIndexSet &i_high_unmapped, const int f) const{ 
    const FermionFieldType &field = i >= nl ? getWh(i_high_unmapped.hit, i_high_unmapped.spin_color): getWl(i);
    bool zero_hint[2] = {false,false};
    if(i >= nl) zero_hint[ !i_high_unmapped.flavor ] = true;
    return zero_hint[f] ? 0 : field.dimpol_site_stride_3d()*field.siteSize();
  }

  template<typename extPolicies>
  void importFields(const A2AvectorWfftw<extPolicies> &r){
    if( !paramsEqual(r) ) ERR.General("A2AvectorWfftw","importFields","External field-vector must share the same underlying parameters\n");
    for(int i=0;i<nl;i++) wl[i]->importField(r.getWl(i));
    for(int i=0;i<12*nhits;i++) wh[i]->importField(r.getWh(i/12,i%12));
  }  

  inline void free_mem(){
    wl.free();
    wh.free();
  }
};

#include "implementation/a2a_impl.tcc"
#include "implementation/a2a_io.tcc"

CPS_END_NAMESPACE

#endif
