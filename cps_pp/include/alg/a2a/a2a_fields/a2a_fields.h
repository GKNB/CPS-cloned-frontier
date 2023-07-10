#ifndef CPS_A2A_FIELDS_H_
#define CPS_A2A_FIELDS_H_

#include<util/lattice.h>
#include<util/time_cps.h>

#ifdef USE_GRID
#include<util/lattice/fgrid.h>
#endif

#include<alg/ktopipi_jobparams.h>
#include<alg/a2a/base/a2a_dilutions.h>
#include<alg/a2a/utils.h>
#include<alg/a2a/lattice.h>

#include "field_array.h"
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
  typedef FermionFieldType LowModeFieldType;
  typedef FermionFieldType HighModeFieldType;  
private:
  CPSfieldArray<FermionFieldType> v;

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

  A2AvectorV(const A2Aparams &_args): StandardIndexDilution(_args){    
    v.resize(nv);
    this->allocInitializeFields(v,NullObject());
  }
  
  A2AvectorV(const A2Aparams &_args, const FieldInputParamType &field_setup_params): StandardIndexDilution(_args){
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

  //Return true if the mode has been allocated
  inline bool modeIsAssigned(const int i) const{ return v[i].assigned(); }
  
  inline const FermionFieldType & getMode(const int i) const{ return *v[i]; }
  inline FermionFieldType & getMode(const int i){ return *v[i]; }  

  inline const FermionFieldType & getLowMode(const int il) const{ return *v[il]; }
  inline const FermionFieldType & getHighMode(const int ih) const{ return *v[nl+ih]; }
  
  //Get a mode from the low mode part
  inline FermionFieldType & getVl(const int il){ return *v[il]; }
  inline const FermionFieldType & getVl(const int il) const{ return *v[il]; }
  
  //Get a mode from the high-mode part
  inline FermionFieldType & getVh(const int ih){ return *v[nl+ih]; }
  inline const FermionFieldType & getVh(const int ih) const{ return *v[nl+ih]; }
 
  class View: public StandardIndexDilution{
    typename CPSfieldArray<FermionFieldType>::View av;
  public:
    typedef mf_Policies Policies;
    typedef typename CPSfieldArray<FermionFieldType>::FieldView FieldView;
    typedef typename FermionFieldType::FieldSiteType FieldSiteType;
    typedef StandardIndexDilution DilutionType;

    View(ViewMode mode, const CPSfieldArray<FermionFieldType> &vin, const StandardIndexDilution &d): av(vin.view(mode)), StandardIndexDilution(d){}
    View(ViewMode mode, const CPSfieldArray<FermionFieldType> &vin, const StandardIndexDilution &d, const std::vector<bool> &modes_used): av(vin.view(mode,modes_used)), StandardIndexDilution(d){}
    
    accelerator_inline FieldView & getMode(const int i) const{ return av[i]; }

    //Get a mode from the low mode part
    accelerator_inline FieldView & getVl(const int il) const{ return av[il]; }
    
    //Get a mode from the high-mode part
    accelerator_inline FieldView & getVh(const int ih) const{ return av[nl+ih]; }

    //Get a particular site/spin/color element of a given *native* (packed) mode. For V this does the same as the above
    //Note: site is the local 4d site offset
    accelerator_inline const FieldSiteType & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
      return *(av[i].site_ptr(site,flavor)+spin_color);
    }

    //Get a particular site/spin/color element of a given mode
    //Note: x3d is the local (on-node) 3d site and t is the local time
    inline const FieldSiteType & elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
      int x4d = av[mode].threeToFour(x3d,t);
      return  *(av[mode].site_ptr(x4d,flavor) + spin_color);
    }

    //For the mode i, get the base pointer and size (in units of FieldSiteType) of the field. Includes all flavors, contiguity guaranteed
    inline void getModeData(FieldSiteType const* &ptr, size_t &size, const int i) const{
      ptr = av[i].ptr(); size = av[i].size();
    }

    //For the mode i, get the base pointers for the provided timeslice (local) for each flavor along with the size (in units of FieldSiteType) of the timeslice field. Contiguity guaranteed
    inline void getModeTimesliceData(FieldSiteType const* &ptr_f0, FieldSiteType const* &ptr_f1, size_t &size, const int i, const int t) const{
      assert(av[i].dimpol_flavor_timeslice_contiguous());
      size = av[i].dimpol_time_stride() * av[i].siteSize();      
      ptr_f0 = av[i].ptr() + t*size;
      ptr_f1 = ptr_f0 + av[i].flav_offset();
    }

    void free(){
      av.free();
    }
	
  };

  View view(ViewMode mode) const{ return View(mode, v, *this); }
  //Open a view only to some subset of modes. Undefined behavior if you access one that you are not supposed to!
  View view(ViewMode mode, const std::vector<bool> &modes_used) const{ return View(mode,v,*this,modes_used); }
  
  void enqueuePrefetch(ViewMode mode, const std::vector<bool> &modes_used) const{
    for(int i=0;i<nv;i++) if(modes_used[i])  v[i]->enqueuePrefetch(mode);
  }   
  static inline void startPrefetches(){ Policies::AllocPolicy::startPrefetches(); }
  static inline void waitPrefetches(){ Policies::AllocPolicy::waitPrefetches(); }
  
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

  //Set all fields to zero
  void zero(){ for(int i=0;i<nv;i++) v[i]->zero(); }
  
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
  typedef FermionFieldType LowModeFieldType;
  typedef FermionFieldType HighModeFieldType;  
  
#define VFFTW_ENABLE_IF_MANUAL_ALLOC(P) typename my_enable_if<  _equal<typename P::A2AvectorVfftwPolicies::FieldAllocStrategy,ManualAllocStrategy>::value , void>::type
private:
  CPSfieldArray<FermionFieldType> v;
  
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
  
  A2AvectorVfftw(const A2Aparams &_args): StandardIndexDilution(_args){
    v.resize(nv);
    this->allocInitializeFields(v,NullObject());
  }
  A2AvectorVfftw(const A2Aparams &_args, const FieldInputParamType &field_setup_params): StandardIndexDilution(_args){
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

  //Return true if the mode has been allocated
  inline bool modeIsAssigned(const int i) const{ return v[i].assigned(); }
  
  inline const FermionFieldType & getMode(const int i) const{ return *v[i]; }
  inline const FermionFieldType & getMode(const int i, const modeIndexSet &i_high_unmapped) const{ return getMode(i); }

  inline FermionFieldType & getMode(const int i){ return *v[i]; }

  inline const FermionFieldType & getLowMode(const int il) const{ return *v[il]; }
  inline const FermionFieldType & getHighMode(const int ih) const{ return *v[nl+ih]; }
  
  //Create a regular fermion field for a given full mode by unpacking the dilution
  void unpackMode(FermionFieldType &into, const int mode) const{
    into = getMode(mode);
  }
  
  class View: public StandardIndexDilution{
    typename CPSfieldArray<FermionFieldType>::View av;
  public:
    typedef mf_Policies Policies;
    typedef typename CPSfieldArray<FermionFieldType>::FieldView FieldView; 
    typedef typename FermionFieldType::FieldSiteType FieldSiteType;
    typedef StandardIndexDilution DilutionType;
    
    View(ViewMode mode, const CPSfieldArray<FermionFieldType> &vin, const StandardIndexDilution &d): av(vin.view(mode)), StandardIndexDilution(d){}
    View(ViewMode mode, const CPSfieldArray<FermionFieldType> &vin, const StandardIndexDilution &d, const std::vector<bool> &modes_used): av(vin.view(mode,modes_used)), StandardIndexDilution(d){}
     
    accelerator_inline FieldView & getMode(const int i) const{ return av[i]; }

    accelerator_inline FieldView & getMode(const int i, const modeIndexSet &i_high_unmapped) const{ return av[i]; }
      
    //Get a particular site/spin/color element of a given 'native' (packed) mode. For V this does the same thing as the above
    accelerator_inline const FieldSiteType & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
      return *(av[i].site_ptr(site,flavor)+spin_color);
    }

    inline const FieldSiteType & elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
      int site = av[mode].threeToFour(x3d,t);
      return *(av[mode].site_ptr(site,flavor) + spin_color);
    }

    //For the mode i, get the base pointer and size (in units of FieldSiteType) of the field. Includes all flavors, contiguity guaranteed
    inline void getModeData(FieldSiteType const* &ptr, size_t &size, const int i) const{
      ptr = av[i].ptr(); size = av[i].size();
    }
    
    //For the mode i, get the base pointers for the provided timeslice (local) for each flavor along with the size (in units of FieldSiteType) of the timeslice field. Contiguity guaranteed
    inline void getModeTimesliceData(FieldSiteType const* &ptr_f0, FieldSiteType const* &ptr_f1, size_t &size, const int i, const int t) const{
      assert(av[i].dimpol_flavor_timeslice_contiguous());
      size = av[i].dimpol_time_stride() * av[i].siteSize();      
      ptr_f0 = av[i].ptr() + t*size;
      ptr_f1 = ptr_f0 + av[i].flav_offset();
    }

    //i_high_unmapped is the index i unmapped to its high mode sub-indices (if it is a high mode of course!)
    inline SCFvectorPtr<FieldSiteType> getFlavorDilutedVect(const int i, const modeIndexSet &i_high_unmapped, const int p3d, const int t) const{
      const FieldView &field = getMode(i);
      const int x4d = field.threeToFour(p3d,t);
      FieldSiteType const *f0 = field.site_ptr(x4d,0);
      return SCFvectorPtr<FieldSiteType>(f0,f0+field.flav_offset());
    }

    void free(){ av.free(); }
  };

  View view(ViewMode mode) const{ return View(mode, v, *this); }
  //Open a view only to some subset of modes. Undefined behavior if you access one that you are not supposed to!
  View view(ViewMode mode, const std::vector<bool> &modes_used) const{ return View(mode,v,*this,modes_used); }

  void enqueuePrefetch(ViewMode mode, const std::vector<bool> &modes_used) const{
    for(int i=0;i<nv;i++) if(modes_used[i])  v[i]->enqueuePrefetch(mode);
  }
  static inline void startPrefetches(){ Policies::AllocPolicy::startPrefetches(); }
  static inline void waitPrefetches(){ Policies::AllocPolicy::waitPrefetches(); }
  
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
  
  //Use the relations between FFTs under translations to compute the FFT for a chosen quark momentum
  //With G-parity BCs there are 2 disjoint sets of momenta hence there are 2 base FFTs
  void getTwistedFFT(const int p[3], A2AvectorVfftw<Policies> const *base_p, A2AvectorVfftw<Policies> const *base_m = NULL);

  //C-shift the fields (with periodic BCs) according to the given shift vector
  void shiftFieldsInPlace(const std::vector<int> &shift);

  //A version of the above that directly shifts the base Wfftw rather than outputting into a separate storage
  //Returns the pointer to the Wfftw acted upon and the *shift required to restore the Wfftw to it's original form* (use shiftFieldsInPlace to restore)
  
  static std::pair< A2AvectorVfftw<mf_Policies>*, std::vector<int> > inPlaceTwistedFFT(const int p[3], A2AvectorVfftw<mf_Policies> *base_p, A2AvectorVfftw<mf_Policies> *base_m = NULL);
  
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

  //Set all fields to zero
  void zero(){ for(int i=0;i<nv;i++) v[i]->zero(); }
  
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

  typedef FermionFieldType LowModeFieldType;
  typedef ComplexFieldType HighModeFieldType;
private:
  CPSfieldArray<FermionFieldType> wl; //The low mode part of the W field, comprised of nl fermion fields
  CPSfieldArray<ComplexFieldType> wh; //The high mode random part of the W field, comprised of nhits complex flavored scalar fields. Note: the spin/color dilution is performed later

  bool wh_rand_performed; //store if the wh random numbers have been set
  
  void initialize(const FieldInputParamType &field_setup_params);
public:
  typedef FullyPackedIndexDilution DilutionType;

  A2AvectorW(const A2AArg &_args);
  A2AvectorW(const A2AArg &_args, const FieldInputParamType &field_setup_params);

  A2AvectorW(const A2Aparams &_args);
  A2AvectorW(const A2Aparams &_args, const FieldInputParamType &field_setup_params);

  //Manually set the Wh random fields. Expects a vector of size nhits
  void setWh(const std::vector<ScalarComplexFieldType> &to);

  //Return whether the random sources have been set
  bool WhRandPerformed() const{ return wh_rand_performed; }

  static double Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params);

  inline FieldInputParamType getFieldInputParams() const{
    if(wl.size() == 0 && wh.size() == 0) ERR.General("A2AvectorW","getFieldInputParams","Wl and Wh sizes are both zero\n");
    if(wl.size() != 0 && wl[0].assigned()) return wl[0]->getDimPolParams();
    if(wh.size() != 0 && wh[0].assigned()) return wh[0]->getDimPolParams();
    
    ERR.General("A2AvectorW","getFieldInputParams","Neither of the zeroth fields are assigned\n");
  }   

  //Return true if the mode has been allocated
  inline bool modeIsAssigned(const int i) const{ return i<nl ? wl[i].assigned() : wh[i-nl].assigned(); }
  
  inline const FermionFieldType & getWl(const int i) const{ return *wl[i]; }
  inline const ComplexFieldType & getWh(const int hit) const{ return *wh[hit]; }

  inline FermionFieldType & getWl(const int i){ return *wl[i]; }
  inline ComplexFieldType & getWh(const int hit){ return *wh[hit]; }

  inline const FermionFieldType & getLowMode(const int il) const{ return *wl[il]; }
  inline const ComplexFieldType & getHighMode(const int ih) const{ return *wh[ih]; }
   
  class View: public FullyPackedIndexDilution{
    typename CPSfieldArray<FermionFieldType>::View awl;
    typename CPSfieldArray<ComplexFieldType>::View awh;
  public:
    typedef mf_Policies Policies;
    typedef typename CPSfieldArray<FermionFieldType>::FieldView FermionFieldView;
    typedef typename CPSfieldArray<ComplexFieldType>::FieldView ComplexFieldView;
    typedef typename FermionFieldType::FieldSiteType FieldSiteType;
    typedef FullyPackedIndexDilution DilutionType;

    View(ViewMode mode,
	 const CPSfieldArray<FermionFieldType> &wlin,
	 const CPSfieldArray<ComplexFieldType> &whin,
	 const FullyPackedIndexDilution &d): awl(wlin.view(mode)), awh(whin.view(mode)), FullyPackedIndexDilution(d){}

    View(ViewMode mode,
	 const CPSfieldArray<FermionFieldType> &wlin,
	 const CPSfieldArray<ComplexFieldType> &whin,
	 const FullyPackedIndexDilution &d, const std::vector<bool> &modes_used):     awl(wlin.view(mode, std::vector<bool>(modes_used.begin(),modes_used.begin()+d.getNl()) ) ),
										      awh(whin.view(mode, std::vector<bool>(modes_used.begin()+d.getNl(),modes_used.end()) ) ),
										      FullyPackedIndexDilution(d){}

    
    accelerator_inline FermionFieldView & getWl(const int i) const{ return awl[i]; }
    accelerator_inline ComplexFieldView & getWh(const int hit) const{ return awh[hit]; }
  
    //The spincolor, flavor and timeslice dilutions are packed so we must treat them differently
    //Mode is a full 'StandardIndex', (unpacked mode index)
    //Note usable on device
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
	if(spin_color != mode_spin_color || flavor != mode_flavor || tblock != mode_tblock) return zero;
	int site = getWh(mode_hit).threeToFour(x3d,t);
	return *(getWh(mode_hit).site_ptr(site,flavor)); //we use different random fields for each time and flavor, although we didn't have to
      }
    }

    //Get a particular site/spin/color element of a given *native* (packed) mode 
    accelerator_inline const FieldSiteType & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
      return i < nl ? 
		 *(awl[i].site_ptr(site,flavor)+spin_color) :
	*(awh[i-nl].site_ptr(site,flavor)); //we use different random fields for each time and flavor, although we didn't have to
    }

    //For the mode i, get the base pointer and size (in units of FieldSiteType) of the field. Includes all flavors, contiguity guaranteed
    inline void getModeData(FieldSiteType const* &ptr, size_t &size, const int i) const{
      if(i<nl){
	ptr = awl[i].ptr(); size = awl[i].size();
      }else{
	int ii = i-nl;
	ptr = awh[ii].ptr(); size = awh[ii].size();
      }
    }

    //For the mode i, get the base pointers for the provided timeslice (local) for each flavor along with the size (in units of FieldSiteType) of the timeslice field. Contiguity guaranteed
    inline void getModeTimesliceData(FieldSiteType const* &ptr_f0, FieldSiteType const* &ptr_f1, size_t &size, const int i, const int t) const{
      if(i<nl){
	assert(awl[i].dimpol_flavor_timeslice_contiguous());
	size = awl[i].dimpol_time_stride() * awl[i].siteSize();      
	ptr_f0 = awl[i].ptr() + t*size;
	ptr_f1 = ptr_f0 + awl[i].flav_offset();
      }else{
	int ii = i-nl;
	assert(awh[ii].dimpol_flavor_timeslice_contiguous());	  
	size = awh[ii].dimpol_time_stride() * awh[ii].siteSize();
	ptr_f0 = awh[ii].ptr() + t*size;
	ptr_f1 = ptr_f0 + awh[ii].flav_offset();
      }
    }

    void free(){ awl.free(); awh.free(); }
  };

  View view(ViewMode mode) const{ return View(mode, wl, wh, *this); }
  //Open a view only to some subset of modes. Undefined behavior if you access one that you are not supposed to!
  View view(ViewMode mode, const std::vector<bool> &modes_used) const{ return View(mode, wl, wh, *this, modes_used); }

  void enqueuePrefetch(ViewMode mode, const std::vector<bool> &modes_used) const{
    for(int i=0;i<nl;i++) if(modes_used[i])  wl[i]->enqueuePrefetch(mode);
    for(int i=0;i<nhits;i++) if(modes_used[i+nl]) wh[i]->enqueuePrefetch(mode);
  }   
  static inline void startPrefetches(){ Policies::AllocPolicy::startPrefetches(); }
  static inline void waitPrefetches(){ Policies::AllocPolicy::waitPrefetches(); }
  
  void importWl(const FermionFieldType &wlin, const int i){
    *wl[i] = wlin;
  }
  void importWh(const ComplexFieldType &whin, const int hit){
    *wh[hit] = whin;
  }

  //Get the diluted source with StandardIndex high-mode index dil_id.
  //Here dil_id is the combined spin-color/flavor/hit/tblock index
  template<typename TargetFermionFieldType>
  void getDilutedSource(TargetFermionFieldType &into, const int dil_id) const;

  //When gauge fixing prior to taking the FFT it is necessary to uncompact the wh field in the spin-color index, as these indices are acted upon by the gauge fixing
  //(I suppose technically only the color indices need uncompacting; this might be considered as a future improvement)
  void getSpinColorDilutedSource(FermionFieldType &into, const int hit, const int sc_id) const;

  //Set each float to a uniform random number in the specified range.
  //WARNING: Uses only the current RNG in LRG, and does not change this based on site. This is therefore only useful for testing*
  void testRandom(const Float &hi = 0.5, const Float &lo = -0.5){
    for(int i=0;i<nl;i++) wl[i]->testRandom(hi,lo);
    for(int i=0;i<nhits;i++) wh[i]->testRandom(hi,lo);
  }

  //Set all fields to zero
  void zero(){
    for(int i=0;i<nl;i++) wl[i]->zero();
    for(int i=0;i<nhits;i++) wh[i]->zero();
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


//General w_FFT^(i)_{sc,f}(p,t) = \rho^(i_sc,i_h)_{sc}(p) \delta_{t,i_t} \delta_{f,i_f}
//Only non-zero elements are    w_FFT^(i)_{sc,i_f}(p,i_t) = w_FFT^(i_sc,i_h,i_f,i_t)_{sc,i_f}(p,i_t)
//Utilize fermion field's flavor and time coordinates for i_f, i_t  leaving only i_sc, i_h to index the new fields
//Store in packed format:   i'=(i_sc,i_h)   t'=i_t f'=i_f    w'_FFT^(i')_{sc,f'}(p,t') = w_FFT^(i'_sc,i'_h,f',t')_{sc,f'}(p,t')
template< typename mf_Policies>
class A2AvectorWfftw: public TimeFlavorPackedIndexDilution, public mf_Policies::A2AvectorWfftwPolicies{
public:
  typedef mf_Policies Policies;
  typedef typename Policies::FermionFieldType FermionFieldType;
  typedef typename Policies::ComplexFieldType ComplexFieldType;
  typedef typename FermionFieldType::FieldSiteType FieldSiteType;
  typedef typename my_enable_if< _equal<typename FermionFieldType::InputParamType, typename ComplexFieldType::InputParamType>::value,  typename FermionFieldType::InputParamType>::type FieldInputParamType;
  typedef FermionFieldType LowModeFieldType;
  typedef FermionFieldType HighModeFieldType;
  
#define WFFTW_ENABLE_IF_MANUAL_ALLOC(P) typename my_enable_if<  _equal<typename P::A2AvectorWfftwPolicies::FieldAllocStrategy,ManualAllocStrategy>::value , int>::type
private:

  CPSfieldArray<FermionFieldType> wl;
  CPSfieldArray<FermionFieldType> wh; //these have been diluted in spin/color but not the other indices, hence there are nhit * 12 fields here (spin/color index changes fastest in mapping)

  void initialize(const FieldInputParamType &field_setup_params);
public:
  typedef TimeFlavorPackedIndexDilution DilutionType;

  A2AvectorWfftw(const A2AArg &_args);
  A2AvectorWfftw(const A2AArg &_args, const FieldInputParamType &field_setup_params);
  A2AvectorWfftw(const A2Aparams &_args);
  A2AvectorWfftw(const A2Aparams &_args, const FieldInputParamType &field_setup_params);

  static double Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params);

  inline FieldInputParamType getFieldInputParams() const{
    if(wl.size() == 0 && wh.size() == 0) ERR.General("A2AvectorWfft","getFieldInputParams","Wl and Wh sizes are both zero\n");
    if(wl.size() != 0 && wl[0].assigned()) return wl[0]->getDimPolParams();
    if(wh.size() != 0 && wh[0].assigned()) return wh[0]->getDimPolParams();
    
    ERR.General("A2AvectorWfftw","getFieldInputParams","Neither of the zeroth fields are assigned\n");
  }   

  //Return true if the mode has been allocated
  inline bool modeIsAssigned(const int i) const{ return i<nl ? wl[i].assigned() : wh[i-nl].assigned(); }
  
  inline const FermionFieldType & getWl(const int i) const{ return *wl[i]; }
  inline const FermionFieldType & getWh(const int hit, const int spin_color) const{ return *wh[spin_color + 12*hit]; }
  
  inline const FermionFieldType & getMode(const int i) const{ return i < nl ? *wl[i] : *wh[i-nl]; }

  inline FermionFieldType & getWl(const int i){ return *wl[i]; }
  inline FermionFieldType & getWh(const int hit, const int spin_color){ return *wh[spin_color + 12*hit]; }

  inline FermionFieldType & getMode(const int i){ return i < nl ? *wl[i] : *wh[i-nl]; }

  inline const FermionFieldType & getLowMode(const int il) const{ return *wl[il]; }
  inline const FermionFieldType & getHighMode(const int ih) const{ return *wh[ih]; }

  //This version allows for the possibility of a different high mode mapping for the index i by passing the unmapped indices: for i>=nl the modeIndexSet is used to obtain the appropriate mode 
  inline const FermionFieldType & getMode(const int i, const modeIndexSet &i_high_unmapped) const{ return i >= nl ? getWh(i_high_unmapped.hit, i_high_unmapped.spin_color): getWl(i); }
 
  //Create a regular fermion field for a given full mode by unpacking the dilution
  void unpackMode(FermionFieldType &into, const int mode) const{
    if(mode < nl){
      into = getWl(mode);
    }else{
      //Data is a delta function in time-block, hit and flavor but not in spin_color     
      into.zero();
      
      int mode_hit, mode_tblock, mode_spin_color,mode_flavor;
      const StandardIndexDilution &dilfull = static_cast<StandardIndexDilution const&>(*this);
      dilfull.indexUnmap(mode-nl,mode_hit,mode_tblock,mode_spin_color,mode_flavor);

      const FermionFieldType &mode_packed = getWh(mode_hit,mode_spin_color);
      size_t size_3d = mode_packed.nodeSites(0)*mode_packed.nodeSites(1)*mode_packed.nodeSites(2);
      assert(mode_packed.nodeSites(3) == GJP.TnodeSites());

      CPSautoView(into_v,into,HostWrite);
      CPSautoView(mode_packed_v,mode_packed,HostRead);

      for(int t=0;t<GJP.TnodeSites();t++){
	int tblock = (t+GJP.TnodeSites()*GJP.TnodeCoor())/args.src_width;
	if(tblock != mode_tblock) continue;
#pragma omp parallel for
	for(size_t x3d=0;x3d<size_3d;x3d++){
	  size_t x4d = mode_packed.threeToFour(x3d,t);
	  FieldSiteType const* from_data = mode_packed_v.site_ptr(x4d,mode_flavor);
	  FieldSiteType *to_data = into_v.site_ptr(x4d,mode_flavor);
	  memcpy(to_data,from_data,12*sizeof(FieldSiteType));
	}
      }
    }
  }
  
  class View: public TimeFlavorPackedIndexDilution{
    typename CPSfieldArray<FermionFieldType>::View awl;
    typename CPSfieldArray<FermionFieldType>::View awh;
  public:
    typedef mf_Policies Policies;
    typedef typename CPSfieldArray<FermionFieldType>::FieldView FermionFieldView;
    typedef typename FermionFieldType::FieldSiteType FieldSiteType;
    typedef TimeFlavorPackedIndexDilution DilutionType;

    View(ViewMode mode,
	 const CPSfieldArray<FermionFieldType> &wlin,
	 const CPSfieldArray<FermionFieldType> &whin,
	 const TimeFlavorPackedIndexDilution &d): awl(wlin.view(mode)), awh(whin.view(mode)), TimeFlavorPackedIndexDilution(d){}

    View(ViewMode mode,
	 const CPSfieldArray<FermionFieldType> &wlin,
	 const CPSfieldArray<FermionFieldType> &whin,
	 const TimeFlavorPackedIndexDilution &d, const std::vector<bool> &modes_used):     awl(wlin.view(mode, std::vector<bool>(modes_used.begin(),modes_used.begin()+d.getNl()) ) ),
											   awh(whin.view(mode, std::vector<bool>(modes_used.begin()+d.getNl(),modes_used.end()) ) ),
											   TimeFlavorPackedIndexDilution(d){}

    
    accelerator_inline FermionFieldView & getWl(const int i) const{ return awl[i]; }
    accelerator_inline FermionFieldView & getWh(const int hit, const int spin_color) const{ return awh[spin_color + 12*hit]; }

    accelerator_inline FermionFieldView & getMode(const int i) const{ return i < nl ? awl[i] : awh[i-nl]; }

    //This version allows for the possibility of a different high mode mapping for the index i by passing the unmapped indices: for i>=nl the modeIndexSet is used to obtain the appropriate mode 
    accelerator_inline FermionFieldView & getMode(const int i, const modeIndexSet &i_high_unmapped) const{ return i >= nl ? getWh(i_high_unmapped.hit, i_high_unmapped.spin_color): getWl(i); }

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

    accelerator_inline const FieldSiteType & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
      return i < nl ? 
		 *(awl[i].site_ptr(site,flavor)+spin_color) :
	*(awh[i-nl].site_ptr(site,flavor)+spin_color); //spin_color index diluted out.
    }

    //BELOW are for use by the meson field
    //For high modes it returns   wFFTP^(j_h,j_sc)_{sc',f'}(p,t) \delta_{f',j_f}   [cf a2a_dilutions.h]
    //i is only used for low mode indices. If i>=nl  the appropriate data is picked using i_high_unmapped
    //t is the local time
    //Note: code not suitable for execution on device
    inline SCFvectorPtr<FieldSiteType> getFlavorDilutedVect(const int i, const modeIndexSet &i_high_unmapped, const int p3d, const int t) const{
      static FieldSiteType zerosc[12]; static bool init=false;
      if(!init){ for(int i=0;i<12;i++) CPSsetZero(zerosc[i]); init = true; }

      if(i >= nl) assert(i_high_unmapped.hit != -1 && i_high_unmapped.flavor != -1 && i_high_unmapped.spin_color != -1);
      const FermionFieldView &field = i >= nl ? getWh(i_high_unmapped.hit, i_high_unmapped.spin_color): getWl(i);
      bool zero_hint[2] = {false,false};
      if(i >= nl) zero_hint[ !i_high_unmapped.flavor ] = true;
      
      const int x4d = field.threeToFour(p3d,t);
      return SCFvectorPtr<FieldSiteType>(zero_hint[0] ? &zerosc[0] : field.site_ptr(x4d,0), zero_hint[1] ? &zerosc[0] : field.site_ptr(x4d,1), zero_hint[0], zero_hint[1]);
    }
    
    //For the mode i, get the base pointer and size (in units of FieldSiteType) of the field. Includes all flavors, contiguity guaranteed
    inline void getModeData(FieldSiteType const* &ptr, size_t &size, const int i) const{
      if(i<nl){
	ptr = awl[i].ptr(); size = awl[i].size();
      }else{
	int ii = i-nl;
	ptr = awh[ii].ptr(); size = awh[ii].size();
      }
    }

    //For the mode i, get the base pointers for the provided timeslice (local) for each flavor along with the size (in units of FieldSiteType) of the timeslice field. Contiguity guaranteed
    inline void getModeTimesliceData(FieldSiteType const* &ptr_f0, FieldSiteType const* &ptr_f1, size_t &size, const int i, const int t) const{
      if(i<nl){
	assert(awl[i].dimpol_flavor_timeslice_contiguous());
	size = awl[i].dimpol_time_stride() * awl[i].siteSize();      
	ptr_f0 = awl[i].ptr() + t*size;
	ptr_f1 = ptr_f0 + awl[i].flav_offset();
      }else{
	int ii = i-nl;
	assert(awh[ii].dimpol_flavor_timeslice_contiguous());	  
	size = awh[ii].dimpol_time_stride() * awh[ii].siteSize();
	ptr_f0 = awh[ii].ptr() + t*size;
	ptr_f1 = ptr_f0 + awh[ii].flav_offset();
      }
    }

    void free(){ awl.free(); awh.free(); }
  };

  View view(ViewMode mode) const{ return View(mode, wl, wh, *this); }
  //Open a view only to some subset of modes. Undefined behavior if you access one that you are not supposed to!
  View view(ViewMode mode, const std::vector<bool> &modes_used) const{ return View(mode, wl, wh, *this, modes_used); }

  void enqueuePrefetch(ViewMode mode, const std::vector<bool> &modes_used) const{
    for(int i=0;i<nl;i++) if(modes_used[i])  wl[i]->enqueuePrefetch(mode);
    for(int i=0;i<12*nhits;i++) if(modes_used[i+nl]) wh[i]->enqueuePrefetch(mode);
  }   
  static inline void startPrefetches(){ Policies::AllocPolicy::startPrefetches(); }
  static inline void waitPrefetches(){ Policies::AllocPolicy::waitPrefetches(); }
  
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

  //Set all fields to zero
  void zero(){ 
    for(int i=0;i<nl;i++) wl[i]->zero();
    for(int i=0;i<12*nhits;i++) wh[i]->zero();
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
