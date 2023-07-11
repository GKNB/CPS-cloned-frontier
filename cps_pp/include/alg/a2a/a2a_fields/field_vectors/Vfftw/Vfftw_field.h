#pragma once

#include<util/lattice.h>
#include<util/time_cps.h>

#ifdef USE_GRID
#include<util/lattice/fgrid.h>
#endif

#include<alg/ktopipi_jobparams.h>
#include<alg/a2a/base/a2a_dilutions.h>
#include<alg/a2a/utils.h>
#include<alg/a2a/lattice.h>

#include <alg/a2a/a2a_fields/field_vectors/field_array.h>
#include <alg/a2a/a2a_fields/field_vectors/field_utils.h>

CPS_START_NAMESPACE

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

#include "implementation/Vfftw_impl.tcc"

CPS_END_NAMESPACE
