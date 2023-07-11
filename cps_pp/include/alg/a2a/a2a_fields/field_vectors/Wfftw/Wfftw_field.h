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


#include "implementation/Wfftw_impl.tcc"

CPS_END_NAMESPACE
