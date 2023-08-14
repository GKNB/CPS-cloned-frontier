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
class A2AvectorWunitary: public TimeSpinColorPackedIndexDilution, public mf_Policies::A2AvectorWunitarypolicies{
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
  typedef TimeSpinColorPackedIndexDilution DilutionType;

  A2AvectorWunitary(const A2AArg &_args);
  A2AvectorWunitary(const A2AArg &_args, const FieldInputParamType &field_setup_params);

  A2AvectorWunitary(const A2Aparams &_args);
  A2AvectorWunitary(const A2Aparams &_args, const FieldInputParamType &field_setup_params);

  //Manually set the Wh random fields. Expects a vector of size nhits
  void setWh(const std::vector<ScalarComplexFieldType> &to);

  //Return whether the random sources have been set
  bool WhRandPerformed() const{ return wh_rand_performed; }

  static double Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params);

  inline FieldInputParamType getFieldInputParams() const{
    if(wl.size() == 0 && wh.size() == 0) ERR.General("A2AvectorWunitary","getFieldInputParams","Wl and Wh sizes are both zero\n");
    if(wl.size() != 0 && wl[0].assigned()) return wl[0]->getDimPolParams();
    if(wh.size() != 0 && wh[0].assigned()) return wh[0]->getDimPolParams();
    
    ERR.General("A2AvectorWunitary","getFieldInputParams","Neither of the zeroth fields are assigned\n");
  }   

  //Return true if the mode has been allocated
  inline bool modeIsAssigned(const int i) const{ return i<nl ? wl[i].assigned() : wh[i-nl].assigned(); }
  
  inline const FermionFieldType & getWl(const int i) const{ return *wl[i]; }
  inline const ComplexFieldType & getWh(const int i) const{ return *wh[i]; }

  inline FermionFieldType & getWl(const int i){ return *wl[i]; }
  inline ComplexFieldType & getWh(const int i){ return *wh[i]; }

  inline const FermionFieldType & getLowMode(const int il) const{ return *wl[il]; }
  inline const ComplexFieldType & getHighMode(const int ih) const{ return *wh[ih]; }
   
  class View: public TimeSpinColorPackedIndexDilution{
    typename CPSfieldArray<FermionFieldType>::View awl;
    typename CPSfieldArray<ComplexFieldType>::View awh;
  public:
    typedef mf_Policies Policies;
    typedef typename CPSfieldArray<FermionFieldType>::FieldView FermionFieldView;
    typedef typename CPSfieldArray<ComplexFieldType>::FieldView ComplexFieldView;
    typedef typename FermionFieldType::FieldSiteType FieldSiteType;
    typedef TimeSpinColorPackedIndexDilution DilutionType;

    View(ViewMode mode,
	 const CPSfieldArray<FermionFieldType> &wlin,
	 const CPSfieldArray<ComplexFieldType> &whin,
	 const TimeSpinColorPackedIndexDilution &d): awl(wlin.view(mode)), awh(whin.view(mode)), TimeSpinColorPackedIndexDilution(d){}

    View(ViewMode mode,
	 const CPSfieldArray<FermionFieldType> &wlin,
	 const CPSfieldArray<ComplexFieldType> &whin,
	 const TimeSpinColorPackedIndexDilution &d, const std::vector<bool> &modes_used):     awl(wlin.view(mode, std::vector<bool>(modes_used.begin(),modes_used.begin()+d.getNl()) ) ),
											      awh(whin.view(mode, std::vector<bool>(modes_used.begin()+d.getNl(),modes_used.end()) ) ),
											      TimeSpinColorPackedIndexDilution(d){}

    
    accelerator_inline FermionFieldView & getWl(const int i) const{ return awl[i]; }
    accelerator_inline ComplexFieldView & getWh(const int i) const{ return awh[i]; }
    accelerator_inline FermionFieldView & getLowMode(const int i) const{ return awl[i]; }
    accelerator_inline ComplexFieldView & getHighMode(const int i) const{ return awh[i]; }
  
    //The spincolor, flavor and timeslice dilutions are packed so we must treat them differently
    //Mode is a full 'StandardIndex', (unpacked mode index)
    //Note usable on device
    inline FieldSiteType elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
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
	if(spin_color != mode_spin_color || tblock != mode_tblock) return zero;
	int idx = this->indexMap(mode_hit,mode_flavor); //packed index
	int site = getWh(idx).threeToFour(x3d,t);

	//As a matrix, the mode gives the column index
	return *(getWh(idx).site_ptr(site,flavor));
      }
    }

    //Get a particular site/spin/color element of a given *native* (packed) mode 
    accelerator_inline const FieldSiteType & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
      return i < nl ? 
		 *(awl[i].site_ptr(site,flavor)+spin_color) :
	*(awh[i-nl].site_ptr(site,flavor));
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
    for(int i=0;i<wh.size();i++) if(modes_used[i+nl]) wh[i]->enqueuePrefetch(mode);
  }   
  static inline void startPrefetches(){ Policies::AllocPolicy::startPrefetches(); }
  static inline void waitPrefetches(){ Policies::AllocPolicy::waitPrefetches(); }
  
  void importWl(const FermionFieldType &wlin, const int i){
    *wl[i] = wlin;
  }
  void importWh(const ComplexFieldType &whin, const int i){
    *wh[i] = whin;
  }

  //Get the diluted source with StandardIndex high-mode index dil_id.
  //Here dil_id is the combined spin-color/flavor/hit/tblock index
  template<typename TargetFermionFieldType>
  void getDilutedSource(TargetFermionFieldType &into, const int dil_id) const;

  //When gauge fixing prior to taking the FFT it is necessary to uncompact the wh field in the spin-color index, as these indices are acted upon by the gauge fixing
  //(I suppose technically only the color indices need uncompacting; this might be considered as a future improvement)
  void getSpinColorDilutedSource(FermionFieldType &into, const int high_mode_idx, const int sc_id) const;

  //Set each float to a uniform random number in the specified range.
  //WARNING: Uses only the current RNG in LRG, and does not change this based on site. This is therefore only useful for testing*
  void testRandom(const Float &hi = 0.5, const Float &lo = -0.5){
    for(int i=0;i<nl;i++) wl[i]->testRandom(hi,lo);
    for(int i=0;i<wh.size();i++) wh[i]->testRandom(hi,lo);
  }

  //Set all fields to zero
  void zero(){
    for(int i=0;i<nl;i++) wl[i]->zero();
    for(int i=0;i<wh.size();i++) wh[i]->zero();
  }

  void writeParallelWithGrid(const std::string &file_stub) const;
  void readParallelWithGrid(const std::string &file_stub);

  inline void free_mem(){
    wl.free();    wh.free();
  }
};

#include "implementation/Wunitary_impl.tcc"

CPS_END_NAMESPACE
