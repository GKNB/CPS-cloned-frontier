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
class A2AvectorWunitaryfftw;

template< typename mf_Policies>
class A2AvectorWtimePacked: public TimePackedIndexDilution, public mf_Policies::A2AvectorWtimePackedPolicies{
public:
  typedef mf_Policies Policies;
  typedef typename Policies::FermionFieldType FermionFieldType;

  typedef typename Policies::ScalarFermionFieldType ScalarFermionFieldType;

  typedef typename FermionFieldType::FieldSiteType FieldSiteType;
  typedef typename FermionFieldType::InputParamType FieldInputParamType;

  typedef FermionFieldType LowModeFieldType;
  typedef FermionFieldType HighModeFieldType;
private:
  CPSfieldArray<FermionFieldType> w;

  bool wh_rand_performed; //store if the wh random numbers have been set
  
  void initialize(const FieldInputParamType &field_setup_params);
public:
  typedef TimePackedIndexDilution DilutionType;
  typedef A2AvectorWunitaryfftw<mf_Policies> FFTvectorType;
  template<typename P> using FFTvectorTemplate = A2AvectorWunitaryfftw<P>;
  template<typename P> using VectorTemplate = A2AvectorWtimePacked<P>;
  
  A2AvectorWtimePacked(const A2AArg &_args);
  A2AvectorWtimePacked(const A2AArg &_args, const FieldInputParamType &field_setup_params);

  A2AvectorWtimePacked(const A2Aparams &_args);
  A2AvectorWtimePacked(const A2Aparams &_args, const FieldInputParamType &field_setup_params);

  //Manually set the Wh random fields. Expects a vector of size nhits
  void setWh(const std::vector<ScalarFermionFieldType> &to);

  //Return whether the random sources have been set
  bool WhRandPerformed() const{ return wh_rand_performed; }

  static double Mbyte_size(const A2AArg &_args, const FieldInputParamType &field_setup_params);

  inline FieldInputParamType getFieldInputParams() const{
    if(w.size() == 0) ERR.General("A2AvectorWtimePacked","getFieldInputParams","W size is zero\n");
    if(w.size() != 0 && w[0].assigned()) return w[0]->getDimPolParams();    
    ERR.General("A2AvectorWtimePacked","getFieldInputParams","Neither of the zeroth fields are assigned\n");
  }   

  //Return true if the mode has been allocated
  inline bool modeIsAssigned(const int i) const{ return w[i].assigned(); }
  
  inline const FermionFieldType & getWl(const int i) const{ return *w[i]; }
  inline const FermionFieldType & getWh(const int i) const{ return *w[i+nl]; }

  inline FermionFieldType & getWl(const int i){ return *w[i]; }
  inline FermionFieldType & getWh(const int i){ return *w[i+nl]; }

  inline const FermionFieldType & getLowMode(const int il) const{ return *w[il]; }
  inline const FermionFieldType & getHighMode(const int ih) const{ return *w[ih+nl]; }
   
  class View: public TimePackedIndexDilution{
    typename CPSfieldArray<FermionFieldType>::View aw;
  public:
    typedef mf_Policies Policies;
    typedef typename CPSfieldArray<FermionFieldType>::FieldView FermionFieldView;
    typedef typename FermionFieldType::FieldSiteType FieldSiteType;
    typedef TimePackedIndexDilution DilutionType;

    View(ViewMode mode,
	 const CPSfieldArray<FermionFieldType> &win,
	 const TimePackedIndexDilution &d): aw(win.view(mode)), TimePackedIndexDilution(d){}

    View(ViewMode mode,
	 const CPSfieldArray<FermionFieldType> &win,
	 const TimePackedIndexDilution &d, const std::vector<bool> &modes_used):     aw(win.view(mode, modes_used)), TimePackedIndexDilution(d){}

    
    accelerator_inline FermionFieldView & getWl(const int i) const{ return aw[i]; }
    accelerator_inline FermionFieldView & getWh(const int i) const{ return aw[i+nl]; }
    accelerator_inline FermionFieldView & getLowMode(const int i) const{ return aw[i]; }
    accelerator_inline FermionFieldView & getHighMode(const int i) const{ return aw[i+nl]; }
  
    //The timeslice dilutions is packed so we must treat them differently
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
	if(tblock != mode_tblock) return zero;
	int idx = this->indexMap(mode_hit,mode_spin_color,mode_flavor); //packed index
	int site = getWh(idx).threeToFour(x3d,t);

	//As a matrix, the mode gives the column index
	return *(getWh(idx).site_ptr(site,flavor) + spin_color);
      }
    }

    //Get a particular site/spin/color element of a given *native* (packed) mode 
    accelerator_inline const FieldSiteType & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
      return *(aw[i].site_ptr(site,flavor)+spin_color);
    }

    //For the mode i, get the base pointer and size (in units of FieldSiteType) of the field. Includes all flavors, contiguity guaranteed
    inline void getModeData(FieldSiteType const* &ptr, size_t &size, const int i) const{
      ptr = aw[i].ptr(); size = aw[i].size();
    }

    //For the mode i, get the base pointers for the provided timeslice (local) for each flavor along with the size (in units of FieldSiteType) of the timeslice field. Contiguity guaranteed
    inline void getModeTimesliceData(FieldSiteType const* &ptr_f0, FieldSiteType const* &ptr_f1, size_t &size, const int i, const int t) const{
      assert(aw[i].dimpol_flavor_timeslice_contiguous());
      size = aw[i].dimpol_time_stride() * aw[i].siteSize();      
      ptr_f0 = aw[i].ptr() + t*size;
      ptr_f1 = ptr_f0 + aw[i].flav_offset();
    }

    void free(){ aw.free(); }
  };

  View view(ViewMode mode) const{ return View(mode, w, *this); }
  //Open a view only to some subset of modes. Undefined behavior if you access one that you are not supposed to!
  View view(ViewMode mode, const std::vector<bool> &modes_used) const{ return View(mode, w, *this, modes_used); }

  void enqueuePrefetch(ViewMode mode, const std::vector<bool> &modes_used) const{
    for(int i=0;i<w.size();i++) if(modes_used[i])  w[i]->enqueuePrefetch(mode);
  }   
  static inline void startPrefetches(){ Policies::AllocPolicy::startPrefetches(); }
  static inline void waitPrefetches(){ Policies::AllocPolicy::waitPrefetches(); }
  
  void importWl(const FermionFieldType &wlin, const int i){
    *w[i] = wlin;
  }
  void importWh(const FermionFieldType &whin, const int i){
    *w[i+nl] = whin;
  }

  //Get the diluted source with StandardIndex high-mode index dil_id.
  //Here dil_id is the combined spin-color/flavor/hit/tblock index
  template<typename TargetFermionFieldType>
  void getDilutedSource(TargetFermionFieldType &into, const int dil_id) const;

  //Set each float to a uniform random number in the specified range.
  //WARNING: Uses only the current RNG in LRG, and does not change this based on site. This is therefore only useful for testing*
  void testRandom(const Float &hi = 0.5, const Float &lo = -0.5){
    for(int i=0;i<w.size();i++) w[i]->testRandom(hi,lo);
  }

  //Set all fields to zero
  void zero(){
    for(int i=0;i<w.size();i++) w[i]->zero();
  }

  void writeParallelWithGrid(const std::string &file_stub) const;
  void readParallelWithGrid(const std::string &file_stub);

  inline void free_mem(){
    w.free();
  }
};

#include "implementation/Wtimepacked_impl.tcc"

CPS_END_NAMESPACE
