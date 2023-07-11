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

#ifdef USE_GRID

//Convert a V field to Grid format
template<typename GridFieldType, typename A2Apolicies>
void convertToGrid(std::vector<GridFieldType> &V_out, const A2AvectorV<A2Apolicies> &V_in, Grid::GridBase *grid);

#endif

#include "implementation/V_impl.tcc"

CPS_END_NAMESPACE
