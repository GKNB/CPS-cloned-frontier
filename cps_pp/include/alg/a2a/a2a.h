#ifndef CK_A2A
#define CK_A2A

#include<util/lattice.h>

#ifdef USE_GRID
#include <util/lattice/fgrid.h>
#endif

#ifdef USE_BFM
#include<util/lattice/bfm_mixed_solver.h>
#include <util/lattice/bfm_evo.h>
#include <alg/eigen/Krylov_5d.h>
#endif

#include<alg/a2a/CPSfield.h>
#include<alg/a2a/CPSfield_utils.h>
#include<alg/a2a/scfvectorptr.h>
#include<alg/a2a/utils.h>

#include<alg/a2a/a2a_params.h>
#include<alg/a2a/a2a_dilutions.h>
#include<alg/a2a/field_operation.h>
#include<alg/a2a/a2a_policies.h>
#include<alg/a2a/evec_interface.h>
#include<alg/a2a/grid_cgne_m_high.h>
CPS_START_NAMESPACE 

template< typename mf_Policies>
class A2AvectorVfftw;

template< typename mf_Policies>
class A2AvectorV: public StandardIndexDilution{
public:
  typedef mf_Policies Policies;
  typedef typename Policies::FermionFieldType FermionFieldType;
  typedef typename FermionFieldType::FieldSiteType FieldSiteType;
  typedef typename FermionFieldType::InputParamType FieldInputParamType;
private:
  std::vector<FermionFieldType> v;
  const std::string cname;

public:
  typedef StandardIndexDilution DilutionType;

  A2AvectorV(const A2AArg &_args): StandardIndexDilution(_args), cname("A2AvectorV"){
    v.resize(nv,FermionFieldType());

    //When computing V and W we can re-use previous V solutions as guesses. Set default to zero here so we have zero guess when no 
    //previously computed solutions
    for(int i=0;i<nv;i++) v[i].zero(); 
  }
  A2AvectorV(const A2AArg &_args, const FieldInputParamType &field_setup_params): StandardIndexDilution(_args), cname("A2AvectorV"){
    v.resize(nv,FermionFieldType(field_setup_params));
    for(int i=0;i<nv;i++) v[i].zero(); 
  }

  
  inline FermionFieldType & getMode(const int i){ return v[i]; }
  inline const FermionFieldType & getMode(const int i) const{ return v[i]; }
  

  //Get a mode from the low mode part
  FermionFieldType & getVl(const int il){ return v[il]; }

  //Get a mode from the high-mode part
  FermionFieldType & getVh(const int ih){ return v[nl+ih]; }

  //Generate the Fourier transformed V fields. This includes gauge fixing and applying the momentum twist
  void computeVfftw(A2AvectorVfftw<Policies> &into);

  //Get a particular site/spin/color element of a given mode 
  const FieldSiteType & elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
    int x4d = v[mode].threeToFour(x3d,t);
    return  *(v[mode].site_ptr(x4d,flavor) + spin_color);
  }
  //Get a particular site/spin/color element of a given *native* (packed) mode. For V this does the same as the above
  inline const FieldSiteType & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
    return *(v[i].site_ptr(site,flavor)+spin_color);
  }

  void importVl(const FermionFieldType &vv, const int il){
    v[il] = vv;
  }
  void importVh(const FermionFieldType &vv, const int ih){
    v[nl+ih] = vv;
  }

};


template< typename mf_Policies>
class A2AvectorVfftw: public StandardIndexDilution{  
public:
  typedef mf_Policies Policies;
  typedef typename Policies::FermionFieldType FermionFieldType;
  typedef typename FermionFieldType::FieldSiteType FieldSiteType;
  typedef typename FermionFieldType::InputParamType FieldInputParamType;
private:
  std::vector<FermionFieldType> v;
  const std::string cname;

public:
  typedef StandardIndexDilution DilutionType;

  A2AvectorVfftw(const A2AArg &_args): StandardIndexDilution(_args), cname("A2AvectorVfftw"){
    v.resize(nv,FermionFieldType());
  }
  A2AvectorVfftw(const A2AArg &_args, const FieldInputParamType &field_setup_params): StandardIndexDilution(_args), cname("A2AvectorVfftw"){
    v.resize(nv,FermionFieldType(field_setup_params));
  }
  
  inline const FermionFieldType & getMode(const int i) const{ return v[i]; }

  //Set this object to be the threaded fast Fourier transform of the input field
  //Can optionally supply an object that performs a transformation on each mode prior to the FFT. 
  //We can use this to avoid intermediate storage for the gauge fixing and momentum phase application steps
  void fft(const A2AvectorV<Policies> &from, fieldOperation<FieldSiteType>* mode_preop = NULL);

  //For each mode, gauge fix, apply the momentum factor, then perform the FFT and store the result in this object
  void gaugeFixTwistFFT(const A2AvectorV<Policies> &from, const int _p[3], Lattice &_lat){
    gaugeFixAndTwist<FieldSiteType> op(_p,_lat); fft(from, &op);
  }

  const FieldSiteType & elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
    int site = v[mode].threeToFour(x3d,t);
    return *(v[mode].site_ptr(site,flavor) + spin_color);
  }
  //Get a particular site/spin/color element of a given 'native' (packed) mode. For V this does the same thing as the above
  inline const FieldSiteType & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
    return *(v[i].site_ptr(site,flavor)+spin_color);
  }

  //i_high_unmapped is the index i unmapped to its high mode sub-indices (if it is a high mode of course!)
  inline SCFvectorPtr<FieldSiteType> getFlavorDilutedVect(const int i, const modeIndexSet &i_high_unmapped, const int site) const{
    const int flav_offset = v[0].flav_offset();
    const int site_offset = v[0].site_offset(site);
    return getFlavorDilutedVect(i,i_high_unmapped,site_offset,flav_offset);
  }
  inline SCFvectorPtr<FieldSiteType> getFlavorDilutedVect(const int i, const modeIndexSet &i_high_unmapped, const int site_offset, const int flav_offset) const{
    const FermionFieldType &field = getMode(i);
    FieldSiteType const* f0_ptr = field.ptr() + site_offset;
    return SCFvectorPtr<FieldSiteType>(f0_ptr, f0_ptr+flav_offset);
  }

  inline SCFvectorPtr<FieldSiteType> getFlavorDilutedVect2(const int i, const modeIndexSet &i_high_unmapped, const int p3d, const int t) const{
    const FermionFieldType &field = getMode(i);
    const int x4d = field.threeToFour(p3d,t);
    return SCFvectorPtr<FieldSiteType>(field.site_ptr(x4d,0),field.site_ptr(x4d,1));
  }

  const CPSfermion4D<FieldSiteType> & getMode(const int i, const modeIndexSet &i_high_unmapped) const{ return getMode(i); }

  //Replace this vector with the average of this another vector, 'with'
  void average(const A2AvectorVfftw<Policies> &with, const bool &parallel = true){
    if( !paramsEqual(with) ) ERR.General("A2AvectorVfftw","average","Second field must share the same underlying parameters\n");
    for(int i=0;i<nv;i++) v[i].average(with.v[i]);
  }
  //Set each float to a uniform random number in the specified range.
  //WARNING: Uses only the current RNG in LRG, and does not change this based on site. This is therefore only useful for testing*
  void testRandom(const Float &hi = 0.5, const Float &lo = -0.5){
    for(int i=0;i<nv;i++) v[i].testRandom(hi,lo);
  }

  template<typename extPolicies>
  void importFields(const A2AvectorVfftw<extPolicies> &r){
    if( !paramsEqual(r) ) ERR.General("A2AvectorVfftw","importFields","External field-vector must share the same underlying parameters\n");
    for(int i=0;i<nv;i++) v[i].importField(r.getMode(i));
  }  

};


template< typename mf_Policies>
class A2AvectorW: public FullyPackedIndexDilution{
public:
  typedef mf_Policies Policies;
  typedef typename Policies::FermionFieldType FermionFieldType;
  typedef typename Policies::ComplexFieldType ComplexFieldType;
  typedef typename my_enable_if< _equal<typename FermionFieldType::FieldSiteType, typename ComplexFieldType::FieldSiteType>::value,  typename FermionFieldType::FieldSiteType>::type FieldSiteType;
  typedef typename my_enable_if< _equal<typename FermionFieldType::InputParamType, typename ComplexFieldType::InputParamType>::value,  typename FermionFieldType::InputParamType>::type FieldInputParamType;
private:
  std::vector<FermionFieldType> wl; //The low mode part of the W field, comprised of nl fermion fields
  std::vector<ComplexFieldType> wh; //The high mode random part of the W field, comprised of nhits complex scalar fields. Note: the dilution is performed later

  const std::string cname;

  //Generate the wh field. We store in a compact notation that knows nothing about any dilution we apply when generating V from this
  //For reproducibility we want to generate the wh field in the same order that Daiqian did originally. Here nhit random numbers are generated for each site/flavor
  void setWhRandom(const RandomType &type);
  
public:
  typedef FullyPackedIndexDilution DilutionType;

  A2AvectorW(const A2AArg &_args): FullyPackedIndexDilution(_args), cname("A2AvectorW"){
    wl.resize(nl,FermionFieldType());
    wh.resize(nhits, ComplexFieldType()); 
  }
  A2AvectorW(const A2AArg &_args, const FieldInputParamType &field_input_params): FullyPackedIndexDilution(_args), cname("A2AvectorW"){
    wl.resize(nl,FermionFieldType(field_input_params));
    wh.resize(nhits, ComplexFieldType(field_input_params)); 
  }
  
  const FermionFieldType & getWl(const int i) const{ return wl[i]; }
  const ComplexFieldType & getWh(const int hit) const{ return wh[hit]; }

  void importWl(const FermionFieldType &wlin, const int i){
    wl[i] = wlin;
  }
  void importWh(const ComplexFieldType &whin, const int hit){
    wh[hit] = whin;
  }

#ifdef USE_GRID
  //Generic Grid VW compute interface that can use either Grid or BFM-computed eigenvectors

  //Compute the low mode part of the W and V vectors.
  void computeVWlow(A2AvectorV<Policies> &V, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass);

  //Compute the high mode parts of V and W. 
  void computeVWhigh(A2AvectorV<Policies> &V, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass, const Float residual, const int max_iter);
#endif

#if defined(USE_BFM_LANCZOS)
  //In the Lanczos class you can choose to store the vectors in single precision (despite the overall precision, which is fixed to double here)
  //Set 'singleprec_evecs' if this has been done
  void computeVWlow(A2AvectorV<Policies> &V, Lattice &lat, BFM_Krylov::Lanczos_5d<double> &eig, bfm_evo<double> &dwf, bool singleprec_evecs);

  //singleprec_evecs specifies whether the input eigenvectors are stored in single precision
  //You can optionally pass a single precision bfm instance, which if given will cause the underlying CG to be performed in mixed precision.
  //WARNING: if using the mixed precision solve, the eigenvectors *MUST* be in single precision (there is a runtime check)
  void computeVWhigh(A2AvectorV<Policies> &V, BFM_Krylov::Lanczos_5d<double> &eig, bool singleprec_evecs, Lattice &lat, bfm_evo<double> &dwf_d, bfm_evo<float> *dwf_fp = NULL);

  void computeVW(A2AvectorV<Policies> &V, Lattice &lat, BFM_Krylov::Lanczos_5d<double> &eig, bool singleprec_evecs, bfm_evo<double> &dwf_d, bfm_evo<float> *dwf_fp = NULL){
    computeVWlow(V,lat,eig,dwf_d,singleprec_evecs);
    computeVWhigh(V,eig,singleprec_evecs,lat,dwf_d,dwf_fp);
  }
#endif


#if defined(USE_GRID_LANCZOS)
  void computeVWlow(A2AvectorV<Policies> &V, Lattice &lat, const std::vector<typename Policies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, const double mass);

  void computeVWhigh(A2AvectorV<Policies> &V, Lattice &lat, const std::vector<typename Policies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, const double mass, const Float residual, const int max_iter);

  void computeVW(A2AvectorV<Policies> &V, Lattice &lat, const std::vector<typename Policies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, const double mass, const Float high_mode_residual, const int high_mode_max_iter){
    computeVWlow(V,lat,evec,eval,mass);
    computeVWhigh(V,lat,evec,eval,mass,high_mode_residual,high_mode_max_iter);
  }

  //Single-precision variants (use mixed_CG internally)
  void computeVWlow(A2AvectorV<Policies> &V, Lattice &lat, const std::vector<typename Policies::GridFermionFieldF> &evec, const std::vector<Grid::RealD> &eval, const double mass);

  void computeVWhigh(A2AvectorV<Policies> &V, Lattice &lat, const std::vector<typename Policies::GridFermionFieldF> &evec, const std::vector<Grid::RealD> &eval, const double mass, const Float residual, const int max_iter);

  void computeVW(A2AvectorV<Policies> &V, Lattice &lat, const std::vector<typename Policies::GridFermionFieldF> &evec, const std::vector<Grid::RealD> &eval, const double mass, const Float high_mode_residual, const int high_mode_max_iter){
    computeVWlow(V,lat,evec,eval,mass);
    computeVWhigh(V,lat,evec,eval,mass,high_mode_residual,high_mode_max_iter);
  }
#endif

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
  const FieldSiteType & elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
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
  inline const FieldSiteType & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
    return i < nl ? 
      *(wl[i].site_ptr(site,flavor)+spin_color) :
      *(wh[i-nl].site_ptr(site,flavor)); //we use different random fields for each time and flavor, although we didn't have to
  }



};


template< typename mf_Policies>
class A2AvectorWfftw: public TimeFlavorPackedIndexDilution{
public:
  typedef mf_Policies Policies;
  typedef typename Policies::FermionFieldType FermionFieldType;
  typedef typename Policies::ComplexFieldType ComplexFieldType;
  typedef typename FermionFieldType::FieldSiteType FieldSiteType;
  typedef typename my_enable_if< _equal<typename FermionFieldType::InputParamType, typename ComplexFieldType::InputParamType>::value,  typename FermionFieldType::InputParamType>::type FieldInputParamType;
private:

  std::vector<FermionFieldType> wl;
  std::vector<FermionFieldType> wh; //these have been diluted in spin/color but not the other indices, hence there are nhit * 12 fields here (spin/color index changes fastest in mapping)

  const std::string cname;

public:
  typedef TimeFlavorPackedIndexDilution DilutionType;

  A2AvectorWfftw(const A2AArg &_args): TimeFlavorPackedIndexDilution(_args), cname("A2AvectorWfftw"){
    wl.resize(nl,FermionFieldType());
    wh.resize(12*nhits, FermionFieldType()); 
  }
  A2AvectorWfftw(const A2AArg &_args, const FieldInputParamType &field_setup_params): TimeFlavorPackedIndexDilution(_args), cname("A2AvectorWfftw"){
    wl.resize(nl,FermionFieldType(field_setup_params));
    wh.resize(12*nhits, FermionFieldType(field_setup_params)); 
  }

  
  inline const FermionFieldType & getWl(const int i) const{ return wl[i]; }
  inline const FermionFieldType & getWh(const int hit, const int spin_color) const{ return wh[spin_color + 12*hit]; }

  inline const FermionFieldType & getMode(const int i) const{ return i < nl ? wl[i] : wh[i-nl]; }

  //Set this object to be the threaded fast Fourier transform of the input field
  //Can optionally supply an object that performs a transformation on each mode prior to the FFT. 
  //We can use this to avoid intermediate storage for the gauge fixing and momentum phase application steps
  void fft(const A2AvectorW<Policies> &from, fieldOperation<FieldSiteType>* mode_preop = NULL);

  //For each mode, gauge fix, apply the momentum factor, then perform the FFT and store the result in this object
  void gaugeFixTwistFFT(const A2AvectorW<Policies> &from, const int _p[3], Lattice &_lat){
    gaugeFixAndTwist<FieldSiteType> op(_p,_lat); fft(from, &op);
  }

  //The flavor and timeslice dilutions are still packed so we must treat them differently
  //Mode is a full 'StandardIndex', (unpacked mode index)
  const FieldSiteType & elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
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
  inline const FieldSiteType & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
    return i < nl ? 
      *(wl[i].site_ptr(site,flavor)+spin_color) :
      *(wh[i-nl].site_ptr(site,flavor)+spin_color); //spin_color index diluted out.
  }



  //Replace this vector with the average of this another vector, 'with'
  void average(const A2AvectorWfftw<Policies> &with, const bool &parallel = true){
    if( !paramsEqual(with) ) ERR.General("A2AvectorWfftw","average","Second field must share the same underlying parameters\n");
    for(int i=0;i<nl;i++) wl[i].average(with.wl[i]);
    for(int i=0;i<12*nhits;i++) wh[i].average(with.wh[i]);
  }

  //Set each float to a uniform random number in the specified range.
  //WARNING: Uses only the current RNG in LRG, and does not change this based on site. This is therefore only useful for testing*
  void testRandom(const Float &hi = 0.5, const Float &lo = -0.5){
    for(int i=0;i<nl;i++) wl[i].testRandom(hi,lo);
    for(int i=0;i<12*nhits;i++) wh[i].testRandom(hi,lo);
  }

  //BELOW are for use by the meson field
  //Meson field W-type indices are described in terms of the timePacked dilution index , where flavor has been diluted out in the process of computing the meson fields
  //This method performs the flavor dilution 'in-place' (i.e. without actually unpacking into another object). 
  //'site' is a local canonical-ordered, packed four-vector
  //i_high_unmapped is the index i unmapped to its high mode sub-indices (if it is a high mode of course!)

  inline SCFvectorPtr<FieldSiteType> getFlavorDilutedVect(const int i, const modeIndexSet &i_high_unmapped, const int site) const{
    const int site_offset = i >= nl ? wh[0].site_offset(site) : wl[0].site_offset(site);
    const int flav_offset = i >= nl ? wh[0].flav_offset() : wl[0].flav_offset();
    return getFlavorDilutedVect(i,i_high_unmapped,site_offset,flav_offset);
  }

  inline SCFvectorPtr<FieldSiteType> getFlavorDilutedVect(const int i, const modeIndexSet &i_high_unmapped, const int site_offset, const int flav_offset) const{
    const FermionFieldType &field = i >= nl ? getWh(i_high_unmapped.hit, i_high_unmapped.spin_color): getWl(i);
    const static FieldSiteType zerosc[12] = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };

    bool zero_hint[2] = {false,false};
    if(i >= nl) zero_hint[ !i_high_unmapped.flavor ] = true;

    FieldSiteType const* f0_ptr = field.ptr() + site_offset;
    FieldSiteType const* lp[2] = { zero_hint[0] ? &zerosc[0] : f0_ptr,
				 zero_hint[1] ? &zerosc[0] : f0_ptr + flav_offset };

    return SCFvectorPtr<FieldSiteType>(lp[0],lp[1],zero_hint[0],zero_hint[1]);
  }

  inline SCFvectorPtr<FieldSiteType> getFlavorDilutedVect2(const int i, const modeIndexSet &i_high_unmapped, const int p3d, const int t) const{
    const FermionFieldType &field = i >= nl ? getWh(i_high_unmapped.hit, i_high_unmapped.spin_color): getWl(i);
    const static FieldSiteType zerosc[12] = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };

    bool zero_hint[2] = {false,false};
    if(i >= nl) zero_hint[ !i_high_unmapped.flavor ] = true;

    const int x4d = field.threeToFour(p3d,t);
    FieldSiteType const* lp[2] = { zero_hint[0] ? &zerosc[0] : field.site_ptr(x4d,0),
				   zero_hint[1] ? &zerosc[0] : field.site_ptr(x4d,1) };

    return SCFvectorPtr<FieldSiteType>(lp[0],lp[1],zero_hint[0],zero_hint[1]);
  }


  //This version allows for the possibility of a different high mode mapping for the index i by passing the unmapped indices
  const FermionFieldType & getMode(const int i, const modeIndexSet &i_high_unmapped) const{ return i >= nl ? getWh(i_high_unmapped.hit, i_high_unmapped.spin_color): getWl(i); }

  template<typename extPolicies>
  void importFields(const A2AvectorWfftw<extPolicies> &r){
    if( !paramsEqual(r) ) ERR.General("A2AvectorWfftw","importFields","External field-vector must share the same underlying parameters\n");
    for(int i=0;i<nl;i++) wl[i].importField(r.getWl(i));
    for(int i=0;i<12*nhits;i++) wh[i].importField(r.getWh(i/12,i%12));
  }  

};

//Generate uniform random V and W vectors for testing
template<typename Policies>
void randomizeVW(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W);




#include <alg/a2a/a2a_impl.h>

#ifdef USE_GRID
#include<alg/a2a/evec_interface_impl.h>
#endif

//Can do Lanczos in BFM or Grid, and A2A in BFM or Grid. I have a BFM Lanczos -> Grid interface

#if defined(USE_BFM_A2A)
# warning "Using BFM A2A"

# ifndef USE_BFM
#  error "Require BFM for USE_BFM_A2A"
# endif

# ifdef USE_GRID_LANCZOS
#  error "No Grid Lanczos -> BFM A2A interface implemented"
# endif

# include <alg/a2a/a2a_impl_vwbfm.h>

#elif defined(USE_GRID_A2A)
# warning "Using Grid A2A"

# ifndef USE_GRID
#  error "Require Grid for USE_GRID_A2A"
# endif

# if defined(USE_BFM_LANCZOS) && !defined(USE_BFM)
#  error "BFM Lanczos -> Grid A2A interface requires BFM!"
# endif

# include <alg/a2a/a2a_impl_vwgrid.h>

#else

# error "Need either BFM or Grid to compute A2A vectors"

#endif


CPS_END_NAMESPACE

#endif
