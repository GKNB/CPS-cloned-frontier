#ifndef CK_A2A
#define CK_A2A

#include<util/lattice.h>
#include<util/lattice/bfm_mixed_solver.h>
#include <util/lattice/bfm_evo.h>
#ifdef USE_GRID
#include <util/lattice/fgrid.h>
#define FGRID FgridGparityMobius
#define GFGRID GnoneFgridGparityMobius
#define DIRAC Grid::QCD::GparityMobiusFermionD
#define LATTICE_FERMION DIRAC ::FermionField
#define FGRID_CLASS_NAME F_CLASS_GRID_GPARITY_MOBIUS
#define GRID_GPARITY
#endif
#include <alg/eigen/Krylov_5d.h>

#include<alg/a2a/CPSfield.h>
#include<alg/a2a/scfvectorptr.h>
#include<alg/a2a/utils.h>

#include<alg/a2a/a2a_params.h>
#include<alg/a2a/a2a_dilutions.h>
#include<alg/a2a/field_operation.h>

CPS_START_NAMESPACE 

template< typename mf_Float>
class A2AvectorVfftw;

template< typename mf_Float>
class A2AvectorV: public StandardIndexDilution{
  std::vector<CPSfermion4D<mf_Float> > v;
  const std::string cname;

public:
  typedef StandardIndexDilution DilutionType;

  A2AvectorV(const A2AArg &_args): StandardIndexDilution(_args), cname("A2AvectorV"){
    v.resize(nv,CPSfermion4D<mf_Float>());

    //When computing V and W we can re-use previous V solutions as guesses. Set default to zero here so we have zero guess when no 
    //previously computed solutions
    for(int i=0;i<nv;i++) v[i].zero(); 
  }

  inline CPSfermion4D<mf_Float> & getMode(const int i){ return v[i]; }
  inline const CPSfermion4D<mf_Float> & getMode(const int i) const{ return v[i]; }
  

  //Get a mode from the low mode part
  CPSfermion4D<mf_Float> & getVl(const int il){ return v[il]; }

  //Get a mode from the high-mode part
  CPSfermion4D<mf_Float> & getVh(const int ih){ return v[nl+ih]; }

  //Generate the Fourier transformed V fields. This includes gauge fixing and applying the momentum twist
  void computeVfftw(A2AvectorVfftw<mf_Float> &into);

  //Get a particular site/spin/color element of a given mode 
  const std::complex<mf_Float> & elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
    int x4d = x3d + GJP.VolNodeSites()/GJP.TnodeSites()*t;
    mf_Float const* p = v[mode].site_ptr(x4d,flavor) + 2*spin_color;
    return reinterpret_cast<std::complex<mf_Float> const&>(*p);
  }
  //Get a particular site/spin/color element of a given *native* (packed) mode. For V this does the same as the above
  inline const std::complex<mf_Float> & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
    return reinterpret_cast<std::complex<mf_Float> const& >(*(v[i].site_ptr(site,flavor)+2*spin_color));
  }

  void importVl(const CPSfermion4D<mf_Float> &vv, const int il){
    v[il] = vv;
  }
  void importVh(const CPSfermion4D<mf_Float> &vv, const int ih){
    v[nl+ih] = vv;
  }

};


template< typename mf_Float>
class A2AvectorVfftw: public StandardIndexDilution{
  std::vector<CPSfermion4D<mf_Float> > v;
  const std::string cname;

public:
  typedef StandardIndexDilution DilutionType;

  A2AvectorVfftw(const A2AArg &_args): StandardIndexDilution(_args), cname("A2AvectorVfftw"){
    v.resize(nv,CPSfermion4D<mf_Float>());
  }
  inline const CPSfermion4D<mf_Float> & getMode(const int i) const{ return v[i]; }

  //Set this object to be the threaded fast Fourier transform of the input field
  //Can optionally supply an object that performs a transformation on each mode prior to the FFT. 
  //We can use this to avoid intermediate storage for the gauge fixing and momentum phase application steps
  void fft(const A2AvectorV<mf_Float> &from, fieldOperation<mf_Float>* mode_preop = NULL);

  //For each mode, gauge fix, apply the momentum factor, then perform the FFT and store the result in this object
  void gaugeFixTwistFFT(const A2AvectorV<mf_Float> &from, const int _p[3], Lattice &_lat){
    gaugeFixAndTwist<mf_Float> op(_p,_lat); fft(from, &op);
  }

  const std::complex<mf_Float> & elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
    int site = x3d + GJP.VolNodeSites()/GJP.TnodeSites()*t;
    mf_Float const* p = v[mode].site_ptr(site,flavor) + 2*spin_color;
    return reinterpret_cast<std::complex<mf_Float> const&>(*p);
  }
  //Get a particular site/spin/color element of a given 'native' (packed) mode. For V this does the same thing as the above
  inline const std::complex<mf_Float> & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
    return reinterpret_cast<std::complex<mf_Float> const& >(*(v[i].site_ptr(site,flavor)+2*spin_color));
  }

  //i_high_unmapped is the index i unmapped to its high mode sub-indices (if it is a high mode of course!)
  inline SCFvectorPtr<mf_Float> getFlavorDilutedVect(const int i, const modeIndexSet &i_high_unmapped, const int site) const{
    const int flav_offset = v[0].flav_offset();
    const int site_offset = v[0].site_offset(site);
    return getFlavorDilutedVect(i,i_high_unmapped,site_offset,flav_offset);
  }
  inline SCFvectorPtr<mf_Float> getFlavorDilutedVect(const int i, const modeIndexSet &i_high_unmapped, const int site_offset, const int flav_offset) const{
    const CPSfermion4D<mf_Float> &field = getMode(i);
    mf_Float const* f0_ptr = field.ptr() + site_offset;
    return SCFvectorPtr<mf_Float>(f0_ptr, f0_ptr+flav_offset);  //( field.site_ptr(site,0) , field.site_ptr(site,1) );
  }

  inline SCFvectorPtr<mf_Float> getFlavorDilutedVect2(const int i, const modeIndexSet &i_high_unmapped, const int p3d, const int t) const{
    const CPSfermion4D<mf_Float> &field = getMode(i);
    const int x4d = field.threeToFour(p3d,t);
    mf_Float const* f0_ptr = field.site_ptr(x4d,0);
    mf_Float const* f1_ptr = field.site_ptr(x4d,1);
    return SCFvectorPtr<mf_Float>(f0_ptr, f1_ptr);
  }


  const CPSfermion4D<mf_Float> & getMode(const int i, const modeIndexSet &i_high_unmapped) const{ return getMode(i); }

  //Replace this vector with the average of this another vector, 'with'
  void average(const A2AvectorVfftw<mf_Float> &with, const bool &parallel = true){
    if( !paramsEqual(with) ) ERR.General("A2AvectorVfftw","average","Second field must share the same underlying parameters\n");
    for(int i=0;i<nv;i++) v[i].average(with.v[i]);
  }
  //Set each float to a uniform random number in the specified range.
  //WARNING: Uses only the current RNG in LRG, and does not change this based on site. This is therefore only useful for testing*
  void testRandom(const Float &hi = 0.5, const Float &lo = -0.5){
    for(int i=0;i<nv;i++) v[i].testRandom(hi,lo);
  }
};



//Unified interface for obtaining evecs and evals from either Grid- or BFM-computed Lanczos
class EvecInterface{
 public:
  //Get an eigenvector and eigenvalue
  virtual Float getEvec(LATTICE_FERMION &into, const int idx) = 0;
  virtual int nEvecs() const = 0;
};


template< typename mf_Float>
class A2AvectorW: public FullyPackedIndexDilution{
  std::vector<CPSfermion4D<mf_Float> > wl; //The low mode part of the W field, comprised of nl fermion fields
  std::vector<CPScomplex4D<mf_Float> > wh; //The high mode random part of the W field, comprised of nhits complex scalar fields. Note: the dilution is performed later

  const std::string cname;

  //Generate the wh field. We store in a compact notation that knows nothing about any dilution we apply when generating V from this
  //For reproducibility we want to generate the wh field in the same order that Daiqian did originally. Here nhit random numbers are generated for each site/flavor
  void setWhRandom(const RandomType &type);
  
public:
  typedef FullyPackedIndexDilution DilutionType;

  A2AvectorW(const A2AArg &_args): FullyPackedIndexDilution(_args), cname("A2AvectorW"){
    wl.resize(nl,CPSfermion4D<mf_Float>());
    wh.resize(nhits, CPScomplex4D<mf_Float>()); 
  }
  const CPSfermion4D<mf_Float> & getWl(const int i) const{ return wl[i]; }
  const CPScomplex4D<mf_Float> & getWh(const int hit) const{ return wh[hit]; }

  void importWl(const CPSfermion4D<mf_Float> &wlin, const int i){
    wl[i] = wlin;
  }
  void importWh(const CPScomplex4D<mf_Float> &whin, const int hit){
    wh[hit] = whin;
  }

  //Compute the low mode part of the W and V vectors.
  void computeVWlow(A2AvectorV<mf_Float> &V, Lattice &lat, EvecInterface &evecs, const Float mass);

  //Compute the high mode parts of V and W. 
  void computeVWhigh(A2AvectorV<mf_Float> &V, Lattice &lat, EvecInterface &evecs, const Float mass, const Float residual, const int max_iter);

#if defined(USE_BFM_LANCZOS)
  //In the Lanczos class you can choose to store the vectors in single precision (despite the overall precision, which is fixed to double here)
  //Set 'singleprec_evecs' if this has been done
  void computeVWlow(A2AvectorV<mf_Float> &V, Lattice &lat, BFM_Krylov::Lanczos_5d<double> &eig, bfm_evo<double> &dwf, bool singleprec_evecs);

  //singleprec_evecs specifies whether the input eigenvectors are stored in single precision
  //You can optionally pass a single precision bfm instance, which if given will cause the underlying CG to be performed in mixed precision.
  //WARNING: if using the mixed precision solve, the eigenvectors *MUST* be in single precision (there is a runtime check)
  void computeVWhigh(A2AvectorV<mf_Float> &V, BFM_Krylov::Lanczos_5d<double> &eig, bool singleprec_evecs, Lattice &lat, bfm_evo<double> &dwf_d, bfm_evo<float> *dwf_fp = NULL);

  void computeVW(A2AvectorV<mf_Float> &V, Lattice &lat, BFM_Krylov::Lanczos_5d<double> &eig, bool singleprec_evecs, bfm_evo<double> &dwf_d, bfm_evo<float> *dwf_fp = NULL){
    computeVWlow(V,lat,eig,dwf_d,singleprec_evecs);
    computeVWhigh(V,eig,singleprec_evecs,lat,dwf_d,dwf_fp);
  }
#endif


#if defined(USE_GRID_LANCZOS)
  void computeVWlow(A2AvectorV<mf_Float> &V, Lattice &lat, const std::vector<LATTICE_FERMION> &evec, const std::vector<Grid::RealD> &eval, const double mass);

  void computeVWhigh(A2AvectorV<mf_Float> &V, Lattice &lat, const std::vector<LATTICE_FERMION> &evec, const std::vector<Grid::RealD> &eval, const double mass, const Float residual, const int max_iter);

  void computeVW(A2AvectorV<mf_Float> &V, Lattice &lat, const std::vector<LATTICE_FERMION> &evec, const std::vector<Grid::RealD> &eval, const double mass, const Float high_mode_residual, const int high_mode_max_iter){
    computeVWlow(V,lat,evec,eval,mass);
    computeVWhigh(V,lat,evec,eval,mass,high_mode_residual,high_mode_max_iter);
  }
#endif

  //Get the diluted source with StandardIndex high-mode index dil_id.
  //We use the same set of random numbers for each spin and dilution as we do not need to rely on stochastic cancellation to separate them
  //For legacy reasons we use different random numbers for the two G-parity flavors, although this is not strictly necessary
  //Here dil_id is the combined spin-color/flavor/hit/tblock index
  template<typename TargetFloat>
  void getDilutedSource(CPSfermion4D<TargetFloat> &into, const int dil_id) const;

  //When gauge fixing prior to taking the FFT it is necessary to uncompact the wh field in the spin-color index, as these indices are acted upon by the gauge fixing
  //(I suppose technically only the color indices need uncompacting; this might be considered as a future improvement)
  void getSpinColorDilutedSource(CPSfermion4D<mf_Float> &into, const int hit, const int sc_id) const;

  //The spincolor, flavor and timeslice dilutions are packed so we must treat them differently
  //Mode is a full 'StandardIndex', (unpacked mode index)
  const std::complex<mf_Float> & elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
    static std::complex<mf_Float> zero(0.0,0.0);
    int site = x3d + GJP.VolNodeSites()/GJP.Tnodes()*t;
    if(mode < nl){
      mf_Float const* p = getWl(mode).site_ptr(site,flavor) + 2*spin_color;
      return reinterpret_cast<std::complex<mf_Float> const&>(*p);
    }else{
      int mode_hit, mode_tblock, mode_spin_color,mode_flavor;
      const StandardIndexDilution &dilfull = static_cast<StandardIndexDilution const&>(*this);
      dilfull.indexUnmap(mode-nl,mode_hit,mode_tblock,mode_spin_color,mode_flavor);
      //flavor and time block indices match those of the mode, the result is zero
      int tblock = (t+GJP.TnodeSites()*GJP.TnodeCoor())/args.src_width;
      if(spin_color != mode_spin_color || flavor != mode_flavor || tblock != mode_tblock) return zero;

      mf_Float const* p = getWh(mode_hit).site_ptr(site,flavor); //we use different random fields for each time and flavor, although we didn't have to
      return reinterpret_cast<std::complex<mf_Float> const&>(*p);
    }
  }
  //Get a particular site/spin/color element of a given *native* (packed) mode 
  inline const std::complex<mf_Float> & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
    return i < nl ? 
	       reinterpret_cast<std::complex<mf_Float> const& >(*(wl[i].site_ptr(site,flavor)+2*spin_color)) :
      reinterpret_cast<std::complex<mf_Float> const& >(*(wh[i-nl].site_ptr(site,flavor))); //we use different random fields for each time and flavor, although we didn't have to
  }



};


template< typename mf_Float>
class A2AvectorWfftw: public TimeFlavorPackedIndexDilution{
  std::vector<CPSfermion4D<mf_Float> > wl;
  std::vector<CPSfermion4D<mf_Float> > wh; //these have been diluted in spin/color but not the other indices, hence there are nhit * 12 fields here (spin/color index changes fastest in mapping)

  const std::string cname;

public:
  typedef TimeFlavorPackedIndexDilution DilutionType;

  A2AvectorWfftw(const A2AArg &_args): TimeFlavorPackedIndexDilution(_args), cname("A2AvectorWfftw"){
    wl.resize(nl,CPSfermion4D<mf_Float>());
    wh.resize(12*nhits, CPSfermion4D<mf_Float>()); 
  }
  inline const CPSfermion4D<mf_Float> & getWl(const int i) const{ return wl[i]; }
  inline const CPSfermion4D<mf_Float> & getWh(const int hit, const int spin_color) const{ return wh[spin_color + 12*hit]; }

  inline const CPSfermion4D<mf_Float> & getMode(const int i) const{ return i < nl ? wl[i] : wh[i-nl]; }

  //Set this object to be the threaded fast Fourier transform of the input field
  //Can optionally supply an object that performs a transformation on each mode prior to the FFT. 
  //We can use this to avoid intermediate storage for the gauge fixing and momentum phase application steps
  void fft(const A2AvectorW<mf_Float> &from, fieldOperation<mf_Float>* mode_preop = NULL);

  //For each mode, gauge fix, apply the momentum factor, then perform the FFT and store the result in this object
  void gaugeFixTwistFFT(const A2AvectorW<mf_Float> &from, const int _p[3], Lattice &_lat){
    gaugeFixAndTwist<mf_Float> op(_p,_lat); fft(from, &op);
  }

  //The flavor and timeslice dilutions are still packed so we must treat them differently
  //Mode is a full 'StandardIndex', (unpacked mode index)
  const std::complex<mf_Float> & elem(const int mode, const int x3d, const int t, const int spin_color, const int flavor) const{
    static std::complex<mf_Float> zero(0.0,0.0);
    int site = x3d + GJP.VolNodeSites()/GJP.Tnodes()*t;
    if(mode < nl){
      mf_Float const* p = getWl(mode).site_ptr(site,flavor) + 2*spin_color;
      return reinterpret_cast<std::complex<mf_Float> const&>(*p);
    }else{
      int mode_hit, mode_tblock, mode_spin_color,mode_flavor;
      const StandardIndexDilution &dilfull = static_cast<StandardIndexDilution const&>(*this);
      dilfull.indexUnmap(mode-nl,mode_hit,mode_tblock,mode_spin_color,mode_flavor);
      //flavor and time block indices match those of the mode, the result is zero
      int tblock = (t+GJP.TnodeSites()*GJP.TnodeCoor())/args.src_width;
      if(flavor != mode_flavor || tblock != mode_tblock) return zero;

      mf_Float const* p = getWh(mode_hit,mode_spin_color).site_ptr(site,flavor) +2*spin_color; //because we multiplied by an SU(3) matrix, the field is not just a delta function in spin/color
      return reinterpret_cast<std::complex<mf_Float> const&>(*p);
    }
  }
  //Get a particular site/spin/color element of a given *native* (packed) mode 
  inline const std::complex<mf_Float> & nativeElem(const int i, const int site, const int spin_color, const int flavor) const{
    return i < nl ? 
	       reinterpret_cast<std::complex<mf_Float> const& >(*(wl[i].site_ptr(site,flavor)+2*spin_color)) :
      reinterpret_cast<std::complex<mf_Float> const& >(*(wh[i-nl].site_ptr(site,flavor)+2*spin_color)); //spin_color index diluted out.
  }



  //Replace this vector with the average of this another vector, 'with'
  void average(const A2AvectorWfftw<mf_Float> &with, const bool &parallel = true){
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

  inline SCFvectorPtr<mf_Float> getFlavorDilutedVect(const int i, const modeIndexSet &i_high_unmapped, const int site) const{
    const int site_offset = i >= nl ? wh[0].site_offset(site) : wl[0].site_offset(site);
    const int flav_offset = i >= nl ? wh[0].flav_offset() : wl[0].flav_offset();
    return getFlavorDilutedVect(i,i_high_unmapped,site_offset,flav_offset);
  }

  inline SCFvectorPtr<mf_Float> getFlavorDilutedVect(const int i, const modeIndexSet &i_high_unmapped, const int site_offset, const int flav_offset) const{
    const CPSfermion4D<mf_Float> &field = i >= nl ? getWh(i_high_unmapped.hit, i_high_unmapped.spin_color): getWl(i);
    const static mf_Float zerosc[24] = {0,0,0,0,0,0,0,0,0,0,
  					0,0,0,0,0,0,0,0,0,0,
  					0,0,0,0};
    bool zero_hint[2] = {false,false};
    if(i >= nl) zero_hint[ !i_high_unmapped.flavor ] = true;

    mf_Float const* f0_ptr = field.ptr() + site_offset;
    mf_Float const* lp[2] = { zero_hint[0] ? &zerosc[0] : f0_ptr,
  			      zero_hint[1] ? &zerosc[0] : f0_ptr + flav_offset };

    return SCFvectorPtr<mf_Float>(lp[0],lp[1],zero_hint[0],zero_hint[1]);
  }

  inline SCFvectorPtr<mf_Float> getFlavorDilutedVect2(const int i, const modeIndexSet &i_high_unmapped, const int p3d, const int t) const{
    const CPSfermion4D<mf_Float> &field = i >= nl ? getWh(i_high_unmapped.hit, i_high_unmapped.spin_color): getWl(i);
    const static mf_Float zerosc[24] = {0,0,0,0,0,0,0,0,0,0,
  					0,0,0,0,0,0,0,0,0,0,
  					0,0,0,0};
    bool zero_hint[2] = {false,false};
    if(i >= nl) zero_hint[ !i_high_unmapped.flavor ] = true;

    const int x4d = field.threeToFour(p3d,t);
    mf_Float const* lp[2] = { zero_hint[0] ? &zerosc[0] : field.site_ptr(x4d,0),
  			      zero_hint[1] ? &zerosc[0] : field.site_ptr(x4d,1) };

    return SCFvectorPtr<mf_Float>(lp[0],lp[1],zero_hint[0],zero_hint[1]);
  }


  //This version allows for the possibility of a different high mode mapping for the index i by passing the unmapped indices
  const CPSfermion4D<mf_Float> & getMode(const int i, const modeIndexSet &i_high_unmapped) const{ return i >= nl ? getWh(i_high_unmapped.hit, i_high_unmapped.spin_color): getWl(i); }
};

//Generate uniform random V and W vectors for testing
template<typename mf_Float>
void randomizeVW(A2AvectorV<mf_Float> &V, A2AvectorW<mf_Float> &W);




#include <alg/a2a/a2a_impl.h>
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

# if defined(USE_GRID_LANCZOS) && !defined(USE_BFM)
#  error "BFM Lanczos -> Grid A2A interface requires BFM!"
# endif

# include <alg/a2a/a2a_impl_vwgrid.h>

#else

# error "Need either BFM or Grid to compute A2A vectors"

#endif


CPS_END_NAMESPACE

#endif
