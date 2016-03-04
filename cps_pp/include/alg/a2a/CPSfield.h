#ifndef CPS_FIELD_H
#define CPS_FIELD_H

#include<algorithm>
#include<alg/a2a/fftw_wrapper.h>
#include<alg/a2a/utils.h>
#include<alg/a2a/CPSfield_policies.h>
CPS_START_NAMESPACE 

//#define CPS_FIELD_MEMALIGN  //128-bit aligned

//A wrapper for a CPS-style field. Most functionality is generic so it can do quite a lot of cool things. Automatically doubles size for G-parity if using DynamicFlavorPolicy

//SiteSize is measured in units of floats, not complex
template< typename mf_Float, int SiteSize, typename DimensionPolicy, typename FlavorPolicy = DynamicFlavorPolicy>
class CPSfield: public DimensionPolicy, public FlavorPolicy{
  mf_Float* f;
protected:
  int site_size; //number of floats per spatial (not including the dynamical flavor index)
  int sites; //number of Euclidean sites
  int flavors; //number of flavors
  int fsites; //number of generalized sites (including flavor)

  int fsize; //number of floats in the array = site_size * fsites
  
  void alloc(){
#ifdef CPS_FIELD_MEMALIGN
    f = (mf_Float *)memalign(128,fsize*sizeof(mf_Float));
#else
    f = (mf_Float *)smalloc("CPSfield", "CPSfield", "alloc" , sizeof(mf_Float) * fsize);
#endif
  }
  void freemem(){
#ifdef CPS_FIELD_MEMALIGN
    free(f);
#else
    sfree("CPSfield","CPSfield","free",f);
#endif
  }

public:
  typedef mf_Float FieldFloatType;
  typedef DimensionPolicy FieldDimensionPolicy;
  typedef FlavorPolicy FieldFlavorPolicy;

  typedef typename DimensionPolicy::ParamType InputParamType;

  CPSfield(const InputParamType &params): site_size(SiteSize), DimensionPolicy(params){
    this->setFlavors(flavors); //from FlavorPolicy
    this->setSites(sites,fsites,flavors); //from DimensionPolicy
    fsize = fsites * site_size;
    alloc(); //zero();  //Don't automatically zero
  }
  CPSfield(const CPSfield<mf_Float,SiteSize,DimensionPolicy,FlavorPolicy> &r): fsize(r.fsize), site_size(r.site_size),flavors(r.flavors),sites(r.sites),fsites(r.fsites), DimensionPolicy(r){
    alloc();
    memcpy(f,r.f,sizeof(mf_Float) * fsize);
  }

  //Set the field to zero
  void zero(){
    memset(f, 0, sizeof(mf_Float) * fsize);      
  }

  

  CPSfield<mf_Float,SiteSize,DimensionPolicy,FlavorPolicy> &operator=(const CPSfield<mf_Float,SiteSize,DimensionPolicy,FlavorPolicy> &r){
    static_cast<DimensionPolicy&>(*this) = r; //copy policy info
    
    site_size = r.site_size;
    sites = r.sites;
    fsites = r.fsites;
    flavors = r.flavors;

    int old_fsize = fsize;
    fsize = r.fsize;

    if(fsize != old_fsize){
      freemem();
      alloc();
    }
    memcpy(f,r.f,sizeof(mf_Float) * fsize);
    return *this;
  }

  //Set the complex number at pointer p to a random value of a chosen type
  //Uses the current LRG for the given FermionFieldDimension. User should choose the range and the particular site-RNG themselves beforehand
  static void rand(mf_Float *p, const RandomType &type, const FermionFieldDimension &frm_dim);

  //Set each float to a uniform random number in the specified range.
  //WARNING: Uses only the current RNG in LRG, and does not change this based on site. This is therefore only useful for testing*
  void testRandom(const Float &hi = 0.5, const Float &lo = -0.5);

  const int nsites() const{ return sites; }
  const int nflavors() const{ return flavors; }
  const int nfsites() const{ return fsites; } //number of generalized sites including flavor

  //Number of floats per site
  const int siteSize() const{ return site_size; }

  //Number of floats in field
  const int size() const{ return fsize; }

  //Accessors
  inline mf_Float* ptr(){ return f; }
  inline mf_Float const* ptr() const{ return f; }

  //Accessors *do not check bounds*
  //int fsite is the linearized N-dimensional site/flavorcoordinate with the mapping specified by the policy class
  inline int fsite_offset(const int fsite) const{ return site_size*fsite; }

  inline mf_Float* fsite_ptr(const int fsite){  //fsite is in the internal flavor/Euclidean mapping of the DimensionPolicy. Use only if you know what you are doing
    return f + site_size*fsite;
  }
  inline mf_Float const* fsite_ptr(const int fsite) const{  //fsite is in the internal flavor/Euclidean mapping of the DimensionPolicy. Use only if you know what you are doing
    return f + site_size*fsite;
  }

  //int site is the linearized N-dimension Euclidean coordinate with mapping specified by the policy class
  inline int site_offset(const int site, const int flav = 0) const{ return site_size*this->siteFsiteConvert(site,flav); }
  inline int site_offset(const int x[], const int flav = 0) const{ return site_size*this->fsiteMap(x,flav); }

  inline mf_Float* site_ptr(const int site, const int flav = 0){  //site is in the internal Euclidean mapping of the DimensionPolicy
    return f + site_size*this->siteFsiteConvert(site,flav);
  }
  inline mf_Float* site_ptr(const int x[], const int flav = 0){ 
    return f + site_size*this->fsiteMap(x,flav);
  }    

  inline mf_Float const* site_ptr(const int site, const int flav = 0) const{  //site is in the internal Euclidean mapping of the DimensionPolicy
    return f + site_size*this->siteFsiteConvert(site,flav);
  }
  inline mf_Float const* site_ptr(const int x[], const int flav = 0) const{ 
    return f + site_size*this->fsiteMap(x,flav);
  }    
 
  inline int flav_offset() const{ return site_size*this->fsiteFlavorOffset(); } //pointer offset between flavors

  //Set this field to the average of this and a second field, r
  void average(const CPSfield<mf_Float,SiteSize,DimensionPolicy,FlavorPolicy> &r, const bool &parallel = true);

  //Self destruct initialized (no more sfree!!)
  virtual ~CPSfield(){
    freemem();
  }

  //Import an export field with arbitrary DimensionPolicy (must have same Euclidean dimension!) and precision. Must have same SiteSize and FlavorPolicy

private:
  template<typename extFloat, typename extDimPol>
  void importFieldInternal(const typename my_enable_if< sameDim<extDimPol,DimensionPolicy>::val, CPSfield<extFloat, SiteSize,extDimPol, FlavorPolicy> >::type &r){
#pragma omp parallel for
    for(int fs=0;fs<this->fsites;fs++){
      int x[5], f; this->fsiteUnmap(fs,x,f);
      mf_Float* to = this->fsite_ptr(fs);
      mf_Float const* from = r.site_ptr(x,f);
      for(int i=0;i<SiteSize;i++) to[i] = from[i];
    }
  }

  template<typename extFloat, typename extDimPol>
  void exportFieldInternal(typename my_enable_if< sameDim<extDimPol,DimensionPolicy>::val, CPSfield<extFloat, SiteSize,extDimPol, FlavorPolicy> >::type &r){
#pragma omp parallel for
    for(int fs=0;fs<this->fsites;fs++){
      int x[5], f; this->fsiteUnmap(fs,x,f);
      mf_Float const* from = this->fsite_ptr(fs);
      mf_Float* to = r.site_ptr(x,f);
      for(int i=0;i<SiteSize;i++) to[i] = from[i];
    }
  }
public:
  
  template<typename extField>
  void importField(const extField &r){
    importFieldInternal<typename extField::FieldFloatType,typename extField::FieldDimensionPolicy>(r);
  }
  template<typename extField>
  void exportField(extField &r){
    exportFieldInternal<typename extField::FieldFloatType,typename extField::FieldDimensionPolicy>(r);
  }

};







template< typename mf_Float, typename DimensionPolicy, typename FlavorPolicy = DynamicFlavorPolicy>
class CPSfermion: public CPSfield<mf_Float,SPINOR_SIZE,DimensionPolicy,FlavorPolicy>{
protected:
  void gauge_fix_site_op(const int x4d[], const int &f, Lattice &lat);

  static void getMomentumUnits(double punits[3]);

  //Apply the phase exp(-ip.x) to each site of this vector, where p is a *three momentum*
  //The units of the momentum are 2pi/L for periodic BCs, pi/L for antiperiodic BCs and pi/2L for G-parity BCs
  //x_lcl is the site in node lattice coords
  void apply_phase_site_op(const int x_lcl[], const int &flav, const int p[], const double punits[]);

public:
  CPSfermion(): CPSfield<mf_Float,SPINOR_SIZE,DimensionPolicy,FlavorPolicy>(NullObject()){}
  CPSfermion(const CPSfermion<mf_Float,DimensionPolicy,FlavorPolicy> &r): CPSfield<mf_Float,SPINOR_SIZE,DimensionPolicy,FlavorPolicy>(r){}
};

template<typename FlavorPolicy>
struct GaugeFix3DInfo{};

template<>
struct GaugeFix3DInfo<DynamicFlavorPolicy>{
  typedef int InfoType;
};
template<>
struct GaugeFix3DInfo<FixedFlavorPolicy<2> >{
  typedef int InfoType;
};
template<>
struct GaugeFix3DInfo<FixedFlavorPolicy<1> >{
  typedef std::pair<int,int> InfoType; //time, flavor (latter ignored if no GPBC)
};

template< typename mf_Float, typename FlavorPolicy = DynamicFlavorPolicy>
class CPSfermion3D: public CPSfermion<mf_Float,SpatialPolicy,FlavorPolicy>{
  void apply_phase_site_op(const int &sf,const int p[],double punits[]);

  template< typename mf_Float2, typename FlavorPolicy2>
  friend struct _ferm3d_gfix_impl;
public:
  CPSfermion3D(): CPSfermion<mf_Float,SpatialPolicy,FlavorPolicy>(){}
  CPSfermion3D(const CPSfermion3D<mf_Float> &r): CPSfermion<mf_Float,SpatialPolicy,FlavorPolicy>(r){}

  //Apply gauge fixing matrices to the field
  //Because this is a 3d field we must also provide a time coordinate.
  //If the field is one flavor we must also provide the flavor
  //We make the field_info type dynamic based on the FlavorPolicy for this reason (pretty cool!)
  void gaugeFix(Lattice &lat, const typename GaugeFix3DInfo<FlavorPolicy>::InfoType &field_info, const bool &parallel);

  //Apply the phase exp(-ip.x) to each site of this vector, where p is a *three momentum*
  //The units of the momentum are 2pi/L for periodic BCs, pi/L for antiperiodic BCs and pi/2L for G-parity BCs
  void applyPhase(const int p[], const bool &parallel);

  //Set this field to be the FFT of 'r'
  void fft(const CPSfermion3D<mf_Float,FlavorPolicy> &r);

  //Set this field to be the FFT of itself
  void fft(){ fft(*this); }
};

template< typename mf_Float, typename FlavorPolicy = DynamicFlavorPolicy>
class CPSfermion4D: public CPSfermion<mf_Float,FourDpolicy,FlavorPolicy>{
  void gauge_fix_site_op(int fi, Lattice &lat);
  void apply_phase_site_op(int sf,const int p[],double punits[]);
public:
  CPSfermion4D(): CPSfermion<mf_Float,FourDpolicy,FlavorPolicy>(){}
  CPSfermion4D(const CPSfermion4D<mf_Float,FlavorPolicy> &r): CPSfermion<mf_Float,FourDpolicy,FlavorPolicy>(r){}

  //Apply gauge fixing matrices to the field. 
  //NOTE: This does not work correctly for GPBC and FlavorPolicy==FixedFlavorPolicy<1> because we need to provide the flavor 
  //that this field represents to obtain the gauge-fixing matrix. I fixed this for CPSfermion3D and a similar implementation will work here
  void gaugeFix(Lattice &lat, const bool &parallel);

  //Apply the phase exp(-ip.x) to each site of this vector, where p is a *three momentum*
  //The units of the momentum are 2pi/L for periodic BCs, pi/L for antiperiodic BCs and pi/2L for G-parity BCs
  void applyPhase(const int p[], const bool &parallel);

  //Set this field to be the FFT of 'r'
  void fft(const CPSfermion4D<mf_Float,FlavorPolicy> &r);

  //Set this field to be the FFT of itself
  void fft(){ fft(*this); }

  //Set the real and imaginary parts to uniform random numbers drawn from the appropriate local RNGs
  void setUniformRandom(const Float &hi = 0.5, const Float &lo = -0.5);
};

template< typename mf_Float, typename FlavorPolicy = DynamicFlavorPolicy>
class CPSfermion5D: public CPSfield<mf_Float,SPINOR_SIZE,FiveDpolicy,FlavorPolicy>{
public:
  CPSfermion5D(): CPSfield<mf_Float,SPINOR_SIZE,FiveDpolicy,FlavorPolicy>(NullObject()){}
  CPSfermion5D(const CPSfermion5D<mf_Float,FlavorPolicy> &r): CPSfield<mf_Float,SPINOR_SIZE,FiveDpolicy,FlavorPolicy>(r){}
  
#ifdef USE_BFM
private:
  template<typename FloatExt>
  void impexFermion(Fermion_t bfm_field, const int cb, const int do_import, bfm_qdp<FloatExt> &dwf){
    if(this->flavors == 2) assert(dwf.gparity);

    const int sc_incr = dwf.nsimd() * 2; //stride between spin-color indices
    FloatExt * bb = (FloatExt*)bfm_field;

#pragma omp parallel for
    for(int fs=0;fs<this->fsites;fs++){
      int x[5], f; this->fsiteUnmap(fs);
      if( (x[0]+x[1]+x[2]+x[3] + (dwf.precon_5d ? x[4] : 0)) % 2 == cb){
	mf_Float* cps_base = this->fsite_ptr(fs);

	int bidx_off = dwf.gparity ? 
	  dwf.bagel_gparity_idx5d(x, x[4], 0, 0, 12, 1, f) :
	  dwf.bagel_idx5d(x, x[4], 0, 0, 12, 1);

	FloatExt * bfm_base = bb + bidx_off;

	for(int i=0;i<12;i++)
	  for(int reim=0;reim<2;reim++)
	    if(do_import)
	      *(cps_base + 2*i + reim) = *(bfm_base + 2*sc_incr*i + reim);
	    else
	      *(bfm_base + 2*sc_incr*i + reim) = *(cps_base + 2*i + reim);
      }
    }
  }
public:
  template<typename FloatExt>
  void importFermion(const Fermion_t bfm_field, const int cb, bfm_qdp<FloatExt> &dwf){
    impexFermion<FloatExt>(const_cast<Fermion_t>(bfm_field), cb, 1, dwf);
  }
  template<typename FloatExt>
  void exportFermion(const Fermion_t bfm_field, const int cb, bfm_qdp<FloatExt> &dwf) const{
    const_cast<CPSfermion5D<mf_Float,FlavorPolicy>*>(this)->impexFermion<FloatExt>(bfm_field, cb, 0, dwf);
  }
#endif





};


template< typename mf_Float>
class CPScomplex4D: public CPSfield<mf_Float,2,FourDpolicy>{

public:
  CPScomplex4D(): CPSfield<mf_Float,2,FourDpolicy>(NullObject()){}
  CPScomplex4D(const CPScomplex4D<mf_Float> &r): CPSfield<mf_Float,2,FourDpolicy>(r){}

  //Make a random complex scalar field of type
  void setRandom(const RandomType &type);

  //Set the real and imaginary parts to uniform random numbers drawn from the appropriate local RNGs
  void setUniformRandom(const Float &hi = 0.5, const Float &lo = -0.5);
};

//3d complex number field
template< typename mf_Float, typename FlavorPolicy = DynamicFlavorPolicy>
class CPScomplexSpatial: public CPSfield<mf_Float,2,SpatialPolicy,FlavorPolicy>{

public:
  CPScomplexSpatial(): CPSfield<mf_Float,2,SpatialPolicy,FlavorPolicy>(NullObject()){}
  CPScomplexSpatial(const CPScomplexSpatial<mf_Float,FlavorPolicy> &r): CPSfield<mf_Float,2,SpatialPolicy,FlavorPolicy>(r){}
};

//Lattice-spanning 'global' 3d complex field
template< typename mf_Float, typename FlavorPolicy = DynamicFlavorPolicy>
class CPSglobalComplexSpatial: public CPSfield<mf_Float,2,GlobalSpatialPolicy,FlavorPolicy>{

public:
  CPSglobalComplexSpatial(): CPSfield<mf_Float,2,GlobalSpatialPolicy,FlavorPolicy>(NullObject()){}
  CPSglobalComplexSpatial(const CPSglobalComplexSpatial<mf_Float,FlavorPolicy> &r): CPSfield<mf_Float,2,GlobalSpatialPolicy,FlavorPolicy>(r){}
  
  //Perform the FFT
  void fft();

  //Scatter to a local field
  void scatter(CPScomplexSpatial<mf_Float,FlavorPolicy> &to) const;
};


//This field contains an entire row of sub-lattices along a particular dimension. Every node along that row contains an identical copy
template< typename mf_Float, int SiteSize, typename DimensionPolicy, typename FlavorPolicy = DynamicFlavorPolicy>
class CPSfieldGlobalInOneDir: public CPSfield<mf_Float,SiteSize,DimensionPolicy,FlavorPolicy>{
  std::string cname;
public:
  CPSfieldGlobalInOneDir(const int &dir): cname("CPSfieldGlobalInOneDir"), CPSfield<mf_Float,SiteSize,DimensionPolicy,FlavorPolicy>(dir){}
  CPSfieldGlobalInOneDir(const CPSfieldGlobalInOneDir<mf_Float,SiteSize,DimensionPolicy,FlavorPolicy> &r): cname("CPSfieldGlobalInOneDir"), CPSfield<mf_Float,SiteSize,DimensionPolicy,FlavorPolicy>(r){}

  //Gather up the row. Involves internode communication
  template<typename LocalDimensionPolicy>
  void gather(const CPSfield<mf_Float,SiteSize,LocalDimensionPolicy,FlavorPolicy> &from);

  //Scatter back out. Involves no communication
  template<typename LocalDimensionPolicy>
  void scatter(CPSfield<mf_Float,SiteSize,LocalDimensionPolicy,FlavorPolicy> &to) const;

  //Perform a fast Fourier transform along the principal direction. It currently assumes the DimensionPolicy has the sites mapped in canonical ordering
  void fft();
};

template< typename mf_Float, typename FlavorPolicy = DynamicFlavorPolicy>
class CPSfermion4DglobalInOneDir: public CPSfieldGlobalInOneDir<mf_Float,SPINOR_SIZE,FourDglobalInOneDir,FlavorPolicy>{
  std::string cname;
public:
  CPSfermion4DglobalInOneDir(const int &dir): cname("CPSfermion4DglobalInOneDir"), CPSfieldGlobalInOneDir<mf_Float,SPINOR_SIZE,FourDglobalInOneDir,FlavorPolicy>(dir){}
  CPSfermion4DglobalInOneDir(const CPSfermion4DglobalInOneDir<mf_Float,FlavorPolicy> &r): cname("CPSfermion4DglobalInOneDir"), CPSfieldGlobalInOneDir<mf_Float,SPINOR_SIZE,FourDglobalInOneDir,FlavorPolicy>(r){}
};
template< typename mf_Float, typename FlavorPolicy = DynamicFlavorPolicy>
class CPSfermion3DglobalInOneDir: public CPSfieldGlobalInOneDir<mf_Float,SPINOR_SIZE,ThreeDglobalInOneDir,FlavorPolicy>{
  std::string cname;
public:
  CPSfermion3DglobalInOneDir(const int &dir): cname("CPSfermion3DglobalInOneDir"), CPSfieldGlobalInOneDir<mf_Float,SPINOR_SIZE,ThreeDglobalInOneDir,FlavorPolicy>(dir){}
  CPSfermion3DglobalInOneDir(const CPSfermion3DglobalInOneDir<mf_Float,FlavorPolicy> &r): cname("CPSfermion4DglobalInOneDir"), CPSfieldGlobalInOneDir<mf_Float,SPINOR_SIZE,ThreeDglobalInOneDir,FlavorPolicy>(r){}
};


////////Checkerboarded types/////////////
template< typename mf_Float, typename CBpolicy, typename FlavorPolicy = DynamicFlavorPolicy>
class CPSfermion5Dprec: public CPSfield<mf_Float,SPINOR_SIZE,FiveDevenOddpolicy<CBpolicy>,FlavorPolicy>{
public:
  CPSfermion5Dprec(): CPSfield<mf_Float,SPINOR_SIZE,FiveDevenOddpolicy<CBpolicy>,FlavorPolicy>(NullObject()){}
  CPSfermion5Dprec(const CPSfermion5Dprec<mf_Float,CBpolicy,FlavorPolicy> &r): CPSfield<mf_Float,SPINOR_SIZE,FiveDevenOddpolicy<CBpolicy>,FlavorPolicy>(r){}
  
//   void importFermion(const CPSfermion5D<mf_Float, FlavorPolicy> &r){
// #pragma omp parallel for
//     for(int fs=0;fs<this->fsites;fs++){
//       int x[5], f; this->fsiteUnmap(fs,x,f);
//       mf_Float* to = this->fsite_ptr(fs);
//       mf_Float const* from = r.site_ptr(x,f);
//       for(int i=0;i<SPINOR_SIZE;i++) to[i] = from[i];
//     }
//   }

//   void exportFermion(CPSfermion5D<mf_Float, FlavorPolicy> &r){
// #pragma omp parallel for
//     for(int fs=0;fs<this->fsites;fs++){
//       int x[5], f; this->fsiteUnmap(fs,x,f);
//       mf_Float const* from = this->fsite_ptr(fs);
//       mf_Float* to = r.site_ptr(x,f);
//       for(int i=0;i<SPINOR_SIZE;i++) to[i] = from[i];
//     }
//   }
};


template< typename mf_Float, typename FlavorPolicy = DynamicFlavorPolicy>
class CPSfermion5Dcb4Deven: public CPSfermion5Dprec<mf_Float,CheckerBoard<4,0>,FlavorPolicy>{
public:
  CPSfermion5Dcb4Deven(): CPSfermion5Dprec<mf_Float,CheckerBoard<4,0>,FlavorPolicy>(){}
  CPSfermion5Dcb4Deven(const CPSfermion5Dcb4Deven<mf_Float,FlavorPolicy> &r): CPSfermion5Dprec<mf_Float,CheckerBoard<4,0>,FlavorPolicy>(r){}
};
template< typename mf_Float, typename FlavorPolicy = DynamicFlavorPolicy>
class CPSfermion5Dcb4Dodd: public CPSfermion5Dprec<mf_Float,CheckerBoard<4,1>,FlavorPolicy>{
public:
  CPSfermion5Dcb4Dodd(): CPSfermion5Dprec<mf_Float,CheckerBoard<4,1>,FlavorPolicy>(){}
  CPSfermion5Dcb4Dodd(const CPSfermion5Dcb4Dodd<mf_Float,FlavorPolicy> &r): CPSfermion5Dprec<mf_Float,CheckerBoard<4,1>,FlavorPolicy>(r){}
};





#include<alg/a2a/CPSfield_impl.h>

CPS_END_NAMESPACE
#endif
