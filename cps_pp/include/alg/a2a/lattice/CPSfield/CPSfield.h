#ifndef CPS_FIELD_H
#define CPS_FIELD_H

#include<algorithm>
#include<comms/scu.h>

#include "CPSfield_policies.h"
#ifdef USE_BFM
#include<util/lattice/bfm_evo.h>
#endif

#include<alg/a2a/utils.h>

CPS_START_NAMESPACE 

typedef std::complex<float> ComplexF;
typedef std::complex<double> ComplexD;

enum CPSfield_checksumType { checksumCRC32, checksumBasic };
inline std::string checksumTypeToString(CPSfield_checksumType type){ return type == checksumCRC32 ? "checksumCRC32" : "checksumBasic"; }
inline CPSfield_checksumType checksumTypeFromString(const std::string &str){
  if(str == "checksumCRC32") return checksumCRC32;
  else if(str == "checksumBasic") return checksumBasic;
  else ERR.General("","checksumTypeFromString","Could not parse checksum type %s\n",str.c_str());
}

//A wrapper for a CPS-style field. Most functionality is generic so it can do quite a lot of cool things
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy = StandardAllocPolicy>
class CPSfield: public MappingPolicy, public AllocPolicy{
  SiteType* f;
protected:
  int fsize; //number of SiteType in the array = SiteSize * fsites
  bool own; //is the memory owned by this object?
  
  void alloc(){
    this->_alloc((void**)&f, fsize*sizeof(SiteType));
  }
  void freemem(){
    this->_free((void*)f);
  }

public:
  enum { FieldSiteSize = SiteSize };
  typedef SiteType FieldSiteType;
  typedef MappingPolicy FieldMappingPolicy;
  typedef AllocPolicy FieldAllocPolicy;
  
  typedef typename MappingPolicy::ParamType InputParamType;

  CPSfield(const InputParamType &params): MappingPolicy(params), own(true){
    fsize = this->nfsites() * SiteSize;
    alloc();
  }
  CPSfield(const CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> &r): fsize(r.fsize), MappingPolicy(r){
    if(copyControl::shallow() && AllocPolicy::UVMenabled == 1){
      //std::cout << "CPSFIELD CONSTRUCTOR SHALLOW COPY" << std::endl;
      f = r.f; own = false;
    }else{
      //std::cout << "CPSFIELD CONSTRUCTOR DEEP COPY" << std::endl;
      alloc();
      memcpy(f,r.f,sizeof(SiteType) * fsize);
      own = true;
    }
  }

  CPSfield(CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> &&r): fsize(r.fsize), MappingPolicy(r), own(r.own), f(r.f){
    r.f = NULL;
    r.own = false;
    r.fsize = 0;
  }

  //Copy from external pointer. Make sure you set the params and policies correctly because it has no way of bounds checking
  CPSfield(SiteType const* copyme, const InputParamType &params): MappingPolicy(params), own(true){
    fsize = this->nfsites() * SiteSize;
    alloc();
    memcpy(f,copyme,sizeof(SiteType) * fsize);
  }

  //Self destruct initialized (no more sfree!!)
  virtual ~CPSfield(){
    if(own){
      //std::cout << "CPSFIELD DESTRUCTOR DEEP FREE" <<std::endl;
      freemem();    
    }//else std::cout << "CPSFIELD DESTRUCTOR SHALLOW FREE" <<std::endl;
  }
  
  //Set the field to zero
  void zero(){
    memset(f, 0, sizeof(SiteType) * fsize);      
  }

  CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> &operator=(const CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> &r){
    if(!own){ own = true; fsize = r.fsize; alloc(); } //was a shallow copy, alloc new memory and take ownership
    
    static_cast<MappingPolicy&>(*this) = r; //copy policy info

    int old_fsize = fsize;
    fsize = r.fsize;

    if(fsize != old_fsize){
      freemem();
      alloc();
    }
    memcpy(f,r.f,sizeof(SiteType) * fsize);
    return *this;
  }

  CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> &operator=(const CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> &&r){
    if(own) freemem();
    f = r.f;
    own = r.own;
    fsize = r.fsize;
    r.f = NULL;
    r.own = false;
    return *this;
  }


  static std::size_t byte_size(const InputParamType &params){
    CPSfield<SiteType,SiteSize,MappingPolicy,NullAllocPolicy> tmp(params); //doesn't allocate
    std::size_t out = SiteSize * sizeof(SiteType);
    return tmp.nfsites() * out;
  }
  std::size_t byte_size() const{
    return this->nfsites() * SiteSize * sizeof(SiteType);
  }
  
  //Set each element to a uniform random number in the specified range.
  //WARNING: Uses only the current RNG in LRG, and does not change this based on site. This is therefore only useful for testing*
  void testRandom(const Float hi = 0.5, const Float lo = -0.5);

  //Number of SiteType per site
  accelerator_inline const int siteSize() const{ return SiteSize; }

  //Number of SiteType in field
  accelerator_inline const int size() const{ return fsize; }

  //Accessors
  accelerator_inline SiteType* ptr(){ return f; }
  accelerator_inline SiteType const* ptr() const{ return f; }

  //Accessors *do not check bounds*
  //int fsite is the linearized N-dimensional site/flavorcoordinate with the mapping specified by the policy class
  accelerator_inline int fsite_offset(const int fsite) const{ return SiteSize*fsite; }
  
  accelerator_inline SiteType* fsite_ptr(const int fsite){  //fsite is in the internal flavor/Euclidean mapping of the MappingPolicy. Use only if you know what you are doing
    return f + SiteSize*fsite;
  }
  accelerator_inline SiteType const* fsite_ptr(const int fsite) const{  //fsite is in the internal flavor/Euclidean mapping of the MappingPolicy. Use only if you know what you are doing
    return f + SiteSize*fsite;
  }

  //int site is the linearized N-dimension Euclidean coordinate with mapping specified by the policy class
  accelerator_inline int site_offset(const int site, const int flav = 0) const{ return SiteSize*this->siteFsiteConvert(site,flav); }
  accelerator_inline int site_offset(const int x[], const int flav = 0) const{ return SiteSize*this->fsiteMap(x,flav); }

  accelerator_inline SiteType* site_ptr(const int site, const int flav = 0){  //site is in the internal Euclidean mapping of the MappingPolicy
    return f + SiteSize*this->siteFsiteConvert(site,flav);
  }
  accelerator_inline SiteType* site_ptr(const int x[], const int flav = 0){ 
    return f + SiteSize*this->fsiteMap(x,flav);
  }    

  accelerator_inline SiteType const* site_ptr(const int site, const int flav = 0) const{  //site is in the internal Euclidean mapping of the MappingPolicy
    return f + SiteSize*this->siteFsiteConvert(site,flav);
  }
  accelerator_inline SiteType const* site_ptr(const int x[], const int flav = 0) const{ 
    return f + SiteSize*this->fsiteMap(x,flav);
  }    
 
  accelerator_inline int flav_offset() const{ return SiteSize*this->fsiteFlavorOffset(); } //pointer offset between flavors

  //Set this field to the average of this and a second field, r
  void average(const CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> &r, const bool &parallel = true);

  //Import and export field with arbitrary MappingPolicy (must have same Euclidean dimension!) and precision. Must have same SiteSize and FlavorPolicy
  //Optional inclusion of a mask for accepting sites from the input field
  template< typename extSiteType, typename extMapPol, typename extAllocPol>
  void importField(const CPSfield<extSiteType,SiteSize,extMapPol,extAllocPol> &r, IncludeSite<extMapPol::EuclideanDimension> const* fromsitemask = NULL);

  template< typename extSiteType, typename extMapPol, typename extAllocPol>
  void exportField(CPSfield<extSiteType,SiteSize,extMapPol,extAllocPol> &r, IncludeSite<MappingPolicy::EuclideanDimension> const* fromsitemask = NULL) const;

  bool equals(const CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> &r) const{
    for(int i=0;i<fsize;i++)
      if(!cps::equals(f[i],r.f[i])) return false;
    return true;
  }

#define CONDITION is_double_or_float<typename extField::FieldSiteType>::value \
  && is_double_or_float<SiteType>::value \
  && _equal<MappingPolicy, typename extField::FieldMappingPolicy>::value
  
  template<typename extField>
  bool equals(const extField &r, typename my_enable_if<CONDITION,const double>::type tolerance) const{
    for(int i=0;i<fsize;i++){
      if( fabs(f[i] - r.f[i]) > tolerance) return false;
    }
    return true;
  }
#undef CONDITION
  
#define CONDITION is_complex_double_or_float<typename extField::FieldSiteType>::value \
  && is_complex_double_or_float<SiteType>::value \
  && _equal<MappingPolicy, typename extField::FieldMappingPolicy>::value
  
  template<typename extField>
  bool equals(const extField &r, typename my_enable_if<CONDITION,const double>::type tolerance, bool verbose = false) const{
    for(int i=0;i<fsize;i++){
      if( fabs(f[i].real() - r.f[i].real()) > tolerance || fabs(f[i].imag() - r.f[i].imag()) > tolerance ){
	if(verbose && !UniqueID()){
	  int rem = i;
	  int s = rem % SiteSize; rem /= SiteSize;
	  int x = rem % this->nsites(); rem /= this->nsites();
	  int flav = rem;
	  int coor[MappingPolicy::EuclideanDimension]; this->siteUnmap(x,coor);
	  std::ostringstream os; for(int a=0;a<MappingPolicy::EuclideanDimension;a++) os << coor[a] << " ";
	  std::string coor_str = os.str();
	  
	  printf("Err: off %d  [s=%d coor=(%s) f=%d] this[%g,%g] vs that[%g,%g] : diff [%g,%g]\n",i, s,coor_str.c_str(),flav,
		 f[i].real(),f[i].imag(),r.f[i].real(),r.f[i].imag(),fabs(f[i].real()-r.f[i].real()), fabs(f[i].imag()-r.f[i].imag()) );
	}
	return false;
      }
    }
    return true;
  }
#undef CONDITION

#ifdef USE_GRID
  
#define CONDITION _equal<  typename ComplexClassify<typename extField::FieldSiteType>::type  ,  grid_vector_complex_mark>::value \
  && _equal<typename ComplexClassify<SiteType>::type,grid_vector_complex_mark>::value \
  && _equal<MappingPolicy, typename extField::FieldMappingPolicy>::value

  template<typename extField>
  bool equals(const extField &r, typename my_enable_if<CONDITION,const double>::type tolerance, bool verbose = false) const{
    typedef typename SiteType::scalar_type ThisScalarType;
    typedef typename extField::FieldSiteType::scalar_type ThatScalarType;
    typedef typename MappingPolicy::EquivalentScalarPolicy ScalarMapPol;
    NullObject null_obj;
    CPSfield<ThisScalarType,SiteSize,ScalarMapPol, StandardAllocPolicy> tmp_this(null_obj);
    CPSfield<ThatScalarType,SiteSize,ScalarMapPol, StandardAllocPolicy> tmp_that(null_obj);
    tmp_this.importField(*this);
    tmp_that.importField(r);
    return tmp_this.equals(tmp_that,tolerance,verbose);
  }
  
#undef CONDITION
#endif
  
  double norm2() const;
  double norm2(const IncludeSite<MappingPolicy::EuclideanDimension> & restrictsites) const;
  
#ifdef USE_GRID
  //Import for Grid Lattice<blah> types
  template<typename GridField>
  void importGridField(const GridField &grid, IncludeSite<MappingPolicy::EuclideanDimension> const* fromsitemask = NULL);
  
  template<typename GridField>
  void exportGridField(GridField &grid, IncludeSite<MappingPolicy::EuclideanDimension> const* fromsitemask = NULL) const;
#endif

  CPSfield & operator+=(const CPSfield &r){
#pragma omp parallel for
    for(int i=0;i<fsize;i++) f[i] += r.f[i];
    return *this;
  }
  CPSfield & operator-=(const CPSfield &r){
#pragma omp parallel for
    for(int i=0;i<fsize;i++) f[i] -= r.f[i];
    return *this;
  }

  CPSfield operator+(const CPSfield &r) const{
    CPSfield out(*this); out += r;
    return out;
  }
  CPSfield operator-(const CPSfield &r){
    CPSfield out(*this); out -= r;
    return out;
  }

  void writeParallel(std::ostream &file, FP_FORMAT fileformat = FP_AUTOMATIC, CPSfield_checksumType cksumtype = checksumBasic) const;
  void writeParallel(const std::string &file_stub, FP_FORMAT fileformat = FP_AUTOMATIC, CPSfield_checksumType cksumtype = checksumBasic) const; //node index is appended
  void readParallel(std::istream &file);
  void readParallel(const std::string &file_stub); 

  void writeParallelSeparateMetadata(const std::string &path, FP_FORMAT fileformat) const;
  void readParallelSeparateMetadata(const std::string &path);
};


//Some useful macros for creating derived types
#define INHERIT_TYPEDEFS(...) \
  typedef typename __VA_ARGS__::FieldSiteType FieldSiteType; \
  typedef typename __VA_ARGS__::FieldMappingPolicy FieldMappingPolicy; \
  typedef typename __VA_ARGS__::FieldAllocPolicy FieldAllocPolicy; \
  typedef typename __VA_ARGS__::InputParamType InputParamType; \
  enum { FieldSiteSize = __VA_ARGS__::FieldSiteSize }


#define DEFINE_ADDSUB_DERIVED(DerivedType) \
  DerivedType & operator+=(const DerivedType &r){ \
    this->CPSfield<FieldSiteType,FieldSiteSize,FieldMappingPolicy,FieldAllocPolicy>::operator+=(r); return *this; \
  } \
  DerivedType & operator-=(const DerivedType &r){ \
    this->CPSfield<FieldSiteType,FieldSiteSize,FieldMappingPolicy,FieldAllocPolicy>::operator-=(r); return *this; \
  } \
  DerivedType operator+(const DerivedType &r) const{ \
    DerivedType out(*this); out += r; \
    return out; \
  } \
  DerivedType operator-(const DerivedType &r){ \
    DerivedType out(*this); out -= r; \
    return out; \
  }

#define CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,DerivedType) \
  DerivedType(): BaseType(NullObject()){} /*default constructor won't compile if policies need arguments*/ \
  DerivedType(const InputParamType &params): BaseType(params){} \
  DerivedType(const DerivedType &r): BaseType(r){} \
  DerivedType(DerivedType &&r): BaseType(std::move(r)){} \
  DerivedType & operator=(const DerivedType &r){ this->BaseType::operator=(r); return *this; } \
  DerivedType & operator=(DerivedType &&r){ this->BaseType::operator=(std::move(r)); return *this; }
  



template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy = StandardAllocPolicy>
class CPSfermion: public CPSfield<mf_Complex,12,MappingPolicy,AllocPolicy>{
protected:
  static void getMomentumUnits(double punits[3]);

public:
  typedef CPSfield<mf_Complex,12,MappingPolicy,AllocPolicy> BaseType;
  INHERIT_TYPEDEFS(BaseType);
  CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,CPSfermion);
};

template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy = StandardAllocPolicy>
class CPSfermion3D4Dcommon: public CPSfermion<mf_Complex,MappingPolicy,AllocPolicy>{
public:
  typedef CPSfermion<mf_Complex,MappingPolicy,AllocPolicy> BaseType;
  INHERIT_TYPEDEFS(BaseType);
  CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,CPSfermion3D4Dcommon);
  
  //Apply the phase exp(-ip.x) to each site of this vector, where p is a *three momentum*
  //The units of the momentum are 2pi/L for periodic BCs, pi/L for antiperiodic BCs and pi/2L for G-parity BCs
  void applyPhase(const int p[], const bool parallel);
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

template< typename mf_Complex, typename MappingPolicy = SpatialPolicy<DynamicFlavorPolicy>, typename AllocPolicy = StandardAllocPolicy>
class CPSfermion3D: public CPSfermion3D4Dcommon<mf_Complex,MappingPolicy,AllocPolicy>{
  StaticAssert<MappingPolicy::EuclideanDimension == 3> check;

  template< typename mf_Complex2, typename FlavorPolicy2>
  friend struct _ferm3d_gfix_impl;
public:
  typedef CPSfermion3D4Dcommon<mf_Complex,MappingPolicy,AllocPolicy> BaseType;
  INHERIT_TYPEDEFS(BaseType);
  CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,CPSfermion3D);

  //Apply gauge fixing matrices to the field
  //Because this is a 3d field we must also provide a time coordinate.
  //If the field is one flavor we must also provide the flavor
  //We make the field_info type dynamic based on the FlavorPolicy for this reason (pretty cool!)
  void gaugeFix(Lattice &lat, const typename GaugeFix3DInfo<typename MappingPolicy::FieldFlavorPolicy>::InfoType &field_info, const bool &parallel);

  DEFINE_ADDSUB_DERIVED(CPSfermion3D);
};

template< typename mf_Complex, typename MappingPolicy = FourDpolicy<DynamicFlavorPolicy>, typename AllocPolicy = StandardAllocPolicy>
class CPSfermion4D: public CPSfermion3D4Dcommon<mf_Complex,MappingPolicy,AllocPolicy>{
  StaticAssert<MappingPolicy::EuclideanDimension == 4> check;
public:
  typedef CPSfermion3D4Dcommon<mf_Complex,MappingPolicy,AllocPolicy> BaseType;
  INHERIT_TYPEDEFS(BaseType);
  CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,CPSfermion4D);

  //Apply gauge fixing matrices to the field. 
  //NOTE: This does not work correctly for GPBC and FlavorPolicy==FixedFlavorPolicy<1> because we need to provide the flavor 
  //that this field represents to obtain the gauge-fixing matrix. I fixed this for CPSfermion3D and a similar implementation will work here
  //dagger = true  applied V^\dagger to the vector to invert a previous gauge fix
  void gaugeFix(Lattice &lat, const bool parallel, const bool dagger = false);

  //Set the real and imaginary parts to uniform random numbers drawn from the appropriate local RNGs
  void setUniformRandom(const Float &hi = 0.5, const Float &lo = -0.5);

  void setGaussianRandom();

  DEFINE_ADDSUB_DERIVED(CPSfermion4D);
};

template< typename mf_Complex, typename MappingPolicy = FiveDpolicy<DynamicFlavorPolicy>, typename AllocPolicy = StandardAllocPolicy>
class CPSfermion5D: public CPSfield<mf_Complex,12,MappingPolicy,AllocPolicy>{
  StaticAssert<MappingPolicy::EuclideanDimension == 5> check;
public:  
  typedef CPSfield<mf_Complex,12,MappingPolicy,AllocPolicy> BaseType;
  INHERIT_TYPEDEFS(BaseType);
  CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,CPSfermion5D);
    
#ifdef USE_BFM
  template<typename FloatExt>
  void importFermion(const Fermion_t bfm_field, const int cb, bfm_qdp<FloatExt> &dwf);

  template<typename FloatExt>
  void exportFermion(const Fermion_t bfm_field, const int cb, bfm_qdp<FloatExt> &dwf) const;
#endif

  void setGaussianRandom();
  
  DEFINE_ADDSUB_DERIVED(CPSfermion5D);
};


template< typename mf_Complex, typename MappingPolicy = FourDpolicy<DynamicFlavorPolicy>, typename AllocPolicy = StandardAllocPolicy>
class CPScomplex4D: public CPSfield<mf_Complex,1,MappingPolicy,AllocPolicy>{
  StaticAssert<MappingPolicy::EuclideanDimension == 4> check;
public:
  typedef CPSfield<mf_Complex,1,MappingPolicy,AllocPolicy> BaseType;
  INHERIT_TYPEDEFS(BaseType);
  CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,CPScomplex4D);  

  //Make a random complex scalar field of type
  void setRandom(const RandomType &type);

  //Set the real and imaginary parts to uniform random numbers drawn from the appropriate local RNGs
  void setUniformRandom(const Float &hi = 0.5, const Float &lo = -0.5);

  DEFINE_ADDSUB_DERIVED(CPScomplex4D);
};

//3d complex number field
template< typename mf_Complex, typename FlavorPolicy = DynamicFlavorPolicy, typename AllocPolicy = StandardAllocPolicy>
class CPScomplexSpatial: public CPSfield<mf_Complex,1,SpatialPolicy<FlavorPolicy>,AllocPolicy>{
  typedef SpatialPolicy<FlavorPolicy> MappingPolicy;
public:
  typedef CPSfield<mf_Complex,1,MappingPolicy,AllocPolicy> BaseType;
  INHERIT_TYPEDEFS(BaseType);
  CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,CPScomplexSpatial);  
  DEFINE_ADDSUB_DERIVED(CPScomplexSpatial);
};

//Lattice-spanning 'global' 3d complex field
template< typename mf_Complex, typename FlavorPolicy = DynamicFlavorPolicy, typename AllocPolicy = StandardAllocPolicy>
class CPSglobalComplexSpatial: public CPSfield<mf_Complex,1,GlobalSpatialPolicy<FlavorPolicy>,AllocPolicy>{
  typedef GlobalSpatialPolicy<FlavorPolicy> MappingPolicy;
public:
  typedef CPSfield<mf_Complex,1,MappingPolicy,AllocPolicy> BaseType;
  INHERIT_TYPEDEFS(BaseType);
  CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,CPSglobalComplexSpatial);  
  
  //Perform the FFT
  void fft();

  //Scatter to a local field
  template<typename extComplex, typename extMapPolicy, typename extAllocPolicy>
  void scatter(CPSfield<extComplex,1,extMapPolicy,extAllocPolicy> &to) const;

  DEFINE_ADDSUB_DERIVED(CPSglobalComplexSpatial);
};


//This field contains an entire row of sub-lattices along a particular dimension. Every node along that row contains an identical copy
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy = StandardAllocPolicy>
class CPSfieldGlobalInOneDir: public CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>{
public:
  typedef CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy> BaseType;
  INHERIT_TYPEDEFS(BaseType);
  CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,CPSfieldGlobalInOneDir);

  //Gather up the row. Involves internode communication
  template<typename extSiteType, typename extMapPol, typename extAllocPol>
  void gather(const CPSfield<extSiteType,SiteSize,extMapPol,extAllocPol> &from);

  //Scatter back out. Involves no communication
  template<typename extSiteType, typename extMapPol, typename extAllocPol>
  void scatter(CPSfield<extSiteType,SiteSize,extMapPol,extAllocPol> &to) const;

  //Perform a fast Fourier transform along the principal direction. It currently assumes the MappingPolicy has the sites mapped in canonical ordering
  void fft(const bool inverse_transform = false);

  DEFINE_ADDSUB_DERIVED(CPSfieldGlobalInOneDir);
};

template< typename mf_Complex, typename FlavorPolicy = DynamicFlavorPolicy, typename AllocPolicy = StandardAllocPolicy>
class CPSfermion4DglobalInOneDir: public CPSfieldGlobalInOneDir<mf_Complex,12,FourDglobalInOneDir<FlavorPolicy>,AllocPolicy>{
public:
  typedef CPSfieldGlobalInOneDir<mf_Complex,12,FourDglobalInOneDir<FlavorPolicy>,AllocPolicy> BaseType;
  INHERIT_TYPEDEFS(BaseType);
  CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,CPSfermion4DglobalInOneDir);  
  DEFINE_ADDSUB_DERIVED(CPSfermion4DglobalInOneDir);
};
template< typename mf_Complex, typename FlavorPolicy = DynamicFlavorPolicy, typename AllocPolicy = StandardAllocPolicy>
class CPSfermion3DglobalInOneDir: public CPSfieldGlobalInOneDir<mf_Complex,12,ThreeDglobalInOneDir<FlavorPolicy>,AllocPolicy>{
public:
  typedef CPSfieldGlobalInOneDir<mf_Complex,12,ThreeDglobalInOneDir<FlavorPolicy>,AllocPolicy> BaseType;
  INHERIT_TYPEDEFS(BaseType);
  CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,CPSfermion3DglobalInOneDir);
  DEFINE_ADDSUB_DERIVED(CPSfermion3DglobalInOneDir);
};


////////Checkerboarded types/////////////
template< typename mf_Complex, typename CBpolicy, typename FlavorPolicy = DynamicFlavorPolicy, typename AllocPolicy = StandardAllocPolicy>
class CPSfermion5Dprec: public CPSfield<mf_Complex,12,FiveDevenOddpolicy<CBpolicy,FlavorPolicy>,AllocPolicy>{
public:
  typedef CPSfield<mf_Complex,12,FiveDevenOddpolicy<CBpolicy,FlavorPolicy>,AllocPolicy> BaseType;
  INHERIT_TYPEDEFS(BaseType);
  CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,CPSfermion5Dprec);
  DEFINE_ADDSUB_DERIVED(CPSfermion5Dprec);
};


template< typename mf_Complex, typename FlavorPolicy = DynamicFlavorPolicy, typename AllocPolicy = StandardAllocPolicy>
class CPSfermion5Dcb4Deven: public CPSfermion5Dprec<mf_Complex,CheckerBoard<4,0>,FlavorPolicy,AllocPolicy>{
public:
  typedef CPSfermion5Dprec<mf_Complex,CheckerBoard<4,0>,FlavorPolicy,AllocPolicy> BaseType;
  INHERIT_TYPEDEFS(BaseType);
  CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,CPSfermion5Dcb4Deven);
  DEFINE_ADDSUB_DERIVED(CPSfermion5Dcb4Deven);
};
template< typename mf_Complex, typename FlavorPolicy = DynamicFlavorPolicy, typename AllocPolicy = StandardAllocPolicy>
class CPSfermion5Dcb4Dodd: public CPSfermion5Dprec<mf_Complex,CheckerBoard<4,1>,FlavorPolicy,AllocPolicy>{
public:
  typedef CPSfermion5Dprec<mf_Complex,CheckerBoard<4,1>,FlavorPolicy,AllocPolicy> BaseType;
  INHERIT_TYPEDEFS(BaseType);
  CPSFIELD_DERIVED_DEFINE_CONSTRUCTORS_AND_COPY_ASSIGNMENT(BaseType,CPSfermion5Dcb4Dodd);
  DEFINE_ADDSUB_DERIVED(CPSfermion5Dcb4Dodd);
};





#include "implementation/CPSfield_impl.tcc"
#include "implementation/CPSfield_copy.tcc"
#include "implementation/CPSfield_io.tcc"

CPS_END_NAMESPACE
#endif
