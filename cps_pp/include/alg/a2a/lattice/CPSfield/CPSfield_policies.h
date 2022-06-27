#ifndef CPS_FIELD_POLICIES
#define CPS_FIELD_POLICIES

#include <malloc.h>
#include <alg/a2a/utils/utils_generic.h>
#include <alg/a2a/utils/utils_malloc.h>
#include <alg/a2a/utils/template_wizardry.h>
#include <alg/a2a/utils/utils_gpu.h>

CPS_START_NAMESPACE

template<typename polA, typename polB>
struct sameDim{ static const bool val = intEq<polA::EuclideanDimension, polB::EuclideanDimension>::val; };


inline void writePolicyName(std::ostream &file, const std::string &policy, const std::string &name, bool newline = true){
  file << policy << " = " << name << (newline ? "\n" : "");
}

inline void checkPolicyName(std::istream &file, const std::string &policy, const std::string &name){
  std::string tmp; getline(file,tmp);
  std::ostringstream expect; writePolicyName(expect,policy,name,false);
  if(tmp != expect.str()){ printf("checkPolicyName expected \"%s\" got \"%s\"\n",expect.str().c_str(), tmp.c_str()); fflush(stdout); exit(-1); }
}


//AllocPolicy controls mem alloc
class StandardAllocPolicy{
protected:
  inline static void _alloc(void** p, const size_t byte_size){
    *p = smalloc("CPSfield", "CPSfield", "alloc" , byte_size);
  }
  inline static void _free(void* p){
    sfree("CPSfield","CPSfield","free",p);
  }

  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "ALLOCPOLICY", "StandardAllocPolicy");
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "ALLOCPOLICY", "StandardAllocPolicy");
  }
public:
  enum { UVMenabled = 0 }; //doesnt' support UVM
};
class Aligned128AllocPolicy{
  void* _ptr;
  size_t _byte_size;

protected:
  inline void _alloc(void** p, const size_t byte_size){
    *p = managed_alloc_check(128,byte_size); //note CUDA ignores alignment
    _ptr = *p;
    _byte_size = byte_size;
  }
  inline static void _free(void* p){
    managed_free(p);
  }
  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "ALLOCPOLICY", "Aligned128AllocPolicy");
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "ALLOCPOLICY", "Aligned128AllocPolicy");
  }
  
public: 
  inline void deviceSetAdviseUVMreadOnly(const bool to) const{
    if(to) device_UVM_advise_readonly(_ptr, _byte_size);
    else device_UVM_advise_unset_readonly(_ptr, _byte_size);
  }
  
  enum { UVMenabled = 1 }; //supports UVM
};
class NullAllocPolicy{
protected:
  inline static void _alloc(void** p, const size_t byte_size){
    *p = NULL;
  }
  inline static void _free(void* p){
  }

  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "ALLOCPOLICY", "NullAllocPolicy");
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "ALLOCPOLICY", "NullAllocPolicy");
  }

public:
  enum { UVMenabled = 0 }; //no data so copy is free
};
class ManualAllocPolicy{
  void** ptr;
  std::size_t bs;
protected:
  inline void _alloc(void** p, const size_t byte_size){
    ptr = p; bs = byte_size; *p = NULL;
  }
  inline static void _free(void* p){
    if(p!=NULL) sfree("CPSfield","CPSfield","free",p);
  }

  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "ALLOCPOLICY", "ManualAllocPolicy");
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "ALLOCPOLICY", "ManualAllocPolicy");
  }

public:
  inline void allocField(){
    if(*ptr == NULL)
      *ptr = smalloc("CPSfield", "CPSfield", "alloc" , bs);
  }
  inline void freeField(){
    if(*ptr != NULL){
      sfree("CPSfield","CPSfield","free",*ptr);
      *ptr = NULL;
    }
  }
  enum { UVMenabled = 0 }; //doesnt' support UVM
};
class ManualAligned128AllocPolicy{
  void** ptr;
  std::size_t bs;
protected:
  inline void _alloc(void** p, const size_t byte_size){
    ptr = p; bs = byte_size; *p = NULL;
  }
  inline static void _free(void* p){
    if(p!=NULL) managed_free(p);
  }
public:
  inline void allocField(){
    if(*ptr == NULL)
      *ptr = managed_alloc_check(128,bs);    
  }
  inline void freeField(){
    if(*ptr != NULL){
      managed_free(*ptr);
      *ptr = NULL;
    }
  }
  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "ALLOCPOLICY", "ManualAligned128AllocPolicy");
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "ALLOCPOLICY", "ManualAligned128AllocPolicy");
  }
  enum { UVMenabled = 1 }; //supports UVM
};


//The FlavorPolicy allows the number of flavors to be fixed or 2/1 if Gparity/noGparity 
//Note if Nf==1 the flavor should be set using the second template parameter
template<int Nf, int f=0>
class FixedFlavorPolicy{
protected:
  inline void writeParams(std::ostream &file) const{
    std::ostringstream os; os << "FixedFlavorPolicy<" << Nf;
    if(Nf == 1) os << "," << f;
    os << '>';
    writePolicyName(file, "FLAVORPOLICY", os.str());
  }
  inline void readParams(std::istream &file){
    std::ostringstream os; os << "FixedFlavorPolicy<" << Nf;
    if(Nf == 1) os << "," << f;
    os << '>';
    checkPolicyName(file, "FLAVORPOLICY", os.str());
  }
public:
  accelerator_inline int nflavors() const{ return Nf; }
  accelerator_inline bool hasFlavor(int flav) const{ return Nf == 2 || flav == f; }
};
typedef FixedFlavorPolicy<1> OneFlavorPolicy;

//Default is to use two flavors if GPBC, 1 otherwise
class DynamicFlavorPolicy{
protected:
  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "FLAVORPOLICY", "DynamicFlavorPolicy");
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "FLAVORPOLICY", "DynamicFlavorPolicy");
  }
  int nf;
public:
  DynamicFlavorPolicy(): nf(GJP.Gparity() + 1){}
  accelerator_inline int nflavors() const{ return nf; }
  accelerator_inline bool hasFlavor(int flav) const{ return flav == 0 || nf == 2; }
};

#define _DEF_REBASE(P) \
  template<typename T> \
  struct Rebase{ \
    typedef P<T> type; \
  }


//The DimensionPolicy controls the mapping between an N-dimensional vector and a flavor index to an integer which is used to compute the pointer offset.
//Each policy contains 2 mappings; a linearization of a Euclidean vector to an index, and a linearization of the Euclidean vector plus a flavor index. The latter is used to compute pointer offsets, the former for convenient site looping
//We generically refer to 'sites' as being those of the Euclidean lattice, and fsites as those of the Euclidean+flavor lattice (as if flavor was another dimension)

template<typename FlavorPolicy  = DynamicFlavorPolicy>
class FourDpolicy: public FlavorPolicy{ //Canonical layout 4D field with second flavor stacked after full 4D block
  size_t vol4d;
  int node_sites[4];  
public:
  typedef FlavorPolicy FieldFlavorPolicy;
  _DEF_REBASE(FourDpolicy);
  
  accelerator_inline size_t nsites() const{ return vol4d; }
  accelerator_inline size_t nfsites() const{ return this->nflavors()*this->nsites(); }
  
  accelerator_inline size_t siteMap(const int x[]) const{ return x[0] + node_sites[0]*( x[1] + node_sites[1]*( x[2] + node_sites[2]*x[3])); }

  accelerator_inline void siteUnmap(size_t site, int x[]) const{
    for(size_t i=0;i<4;i++){ 
      x[i] = site % node_sites[i]; site /= node_sites[i];
    }
  }

  accelerator_inline size_t fsiteMap(const int x[], const int f) const{ return siteMap(x) + f*vol4d; }

  accelerator_inline void fsiteUnmap(size_t fsite, int x[], int &f) const{
    siteUnmap(fsite,x); f = fsite / vol4d;
  }

  accelerator_inline size_t fsiteFlavorOffset() const{ return vol4d; } //increment of linearized coordinate between flavors
  accelerator_inline size_t dimpol_site_stride_3d() const{ return 1; }
 
  accelerator_inline size_t siteFsiteConvert(const size_t site, const int f) const{ return site + vol4d*f; } //convert a site-flavor pair to an fsite

  accelerator_inline int nodeSites(const int dir) const{ return node_sites[dir]; }

  accelerator_inline bool contains(const int x[], int flav = 0) const { return this->hasFlavor(flav); }
  
  typedef NullObject ParamType;
  FourDpolicy(){
    vol4d = GJP.VolNodeSites();
    for(int i=0;i<4;i++) node_sites[i] = GJP.NodeSites(i);
  }
  FourDpolicy(const ParamType &p): FourDpolicy(){}

  const static int EuclideanDimension = 4;

  accelerator_inline size_t threeToFour(const size_t x3d, const int t) const{ return x3d + vol4d/node_sites[3]*t; } //convert 3d index to 4d index

  ParamType getDimPolParams() const{ return ParamType(); }

  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "DIMENSIONPOLICY", "FourDpolicy");
    this->FlavorPolicy::writeParams(file);
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "DIMENSIONPOLICY", "FourDpolicy");
    this->FlavorPolicy::readParams(file);
  }
};
//Canonical layout 5D field. The fsite second flavor is stacked inside the s-loop. The site is just linearized in the canonical format
template<typename FlavorPolicy  = DynamicFlavorPolicy>
class FiveDpolicy: public FlavorPolicy{ 
  size_t vol4d;
  size_t vol5d;
  int node_sites[5];  
public:
  typedef FlavorPolicy FieldFlavorPolicy;
  _DEF_REBASE(FiveDpolicy);
  
  accelerator_inline size_t nsites() const{ return vol5d; }
  accelerator_inline size_t nfsites() const{ return this->nflavors()*this->nsites(); }
  
  //Note this does not correspond to the ordering of the data in memory. To compute the offset use fsiteMap or siteMap followed by siteFsiteConvert
  accelerator_inline size_t siteMap(const int x[]) const{ return x[0] + node_sites[0]*( x[1] + node_sites[1]*( x[2] + node_sites[2]*(x[3] + node_sites[3]*x[4]))); }

  accelerator_inline void siteUnmap(size_t site, int x[]) const{
    for(int i=0;i<5;i++){ 
      x[i] = site % node_sites[i]; site /= node_sites[i];
    }
  }

  accelerator_inline size_t fsiteMap(const int x[], const int f) const{ return x[0] + node_sites[0]*( x[1] + node_sites[1]*( x[2] + node_sites[2]*(x[3] + node_sites[3]*( f + this->nflavors()*x[4]) ))); }

  accelerator_inline void fsiteUnmap(size_t fsite, int x[], int &f) const{
    for(int i=0;i<4;i++){ 
      x[i] = fsite % node_sites[i]; fsite /= node_sites[i];
    }
    f = fsite % this->nflavors(); fsite /= this->nflavors();
    x[4] = fsite;
  }

  accelerator_inline size_t fsiteFlavorOffset() const{ return vol4d; /*Flavor index inside s-index*/   }

  accelerator_inline size_t siteFsiteConvert(const size_t site, const int f) const{
    size_t x4d = site % vol4d;
    size_t s = site / vol4d;
    return x4d + vol4d*(f + this->nflavors()*s);
  }

  accelerator_inline int nodeSites(const int dir) const{ return node_sites[dir]; }
  
  accelerator_inline bool contains(const int x[], int flav = 0) const { return this->hasFlavor(flav); }

  typedef NullObject ParamType;
  FiveDpolicy(){
    vol4d = GJP.VolNodeSites();
    vol5d = GJP.VolNodeSites()*GJP.SnodeSites();
    for(int i=0;i<5;i++) node_sites[i] = GJP.NodeSites(i);
  }

  FiveDpolicy(const ParamType &p): FiveDpolicy(){}

  const static int EuclideanDimension = 5;

  ParamType getDimPolParams() const{ return ParamType(); }
  
  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "DIMENSIONPOLICY", "FiveDpolicy");
    this->FlavorPolicy::writeParams(file);
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "DIMENSIONPOLICY", "FiveDpolicy");
    this->FlavorPolicy::readParams(file);
  }
};

template<typename FlavorPolicy = DynamicFlavorPolicy>
class FourDglobalInOneDir: public FlavorPolicy{ //4D field where one direction 'dir' spans the entire lattice on each node separately. The ordering is setup so that the 'dir' points are blocked (change most quickly)
  int lmap[4]; //map of local dimension to physical X,Y,Z,T dimension. e.g.  [1,0,2,3] means local dimension 0 is the Y dimension, local dimension 1 is the X-direction and so on
  int dims[4]; //node sites
  int dir; //direction in which it is global
  size_t dvol; //node volume

  void setDir(const int &_dir){
    dir = _dir;
    for(int i=0;i<4;i++) lmap[i] = i;
    std::swap(lmap[dir],lmap[0]); //make dir direction change fastest

    for(int i=0;i<4;i++) dims[i] = GJP.NodeSites(lmap[i]);
    dims[0] *= GJP.Nodes(dir);

    dvol = dims[0]*dims[1]*dims[2]*dims[3];
  }
public:
  typedef FlavorPolicy FieldFlavorPolicy;
  _DEF_REBASE(FourDglobalInOneDir);
  
  accelerator_inline size_t nsites() const{ return dvol; }
  accelerator_inline size_t nfsites() const{ return this->nflavors()*this->nsites(); }
  
  accelerator_inline size_t siteMap(const int x[]) const{ return x[lmap[0]] + dims[0]*( x[lmap[1]] + dims[1]*( x[lmap[2]] + dims[2]*x[lmap[3]])); }

  accelerator_inline void siteUnmap(size_t site, int x[]) const{
    for(int i=0;i<4;i++){ 
      x[lmap[i]] = site % dims[i]; site /= dims[i];
    }
  }

  accelerator_inline size_t fsiteMap(const int x[], const int f) const{ return siteMap(x) + dvol*f; }

  accelerator_inline void fsiteUnmap(size_t fsite, int x[], int &f) const{
    siteUnmap(fsite,x);
    f = fsite/dvol;
  }

  accelerator_inline size_t fsiteFlavorOffset() const{ return dvol; }

  accelerator_inline size_t siteFsiteConvert(const size_t site, const int f) const{ 
    return site + dvol * f;
  }

  accelerator_inline bool contains(const int x[], int flav = 0) const { return this->hasFlavor(flav); }

  typedef int ParamType;

  accelerator_inline int getDir() const{ return dir; }

  FourDglobalInOneDir(const int &_dir): dir(-1){
    setDir(_dir);
  }
  const static int EuclideanDimension = 4;

  ParamType getDimPolParams() const{ return dir; }

  typedef FourDpolicy<FlavorPolicy> EquivalentLocalPolicy;

  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "DIMENSIONPOLICY", "FourDglobalInOneDir");
    file << "LMAP = " << lmap[0] << " " << lmap[1] << " " << lmap[2] << " " << lmap[3] << "\n";
    file << "DIMS = " << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3] << "\n";
    file << "DIR = " << dir << "\n";
    file << "DVOL = " << dvol << "\n";
    this->FlavorPolicy::writeParams(file);
  }
  inline void readParams(std::istream &file){ //overwrite existing dir
    checkPolicyName(file, "DIMENSIONPOLICY", "FourDglobalInOneDir");
    std::string str;
    getline(file,str); assert( sscanf(str.c_str(),"LMAP = %d %d %d %d",&lmap[0],&lmap[1],&lmap[2],&lmap[3]) == 4 );
    getline(file,str); assert( sscanf(str.c_str(),"DIMS = %d %d %d %d",&dims[0],&dims[1],&dims[2],&dims[3]) == 4 );
    getline(file,str); assert( sscanf(str.c_str(),"DIR = %d",&dir) == 1 );
    getline(file,str); assert( sscanf(str.c_str(),"DVOL = %d",&dvol) == 1 );
    this->FlavorPolicy::readParams(file);
  }
};

template<typename FlavorPolicy = DynamicFlavorPolicy >
class SpatialPolicy: public FlavorPolicy{ //Canonical layout 3D field
  size_t threevol;
  int node_sites[3];
public:
  typedef FlavorPolicy FieldFlavorPolicy;
  _DEF_REBASE(SpatialPolicy);
  
  accelerator_inline size_t nsites() const{ return threevol; }
  accelerator_inline size_t nfsites() const{ return this->nflavors()*this->nsites(); }
  
  accelerator_inline size_t siteMap(const int x[]) const{ return x[0] + node_sites[0]*( x[1] + node_sites[1]*x[2]); }
  accelerator_inline void siteUnmap(size_t site, int x[]) const{
    for(int i=0;i<3;i++){ 
      x[i] = site % node_sites[i]; site /= node_sites[i];
    }
  }

  accelerator_inline size_t fsiteMap(const int x[], const int f) const{ return siteMap(x) + threevol*f; }

  accelerator_inline void fsiteUnmap(size_t fsite, int x[], int &f) const{
    siteUnmap(fsite,x);
    f = fsite/threevol;
  }

  accelerator_inline size_t fsiteFlavorOffset() const{ return threevol; }

  accelerator_inline size_t siteFsiteConvert(const size_t site, const int f) const{ 
    return site + threevol * f;
  }

  accelerator_inline int nodeSites(const int dir) const{ return node_sites[dir]; }
  
  accelerator_inline bool contains(const int x[], int flav = 0) const { return this->hasFlavor(flav); }

  typedef NullObject ParamType;
  SpatialPolicy():  threevol(GJP.VolNodeSites()/GJP.TnodeSites()){
    for(int i=0;i<3;i++) node_sites[i] = GJP.NodeSites(i);
  }
  SpatialPolicy(const ParamType &p): SpatialPolicy(){}

  const static int EuclideanDimension = 3;

  ParamType getDimPolParams() const{ return ParamType(); }

  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "DIMENSIONPOLICY", "SpatialPolicy");
    this->FlavorPolicy::writeParams(file);
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "DIMENSIONPOLICY", "SpatialPolicy");
    this->FlavorPolicy::readParams(file);
  }
};

template<typename FlavorPolicy = DynamicFlavorPolicy>
class GlobalSpatialPolicy: public FlavorPolicy{ //Global canonical 3D field
protected:
  int glb_size[3];
  size_t glb_vol;
public:
  typedef FlavorPolicy FieldFlavorPolicy;
    _DEF_REBASE(GlobalSpatialPolicy);
  
  accelerator_inline size_t nsites() const{ return glb_vol; }
  accelerator_inline size_t nfsites() const{ return this->nflavors()*this->nsites(); }
  
  accelerator_inline size_t siteMap(const int x[]) const{ return x[0] + glb_size[0]*( x[1] + glb_size[1]*x[2]); }

  accelerator_inline void siteUnmap(size_t site, int x[]){
    for(int i=0;i<3;i++){ 
      x[i] = site % glb_size[i]; site /= glb_size[i];
    }
  }

  accelerator_inline size_t fsiteMap(const int x[], const int f) const{ return siteMap(x) + f*glb_vol; }

  accelerator_inline void fsiteUnmap(size_t fsite, int x[], int &f){
    siteUnmap(fsite,x);
    f = fsite / glb_vol;
  }

  accelerator_inline size_t siteFsiteConvert(const size_t site, const int f) const{ 
    return site + glb_vol * f;
  }

  accelerator_inline size_t fsiteFlavorOffset() const{ return glb_vol; }

  accelerator_inline size_t nodeSites(const int dir) const{ return glb_size[dir]; }

  accelerator_inline bool contains(const int x[], int flav = 0) const { return this->hasFlavor(flav); }

  typedef NullObject ParamType;
  GlobalSpatialPolicy(){
    for(int i=0;i<3;i++) glb_size[i] = GJP.NodeSites(i)*GJP.Nodes(i);
    glb_vol = glb_size[0]*glb_size[1]*glb_size[2];
  }
  GlobalSpatialPolicy(const ParamType &p): GlobalSpatialPolicy(){}

  const static int EuclideanDimension = 3;

  ParamType getDimPolParams() const{ return ParamType(); }

  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "DIMENSIONPOLICY", "GlobalSpatialPolicy");
    this->FlavorPolicy::writeParams(file);
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "DIMENSIONPOLICY", "GlobalSpatialPolicy");
    this->FlavorPolicy::readParams(file);
  }
  
};

template<typename FlavorPolicy = DynamicFlavorPolicy>
class ThreeDglobalInOneDir: public FlavorPolicy{ //3D field where one direction 'dir' spans the entire lattice on each node separately. The ordering is setup so that the 'dir' points are blocked (change most quickly)
  int lmap[3]; //map of local dimension to physical X,Y,Z dimension. e.g.  [1,0,2] means local dimension 0 is the Y dimension, local dimension 1 is the X-direction and so on
  int dims[3]; //node sites
  int dir; //direction in which global
  size_t dvol; //node volume

  void setDir(const int &_dir){
    dir = _dir;
    for(int i=0;i<3;i++) lmap[i] = i;
    std::swap(lmap[dir],lmap[0]); //make dir direction change fastest

    for(int i=0;i<3;i++) dims[i] = GJP.NodeSites(lmap[i]);
    dims[0] *= GJP.Nodes(dir);

    dvol = dims[0]*dims[1]*dims[2];
  }
public:
  typedef FlavorPolicy FieldFlavorPolicy;
  _DEF_REBASE(ThreeDglobalInOneDir);
  
  accelerator_inline size_t nsites() const{ return dvol; }
  accelerator_inline size_t nfsites() const{ return this->nflavors()*this->nsites(); }
  
  accelerator_inline size_t siteMap(const int x[]) const{ return x[lmap[0]] + dims[0]*( x[lmap[1]] + dims[1]*x[lmap[2]]); }
  accelerator_inline void siteUnmap(size_t site, int x[]) const{
    for(int i=0;i<3;i++){ 
      x[lmap[i]] = site % dims[i]; site /= dims[i];
    }
  }

  accelerator_inline size_t fsiteMap(const int x[], const int f) const{ return siteMap(x) + f*dvol; }

  accelerator_inline void fsiteUnmap(size_t fsite, int x[], int &f) const{
    siteUnmap(fsite,x);
    f = fsite / dvol;
  }

  accelerator_inline size_t fsiteFlavorOffset() const{ return dvol; }

  accelerator_inline size_t siteFsiteConvert(const size_t site, const int f) const{ 
    return site + dvol * f;
  }

  accelerator_inline bool contains(const int x[], int flav = 0) const { return this->hasFlavor(flav); }

  typedef int ParamType;

  accelerator_inline int getDir() const{ return dir; }

  ThreeDglobalInOneDir(const int &_dir): dir(-1){
    setDir(_dir);
  }
  const static int EuclideanDimension = 3;

  ParamType getDimPolParams() const{ return dir; }

  typedef SpatialPolicy<FlavorPolicy> EquivalentLocalPolicy;

  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "DIMENSIONPOLICY", "ThreeDglobalInOneDir");
    file << "LMAP = " << lmap[0] << " " << lmap[1] << " " << lmap[2] << "\n";
    file << "DIMS = " << dims[0] << " " << dims[1] << " " << dims[2] << "\n";
    file << "DIR = " << dir << "\n";
    file << "DVOL = " << dvol << "\n";
    this->FlavorPolicy::writeParams(file);
  }
  inline void readParams(std::istream &file){ //overwrite existing dir
    checkPolicyName(file, "DIMENSIONPOLICY", "ThreeDglobalInOneDir");
    std::string str;
    getline(file,str); assert( sscanf(str.c_str(),"LMAP = %d %d %d",&lmap[0],&lmap[1],&lmap[2]) == 3 );
    getline(file,str); assert( sscanf(str.c_str(),"DIMS = %d %d %d",&dims[0],&dims[1],&dims[2]) == 3 );
    getline(file,str); assert( sscanf(str.c_str(),"DIR = %d",&dir) == 1 );
    getline(file,str); assert( sscanf(str.c_str(),"DVOL = %d",&dvol) == 1 );
    this->FlavorPolicy::readParams(file);
  }
};



//////////////////////////
//Checkerboarding
template<int _CBdim, int _CB>   //CBdim is the amount of elements of the coordinate vector (starting at 0) included in the computation (i.e. 4d or 5d even-odd),  CB is the checkerboard
class CheckerBoard{
public:
  enum { CB = _CB, CBdim = _CBdim };

  CheckerBoard(){
    //Currently require even-sized sublattices- This needs to be generalized
    assert(GJP.NodeSites(0)%2==0 && GJP.NodeSites(1)%2==0 && GJP.NodeSites(2)%2==0 && GJP.NodeSites(3)%2==0 && CBdim > 4 ? GJP.NodeSites(4)%2==0 : true);
  }
  
  accelerator_inline int cb() const{ return _CB; }
  accelerator_inline int cbDim() const{ return _CBdim; }
  accelerator_inline bool onCb(const int x[]) const{
    size_t c = 0; for(int i=0;i<_CBdim;i++) c += x[i];
    return c % 2 == _CB;
  }

  inline std::string getPolicyName() const{ std::ostringstream os; os << "CheckerBoard<" << _CBdim << "," << _CB << ">"; return os.str(); }
  
};

template<typename T>
class CPSfieldIsCheckerboarded
{
  template <typename U, U> struct Check;
  template <typename U> static char func(Check<int, U::CB> *);
  template <typename U> static int func(...);
public:
  enum { value = sizeof(func<T>(0)) == sizeof(char) };
};



//Checkerboarded 5D field. The fsite second flavor is stacked inside the s-loop. The checkerboard dimension and which checkerboard it is are handled by the policy CheckerBoard
//Note, the mappings do not check that the site is on the checkerboard; you should do that using onCb(x[])
template<typename CheckerBoardType,typename FlavorPolicy = DynamicFlavorPolicy>
class FiveDevenOddpolicy: public CheckerBoardType, public FlavorPolicy{ 
  size_t vol4d; //uncheckerboarded
  size_t vol5d; //uncheckerboarded
  int node_sites[5];
public:
  typedef FlavorPolicy FieldFlavorPolicy;
  _DEF_REBASE(FiveDevenOddpolicy);
  
  accelerator_inline size_t nsites() const{ return vol5d/2; }
  accelerator_inline size_t nfsites() const{ return this->nflavors()*this->nsites(); }
  
  accelerator_inline size_t siteMap(const int x[]) const{ 
    return (x[0] + node_sites[0]*( x[1] + node_sites[1]*( x[2] + node_sites[2]*(x[3] + node_sites[3]*x[4]))))/2; 
  }

  accelerator_inline void siteUnmap(size_t site, int x[]) const{
    site *= 2;
    for(int i=0;i<5;i++){ 
      x[i] = site % node_sites[i]; site /= node_sites[i];
    }
    if(!this->onCb(x)) x[0] += 1; //deal with int convert x[0]/2 giving same number for 0,1 etc
  }

  accelerator_inline size_t fsiteMap(const int x[], const int f) const{ 
    return (x[0] + node_sites[0]*( x[1] + node_sites[1]*( x[2] + node_sites[2]*(x[3] + node_sites[3]*( f + this->nflavors()*x[4]) ))))/2;
  }

  accelerator_inline void fsiteUnmap(size_t fsite, int x[], int &f) const{
    fsite *= 2;
    for(int i=0;i<4;i++){ 
      x[i] = fsite % node_sites[i]; fsite /= node_sites[i];
    }
    f = fsite % this->nflavors(); fsite /= this->nflavors();
    x[4] = fsite;
    if(!this->onCb(x)) x[0] += 1; //deal with int convert x[0]/2 giving same number for 0,1 etc
  }

  accelerator_inline size_t fsiteFlavorOffset() const{ return vol4d/2; }

  accelerator_inline size_t siteFsiteConvert(const size_t site, const int f) const{ 
    size_t x4d = site % (vol4d/2); //checkerboarded
    size_t s = site / (vol4d/2);
    return x4d + vol4d*(f + this->nflavors()*s)/2;
  }

  accelerator_inline bool contains(const int x[], int flav = 0) const { return this->onCb(x) && this->hasFlavor(flav); }

  typedef NullObject ParamType;
  FiveDevenOddpolicy(){
    vol4d = GJP.VolNodeSites();
    vol5d = GJP.VolNodeSites()*GJP.SnodeSites();
    for(int i=0;i<5;i++) node_sites[i] = GJP.NodeSites(i);    
  }
  FiveDevenOddpolicy(const ParamType &p): FiveDevenOddpolicy(){}

  const static int EuclideanDimension = 5;

  ParamType getDimPolParams() const{ return ParamType(); }

  inline void writeParams(std::ostream &file) const{
    std::ostringstream os; os << "FiveDevenOddpolicy< " << this->CheckerBoardType::getPolicyName() << " >";    
    writePolicyName(file, "DIMENSIONPOLICY", os.str());
    this->FlavorPolicy::writeParams(file);
  }
  inline void readParams(std::istream &file){
    std::ostringstream os; os << "FiveDevenOddpolicy< " << this->CheckerBoardType::getPolicyName() << " >";  
    checkPolicyName(file, "DIMENSIONPOLICY", os.str());
    this->FlavorPolicy::readParams(file);
  }
};

template<int N>
class SIMDdims{
  int v[N];
public:
  accelerator_inline int & operator[](const int i){ return v[i]; }
  accelerator_inline int operator[](const int i) const{ return v[i]; }
  accelerator_inline int* ptr(){ return &v[0]; }
  accelerator_inline int const* ptr() const{ return &v[0]; }
  accelerator_inline void set(const int* f){ for(int i=0;i<N;i++) v[i] = f[i]; }
  SIMDdims(){ for(int i=0;i<N;i++) v[i] = 1; }
  SIMDdims(const int* f){ set(f); }
};
template<int N>
std::ostream & operator<<(std::ostream &os, const SIMDdims<N> &v){
  os << "SIMDdims(";
  for(int i=0;i<N-1;i++) os << v[i] << ", ";
  os << v[N-1] << ")";
  return os;
}

template<int Dimension>
class SIMDpolicyBase{
 public:
  typedef SIMDdims<Dimension> ParamType;
  
  //Given the list of base site pointers to be packed, apply the packing for n site elements
  template<typename Vtype, typename Stype>
  static inline void SIMDpack(Vtype *into, const std::vector<Stype const*> &from, const int n = 1){
    int nsimd = Vtype::Nsimd();
    typename Vtype::scalar_type tmp[nsimd];
    for(int idx=0;idx<n;idx++){ //offset of elements on site
      for(int s=0;s<nsimd;s++)
	tmp[s] = *(from[s] + idx); //gather from the different sites with fixed element offset
      vset(*(into+idx),tmp);
    }
  }
    
  template<typename Vtype, typename Stype>
  static inline void SIMDunpack(std::vector<Stype*> &into, const Vtype *from, const int n = 1){
    int nsimd = Vtype::Nsimd();
    typename Vtype::scalar_type* tmp = (typename Vtype::scalar_type*)memalign_check(128,nsimd*sizeof(typename Vtype::scalar_type));
    for(int idx=0;idx<n;idx++){ //offset of elements on site
      vstore(*(from+idx),tmp);      
      for(int s=0;s<nsimd;s++)
	*(into[s] + idx) = tmp[s];
    }
    free(tmp);
  }
  
  //Iteratively divide the dimensions over the SIMD lanes up to a chosen maximum dimension (so we can exclude the time dimension for example, by setting max_dim_idx = 2)
  inline static void SIMDdefaultLayout(ParamType &simd_dims, const int nsimd, const int max_dim_idx = Dimension-1){
    for(int i=0;i<Dimension;i++) simd_dims[i] = 1;
    if(nsimd == 1) return;

    assert(nsimd % 2 == 0);
    int rem = nsimd;
    int i=0;
    while(rem != 1){
      simd_dims[i] *= 2;
      rem /= 2;
      i = (i+1) % (max_dim_idx+1);
    }
    int p = 1;
    for(int i=0;i<Dimension;i++) p *= simd_dims[i];
    assert(p == nsimd);
  }
};

template<typename FlavorPolicy = DynamicFlavorPolicy>
class FourDSIMDPolicy: public SIMDpolicyBase<4>, public FlavorPolicy{ //4D field with the dimensions blocked into logical nodes to be mapped into elements of SIMD vectors
  int simd_dims[4]; //number of SIMD logical nodes in each direction
  int logical_dim[4]; //dimension of logical nodes
  size_t logical_vol;
  int nsimd;
public:
  typedef FlavorPolicy FieldFlavorPolicy;
  _DEF_REBASE(FourDSIMDPolicy);
  
  accelerator_inline size_t nsites() const{ return logical_vol; }
  accelerator_inline size_t nfsites() const{ return this->nflavors()*this->nsites(); }
  
  accelerator_inline int Nsimd() const{ return nsimd; }
  accelerator_inline int SIMDlogicalNodes(const int dir) const{ return simd_dims[dir]; } 
  
  //Coordinate of SIMD block containing full 4D site x
  accelerator_inline size_t siteMap(const int x[]) const{ return (x[0] % logical_dim[0]) + logical_dim[0]*(
											    (x[1] % logical_dim[1]) + logical_dim[1]*(
																      (x[2] % logical_dim[2]) + logical_dim[2]*(
																						(x[3] % logical_dim[3]))));
  }
  //Returns coordinate in logical volume. Other coordinates within SIMD vector can be found by adding logical_dim[i] up to simd_dims[i] times for each direction i
  accelerator_inline void siteUnmap(size_t site, int x[]) const{
    for(int i=0;i<4;i++){ 
      x[i] = site % logical_dim[i]; site /= logical_dim[i];
    }
  }

  //Offset in units of complex of the site x within the SIMD block
  accelerator_inline size_t SIMDmap(const int x[]) const{
    return (x[0] / logical_dim[0]) + simd_dims[0] * (
						 (x[1] / logical_dim[1]) + simd_dims[1]*(
										     (x[2] / logical_dim[2]) + simd_dims[2]*(
															 (x[3] / logical_dim[3]))));
  }
  //Returns an offset from the root site coordinate returned by siteUnmap for the site packed into SIMD index idx
  accelerator_inline void SIMDunmap(size_t idx, int x[]) const{
    x[0] = (idx % simd_dims[0]) * logical_dim[0]; idx /= simd_dims[0];
    x[1] = (idx % simd_dims[1]) * logical_dim[1]; idx /= simd_dims[1];
    x[2] = (idx % simd_dims[2]) * logical_dim[2]; idx /= simd_dims[2];	
    x[3] = (idx % simd_dims[3]) * logical_dim[3];
  }
    
  accelerator_inline size_t fsiteMap(const int x[], const int f) const{ return siteMap(x) + f*logical_vol; } //second flavor still stacked after first

  accelerator_inline void fsiteUnmap(size_t fsite, int x[], int &f) const{
    siteUnmap(fsite,x);
    f = fsite / logical_vol;
  }

  accelerator_inline size_t fsiteFlavorOffset() const{ return logical_vol; }
  accelerator_inline size_t dimpol_site_stride_3d() const{ return 1; }

  //Stride between consecutive timeslices
  accelerator_inline size_t dimpol_time_stride() const{ return logical_dim[0]*logical_dim[1]*logical_dim[2]; }

  //Return true if the data for a single flavor for a given timeslice is contiguous
  accelerator_inline static bool dimpol_flavor_timeslice_contiguous(){ return true; }
  
  accelerator_inline size_t siteFsiteConvert(const size_t site, const int f) const{ 
    return site + logical_vol * f;
  }

  accelerator_inline int nodeSites(const int dir) const{ return logical_dim[dir]; }

  accelerator_inline bool contains(const int x[], int flav = 0) const { return this->hasFlavor(flav); }
  
  typedef SIMDpolicyBase<4>::ParamType ParamType;

private:
  void setup(const ParamType &_simd_dims){
    logical_vol = 1;
    for(int i=0;i<4;i++){
      simd_dims[i] = _simd_dims[i];
      assert(GJP.NodeSites(i) % simd_dims[i] == 0);
      logical_dim[i] = GJP.NodeSites(i)/simd_dims[i];
      logical_vol *= logical_dim[i];
    }
    nsimd = simd_dims[0]*simd_dims[1]*simd_dims[2]*simd_dims[3];
  }
public:
  FourDSIMDPolicy(const ParamType &_simd_dims){
    setup(_simd_dims);
  }
  const static int EuclideanDimension = 4;

  //Convert space-time indices on logical volume
  accelerator_inline size_t threeToFour(const size_t x3d, const int t) const{ return x3d + logical_vol/logical_dim[3]*t; } //convert 3d index to 4d index

  accelerator_inline void fourToThree(size_t &x3d, int &t, const size_t x4d) const{ 
    size_t vol3d = logical_vol/logical_dim[3];
    x3d = x4d % vol3d;
    t = x4d / vol3d;
  }

  ParamType getDimPolParams() const{
    return ParamType(simd_dims);
  }

  typedef FourDpolicy<FlavorPolicy> EquivalentScalarPolicy;

  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "DIMENSIONPOLICY", "FourDSIMDPolicy");
    file << "SIMD_DIMS = " << simd_dims[0] << " " << simd_dims[1] << " " << simd_dims[2] << " " << simd_dims[3] << "\n";
    this->FlavorPolicy::writeParams(file);
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "DIMENSIONPOLICY", "FourDSIMDPolicy");
    ParamType rd_simd_dims; std::string str;
    getline(file,str); assert( sscanf(str.c_str(),"SIMD_DIMS = %d %d %d %d",&rd_simd_dims[0],&rd_simd_dims[1],&rd_simd_dims[2],&rd_simd_dims[3]) == 4 );
    setup(rd_simd_dims);
    this->FlavorPolicy::readParams(file);
  }
};



template<typename FlavorPolicy = DynamicFlavorPolicy>
class ThreeDSIMDPolicy: public SIMDpolicyBase<3>, public FlavorPolicy{ //3D field with the dimensions blocked into logical nodes to be mapped into elements of SIMD vectors
  int simd_dims[3]; //number of SIMD logical nodes in each direction
  int logical_dim[3]; //dimension of logical nodes
  size_t logical_vol;
  int nsimd;
public:
  typedef FlavorPolicy FieldFlavorPolicy;
   _DEF_REBASE(ThreeDSIMDPolicy);
  
  accelerator_inline size_t nsites() const{ return logical_vol; }
  accelerator_inline size_t nfsites() const{ return this->nflavors()*this->nsites(); }
  
  accelerator_inline int Nsimd() const{ return nsimd; }
  accelerator_inline int SIMDlogicalNodes(const int dir) const{ return simd_dims[dir]; }
  
  //Coordinate of SIMD block containing full 4D site x
  accelerator_inline size_t siteMap(const int x[]) const{ return (x[0] % logical_dim[0]) + logical_dim[0]*(
											    (x[1] % logical_dim[1]) + logical_dim[1]*(
																      (x[2] % logical_dim[2])));
  }
  //Returns coordinate in logical volume. Other coordinates within SIMD vector can be found by adding logical_dim[i] up to simd_dims[i] times for each direction i
  accelerator_inline void siteUnmap(size_t site, int x[]) const{
    for(int i=0;i<3;i++){ 
      x[i] = site % logical_dim[i]; site /= logical_dim[i];
    }
  }

  //Offset in units of complex of the site x within the SIMD block
  accelerator_inline size_t SIMDmap(const int x[]) const{
    return (x[0] / logical_dim[0]) + simd_dims[0] * (
						 (x[1] / logical_dim[1]) + simd_dims[1]*(
											 (x[2] / logical_dim[2])));
  }
  //Returns an offset from the root site coordinate returned by siteUnmap for the site packed into SIMD index idx
  accelerator_inline void SIMDunmap(size_t idx, int x[]) const{
    x[0] = (idx % simd_dims[0]) * logical_dim[0]; idx /= simd_dims[0];
    x[1] = (idx % simd_dims[1]) * logical_dim[1]; idx /= simd_dims[1];
    x[2] = (idx % simd_dims[2]) * logical_dim[2];
  }
    
  accelerator_inline size_t fsiteMap(const int x[], const int f) const{ return siteMap(x) + f*logical_vol; } //second flavor still stacked after first

  accelerator_inline void fsiteUnmap(size_t fsite, int x[], int &f) const{
    siteUnmap(fsite,x);
    f = fsite / logical_vol;
  }

  accelerator_inline size_t fsiteFlavorOffset() const{ return logical_vol; }

  accelerator_inline size_t siteFsiteConvert(const size_t site, const int f) const{ 
    return site + logical_vol * f;
  }

  accelerator_inline int nodeSites(const int dir) const{ return logical_dim[dir]; }

  accelerator_inline bool contains(const int x[], int flav = 0) const { return this->hasFlavor(flav); }

  typedef SIMDpolicyBase<3>::ParamType ParamType;

private:
  void setup(const ParamType &_simd_dims){
    logical_vol = 1;
    for(int i=0;i<3;i++){
      simd_dims[i] = _simd_dims[i];
      assert(GJP.NodeSites(i) % simd_dims[i] == 0);
      logical_dim[i] = GJP.NodeSites(i)/simd_dims[i];
      logical_vol *= logical_dim[i];
    }
    nsimd = simd_dims[0]*simd_dims[1]*simd_dims[2];
  }
public:
  ThreeDSIMDPolicy(const ParamType &_simd_dims){
    setup(_simd_dims);
  }
  const static int EuclideanDimension = 3;

  ParamType getDimPolParams() const{
    return ParamType(simd_dims);
  }

  typedef SpatialPolicy<FlavorPolicy> EquivalentScalarPolicy;

  inline void writeParams(std::ostream &file) const{
    writePolicyName(file, "DIMENSIONPOLICY", "ThreeDSIMDPolicy");
    file << "SIMD_DIMS = " << simd_dims[0] << " " << simd_dims[1] << " " << simd_dims[2] << "\n";
    this->FlavorPolicy::writeParams(file);
  }
  inline void readParams(std::istream &file){
    checkPolicyName(file, "DIMENSIONPOLICY", "ThreeDSIMDPolicy");
    ParamType rd_simd_dims; std::string str;
    getline(file,str); assert( sscanf(str.c_str(),"SIMD_DIMS = %d %d %d",&rd_simd_dims[0],&rd_simd_dims[1],&rd_simd_dims[2]) == 3 );
    setup(rd_simd_dims);
    this->FlavorPolicy::readParams(file);
  }
};


template<typename T>
class isSIMDdimensionPolicy{
  template<typename U, int (U::*)() const> struct SFINAE {};
  template<typename U> static char Test(SFINAE<U, &U::Nsimd>*);
  template<typename U> static std::pair<char,char> Test(...);
public:
  enum { value = sizeof(Test<T>(0)) == sizeof(char) };
};


//Structs to get a scalar mapping policy from a SIMD mapping policy
template<typename MappingPolicy, int is_SIMD = isSIMDdimensionPolicy<MappingPolicy>::value>
struct getScalarMappingPolicy{};

template<typename MappingPolicy>
struct getScalarMappingPolicy<MappingPolicy, 0>{
  typedef MappingPolicy type;
};
template<typename MappingPolicy>
struct getScalarMappingPolicy<MappingPolicy, 1>{
  typedef typename MappingPolicy::EquivalentScalarPolicy type;
};


//Some helper structs to get policies for common field types
template<int Nd, typename FlavorPolicy = DynamicFlavorPolicy>
struct StandardDimensionPolicy{};

template<typename FlavorPolicy>
struct StandardDimensionPolicy<3,FlavorPolicy>{
  typedef SpatialPolicy<FlavorPolicy> type;
};
template<typename FlavorPolicy>
struct StandardDimensionPolicy<4,FlavorPolicy>{
  typedef FourDpolicy<FlavorPolicy> type;
};
template<typename FlavorPolicy>
struct StandardDimensionPolicy<5,FlavorPolicy>{
  typedef FiveDpolicy<FlavorPolicy> type;
};

//Mapping between local and global-in-one-dir policies
template<typename LocalDimPol>
struct LocalToGlobalInOneDirMap{};

template<typename FlavorPolicy>
struct LocalToGlobalInOneDirMap<FourDpolicy<FlavorPolicy> >{
  typedef FourDglobalInOneDir<FlavorPolicy> type;
};
template<typename FlavorPolicy>
struct LocalToGlobalInOneDirMap<SpatialPolicy<FlavorPolicy> >{
  typedef ThreeDglobalInOneDir<FlavorPolicy> type;
};



template<int Dimension>
struct IncludeSite{
  accelerator_inline virtual bool query(const int x[Dimension], const int f = 0) const = 0;
};
template<int Dimension>
struct IncludeCBsite: public IncludeSite<Dimension>{
  int cb;
  int excludeflav;
  
  IncludeCBsite(const int _cb, int frestrict = -1): cb(_cb),  excludeflav(frestrict == -1 ? -1 : !frestrict){  //frestrict : restrict to a particular flavor
    for(int i=0;i<Dimension;i++) assert(GJP.NodeSites(i) % 2 == 0);
  }
  
  accelerator_inline bool query(const int x[Dimension], const int f = 0) const{
    size_t c = 0;
    for(int i=0;i<Dimension;i++) c += x[i];
    return c % 2 == cb && f != excludeflav;    
  }
};
template<>
struct IncludeCBsite<5>: public IncludeSite<5>{
  int cb;
  bool fived_prec;
  int excludeflav;
  
  IncludeCBsite(const int _cb, bool _fived_prec = false, int frestrict = -1): cb(_cb), fived_prec(_fived_prec),  excludeflav(frestrict == -1 ? -1 : !frestrict){
    for(int i=0;i<4;i++) assert(GJP.NodeSites(i) % 2 == 0);
    if(fived_prec) assert(GJP.SnodeSites() % 2 == 0);
  }
  
  accelerator_inline bool query(const int x[5], const int f = 0) const{
    size_t c = 0;
    for(int i=0;i<4;i++) c += x[i];
    if(fived_prec) c += x[4];
    return c % 2 == cb && f != excludeflav;        
  }
};


CPS_END_NAMESPACE
#endif
