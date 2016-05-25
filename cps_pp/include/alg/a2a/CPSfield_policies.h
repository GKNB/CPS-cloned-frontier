#ifndef CPS_FIELD_POLICIES
#define CPS_FIELD_POLICIES
CPS_START_NAMESPACE

class NullObject
{
 public:
  NullObject(){}
};
template<typename polA, typename polB>
struct sameDim{ static const bool val = intEq<polA::EuclideanDimension, polB::EuclideanDimension>::val; };

//AllocPolicy controls mem alloc
class StandardAllocPolicy{
 protected:
  inline static void* _alloc(const size_t byte_size){
    return smalloc("CPSfield", "CPSfield", "alloc" , byte_size);
  }
  inline static _free(void* p){
    sfree("CPSfield","CPSfield","free",p);
  }
};
class Aligned128AllocPolicy{
 protected:
  inline static void* _alloc(const size_t byte_size){
    return memalign(128,byte_size);
  }
  inline static _free(void* p){
    free(p);
  }
};

//The FlavorPolicy allows the number of flavors to be fixed or 2/1 if Gparity/noGparity 
template<int Nf>
class FixedFlavorPolicy{
 protected:
  void setFlavors(int &flavors) const{ flavors = Nf; }
};
typedef FixedFlavorPolicy<1> OneFlavorPolicy;

//Default is to use two flavors if GPBC, 1 otherwise
class DynamicFlavorPolicy{
 protected:
  void setFlavors(int &flavors) const{ flavors = GJP.Gparity() ? 2:1 ; }
};


//The DimensionPolicy controls the mapping between an N-dimensional vector and a flavor index to an integer which is used to compute the pointer offset.
//Each policy contains 2 mappings; a linearization of a Euclidean vector to an index, and a linearization of the Euclidean vector plus a flavor index. The latter is used to compute pointer offsets, the former for convenient site looping
//We generically refer to 'sites' as being those of the Euclidean lattice, and fsites as those of the Euclidean+flavor lattice (as if flavor was another dimension)

class FourDpolicy{ //Canonical layout 4D field with second flavor stacked after full 4D block
protected:
  void setSites(int &sites, int &fsites, const int nf) const{ sites = GJP.VolNodeSites(); fsites = nf * sites; }
public:
  inline int siteMap(const int x[]) const{ return x[0] + GJP.XnodeSites()*( x[1] + GJP.YnodeSites()*( x[2] + GJP.ZnodeSites()*x[3])); }

  inline void siteUnmap(int site, int x[]) const{
    for(int i=0;i<4;i++){ 
      x[i] = site % GJP.NodeSites(i); site /= GJP.NodeSites(i);
    }
  }

  inline int fsiteMap(const int x[], const int f) const{ return siteMap(x) + f*GJP.VolNodeSites(); }

  inline void fsiteUnmap(int fsite, int x[], int &f) const{
    siteUnmap(fsite,x); f = fsite / GJP.VolNodeSites();
  }

  inline int fsiteFlavorOffset() const{ return GJP.VolNodeSites(); } //increment of linearized coordinate between flavors

  inline int siteFsiteConvert(const int site, const int f) const{ return site + GJP.VolNodeSites()*f; } //convert a site-flavor pair to an fsite

  typedef NullObject ParamType;
  FourDpolicy(const ParamType &p){}
  FourDpolicy(){}
  const static int EuclideanDimension = 4;

  inline int threeToFour(const int x3d, const int t) const{ return x3d + GJP.VolNodeSites()/GJP.TnodeSites()*t; } //convert 3d index to 4d index
};
//Canonical layout 5D field. The fsite second flavor is stacked inside the s-loop. The site is just linearized in the canonical format
class FiveDpolicy{ 
  int nf; //store nf so we don't have to keep passing it
protected:
  void setSites(int &sites, int &fsites, const int _nf){ nf = _nf; sites = GJP.VolNodeSites()*GJP.SnodeSites(); fsites = nf*sites; }
public:
  inline int siteMap(const int x[]) const{ return x[0] + GJP.XnodeSites()*( x[1] + GJP.YnodeSites()*( x[2] + GJP.ZnodeSites()*(x[3] + GJP.TnodeSites()*x[4]))); }

  inline void siteUnmap(int site, int x[]){
    for(int i=0;i<5;i++){ 
      x[i] = site % GJP.NodeSites(i); site /= GJP.NodeSites(i);
    }
  }

  inline int fsiteMap(const int x[], const int f) const{ return x[0] + GJP.XnodeSites()*( x[1] + GJP.YnodeSites()*( x[2] + GJP.ZnodeSites()*(x[3] + GJP.TnodeSites()*( f + nf*x[4]) ))); }

  inline void fsiteUnmap(int fsite, int x[], int &f){
    for(int i=0;i<4;i++){ 
      x[i] = fsite % GJP.NodeSites(i); fsite /= GJP.NodeSites(i);
    }
    f = fsite % nf; fsite /= nf;
    x[4] = fsite;
  }

  inline int fsiteFlavorOffset() const{ return GJP.VolNodeSites(); }

  inline int siteFsiteConvert(const int site, const int f) const{ 
    int x4d = site % GJP.VolNodeSites();
    int s = site / GJP.VolNodeSites();
    return x4d + GJP.VolNodeSites()*(f + nf*s);
  }

  typedef NullObject ParamType;
  FiveDpolicy(const ParamType &p){}

  const static int EuclideanDimension = 5;
};
class FourDglobalInOneDir{ //4D field where one direction 'dir' spans the entire lattice on each node separately. The ordering is setup so that the 'dir' points are blocked (change most quickly)
  int lmap[4]; //map of local dimension to physical X,Y,Z,T dimension. e.g.  [1,0,2,3] means local dimension 0 is the Y dimension, local dimension 1 is the X-direction and so on
  int dims[4];
  int dir;
  int dvol;

  void setDir(const int &_dir){
    dir = _dir;
    for(int i=0;i<4;i++) lmap[i] = i;
    std::swap(lmap[dir],lmap[0]); //make dir direction change fastest

    for(int i=0;i<4;i++) dims[i] = GJP.NodeSites(lmap[i]);
    dims[0] *= GJP.Nodes(dir);

    dvol = dims[0]*dims[1]*dims[2]*dims[3];
  }
protected:
  void setSites(int &sites, int &fsites, const int nf) const{ sites = dvol; fsites = nf * sites; }

public:
  inline int siteMap(const int x[]) const{ return x[lmap[0]] + dims[0]*( x[lmap[1]] + dims[1]*( x[lmap[2]] + dims[2]*x[lmap[3]])); }

  inline void siteUnmap(int site, int x[]){
    for(int i=0;i<4;i++){ 
      x[lmap[i]] = site % dims[i]; site /= dims[i];
    }
  }

  inline int fsiteMap(const int x[], const int f) const{ return siteMap(x) + dvol*f; }

  inline void fsiteUnmap(int fsite, int x[], int &f){
    siteUnmap(fsite,x);
    f = fsite/dvol;
  }

  inline int fsiteFlavorOffset() const{ return dvol; }

  inline int siteFsiteConvert(const int site, const int f) const{ 
    return site + dvol * f;
  }

  typedef int ParamType;

  const int &getDir() const{ return dir; }

  FourDglobalInOneDir(const int &_dir): dir(-1){
    setDir(_dir);
  }
  const static int EuclideanDimension = 4;
};

class SpatialPolicy{ //Canonical layout 3D field
  int threevol;
protected:
  void setSites(int &sites, int &fsites, const int nf) const{ sites = threevol; fsites = nf*sites; }
public:
  inline int siteMap(const int x[]) const{ return x[0] + GJP.XnodeSites()*( x[1] + GJP.YnodeSites()*x[2]); }
  inline void siteUnmap(int site, int x[]){
    for(int i=0;i<3;i++){ 
      x[i] = site % GJP.NodeSites(i); site /= GJP.NodeSites(i);
    }
  }

  inline int fsiteMap(const int x[], const int f) const{ return siteMap(x) + GJP.VolNodeSites()/GJP.TnodeSites()*f; }

  inline void fsiteUnmap(int fsite, int x[], int &f){
    siteUnmap(fsite,x);
    f = fsite/threevol;
  }

  inline int fsiteFlavorOffset() const{ return threevol; }

  inline int siteFsiteConvert(const int site, const int f) const{ 
    return site + threevol * f;
  }

  typedef NullObject ParamType;
  SpatialPolicy(const ParamType &p): threevol(GJP.VolNodeSites()/GJP.TnodeSites()){}

  const static int EuclideanDimension = 3;
};


class GlobalSpatialPolicy{ //Global canonical 3D field
protected:
  int glb_size[3];
  int glb_vol;
  void setSites(int &sites, int &fsites, const int nf) const{ sites = glb_vol; fsites = nf*sites; }
public:
  inline int siteMap(const int x[]) const{ return x[0] + glb_size[0]*( x[1] + glb_size[1]*x[2]); }

  inline void siteUnmap(int site, int x[]){
    for(int i=0;i<3;i++){ 
      x[i] = site % glb_size[i]; site /= glb_size[i];
    }
  }

  inline int fsiteMap(const int x[], const int f) const{ return siteMap(x) + f*glb_vol; }

  inline void fsiteUnmap(int fsite, int x[], int &f){
    siteUnmap(fsite,x);
    f = fsite / glb_vol;
  }

  inline int siteFsiteConvert(const int site, const int f) const{ 
    return site + glb_vol * f;
  }

  inline int fsiteFlavorOffset() const{ return glb_vol; }

  typedef NullObject ParamType;
  GlobalSpatialPolicy(const ParamType &p){
    for(int i=0;i<3;i++) glb_size[i] = GJP.NodeSites(i)*GJP.Nodes(i);
    glb_vol = glb_size[0]*glb_size[1]*glb_size[2];
  }

  const static int EuclideanDimension = 3;
};

class ThreeDglobalInOneDir{ //3D field where one direction 'dir' spans the entire lattice on each node separately. The ordering is setup so that the 'dir' points are blocked (change most quickly)
  int lmap[3]; //map of local dimension to physical X,Y,Z dimension. e.g.  [1,0,2] means local dimension 0 is the Y dimension, local dimension 1 is the X-direction and so on
  int dims[3];
  int dir;
  int dvol;

  void setDir(const int &_dir){
    dir = _dir;
    for(int i=0;i<3;i++) lmap[i] = i;
    std::swap(lmap[dir],lmap[0]); //make dir direction change fastest

    for(int i=0;i<3;i++) dims[i] = GJP.NodeSites(lmap[i]);
    dims[0] *= GJP.Nodes(dir);

    dvol = dims[0]*dims[1]*dims[2];
  }
protected:
  void setSites(int &sites, int &fsites, const int nf) const{ sites = dvol; fsites = nf*sites; }

public:
  inline int siteMap(const int x[]) const{ return x[lmap[0]] + dims[0]*( x[lmap[1]] + dims[1]*x[lmap[2]]); }
  inline void siteUnmap(int site, int x[]){
    for(int i=0;i<3;i++){ 
      x[lmap[i]] = site % dims[i]; site /= dims[i];
    }
  }

  inline int fsiteMap(const int x[], const int f) const{ return siteMap(x) + f*dvol; }

  inline void fsiteUnmap(int fsite, int x[], int &f){
    siteUnmap(fsite,x);
    f = fsite / dvol;
  }

  inline int fsiteFlavorOffset() const{ return dvol; }

  inline int siteFsiteConvert(const int site, const int f) const{ 
    return site + dvol * f;
  }

  typedef int ParamType;

  const int &getDir() const{ return dir; }

  ThreeDglobalInOneDir(const int &_dir): dir(-1){
    setDir(_dir);
  }
  const static int EuclideanDimension = 3;
};



//////////////////////////
//Checkerboarding
template<int CBdim, int CB>   //CBdim is the amount of elements of the coordinate vector (starting at 0) included in the computation,  CB is the checkerboard
class CheckerBoard{
public:
  inline int cb() const{ return CB; }
  inline bool cbDim() const{ return CBdim; }
  inline bool onCb(const int x[]) const{ 
    int c = 0; for(int i=0;i<CBdim;i++) c += x[i];
    return c % 2 == CB;
  }
};

//Checkerboarded 5D field. The fsite second flavor is stacked inside the s-loop. The checkerboard dimension and which checkerboard it is are handled by the policy CheckerBoard
//Note, the mappings do not check that the site is on the checkerboard; you should do that using onCb(x[])
template<typename CheckerBoardType>
class FiveDevenOddpolicy: public CheckerBoardType{ 
  int nf; //store nf so we don't have to keep passing it
protected:
  void setSites(int &sites, int &fsites, const int _nf){ nf = _nf; sites = GJP.VolNodeSites()*GJP.SnodeSites()/2; fsites = nf*sites; }
public:
  inline int siteMap(const int x[]) const{ 
    return (x[0] + GJP.XnodeSites()*( x[1] + GJP.YnodeSites()*( x[2] + GJP.ZnodeSites()*(x[3] + GJP.TnodeSites()*x[4]))))/2; 
  }

  inline void siteUnmap(int site, int x[]){
    site *= 2;
    for(int i=0;i<5;i++){ 
      x[i] = site % GJP.NodeSites(i); site /= GJP.NodeSites(i);
    }
    if(!this->onCb(x)) x[0] += this->cb(); //deal with int convert x[0]/2 giving same number for 0,1 etc
  }

  inline int fsiteMap(const int x[], const int f) const{ return (x[0] + GJP.XnodeSites()*( x[1] + GJP.YnodeSites()*( x[2] + GJP.ZnodeSites()*(x[3] + GJP.TnodeSites()*( f + nf*x[4]) ))))/2; }

  inline void fsiteUnmap(int fsite, int x[], int &f){
    fsite *= 2;
    for(int i=0;i<4;i++){ 
      x[i] = fsite % GJP.NodeSites(i); fsite /= GJP.NodeSites(i);
    }
    f = fsite % nf; fsite /= nf;
    x[4] = fsite;
    if(!this->onCb(x)) x[0] += this->cb(); //deal with int convert x[0]/2 giving same number for 0,1 etc
  }

  inline int fsiteFlavorOffset() const{ return GJP.VolNodeSites()/2; }

  inline int siteFsiteConvert(const int site, const int f) const{ 
    int x4d = site % (GJP.VolNodeSites()/2);
    int s = site / (GJP.VolNodeSites()/2);
    return x4d + GJP.VolNodeSites()*(f + nf*s)/2;
  }

  typedef NullObject ParamType;
  FiveDevenOddpolicy(const ParamType &p){}

  const static int EuclideanDimension = 5;
};





CPS_END_NAMESPACE
#endif
