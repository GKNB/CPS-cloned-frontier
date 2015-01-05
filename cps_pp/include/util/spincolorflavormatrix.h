#ifndef SPIN_COLOR_FLAVOR_MATRIX_H
#define SPIN_COLOR_FLAVOR_MATRIX_H

#include<config.h>
#include <alg/alg_base.h>
#include <alg/qpropw.h>
#include <alg/prop_attribute_arg.h>
#include <alg/propagatorcontainer.h>
#include <alg/propmanager.h>
#include <util/spinflavormatrix.h>
CPS_START_NAMESPACE

enum FlavorMatrixType {F0, F1, Fud, sigma0, sigma1, sigma2, sigma3};
enum SpinMatrixType { gamma1, gamma2, gamma3, gamma4, gamma5, spin_unit };

class SpinColorFlavorMatrix{
public:
  enum PropSplane { SPLANE_BOUNDARY, SPLANE_MIDPOINT }; //Usual boundary 5d propagator or the midpoint propagator
protected:
  WilsonMatrix** wmat;
  const char *cname;

  inline static WilsonMatrix & getSite(const int &site, const int &flav, QPropWcontainer &from,  Lattice &lattice, const PropSplane &splane){
    return splane == SPLANE_BOUNDARY ? from.getProp(lattice).SiteMatrix(site,flav) : from.getProp(lattice).MidPlaneSiteMatrix(site,flav);
  }
public:
  inline void allocate(){
    wmat = new WilsonMatrix* [2];
    wmat[0] = new WilsonMatrix[2];
    wmat[1] = new WilsonMatrix[2];
  }
  inline void free(){
    if(wmat!=NULL){
      delete[] wmat[0];
      delete[] wmat[1];
      delete[] wmat;
      wmat=NULL;
    }
  }

  SpinColorFlavorMatrix(QPropWcontainer &from, Lattice &lattice, const int &site, const PropSplane &splane = SPLANE_BOUNDARY): wmat(NULL), cname("SpinColorFlavorMatrix"){
    generate(from,lattice,site,splane);
  }
  SpinColorFlavorMatrix(const SpinColorFlavorMatrix &from): wmat(NULL), cname("SpinColorFlavorMatrix"){
    allocate();
    wmat[0][0] = from.wmat[0][0];
    wmat[0][1] = from.wmat[0][1];
    wmat[1][0] = from.wmat[1][0];
    wmat[1][1] = from.wmat[1][1];
  }
  SpinColorFlavorMatrix(const Float &rhs): cname("SpinColorFlavorMatrix"){
    allocate();
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j] = rhs;
  }

  SpinColorFlavorMatrix(const SpinMatrixType &g = spin_unit, const FlavorMatrixType &f = sigma0): cname("SpinColorFlavorMatrix"){
    allocate();
    static const int spin_map[] = { 0,1,2,3,-5 };
    unit();
    if(g != spin_unit) gr(spin_map[ (int)g - (int)gamma1 ]);
    if(f != sigma0) pr(f);
  }

  void generate(QPropWcontainer &from_f0, QPropWcontainer &from_f1, Lattice &lattice, const int &site, const PropSplane &splane = SPLANE_BOUNDARY){
    if(wmat!=NULL) free();
    allocate();
    
    wmat[0][0] = getSite(site,0,from_f0,lattice,splane);
    wmat[1][0] = getSite(site,1,from_f0,lattice,splane);
    wmat[0][1] = getSite(site,0,from_f1,lattice,splane);
    wmat[1][1] = getSite(site,1,from_f1,lattice,splane);
  }

  void generate(QPropWcontainer &from, Lattice &lattice, const int &site, const PropSplane &splane = SPLANE_BOUNDARY){
    const char* fname = "generate(PropagatorContainer &from, const int &site)";
    
    if(!GJP.Gparity()){
      ERR.General(cname,fname,"Require 2f G-parity BCs to be active");
    }
    PointSourceAttrArg *pt;
    MomentumAttrArg *mom;
    MomCosAttrArg *cos;
    WallSourceAttrArg *wall;
    if( ( from.getAttr(mom) && from.getAttr(cos) ) || ( from.getAttr(pt) && !from.getAttr(mom) ) || ( from.getAttr(wall) && !from.getAttr(mom) ) ){
      //cos source (point or wall) or zero momentum point source
    }else{
      GparityOtherFlavPropAttrArg* otherfarg; //if a propagator with the same properties but the other flavor exists then use both to generate the matrix
      if(from.getAttr(otherfarg)){
	QPropWcontainer &otherfprop = PropManager::getProp(otherfarg->tag).convert<QPropWcontainer>();
	if(otherfprop.flavor() == from.flavor()) ERR.General(cname,fname,"Found a propagator %s with supposedly the other flavor to propagator %s, but in fact the flavors are identical!",otherfarg->tag,from.tag());
	QPropWcontainer *f0prop; QPropWcontainer *f1prop;
	if(from.flavor() == 0){ f0prop = &from; f1prop = &otherfprop; }
	else { f1prop = &from; f0prop = &otherfprop; }
	
	return generate(*f0prop,*f1prop,lattice,site,splane);
      }else ERR.General(cname,fname,"Cannot generate prop elements without a real source, e.g. a Cos (Point or Wall) or Zero-Momentum Point or Wall source, or else two props with the same attributes but different flavors and a GparityOtherFlavPropAttrArg given");
    }
    
    if(wmat!=NULL) free();
    allocate();

    int flav = from.flavor();
    if(flav == 0){
      wmat[0][0] = getSite(site,0,from,lattice,splane);
      wmat[1][0] = getSite(site,1,from,lattice,splane);
      wmat[0][1] = wmat[1][0];
      wmat[1][1] = wmat[0][0];

      wmat[0][1].cconj();
      wmat[1][1].cconj();

      wmat[0][1].ccl(-1).gl(-5).ccr(1).gr(-5); //ccl(-1) mults by C from left, ccr(1) mults by C=-C^dag from right
      wmat[1][1].ccl(-1).gl(-5).ccr(-1).gr(-5); //has opposite sign to the above 
    }else{ //flavour 1 source
      wmat[0][1] = getSite(site,0,from,lattice,splane);
      wmat[1][1] = getSite(site,1,from,lattice,splane);
      wmat[1][0] = wmat[0][1];
      wmat[0][0] = wmat[1][1];

      wmat[1][0].cconj();
      wmat[0][0].cconj();

      wmat[1][0].ccl(-1).gl(-5).ccr(1).gr(-5); 
      wmat[0][0].ccl(-1).gl(-5).ccr(-1).gr(-5);
    }
  }

  ~SpinColorFlavorMatrix(){ free(); }
  
  //multiply on left by a flavor matrix
  SpinColorFlavorMatrix & pl(const FlavorMatrixType &type){
    if(type == F0){
      wmat[1][0] = 0.0;
      wmat[1][1] = 0.0;
      return *this;
    }else if(type == F1){
      wmat[0][0] = 0.0;
      wmat[0][1] = 0.0;
      return *this;
    }else if(type == Fud){
      WilsonMatrix _00(wmat[0][0]);
      WilsonMatrix _01(wmat[0][1]);
      wmat[0][0] = wmat[1][0];
      wmat[0][1] = wmat[1][1];
      wmat[1][0] = _00;
      wmat[1][1] = _01;
      return *this;
    }else if(type == sigma0){
      return *this;
    }else if(type == sigma1){
      return pl(Fud);
    }else if(type == sigma2){
      WilsonMatrix _i00(wmat[0][0]); _i00*=Complex(0.0,1.0);
      WilsonMatrix _i01(wmat[0][1]); _i01*=Complex(0.0,1.0);
      wmat[0][0] = wmat[1][0]; wmat[0][0]*=Complex(0.0,-1.0);
      wmat[0][1] = wmat[1][1]; wmat[0][1]*=Complex(0.0,-1.0);
      wmat[1][0] = _i00;
      wmat[1][1] = _i01;
      return *this;
    }else if(type == sigma3){
      wmat[1][0]*=-1.0;
      wmat[1][1]*=-1.0;
      return *this;
    }
    ERR.General(cname,"pl(const FlavorMatrixType &type)","Unknown FlavorMatrixType");
  }
  //multiply on right by a flavor matrix
  SpinColorFlavorMatrix & pr(const FlavorMatrixType &type){
    if(type == F0){
      wmat[0][1] = 0.0;
      wmat[1][1] = 0.0;
      return *this;
    }else if(type == F1){
      wmat[0][0] = 0.0;
      wmat[1][0] = 0.0;
      return *this;
    }else if(type == Fud){
      WilsonMatrix _00(wmat[0][0]);
      WilsonMatrix _10(wmat[1][0]);
      wmat[0][0] = wmat[0][1];
      wmat[1][0] = wmat[1][1];
      wmat[0][1] = _00;
      wmat[1][1] = _10;
      return *this;
    }else if(type == sigma0){
      return *this;
    }else if(type == sigma1){
      return pr(Fud);
    }else if(type == sigma2){
      WilsonMatrix _mi00(wmat[0][0]); _mi00 *= Complex(0.0,-1.0);
      WilsonMatrix _mi10(wmat[1][0]); _mi10 *= Complex(0.0,-1.0);
      wmat[0][0] = wmat[0][1]; wmat[0][0] *= Complex(0.0,1.0); 
      wmat[1][0] = wmat[1][1]; wmat[1][0] *= Complex(0.0,1.0);
      wmat[0][1] = _mi00;
      wmat[1][1] = _mi10;
      return *this;
    }else if(type == sigma3){
      wmat[0][1]*=-1.0;
      wmat[1][1]*=-1.0;
      return *this;
    }
    ERR.General(cname,"pr(const FlavorMatrixType &type)","Unknown FlavorMatrixType");
  }
  WilsonMatrix FlavourTrace() const{
    WilsonMatrix out(wmat[0][0]);
    out+=wmat[1][1];
    return out;
  }
  Rcomplex Trace() const{
    Rcomplex out(0);
    for(int f=0;f<2;f++) out += wmat[f][f].Trace();
    return out;
  }
  
  SpinColorFlavorMatrix operator*(const SpinColorFlavorMatrix& rhs) const{
    SpinColorFlavorMatrix out(*this);
    out.wmat[0][0] = wmat[0][0]*rhs.wmat[0][0] + wmat[0][1]*rhs.wmat[1][0];
    out.wmat[1][0] = wmat[1][0]*rhs.wmat[0][0] + wmat[1][1]*rhs.wmat[1][0];
    out.wmat[0][1] = wmat[0][0]*rhs.wmat[0][1] + wmat[0][1]*rhs.wmat[1][1];
    out.wmat[1][1] = wmat[1][0]*rhs.wmat[0][1] + wmat[1][1]*rhs.wmat[1][1];
    return out;
  }

  SpinColorFlavorMatrix& operator*=(const SpinColorFlavorMatrix& rhs){
    SpinColorFlavorMatrix cp(*this);
    wmat[0][0] = cp.wmat[0][0]*rhs.wmat[0][0] + cp.wmat[0][1]*rhs.wmat[1][0];
    wmat[1][0] = cp.wmat[1][0]*rhs.wmat[0][0] + cp.wmat[1][1]*rhs.wmat[1][0];
    wmat[0][1] = cp.wmat[0][0]*rhs.wmat[0][1] + cp.wmat[0][1]*rhs.wmat[1][1];
    wmat[1][1] = cp.wmat[1][0]*rhs.wmat[0][1] + cp.wmat[1][1]*rhs.wmat[1][1];
    return *this;
  }

  SpinColorFlavorMatrix& operator*=(const Float& rhs){
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j]*=rhs;
    return *this;
  }
  SpinColorFlavorMatrix operator*(const Float& rhs) const{
    SpinColorFlavorMatrix out(*this);
    out*=rhs;
    return out;
  }
  SpinColorFlavorMatrix& operator*=(const Rcomplex& rhs){
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j]*=rhs;
    return *this;
  }
  SpinColorFlavorMatrix operator*(const Rcomplex& rhs) const{
    SpinColorFlavorMatrix out(*this);
    out*=rhs;
    return out;
  }
  SpinColorFlavorMatrix& operator*=(const WilsonMatrix& rhs){
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j]*=rhs;
    return *this;
  }
  SpinColorFlavorMatrix operator*(const WilsonMatrix& rhs) const{
    SpinColorFlavorMatrix out(*this);
    out*=rhs;
    return out;
  }

  SpinColorFlavorMatrix& operator+=(const SpinColorFlavorMatrix& rhs){
    wmat[0][0]+= rhs.wmat[0][0];
    wmat[0][1]+= rhs.wmat[0][1];
    wmat[1][0]+= rhs.wmat[1][0];
    wmat[1][1]+= rhs.wmat[1][1];
    return *this;
  }
  SpinColorFlavorMatrix operator+(const SpinColorFlavorMatrix& rhs) const{
    SpinColorFlavorMatrix out(*this);
    out+=rhs;
    return out;
  }
  SpinColorFlavorMatrix& operator-=(const SpinColorFlavorMatrix& rhs){
    wmat[0][0]-= rhs.wmat[0][0];
    wmat[0][1]-= rhs.wmat[0][1];
    wmat[1][0]-= rhs.wmat[1][0];
    wmat[1][1]-= rhs.wmat[1][1];
    return *this;
  }
  SpinColorFlavorMatrix operator-(const SpinColorFlavorMatrix& rhs) const{
    SpinColorFlavorMatrix out(*this);
    out-=rhs;
    return out;
  }
    
  SpinColorFlavorMatrix& operator= (const Float& rhs){
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j] = rhs;
    return *this;
  }
  SpinColorFlavorMatrix& operator= (const Rcomplex& rhs){
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j] = rhs;
    return *this;
  }
  SpinColorFlavorMatrix& operator=(const SpinColorFlavorMatrix &from){
    wmat[0][0] = from.wmat[0][0];
    wmat[0][1] = from.wmat[0][1];
    wmat[1][0] = from.wmat[1][0];
    wmat[1][1] = from.wmat[1][1];
    return *this;
  }
  
  SpinColorFlavorMatrix& gl(int dir){
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j].gl(dir);
    return *this;
  }
  SpinColorFlavorMatrix& gr(int dir){
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j].gr(dir);
    return *this;
  }

  SpinColorFlavorMatrix& ccl(int dir){
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j].ccl(dir);
    return *this;
  }
  SpinColorFlavorMatrix& ccr(int dir){
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j].ccr(dir);
    return *this;
  }
 //! left multiply by 1/2(1-gamma_5)
  SpinColorFlavorMatrix& glPL(){
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j].glPL();
    return *this;
  }
  //! left multiply by 1/2(1+gamma_5)
  SpinColorFlavorMatrix& glPR(){
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j].glPR();
    return *this;
  }

  //! right multiply by 1/2(1-gamma_5)
  SpinColorFlavorMatrix& grPL(){
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j].grPL();
    return *this;
  }

  //! right multiply by 1/2(1+gamma_5)
  SpinColorFlavorMatrix& grPR(){
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j].grPR();
    return *this;
  }
  IFloat norm() const{
    IFloat out(0.0);
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	out += wmat[i][j].norm();
    return out;
  }
  //! hermitean conjugate
  void hconj(){
    WilsonMatrix hc01 = wmat[0][1]; hc01.hconj();
    wmat[0][0].hconj();
    wmat[1][1].hconj();
    wmat[0][1] = wmat[1][0];
    wmat[0][1].hconj();
    wmat[1][0] = hc01;
  }
 
  //! complex conjugate
  void cconj(){
    for(int i=0;i<2;i++)
      for(int j=0;j<2;j++)
	wmat[i][j].cconj();
  }

  //spin color and flavor transpose
  SpinColorFlavorMatrix & transpose(){
    WilsonMatrix _01tr = wmat[0][1]; _01tr.transpose();
    wmat[0][0].transpose();
    wmat[1][1].transpose();
    wmat[0][1] = wmat[1][0];
    wmat[0][1].transpose();
    wmat[1][0] = _01tr;
    return *this;
  }
  //just flavor
  SpinColorFlavorMatrix & tranpose_flavor(){
    WilsonMatrix _01 = wmat[0][1];
    wmat[0][1] = wmat[1][0];
    wmat[1][0] = _01;
    return *this;
  }

  Complex& operator()(int s1, int c1, int f1, int s2, int c2, int f2){
    return wmat[f1][f2](s1,c1,s2,c2);
  }  
  const Complex& operator()(int s1, int c1, int f1, int s2, int c2, int f2) const{
    return wmat[f1][f2](s1,c1,s2,c2);
  }
  WilsonMatrix &operator()(int f1,int f2){
    return wmat[f1][f2];
  }
  const WilsonMatrix &operator()(int f1,int f2) const{
    return wmat[f1][f2];
  }
  SpinColorFlavorMatrix& LeftTimesEqual(const SpinColorFlavorMatrix& lhs){
    SpinColorFlavorMatrix tmp(*this);
    for(int f1=0;f1<2;f1++) for(int f2=0;f2<2;f2++){
	wmat[f1][f2] = 0.0;
	for(int f3=0;f3<2;f3++) wmat[f1][f2] += lhs.wmat[f1][f3]*tmp.wmat[f3][f2];
    }
    return *this;
  }
  SpinColorFlavorMatrix& LeftTimesEqual(const WilsonMatrix& lhs){
    for(int f1=0;f1<2;f1++) for(int f2=0;f2<2;f2++) wmat[f1][f2].LeftTimesEqual(lhs);
    return *this;
  }
  SpinColorFlavorMatrix& LeftTimesEqual(const Matrix& lhs){
    for(int f1=0;f1<2;f1++) for(int f2=0;f2<2;f2++) wmat[f1][f2].LeftTimesEqual(lhs);
    return *this;
  }

  void unit(){
    *this = 0.0;
    for(int f=0;f<2;f++) for(int s=0;s<4;s++) for(int c=0;c<3;c++) wmat[f][f](s,c,s,c) = 1.0;
  }
};

Rcomplex Trace(const SpinColorFlavorMatrix& a, const SpinColorFlavorMatrix& b);
Matrix SpinFlavorTrace(const SpinColorFlavorMatrix& a, const SpinColorFlavorMatrix& b);
SpinFlavorMatrix ColorTrace(const SpinColorFlavorMatrix& a, const SpinColorFlavorMatrix& b);

typedef struct { wilson_vector f[2]; } flav_spin_color_vector;
typedef struct { flav_spin_color_vector c[3]; } color_flav_spin_color_vector;
typedef struct { color_flav_spin_color_vector d[4]; } spin_color_flav_spin_color_vector;
typedef struct { spin_color_flav_spin_color_vector f[2]; } spin_color_flavor_matrix;

class FlavorSpinColorMatrix{
public:
  enum PropSplane { SPLANE_BOUNDARY, SPLANE_MIDPOINT }; //Usual boundary 5d propagator or the midpoint propagator
protected:
  spin_color_flavor_matrix p;

  inline static WilsonMatrix & getSite(const int &site, const int &flav, QPropWcontainer &from,  Lattice &lattice, const PropSplane &splane){
    return splane == SPLANE_BOUNDARY ? from.getProp(lattice).SiteMatrix(site,flav) : from.getProp(lattice).MidPlaneSiteMatrix(site,flav);
  }
public:
  Rcomplex* ptr() { return reinterpret_cast<Rcomplex*>(&p); }
  const Rcomplex* ptr() const { return reinterpret_cast<const Rcomplex*>(&p); }

  //Complex pointer offset to element
  inline static int wilson_matrix_map(const int &rs,const int&rc, const int &cs, const int &cc){
    return (cc+3*(cs+4*(rc+3*rs)));
  }
  inline static void wilson_matrix_invmap(int idx, int &rs, int&rc, int &cs, int &cc){
    cc = idx%3; idx/=3;
    cs = idx%4; idx/=4;
    rc = idx%3; idx/=3;
    rs = idx;
  }
  inline static int fsc_matrix_map(const int &rf, const int &rs,const int&rc, const int &cf, const int &cs, const int &cc){
    return (cc+3*(cs+4*(cf+2*(rc+3*(rs+4*rf)))));
  }
  inline static int fsc_matrix_invmap(int idx, int &rf, int &rs, int&rc, int &cf, int &cs, int &cc){
    cc = idx%3; idx/=3;
    cs = idx%4; idx/=4;
    cf = idx%2; idx/=2;
    rc = idx%3; idx/=3;
    rs = idx%4; idx/=4;
    rf = idx;
  }

  FlavorSpinColorMatrix(QPropWcontainer &from, Lattice &lattice, const int &site, const PropSplane &splane = SPLANE_BOUNDARY){
    generate(from,lattice,site,splane);
  }
  FlavorSpinColorMatrix(){}

  FlavorSpinColorMatrix(const FlavorSpinColorMatrix &from){
    Float* top = (Float*)&p;
    Float* fromp = (Float*)&from.p;
    memcpy((void*)top,(void*)fromp,1152*sizeof(Float));
  }
  FlavorSpinColorMatrix(const Float &rhs){
    Float* to = (Float*)&p;
    for(int i=0;i<1152;++i) to[i] = rhs;
  }
  Complex& operator()(int f1, int s1, int c1, int f2, int s2, int c2){
    return p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2];
  }  
  const Complex& operator()(int f1, int s1, int c1, int f2, int s2, int c2) const{
    return p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2];
  }
  WilsonMatrix operator()(int f1,int f2){
    WilsonMatrix out;
    for(int s1=0;s1<4;++s1) for(int c1=0;c1<3;++c1)
    for(int s2=0;s2<4;++s2) for(int c2=0;c2<3;++c2)
       out(s1,c1,s2,c2) = p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2];
  }

  void generate(QPropWcontainer &from_f0, QPropWcontainer &from_f1, Lattice &lattice, const int &site, const PropSplane &splane = SPLANE_BOUNDARY){
    WilsonMatrix* wmat[2][2] = { { &getSite(site,0,from_f0,lattice,splane), &getSite(site,0,from_f1,lattice,splane) },
				 { &getSite(site,1,from_f0,lattice,splane), &getSite(site,1,from_f1,lattice,splane) } };

    for(int f1=0;f1<2;++f1) for(int s1=0;s1<4;++s1) for(int c1=0;c1<3;++c1)
    for(int f2=0;f2<2;++f2) for(int s2=0;s2<4;++s2) for(int c2=0;c2<3;++c2)
       p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2] = (*wmat[f1][f2])(s1,c1,s2,c2);
  }

  void generate(QPropWcontainer &from, Lattice &lattice, const int &site, const PropSplane &splane = SPLANE_BOUNDARY);

  //multiply on left by a flavor matrix
  FlavorSpinColorMatrix & pl(const FlavorMatrixType &type);
  //multiply on right by a flavor matrix
  FlavorSpinColorMatrix & pr(const FlavorMatrixType &type);

  WilsonMatrix FlavourTrace(){
    WilsonMatrix out(0);
    for(int f=0;f<2;++f)
    for(int s1=0;s1<4;++s1) for(int c1=0;c1<3;++c1)
    for(int s2=0;s2<4;++s2) for(int c2=0;c2<3;++c2)
       out(s1,c1,s2,c2) += p.f[f].d[s1].c[c1].f[f].d[s2].c[c2];

    return out;
  }
  Rcomplex Trace(){
    Rcomplex out(0);
    for(int f=0;f<2;++f) for(int s=0;s<4;++s) for(int c=0;c<3;++c)
	out += p.f[f].d[s].c[c].f[f].d[s].c[c];					      
    return out;
  }

  FlavorSpinColorMatrix operator*(const FlavorSpinColorMatrix& rhs){
    FlavorSpinColorMatrix out;
    for(int f1=0;f1<2;++f1) for(int s1=0;s1<4;++s1) for(int c1=0;c1<3;++c1)
    for(int f2=0;f2<2;++f2) for(int s2=0;s2<4;++s2) for(int c2=0;c2<3;++c2){
      out(f1,s1,c1,f2,s2,c2) = 0.0;
      for(int f3=0;f3<2;++f3) for(int s3=0;s3<4;++s3) for(int c3=0;c3<3;++c3) out(f1,s1,c1,f2,s2,c2) +=  (*this)(f1,s1,c1,f3,s3,c3) * rhs(f3,s3,c3,f2,s2,c2);
    }
    return out;
  }
  
  FlavorSpinColorMatrix& operator*=(const FlavorSpinColorMatrix& rhs){
    FlavorSpinColorMatrix cp(*this);
    for(int f1=0;f1<2;++f1) for(int s1=0;s1<4;++s1) for(int c1=0;c1<3;++c1)
    for(int f2=0;f2<2;++f2) for(int s2=0;s2<4;++s2) for(int c2=0;c2<3;++c2){
      p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2] = 0.0;
      for(int f3=0;f3<2;++f3) for(int s3=0;s3<4;++s3) for(int c3=0;c3<3;++c3) p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2] +=  cp(f1,s1,c1,f3,s3,c3) * rhs(f3,s3,c3,f2,s2,c2);
    }
    return *this;
  }

  FlavorSpinColorMatrix& operator*=(const Float& rhs){
    Float *to = (Float*)&p;
    for(int i=0;i<1152;++i) to[i]*=rhs;
    return *this;
  }
  FlavorSpinColorMatrix operator*(const Float& rhs){
    FlavorSpinColorMatrix out(*this);
    out*=rhs;
    return out;
  }
  FlavorSpinColorMatrix& operator*=(const Rcomplex& rhs){
    Complex *to = (Complex*)&p;
    for(int i=0;i<576;++i) to[i]*=rhs;
    return *this;
  }
  FlavorSpinColorMatrix operator*(const Rcomplex& rhs){
    FlavorSpinColorMatrix out(*this);
    out*=rhs;
    return out;
  }
  FlavorSpinColorMatrix& operator*=(const WilsonMatrix& rhs){
    FlavorSpinColorMatrix cp(*this);
    for(int f1=0;f1<2;++f1) for(int s1=0;s1<4;++s1) for(int c1=0;c1<3;++c1)
    for(int f2=0;f2<2;++f2) for(int s2=0;s2<4;++s2) for(int c2=0;c2<3;++c2){
      p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2] = 0.0;
      for(int s3=0;s3<4;++s3) for(int c3=0;c3<3;++c3) p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2] +=  cp(f1,s1,c1,f2,s3,c3) * rhs(s3,c3,s2,c2);
    }
    return *this;
  }
  FlavorSpinColorMatrix operator*(const WilsonMatrix& rhs){
    FlavorSpinColorMatrix out(*this);
    out*=rhs;
    return out;
  }

  FlavorSpinColorMatrix& operator+=(const FlavorSpinColorMatrix& rhs){
    Float* top = (Float*)&p;
    Float* addp = (Float*)&rhs.p;
    for(int i=0;i<1152;++i) top[i] += addp[i];
    return *this;
  }
  FlavorSpinColorMatrix operator+(const FlavorSpinColorMatrix& rhs){
    FlavorSpinColorMatrix out(*this);
    out+=rhs;
    return out;
  }
  FlavorSpinColorMatrix& operator-=(const FlavorSpinColorMatrix& rhs){
    Float* top = (Float*)&p;
    Float* addp = (Float*)&rhs.p;
    for(int i=0;i<1152;++i) top[i] -= addp[i];
    return *this;
  }
  FlavorSpinColorMatrix operator-(const FlavorSpinColorMatrix& rhs){
    FlavorSpinColorMatrix out(*this);
    out-=rhs;
    return out;
  }
    
  FlavorSpinColorMatrix& operator= (const Float& rhs){
    Float* top = (Float*)&p;
    for(int i=0;i<1152;++i) top[i] = rhs;
    return *this;
  }
  FlavorSpinColorMatrix& operator= (const Rcomplex& rhs){
    Complex *to = (Complex*)&p;
    for(int i=0;i<576;++i) to[i]=rhs;
    return *this;
  }
  FlavorSpinColorMatrix& operator=(const FlavorSpinColorMatrix &from){
    Float* top = (Float*)&p;
    Float* fromp = (Float*)&from.p;
    memcpy((void*)top,(void*)fromp,1152*sizeof(Float));
    return *this;
  }
  
  FlavorSpinColorMatrix& gl(int dir);
  FlavorSpinColorMatrix& gr(int dir);

  FlavorSpinColorMatrix& ccl(int dir);//Note, this has the same incorrect ordering as for WilsonMatrix:   ccl(1)  =  C^{-1}M   ccl(-1) = CM
  FlavorSpinColorMatrix& ccr(int dir);

 //! left multiply by 1/2(1-gamma_5)
  FlavorSpinColorMatrix& glPL();

  //! left multiply by 1/2(1+gamma_5)
  FlavorSpinColorMatrix& glPR();

  //! right multiply by 1/2(1-gamma_5)
  FlavorSpinColorMatrix& grPL();

  //! right multiply by 1/2(1+gamma_5)
  FlavorSpinColorMatrix& grPR();

  IFloat norm() const{
    Float out(0.0);
    Complex *c = (Complex*)&p;
    for(int i=0;i<576;++i) out += std::norm(c[i]);
    return out;
  }
  //! hermitean conjugate
  void hconj(){
    FlavorSpinColorMatrix tmp(*this);   
    for(int f1=0;f1<2;++f1) for(int s1=0;s1<4;++s1) for(int c1=0;c1<3;++c1)
    for(int f2=0;f2<2;++f2) for(int s2=0;s2<4;++s2) for(int c2=0;c2<3;++c2){
      p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2].real() = tmp.p.f[f2].d[s2].c[c2].f[f1].d[s1].c[c1].real();
      p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2].imag() = -tmp.p.f[f2].d[s2].c[c2].f[f1].d[s1].c[c1].imag();
    }
  }
 
  //! complex conjugate
  void cconj(){
    Float *f = (Float*)&p;
    for(int i=1;i<1152;i+=2) f[i]*=-1;
  }

  //spin color and flavor transpose
  void transpose(){
    FlavorSpinColorMatrix tmp(*this);   
    for(int f1=0;f1<2;++f1) for(int s1=0;s1<4;++s1) for(int c1=0;c1<3;++c1)
    for(int f2=0;f2<2;++f2) for(int s2=0;s2<4;++s2) for(int c2=0;c2<3;++c2)
      p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2] = tmp.p.f[f2].d[s2].c[c2].f[f1].d[s1].c[c1];
  }
  //just flavor
  void tranpose_flavor(){
    FlavorSpinColorMatrix tmp(*this); 
    for(int f1=0;f1<2;++f1) for(int s1=0;s1<4;++s1) for(int c1=0;c1<3;++c1)
    for(int f2=0;f2<2;++f2) for(int s2=0;s2<4;++s2) for(int c2=0;c2<3;++c2)
      p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2] = tmp.p.f[f2].d[s1].c[c1].f[f1].d[s2].c[c2];
  }

  FlavorSpinColorMatrix& LeftTimesEqual(const FlavorSpinColorMatrix& lhs){
    FlavorSpinColorMatrix cp(*this);
    for(int f1=0;f1<2;++f1) for(int s1=0;s1<4;++s1) for(int c1=0;c1<3;++c1)
    for(int f2=0;f2<2;++f2) for(int s2=0;s2<4;++s2) for(int c2=0;c2<3;++c2){
      p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2] = 0.0;
      for(int f3=0;f3<2;++f3) for(int s3=0;s3<4;++s3) for(int c3=0;c3<3;++c3) p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2] +=  lhs(f1,s1,c1,f3,s3,c3) * cp(f3,s3,c3,f2,s2,c2);
    }
    return *this;
  }

  FlavorSpinColorMatrix& LeftTimesEqual(const WilsonMatrix& lhs){
    FlavorSpinColorMatrix cp(*this);
    for(int f1=0;f1<2;++f1) for(int s1=0;s1<4;++s1) for(int c1=0;c1<3;++c1)
    for(int f2=0;f2<2;++f2) for(int s2=0;s2<4;++s2) for(int c2=0;c2<3;++c2){
      p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2] = 0.0;
      for(int s3=0;s3<4;++s3) for(int c3=0;c3<3;++c3) p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2] +=  lhs(s1,c1,s3,c3) * cp(f1,s3,c3,f2,s2,c2);
    }
    return *this;
  }
  FlavorSpinColorMatrix& LeftTimesEqual(const Matrix& lhs){
    FlavorSpinColorMatrix cp(*this);
    for(int f1=0;f1<2;++f1) for(int s1=0;s1<4;++s1) for(int c1=0;c1<3;++c1)
    for(int f2=0;f2<2;++f2) for(int s2=0;s2<4;++s2) for(int c2=0;c2<3;++c2){
      p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2] = 0.0;
      for(int c3=0;c3<3;++c3) p.f[f1].d[s1].c[c1].f[f2].d[s2].c[c2] +=  lhs(c1,c3) * cp(f1,s1,c3,f2,s2,c2);
    }
    return *this;
  }

  void unit(){
    *this = 0.0;
    for(int f=0;f<2;f++) for(int s=0;s<4;s++) for(int c=0;c<3;c++) p.f[f].d[s].c[c].f[f].d[s].c[c] = 1.0;
  }
};


inline static SpinColorFlavorMatrix ColorTranspose(const SpinColorFlavorMatrix &m){
  SpinColorFlavorMatrix out;
  for(int f1=0;f1<2;++f1) for(int s1=0;s1<4;++s1) for(int c1=0;c1<3;++c1)
  for(int f2=0;f2<2;++f2) for(int s2=0;s2<4;++s2) for(int c2=0;c2<3;++c2)
    out(s1,c1,f1,s2,c2,f2) = m(s1,c2,f1,s2,c1,f2);
  return out;
}




















CPS_END_NAMESPACE

#endif

