CPS_START_NAMESPACE
#ifndef SPIN_COLOR_FLAVOR_MATRIX_H
#define SPIN_COLOR_FLAVOR_MATRIX_H
CPS_END_NAMESPACE

#include<config.h>
#include <alg/alg_base.h>
#include <alg/qpropw.h>
#include <alg/prop_attribute_arg.h>

CPS_START_NAMESPACE

enum FlavorMatrixType {F0, F1, Fud, sigma3};

class SpinColorFlavorMatrix{
protected:
  WilsonMatrix** wmat;
  const char *cname;
public:
  SpinColorFlavorMatrix(PropagatorContainer &from, Lattice &lattice, const int &site): wmat(NULL), cname("SpinColorFlavorMatrix"){
    generate(from,lattice,site);
  }
  SpinColorFlavorMatrix(const SpinColorFlavorMatrix &from): wmat(NULL), cname("SpinColorFlavorMatrix"){
    wmat = new WilsonMatrix* [2];
    wmat[0] = new WilsonMatrix[2];
    wmat[1] = new WilsonMatrix[2];

    wmat[0][0] = from.wmat[0][0];
    wmat[0][1] = from.wmat[0][1];
    wmat[1][0] = from.wmat[1][0];
    wmat[1][1] = from.wmat[1][1];
  }

  void generate(PropagatorContainer &from, Lattice &lattice, const int &site){
    const char* fname = "generate(PropagatorContainer &from, const int &site)";
    
    if(!GJP.Gparity()){
      ERR.General(cname,fname,"Require 2f G-parity BCs to be active");
    }
    PointSourceAttrArg *pt;
    MomentumAttrArg *mom;
    MomCosAttrArg *cos;
    WallSourceAttrArg *wl;
    if( from.getAttr(pt) ){
      if(from.getAttr(mom) && (mom->p[0]!=0 || mom->p[1]!=0 || mom->p[2]!=0) ){
	ERR.General(cname,fname,"Cannot generate prop elements without a real source. Prop is a point source but with non-zero momentum");
      }
    }else if( from.getAttr(wl) && from.getAttr(cos) && from.getAttr(mom) ){
      //this is a cos source
    }else{
      ERR.General(cname,fname,"Cannot generate prop elements without a real source, e.g. a Cos or Point source");
    }
    
    if(wmat!=NULL) free();
    wmat = new WilsonMatrix* [2];
    wmat[0] = new WilsonMatrix[2];
    wmat[1] = new WilsonMatrix[2];

    int flav = from.flavor();
    if(flav == 0){
      wmat[0][0] = from.getProp(lattice).SiteMatrix(site,0);
      wmat[1][0] = from.getProp(lattice).SiteMatrix(site,1);
      wmat[0][1] = wmat[1][0];
      wmat[1][1] = wmat[0][0];

      wmat[0][1].cconj();
      wmat[1][1].cconj();

      wmat[0][1].ccl(-1).gl(-5).ccr(1).gr(-5); //ccl(-1) mults by C from left, ccr(1) mults by C=-C^dag from right
      wmat[1][1].ccl(-1).gl(-5).ccr(-1).gr(-5); //has opposite sign to the above 
    }else{ //flavour 1 source
      wmat[0][1] = from.getProp(lattice).SiteMatrix(site,0);
      wmat[1][1] = from.getProp(lattice).SiteMatrix(site,1);
      wmat[1][0] = wmat[0][1];
      wmat[0][0] = wmat[1][1];

      wmat[1][0].cconj();
      wmat[0][0].cconj();

      wmat[1][0].ccl(-1).gl(-5).ccr(1).gr(-5); 
      wmat[0][0].ccl(-1).gl(-5).ccr(-1).gr(-5);
    }
  }
  void free(){
    if(wmat!=NULL){
      delete[] wmat[0];
      delete[] wmat[1];
      delete wmat;
      wmat=NULL;
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
    }else if(type == sigma3){
      wmat[0][1]*=-1.0;
      wmat[1][1]*=-1.0;
      return *this;
    }
    ERR.General(cname,"pr(const FlavorMatrixType &type)","Unknown FlavorMatrixType");
  }
  WilsonMatrix FlavourTrace(){
    WilsonMatrix out(wmat[0][0]);
    out+=wmat[1][1];
    return out;
  }
  Rcomplex Trace(){
    return FlavourTrace().Trace();
  }
  
  SpinColorFlavorMatrix operator*(const SpinColorFlavorMatrix& rhs){
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
  SpinColorFlavorMatrix operator*(const Float& rhs){
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
  SpinColorFlavorMatrix operator*(const Rcomplex& rhs){
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
  SpinColorFlavorMatrix operator*(const WilsonMatrix& rhs){
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
  SpinColorFlavorMatrix operator+(const SpinColorFlavorMatrix& rhs){
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
  SpinColorFlavorMatrix operator-(const SpinColorFlavorMatrix& rhs){
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
  void transpose(){
    WilsonMatrix _01tr = wmat[0][1]; _01tr.transpose();
    wmat[0][0].transpose();
    wmat[1][1].transpose();
    wmat[0][1] = wmat[1][0];
    wmat[0][1].transpose();
    wmat[1][0] = _01tr;
  }
  //just flavor
  void tranpose_flavor(){
    WilsonMatrix _01 = wmat[0][1];
    wmat[0][1] = wmat[1][0];
    wmat[1][0] = _01;
  }
  Complex& operator()(int s1, int c1, int f1, int s2, int c2, int f2){
    return wmat[f1][f2](s1,c1,s2,c2);
  }
  WilsonMatrix &operator()(int f1,int f2){
    return wmat[f1][f2];
  }
  const WilsonMatrix &operator()(int f1,int f2) const{
    return wmat[f1][f2];
  }
};
#endif
CPS_END_NAMESPACE
