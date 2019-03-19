#ifndef _AMA_PROP_SITEMATRIX_GETTER_H
#define _AMA_PROP_SITEMATRIX_GETTER_H

#include "propwrapper.h"
#include "propmomcontainer.h"
#include "wallsinkprop.h"

CPS_START_NAMESPACE

//A class to manage the getting of the propagator for a particular local 3d coordinate and a *global* source->sink separation tdis_glb.  tdis_glb is linear, i.e. the user doesn't need to worry about the modulo Lt wrapping
//The class knows about the periodicity of the propagator
//Note, the resulting site must be available on this node - no comms are performed. This is not a huge restriction as the shifts we perform are in units of Lt
class PropSiteMatrixGetter{
public:
  //(tsrc + tdis_glb) % Lt must be on node (runtime check performed)
  virtual void siteMatrix(WilsonMatrix &into, const int x3d_lcl, const int tdis_glb, const PropSplane splane = SPLANE_BOUNDARY) const = 0;
  virtual void siteMatrix(SpinColorFlavorMatrix &into, const int x3d_lcl, const int tdis_glb, const PropSplane splane = SPLANE_BOUNDARY) const = 0;
  virtual void shiftSourcenLt(const int n) = 0; //use the known periodicity to shift the source timeslice by n*Lt
  virtual ~PropSiteMatrixGetter(){}
};

//For propagators with modulo-Lt periodicity or antiperiodicity
class PropSiteMatrixStandard: public PropSiteMatrixGetter{
  PropWrapper prop;
  BndCndType bc;
  int tsrc;
  int base_sgn;
  int tdis_shift;  //if we translate the source internally using the BCs we also have to translate the src->snk sep
  //G(x,y) = G'(x+nLt,y) = G'(x',y)
  //y = tdis + x      y = tdis + x'-nLt

  
  inline void get4dcoordAndSign(int &x4d_lcl, int &sgn, const int x3d_lcl, const int tdis_glb) const{
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    sgn = base_sgn;
    int t = tdis_glb + tdis_shift + tsrc;
    while(t<0 || t>=Lt){ //use periodicity in sink coordinate
      if(t<0) t+= Lt;
      else t-=Lt;
      sgn *= (BND_CND_APRD ? -1 : 1); //APRD   G(t-Lt,t') = -G(t,t')
    }
    int t_lcl = t - GJP.TnodeCoor() * GJP.TnodeSites();
    assert(t_lcl>=0 && t_lcl<GJP.TnodeSites());
    x4d_lcl = x3d_lcl + GJP.VolNodeSites()/GJP.TnodeSites()*t_lcl;
  }


public:
 PropSiteMatrixStandard(const PropWrapper &_prop, const BndCndType _bc, const int _tsrc): prop(_prop), bc(_bc), tsrc(_tsrc), base_sgn(1), tdis_shift(0){}

  void siteMatrix(WilsonMatrix &into, const int x3d_lcl, const int tdis_glb, const PropSplane splane = SPLANE_BOUNDARY) const{
    int sgn, x4d_lcl;
    get4dcoordAndSign(x4d_lcl,sgn,x3d_lcl,tdis_glb);
    prop.siteMatrix(into,x4d_lcl,splane);
    if(!sgn) into *= Float(-1);
  }
  void siteMatrix(SpinColorFlavorMatrix &into, const int x3d_lcl, const int tdis_glb, const PropSplane splane = SPLANE_BOUNDARY) const{
    int sgn, x4d_lcl;
    get4dcoordAndSign(x4d_lcl,sgn,x3d_lcl,tdis_glb);
    prop.siteMatrix(into,x4d_lcl,splane);
    if(!sgn) into *= Float(-1);
  }
  void shiftSourcenLt(const int n){
    int absn = abs(n);
    tdis_shift -= n*GJP.TnodeSites()*GJP.Tnodes();
    for(int i=0;i<absn;i++) base_sgn *= (BND_CND_APRD ? -1 : 1); //APRD   G(t,t') = -G(t,t'+/-Lt)
  }
};


//For F=P+A and B=P-A propagators with modulo-2Lt periodicity
//Utilize F(t+Lt,t') = B(t,t') and B(t+Lt,t') = F(t,t')
//Need both the F and B propagators. The 'base' propagator is either F or B, and the 'shift' propagator is B or F respectively
class PropSiteMatrixFB: public PropSiteMatrixGetter{
  PropWrapper prop_base;
  PropWrapper prop_shift;
  int tsrc;
  int use_base;
  int tdis_shift; //cf above

  
  inline PropWrapper const* get4dcoordAndProp(int &x4d_lcl, const int x3d_lcl, const int tdis_glb) const{
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    int t = tdis_glb + tdis_shift + tsrc;
    PropWrapper const* props[2] = { &prop_base, &prop_shift };
    int use = use_base;
    while(t<0 || t>=Lt){
      if(t<0) t+= Lt;
      else t-=Lt;
      use = (use + 1) % 2;
    }
    int t_lcl = t - GJP.TnodeCoor() * GJP.TnodeSites();
    if(t_lcl < 0 || t_lcl >= GJP.TnodeSites()){
      printf("PropSiteMatrixFB::get4dcoordAndProp for Node %d with t range %d to %d, tdis_glb=%d and tsrc=%d, got lattice t=%d and prop %d. Lattice t is not on node!\n",
	     UniqueID(),
	     GJP.TnodeCoor()*GJP.TnodeSites(), (GJP.TnodeCoor()+1)*GJP.TnodeSites()-1,
	     tdis_glb,tsrc,t,use); 
      fflush(stdout);
      exit(-1);
    }
    x4d_lcl = x3d_lcl + GJP.VolNodeSites()/GJP.TnodeSites()*t_lcl;
    // if(!UniqueID()){ printf("PropSiteMatrixFB::get4dcoordAndProp for Node %d with t range %d to %d, tdis_glb=%d and tsrc=%d, got lattice t=%d and prop %d.\n",
    // 			    UniqueID(),
    // 			    GJP.TnodeCoor()*GJP.TnodeSites(), (GJP.TnodeCoor()+1)*GJP.TnodeSites()-1,
    // 			    tdis_glb,tsrc,t,use); fflush(stdout); }
    return props[use];
  }


public:
  PropSiteMatrixFB(const PropWrapper &_prop_base, const PropWrapper &_prop_shift, const int _tsrc): prop_base(_prop_base),prop_shift(_prop_shift),tsrc(_tsrc),use_base(0), tdis_shift(0){}

  void siteMatrix(WilsonMatrix &into, const int x3d_lcl, const int tdis_glb, const PropSplane splane = SPLANE_BOUNDARY) const{
    int x4d_lcl;
    PropWrapper const* prop_use = get4dcoordAndProp(x4d_lcl,x3d_lcl,tdis_glb);
    prop_use->siteMatrix(into,x4d_lcl,splane);
  }
  void siteMatrix(SpinColorFlavorMatrix &into, const int x3d_lcl, const int tdis_glb, const PropSplane splane = SPLANE_BOUNDARY) const{
    int x4d_lcl;
    PropWrapper const* prop_use = get4dcoordAndProp(x4d_lcl,x3d_lcl,tdis_glb);
    prop_use->siteMatrix(into,x4d_lcl,splane);
  }
  void shiftSourcenLt(const int n){
    int absn = abs(n);
    tdis_shift -= n*GJP.TnodeSites()*GJP.Tnodes();
    for(int i=0;i<absn;i++) use_base = (use_base + 1) % 2; //F(t,t'+/-Lt) = B(t,t'),  B(t,t'+/-Lt) = F(t,t')
  }
};


template<typename MatrixType>
class WallSinkPropSiteMatrixGetter{
public:
  virtual void siteMatrix(MatrixType &into, const int tdis_glb) const = 0;
  virtual ~WallSinkPropSiteMatrixGetter(){}
};

template<typename MatrixType>
class WallSinkPropSiteMatrixStandard: public WallSinkPropSiteMatrixGetter<MatrixType>{
  WallSinkProp<MatrixType> prop;
  BndCndType bc;
  int tsrc;

  inline void getSinkTimeAndSign(int &t, int &sgn, const int tdis_glb) const{
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    sgn = 1;
    t = tdis_glb + tsrc;
    while(t<0 || t>=Lt){
      if(t<0) t+= Lt;
      else t-=Lt;
      sgn *= (BND_CND_APRD ? -1 : 1); //APRD   G(t-Lt) = -G(t)
    }
  }

public:
  WallSinkPropSiteMatrixStandard(const PropWrapper &prop_in, const BndCndType _bc, const int _tsrc, const ThreeMomentum &p_snk, Lattice &lat, const bool gauge_fix_sink = true): bc(_bc), tsrc(_tsrc), prop(gauge_fix_sink){
    prop.setProp(prop_in);
    prop.compute(lat,p_snk);
  }
  WallSinkPropSiteMatrixStandard(const PropWrapper &prop_in, const BndCndType _bc, const int _tsrc, const double *p_snk, Lattice &lat, const bool gauge_fix_sink = true): bc(_bc), tsrc(_tsrc), prop(gauge_fix_sink){
    prop.setProp(prop_in);
    prop.compute(lat,p_snk);
  }

  void siteMatrix(MatrixType &into, const int tdis_glb) const{
    int sgn, t_glb;
    getSinkTimeAndSign(t_glb,sgn,tdis_glb);
    into = prop(t_glb);
    if(!sgn) into *= Float(-1);
  }

  WallSinkProp<MatrixType> & getWallSinkProp(){ return prop; }
};

template<typename MatrixType>
class WallSinkPropSiteMatrixFB: public WallSinkPropSiteMatrixGetter<MatrixType>{
  WallSinkProp<MatrixType> prop_base;
  WallSinkProp<MatrixType> prop_shift;
  int tsrc;

  inline WallSinkProp<MatrixType> const* getSinkTimeAndProp(int &t, const int tdis_glb) const{
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    t = tdis_glb + tsrc;
    WallSinkProp<MatrixType> const* props[2] = { &prop_base, &prop_shift };
    int use = 0;
    while(t<0 || t>=Lt){
      if(t<0) t+= Lt;
      else t-=Lt;
      use = (use + 1) % 2;
    }
    return props[use];
  }

public:
  WallSinkPropSiteMatrixFB(const PropWrapper &prop_base_in, const PropWrapper &prop_shift_in, const int _tsrc, 
			   const ThreeMomentum &p_snk, Lattice &lat, const bool gauge_fix_sink = true): tsrc(_tsrc),prop_base(gauge_fix_sink), prop_shift(gauge_fix_sink){
    prop_base.setProp(prop_base_in);
    prop_base.compute(lat,p_snk);

    prop_shift.setProp(prop_shift_in);
    prop_shift.compute(lat,p_snk);
  }


  void siteMatrix(MatrixType &into, const int tdis_glb) const{
    int t_glb;
    WallSinkProp<MatrixType> const* prop_use = getSinkTimeAndProp(t_glb,tdis_glb);
    into = prop_use->operator()(t_glb);
  }
};


class PropGetter{
public:
  //0<=t<Lt
  virtual std::auto_ptr<PropSiteMatrixGetter> get(const int t_torus, const ThreeMomentum &psrc) const = 0;
  
  //-\infty <= t <= \infty
  std::auto_ptr<PropSiteMatrixGetter> operator()(const int t, const ThreeMomentum &psrc) const{
    //If t is across the boundary we must use the periodicity of the propagators to shift it back to 0<=t<Lt
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    const int nLt_shift = -t/Lt;
    const int t_torus = t % Lt;
    std::auto_ptr<PropSiteMatrixGetter> ret( this->get(t_torus,psrc) );
    if(nLt_shift != 0) ret->shiftSourcenLt(nLt_shift);
    return ret;
  }

  //-\infty <= t <= \infty
  virtual bool contains(const int t, const ThreeMomentum &psrc) const = 0;  
};
class PropGetterFB: public PropGetter{
  const Props &props_base;
  const Props &props_shift;

public:
  PropGetterFB(const Props &_props_base, const Props &_props_shift): props_base(_props_base),  props_shift(_props_shift){}
  
  std::auto_ptr<PropSiteMatrixGetter> get(const int t_torus, const ThreeMomentum &psrc) const{
    const PropWrapper &pF = props_base(t_torus,psrc);
    const PropWrapper &pB = props_shift(t_torus,psrc);
    return std::auto_ptr<PropSiteMatrixGetter>(new PropSiteMatrixFB(pF,pB,t_torus));
  }
  bool contains(const int t, const ThreeMomentum &psrc) const{
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    const int t_torus = t % Lt;
    return props_base.contains(t_torus,psrc) && props_shift.contains(t_torus,psrc);
  }
    
};
class PropGetterStd: public PropGetter{
  const Props &props;
  const BndCndType time_bc;
public:
 PropGetterStd(const Props &_props, const BndCndType _time_bc): props(_props), time_bc(GJP.Tbc()){}
  
  std::auto_ptr<PropSiteMatrixGetter> get(const int t_torus, const ThreeMomentum &psrc) const{
    const PropWrapper &p = props(t_torus,psrc);
    return std::auto_ptr<PropSiteMatrixGetter>(new PropSiteMatrixStandard(p,time_bc,t_torus));
  }

  bool contains(const int t, const ThreeMomentum &psrc) const{
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();
    const int t_torus = t % Lt;
    return props.contains(t_torus,psrc);
  }
};


template<typename MatrixType>
class WallSinkPropGetter{
protected:
  Lattice &lattice;
  bool gauge_fix_sink;
public:
  WallSinkPropGetter(Lattice &_lattice, bool _gauge_fix_sink = true): lattice(_lattice), gauge_fix_sink(_gauge_fix_sink){}
  
  virtual std::auto_ptr<WallSinkPropSiteMatrixGetter<MatrixType> > operator()(const int t, const ThreeMomentum &psrc, const ThreeMomentum &psnk) const = 0;
};
template<typename MatrixType>
class WallSinkPropGetterFB: public WallSinkPropGetter<MatrixType>{
  const Props &props_base;
  const Props &props_shift;

public:
  WallSinkPropGetterFB(const Props &_props_base, const Props &_props_shift, Lattice &_lattice, bool _gauge_fix_sink = true): props_base(_props_base),  props_shift(_props_shift),
															     WallSinkPropGetter<MatrixType>(_lattice,_gauge_fix_sink){}
  
  std::auto_ptr<WallSinkPropSiteMatrixGetter<MatrixType> > operator()(const int t, const ThreeMomentum &psrc, const ThreeMomentum &psnk) const{
    const PropWrapper &pF = props_base(t,psrc);
    const PropWrapper &pB = props_shift(t,psrc);
    return std::auto_ptr<WallSinkPropSiteMatrixGetter<MatrixType> >(new WallSinkPropSiteMatrixFB<MatrixType>(pF,pB,t,psnk,this->lattice,this->gauge_fix_sink));
  }
};
template<typename MatrixType>
class WallSinkPropGetterStd: public WallSinkPropGetter<MatrixType>{
  const Props &props;
  const BndCndType time_bc;
public:
  WallSinkPropGetterStd(const Props &_props, const BndCndType _time_bc, Lattice &_lattice, bool _gauge_fix_sink = true): props(_props), time_bc(_time_bc),
    WallSinkPropGetter<MatrixType>(_lattice,_gauge_fix_sink){}
  
  std::auto_ptr<WallSinkPropSiteMatrixGetter<MatrixType> > operator()(const int t, const ThreeMomentum &psrc, const ThreeMomentum &psnk) const{
    const PropWrapper &p = props(t,psrc);
    return std::auto_ptr<WallSinkPropSiteMatrixGetter<MatrixType> >(new WallSinkPropSiteMatrixStandard<MatrixType>(p,time_bc,t,psnk,this->lattice,this->gauge_fix_sink));
  }
};


CPS_END_NAMESPACE

#endif
