#ifndef _PROP_WRAPPER_H
#define _PROP_WRAPPER_H

#include <util/spincolorflavormatrix.h>

CPS_START_NAMESPACE
//The G-parity complex-conjugate relation allows us to in-place flip the source momentum. We therefore only compute the propagator for one parity. This class wraps that result and computes
//the parity partner on the fly in a way that hides this complexity

class PropWrapper{
  QPropW *prop_f0;
  QPropW *prop_f1;
  bool flip;
public:
  PropWrapper(): prop_f0(NULL), prop_f1(NULL), flip(false){}
  PropWrapper(QPropW *_prop_f0, QPropW *_prop_f1 = NULL, const bool _flip = false): prop_f0(_prop_f0), prop_f1(_prop_f1), flip(_flip){
    if(_prop_f1 != NULL && !GJP.Gparity()) ERR.General("PropWrapper","PropWrapper","For standard BCs expect only one propagator flavor\n");
    if(flip && !GJP.Gparity()) ERR.General("PropWrapper","PropWrapper","Momentum flip operation only possible for G-parity propagators\n");
  }
    
  inline QPropW* getPtr(const int f = 0){ return f==0 ? prop_f0 : prop_f1; }

  void siteMatrix(SpinColorFlavorMatrix &into, const int x, const PropSplane splane = SPLANE_BOUNDARY) const{
    assert(GJP.Gparity());
    into.generate(*prop_f0, *prop_f1, x, splane);
    if(flip) into.flipSourceMomentum();
  }
  void siteMatrix(WilsonMatrix &into, const int x, const PropSplane splane = SPLANE_BOUNDARY) const{
    assert(!GJP.Gparity());
    into = (splane == SPLANE_BOUNDARY ? prop_f0->SiteMatrix(x) : prop_f0->MidPlaneSiteMatrix(x));
  }

};




CPS_END_NAMESPACE
#endif
