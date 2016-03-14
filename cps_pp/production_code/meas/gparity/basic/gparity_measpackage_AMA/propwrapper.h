#ifndef _PROP_WRAPPER_H
#define _PROP_WRAPPER_H
CPS_START_NAMESPACE
//The G-parity complex-conjugate relation allows us to in-place flip the source momentum. We therefore only compute the propagator for one parity. This class wraps that result and computes
//the parity partner on the fly in a way that hides this complexity

class PropWrapper{
  QPropW *prop_f0;
  QPropW *prop_f1;
  bool flip;
public:
  PropWrapper(): prop_f0(NULL), prop_f1(NULL), flip(false){}
  PropWrapper(QPropW *_prop_f0, QPropW *_prop_f1, const bool _flip = false): prop_f0(_prop_f0), prop_f1(_prop_f1), flip(_flip){}
    
  inline QPropW* getPtr(const int f){ return f==0 ? prop_f0 : prop_f1; }

  void siteMatrix(SpinColorFlavorMatrix &into, const int x) const{
    into.generate(*prop_f0, *prop_f1, x);
    if(flip) into.flipSourceMomentum();
  }
};




CPS_END_NAMESPACE
#endif
