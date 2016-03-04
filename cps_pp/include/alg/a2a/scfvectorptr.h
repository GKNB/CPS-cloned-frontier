#ifndef _SCF_VECTOR_PTR_H
#define _SCF_VECTOR_PTR_H

//A class that hides away the pointer arithmetic for a spin-color-flavor vector
//mf_Float is the storage precision for the vectors
template<typename mf_Float>
class SCFvectorPtr{
  mf_Float const* p[2];
  bool zero_hint[2]; //a hint that the pointer points to a set of zeroes
public:
  inline SCFvectorPtr(mf_Float const* f0, mf_Float const* f1 = NULL, bool zero_f0=false, bool zero_f1=false )
 #if __cplusplus >= 201103L
     : p{f0,f1}, zero_hint{zero_f0,zero_f1}{}
#else
{ p[0] = f0; p[1] = f1; zero_hint[0] = zero_f0; zero_hint[1] = zero_f1; }
#endif
  inline void assign(const int f, mf_Float* fp){ p[f] = fp; }
  inline const std::complex<mf_Float> & operator()(const int s, const int c, const int f = 0) const{
    mf_Float const* cbase = p[f] + 2*(c+3*s);
    return *reinterpret_cast<std::complex<mf_Float> const* const>(cbase);
  }
  inline const std::complex<mf_Float> & scelem(const int sc,const int f = 0) const{
    mf_Float const* cbase = p[f] + 2*sc;
    return *reinterpret_cast<std::complex<mf_Float> const* const>(cbase);
  }
  inline mf_Float const* getPtr(const int f) const{ return p[f]; }
  inline bool isZero(const int f) const{ return zero_hint[f]; }
};




#endif
