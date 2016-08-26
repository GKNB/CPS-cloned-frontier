#ifndef _SCF_VECTOR_PTR_H
#define _SCF_VECTOR_PTR_H

//A class that hides away the pointer arithmetic for a spin-color-flavor vector
//ComplexType is the basic type of each element of the vectors. Usually std::complex<double> or same for float
template<typename ComplexType>
class SCFvectorPtr{
  ComplexType const* p[2]; //one for each *flavor*
  bool zero_hint[2]; //a hint that the pointer points to a set of zeroes
public:
  inline SCFvectorPtr(ComplexType const* f0, ComplexType const* f1 = NULL, bool zero_f0=false, bool zero_f1=false )
 #if __cplusplus >= 201103L
     : p{f0,f1}, zero_hint{zero_f0,zero_f1}{}
#else
{ p[0] = f0; p[1] = f1; zero_hint[0] = zero_f0; zero_hint[1] = zero_f1; }
#endif
  inline void assign(const int f, ComplexType* fp){ p[f] = fp; }
  inline const ComplexType & operator()(const int s, const int c, const int f = 0) const{
    return p[f][c+3*s];
  }
  inline const ComplexType & scelem(const int sc,const int f = 0) const{ //sc = c+3*s
    return p[f][sc];
  }
  inline ComplexType const* getPtr(const int f) const{ return p[f]; }
  inline bool isZero(const int f) const{ return zero_hint[f]; }
};




#endif
