#ifndef _SCF_VECTOR_PTR_H
#define _SCF_VECTOR_PTR_H

#include<utility>
#include<config.h>

CPS_START_NAMESPACE

//A class that hides away the pointer arithmetic for a spin-color-flavor vector
//ComplexType is the basic type of each element of the vectors. Usually std::complex<double> or same for float
template<typename ComplexType>
class SCFvectorPtr{
  ComplexType const* p[2]; //one for each *flavor*
  bool zero_hint[2]; //a hint that the pointer points to a set of zeroes
public:
  accelerator_inline SCFvectorPtr(ComplexType const* f0, ComplexType const* f1 = NULL, bool zero_f0=false, bool zero_f1=false )
#if __cplusplus >= 201103L
     : p{f0,f1}, zero_hint{zero_f0,zero_f1}{}
#else
{ p[0] = f0; p[1] = f1; zero_hint[0] = zero_f0; zero_hint[1] = zero_f1; }
#endif

  accelerator_inline SCFvectorPtr()
#if __cplusplus >= 201103L
    : p{NULL,NULL}, zero_hint{false,false}{}
#else
{ p[0] = NULL; p[1] = NULL; zero_hint[0] = false; zero_hint[1] = false; }
#endif

  accelerator_inline SCFvectorPtr(const SCFvectorPtr &r)
#if __cplusplus >= 201103L
    : p{r.p[0],r.p[1]}, zero_hint{r.zero_hint[0],r.zero_hint[1]}{}
#else
  { p[0] = r.p[0]; p[1] = r.p[1]; zero_hint[0] = r.zero_hint[0]; zero_hint[1] = r.zero_hint[1]; }
#endif

  //Shift a base pointer container by some number of sites. site_incr is the stride between sites
  accelerator_inline SCFvectorPtr(const SCFvectorPtr &r, const std::pair<int,int> &site_incr, const int sites = 1)
#if __cplusplus >= 201103L
    : p{r.p[0]+sites*site_incr.first, r.p[1]+sites*site_incr.second}, zero_hint{r.zero_hint[0],r.zero_hint[1]}{}
#else
  { p[0] = r.p[0]+sites*site_incr.first; p[1] = r.p[1]+sites*site_incr.second; zero_hint[0] = r.zero_hint[0]; zero_hint[1] = r.zero_hint[1]; }
#endif

  accelerator_inline void assign(const int f, ComplexType* fp){ p[f] = fp; }
  accelerator_inline const ComplexType & operator()(const int s, const int c, const int f = 0) const{
    return p[f][c+3*s];
  }
  accelerator_inline const ComplexType & scelem(const int sc,const int f = 0) const{ //sc = c+3*s
    return p[f][sc];
  }
  accelerator_inline ComplexType const* getPtr(const int f) const{ return p[f]; }
  accelerator_inline bool isZero(const int f) const{ return zero_hint[f]; }
  accelerator_inline void incrementPointers(const int df0, const int df1){ p[0] += df0; p[1] += df1; }
  accelerator_inline void incrementPointers(const std::pair<int,int> df, const int sites = 1){ p[0] += df.first*sites; p[1] += df.second*sites; }
  accelerator_inline void setHint(const int f, const bool to){ zero_hint[f] = to; }
};


CPS_END_NAMESPACE

#endif
