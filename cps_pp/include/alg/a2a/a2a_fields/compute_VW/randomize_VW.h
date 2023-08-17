#include<alg/a2a/a2a_fields/field_vectors.h>

CPS_START_NAMESPACE

template<typename Vtype, typename Wtype,typename ComplexClass>
struct _randomizeVWimpl{};

//Randomize the V and W vectors
template<typename Vtype, typename Wtype>
void randomizeVW(Vtype &V, Wtype &W){
#ifndef MEMTEST_MODE
  return _randomizeVWimpl<Vtype,Wtype,typename ComplexClassify<typename Vtype::Policies::ComplexType>::type>::randomizeVW(V,W);
#endif
}

#include "implementation/randomize_VW.tcc"

CPS_END_NAMESPACE
