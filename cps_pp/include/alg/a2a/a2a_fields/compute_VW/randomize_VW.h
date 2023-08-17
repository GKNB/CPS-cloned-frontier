#include<alg/a2a/a2a_fields/field_vectors.h>

CPS_START_NAMESPACE

template<typename mf_Policies,typename ComplexClass>
struct _randomizeVWimpl{};

//Randomize the V and W vectors
template<typename mf_Policies>
void randomizeVW(A2AvectorV<mf_Policies> &V, A2AvectorW<mf_Policies> &W){
#ifndef MEMTEST_MODE
  return _randomizeVWimpl<mf_Policies,typename ComplexClassify<typename mf_Policies::ComplexType>::type>::randomizeVW(V,W);
#endif
}

#include "implementation/randomize_VW.tcc"

CPS_END_NAMESPACE
