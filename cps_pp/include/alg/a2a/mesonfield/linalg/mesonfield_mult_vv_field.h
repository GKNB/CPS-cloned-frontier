#ifndef _A2A_MESONFIELD_MULT_VV_FIELD_H
#define _A2A_MESONFIELD_MULT_VV_FIELD_H

#include<alg/a2a/mesonfield/mesonfield.h>
#include "implementation/mesonfield_mult_vv_field_offload.h";

CPS_START_NAMESPACE

//Multiply v * v  (where v are A2A fields) for all sites
//Output is a CPSfield<Matrix>  where Matrix is a spin-color(-flavor) matrix
//conj_l and conj_r control whether the left/right vectors have the complex conjugate applied
template<typename mf_Policies, 
	 template <typename> class lA2Afield,  template <typename> class rA2Afield>
void mult(typename mult_vv_field<mf_Policies, lA2Afield, rA2Afield>::PropagatorField &into,
	  const lA2Afield<mf_Policies> &l, 
	  const rA2Afield<mf_Policies> &r,
	  bool conj_l, bool conj_r){
  return mult_vv_field<mf_Policies, lA2Afield, rA2Afield>::implementation(into, l, r, conj_l, conj_r);
}

CPS_END_NAMESPACE
#endif
