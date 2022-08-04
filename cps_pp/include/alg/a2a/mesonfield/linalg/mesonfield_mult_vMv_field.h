#ifndef _A2A_MESONFIELD_MULT_VMV_FIELD_H
#define _A2A_MESONFIELD_MULT_VMV_FIELD_H

#include<alg/a2a/mesonfield/mesonfield.h>
#include "implementation/mesonfield_mult_vMv_field_offload.h";

CPS_START_NAMESPACE

//Multiply v * M * v  (where v are A2A fields and M is a meson field) for all sites
//Output is a CPSfield<Matrix>  where Matrix is a spin-color(-flavor) matrix
//conj_l and conj_r control whether the left/right vectors have the complex conjugate applied

//Function is *node local*
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR>
void mult(typename mult_vMv_field<mf_Policies, lA2AfieldL, lA2AfieldR, rA2AfieldL, rA2AfieldR>::PropagatorField &into,
	  const lA2AfieldL<mf_Policies> &l, 
	  const A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> &M, 
	  const rA2AfieldR<mf_Policies> &r,
	  bool conj_l, bool conj_r){
  return mult_vMv_field<mf_Policies, lA2AfieldL, lA2AfieldR, rA2AfieldL, rA2AfieldR>::implementation(into, l, M, r, conj_l, conj_r);
}


//Multiply v * M * v  (where v are A2A fields and M is a meson field) for all spatial sites and all full lattice timeslices  0 <= t < Lt for which t_start <= t <= t_start + t_dis and for which t is on the current node
//Output is a CPSfield<Matrix>  where Matrix is a spin-color(-flavor) matrix
//conj_l and conj_r control whether the left/right vectors have the complex conjugate applied

//Function is *node local*
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR>
void mult(typename mult_vMv_field<mf_Policies, lA2AfieldL, lA2AfieldR, rA2AfieldL, rA2AfieldR>::PropagatorField &into,
	  const lA2AfieldL<mf_Policies> &l, 
	  const A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> &M, 
	  const rA2AfieldR<mf_Policies> &r,
	  bool conj_l, bool conj_r,
	  const int t_start, const int t_dis){
  return mult_vMv_field<mf_Policies, lA2AfieldL, lA2AfieldR, rA2AfieldL, rA2AfieldR>::implementation(into, l, M, r, conj_l, conj_r, t_start, t_dis);
}



CPS_END_NAMESPACE
#endif
