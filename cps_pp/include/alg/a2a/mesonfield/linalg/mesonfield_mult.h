#ifndef _A2A_MESONFIELD_MULT_H
#define _A2A_MESONFIELD_MULT_H

#include<alg/a2a/mesonfield/mesonfield.h>

CPS_START_NAMESPACE

//Matrix product of meson field pairs
//out(t1,t4) = l(t1,t2) * r(t3,t4)     (The stored timeslices are only used to unpack TimePackedIndex so it doesn't matter if t2 and t3 are thrown away; their indices are contracted over hence the times are not needed)
//Threaded and node distributed. 
//Node-locality can be enabled with 'node_local = true'
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
void mult(A2AmesonField<mf_Policies,lA2AfieldL,rA2AfieldR> &out, const A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> &l, const A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> &r, const bool node_local = false);


// l^i(xop,top) M^ij r^j(xop,top)
//argument xop is the *local* 3d site index in canonical ordering, top is the *local* time coordinate
// Node local and unthreaded
//For G-parity BCs
template<typename mf_Policies, 
	 template <typename> class lA2Afield,  
	 template <typename> class MA2AfieldL,  template <typename> class MA2AfieldR,
	 template <typename> class rA2Afield  
	 >
void mult(CPSspinColorFlavorMatrix<typename mf_Policies::ComplexType> &out, const typename lA2Afield<mf_Policies>::View &l,  const A2AmesonField<mf_Policies,MA2AfieldL,MA2AfieldR> &M, const typename rA2Afield<mf_Policies>::View &r, 
	  const int xop, const int top, const bool conj_l, const bool conj_r);

//For regular BCs
template<typename mf_Policies, 
	 template <typename> class lA2Afield,  
	 template <typename> class MA2AfieldL,  template <typename> class MA2AfieldR,
	 template <typename> class rA2Afield  
	 >
void mult(CPSspinColorMatrix<typename mf_Policies::ComplexType> &out, const typename lA2Afield<mf_Policies>::View &l,  const A2AmesonField<mf_Policies,MA2AfieldL,MA2AfieldR> &M, const typename rA2Afield<mf_Policies>::View &r, 
	  const int xop, const int top, const bool conj_l, const bool conj_r);


// l^i(xop,top) r^i(xop,top)
//argument xop is the *local* 3d site index in canonical ordering, top is the *local* time coordinate
// Node local and unthreaded
//For G-parity BCs
template<class lA2AfieldView, class rA2AfieldView, typename std::enable_if<std::is_same<typename lA2AfieldView::Policies,typename rA2AfieldView::Policies>::value,int>::type = 0 >
void mult(CPSspinColorFlavorMatrix<typename lA2AfieldView::Policies::ComplexType> &out, const lA2AfieldView &l, const rA2AfieldView &r, 
	  const int xop, const int top, const bool conj_l, const bool conj_r);

//For regular BCs
template<class lA2AfieldView, class rA2AfieldView, typename std::enable_if<std::is_same<typename lA2AfieldView::Policies,typename rA2AfieldView::Policies>::value,int>::type = 0 >
void mult(CPSspinColorMatrix<typename lA2AfieldView::Policies::ComplexType> &out, const lA2AfieldView &l, const rA2AfieldView &r, 
	  const int xop, const int top, const bool conj_l, const bool conj_r);

#include "implementation/mesonfield_mult_impl.tcc"
#include "implementation/mesonfield_mult_vMv_impl.tcc"
#include "implementation/mesonfield_mult_vv_impl.tcc"


CPS_END_NAMESPACE

#endif
