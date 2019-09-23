#ifndef _INNER_PRODUCT_FMATSPINCOLOR_CONTRACT
#define _INNER_PRODUCT_FMATSPINCOLOR_CONTRACT

#include<alg/a2a/inner_product_spincolorcontract.h>

//Optimized code for taking the spin-color contraction of 12-component complex vectors in a G-parity context, forming a flavor matrix
CPS_START_NAMESPACE

//Tie together the spin-color structure to form a flavor matrix   lMr[f1,f3] =  l[sc1,f1]^T M[sc1,sc2] r[sc2,f3]
template<int smatidx, bool conj_left, bool conj_right>
struct flavorMatrixSpinColorContract{

  //std::complex type
  template<typename ComplexType>
  static inline typename my_enable_if<_equal<typename ComplexClassify<ComplexType>::type,complex_double_or_float_mark>::value, void>::type
  spinColorContract(FlavorMatrixGeneral<ComplexType> &lMr, const SCFvectorPtr<ComplexType> &l, const SCFvectorPtr<ComplexType> &r){
#ifdef USE_GRID_SCCON
    grid_scf_contract_select<smatidx,ComplexType,conj_left,conj_right>::doit(lMr,l,r);
#else
    const ComplexType zero(0.,0.);
    for(int f1=0;f1<2;f1++)
      for(int f3=0;f3<2;f3++)
	lMr(f1,f3) = l.isZero(f1) || r.isZero(f3) ? zero : SpinColorContractSelect<smatidx,ComplexType,conj_left,conj_right>::doit(l.getPtr(f1),r.getPtr(f3));
#endif
  }

#ifdef USE_GRID
  //Grid SIMD complex type
  template<typename ComplexType>
  static accelerator_inline typename my_enable_if<_equal<typename ComplexClassify<ComplexType>::type,grid_vector_complex_mark>::value, void>::type
  spinColorContract(FlavorMatrixGeneral<typename SIMT<ComplexType>::value_type> &lMr, const SCFvectorPtr<ComplexType> &l, const SCFvectorPtr<ComplexType> &r){
    typedef typename SIMT<ComplexType>::value_type SIMT_ztype;
    
    lMr(0,0) = l.isZero(0) || r.isZero(0) ? SIMT_ztype(0.) : GridVectorizedSpinColorContractSelect<smatidx,ComplexType,conj_left,conj_right>::doit(l.getPtr(0),r.getPtr(0));
    lMr(0,1) = l.isZero(0) || r.isZero(1) ? SIMT_ztype(0.) : GridVectorizedSpinColorContractSelect<smatidx,ComplexType,conj_left,conj_right>::doit(l.getPtr(0),r.getPtr(1));
    lMr(1,0) = l.isZero(1) || r.isZero(0) ? SIMT_ztype(0.) : GridVectorizedSpinColorContractSelect<smatidx,ComplexType,conj_left,conj_right>::doit(l.getPtr(1),r.getPtr(0));
    lMr(1,1) = l.isZero(1) || r.isZero(1) ? SIMT_ztype(0.) : GridVectorizedSpinColorContractSelect<smatidx,ComplexType,conj_left,conj_right>::doit(l.getPtr(1),r.getPtr(1));
  }
#endif
};



CPS_END_NAMESPACE

#endif
