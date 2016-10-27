#ifndef _INNER_PRODUCT_FMATSPINCOLOR_CONTRACT
#define _INNER_PRODUCT_FMATSPINCOLOR_CONTRACT

#include<alg/a2a/inner_product_spincolorcontract.h>

//Optimized code for taking the spin-color contraction of 12-component complex vectors in a G-parity context, forming a flavor matrix
CPS_START_NAMESPACE

//Tie together the spin-color structure to form a flavor matrix   lMr[f1,f3] =  l[sc1,f1]^T M[sc1,sc2] r[sc2,f3]
template<int smatidx, typename mf_Complex, bool conj_left, bool conj_right, typename Dummy>
struct flavorMatrixSpinColorContract{};

//std::complex types
template<int smatidx, typename mf_Complex, bool conj_left, bool conj_right>
struct flavorMatrixSpinColorContract<smatidx, mf_Complex, conj_left, conj_right,  complex_double_or_float_mark>{
  
  inline static void doit(FlavorMatrix &lMr, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r){
#ifdef USE_GRID_SCCON
    grid_scf_contract_select<smatidx,mf_Complex,conj_left,conj_right>::doit(lMr,l,r);
#else
    const std::complex<double> zero(0.,0.);
    for(int f1=0;f1<2;f1++)
      for(int f3=0;f3<2;f3++)
	lMr(f1,f3) = l.isZero(f1) || r.isZero(f3) ? zero : SpinColorContractSelect<smatidx,mf_Complex,conj_left,conj_right>::doit(l.getPtr(f1),r.getPtr(f3));
#endif
  }
};

//Grid SIMD vectorized types
template<int smatidx, typename mf_Complex, bool conj_left, bool conj_right>
struct flavorMatrixSpinColorContract<smatidx, mf_Complex, conj_left, conj_right,  grid_vector_complex_mark>{
  
  inline static void doit(FlavorMatrixGeneral<mf_Complex> &lMr, const SCFvectorPtr<mf_Complex> &l, const SCFvectorPtr<mf_Complex> &r){
    const mf_Complex zero(0.);

    for(int f1=0;f1<2;f1++)
      for(int f3=0;f3<2;f3++)
    	lMr(f1,f3) = l.isZero(f1) || r.isZero(f3) ? zero : GridVectorizedSpinColorContractSelect<smatidx,mf_Complex,conj_left,conj_right>::doit(l.getPtr(f1),r.getPtr(f3));
  }
};


CPS_END_NAMESPACE

#endif
