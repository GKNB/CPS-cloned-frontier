#ifndef _MULT_IMPL
#define _MULT_IMPL


//Options for _mult_impl:
//MULT_IMPL_BASIC        :not using any external libraries
//MULT_IMPL_BLOCK_BASIC  :blocked matrix implementation without external libraries
//MULT_IMPL_GSL          :blocked matrix implementation using GSL BLAS (and other BLAS can be slotted in by linking to the appropriate libraries)
//MULT_IMPL_GPUBLAS      :using CUDA CUBLASXT or HIP ROCBLAS library
//MULT_IMPL_ONEMKL       :using Intel oneMKL

#if defined(GRID_CUDA) || defined(GRID_HIP)
//Use cuda/hip version
#define MULT_IMPL_GPUBLAS

#elif defined(GRID_SYCL)
//Use oneMKL version
#define MULT_IMPL_ONEMKL

#else
//Default to GSL version
#define MULT_IMPL_GSL
//#define MULT_IMPL_BASIC
//#define MULT_IMPL_BLOCK_BASIC
#endif

#if defined(MULT_IMPL_BASIC)
#  include "mesonfield_mult_impl_basic.tcc"
#elif defined(MULT_IMPL_BLOCK_BASIC)
#  include "mesonfield_mult_impl_block_basic.tcc"
#elif defined(MULT_IMPL_GSL)
#  include "mesonfield_mult_impl_gsl.tcc"
#elif defined(MULT_IMPL_GPUBLAS)
#  include "mesonfield_mult_impl_gpublas.tcc"
#elif defined(MULT_IMPL_ONEMKL)
#  include "mesonfield_mult_impl_onemkl.tcc"
#else
#  error Must specify a MULT_IMPL_* in mesonfield_mult_impl.tcc
#endif

//Matrix product of meson field pairs
//out(t1,t4) = l(t1,t2) * r(t3,t4)     (The stored timeslices are only used to unpack TimePackedIndex so it doesn't matter if t2 and t3 are thrown away; their indices are contracted over hence the times are not needed)
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
void mult(A2AmesonField<mf_Policies,lA2AfieldL,rA2AfieldR> &out, const A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> &l, const A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> &r, const bool node_local){
  _mult_impl<mf_Policies,lA2AfieldL,lA2AfieldR,rA2AfieldL,rA2AfieldR>::mult(out,l,r, node_local);
}


#endif
