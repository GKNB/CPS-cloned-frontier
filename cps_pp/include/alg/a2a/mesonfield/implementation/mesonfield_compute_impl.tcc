//Compute meson fields from V and W vectors. Include host implementation always and offload (GPU) version if offloading. Choose latter for default function if offloading with Grid vector policies
#include "mesonfield_compute_impl_cpu.tcc" //host implementation

#ifdef GRID_NVCC
#include "mesonfield_compute_impl_offload.tcc" //offload implementation

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
template<typename InnerProduct, typename Allocator>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::compute(std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator > &mf_t,
							     const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
  typedef typename _choose_mf_mult_impl_offload<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct,Allocator, typename ComplexClassify<typename mf_Policies::ComplexType>::type>::ComputeImplSingle Impl;  
  Impl cg;
  cg.compute(mf_t,l,M,r,do_setup);
}

//Version of the above for multi-src inner products (output vector indexed by [src idx][t]
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
template<typename InnerProduct, typename Allocator>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::compute(std::vector< std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator >* > &mf_st,
							     const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
  typedef typename _choose_mf_mult_impl_offload<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct,Allocator, typename ComplexClassify<typename mf_Policies::ComplexType>::type>::ComputeImplMulti Impl;  
  Impl cg;
  cg.compute(mf_st,l,M,r,do_setup);
}

#else

template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
template<typename InnerProduct, typename Allocator>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::compute(std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator > &mf_t,
							     const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
  typedef typename _choose_vector_policies<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct,Allocator, typename ComplexClassify<typename mf_Policies::ComplexType>::type>::SingleSrcVectorPoliciesT VectorPolicies;  
  mfComputeGeneral<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct, VectorPolicies> cg;
  cg.compute(mf_t,l,M,r,do_setup);
}

//Version of the above for multi-src inner products (output vector indexed by [src idx][t]
template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
template<typename InnerProduct, typename Allocator>
void A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::compute(std::vector< std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator >* > &mf_st,
							     const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup){
  typedef typename _choose_vector_policies<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct,Allocator, typename ComplexClassify<typename mf_Policies::ComplexType>::type>::MultiSrcVectorPoliciesT VectorPolicies;  
  mfComputeGeneral<mf_Policies,A2AfieldL,A2AfieldR,InnerProduct, VectorPolicies> cg;
  cg.compute(mf_st,l,M,r,do_setup);
}

#endif
