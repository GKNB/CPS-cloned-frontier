#ifndef _MESONFIELD_MULT_VMV_SPLIT_LITE_IMPL_TCC
#define  _MESONFIELD_MULT_VMV_SPLIT_LITE_IMPL_TCC

template<typename mf_Policies,
	 typename ComplexClass>
struct mult_vMv_split_lite_cnum_policy;

template<typename mf_Policies>
struct mult_vMv_split_lite_cnum_policy<mf_Policies, complex_double_or_float_mark>{
  typedef typename mf_Policies::ComplexType ComplexType;

  template<typename FermionFieldType>
  inline static void checkDecomp(const FermionFieldType &field){}
  
  inline static void zeroit(ComplexType &z){
    memset(&z, 0, sizeof(ComplexType));
  }

  inline static void splat(ComplexType &out, const ComplexType &in){
    out = in;
  }
  inline static ComplexType cconj(const ComplexType &in){
    return conj(in);
  }
};

#ifdef USE_GRID

template<typename mf_Policies>
struct mult_vMv_split_lite_cnum_policy<mf_Policies, grid_vector_complex_mark>{  //for SIMD vectorized W and V vectors
  typedef typename mf_Policies::ComplexType ComplexType;
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;

  template<typename FermionFieldType>
  inline static void checkDecomp(const FermionFieldType &field){
    assert(field.SIMDlogicalNodes(3) == 1);    
  }

  inline static void zeroit(ComplexType &z){
    Grid::zeroit(z);
  }

  inline static void splat(ComplexType &out, const ScalarComplexType &in){
    Grid::vsplat(out, in);
  }
  inline static ComplexType cconj(const ComplexType &in){
    return Grid::conjugate(in);
  }
};

#endif


#endif
