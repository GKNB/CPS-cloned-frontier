#ifndef _INNER_PRODUCT_SPINCOLOR_CONTRACT
#define _INNER_PRODUCT_SPINCOLOR_CONTRACT

#include<alg/a2a/inner_product_grid.h>
#include<alg/a2a/inner_product_avx512.h>
#include<alg/a2a/conj_zmul.h>
#include<alg/a2a/template_wizardry.h>

#ifdef USE_GRID
#define USE_GRID_SCCON //switch on inner_product_grid code
#endif

//Optimized code for taking the spin-color contraction of two 12-component complex vectors
CPS_START_NAMESPACE

//For non-SIMD complex types
//Option USE_GRID_SCCON to use SIMD intrinsics underneath for minor speed-up. Not as optimal as using Grid SIMD vectorized types as native complex type due to additional overheads
template<typename ComplexType, bool conj_left, bool conj_right>
class OptimizedSpinColorContract{
public:
  inline static ComplexType g5(const ComplexType *const l, const ComplexType *const r){
    const static int sc_size =12;
    const static int half_sc = 6;
    
    ComplexType v3(0,0);

#if defined(USE_GRID_SCCON)
    grid_g5contract<ComplexType,conj_left,conj_right>::doit(v3,l,r);
#else
    for(int i = half_sc; i < sc_size; i++){ 
      v3 += Mconj<ComplexType,conj_left,conj_right>::doit(l+i,r+i);
    }
    v3 *= -1;
      
    for(int i = 0; i < half_sc; i ++){ 
      v3 += Mconj<ComplexType,conj_left,conj_right>::doit(l+i,r+i);
    }
#endif

    return v3;
  }
  inline static ComplexType unit(const ComplexType *const l, const ComplexType *const r){
    const static int sc_size =12;
    ComplexType v3(0,0);

#ifdef USE_GSL_SCCON    
    typedef gsl_wrapper<typename ComplexType::value_type> gw;
    
    typename gw::block_complex_struct lblock;
    typename gw::vector_complex lgsl;
    typename gw::block_complex_struct rblock;
    typename gw::vector_complex rgsl;
    typename gw::complex result;

    lblock.size = sc_size;
    lgsl.block = &lblock;
    lgsl.size = sc_size;
    lgsl.stride = 1;
    lgsl.owner = 0;
      
    rblock.size = sc_size;
    rgsl.block = &rblock;
    rgsl.size = sc_size;
    rgsl.stride = 1;
    rgsl.owner = 0;

    lblock.data = lgsl.data = l;
    rblock.data = rgsl.data = r;

    gsl_dotproduct<typename ComplexType::value_type,conj_left,conj_right>::doit(&lgsl,&rgsl,&result);
    double(&v3_a)[2] = reinterpret_cast<double(&)[2]>(v3);
    v3_a[0] = GSL_REAL(result);
    v3_a[1] = GSL_IMAG(result);
 
#else
    for(int i = 0; i < sc_size; i ++){
      v3 += Mconj<ComplexType,conj_left,conj_right>::doit(l+i,r+i);
    }
#endif

    return v3;
  }

};

template<int smatidx,typename ComplexType, bool conj_left, bool conj_right>
struct SpinColorContractSelect{};

template<typename ComplexType, bool conj_left, bool conj_right>
struct SpinColorContractSelect<15,ComplexType,conj_left,conj_right>{
  inline static ComplexType doit(const ComplexType *const l, const ComplexType *const r){
    return OptimizedSpinColorContract<ComplexType,conj_left,conj_right>::g5(l,r);
  }
};
template<typename ComplexType, bool conj_left, bool conj_right>
struct SpinColorContractSelect<0,ComplexType,conj_left,conj_right>{
  inline static ComplexType doit(const ComplexType *const l, const ComplexType *const r){
    return OptimizedSpinColorContract<ComplexType,conj_left,conj_right>::unit(l,r);
  }
};


//For Grid SIMD vectorized complex numbers
#ifdef USE_GRID

template<typename vComplexType, bool conj_left, bool conj_right>
class GridVectorizedSpinColorContract{
public:
  accelerator_inline static typename SIMT<vComplexType>::value_type g5(const vComplexType *const l, const vComplexType *const r){
    const static int sc_size =12;
    const static int half_sc = 6;

    typename SIMT<vComplexType>::value_type v3; Grid::zeroit(v3);

    for(int i = half_sc; i < sc_size; i++){ 
      v3 -= MconjGrid<vComplexType,conj_left,conj_right>::doit(l+i,r+i);
    }
    for(int i = 0; i < half_sc; i ++){ 
      v3 += MconjGrid<vComplexType,conj_left,conj_right>::doit(l+i,r+i);
    }
    return v3;
  }
  accelerator_inline static typename SIMT<vComplexType>::value_type unit(const vComplexType *const l, const vComplexType *const r){
    const static int sc_size =12;
    typename SIMT<vComplexType>::value_type v3; Grid::zeroit(v3);

    for(int i = 0; i < sc_size; i ++){
      v3 += MconjGrid<vComplexType,conj_left,conj_right>::doit(l+i,r+i);
    }
    return v3;
  }
  //numbering g<0,1,2> follows wilson_matrix.h convention
  //gamma_x
  accelerator_inline static typename SIMT<vComplexType>::value_type g0(const vComplexType *const l, const vComplexType *const r){
    //const static int sc_size =12;
    const static int half_sc = 6;
    const static int c = 3;

    typename SIMT<vComplexType>::value_type v3; Grid::zeroit(v3);

    for(int j = 0; j < c; j++){ 
      v3 += MconjGrid<vComplexType,conj_left,conj_right>::doit(l+j,r+half_sc+c+j);
      v3 -= MconjGrid<vComplexType,conj_left,conj_right>::doit(l+half_sc+c+j,r+j);
    }
    for(int j = c; j < half_sc; j ++){ 
      v3 += MconjGrid<vComplexType,conj_left,conj_right>::doit(l+j,r+c+j);
      v3 -= MconjGrid<vComplexType,conj_left,conj_right>::doit(l+c+j,r+j);
    }
    //mult by i
    return timesI(v3);
  }
  //gamma_y
  accelerator_inline static vComplexType g1(const vComplexType *const l, const vComplexType *const r){
    //const static int sc_size =12;
    const static int half_sc = 6;
    const static int c = 3;

    typename SIMT<vComplexType>::value_type v3; Grid::zeroit(v3);

    for(int j = 0; j < c; j++){ 
      v3 -= MconjGrid<vComplexType,conj_left,conj_right>::doit(l+j,r+half_sc+c+j);
      v3 -= MconjGrid<vComplexType,conj_left,conj_right>::doit(l+half_sc+c+j,r+j);
    }
    for(int j = c; j < half_sc; j ++){ 
      v3 += MconjGrid<vComplexType,conj_left,conj_right>::doit(l+j,r+c+j);
      v3 += MconjGrid<vComplexType,conj_left,conj_right>::doit(l+c+j,r+j);
    }
    return v3;
  }
  //gamma_z
  accelerator_inline static typename SIMT<vComplexType>::value_type g2(const vComplexType *const l, const vComplexType *const r){
    //const static int sc_size =12;
    const static int half_sc = 6;
    const static int c = 3;

    typename SIMT<vComplexType>::value_type v3; Grid::zeroit(v3);

    for(int j = 0; j < c; j++){ 
      v3 += MconjGrid<vComplexType,conj_left,conj_right>::doit(l+j,r+half_sc+j);
      v3 -= MconjGrid<vComplexType,conj_left,conj_right>::doit(l+half_sc+j,r+j);
    }
    for(int j = c; j < half_sc; j ++){ 
      v3 -= MconjGrid<vComplexType,conj_left,conj_right>::doit(l+j,r+half_sc+j);
      v3 += MconjGrid<vComplexType,conj_left,conj_right>::doit(l+half_sc+j,r+j);
    }
    //mult by i
    return timesI(v3);
  }

};


//Hand-optimized AVX512 kernel
# ifdef AVX512

template<>
inline static Grid::vComplexD GridVectorizedSpinColorContract<Grid::vComplexD,true,false>::g5(const Grid::vComplexD *const l, const Grid::vComplexD *const r){
    Grid::vComplexD v3;
    v3.v = g5d_conjl_r_asm_avx512( (__m512d const*)l, (__m512d const*)r );
    return v3;
}
template<>
inline static Grid::vComplexD GridVectorizedSpinColorContract<Grid::vComplexD,true,false>::unit(const Grid::vComplexD *const l, const Grid::vComplexD *const r){
    Grid::vComplexD v3;
    v3.v = gunitd_conjl_r_asm_avx512( (__m512d const*)l, (__m512d const*)r );
    return v3;
}

# endif //AVX512

template<int smatidx,typename vComplexType, bool conj_left, bool conj_right>
struct GridVectorizedSpinColorContractSelect{};

template<typename vComplexType, bool conj_left, bool conj_right>
struct GridVectorizedSpinColorContractSelect<15,vComplexType,conj_left,conj_right>{
  accelerator_inline static typename SIMT<vComplexType>::value_type doit(const vComplexType *const l, const vComplexType *const r){
    return GridVectorizedSpinColorContract<vComplexType,conj_left,conj_right>::g5(l,r);
  }
};
template<typename vComplexType, bool conj_left, bool conj_right>
struct GridVectorizedSpinColorContractSelect<0,vComplexType,conj_left,conj_right>{
  accelerator_inline static typename SIMT<vComplexType>::value_type doit(const vComplexType *const l, const vComplexType *const r){
    return GridVectorizedSpinColorContract<vComplexType,conj_left,conj_right>::unit(l,r);
  }
};
template<typename vComplexType, bool conj_left, bool conj_right>
struct GridVectorizedSpinColorContractSelect<1,vComplexType,conj_left,conj_right>{
  accelerator_inline static typename SIMT<vComplexType>::value_type doit(const vComplexType *const l, const vComplexType *const r){
    return GridVectorizedSpinColorContract<vComplexType,conj_left,conj_right>::g0(l,r);
  }
};
template<typename vComplexType, bool conj_left, bool conj_right>
struct GridVectorizedSpinColorContractSelect<2,vComplexType,conj_left,conj_right>{
  accelerator_inline static typename SIMT<vComplexType>::value_type doit(const vComplexType *const l, const vComplexType *const r){
    return GridVectorizedSpinColorContract<vComplexType,conj_left,conj_right>::g1(l,r);
  }
};
template<typename vComplexType, bool conj_left, bool conj_right>
struct GridVectorizedSpinColorContractSelect<4,vComplexType,conj_left,conj_right>{
  accelerator_inline static typename SIMT<vComplexType>::value_type doit(const vComplexType *const l, const vComplexType *const r){
    return GridVectorizedSpinColorContract<vComplexType,conj_left,conj_right>::g2(l,r);
  }
};
#endif //USE_GRID


template<int smatidx,typename ComplexType, bool conj_left, bool conj_right, typename ComplexClass>
struct GeneralSpinColorContractSelect{};

template<typename ComplexType, bool conj_left, bool conj_right>
struct GeneralSpinColorContractSelect<15,ComplexType,conj_left,conj_right,complex_double_or_float_mark>{
  inline static ComplexType doit(const ComplexType *const l, const ComplexType *const r){ return OptimizedSpinColorContract<ComplexType,conj_left,conj_right>::g5(l,r); }    
};
template<typename ComplexType, bool conj_left, bool conj_right>
struct GeneralSpinColorContractSelect<0,ComplexType,conj_left,conj_right,complex_double_or_float_mark>{
  inline static ComplexType doit(const ComplexType *const l, const ComplexType *const r){ return OptimizedSpinColorContract<ComplexType,conj_left,conj_right>::unit(l,r); }
};

#ifdef USE_GRID
template<typename ComplexType, bool conj_left, bool conj_right>
struct GeneralSpinColorContractSelect<15,ComplexType,conj_left,conj_right,grid_vector_complex_mark>{
  inline static ComplexType doit(const ComplexType *const l, const ComplexType *const r){ return GridVectorizedSpinColorContract<ComplexType,conj_left,conj_right>::g5(l,r); }
};
template<typename ComplexType, bool conj_left, bool conj_right>
struct GeneralSpinColorContractSelect<0,ComplexType,conj_left,conj_right,grid_vector_complex_mark>{
  inline static ComplexType doit(const ComplexType *const l, const ComplexType *const r){ return GridVectorizedSpinColorContract<ComplexType,conj_left,conj_right>::unit(l,r); }
};
template<typename ComplexType, bool conj_left, bool conj_right>
struct GeneralSpinColorContractSelect<1,ComplexType,conj_left,conj_right,grid_vector_complex_mark>{
  inline static ComplexType doit(const ComplexType *const l, const ComplexType *const r){ return GridVectorizedSpinColorContract<ComplexType,conj_left,conj_right>::g0(l,r); }
};
template<typename ComplexType, bool conj_left, bool conj_right>
struct GeneralSpinColorContractSelect<2,ComplexType,conj_left,conj_right,grid_vector_complex_mark>{
  inline static ComplexType doit(const ComplexType *const l, const ComplexType *const r){ return GridVectorizedSpinColorContract<ComplexType,conj_left,conj_right>::g1(l,r); }
};
template<typename ComplexType, bool conj_left, bool conj_right>
struct GeneralSpinColorContractSelect<4,ComplexType,conj_left,conj_right,grid_vector_complex_mark>{
  inline static ComplexType doit(const ComplexType *const l, const ComplexType *const r){ return GridVectorizedSpinColorContract<ComplexType,conj_left,conj_right>::g2(l,r); }
};
#endif




CPS_END_NAMESPACE

#endif
