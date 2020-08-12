#ifndef A2A_SIMT_H_
#define A2A_SIMT_H_

#include<config.h>
#include "utils_gpu.h"

CPS_START_NAMESPACE

//In the SIMT model we cast to a SIMTtype that allows access to the individual SIMT lanes. If we are not using Grid SIMD this should be transparent
//In the context of a device function being compiled for device execution, we use the fact that a preprocessor directive is defined only when compiling for a device
//to modify the output type to the underlying scalar type and extract a scalar element from the vectorized data type


//NOTE: Do not use these classes outside of a kernel as it will always return the vector type and not the underlying scalar type

template<typename T> struct SIMT{
  typedef T vector_type;
  typedef T value_type;

  static accelerator_inline value_type read(const vector_type & __restrict__ vec,int lane=0){
    return vec;
  }
  static accelerator_inline void write(vector_type & __restrict__ vec,const value_type & __restrict__ extracted,int lane=0){
    vec = extracted;
  }
};

#define SIMT_INHERIT(BASE)		 \
  typedef BASE::vector_type vector_type; \
  typedef BASE::value_type value_type;


#ifdef GPU_VEC
#ifdef GRID_SIMT
//The above is necessary such that the functions revert to normal for host code

template<typename _vector_type>
struct SIMT_GridVector{
  typedef _vector_type vector_type;
  typedef typename vector_type::scalar_type value_type;
  typedef Grid::iScalar<vector_type> accessor_type;

  static accelerator_inline value_type read(const vector_type & __restrict__ vec,int lane=Grid::acceleratorSIMTlane(vector_type::Nsimd()) ){
    const accessor_type &v = (const accessor_type &)vec;
    typename accessor_type::scalar_object s = coalescedRead(v, lane);
    return s._internal;
  }

  static accelerator_inline void write(vector_type & __restrict__ vec,const value_type & __restrict__ extracted,int lane=Grid::acceleratorSIMTlane(vector_type::Nsimd()) ){
    accessor_type &v = (accessor_type &)vec;
    coalescedWrite(v, extracted, lane);
  }
};

template<> struct SIMT< Grid::vComplexD >: public SIMT_GridVector<Grid::vComplexD>{ SIMT_INHERIT(SIMT_GridVector<Grid::vComplexD>); };
template<> struct SIMT< Grid::vComplexF >: public SIMT_GridVector<Grid::vComplexF>{ SIMT_INHERIT(SIMT_GridVector<Grid::vComplexF>); };

#endif
#endif

CPS_END_NAMESPACE

#endif
