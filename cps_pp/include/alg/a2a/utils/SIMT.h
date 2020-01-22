#ifndef A2A_SIMT_H_
#define A2A_SIMT_H_

#include<config.h>

#ifdef USE_GRID
#include<Grid/Grid.h>
#else
#define accelerator_inline inline
#endif

CPS_START_NAMESPACE

//In the SIMT model we cast to a SIMTtype that allows access to the individual SIMT lanes. If we are not using Grid SIMD this should be transparent
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


#ifdef GRID_NVCC
#ifdef __CUDA_ARCH__
//The above is necessary such that the functions revert to normal for host code

template<typename _vector_type>
struct SIMT_GridVector{
  typedef _vector_type vector_type;
  typedef typename vector_type::scalar_type value_type;
  typedef Grid::iScalar<vector_type> accessor_type;

  static accelerator_inline value_type read(const vector_type & __restrict__ vec,int lane=Grid::SIMTlane(vector_type::Nsimd()) ){
    const accessor_type &v = (const accessor_type &)vec;
    typename accessor_type::scalar_object s = coalescedRead(v, lane);
    return s._internal;
  }

  static accelerator_inline void write(vector_type & __restrict__ vec,const value_type & __restrict__ extracted,int lane=Grid::SIMTlane(vector_type::Nsimd()) ){
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
