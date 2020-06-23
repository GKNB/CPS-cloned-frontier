#ifndef _SPIN_COLOR_MATRICES_SIMT_H__
#define _SPIN_COLOR_MATRICES_SIMT_H__

#ifdef USE_GRID
#include<Grid.h>
#endif
#include "spin_color_matrices.h"
#include <alg/a2a/utils/SIMT.h>

CPS_START_NAMESPACE 

//SIMT implementations for CPSsquareMatrix and derivatives

#ifdef GRID_NVCC
#ifdef __CUDA_ARCH__
//The above is necessary such that the functions revert to normal for host code

template<typename VectorMatrixType, typename std::enable_if<VectorMatrixType::isDerivedFromCPSsquareMatrix,int>::type = 0>
struct SIMT_VectorMatrix{
  typedef VectorMatrixType vector_type;
  typedef typename VectorMatrixType::scalar_type vector_complex_type; //the scalar_type of a matrix is the underlying complex type; it can be SIMD vectorized!
  typedef typename vector_complex_type::scalar_type scalar_complex_type; //this is a non-SIMD numerical type
  typedef typename VectorMatrixType::RebaseScalarType<scalar_complex_type>::type value_type; //non-simd Matrix type
  static const int N = sizeof(vector_type)/sizeof(vector_complex_type);

  static accelerator_inline value_type read(const vector_type & __restrict__ vec,int lane=Grid::SIMTlane(vector_complex_type::Nsimd()) ){
    value_type out;
    vector_complex_type const* inp = (vector_complex_type const*)&vec;
    scalar_complex_type *outp = (scalar_complex_type*)&out;
    for(int i=0;i<N;i++)
      *outp++ = SIMT<vector_complex_type>::read(*inp++);
    return out;
  }

  static accelerator_inline void write(vector_type & __restrict__ vec,const value_type & __restrict__ extracted,int lane=Grid::SIMTlane(vector_complex_type::Nsimd()) ){
    vector_complex_type* outp = (vector_complex_type*)&vec;
    scalar_complex_type const* inp = (scalar_complex_type const*)&extracted;
    for(int i=0;i<N;i++)
      SIMT<vector_complex_type>::write(*outp++, *inp++);
  }
};

template<typename T, int N> 
struct SIMT< CPSsquareMatrix<T,N> >: public SIMT_VectorMatrix< CPSsquareMatrix<T,N> >{ 
  typedef typename SIMT_VectorMatrix< CPSsquareMatrix<T,N> >::vector_type vector_type;
  typedef typename SIMT_VectorMatrix< CPSsquareMatrix<T,N> >::value_type value_type;
};

template<typename Z> 
struct SIMT< CPSspinColorFlavorMatrix<Z> >: public SIMT_VectorMatrix< CPSspinColorFlavorMatrix<Z>   >{ 
  typedef typename SIMT_VectorMatrix< CPSspinColorFlavorMatrix<Z> >::vector_type vector_type;
  typedef typename SIMT_VectorMatrix< CPSspinColorFlavorMatrix<Z> >::value_type value_type;
};

template<typename Z> 
struct SIMT< CPSspinMatrix<Z> >: public SIMT_VectorMatrix< CPSspinMatrix<Z>   >{ 
  typedef typename SIMT_VectorMatrix< CPSspinMatrix<Z> >::vector_type vector_type;
  typedef typename SIMT_VectorMatrix< CPSspinMatrix<Z> >::value_type value_type;
};

template<typename Z> 
struct SIMT< CPScolorMatrix<Z> >: public SIMT_VectorMatrix< CPScolorMatrix<Z>   >{ 
  typedef typename SIMT_VectorMatrix< CPScolorMatrix<Z> >::vector_type vector_type;
  typedef typename SIMT_VectorMatrix< CPScolorMatrix<Z> >::value_type value_type;
};

template<typename Z> 
struct SIMT< CPSflavorMatrix<Z> >: public SIMT_VectorMatrix< CPSflavorMatrix<Z>   >{ 
  typedef typename SIMT_VectorMatrix< CPSflavorMatrix<Z> >::vector_type vector_type;
  typedef typename SIMT_VectorMatrix< CPSflavorMatrix<Z> >::value_type value_type;
};


#endif
#endif

CPS_END_NAMESPACE
#endif
