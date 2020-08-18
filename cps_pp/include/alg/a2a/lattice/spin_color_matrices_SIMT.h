#ifndef _SPIN_COLOR_MATRICES_SIMT_H__
#define _SPIN_COLOR_MATRICES_SIMT_H__

#ifdef USE_GRID
#include<Grid.h>
#endif
#include "spin_color_matrices.h"
#include <alg/a2a/utils/SIMT.h>

CPS_START_NAMESPACE 

//SIMT implementations for CPSsquareMatrix and derivatives

#ifdef GPU_VEC
#ifdef GRID_SIMT
//The above is necessary such that the functions revert to normal for host code

template<typename VectorMatrixType, typename std::enable_if<VectorMatrixType::isDerivedFromCPSsquareMatrix,int>::type = 0>
struct SIMT_VectorMatrix{
  typedef VectorMatrixType vector_type;
  typedef typename VectorMatrixType::scalar_type vector_complex_type; //the scalar_type of a matrix is the underlying complex type; it can be SIMD vectorized!
  typedef typename vector_complex_type::scalar_type scalar_complex_type; //this is a non-SIMD numerical type
  typedef typename VectorMatrixType::RebaseScalarType<scalar_complex_type>::type value_type; //non-simd Matrix type
  static const int N = sizeof(vector_type)/sizeof(vector_complex_type);

  static accelerator_inline value_type read(const vector_type & __restrict__ vec,int lane=Grid::acceleratorSIMTlane(vector_complex_type::Nsimd()) ){
    value_type out;
    vector_complex_type const* inp = (vector_complex_type const*)&vec;
    scalar_complex_type *outp = (scalar_complex_type*)&out;
    for(int i=0;i<N;i++)
      *outp++ = SIMT<vector_complex_type>::read(*inp++);
    return out;
  }

  static accelerator_inline void write(vector_type & __restrict__ vec,const value_type & __restrict__ extracted,int lane=Grid::acceleratorSIMTlane(vector_complex_type::Nsimd()) ){
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

//For optimal performance in offloaded routines, the naive method in which the non-SIMD matrix is first extracted from the SIMD matrix on each thread and the 
//routine applied to the non-SIMD matrix, appears to perform very poorly due to register spilling to local memory. To counteract this we need versions of the
//matrix operations that act only for a specific SIMD lane


///////////////////////////// TRACE ////////////////////////////////////////////////

template<typename scalar_type, typename T, typename TypeClass>
struct _LaneRecursiveTraceImpl{};

template<typename scalar_type, typename T>
struct _LaneRecursiveTraceImpl<scalar_type, T, cps_square_matrix_mark>{
  accelerator_inline static void doit(scalar_type &into, const T &what, const int lane){
    for(int i=0;i<T::Size;i++)
      _LaneRecursiveTraceImpl<scalar_type, typename T::value_type, typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::doit(into,what(i,i),lane);    
  } 
};
template<typename scalar_type>
struct _LaneRecursiveTraceImpl<scalar_type, scalar_type, no_mark>{  
  accelerator_inline static void doit(scalar_type &into, const scalar_type &what, const int lane){
    typedef SIMT<scalar_type> ACC;
    ACC::write(into,
	       ACC::read(into,lane) + ACC::read(what,lane),
	       lane);
  }
};

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
accelerator_inline void Trace(typename VectorMatrixType::scalar_type &out, const VectorMatrixType &in, const int lane){
  SIMT<typename VectorMatrixType::scalar_type>::write(out,0,lane);
  _LaneRecursiveTraceImpl<typename VectorMatrixType::scalar_type, VectorMatrixType, cps_square_matrix_mark>::doit(out, in, lane);
}

//////////////////////////////////// MATRIX MULT ///////////////////////////////////////////////////

template<typename U> 
accelerator_inline typename my_enable_if<is_grid_vector_complex<U>::value, void>::type madd(U &out, const U &a, const U &b, const int lane){
  SIMT<U>::write(out,  
		 SIMT<U>::read(out,lane) + SIMT<U>::read(a,lane) * SIMT<U>::read(b,lane), 
		 lane);
}


template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type madd(U &out, const U &a, const U &b, const int lane){
  for(int i=0;i<U::Size;i++)
    for(int k=0;k<U::Size;k++){
      madd(out(i,k),  a(i,0), b(0,k), lane);
      for(int j=1;j<U::Size;j++)
	madd(out(i,k),  a(i,j), b(j,k), lane);
    }      
}


template<typename U> 
accelerator_inline typename my_enable_if<is_grid_vector_complex<U>::value, void>::type mult(U &out, const U &a, const U &b, const int lane){
  SIMT<U>::write(out,  
		 SIMT<U>::read(a,lane) * SIMT<U>::read(b,lane), 
		 lane);
}

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type mult(U &out, const U &a, const U &b, const int lane){
  for(int i=0;i<U::Size;i++)
    for(int k=0;k<U::Size;k++){
      mult(out(i,k),  a(i,0), b(0,k), lane);
      for(int j=1;j<U::Size;j++)
	madd(out(i,k),  a(i,j), b(j,k), lane);
    }      
}









CPS_END_NAMESPACE
#endif
