#ifndef _SPIN_COLOR_MATRICES_SIMT_H__
#define _SPIN_COLOR_MATRICES_SIMT_H__

#ifdef USE_GRID
#include<Grid.h>
#endif
#include "spin_color_matrices.h"
#include <alg/a2a/utils/SIMT.h>
#include <alg/a2a/utils/utils_complex.h>

CPS_START_NAMESPACE 

//SIMT implementations for CPSsquareMatrix and derivatives

//For optimal performance in offloaded routines, the naive method in which the non-SIMD matrix is first extracted from the SIMD matrix on each thread and the 
//routine applied to the non-SIMD matrix, appears to perform very poorly due to register spilling to local memory. To counteract this we need versions of the
//matrix operations that act only for a specific SIMD lane

template<typename U> 
accelerator_inline typename my_enable_if<is_grid_vector_complex<U>::value, void>::type equals(U &out, const U &a, const int lane);

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type equals(U &out, const U &a, const int lane);

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
accelerator_inline void Trace(typename VectorMatrixType::scalar_type &out, const VectorMatrixType &in, const int lane);

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
accelerator_inline void Trace(typename VectorMatrixType::scalar_type &out, const VectorMatrixType &a, const VectorMatrixType &b, const int lane);

//multiply-add
template<typename U> 
accelerator_inline typename my_enable_if<is_grid_vector_complex<U>::value, void>::type madd(U &out, const U &a, const U &b, const int lane);


template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type madd(U &out, const U &a, const U &b, const int lane);

//matrix * matrix
template<typename U> 
accelerator_inline typename my_enable_if<is_grid_vector_complex<U>::value, void>::type mult(U &out, const U &a, const U &b, const int lane);

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type mult(U &out, const U &a, const U &b, const int lane);

//scalar * matrix
template<typename T, typename U> 
accelerator_inline typename my_enable_if<is_complex_double_or_float<T>::value && is_grid_vector_complex<U>::value, void>::type scalar_mult_pre(U &out, const T &a, const U &b, const int lane);

template<typename T, typename U> 
accelerator_inline typename my_enable_if<is_complex_double_or_float<T>::value && isCPSsquareMatrix<U>::value, void>::type scalar_mult_pre(U &out, const T &a, const U &b, const int lane);

//vscalar * matrix
template<typename T, typename U> 
accelerator_inline typename my_enable_if<is_grid_vector_complex<T>::value && is_grid_vector_complex<U>::value, void>::type vscalar_mult_pre(U &out, const T &a, const U &b, const int lane);

template<typename T, typename U> 
accelerator_inline typename my_enable_if<is_grid_vector_complex<T>::value && isCPSsquareMatrix<U>::value, void>::type vscalar_mult_pre(U &out, const T &a, const U &b, const int lane);

//add
template<typename U> 
accelerator_inline typename my_enable_if<is_grid_vector_complex<U>::value, void>::type add(U &out, const U &a, const U &b, const int lane);

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type add(U &out, const U &a, const U &b, const int lane);

template<typename U> 
accelerator_inline typename my_enable_if<is_grid_vector_complex<U>::value, void>::type sub(U &out, const U &a, const U &b, const int lane);

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type sub(U &out, const U &a, const U &b, const int lane);


template<typename U> 
accelerator_inline typename my_enable_if<is_grid_vector_complex<U>::value, void>::type plus_equals(U &out, const U &a, const int lane);

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type plus_equals(U &out, const U &a, const int lane);

//self-multiply by 1
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type unit(U &out, const int lane);

//self set zero
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type zeroit(U &out, const int lane);

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type timesI(U &out, const U &in, const int lane);

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type timesMinusI(U &out, const U &in, const int lane);

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type timesMinusOne(U &out, const U &in, const int lane);

//complex conjugate
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type cconj(U &out, const U &in, const int lane);

//selt complex conjugate
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type cconj(U &inout, const int lane);

//trace over single index
template<int RemoveDepth, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
accelerator_inline void TraceIndex(typename _PartialTraceFindReducedType<VectorMatrixType, RemoveDepth>::type &into,
				   const VectorMatrixType &from, int lane);

//trace over two indices
template<int RemoveDepth1, int RemoveDepth2, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
accelerator_inline void TraceTwoIndices(typename _PartialDoubleTraceFindReducedType<VectorMatrixType,RemoveDepth1,RemoveDepth2>::type &into,
					const VectorMatrixType &from, int lane);

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
accelerator_inline void Transpose(VectorMatrixType &into, const VectorMatrixType &from, int lane);

template<int TransposeDepth, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
accelerator_inline void TransposeOnIndex(VectorMatrixType &out, const VectorMatrixType &in, int lane);




//left-multiply by gamma matrix in-place
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void gl(MatrixType &M, int dir, int lane);

//Non in-place version 
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void gl_r(MatrixType &O, const MatrixType &M, int dir, int lane);

//Version for specific spin column and flavor-color element
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void gl_r(MatrixType &O, const MatrixType &M, int dir, int s2, int ffcc, int lane);

//right-multiply by gamma matrix in-place
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void gr(MatrixType &M, int dir, int lane);

//Non in-place version
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void gr_r(MatrixType &O, const MatrixType &M, int dir, int lane);

//Version for specific spin column and flavor-color element
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void gr_r(MatrixType &O, const MatrixType &M, int dir, int s1, int ffcc, int lane);

//multiply gamma(i)gamma(5) on the left: result = gamma(i)*gamma(5)*fro
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void glAx(MatrixType &M, int dir, int lane);

//multiply gamma(i)gamma(5) on the left: result = gamma(i)*gamma(5)*fro   non self-op
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void glAx_r(MatrixType &O, const MatrixType &M, int dir, int lane);

//Version for specific spin column and flavor-color element
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void glAx_r(MatrixType &O, const MatrixType &M, int dir, int s2, int ffcc, int lane);

//multiply gamma(i)gamma(5) on the left: result = gamma(i)*gamma(5)*fro
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void grAx(MatrixType &M, int dir, int lane);

//multiply gamma(i)gamma(5) on the left: result = gamma(i)*gamma(5)*fro
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void grAx_r(MatrixType &O, const MatrixType &M, int dir, int lane);

//multiply gamma(i)gamma(5) on the left: result = gamma(i)*gamma(5)*fro
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void grAx_r(MatrixType &O, const MatrixType &M, int dir, int s1, int ffcc, int lane);

//left-multiply by flavor matrix in-placew
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void pl(MatrixType &M, const FlavorMatrixType type, int lane);

template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void pr(MatrixType &M, const FlavorMatrixType type, int lane);

#include "implementation/spin_color_matrices_SIMT_impl.tcc"
#include "implementation/spin_color_matrices_SIMT_derived_impl.tcc"

CPS_END_NAMESPACE
#endif
