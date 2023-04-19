#ifndef _CPS_MATRIX_FIELD_H__
#define _CPS_MATRIX_FIELD_H__

#include "CPSfield.h"
#include <alg/a2a/lattice/spin_color_matrices.h>

//CPSfields of SIMD-vectorized matrices and associated functionality

CPS_START_NAMESPACE 

//Definition of CPSmatrixField
template<typename VectorMatrixType>
using CPSmatrixField = CPSfield<VectorMatrixType,1, FourDSIMDPolicy<OneFlavorPolicy>, UVMallocPolicy>;

template<typename VectorMatrixType>
double CPSmatrixFieldNorm2(const CPSmatrixField<VectorMatrixType> &f);

//For testRandom
template<typename T>
class _testRandom<T, typename std::enable_if<isCPSsquareMatrix<T>::value, void>::type>{
public:
  static void rand(T* f, size_t fsize, const Float hi, const Float lo);
};

#include "implementation/CPSmatrixField_meta.tcc"
#include "implementation/CPSmatrixField_functors.tcc"

template<typename VectorMatrixType>
inline auto Trace(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trV<VectorMatrixType>()) );

template<int Index, typename VectorMatrixType>
inline auto TraceIndex(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trIndexV<Index,VectorMatrixType>()) );

template<int Index, typename VectorMatrixType>
inline void TraceIndex(CPSmatrixField<typename _trIndexV<Index,VectorMatrixType>::OutputType > &out,  const CPSmatrixField<VectorMatrixType> &in);

template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > ColorTrace(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in);

template<typename ComplexType>
inline void ColorTrace(CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in);

//Trace over two indices of a nested matrix. Requires  Index1 < Index2
template<int Index1, int Index2, typename VectorMatrixType>
inline auto TraceTwoIndices(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trTwoIndicesV<Index1,Index2,VectorMatrixType>()) );

template<int Index1, int Index2, typename VectorMatrixType>
inline void TraceTwoIndices(CPSmatrixField<typename _trTwoIndicesV<Index1,Index2,VectorMatrixType>::OutputType > &out, const CPSmatrixField<VectorMatrixType> &in);

template<typename ComplexType>
inline CPSmatrixField<CPScolorMatrix<ComplexType> > SpinFlavorTrace(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in);

template<typename ComplexType>
inline void SpinFlavorTrace(CPSmatrixField<CPScolorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in);

template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> Transpose(const CPSmatrixField<VectorMatrixType> &a);

template<int Index, typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> TransposeOnIndex(const CPSmatrixField<VectorMatrixType> &in);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > TransposeColor(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in);

template<int Index, typename VectorMatrixType>
inline void TransposeOnIndex(CPSmatrixField<VectorMatrixType> &out, const CPSmatrixField<VectorMatrixType> &in);

template<typename ComplexType>
inline void TransposeColor(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in);

//Complex conjugate
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> cconj(const CPSmatrixField<VectorMatrixType> &a);

//Left multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > gl_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > gl_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline void gl_r(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

//Right multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > gr_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > gr_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline void gr_r(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

//Left multiplication by gamma(dir)gamma(5)
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > glAx_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > glAx_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline void glAx_r(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

//Right multiplication by gamma(dir)gamma(5)
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > grAx_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > grAx_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline void grAx_r(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> operator*(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b);

template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> operator+(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b);

template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> operator-(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b);

//Trace(a*b) = \sum_{ij} a_{ij}b_{ji}
template<typename VectorMatrixType>
inline CPSmatrixField<typename _traceProdV<VectorMatrixType>::OutputType> Trace(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b);

//in -> in
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & unit(CPSmatrixField<VectorMatrixType> &in);

//in -> i * in
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & timesI(CPSmatrixField<VectorMatrixType> &in);

//in -> -i * in
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & timesMinusI(CPSmatrixField<VectorMatrixType> &in);

//in -> -in
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & timesMinusOne(CPSmatrixField<VectorMatrixType> &in);

//Left multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & gl(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & gl(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

//Right multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & gr(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & gr(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

//Left multiplication by gamma^dir gamma^5
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & glAx(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & glAx(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);

//Right multiplication by gamma^dir gamma^5
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & grAx(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & grAx(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir);


//Left multiplication by flavor matrix
template<typename ComplexType>
inline CPSmatrixField<CPSflavorMatrix<ComplexType> > & pl(CPSmatrixField<CPSflavorMatrix<ComplexType> > &in, const FlavorMatrixType type);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & pl(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const FlavorMatrixType type);

//Right multiplication by flavor matrix
template<typename ComplexType>
inline CPSmatrixField<CPSflavorMatrix<ComplexType> > & pr(CPSmatrixField<CPSflavorMatrix<ComplexType> > &in, const FlavorMatrixType type);

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & pr(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const FlavorMatrixType type);

//Tr(a * b)
template<typename VectorMatrixType>			
CPSmatrixField<typename VectorMatrixType::scalar_type> Trace(const CPSmatrixField<VectorMatrixType> &a, const VectorMatrixType &b);

//Sum the matrix field over sides on this node
//Slow implementation
template<typename VectorMatrixType>			
VectorMatrixType localNodeSumSimple(const CPSmatrixField<VectorMatrixType> &a);
//Fast implementation
template<typename VectorMatrixType>			
VectorMatrixType localNodeSum(const CPSmatrixField<VectorMatrixType> &a);

//Simultaneous global and SIMD reduction (if applicable)
template<typename VectorMatrixType>			
inline auto globalSumReduce(const CPSmatrixField<VectorMatrixType> &a) ->decltype(Reduce(localNodeSum(a)));

//Perform the local-node 3d slice sum
//Output is an array of size GJP.TnodeSites()  (i.e. the local time coordinate)
//Slow implementation
template<typename VectorMatrixType>			
ManagedVector<VectorMatrixType>  localNodeSpatialSumSimple(const CPSmatrixField<VectorMatrixType> &a);

//Perform the local-node 3d slice sum
//Output is an array of size GJP.TnodeSites()  (i.e. the local time coordinate)
//Fast implementation
template<typename VectorMatrixType>			
ManagedVector<VectorMatrixType> localNodeSpatialSum(const CPSmatrixField<VectorMatrixType> &a);

#include "implementation/CPSmatrixField_func_templates.tcc"
#include "implementation/CPSmatrixField_impl.tcc"

CPS_END_NAMESPACE

#endif
