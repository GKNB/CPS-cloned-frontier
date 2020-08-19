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


///////////////////////////// ASSIGMENT ////////////////////////////////////

template<typename U> 
accelerator_inline typename my_enable_if<is_grid_vector_complex<U>::value, void>::type equals(U &out, const U &a, const int lane){
  SIMT<U>::write(out,  
		 SIMT<U>::read(a,lane), 
		 lane);
}

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type equals(U &out, const U &a, const int lane){
  for(int i=0;i<U::Size;i++)
    for(int k=0;k<U::Size;k++)
      equals(out(i,k),  a(i,k), lane);
}

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


//////////////////////////////////// ADD/SUBTRACT ///////////////////////////////////////////////////

template<typename U> 
accelerator_inline typename my_enable_if<is_grid_vector_complex<U>::value, void>::type add(U &out, const U &a, const U &b, const int lane){
  SIMT<U>::write(out,  
		 SIMT<U>::read(a,lane) + SIMT<U>::read(b,lane), 
		 lane);
}

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type add(U &out, const U &a, const U &b, const int lane){
  for(int i=0;i<U::Size;i++)
    for(int k=0;k<U::Size;k++)
      add(out(i,k),  a(i,k), b(i,k), lane);
}

template<typename U> 
accelerator_inline typename my_enable_if<is_grid_vector_complex<U>::value, void>::type sub(U &out, const U &a, const U &b, const int lane){
  SIMT<U>::write(out,  
		 SIMT<U>::read(a,lane) - SIMT<U>::read(b,lane), 
		 lane);
}

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type sub(U &out, const U &a, const U &b, const int lane){
  for(int i=0;i<U::Size;i++)
    for(int k=0;k<U::Size;k++)
      sub(out(i,k),  a(i,k), b(i,k), lane);
}


template<typename U> 
accelerator_inline typename my_enable_if<is_grid_vector_complex<U>::value, void>::type plus_equals(U &out, const U &a, const int lane){
  SIMT<U>::write(out,  
		 SIMT<U>::read(out,lane) + SIMT<U>::read(a,lane), 
		 lane);
}

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type plus_equals(U &out, const U &a, const int lane){
  for(int i=0;i<U::Size;i++)
    for(int k=0;k<U::Size;k++)
      plus_equals(out(i,k),  a(i,k), lane);
}


//////////////////////////////// ZERO / UNIT MATRIX ////////////////////////////////////////////////

template<typename T, typename TypeClass>
struct _CPSsetZeroOneSIMT{};

template<typename T>
struct _CPSsetZeroOneSIMT<T,no_mark>{
  accelerator_inline static void setone(T &what, int lane){
    SIMT<T>::write(what, typename T::scalar_type(1.), lane);
  }
  accelerator_inline static void setzero(T &what, int lane){
    SIMT<T>::write(what, typename T::scalar_type(0.), lane);
  }
};
template<typename T>
struct _CPSsetZeroOneSIMT<T, cps_square_matrix_mark>{
  accelerator_inline static void setone(T &what, int lane){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	i == j ? 
	  _CPSsetZeroOneSIMT<typename T::value_type, typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::setone(what(i,j),lane) :
	  _CPSsetZeroOneSIMT<typename T::value_type, typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::setzero(what(i,j),lane);
  }
  accelerator_inline static void setzero(T &what, int lane){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_CPSsetZeroOneSIMT<typename T::value_type, typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::setzero(what(i,j),lane);
  }
};

//Note these are self-ops
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type unit(U &out, const int lane){
  _CPSsetZeroOneSIMT<U, cps_square_matrix_mark>::setone(out, lane);
}
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type zeroit(U &out, const int lane){
  _CPSsetZeroOneSIMT<U, cps_square_matrix_mark>::setzero(out, lane);
}

////////////////////// MULTIPLY BY -1, I, -I

template<typename T, typename Tclass>
struct _timespmI_SIMT{};


template<typename T>
struct _timespmI_SIMT<T, no_mark>{
  accelerator_inline static void timesMinusOne(T &out, const T &in, int lane){
    SIMT<T>::write(out, -SIMT<T>::read(in, lane), lane);
  }  
  accelerator_inline static void timesI(T &out, const T &in, int lane){
    auto v = SIMT<T>::read(in, lane);
    v = cps::timesI(v);
    SIMT<T>::write(out, v, lane);
  }
  accelerator_inline static void timesMinusI(T &out, const T &in, int lane){
    auto v = SIMT<T>::read(in, lane);
    v = cps::timesMinusI(v);
    SIMT<T>::write(out, v, lane);
  }
};
template<typename T>
struct _timespmI_SIMT<T,cps_square_matrix_mark>{
  accelerator_inline static void timesMinusOne(T &out, const T &in, int lane){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_timespmI_SIMT<typename T::value_type, typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::timesMinusOne(out(i,j), in(i,j), lane);
  }    
  accelerator_inline static void timesI(T &out, const T &in, int lane){    
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_timespmI_SIMT<typename T::value_type, typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::timesI(out(i,j), in(i,j), lane);
  }
  accelerator_inline static void timesMinusI(T &out, const T &in, int lane){    
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_timespmI_SIMT<typename T::value_type, typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::timesMinusI(out(i,j), in(i,j), lane);
  }
};


template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type timesI(U &out, const U &in, const int lane){
  _timespmI_SIMT<U, cps_square_matrix_mark>::timesI(out, in, lane);
}
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type timesMinusI(U &out, const U &in, const int lane){
  _timespmI_SIMT<U, cps_square_matrix_mark>::timesMinusI(out, in, lane);
}
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, void>::type timesMinusOne(U &out, const U &in, const int lane){
  _timespmI_SIMT<U, cps_square_matrix_mark>::timesMinusOne(out, in, lane);
}



////////////////////// PARTIAL TRACE ////////////////////////////////////////////////

template<typename U,typename T, int RemoveDepth>
struct _PartialTraceImplSIMT{
  accelerator_inline static void doit(U &into, const T&from, int lane){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_PartialTraceImplSIMT<typename U::value_type, typename T::value_type, RemoveDepth-1>::doit(into(i,j), from(i,j), lane);
  }
};
template<typename U, typename T>
struct _PartialTraceImplSIMT<U,T,0>{
  accelerator_inline static void doit(U &into, const T&from, int lane){    
    equals(into, from(0,0), lane);
    for(int i=1;i<T::Size;i++)
      plus_equals(into, from(i,i), lane);
  }
};


template<int RemoveDepth, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
accelerator_inline void TraceIndex(typename _PartialTraceFindReducedType<VectorMatrixType, RemoveDepth>::type &into,
				   const VectorMatrixType &from, int lane){
  _PartialTraceImplSIMT<typename _PartialTraceFindReducedType<VectorMatrixType, RemoveDepth>::type,
			VectorMatrixType, RemoveDepth>::doit(into, from, lane);
}


///////////////////////////////////// PARTIAL DOUBLE TRACE /////////////////////////////////////////////
template<typename U,typename T, int RemoveDepth>
struct _PartialDoubleTraceImplSIMT_2{
  accelerator_inline static void doit(U &into, const T&from, int lane){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_PartialDoubleTraceImplSIMT_2<typename U::value_type, typename T::value_type, RemoveDepth-1>::doit(into(i,j), from(i,j), lane);
  }
};
template<typename U, typename T>
struct _PartialDoubleTraceImplSIMT_2<U,T,0>{
  accelerator_inline static void doit(U &into, const T&from, int lane){    
    for(int i=0;i<T::Size;i++)
      plus_equals(into, from(i,i), lane);
  }
};


template<typename U,typename T, int RemoveDepth1,int RemoveDepth2>
struct _PartialDoubleTraceImplSIMT{
  accelerator_inline static void doit(U &into, const T&from, int lane){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_PartialDoubleTraceImplSIMT<typename U::value_type, typename T::value_type, RemoveDepth1-1,RemoveDepth2-1>::doit(into(i,j), from(i,j), lane);
  }
};
template<typename U, typename T, int RemoveDepth2>
struct _PartialDoubleTraceImplSIMT<U,T,0,RemoveDepth2>{
  accelerator_inline static void doit(U &into, const T&from, int lane){
    _PartialTraceImplSIMT<U,typename T::value_type, RemoveDepth2-1>::doit(into, from(0,0), lane); //into = tr<Depth>( from(0,0) )
    for(int i=1;i<T::Size;i++)
      _PartialDoubleTraceImplSIMT_2<U,typename T::value_type, RemoveDepth2-1>::doit(into, from(i,i), lane); //into += tr<Depth>( from(i,i) )
  }
};

template<int RemoveDepth1, int RemoveDepth2, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
accelerator_inline void TraceTwoIndices(typename _PartialDoubleTraceFindReducedType<VectorMatrixType,RemoveDepth1,RemoveDepth2>::type &into,
					const VectorMatrixType &from, int lane){
  _PartialDoubleTraceImplSIMT<typename _PartialDoubleTraceFindReducedType<VectorMatrixType,RemoveDepth1,RemoveDepth2>::type,
			      VectorMatrixType, RemoveDepth1, RemoveDepth2>::doit(into, from, lane);
}


//////////////////////////////////////// FULL TRANSPOSE ///////////////////////

template<typename T, typename TypeClass>
struct _RecursiveTransposeImplSIMT{};

template<typename T>
struct _RecursiveTransposeImplSIMT<T, cps_square_matrix_mark>{
  accelerator_inline static void doit(T &into, const T &what, int lane){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_RecursiveTransposeImplSIMT<typename T::value_type, typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::doit(into(i,j), what(j,i),lane);
  } 
};
template<typename scalar_type>
struct _RecursiveTransposeImplSIMT<scalar_type, no_mark>{
  accelerator_inline static void doit(scalar_type &into, const scalar_type &what, int lane){
    SIMT<scalar_type>::write(into, SIMT<scalar_type>::read(what,lane), lane);
  }
};

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
accelerator_inline void Transpose(VectorMatrixType &into, const VectorMatrixType &from, int lane){
  _RecursiveTransposeImplSIMT<VectorMatrixType, cps_square_matrix_mark>::doit(into, from, lane);
}

////////////////////////// TRANSPOSE ON SINGLE INDEX ////////////////////

//Transpose the matrix at depth TransposeDepth
template<typename T, int TransposeDepth>
struct _IndexTransposeImplSIMT{
  accelerator_inline static void doit(T &into, const T&from, int lane){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_IndexTransposeImplSIMT<typename T::value_type, TransposeDepth-1>::doit(into(i,j), from(i,j), lane);
  }
};
template<typename T>
struct _IndexTransposeImplSIMT<T,0>{
  accelerator_inline static void doit(T &into, const T&from, int lane){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	equals(into(i,j), from(j,i), lane);
  }
};

template<int TransposeDepth, typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
accelerator_inline void TransposeOnIndex(VectorMatrixType &out, const VectorMatrixType &in, int lane){
  _IndexTransposeImplSIMT<VectorMatrixType, TransposeDepth>::doit(out,in,lane);
}


CPS_END_NAMESPACE
#endif
