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


////////////////////////////////// GAMMA MULT ////////////////////////////
#define TIMESPLUSONE(a,b) { SIMT<ComplexType>::write(a, b, lane); }
#define TIMESMINUSONE(a,b) { SIMT<ComplexType>::write(a, -b, lane); }
#define TIMESPLUSI(a,b) { SIMT<ComplexType>::write(a, cps::timesI(b), lane); }
#define TIMESMINUSI(a,b) { SIMT<ComplexType>::write(a, cps::timesMinusI(b), lane); }
#define SETZERO(a){ SIMT<ComplexType>::write(a, typename SIMT<ComplexType>::value_type(0), lane); }
#define WR(A,B) SIMT<ComplexType>::write(A,B,lane)
#define RD(A) SIMT<ComplexType>::read(A,lane)


template<typename ComplexType>
struct gl_spinMatrixIterator{
  int i;
  accelerator_inline gl_spinMatrixIterator(): i(0){}
  
  accelerator_inline gl_spinMatrixIterator& operator++(){
    ++i;
    return *this;
  }  
  accelerator_inline ComplexType & elem(CPSspinMatrix<ComplexType> &M, const int s1, const int s2){ return M(s1,s2); }
  accelerator_inline const ComplexType & elem(const CPSspinMatrix<ComplexType> &M, const int s1, const int s2){ return M(s1,s2); }

  accelerator_inline bool end() const{ return i==1; }
};

template<typename ComplexType>
struct gl_spinColorFlavorMatrixIterator{
  int c1,c2,f1,f2;
  int i;
  accelerator_inline gl_spinColorFlavorMatrixIterator(): i(0),c1(0),c2(0),f1(0),f2(0){}
  
  accelerator_inline gl_spinColorFlavorMatrixIterator& operator++(){
    //mapping f2 + 2*(f1 + 2*(c2 + 3*c1))
    ++i;
    int rem = i;
    f2 = rem % 2; rem /= 2;
    f1 = rem % 2; rem /= 2;
    c2 = rem % 3; rem /= 3;
    c1 = rem;

    return *this;
  }  
  accelerator_inline ComplexType & elem(CPSspinColorFlavorMatrix<ComplexType> &M, const int s1, const int s2){ return M(s1,s2)(c1,c2)(f1,f2); }
  accelerator_inline const ComplexType & elem(const CPSspinColorFlavorMatrix<ComplexType> &M, const int s1, const int s2){ return M(s1,s2)(c1,c2)(f1,f2); }

  accelerator_inline bool end() const{ return i==36; }
};



template<typename MatrixIterator, typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void gl(MatrixType &M, int dir, int lane){
  int s2, s1;
  typedef typename MatrixType::scalar_type ComplexType;
  typename SIMT<ComplexType>::value_type tmp[4];
  MatrixIterator it;

  switch(dir){
  case 0:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	for(s1=0;s1<4;s1++) tmp[s1] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESPLUSI(  it.elem(M,0,s2), tmp[3] );
	TIMESPLUSI(  it.elem(M,1,s2), tmp[2] );
	TIMESMINUSI( it.elem(M,2,s2), tmp[1] );
	TIMESMINUSI( it.elem(M,3,s2), tmp[0] );
      }
      ++it;
    }
    break;
  case 1:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	for(s1=0;s1<4;s1++) tmp[s1] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESMINUSONE( it.elem(M,0,s2), tmp[3] );
	TIMESPLUSONE(  it.elem(M,1,s2), tmp[2] );
	TIMESPLUSONE(  it.elem(M,2,s2), tmp[1] );
	TIMESMINUSONE( it.elem(M,3,s2), tmp[0] );
      }
      ++it;
    }    
    break;
  case 2:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	for(s1=0;s1<4;s1++) tmp[s1] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESPLUSI(  it.elem(M,0,s2), tmp[2] );
	TIMESMINUSI( it.elem(M,1,s2), tmp[3] );
	TIMESMINUSI( it.elem(M,2,s2), tmp[0] );
	TIMESPLUSI(  it.elem(M,3,s2), tmp[1] );
      }
      ++it;
    }
    break;
  case 3:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	for(s1=0;s1<4;s1++) tmp[s1] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESPLUSONE( it.elem(M,0,s2), tmp[2] );
	TIMESPLUSONE( it.elem(M,1,s2), tmp[3] );
	TIMESPLUSONE( it.elem(M,2,s2), tmp[0] );
	TIMESPLUSONE( it.elem(M,3,s2), tmp[1] );
      }
      ++it;
    }
    break;
  case -5:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	for(s1=0;s1<4;s1++) tmp[s1] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESPLUSONE(  it.elem(M,0,s2), tmp[0] );
	TIMESPLUSONE(  it.elem(M,1,s2), tmp[1] );
	TIMESMINUSONE( it.elem(M,2,s2), tmp[2] );
	TIMESMINUSONE( it.elem(M,3,s2), tmp[3] );
      }
      ++it;
    }
    break;
  default:
    assert(0);
    break;
  }
}

//Non self-op version 
template<typename MatrixIterator, typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void gl_r(MatrixType &O, const MatrixType &M, int dir, int lane){
  int s2, s1;
  typedef typename MatrixType::scalar_type ComplexType;
  MatrixIterator it;

  switch(dir){
  case 0:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	TIMESPLUSI(  it.elem(O,0,s2), RD(it.elem(M,3,s2)) );
	TIMESPLUSI(  it.elem(O,1,s2), RD(it.elem(M,2,s2)) );
	TIMESMINUSI( it.elem(O,2,s2), RD(it.elem(M,1,s2)) );
	TIMESMINUSI( it.elem(O,3,s2), RD(it.elem(M,0,s2)) );
      }
      ++it;
    }
    break;
  case 1:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	TIMESMINUSONE( it.elem(O,0,s2), RD(it.elem(M,3,s2)) );
	TIMESPLUSONE(  it.elem(O,1,s2), RD(it.elem(M,2,s2)) );
	TIMESPLUSONE(  it.elem(O,2,s2), RD(it.elem(M,1,s2)) );
	TIMESMINUSONE( it.elem(O,3,s2), RD(it.elem(M,0,s2)) );
      }
      ++it;
    }    
    break;
  case 2:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	TIMESPLUSI(  it.elem(O,0,s2), RD(it.elem(M,2,s2)) );
	TIMESMINUSI( it.elem(O,1,s2), RD(it.elem(M,3,s2)) );
	TIMESMINUSI( it.elem(O,2,s2), RD(it.elem(M,0,s2)) );
	TIMESPLUSI(  it.elem(O,3,s2), RD(it.elem(M,1,s2)) );
      }
      ++it;
    }
    break;
  case 3:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	TIMESPLUSONE( it.elem(O,0,s2), RD(it.elem(M,2,s2)) );
	TIMESPLUSONE( it.elem(O,1,s2), RD(it.elem(M,3,s2)) );
	TIMESPLUSONE( it.elem(O,2,s2), RD(it.elem(M,0,s2)) );
	TIMESPLUSONE( it.elem(O,3,s2), RD(it.elem(M,1,s2)) );
      }
      ++it;
    }
    break;
  case -5:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	TIMESPLUSONE(  it.elem(O,0,s2), RD(it.elem(M,0,s2)) );
	TIMESPLUSONE(  it.elem(O,1,s2), RD(it.elem(M,1,s2)) );
	TIMESMINUSONE( it.elem(O,2,s2), RD(it.elem(M,2,s2)) );
	TIMESMINUSONE( it.elem(O,3,s2), RD(it.elem(M,3,s2)) );
      }
      ++it;
    }
    break;
  default:
    assert(0);
    break;
  }
}













//multiply gamma(i)gamma(5) on the left: result = gamma(i)*gamma(5)*fro
template<typename MatrixIterator, typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void glAx(MatrixType &M, int dir, int lane){
  int s2, s1;
  typedef typename MatrixType::scalar_type ComplexType;
  typename SIMT<ComplexType>::value_type tmp[4];
  MatrixIterator it;

  switch(dir){
  case 0:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	for(s1=0;s1<4;s1++) tmp[s1] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESMINUSI( it.elem(M,0,s2), tmp[3] );
	TIMESMINUSI( it.elem(M,1,s2), tmp[2] );
	TIMESMINUSI( it.elem(M,2,s2), tmp[1] );
	TIMESMINUSI( it.elem(M,3,s2), tmp[0] );
      }
      ++it;
    }
    break;
  case 1:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	for(s1=0;s1<4;s1++) tmp[s1] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESPLUSONE(  it.elem(M,0,s2), tmp[3] );
	TIMESMINUSONE( it.elem(M,1,s2), tmp[2] );
	TIMESPLUSONE(  it.elem(M,2,s2), tmp[1] );
	TIMESMINUSONE( it.elem(M,3,s2), tmp[0] );
      }
      ++it;
    }
    break;
  case 2:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	for(s1=0;s1<4;s1++) tmp[s1] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESMINUSI( it.elem(M,0,s2), tmp[2] );
	TIMESPLUSI( it.elem(M,1,s2), tmp[3] );
	TIMESMINUSI( it.elem(M,2,s2), tmp[0] );
	TIMESPLUSI( it.elem(M,3,s2), tmp[1] );
      }
      ++it;
    }
    break;
  case 3:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	for(s1=0;s1<4;s1++) tmp[s1] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESMINUSONE( it.elem(M,0,s2), tmp[2] );
	TIMESMINUSONE( it.elem(M,1,s2), tmp[3] );
	TIMESPLUSONE( it.elem(M,2,s2), tmp[0] );
	TIMESPLUSONE( it.elem(M,3,s2), tmp[1] );
      }
      ++it;
    }
    break;
  default:
    assert(0);
    break;
  }
}


//multiply gamma(i)gamma(5) on the left: result = gamma(i)*gamma(5)*fro   non self-op
template<typename MatrixIterator, typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void glAx_r(MatrixType &O, const MatrixType &M, int dir, int lane){
  int s2, s1;
  typedef typename MatrixType::scalar_type ComplexType;
  MatrixIterator it;

  switch(dir){
  case 0:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	TIMESMINUSI( it.elem(O,0,s2), RD(it.elem(M,3,s2)) );
	TIMESMINUSI( it.elem(O,1,s2), RD(it.elem(M,2,s2)) );
	TIMESMINUSI( it.elem(O,2,s2), RD(it.elem(M,1,s2)) );
	TIMESMINUSI( it.elem(O,3,s2), RD(it.elem(M,0,s2)) );
      }
      ++it;
    }
    break;
  case 1:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	TIMESPLUSONE(  it.elem(O,0,s2), RD(it.elem(M,3,s2)) );
	TIMESMINUSONE( it.elem(O,1,s2), RD(it.elem(M,2,s2)) );
	TIMESPLUSONE(  it.elem(O,2,s2), RD(it.elem(M,1,s2)) );
	TIMESMINUSONE( it.elem(O,3,s2), RD(it.elem(M,0,s2)) );
      }
      ++it;
    }
    break;
  case 2:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	TIMESMINUSI( it.elem(O,0,s2), RD(it.elem(M,2,s2)) );
	TIMESPLUSI( it.elem(O,1,s2), RD(it.elem(M,3,s2)) );
	TIMESMINUSI( it.elem(O,2,s2), RD(it.elem(M,0,s2)) );
	TIMESPLUSI( it.elem(O,3,s2), RD(it.elem(M,1,s2)) );
      }
      ++it;
    }
    break;
  case 3:
    while(!it.end()){
      for(s2=0;s2<4;s2++){
	TIMESMINUSONE( it.elem(O,0,s2), RD(it.elem(M,2,s2)) );
	TIMESMINUSONE( it.elem(O,1,s2), RD(it.elem(M,3,s2)) );
	TIMESPLUSONE( it.elem(O,2,s2), RD(it.elem(M,0,s2)) );
	TIMESPLUSONE( it.elem(O,3,s2), RD(it.elem(M,1,s2)) );
      }
      ++it;
    }
    break;
  default:
    assert(0);
    break;
  }
}




//////////////////// FLAVOR MULT ///////////////////////////

template<typename ComplexType>
struct pl_flavorMatrixIterator{
  int i;
  accelerator_inline pl_flavorMatrixIterator(): i(0){}
  
  accelerator_inline pl_flavorMatrixIterator& operator++(){
    ++i;
    return *this;
  }  
  accelerator_inline ComplexType & elem(CPSflavorMatrix<ComplexType> &M, const int f1, const int f2){ return M(f1,f2); }
  accelerator_inline const ComplexType & elem(const CPSflavorMatrix<ComplexType> &M, const int f1, const int f2){ return M(f1,f2); }

  accelerator_inline bool end() const{ return i==1; }
};

template<typename ComplexType>
struct pl_spinColorFlavorMatrixIterator{
  int s1,s2,c1,c2;
  int i;
  accelerator_inline pl_spinColorFlavorMatrixIterator(): i(0),s1(0),s2(0),c1(0),c2(0){}
  
  accelerator_inline pl_spinColorFlavorMatrixIterator& operator++(){
    //mapping c2 + 3*(c1 + 3*(s2 + 4*s1))
    ++i;
    int rem = i;
    c2 = rem % 3; rem /= 3;
    c1 = rem % 3; rem /= 3;
    s2 = rem % 4; rem /= 4;
    s1 = rem;

    return *this;
  }  
  accelerator_inline ComplexType & elem(CPSspinColorFlavorMatrix<ComplexType> &M, const int f1, const int f2){ return M(s1,s2)(c1,c2)(f1,f2); }
  accelerator_inline const ComplexType & elem(const CPSspinColorFlavorMatrix<ComplexType> &M, const int f1, const int f2){ return M(s1,s2)(c1,c2)(f1,f2); }

  accelerator_inline bool end() const{ return i==144; }
};



template<typename MatrixIterator, typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
accelerator_inline void pl(MatrixType &M, const FlavorMatrixType type, int lane){
  typedef typename MatrixType::scalar_type ComplexType;
  typename SIMT<ComplexType>::value_type tmp1, tmp2;
  MatrixIterator it;

  switch( type ){
  case F0:
    while(!it.end()){
      SETZERO(it.elem(M,1,0));
      SETZERO(it.elem(M,1,1));
      ++it;
    }
    break;
  case F1:
    while(!it.end()){     
      SETZERO(it.elem(M,0,0));
      SETZERO(it.elem(M,0,1));
      ++it;
    }
    break;
  case Fud:
    while(!it.end()){
      tmp1 = RD(it.elem(M,0,0));
      tmp2 = RD(it.elem(M,0,1));
      WR(it.elem(M,0,0), RD(it.elem(M,1,0)) );
      WR(it.elem(M,0,1), RD(it.elem(M,1,1)) );
      WR(it.elem(M,1,0), tmp1);
      WR(it.elem(M,1,1), tmp2);
      ++it;
    }
    break;
  case sigma0:
    break;
  case sigma1:
    while(!it.end()){
      tmp1 = RD(it.elem(M,0,0));
      tmp2 = RD(it.elem(M,0,1));
      WR(it.elem(M,0,0), RD(it.elem(M,1,0)) );
      WR(it.elem(M,0,1), RD(it.elem(M,1,1)) );
      WR(it.elem(M,1,0), tmp1);
      WR(it.elem(M,1,1), tmp2);
      ++it;
    }
    break;      
  case sigma2:
    while(!it.end()){
      tmp1 = RD(it.elem(M,0,0));
      tmp2 = RD(it.elem(M,0,1));
      TIMESMINUSI(it.elem(M,0,0), RD(it.elem(M,1,0)) );
      TIMESMINUSI(it.elem(M,0,1), RD(it.elem(M,1,1)) );
      TIMESPLUSI(it.elem(M,1,0), tmp1);
      TIMESPLUSI(it.elem(M,1,1), tmp2);
      ++it;
    }
    break;
  case sigma3:
    while(!it.end()){
      TIMESMINUSONE(it.elem(M,1,0), RD(it.elem(M,1,0)) );
      TIMESMINUSONE(it.elem(M,1,1), RD(it.elem(M,1,1)) );
      ++it;
    }
    break;
  default:
    assert(0);
    //ERR.General("FlavorMatrixGeneral","pl(const FlavorMatrixGeneralType &type)","Unknown FlavorMatrixGeneralType");
    break;
  }
}


#undef RD
#undef WR
#undef TIMESPLUSONE
#undef TIMESMINUSONE
#undef TIMESPLUSI
#undef TIMESMINUSI
#undef SETZERO


CPS_END_NAMESPACE
#endif
