#ifndef _UTILS_MATRIX_H_
#define _UTILS_MATRIX_H_

#include <util/spincolorflavormatrix.h>
#include <alg/a2a/gsl_wrapper.h>
//Utilities for matrices and vectors

CPS_START_NAMESPACE

//3x3 complex vector multiplication with different precision matrices and vectors
template<typename VecFloat, typename MatFloat>
void colorMatrixMultiplyVector(VecFloat* y, const MatFloat* u, const VecFloat* x){
	*y     =  *u      * *x     - *(u+1)  * *(x+1) + *(u+2)  * *(x+2)
		- *(u+3)  * *(x+3) + *(u+4)  * *(x+4) - *(u+5)  * *(x+5);
	*(y+1) =  *u      * *(x+1) + *(u+1)  * *x     + *(u+2)  * *(x+3)
		+ *(u+3)  * *(x+2) + *(u+4)  * *(x+5) + *(u+5)  * *(x+4);
	*(y+2) =  *(u+6)  * *x     - *(u+7)  * *(x+1) + *(u+8)  * *(x+2)
		- *(u+9)  * *(x+3) + *(u+10) * *(x+4) - *(u+11) * *(x+5);
	*(y+3) =  *(u+6)  * *(x+1) + *(u+7)  * *x     + *(u+8)  * *(x+3)
		+ *(u+9)  * *(x+2) + *(u+10) * *(x+5) + *(u+11) * *(x+4);
	*(y+4) =  *(u+12) * *x     - *(u+13) * *(x+1) + *(u+14) * *(x+2)
		- *(u+15) * *(x+3) + *(u+16) * *(x+4) - *(u+17) * *(x+5);
	*(y+5) =  *(u+12) * *(x+1) + *(u+13) * *x     + *(u+14) * *(x+3)
		+ *(u+15) * *(x+2) + *(u+16) * *(x+5) + *(u+17) * *(x+4);
}
//M^\dagger v

//0 ,1    2 ,3    4 ,5
//6 ,7    8 ,9    10,11
//12,13   14,15   16,17
//->
//0 ,-1   6 ,-7   12,-13
//2 ,-3   8 ,-9   14,-15 
//4 ,-5   10,-11  16,-17

template<typename VecFloat, typename MatFloat>
void colorMatrixDaggerMultiplyVector(VecFloat* y, const MatFloat* u, const VecFloat* x){
	*y     =  *u      * *x     + *(u+1)  * *(x+1) + *(u+6)  * *(x+2)	  
		+ *(u+7)  * *(x+3) + *(u+12)  * *(x+4) + *(u+13)  * *(x+5);
	*(y+1) =  *u      * *(x+1) - *(u+1)  * *x     + *(u+6)  * *(x+3)	  
		- *(u+7)  * *(x+2) + *(u+12)  * *(x+5) - *(u+13)  * *(x+4);	
	*(y+2) =  *(u+2)  * *x     + *(u+3)  * *(x+1) + *(u+8)  * *(x+2)	  
		+ *(u+9)  * *(x+3) + *(u+14) * *(x+4) + *(u+15) * *(x+5);	
	*(y+3) =  *(u+2)  * *(x+1) - *(u+3)  * *x     + *(u+8)  * *(x+3)	  
		- *(u+9)  * *(x+2) + *(u+14) * *(x+5) - *(u+15) * *(x+4);	
	*(y+4) =  *(u+4) * *x     + *(u+5) * *(x+1) + *(u+10) * *(x+2)	  
		+ *(u+11) * *(x+3) + *(u+16) * *(x+4) + *(u+17) * *(x+5);	
	*(y+5) =  *(u+4) * *(x+1) - *(u+5) * *x     + *(u+10) * *(x+3)	  
		- *(u+11) * *(x+2) + *(u+16) * *(x+5) - *(u+17) * *(x+4);
}

//Array *= with cps::Float(=double) input and arbitrary precision output
template<typename FloatOut,typename FloatIn>
void VecTimesEquFloat(FloatOut *out, FloatIn *in, const Float fac, const int len) 
{
#pragma omp parallel for
	for(int i = 0; i < len; i++) out[i] = in[i] * fac;
}


//Invert 3x3 complex matrix. Expect elements accessible as  row*3 + col
//0 1 2
//3 4 5
//6 7 8

//+ - +
//- + -
//+ - +

template<typename Zout, typename Zin>
void z3x3_invert(Zout* out, Zin const* in){
  out[0] = in[4]*in[8]-in[7]*in[5];
  out[1] = -in[3]*in[8]+in[6]*in[5];
  out[2] = in[3]*in[7]-in[6]*in[4];

  out[3] = -in[1]*in[8]+in[7]*in[2];
  out[4] = in[0]*in[8]-in[6]*in[2];
  out[5] = -in[0]*in[7]+in[6]*in[1];

  out[6] = in[1]*in[5]-in[4]*in[2];
  out[7] = -in[0]*in[5]+in[3]*in[2];
  out[8] = in[0]*in[4]-in[3]*in[1];
  
  Zout det = in[0]*out[0] + in[1]*out[1] + in[2]*out[2];

  out[0] /= det; out[1] /= det; out[2] /= det;
  out[3] /= det; out[4] /= det; out[5] /= det;
  out[6] /= det; out[7] /= det; out[8] /= det;
}


//Trace of two SpinColorFlavorMatrix using GSL
inline std::complex<double> GSLtrace(const SpinColorFlavorMatrix& a, const SpinColorFlavorMatrix& b){
  const int scf_size = 24;
  std::complex<double> _a[scf_size][scf_size];
  std::complex<double> _bT[scf_size][scf_size];   //In-place transpose of b so rows are contiguous
  for(int i=0;i<scf_size;i++){
    int rem = i;
    int ci = rem % 3; rem /= 3;
    int si = rem % 4; rem /= 4;
    int fi = rem;
    
    for(int j=0;j<scf_size;j++){
      rem = j;
      int cj = rem % 3; rem /= 3;
      int sj = rem % 4; rem /= 4;
      int fj = rem;
      
      _bT[i][j] = b(sj,cj,fj, si,ci,fi);
      _a[i][j] = a(si,ci,fi, sj,cj,fj);
    }
  }

  double* ad = (double*)&_a[0][0];
  double* bd = (double*)&_bT[0][0];

  gsl_block_complex_struct ablock;
  ablock.size = 24*24;
  ablock.data = ad;

  gsl_vector_complex arow; //single row of a
  arow.block = &ablock;
  arow.owner = 0;
  arow.size = 24;
  arow.stride = 1;
  
  gsl_block_complex_struct bblock;
  bblock.size = 24*24;
  bblock.data = bd;

  gsl_vector_complex bcol; //single col of b
  bcol.block = &bblock;
  bcol.owner = 0;
  bcol.size = 24;
  bcol.stride = 1;

  //gsl_blas_zdotu (const gsl_vector_complex * x, const gsl_vector_complex * y, gsl_complex * dotu)
  //   //  a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0] + ...
  //   //+ a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1] + ....
  //   //...

  std::complex<double> out(0.0);
  gsl_complex tmp;
  for(int i=0;i<24;i++){
    arow.data = ad + 24*2*i; //i'th row offset
    bcol.data = bd + 24*2*i; //i'th col offset (remember we transposed it)

    gsl_blas_zdotu(&arow, &bcol, &tmp);
    reinterpret_cast<double(&)[2]>(out)[0] += GSL_REAL(tmp);
    reinterpret_cast<double(&)[2]>(out)[1] += GSL_IMAG(tmp);
  }
  return out;
}


//For a Nrows*Ncols matrix 'to' with elements in the standard order  idx=(Ncols*i + j), poke a submatrix into it with origin (i0,j0) and size (ni,nj)
template<typename T>
void pokeSubmatrix(T* to, const T* sub, const int Nrows, const int Ncols, const int i0, const int j0, const int ni, const int nj, const bool threaded = false){
  #define DOIT \
    for(int row = i0; row < i0+ni; row++){ \
      T* to_block = to + row*Ncols + j0;	  \
      const T* from_block = sub + (row-i0)*nj;	\
      memcpy(to_block,from_block,nj*sizeof(T));	\
    }
  if(threaded){
#pragma omp parallel for
    DOIT;
  }else{
    DOIT;
  }
  #undef DOIT
}

//For a Nrows*Ncols matrix 'from' with elements in the standard order  idx=(Ncols*i + j), get a submatrix with origin (i0,j0) and size (ni,nj) and store in sub
template<typename T>
void getSubmatrix(T* sub, const T* from, const int Nrows, const int Ncols, const int i0, const int j0, const int ni, const int nj, const bool threaded = false){
  #define DOIT \
    for(int row = i0; row < i0+ni; row++){		\
      const T* from_block = from + row*Ncols + j0;	\
      T* to_block = sub + (row-i0)*nj;			\
      memcpy(to_block,from_block,nj*sizeof(T));		\
    }
  if(threaded){
#pragma omp parallel for
    DOIT;
  }else{
    DOIT;
  }
  #undef DOIT
}

//Grid peeking and poking to SIMD lanes
#ifdef USE_GRID

template<typename T>
struct TensorPeekPoke{};

template<typename C>
struct ScalarPeekPoke{
  typedef typename Grid::GridTypeMapper<C>::scalar_type ScalarC;
  inline static void pokeLane(C &into, const ScalarC &from, const int lane){
    ScalarC* Cp = (ScalarC*)&into;
    Cp[lane] = from;
  }
  inline static void peekLane(ScalarC &into, const C &from, const int lane){
    ScalarC const* Cp = (ScalarC const*)&from;
    into = Cp[lane];
  }
};

template<typename T, int isScalar>
struct TensorScalarPeekPokeReroute{};

template<typename T>
struct TensorScalarPeekPokeReroute<T,0>{
  typedef TensorPeekPoke<T> Route;
};
template<typename T>
struct TensorScalarPeekPokeReroute<T,1>{
  typedef ScalarPeekPoke<T> Route;
};



template<typename U>
struct TensorPeekPoke<Grid::iScalar<U> >{
  typedef typename Grid::iScalar<U>::scalar_object ScalarType;
  inline static void pokeLane(Grid::iScalar<U> &into, const ScalarType &from, const int lane){
    TensorScalarPeekPokeReroute<U, !Grid::isGridTensor<U>::value>::Route::pokeLane(into._internal, from._internal, lane);
  }
  inline static void peekLane(ScalarType &into, const Grid::iScalar<U> &from, const int lane){
    TensorScalarPeekPokeReroute<U, !Grid::isGridTensor<U>::value>::Route::peekLane(into._internal, from._internal, lane);
  }
};

template<typename U, int N>
struct TensorPeekPoke<Grid::iVector<U,N> >{
  typedef typename Grid::iVector<U,N>::scalar_object ScalarType;
  inline static void pokeLane(Grid::iVector<U,N> &into, const ScalarType &from, const int lane){
    for(int i=0;i<N;i++)
      TensorScalarPeekPokeReroute<U, !Grid::isGridTensor<U>::value>::Route::pokeLane(into._internal[i], from._internal[i], lane);
  }
  inline static void peekLane(ScalarType &into, const Grid::iVector<U,N> &from, const int lane){
    for(int i=0;i<N;i++)
      TensorScalarPeekPokeReroute<U, !Grid::isGridTensor<U>::value>::Route::peekLane(into._internal[i], from._internal[i], lane);
  }
};

template<typename U, int N>
struct TensorPeekPoke<Grid::iMatrix<U,N> >{
  typedef typename Grid::iMatrix<U,N>::scalar_object ScalarType;
  inline static void pokeLane(Grid::iMatrix<U,N> &into, const ScalarType &from, const int lane){
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++)
	TensorScalarPeekPokeReroute<U, !Grid::isGridTensor<U>::value>::Route::pokeLane(into._internal[i][j], from._internal[i][j], lane);
  }
  inline static void peekLane(ScalarType &into, const Grid::iMatrix<U,N> &from, const int lane){
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++)
	TensorScalarPeekPokeReroute<U, !Grid::isGridTensor<U>::value>::Route::peekLane(into._internal[i][j], from._internal[i][j], lane);
  }
};


template<typename T>
void pokeLane(T &into, const typename Grid::GridTypeMapper<T>::scalar_object &from, const int lane){
  TensorScalarPeekPokeReroute<T, !Grid::isGridTensor<T>::value >::Route::pokeLane(into,from,lane);
}
template<typename T>
void peekLane(typename Grid::GridTypeMapper<T>::scalar_object &into, const T &from, const int lane){
  TensorScalarPeekPokeReroute<T, !Grid::isGridTensor<T>::value >::Route::peekLane(into,from,lane);
}

#endif





CPS_END_NAMESPACE

#endif
