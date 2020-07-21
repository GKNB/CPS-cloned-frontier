#ifndef _SPIN_COLOR_MATRICES_H
#define _SPIN_COLOR_MATRICES_H

//These are alternatives to Matrix, WilsonMatrix, SpinColorFlavorMatrix
#include<util/flavormatrix.h>
#include <alg/a2a/utils/template_wizardry.h>

CPS_START_NAMESPACE

template<typename T>
accelerator_inline void CPSsetZero(T &what){
  what = 0.;
}
#ifdef USE_GRID
template<>
accelerator_inline void CPSsetZero(Grid::vComplexD &what){
  zeroit(what);
}
template<>
accelerator_inline void CPSsetZero(Grid::vComplexF &what){
  zeroit(what);
}
#endif

template<typename T>
inline void CPSprintT(std::ostream &into, const T &what){
  into << what;
}
template<>
inline void CPSprintT(std::ostream &into, const std::complex<double> &what){
  into << "(" << what.real() << "," << what.imag() << ")";
}
template<>
inline void CPSprintT(std::ostream &into, const std::complex<float> &what){
  into << "(" << what.real() << "," << what.imag() << ")";
}

template<typename T>
class isCPSsquareMatrix{

  template<typename U, U> struct Check;

  template<typename U>
  static char test(Check<int, U::isDerivedFromCPSsquareMatrix> *);

  template<typename U>
  static double test(...);
  
public:

  enum {value = sizeof(test<T>(0)) == sizeof(char) };
};


struct cps_square_matrix_mark;

template<typename T>
struct ClassifyMatrixOrNotMatrix{
  typedef typename TestElem< isCPSsquareMatrix<T>::value, cps_square_matrix_mark,LastElem >::type type;
};

template<typename T, typename Tclass>
struct _timespmI{};


template<typename T>
struct _timespmI<T, no_mark>{
  accelerator_inline static void timesMinusOne(T &out, const T &in){
    out = -in;
  }  
  accelerator_inline static void timesI(T &out, const T &in){
    out = cps::timesI(in);
  }
  accelerator_inline static void timesMinusI(T &out, const T &in){
    out = cps::timesMinusI(in);
  }
};
template<typename T>
struct _timespmI<T,cps_square_matrix_mark>{
  accelerator_inline static void timesMinusOne(T &out, const T &in){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_timespmI<typename T::value_type, typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::timesMinusOne(out(i,j), in(i,j));
  }    
  accelerator_inline static void timesI(T &out, const T &in){    
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_timespmI<typename T::value_type, typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::timesI(out(i,j), in(i,j));
  }
  accelerator_inline static void timesMinusI(T &out, const T &in){    
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_timespmI<typename T::value_type, typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::timesMinusI(out(i,j), in(i,j));
  }
};

template<typename T, typename TypeClass>
struct _CPSsetZeroOne{};

template<typename T>
struct _CPSsetZeroOne<T,no_mark>{
  accelerator_inline static void setone(T &what){
    what = 1.0;
  }
  accelerator_inline static void setzero(T &what){
    CPSsetZero(what);
  }
};
template<typename T>
struct _CPSsetZeroOne<T, cps_square_matrix_mark>{
  accelerator_inline static void setone(T &what){
    what.unit();
  }
  accelerator_inline static void setzero(T &what){
    what.zero();
  }
};

//Find the underlying scalar type
template<typename T, typename TypeClass>
struct _RecursiveTraceFindScalarType{};

template<typename T>
struct _RecursiveTraceFindScalarType<T, cps_square_matrix_mark>{
  typedef typename _RecursiveTraceFindScalarType<typename T::value_type,typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::scalar_type scalar_type;
};
template<typename T>
struct _RecursiveTraceFindScalarType<T, no_mark>{
  typedef T scalar_type;
};


//Count the number of fundamental scalars in the nested matrix
template<typename T, typename TypeClass>
struct _RecursiveCountScalarType{};

template<typename T>
struct _RecursiveCountScalarType<T, cps_square_matrix_mark>{
  static constexpr size_t count(){
    return T::Size*T::Size*_RecursiveCountScalarType<typename T::value_type,typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type >::count();
  }
};
template<typename T>
struct _RecursiveCountScalarType<T, no_mark>{
  static constexpr size_t count(){
    return 1;
  }
};
 

//Perform a trace of an arbitrary nested square matrix type
template<typename scalar_type, typename T, typename TypeClass>
struct _RecursiveTraceImpl{};

template<typename scalar_type, typename T>
struct _RecursiveTraceImpl<scalar_type, T, cps_square_matrix_mark>{
  accelerator_inline static void doit(scalar_type &into, const T &what){
    for(int i=0;i<T::Size;i++)
      _RecursiveTraceImpl<scalar_type, typename T::value_type, typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::doit(into,what(i,i));    
  }
  accelerator_inline static void trace_prod(scalar_type &into, const T &a, const T&b){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_RecursiveTraceImpl<scalar_type, typename T::value_type, typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::trace_prod(into, a(i,j), b(j,i));
  }
  
};
template<typename scalar_type>
struct _RecursiveTraceImpl<scalar_type, scalar_type, no_mark>{
  accelerator_inline static void doit(scalar_type &into, const scalar_type &what){
    into = into + what;
  }
  accelerator_inline static void trace_prod(scalar_type &into, const scalar_type &a, const scalar_type &b){
    into = into + a*b;
  }
};

//Perform a trace over the nested matrix at level RemoveDepth
template<typename T, int RemoveDepth>
struct _PartialTraceFindReducedType{
  typedef typename T::template Rebase< typename _PartialTraceFindReducedType<typename T::value_type,RemoveDepth-1>::type >::type type;
};
template<typename T>
struct _PartialTraceFindReducedType<T,0>{
  typedef typename T::value_type type;
};

template<typename U,typename T, int RemoveDepth>
struct _PartialTraceImpl{
  accelerator_inline static void doit(U &into, const T&from){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_PartialTraceImpl<typename U::value_type, typename T::value_type, RemoveDepth-1>::doit(into(i,j), from(i,j));
  }
};
template<typename U, typename T>
struct _PartialTraceImpl<U,T,0>{
  accelerator_inline static void doit(U &into, const T&from){
    for(int i=0;i<T::Size;i++)
      into += from(i,i);
  }
};


//Perform a trace over the nested matrix at level RemoveDepth1 and that at RemoveDepth2
template<typename T, int RemoveDepth1, int RemoveDepth2> 
struct _PartialDoubleTraceFindReducedType{
  typedef typename my_enable_if<RemoveDepth1 < RemoveDepth2,int>::type test;
  typedef typename T::template Rebase< typename _PartialDoubleTraceFindReducedType<typename T::value_type,RemoveDepth1-1,RemoveDepth2-1>::type >::type type;
};
template<typename T,int RemoveDepth2>
struct _PartialDoubleTraceFindReducedType<T,0,RemoveDepth2>{
  typedef typename _PartialTraceFindReducedType<typename T::value_type,RemoveDepth2-1>::type type;
};



template<typename U,typename T, int RemoveDepth1,int RemoveDepth2>
struct _PartialDoubleTraceImpl{
  accelerator_inline static void doit(U &into, const T&from){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_PartialDoubleTraceImpl<typename U::value_type, typename T::value_type, RemoveDepth1-1,RemoveDepth2-1>::doit(into(i,j), from(i,j));
  }
};
template<typename U, typename T, int RemoveDepth2>
struct _PartialDoubleTraceImpl<U,T,0,RemoveDepth2>{
  accelerator_inline static void doit(U &into, const T&from){
    for(int i=0;i<T::Size;i++)
      _PartialTraceImpl<U,typename T::value_type, RemoveDepth2-1>::doit(into, from(i,i));
  }
};

//Transpose the matrix at depth TransposeDepth
template<typename T, int TransposeDepth>
struct _IndexTransposeImpl{
  accelerator_inline static void doit(T &into, const T&from){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_IndexTransposeImpl<typename T::value_type, TransposeDepth-1>::doit(into(i,j), from(i,j));
  }
};
template<typename T>
struct _IndexTransposeImpl<T,0>{
  accelerator_inline static void doit(T &into, const T&from){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	into(i,j) = from(j,i);
  }
};

//Get the type with a different underlying numerical type
template<typename T, typename TypeClass, typename NewNumericalType>
struct _rebaseScalarType{};

template<typename T, typename NewNumericalType>
struct _rebaseScalarType<T, cps_square_matrix_mark, NewNumericalType>{
  typedef typename _rebaseScalarType<typename T::value_type,typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type, NewNumericalType>::type subType;
  typedef typename T::template Rebase<subType>::type type;
};
template<typename T, typename NewNumericalType>
struct _rebaseScalarType<T, no_mark, NewNumericalType>{
  typedef NewNumericalType type;
};




template<typename T, int N>
class CPSsquareMatrix{
protected:
  T v[N][N];
  template<typename U>  //, typename my_enable_if< my_is_base_of<U, CPSsquareMatrix<T,N> >::value, int>::type = 0
  accelerator_inline static void mult(U &into, const U &a, const U &b){
    into.zero();
    for(int i=0;i<N;i++)
      for(int k=0;k<N;k++)
	for(int j=0;j<N;j++)
	  into(i,k) = into(i,k) + a(i,j)*b(j,k);      
  }
public:
  enum { isDerivedFromCPSsquareMatrix=1 };
  enum { Size = N };
  typedef T value_type; //the type of the elements (can be another matrix)
  typedef typename _RecursiveTraceFindScalarType<CPSsquareMatrix<T,N>,cps_square_matrix_mark>::scalar_type scalar_type; //the underlying numerical type
  
  //Get the type were the value_type of this matrix to be replaced by matrix U
  template<typename U>
  struct Rebase{
    typedef CPSsquareMatrix<U,N> type;
  };
  //Get the type were the underlying numerical type (scalar_type) to be replaced by type U
  template<typename U>
  struct RebaseScalarType{
    typedef typename _rebaseScalarType<CPSsquareMatrix<T,N>,cps_square_matrix_mark,U>::type type;
  };

  //The number of fundamental "scalar_type" data in memory
  static constexpr size_t nScalarType(){
    return _RecursiveCountScalarType<CPSsquareMatrix<T,N>, cps_square_matrix_mark>::count();
  }
  //Return a pointer to this object as an array of scalar_type of size nScalarType()
  scalar_type const* scalarTypePtr() const{ return (scalar_type const*)this; }
  scalar_type * scalarTypePtr(){ return (scalar_type*)this; }


  accelerator CPSsquareMatrix() = default;
  accelerator CPSsquareMatrix(const CPSsquareMatrix &r) = default;
  
  accelerator_inline T & operator()(const int i, const int j){ return v[i][j]; }
  accelerator_inline const T & operator()(const int i, const int j) const{ return v[i][j]; }

  accelerator_inline void equalsTranspose(const CPSsquareMatrix<T,N> &r){
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++)
	v[i][j] = r.v[j][i];
  }
  accelerator_inline CPSsquareMatrix<T,N> Transpose() const{
    CPSsquareMatrix<T,N> out; out.equalsTranspose(*this);
    return out;
  }				
  
  //Trace on just the index associated with this matrix
  // T traceIndex() const{
  //   T out; CPSsetZero(out);
  //   for(int i=0;i<N;i++) out = out + v[i][i];
  //   return out;
  // }
  //Trace on all indices recursively
  accelerator_inline scalar_type Trace() const{
    scalar_type ret; CPSsetZero(ret);
    _RecursiveTraceImpl<scalar_type, CPSsquareMatrix<T,N>, cps_square_matrix_mark>::doit(ret, *this);
    return ret;
  }

  template<int RemoveDepth>
  accelerator_inline typename _PartialTraceFindReducedType<CPSsquareMatrix<T,N>, RemoveDepth>::type TraceIndex() const{
    typedef typename _PartialTraceFindReducedType<CPSsquareMatrix<T,N>, RemoveDepth>::type ReducedType;
    ReducedType into; _CPSsetZeroOne<ReducedType, typename ClassifyMatrixOrNotMatrix<ReducedType>::type>::setzero(into);//  into.zero();
    _PartialTraceImpl<ReducedType, CPSsquareMatrix<T,N>, RemoveDepth>::doit(into, *this);
    return into;
  }

  template<int RemoveDepth1, int RemoveDepth2>
  accelerator_inline typename _PartialDoubleTraceFindReducedType<CPSsquareMatrix<T,N>,RemoveDepth1,RemoveDepth2>::type TraceTwoIndices() const{
    typedef typename _PartialDoubleTraceFindReducedType<CPSsquareMatrix<T,N>,RemoveDepth1,RemoveDepth2>::type ReducedType;
    ReducedType into; _CPSsetZeroOne<ReducedType, typename ClassifyMatrixOrNotMatrix<ReducedType>::type>::setzero(into);//  into.zero();
    _PartialDoubleTraceImpl<ReducedType, CPSsquareMatrix<T,N>, RemoveDepth1,RemoveDepth2>::doit(into, *this);
    return into;
  }

  //Set this matrix equal to the transpose of r on its compound tensor index TransposeDepth. If it is not a compound matrix then TranposeDepth=0 does the same thing as equalsTranspose above
  //i.e. in(i,j)(k,l)(m,n)  transpose on index depth 1:   out(i,j)(k,l)(m,n) = in(i,j)(l,k)(m,n)
  template<int TransposeDepth>
  accelerator_inline void equalsTransposeOnIndex(const CPSsquareMatrix<T,N> &r){
    assert(&r != this);
    _IndexTransposeImpl<CPSsquareMatrix<T,N>, TransposeDepth>::doit(*this,r);
  }
  template<int TransposeDepth>
  accelerator_inline CPSsquareMatrix<T,N> TransposeOnIndex() const{
    CPSsquareMatrix<T,N> out(*this);
    out.equalsTransposeOnIndex<TransposeDepth>(*this);
    return out;
  }

  accelerator_inline CPSsquareMatrix<T,N> & zero(){
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++)
	_CPSsetZeroOne<T,  typename ClassifyMatrixOrNotMatrix<T>::type>::setzero(v[i][j]);
    return *this;    
  }
  accelerator_inline CPSsquareMatrix<T,N>& unit(){
    zero();
    for(int i=0;i<N;i++)
      _CPSsetZeroOne<T,  typename ClassifyMatrixOrNotMatrix<T>::type>::setone(v[i][i]);
    return *this;
  }
  //this = -this
  accelerator_inline CPSsquareMatrix<T,N> & timesMinusOne(){
    _timespmI<CPSsquareMatrix<T,N>,cps_square_matrix_mark>::timesMinusOne(*this,*this);
    return *this;
  }
  //this = i*this
  accelerator_inline CPSsquareMatrix<T,N> & timesI(){
    _timespmI<CPSsquareMatrix<T,N>,cps_square_matrix_mark>::timesI(*this,*this);
    return *this;
  }
  //this = -i*this
  accelerator_inline CPSsquareMatrix<T,N> & timesMinusI(){
    _timespmI<CPSsquareMatrix<T,N>,cps_square_matrix_mark>::timesMinusI(*this,*this);
    return *this;
  }
    
  accelerator_inline CPSsquareMatrix<T,N> & operator+=(const CPSsquareMatrix<T,N> &r){
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++)
	v[i][j] = v[i][j] + r.v[i][j];
    return *this;
  }
  accelerator_inline CPSsquareMatrix<T,N> & operator*=(const T &r){
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++)
	v[i][j] = v[i][j] * r;
    return *this;
  }
  accelerator_inline CPSsquareMatrix<T,N> & operator-=(const CPSsquareMatrix<T,N> &r){
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++)
	v[i][j] = v[i][j] - r.v[i][j];
    return *this;
  }


  accelerator_inline bool operator==(const CPSsquareMatrix<T,N> &r) const{
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++)
	if(v[i][j] != r.v[i][j]) return false;
    return true;
  }
  friend std::ostream & operator<<(std::ostream &os, const CPSsquareMatrix<T,N> &m){
    for(int i=0;i<N;i++){
      for(int j=0;j<N;j++){
	CPSprintT(os, m.v[i][j]);
	if(j<N-1) os << " ";
      }
      os << '\n';
    }
    return os;
  }
};

template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, U>::type operator*(const U &a, const U &b){
  U into;
  into.zero();
  for(int i=0;i<U::Size;i++)
    for(int k=0;k<U::Size;k++)
      for(int j=0;j<U::Size;j++)
	into(i,k) = into(i,k) + a(i,j)*b(j,k);
  return into;
}
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, U>::type operator*(const U &a, const typename U::scalar_type &b){
  U into;
  into.zero();
  for(int i=0;i<U::Size;i++)
    for(int j=0;j<U::Size;j++)
      into(i,j) = a(i,j) * b;
  return into;
}
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, U>::type operator+(const U &a, const U &b){
  U into;
  for(int i=0;i<U::Size;i++)
    for(int j=0;j<U::Size;j++)
      into(i,j) = a(i,j) + b(i,j);
  return into;
}
template<typename U> 
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, U>::type operator-(const U &a, const U &b){
  U into;
  for(int i=0;i<U::Size;i++)
    for(int j=0;j<U::Size;j++)
      into(i,j) = a(i,j) - b(i,j);
  return into;
}



template<typename U>
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, typename U::scalar_type>::type Trace(const U &a, const U &b){
  typename U::scalar_type out;
  CPSsetZero(out);
  _RecursiveTraceImpl<typename U::scalar_type, U, cps_square_matrix_mark>::trace_prod(out, a,b);
  return out;
}

template<typename T>
accelerator_inline T Transpose(const T& r){
  T out;
  out.equalsTranspose(r);
  return out;
}

//Perform SIMD reductions
#ifdef USE_GRID
template<typename VectorMatrixType, typename my_enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
inline typename VectorMatrixType::template RebaseScalarType<typename VectorMatrixType::scalar_type::scalar_type>::type
Reduce(const VectorMatrixType &v){
  typedef typename VectorMatrixType::template RebaseScalarType<typename VectorMatrixType::scalar_type::scalar_type>::type OutType;
  OutType out;
  typename OutType::scalar_type *out_p = out.scalarTypePtr();
  typename VectorMatrixType::scalar_type const* in_p = v.scalarTypePtr();
  static const size_t NN = v.nScalarType();
  for(size_t i=0;i<NN;i++) out_p[i] = Reduce(in_p[i]);
  return out;
}
#endif

//Sum matrix over all nodes
template<typename MatrixType, typename my_enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type = 0>
void globalSum(MatrixType *m){
  globalSum( m->scalarTypePtr(), m->nScalarType() );
}

//Old annoying CPS conventions with output on RHS (Fortran user or something??)
#define TIMESPLUSONE(a,b) { b=a; }
#define TIMESMINUSONE(a,b) { _timespmI<T, typename ClassifyMatrixOrNotMatrix<T>::type>::timesMinusOne(b,a); }
#define TIMESPLUSI(a,b) { _timespmI<T, typename ClassifyMatrixOrNotMatrix<T>::type>::timesI(b,a); }
#define TIMESMINUSI(a,b) { _timespmI<T, typename ClassifyMatrixOrNotMatrix<T>::type>::timesMinusI(b,a); }
#define SETZERO(a){ _CPSsetZeroOne<T, typename ClassifyMatrixOrNotMatrix<T>::type>::setzero(a); }


//For derived classes we want methods to return references or instances of the derived type for inherited functions
//Must have "base_type" defined
//DERIVED = the full derived class name, eg CPSflavorMatrix<T>
//DERIVED_CON = the derived class constructor name, eg CPSflavorMatrix
#define INHERIT_METHODS_AND_TYPES(DERIVED, DERIVED_CON)		\
  typedef typename base_type::value_type value_type;	\
  typedef typename base_type::scalar_type scalar_type;			\
  accelerator_inline operator base_type(){ return static_cast<base_type &>(*this); } \
									\
  accelerator DERIVED_CON(const base_type &r): base_type(r){} \
  accelerator DERIVED_CON(): base_type(){}			\
  accelerator DERIVED_CON(base_type &&r): base_type(std::move(r)){}	\
									\
  accelerator_inline DERIVED Transpose() const{ return this->base_type::Transpose(); } \
  template<int TransposeDepth>						\
  accelerator_inline DERIVED TransposeOnIndex() const{ return this->base_type::template TransposeOnIndex<TransposeDepth>(); } \
									\
  accelerator_inline DERIVED & zero(){ return static_cast<DERIVED &>(this->base_type::zero()); } \  
  accelerator_inline DERIVED & unit(){ return static_cast<DERIVED &>(this->base_type::unit()); } \  
  accelerator_inline DERIVED & timesMinusOne(){ return static_cast<DERIVED &>(this->base_type::timesMinusOne()); } \  
  accelerator_inline DERIVED & timesI(){ return static_cast<DERIVED &>(this->base_type::timesI()); } \  
  accelerator_inline DERIVED & timesMinusI(){ return static_cast<DERIVED &>(this->base_type::timesMinusI()); } \  
									\
  accelerator_inline DERIVED & operator=(const base_type &r){ return static_cast<DERIVED &>(this->base_type::operator=(r)); } \
  accelerator_inline DERIVED & operator=(base_type &&r){ return static_cast<DERIVED &>(this->base_type::operator=(std::move(r))); } \
  accelerator_inline DERIVED & operator+=(const DERIVED &r){ return static_cast<DERIVED &>(this->base_type::operator+=(r)); } \
  accelerator_inline DERIVED & operator*=(const scalar_type &r){ return static_cast<DERIVED &>(this->base_type::operator*=(r)); } \
  accelerator_inline DERIVED & operator-=(const DERIVED &r){ return static_cast<DERIVED &>(this->base_type::operator-=(r)); } 



template<typename T>
class CPSflavorMatrix: public CPSsquareMatrix<T,2>{
public:
  typedef CPSsquareMatrix<T,2> base_type;
  INHERIT_METHODS_AND_TYPES(CPSflavorMatrix<T>, CPSflavorMatrix);
  
  template<typename U>
  struct Rebase{
    typedef CPSflavorMatrix<U> type;
  };
  //Get the type were the underlying numerical type (scalar_type) to be replaced by type U
  template<typename U>
  struct RebaseScalarType{
    typedef typename _rebaseScalarType<CPSflavorMatrix<T>,cps_square_matrix_mark,U>::type type;
  };

  
  //multiply on left by a flavor matrix
  accelerator_inline CPSflavorMatrix<T> & pl(const FlavorMatrixType &type){
    T tmp1, tmp2;
    T (&v)[2][2] = this->v;
    
    switch( type ){
    case F0:
      SETZERO(v[1][0]);
      SETZERO(v[1][1]);
      break;
    case F1:
      SETZERO(v[0][0]);
      SETZERO(v[0][1]);
      break;
    case Fud:
      tmp1 = v[0][0];
      tmp2 = v[0][1];
      v[0][0] = v[1][0];
      v[0][1] = v[1][1];
      v[1][0] = tmp1;
      v[1][1] = tmp2;
      break;
    case sigma0:
      break;
    case sigma1:
      tmp1 = v[0][0];
      tmp2 = v[0][1];
      v[0][0] = v[1][0];
      v[0][1] = v[1][1];
      v[1][0] = tmp1;
      v[1][1] = tmp2;
      break;      
    case sigma2:
      TIMESPLUSI(v[0][0], tmp1);
      TIMESPLUSI(v[0][1], tmp2);
      TIMESMINUSI(v[1][0], v[0][0]);
      TIMESMINUSI(v[1][1], v[0][1]);
      v[1][0] = tmp1;
      v[1][1] = tmp2;
      break;
    case sigma3:
      TIMESMINUSONE(v[1][0],v[1][0]);
      TIMESMINUSONE(v[1][1],v[1][1]);
      break;
    default:
      assert(0);
      //ERR.General("FlavorMatrixGeneral","pl(const FlavorMatrixGeneralType &type)","Unknown FlavorMatrixGeneralType");
      break;
    }
    return *this;

  }

  //multiply on right by a flavor matrix
  accelerator_inline CPSflavorMatrix<T> & pr(const FlavorMatrixType &type){
    T tmp1, tmp2;
    T (&v)[2][2] = this->v;
    
    switch(type){
    case F0:     
      SETZERO(v[0][1]);
      SETZERO(v[1][1]);
      break;
    case F1:
      SETZERO(v[0][0]);
      SETZERO(v[1][0]);
      break;
    case Fud:
      tmp1 = v[0][0];
      tmp2 = v[1][0];
      v[0][0] = v[0][1];
      v[1][0] = v[1][1];
      v[0][1] = tmp1;
      v[1][1] = tmp2;
      break;
    case sigma0:
      break;
    case sigma1:
      tmp1 = v[0][0];
      tmp2 = v[1][0];
      v[0][0] = v[0][1];
      v[1][0] = v[1][1];
      v[0][1] = tmp1;
      v[1][1] = tmp2;
      break;      
    case sigma2:
      TIMESMINUSI(v[0][0], tmp1);
      TIMESMINUSI(v[1][0], tmp2);
      TIMESPLUSI(v[0][1], v[0][0]);
      TIMESPLUSI(v[1][1], v[1][0]);
      v[0][1] = tmp1;
      v[1][1] = tmp2;
      break;
    case sigma3:
      TIMESMINUSONE(v[0][1],v[0][1]);
      TIMESMINUSONE(v[1][1],v[1][1]);
      break;
    default:
      //ERR.General("FlavorMatrixGeneral","pr(const FlavorMatrixGeneralType &type)","Unknown FlavorMatrixGeneralType");
      assert(0);
      break;
    }
    return *this;
  }

};


  
  
//  Chiral basis
//  gamma(XUP)    gamma(YUP)    gamma(ZUP)    gamma(TUP)    gamma(FIVE)
//  0  0  0  i    0  0  0 -1    0  0  i  0    0  0  1  0    1  0  0  0
//  0  0  i  0    0  0  1  0    0  0  0 -i    0  0  0  1    0  1  0  0
//  0 -i  0  0    0  1  0  0   -i  0  0  0    1  0  0  0    0  0 -1  0
// -i  0  0  0   -1  0  0  0    0  i  0  0    0  1  0  0    0  0  0 -1

template<typename T>
class CPSspinMatrix: public CPSsquareMatrix<T,4>{
public:
  typedef CPSsquareMatrix<T,4> base_type;
  INHERIT_METHODS_AND_TYPES(CPSspinMatrix<T>, CPSspinMatrix);
  
  template<typename U>
  struct Rebase{
    typedef CPSspinMatrix<U> type;
  };
  //Get the type were the underlying numerical type (scalar_type) to be replaced by type U
  template<typename U>
  struct RebaseScalarType{
    typedef typename _rebaseScalarType<CPSspinMatrix<T>,cps_square_matrix_mark,U>::type type;
  };

  
  //Left Multiplication by Dirac gamma's
  accelerator_inline CPSspinMatrix<T> & gl(int dir){
    int s2;
    CPSspinMatrix<T> cp(*this);
    const T (&src)[4][4] = cp.v;
    T (&p)[4][4] = this->v;
    
    switch(dir){
    case 0:
      for(s2=0;s2<4;s2++){
	TIMESPLUSI(  src[3][s2], p[0][s2] );
	TIMESPLUSI(  src[2][s2], p[1][s2] );
	TIMESMINUSI( src[1][s2], p[2][s2] );
	TIMESMINUSI( src[0][s2], p[3][s2] );
      }
      break;
    case 1:
      for(s2=0;s2<4;s2++){
	TIMESMINUSONE( src[3][s2], p[0][s2] );
	TIMESPLUSONE(  src[2][s2], p[1][s2] );
	TIMESPLUSONE(  src[1][s2], p[2][s2] );
	TIMESMINUSONE( src[0][s2], p[3][s2] );
      }
      break;
    case 2:
      for(s2=0;s2<4;s2++){
	TIMESPLUSI(  src[2][s2], p[0][s2] );
	TIMESMINUSI( src[3][s2], p[1][s2] );
	TIMESMINUSI( src[0][s2], p[2][s2] );
	TIMESPLUSI(  src[1][s2], p[3][s2] );
      }
      break;
    case 3:
      for(s2=0;s2<4;s2++){
	TIMESPLUSONE( src[2][s2], p[0][s2] );
	TIMESPLUSONE( src[3][s2], p[1][s2] );
	TIMESPLUSONE( src[0][s2], p[2][s2] );
	TIMESPLUSONE( src[1][s2], p[3][s2] );
      }
      break;
    case -5:
      for(s2=0;s2<4;s2++){
	TIMESPLUSONE(  src[0][s2], p[0][s2] );
	TIMESPLUSONE(  src[1][s2], p[1][s2] );
	TIMESMINUSONE( src[2][s2], p[2][s2] );
	TIMESMINUSONE( src[3][s2], p[3][s2] );
      }
      break;
    default:
      assert(0);
      break;
    }
    return *this;
  }


  //Right Multiplication by Dirac gamma's

  accelerator_inline CPSspinMatrix<T>& gr(int dir)
  {
    int s1;
    CPSspinMatrix<T> cp(*this);
    const T (&src)[4][4] = cp.v;
    T (&p)[4][4] = this->v;

    switch(dir){
    case 0:
      for(s1=0;s1<4;s1++){
	TIMESMINUSI( src[s1][3], p[s1][0] );
	TIMESMINUSI( src[s1][2], p[s1][1] );
	TIMESPLUSI(  src[s1][1], p[s1][2] );
	TIMESPLUSI(  src[s1][0], p[s1][3] );
      }
      break;
    case 1:
      for(s1=0;s1<4;s1++){
	TIMESMINUSONE( src[s1][3], p[s1][0] );
	TIMESPLUSONE(  src[s1][2], p[s1][1] );
	TIMESPLUSONE(  src[s1][1], p[s1][2] );
	TIMESMINUSONE( src[s1][0], p[s1][3] );
      }
      break;
    case 2:
      for(s1=0;s1<4;s1++){
	TIMESMINUSI( src[s1][2], p[s1][0] );
	TIMESPLUSI(  src[s1][3], p[s1][1] );
	TIMESPLUSI(  src[s1][0], p[s1][2] );
	TIMESMINUSI( src[s1][1], p[s1][3] );
      }
      break;
    case 3:
      for(s1=0;s1<4;s1++){
	TIMESPLUSONE( src[s1][2], p[s1][0] );
	TIMESPLUSONE( src[s1][3], p[s1][1] );
	TIMESPLUSONE( src[s1][0], p[s1][2] );
	TIMESPLUSONE( src[s1][1], p[s1][3] );
      }
      break;
    case -5:
      for(s1=0;s1<4;s1++){
	TIMESPLUSONE(  src[s1][0], p[s1][0] );
	TIMESPLUSONE(  src[s1][1], p[s1][1] );
	TIMESMINUSONE( src[s1][2], p[s1][2] );
	TIMESMINUSONE( src[s1][3], p[s1][3] );
      }
      break;
    default:
      assert(0);
      break;
    }
    return *this;
  }

  //multiply gamma(i)gamma(5) on the left: result = gamma(i)*gamma(5)*from
  accelerator_inline CPSspinMatrix<T>& glAx(const int dir){
    int s2;
    CPSspinMatrix<T> cp(*this);
    const T (&from_mat)[4][4] = cp.v;
    T (&p)[4][4] = this->v;
    
    switch(dir){
    case 0:
      for(s2=0;s2<4;s2++){
            TIMESMINUSI( from_mat[3][s2], p[0][s2] );
            TIMESMINUSI( from_mat[2][s2], p[1][s2] );
            TIMESMINUSI( from_mat[1][s2], p[2][s2] );
            TIMESMINUSI( from_mat[0][s2], p[3][s2] );
        }
        break;
    case 1:
        for(s2=0;s2<4;s2++){
            TIMESPLUSONE(  from_mat[3][s2], p[0][s2] );
            TIMESMINUSONE( from_mat[2][s2], p[1][s2] );
            TIMESPLUSONE(  from_mat[1][s2], p[2][s2] );
            TIMESMINUSONE( from_mat[0][s2], p[3][s2] );
        }
        break;
    case 2:
        for(s2=0;s2<4;s2++){
            TIMESMINUSI( from_mat[2][s2], p[0][s2] );
            TIMESPLUSI(  from_mat[3][s2], p[1][s2] );
            TIMESMINUSI( from_mat[0][s2], p[2][s2] );
            TIMESPLUSI(  from_mat[1][s2], p[3][s2] );
        }
	break;
    case 3:
        for(s2=0;s2<4;s2++){
            TIMESMINUSONE( from_mat[2][s2], p[0][s2] );
            TIMESMINUSONE( from_mat[3][s2], p[1][s2] );
            TIMESPLUSONE(  from_mat[0][s2], p[2][s2] );
            TIMESPLUSONE(  from_mat[1][s2], p[3][s2] );
        }
        break;
    default:
      assert(0);
      break;
    }
    return *this;
  }

  //multiply gamma(i)gamma(5) on the right: result = from*gamma(i)*gamma(5)
  accelerator_inline CPSspinMatrix<T>& grAx(int dir)
  {
    int s1;
    CPSspinMatrix<T> cp(*this);
    const T (&src)[4][4] = cp.v;
    T (&p)[4][4] = this->v;

    switch(dir){
    case 0:
      for(s1=0;s1<4;s1++){
	TIMESMINUSI( src[s1][3], p[s1][0] );
	TIMESMINUSI( src[s1][2], p[s1][1] );
	TIMESMINUSI(  src[s1][1], p[s1][2] );
	TIMESMINUSI(  src[s1][0], p[s1][3] );
      }
      break;
    case 1:
      for(s1=0;s1<4;s1++){
	TIMESMINUSONE( src[s1][3], p[s1][0] );
	TIMESPLUSONE(  src[s1][2], p[s1][1] );
	TIMESMINUSONE(  src[s1][1], p[s1][2] );
	TIMESPLUSONE( src[s1][0], p[s1][3] );
      }
      break;
    case 2:
      for(s1=0;s1<4;s1++){
	TIMESMINUSI( src[s1][2], p[s1][0] );
	TIMESPLUSI(  src[s1][3], p[s1][1] );
	TIMESMINUSI(  src[s1][0], p[s1][2] );
	TIMESPLUSI( src[s1][1], p[s1][3] );
      }
      break;
    case 3:
      for(s1=0;s1<4;s1++){
	TIMESPLUSONE( src[s1][2], p[s1][0] );
	TIMESPLUSONE( src[s1][3], p[s1][1] );
	TIMESMINUSONE( src[s1][0], p[s1][2] );
	TIMESMINUSONE( src[s1][1], p[s1][3] );
      }
      break;
    default:
      assert(0);
      break;
    }
    return *this;
  }

  
};



template<typename T>
class CPScolorMatrix: public CPSsquareMatrix<T,3>{
public:
  typedef CPSsquareMatrix<T,3> base_type;
  INHERIT_METHODS_AND_TYPES(CPScolorMatrix<T>, CPScolorMatrix);

  template<typename U>
  struct Rebase{
    typedef CPScolorMatrix<U> type;
  };
  //Get the type were the underlying numerical type (scalar_type) to be replaced by type U
  template<typename U>
  struct RebaseScalarType{
    typedef typename _rebaseScalarType<CPScolorMatrix<T>,cps_square_matrix_mark,U>::type type;
  };
};

template<typename ComplexType>
class CPSspinColorFlavorMatrix: public CPSspinMatrix<CPScolorMatrix<CPSflavorMatrix<ComplexType> > >{
public:
  typedef CPSspinMatrix<CPScolorMatrix<CPSflavorMatrix<ComplexType> > > SCFmat;
  typedef CPSspinMatrix<CPScolorMatrix<CPSflavorMatrix<ComplexType> > > base_type;
  INHERIT_METHODS_AND_TYPES(CPSspinColorFlavorMatrix<ComplexType>, CPSspinColorFlavorMatrix);
  
  template<typename U>
  struct Rebase{
    typedef CPSspinMatrix<U> type;
  };
  //Get the type were the underlying numerical type (scalar_type) to be replaced by type U
  template<typename U>
  struct RebaseScalarType{
    typedef CPSspinColorFlavorMatrix<U> type;
  };

  accelerator_inline CPScolorMatrix<ComplexType> SpinFlavorTrace() const{
    return this->CPSsquareMatrix<value_type,4>::template TraceTwoIndices<0,2>();
  }
  accelerator_inline CPSspinMatrix<CPSflavorMatrix<ComplexType> > ColorTrace() const{
    return this->CPSsquareMatrix<value_type,4>::template TraceIndex<1>();
  }

  accelerator_inline CPSspinColorFlavorMatrix<ComplexType> TransposeColor() const{
    return CPSspinColorFlavorMatrix<ComplexType>(this->CPSsquareMatrix<value_type,4>::template TransposeOnIndex<1>());
  }
  accelerator_inline void equalsColorTranspose(const CPSspinColorFlavorMatrix<ComplexType> &r){
    this->CPSsquareMatrix<value_type,4>::template equalsTransposeOnIndex<1>(r);
  }
  
  //multiply on left by a flavor matrix
  accelerator_inline CPSspinColorFlavorMatrix<ComplexType> & pl(const FlavorMatrixType type){
    for(int s1=0;s1<4;s1++)
      for(int s2=0;s2<4;s2++)
	for(int c1=0;c1<3;c1++)
	  for(int c2=0;c2<3;c2++)
	    this->operator()(s1,s2)(c1,c2).pl(type);
    return *this;
  }
  //multiply on left by a flavor matrix
  accelerator_inline CPSspinColorFlavorMatrix<ComplexType> & pr(const FlavorMatrixType type){
    for(int s1=0;s1<4;s1++)
      for(int s2=0;s2<4;s2++)
	for(int c1=0;c1<3;c1++)
	  for(int c2=0;c2<3;c2++)
	    this->operator()(s1,s2)(c1,c2).pr(type);
    return *this;
  }

  
};





#undef TIMESPLUSONE
#undef TIMESMINUSONE
#undef TIMESPLUSI
#undef TIMESMINUSI
#undef SETZERO



CPS_END_NAMESPACE

#endif
