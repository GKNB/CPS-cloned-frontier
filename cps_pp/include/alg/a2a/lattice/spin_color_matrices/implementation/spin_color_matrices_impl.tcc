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

  //this = -this
template<typename T, int N>
accelerator_inline CPSsquareMatrix<T,N> & CPSsquareMatrix<T,N>::timesMinusOne(){
  _timespmI<CPSsquareMatrix<T,N>,cps_square_matrix_mark>::timesMinusOne(*this,*this);
  return *this;
}
//this = i*this
template<typename T, int N>
accelerator_inline CPSsquareMatrix<T,N> & CPSsquareMatrix<T,N>::timesI(){
  _timespmI<CPSsquareMatrix<T,N>,cps_square_matrix_mark>::timesI(*this,*this);
  return *this;
}
//this = -i*this
template<typename T, int N>
accelerator_inline CPSsquareMatrix<T,N> & CPSsquareMatrix<T,N>::timesMinusI(){
  _timespmI<CPSsquareMatrix<T,N>,cps_square_matrix_mark>::timesMinusI(*this,*this);
  return *this;
}


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
template<typename T, int N>
accelerator_inline CPSsquareMatrix<T,N> & CPSsquareMatrix<T,N>::zero(){
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      _CPSsetZeroOne<T,  typename ClassifyMatrixOrNotMatrix<T>::type>::setzero(v[i][j]);
  return *this;    
}
template<typename T, int N>
accelerator_inline CPSsquareMatrix<T,N>& CPSsquareMatrix<T,N>::unit(){
  zero();
  for(int i=0;i<N;i++)
    _CPSsetZeroOne<T,  typename ClassifyMatrixOrNotMatrix<T>::type>::setone(v[i][i]);
  return *this;
}

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

template<typename T, int N>
accelerator_inline typename CPSsquareMatrix<T,N>::scalar_type CPSsquareMatrix<T,N>::Trace() const{
  scalar_type ret; CPSsetZero(ret);
  _RecursiveTraceImpl<scalar_type, CPSsquareMatrix<T,N>, cps_square_matrix_mark>::doit(ret, *this);
  return ret;
}



//Perform a trace over the nested matrix at level RemoveDepth
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

template<typename T, int N>
template<int RemoveDepth>
accelerator_inline typename _PartialTraceFindReducedType<CPSsquareMatrix<T,N>, RemoveDepth>::type CPSsquareMatrix<T,N>::TraceIndex() const{
  typedef typename _PartialTraceFindReducedType<CPSsquareMatrix<T,N>, RemoveDepth>::type ReducedType;
  ReducedType into; _CPSsetZeroOne<ReducedType, typename ClassifyMatrixOrNotMatrix<ReducedType>::type>::setzero(into);//  into.zero();
  _PartialTraceImpl<ReducedType, CPSsquareMatrix<T,N>, RemoveDepth>::doit(into, *this);
  return into;
}



//Perform a trace over the nested matrix at level RemoveDepth1 and that at RemoveDepth2
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

template<typename T, int N>
template<int RemoveDepth1, int RemoveDepth2>
accelerator_inline typename _PartialDoubleTraceFindReducedType<CPSsquareMatrix<T,N>,RemoveDepth1,RemoveDepth2>::type CPSsquareMatrix<T,N>::TraceTwoIndices() const{
  typedef typename _PartialDoubleTraceFindReducedType<CPSsquareMatrix<T,N>,RemoveDepth1,RemoveDepth2>::type ReducedType;
  ReducedType into; _CPSsetZeroOne<ReducedType, typename ClassifyMatrixOrNotMatrix<ReducedType>::type>::setzero(into);//  into.zero();
  _PartialDoubleTraceImpl<ReducedType, CPSsquareMatrix<T,N>, RemoveDepth1,RemoveDepth2>::doit(into, *this);
  return into;
}


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

template<typename T, int N>
template<int TransposeDepth>
accelerator_inline void CPSsquareMatrix<T,N>::equalsTransposeOnIndex(const CPSsquareMatrix<T,N> &r){
  assert(&r != this);
  _IndexTransposeImpl<CPSsquareMatrix<T,N>, TransposeDepth>::doit(*this,r);
}

template<typename T, int N>
template<int TransposeDepth>
accelerator_inline CPSsquareMatrix<T,N> CPSsquareMatrix<T,N>::TransposeOnIndex() const{
  CPSsquareMatrix<T,N> out(*this);
  out.equalsTransposeOnIndex<TransposeDepth>(*this);
  return out;
}

//Transpose a nested matrix on all indices
template<typename T, typename TypeClass>
struct _RecursiveTransposeImpl{};

template<typename T>
struct _RecursiveTransposeImpl<T, cps_square_matrix_mark>{
  accelerator_inline static void doit(T &into, const T &what){
    for(int i=0;i<T::Size;i++)
      for(int j=0;j<T::Size;j++)
	_RecursiveTransposeImpl<typename T::value_type, typename ClassifyMatrixOrNotMatrix<typename T::value_type>::type>::doit(into(i,j), what(j,i));
  } 
};
template<typename scalar_type>
struct _RecursiveTransposeImpl<scalar_type, no_mark>{
  accelerator_inline static void doit(scalar_type &into, const scalar_type &what){
    into = what;
  }
};

template<typename T, int N>
accelerator_inline void CPSsquareMatrix<T,N>::equalsTranspose(const CPSsquareMatrix<T,N> &r){
  _RecursiveTransposeImpl<CPSsquareMatrix<T,N>, cps_square_matrix_mark>::doit(*this, r);
}
template<typename T, int N>
accelerator_inline CPSsquareMatrix<T,N> CPSsquareMatrix<T,N>::Transpose() const{
  CPSsquareMatrix<T,N> out; out.equalsTranspose(*this);
  return out;
}	
template<typename T, int N>
accelerator_inline CPSsquareMatrix<T,N> & CPSsquareMatrix<T,N>::operator+=(const CPSsquareMatrix<T,N> &r){
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      v[i][j] = v[i][j] + r.v[i][j];
  return *this;
}
template<typename T, int N>
accelerator_inline CPSsquareMatrix<T,N> & CPSsquareMatrix<T,N>::operator*=(const T &r){
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      v[i][j] = v[i][j] * r;
  return *this;
}
template<typename T, int N>
accelerator_inline CPSsquareMatrix<T,N> & CPSsquareMatrix<T,N>::operator-=(const CPSsquareMatrix<T,N> &r){
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      v[i][j] = v[i][j] - r.v[i][j];
  return *this;
}

template<typename T, int N>
accelerator_inline bool CPSsquareMatrix<T,N>::operator==(const CPSsquareMatrix<T,N> &r) const{
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
      if(v[i][j] != r.v[i][j]) return false;
  return true;
}


template<typename T>
inline void _print_T_sqmat(std::ostream &into, const T &what){
  into << what;
}
template<>
inline void _print_T_sqmat(std::ostream &into, const std::complex<double> &what){
  into << "(" << what.real() << "," << what.imag() << ")";
}
template<>
inline void _print_T_sqmat(std::ostream &into, const std::complex<float> &what){
  into << "(" << what.real() << "," << what.imag() << ")";
}

template<typename T, int N>
std::ostream & operator<<(std::ostream &os, const CPSsquareMatrix<T,N> &m){
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      _print_T_sqmat(os, m(i,j));
      if(j<N-1) os << " ";
    }
    os << '\n';
  }
  return os;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Functions acting on CPSsquareMatrix

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
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, U>::type operator*(const typename U::scalar_type &a, const U &b){
  U into;
  into.zero();
  for(int i=0;i<U::Size;i++)
    for(int j=0;j<U::Size;j++)
      into(i,j) = a * b(i,j);
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
accelerator_inline typename my_enable_if<isCPSsquareMatrix<U>::value, U>::type operator-(const U &a){
  U into;
  for(int i=0;i<U::Size;i++)
    for(int j=0;j<U::Size;j++)
      into(i,j) = -a(i,j);
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

