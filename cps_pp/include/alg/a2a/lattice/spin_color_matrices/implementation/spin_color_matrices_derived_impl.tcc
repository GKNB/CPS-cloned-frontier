//Old annoying CPS conventions with output on RHS (Fortran user or something??)
#define TIMESPLUSONE(a,b) { b=a; }
#define TIMESMINUSONE(a,b) { _timespmI<T, typename ClassifyMatrixOrNotMatrix<T>::type>::timesMinusOne(b,a); }
#define TIMESPLUSI(a,b) { _timespmI<T, typename ClassifyMatrixOrNotMatrix<T>::type>::timesI(b,a); }
#define TIMESMINUSI(a,b) { _timespmI<T, typename ClassifyMatrixOrNotMatrix<T>::type>::timesMinusI(b,a); }
#define SETZERO(a){ _CPSsetZeroOne<T, typename ClassifyMatrixOrNotMatrix<T>::type>::setzero(a); }

template<typename T>
accelerator_inline CPSflavorMatrix<T> & CPSflavorMatrix<T>::pl(const FlavorMatrixType &type){
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
template<typename T>
accelerator_inline CPSflavorMatrix<T> & CPSflavorMatrix<T>::pr(const FlavorMatrixType &type){
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



//Left Multiplication by Dirac gamma's
template<typename T>
accelerator_inline CPSspinMatrix<T> & CPSspinMatrix<T>::gl(int dir){
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
template<typename T>
accelerator_inline CPSspinMatrix<T>& CPSspinMatrix<T>::gr(int dir)
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
template<typename T>
accelerator_inline CPSspinMatrix<T>& CPSspinMatrix<T>::glAx(const int dir){
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
template<typename T>
accelerator_inline CPSspinMatrix<T>& CPSspinMatrix<T>::grAx(int dir)
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

#undef TIMESPLUSONE
#undef TIMESMINUSONE
#undef TIMESPLUSI
#undef TIMESMINUSI
#undef SETZERO


template<typename ComplexType>
accelerator_inline CPScolorMatrix<ComplexType> CPSspinColorFlavorMatrix<ComplexType>::SpinFlavorTrace() const{
  return this->CPSsquareMatrix<value_type,4>::template TraceTwoIndices<0,2>();
}
template<typename ComplexType>
accelerator_inline CPSspinMatrix<CPSflavorMatrix<ComplexType> > CPSspinColorFlavorMatrix<ComplexType>::ColorTrace() const{
  return this->CPSsquareMatrix<value_type,4>::template TraceIndex<1>();
}
template<typename ComplexType>
accelerator_inline CPSspinColorFlavorMatrix<ComplexType> CPSspinColorFlavorMatrix<ComplexType>::TransposeColor() const{
  return CPSspinColorFlavorMatrix<ComplexType>(this->CPSsquareMatrix<value_type,4>::template TransposeOnIndex<1>());
}
template<typename ComplexType>
accelerator_inline void CPSspinColorFlavorMatrix<ComplexType>::equalsColorTranspose(const CPSspinColorFlavorMatrix<ComplexType> &r){
  this->CPSsquareMatrix<value_type,4>::template equalsTransposeOnIndex<1>(r);
}
  
//multiply on left by a flavor matrix
template<typename ComplexType>
accelerator_inline CPSspinColorFlavorMatrix<ComplexType> & CPSspinColorFlavorMatrix<ComplexType>::pl(const FlavorMatrixType type){
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  this->operator()(s1,s2)(c1,c2).pl(type);
  return *this;
}
//multiply on left by a flavor matrix
template<typename ComplexType>
accelerator_inline CPSspinColorFlavorMatrix<ComplexType> & CPSspinColorFlavorMatrix<ComplexType>::pr(const FlavorMatrixType type){
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  this->operator()(s1,s2)(c1,c2).pr(type);
  return *this;
}

template<typename ComplexType>
accelerator_inline CPScolorMatrix<ComplexType> CPSspinColorMatrix<ComplexType>::SpinTrace() const{
  return this->CPSsquareMatrix<value_type,4>::template TraceIndex<0>();
}
template<typename ComplexType>
accelerator_inline CPSspinMatrix<ComplexType> CPSspinColorMatrix<ComplexType>::ColorTrace() const{
  return this->CPSsquareMatrix<value_type,4>::template TraceIndex<1>();
}
template<typename ComplexType>
accelerator_inline CPSspinColorMatrix<ComplexType> CPSspinColorMatrix<ComplexType>::TransposeColor() const{
  return CPSspinColorMatrix<ComplexType>(this->CPSsquareMatrix<value_type,4>::template TransposeOnIndex<1>());
}
template<typename ComplexType>
accelerator_inline void CPSspinColorMatrix<ComplexType>::equalsColorTranspose(const CPSspinColorMatrix<ComplexType> &r){
  this->CPSsquareMatrix<value_type,4>::template equalsTransposeOnIndex<1>(r);
}  
