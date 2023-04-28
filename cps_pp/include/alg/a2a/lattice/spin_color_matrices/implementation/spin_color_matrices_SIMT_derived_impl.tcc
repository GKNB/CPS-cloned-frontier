////////////////////////////////// GAMMA MULT ////////////////////////////
#define TIMESPLUSONE(a,b) { SIMT<ComplexType>::write(a, b, lane); }
#define TIMESMINUSONE(a,b) { SIMT<ComplexType>::write(a, -b, lane); }
#define TIMESPLUSI(a,b) { SIMT<ComplexType>::write(a, cps::timesI(b), lane); }
#define TIMESMINUSI(a,b) { SIMT<ComplexType>::write(a, cps::timesMinusI(b), lane); }
#define SETZERO(a){ SIMT<ComplexType>::write(a, typename SIMT<ComplexType>::value_type(0), lane); }
#define WR(A,B) SIMT<ComplexType>::write(A,B,lane)
#define RD(A) SIMT<ComplexType>::read(A,lane)

//Use iterator classes to write generic implementations for the different derived types
template<typename>
struct CPSsquareMatrixSpinIterator{};

template<typename ComplexType>
struct CPSsquareMatrixSpinIterator<CPSspinMatrix<ComplexType> >{
  int i;
  accelerator_inline CPSsquareMatrixSpinIterator(): i(0){}
  
  accelerator_inline CPSsquareMatrixSpinIterator& operator++(){
    ++i;
    return *this;
  }  
  accelerator_inline ComplexType & elem(CPSspinMatrix<ComplexType> &M, const int s1, const int s2){ return M(s1,s2); }
  accelerator_inline const ComplexType & elem(const CPSspinMatrix<ComplexType> &M, const int s1, const int s2){ return M(s1,s2); }

  accelerator_inline bool end() const{ return i==1; }

  //static versions that take a subindex that is ignored here
  static accelerator_inline ComplexType & elem(CPSspinMatrix<ComplexType> &M, const int s1, const int s2, int ccff){
    return M(s1,s2);
  }
  static accelerator_inline const ComplexType & elem(CPSspinMatrix<ComplexType> const& M, const int s1, const int s2, int ccff){
    return M(s1,s2);
  }

};

template<typename ComplexType>
struct CPSsquareMatrixSpinIterator<CPSspinColorFlavorMatrix<ComplexType> >{
  int c1,c2,f1,f2;
  int i;
  accelerator_inline CPSsquareMatrixSpinIterator(): i(0),c1(0),c2(0),f1(0),f2(0){}
  
  accelerator_inline CPSsquareMatrixSpinIterator& operator++(){
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

  //static versions that take a color*color*flavor*flavor compound offset 'ccff'
  static accelerator_inline ComplexType & elem(CPSspinColorFlavorMatrix<ComplexType> &M, const int s1, const int s2, int ccff){
    return *(M(s1,s2).scalarTypePtr() + ccff);
  }
  static accelerator_inline const ComplexType & elem(CPSspinColorFlavorMatrix<ComplexType> const& M, const int s1, const int s2, int ccff){
    return *(M(s1,s2).scalarTypePtr() + ccff);
  }

};

template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type>
accelerator_inline void gl(MatrixType &M, int dir, int lane){
  int s2, s1;
  typedef CPSsquareMatrixSpinIterator<MatrixType> MatrixIterator;
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





//Non in-place version 
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type>
accelerator_inline void gl_r(MatrixType &O, const MatrixType &M, int dir, int lane){
  int s2, s1;
  typedef CPSsquareMatrixSpinIterator<MatrixType> MatrixIterator;
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

//Version for specific spin column and flavor-color element
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type>
accelerator_inline void gl_r(MatrixType &O, const MatrixType &M, int dir, int s2, int ffcc, int lane){
  typedef CPSsquareMatrixSpinIterator<MatrixType> MatrixIterator;
  typedef typename MatrixType::scalar_type ComplexType;

  switch(dir){
  case 0:
    TIMESPLUSI(  MatrixIterator::elem(O,0,s2,ffcc), RD(MatrixIterator::elem(M,3,s2,ffcc)) );
    TIMESPLUSI(  MatrixIterator::elem(O,1,s2,ffcc), RD(MatrixIterator::elem(M,2,s2,ffcc)) );
    TIMESMINUSI( MatrixIterator::elem(O,2,s2,ffcc), RD(MatrixIterator::elem(M,1,s2,ffcc)) );
    TIMESMINUSI( MatrixIterator::elem(O,3,s2,ffcc), RD(MatrixIterator::elem(M,0,s2,ffcc)) );
    break;
  case 1:
    TIMESMINUSONE( MatrixIterator::elem(O,0,s2,ffcc), RD(MatrixIterator::elem(M,3,s2,ffcc)) );
    TIMESPLUSONE(  MatrixIterator::elem(O,1,s2,ffcc), RD(MatrixIterator::elem(M,2,s2,ffcc)) );
    TIMESPLUSONE(  MatrixIterator::elem(O,2,s2,ffcc), RD(MatrixIterator::elem(M,1,s2,ffcc)) );
    TIMESMINUSONE( MatrixIterator::elem(O,3,s2,ffcc), RD(MatrixIterator::elem(M,0,s2,ffcc)) );
    break;
  case 2:
    TIMESPLUSI(  MatrixIterator::elem(O,0,s2,ffcc), RD(MatrixIterator::elem(M,2,s2,ffcc)) );
    TIMESMINUSI( MatrixIterator::elem(O,1,s2,ffcc), RD(MatrixIterator::elem(M,3,s2,ffcc)) );
    TIMESMINUSI( MatrixIterator::elem(O,2,s2,ffcc), RD(MatrixIterator::elem(M,0,s2,ffcc)) );
    TIMESPLUSI(  MatrixIterator::elem(O,3,s2,ffcc), RD(MatrixIterator::elem(M,1,s2,ffcc)) );
    break;
  case 3:
    TIMESPLUSONE( MatrixIterator::elem(O,0,s2,ffcc), RD(MatrixIterator::elem(M,2,s2,ffcc)) );
    TIMESPLUSONE( MatrixIterator::elem(O,1,s2,ffcc), RD(MatrixIterator::elem(M,3,s2,ffcc)) );
    TIMESPLUSONE( MatrixIterator::elem(O,2,s2,ffcc), RD(MatrixIterator::elem(M,0,s2,ffcc)) );
    TIMESPLUSONE( MatrixIterator::elem(O,3,s2,ffcc), RD(MatrixIterator::elem(M,1,s2,ffcc)) );
    break;
  case -5:
    TIMESPLUSONE(  MatrixIterator::elem(O,0,s2,ffcc), RD(MatrixIterator::elem(M,0,s2,ffcc)) );
    TIMESPLUSONE(  MatrixIterator::elem(O,1,s2,ffcc), RD(MatrixIterator::elem(M,1,s2,ffcc)) );
    TIMESMINUSONE( MatrixIterator::elem(O,2,s2,ffcc), RD(MatrixIterator::elem(M,2,s2,ffcc)) );
    TIMESMINUSONE( MatrixIterator::elem(O,3,s2,ffcc), RD(MatrixIterator::elem(M,3,s2,ffcc)) );
    break;
  default:
    assert(0);
    break;
  }
}



template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type>
accelerator_inline void gr(MatrixType &M, int dir, int lane){
  int s2, s1;
  typedef CPSsquareMatrixSpinIterator<MatrixType> MatrixIterator;
  typedef typename MatrixType::scalar_type ComplexType;
  typename SIMT<ComplexType>::value_type tmp[4];
  MatrixIterator it;

  switch(dir){
  case 0:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	for(s2=0;s2<4;s2++) tmp[s2] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESMINUSI( it.elem(M,s1,0), tmp[3] );
	TIMESMINUSI( it.elem(M,s1,1), tmp[2] );
	TIMESPLUSI(  it.elem(M,s1,2), tmp[1] );
	TIMESPLUSI(  it.elem(M,s1,3), tmp[0] );
      }
      ++it;
    }
    break;
  case 1:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	for(s2=0;s2<4;s2++) tmp[s2] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESMINUSONE( it.elem(M,s1,0), tmp[3] );
	TIMESPLUSONE( it.elem(M,s1,1), tmp[2] );
	TIMESPLUSONE( it.elem(M,s1,2), tmp[1] );
	TIMESMINUSONE( it.elem(M,s1,3), tmp[0] );
      }
      ++it;      
    }
    break;
  case 2:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	for(s2=0;s2<4;s2++) tmp[s2] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESMINUSI( it.elem(M,s1,0), tmp[2] );
	TIMESPLUSI( it.elem(M,s1,1), tmp[3] );
	TIMESPLUSI( it.elem(M,s1,2), tmp[0] );
	TIMESMINUSI( it.elem(M,s1,3), tmp[1] );
      }
      ++it;
    }
    break;
  case 3:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	for(s2=0;s2<4;s2++) tmp[s2] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESPLUSONE( it.elem(M,s1,0), tmp[2] );
	TIMESPLUSONE( it.elem(M,s1,1), tmp[3] );
	TIMESPLUSONE( it.elem(M,s1,2), tmp[0] );
	TIMESPLUSONE( it.elem(M,s1,3), tmp[1] );
      }
      ++it;
    }
    break;
  case -5:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	for(s2=0;s2<4;s2++) tmp[s2] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESPLUSONE( it.elem(M,s1,0), tmp[0] );
	TIMESPLUSONE( it.elem(M,s1,1), tmp[1] );
	TIMESMINUSONE( it.elem(M,s1,2), tmp[2] );
	TIMESMINUSONE( it.elem(M,s1,3), tmp[3] );
      }
      ++it;
    }
    break;
  default:
    assert(0);
    break;
  }
}




template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type>
accelerator_inline void gr_r(MatrixType &O, const MatrixType &M, int dir, int lane){
  int s2, s1;
  typedef CPSsquareMatrixSpinIterator<MatrixType> MatrixIterator;
  typedef typename MatrixType::scalar_type ComplexType;
  MatrixIterator it;

  switch(dir){
  case 0:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	TIMESMINUSI( it.elem(O,s1,0), RD(it.elem(M,s1,3)) );
	TIMESMINUSI( it.elem(O,s1,1), RD(it.elem(M,s1,2)) );
	TIMESPLUSI(  it.elem(O,s1,2), RD(it.elem(M,s1,1)) );
	TIMESPLUSI(  it.elem(O,s1,3), RD(it.elem(M,s1,0)) );
      }
      ++it;
    }
    break;
  case 1:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	TIMESMINUSONE( it.elem(O,s1,0), RD(it.elem(M,s1,3)) );
	TIMESPLUSONE( it.elem(O,s1,1), RD(it.elem(M,s1,2)) );
	TIMESPLUSONE( it.elem(O,s1,2), RD(it.elem(M,s1,1)) );
	TIMESMINUSONE( it.elem(O,s1,3), RD(it.elem(M,s1,0)) );
      }
      ++it;      
    }
    break;
  case 2:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	TIMESMINUSI( it.elem(O,s1,0), RD(it.elem(M,s1,2)) );
	TIMESPLUSI( it.elem(O,s1,1), RD(it.elem(M,s1,3)) );
	TIMESPLUSI( it.elem(O,s1,2), RD(it.elem(M,s1,0)) );
	TIMESMINUSI( it.elem(O,s1,3), RD(it.elem(M,s1,1)) );
      }
      ++it;
    }
    break;
  case 3:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	TIMESPLUSONE( it.elem(O,s1,0), RD(it.elem(M,s1,2)) );
	TIMESPLUSONE( it.elem(O,s1,1), RD(it.elem(M,s1,3)) );
	TIMESPLUSONE( it.elem(O,s1,2), RD(it.elem(M,s1,0)) );
	TIMESPLUSONE( it.elem(O,s1,3), RD(it.elem(M,s1,1)) );
      }
      ++it;
    }
    break;
  case -5:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	TIMESPLUSONE( it.elem(O,s1,0), RD(it.elem(M,s1,0)) );
	TIMESPLUSONE( it.elem(O,s1,1), RD(it.elem(M,s1,1)) );
	TIMESMINUSONE( it.elem(O,s1,2), RD(it.elem(M,s1,2)) );
	TIMESMINUSONE( it.elem(O,s1,3), RD(it.elem(M,s1,3)) );
      }
      ++it;
    }
    break;
  default:
    assert(0);
    break;
  }
}

template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type>
accelerator_inline void gr_r(MatrixType &O, const MatrixType &M, int dir, int s1, int ffcc, int lane){
  typedef CPSsquareMatrixSpinIterator<MatrixType> MatrixIterator;
  typedef typename MatrixType::scalar_type ComplexType;

  switch(dir){
  case 0:
    TIMESMINUSI( MatrixIterator::elem(O,s1,0,ffcc), RD(MatrixIterator::elem(M,s1,3,ffcc)) );
    TIMESMINUSI( MatrixIterator::elem(O,s1,1,ffcc), RD(MatrixIterator::elem(M,s1,2,ffcc)) );
    TIMESPLUSI(  MatrixIterator::elem(O,s1,2,ffcc), RD(MatrixIterator::elem(M,s1,1,ffcc)) );
    TIMESPLUSI(  MatrixIterator::elem(O,s1,3,ffcc), RD(MatrixIterator::elem(M,s1,0,ffcc)) );    
    break;
  case 1:
    TIMESMINUSONE( MatrixIterator::elem(O,s1,0,ffcc), RD(MatrixIterator::elem(M,s1,3,ffcc)) );
    TIMESPLUSONE( MatrixIterator::elem(O,s1,1,ffcc), RD(MatrixIterator::elem(M,s1,2,ffcc)) );
    TIMESPLUSONE( MatrixIterator::elem(O,s1,2,ffcc), RD(MatrixIterator::elem(M,s1,1,ffcc)) );
    TIMESMINUSONE( MatrixIterator::elem(O,s1,3,ffcc), RD(MatrixIterator::elem(M,s1,0,ffcc)) );
    break;
  case 2:
    TIMESMINUSI( MatrixIterator::elem(O,s1,0,ffcc), RD(MatrixIterator::elem(M,s1,2,ffcc)) );
    TIMESPLUSI( MatrixIterator::elem(O,s1,1,ffcc), RD(MatrixIterator::elem(M,s1,3,ffcc)) );
    TIMESPLUSI( MatrixIterator::elem(O,s1,2,ffcc), RD(MatrixIterator::elem(M,s1,0,ffcc)) );
    TIMESMINUSI( MatrixIterator::elem(O,s1,3,ffcc), RD(MatrixIterator::elem(M,s1,1,ffcc)) );
    break;
  case 3:
    TIMESPLUSONE( MatrixIterator::elem(O,s1,0,ffcc), RD(MatrixIterator::elem(M,s1,2,ffcc)) );
    TIMESPLUSONE( MatrixIterator::elem(O,s1,1,ffcc), RD(MatrixIterator::elem(M,s1,3,ffcc)) );
    TIMESPLUSONE( MatrixIterator::elem(O,s1,2,ffcc), RD(MatrixIterator::elem(M,s1,0,ffcc)) );
    TIMESPLUSONE( MatrixIterator::elem(O,s1,3,ffcc), RD(MatrixIterator::elem(M,s1,1,ffcc)) );
    break;
  case -5:
    TIMESPLUSONE( MatrixIterator::elem(O,s1,0,ffcc), RD(MatrixIterator::elem(M,s1,0,ffcc)) );
    TIMESPLUSONE( MatrixIterator::elem(O,s1,1,ffcc), RD(MatrixIterator::elem(M,s1,1,ffcc)) );
    TIMESMINUSONE( MatrixIterator::elem(O,s1,2,ffcc), RD(MatrixIterator::elem(M,s1,2,ffcc)) );
    TIMESMINUSONE( MatrixIterator::elem(O,s1,3,ffcc), RD(MatrixIterator::elem(M,s1,3,ffcc)) );
    break;
  default:
    assert(0);
    break;
  }
}




//multiply gamma(i)gamma(5) on the left: result = gamma(i)*gamma(5)*fro
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type>
accelerator_inline void glAx(MatrixType &M, int dir, int lane){
  int s2, s1;
  typedef CPSsquareMatrixSpinIterator<MatrixType> MatrixIterator;
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
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type>
accelerator_inline void glAx_r(MatrixType &O, const MatrixType &M, int dir, int lane){
  int s2, s1;
  typedef CPSsquareMatrixSpinIterator<MatrixType> MatrixIterator;
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

//Version for specific spin column and flavor-color element
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type>
accelerator_inline void glAx_r(MatrixType &O, const MatrixType &M, int dir, int s2, int ffcc, int lane){
  typedef CPSsquareMatrixSpinIterator<MatrixType> MatrixIterator;
  typedef typename MatrixType::scalar_type ComplexType;

  switch(dir){
  case 0:
    TIMESMINUSI( MatrixIterator::elem(O,0,s2,ffcc), RD(MatrixIterator::elem(M,3,s2,ffcc)) );
    TIMESMINUSI( MatrixIterator::elem(O,1,s2,ffcc), RD(MatrixIterator::elem(M,2,s2,ffcc)) );
    TIMESMINUSI( MatrixIterator::elem(O,2,s2,ffcc), RD(MatrixIterator::elem(M,1,s2,ffcc)) );
    TIMESMINUSI( MatrixIterator::elem(O,3,s2,ffcc), RD(MatrixIterator::elem(M,0,s2,ffcc)) );
    break;
  case 1:
    TIMESPLUSONE(  MatrixIterator::elem(O,0,s2,ffcc), RD(MatrixIterator::elem(M,3,s2,ffcc)) );
    TIMESMINUSONE( MatrixIterator::elem(O,1,s2,ffcc), RD(MatrixIterator::elem(M,2,s2,ffcc)) );
    TIMESPLUSONE(  MatrixIterator::elem(O,2,s2,ffcc), RD(MatrixIterator::elem(M,1,s2,ffcc)) );
    TIMESMINUSONE( MatrixIterator::elem(O,3,s2,ffcc), RD(MatrixIterator::elem(M,0,s2,ffcc)) );
    break;
  case 2:
    TIMESMINUSI( MatrixIterator::elem(O,0,s2,ffcc), RD(MatrixIterator::elem(M,2,s2,ffcc)) );
    TIMESPLUSI( MatrixIterator::elem(O,1,s2,ffcc), RD(MatrixIterator::elem(M,3,s2,ffcc)) );
    TIMESMINUSI( MatrixIterator::elem(O,2,s2,ffcc), RD(MatrixIterator::elem(M,0,s2,ffcc)) );
    TIMESPLUSI( MatrixIterator::elem(O,3,s2,ffcc), RD(MatrixIterator::elem(M,1,s2,ffcc)) );
    break;
  case 3:
    TIMESMINUSONE( MatrixIterator::elem(O,0,s2,ffcc), RD(MatrixIterator::elem(M,2,s2,ffcc)) );
    TIMESMINUSONE( MatrixIterator::elem(O,1,s2,ffcc), RD(MatrixIterator::elem(M,3,s2,ffcc)) );
    TIMESPLUSONE( MatrixIterator::elem(O,2,s2,ffcc), RD(MatrixIterator::elem(M,0,s2,ffcc)) );
    TIMESPLUSONE( MatrixIterator::elem(O,3,s2,ffcc), RD(MatrixIterator::elem(M,1,s2,ffcc)) );
    break;
  default:
    assert(0);
    break;
  }
}

//multiply gamma(i)gamma(5) on the left: result = gamma(i)*gamma(5)*fro
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type>
accelerator_inline void grAx(MatrixType &M, int dir, int lane){
  int s2, s1;
  typedef CPSsquareMatrixSpinIterator<MatrixType> MatrixIterator;
  typedef typename MatrixType::scalar_type ComplexType;
  typename SIMT<ComplexType>::value_type tmp[4];
  MatrixIterator it;

  switch(dir){
  case 0:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	for(s2=0;s2<4;s2++) tmp[s2] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESMINUSI( it.elem(M,s1,0), tmp[3] );
	TIMESMINUSI( it.elem(M,s1,1), tmp[2]  );
	TIMESMINUSI( it.elem(M,s1,2), tmp[1] );
	TIMESMINUSI( it.elem(M,s1,3), tmp[0] );
      }
      ++it;
    }
    break;
  case 1:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	for(s2=0;s2<4;s2++) tmp[s2] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESMINUSONE( it.elem(M,s1,0), tmp[3] );
	TIMESPLUSONE( it.elem(M,s1,1), tmp[2] );
	TIMESMINUSONE( it.elem(M,s1,2), tmp[1]  );
	TIMESPLUSONE( it.elem(M,s1,3), tmp[0] );
      }
      ++it;
    }
    break;
  case 2:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	for(s2=0;s2<4;s2++) tmp[s2] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESMINUSI( it.elem(M,s1,0), tmp[2] );
	TIMESPLUSI( it.elem(M,s1,1), tmp[3] );
	TIMESMINUSI( it.elem(M,s1,2), tmp[0] );
	TIMESPLUSI( it.elem(M,s1,3), tmp[1] );
      }
      ++it;
    }
    break;
  case 3:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	for(s2=0;s2<4;s2++) tmp[s2] = SIMT<ComplexType>::read(it.elem(M,s1,s2),lane);	
	TIMESPLUSONE( it.elem(M,s1,0), tmp[2] );
	TIMESPLUSONE( it.elem(M,s1,1), tmp[3] );
	TIMESMINUSONE( it.elem(M,s1,2), tmp[0] );
	TIMESMINUSONE( it.elem(M,s1,3), tmp[1] );
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
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type>
accelerator_inline void grAx_r(MatrixType &O, const MatrixType &M, int dir, int lane){
  int s2, s1;
  typedef CPSsquareMatrixSpinIterator<MatrixType> MatrixIterator;
  typedef typename MatrixType::scalar_type ComplexType;
  MatrixIterator it;

  switch(dir){
  case 0:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	TIMESMINUSI( it.elem(O,s1,0), RD(it.elem(M,s1,3)) );
	TIMESMINUSI( it.elem(O,s1,1), RD(it.elem(M,s1,2))  );
	TIMESMINUSI( it.elem(O,s1,2), RD(it.elem(M,s1,1)) );
	TIMESMINUSI( it.elem(O,s1,3), RD(it.elem(M,s1,0)) );
      }
      ++it;
    }
    break;
  case 1:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	TIMESMINUSONE( it.elem(O,s1,0), RD(it.elem(M,s1,3)) );
	TIMESPLUSONE( it.elem(O,s1,1), RD(it.elem(M,s1,2)) );
	TIMESMINUSONE( it.elem(O,s1,2), RD(it.elem(M,s1,1))  );
	TIMESPLUSONE( it.elem(O,s1,3), RD(it.elem(M,s1,0)) );
      }
      ++it;
    }
    break;
  case 2:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	TIMESMINUSI( it.elem(O,s1,0), RD(it.elem(M,s1,2)) );
	TIMESPLUSI( it.elem(O,s1,1), RD(it.elem(M,s1,3)) );
	TIMESMINUSI( it.elem(O,s1,2), RD(it.elem(M,s1,0)) );
	TIMESPLUSI( it.elem(O,s1,3), RD(it.elem(M,s1,1)) );
      }
      ++it;
    }
    break;
  case 3:
    while(!it.end()){
      for(s1=0;s1<4;s1++){
	TIMESPLUSONE( it.elem(O,s1,0), RD(it.elem(M,s1,2)) );
	TIMESPLUSONE( it.elem(O,s1,1), RD(it.elem(M,s1,3)) );
	TIMESMINUSONE( it.elem(O,s1,2), RD(it.elem(M,s1,0)) );
	TIMESMINUSONE( it.elem(O,s1,3), RD(it.elem(M,s1,1)) );
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
template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type>
accelerator_inline void grAx_r(MatrixType &O, const MatrixType &M, int dir, int s1, int ffcc, int lane){
  typedef CPSsquareMatrixSpinIterator<MatrixType> MatrixIterator;
  typedef typename MatrixType::scalar_type ComplexType;

  switch(dir){
  case 0:
    TIMESMINUSI( MatrixIterator::elem(O,s1,0,ffcc), RD(MatrixIterator::elem(M,s1,3,ffcc)) );
    TIMESMINUSI( MatrixIterator::elem(O,s1,1,ffcc), RD(MatrixIterator::elem(M,s1,2,ffcc))  );
    TIMESMINUSI( MatrixIterator::elem(O,s1,2,ffcc), RD(MatrixIterator::elem(M,s1,1,ffcc)) );
    TIMESMINUSI( MatrixIterator::elem(O,s1,3,ffcc), RD(MatrixIterator::elem(M,s1,0,ffcc)) );
    break;
  case 1:
    TIMESMINUSONE( MatrixIterator::elem(O,s1,0,ffcc), RD(MatrixIterator::elem(M,s1,3,ffcc)) );
    TIMESPLUSONE( MatrixIterator::elem(O,s1,1,ffcc), RD(MatrixIterator::elem(M,s1,2,ffcc)) );
    TIMESMINUSONE( MatrixIterator::elem(O,s1,2,ffcc), RD(MatrixIterator::elem(M,s1,1,ffcc))  );
    TIMESPLUSONE( MatrixIterator::elem(O,s1,3,ffcc), RD(MatrixIterator::elem(M,s1,0,ffcc)) );
    break;
  case 2:
    TIMESMINUSI( MatrixIterator::elem(O,s1,0,ffcc), RD(MatrixIterator::elem(M,s1,2,ffcc)) );
    TIMESPLUSI( MatrixIterator::elem(O,s1,1,ffcc), RD(MatrixIterator::elem(M,s1,3,ffcc)) );
    TIMESMINUSI( MatrixIterator::elem(O,s1,2,ffcc), RD(MatrixIterator::elem(M,s1,0,ffcc)) );
    TIMESPLUSI( MatrixIterator::elem(O,s1,3,ffcc), RD(MatrixIterator::elem(M,s1,1,ffcc)) );
    break;
  case 3:
    TIMESPLUSONE( MatrixIterator::elem(O,s1,0,ffcc), RD(MatrixIterator::elem(M,s1,2,ffcc)) );
    TIMESPLUSONE( MatrixIterator::elem(O,s1,1,ffcc), RD(MatrixIterator::elem(M,s1,3,ffcc)) );
    TIMESMINUSONE( MatrixIterator::elem(O,s1,2,ffcc), RD(MatrixIterator::elem(M,s1,0,ffcc)) );
    TIMESMINUSONE( MatrixIterator::elem(O,s1,3,ffcc), RD(MatrixIterator::elem(M,s1,1,ffcc)) );
    break;
  default:
    assert(0);
    break;
  }
}



//////////////////// FLAVOR MULT ///////////////////////////

template<typename MatrixType>
struct CPSsquareMatrixFlavorIterator{};

template<typename ComplexType>
struct CPSsquareMatrixFlavorIterator<CPSflavorMatrix<ComplexType> >{
  int i;
  accelerator_inline CPSsquareMatrixFlavorIterator(): i(0){}
  
  accelerator_inline CPSsquareMatrixFlavorIterator& operator++(){
    ++i;
    return *this;
  }  
  accelerator_inline ComplexType & elem(CPSflavorMatrix<ComplexType> &M, const int f1, const int f2){ return M(f1,f2); }
  accelerator_inline const ComplexType & elem(const CPSflavorMatrix<ComplexType> &M, const int f1, const int f2){ return M(f1,f2); }

  accelerator_inline bool end() const{ return i==1; }
};

template<typename ComplexType>
struct CPSsquareMatrixFlavorIterator<CPSspinColorFlavorMatrix<ComplexType> >{
  int s1,s2,c1,c2;
  int i;
  accelerator_inline CPSsquareMatrixFlavorIterator(): i(0),s1(0),s2(0),c1(0),c2(0){}
  
  accelerator_inline CPSsquareMatrixFlavorIterator& operator++(){
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



template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type>
accelerator_inline void pl(MatrixType &M, const FlavorMatrixType type, int lane){
  typedef CPSsquareMatrixFlavorIterator<MatrixType> MatrixIterator;
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


template<typename MatrixType, typename std::enable_if<isCPSsquareMatrix<MatrixType>::value, int>::type>
accelerator_inline void pr(MatrixType &M, const FlavorMatrixType type, int lane){
  typedef CPSsquareMatrixFlavorIterator<MatrixType> MatrixIterator;
  typedef typename MatrixType::scalar_type ComplexType;
  typename SIMT<ComplexType>::value_type tmp1, tmp2;
  MatrixIterator it;

  switch(type){    
  case F0:   
    while(!it.end()){
      SETZERO(it.elem(M,0,1));
      SETZERO(it.elem(M,1,1));
      ++it;
    }
    break;
  case F1:
    while(!it.end()){
      SETZERO(it.elem(M,0,0) );
      SETZERO(it.elem(M,1,0) );
      ++it;
    }
    break;
  case Fud:
    while(!it.end()){
      tmp1 = RD(it.elem(M,0,0));
      tmp2 = RD(it.elem(M,1,0));
      WR(it.elem(M,0,0), RD(it.elem(M,0,1) ) );
      WR(it.elem(M,1,0), RD(it.elem(M,1,1) ) );
      WR(it.elem(M,0,1), tmp1);
      WR(it.elem(M,1,1), tmp2);
      ++it;
    }
    break;
  case sigma0:
    break;
  case sigma1:
    while(!it.end()){
      tmp1 = RD(it.elem(M,0,0));
      tmp2 = RD(it.elem(M,1,0));
      WR(it.elem(M,0,0), RD(it.elem(M,0,1)) );
      WR(it.elem(M,1,0), RD(it.elem(M,1,1)) );
      WR(it.elem(M,0,1), tmp1);
      WR(it.elem(M,1,1), tmp2);
      ++it;
    }
    break;      
  case sigma2:
    while(!it.end()){
      tmp1 = RD(it.elem(M,0,0));
      tmp2 = RD(it.elem(M,1,0));
      TIMESPLUSI(it.elem(M,0,0), RD(it.elem(M,0,1)) );
      TIMESPLUSI(it.elem(M,1,0), RD(it.elem(M,1,1)) );
      TIMESMINUSI(it.elem(M,0,1), tmp1);
      TIMESMINUSI(it.elem(M,1,1), tmp2);
      ++it;
    }
    break;
  case sigma3:
    while(!it.end()){
      TIMESMINUSONE(it.elem(M,0,1), RD(it.elem(M,0,1)) );
      TIMESMINUSONE(it.elem(M,1,1), RD(it.elem(M,1,1)) );
      ++it;
    }
    break;
  default:
    //ERR.General("FlavorMatrixGeneral","pr(const FlavorMatrixGeneralType &type)","Unknown FlavorMatrixGeneralType");
    assert(0);
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
