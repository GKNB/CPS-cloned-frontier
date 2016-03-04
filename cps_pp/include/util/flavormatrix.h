#ifndef _FLAVOR_MATRIX_H
#define _FLAVOR_MATRIX_H
CPS_START_NAMESPACE

enum FlavorMatrixType {F0, F1, Fud, sigma0, sigma1, sigma2, sigma3};

//Added by CK, optimized from Daiqian's equivalent
class FlavorMatrix{
	// The matrix is like:
	//
	//  fmat[0]  fmat[1]
	//
	//  fmat[2]  fmat[3]
protected:
  Complex fmat[2][2];
public:
  FlavorMatrix(){
  }
  FlavorMatrix(const FlavorMatrix &from){
    for(int i = 0 ; i < 2; i++)
      for(int j = 0 ; j < 2; j++)
	fmat[i][j] = from.fmat[i][j];
  }

  FlavorMatrix(Float rhs){
    for(int i = 0 ; i < 2; i++)
      for(int j = 0 ; j < 2; j++)
	fmat[i][j] = rhs;
  }

  void zero(){
    for(int i = 0 ; i < 2; i++)
      for(int j = 0 ; j < 2; j++)
	fmat[i][j] = 0.0;
  }

  FlavorMatrix& operator=(const FlavorMatrix &from){
    for(int i = 0 ; i < 2; i++)
      for(int j = 0 ; j < 2; j++)
	fmat[i][j] = from.fmat[i][j];
    return *this;
  }

  inline Complex &operator()(int f_row,int f_col){
    return fmat[f_row][f_col];
  }
  inline const Complex &operator()(int f_row,int f_col) const{
    return fmat[f_row][f_col];
  }

  FlavorMatrix & operator+=(const FlavorMatrix &r){
    for(int i = 0 ; i < 2; i++)
      for(int j = 0 ; j < 2; j++)
	fmat[i][j] += r.fmat[i][j];
    return *this;
  }

  FlavorMatrix & operator*=(const Float &rhs){ 
    for(int i = 0 ; i < 2; i++)
      for(int j = 0 ; j < 2; j++)
	fmat[i][j] *= rhs;
    return *this;
  }

  FlavorMatrix operator*(const FlavorMatrix& rhs) const{
    FlavorMatrix out;
    out.fmat[0][0] = fmat[0][0]*rhs.fmat[0][0] + fmat[0][1]*rhs.fmat[1][0];
    out.fmat[0][1] = fmat[0][0]*rhs.fmat[0][1] + fmat[0][1]*rhs.fmat[1][1];
    out.fmat[1][0] = fmat[1][0]*rhs.fmat[0][0] + fmat[1][1]*rhs.fmat[1][0];
    out.fmat[1][1] = fmat[1][0]*rhs.fmat[0][1] + fmat[1][1]*rhs.fmat[1][1];
    return out;
  }

  FlavorMatrix transpose() const{
    FlavorMatrix out;
    out.fmat[0][0] = fmat[0][0];
    out.fmat[0][1] = fmat[1][0];
    out.fmat[1][0] = fmat[0][1];
    out.fmat[1][1] = fmat[1][1];
    return out;
  }

  Complex Trace() const{
    return fmat[0][0] + fmat[1][1];
  }

  //Added by CK
  //multiply on left by a flavor matrix
  FlavorMatrix & pl(const FlavorMatrixType &type){
    Complex tmp1, tmp2;

    switch( type ){
    case F0:
      (*this)(1,0) = 0.0;
      (*this)(1,1) = 0.0;
      break;
    case F1:
      (*this)(0,0) = 0.0;
      (*this)(0,1) = 0.0;
      break;
    case Fud:
      tmp1 = (*this)(0,0);
      tmp2 = (*this)(0,1);
      (*this)(0,0) = (*this)(1,0);
      (*this)(0,1) = (*this)(1,1);
      (*this)(1,0) = tmp1;
      (*this)(1,1) = tmp2;
      break;
    case sigma0:
      break;
    case sigma1:
      tmp1 = (*this)(0,0);
      tmp2 = (*this)(0,1);
      (*this)(0,0) = (*this)(1,0);
      (*this)(0,1) = (*this)(1,1);
      (*this)(1,0) = tmp1;
      (*this)(1,1) = tmp2;
      break;      
    case sigma2:
      tmp1 = (*this)(0,0)*Complex(0.0,1.0);
      tmp2 = (*this)(0,1)*Complex(0.0,1.0);
      (*this)(0,0) = (*this)(1,0)*Complex(0.0,-1.0);
      (*this)(0,1) = (*this)(1,1)*Complex(0.0,-1.0);
      (*this)(1,0) = tmp1;
      (*this)(1,1) = tmp2;
      break;
    case sigma3:
      (*this)(1,0)*=-1.0;
      (*this)(1,1)*=-1.0;
      break;
    default:
      ERR.General("FlavorMatrix","pl(const FlavorMatrixType &type)","Unknown FlavorMatrixType");
      break;
    }
    return *this;

  }

  //multiply on right by a flavor matrix
  FlavorMatrix & pr(const FlavorMatrixType &type){
    Complex tmp1, tmp2;

    switch(type){
    case F0:
      (*this)(0,1) = 0.0;
      (*this)(1,1) = 0.0;
      break;
    case F1:
      (*this)(0,0) = 0.0;
      (*this)(1,0) = 0.0;
      break;
    case Fud:
      tmp1 = (*this)(0,0);
      tmp2 = (*this)(1,0);
      (*this)(0,0) = (*this)(0,1);
      (*this)(1,0) = (*this)(1,1);
      (*this)(0,1) = tmp1;
      (*this)(1,1) = tmp2;
      break;
    case sigma0:
      break;
    case sigma1:
      tmp1 = (*this)(0,0);
      tmp2 = (*this)(1,0);
      (*this)(0,0) = (*this)(0,1);
      (*this)(1,0) = (*this)(1,1);
      (*this)(0,1) = tmp1;
      (*this)(1,1) = tmp2;
      break;      
    case sigma2:
      tmp1 = (*this)(0,0) *  Complex(0.0,-1.0);
      tmp2 = (*this)(1,0) *  Complex(0.0,-1.0);
      (*this)(0,0) = (*this)(0,1)* Complex(0.0,1.0); 
      (*this)(1,0) = (*this)(1,1)* Complex(0.0,1.0);
      (*this)(0,1) = tmp1;
      (*this)(1,1) = tmp2;
      break;
    case sigma3:
      (*this)(0,1)*=-1.0;
      (*this)(1,1)*=-1.0;
      break;
    default:
      ERR.General("FlavorMatrix","pr(const FlavorMatrixType &type)","Unknown FlavorMatrixType");
      break;
    }
    return *this;
  }
};
//Trace(A * B);
inline static Complex Trace(const FlavorMatrix &a, const FlavorMatrix &b){
  return a(0,0)*b(0,0) + a(0,1)*b(1,0) + a(1,0)*b(0,1) + a(1,1)*b(1,1);
}
//Trace(A^T * B)
inline static Complex TransLeftTrace(const FlavorMatrix &a, const FlavorMatrix &b){
  return a(0,0)*b(0,0) + a(1,0)*b(1,0) + a(0,1)*b(0,1) + a(1,1)*b(1,1);
}


CPS_END_NAMESPACE
#endif
