#include "essl_interface.h"
#include<iostream>

using namespace essl_interface;

//Print row-major matrix
template<typename T>
void printMatrix(T const* a, int n, int m){
  for(int i=0;i<n;i++){
    for(int j=0;j<m;j++){
      std::cout << a[j + m*i] << " ";
    }
    std::cout << std::endl;
  }
}

template<typename T>
std::ostream & operator<<(std::ostream &os, const std::complex<T> &M){
  os << "(" << M.real() << ", " << M.imag() << ")"; return os;
}


template<typename T>
struct MatrixWrapper{
  T* v;
  int n;
  int m;
  bool own_mem;

  MatrixWrapper(T* v, int n, int m): v(v), n(n), m(m), own_mem(false){}

  MatrixWrapper(int n, int m): v(v), n(n), m(m), own_mem(true){
    v = (T*)malloc(size_t(n)*size_t(m)*sizeof(T));    
  }
  
  ~MatrixWrapper(){ if(own_mem) free(v); }
  
  T& operator()(const int i, const int j){ return v[j + m*i]; }
};

template<typename T>
std::ostream & operator<<(std::ostream &os, const MatrixWrapper<T> &M){
  for(int i=0;i<M.n;i++){
    for(int j=0;j<M.m;j++){
      std::cout << M(i,j) << " ";
    }
    std::cout << std::endl;
  }  
  return os;
}


//c = l*n   a=l*m (m*l if transpose)  b=m*n  (n*m if transpose)
template<typename T>
void testMul(T* c, T* a, T* b, const int l, const int m, const int n, bool trans_A, bool trans_B){
  MatrixWrapper<T> C(c,l,n);
  MatrixWrapper<T> A(a,trans_A ? m : l, trans_A ? l : m);
  MatrixWrapper<T> B(b,trans_B ? n : m, trans_B ? m : n);

  for(int i=0;i<l;i++){
    for(int k=0;k<n;k++){
      C(i,k) = 0.;
      for(int j=0;j<m;j++)
	C(i,k) += ( trans_A ? A(j,i) : A(i,j) ) * ( trans_B ? B(k,j) : B(j,k) );
    }
  }
}

int main(void){
  //Matrix must be in column-major order, ie.  i = row + nrows*col

  {
    float a_cm[2*3] = { 1, 2, 3, 
			4, 5, 6};

    float b_cm[3*2] = { 1, 2, 
			3, 4, 
			5, 6};

    float c_cm_test[2*2];
    testMul(c_cm_test, a_cm, b_cm, 2, 3, 2, false, false);
  
    float c_cm[2*2];
    essl_gemul_rowmajor(c_cm, a_cm, b_cm, 2, 3, 2, false, false);

    std::cout << "A:\n";
    printMatrix(a_cm, 2,3);

    std::cout << "B:\n";
    printMatrix(b_cm, 3,2);

    std::cout << "C=A*B expect:\n";
    printMatrix(c_cm_test, 2,2);

    std::cout << "C=A*B got:\n";
    printMatrix(c_cm, 2,2);
  }


  {
    typedef std::complex<float> T;

    T a_cm[2*3] = { 1, 2, 3, 
		    4, 5, 6};

    T b_cm[3*2] = { 1, 2, 
		    3, 4, 
		    5, 6};

    T c_cm_test[2*2];
    testMul(c_cm_test, b_cm, a_cm, 2, 3, 2, true, true);
  
    T c_cm[2*2];
    essl_gemul_rowmajor(c_cm, b_cm, a_cm, 2, 3, 2, true, true);

    std::cout << "A:\n";
    printMatrix(a_cm, 2,3);

    std::cout << "B:\n";
    printMatrix(b_cm, 3,2);

    std::cout << "C=B^T*A^T expect:\n";
    printMatrix(c_cm_test, 2,2);

    std::cout << "C=B^T*A^T got:\n";
    printMatrix(c_cm, 2,2);
  }


  



  return 0;
}
