#include<complex>
#include<cassert>
#include<cstdlib>

namespace essl_interface{

  //ESSL matrices must be in column major (i = row + nrows*col) vs the usual C-convention of (i = col + ncols*row)
  template<typename T>
  inline void convertRowMajorToColumnMajor(T* out, T const* in, int nrows, int ncols){
    for(int i=0;i<nrows;i++){
      for(int j=0;j<ncols;j++){
	int iin = j + ncols*i;
	int iout = i + nrows*j;
	out[iout] = in[iin];
      }
    }
  }

  template<typename T>
  inline void convertColumnMajorToRowMajor(T* out, T const* in, int nrows, int ncols){
    for(int i=0;i<nrows;i++){
      for(int j=0;j<ncols;j++){
	int iout = j + ncols*i;
	int iin = i + nrows*j;
	out[iout] = in[iin];
      }
    }
  }

  //It seems bgclang and xlc don't play nice when it comes to complex numbers (probably a library version mismatch). 
  //To deal with this I define an intermediate complex type that can be cast as appropriate
  template<typename T>
  struct icomplex{
    T v[2];
  };

  //BLAS calls
  

  //Matrix multiplication with both matrices in column major format
  //C = AB   with optional transpose
  //l, m, n are the matrix dimensions.  C=l*n  A=l*m (or m*l if using transpose)  B=m*n (or n*m if using transpose)
  //C must have size ra * cb (not checked)
  void essl_gemul_colmajor(float* c,
			   float const* a,
			   float const* b,
			   int l,
			   int m,
			   int n,
			   bool trans_A,
			   bool trans_B);
  
  void essl_gemul_colmajor(double* c,
			   double const* a,
			   double const* b,
			   int l,
			   int m,
			   int n,
			   bool trans_A,
			   bool trans_B);

  void essl_gemul_zcolmajor(icomplex<float>* c,
			    icomplex<float> const* a,
			    icomplex<float> const* b,
			    int l,
			    int m,
			    int n,
			    bool trans_A,
			    bool trans_B);
 
  void essl_gemul_zcolmajor(icomplex<double>* c,
			    icomplex<double> const* a,
			    icomplex<double> const* b,
			    int l,
			    int m,
			    int n,
			    bool trans_A,
			    bool trans_B);

  template<typename T>
  void essl_gemul_colmajor(std::complex<T>* c,
			   std::complex<T> const* a,
			   std::complex<T> const* b,
			   int l,
			   int m,
			   int n,
			   bool trans_A,
			   bool trans_B){
    essl_gemul_zcolmajor( (icomplex<T>*)c,
			 (icomplex<T> const*)a,
			 (icomplex<T> const*)b,
			 l,m,n,trans_A,trans_B );
  }

  //buf must be at least l*m + m*n + l*n elements
  template<typename T>
  void essl_gemul_rowmajor(T* c,
			   T const* a,
			   T const* b,
			   T * buf,
			   int l,
			   int m,
			   int n,
			   bool trans_A,
			   bool trans_B){
    T* acm = buf;
    T* bcm = acm + l*m;
    T* ccm = bcm + m*n;
    convertRowMajorToColumnMajor(acm, a, trans_A ? m : l, trans_A ? l : m);
    convertRowMajorToColumnMajor(bcm, b, trans_B ? n : m, trans_B ? m : n);
    essl_gemul_colmajor(ccm,acm,bcm,l,m,n,trans_A,trans_B);
    convertColumnMajorToRowMajor(c, ccm, l, n);
  }

  template<typename T>
  void essl_gemul_rowmajor(T* c,
			   T const* a,
			   T const* b,
			   int l,
			   int m,
			   int n,
			   bool trans_A,
			   bool trans_B){
    int e = l*m + m*n + l*n;
    T* buf = (T*)malloc(e * sizeof(T));
    assert(buf != NULL);
    essl_gemul_rowmajor(c,a,b,buf,l,m,n,trans_A,trans_B);
    free(buf);
  }

}
