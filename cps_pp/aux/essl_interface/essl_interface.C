#include "essl_interface.h"
#include<essl.h>
#include<cstdio>

namespace essl_interface{

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
			   bool trans_B){
    sgemul(a, trans_A ? m : l, trans_A ? "T" : "N",
    	   b, trans_B ? n : m, trans_B ? "T" : "N",
    	   c, l, 
    	   l,m,n);
  }

  void essl_gemul_colmajor(double* c,
			   double const* a,
			   double const* b,
			   int l,
			   int m,
			   int n,
			   bool trans_A,
			   bool trans_B){
    dgemul(a, trans_A ? m : l, trans_A ? "T" : "N",
    	   b, trans_B ? n : m, trans_B ? "T" : "N",
    	   c, l, 
    	   l,m,n);
  }

  void essl_gemul_zcolmajor(icomplex<float>* c,
			   icomplex<float> const* a,
			   icomplex<float> const* b,
			   int l,
			   int m,
			   int n,
			   bool trans_A,
			   bool trans_B){
    cgemul(a, trans_A ? m : l, trans_A ? "T" : "N",
    	   b, trans_B ? n : m, trans_B ? "T" : "N",
    	   c, l, 
    	   l,m,n);
  }

  void essl_gemul_zcolmajor(icomplex<double>* c,
			   icomplex<double> const* a,
			   icomplex<double> const* b,
			   int l,
			   int m,
			   int n,
			   bool trans_A,
			   bool trans_B){
    zgemul(a, trans_A ? m : l, trans_A ? "T" : "N",
    	   b, trans_B ? n : m, trans_B ? "T" : "N",
    	   c, l, 
    	   l,m,n);
  }
}
