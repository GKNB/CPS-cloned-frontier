#ifndef __QBLAS__CD
#define __QBLAS__CD

/*
*/
extern "C"{

  void cblas_daxpy(const int N, 
                   const double alpha, 
                   const double *X,
                   const int incX, 
                   double *Y, 
                   const int incY);
  
  void cblas_zaxpy(const int N, 
		   const double* alpha, 
		   const double *X,
		   const int incX, 
		   double *Y, 
		   const int incY);

  void cblas_caxpy(const int N, 
		  const float* alpha, 
		  const float *X,
		  const int incX, 
		  float *Y, 
		  const int incY);
  
  double cblas_dnrm2(const int N, 
                     const double *X, 
                     const int incX  );
  
  void cblas_dcopy(const int N, 
                   const double *X, 
                   const int incX, 
                   double *Y, 
                   const int incY);
  
  void cblas_dscal(const int N, 
                   const double alpha, 
                   double *X, 
                   const int incX);
 
  double cblas_ddot(const int N    , 
		    const double *X, 
		    const int incX ,
		    const double *Y,
		    const int incY  );

  void cblas_zdotc_sub(const int N    , 
		      const double *X, 
		      const int incX ,
		      const double *Y,
		      const int incY,
		      double* dot);
 void cblas_cdotc_sub(const int N    , 
		      const float *X, 
		      const int incX ,
		      const float *Y,
		      const int incY,
		      float* dot);
  
} /* extern "C" */

#endif 
