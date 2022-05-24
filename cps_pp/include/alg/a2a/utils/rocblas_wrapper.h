#ifndef _ROCBLAS_WRAPPER_H_
#define _ROCBLAS_WRAPPER_H_

#include <Grid.h>

#ifdef GRID_HIP

#include "utils_malloc.h"
#include "rocblas.h"
#include <thrust/complex.h>
#include <cassert>

CPS_START_NAMESPACE


//A simple matrix class with managed memory
class gpuMatrix{
public:
  typedef typename thrust::complex<double> complexD;
private:
  complexD *d;
  size_t m_rows;
  size_t m_cols;
  
  void alloc()
  {
    assert( hipMallocManaged(&d, m_rows*m_cols*sizeof(complexD)) == hipSuccess );  
  }
public:
  gpuMatrix(size_t rows, size_t cols): m_rows(rows), m_cols(cols){
    alloc();
  }
  gpuMatrix(const gpuMatrix &r): m_rows(r.m_rows), m_cols(r.m_cols){    
    alloc();
    memcpy(d, r.d, m_rows*m_cols*sizeof(complexD));
  }
  gpuMatrix(gpuMatrix &&r): m_rows(r.m_rows), m_cols(r.m_cols), d(r.d){
    r.d = NULL;
  }

  __host__ __device__ inline complexD & operator()(const size_t i, const size_t j){
    return d[j + m_cols*i];
  }
  __host__ __device__ inline const complexD & operator()(const size_t i, const size_t j) const{
    return d[j + m_cols*i];
  }

  __host__ __device__ inline size_t rows() const{ return m_rows; }
  __host__ __device__ inline size_t cols() const{ return m_cols; }

  rocblas_double_complex* ptr(){ return (rocblas_double_complex*)d; }
  rocblas_double_complex const* ptr() const{ return (rocblas_double_complex*)d; }
  
  ~gpuMatrix(){
    if(d) hipFree(d);
  }

  bool equals(const gpuMatrix &r, const double tol = 1e-8) const{
    if(rows() != r.rows() || cols() != r.cols()) return false;
    for(size_t i=0;i<rows();i++)
      for(size_t j=0;j<cols();j++){
	double dr = fabs( (*this)(i,j).real() - r(i,j).real() );
	if(dr > tol) return false;

	double di = fabs( (*this)(i,j).imag() - r(i,j).imag() );
	if(di > tol) return false;
      }
    return true;
  }
  void print() const{
    for(size_t i=0;i<rows();i++){
      for(size_t j=0;j<cols();j++)
	std::cout << "(" << (*this)(i,j).real() << "," << (*this)(i,j).imag() << ") ";
      std::cout << std::endl;
    }
  }
  
};


class gpuDeviceMatrix{
public:
  typedef typename thrust::complex<double> complexD;
private:
  complexD *d; //device
  size_t *m_rows_d; //device
  size_t *m_cols_d; //device

  size_t m_rows_h; //host
  size_t m_cols_h; //host

  static inline void mallocd(void** ptr, size_t n){
    assert( hipMalloc(ptr, n) == hipSuccess ); 
  }
  static inline void copyd(void* to, void* from, size_t n){
    assert( hipMemcpy(to, from, n, hipMemcpyHostToDevice) == hipSuccess );
  }
  static inline void copyh(void* to, void* from, size_t n){
    assert( hipMemcpy(to, from, n, hipMemcpyDeviceToHost) == hipSuccess );
  }
  static inline void copydd(void* to, void* from, size_t n){
    assert( hipMemcpy(to, from, n, hipMemcpyDeviceToDevice) == hipSuccess );
  }

  __host__ void alloc(){
    mallocd((void**)&d, m_rows_h*m_cols_h*sizeof(complexD));
    mallocd((void**)&m_rows_d, sizeof(size_t));
    mallocd((void**)&m_cols_d, sizeof(size_t));
    copyd(m_rows_d, &m_rows_h, sizeof(size_t));
    copyd(m_cols_d, &m_cols_h, sizeof(size_t));
  }
public:
  __host__ gpuDeviceMatrix(size_t rows, size_t cols): m_rows_h(rows), m_cols_h(cols){
    alloc();
  }
  __host__ gpuDeviceMatrix(const gpuDeviceMatrix &r): m_rows_h(r.m_rows_h), m_cols_h(r.m_cols_h){    
    alloc();
    copydd(d, r.d, m_rows_h*m_cols_h*sizeof(complexD));
  }
  __device__ inline complexD & operator()(const size_t i, const size_t j){
    return d[j + (*m_cols_d)*i];
  }
  __device__ inline const complexD & operator()(const size_t i, const size_t j) const{
    return d[j + (*m_cols_d)*i];
  }

  __host__ __device__ inline size_t rows() const{
#if __HIP_DEVICE_COMPILE__	  
    return *m_rows_d;
#else
    return m_rows_h;
#endif
  }
  __host__ __device__ inline size_t cols() const{
#if __HIP_DEVICE_COMPILE__	  
    return *m_cols_d;
#else
    return m_cols_h;
#endif
  }

  __host__ __device__ rocblas_double_complex* ptr(){ return (rocblas_double_complex*)d; }
  __host__ __device__ rocblas_double_complex const* ptr() const{ return (rocblas_double_complex*)d; }

  //Copy nelem consecutive data from host to device
  inline void copyFromHost(const size_t i, const size_t j, complexD const* ptr, const size_t nelem = 1){
    copyd((void*)(d + j + m_cols_h*i), (void*)ptr, nelem*sizeof(complexD) );
  }

  //Copy nelem consecutive data from device to host
  inline void copyToHost(complexD *ptr, const size_t i, const size_t j, const size_t nelem = 1) const{
    copyh((void*)ptr, (void*)(d + j + m_cols_h*i), nelem*sizeof(complexD) );
  }

  ~gpuDeviceMatrix(){
    hipFree(d);
    hipFree(m_rows_d);
    hipFree(m_cols_d);
  }
  
};


//Matrix on host with pinned memory
//Pinning prevents the host from paging out the data and allows for asynchronous (concurrent) memory transfers
#define GPU_HOST_PINNED_MATRIX_USE_CACHE

class gpuHostPinnedMatrix{
public:
  typedef typename thrust::complex<double> complexD;
private:
  complexD *d;
  size_t m_rows;
  size_t m_cols;
  
  void alloc(){
#ifdef GPU_HOST_PINNED_MATRIX_USE_CACHE
    d = (complexD*)PinnedHostMemoryCache::alloc(m_rows*m_cols*sizeof(complexD));
#else
    assert( hipHostMalloc(&d, m_rows*m_cols*sizeof(complexD), hipHostMallocDefault) == hipSuccess );    
#endif
  }
public:
  gpuHostPinnedMatrix(size_t rows, size_t cols): m_rows(rows), m_cols(cols){
    alloc();
  }
  gpuHostPinnedMatrix(const gpuHostPinnedMatrix &r): m_rows(r.m_rows), m_cols(r.m_cols){    
    alloc();
    memcpy(d, r.d, m_rows*m_cols*sizeof(complexD));
  }
  gpuHostPinnedMatrix(gpuHostPinnedMatrix &&r): m_rows(r.m_rows), m_cols(r.m_cols), d(r.d){
    r.d = NULL;
  }

  __host__ inline complexD & operator()(const size_t i, const size_t j){
    return d[j + m_cols*i];
  }
  __host__ inline const complexD & operator()(const size_t i, const size_t j) const{
    return d[j + m_cols*i];
  }

  __host__ inline size_t rows() const{ return m_rows; }
  __host__ inline size_t cols() const{ return m_cols; }

  rocblas_double_complex* ptr(){ return (rocblas_double_complex*)d; }
  rocblas_double_complex const* ptr() const{ return (rocblas_double_complex*)d; }
  
  ~gpuHostPinnedMatrix(){
    if(d!=NULL){
#ifdef GPU_HOST_PINNED_MATRIX_USE_CACHE
      PinnedHostMemoryCache::free(d,m_rows*m_cols*sizeof(complexD));
#else
      assert( hipHostFree(d) == hipSuccess );
#endif
    }
  }
};


struct rocBLAShandles{
  inline static double & time(){ static double t; return t; }
  
  rocblas_handle handle;
  
  rocBLAShandles(){
    time() -= dclock();
    assert( rocblas_create_handle(&handle) == rocblas_status_success );
    time() += dclock();
  }

  ~rocBLAShandles(){
    assert( rocblas_destroy_handle(handle) == rocblas_status_success );
  }
};

//Multiply matrices C = A*B
//m = A.rows()  n=B.cols()  k=A.cols()
inline void mult_offload_rocBLAS(rocblas_double_complex* C,
			         rocblas_double_complex const* A,
			         rocblas_double_complex const* B,
			         const size_t m, const size_t n, const size_t k){
  static rocBLAShandles handles; //perform initialization crud only once
  
  //cuBLAS uses awful column-major order
  //use trick from https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication to workaround
  gpuMatrix::complexD one(1.0,0.0);
  gpuMatrix::complexD zero(0.0,0.0);

  rocblas_status err = rocblas_zgemm(handles.handle,
		  		      rocblas_operation_none, rocblas_operation_none,
				      n,m,k,
				      (rocblas_double_complex*)&one,
				      B, n,
				      A, k,
				      (rocblas_double_complex*)&zero,
				      C, n);
  if(err!=rocblas_status_success)
    ERR.General("","mult_offload_rocBLAS","rocblas_zgemm call failed with error: %s", rocblas_status_to_string(err));
  hipDeviceSynchronize();	//FIXME: Tianle: Need to figure out where to put this to improve performance
}




inline gpuMatrix mult_offload_rocBLAS(const gpuMatrix &A, const gpuMatrix &B){
  size_t m =A.rows(), n=B.cols(), k=A.cols();
  gpuMatrix C_2(m,n);
  mult_offload_rocBLAS(C_2.ptr(), A.ptr(), B.ptr(),
		       m,n,k);
  return C_2;
}

inline gpuMatrix mult_offload_rocBLAS(const gpuDeviceMatrix &A, const gpuDeviceMatrix &B){
  size_t m =A.rows(), n=B.cols(), k=A.cols();
  gpuMatrix C_2(m,n);
  mult_offload_rocBLAS(C_2.ptr(), A.ptr(), B.ptr(),
		       m,n,k);
  return C_2;
}

inline gpuHostPinnedMatrix mult_offload_rocBLAS(const gpuHostPinnedMatrix &A, const gpuHostPinnedMatrix &B){
  size_t m =A.rows(), n=B.cols(), k=A.cols();
  gpuHostPinnedMatrix C_2(m,n);
  mult_offload_rocBLAS(C_2.ptr(), A.ptr(), B.ptr(),
		       m,n,k);
  return C_2;
}

//A version of the above using preallocated output matrix
inline void mult_offload_rocBLAS(gpuHostPinnedMatrix &C, const gpuHostPinnedMatrix &A, const gpuHostPinnedMatrix &B){
  size_t m =A.rows(), n=B.cols(), k=A.cols();
  assert(C.rows() == m && C.cols() == n);
  mult_offload_rocBLAS(C.ptr(), A.ptr(), B.ptr(),
		       m,n,k);
}



CPS_END_NAMESPACE
#endif
//GRID_HIP

#endif
