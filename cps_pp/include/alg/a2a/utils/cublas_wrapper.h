#ifndef _CUBLAS_WRAPPER_H_
#define _CUBLAS_WRAPPER_H_

#ifdef GRID_CUDA

#include "utils_malloc.h"
#include<cublasXt.h>
#include <thrust/complex.h>
#include <cassert>

CPS_START_NAMESPACE


//A simple matrix class with managed memory and controllable shallow/deepcopy
class gpuMatrix{
public:
  typedef typename thrust::complex<double> complexD;
private:
  complexD *d;
  size_t m_rows;
  size_t m_cols;
  
  void alloc(){
    assert( cudaMallocManaged(&d, m_rows*m_cols*sizeof(complexD)) == cudaSuccess );
    
  }
public:
  gpuMatrix(size_t rows, size_t cols): m_rows(rows), m_cols(cols){
    alloc();
  }
  gpuMatrix(const gpuMatrix &r): m_rows(r.m_rows), m_cols(r.m_cols){    
    if(!copyControl::shallow()){
      alloc();
      memcpy(d, r.d, m_rows*m_cols*sizeof(complexD));
    }else{
      d = r.d;
    }
  }
  gpuMatrix(gpuMatrix &&r): m_rows(r.m_rows), m_cols(r.m_cols), d(r.d){
    assert(!copyControl::shallow());
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

  cuDoubleComplex* ptr(){ return (cuDoubleComplex*)d; }
  cuDoubleComplex const* ptr() const{ return (cuDoubleComplex*)d; }
  
  ~gpuMatrix(){
    if(!copyControl::shallow() && d != NULL){
      cudaFree(d);
    }
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
    assert( cudaMalloc(ptr, n) == cudaSuccess ); 
  }
  static inline void copyd(void* to, void* from, size_t n){
    assert( cudaMemcpy(to, from, n, cudaMemcpyHostToDevice) == cudaSuccess );
  }
  static inline void copyh(void* to, void* from, size_t n){
    assert( cudaMemcpy(to, from, n, cudaMemcpyDeviceToHost) == cudaSuccess );
  }
  static inline void copydd(void* to, void* from, size_t n){
    assert( cudaMemcpy(to, from, n, cudaMemcpyDeviceToDevice) == cudaSuccess );
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
#ifdef __CUDA_ARCH__
    return *m_rows_d;
#else
    return m_rows_h;
#endif
  }
  __host__ __device__ inline size_t cols() const{ 
#ifdef __CUDA_ARCH__
    return *m_cols_d;
#else
    return m_cols_h;
#endif
  }

  __host__ __device__ cuDoubleComplex* ptr(){ return (cuDoubleComplex*)d; }
  __host__ __device__ cuDoubleComplex const* ptr() const{ return (cuDoubleComplex*)d; }

  //Copy nelem consecutive data from host to device
  inline void copyFromHost(const size_t i, const size_t j, complexD const* ptr, const size_t nelem = 1){
    copyd((void*)(d + j + m_cols_h*i), (void*)ptr, nelem*sizeof(complexD) );
  }

  //Copy nelem consecutive data from device to host
  inline void copyToHost(complexD *ptr, const size_t i, const size_t j, const size_t nelem = 1) const{
    copyh((void*)ptr,  (void*)(d + j + m_cols_h*i), nelem*sizeof(complexD) );
  }

  ~gpuDeviceMatrix(){
    cudaFree(d);
    cudaFree(m_rows_d);
    cudaFree(m_cols_d);
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
    assert( cudaMallocHost(&d, m_rows*m_cols*sizeof(complexD), cudaHostAllocDefault) == cudaSuccess );    
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

  cuDoubleComplex* ptr(){ return (cuDoubleComplex*)d; }
  cuDoubleComplex const* ptr() const{ return (cuDoubleComplex*)d; }
  
  ~gpuHostPinnedMatrix(){
    if(d!=NULL){
#ifdef GPU_HOST_PINNED_MATRIX_USE_CACHE
      PinnedHostMemoryCache::free(d,m_rows*m_cols*sizeof(complexD));
#else
      cudaFree(d);
#endif
    }
  }
};











inline void mult_offload_cuBLASxt(cuDoubleComplex* C,
			   cuDoubleComplex const* A,
			   cuDoubleComplex const* B,
			   const size_t m, const size_t n, const size_t k){

  cublasXtHandle_t handle_xt;
  assert( cublasXtCreate(&handle_xt) == CUBLAS_STATUS_SUCCESS );

  int devices[1] = { Grid::GlobalSharedMemory::WorldShmRank };   //use the same GPU Grid does
  assert(cublasXtDeviceSelect(handle_xt, 1, devices) == CUBLAS_STATUS_SUCCESS);
  
  //cuBLAS uses awful column-major order
  //use trick from https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication to workaround
  gpuMatrix::complexD one(1.0,0.0);
  gpuMatrix::complexD zero(0.0,0.0);

  assert( cublasXtZgemm(handle_xt,
  			CUBLAS_OP_N,CUBLAS_OP_N,
  			n,m,k,
  			(cuDoubleComplex*)&one,
  			B, n,
  			A, k,
  			(cuDoubleComplex*)&zero,
  			C, n) == CUBLAS_STATUS_SUCCESS );

  assert( cublasXtDestroy(handle_xt) == CUBLAS_STATUS_SUCCESS );
}




inline gpuMatrix mult_offload_cuBLASxt(const gpuMatrix &A, const gpuMatrix &B){
  size_t m =A.rows(), n=B.cols(), k=A.cols();
  gpuMatrix C_2(m,n);
  mult_offload_cuBLASxt(C_2.ptr(), A.ptr(), B.ptr(),
			m,n,k);
  return C_2;
}

inline gpuMatrix mult_offload_cuBLASxt(const gpuDeviceMatrix &A, const gpuDeviceMatrix &B){
  size_t m =A.rows(), n=B.cols(), k=A.cols();
  gpuMatrix C_2(m,n);
  mult_offload_cuBLASxt(C_2.ptr(), A.ptr(), B.ptr(),
			m,n,k);
  return C_2;
}

inline gpuHostPinnedMatrix mult_offload_cuBLASxt(const gpuHostPinnedMatrix &A, const gpuHostPinnedMatrix &B){
  size_t m =A.rows(), n=B.cols(), k=A.cols();
  gpuHostPinnedMatrix C_2(m,n);
  mult_offload_cuBLASxt(C_2.ptr(), A.ptr(), B.ptr(),
			m,n,k);
  return C_2;
}



CPS_END_NAMESPACE
#endif
//GRID_CUDA

#endif
