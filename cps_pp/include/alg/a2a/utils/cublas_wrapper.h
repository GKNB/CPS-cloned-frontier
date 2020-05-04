#ifndef _CUBLAS_WRAPPER_H_
#define _CUBLAS_WRAPPER_H_

#ifdef GRID_NVCC

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
    if(!copyControl::shallow()){
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

gpuMatrix mult_offload_cuBLASxt(const gpuMatrix &A, const gpuMatrix &B){
  typedef gpuMatrix::complexD complexD;

  cublasXtHandle_t handle_xt;
  assert( cublasXtCreate(&handle_xt) == CUBLAS_STATUS_SUCCESS );

  int devices[1] = { Grid::GlobalSharedMemory::WorldShmRank };   //use the same GPU Grid does
  assert(cublasXtDeviceSelect(handle_xt, 1, devices) == CUBLAS_STATUS_SUCCESS);
  
  //cuBLAS uses awful column-major order
  //use trick from https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication to workaround
  complexD one(1.0,0.0);
  complexD zero(0.0,0.0);
  size_t m =A.rows(), n=B.cols(), k=A.cols();
  gpuMatrix C_2(m,n);

  assert( cublasXtZgemm(handle_xt,
  			CUBLAS_OP_N,CUBLAS_OP_N,
  			n,m,k,
  			(cuDoubleComplex*)&one,
  			B.ptr(), n,
  			A.ptr(), k,
  			(cuDoubleComplex*)&zero,
  			C_2.ptr(), n) == CUBLAS_STATUS_SUCCESS );

  assert( cublasXtDestroy(handle_xt) == CUBLAS_STATUS_SUCCESS );

  return C_2;
}




#endif
//GRID_NVCC



CPS_END_NAMESPACE

#endif
