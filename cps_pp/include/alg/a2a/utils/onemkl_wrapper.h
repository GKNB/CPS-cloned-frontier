#ifndef ONEMKL_WRAPPER
#define ONEMKL_WRAPPER

#include<Grid.h>
#ifdef GRID_SYCL

#undef PRECISION

//hack to prevent collision with GSL
#ifdef __GSL_CBLAS_H__
#define __MKL_CBLAS_H__
//#define _MKL_TYPES_H_
#endif
#include<oneapi/mkl.hpp>

CPS_START_NAMESPACE

//Copy-pasta from oneMKL examples
template <typename T, int align>
struct mkl_allocator
{
    typedef T*          pointer;
    typedef const T*    const_pointer;
    typedef void*       void_pointer;
    typedef const void* const_void_pointer;
    typedef T           value_type;
    typedef size_t      size_type;
    typedef ptrdiff_t   difference_type;

    template <typename U> struct rebind { typedef mkl_allocator<U,align> other; };

    mkl_allocator() noexcept {}
    template <typename U, int align2> mkl_allocator(mkl_allocator<U,align2> &other)  noexcept {}
    template <typename U, int align2> mkl_allocator(mkl_allocator<U,align2> &&other) noexcept {}

    T* allocate(size_t n) {
        void *mem = mkl_malloc(n * sizeof(T), align);
        if (!mem) throw std::bad_alloc();

        return static_cast<T*>(mem);
    }

    void deallocate(T *p, size_t n) noexcept {
      mkl_free(p);
    }

    constexpr size_t max_size() const noexcept {
        return std::numeric_limits<size_t>::max() / sizeof(T);
    }

    template <typename U, int align2> constexpr bool operator==(const mkl_allocator<U,align2>) const noexcept { return true;  }
    template <typename U, int align2> constexpr bool operator!=(const mkl_allocator<U,align2>) const noexcept { return false; }

    typedef std::true_type is_always_equal;
};


//Matrix for oneMKL, row major layout
template<typename T>
class oneMKLmatrix{
  int m_rows;
  int m_cols;
  std::vector <T, mkl_allocator<T, 64>> d;
public:
  oneMKLmatrix(const int rows, const int cols): m_rows(rows), m_cols(cols), d(rows*cols){}
  oneMKLmatrix(): m_rows(0), m_cols(0), d(0) {}

  void resize(const int rows, const int cols){
    m_rows = rows;
    m_cols = cols;
    d.resize(m_rows*m_cols);
  }
     
  inline T & operator()(const size_t i, const size_t j){
    return d[j + m_cols*i];
  }
  inline const T & operator()(const size_t i, const size_t j) const{
    return d[j + m_cols*i];
  }

  inline int rows() const { return m_rows; }
  inline int cols() const { return m_cols; }

  cl::sycl::buffer<T,1> getBuffer() const{ return cl::sycl::buffer<T,1>(const_cast<T*>(d.data()),d.size()); }
};


//Multiply matrices C = A*B
//m = A.rows()  n=B.cols()  k=A.cols()
template<typename T>
inline void mult_offload_oneMKL(oneMKLmatrix<T> &C,
				const oneMKLmatrix<T> &A,
				const oneMKLmatrix<T> &B){
  cl::sycl::queue &main_queue = *Grid::theGridAccelerator;
  
  int m = A.rows();
  int k = A.cols();
  int n = B.cols();

  assert(A.cols() == B.rows());
  assert(C.rows() == A.rows());
  assert(C.cols() == B.cols());
  
  //C' <-  alpha * op(A') * op(B') + beta * C'
  //alpha = -1.
  //beta = 0

  //oneMKL also uses awful column-major order
  //use trick from https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication to workaround
  //C^T = B^T A^T
  
  const static oneapi::mkl::transpose noTrans = oneapi::mkl::transpose::nontrans;

  const static T alpha(1.);
  const static T beta(0.);

  cl::sycl::buffer<T,1> A_wrapper(A.getBuffer());
  cl::sycl::buffer<T,1> B_wrapper(B.getBuffer());
  cl::sycl::buffer<T,1> C_wrapper(C.getBuffer());

  //A'= B^T = [k*n]^T = n*k
  //B'= A^T = [m*k]^T = k*m
  //C'= C^T = [m*n]^T = n*m

  //args
  //rows of A'=n
  //cols of B'=m
  //cols of A'=k

  //lda(A') = rows A' = n
  //lda(B') = rows B' = k
  //lda(C') = rows C' = n
  
  try {   
    oneapi::mkl::blas::gemm(main_queue, noTrans, noTrans, n, m, k, alpha, B_wrapper, n, A_wrapper, k, beta, C_wrapper, n);  }
  catch(cl::sycl::exception const& e) {
    std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
	      << e.what() << std::endl << "OpenCL status: " << e.get_cl_code() << std::endl;
    ERR.General("","mult_offload_oneMKL","Offloaded GEMM failed\n");
  }
}

CPS_END_NAMESPACE


#endif //GRID_SYCL
#endif //include guard
