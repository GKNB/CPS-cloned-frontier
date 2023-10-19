#ifndef _UTILS_PARALLEL_GLOBALSUM_H_
#define _UTILS_PARALLEL_GLOBALSUM_H_

#include <util/gjp.h>
#ifdef USE_GRID
#include <Grid/Grid.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "utils_parallel.h"
#include "utils_logging.h"

CPS_START_NAMESPACE

inline void globalSum(double *result, size_t len = 1){
#ifdef A2A_GLOBALSUM_DISK
  LOGA2A << "Performing reduction of " << byte_to_MB(len*sizeof(double)) << " MB through disk" << std::endl;
  disk_reduce(result,len);
#elif defined(A2A_GLOBALSUM_MAX_ELEM) //if this is defined, the global sum will be broken up into many of this size
  size_t b = A2A_GLOBALSUM_MAX_ELEM;
  LOGA2A << "Performing reduction of " << len << " doubles in " << (len+b-1) / b << " blocks of size " << b << std::endl;
  size_t lencp = len;
  for(size_t off = 0; off < lencp; off += b){
    size_t rlen = std::min(len, b);
    MPI_Allreduce(MPI_IN_PLACE, result, rlen, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    len -= rlen;
    result += rlen;
  }
#elif defined(USE_QMP)
  QMP_sum_double_array(result, len);
#elif defined(USE_MPI)
  MPI_Allreduce(MPI_IN_PLACE, result, len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  if(GJP.TotalNodes()!=1) ERR.General("","globalSum(double *result, int len)","Only implemented for QMP/MPI on parallel machines");
#endif
}

inline void globalSum(float *result, size_t len = 1){
#ifdef A2A_GLOBALSUM_DISK
  LOGA2A << "Performing reduction of " << byte_to_MB(len*sizeof(float)) << " MB through disk" << std::endl;
  disk_reduce(result,len);
#elif defined(A2A_GLOBALSUM_MAX_ELEM) //if this is defined, the global sum will be broken up into many of this size
  size_t b = A2A_GLOBALSUM_MAX_ELEM;
  LOGA2A << "Performing reduction of " << len << " floats in " << (len+b-1) / b << " blocks of size " << b << std::endl;
  size_t lencp = len;
  for(size_t off = 0; off < lencp; off += b){
    size_t rlen = std::min(len, b);
    MPI_Allreduce(MPI_IN_PLACE, result, rlen, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    len -= rlen;
    result += rlen;
  }
#elif defined(USE_QMP)
  QMP_sum_float_array(result, len);
#elif defined(USE_MPI)
  MPI_Allreduce(MPI_IN_PLACE, result, len, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
  if(GJP.TotalNodes()!=1) ERR.General("","globalSum(float *result, int len)","Only implemented for QMP/MPI on parallel machines");
#endif
}

inline void globalSum(std::complex<double> *result, size_t len = 1){
  globalSum( (double*)result, 2*len );
}
inline void globalSum(std::complex<float> *result, size_t len = 1){
  globalSum( (float*)result, 2*len );
}



#ifdef USE_GRID

#if defined(GRID_CUDA) || defined(GRID_HIP)
inline void globalSum(thrust::complex<double>* v, const size_t n = 1){
  globalSum( (double*)v,2*n);
}
inline void globalSum(thrust::complex<float>* v, const size_t n = 1){
  globalSum( (float*)v,2*n);
}


#endif //GRID_CUDA || GRID_HIP

template<typename T>
struct _globalSumComplexGrid{
  static inline void doit(T *v, const int n){
    typedef typename T::scalar_type scalar_type; //an std::complex type
    typedef typename scalar_type::value_type floatType;    
    int vmult = sizeof(T)/sizeof(scalar_type);
    floatType * ptr = (floatType *)v; 
    globalSum(ptr,2*n*vmult);
  }
};

inline void globalSum(Grid::vComplexD* v, const size_t n = 1){
  _globalSumComplexGrid<Grid::vComplexD>::doit(v,n);
}
inline void globalSum(Grid::vComplexF* v, const size_t n = 1){
  _globalSumComplexGrid<Grid::vComplexF>::doit(v,n);
}

#endif //GRID


CPS_END_NAMESPACE

#endif
