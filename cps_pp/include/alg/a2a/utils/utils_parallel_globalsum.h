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

CPS_START_NAMESPACE

inline void globalSum(double *result, size_t len = 1){
#ifdef USE_QMP
  QMP_sum_double_array(result, len);
#elif defined(USE_MPI)
  MPI_Allreduce(MPI_IN_PLACE, result, len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  if(GJP.TotalNodes()!=1) ERR.General("","globalSum(double *result, int len)","Only implemented for QMP/MPI on parallel machines");
#endif
}

inline void globalSum(float *result, size_t len = 1){
#ifdef USE_QMP
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

#ifdef GRID_NVCC
void globalSum(thrust::complex<double>* v, const size_t n = 1){
  globalSum( (double*)v,2*n);
}
void globalSum(thrust::complex<float>* v, const size_t n = 1){
  globalSum( (float*)v,2*n);
}


#endif //GRID_NVCC

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
