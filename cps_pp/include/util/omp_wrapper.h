#ifndef FAKE_OMP_H
#define FAKE_OMP_H
#ifdef USE_OMP
#warning "omp_wrapper.h: Using OpenMP"
#include <omp.h>
#else
#warning "omp_wrapper.h: NOT using OpenMP"
inline void omp_set_dynamic(bool val){}
inline int omp_get_num_threads(void) {return 1;}
inline int omp_get_max_threads(void) {return 1;}
inline int omp_get_thread_num(void) {return 0;}
inline void omp_set_num_threads(int n) {}
#endif
#endif
