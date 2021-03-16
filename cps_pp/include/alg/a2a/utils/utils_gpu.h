#ifndef _CPS_UTILS_GPU_H__
#define _CPS_UTILS_GPU_H__

#include<config.h>

#ifdef USE_GRID
#include<Grid/Grid.h>
#else
#define accelerator_inline inline
#endif //USE_GRID

#ifdef GRID_CUDA

//Duplicates of Grid's wrappers but allowing for shared memory
//Shared memory size is *per block*. The number of threads in a block is nsimd*gpu_threads where gpu_threads is a Grid global variable set by the user on the command line (default 8)
//Note that unlike the regular accelerator_for, this currently doesn't work on the host as we need to abstract the shared memory concept
#define accelerator_forNB_shmem( iterator, num, nsimd, shmem_size, ... ) \
  {\
  typedef uint64_t Iterator;\
  auto lambda = [=] __device__ (Iterator iterator, Iterator ignored, Iterator lane) mutable { \
    __VA_ARGS__;\
  };\
  int nt=acceleratorThreads();						\
  dim3 cu_threads(nt,1,nsimd);			\
  dim3 cu_blocks ((num+nt-1)/nt,1,1);				\
  LambdaApply<<<cu_blocks,cu_threads,shmem_size>>>(num,1,nsimd,lambda); \
  }

// Copy the for_each_n style ; Non-blocking variant (default
#define accelerator_for_shmem( iterator, num, nsimd, shmem_size, ... )	\
  accelerator_forNB_shmem(iterator, num, nsimd, shmem_size, { __VA_ARGS__ } ); \
  accelerator_barrier(dummy);

#elif defined(GPU_VEC)
//GPU but not CUDA; not implemented

#define accelerator_forNB_shmem( iterator, num, nsimd, shmem_size, ... ) { static_assert(false, "accelerator_forNB_shmem not defined for GPU other than CUDA"); }
  
#define accelerator_shmem( iterator, num, nsimd, shmem_size, ... ) { static_assert(false, "accelerator_for_shmem not defined for GPU other than CUDA"); }

#else

#define accelerator_for_shmem(iterator,num,nsimd, shmem_size, ... )   thread_for(iterator, num, { __VA_ARGS__ });
#define accelerator_forNB_shmem(iterator,num,nsimd, shmem_size, ... ) thread_for(iterator, num, { __VA_ARGS__ });

#endif //GRID_CUDA


CPS_START_NAMESPACE

#if defined(GRID_CUDA) || !defined(GPU_VEC)
//Only implemented for CUDA right now but I want to actually compile with non-CUDA devices!

//query the max bytes allocatable as block shared memory for a given device. If the device index is -1 it will be inferred from the current device
//Returns 0 if not using a CUDA GPU
inline int maxDeviceShmemPerBlock(int device = -1){
#ifdef GRID_CUDA
  if(device == -1) cudaGetDevice(&device);
  int smemSize;
  cudaDeviceGetAttribute(&smemSize, cudaDevAttrMaxSharedMemoryPerBlock, device);
  return smemSize;
#elif defined(GPU_VEC)
  static_assert(false, "maxDeviceShmemPerBlock not defined for GPU other than CUDA");
#else
  return 0;
#endif
}

#endif



//Wrappers for device copy; default to memcpy on non-GPU machine
inline void copy_host_to_device(void* to, void const* from, size_t bytes){
#ifdef GPU_VEC
  Grid::acceleratorCopyToDevice((void*)from, to, bytes); //arg I hate why Grid messes with the ordering sometimes!!
#else
  memcpy(to, from, bytes);
#endif
}
inline void copy_device_to_host(void* to, void const* from, size_t bytes){
#ifdef GPU_VEC
  Grid::acceleratorCopyFromDevice((void*)from, to, bytes); //***cries***
#else
  memcpy(to, from, bytes);
#endif
}





CPS_END_NAMESPACE

#endif
