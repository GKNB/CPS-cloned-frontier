#ifndef _CPS_UTILS_GPU_H__
#define _CPS_UTILS_GPU_H__

#include<config.h>
#include "template_wizardry.h"
#include "utils_malloc.h"

#ifdef USE_GRID
#include<Grid/Grid.h>
#else
#define accelerator_inline inline
#endif //USE_GRID

#if defined(GRID_CUDA) || defined(GRID_HIP)

//Duplicates of Grid's wrappers but allowing for shared memory
//Shared memory size is *per block*. The number of threads in a block is nsimd*gpu_threads where gpu_threads is a Grid global variable set by the user on the command line (default 8)
//Note that unlike the regular accelerator_for, this currently doesn't work on the host as we need to abstract the shared memory concept
#define accelerator_forNB_shmem( iterator, num, nsimd, shmem_size, ... ) \
  {\
  typedef uint64_t Iterator;\
  auto lambda = [=] __device__ (Iterator iterator, Iterator ignored, Iterator lane) mutable { \
    __VA_ARGS__;\
  };\
  int nt=acceleratorThreads();					\
  dim3 cu_threads(nsimd, nt,1);					\
  dim3 cu_blocks ((num+nt-1)/nt,1,1);				\
  LambdaApply<<<cu_blocks,cu_threads,shmem_size>>>(num,1,nsimd,lambda); \
  }

// Copy the for_each_n style ; Non-blocking variant (default
#define accelerator_for_shmem( iterator, num, nsimd, shmem_size, ... )	\
  accelerator_forNB_shmem(iterator, num, nsimd, shmem_size, { __VA_ARGS__ } ); \
  accelerator_barrier(dummy);

#elif defined(GPU_VEC)
//GPU but not CUDA/HIP; not implemented

#define accelerator_forNB_shmem( iterator, num, nsimd, shmem_size, ... ) { static_assert(false, "accelerator_forNB_shmem not defined for GPU other than CUDA/HIP"); }
  
#define accelerator_shmem( iterator, num, nsimd, shmem_size, ... ) { static_assert(false, "accelerator_for_shmem not defined for GPU other than CUDA/HIP"); }

#else

#define accelerator_for_shmem(iterator,num,nsimd, shmem_size, ... )   thread_for(iterator, num, { __VA_ARGS__ });
#define accelerator_forNB_shmem(iterator,num,nsimd, shmem_size, ... ) thread_for(iterator, num, { __VA_ARGS__ });

#endif //GRID_CUDA || GRID_HIP


CPS_START_NAMESPACE

#if defined(GRID_CUDA) || defined(GRID_HIP) || !defined(GPU_VEC)
//FIXME: Need to test HIP implementation after mf_contract since it is needed in mesonfield_mult_vv_field_offload.h. Logic with GPU_VEC seems weird
//Only implemented for CUDA/HIP right now but I want to actually compile with non-CUDA devices!
//query the max bytes allocatable as block shared memory for a given device. If the device index is -1 it will be inferred from the current device
//Returns 0 if not using a CUDA/HIP GPU
inline int maxDeviceShmemPerBlock(int device = -1){
#ifdef GRID_CUDA
  if(device == -1) cudaGetDevice(&device);
  int smemSize;
  cudaDeviceGetAttribute(&smemSize, cudaDevAttrMaxSharedMemoryPerBlock, device);
  return smemSize;
#elif defined(GRID_HIP)
  if(device == -1) hipGetDevice(&device);
  int smemSize;
  hipDeviceGetAttribute(&smemSize, hipDeviceAttributeMaxSharedMemoryPerBlock, device);
  return smemSize;
#elif defined(GPU_VEC)
  static_assert(false, "maxDeviceShmemPerBlock not defined for GPU other than CUDA/HIP");
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

//memset for device memory
inline void device_memset(void *ptr, int value, size_t count){
#ifdef GPU_VEC
  Grid::acceleratorMemSet(ptr, value, count);
#else
  memset(ptr, value, count);
#endif
}

//Check if a class T has a method "free"
template<typename T, typename U = void>
struct hasFreeMethod{
  enum{ value = 0 };
};
template<typename T>
struct hasFreeMethod<T, typename Void<decltype( ((T*)(NULL))->free() )>::type>{
  enum{ value = 1 };
};

//Because View classes cannot have non-trivial destructors, if the view requires a free it needs to be managed externally
//This class calls free on the view (if it has a free method). It should be constructed after the view (and be destroyed before, which should happen automatically at the end of the scope)
//Static methods are also provided to wrap the free call if not present
template<typename ViewType, int hasFreeMethod>
struct _viewDeallocator{};

template<typename ViewType>
struct _viewDeallocator<ViewType,0>{
  ViewType &v;
  _viewDeallocator(ViewType &v): v(v){}

  ~_viewDeallocator(){
    //std::cout << "ViewType doesn't require free" << std::endl;
  }

  static void free(ViewType &v){}
};

template<typename ViewType>
struct _viewDeallocator<ViewType,1>{
  ViewType &v;
  _viewDeallocator(ViewType &v): v(v){}

  ~_viewDeallocator(){
    //std::cout << "ViewType does require free" << std::endl;	
    v.free();
  }

  static void free(ViewType &v){ v.free(); }
};

template<typename ViewType>
using viewDeallocator = _viewDeallocator<ViewType, hasFreeMethod<ViewType>::value>;  

#define CPSautoView(ViewName, ObjName) \
  auto ViewName = ObjName .view(); \
  viewDeallocator<typename std::decay<decltype(ViewName)>::type> ViewName##_d(ViewName);








//A class to contain an array of views. The views are assigned on the host but copied to the device
template<typename vtype>
class ViewArray{
public:
  typedef vtype ViewType;
private:
  ViewType *v;
  size_t sz;
public:
  
  accelerator_inline size_t size() const{ return sz; }
  
  ViewArray(): v(nullptr), sz(0){}

  //Create the viewarray object from an array of views vin of size n generated on the host
  //Use inplace new to generate the array because views don't tend to have default constructors
  ViewArray(ViewType *vin, size_t n): ViewArray(){ assign(vin, n); }

  //This version generates the host array of views automatically  
  template<typename T>
  ViewArray(const std::vector<T*> &obj_ptrs) : ViewArray(){ assign(obj_ptrs); }
  
  ViewArray(const ViewArray &r) = default;
  ViewArray(ViewArray &&r) = default;

  //This version generates the host array of views automatically
  template<typename T>
  void assign(const std::vector<T*> &obj_ptrs){
    size_t n = obj_ptrs.size();
    size_t byte_size = n*sizeof(ViewType);
    ViewType* tmpv = (ViewType*)malloc(byte_size);
    for(size_t i=0;i<n;i++){
      new (tmpv+i) ViewType(obj_ptrs[i]->view());
    }
    assign(tmpv,n);
    ::free(tmpv);
  }  

  //Create the viewarray object from an array of views vin of size n generated on the host
  //Use inplace new to generate the array because views don't tend to have default constructors
  void assign(ViewType *vin, size_t n){
    if(v != nullptr) device_free(v); //could be reused
      
    size_t byte_size = n * sizeof(ViewType);
    v = (ViewType*)device_alloc_check(byte_size);
    sz = n;
    copy_host_to_device(v, vin, byte_size);
  }

  //Deallocation must be either manually called or use CPSautoView  
  void free(){
    if(v){
      //The views may have allocs that need to be freed. This can only be done from the host, so we need to copy back
      if(hasFreeMethod<ViewType>::value){
	size_t byte_size = sz*sizeof(ViewType);
	ViewType* tmpv = (ViewType*)malloc(byte_size);
	copy_device_to_host(tmpv, v, byte_size);
	for(size_t i=0;i<sz;i++)
	  viewDeallocator<ViewType>::free(tmpv[i]); //so it compiles even if the view doesn't have a free method
	::free(tmpv);
      }
      device_free(v);
    }
  }
    
  accelerator_inline ViewType & operator[](const size_t i) const{ return v[i]; }
};



//A class to contain a pointer to a view that is automatically copied to the device.
//This, for example, allows the creation of dynamic containers of view objects that don't have default constructors
//In terms of functionality it is the same as a ViewArray of size 1
template<typename vtype>
class ViewPointerWrapper{
public:
  typedef vtype ViewType;
private:
  ViewType *v;
public:
  
  ViewPointerWrapper(): v(nullptr){}

  ViewPointerWrapper(const ViewType &vin): ViewPointerWrapper(){ assign(vin); }
 
  ViewPointerWrapper(const ViewPointerWrapper &r) = default;
  ViewPointerWrapper(ViewPointerWrapper &&r) = default;

  //Create the viewarray object from an array of views vin of size n generated on the host
  //Use inplace new to generate the array because views don't tend to have default constructors
  void assign(const ViewType &vin){
    if(v != nullptr) device_free(v); //could be reused
      
    size_t byte_size = sizeof(ViewType);
    v = (ViewType*)device_alloc_check(byte_size);
    copy_host_to_device(v, &vin, byte_size);
  }

  //Deallocation must be either manually called or use CPSautoView  
  void free(){
    if(v){
      //The views may have allocs that need to be freed. This can only be done from the host, so we need to copy back
      if(hasFreeMethod<ViewType>::value){
	size_t byte_size = sizeof(ViewType);
	ViewType* tmpv = (ViewType*)malloc(byte_size);
	copy_device_to_host(tmpv, v, byte_size);
	viewDeallocator<ViewType>::free(*tmpv); //so it compiles even if the view doesn't have a free method
	::free(tmpv);
      }
      device_free(v);
    }
  }

  accelerator_inline ViewType* ptr() const{ return v; }
  accelerator_inline ViewType& operator*() const{ return *v; }
  accelerator_inline ViewType* operator->() const{ return v; }
};


//A class that contains a pair of pointers, one on the device and one on the host, and facilities to sync between them
//For non-GPU applications it retains the same semantics but the pointer remains the same
template<typename T>
class hostDeviceMirroredContainer{
  T* host;
  T* device;
  bool host_in_sync;
  bool device_in_sync;  
  size_t n; //number of elements
  bool use_pinned_mem;
public:
  size_t byte_size() const{ return n*sizeof(T); }

  //pinned memory has a faster copy as it avoids a host-side copy in Cuda, but it can take a while to allocate
  hostDeviceMirroredContainer(size_t n, bool use_pinned_mem = true): n(n), device_in_sync(true), host_in_sync(true), use_pinned_mem(use_pinned_mem){
    host = use_pinned_mem ? ((T*)pinned_alloc_check(128,byte_size())) : ((T*)memalign_check(128,byte_size())); 
#ifdef GPU_VEC
    device = (T*)device_alloc_check(byte_size());
#else
    host = device;
#endif
  }

  //Return the status to its original, allowing the container to forget about its state and be reused
  void reset(){
    host_in_sync = device_in_sync = true;
  }  

  T* getHostWritePtr(){
    host_in_sync = true;
    device_in_sync = false;
    return host;
  }
  T* getDeviceWritePtr(){
    device_in_sync = true;
    host_in_sync = false;
    return device;
  }
  
  //Get the host pointer for read access. Will copy from the GPU if host not in sync
  T const* getHostReadPtr(){
    if(!host_in_sync){
#ifdef GPU_VEC            
      copy_device_to_host(host,device,byte_size());
#endif
      host_in_sync = true;
    }
    return host;
  }
  T const* getDeviceReadPtr(){
    if(!device_in_sync){
#ifdef GPU_VEC      
      copy_host_to_device(device,host,byte_size());
#endif
      device_in_sync = true;
    }
    return device;
  }

  //Synchronize the host and device with an asynchronous copy. Requires pinned memory
  //NOTE: While the sync flag will be set, the state is undefined until the copy stream has been synchronized
  void asyncHostDeviceSync(){
#if defined(GRID_CUDA)
    if(!use_pinned_mem) ERR.General("hostDeviceMirroredContainer","asyncHostDeviceSync","Requires use of pinned memoruy");
    if(!device_in_sync && !host_in_sync) ERR.General("hostDeviceMirroredContainer","asyncHostDeviceSync","Invalid state");
    if(!device_in_sync){
      cudaMemcpyAsync(device,host,byte_size(), cudaMemcpyHostToDevice,Grid::copyStream);
      device_in_sync = true;
    }
    if(!host_in_sync){
      cudaMemcpyAsync(device,host,byte_size(), cudaMemcpyDeviceToHost,Grid::copyStream);
      host_in_sync = true;
    }
#elif defined(GRID_HIP)
    if(!use_pinned_mem) ERR.General("hostDeviceMirroredContainer","asyncHostDeviceSync","Requires use of pinned memoruy");
    if(!device_in_sync && !host_in_sync) ERR.General("hostDeviceMirroredContainer","asyncHostDeviceSync","Invalid state");
    if(!device_in_sync){
      hipMemcpyAsync(device,host,byte_size(), hipMemcpyHostToDevice ,Grid::copyStream);
      device_in_sync = true;
    }
    if(!host_in_sync){
      hipMemcpyAsync(device,host,byte_size(), hipMemcpyDeviceToHost,Grid::copyStream);
      host_in_sync = true;
    }
#else    
    ERR.General("hostDeviceMirroredContainer","asyncHostDeviceSync","Asynchronous copies only currently supported for CUDA and HIP");
#endif
  }
  
  ~hostDeviceMirroredContainer(){
    if(use_pinned_mem) pinned_free(host);
    else free(host);
#ifdef GPU_VEC
    device_free(device);
#endif
  }
};
  

    
    
    


  


CPS_END_NAMESPACE

#endif
