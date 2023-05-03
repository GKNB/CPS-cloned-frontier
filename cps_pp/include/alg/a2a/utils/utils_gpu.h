#ifndef _CPS_UTILS_GPU_H__
#define _CPS_UTILS_GPU_H__

#include<thread>

#include<config.h>
#include "template_wizardry.h"
#include "utils_malloc.h"

#ifdef USE_GRID
#include<Grid/Grid.h>
#else
#define accelerator_inline inline
#endif //USE_GRID

#ifdef CPS_ENABLE_DEVICE_PROFILING
# if defined(GRID_CUDA)
#  include<cuda_profiler_api.h>
# elif defined(GRID_HIP)
#  include <hip/hip_profile.h>
# endif
#endif

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
#if defined(GRID_CUDA)
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

//Advise the UVM driver that the memory region will be accessed in read-only fashion
inline void device_UVM_advise_readonly(const void* ptr, size_t count){
#if defined(GRID_CUDA)
  assert( cudaMemAdvise(ptr, count, cudaMemAdviseSetReadMostly, 0) == cudaSuccess );
#elif defined(GRID_HIP)
  assert( hipMemAdvise(ptr, count, hipMemAdviseSetReadMostly, 0) == hipSuccess );
#else
  assert(0);
#endif
}

//Unset advice to the UVM driver that the memory region will be accessed in read-only fashion
inline void device_UVM_advise_unset_readonly(const void* ptr, size_t count){
#if defined(GRID_CUDA)
  assert( cudaMemAdvise(ptr, count, cudaMemAdviseUnsetReadMostly, 0) == cudaSuccess );
#elif defined(GRID_HIP)  
  assert( hipMemAdvise(ptr, count, hipMemAdviseUnsetReadMostly, 0) == hipSuccess );
#else
  assert(0);
#endif
}

inline void device_UVM_prefetch_device(const void* ptr, size_t sz){
#if defined(GRID_CUDA)
  int device;
  cudaGetDevice(&device);
  assert( cudaMemPrefetchAsync( ptr, sz, device, Grid::copyStream ) == cudaSuccess );
#elif defined(GRID_HIP)
  int device;
  hipGetDevice(&device);
  assert( hipMemPrefetchAsync( ptr, sz, device, Grid::copyStream ) == hipSuccess );
#else
  //do nothing
#endif
}

inline void device_UVM_prefetch_host(const void* ptr, size_t sz){
#if defined(GRID_CUDA)
  assert( cudaMemPrefetchAsync( ptr, sz, cudaCpuDeviceId, Grid::copyStream ) == cudaSuccess );
#elif defined(GRID_HIP)
  assert( hipMemPrefetchAsync( ptr, sz, hipCpuDeviceId, Grid::copyStream ) == hipSuccess );
#else
  //do nothing
#endif
}

//NB: for HIP,CUDA this requires the host pointer to be allocated in pinned memory
inline void copy_host_to_device_async(void* to, void const* from, size_t bytes){
#if defined(GRID_CUDA)
  cudaMemcpyAsync(to,from,bytes, cudaMemcpyHostToDevice,Grid::copyStream);
#elif defined(GRID_HIP)
  hipMemcpyAsync(to,from,bytes, hipMemcpyHostToDevice ,Grid::copyStream);
#elif defined(GRID_SYCL)
  Grid::theCopyAccelerator->memcpy(to,from,bytes);
#else
  //assume host
  memcpy(to,from,bytes);
#endif
}

inline void copy_device_to_host_async(void* to, void const* from, size_t bytes){
#if defined(GRID_CUDA)
  cudaMemcpyAsync(to,from,bytes, cudaMemcpyDeviceToHost,Grid::copyStream);
#elif defined(GRID_HIP)
  hipMemcpyAsync(to,from,bytes, hipMemcpyDeviceToHost,Grid::copyStream);
#elif defined(GRID_SYCL)
  Grid::theCopyAccelerator->memcpy(to,from,bytes);
#else
  //assume host
  memcpy(to,from,bytes);
#endif
}

//Synchronize the compute and copy stream ensuring all asynchronous copies and kernels have completed
inline void device_synchronize_all(){
#ifdef GPU_VEC
  using namespace Grid;
  accelerator_barrier(); //computeStream
  acceleratorCopySynchronise(); //copyStream
#endif
}

//Synchronize asynchronous copies and UVM prefetches
inline void device_synchronize_copies(){
#ifdef GPU_VEC
  using namespace Grid;
  acceleratorCopySynchronise(); //copyStream
#endif
}


inline void device_profile_start(){
#ifdef CPS_ENABLE_DEVICE_PROFILING
# if defined(GRID_CUDA)
  cudaProfilerStart();
# elif defined(GRID_HIP)
  hipProfilerStart();
# endif
#endif
}
inline void device_profile_stop(){
#ifdef CPS_ENABLE_DEVICE_PROFILING
# if defined(GRID_CUDA)
  cudaProfilerStop();
# elif defined(GRID_HIP)
  hipProfilerStop();
# endif
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

enum ViewMode { HostRead, HostWrite, DeviceRead, DeviceWrite };

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
  }

  static void free(ViewType &v){}
};

template<typename ViewType>
struct _viewDeallocator<ViewType,1>{
  ViewType &v;
  _viewDeallocator(ViewType &v): v(v){}

  ~_viewDeallocator(){
    v.free();
  }

  static void free(ViewType &v){ v.free(); }
};

template<typename ViewType>
using viewDeallocator = _viewDeallocator<ViewType, hasFreeMethod<ViewType>::value>;  

#define CPSautoView(ViewName, ObjName, mode)		\
  auto ViewName = ObjName .view(mode); \
  viewDeallocator<typename std::decay<decltype(ViewName)>::type> ViewName##_d(ViewName);

//A class that contains a view. This should *not* be copied to the device, it merely holds the view and ensures the view is
//freed upon destruction. This allows us to have views as class members that are destroyed when the instance is destroyed
template<typename vtype>
class ViewAutoDestructWrapper{
public:
  typedef vtype ViewType;
private:
  ViewType* v;
public:
  ViewAutoDestructWrapper(): v(nullptr){}
  //Note, takes ownership of pointer!
  ViewAutoDestructWrapper(ViewType *vin): v(vin){}

  ViewAutoDestructWrapper(ViewType &&vin): v(new ViewType(vin)){}

  //Cannot be copied
  ViewAutoDestructWrapper(const ViewAutoDestructWrapper &r) = delete;
  //Can be moved
  ViewAutoDestructWrapper(ViewAutoDestructWrapper &&r) = default;

  void reset(ViewType *vin){
    if(v){ 
      viewDeallocator<ViewType>::free(*v);
      delete v;
    }
    v = vin;
  }  
  void reset(ViewType &&vin){ reset(new ViewType(vin)); }
 
  bool isSet() const{ return v!=nullptr; }
  ViewType* ptr(){ return v; }
  ViewType const* ptr() const{ return v; }
  
  ViewType & operator*(){ return *v; }
  ViewType const & operator*() const{ return *v; }
  ViewType* operator->(){ return v; }
  ViewType const* operator->() const{ return v; }

  ~ViewAutoDestructWrapper(){
    if(v){ 
      viewDeallocator<ViewType>::free(*v);
      delete v;
    }
  }
};



//A class to contain an array of views. The views are assigned on the host but copied to the device
template<typename vtype>
class ViewArray{
public:
  typedef vtype ViewType;
private:
  ViewType *v;
  size_t sz;

  enum DataLoc { Host, Device };
  DataLoc loc;

  void placeData(ViewMode mode, ViewType* host_views, size_t n){
    freeData();
    sz = n;
    size_t byte_size = n*sizeof(ViewType);
    if(mode == DeviceRead || mode == DeviceWrite){
      v = (ViewType *)device_alloc_check(byte_size);
      copy_host_to_device(v, host_views, byte_size);
      loc = Device;
    }else{
      v = (ViewType *)malloc(byte_size);
      memcpy(v,host_views,byte_size);
      loc = Host;
    }
  }
  void freeData(){
    if(v){
      if(loc == Host){
	if(hasFreeMethod<ViewType>::value){
	  for(size_t i=0;i<sz;i++) viewDeallocator<ViewType>::free(v[i]); //so it compiles even if the view doesn't have a free method
	}
	::free(v);
      }else{
	//The views may have allocs that need to be freed. This can only be done from the host, so we need to copy back
	if(hasFreeMethod<ViewType>::value){
	  size_t byte_size = sz*sizeof(ViewType);
	  ViewType* tmpv = (ViewType*)malloc(byte_size);
	  copy_device_to_host(tmpv, v, byte_size);
	  for(size_t i=0;i<sz;i++) viewDeallocator<ViewType>::free(tmpv[i]); //so it compiles even if the view doesn't have a free method
	  ::free(tmpv);
	}
	device_free(v);
      }
      
    }
    v=nullptr;
    sz=0;
  }

public:
  
  accelerator_inline size_t size() const{ return sz; }
  
  ViewArray(): v(nullptr), sz(0){}

  //Create the viewarray object from an array of views vin of size n generated on the host
  //Use inplace new to generate the array because views don't tend to have default constructors
  ViewArray(ViewMode mode, ViewType *vin, size_t n): ViewArray(){ assign(mode, vin, n); }

  //This version generates the host array of views automatically  
  template<typename T>
  ViewArray(ViewMode mode, const std::vector<T*> &obj_ptrs) : ViewArray(){ assign(mode,obj_ptrs); }

  template<typename T>
  ViewArray(ViewMode mode, const std::vector<T> &objs) : ViewArray(){ assign(mode,objs); }

  
  ViewArray(const ViewArray &r) = default;
  ViewArray(ViewArray &&r) = default;

  //This version generates the host array of views automatically
  template<typename T>
  void assign(ViewMode mode, const std::vector<T*> &obj_ptrs){
    size_t n = obj_ptrs.size();
    size_t byte_size = n*sizeof(ViewType);
    ViewType* tmpv = (ViewType*)malloc(byte_size);
    for(size_t i=0;i<n;i++){
      new (tmpv+i) ViewType(obj_ptrs[i]->view(mode));
    }
    assign(mode,tmpv,n);
    ::free(tmpv);
  }  

  template<typename T>
  void assign(ViewMode mode, const std::vector<T> &objs){
    size_t n = objs.size();
    size_t byte_size = n*sizeof(ViewType);
    ViewType* tmpv = (ViewType*)malloc(byte_size);
    for(size_t i=0;i<n;i++){
      new (tmpv+i) ViewType(objs[i].view(mode));
    }
    assign(mode,tmpv,n);
    ::free(tmpv);
  }  


  //Create the viewarray object from an array of views vin of size n generated on the host
  //Use inplace new to generate the array because views don't tend to have default constructors
  void assign(ViewMode mode, ViewType *vin, size_t n){
    placeData(mode,vin,n);
  }

  //Deallocation must be either manually called or use CPSautoView  
  void free(){
    freeData();
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

  enum DataLoc { Host, Device };
  DataLoc loc;

  void placeData(ViewMode mode, ViewType* host_view){
    freeData();
    size_t byte_size = sizeof(ViewType);
    if(mode == DeviceRead || mode == DeviceWrite){
      v = (ViewType *)device_alloc_check(byte_size);
      copy_host_to_device(v, host_view, byte_size);
      loc = Device;
    }else{
      v = (ViewType *)malloc(byte_size);
      memcpy(v,host_view,byte_size);
      loc = Host;
    }
  }
  void freeData(){
    if(v){
      if(loc == Host){
	if(hasFreeMethod<ViewType>::value){
	  viewDeallocator<ViewType>::free(*v); //so it compiles even if the view doesn't have a free method
	}
	::free(v);
      }else{
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
    v=nullptr;
  }


public:
  
  ViewPointerWrapper(): v(nullptr){}

  ViewPointerWrapper(ViewMode mode, const ViewType &vin): ViewPointerWrapper(){ assign(mode, vin); }
 
  ViewPointerWrapper(const ViewPointerWrapper &r) = default;
  ViewPointerWrapper(ViewPointerWrapper &&r) = default;

  //Create the viewarray object from an array of views vin of size n generated on the host
  //Use inplace new to generate the array because views don't tend to have default constructors
  void assign(ViewMode mode, const ViewType &vin){
    placeData(mode, (ViewType*)&vin);
  }

  //Deallocation must be either manually called or use CPSautoView  
  void free(){
    freeData();
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
  size_t size() const{ return n; }
  
  size_t byte_size() const{ return n*sizeof(T); }

  //pinned memory has a faster copy as it avoids a host-side copy in Cuda, but it can take a while to allocate
  hostDeviceMirroredContainer(size_t n, bool use_pinned_mem = true): n(n), device_in_sync(true), host_in_sync(true), use_pinned_mem(use_pinned_mem){
    host = use_pinned_mem ? ((T*)pinned_alloc_check(128,byte_size())) : ((T*)memalign_check(128,byte_size())); 
#ifdef GPU_VEC
    device = (T*)device_alloc_check(byte_size());
#else
    device = host;
#endif
  }

  bool hostInSync() const{ return host_in_sync; }
  bool deviceInSync() const{ return device_in_sync; }
  
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
#ifdef GPU_VEC
#if defined(GRID_CUDA) || defined(GRID_HIP)
    if(!use_pinned_mem) ERR.General("hostDeviceMirroredContainer","asyncHostDeviceSync","Requires use of pinned memory");
#endif   
    if(!device_in_sync && !host_in_sync) ERR.General("hostDeviceMirroredContainer","asyncHostDeviceSync","Invalid state");
    if(!device_in_sync){
      copy_host_to_device_async(device,host,byte_size());
      device_in_sync = true;
    }
    if(!host_in_sync){
      copy_device_to_host_async(host,device,byte_size());
      host_in_sync = true;
    }
#endif
  }
  
  ~hostDeviceMirroredContainer(){
    if(use_pinned_mem) pinned_free(host);
    else free(host);
#ifdef GPU_VEC
    device_free(device);
#endif    
  }

  class View{
    T* ptr;
    size_t n;
  public:
    View(ViewMode mode, hostDeviceMirroredContainer &con): n(con.n){
      switch(mode){
      case HostRead:
	ptr = (T*)con.getHostReadPtr(); break;
      case HostWrite:
	ptr = (T*)con.getHostWritePtr(); break;
      case DeviceRead:
	ptr = (T*)con.getDeviceReadPtr(); break;
      case DeviceWrite:
	ptr = (T*)con.getDeviceWritePtr(); break;
      default:
	assert(0); break;
      };
    }
    View(const View &v)=default;
    View(View &&v)=default;

    accelerator_inline T& operator[](size_t i){ return ptr[i]; }
    accelerator_inline const T& operator[](size_t i) const{ return ptr[i]; }
    accelerator_inline size_t size() const{ return n; }
  };
  
  View view(ViewMode mode) const{
    return View(mode, const_cast<hostDeviceMirroredContainer&>(*this));
  }

};

//Current support only transfers to device
class asyncTransferManager{
  struct entry{
    void* to;
    void const* from;
    size_t bytes;    
  };
  std::list<entry> queue;

  std::thread *eng;
  bool lock;
  bool verbose;
public:
  asyncTransferManager(): lock(false), eng(nullptr), verbose(false){}

  void setVerbose(bool to){ verbose = to; }

  void enqueue(void* to, void const* from, size_t bytes){
    if(lock) ERR.General("asyncTransferManager","enqueue","Lock is engaged");
    entry e; e.to = to; e.from = from; e.bytes = bytes;
    queue.push_front(e);
  }
  void start(){
    lock = true;
    auto &q = queue;
    bool vrb = verbose;
    eng = new std::thread([&q,vrb]{
      //Find the largest entry for the pinned memory allocation
      size_t lrg=0;
      for(auto it=q.begin();it!=q.end();it++){
	lrg = std::max(lrg,it->bytes);
      }
      if(vrb) std::cout << Grid::GridLogMessage << "Allocating " << lrg << " bytes of pinned memory" << std::endl;
      void *pmem = pinned_alloc_check(128,lrg);

      int i=0;
      while(!q.empty()){
	entry e = q.back();
	if(vrb) std::cout << Grid::GridLogMessage << "Queue entry " << i << " of size " << e.bytes << std::endl;
	q.pop_back();
	memcpy(pmem,e.from,e.bytes);
	copy_host_to_device_async(e.to,pmem,e.bytes);
	//copy_host_to_device_async(e.to,e.from,e.bytes);
	//copy_host_to_device(e.to,e.from,e.bytes);
#ifdef GPU_VEC
	{
	  using namespace Grid;
	  acceleratorCopySynchronise(); //copyStream
	}
#endif
	++i;
      }
      pinned_free(pmem);
      if(vrb) std::cout << Grid::GridLogMessage << "Transfers complete" << std::endl;
    });
  
  }
  void wait(){
    if(!lock) return;
    eng->join(); //wait for the thread to finish its business
    delete eng; eng = nullptr;
    lock = false;
  }

  inline static asyncTransferManager & globalInstance(){
    static asyncTransferManager man;
    return man;
  }
  
};
    


CPS_END_NAMESPACE

#endif
