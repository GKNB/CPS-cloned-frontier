#ifndef _CPS_UTILS_GPU_H__
#define _CPS_UTILS_GPU_H__

#include<thread>

#include<config.h>
#include "template_wizardry.h"
#include "utils_malloc.h"
#include<util/time_cps.h>

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

inline void device_pin_memory(void const* ptr, size_t bytes){
#if defined(GRID_CUDA)
  assert(cudaHostRegister((void*)ptr,bytes,cudaHostRegisterDefault) == cudaSuccess);
#elif defined(GRID_HIP)
  assert(hipHostRegister((void*)ptr,bytes,hipHostRegisterDefault) == hipSuccess);
#endif
}
inline void device_unpin_memory(void const* ptr){
#if defined(GRID_CUDA)
  assert(cudaHostUnregister((void*)ptr) == cudaSuccess);
#elif defined(GRID_HIP)
  assert(hipHostUnregister((void*)ptr) == hipSuccess);
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

enum ViewMode { HostRead, HostWrite, DeviceRead, DeviceWrite, HostReadWrite, DeviceReadWrite };

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
    if(mode == DeviceRead || mode == DeviceWrite || mode == DeviceReadWrite){
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
    if(mode == DeviceRead || mode == DeviceWrite || mode == DeviceReadWrite){
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
      case HostReadWrite:
	con.getHostReadPtr();
	ptr = (T*)con.getHostWritePtr(); break;
      case DeviceReadWrite:
	con.getDeviceReadPtr();
	ptr = (T*)con.getDeviceWritePtr(); break;	
      default:
	assert(0); break;
      };
    }
    View(const View &v)=default;
    View(View &&v)=default;

    accelerator_inline T* data(){ return ptr; }
    accelerator_inline T const* data() const{ return ptr; }
    
    accelerator_inline T& operator[](size_t i){ return ptr[i]; }
    accelerator_inline const T& operator[](size_t i) const{ return ptr[i]; }
    accelerator_inline size_t size() const{ return n; }
    void free(){}
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
      double time = -dclock();
      size_t total_bytes =0;
      
      //Find the largest entry for the pinned memory allocation
      size_t lrg=0;
      for(auto it=q.begin();it!=q.end();it++){
	lrg = std::max(lrg,it->bytes);
	total_bytes += it->bytes;
      }
      //#define PIN_IN_PLACE    #Much slower and does not overlap with kernel
#ifndef PIN_IN_PLACE
      if(vrb) std::cout << Grid::GridLogMessage << "Allocating " << lrg << " bytes of pinned memory" << std::endl;
      void *pmem = pinned_alloc_check(128,lrg);
#endif
      
      int i=0;
      while(!q.empty()){
	entry e = q.back();
	if(vrb) std::cout << Grid::GridLogMessage << "Queue entry " << i << " of size " << e.bytes << std::endl;
	q.pop_back();
#ifndef PIN_IN_PLACE
	memcpy(pmem,e.from,e.bytes);
	copy_host_to_device_async(e.to,pmem,e.bytes);
#else
	device_pin_memory(e.from,e.bytes);
	copy_host_to_device_async(e.to,e.from,e.bytes);
#endif
	
	//copy_host_to_device_async(e.to,e.from,e.bytes);
	//copy_host_to_device(e.to,e.from,e.bytes);
#ifdef GPU_VEC
	{
	  using namespace Grid;
	  acceleratorCopySynchronise(); //copyStream
	}
#endif

#ifdef PIN_IN_PLACE
	device_unpin_memory(e.from);
#endif
	
	++i;
      }
      
#ifndef PIN_IN_PLACE
      pinned_free(pmem);
#endif           
     
      if(vrb){
	time += dclock();
	double total_MB = double(total_bytes)/1024./1024.;
	double rate = total_MB/time;
	std::cout << Grid::GridLogMessage << "Transfers complete bytes " << total_MB << " MB in " << time << "s : rate " << rate << "MB/s" << std::endl;
      }
#undef PIN_IN_PLACE
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


class DeviceMemoryPoolManager{
public:
  struct Handle;

  struct Entry{
    size_t bytes;
    void* ptr;
    Handle* owned_by;
  };
  typedef std::list<Entry>::iterator EntryIterator;

  struct Handle{
    bool valid;
    bool lock_entry;
    EntryIterator entry;
    size_t bytes;

    bool device_in_sync;
    bool host_in_sync;
    void* host_ptr;
  };

  typedef std::list<Handle>::iterator HandleIterator;

  bool verbose;
protected:

  std::list<Entry> in_use_pool; //LRU
  std::map<size_t,std::list<Entry>, std::greater<size_t> > free_pool; //sorted by size in descending order

  std::list<Handle> handles; //active fields

  std::list<HandleIterator> queued_prefetches; //track open prefetches
  
  size_t allocated;
  size_t pool_max_size;

  //Move the entry to the end and return a new iterator
  void touchEntry(EntryIterator entry){
    if(verbose) std::cout << "Touching entry " << entry->ptr << std::endl;
    in_use_pool.splice(in_use_pool.end(),in_use_pool,entry); //doesn't invalidate any iterators :)
  }

  EntryIterator evictEntry(EntryIterator entry, bool free_it){
    if(verbose) std::cout << "Evicting entry " << entry->ptr << std::endl;
	
    if(entry->owned_by != nullptr){
      if(verbose) std::cout << "Entry is owned by handle " << entry->owned_by << ", detaching" << std::endl;
      Handle &hown = *entry->owned_by;
      if(hown.lock_entry) ERR.General("DeviceMemoryPoolManager","evictEntry","Cannot evict a locked entry!");
      //Copy data back to host if not in sync
      if(!hown.host_in_sync){	
	if(verbose) std::cout << "Host is not in sync with device, copying back before detach" << std::endl;
	copy_device_to_host(hown.host_ptr,entry->ptr,hown.bytes);
	hown.host_in_sync = true;
      }
      hown.device_in_sync = false;
      entry->owned_by->valid = false; //evict
    }
    if(free_it){ 
      device_free(entry->ptr); allocated -= entry->bytes; 
      if(verbose) std::cout << "Freed memory " << entry->ptr << " of size " << entry->bytes << ". Allocated amount is now " << allocated << " vs max " << pool_max_size << std::endl;
    }
    return in_use_pool.erase(entry); //remove from list
  }

  void deallocateFreePool(size_t until_allocated_lte = 0){
    //Start from the largest    
    auto sit = free_pool.begin();
    while(sit != free_pool.end()){
      auto& entry_list = sit->second;

      auto it = entry_list.begin();
      while(it != entry_list.end()){
	device_free(it->ptr); allocated -= it->bytes;
	if(verbose) std::cout << "Freed memory " << it->ptr << " of size " << it->bytes << ". Allocated amount is now " << allocated << " vs max " << pool_max_size << std::endl;
	it = entry_list.erase(it);

	if(allocated <= until_allocated_lte){
	  if(entry_list.size() == 0) free_pool.erase(sit); //if we break out after draining the list for a particular size, we need to remove that list from the map
	  if(verbose) std::cout << "deallocateFreePool has freed enough memory" << std::endl;
	  return;
	}
      }
      sit = free_pool.erase(sit);
    }
    if(verbose) std::cout << "deallocateFreePool has freed all of its memory" << std::endl;
  }

  //Allocate a new entry of the given size and move to the end of the LRU queue, returning a pointer
  EntryIterator allocEntry(size_t bytes){
    Entry e;
    e.bytes = bytes;
    e.ptr = device_alloc_check(128,bytes);
    allocated += bytes;
    e.owned_by = nullptr;
    if(verbose) std::cout << "Allocated entry " << e.ptr << " of size " << bytes << ". Allocated amount is now " << allocated << " vs max " << pool_max_size << std::endl;
    return in_use_pool.insert(in_use_pool.end(),e);
  }    

  //Get an entry either new or from the pool
  //It will automatically be moved to the end of the in_use_pool list
  EntryIterator getEntry(size_t bytes){
    if(verbose) std::cout << "Getting an entry of size " << bytes << std::endl;
    if(bytes > pool_max_size) ERR.General("DeviceMemoryPoolManager","getEntry","Requested size is larger than the maximum pool size!");

    //First check if we have an entry of the right size in the pool
    auto fit = free_pool.find(bytes);
    if(fit != free_pool.end()){
      assert(fit->second.size() > 0);
      Entry e = fit->second.back();
      if(verbose) std::cout << "Found entry " << e.ptr << " in free pool" << std::endl;
      if(fit->second.size() == 1) free_pool.erase(fit); //remove the entire, now-empty list
      else fit->second.pop_back();
      return in_use_pool.insert(in_use_pool.end(),e);
    }
    //Next, if we have enough room, allocate new memory
    if(allocated + bytes <= pool_max_size){
      if(verbose) std::cout << "Allocating new memory for entry" << std::endl;
      return allocEntry(bytes);
    }
    //Next, we should free up unused blocks from the free pool
    if(verbose) std::cout << "Clearing up space from the free pool to make room" << std::endl;
    deallocateFreePool(pool_max_size - bytes);
    if(allocated + bytes <= pool_max_size){
      if(verbose) std::cout << "Allocating new memory for entry" << std::endl;
      return allocEntry(bytes);
    }

    //Evict old data until we have enough room
    //If we hit an entry with just the right size, reuse the pointer
    if(verbose) std::cout << "Evicting data to make room" << std::endl;
    auto it = in_use_pool.begin();
    while(it != in_use_pool.end()){
      if(verbose) std::cout << "Attempting to evict entry " << it->ptr << std::endl;

      if(it->owned_by->lock_entry){ //don't evict an entry that is currently in use
      	if(verbose) std::cout << "Entry is assigned to an open view or prefetch for handle " << it->owned_by << ", skipping" << std::endl;
      	++it;
      	continue;
      }

      bool erase = true;
      void* reuse;
      if(it->bytes == bytes){
	if(verbose) std::cout << "Found entry " << it->ptr << " has the right size, yoink" << std::endl;
	reuse = it->ptr;
	erase = false;
      }
      it = evictEntry(it, erase);

      if(!erase){
	if(verbose) std::cout << "Reusing memory " << reuse << std::endl;
	//reuse existing allocation
	Entry e;
	e.bytes = bytes;
	e.ptr = reuse;
	e.owned_by = nullptr;
	return in_use_pool.insert(in_use_pool.end(),e);
      }else if(allocated + bytes <= pool_max_size){ //allocate if we have enough room
	if(verbose) std::cout << "Memory available " << allocated << " is now sufficient, allocating" << std::endl;
	return allocEntry(bytes);
      }
    }	
    ERR.General("DeviceMemoryPoolManager","getEntry","Was not able to get an entry for %lu bytes",bytes);
  }


public:

  DeviceMemoryPoolManager(): allocated(0), pool_max_size(1024*1024*1024), verbose(false){}
  DeviceMemoryPoolManager(size_t max_size): DeviceMemoryPoolManager(){
    pool_max_size = max_size;
  }

  ~DeviceMemoryPoolManager(){
    auto it = in_use_pool.begin();
    while(it != in_use_pool.end()){
      it = evictEntry(it, true);
    }
    deallocateFreePool();
  }

  void setVerbose(bool to){ verbose = to; }

  //Set the pool max size. When the next eviction cycle happens the extra memory will be deallocated
  void setPoolMaxSize(size_t to){ pool_max_size = to; }

  size_t getAllocated() const{ return allocated; }

  HandleIterator allocate(size_t bytes){
    if(verbose) std::cout << "Request for allocation of size " << bytes << std::endl;
    Handle h;
    h.valid = true;
    h.entry = getEntry(bytes);
    h.bytes = bytes;
    h.host_ptr = memalign_check(128,bytes);
    h.host_in_sync = true;
    h.device_in_sync = true;
    h.lock_entry = false;

    HandleIterator it = handles.insert(handles.end(),h);
    it->entry->owned_by = &(*it);
    return it;
  }

  void* openView(ViewMode mode, HandleIterator h){
    h->lock_entry = true; //make sure it isn't evicted!
    bool is_host = (mode == HostRead || mode == HostWrite || mode == HostReadWrite);
    bool read(false), write(false);

    switch(mode){
    case HostRead:
    case DeviceRead:
      read=true; break;
    case HostWrite:
    case DeviceWrite:
      write=true; break;
    case HostReadWrite:
    case DeviceReadWrite:
      write=read=true; break;
    }

    if(is_host){
      if(read && !h->host_in_sync){
	if(!h->valid) ERR.General("DeviceMemoryPoolManager","openView","Host is not in sync but device side has been evicted!");
	copy_device_to_host(h->host_ptr,h->entry->ptr,h->bytes);
	h->host_in_sync=true;
      }
      if(write){
	h->host_in_sync = true;
	h->device_in_sync = false;
      }
      return h->host_ptr;
    }else{ //device
      if(h->valid){
	touchEntry(h->entry); //touch the entry and refresh the iterator
      }else{
	//find a new entry
	h->entry = getEntry(h->bytes);
	h->valid = true;
	h->entry->owned_by = &(*h);
	assert(!h->device_in_sync);
      }
      
      if(read && !h->device_in_sync){
	copy_host_to_device(h->entry->ptr,h->host_ptr,h->bytes);
	h->device_in_sync = true;
      }
      if(write){
	h->host_in_sync = false;
	h->device_in_sync = true;
      }
      return h->entry->ptr;
    }
    
  }

  void closeView(HandleIterator h){
    h->lock_entry = false;
  }

  void enqueuePrefetch(ViewMode mode, HandleIterator h){    
    if(mode == HostRead || mode == HostReadWrite){
      //no support for device->host async copies yet
    }else if(mode == DeviceRead || mode == DeviceReadWrite){
      if(h->valid){
	touchEntry(h->entry); //touch the entry to make it less likely to be evicted; benefit even if already in sync
      }else{
	//find a new entry
	h->entry = getEntry(h->bytes);
	h->valid = true;
	h->entry->owned_by = &(*h);
	assert(!h->device_in_sync);
      }
      if(!h->device_in_sync){
	asyncTransferManager::globalInstance().enqueue(h->entry->ptr,h->host_ptr,h->bytes);
	h->device_in_sync = true; //technically true only if the prefetch is complete; make sure to wait!!
	h->lock_entry = true; //use this flag also for prefetches to ensure the memory region is not evicted while the async copy is happening
	queued_prefetches.push_back(h);
      }
    }
  }

  void startPrefetches(){
    if(queued_prefetches.size()==0) return;
    asyncTransferManager::globalInstance().start();
  }
  void waitPrefetches(){
    if(queued_prefetches.size()==0) return;   
    asyncTransferManager::globalInstance().wait();
    for(auto h : queued_prefetches) h->lock_entry=false; //unlock
    queued_prefetches.clear();
  }
  
  void free(HandleIterator h){
    if(h->valid){
      if(verbose) std::cout << "Freeing ptr " << h->entry->ptr << " of size " << h->entry->bytes << " into free pool" << std::endl;
      //Remove entry from in-use pool
      Entry e = *(h->entry);
      in_use_pool.erase(h->entry);
      e.owned_by = nullptr;
      free_pool[e.bytes].push_back(e);
    }
    //Free host memory
    ::free(h->host_ptr);
    //Remove handle
    if(verbose) std::cout << "Freed host ptr " << h->host_ptr << ", removing handle" << std::endl;
    handles.erase(h);
  }

  inline static DeviceMemoryPoolManager & globalPool(){
    static DeviceMemoryPoolManager pool;
    return pool;
  }      

};





//3-level memory pool manager with device,host,disk storage locations with eviction possible from device, host
class HolisticMemoryPoolManager{
public:
  struct Handle;

  struct Entry{
    size_t bytes;
    void* ptr;
    Handle* owned_by;
  };
  typedef std::list<Entry>::iterator EntryIterator;

  struct Handle{
    bool lock_entry;

    bool device_valid;
    EntryIterator device_entry;

    bool host_valid;
    EntryIterator host_entry;

    size_t bytes;

    bool device_in_sync;
    bool host_in_sync;
    bool disk_in_sync;
    std::string disk_file;

    bool initialized; //keep track of whether the data has had a write
  };

  typedef std::list<Handle>::iterator HandleIterator;

  enum Pool { DevicePool, HostPool };
  
protected:
  bool verbose;
  std::list<Handle> handles; //active fields

  std::list<Entry> device_in_use_pool; //LRU
  std::map<size_t,std::list<Entry>, std::greater<size_t> > device_free_pool; //sorted by size in descending order
  std::list<HandleIterator> device_queued_prefetches; //track open prefetches to device from host

  std::list<Entry> host_in_use_pool; //LRU
  std::map<size_t,std::list<Entry>, std::greater<size_t> > host_free_pool; //sorted by size in descending order
  std::list<HandleIterator> host_queued_prefetches; //track open prefetches to host from disk

  inline std::list<Entry> & getLRUpool(Pool pool){ return pool == DevicePool ? device_in_use_pool : host_in_use_pool; }
  inline std::map<size_t,std::list<Entry>, std::greater<size_t> > & getFreePool(Pool pool){ return pool == DevicePool ? device_free_pool : host_free_pool; }
  inline std::string poolName(Pool pool){ return pool == DevicePool ? "DevicePool" : "HostPool"; }   
  
  size_t device_allocated;
  size_t host_allocated;
  size_t device_pool_max_size;
  size_t host_pool_max_size;

  std::string disk_root; //root location for temp files, default "."
  
  //Allocate a new entry of the given size and move to the end of the LRU queue, returning a pointer
  EntryIterator allocEntry(size_t bytes, Pool pool){
    Entry e;
    e.bytes = bytes;
    e.owned_by = nullptr;
    if(pool == DevicePool){
      e.ptr = device_alloc_check(128,bytes);
      device_allocated += bytes;
      if(verbose) std::cout << "HolisticMemoryPoolManager: Allocated device entry " << e.ptr << " of size " << bytes << ". Allocated amount is now " << device_allocated << " vs max " << device_pool_max_size << std::endl;
    }else{ //HostPool
      e.ptr = memalign_check(128,bytes);
      host_allocated += bytes;
      if(verbose) std::cout << "HolisticMemoryPoolManager: Allocated host entry " << e.ptr << " of size " << bytes << ". Allocated amount is now " << host_allocated << " vs max " << host_pool_max_size << std::endl;
    }
    auto &p = getLRUpool(pool);
    return p.insert(p.end(),e);
  }

  //Relinquish the entry from the LRU and put in the free pool
  void moveEntryToFreePool(EntryIterator it, Pool pool){
    if(verbose) std::cout << "HolisticMemoryPoolManager: Relinquishing " << it->ptr << " of size " << it->bytes << " from " << poolName(pool) << std::endl;
    Entry e = *it;
    getLRUpool(pool).erase(it);
    e.owned_by = nullptr;
    getFreePool(pool)[e.bytes].push_back(e);
  }
  
  //Free the memory associated with an entry
  void freeEntry(EntryIterator it, Pool pool){       
    if(pool == DevicePool){
      device_free(it->ptr); device_allocated -= it->bytes;
      if(verbose) std::cout << "HolisticMemoryPoolManager: Freed device memory " << it->ptr << " of size " << it->bytes << ". Allocated amount is now " << device_allocated << " vs max " << device_pool_max_size << std::endl;
    }else{ //DevicePool
      ::free(it->ptr); host_allocated -= it->bytes;
      if(verbose) std::cout << "HolisticMemoryPoolManager: Freed host memory " << it->ptr << " of size " << it->bytes << ". Allocated amount is now " << host_allocated << " vs max " << host_pool_max_size << std::endl;	
    }
  }
  
  void deallocateFreePool(Pool pool, size_t until_allocated_lte = 0){
    size_t &allocated = (pool == DevicePool ? device_allocated : host_allocated);
    if(verbose) std::cout << "HolisticMemoryPoolManager: Deallocating free " << poolName(pool) << " until " << until_allocated_lte << " remaining. Current " << allocated << std::endl;
    auto &free_pool = getFreePool(pool);
    
    //Start from the largest    
    auto sit = free_pool.begin();
    while(sit != free_pool.end()){
      auto& entry_list = sit->second;

      auto it = entry_list.begin();
      while(it != entry_list.end()){
	freeEntry(it, pool);
	it = entry_list.erase(it);

	if(allocated <= until_allocated_lte){
	  if(entry_list.size() == 0) free_pool.erase(sit); //if we break out after draining the list for a particular size, we need to remove that list from the map
	  if(verbose) std::cout << "HolisticMemoryPoolManager: deallocateFreePool has freed enough memory" << std::endl;
	  return;
	}
      }
      sit = free_pool.erase(sit);
    }
    if(verbose) std::cout << "HolisticMemoryPoolManager: deallocateFreePool has freed all of its memory" << std::endl;
  }

  //Get an entry either new or from the pool
  //It will automatically be moved to the end of the in_use_pool list
  EntryIterator getEntry(size_t bytes, Pool pool){
    if(verbose) std::cout << "HolisticMemoryPoolManager: Getting an entry of size " << bytes << " from " << poolName(pool) << std::endl;
    size_t pool_max_size = ( pool == DevicePool ? device_pool_max_size : host_pool_max_size );
    size_t &allocated = (pool == DevicePool ? device_allocated : host_allocated);
    
    if(bytes > pool_max_size) ERR.General("HolisticMemoryPoolManager","getEntry","Requested size is larger than the maximum pool size!");

    auto &LRUpool = getLRUpool(pool);
    auto &free_pool = getFreePool(pool);
    
    //First check if we have an entry of the right size in the pool
    auto fit = free_pool.find(bytes);
    if(fit != free_pool.end()){
      assert(fit->second.size() > 0);
      Entry e = fit->second.back();
      if(verbose) std::cout << "HolisticMemoryPoolManager: Found entry " << e.ptr << " in free pool" << std::endl;
      if(fit->second.size() == 1) free_pool.erase(fit); //remove the entire, now-empty list
      else fit->second.pop_back();
      return LRUpool.insert(LRUpool.end(),e);
    }
    //Next, if we have enough room, allocate new memory
    if(allocated + bytes <= pool_max_size){
      if(verbose) std::cout << "HolisticMemoryPoolManager: Allocating new memory for entry" << std::endl;
      return allocEntry(bytes, pool);
    }
    //Next, we should free up unused blocks from the free pool
    if(verbose) std::cout << "HolisticMemoryPoolManager: Clearing up space from the free pool to make room" << std::endl;
    deallocateFreePool(pool, pool_max_size - bytes);
    if(allocated + bytes <= pool_max_size){
      if(verbose) std::cout << "HolisticMemoryPoolManager: Allocating new memory for entry" << std::endl;
      return allocEntry(bytes, pool);
    }

    //Evict old data until we have enough room
    //If we hit an entry with just the right size, reuse the pointer
    if(verbose) std::cout << "HolisticMemoryPoolManager: Evicting data to make room" << std::endl;
    auto it = LRUpool.begin();
    while(it != LRUpool.end()){
      if(verbose) std::cout << "HolisticMemoryPoolManager: Attempting to evict entry " << it->ptr << std::endl;

      if(it->owned_by->lock_entry){ //don't evict an entry that is currently in use
      	if(verbose) std::cout << "HolisticMemoryPoolManager: Entry is assigned to an open view or prefetch for handle " << it->owned_by << ", skipping" << std::endl;
      	++it;
      	continue;
      }

      bool erase = true;
      void* reuse;
      if(it->bytes == bytes){
	if(verbose) std::cout << "HolisticMemoryPoolManager: Found entry " << it->ptr << " has the right size, yoink" << std::endl;
	reuse = it->ptr;
	erase = false;
      }
      it = evictEntry(it, erase, pool);

      if(!erase){
	if(verbose) std::cout << "HolisticMemoryPoolManager: Reusing memory " << reuse << std::endl;
	//reuse existing allocation
	Entry e;
	e.bytes = bytes;
	e.ptr = reuse;
	e.owned_by = nullptr;
	return LRUpool.insert(LRUpool.end(),e);
      }else if(allocated + bytes <= pool_max_size){ //allocate if we have enough room
	if(verbose) std::cout << "HolisticMemoryPoolManager: Memory available " << allocated << " is now sufficient, allocating" << std::endl;
	return allocEntry(bytes,pool);
      }
    }	
    ERR.General("HolisticMemoryPoolManager","getEntry","Was not able to get an entry for %lu bytes",bytes);
  }

  void attachEntry(Handle &handle, Pool pool){
    if(pool == DevicePool){
      handle.device_entry = getEntry(handle.bytes, DevicePool);
      handle.device_valid = true;
      handle.device_entry->owned_by = &handle;
    }else{
      handle.host_entry = getEntry(handle.bytes, HostPool);
      handle.host_valid = true;
      handle.host_entry->owned_by = &handle;
    }
  }
  //Move the entry to the end and return a new iterator
  void touchEntry(Handle &handle, Pool pool){
    EntryIterator entry = pool == DevicePool ? handle.device_entry : handle.host_entry;
    if(verbose) std::cout << "HolisticMemoryPoolManager: Touching entry " << entry->ptr << " in " << poolName(pool) << std::endl;
    auto &p = getLRUpool(pool);
    p.splice(p.end(),p,entry); //doesn't invalidate any iterators :)
  }
  
  void syncDeviceToHost(Handle &handle){
    assert(handle.initialized);
    if(!handle.host_in_sync){
      assert(handle.device_in_sync && handle.device_valid);
      if(!handle.host_valid) attachEntry(handle, HostPool);
      if(verbose) std::cout << "HolisticMemoryPoolManager: Synchronizing device " << handle.device_entry->ptr << " to host " << handle.host_entry->ptr << std::endl;
      copy_device_to_host(handle.host_entry->ptr, handle.device_entry->ptr, handle.bytes);
      handle.host_in_sync = true;
    }
  }
  void syncHostToDevice(Handle &handle){
    assert(handle.initialized);
    if(!handle.device_in_sync){
      assert(handle.host_in_sync && handle.host_valid);
      if(!handle.device_valid) attachEntry(handle, DevicePool);
      if(verbose) std::cout << "HolisticMemoryPoolManager: Synchronizing host " << handle.host_entry->ptr << " to device " << handle.device_entry->ptr << std::endl;
      copy_host_to_device(handle.device_entry->ptr, handle.host_entry->ptr, handle.bytes);
      handle.device_in_sync = true;
    }
  }  
  void syncHostToDisk(Handle &handle){
    assert(handle.initialized);
    if(!handle.disk_in_sync){
      assert(handle.host_in_sync && handle.host_valid);
      static size_t idx = 0;
      if(handle.disk_file == ""){
	handle.disk_file = disk_root + "/mempool." + std::to_string(UniqueID()) + "." + std::to_string(idx++);
      }
      if(verbose) std::cout << "HolisticMemoryPoolManager: Synchronizing host " << handle.host_entry->ptr << " to disk " << handle.disk_file << std::endl;
      std::fstream f(handle.disk_file.c_str(), std::ios::out | std::ios::binary);
      if(!f.good()) ERR.General("HolisticMemoryPoolManager","syncHostToDisk","Failed to open file %s for write\n",handle.disk_file.c_str());
      f.write((char*)handle.host_entry->ptr, handle.bytes);
      if(!f.good()) ERR.General("HolisticMemoryPoolManager","syncHostToDisk","Write error in file %s\n",handle.disk_file.c_str());
      f.flush(); //should ensure data is written to disk immediately and not kept around in some memory buffer, but may slow things down

      handle.disk_in_sync = true;
    }
  }
  void syncDiskToHost(Handle &handle){
    assert(handle.initialized);
    if(!handle.host_in_sync){
      assert(handle.disk_in_sync && handle.disk_file != "");
      if(!handle.host_valid) attachEntry(handle, HostPool);      
      if(verbose) std::cout << "HolisticMemoryPoolManager: Synchronizing disk " << handle.disk_file << " to host " << handle.host_entry->ptr << std::endl;
      std::fstream f(handle.disk_file.c_str(), std::ios::in | std::ios::binary);
      if(!f.good()) ERR.General("HolisticMemoryPoolManager","syncDiskToHost","Failed to open file %s for write\n",handle.disk_file.c_str());
      f.read((char*)handle.host_entry->ptr, handle.bytes);
      if(!f.good()) ERR.General("HolisticMemoryPoolManager","syncDiskToHost","Write error in file %s\n",handle.disk_file.c_str());
      handle.host_in_sync = true;
    }
  }

  void syncForRead(Handle &handle, Pool pool){
    if(pool == HostPool){    
      if(!handle.host_in_sync){
	if(handle.device_in_sync) syncDeviceToHost(handle);
	else if(handle.disk_in_sync) syncDiskToHost(handle);
	else if(handle.initialized) ERR.General("HolisticMemoryPoolManager","syncForRead (HostRead)","Data has been initialized but no active copy!");
	//Allow copies from uninitialized data, eg in copy constructor called during initialization of vector of fields
      }
    }else{ //DevicePool
      if(!handle.device_in_sync){	      
	if(handle.host_in_sync) syncHostToDevice(handle);
	else if(handle.disk_in_sync){
	  syncDiskToHost(handle);
	  syncHostToDevice(handle);
	}
	else if(handle.initialized) ERR.General("HolisticMemoryPoolManager","syncForRead (DeviceRead)","Data has been initialized but no active copy!");
      }
    }
  }

  void markForWrite(Handle &handle, Pool pool){
    if(pool == HostPool){
      handle.host_in_sync = true;
      handle.device_in_sync = false;
      handle.disk_in_sync = false;
      handle.initialized = true;
    }else{ //DevicePool
      handle.host_in_sync = false;
      handle.device_in_sync = true;
      handle.disk_in_sync = false;
      handle.initialized = true;
    }
  }   

  void prepareEntryForView(Handle &handle, Pool pool){
    bool valid = pool == DevicePool ? handle.device_valid : handle.host_valid;
    if(!valid) attachEntry(handle,pool);
    else touchEntry(handle, pool); //move to end of LRU
  }
 
  //Evict an entry (with optional freeing of associated memory), and return an entry to the next item in the LRU
  EntryIterator evictEntry(EntryIterator entry, bool free_it, Pool pool){
    if(verbose) std::cout << "HolisticMemoryPoolManager: Evicting entry " << entry->ptr << " from " << poolName(pool) << std::endl;
	
    if(entry->owned_by != nullptr){
      if(verbose) std::cout << "HolisticMemoryPoolManager: Entry is owned by handle " << entry->owned_by << ", detaching" << std::endl;
      Handle &handle = *entry->owned_by;
      if(handle.lock_entry) ERR.General("DeviceMemoryPoolManager","evictEntry","Cannot evict a locked entry!");

      if(pool == DevicePool){
	//Copy data back to host if not in sync
	if(handle.device_in_sync && !handle.host_in_sync){
	  if(verbose) std::cout << "HolisticMemoryPoolManager: Host is not in sync with device, copying back before detach" << std::endl;
	  syncDeviceToHost(handle);
	}
	handle.device_entry->owned_by = nullptr;
	handle.device_in_sync = false;      
	handle.device_valid = false; //evict
      }else{
	//Copy data to disk if not in sync
	if(handle.host_in_sync && !handle.disk_in_sync){
	  if(verbose) std::cout << "HolisticMemoryPoolManager: Disk is not in sync with device, copying back before detach" << std::endl;
	  syncHostToDisk(handle);
	}
	handle.host_entry->owned_by = nullptr;
	handle.host_in_sync = false;
	handle.host_valid = false; //evict
      }    
    }
    if(free_it) freeEntry(entry, pool); //deallocate the memory entirely (optional, we might want to reuse it)
    return getLRUpool(pool).erase(entry); //remove from list
  }

public:

  HolisticMemoryPoolManager(): device_allocated(0), device_pool_max_size(1024*1024*1024), host_allocated(0), host_pool_max_size(1024*1024*1024), verbose(false), disk_root("."){}
  HolisticMemoryPoolManager(size_t max_size_device, size_t max_size_host): HolisticMemoryPoolManager(){
    device_pool_max_size = max_size_device;
    host_pool_max_size = max_size_host;
  }

  ~HolisticMemoryPoolManager(){
    std::cout << "~HolisticMemoryPoolManager handles.size()=" << handles.size() << " device_in_use_pool.size()=" << device_in_use_pool.size() << " host_in_use_pool.size()=" << host_in_use_pool.size() << std::endl;
    auto it = device_in_use_pool.begin();
    while(it != device_in_use_pool.end()){
      freeEntry(it, DevicePool);
      it = device_in_use_pool.erase(it);
    }
    it = host_in_use_pool.begin();
    while(it != host_in_use_pool.end()){
      freeEntry(it, HostPool);
      it = host_in_use_pool.erase(it);
    }
    deallocateFreePool(HostPool);
    deallocateFreePool(DevicePool);
  }

  void setVerbose(bool to){ verbose = to; }

  void setDiskRoot(const std::string &to){ disk_root = to; }
  
  //Set the pool max size. When the next eviction cycle happens the extra memory will be deallocated
  void setPoolMaxSize(size_t to, Pool pool){
    auto &m = (pool == DevicePool ? device_pool_max_size : host_pool_max_size );
    m = to;
  }

  size_t getAllocated(Pool pool) const{ return pool == DevicePool ? device_allocated : host_allocated; }

  HandleIterator allocate(size_t bytes, Pool pool = DevicePool){
    if(verbose) std::cout << "HolisticMemoryPoolManager: Request for allocation of size " << bytes << " in " << poolName(pool) << std::endl;

    HandleIterator it = handles.insert(handles.end(),Handle());
    Handle &h = *it;
    h.host_in_sync = false;
    h.device_in_sync = false;
    h.disk_in_sync = false;
    h.lock_entry = false;
    h.device_valid = false;
    h.host_valid = false;
    h.bytes = bytes;
    h.disk_file = "";
    h.initialized = false;
    attachEntry(h,pool);
    return it;
  }

  void* openView(ViewMode mode, HandleIterator h){
    h->lock_entry = true; //make sure it isn't evicted!
    Pool pool = (mode == HostRead || mode == HostWrite || mode == HostReadWrite) ? HostPool : DevicePool;   
    prepareEntryForView(*h,pool); 
    bool read(false), write(false);
    
    switch(mode){
    case HostRead:
    case DeviceRead:
      read=true; break;
    case HostWrite:
    case DeviceWrite:
      write=true; break;
    case HostReadWrite:
    case DeviceReadWrite:
      write=read=true; break;
    }

    if(read) syncForRead(*h,pool);
    if(write) markForWrite(*h,pool);

    return pool == HostPool ? h->host_entry->ptr : h->device_entry->ptr;
  }  

  void closeView(HandleIterator h){
    h->lock_entry = false;
  }

  void enqueuePrefetch(ViewMode mode, HandleIterator h){    
    if(mode == HostRead || mode == HostReadWrite){
      //no support for device->host async copies yet
    }else if( (mode == DeviceRead || mode == DeviceReadWrite) && h->host_valid && h->host_in_sync){
      prepareEntryForView(*h,DevicePool); 
      if(!h->device_in_sync){
	asyncTransferManager::globalInstance().enqueue(h->device_entry->ptr,h->host_entry->ptr,h->bytes);
	h->device_in_sync = true; //technically true only if the prefetch is complete; make sure to wait!!
	h->lock_entry = true; //use this flag also for prefetches to ensure the memory region is not evicted while the async copy is happening
	device_queued_prefetches.push_back(h);
      }
    }
  }

  void startPrefetches(){
    if(device_queued_prefetches.size()==0) return;
    asyncTransferManager::globalInstance().start();
  }
  void waitPrefetches(){
    if(device_queued_prefetches.size()==0) return;   
    asyncTransferManager::globalInstance().wait();
    for(auto h : device_queued_prefetches) h->lock_entry=false; //unlock
    device_queued_prefetches.clear();
  }
  
  void free(HandleIterator h){
    if(h->device_valid) moveEntryToFreePool(h->device_entry, DevicePool);
    if(h->host_valid) moveEntryToFreePool(h->host_entry, HostPool);
    if(h->disk_file != ""){
      if(verbose) std::cout << "HolisticMemoryPoolManager: Erasing cache file " << h->disk_file << std::endl;
      remove(h->disk_file.c_str());
    }
    handles.erase(h);
  }

  inline static HolisticMemoryPoolManager & globalPool(){
    static HolisticMemoryPoolManager pool;
    return pool;
  }      

};



CPS_END_NAMESPACE

#endif
