#ifndef _UTILS_MALLOC_H_
#define _UTILS_MALLOC_H_

#include <malloc.h>
#include <fstream>
#include <vector>

#include "utils_memory.h"

//Allocators and mallocators!

CPS_START_NAMESPACE

//Align a pointer. Make sure it has buffer space of at least alignment-1 bytes
inline void* aligned_ptr(void* p, size_t alignment){
  uintptr_t num = (uintptr_t)p;
  num = (num + (alignment - 1) ) & ~(alignment - 1);
  return (void*)num;
}

//Check if a pointer is aligned
inline bool isAligned(void *p, const size_t alignment){
  return (((uintptr_t)p) % alignment == 0 );
}

//memalign with alloc check
inline void* memalign_check(const size_t align, const size_t sz){
#define MEMALIGN_CHECK_USE_POSIX_MEMALIGN
#ifdef MEMALIGN_CHECK_USE_POSIX_MEMALIGN
  void* p;
  int err = posix_memalign(&p, align, sz);
  if(err){ 
    std::string errnm;
    switch(err){
    case EINVAL:
      errnm = "EINVAL"; break;
    case ENOMEM:
      errnm = "ENOMEM"; break;
    default:
      errnm = "UNKNOWN"; break; //this shouldn't happen
    }
    printf("memalign_check alloc of alignment %d and size %f MB failed on node %d with error %s. Stack trace:\n", 
	   align, double(sz)/1024./1024., UniqueID(), errnm.c_str());
    printBacktrace(std::cout);
    printMem("Memory status on fail", UniqueID());
    std::cout.flush(); fflush(stdout);
    ERR.General("","memalign_check","Mem alloc failed\n");
  }
#else
  void* p = memalign(align,sz);
  if(p == NULL){ 
    printf("memalign_check alloc of alignment %d and size %f MB failed on node %d. Stack trace:\n", align, double(sz)/1024./1024., UniqueID());
    printBacktrace(std::cout);
    printMem("Memory status on fail", UniqueID());
    std::cout.flush(); fflush(stdout);
    ERR.General("","memalign_check","Mem alloc failed\n");
  }
#endif
#undef MEMALIGN_CHECK_USE_POSIX_MEMALIGN
  return p;
}

//malloc with alloc check
inline void* malloc_check(const size_t sz){
  void* p = malloc(sz);
  if(p == NULL){ 
    printf("malloc_check alloc of size %f MB failed on node %d. Stack trace:\n", double(sz)/1024./1024., UniqueID());
    perror("reason");
    printBacktrace(std::cout);
    printMem("Memory status on fail", UniqueID());
    std::cout.flush(); fflush(stdout);
    ERR.General("","malloc_check","Mem alloc failed\n");
  }
  return p;
}

inline void* mmap_alloc_check(const size_t align, const size_t byte_size){
  assert(align > 0);

  //Need space for an extra size_t for length, a ptrdiff_t
  size_t asize = sizeof(size_t) + sizeof(ptrdiff_t) + align + byte_size;
  void *p = mmap (NULL, asize, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, (off_t)0);
  if(p == MAP_FAILED){
    printf("mmap_alloc_check alloc of alignment %d and size %f MB failed on node %d. Stack trace:\n", align, double(asize)/1024./1024., UniqueID());
    perror("reason");
    printBacktrace(std::cout);
    printMem("Memory status on fail", UniqueID());
    std::cout.flush(); fflush(stdout);
    ERR.General("","mmap_alloc_check","Mem alloc failed\n");
  }

  char* v = (char*)p;
  *( (size_t*)v ) = asize;
  v += sizeof(size_t) + sizeof(ptrdiff_t);
  
  char* valigned = (char*)aligned_ptr(v, align);
  ptrdiff_t vshift = valigned - v;

  v = valigned - sizeof(ptrdiff_t);
  *( (ptrdiff_t*)v ) = vshift;
 
  //std::cout << "mmap_alloc_check(" << align << "," << byte_size << ") actual size " << asize << " base ptr " << p << " aligned ptr " << valigned << " alignment shift " << vshift << std::endl;
  return valigned;
}
inline void mmap_free(void* pv){
  char* c = (char*)pv;
  c -= sizeof(ptrdiff_t);
  ptrdiff_t vshift = *( (ptrdiff_t*)c );
  c -= vshift;
  c -= sizeof(size_t);
  size_t asize = *( (size_t*)c );
  
  //std::cout << "mmap_free(" << pv << ") actual size " << asize << " base ptr "<<  (void*)c << " alignment shift " << vshift << std::endl;

  if( munmap(c, asize) != 0){
    ERR.General("","mmap_free","Mem free failed\n");
  }
}


//Note, CUDA does not allow alignment; apparently it is always "sufficient"
inline void* managed_alloc_check(const size_t align, const size_t byte_size){
#ifdef GPU_VEC
  return Grid::acceleratorAllocShared(byte_size);
#else
  return memalign_check(align, byte_size);
#endif
}

inline void* managed_alloc_check(const size_t byte_size){
#ifdef GPU_VEC
  return Grid::acceleratorAllocShared(byte_size);
#else
  return malloc_check(byte_size);
#endif
}

inline void managed_free(void* p){
#ifdef GPU_VEC
  Grid::acceleratorFreeShared(p);
#else
  free(p);
#endif
}



//Allocate memory on GPU device if in use; otherwise do memalign
inline void* device_alloc_check(const size_t align, const size_t byte_size){
#ifdef GPU_VEC
  return Grid::acceleratorAllocDevice(byte_size);
#else
  return memalign_check(align, byte_size);
#endif
}

inline void* device_alloc_check(const size_t byte_size){
#ifdef GPU_VEC
  return Grid::acceleratorAllocDevice(byte_size);
#else
  return malloc_check(byte_size);
#endif
}

inline void device_free(void* p){
#ifdef GPU_VEC
  Grid::acceleratorFreeDevice(p);
#else
  free(p);
#endif
}






//Allocate pinned memory on host if in use; otherwise do memalign
inline void* pinned_alloc_check(const size_t align, const size_t byte_size){
#if defined(GRID_CUDA)
  void *p;
  auto err = cudaMallocHost(&p,byte_size);
  if( err != cudaSuccess ) {
    p = (void*)NULL;
    std::cerr << "pinned_alloc_check: cudaHostMalloc failed for " << byte_size<<" bytes " <<cudaGetErrorString(err)<< std::endl;
    printMem("malloc failed",UniqueID());
    assert(0);
  }
  return p;
#elif defined(GRID_HIP)
  void *p;
  auto err = hipHostMalloc(&p,byte_size,hipHostMallocDefault);
  if( err != hipSuccess ) {
    p = (void*)NULL;
    std::cerr << "pinned_alloc_check: hipHostMalloc failed for " << byte_size<<" bytes " <<hipGetErrorString(err)<< std::endl;
    printMem("malloc failed",UniqueID());
    assert(0);
  }
  return p;
#else 
  return memalign_check(align, byte_size);
#endif
}

inline void* pinned_alloc_check(const size_t byte_size){
#if defined(GRID_CUDA)
  void *p;
  auto err = cudaMallocHost(&p,byte_size);
  if( err != cudaSuccess ) {
    p = (void*)NULL;
    std::cerr << "device_alloc_check: cudaHostMalloc failed for " << byte_size<<" bytes " <<cudaGetErrorString(err)<< std::endl;
    printMem("malloc failed",UniqueID());
    assert(0);
  }
  return p;
#elif defined(GRID_HIP)
  void *p;
  auto err = hipHostMalloc(&p,byte_size,hipHostMallocDefault);
  if( err != hipSuccess ) {
    p = (void*)NULL;
    std::cerr << "device_alloc_check: hipHostMalloc failed for " << byte_size<<" bytes " <<hipGetErrorString(err)<< std::endl;
    printMem("malloc failed",UniqueID());
    assert(0);
  }
  return p;
#else
  return malloc_check(byte_size);
#endif
}

inline void pinned_free(void* p){
#if defined(GRID_CUDA)
  auto err = cudaFreeHost(p);
  if( err != cudaSuccess ) {
    std::cerr << "pinned_free: cudaFree failed with error " <<cudaGetErrorString(err)<< std::endl;
    assert(0);
  }  
#elif defined(GRID_HIP)
  auto err = hipHostFree(p);	//Here hipFreeHost is deprecated, and hipHostFree does an implicit deviceSynchronize
  if( err != hipSuccess ) {
    std::cerr << "pinned_free: hipFree failed with error " <<hipGetErrorString(err)<< std::endl;
    assert(0);
  }  
#else
  free(p);
#endif
}




//Allocate mapped memory on host and device (if CUDA/HIP, otherwise do memalign)
inline void mapped_alloc_check(void** hostptr, void **deviceptr,  const size_t align, const size_t byte_size){
#if defined(GRID_CUDA)
  auto err1 = cudaHostAlloc(hostptr,byte_size,cudaHostAllocMapped);
  auto err2 = cudaHostGetDevicePointer(deviceptr, *hostptr, 0);	

  if( err1 != cudaSuccess || err2 != cudaSuccess ) {
    *hostptr = (void*)NULL;
    *deviceptr = (void*)NULL;
    std::cerr << "mapped_alloc_check: cudaMallocHost failed for " << byte_size<<" bytes " <<cudaGetErrorString(err1)<< " " << cudaGetErrorString(err2) << std::endl;
    printMem("malloc failed",UniqueID());
    assert(0);
  }
#elif defined(GRID_HIP)
  //FIXME: This version needs testing to see if it works as imagined. The official documentation is stupid
  auto err1 = hipHostMalloc(hostptr,byte_size,hipHostMallocMapped);
  auto err2 = hipHostGetDevicePointer(deviceptr, *hostptr, 0);	

  if( err1 != hipSuccess || err2 != hipSuccess ) {
    *hostptr = (void*)NULL;
    *deviceptr = (void*)NULL;
    std::cerr << "mapped_alloc_check: hipMallocHost failed for " << byte_size<<" bytes " <<hipGetErrorString(err1)<< " " << hipGetErrorString(err2) << std::endl;
    printMem("malloc failed",UniqueID());
    assert(0);
  }
#else
  *hostptr = memalign_check(align, byte_size);
  *deviceptr = *hostptr;
#endif
}

inline void mapped_alloc_check(void** hostptr, void **deviceptr, const size_t byte_size){
#if defined(GRID_CUDA)
  auto err1 = cudaHostAlloc(hostptr,byte_size,cudaHostAllocMapped);
  auto err2 = cudaHostGetDevicePointer(deviceptr, *hostptr, 0);	

  if( err1 != cudaSuccess || err2 != cudaSuccess ) {
    *hostptr = (void*)NULL;
    *deviceptr = (void*)NULL;
    std::cerr << "mapped_alloc_check: cudaMallocHost failed for " << byte_size<<" bytes " <<cudaGetErrorString(err1)<< " " << cudaGetErrorString(err2) << std::endl;
    printMem("malloc failed",UniqueID());
    assert(0);
  }
#elif defined(GRID_HIP)
  //FIXME: This version needs testing to see if it works as imagined. The official documentation is stupid
  auto err1 = hipHostMalloc(hostptr,byte_size,hipHostMallocMapped);
  auto err2 = hipHostGetDevicePointer(deviceptr, *hostptr, 0);	

  if( err1 != hipSuccess || err2 != hipSuccess ) {
    *hostptr = (void*)NULL;
    *deviceptr = (void*)NULL;
    std::cerr << "mapped_alloc_check: hipMallocHost failed for " << byte_size<<" bytes " <<hipGetErrorString(err1)<< " " << hipGetErrorString(err2) << std::endl;
    printMem("malloc failed",UniqueID());
    assert(0);
  }
#else
  *hostptr = malloc_check(byte_size);
  *deviceptr = *hostptr;
#endif
}



inline void mapped_free(void* hostptr){
#if defined(GRID_CUDA)
  auto err = cudaFreeHost(hostptr);
  if( err != cudaSuccess ) {
    std::cerr << "mapped_free: cudaFree failed with error " <<cudaGetErrorString(err)<< std::endl;
    assert(0);
  }  
#elif defined(GRID_HIP)
  auto err = hipHostFree(hostptr);
  if( err != hipSuccess ) {
    std::cerr << "mapped_free: hipFree failed with error " <<hipGetErrorString(err)<< std::endl;
    assert(0);
  }  
#else
  free(hostptr);
#endif
}









//Simple test  standard library allocator to find out when memory is allocated
template <typename T>
class mmap_allocator: public std::allocator<T>{
public:
  typedef size_t size_type;
  typedef T* pointer;
  typedef const T* const_pointer;

  template<typename _Tp1>
  struct rebind{
    typedef mmap_allocator<_Tp1> other;
  };

  pointer allocate(size_type n, const void *hint=0){
    fprintf(stderr, "Alloc %d bytes.\n", n*sizeof(T));
    return std::allocator<T>::allocate(n, hint);
  }

  void deallocate(pointer p, size_type n){
    fprintf(stderr, "Dealloc %d bytes (%p).\n", n*sizeof(T), p);
    return std::allocator<T>::deallocate(p, n);
  }

  mmap_allocator() throw(): std::allocator<T>() { fprintf(stderr, "Hello allocator!\n"); }
  mmap_allocator(const mmap_allocator &a) throw(): std::allocator<T>(a) { }
  template <class U>                    
  mmap_allocator(const mmap_allocator<U> &a) throw(): std::allocator<T>(a) { }
  ~mmap_allocator() throw() { }
};



//A standard library allocator for aligned memory
template<typename _Tp>
class BasicAlignedAllocator {
public: 
  typedef std::size_t     size_type;
  typedef std::ptrdiff_t  difference_type;
  typedef _Tp*       pointer;
  typedef const _Tp* const_pointer;
  typedef _Tp&       reference;
  typedef const _Tp& const_reference;
  typedef _Tp        value_type;

  template<typename _Tp1>  struct rebind { typedef BasicAlignedAllocator<_Tp1> other; };
  BasicAlignedAllocator() throw() { }
  BasicAlignedAllocator(const BasicAlignedAllocator&) throw() { }
  template<typename _Tp1> BasicAlignedAllocator(const BasicAlignedAllocator<_Tp1>&) throw() { }
  ~BasicAlignedAllocator() throw() { }
  pointer       address(reference __x)       const { return &__x; }
  size_type  max_size() const throw() { return size_t(-1) / sizeof(_Tp); }

  pointer allocate(size_type __n, const void* _p= 0)
  { 
    size_type bytes = __n*sizeof(_Tp);
    pointer ptr = (pointer) memalign_check(128,bytes);
    return ptr;
  }

  void deallocate(pointer __p, size_type __n) { 
    free((void *)__p);
  }
  void construct(pointer __p, const _Tp& __val) { new((void *)__p) _Tp(__val); };
  void construct(pointer __p) { new((void *)__p) _Tp();  };
  void destroy(pointer __p) { ((_Tp*)__p)->~_Tp(); };
};
template<typename _Tp>  inline bool operator==(const BasicAlignedAllocator<_Tp>&, const BasicAlignedAllocator<_Tp>&){ return true; }
template<typename _Tp>  inline bool operator!=(const BasicAlignedAllocator<_Tp>&, const BasicAlignedAllocator<_Tp>&){ return false; }

//Wrapper to get an std::vector with the aligned allocator
template<typename T>
struct AlignedVector{
  typedef std::vector<T,BasicAlignedAllocator<T> > type;
};


#if defined(GRID_CUDA) || defined(GRID_HIP)

//Cache for pinned host memory; a modified version of Grid's pointer cache
class PinnedHostMemoryCache {
private:
  static const int Ncache=128;

  inline static int & getVictim(){
    static int victim = 0;
    return victim;
  }

  typedef struct { 
    void *address;
    size_t bytes;
    int valid;
  } PointerCacheEntry;
    
  inline static PointerCacheEntry* getEntries(){
    static PointerCacheEntry Entries[Ncache];
    return Entries;
  }

public:

  //Report the memory usage in MB
  static double report_usage(){
    double sz = 0;
    static double MB=1024*1024;
    PointerCacheEntry* Entries = getEntries();
    for(int e=0;e<Ncache;e++)
      sz += double(Entries[e].bytes)/MB;
    return sz;
  }   
  
  static void free(void *ptr,size_t bytes){
#ifdef GRID_OMP
    assert(omp_in_parallel()==0);
#endif 
    PointerCacheEntry* Entries = getEntries();
    int & victim = getVictim();

    void * ret = NULL;
    int v = -1;

    for(int e=0;e<Ncache;e++) {
      if ( Entries[e].valid==0 ) {
	v=e; 
	break;
      }
    }

    if ( v==-1 ) {
      v=victim;
      victim = (victim+1)%Ncache;
    }

    if ( Entries[v].valid ) {
      ret = Entries[v].address;
      Entries[v].valid = 0;
      Entries[v].address = NULL;
      Entries[v].bytes = 0;
    }

    Entries[v].address=ptr;
    Entries[v].bytes  =bytes;
    Entries[v].valid  =1;
# if defined(GRID_CUDA)
    if(ret != NULL) assert( cudaFreeHost(ret) == cudaSuccess );
# elif defined(GRID_HIP)
    if(ret != NULL) assert( hipHostFree(ret) == hipSuccess );
# endif
  }

  static void * alloc(size_t bytes){
#ifdef GRID_OMP
    assert(omp_in_parallel()==0);
#endif 

    PointerCacheEntry* Entries = getEntries();

    for(int e=0;e<Ncache;e++){
      if ( Entries[e].valid && ( Entries[e].bytes == bytes ) ) {
	Entries[e].valid = 0;
	return Entries[e].address;
      }
    }
    void* ret;
# if defined(GRID_CUDA)
    assert( cudaMallocHost(&ret, bytes, cudaHostAllocDefault) == cudaSuccess );    
# elif defined(GRID_HIP)
    assert( hipHostMalloc(&ret, bytes, hipHostMallocDefault) == hipSuccess );    
# endif
    return ret;
  }
};

#endif //GRID_CUDA || GRID_HIP

class StandardAllocPolicy{
protected:
  inline static void _alloc(void** p, const size_t byte_size){
    *p = smalloc("CPSfield", "CPSfield", "alloc" , byte_size);
  }
  inline static void _free(void* p){
    sfree("CPSfield","CPSfield","free",p);
  }
public:
  enum { UVMenabled = 0 }; //doesnt' support UVM
};

inline void device_UVM_advise_readonly(const void* ptr, size_t count);
inline void device_UVM_advise_unset_readonly(const void* ptr, size_t count);

class Aligned128AllocPolicy{
  void* _ptr;
  size_t _byte_size;

protected:
  inline void _alloc(void** p, const size_t byte_size){
    *p = managed_alloc_check(128,byte_size); //note CUDA ignores alignment
    _ptr = *p;
    _byte_size = byte_size;
  }
  inline static void _free(void* p){
    managed_free(p);
  } 
public: 
  inline void deviceSetAdviseUVMreadOnly(const bool to) const{
    if(to) device_UVM_advise_readonly(_ptr, _byte_size);
    else device_UVM_advise_unset_readonly(_ptr, _byte_size);
  }
  
  enum { UVMenabled = 1 }; //supports UVM
};

CPS_END_NAMESPACE

#endif
