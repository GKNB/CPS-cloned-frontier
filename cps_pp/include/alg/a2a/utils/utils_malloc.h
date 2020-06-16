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
    printBacktrace(std::cout);
    printMem("Memory status on fail", UniqueID());
    std::cout.flush(); fflush(stdout);
    ERR.General("","malloc_check","Mem alloc failed\n");
  }
  return p;
}

//Note, CUDA does not allow alignment; apparently it is always "sufficient"
inline void* managed_alloc_check(const size_t align, const size_t byte_size){
#ifdef GRID_NVCC
  void *p;
  auto err = cudaMallocManaged(&p,byte_size);
  if( err != cudaSuccess ) {
    p = (void*)NULL;
    std::cerr << "managed_alloc_check: cudaMallocManaged failed for " << byte_size<<" bytes " <<cudaGetErrorString(err)<< std::endl;
    printMem("malloc failed",UniqueID());
    assert(0);
  }
  return p;
#else
  return memalign_check(align, byte_size);
#endif
}

inline void* managed_alloc_check(const size_t byte_size){
#ifdef GRID_NVCC
  void *p;
  auto err = cudaMallocManaged(&p,byte_size);
  if( err != cudaSuccess ) {
    p = (void*)NULL;
    std::cerr << "managed_alloc_check: cudaMallocManaged failed for " << byte_size<<" bytes " <<cudaGetErrorString(err)<< std::endl;
    printMem("malloc failed",UniqueID());
    assert(0);
  }
  return p;
#else
  return malloc_check(byte_size);
#endif
}

inline void managed_free(void* p){
#ifdef GRID_NVCC
  auto err = cudaFree(p);
  if( err != cudaSuccess ) {
    std::cerr << "managed_free: cudaFree failed with error " <<cudaGetErrorString(err)<< std::endl;
    assert(0);
  }  
#else
  free(p);
#endif
}



//Allocate memory on GPU device if in use; otherwise do memalign
inline void* device_alloc_check(const size_t align, const size_t byte_size){
#ifdef GRID_NVCC
  void *p;
  auto err = cudaMalloc(&p,byte_size);
  if( err != cudaSuccess ) {
    p = (void*)NULL;
    std::cerr << "device_alloc_check: cudaMalloc failed for " << byte_size<<" bytes " <<cudaGetErrorString(err)<< std::endl;
    printMem("malloc failed",UniqueID());
    assert(0);
  }
  return p;
#else
  return memalign_check(align, byte_size);
#endif
}

inline void* device_alloc_check(const size_t byte_size){
#ifdef GRID_NVCC
  void *p;
  auto err = cudaMalloc(&p,byte_size);
  if( err != cudaSuccess ) {
    p = (void*)NULL;
    std::cerr << "device_alloc_check: cudaMalloc failed for " << byte_size<<" bytes " <<cudaGetErrorString(err)<< std::endl;
    printMem("malloc failed",UniqueID());
    assert(0);
  }
  return p;
#else
  return malloc_check(byte_size);
#endif
}

inline void device_free(void* p){
#ifdef GRID_NVCC
  auto err = cudaFree(p);
  if( err != cudaSuccess ) {
    std::cerr << "device_free: cudaFree failed with error " <<cudaGetErrorString(err)<< std::endl;
    assert(0);
  }  
#else
  free(p);
#endif
}






//Allocate pinned memory on host if in use; otherwise do memalign
inline void* pinned_alloc_check(const size_t align, const size_t byte_size){
#ifdef GRID_NVCC
  void *p;
  auto err = cudaMallocHost(&p,byte_size);
  if( err != cudaSuccess ) {
    p = (void*)NULL;
    std::cerr << "pinned_alloc_check: cudaHostMalloc failed for " << byte_size<<" bytes " <<cudaGetErrorString(err)<< std::endl;
    printMem("malloc failed",UniqueID());
    assert(0);
  }
  return p;
#else
  return memalign_check(align, byte_size);
#endif
}

inline void* pinned_alloc_check(const size_t byte_size){
#ifdef GRID_NVCC
  void *p;
  auto err = cudaMallocHost(&p,byte_size);
  if( err != cudaSuccess ) {
    p = (void*)NULL;
    std::cerr << "device_alloc_check: cudaHostMalloc failed for " << byte_size<<" bytes " <<cudaGetErrorString(err)<< std::endl;
    printMem("malloc failed",UniqueID());
    assert(0);
  }
  return p;
#else
  return malloc_check(byte_size);
#endif
}

inline void pinned_free(void* p){
#ifdef GRID_NVCC
  auto err = cudaFreeHost(p);
  if( err != cudaSuccess ) {
    std::cerr << "pinned_free: cudaFree failed with error " <<cudaGetErrorString(err)<< std::endl;
    assert(0);
  }  
#else
  free(p);
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


//Global variable controlling whether participating copy constructors are shallow or deep copies. Switch to shallow for structs with managed memory pointers to prevent extra copy
struct copyControl{
  static inline bool & shallow(){ static bool s=false; return s; }
};

#ifdef GRID_NVCC

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

    if(ret != NULL) assert( cudaFree(ret) == cudaSuccess );
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
    assert( cudaMallocHost(&ret, bytes, cudaHostAllocDefault) == cudaSuccess );    
    return ret;
  }
};

#endif //GRID_NVCC



CPS_END_NAMESPACE

#endif
