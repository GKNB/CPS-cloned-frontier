#ifndef _UTILS_MEMORY_H_
#define _UTILS_MEMORY_H_
#include <fstream>
#include <malloc.h>
#include <vector>
#ifdef ARCH_BGQ
#include <spi/include/kernel/memory.h>
#else
#include <sys/sysinfo.h>
#endif
#include <util/gjp.h>
#include<errno.h>
#ifdef USE_MPI
#include<mpi.h>
#endif

#ifdef PRINTMEM_HEAPDUMP_GPERFTOOLS
//Allows dumping of heap state. Requires linking against libtcmalloc  -ltcmalloc
#include<gperftools/heap-profiler.h>
#endif

#include<alg/a2a/reuse_block_allocator.h>

//Utilities for memory control

CPS_START_NAMESPACE

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
    return (pointer) memalign(128,bytes);
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


//A class that owns data via a pointer that has an assignment and copy constructor which does a deep copy.
template<typename T>
class PtrWrapper{
  T* t;
public:
  inline T* ptr(){ return t; }
  inline T const* ptr() const{ return t; }
  inline T& operator*(){ return *t; }
  inline T* operator->(){ return t; }
  inline T const& operator*() const{ return *t; }
  inline T const* operator->() const{ return t; }
  
  inline PtrWrapper(): t(NULL){};
  inline PtrWrapper(T* _t): t(_t){}
  inline ~PtrWrapper(){ if(t!=NULL) delete t; }

  inline const bool assigned() const{ return t != NULL; }
  
  inline void set(T* _t){
    if(t!=NULL) delete t;
    t = _t;
  }
  inline void free(){
    if(t!=NULL) delete t;
    t = NULL;
  }
  
  //Deep copies
  inline PtrWrapper(const PtrWrapper &r): t(NULL){
    if(r.t != NULL) t = new T(*r.t);
  }

  inline PtrWrapper & operator=(const PtrWrapper &r){
    if(t!=NULL){ delete t; t = NULL; }
    if(r.t!=NULL) t = new T(*r.t);
  } 
};


inline double byte_to_MB(const size_t b){
  return double(b)/1024./1024.;
}

bool isAligned(void *p, const size_t alignment){
  return (((uintptr_t)p) % alignment == 0 );
}

//Empty shells for google perftools heap profile funcs
#ifndef BASE_HEAP_PROFILER_H_
inline void HeapProfilerStart(const char* nm){}
inline void HeapProfilerStop(){}
inline void HeapProfilerDump(const char *reason){}
#endif


//Print memory usage
inline void printMem(const std::string &reason = "", int node = 0, FILE* stream = stdout){
  if(UniqueID()==node && reason != "") fprintf(stream, "printMem node %d called with reason: %s\n", node, reason.c_str());
  
#ifdef ARCH_BGQ
  #warning "printMem using ARCH_BGQ"
  uint64_t shared, persist, heapavail, stackavail, stack, heap, guard, mmap;
  Kernel_GetMemorySize(KERNEL_MEMSIZE_SHARED, &shared);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_PERSIST, &persist);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &heapavail);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_STACKAVAIL, &stackavail);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_STACK, &stack);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &heap);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_GUARD, &guard);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_MMAP, &mmap);

  if(UniqueID()==node){
    fprintf(stream,"printMem node %d: Allocated heap: %.2f MB, avail. heap: %.2f MB\n", node, (double)heap/(1024*1024),(double)heapavail/(1024*1024));
    fprintf(stream,"printMem node %d: Allocated stack: %.2f MB, avail. stack: %.2f MB\n", node, (double)stack/(1024*1024), (double)stackavail/(1024*1024));
    fprintf(stream,"printMem node %d: Memory: shared: %.2f MB, persist: %.2f MB, guard: %.2f MB, mmap: %.2f MB\n", node, (double)shared/(1024*1024), (double)persist/(1024*1024), (double)guard/(1024*1024), (double)mmap/(1024*1024));
  }
#else
#warning "printMem using NOARCH"
  /* unsigned long totalram;  /\* Total usable main memory size *\/ */
  /* unsigned long freeram;   /\* Available memory size *\/ */
  /* unsigned long sharedram; /\* Amount of shared memory *\/ */
  /* unsigned long bufferram; /\* Memory used by buffers *\/ */
  /* unsigned long totalswap; /\* Total swap space size *\/ */
  /* unsigned long freeswap;  /\* swap space still available *\/ */
  /* unsigned short procs;    /\* Number of current processes *\/ */
  /* unsigned long totalhigh; /\* Total high memory size *\/ */
  /* unsigned long freehigh;  /\* Available high memory size *\/ */
  /* unsigned int mem_unit;   /\* Memory unit size in bytes *\/ */

  struct sysinfo myinfo;
  sysinfo(&myinfo);
  double total_mem = myinfo.mem_unit * myinfo.totalram;
  total_mem /= (1024.*1024.);
  double free_mem = myinfo.mem_unit * myinfo.freeram;
  free_mem /= (1024.*1024.);
  
  if(UniqueID()==node){
    fprintf(stream,"printMem node %d: Memory: total: %.2f MB, avail: %.2f MB, used %.2f MB\n",node,total_mem, free_mem, total_mem-free_mem);
  }

  //# define PRINT_MALLOC_INFO    //Use of int means this is garbage for large memory systems
# ifdef PRINT_MALLOC_INFO
  struct mallinfo mi;
  mi = mallinfo();

  // int arena;     /* Non-mmapped space allocated (bytes) */
  // int ordblks;   /* Number of free chunks */
  // int smblks;    /* Number of free fastbin blocks */
  // int hblks;     /* Number of mmapped regions */
  // int hblkhd;    /* Space allocated in mmapped regions (bytes) */
  // int usmblks;   /* Maximum total allocated space (bytes) */
  // int fsmblks;   /* Space in freed fastbin blocks (bytes) */
  // int uordblks;  /* Total allocated space (bytes) */
  // int fordblks;  /* Total free space (bytes) */
  // int keepcost;  /* Top-most, releasable space (bytes) */

  if(UniqueID()==node){
    fprintf(stream,"printMem node %d: Malloc info: arena %f MB, ordblks %d, smblks %d, hblks %d, hblkhd %f MB, fsmblks %f MB, uordblks %f MB, fordblks %f MB, keepcost %f MB\n",
	   node, byte_to_MB(mi.arena), mi.ordblks, mi.smblks, mi.hblks, byte_to_MB(mi.hblkhd), byte_to_MB(mi.fsmblks), byte_to_MB(mi.uordblks), byte_to_MB(mi.fordblks), byte_to_MB(mi.keepcost) );
  }

# endif

  //# define PRINT_MALLOC_STATS  Also doesn't work well
# ifdef PRINT_MALLOC_STATS
  if(UniqueID()==node) malloc_stats();
# endif
  
#endif

#ifdef PRINTMEM_HEAPDUMP_GPERFTOOLS
  //if(UniqueID()==node) reason == "" ? HeapProfilerDump("printMem") : HeapProfilerDump(reason.c_str());
  HeapProfilerDump(reason.c_str());
#endif
  fflush(stream);
}

inline void printMemNodeFile(const std::string &msg = ""){
  static int calls = 0;

  std::ostringstream os; os << "mem_status." << UniqueID();
  FILE* out = fopen (os.str().c_str(), calls == 0 ? "w" : "a");
  if(out == NULL){
    printf("Non-fatal error in printMemNodeFile on node %d: could not open file %s with mode %c\n",UniqueID(),os.str().c_str(),calls==0 ? 'w' : 'a');
    fflush(stdout);
  }else{
    printMem(msg,UniqueID(),out);
    fclose(out);
  }
  calls++;
}


#ifdef USE_MPI

struct _MPI_UniqueID_map{
  std::map<int,int> mpi_rank_to_uid;
  std::map<int,int> uid_to_mpi_rank;
  
  void setup(){
    int nodes = 1;
    for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);

    int* mpi_ranks = (int*)malloc(nodes *  sizeof(int));
    memset(mpi_ranks, 0, nodes *  sizeof(int));
    assert( MPI_Comm_rank(MPI_COMM_WORLD, mpi_ranks + UniqueID() ) == MPI_SUCCESS );

    int* mpi_ranks_all = (int*)malloc(nodes *  sizeof(int));
    assert( MPI_Allreduce(mpi_ranks, mpi_ranks_all, nodes, MPI_INT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS );

    for(int i=0;i<nodes;i++){
      int uid = i;
      int rank = mpi_ranks_all[i];
      
      mpi_rank_to_uid[rank] = uid;
      uid_to_mpi_rank[uid] = rank;
    }
    
    free(mpi_ranks);
    free(mpi_ranks_all);
  }
};

class MPI_UniqueID_map{
  static _MPI_UniqueID_map *getMap(){
    static _MPI_UniqueID_map* mp = NULL;
    if(mp == NULL){
      mp = new _MPI_UniqueID_map;
      mp->setup();
    }
    return mp;
  }
public:
  
  static int MPIrankToUid(const int rank){ return getMap()->mpi_rank_to_uid[rank]; }
  static int UidToMPIrank(const int uid){ return getMap()->uid_to_mpi_rank[uid]; }
};

#endif

struct nodeDistributeCounter{
  //Get a rank idx that cycles between 0... nodes-1
  static int getNext(){
    static int cur = 0;
    static int nodes = -1;
    if(nodes == -1){
      nodes = 1; for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);
    }
    int out = cur;
    cur = (cur + 1) % nodes;
    return out;
  }

  //Keep tally of the number of MF uniquely stored on this node
  static int incrOnNodeCount(const int by){
    static int i = 0;
    i += by;
    return i;
  }
  static int onNodeCount(){
    return incrOnNodeCount(0);
  }

};

class DistributedMemoryStorage{
  void *ptr;
  int _alignment;
  size_t _size;
  int _master_uid;
  int _master_mpirank;

  void initMaster(){
    if(_master_uid == -1){
      int master_uid = nodeDistributeCounter::getNext(); //round-robin assignment
      
      int nodes = 1;
      for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);
#ifndef USE_MPI
      _master_uid = master_uid;
      if(nodes > 1) ERR.General("DistributedMemoryStorage","initMaster","Implementation requires MPI\n");
#else
      //Check all nodes are decided on the same master node
      int* masters = (int*)malloc(nodes *  sizeof(int));
      memset(masters, 0, nodes *  sizeof(int));
      masters[UniqueID()] = master_uid;

      int* masters_all = (int*)malloc(nodes *  sizeof(int));
      assert( MPI_Allreduce(masters, masters_all, nodes, MPI_INT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS );

      for(int i=0;i<nodes;i++) assert(masters_all[i] == master_uid);

      free(masters); free(masters_all);
      _master_uid = master_uid;
      _master_mpirank = MPI_UniqueID_map::UidToMPIrank(master_uid);
#endif
    }
  }

public:
  struct GatherPerf{
    size_t check_calls;
    double check_time;
    size_t alloc_calls;
    double alloc_time;
    size_t gather_calls;
    double gather_time;
    size_t bytes;
    size_t free_calls;
    double free_time;
    
    void reset(){
      check_calls=alloc_calls=gather_calls=bytes=free_calls=0;
      check_time=alloc_time=gather_time=free_time=0;
    }
    GatherPerf(){
      reset();
    }
    void print(){
      if(!UniqueID()){
	double avg_check_time = check_time / double(check_calls);
	double avg_alloc_time = alloc_time / double(alloc_calls);
	double avg_gather_time = gather_time / double(gather_calls);
	double avg_free_time = free_time / double(free_calls);
	double avg_bandwidth = double(bytes)/gather_time/(1024*1024); //MB/s
	std::ostringstream os; 
	os << "DistributedMemoryStorage::GatherPerf avg check time " << avg_check_time << "s, avg alloc time " << avg_alloc_time << "s, avg gather time " << avg_gather_time << "s, gather bandwidth " << avg_bandwidth << "MB/s, avg free time " << avg_free_time << "s\n";
	printf(os.str().c_str()); fflush(stdout);
      }
    }
  };
  static GatherPerf & perf(){ static GatherPerf p; return p; }

  DistributedMemoryStorage(): ptr(NULL), _master_uid(-1){}

  DistributedMemoryStorage(const DistributedMemoryStorage &r): ptr(NULL){
    if(r.ptr != NULL){    
      alloc(r._alignment, r._size);
      memcpy(ptr, r.ptr, r._size);
    }else{
      _size = r._size;
      _alignment = r._alignment;
    }      
    _master_uid = r._master_uid;
    _master_mpirank = r._master_mpirank;
  }

  DistributedMemoryStorage & operator=(const DistributedMemoryStorage &r){
    freeMem();
    if(r.ptr != NULL){
      alloc(r._alignment, r._size);
      memcpy(ptr, r.ptr, r._size);      
    }else{
      _size = r._size;
      _alignment = r._alignment;
    }      
    _master_uid = r._master_uid;
    _master_mpirank = r._master_mpirank;
    return *this;
  }

  inline bool isOnNode() const{ return ptr != NULL; }
  
  void move(DistributedMemoryStorage &r){
    ptr = r.ptr;
    _alignment = r._alignment;
    _size = r._size;
    _master_uid = r._master_uid;
    _master_mpirank = r._master_mpirank;
    r._size = 0;
    r.ptr = NULL;
  }


#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
  inline static ReuseBlockAllocatorsAligned & block_allocator(){ static ReuseBlockAllocatorsAligned r; return r; }
#endif


  void alloc(int alignment, size_t size){
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY

    if(ptr != NULL){
      if(alignment == _alignment && size == _size) return;
      else{ block_allocator().free(ptr); ptr = NULL; }
    }
    ptr = block_allocator().alloc(alignment,size);

#else

    if(ptr != NULL){
      if(alignment == _alignment && size == _size) return;
      else{ free(ptr); ptr = NULL; }
    }
    int r = posix_memalign(&ptr, alignment, size);
    if(r){
#  ifdef USE_MPI
      int mpi_rank; assert( MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank) == MPI_SUCCESS );
      printf("Error: rank %d (uid %d) failed to allocate memory! posix_memalign return code %d (EINVAL=%d ENOMEM=%d). Require %g MB. Memory status\n", 
	     mpi_rank, UniqueID(), r, EINVAL, ENOMEM, byte_to_MB(size) );
#  else
      printf("Error: uid %d failed to allocate memory! posix_memalign return code %d (EINVAL=%d ENOMEM=%d). Require %g MB. Memory status\n", UniqueID(), r,EINVAL, ENOMEM, byte_to_MB(size) ); 
#  endif
      printMem("Error",UniqueID());
    }

#endif
    _size = size;
    _alignment = alignment;
  }

  void freeMem(){
    if(ptr != NULL){
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
      block_allocator().free(ptr);
#else
      free(ptr);
#endif
      ptr = NULL;
    }
  }

  inline void* data(){ return ptr; }
  inline void const* data() const{ return ptr; }
  
  //Every node performs gather but if not required and not master, data is not kept
  void gather(bool require){
#ifndef USE_MPI
    if(ptr != NULL) ERR.General("DistributedMemoryStorage","gather","Implementation requires MPI\n");
#else
    double time = dclock();
    int do_gather_node = (require && ptr == NULL);

#define ENABLE_GATHER_PRECHECK
#ifdef ENABLE_GATHER_PRECHECK
    //Check to see if a gather is actually necessary
    int do_gather_any = 0;
    assert( MPI_Allreduce(&do_gather_node, &do_gather_any, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS );

    perf().check_calls++;
    perf().check_time += dclock() - time;
#else
    int do_gather_any = 1;
#endif

    //Do the gather. All nodes need memory space, albeit temporarily
    if(do_gather_any){
      if(UniqueID() != _master_uid && ptr == NULL){
	time = dclock();
	alloc(_alignment, _size);      
	perf().alloc_calls++;
	perf().alloc_time += dclock() - time;
      }
      time = dclock();
      assert( MPI_Bcast(ptr, _size, MPI_BYTE, _master_mpirank, MPI_COMM_WORLD) == MPI_SUCCESS );
      perf().gather_calls++;
      perf().gather_time += dclock() - time;
      perf().bytes += _size;
    }
    
    //Non-master copies safe to throw away data if not required. If data was already present we don't throw away because it may have been pulled by a different call to gather
    if(!require && UniqueID() != _master_uid && ptr != NULL && do_gather_node){
      time = dclock();
      freeMem();
      perf().free_calls++;
      perf().free_time += dclock() - time;
    }

#endif
  }

  void distribute(){
    initMaster();
    if(UniqueID() != _master_uid) freeMem();
  }
  
  ~DistributedMemoryStorage(){
    freeMem();
  }
  
};


//A similar class to the above but which stores to disk rather than distributing over the memory systems of a multi-node machine. This version is designed to work on systems with burst buffers that are on a shared filesystem such that only the head node writes to disk
class BurstBufferMemoryStorage{
protected:
  void *ptr;
  int _alignment;
  size_t _size;
  std::string file;
  bool ondisk;
  unsigned int ondisk_checksum; //so we can tell if the data has changed since we wrote it previously
  
  unsigned int checksum() const{
    static bool init = false;
    static FPConv fp;    
    if(!init){
      fp.setFileFormat(FP_AUTOMATIC);
      init = true;
    }
    assert(sizeof(Float) == sizeof(double));
    assert(_size % sizeof(double) == 0);   
    return fp.checksum((char*)ptr, _size/sizeof(double), FP_AUTOMATIC);
  }  
public:
  BurstBufferMemoryStorage(): ptr(NULL), ondisk(false){}

  BurstBufferMemoryStorage(const BurstBufferMemoryStorage &r): ptr(NULL){
    if(r.ptr != NULL){
      alloc(r._alignment, r._size);
      memcpy(ptr, r.ptr, r._size);
    }else{
      _size = r._size;
      _alignment = r._alignment;
    }      
    file = r.file;
    ondisk = r.ondisk;
    ondisk_checksum = r.ondisk_checksum;
  }

  BurstBufferMemoryStorage & operator=(const BurstBufferMemoryStorage &r){
    freeMem();
    if(r.ptr != NULL){
      alloc(r._alignment, r._size);
      memcpy(ptr, r.ptr, r._size);
    }else{
      _size = r._size;
      _alignment = r._alignment;
    }
    file = r.file;
    ondisk = r.ondisk;
    ondisk_checksum = r.ondisk_checksum;
    return *this;
  }

  static std::string & filestub(){ static std::string b = "burststore"; return b; } //change this to the base filename (an _<IDX>.dat is appended)
  
  inline bool isOnNode() const{ return ptr != NULL; }
  
  void move(BurstBufferMemoryStorage &r){
    ptr = r.ptr;
    _alignment = r._alignment;
    _size = r._size;
    file = r.file;
    ondisk = r.ondisk;
    ondisk_checksum = r.ondisk_checksum;
    r._size = 0;
    r.ondisk = false;
    r.file = "";
    r.ptr = NULL;
  }

  
  void alloc(int alignment, size_t size){
    if(ptr != NULL){
      if(alignment == _alignment && size == _size) return;
      else{ free(ptr); ptr = NULL; }
    }
    int r = posix_memalign(&ptr, alignment, size);
    if(r){
      printf("Error: Node %d failed to allocate memory! posix_memalign return code %d (EINVAL=%d ENOMEM=%d). Require %g MB. Memory status\n", UniqueID(), r, EINVAL, ENOMEM, byte_to_MB(size) );
      printMem("Error",UniqueID());
    }
    _size = size;
    _alignment = alignment;
  }

  void freeMem(){
    if(ptr != NULL){
      free(ptr);
      ptr = NULL;
    }
  }

  inline void* data(){ return ptr; }
  inline void const* data() const{ return ptr; }

#define DISTRIBUTE_FLUSH_MEMBUF
  
  //Load from disk (optional)
  void gather(bool require){
    int do_gather_node = (require && ptr == NULL);
    if(do_gather_node){
      alloc(_alignment,_size);
      std::fstream f(file.c_str(), std::ios::in | std::ios::binary);
      if(!f.good()) ERR.General("BurstBufferMemoryStorage","gather(bool)","Failed to open file %s for read\n",file.c_str());
      size_t rd_size; f.read((char*)&rd_size,sizeof(size_t));
      if(rd_size != _size) ERR.General("BurstBufferMemoryStorage","gather(bool)","Data size %lu in file %s different from expected %lu\n", (unsigned long)rd_size, file.c_str(), (unsigned long)_size );
      unsigned int cksum_rd; f.read((char*)&cksum_rd,sizeof(unsigned int));
      f.read((char*)ptr,_size);
      if(!f.good()) ERR.General("BurstBufferMemoryStorage","gather(bool)","Read error in file %s\n",file.c_str());      
      f.close();
      unsigned int cksum_calc = checksum();
      if(cksum_calc != cksum_rd) ERR.General("BurstBufferMemoryStorage","gather(bool)","Checksum error on reading file %s, expected %u, got %u\n",file.c_str(),cksum_rd,cksum_calc);
    }
    cps::sync();
  }

  void distribute(){
    if(ptr == NULL) return;
    unsigned int cksum = checksum();
    if(!ondisk || (ondisk && ondisk_checksum != cksum) ){ //it is assumed here that node 0 has a copy
      static int fidx = 0;
      if(!ondisk){
	std::ostringstream os; os << filestub() << "_" << fidx++ << ".dat";
	file = os.str();
      }
	
      if(!UniqueID()){
	assert(ptr != NULL);
	std::fstream f(file.c_str(), std::ios::out | std::ios::binary);
	if(!f.good()) ERR.General("BurstBufferMemoryStorage","gather(bool)","Failed to open file %s for write\n",file.c_str());
	f.write((char*)&_size,sizeof(size_t));
	f.write((char*)&cksum,sizeof(unsigned int));
	f.write((char*)ptr,_size);
	if(!f.good()) ERR.General("BurstBufferMemoryStorage","gather(bool)","Write error in file %s\n",file.c_str());
#ifdef DISTRIBUTE_FLUSH_MEMBUF
	f.flush(); //should ensure data is written to disk immediately and not kept around in some memory buffer, but may slow things down
#endif	
      }
      ondisk = true;
      ondisk_checksum = cksum;
    }
    cps::sync();
    freeMem();
  }
  
  ~BurstBufferMemoryStorage(){
    freeMem();
  }
  
};






//This works similarly to the above only each node writes its own unique copy rather than just the head node. This is intended for systems where nodes have independent scratch disks
class IndependentDiskWriteStorage: public BurstBufferMemoryStorage{
public:
  IndependentDiskWriteStorage(): BurstBufferMemoryStorage(){}

  IndependentDiskWriteStorage(const IndependentDiskWriteStorage &r): BurstBufferMemoryStorage(r){}

  inline IndependentDiskWriteStorage & operator=(const IndependentDiskWriteStorage &r){
    static_cast<BurstBufferMemoryStorage&>(*this) = r;
  }

  void distribute(){
    if(ptr == NULL) return;
    unsigned int cksum = checksum();
    if(!ondisk || (ondisk && ondisk_checksum != cksum) ){
      static int fidx = 0;
      if(!ondisk){
	std::ostringstream os; os << filestub() << "_" << fidx++ << "_node" << UniqueID() << ".dat";
	file = os.str();
      }
      assert(ptr != NULL);
      std::fstream f(file.c_str(), std::ios::out | std::ios::binary);
      if(!f.good()) ERR.General("IndependentDiskWriteStorage","gather(bool)","Failed to open file %s for write\n",file.c_str());
      f.write((char*)&_size,sizeof(size_t));
      f.write((char*)&cksum,sizeof(unsigned int));
      f.write((char*)ptr,_size);
      if(!f.good()) ERR.General("IndependentDiskWriteStorage","gather(bool)","Write error in file %s\n",file.c_str());
#ifdef DISTRIBUTE_FLUSH_MEMBUF
      f.flush(); //should ensure data is written to disk immediately and not kept around in some memory buffer, but may slow things down
#endif
      ondisk = true;
      ondisk_checksum = cksum;      
    }
    freeMem();
  }
};


CPS_END_NAMESPACE

#endif
