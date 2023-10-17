#ifndef _UTILS_MEMORY_H_
#define _UTILS_MEMORY_H_

//Utilities for memory status and control

#include <fstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>

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

#ifdef USE_GRID
#include<Grid.h>
#endif

#include<execinfo.h>
#include <atomic>
#include <regex>
#include <thread>
#include <omp.h>

#ifdef PRINTMEM_HEAPDUMP_GPERFTOOLS
//Allows dumping of heap state. Requires linking against libtcmalloc  -ltcmalloc
#include<gperftools/heap-profiler.h>
#endif

CPS_START_NAMESPACE

inline double byte_to_MB(const size_t b){
  return double(b)/1024./1024.;
}

inline long int procParseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

//Get the "resident set size"
inline long int getRSS(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    long int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
	  result = procParseLine(line);
	  break;
        }
    }
    fclose(file);
    return result;
}

//Get the "resident set size" breakdown: anonymous (RAM), file, shared
//RssAnon:	   13524 kB
//RssFile:	   10200 kB
//RssShmem:	      20 kB
inline void getRSSbreakdown(long int &anon, long int &file, long int &shmem){ //Note: this value is in KB!  
    FILE* f = fopen("/proc/self/status", "r");
    anon = file = shmem = -1;
    char line[128];

    while (fgets(line, 128, f) != NULL){
        if (strncmp(line, "RssAnon:", 6) == 0){
	  anon = procParseLine(line);
        }else if (strncmp(line, "RssFile:", 6) == 0){
	  file = procParseLine(line);
	}else if (strncmp(line, "RssShmem:", 6) == 0){
	  shmem = procParseLine(line);
	  break;
	}	  
    }
    fclose(f);
}



inline std::string memPoolManagerReport(bool detailed);

//Print memory usage
inline void printMem(const std::string &reason = "", int node = 0, FILE* stream = stdout){
#ifdef USE_GRID
  std::string time_str = "unknown";
  if(!omp_in_parallel()){ //Grid's global stopwatch is not thread safe
    using namespace Grid;
    Logger::GlobalStopWatch.Stop();
    GridTime time = Grid::Logger::GlobalStopWatch.Elapsed();
    Logger::GlobalStopWatch.Start();
    std::ostringstream ss; ss << time; time_str = ss.str();
  }
  if(UniqueID()==node){
    fprintf(stream,"printMem node %d : %s s",node,time_str.c_str());    
    if(reason != "") fprintf(stream, ": called with reason: %s", reason.c_str());
    fprintf(stream,"\n");
  }
#else    
  if(UniqueID()==node && reason != "") fprintf(stream, "printMem node %d called with reason: %s\n", node, reason.c_str());
#endif
  
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

#if defined(GRID_CUDA)
  size_t gpu_free, gpu_tot;
  cudaError_t err = cudaMemGetInfo(&gpu_free, &gpu_tot);
  if( err != cudaSuccess ) {
    std::cerr << "printMem: cudaMemGetInfo failed: " <<cudaGetErrorString(err)<< std::endl;
    assert(0);
  }
  if(UniqueID()==node){
    fprintf(stream,"printMem node %d: GPU memory free %f MB, used %f MB, total %f MB\n",
	    node, byte_to_MB(gpu_free), byte_to_MB(gpu_tot-gpu_free), byte_to_MB(gpu_tot) );
  }
#elif defined(GRID_HIP)
  size_t gpu_free, gpu_tot;
  hipError_t err = hipMemGetInfo(&gpu_free, &gpu_tot);
  if( err != hipSuccess ) {
    std::cerr << "printMem: hipMemGetInfo failed: " <<hipGetErrorString(err)<< std::endl;
    assert(0);
  }
  if(UniqueID()==node){
    fprintf(stream,"printMem node %d: GPU memory free %f MB, used %f MB, total %f MB\n",
	    node, byte_to_MB(gpu_free), byte_to_MB(gpu_tot-gpu_free), byte_to_MB(gpu_tot) );
  }
#endif

  if(UniqueID()==node){
    fprintf(stream, "printMem node %d: Resident set size %f MB\n", node, double(getRSS())/1024.);
  }

  if(UniqueID()==node){
#ifdef A2A_PRINTMEM_MEMPOOL_DETAILED_REPORT
    fprintf(stream, "%s\n", memPoolManagerReport(true).c_str());
#else
    fprintf(stream, "%s\n", memPoolManagerReport(false).c_str());
#endif
  }

#ifdef USE_GRID
  Grid::MemoryManager::PrintBytes(); //only output for node 0 unfortunately
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

inline void printBacktrace(std::ostream &to){
  void* tr[10];
  int n = backtrace(tr,10);
  for(int i=0;i<n;i++) to << tr[i] << std::endl;
}

//Empty shells for google perftools heap profile funcs
#ifndef BASE_HEAP_PROFILER_H_
inline void HeapProfilerStart(const char* nm){}
inline void HeapProfilerStop(){}
inline void HeapProfilerDump(const char *reason){}
#endif


class MemoryMonitor{
  std::thread *t;
  std::atomic<bool> stop;
public:
  MemoryMonitor(): t(nullptr){}

  void Start(){
    if(t) return;

    using namespace Grid;
    Logger::GlobalStopWatch.Stop();
    GridTime toff = Grid::Logger::GlobalStopWatch.Elapsed();
    Logger::GlobalStopWatch.Start();
    GridTimePoint start = GridClock::now();
    std::cout << GridLogMessage << "Starting MemoryMonitor at " << toff << std::endl;

    stop = false;
    std::atomic<bool>* stp = &stop;

    t = new std::thread([toff,start,stp]{
	std::ofstream out("mem." + std::to_string(UniqueID()));

	long anon_last=0, file_last=0, shmem_last=0, usedmem_last=0;

	while(!stp->load()){
	  GridTime t = toff + std::chrono::duration_cast<GridUsecs>(GridClock::now() - start);
	  GridUsecs tu = std::chrono::duration_cast<GridUsecs>(t);

	  long anon, file, shmem;
	  getRSSbreakdown(anon, file, shmem);
	  
	  //VmRSS:   38960 kB	  
	  // std::string VmRSS;
	  // {
	  //   std::ifstream f("/proc/self/status");
	  //   for(int i=0;i<18;i++) std::getline(f,VmRSS);
	  // }
	  // std::smatch m;
	  // assert(std::regex_search(VmRSS, m, std::regex(R"((\d+)\skB)")));
	  // VmRSS = m[1];
	  // long VmRSS_i = std::stol(VmRSS);

	  struct sysinfo sys_info;
	  sysinfo(&sys_info);
	  long total_mem = ((uint64_t)sys_info.totalram * sys_info.mem_unit)/1024;
	  long free_mem = ((uint64_t)sys_info.freeram * sys_info.mem_unit)/1024;
	  long usedmem = total_mem - free_mem;

	  out << tu << " anon:" << anon << " danon:" << anon-anon_last << " file:" << file << " dfile:" << file-file_last << " shmem:" << shmem << " dshmem:" << shmem-shmem_last << " used:" << usedmem << " dused:" << usedmem - usedmem_last << " total:" << total_mem << std::endl << std::flush;

	  anon_last = anon;
	  file_last = file;
	  shmem_last = shmem;
	  usedmem_last = usedmem;
	  
	  std::this_thread::sleep_for(GridMillisecs(1000));
	}
	out.close();
      }
      );
  }
  

  void Stop(){
    if(!t) return;
    stop = true;
    t->join();
  }

};

CPS_END_NAMESPACE

#endif
