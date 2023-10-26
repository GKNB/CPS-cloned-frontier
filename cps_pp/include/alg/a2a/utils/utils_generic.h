#ifndef __UTILS_GENERIC_H_
#define __UTILS_GENERIC_H_

#include <cstdarg>
#include <cxxabi.h>
#include <chrono>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <util/lattice.h>
#include <alg/fix_gauge_arg.h>
#include <zlib.h>
#include "utils_malloc.h"

CPS_START_NAMESPACE

//An empty class
class NullObject
{
 public:
  NullObject(){}
};

//A class inheriting from this type must have template parameter T as a double or float
#define EXISTS_IF_DOUBLE_OR_FLOAT(T) public my_enable_if<is_double_or_float<mf_Float>::value,NullObject>::type

//Skip gauge fixing and set all gauge fixing matrices to unity
inline void gaugeFixUnity(Lattice &lat, const FixGaugeArg &fix_gauge_arg){
  FixGaugeType fix = fix_gauge_arg.fix_gauge_kind;
  int start = fix_gauge_arg.hyperplane_start;
  int step = fix_gauge_arg.hyperplane_step;
  int num = fix_gauge_arg.hyperplane_num;

  int h_planes[num];
  for(int i=0; i<num; i++) h_planes[i] = start + step * i;

  if(lat.FixGaugePtr() != nullptr) lat.FixGaugeFree();  
  lat.FixGaugeAllocate(fix, num, h_planes);
  
#pragma omp parallel for
  for(int sf=0;sf<(GJP.Gparity()+1)*GJP.VolNodeSites();sf++){
    //s + vol*f
    int s = sf % GJP.VolNodeSites();
    int f = sf / GJP.VolNodeSites();
    
    const Matrix* mat = lat.FixGaugeMatrix(s,f);
    if(mat == NULL) continue;
    else{
      Matrix* mm = const_cast<Matrix*>(mat); //evil, I know, but it saves duplicating the accessor (which is overly complicated)
      mm->UnitMatrix();
    }
  }
}


//The base G-parity momentum vector for quark fields with arbitrary sign
inline void GparityBaseMomentum(int p[3], const int sgn){
  for(int i=0;i<3;i++)
    if(GJP.Bc(i) == BND_CND_GPARITY)
      p[i] = sgn;
    else p[i] = 0;
}

//String to anything conversion
template<typename T>
inline T strToAny(const char* a){
  std::stringstream ss; ss << a; T o; ss >> o;
  return o;
}
template<typename T>
inline std::string anyToStr(const T &t){
  std::ostringstream os; os << t;
  return os.str();
}

inline std::string demangle( const char* mangled_name ) {

  std::size_t len = 0 ;
  int status = 0 ;
  char* unmangled = __cxxabiv1::__cxa_demangle( mangled_name, NULL, &len, &status );
  if(status == 0){  
    std::string out(unmangled);
    free(unmangled);
    return out;
  }else return "<demangling failed>";
  
  /* std::unique_ptr< char, decltype(&std::free) > ptr( */
  /* 						    __cxxabiv1::__cxa_demangle( mangled_name, nullptr, &len, &status ), &std::free ) ; */
  /* return ptr.get() ; */
}

//Print a type as a string (useful for debugging)
template<typename T>
inline std::string printType(){ return demangle(typeid(T).name()); }

inline std::string stringize(const char* format, ...){
  int n; //not counting null character
  {
    char buf[1024];
    va_list argptr;
    va_start(argptr, format);    
    n = vsnprintf(buf, 1024, format, argptr);
    va_end(argptr);
    if(n < 1024) return std::string(buf);
  }
  
  char buf[n+1];
  va_list argptr;
  va_start(argptr, format);    
  int n2 = vsnprintf(buf, 1024, format, argptr);
  va_end(argptr);
  assert(n2 <= n);
  return std::string(buf);
}

//Seconds since first call with us accuracy
inline double secs_since_first_call(){
  static std::chrono::time_point<std::chrono::high_resolution_clock> start;
  static bool start_set = false;

  auto now = std::chrono::high_resolution_clock::now();
  if(!start_set){
    start = now; start_set = true; return 0;
  }else{
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return double(us)/1e6;
  }
}

inline void write_data_bypass_cache(const std::string &file, char const* data, size_t bytes){
  int fd = open(file.c_str(), O_WRONLY | O_CREAT | O_DIRECT | O_DSYNC | O_TRUNC, S_IRWXU);
  if(fd == -1){
    perror("Failed to open file");
    ERR.General("","write_data_bypass_cache","Failed to open file %s", file.c_str());
  }

  /* //Need an aligned memory region */

  //Can't get statx working on qcdserver??
  /* struct statx st; */
  /* int e = statx(fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &st); */
  /* if(e == -1){ */
  /*   perror(errno); */
  /*   ERR.General("","write_data_bypass_cache","Failed to query file alignment");     */
  /* } */
  /* if(st.stx_dio_mem_align == 0) ERR.General("","write_data_bypass_cache","Direct IO not supported on this file");     */
  
  /* size_t bufsz = (10*1024*1024 + st.stx_dio_offset_align) % st.stx_dio_offset_align;   */
  /* void* buf = memalign_check(st.stx_dio_mem_align, bufsz); */

  size_t bufsz = 10*1024*1024;
  void* buf = memalign_check(4096, bufsz);
  
  while(bytes > 0){
    size_t count = std::min(bytes, bufsz);
    size_t cpcount = count;
    memcpy(buf, data, cpcount);
    if(count % 4096 != 0){
      count = ((count + 4096) / 4096) * 4096; //round up to nearest disk block size to avoid errors. This can only happen for the last snippet of data
      assert(count <= bufsz); //sanity check! Should always pass because bufsz is a multiple of 4kB
    }

    ssize_t f = write(fd, buf, count);
    if(f==-1){
      perror("Write failed");
      ERR.General("","write_data_bypass_cache","Write failed for bytes %lu", count);    
    }else if(f != count){
      ERR.General("","write_data_bypass_cache","Write did not write expected number of bytes, wrote %lu requested %lu", f, count);    
    }
    bytes -= cpcount;
    data += cpcount;
  }  
  int e = close(fd);
  if(e == -1){
    perror("File close failed");
    ERR.General("","write_data_bypass_cache","Failed to close file %s", file.c_str());
  }
  free(buf);
}


inline void read_data_bypass_cache(const std::string &file, char * data, size_t bytes){
  int fd = open(file.c_str(), O_RDONLY | O_DIRECT);
  if(fd == -1){
    perror("Failed to open file");
    ERR.General("","read_data_bypass_cache","Failed to open %s", file.c_str());
  }

  /* //Need an aligned memory region */
  size_t bufsz = 10*1024*1024;
  void* buf = memalign_check(4096, bufsz);
  
  while(bytes > 0){
    size_t count = std::min(bytes, bufsz);
    size_t cpcount = count;
    if(count % 4096 != 0){
      count = ((count + 4096) / 4096) * 4096; //round up to nearest disk block size to avoid errors. This can only happen for the last snippet of data
      assert(count <= bufsz); //sanity check! Should always pass because bufsz is a multiple of 4kB
    }

    ssize_t f = read(fd, buf, count);
    if(f==-1){
      perror("Read failed");
      ERR.General("","read_data_bypass_cache","Read failed for bytes %lu", count);
    }else if(f != count){
      ERR.General("","read_data_bypass_cache","Read did not read expected number of bytes, got %lu requested %lu", f, count);
    }
    memcpy(data, buf, cpcount);
    bytes -= cpcount;
    data += cpcount;
  }  
  int e = close(fd);
  if(e == -1){
    perror("Failed to close file");
    ERR.General("","read_data_bypass_cache","Failed to close file %s", file.c_str());
  }
  free(buf);
}

inline void disk_write_immediate(const std::string &file, void* data, size_t len){
  int fd = open(file.c_str(), O_WRONLY | O_CREAT | O_DSYNC | O_TRUNC, S_IRWXU);
  if(fd == -1){
    perror("Failed to open file");
    ERR.General("","disk_write_immediate","Failed to open file %s", file.c_str());
  }
  ssize_t f = write(fd, data, len);
  if(f==-1){
    perror("Write failed");
    ERR.General("","disk_write_immediate","Write failed for bytes %lu", len);    
  }else if(f != len){
    ERR.General("","disk_write_immediate","Write did not write expected number of bytes, wrote %lu requested %lu", f, len);
  }
  int e = close(fd);
  if(e == -1){
    perror("Failed to close file");
    ERR.General("","disk_write_immediate","Failed to close file %s", file.c_str());
  }
}

inline void disk_read(const std::string &file, void* data, size_t len){
  int fd = open(file.c_str(), O_RDONLY);
  if(fd == -1){
    perror("Failed to open file");
    ERR.General("","disk_read","Failed to open file %s", file.c_str());
  }
  ssize_t f = read(fd, data, len);
  if(f==-1){
    perror("Read failed");
    ERR.General("","disk_read","Read failed for bytes %lu", len);    
  }else if(f != len){
    ERR.General("","disk_read","Write did not write expected number of bytes, wrote %lu requested %lu", f, len);    
  }
  int e = close(fd);
  if(e == -1){
    perror("Failed to close file");
    ERR.General("","disk_read","Failed to close file %s", file.c_str());
  }
}

class cpsBinaryWriter{
  int fd;
  const std::string file;
public:
  void open(const std::string &_file, bool immediate = false){
    int flags =  O_WRONLY | O_CREAT  | O_TRUNC;
    if(immediate) flags = flags | O_DSYNC;

    fd = ::open(_file.c_str(), flags, S_IRWXU);
    if(fd == -1){
      perror("Failed to open file");
      ERR.General("cpsBinaryWriter","open","Failed to open file %s", _file.c_str());
    }
    file = _file;
  }
  
  cpsBinaryWriter(): fd(-1){}
  cpsBinaryWriter(const std::string &_file, bool immediate = false): fd(-1){ open(_file, immediate); }

  void write(void* data, size_t len, bool checksum = true) const{
    if(fd == -1) ERR.General("cpsBinaryWriter","write","No file is open");

    if(checksum){
      uint32_t crc = crc32(0L, data, len);
      ssize_t f = ::write(fd, &crc, sizeof(uint32_t));
      if(f==-1){
	perror("Write failed");
	ERR.General("cpsBinaryWriter","write","CRC write failed");    
      }else if(f != sizeof(uint32_t)){
	ERR.General("cpsBinaryWriter","write","CRC write did not write expected number of bytes, wrote %lu requested %lu", f, sizeof(uint32_t));
      }
    }
    ssize_t f = ::write(fd, data, len);
    if(f==-1){
      perror("Write failed");
      ERR.General("cpsBinaryWriter","write","Write failed for bytes %lu", len);
    }else if(f != len){
      ERR.General("cpsBinaryWriter","write","Write did not write expected number of bytes, wrote %lu requested %lu", f, len);
    }
  }

  void close(){
    if(fd!=-1){
      int e = ::close(fd);
      if(e == -1){
	perror("Failed to close file");
	ERR.General("cpsBinaryWriter","close","Failed to close file %s", file.c_str());
      }
      fd=-1;
    }
  }
  ~cpsBinaryWriter(){ close(); }
};


class cpsBinaryReader{
  int fd;
  const std::string file;
public:
  void open(const std::string &_file){
    fd = ::open(_file.c_str(), O_RDONLY);
    if(fd == -1){
      perror("Failed to open file");
      ERR.General("cpsBinaryReader","open","Failed to open file %s", _file.c_str());
    }
    file = _file;
  }
  
  cpsBinaryReader(): fd(-1){}
  cpsBinaryReader(const std::string &_file): fd(-1){ open(_file); }

  void read(void* data, size_t len, bool checksum = true) const{
    if(fd == -1) ERR.General("cpsBinaryReader","read","No file is open");

    uint32_t crc;
    if(checksum){
      ssize_t f = ::read(fd, &crc, sizeof(uint32_t));
      if(f==-1){
	perror("Read failed");
	ERR.General("cpsBinaryReader","read","CRC read failed");    
      }else if(f != sizeof(uint32_t)){
	ERR.General("cpsBinaryReader","read","CRC read did not read expected number of bytes, wrote %lu requested %lu", f, sizeof(uint32_t));
      }
    }
    ssize_t f = ::read(fd, data, len);
    if(f==-1){
      perror("Read failed");
      ERR.General("cpsBinaryReader","read","Read failed for bytes %lu", len);
    }else if(f != len){
      ERR.General("cpsBinaryReader","read","Read did not read expected number of bytes, wrote %lu requested %lu", f, len);
    }

    uint32_t crc_got = crc32(0L, data, len);
    if(crc_got != crc) ERR.General("cpsBinaryReader","read","CRC checksum failed");
  }

  void close(){
    if(fd!=-1){
      int e = ::close(fd);
      if(e == -1){
	perror("Failed to close file");
	ERR.General("cpsBinaryReader","close","Failed to close file %s", file.c_str());
      }
      fd=-1;
    }
  }
  ~cpsBinaryReader(){ close(); }
};




//Perform reduction via the disk rather than MPI
template<typename T>
void disk_reduce(T* data, size_t size){
  static int rc = 0;
  std::string df = "reddata." + std::to_string(UniqueID()) + "." + std::to_string(rc);

  disk_write_immediate(df,(void*)data,size*sizeof(T));
  cps::sync(); //Assume MPI barrier is available
  
  int nodes = GJP.TotalNodes();
  
  memset(data, 0, size*sizeof(T));

  T* buf = (T*)malloc_check(size*sizeof(T));
  for(int i=0;i<nodes;i++){
    std::string dfi = "reddata." + std::to_string(i) + "." + std::to_string(rc);
    disk_read(dfi, (void*)buf, size*sizeof(T));
    for(int j=0;j<size;j++) data[j] += buf[j];
  }
  free(buf);

  cps::sync();
  remove(df.c_str());
  ++rc;
}
  // std::string ddone = "done." + std::to_string(UniqueID()) + "." + std::to_string(rc);
  // disk_write_immediate(ddone,(void*)&rc,sizeof(int));  
  // //Wait for all done files to be written
  // std::vector<bool> done(GJP.Vol
  // 			 while(1){    
  // 	std::this_thread::sleep_for (std::chrono::milliseconds(100));

CPS_END_NAMESPACE

#endif
