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

void write_data_bypass_cache(const std::string &file, char const* data, size_t bytes){
  int fd = open(file.c_str(), O_WRONLY | O_CREAT | O_DIRECT | O_DSYNC | O_TRUNC, S_IRWXU);
  if(fd == -1){
    perror(errno);
    ERR.General("","write_data_bypass_cache","Failed to open %s", file.c_str());
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
    memcpy(buf, data, count);
    ssize_t f = write(fd, buf, count);
    if(f==-1){
      perror(errno);
      ERR.General("","write_data_bypass_cache","Write failed");    
    }else if(f != count){
      ERR.General("","write_data_bypass_cache","Write did not write expected number of bytes");    
    }
    bytes -= count;
    data += count;
  }  
  int e = close(fd);
  if(e == -1){
    perror(errno);
    ERR.General("","write_data_bypass_cache","Failed to close file");
  }
  free(buf);
}


void read_data_bypass_cache(const std::string &file, char * data, size_t bytes){
  int fd = open(file.c_str(), O_RDONLY | O_DIRECT);
  if(fd == -1){
    perror(errno);
    ERR.General("","read_data_bypass_cache","Failed to open %s", file.c_str());
  }

  /* //Need an aligned memory region */
  size_t bufsz = 10*1024*1024;
  void* buf = memalign_check(4096, bufsz);
  
  while(bytes > 0){
    size_t count = std::min(bytes, bufsz);
    ssize_t f = read(fd, buf, count);
    if(f==-1){
      perror(errno);
      ERR.General("","read_data_bypass_cache","Read failed");    
    }else if(f != count){
      ERR.General("","read_data_bypass_cache","Read did not write expected number of bytes");    
    }
    memcpy(data, buf, count);
    bytes -= count;
    data += count;
  }  
  int e = close(fd);
  if(e == -1){
    perror(errno);
    ERR.General("","read_data_bypass_cache","Failed to close file");
  }
  free(buf);
}



CPS_END_NAMESPACE

#endif
