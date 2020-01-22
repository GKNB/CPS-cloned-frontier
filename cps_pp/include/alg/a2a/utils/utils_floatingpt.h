#ifndef _UTILS_FLOATINGPOINT_H_
#define _UTILS_FLOATINGPOINT_H_

#include <cassert>
#include <util/gjp.h>
#include <util/fpconv.h>
#include "utils_malloc.h"

//Utilities for floating point numbers

CPS_START_NAMESPACE

//Get the endian-ness of the machine
struct hostEndian{
  enum EndianType { BIG, LITTLE };
  inline static EndianType get(){ //copied from fpconv
    char end_check[4] = {1,0,0,0};
    uint32_t *lp = (uint32_t *)end_check;
    if ( *lp == 0x1 ) { 
      return LITTLE;
    } else {
      return BIG;
    }
  }
};

//Get the floating point standard of the machine for a type T
template<typename T>
struct FPformat{
  inline static FP_FORMAT get(){ //also taken from fpconv
    assert(sizeof(T) == 4 || sizeof(T) == 8);
    static const hostEndian::EndianType endian = hostEndian::get();
    
    if(sizeof(T) == 8){
      return endian == hostEndian::LITTLE ? FP_IEEE64LITTLE : FP_IEEE64BIG;
    }else {  // 32 bits
      union { 
	float pinum;
	char pichar[4];
      }cpspi;

      FP_FORMAT format;
      
      cpspi.pinum = FPConv_PI;
      if(endian == hostEndian::BIG) {
	format = FP_IEEE32BIG;
	for(int i=0;i<4;i++) {
	  if(cpspi.pichar[i] != FPConv_ieee32pi_big[i]) {
	    format = FP_TIDSP32;
	    break;
	  }
	}
      }
      else {
	format = FP_IEEE32LITTLE;
	for(int i=0;i<4;i++) {
	  if(cpspi.pichar[i] != FPConv_ieee32pi_big[3-i]) {
	    format = FP_TIDSP32;
	    break;
	  }
	}
      }
      return format;
    } // end of 32 bits
  }   
};

template<typename T>
struct FPformat<std::complex<T> >{
  inline static FP_FORMAT get(){ return FPformat<T>::get(); }
};

#ifdef USE_GRID
template<>
struct FPformat<Grid::vComplexD>{
  inline static FP_FORMAT get(){ return FPformat<double>::get(); }
};
template<>
struct FPformat<Grid::vComplexF>{
  inline static FP_FORMAT get(){ return FPformat<float>::get(); }
};
#endif

template<typename T>
class arrayIO{
  FPConv conv;
  FP_FORMAT fileformat;
  FP_FORMAT dataformat;  
  int dsize; //underlying floating point data type
  int nd_in_T; //number of floats in T
public:
  arrayIO(FP_FORMAT _fileformat = FP_AUTOMATIC): fileformat(_fileformat){
    dataformat = FPformat<T>::get();
    if(fileformat == FP_AUTOMATIC)
      fileformat = dataformat;
    else if(conv.size(fileformat) != conv.size(dataformat))
      ERR.General("arrayIO","arrayIO","Size of fileformat %s differs from size of data format %s\n",conv.name(fileformat),conv.name(dataformat));

    conv.setHostFormat(dataformat);
    conv.setFileFormat(fileformat);

    dsize = conv.size(dataformat); //underlying floating point data type
    assert(sizeof(T) % dsize == 0);
    nd_in_T = sizeof(T)/dsize;
  }
  arrayIO(const char* _fileformat){
    dataformat = FPformat<T>::get();

    conv.setHostFormat(dataformat);
    conv.setFileFormat(_fileformat);
    fileformat = conv.fileFormat;

    if(conv.size(fileformat) != conv.size(dataformat))
      ERR.General("arrayIO","arrayIO","Size of fileformat %s differs from size of data format %s\n",conv.name(fileformat),conv.name(dataformat));
    
    dsize = conv.size(dataformat); //underlying floating point data type
    assert(sizeof(T) % dsize == 0);
    nd_in_T = sizeof(T)/dsize;
  }
  std::string getFileFormatString() const{ return conv.name(fileformat); }
  
  unsigned int checksum(T const* v, const int size){
    return conv.checksum( (char*)v, nd_in_T*size, dataformat);    
  }
  void write(std::ostream &to, T const* v, const int size){
    static const int chunk = 32768; //32kb chunks
    assert(chunk % dsize == 0);
    int fdinchunk = chunk/dsize;
    char* wbuf = (char*)malloc_check(chunk * sizeof(char));     
    char const* dptr = (char const*)v;
    
    int off = 0;
    int nfd = nd_in_T*size;
    while(off < nfd){
      int grab = std::min(nfd-off, fdinchunk); //How many data elements to grab
      int grabchars = grab * dsize;
      conv.host2file(wbuf,dptr,grab);
      to.write(wbuf,grabchars);
      off += grab;
      dptr += grabchars;
    }
    free(wbuf);
    to << '\n';
  }

  void read(std::istream &from, T* v, const int size){
    static const int chunk = 32768; //32kb chunks
    assert(chunk % dsize == 0);
    int fdinchunk = chunk/dsize;
    char *rbuf = (char *)malloc_check(chunk * sizeof(char)); //leave room for auto null char      
    char *dptr = (char *)v;

    int off = 0;
    int nfd = nd_in_T*size;
    while(off < nfd){
      int grab = std::min(nfd-off, fdinchunk); //How many data elements to grab
      int grabchars = grab * dsize;
      
      from.read(rbuf,grabchars);
      int got = from.gcount();
      
      if(from.gcount() != grabchars)
	ERR.General("arrayIO","read","Only managed to read %d chars, needed %d\n",from.gcount(),grabchars);
      
      conv.file2host(dptr,rbuf,grab);
      
      off += grab;
      dptr += grabchars;
    }
    free(rbuf);

    from.ignore(1); //newline
  }
  
};








CPS_END_NAMESPACE

#endif
