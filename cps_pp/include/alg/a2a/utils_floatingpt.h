#ifndef _UTILS_FLOATINGPOINT_H_
#define _UTILS_FLOATINGPOINT_H_

#include <cassert>
#include <util/gjp.h>
#include <util/fpconv.h>
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


CPS_END_NAMESPACE

#endif
