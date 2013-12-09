#include <stdio.h>
#include <alg/lanc_arg.h>
#include <util/spincolorflavormatrix.h>

CPS_START_NAMESPACE

Rcomplex Trace(const SpinColorFlavorMatrix& a, const SpinColorFlavorMatrix& b){
  Rcomplex trace(0.0,0.0);
  for(int f2=0;f2<2;f2++){
    for(int f1=0;f1<2;f1++){
      for(int s2=0;s2<4;++s2){
	for(int c2=0;c2<3;++c2){
	  for(int s1=0;s1<4;++s1){
	    for(int c1=0;c1<3;++c1){
	      trace += a(s1,c1,f1,s2,c2,f2)*b(s2,c2,f2,s1,c1,f1);
	    }
	  }
	}
      }
    }
  }
  return trace;
}

CPS_END_NAMESPACE
