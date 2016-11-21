#include<simd/Intel512common.h>

CPS_START_NAMESPACE

#define ROUT %zmm0 
#define TMP1 %zmm1 
#define TMP2 %zmm2 
#define TMP3 %zmm3 
#define TMP4 %zmm4
#define TMP5 %zmm29
#define TMP6 %zmm30
#define TMP7 %zmm31

#define LPTR %rdi 
#define RPTR %rsi 

  //Some new macro definitions
#define GETREAL(FROM,INTO) "vmovddup " #FROM "," #INTO ";\n"
#define GETREALMEM(OFF,PTR,INTO) "vmovddup " #OFF"*64("#PTR")," #INTO ";\n"

#define GETIMAG(FROM,INTO) "vpshufd $0xFF," #FROM"," #INTO ";\n"
#define GETIMAGMEM(OFF,PTR,INTO) "vpshufd $0xFF," #OFF"*64("#PTR")," #INTO ";\n"

#define PERMUTEREIM(FROM,INTO) "vpermilpd $0x55," #FROM"," #INTO ";\n"
#define PERMUTEREIMMEM(OFF,PTR,INTO) "vpermilpd $0x55," #OFF"*64("#PTR")," #INTO ";\n"

  //In Intel notation  vfmaddsub132pd A1 A2 A3    A1 = A1*A3 +/-(even/odd) A2.  
  //In GAS notation the argument order is reversed so   (INTEL) vfmaddsub132pd A1 A2 A3 = (GAS) vfmaddsub132pd A3 A2 A1
  
  //Result is placed in MULB
  //Here the parity of a word is counted from the right, eg (7,6,5,4,3,2,1,0)  with 0, 2, 4, 6 even, 1, 3, 5, 7 odd
#define MULSUBEVENADDODD(ADDSUB, MULA,MULB) "vfmaddsub132pd " #MULA "," #ADDSUB "," #MULB ";\n"
#define MULADDEVENSUBODD(ADDSUB, MULA,MULB) "vfmsubadd132pd " #MULA "," #ADDSUB "," #MULB ";\n"
#define VNMADDd(A,B,DEST)       "vfnmadd231pd   " #A "," #B "," #DEST  ";\n"

  //Work around macro expansion rules to expand arguments first
#define _GETREAL(FROM,INTO) GETREAL(FROM,INTO)
#define _GETREALMEM(OFF,PTR,INTO) GETREALMEM(OFF,PTR,INTO)
#define _GETIMAG(FROM,INTO) GETIMAG(FROM,INTO)
#define _GETIMAGMEM(OFF,PTR,INTO) GETIMAGMEM(OFF,PTR,INTO)
#define _PERMUTEREIM(FROM,INTO) PERMUTEREIM(FROM,INTO)
#define _PERMUTEREIMMEM(OFF,PTR,INTO) PERMUTEREIMMEM(OFF,PTR,INTO)
#define _MULADDEVENSUBODD(ADDSUB, MULA,MULB) MULADDEVENSUBODD(ADDSUB, MULA,MULB)
#define _MULSUBEVENADDODD(ADDSUB, MULA,MULB) MULSUBEVENADDODD(ADDSUB, MULA,MULB)

#define _VLOADd(OFF,PTR,REG) VLOADd(OFF,PTR,REG)
#define _VMULd(A,B,DEST) VMULd(A,B,DEST)
#define _VADDd(A,B,DEST) VADDd(A,B,DEST)
#define _VSUBd(A,B,DEST) VSUBd(A,B,DEST)
#define _VZEROd(A) VZEROd(A)
#define _VMADDd(A,B,DEST) VMADDd(A,B,DEST)
#define _VNMADDd(A,B,DEST) VNMADDd(A,B,DEST)
#define _VMOVd(A,DEST) VMOVd(A,DEST)
#define _VSTOREd(OFF,PTR,SRC) VSTOREd(OFF,PTR,SRC) 
#define _VPREFETCH1(O,A) VPREFETCH1(O,A)

  //12 __m512d for left and 12 for right but no reuse. Could use 24 registers to load
#define L0 %zmm5 
#define L1 %zmm6
#define L2 %zmm7
#define L3 %zmm8
#define L4 %zmm9
#define L5 %zmm10
#define L6 %zmm11
#define L7 %zmm12
#define L8 %zmm13
#define L9 %zmm14
#define L10 %zmm15
#define L11 %zmm16

#define R0 %zmm17
#define R1 %zmm18
#define R2 %zmm19
#define R3 %zmm20
#define R4 %zmm21
#define R5 %zmm22
#define R6 %zmm23
#define R7 %zmm24
#define R8 %zmm25
#define R9 %zmm26
#define R10 %zmm27
#define R11 %zmm28



#define LOADL	    \
  _VLOADd(0,LPTR,L0)				\
  _VLOADd(1,LPTR,L1)				\
  _VLOADd(2,LPTR,L2)				\
  _VLOADd(3,LPTR,L3)				\
  _VLOADd(4,LPTR,L4)				\
  _VLOADd(5,LPTR,L5)				\
  _VLOADd(6,LPTR,L6)				\
  _VLOADd(7,LPTR,L7)				\
  _VLOADd(8,LPTR,L8)				\
  _VLOADd(9,LPTR,L9)				\
  _VLOADd(10,LPTR,L10)				\
  _VLOADd(11,LPTR,L11)

#define LOADR	    \
    _VLOADd(0,RPTR,R0)				\
    _VLOADd(1,RPTR,R1)				\
    _VLOADd(2,RPTR,R2)				\
    _VLOADd(3,RPTR,R3)				\
    _VLOADd(4,RPTR,R4)				\
    _VLOADd(5,RPTR,R5)				\
    _VLOADd(6,RPTR,R6)				\
    _VLOADd(7,RPTR,R7)				\
    _VLOADd(8,RPTR,R8)				\
    _VLOADd(9,RPTR,R9)				\
    _VLOADd(10,RPTR,R10)				\
    _VLOADd(11,RPTR,R11)

  //a_real -> OUT   (a_re, a_re)
  //a_imag -> TMP1   (a_im, a_im)
  //swap re/im of b -> TMP2  (b_im, b_re) -> (b_re, b_im)
  //(a_im,a_im)*(b_re, b_im) = (a_im*b_re, a_im*b_im)  TMP1 * TMP2 -> TMP1 
  //(a_re, a_re) * (b_im, b_re) +/- TMP1 = (a_re * b_im + a_im*b_re, a_re * b_re - a_im*b_im)   -> OUT

  //OUT cannot be MULA or MULB, or TMP1, TMP2
#define ZMUL(MULA,MULB,OUT) \
  _GETREAL(MULA,OUT) \
  _GETIMAG(MULA,TMP1) \  
  _PERMUTEREIM(MULB,TMP2) \
  _VMULd(TMP1,TMP2,TMP1) \
  _MULSUBEVENADDODD(TMP1,MULB,OUT)

    //conj(a)*b =  (a_re * b_im - a_im*b_re, a_re * b_re + a_im*b_im)

#define ZCONJMUL(MULA,MULB,OUT) \
  _GETREAL(MULA,OUT) \
  _GETIMAG(MULA,TMP1) \  
  _PERMUTEREIM(MULB,TMP2) \
  _VMULd(TMP1,TMP2,TMP1) \
  _MULADDEVENSUBODD(TMP1,MULB,OUT)

#define ZCONJMUL_TMPREGPASS(MULA,MULB,OUT,MYTMP1,MYTMP2)	\
  _GETREAL(MULA,OUT) \
  _GETIMAG(MULA,MYTMP1) \  
  _PERMUTEREIM(MULB,MYTMP2) \
  _VMULd(MYTMP1,MYTMP2,MYTMP1) \
  _MULADDEVENSUBODD(MYTMP1,MULB,OUT)

    //Doesn't seem to be a mem version of vfmsubadd132pd, grrr
#define ZCONJMUL_TMPREGPASS_MEM(OFF,LPTR,RPTR,OUT,MYTMP1,MYTMP2,MYTMP3)	\
  _VLOADd(OFF,RPTR,MYTMP3) \
  _GETREALMEM(OFF,LPTR,OUT)						\
  _GETIMAGMEM(OFF,LPTR,MYTMP1) \
  _PERMUTEREIM(MYTMP3,MYTMP2)		\
  _VMULd(MYTMP1,MYTMP2,MYTMP1) \
  _MULADDEVENSUBODD(MYTMP1,MYTMP3,OUT)


#define ZCONJMUL2_TMPREGPASS_MEM(OFF1,LPTR1,RPTR1,OUT1,		OFF2,LPTR2,RPTR2,OUT2,	       MYTMP11,MYTMP12,MYTMP13, MYTMP21,MYTMP22,MYTMP23) \
  _VLOADd(OFF1,RPTR1,MYTMP13)			\
  _VLOADd(OFF2,RPTR2,MYTMP23)			\
  _GETREALMEM(OFF1,LPTR1,OUT1)			\
  _GETREALMEM(OFF2,LPTR2,OUT2)			\
  _GETIMAGMEM(OFF1,LPTR1,MYTMP11) \
  _GETIMAGMEM(OFF2,LPTR2,MYTMP21) \
  _PERMUTEREIM(MYTMP13,MYTMP12)		\
  _PERMUTEREIM(MYTMP23,MYTMP22)		\
  _VMULd(MYTMP11,MYTMP12,MYTMP11) \
  _VMULd(MYTMP21,MYTMP22,MYTMP21) \
  _MULADDEVENSUBODD(MYTMP11,MYTMP13,OUT1) \
  _MULADDEVENSUBODD(MYTMP21,MYTMP23,OUT2)


/* #define ZCONJMUL2_TMPREGPASS_MEM(OFF1,LPTR1,RPTR1,OUT1,		OFF2,LPTR2,RPTR2,OUT2,	       MYTMP11,MYTMP12,MYTMP13, MYTMP21,MYTMP22,MYTMP23) \ */
/*   ZCONJMUL_TMPREGPASS_MEM(OFF1,LPTR1,RPTR1,OUT1, MYTMP11,MYTMP12,MYTMP13) \ */
/*   ZCONJMUL_TMPREGPASS_MEM(OFF2,LPTR2,RPTR2,OUT2, MYTMP21,MYTMP22,MYTMP23) */

__attribute__((noinline)) __m512d g5d_conjl_r_asm_avx512(__m512d const*l, __m512d const*r){
   // __asm__ ( 
   // 	     LOADL \
   // 	     LOADR \
   // 	     _VZEROd(ROUT) \
   // 	     ZCONJMUL(L0,R0,TMP3) \
   // 	     _VADDd(TMP3,ROUT,ROUT) \
   // 	     ZCONJMUL(L1,R1,TMP3) \
   // 	     _VADDd(TMP3,ROUT,ROUT) \
   // 	     ZCONJMUL(L2,R2,TMP3) \
   // 	     _VADDd(TMP3,ROUT,ROUT) \
   // 	     ZCONJMUL(L3,R3,TMP3) \
   // 	     _VADDd(TMP3,ROUT,ROUT) \
   // 	     ZCONJMUL(L4,R4,TMP3) \
   // 	     _VADDd(TMP3,ROUT,ROUT) \
   // 	     ZCONJMUL(L5,R5,TMP3) \
   // 	     _VADDd(TMP3,ROUT,ROUT) \
   // 	     ZCONJMUL(L6,R6,TMP3) \
   // 	     _VSUBd(TMP3,ROUT,ROUT) \
   // 	     ZCONJMUL(L7,R7,TMP3) \
   // 	     _VSUBd(TMP3,ROUT,ROUT) \
   // 	     ZCONJMUL(L8,R8,TMP3) \
   // 	     _VSUBd(TMP3,ROUT,ROUT) \
   // 	     ZCONJMUL(L9,R9,TMP3) \
   // 	     _VSUBd(TMP3,ROUT,ROUT) \
   // 	     ZCONJMUL(L10,R10,TMP3) \
   // 	     _VSUBd(TMP3,ROUT,ROUT) \
   // 	     ZCONJMUL(L11,R11,TMP3) \
   // 	     _VSUBd(TMP3,ROUT,ROUT) \
   // 	      );


    //sigma_gen_exp3.qcdknl0.phys.columbia.edu
   // __asm__ (
   // 	     LOADL \
   // 	     LOADR \
   // 	     ZCONJMUL(L0,R0,ROUT) \
   // 	     ZCONJMUL(L1,R1,TMP3) \
   // 	     ZCONJMUL(L2,R2,TMP4) \
   // 	     ZCONJMUL(L3,R3,TMP5) \
   // 	     ZCONJMUL(L4,R4,TMP6) \
   // 	     ZCONJMUL(L5,R5,TMP7) \
   // 	     _VADDd(TMP3,ROUT,ROUT) \
   // 	     _VADDd(TMP4,ROUT,ROUT) \
   // 	     _VADDd(TMP5,ROUT,ROUT) \
   // 	     _VADDd(TMP6,ROUT,ROUT) \
   // 	     _VADDd(TMP7,ROUT,ROUT) \
   // 	     ZCONJMUL(L6,R6,TMP3) \
   // 	     ZCONJMUL(L7,R7,TMP4) \
   // 	     ZCONJMUL(L8,R8,TMP5) \
   // 	     ZCONJMUL(L9,R9,TMP6) \
   // 	     ZCONJMUL(L10,R10,TMP7) \
   // 	     _VSUBd(TMP3,ROUT,ROUT) \
   // 	     _VSUBd(TMP4,ROUT,ROUT) \
   // 	     _VSUBd(TMP5,ROUT,ROUT) \
   // 	     _VSUBd(TMP6,ROUT,ROUT) \
   // 	     _VSUBd(TMP7,ROUT,ROUT) \
   // 	     ZCONJMUL(L11,R11,TMP3) \
   // 	     _VSUBd(TMP3,ROUT,ROUT)
   // 	      );

    //sigma_gen_exp4.qcdknl0.phys.columbia.edu
   // __asm__ (
   // 	     LOADL \
   // 	     LOADR \
   // 	     ZCONJMUL(L0,R0,ROUT) \
   // 	     ZCONJMUL(L1,R1,TMP3) \
   // 	     _VADDd(TMP3,ROUT,ROUT) \
   // 	     ZCONJMUL(L2,R2,TMP4) \
   // 	     _VADDd(TMP4,ROUT,ROUT) \
   // 	     ZCONJMUL(L3,R3,TMP5) \
   // 	     _VADDd(TMP5,ROUT,ROUT) \
   // 	     ZCONJMUL(L4,R4,TMP6) \
   // 	     _VADDd(TMP6,ROUT,ROUT) \
   // 	     ZCONJMUL(L5,R5,TMP7) \
   // 	     _VADDd(TMP7,ROUT,ROUT) \
   // 	     ZCONJMUL(L6,R6,TMP3) \
   // 	     ZCONJMUL(L7,R7,TMP4) \
   // 	     ZCONJMUL(L8,R8,TMP5) \
   // 	     ZCONJMUL(L9,R9,TMP6) \
   // 	     ZCONJMUL(L10,R10,TMP7) \
   // 	     _VSUBd(TMP3,ROUT,ROUT) \
   // 	     _VSUBd(TMP4,ROUT,ROUT) \
   // 	     _VSUBd(TMP5,ROUT,ROUT) \
   // 	     _VSUBd(TMP6,ROUT,ROUT) \
   // 	     _VSUBd(TMP7,ROUT,ROUT) \
   // 	     ZCONJMUL(L11,R11,TMP3) \
   // 	     _VSUBd(TMP3,ROUT,ROUT)
   // 	      );

    //    sigma_gen_exp5.qcdknl0.phys.columbia.edu
   // __asm__ (
   // 	     LOADL \
   // 	     LOADR \
   // 	     ZCONJMUL(L0,R0,ROUT) \
   // 	     ZCONJMUL(L1,R1,TMP3) \
   // 	     _VADDd(TMP3,ROUT,ROUT) \
   // 	     ZCONJMUL(L2,R2,TMP4) \
   // 	     _VADDd(TMP4,ROUT,ROUT) \
   // 	     ZCONJMUL(L3,R3,TMP5) \
   // 	     _VADDd(TMP5,ROUT,ROUT) \
   // 	     ZCONJMUL(L4,R4,TMP6) \
   // 	     _VADDd(TMP6,ROUT,ROUT) \
   // 	     ZCONJMUL(L5,R5,TMP7) \
   // 	     _VADDd(TMP7,ROUT,ROUT) \
   // 	     ZCONJMUL(L6,R6,TMP3) \
   // 	     _VSUBd(TMP3,ROUT,ROUT) \
   // 	     ZCONJMUL(L7,R7,TMP4) \
   // 	     _VSUBd(TMP4,ROUT,ROUT) \
   // 	     ZCONJMUL(L8,R8,TMP5) \
   // 	     _VSUBd(TMP5,ROUT,ROUT) \
   // 	     ZCONJMUL(L9,R9,TMP6) \
   // 	     _VSUBd(TMP6,ROUT,ROUT) \
   // 	     ZCONJMUL(L10,R10,TMP7) \
   // 	     _VSUBd(TMP7,ROUT,ROUT) \
   // 	     ZCONJMUL(L11,R11,L0) \
   // 	     _VSUBd(L0,ROUT,ROUT)
   // 	    ); //note reuse of L0 at end of chain


  //sigma_gen_exp6.qcdknl0.phys.columbia.edu
   /* __asm__ ( */
   /* 	     LOADL \ */
   /* 	     LOADR \ */
   /* 	     ZCONJMUL_TMPREGPASS(L0,R0,ROUT,  TMP1,TMP2)	\ */
   /* 	     ZCONJMUL_TMPREGPASS(L1,R1,TMP3,  TMP4,TMP5)	\ */
   /* 	     _VADDd(TMP3,ROUT,ROUT) \ */
   /* 	     ZCONJMUL_TMPREGPASS(L2,R2,TMP6,  TMP7,TMP1)	\ */
   /* 	     _VADDd(TMP6,ROUT,ROUT) \ */
   /* 	     ZCONJMUL_TMPREGPASS(L3,R3,TMP2, TMP3,TMP4)	\ */
   /* 	     _VADDd(TMP2,ROUT,ROUT) \ */
   /* 	     ZCONJMUL_TMPREGPASS(L4,R4,TMP5,  TMP6,TMP7)		\ */
   /* 	     _VADDd(TMP5,ROUT,ROUT) \ */
   /* 	     ZCONJMUL_TMPREGPASS(L5,R5,TMP1, TMP2,TMP3)		\ */
   /* 	     _VADDd(TMP1,ROUT,ROUT) \ */
   /* 	     ZCONJMUL_TMPREGPASS(L6,R6,TMP4,  TMP5,TMP6)	\ */
   /* 	     _VSUBd(TMP4,ROUT,ROUT) \ */
   /* 	     ZCONJMUL_TMPREGPASS(L7,R7,TMP7,  TMP1,TMP2)	\ */
   /* 	     _VSUBd(TMP7,ROUT,ROUT) \ */
   /* 	     ZCONJMUL_TMPREGPASS(L8,R8,TMP3, TMP4,TMP5)	\ */
   /* 	     _VSUBd(TMP3,ROUT,ROUT) \ */
   /* 	     ZCONJMUL_TMPREGPASS(L9,R9,TMP6,  TMP7,TMP1)		\ */
   /* 	     _VSUBd(TMP6,ROUT,ROUT) \ */
   /* 	     ZCONJMUL_TMPREGPASS(L10,R10,TMP2,  TMP3,TMP4)	\ */
   /* 	     _VSUBd(TMP2,ROUT,ROUT) \ */
   /* 	     ZCONJMUL_TMPREGPASS(L11,R11,TMP5,  TMP6,TMP7)	\ */
   /* 	     _VSUBd(TMP5,ROUT,ROUT) */
   /* 	    );  */

    //This version eliminates the loads at the beginning in favor of fused operations from L1. It also frees more registers to cycle through.
	    /* _VPREFETCH1(0,LPTR) \ */
	    /* _VPREFETCH1(0,RPTR) \ */
	    /* _VPREFETCH1(1,LPTR) \ */
	    /* _VPREFETCH1(1,RPTR) \ */
	    /* _VPREFETCH1(2,LPTR) \ */
	    /* _VPREFETCH1(2,RPTR) \ */
	    /* _VPREFETCH1(3,LPTR) \ */
	    /* _VPREFETCH1(3,RPTR) \ */
	    /* _VPREFETCH1(4,LPTR) \ */
	    /* _VPREFETCH1(4,RPTR) \ */
	    /* _VPREFETCH1(5,LPTR) \ */
	    /* _VPREFETCH1(5,RPTR) \ */
	    /* _VPREFETCH1(6,LPTR) \ */
	    /* _VPREFETCH1(6,RPTR) \ */
	    /* _VPREFETCH1(7,LPTR) \ */
	    /* _VPREFETCH1(7,RPTR) \ */
	    /* _VPREFETCH1(8,LPTR) \ */
	    /* _VPREFETCH1(8,RPTR) \ */
	    /* _VPREFETCH1(9,LPTR) \ */
	    /* _VPREFETCH1(9,RPTR) \ */
	    /* _VPREFETCH1(10,LPTR) \ */
	    /* _VPREFETCH1(10,RPTR) \ */
	    /* _VPREFETCH1(11,LPTR) \ */
	    /* _VPREFETCH1(11,RPTR) \ */

   /* __asm__ ( */
   /* 	    ZCONJMUL_TMPREGPASS_MEM(0,LPTR,RPTR,ROUT,  L0,L1,L2)	\ */
   /* 	    ZCONJMUL_TMPREGPASS_MEM(1,LPTR,RPTR,L3,    L4,L5,L6)		\ */
   /* 	    ZCONJMUL_TMPREGPASS_MEM(2,LPTR,RPTR,L7,    L8,L9,L10)	\ */
   /* 	    _VADDd(L3,ROUT,ROUT) \ */
   /* 	    ZCONJMUL_TMPREGPASS_MEM(3,LPTR,RPTR,L11,   R0,R1,R2)	\ */
   /* 	    _VADDd(L7,ROUT,ROUT) \ */
   /* 	    ZCONJMUL_TMPREGPASS_MEM(4,LPTR,RPTR,R3,    R4,R5,R6)	\ */
   /* 	    _VADDd(L11,ROUT,ROUT) \ */
   /* 	    ZCONJMUL_TMPREGPASS_MEM(5,LPTR,RPTR,R7,    R8,R9,R10)	\ */
   /* 	    _VADDd(R3,ROUT,ROUT) \ */
   /* 	    ZCONJMUL_TMPREGPASS_MEM(6,LPTR,RPTR,R11,   TMP1,TMP2,TMP3)	\ */
   /* 	    _VADDd(R7,ROUT,ROUT) \ */
   /* 	    ZCONJMUL_TMPREGPASS_MEM(7,LPTR,RPTR,TMP4,  TMP5,TMP6,TMP7)	\ */
   /* 	    _VSUBd(R11,ROUT,ROUT) \ */
   /* 	    ZCONJMUL_TMPREGPASS_MEM(8,LPTR,RPTR,L0,    L1,L2,L3)		\ */
   /* 	    _VSUBd(TMP4,ROUT,ROUT) \ */
   /* 	    ZCONJMUL_TMPREGPASS_MEM(9,LPTR,RPTR,L4,    L5,L6,L7)		\ */
   /* 	    _VSUBd(L0,ROUT,ROUT) \ */
   /* 	    ZCONJMUL_TMPREGPASS_MEM(10,LPTR,RPTR,L8,   L9,L10,L11)		\ */
   /* 	    _VSUBd(L4,ROUT,ROUT) \ */
   /* 	    ZCONJMUL_TMPREGPASS_MEM(11,LPTR,RPTR,R0,   R1,R2,R3)		\ */
   /* 	    _VSUBd(L8,ROUT,ROUT) \ */
   /* 	    _VSUBd(R0,ROUT,ROUT) */
   /* 	    ); */


	    /* _VPREFETCH1(1,LPTR) \ */
	    /* _VPREFETCH1(1,RPTR) \ */
	    /* _VPREFETCH1(2,LPTR) \ */
	    /* _VPREFETCH1(2,RPTR) \ */
	    /* _VPREFETCH1(3,LPTR) \ */
	    /* _VPREFETCH1(3,RPTR) \ */
	    /* _VPREFETCH1(4,LPTR) \ */
	    /* _VPREFETCH1(4,RPTR) \ */
	    /* _VPREFETCH1(5,LPTR) \ */
	    /* _VPREFETCH1(5,RPTR) \ */
	    /* _VPREFETCH1(6,LPTR) \ */
	    /* _VPREFETCH1(6,RPTR) \ */
	    /* _VPREFETCH1(7,LPTR) \ */
	    /* _VPREFETCH1(7,RPTR) \ */
	    /* _VPREFETCH1(8,LPTR) \ */
	    /* _VPREFETCH1(8,RPTR) \ */
	    /* _VPREFETCH1(9,LPTR) \ */
	    /* _VPREFETCH1(9,RPTR) \ */
	    /* _VPREFETCH1(10,LPTR) \ */
	    /* _VPREFETCH1(10,RPTR) \ */
	    /* _VPREFETCH1(11,LPTR) \ */
	    /* _VPREFETCH1(11,RPTR) \ */

   __asm__ (
   	    ZCONJMUL2_TMPREGPASS_MEM(0,LPTR,RPTR,ROUT,  1,LPTR,RPTR,L0,   L1,L2,L3,L4,L5,L6) \
   	    ZCONJMUL2_TMPREGPASS_MEM(2,LPTR,RPTR,L7,    3,LPTR,RPTR,L8,   L9,L10,L11,R0,R1,R2) \
   	    _VADDd(L0,ROUT,ROUT) \
   	    ZCONJMUL2_TMPREGPASS_MEM(4,LPTR,RPTR,R3,    5,LPTR,RPTR,R4,   R5,R6,R7,R8,R9,R10) \
   	    _VADDd(L7,ROUT,ROUT) \
   	    ZCONJMUL2_TMPREGPASS_MEM(6,LPTR,RPTR,R11,   7,LPTR,RPTR,TMP1, TMP2,TMP3,TMP4,TMP5,TMP6,TMP7) \
   	    _VADDd(L8,ROUT,ROUT) \
   	    ZCONJMUL2_TMPREGPASS_MEM(8,LPTR,RPTR,L0,     9,LPTR,RPTR,L1,  L2,L3,L4,L5,L6,L7)	\
   	    _VADDd(R3,ROUT,ROUT) \
   	    ZCONJMUL2_TMPREGPASS_MEM(10,LPTR,RPTR,L8,    11,LPTR,RPTR,L9, L10,L11,R0,R1,R2,R3) \
   	    _VADDd(R4,ROUT,ROUT) \
   	    _VADDd(R11,TMP1,TMP1) \
   	    _VADDd(L0,L1,L1) \
   	    _VADDd(L8,L9,L9) \
   	    _VSUBd(TMP1,ROUT,ROUT) \
   	    _VADDd(L1,L9,L9) \
   	    _VSUBd(L9,ROUT,ROUT)
   	    );



 	    /* _VADDd(R4,ROUT,ROUT) \ */
   	    /* _VSUBd(R11,ROUT,ROUT) \   */
 	    /* _VSUBd(TMP1,ROUT,ROUT) \ */
   	    /* _VSUBd(L0,ROUT,ROUT) \ */
   	    /* _VSUBd(L1,ROUT,ROUT) \ */
   	    /* _VSUBd(L8,ROUT,ROUT) \ */
   	    /* _VSUBd(L9,ROUT,ROUT) */





}



__attribute__((noinline)) __m512d gunitd_conjl_r_asm_avx512(__m512d const*l, __m512d const*r){
   __asm__ (
	    ZCONJMUL_TMPREGPASS_MEM(0,LPTR,RPTR,ROUT,  L0,L1,L2)	\
   	    ZCONJMUL_TMPREGPASS_MEM(1,LPTR,RPTR,L3,    L4,L5,L6)		\
   	    ZCONJMUL_TMPREGPASS_MEM(2,LPTR,RPTR,L7,    L8,L9,L10)	\
   	    _VADDd(L3,ROUT,ROUT) \
   	    ZCONJMUL_TMPREGPASS_MEM(3,LPTR,RPTR,L11,   R0,R1,R2)	\
   	    _VADDd(L7,ROUT,ROUT) \
   	    ZCONJMUL_TMPREGPASS_MEM(4,LPTR,RPTR,R3,    R4,R5,R6)	\
   	    _VADDd(L11,ROUT,ROUT) \
   	    ZCONJMUL_TMPREGPASS_MEM(5,LPTR,RPTR,R7,    R8,R9,R10)	\
   	    _VADDd(R3,ROUT,ROUT) \
   	    ZCONJMUL_TMPREGPASS_MEM(6,LPTR,RPTR,R11,   TMP1,TMP2,TMP3)	\
   	    _VADDd(R7,ROUT,ROUT) \
   	    ZCONJMUL_TMPREGPASS_MEM(7,LPTR,RPTR,TMP4,  TMP5,TMP6,TMP7)	\
   	    _VADDd(R11,ROUT,ROUT) \
   	    ZCONJMUL_TMPREGPASS_MEM(8,LPTR,RPTR,L0,    L1,L2,L3)		\
   	    _VADDd(TMP4,ROUT,ROUT) \
   	    ZCONJMUL_TMPREGPASS_MEM(9,LPTR,RPTR,L4,    L5,L6,L7)		\
   	    _VADDd(L0,ROUT,ROUT) \
   	    ZCONJMUL_TMPREGPASS_MEM(10,LPTR,RPTR,L8,   L9,L10,L11)		\
   	    _VADDd(L4,ROUT,ROUT) \
   	    ZCONJMUL_TMPREGPASS_MEM(11,LPTR,RPTR,R0,   R1,R2,R3)		\
   	    _VADDd(L8,ROUT,ROUT) \
   	    _VADDd(R0,ROUT,ROUT)
   	    );
}




CPS_END_NAMESPACE
