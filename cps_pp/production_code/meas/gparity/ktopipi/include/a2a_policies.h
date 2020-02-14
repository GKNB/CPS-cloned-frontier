#ifndef _KTOPIPI_MAIN_A2A_POLICIES_H_
#define _KTOPIPI_MAIN_A2A_POLICIES_H_

//Setup the A2A policy
#ifdef USE_DESTRUCTIVE_FFT

#ifdef A2A_PREC_DOUBLE
typedef A2ApoliciesDoubleManualAllocGparity A2Apolicies;
#elif defined(A2A_PREC_SINGLE)
typedef A2ApoliciesSingleManualAllocGparity A2Apolicies;
#elif defined(A2A_PREC_SIMD_DOUBLE)
typedef A2ApoliciesSIMDdoubleManualAllocGparity A2Apolicies;
#elif defined(A2A_PREC_SIMD_SINGLE)
typedef A2ApoliciesSIMDsingleManualAllocGparity A2Apolicies;
#else
#error "Must provide an A2A precision"
#endif

#else

#ifdef A2A_PREC_DOUBLE
typedef A2ApoliciesDoubleAutoAllocGparity A2Apolicies;
#elif defined(A2A_PREC_SINGLE)
typedef A2ApoliciesSingleAutoAllocGparity A2Apolicies;
#elif defined(A2A_PREC_SIMD_DOUBLE)
typedef A2ApoliciesSIMDdoubleAutoAllocGparity A2Apolicies;
#elif defined(A2A_PREC_SIMD_SINGLE)
typedef A2ApoliciesSIMDsingleAutoAllocGparity A2Apolicies;
#else
#error "Must provide an A2A precision"
#endif

#endif

#endif
