#ifndef _KTOPIPI_MAIN_A2A_POLICIES_H_
#define _KTOPIPI_MAIN_A2A_POLICIES_H_

//Get the allocation type
#ifdef USE_DESTRUCTIVE_FFT
#define ALLOC_TYPE SET_A2AVECTOR_MANUAL_ALLOC
#else
#define ALLOC_TYPE SET_A2AVECTOR_AUTOMATIC_ALLOC
#endif

//Get the mesonfield storage type;  override default using -DMF_STORAGE_TYPE=<type>
#ifndef MF_STORAGE_TYPE
#define MF_STORAGE_TYPE SET_MFSTORAGE_DISTRIBUTED
#endif

//Get whether to use SIMD or non-SIMD policy
#if defined(A2A_PREC_SIMD_DOUBLE) || defined(A2A_PREC_SIMD_DOUBLE)
#define POLICIES_MACRO A2APOLICIES_SIMD_TEMPLATE
#else
#define POLICIES_MACRO A2APOLICIES_TEMPLATE
#endif

#ifndef A2A_ALLOC_POLICY
#define A2A_ALLOC_POLICY ExplicitCopyDiskBackedPoolAllocPolicy
#endif

POLICIES_MACRO(A2Apolicies, 1, BaseGridPoliciesGparity, ALLOC_TYPE, MF_STORAGE_TYPE, A2A_ALLOC_POLICY);

#endif
