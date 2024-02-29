include ../../../../Makefile.users



DFLAGS += -DWRITE_HEX_OUTPUT
DFLAGS += -DDISABLE_EVALUATION_OF_SECOND_PIPI_TOPOLOGY
DFLAGS += -DDO_EXTENDED_CALC_V1

#Choose A2A precision (use double unless you run out of memory in the A2A contraction part)
DFLAGS += -DA2A_PREC_SIMD_DOUBLE
DFLAGS += -DA2A_USE_GRID_GFIX
DFLAGS += -DUSE_DESTRUCTIVE_FFT -DCOMPUTEMANY_INPLACE_SHIFT

#Write down V/W checkpoints for mission failure in the middle
DFLAGS += -DA2A_CHECKPOINT_INVERSIONS

##Async allreduce and compute when doing mesonfield
DFLAGS += -DMF_OFFLOAD_NODESUM_ASYNC
DFLAGS += -DMF_ASYNCREDUCE_THREAD

#DFLAGS += -DMF_STORAGE_TYPE=SET_MFSTORAGE_DISTRIBUTEDONESIDED
DFLAGS += -DMF_STORAGE_TYPE=SET_MFSTORAGE_MEMPOOL

LDFLAGS += -lgsl -lgslcblas -fopenmp -lrocblas -lrocfft -L/ccs/home/tianle/test_diff_module/rocm-5.3.0-mpich-8.1.17/install/Grid_crusher_Xconj/lib

#possible mpich flags for gpu-aware MPI
LDFLAGS += -L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa
