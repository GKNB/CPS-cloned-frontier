#ifndef _UTILS_MAIN_H_
#define _UTILS_MAIN_H_

#include <sys/mman.h>

#include <util/time_cps.h>
#include <util/lattice/fgrid.h>
#include <alg/meas_arg.h>
#include <alg/ktopipi_jobparams.h>
#include <alg/a2a/lattice/CPSfield.h>
#include <alg/alg_fix_gauge.h>

//Useful functions for main programs
CPS_START_NAMESPACE


inline void printLogNodeFile(const char* fmt, ...){
  va_list argptr;
  va_start(argptr, fmt);

  static int calls = 0;

  std::ostringstream os; os << "node_log." << UniqueID();
  FILE* out = fopen (os.str().c_str(), calls == 0 ? "w" : "a");
  if(out == NULL){
    printf("Non-fatal error in printLogNodeFile on node %d: could not open file %s with mode %c\n",UniqueID(),os.str().c_str(),calls==0 ? 'w' : 'a');
    fflush(stdout);
  }else{
    vfprintf(out,fmt,argptr);
    fclose(out);
  }
  calls++;

  va_end(argptr);
}


//Gauge and RNG read
void ReadGaugeField(const MeasArg &meas_arg, bool double_latt = false);

void ReadRngFile(const MeasArg &meas_arg, bool double_latt = false);

//Read both gauge and RNG field contingent on DoArgs start_conf_kind and start_seed_kind, allowing easy use of random lattices for testing
void readGaugeRNG(const DoArg &do_arg, const MeasArg &meas_arg, const bool double_latt);

//Rotate the temporal boundary links by a phase exp(i degrees/180 *pi)
void TboundaryTwist(const double degrees);

//Initialize OpenMP, GJP and QDP
void initCPS(int argc, char **argv, const DoArg &do_arg, const int nthreads);

//Do the gauge fixing if !skip_gauge_fix
void doGaugeFix(Lattice &lat, const bool skip_gauge_fix, const FixGaugeArg &fix_gauge_arg);


inline void freeGridSharedMem(){
#if defined(USE_GRID_A2A) || defined(USE_GRID_LANCZOS)

#ifdef GRID_SHMEM_FREE_OLD //For older versions before MPI/MPI3 union (late Dec 2017)
  munmap(Grid::CartesianCommunicator::ShmCommBuf, Grid::CartesianCommunicator::MAX_MPI_SHM_BYTES);
#elif !defined(GPU_VEC)
  //Current version does not appear to properly dealloc the memory under the GPU compile
  Grid::GlobalSharedMemory::SharedMemoryFree();
#endif

#endif
}

CPS_END_NAMESPACE


#endif
