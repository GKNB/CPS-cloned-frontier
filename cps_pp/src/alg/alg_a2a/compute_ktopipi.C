#ifdef USE_MPI
//mpi headers
#warning "WARNING : USING MPI"
#include<mpi.h>
#endif

#include <cassert>
#include <vector>
#include <fftw3.h>

#include <util/spincolorflavormatrix.h>
#include <util/vector.h>
#include <alg/fix_gauge_arg.h>
#include <alg/a2a/ktopipi_gparity/compute_ktopipi_base.h>


CPS_START_NAMESPACE

CPS_END_NAMESPACE
