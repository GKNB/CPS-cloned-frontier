#ifdef USE_MPI
//mpi headers
#warning "WARNING : USING MPI"
#include<mpi.h>
#endif

#include <vector>
#include <fftw3.h>

#include <alg/eigen/Krylov_5d.h>
#include <util/spincolorflavormatrix.h>
#include <util/vector.h>
#include <alg/a2a/compute_ktopipi_base.h>


CPS_START_NAMESPACE

SpinColorFlavorMatrix ComputeKtoPiPiGparityBase::S2(gamma5,sigma3);
SpinColorFlavorMatrix ComputeKtoPiPiGparityBase::_F0(spin_unit,F0);
SpinColorFlavorMatrix ComputeKtoPiPiGparityBase::_F1(spin_unit,F1);
SpinColorFlavorMatrix ComputeKtoPiPiGparityBase::g5(gamma5,sigma0);
SpinColorFlavorMatrix ComputeKtoPiPiGparityBase::unit(spin_unit,sigma0);
SpinColorFlavorMatrix ComputeKtoPiPiGparityBase::gmu[4] = { SpinColorFlavorMatrix(gamma1,sigma0), 
						  SpinColorFlavorMatrix(gamma2,sigma0),
						  SpinColorFlavorMatrix(gamma3,sigma0),
						  SpinColorFlavorMatrix(gamma4,sigma0) };

CPS_END_NAMESPACE
