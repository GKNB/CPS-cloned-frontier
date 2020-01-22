#ifndef _EVEC_INTERFACE_H
#define _EVEC_INTERFACE_H
#if defined(USE_GRID) && defined(USE_GRID_A2A)

#include<config.h>
#include<cstdint>
#include<precision.h>
#include<Grid/Grid.h>
#include<util/gjp.h>
#include<alg/ktopipi_jobparams.h>
#include "grid_split_cg.h"
CPS_START_NAMESPACE

//Unified interface for obtaining evecs and evals from either Grid- or BFM-computed Lanczos
template<typename GridPolicies>
class EvecInterface{
  typedef typename GridPolicies::GridFermionField GridFermionField;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDirac GridDirac;
 public:
  
  //Get an eigenvector and eigenvalue
  virtual Float getEvec(GridFermionField &into, const int idx) = 0;
  virtual int nEvecs() const = 0;

  //Allow the interface to choose which function computes the preconditioned M^dag M matrix inverse. Default is CG
  virtual void CGNE_MdagM(Grid::SchurDiagMooeeOperator<GridDirac,GridFermionField> &linop,
			  GridFermionField &solution, const GridFermionField &source,
			  const CGcontrols &cg_controls);

  //Multi-RHS version of the above
  virtual void CGNE_MdagM_multi(Grid::SchurDiagMooeeOperator<GridDirac,GridFermionField> &linop,
				std::vector<GridFermionField> &solution, const std::vector<GridFermionField> &source,
				const CGcontrols &cg_controls);
  virtual void Report() const{}
};

#include "implementation/evec_interface_impl.tcc"

CPS_END_NAMESPACE

#endif
#endif
