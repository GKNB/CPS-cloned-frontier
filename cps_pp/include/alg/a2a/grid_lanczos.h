#ifndef _A2A_GRID_LANCZOS_H
#define _A2A_GRID_LANCZOS_H

#ifdef USE_GRID
#include<util/lattice/fgrid.h>

CPS_START_NAMESPACE

template<typename GridPolicies>
inline void gridLanczos(std::vector<Grid::RealD> &eval, std::vector<typename GridPolicies::GridFermionField> &evec, const LancArg &lanc_arg, typename GridPolicies::FgridGFclass &lattice){
  typedef typename GridPolicies::GridFermionField GridFermionField;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDirac GridDirac;
  
  if(lanc_arg.N_true_get == 0){
    eval.clear(); evec.clear();
    if(!UniqueID()) printf("gridLanczos skipping because N_true_get = 0\n");
    return;
  }
  
  Grid::GridCartesian *UGrid = lattice.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();
  Grid::GridCartesian *FGrid = lattice.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();
  Grid::QCD::LatticeGaugeFieldD *Umu = lattice.getUmu();

  double mob_b = lattice.get_mob_b();
  double mob_c = mob_b - 1.;   //b-c = 1
  double M5 = GJP.DwfHeight();
  if(!UniqueID()) printf("Grid b=%g c=%g b+c=%g\n",mob_b,mob_c,mob_b+mob_c);

  typename GridDirac::ImplParams params;
  lattice.SetParams(params);

  for(int i=0;i<4;i++)
    VRB.Result("","gridLanczos","UGrid.gdimensions[%d]=%d\n",i,UGrid->_gdimensions[i]);
  for(int i=0;i<4;i++)
    VRB.Result("","gridLanczos","UrbGrid.gdimensions[%d]=%d\n",i,UrbGrid->_gdimensions[i]);
  for(int i=0;i<5;i++)
    VRB.Result("","gridLanczos","FGrid.gdimensions[%d]=%d\n",i,FGrid->_gdimensions[i]);
   for(int i=0;i<5;i++)
    VRB.Result("","gridLanczos","FrbGrid.gdimensions[%d]=%d\n",i,FrbGrid->_gdimensions[i]);

   for(int i=0;i<4;i++)
    VRB.Result("","gridLanczos","UGrid.fdimensions[%d]=%d\n",i,UGrid->_fdimensions[i]);
  for(int i=0;i<4;i++)
    VRB.Result("","gridLanczos","UrbGrid.fdimensions[%d]=%d\n",i,UrbGrid->_fdimensions[i]);
  for(int i=0;i<5;i++)
    VRB.Result("","gridLanczos","FGrid.fdimensions[%d]=%d\n",i,FGrid->_fdimensions[i]);
   for(int i=0;i<5;i++)
    VRB.Result("","gridLanczos","FrbGrid.fdimensions[%d]=%d\n",i,FrbGrid->_fdimensions[i]);

   
  GridDirac Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,lanc_arg.mass,M5,mob_b,mob_c, params);
  assert(lanc_arg.precon);
  Grid::SchurDiagMooeeOperator<GridDirac, GridFermionField> HermOp(Ddwf);

    // int Nstop;   // Number of evecs checked for convergence
    // int Nk;      // Number of converged sought
    // int Np;      // Np -- Number of spare vecs in kryloc space
    // int Nm;      // Nm -- total number of vectors

  const int Nstop = lanc_arg.N_true_get;
  const int Nk = lanc_arg.N_get;
  const int Np = lanc_arg.N_use - lanc_arg.N_get;
  const int Nm = lanc_arg.N_use;
  const int MaxIt= lanc_arg.maxits;
  Grid::RealD resid = lanc_arg.stop_rsd;

  double lo = lanc_arg.ch_beta * lanc_arg.ch_beta;
  double hi = lanc_arg.ch_alpha * lanc_arg.ch_alpha;
  int ord = lanc_arg.ch_ord + 1; //different conventions

  Grid::Chebyshev<GridFermionField> Cheb(lo,hi,ord);
  Grid::ImplicitlyRestartedLanczos<GridFermionField> IRL(HermOp,Cheb,Nstop,Nk,Nm,resid,MaxIt);

  if(lanc_arg.lock) IRL.lock = 1;

  eval.resize(Nm);
  evec.reserve(Nm);
  evec.resize(Nm, FrbGrid);
 
  for(int i=0;i<Nm;i++){
    evec[i].checkerboard = Grid::Odd;
  }

  std::vector<int> seeds5({5,6,7,8});
  Grid::GridParallelRNG RNG5rb(FrbGrid);  RNG5rb.SeedFixedIntegers(seeds5);

  GridFermionField src(FrbGrid);
  gaussian(RNG5rb,src);
  src.checkerboard = Grid::Odd;

  int Nconv;
  IRL.calc(eval,evec,
	   src,
	   Nconv);
}




CPS_END_NAMESPACE

#endif
#endif
