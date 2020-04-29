#ifndef _A2A_GRID_LANCZOS_H
#define _A2A_GRID_LANCZOS_H

#ifdef USE_GRID
#include<util/time_cps.h>
#include<util/lattice/fgrid.h>
#include<alg/lanc_arg.h>
#include<alg/a2a/lattice/CPSfield.h>

CPS_START_NAMESPACE

//Call Grid Lanczos with given Dirac operator
template<typename GridFermionField, typename GridGaugeField, typename GridDirac>
void gridLanczos(std::vector<Grid::RealD> &eval, std::vector<GridFermionField> &evec, const LancArg &lanc_arg,		 
		 GridDirac &Ddwf, const GridGaugeField& Umu,
		 Grid::GridCartesian *UGrid, Grid::GridRedBlackCartesian *UrbGrid,
		 Grid::GridCartesian *FGrid, Grid::GridRedBlackCartesian *FrbGrid){
  
  if(lanc_arg.N_true_get == 0){
    std::vector<Grid::RealD>().swap(eval); 	std::vector<GridFermionField>().swap(evec);      
    //eval.clear(); evec.clear();
    if(!UniqueID()) printf("gridLanczos skipping because N_true_get = 0\n");
    return;
  }

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

  if(!UniqueID()) printf("Chebyshev lo=%g hi=%g ord=%d\n",lo,hi,ord);
  
  Grid::Chebyshev<GridFermionField> Cheb(lo,hi,ord);
#ifdef USE_CHULWOOS_LANCZOS
#warning "Using Chulwoo's Grid Lanczos implementation"
  Grid::ImplicitlyRestartedLanczosCJ<GridFermionField> IRL(HermOp,Cheb,Nstop,Nk,Nm,resid,MaxIt);
#else
#warning "Using default Grid Lanczos implementation"
  Grid::PlainHermOp<GridFermionField> HermOpF(HermOp);
  Grid::FunctionHermOp<GridFermionField> ChebF(Cheb,HermOp); 
  Grid::ImplicitlyRestartedLanczos<GridFermionField> IRL(ChebF,HermOpF,Nstop,Nk,Nm,resid,MaxIt);
  
  //Grid::ImplicitlyRestartedLanczos<GridFermionField> IRL(HermOp,Cheb,Nstop,Nk,Nm,resid,MaxIt);
#endif
  //if(lanc_arg.lock) IRL.lock = 1;
  if(lanc_arg.lock) ERR.General("::","gridLanczos","Grid Lanczos does not currently support locking\n");
  
  eval.resize(Nm);
  evec.reserve(Nm);
  evec.resize(Nm, FrbGrid);
 
  for(int i=0;i<Nm;i++){
    evec[i].Checkerboard() = Grid::Odd;
  }
  GridFermionField src(FrbGrid);
  
#ifndef MEMTEST_MODE
  if(!UniqueID()) printf("Starting Grid RNG seeding for Lanczos\n");
  double time = -dclock();
  
# if 0
  std::vector<int> seeds5({5,6,7,8});
  Grid::GridParallelRNG RNG5rb(FrbGrid);  RNG5rb.SeedFixedIntegers(seeds5);

  print_time("gridLanczos","RNG seeding",time+dclock());
  time = -dclock();

  if(!UniqueID()) printf("Initializing Gaussian src\n");
  gaussian(RNG5rb,src);
  src.checkerboard = Grid::Odd;
# else
  {
    LatRanGen lrgbak(LRG);
    
    CPSfermion5D<cps::ComplexD> tmp;
    tmp.setGaussianRandom();

    LRG = lrgbak;
    
    GridFermionField src_all(FGrid);
    tmp.exportGridField(src_all);
    pickCheckerboard(Grid::Odd,src,src_all);
  }
# endif

  print_time("gridLanczos","Gaussian src",time+dclock());
  time = -dclock();

  
  if(!UniqueID()) printf("Starting Lanczos algorithm with %d threads (omp_get_max_threads %d)\n", Grid::GridThread::GetThreads(),omp_get_max_threads());
  int Nconv;
  IRL.normalise(src);
  IRL.calc(eval,evec,
	   src,
	   Nconv
#ifndef USE_CHULWOOS_LANCZOS
	   , true
#endif
	   );
  
  print_time("gridLanczos","Algorithm",time+dclock());
#endif
}

  
//Construct Grids and Dirac operator
template<typename GridPolicies>
void gridLanczos(std::vector<Grid::RealD> &eval, std::vector<typename GridPolicies::GridFermionField> &evec, const LancArg &lanc_arg, typename GridPolicies::FgridGFclass &lattice){
  typedef typename GridPolicies::GridFermionField GridFermionField;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDirac GridDirac;
  
  Grid::GridCartesian *UGrid = lattice.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();
  Grid::GridCartesian *FGrid = lattice.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();
  Grid::LatticeGaugeFieldD *Umu = lattice.getUmu();

  double mob_b = lattice.get_mob_b();
  double mob_c = mob_b - 1.;   //b-c = 1
  double M5 = GJP.DwfHeight();
  if(!UniqueID()) printf("Grid b=%g c=%g b+c=%g\n",mob_b,mob_c,mob_b+mob_c);

  typename GridDirac::ImplParams params;
  lattice.SetParams(params);

  GridDirac Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,lanc_arg.mass,M5,mob_b,mob_c, params);

  gridLanczos(eval, evec, lanc_arg, Ddwf, *Umu, UGrid, UrbGrid, FGrid, FrbGrid);

  //Ddwf.Report();
}

template<typename GridPolicies>
void gridSinglePrecLanczos(std::vector<Grid::RealD> &eval, std::vector<typename GridPolicies::GridFermionFieldF> &evec, const LancArg &lanc_arg, typename GridPolicies::FgridGFclass &lattice,
			   Grid::GridCartesian *UGrid_f, Grid::GridRedBlackCartesian *UrbGrid_f,
			   Grid::GridCartesian *FGrid_f, Grid::GridRedBlackCartesian *FrbGrid_f
			   ){
  typedef typename GridPolicies::GridFermionFieldF GridFermionFieldF;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDiracF GridDiracF;
  
  Grid::LatticeGaugeFieldD *Umu = lattice.getUmu();

  Grid::LatticeGaugeFieldF Umu_f(UGrid_f);
  //Grid::precisionChange(Umu_f,*Umu);
  
  NullObject null_obj;
  lattice.BondCond();
  CPSfield<cps::ComplexD,4*9,FourDpolicy<OneFlavorPolicy> > cps_gauge((cps::ComplexD*)lattice.GaugeField(),null_obj);
  cps_gauge.exportGridField(Umu_f);
  lattice.BondCond();

  double mob_b = lattice.get_mob_b();
  double mob_c = mob_b - 1.;   //b-c = 1
  double M5 = GJP.DwfHeight();
  if(!UniqueID()) printf("Grid b=%g c=%g b+c=%g\n",mob_b,mob_c,mob_b+mob_c);

  typename GridDiracF::ImplParams params;
  lattice.SetParams(params);

  GridDiracF Ddwf(Umu_f,*FGrid_f,*FrbGrid_f,*UGrid_f,*UrbGrid_f,lanc_arg.mass,M5,mob_b,mob_c, params);

  gridLanczos(eval, evec, lanc_arg, Ddwf, Umu_f, UGrid_f, UrbGrid_f, FGrid_f, FrbGrid_f);

  //Ddwf.Report();
}



CPS_END_NAMESPACE

#endif
#endif