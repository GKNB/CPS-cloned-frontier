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
		 Grid::GridCartesian *FGrid, Grid::GridRedBlackCartesian *FrbGrid,
		 Grid::innerProductImplementation<GridFermionField> &inner_prod,
		 A2Apreconditioning precon_type = SchurOriginal){
  
  if(lanc_arg.N_true_get == 0){
    std::vector<Grid::RealD>().swap(eval); 	std::vector<GridFermionField>().swap(evec);      
    //eval.clear(); evec.clear();
    LOGA2A << "gridLanczos skipping because N_true_get = 0" << std::endl;
    return;
  }

  assert(lanc_arg.precon);
  Grid::SchurOperatorBase<GridFermionField>  *HermOp;
  if(precon_type == SchurOriginal) HermOp = new Grid::SchurDiagMooeeOperator<GridDirac, GridFermionField>(Ddwf);
  else if(precon_type == SchurDiagTwo) HermOp = new Grid::SchurDiagTwoOperator<GridDirac, GridFermionField>(Ddwf);
  else assert(0);
  
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

  a2a_printf("Chebyshev lo=%g hi=%g ord=%d\n",lo,hi,ord);
  
  Grid::Chebyshev<GridFermionField> Cheb(lo,hi,ord);
  Grid::PlainHermOp<GridFermionField> HermOpF(*HermOp);
  Grid::FunctionHermOp<GridFermionField> ChebF(Cheb,*HermOp); 
  Grid::ImplicitlyRestartedLanczosHermOpTester<GridFermionField> tester(HermOpF, inner_prod);
  Grid::ImplicitlyRestartedLanczos<GridFermionField> IRL(ChebF,HermOpF,tester,inner_prod,Nstop,Nk,Nm,resid,MaxIt);
  
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
  a2a_printf("Starting Grid RNG seeding for Lanczos\n");
  double time = -dclock();
  
  {
    Grid::GridParallelRNG RNG(FGrid);  
    RNG.SeedFixedIntegers({1,2,3,4});
    GridFermionField gauss(FGrid);
    gaussian(RNG, gauss);
    pickCheckerboard(Odd, src, gauss);
  }

  // //Use CPS RNG to generate the initial field
  // {
  //   LatRanGen lrgbak(LRG);

  //   typedef typename GridCPSfieldFermionFlavorPolicyMap<GridFermionField>::value FlavorPolicy;   
  //   CPSfermion5D<cps::ComplexD, FiveDpolicy<FlavorPolicy> > tmp;
  //   tmp.setGaussianRandom();

  //   LRG = lrgbak;
    
  //   GridFermionField src_all(FGrid);
  //   tmp.exportGridField(src_all);
  //   pickCheckerboard(Grid::Odd,src,src_all);
  // }

  a2a_print_time("gridLanczos","Gaussian src",time+dclock());
  time = -dclock();

  
  a2a_printf("Starting Lanczos algorithm with %d threads (omp_get_max_threads %d)\n", Grid::GridThread::GetThreads(),omp_get_max_threads());
  int Nconv;
  //IRL.normalise(src);
  IRL.calc(eval,evec,
	   src,
	   Nconv
	   , true
	   );
  
  a2a_print_time("gridLanczos","Algorithm",time+dclock());
#endif
  delete HermOp;
}
template<typename GridFermionField, typename GridGaugeField, typename GridDirac>
void gridLanczos(std::vector<Grid::RealD> &eval, std::vector<GridFermionField> &evec, const LancArg &lanc_arg,		 
		 GridDirac &Ddwf, const GridGaugeField& Umu,
		 Grid::GridCartesian *UGrid, Grid::GridRedBlackCartesian *UrbGrid,
		 Grid::GridCartesian *FGrid, Grid::GridRedBlackCartesian *FrbGrid,
		 A2Apreconditioning precon_type = SchurOriginal){
  Grid::innerProductImplementation<GridFermionField> inner;
  gridLanczos(eval,evec,lanc_arg,Ddwf,Umu,UGrid,UrbGrid,FGrid,FrbGrid,inner,precon_type);
}

  
//Construct Grids and Dirac operator
template<typename GridPolicies>
void gridLanczos(std::vector<Grid::RealD> &eval, 
		 std::vector<typename GridPolicies::GridFermionField> &evec, 
		 const LancArg &lanc_arg, typename GridPolicies::FgridGFclass &lattice,
		 A2Apreconditioning precon_type = SchurOriginal){
  typedef typename GridPolicies::GridFermionField GridFermionField;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDirac GridDirac;

  if(!lattice.getGridFullyInitted()) ERR.General("","gridLanczos","Grid/Grids are not initialized!");
  
  Grid::GridCartesian *UGrid = lattice.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();
  Grid::GridCartesian *FGrid = lattice.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();
  Grid::LatticeGaugeFieldD *Umu = lattice.getUmu();

  double mob_b = lattice.get_mob_b();
  double mob_c = mob_b - 1.;   //b-c = 1
  double M5 = GJP.DwfHeight();
  a2a_printf("Creating Grid Dirac operator with b=%g c=%g b+c=%g mass=%g M5=%g\n",mob_b,mob_c,mob_b+mob_c,lanc_arg.mass,M5);

  typename GridDirac::ImplParams params;
  lattice.SetParams(params);
 
  GridDirac Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,lanc_arg.mass,M5,mob_b,mob_c, params);

  gridLanczos(eval, evec, lanc_arg, Ddwf, *Umu, UGrid, UrbGrid, FGrid, FrbGrid, precon_type);

  //Ddwf.Report();
}

template<typename GridPolicies>
void gridSinglePrecLanczos(std::vector<Grid::RealD> &eval, std::vector<typename GridPolicies::GridFermionFieldF> &evec, const LancArg &lanc_arg, typename GridPolicies::FgridGFclass &lattice,
			   Grid::GridCartesian *UGrid_f, Grid::GridRedBlackCartesian *UrbGrid_f,
			   Grid::GridCartesian *FGrid_f, Grid::GridRedBlackCartesian *FrbGrid_f,
			   A2Apreconditioning precon_type = SchurOriginal
			   ){
  typedef typename GridPolicies::GridFermionFieldF GridFermionFieldF;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDiracF GridDiracF;

  if(!lattice.getGridFullyInitted()) ERR.General("","gridSinglePrecLanczos","Grid/Grids are not initialized!");
  
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
  a2a_printf("Creating Grid Dirac operator with b=%g c=%g b+c=%g mass=%g M5=%g\n",mob_b,mob_c,mob_b+mob_c,lanc_arg.mass,M5);

  typename GridDiracF::ImplParams params;
  lattice.SetParams(params);

  GridDiracF Ddwf(Umu_f,*FGrid_f,*FrbGrid_f,*UGrid_f,*UrbGrid_f,lanc_arg.mass,M5,mob_b,mob_c, params);

  gridLanczos(eval, evec, lanc_arg, Ddwf, Umu_f, UGrid_f, UrbGrid_f, FGrid_f, FrbGrid_f, precon_type);

  //Ddwf.Report();
}

template<typename GridPolicies>
void gridLanczosXconj(std::vector<Grid::RealD> &eval, 
		      std::vector<typename GridPolicies::GridXconjFermionField> &evec, 
		      const LancArg &lanc_arg, typename GridPolicies::FgridGFclass &lattice,
		      A2Apreconditioning precon_type = SchurOriginal){
  typedef typename GridPolicies::GridXconjFermionField GridFermionField;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDiracXconj GridDirac;
  typedef typename GridPolicies::GridDirac GridDiracGP;
  typedef typename GridPolicies::GridFermionField GridFermionFieldGP;

  if(!lattice.getGridFullyInitted()) ERR.General("","gridLanczosXconj","Grid/Grids are not initialized!");
  
  Grid::GridCartesian *UGrid = lattice.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();
  Grid::GridCartesian *FGrid = lattice.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();
  Grid::LatticeGaugeFieldD *Umu = lattice.getUmu();

  double mob_b = lattice.get_mob_b();
  double mob_c = mob_b - 1.;   //b-c = 1
  double M5 = GJP.DwfHeight();
  typename GridDiracGP::ImplParams gp_params;
  lattice.SetParams(gp_params);

  typename GridDirac::ImplParams params;
  params.twists = gp_params.twists;
  params.boundary_phase = 1.0;

  a2a_printf("Creating X-conjugate Grid Dirac operator with b=%g c=%g b+c=%g mass=%g M5=%g phase=(%f,%f) twists=(%d,%d,%d,%d)\n",mob_b,mob_c,mob_b+mob_c,lanc_arg.mass,M5,params.boundary_phase.real(),params.boundary_phase.imag(),params.twists[0],params.twists[1],params.twists[2],params.twists[3]);

  GridDirac Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,lanc_arg.mass,M5,mob_b,mob_c, params);
  Grid::innerProductImplementationXconjugate<GridFermionField> inner;
  gridLanczos(eval,evec,lanc_arg,Ddwf,Umu,UGrid,UrbGrid,FGrid,FrbGrid,inner,precon_type);
}




CPS_END_NAMESPACE

#endif
#endif
