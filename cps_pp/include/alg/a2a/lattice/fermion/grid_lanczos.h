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
	   , false
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









//Call Grid Block Lanczos with given Dirac operator
template<typename GridFermionField, typename GridDirac>
void gridBlockLanczos(std::vector<Grid::RealD> &eval, std::vector<GridFermionField> &evec, const LancArg &lanc_arg,		 
		      GridDirac &Ddwf, GridDirac &Ddwf_split, int Nsplit,
		      Grid::innerProductImplementation<GridFermionField> &inner_prod,
		      A2Apreconditioning precon_type = SchurOriginal){
  
  if(lanc_arg.N_true_get == 0){
    std::vector<Grid::RealD>().swap(eval); 	std::vector<GridFermionField>().swap(evec);      
    LOGA2A << "gridBlockLanczos skipping because N_true_get = 0" << std::endl;
    return;
  }

  assert(lanc_arg.precon);
  Grid::SchurOperatorBase<GridFermionField>  *HermOp, *HermOp_s;
  if(precon_type == SchurOriginal){
    HermOp = new Grid::SchurDiagMooeeOperator<GridDirac, GridFermionField>(Ddwf);
    HermOp_s = new Grid::SchurDiagMooeeOperator<GridDirac, GridFermionField>(Ddwf_split);
  }
  else if(precon_type == SchurDiagTwo){
    HermOp = new Grid::SchurDiagTwoOperator<GridDirac, GridFermionField>(Ddwf);
    HermOp_s = new Grid::SchurDiagTwoOperator<GridDirac, GridFermionField>(Ddwf_split);
  }
  else assert(0);
 
    // int Nstop;   // Number of evecs checked for convergence
    // int Nk;      // Number of converged sought
    // int Np;      // Np -- Number of spare vecs in kryloc space
    // int Nm;      // Nm -- total number of vectors

  const int Nu = Nsplit; //number of parallel Lanczos' = nsplit

  const int Nstop = lanc_arg.N_true_get;
  const int Nk = lanc_arg.N_get; //Must be divisible by Nsplit
  const int Np = lanc_arg.N_use - lanc_arg.N_get; //NOTE: for block Lanczos, this should be divisible by Nu=Nsplit

  const int MaxIt= lanc_arg.maxits;  //NOTE: For block Lanczos, this is the max number of restarts; it should be small because the number of vectors scales like maxits!
  if(MaxIt > 50) ERR.General("::","gridBlockLanczos","Maxiters much too large! This will use up a tonne of RAM. You should expect only a small number of restarts");

  const int Nm = lanc_arg.N_use + Np * MaxIt; //NOTE: Must be divisible by Nsplit

  Grid::RealD resid = lanc_arg.stop_rsd; //NOTE: For block Lanczos, this should be larger than the normal residual by a factor of the largest eval of the op (obtainable eg using power method)

  double lo = lanc_arg.ch_beta * lanc_arg.ch_beta;
  double hi = lanc_arg.ch_alpha * lanc_arg.ch_alpha;
  int ord = lanc_arg.ch_ord + 1; //different conventions

  //NOTE:  Block Lanczos also requires a "skip" quantity which is used when doing the convergence check; it skips this many evecs between checks so as to avoid checking every evec. For use here we will hijack "ch_mu" and interpret it as a fraction of evecs to check
  int Nskip = int(lanc_arg.ch_mu * Nstop);
  if(Nskip == 0) Nskip = 1;

  LOGA2A << "Doing convergence check on every " << Nskip << "'th evec" << std::endl;

  if(lanc_arg.lock) ERR.General("::","gridBlockLanczos","Grid Lanczos does not currently support locking\n");

  a2a_printf("Chebyshev lo=%g hi=%g ord=%d\n",lo,hi,ord);

  Grid::Chebyshev<GridFermionField> Cheb(lo,hi,ord);
  Grid::ImplicitlyRestartedBlockLanczos<GridFermionField> IRL(*HermOp, *HermOp_s, (Grid::GridRedBlackCartesian *)Ddwf.FermionRedBlackGrid(), (Grid::GridRedBlackCartesian *)Ddwf_split.FermionRedBlackGrid(), Nsplit, 
							      Cheb, Nstop, Nskip,
							      Nu, Nk, Nm, resid, MaxIt, Grid::IRBLdiagonaliseWithEigen, inner_prod);

  eval.resize(Nm);
  evec.reserve(Nm);
  evec.resize(Nm, Ddwf.FermionRedBlackGrid());
 
  for(int i=0;i<Nm;i++){
    evec[i].Checkerboard() = Grid::Odd;
  }
  std::vector<GridFermionField> src(Nu, Ddwf.FermionRedBlackGrid());
  
#ifndef MEMTEST_MODE
  a2a_printf("Starting Grid RNG seeding for Lanczos\n");
  double time = -dclock();

  Grid::GridParallelRNG RNG(Ddwf.FermionGrid());
  //Grid::GridParallelRNG RNG(Ddwf.FermionRedBlackGrid());   //throws errors when seeding, likely not designed for preconditioned grids
  RNG.SeedFixedIntegers({1,2,3,4});
 
  GridFermionField gauss(Ddwf.FermionGrid());

  for(int i=0;i<Nu;i++){
    //gaussian(RNG, src[i]);
    //src[i].Checkerboard() = Grid::Odd;
    gaussian(RNG, gauss); 
    pickCheckerboard(Odd, src[i], gauss);
  }

  a2a_print_time("gridLanczos","Gaussian src",time+dclock());

#define TEST_SPLIT_GRID
#ifdef TEST_SPLIT_GRID
  {
    std::vector<GridFermionField> tmp(Nsplit, Ddwf.FermionRedBlackGrid());
    //gaussian(RNG, tmp[0]);
    gaussian(RNG, gauss); 
    pickCheckerboard(Odd, tmp[0], gauss);
    
    GridFermionField expect(Ddwf.FermionRedBlackGrid());
    HermOp->Mpc(tmp[0],expect);

    for(int i=1;i<Nsplit;i++) tmp[i] = tmp[0];
    GridFermionField tmp_split(Ddwf_split.FermionRedBlackGrid()),  got_split(Ddwf_split.FermionRedBlackGrid());
    Grid::Grid_split(tmp,tmp_split);
    HermOp_s->Mpc(tmp_split,got_split);

    std::vector<GridFermionField> got(Nsplit, Ddwf.FermionRedBlackGrid());
    Grid::Grid_unsplit(got,got_split);

    GridFermionField diff(Ddwf.FermionRedBlackGrid());
    for(int i=0;i<Nsplit;i++){
      got[i].Checkerboard() = Odd;

      diff = got[i] - expect;
      Grid::RealD n2diff = inner_prod.norm2(diff);
      LOGA2A << "Split test Mpc " << i << " got " << inner_prod.norm2(got[i]) << " expect " << inner_prod.norm2(expect) << " diff " << n2diff << std::endl;
    }

    Cheb(*HermOp, tmp[0], expect);
    Cheb(*HermOp_s, tmp_split, got_split);
    Grid::Grid_unsplit(got,got_split);

    for(int i=0;i<Nsplit;i++){
      diff = got[i] - expect;
      Grid::RealD n2diff = inner_prod.norm2(diff);
      LOGA2A << "Split test Cheby " << i << " got " << inner_prod.norm2(got[i]) << " expect " << inner_prod.norm2(expect) << " diff " << n2diff << std::endl;
    }

  }
#endif

  time = -dclock();

  LOGA2A << "Starting block Lanczos algorithm" << std::endl;
  
  int Nconv; //ignore this, the evecs will be resized to Nstop  
  IRL.calc(eval,evec,src,Nconv,Grid::LanczosType::rbl);
  //Block Lanczos sorts the output in descending order, we want ascending
  std::reverse(eval.begin(),eval.end());
  std::reverse(evec.begin(),evec.end());

  a2a_print_time("gridLanczos","Algorithm",time+dclock());
#endif
  delete HermOp;
  delete HermOp_s;
}

template<typename GridPolicies>
void gridBlockLanczosXconj(std::vector<Grid::RealD> &eval, 
			   std::vector<typename GridPolicies::GridXconjFermionField> &evec, 
			   const LancArg &lanc_arg, typename GridPolicies::FgridGFclass &lattice,
			   const std::vector<int> &split_grid_geom, //rank size of split grids
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

  Grid::Coordinate latt(4);
  int nsplit = 1;
  for(int i=0;i<4;i++){
    latt[i] = GJP.NodeSites(i)*GJP.Nodes(i);
    assert(GJP.Nodes(i) % split_grid_geom[i] == 0);
    nsplit *= GJP.Nodes(i)/split_grid_geom[i];
  }

  LOGA2A << "Setting up block Lanczos with " << nsplit << " subgrids" << std::endl;
  int Ls = GJP.SnodeSites()*GJP.Snodes();

  Grid::GridCartesian         * SUGrid = new Grid::GridCartesian(latt,
								 Grid::GridDefaultSimd(4,GridFermionField::vector_type::Nsimd()),
								 split_grid_geom,
								 *UGrid);

  Grid::GridCartesian         * SFGrid   = Grid::SpaceTimeGrid::makeFiveDimGrid(Ls,SUGrid);
  Grid::GridRedBlackCartesian * SUrbGrid  = Grid::SpaceTimeGrid::makeFourDimRedBlackGrid(SUGrid);
  Grid::GridRedBlackCartesian * SFrbGrid = Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,SUGrid);

  Grid::LatticeGaugeFieldD Umu_s(SUGrid);
  Grid::Grid_split(*Umu,Umu_s);


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
  GridDirac Ddwf_s(Umu_s,*SFGrid,*SFrbGrid,*SUGrid,*SUrbGrid,lanc_arg.mass,M5,mob_b,mob_c, params);

  Grid::innerProductImplementationXconjugate<GridFermionField> inner;
  gridBlockLanczos(eval,evec,lanc_arg,Ddwf,Ddwf_s,nsplit,inner,precon_type);
}



template<typename GridPolicies>
void gridBlockLanczosXconjSingle(std::vector<Grid::RealD> &eval, 
			   std::vector<typename GridPolicies::GridXconjFermionFieldF> &evec, 
			   const LancArg &lanc_arg, typename GridPolicies::FgridGFclass &lattice,
			   const std::vector<int> &split_grid_geom, //rank size of split grids
			   A2Apreconditioning precon_type = SchurOriginal){
  typedef typename GridPolicies::GridXconjFermionField GridFermionField;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDiracXconj GridDirac;
  typedef typename GridPolicies::GridDirac GridDiracGP;
  typedef typename GridPolicies::GridFermionField GridFermionFieldGP;

  typedef typename GridPolicies::GridXconjFermionFieldF GridFermionFieldF;
  typedef typename GridPolicies::GridDiracXconjF GridDiracF;
  typedef typename GridPolicies::GridDiracF GridDiracGPF;
  typedef typename GridPolicies::GridFermionFieldF GridFermionFieldGPF;

  if(!lattice.getGridFullyInitted()) ERR.General("","gridBlockLanczosXconjSingle","Grid/Grids are not initialized!");
  
  Grid::GridCartesian *UGrid = lattice.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();
  Grid::GridCartesian *FGrid = lattice.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();
  Grid::LatticeGaugeFieldD *Umu = lattice.getUmu();

  Grid::GridCartesian *UGridF = lattice.getUGridF();
  Grid::GridRedBlackCartesian *UrbGridF = lattice.getUrbGridF();
  Grid::GridCartesian *FGridF = lattice.getFGridF();
  Grid::GridRedBlackCartesian *FrbGridF = lattice.getFrbGridF();

  Grid::LatticeGaugeFieldF Umu_f(UGridF);
  precisionChange(Umu_f,*Umu);

  Grid::Coordinate latt(4);
  int nsplit = 1;
  for(int i=0;i<4;i++){
    latt[i] = GJP.NodeSites(i)*GJP.Nodes(i);
    assert(GJP.Nodes(i) % split_grid_geom[i] == 0);
    nsplit *= GJP.Nodes(i)/split_grid_geom[i];
  }

  LOGA2A << "Setting up block Lanczos with " << nsplit << " subgrids" << std::endl;
  int Ls = GJP.SnodeSites()*GJP.Snodes();

  Grid::GridCartesian         * SUGridF = new Grid::GridCartesian(latt,
								 Grid::GridDefaultSimd(4,GridFermionFieldF::vector_type::Nsimd()),
								 split_grid_geom,
								 *UGridF);

  Grid::GridCartesian         * SFGridF   = Grid::SpaceTimeGrid::makeFiveDimGrid(Ls,SUGridF);
  Grid::GridRedBlackCartesian * SUrbGridF  = Grid::SpaceTimeGrid::makeFourDimRedBlackGrid(SUGridF);
  Grid::GridRedBlackCartesian * SFrbGridF = Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,SUGridF);

  Grid::LatticeGaugeFieldF Umu_sf(SUGridF);
  Grid::Grid_split(Umu_f,Umu_sf);

  double mob_b = lattice.get_mob_b();
  double mob_c = mob_b - 1.;   //b-c = 1
  double M5 = GJP.DwfHeight();
  typename GridDiracGP::ImplParams gp_params;
  lattice.SetParams(gp_params);

  typename GridDirac::ImplParams params;
  params.twists = gp_params.twists;
  params.boundary_phase = 1.0;

  a2a_printf("Creating single precision X-conjugate Grid Dirac operator with b=%g c=%g b+c=%g mass=%g M5=%g phase=(%f,%f) twists=(%d,%d,%d,%d)\n",mob_b,mob_c,mob_b+mob_c,lanc_arg.mass,M5,params.boundary_phase.real(),params.boundary_phase.imag(),params.twists[0],params.twists[1],params.twists[2],params.twists[3]);

  GridDiracF Ddwf(Umu_f,*FGridF,*FrbGridF,*UGridF,*UrbGridF,lanc_arg.mass,M5,mob_b,mob_c, params);
  GridDiracF Ddwf_s(Umu_sf,*SFGridF,*SFrbGridF,*SUGridF,*SUrbGridF,lanc_arg.mass,M5,mob_b,mob_c, params);

  Grid::innerProductImplementationXconjugate<GridFermionFieldF> inner;
  gridBlockLanczos(eval,evec,lanc_arg,Ddwf,Ddwf_s,nsplit,inner,precon_type);
}





CPS_END_NAMESPACE

#endif
#endif
