#ifndef _A2A_GRID_LANCZOS_H
#define _A2A_GRID_LANCZOS_H

#ifdef USE_GRID
#include<util/lattice/fgrid.h>

CPS_START_NAMESPACE

template<typename GridFermionField, typename GridGaugeField, typename GridDirac>
void gridLanczos(GridFermionField &src, std::vector<Grid::RealD> &eval, std::vector<GridFermionField> &evec, const LancArg &lanc_arg,		 
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
  HermOp.MpcNorm=false;
  HermOp.MpcDagNorm=false;


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
  if(!UniqueID()) printf("Chebyshev lo=%g hi=%g ord=%d Cheb(0)=%g \n",lo,hi,ord,Cheb.approx(0));
  Grid::ImplicitlyRestartedLanczos<GridFermionField> IRL(HermOp,Cheb,Nstop,Nk,Nm,resid,MaxIt);

  if(lanc_arg.lock) IRL.lock = 1;

  eval.resize(Nm);
  evec.reserve(Nm);
  evec.resize(Nm, FrbGrid);
 
  for(int i=0;i<Nm;i++){
    evec[i].checkerboard = Grid::Odd;
  }
#ifndef MEMTEST_MODE
#if 0
  GridFermionField src(FrbGrid);
#if 1
  std::vector<int> seeds5({5,6,7,8});
  Grid::GridParallelRNG RNG5rb(FrbGrid);  RNG5rb.SeedFixedIntegers(seeds5);

  gaussian(RNG5rb,src);
//  Grid::CartesianCommunicator::Barrier();
#else
//  src=1.; //hack to save init time during testing
#endif
#endif
  Float temp=0.; glb_sum(&temp);
  src.checkerboard = Grid::Odd;
  
  int Nconv;
  IRL.calc(eval,evec,
	   src,
	   Nconv);
#endif
}

  

template<typename GridPolicies>
void gridLanczos(std::vector<Grid::RealD> &eval, std::vector<typename GridPolicies::GridFermionField> &evec, const LancArg &lanc_arg, typename GridPolicies::FgridGFclass &lattice){
  typedef typename GridPolicies::GridFermionField GridFermionField;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDirac GridDirac;
  
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

  GridFermionField src(FrbGrid);
  GridFermionField src_all(FGrid);
{
  int n_gp=1; if (GJP.Gparity()) n_gp++;

  Vector *X_in =
        (Vector*)smalloc(GJP.VolNodeSites()*n_gp*lattice.FsiteSize()*sizeof(IFloat));
  lattice.RandGaussVector(X_in,0.5,1);
  lattice.ImportFermion(src_all,X_in,FgridBase::Odd);
  pickCheckerboard(Grid::Odd,src,src_all);
  sfree(X_in);
}

  GridDirac Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,lanc_arg.mass,M5,mob_b,mob_c, params);

  gridLanczos(src,eval, evec, lanc_arg, Ddwf, *Umu, UGrid, UrbGrid, FGrid, FrbGrid);
  int n_gp=1; if (GJP.Gparity()) n_gp++;
}

template<typename GridPolicies>
void gridSinglePrecLanczos(std::vector<Grid::RealD> &eval, std::vector<typename GridPolicies::GridFermionFieldF> &evec, const LancArg &lanc_arg, typename GridPolicies::FgridGFclass &lattice,
			   Grid::GridCartesian *UGrid_f, Grid::GridRedBlackCartesian *UrbGrid_f,
			   Grid::GridCartesian *FGrid_f, Grid::GridRedBlackCartesian *FrbGrid_f
			   ){
  typedef typename GridPolicies::GridFermionFieldF GridFermionFieldF;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDiracF GridDiracF;
  
  Grid::QCD::LatticeGaugeFieldD *Umu = lattice.getUmu();

  Grid::QCD::LatticeGaugeFieldF Umu_f(UGrid_f);
  //Grid::precisionChange(Umu_f,*Umu);
  
  NullObject null_obj;
  lattice.BondCond();
  CPSfield<ComplexD,4*9,FourDpolicy,OneFlavorPolicy> cps_gauge((ComplexD*)lattice.GaugeField(),null_obj);
  cps_gauge.exportGridField(Umu_f);
  lattice.BondCond();

  double mob_b = lattice.get_mob_b();
  double mob_c = mob_b - 1.;   //b-c = 1
  double M5 = GJP.DwfHeight();
  if(!UniqueID()) printf("Grid b=%g c=%g b+c=%g\n",mob_b,mob_c,mob_b+mob_c);

  typename GridDiracF::ImplParams params;
  lattice.SetParams(params);

  GridFermionFieldF src(FrbGrid_f);
  GridFermionFieldF src_all(FGrid_f);
{
  int n_gp=1; if (GJP.Gparity()) n_gp++;
  Vector *X_in =
        (Vector*)smalloc(GJP.VolNodeSites()*n_gp*lattice.FsiteSize()*sizeof(IFloat));
  lattice.RandGaussVector(X_in,0.5,1);
  lattice.ImportFermion(src_all,X_in,FgridBase::Odd);
  pickCheckerboard(Grid::Odd,src,src_all);
  sfree(X_in);
}

  GridDiracF Ddwf(Umu_f,*FGrid_f,*FrbGrid_f,*UGrid_f,*UrbGrid_f,lanc_arg.mass,M5,mob_b,mob_c, params);

  gridLanczos(src,eval, evec, lanc_arg, Ddwf, Umu_f, UGrid_f, UrbGrid_f, FGrid_f, FrbGrid_f);
}



CPS_END_NAMESPACE

#endif
#endif
