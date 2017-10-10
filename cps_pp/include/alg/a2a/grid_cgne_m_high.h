#ifndef _GRID_CGNE_M_HIGH_H
#define _GRID_CGNE_M_HIGH_H

#if defined(USE_GRID) && defined(USE_GRID_A2A)
#include<util/lattice/fgrid.h>
#include<alg/a2a/evec_interface.h>

CPS_START_NAMESPACE

//nLowMode is the number of modes we actually use to deflate. This must be <= evals.size(). The full set of computed eigenvectors is used to improve the guess.
template<typename GridPolicies>
inline void Grid_CGNE_M_high(typename GridPolicies::GridFermionField &solution, const typename GridPolicies::GridFermionField &source, const CGcontrols &cg_controls,
			     EvecInterface<GridPolicies> &evecs, int nLowMode, 
			     typename GridPolicies::FgridFclass &latg, typename GridPolicies::GridDirac &Ddwf, Grid::GridCartesian *FGrid, Grid::GridRedBlackCartesian *FrbGrid){
  typedef typename GridPolicies::GridFermionField GridFermionField;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDirac GridDirac;
  
  double f = norm2(source);
  if (!UniqueID()) printf("Grid_CGNE_M_high: Source norm is %le\n",f);
  f = norm2(solution);
  if (!UniqueID()) printf("Grid_CGNE_M_high: Guess norm is %le\n",f);

  Grid::SchurDiagMooeeOperator<GridDirac, GridFermionField> linop(Ddwf);

  GridFermionField tmp_cb1(FrbGrid);
  GridFermionField tmp_cb2(FrbGrid);
  GridFermionField tmp_cb3(FrbGrid);
  GridFermionField tmp_cb4(FrbGrid);

  GridFermionField tmp_full(FGrid);

  // src_o = Mprecdag * (source_o - Moe MeeInv source_e)  , cf Daiqian's thesis page 60
  GridFermionField src_o(FrbGrid);

  pickCheckerboard(Grid::Even,tmp_cb1,source);  //tmp_cb1 = source_e
  pickCheckerboard(Grid::Odd,tmp_cb2,source);   //tmp_cb2 = source_o

  Ddwf.MooeeInv(tmp_cb1,tmp_cb3);
  Ddwf.Meooe     (tmp_cb3,tmp_cb4); //tmp_cb4 = Moe MeeInv source_e       (tmp_cb3 free)
  axpy    (tmp_cb3,-1.0,tmp_cb4, tmp_cb2); //tmp_cb3 = (source_o - Moe MeeInv source_e)    (tmp_cb4 free)
  linop.MpcDag(tmp_cb3, src_o); //src_o = Mprecdag * (source_o - Moe MeeInv source_e)    (tmp_cb3, tmp_cb4 free)

  //Compute low-mode projection and CG guess
  int Nev = evecs.nEvecs();

  GridFermionField lsol_full(FrbGrid); //full low-mode part (all evecs)
  lsol_full = Grid::zero;

  GridFermionField lsol_defl(FrbGrid); //low-mode part for subset of evecs with index < nLowMode
  lsol_defl = Grid::zero;
  lsol_defl.checkerboard = Grid::Odd;
  
  GridFermionField sol_o(FrbGrid); //CG solution
  sol_o = Grid::zero;

  if(Nev < nLowMode)
    ERR.General("","Grid_CGNE_M_High","Number of low eigen modes to do deflation is smaller than number of low modes to be substracted!\n");

  if(Nev > 0){
    if (!UniqueID()) printf("Grid_CGNE_M_High: deflating with %d evecs\n",Nev);

    for(int n = 0; n < Nev; n++){
      double eval = evecs.getEvec(tmp_cb1,n);
      Grid::ComplexD cn = innerProduct(tmp_cb1, src_o);	
      axpy(lsol_full, cn / eval, tmp_cb1, lsol_full);

      if(n == nLowMode - 1) lsol_defl = lsol_full;
    }
    sol_o = lsol_full; //sol_o = lsol   Set guess equal to low mode projection 
  }

  f = norm2(src_o);
  if (!UniqueID()) printf("Grid_CGNE_M_high: CGNE_prec_MdagM src norm %le\n",f);
  f = norm2(sol_o);
  if (!UniqueID()) printf("Grid_CGNE_M_high: CGNE_prec_MdagM guess norm %le\n",f);

  //MdagM inverse controlled by evec interface
#ifndef MEMTEST_MODE
  evecs.CGNE_MdagM(linop, sol_o, src_o, cg_controls);
#endif
  
  f = norm2(sol_o);
  if (!UniqueID()) printf("Grid_CGNE_M_high: CGNE_prec_MdagM sol norm %le\n",f);


  //Pull low-mode part out of solution
  axpy(sol_o, -1.0, lsol_defl, sol_o);

  f = norm2(sol_o);
  if (!UniqueID()) printf("Grid_CGNE_M_high: sol norm after subtracting low-mode part %le\n",f);

  assert(sol_o.checkerboard == Grid::Odd);
  setCheckerboard(solution, sol_o);
  
  // sol_e = M_ee^-1 * ( src_e - Meo sol_o )...
  pickCheckerboard(Grid::Even,tmp_cb1,source);  //tmp_cb1 = src_e
  
  Ddwf.Meooe(sol_o,tmp_cb2); //tmp_cb2 = Meo sol_o
  assert(tmp_cb2.checkerboard == Grid::Even);

  axpy(tmp_cb1, -1.0, tmp_cb2, tmp_cb1); //tmp_cb1 = (-Meo sol_o + src_e)   (tmp_cb2 free)
  
  Ddwf.MooeeInv(tmp_cb1,tmp_cb2);  //tmp_cb2 = Mee^-1(-Meo sol_o + src_e)   (tmp_cb1 free)

  f = norm2(tmp_cb2);
  if (!UniqueID()) printf("Grid_CGNE_M_high: even checkerboard of sol %le\n",f);

  assert(tmp_cb2.checkerboard == Grid::Even);
  setCheckerboard(solution, tmp_cb2);

  f = norm2(solution);
  if (!UniqueID()) printf("Grid_CGNE_M_high: unprec sol norm is %le\n",f);
}






//Version of the above that acts on multiple src/solution pairs
template<typename GridPolicies>
inline void Grid_CGNE_M_high_multi(std::vector<typename GridPolicies::GridFermionField> &solutions, const std::vector<typename GridPolicies::GridFermionField> &sources, const CGcontrols &cg_controls,
			     EvecInterface<GridPolicies> &evecs, int nLowMode, 
			     typename GridPolicies::FgridFclass &latg, typename GridPolicies::GridDirac &Ddwf, Grid::GridCartesian *FGrid, Grid::GridRedBlackCartesian *FrbGrid){
  typedef typename GridPolicies::GridFermionField GridFermionField;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDirac GridDirac;
  
  Grid::SchurDiagMooeeOperator<GridDirac, GridFermionField> linop(Ddwf);

  GridFermionField tmp_cb1(FrbGrid);
  GridFermionField tmp_cb2(FrbGrid);
  GridFermionField tmp_cb3(FrbGrid);
  GridFermionField tmp_cb4(FrbGrid);

  GridFermionField tmp_full(FGrid);
  GridFermionField lsol_full(FrbGrid); //full low-mode part (all evecs)
  
  const int Nev = evecs.nEvecs();
  if(Nev < nLowMode)
    ERR.General("","Grid_CGNE_M_high_multi","Number of low eigen modes to do deflation is smaller than number of low modes to be substracted!\n");
  
  assert(sources.size() == solutions.size());

  const int nsolve = sources.size();

  //Fields that are passed into multi-solve CG
  std::vector<GridFermionField> src_o(nsolve,GridFermionField(FrbGrid));
  std::vector<GridFermionField> sol_o(nsolve,GridFermionField(FrbGrid)); //CG solution

  //Low-mode part for subset of evecs with index < nLowMode
  std::vector<GridFermionField> lsol_defl(nsolve, GridFermionField(FrbGrid)); 
  
  for(int s=0;s<nsolve;s++){
    double f = norm2(sources[s]);
    if (!UniqueID()) printf("Grid_CGNE_M_high_multi: Source %d norm is %le\n",s,f);
    f = norm2(solutions[s]);
    if (!UniqueID()) printf("Grid_CGNE_M_high_multi: Guess %d norm is %le\n",s,f);
    
    pickCheckerboard(Grid::Even,tmp_cb1,sources[s]);  //tmp_cb1 = source_e
    pickCheckerboard(Grid::Odd,tmp_cb2,sources[s]);   //tmp_cb2 = source_o

    // src_o = Mprecdag * (source_o - Moe MeeInv source_e)  , cf Daiqian's thesis page 60
    Ddwf.MooeeInv(tmp_cb1,tmp_cb3);
    Ddwf.Meooe     (tmp_cb3,tmp_cb4); //tmp_cb4 = Moe MeeInv source_e       (tmp_cb3 free)
    axpy    (tmp_cb3,-1.0,tmp_cb4, tmp_cb2); //tmp_cb3 = (source_o - Moe MeeInv source_e)    (tmp_cb4 free)
    linop.MpcDag(tmp_cb3, src_o[s]); //src_o = Mprecdag * (source_o - Moe MeeInv source_e)    (tmp_cb3, tmp_cb4 free)

    //Compute low-mode projection and CG guess (=low-mode part)
    lsol_full = Grid::zero;
    lsol_defl[s].checkerboard = Grid::Odd;

    if(Nev > 0){
      if (!UniqueID()) printf("Grid_CGNE_M_High: deflating src %d with %d evecs\n",s,Nev);

      for(int n = 0; n < Nev; n++){
	double eval = evecs.getEvec(tmp_cb1,n);
	Grid::ComplexD cn = innerProduct(tmp_cb1, src_o[s]);	
	axpy(lsol_full, cn / eval, tmp_cb1, lsol_full);
	
	if(n == nLowMode - 1) lsol_defl[s] = lsol_full; //store until after CG
      }
      sol_o[s] = lsol_full; //sol_o = lsol   Set guess equal to low mode projection 
    }

    f = norm2(src_o[s]);
    if (!UniqueID()) printf("Grid_CGNE_M_high: CGNE_prec_MdagM src %d norm %le\n",s,f);
    f = norm2(sol_o[s]);
    if (!UniqueID()) printf("Grid_CGNE_M_high: CGNE_prec_MdagM guess %d norm %le\n",s,f);
  }

  //Do the CG on multiple src/solution pairs 
#ifndef MEMTEST_MODE
  evecs.CGNE_MdagM_multi(linop, sol_o, src_o, cg_controls); //MdagM inverse controlled by evec interface
#endif

  //Remove the low-mode part from the odd-checkerboard solution and generate full deflated inverse
  for(int s=0;s<nsolve;s++){  
    double f = norm2(sol_o[s]);
    if (!UniqueID()) printf("Grid_CGNE_M_high: CGNE_prec_MdagM sol %d norm %le\n",s,f);

    //Pull low-mode part out of solution
    axpy(sol_o[s], -1.0, lsol_defl[s], sol_o[s]);
    
    f = norm2(sol_o[s]);
    if (!UniqueID()) printf("Grid_CGNE_M_high: sol %d norm after subtracting low-mode part %le\n",s,f);

    assert(sol_o[s].checkerboard == Grid::Odd);
    setCheckerboard(solutions[s], sol_o[s]);
  
    // sol_e = M_ee^-1 * ( src_e - Meo sol_o )...
    pickCheckerboard(Grid::Even,tmp_cb1,sources[s]);  //tmp_cb1 = src_e
    
    Ddwf.Meooe(sol_o[s],tmp_cb2); //tmp_cb2 = Meo sol_o
    assert(tmp_cb2.checkerboard == Grid::Even);

    axpy(tmp_cb1, -1.0, tmp_cb2, tmp_cb1); //tmp_cb1 = (-Meo sol_o + src_e)   (tmp_cb2 free)
  
    Ddwf.MooeeInv(tmp_cb1,tmp_cb2);  //tmp_cb2 = Mee^-1(-Meo sol_o + src_e)   (tmp_cb1 free)

    f = norm2(tmp_cb2);
    if (!UniqueID()) printf("Grid_CGNE_M_high: even checkerboard of sol %d is %le\n",s,f);

    assert(tmp_cb2.checkerboard == Grid::Even);
    setCheckerboard(solutions[s], tmp_cb2);

    f = norm2(solutions[s]);
    if (!UniqueID()) printf("Grid_CGNE_M_high: unprec sol %d norm is %le\n",s,f);
  }
}

CPS_END_NAMESPACE

#endif
#endif
