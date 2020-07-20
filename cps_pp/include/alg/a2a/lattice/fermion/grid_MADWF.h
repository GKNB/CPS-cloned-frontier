#ifndef A2A_GRID_MADWF_H
#define A2A_GRID_MADWF_H

#include<config.h>

#if defined(USE_GRID) && defined(USE_GRID_A2A)
#include<util/lattice/fgrid.h>
#include "evec_interface.h"

CPS_START_NAMESPACE

struct ZmobParams{
  double b_plus_c_inner;
  int Ls_inner;
  double b_plus_c_outer;
  int Ls_outer;
  double lambda_max;
  bool complex_coeffs; //use Zmobius or regular Mobius
  
  ZmobParams(double b_plus_c_inner, int Ls_inner, double b_plus_c_outer, int Ls_outer, double lambda_max, bool complex_coeffs): b_plus_c_inner(b_plus_c_inner), Ls_inner(Ls_inner),
																b_plus_c_outer(b_plus_c_outer), Ls_outer(Ls_outer), 
																lambda_max(lambda_max), complex_coeffs(complex_coeffs){}
  
  bool operator<(const ZmobParams &r) const{
    if(b_plus_c_inner == r.b_plus_c_inner){
      if(Ls_inner == r.Ls_inner){
	if(b_plus_c_outer == r.b_plus_c_outer){
	  if(Ls_outer == r.Ls_outer){
	    if(lambda_max == r.lambda_max){
	      return complex_coeffs < r.complex_coeffs;
	    }else return lambda_max < r.lambda_max;
	  }else return Ls_outer < r.Ls_outer;
	}else return b_plus_c_outer < r.b_plus_c_outer;
      }else return Ls_inner<r.Ls_inner;
    }else return b_plus_c_inner < r.b_plus_c_inner;
  }
};

//Compute the (Z)Mobius gamma vector with caching. Here complex_coeffs == true implies ZMobius, false implies regular Mobius
std::vector<Grid::ComplexD> computeZmobiusGammaWithCache(double b_plus_c_inner, int Ls_inner, double b_plus_c_outer, int Ls_outer, double lambda_max, bool complex_coeffs){
  ZmobParams pstruct(b_plus_c_inner, Ls_inner, b_plus_c_outer, Ls_outer, lambda_max, complex_coeffs);
  static std::map<ZmobParams, std::vector<Grid::ComplexD> > cache;
  auto it = cache.find(pstruct);
  if(it == cache.end()){    
    std::vector<Grid::ComplexD> gamma_inner;
    
    std::cout << "MADWF Compute parameters with inner Ls = " << Ls_inner << std::endl;
    if(complex_coeffs){
      Grid::Approx::computeZmobiusGamma(gamma_inner, b_plus_c_inner, Ls_inner, b_plus_c_outer, Ls_outer, lambda_max);
    }else{
      Grid::Approx::zolotarev_data *zdata = Grid::Approx::higham(1.0,Ls_inner);
      gamma_inner.resize(Ls_inner);
      for(int s=0;s<Ls_inner;s++) gamma_inner[s] = zdata->gamma[s];
      Grid::Approx::zolotarev_free(zdata);
    }
    std::cout << "gamma:\n";
    for(int s=0;s<Ls_inner;s++) std::cout << s << " " << gamma_inner[s] << std::endl;
    
    cache[pstruct] = gamma_inner;
    return gamma_inner;
  }else{
    std::cout << "gamma (from cache):\n";
    for(int s=0;s<Ls_inner;s++) std::cout << s << " " << it->second[s] << std::endl;
    return it->second;
  }
}

//Get the (Z)Mobius parameters using the parameters in cg_controls, either through direct computation or from the struct directly
std::vector<Grid::ComplexD> getZMobiusGamma(const double b_plus_c_outer, const int Ls_outer,
					    const MADWFparams &madwf_p){
  const ZMobiusParams &zmp = madwf_p.ZMobius_params;

  std::vector<Grid::ComplexD> gamma_inner;

  //Get the parameters from the input struct
  if(zmp.gamma_src == A2A_ZMobiusGammaSourceInput){
    assert(zmp.gamma_real.gamma_real_len == madwf_p.Ls_inner);
    assert(zmp.gamma_imag.gamma_imag_len == madwf_p.Ls_inner);

    gamma_inner.resize(madwf_p.Ls_inner);
    for(int s=0;s<madwf_p.Ls_inner;s++)
      gamma_inner[s] = Grid::ComplexD( zmp.gamma_real.gamma_real_val[s], zmp.gamma_imag.gamma_imag_val[s] );
  }else{
    //Compute the parameters directly
    gamma_inner = computeZmobiusGammaWithCache(madwf_p.b_plus_c_inner, 
					       madwf_p.Ls_inner, 
					       b_plus_c_outer, Ls_outer,
					       zmp.compute_lambda_max, madwf_p.use_ZMobius);
  }
  return gamma_inner;
}

  



template<typename FermionFieldType>
struct CGincreaseTol : public Grid::MADWFinnerIterCallbackBase{
  Grid::ConjugateGradient<FermionFieldType> &cg_inner;  
  Grid::RealD outer_resid;

  CGincreaseTol(Grid::ConjugateGradient<FermionFieldType> &cg_inner,
	       Grid::RealD outer_resid): cg_inner(cg_inner), outer_resid(outer_resid){}
  
  void operator()(const Grid::RealD current_resid){
    std::cout << "CGincreaseTol with current residual " << current_resid << " changing inner tolerance " << cg_inner.Tolerance << " -> ";
    while(cg_inner.Tolerance < current_resid) cg_inner.Tolerance *= 2;    
    //cg_inner.Tolerance = outer_resid/current_resid;
    std::cout << cg_inner.Tolerance << std::endl;
  }
};


//Perform MADWF inversion upon the source. Interior Dirac operator is zMobius but can of course use real coefficients; this enables both MADWF and zMADWF
//Interior zMobius Dirac operator is in lower precision
//Eigenvectors must be for the inner, zMobius Dirac operator and in same precision. All eigenvectors will be used only to generate a guess for the solver
//Source must be *4D* and solution will also be *4D*

//****This code does not perform the low-mode subtraction*****

//if revert_to_mixedCG MADWF will be replaced by a standard mixed CG if the inner Dirac op is the same as the outer (useful for testing)
template<typename GridPolicies, typename GuessTypeF>
inline void Grid_MADWF_mixedprec_invert(typename GridPolicies::GridFermionField &solution, const typename GridPolicies::GridFermionField &source, const CGcontrols &cg_controls,
					const typename GridPolicies::GridDirac::GaugeField & Umu, 
					typename GridPolicies::GridDirac &DOuter, typename GridPolicies::GridDiracFZMobiusInner &DZMobInner,
					GuessTypeF &Guess_inner, A2Apreconditioning precond,
					bool revert_to_mixedCG = true){
  const MADWFparams &madwf_p = cg_controls.madwf_params;
  using namespace Grid;

  typedef typename GridPolicies::GridFermionField GridFermionField;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDirac GridDirac;

  typedef typename GridPolicies::GridFermionFieldF GridFermionFieldF;
  
  //Fields are 4D!
  assert(source.Grid() == Umu.Grid());
  assert(solution.Grid() == Umu.Grid());

  //If the inner Dirac operator is identical to the outer Dirac operator (up to precision), use the regular (mixed-precision) restarted CG [useful for testing]
  bool inner_Dop_same = 
    madwf_p.b_plus_c_inner == DOuter._b + DOuter._c &&
    madwf_p.Ls_inner == GJP.Snodes()*GJP.SnodeSites() &&
    !madwf_p.use_ZMobius;
  
  bool use_MADWF = !inner_Dop_same || (inner_Dop_same && !revert_to_mixedCG);

  //Setup outer double-prec Pauli-Villars solve
  ConjugateGradient<GridFermionField> CG_outer(cg_controls.CG_tolerance, cg_controls.CG_max_iters);  
  typedef PauliVillarsSolverFourierAccel<GridFermionField, LatticeGaugeFieldD> PVtype;
  PVtype PV_outer(const_cast<typename GridPolicies::GridDirac::GaugeField &>(Umu), CG_outer);
  
  //Setup inner preconditioned solver
  ConjugateGradient<GridFermionFieldF> CG_inner(cg_controls.mixedCG_init_inner_tolerance, cg_controls.CG_max_iters, 0);
  
  //Setup update control
  CGincreaseTol<GridFermionFieldF> tol_control_inner(CG_inner, cg_controls.CG_tolerance);
  
  GridFermionField sol5(DOuter.FermionGrid());
  if(use_MADWF){
    std::cout << "Doing " << (madwf_p.use_ZMobius ? "Z" : "") << "MADWF solve with "
	      << " Ls_outer=" << GJP.Snodes()*GJP.SnodeSites() << " (b+c)_outer=" << DOuter._b + DOuter._c
	      << " Ls_inner=" << madwf_p.Ls_inner << " (b+c)_inner=" << madwf_p.b_plus_c_inner << std::endl;
    
    if(precond == SchurDiagTwo){
      typedef SchurRedBlackDiagTwoSolve<GridFermionFieldF> SchurSolverTypeF;
      SchurSolverTypeF SchurSolver_inner(CG_inner);
       
      //Do solve
      MADWF<typename GridPolicies::GridDirac, 
	    typename GridPolicies::GridDiracFZMobiusInner,
	    PVtype, SchurSolverTypeF, GuessTypeF > madwf(DOuter, DZMobInner, PV_outer, SchurSolver_inner, Guess_inner, cg_controls.CG_tolerance, 100, &tol_control_inner);
    
      madwf(source, sol5);
    }else if(precond == SchurOriginal){
      typedef SchurRedBlackDiagMooeeSolve<GridFermionFieldF> SchurSolverTypeF;
      SchurSolverTypeF SchurSolver_inner(CG_inner);
       
      //Do solve
      MADWF<typename GridPolicies::GridDirac, 
	    typename GridPolicies::GridDiracFZMobiusInner,
	    PVtype, SchurSolverTypeF, GuessTypeF > madwf(DOuter, DZMobInner, PV_outer, SchurSolver_inner, Guess_inner, cg_controls.CG_tolerance, 100, &tol_control_inner);
    
      madwf(source, sol5);
    }else assert(0);
    
  }else{
    std::cout << "Doing mixed-precision CG solve rather than MADWF because inner Dirac op is the same as the outer" << std::endl;

    GridFermionField src_5d(DOuter.FermionGrid());
    DOuter.ImportPhysicalFermionSource(source, src_5d);

    SchurRedBlackBase<GridFermionField> *SchurSolver_outer;
    SchurOperatorBase<GridFermionField> *linop_outer;
    SchurOperatorBase<GridFermionFieldF> *linop_inner;

    if(precond == SchurOriginal){
      SchurSolver_outer = new SchurRedBlackDiagMooeeSolve<GridFermionField>(CG_outer);
      linop_outer = new SchurDiagMooeeOperator<typename GridPolicies::GridDirac, GridFermionField>(DOuter);
      linop_inner = new SchurDiagMooeeOperator<typename GridPolicies::GridDiracFZMobiusInner, GridFermionFieldF>(DZMobInner);      
    }else if(precond == SchurDiagTwo){
      SchurSolver_outer = new SchurRedBlackDiagTwoSolve<GridFermionField>(CG_outer);
      linop_outer = new SchurDiagTwoOperator<typename GridPolicies::GridDirac, GridFermionField>(DOuter);
      linop_inner = new SchurDiagTwoOperator<typename GridPolicies::GridDiracFZMobiusInner, GridFermionFieldF>(DZMobInner);
    }else assert(0);
    

    GridFermionField src_5d_o(DOuter.FermionRedBlackGrid()), src_5d_e(DOuter.FermionRedBlackGrid());
    SchurSolver_outer->RedBlackSource(DOuter, src_5d, src_5d_e, src_5d_o);

    MixedPrecisionConjugateGradient<GridFermionField,  GridFermionFieldF> mcg(cg_controls.CG_tolerance, cg_controls.CG_max_iters, cg_controls.CG_max_iters,
									      DZMobInner.FermionRedBlackGrid(), *linop_inner, *linop_outer);
    
    //Get a guess
    GridFermionFieldF src_5d_o_f(DZMobInner.FermionRedBlackGrid());
    precisionChange(src_5d_o_f, src_5d_o);
    GridFermionFieldF sol_guess_o_f(DZMobInner.FermionRedBlackGrid());
    Guess_inner(src_5d_o_f, sol_guess_o_f);
    GridFermionField sol_5d_o(DOuter.FermionRedBlackGrid()); 
    precisionChange(sol_5d_o, sol_guess_o_f);

    //Run mixed CG
    mcg.useGuesser(Guess_inner);
    mcg(src_5d_o, sol_5d_o);
    
    SchurSolver_outer->RedBlackSolution(DOuter, sol_5d_o, src_5d_e, sol5);

    //TESTING
    GridFermionField sol_5d_e(DOuter.FermionRedBlackGrid()); 
    pickCheckerboard(Even,sol_5d_e,sol5);
    std::cout << "Solution norms odd:" << norm2(sol_5d_o) << " even:" << norm2(sol_5d_e) << std::endl;
    //TESTING

    delete SchurSolver_outer;
    delete linop_outer;
    delete linop_inner;
  }
  
  DOuter.ExportPhysicalFermionSolution(sol5, solution);
}

template<typename GridPolicies>
class EvecInterface;

//Compute the low-mode contribution as a 4D field for use in MADWF version
//uses first nl eigenvectors
template<typename GridPolicies>
void computeMADWF_lowmode_contrib_4D(typename GridPolicies::GridFermionField &lowmode_contrib, 
				     const typename GridPolicies::GridFermionField &source, 
				     const int nl, const EvecInterface<GridPolicies> &evecs,
				     typename GridPolicies::GridDiracZMobius &Dop_inner,
				     A2Apreconditioning precond){
  using namespace Grid;
  typedef typename GridPolicies::GridFermionField GridFermionField;

  lowmode_contrib = GridFermionField(source.Grid()); //UGrid
    
  Grid::GridBase *FGrid_inner = Dop_inner.FermionGrid();
  Grid::GridBase *FrbGrid_inner = Dop_inner.FermionRedBlackGrid();

  GridFermionField source_5d(FGrid_inner);
  Dop_inner.ImportPhysicalFermionSource(source, source_5d); //applies D- appropriately

  //We want to accumulate in double
  GridFermionField lowmode_contrib_5d_o(FrbGrid_inner);
  lowmode_contrib_5d_o = Grid::Zero();
  lowmode_contrib_5d_o.Checkerboard() = Grid::Odd;

  //Get the checkerboarded source
  ConjugateGradient<GridFermionField> CG_outer(1e-8,10000); //not actually used, we just need some other functionality of SchurRedBlackDiagTwoSolve
  SchurRedBlackBase<GridFermionField> *schur;
  if(precond == SchurOriginal) schur = new SchurRedBlackDiagMooeeSolve<GridFermionField>(CG_outer);
  else if(precond == SchurDiagTwo) schur = new SchurRedBlackDiagTwoSolve<GridFermionField>(CG_outer);
  else assert(0);

  GridFermionField source_5d_e(FrbGrid_inner);
  GridFermionField source_5d_o(FrbGrid_inner);
  schur->RedBlackSource(Dop_inner, source_5d, source_5d_e, source_5d_o);

  GridFermionField evec_tmp(FrbGrid_inner);

  for(int i=0;i<nl;i++){
    Float eval = evecs.getEvec(evec_tmp, i);
    Grid::ComplexD dot = innerProduct(evec_tmp, source_5d_o);
    dot = dot / eval;
      
    evec_tmp = evec_tmp * dot;
      
    lowmode_contrib_5d_o = lowmode_contrib_5d_o + evec_tmp;
  }
    
  //Because the low-mode part is computed only through the odd-checkerboard eigenvectors, the even component of the solution comes only through -Mee^-1 Meo applied to the odd solution.
  //i.e. there is no M_ee^-1 src_e in the even part of the solution
  //We can spoof that behavior by zeroing the even part of the source in RedBlackSolution such that M_ee^-1 src_e = 0
  source_5d_e = Grid::Zero();

  GridFermionField lowmode_contrib_5d(FGrid_inner);
  schur->RedBlackSolution(Dop_inner, lowmode_contrib_5d_o, source_5d_e, lowmode_contrib_5d);

  Dop_inner.ExportPhysicalFermionSolution(lowmode_contrib_5d, lowmode_contrib);

  delete schur;
}


CPS_END_NAMESPACE

#endif

#endif
