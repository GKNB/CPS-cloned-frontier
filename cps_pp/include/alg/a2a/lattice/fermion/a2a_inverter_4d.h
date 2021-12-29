#pragma once
#include<config.h>

#if defined(USE_GRID) && defined(USE_GRID_A2A)
#include<Grid/Grid.h>
#include "grid_MADWF.h"
#include "a2a_inverter_5d.h"
#include "schur_operator.h"

CPS_START_NAMESPACE

//Base class for 4D->4D inverters
template<typename _FermionOperatorTypeD>
class A2Ainverter4dBase{
public:
  typedef _FermionOperatorTypeD FermionOperatorTypeD; //Double precision operator
  typedef typename _FermionOperatorTypeD::FermionField GridFermionFieldD;

  virtual void invert4Dto4D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const = 0;

  virtual ~A2Ainverter4dBase(){}
};

//4D->4D inversion with internal Schur preconditioning
template<typename _FermionOperatorTypeD>
class A2Ainverter4dSchurPreconditioned: public A2Ainverter4dBase<_FermionOperatorTypeD>{
public:
  typedef _FermionOperatorTypeD FermionOperatorTypeD; //Double precision operator
  typedef typename _FermionOperatorTypeD::FermionField GridFermionFieldD;

protected:
  A2ASchurOperatorImpl<FermionOperatorTypeD> & OpD;
  const A2Ainverter5dBase<GridFermionFieldD> &inv5D;
public:
  A2Ainverter4dSchurPreconditioned(A2ASchurOperatorImpl<FermionOperatorTypeD> &OpD, const A2Ainverter5dBase<GridFermionFieldD> &inv5D): OpD(OpD), inv5D(inv5D){}

  void invert4Dto4D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const{
    Grid::SchurRedBlackBase<GridFermionFieldD> & solver = OpD.getSolver();
    FermionOperatorTypeD & fermop = OpD.getOp();
    int N=in.size();
    assert(out.size() == N);
    Grid::GridBase* FrbGridD = fermop.FermionRedBlackGrid();
    Grid::GridBase* FGridD = fermop.FermionGrid();

    GridFermionFieldD src_5d(FGridD), sol_5d(FGridD);
    std::vector<GridFermionFieldD> src_5d_o(N,FrbGridD), src_5d_e(N,FrbGridD);
    for(int s=0;s<N;s++){
      fermop.ImportPhysicalFermionSource(in[s], src_5d);
      solver.RedBlackSource(fermop, src_5d, src_5d_e[s], src_5d_o[s]);
    }
    std::vector<GridFermionFieldD> sol_5d_o(N,FrbGridD);
    for(int s=0;s<N;s++) sol_5d_o[s].Checkerboard() = Grid::Odd;

    inv5D.invert5Dto5D(sol_5d_o, src_5d_o);
      
    for(int s=0;s<N;s++){
      solver.RedBlackSolution(fermop, sol_5d_o[s], src_5d_e[s], sol_5d);
      fermop.ExportPhysicalFermionSolution(sol_5d, out[s]);
    }
  }
};


//4D->4D inversion using (Z)MADWF with internal Schur preconditioning
//Outer Dirac operator should be double precision, inner could be either double or single
template<typename _OuterFermionOperatorTypeD, typename _InnerFermionOperatorType, typename _InnerGridFermionFieldD, typename _InnerGridFermionFieldF>
class A2Ainverter4dSchurPreconditionedMADWF: public A2Ainverter4dBase<_OuterFermionOperatorTypeD>{
public:
  typedef _OuterFermionOperatorTypeD OuterFermionOperatorTypeD; //Double precision operator
  typedef typename _OuterFermionOperatorTypeD::FermionField OuterGridFermionFieldD;

  typedef _InnerFermionOperatorType InnerFermionOperatorType;
  typedef typename _InnerFermionOperatorType::FermionField InnerGridFermionField;
  
protected:
  //Outer operator (5 space)
  const OuterFermionOperatorTypeD &OpOuter;  

  //Inner operator (5' space)
  const A2ASchurOperatorImpl<InnerFermionOperatorType> & SchurOpInner;
  const EvecInterfaceMixedPrec<_InnerGridFermionFieldD,_InnerGridFermionFieldF> &evecsInner;

  //Other parameters
  const Grid::LatticeGaugeFieldD &Umu;

  double tol;
  double inner_tol;
  int max_iters;

public:
  A2Ainverter4dSchurPreconditionedMADWF(const OuterFermionOperatorTypeD &OpOuter, 
					const A2ASchurOperatorImpl<InnerFermionOperatorType> & SchurOpInner, const EvecInterfaceMixedPrec<_InnerGridFermionFieldD,_InnerGridFermionFieldF> &evecsInner,
					const Grid::LatticeGaugeFieldD &Umu,
					double tol, double inner_tol, int max_iters): OpOuter(OpOuter), SchurOpInner(SchurOpInner), evecsInner(evecsInner), Umu(Umu), tol(tol), inner_tol(inner_tol),
										      max_iters(max_iters){}
					

  void invert4Dto4D(std::vector<OuterGridFermionFieldD> &out, const std::vector<OuterGridFermionFieldD> &in) const{
    int N=in.size();
    assert(out.size() == N);

    //Outer operator
    Grid::ConjugateGradient<OuterGridFermionFieldD> CG_outer(tol, max_iters);
    typedef Grid::PauliVillarsSolverFourierAccel<OuterGridFermionFieldD, Grid::LatticeGaugeFieldD> PVtype;
    PVtype PV_outer(const_cast<Grid::LatticeGaugeFieldD &>(Umu), CG_outer);
    OuterFermionOperatorTypeD & OpOuter_ = const_cast<OuterFermionOperatorTypeD &>(OpOuter);

    //Inner operator
    InnerFermionOperatorType & OpInner_ = SchurOpInner.getOp();
    Grid::ConjugateGradient<InnerGridFermionField> CG_inner(inner_tol, max_iters, 0);
    Grid::SchurRedBlackBase<InnerGridFermionField> *SchurSolver_inner = SchurOpInner.getNewSolver(CG_inner);

    //Setup update control
    CGincreaseTol<InnerGridFermionField> tol_control_inner(CG_inner, tol);
       
    //Inner guesser
    typedef EvecInterfaceMixedPrecGuesser<InnerGridFermionField, _InnerGridFermionFieldD,_InnerGridFermionFieldF> GuessTypeF;
    GuessTypeF Guess_inner(evecsInner);

    //Put it all together
    Grid::MADWF<OuterFermionOperatorTypeD,
		InnerFermionOperatorType,
		PVtype, Grid::SchurRedBlackBase<InnerGridFermionField>, GuessTypeF > madwf(OpOuter_, OpInner_, PV_outer, *SchurSolver_inner, Guess_inner, tol, 100, &tol_control_inner);

    //Do solve
    OuterGridFermionFieldD sol5(OpOuter_.FermionGrid());
    for(int s=0;s<N;s++){
      madwf(in[s], sol5);
      OpOuter_.ExportPhysicalFermionSolution(sol5, out[s]);
    }
    
    delete SchurSolver_inner;
  }
};


CPS_END_NAMESPACE

#endif
