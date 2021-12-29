#pragma once
#include<config.h>

#if defined(USE_GRID) && defined(USE_GRID_A2A)
#include<Grid/Grid.h>

CPS_START_NAMESPACE

//Class containing the various constructions needed for A2A with Schur preconditioning
template<typename _FermionOperatorTypeD>
class A2ASchurOperatorImpl{
public:
  typedef _FermionOperatorTypeD FermionOperatorTypeD; //Double precision operator
  typedef typename _FermionOperatorTypeD::FermionField GridFermionFieldD;

protected:
  FermionOperatorTypeD & OpD;
public:
  A2ASchurOperatorImpl(FermionOperatorTypeD &OpD): OpD(OpD){}

  //Operation performed on the evec for computing Vl, differs depending on preconditioning scheme 
  virtual void SchurEvecMulVl(GridFermionFieldD &out, const GridFermionFieldD &in) const = 0;

  FermionOperatorTypeD & getOp() const{ return const_cast<FermionOperatorTypeD &>(OpD); }
  virtual Grid::LinearOperatorBase<GridFermionFieldD> & getLinOp() const = 0;
  virtual Grid::SchurOperatorBase<GridFermionFieldD> & getSchurOp() const = 0;

  //Return a solver with a dummy inner operator function to expose certain functionality
  virtual Grid::SchurRedBlackBase<GridFermionFieldD> & getSolver() const = 0;

  //Get a new Solver instance with the operator function provided
  virtual Grid::SchurRedBlackBase<GridFermionFieldD>* getNewSolver(Grid::OperatorFunction<GridFermionFieldD> &herm_solver) const = 0;

  virtual ~A2ASchurOperatorImpl(){}
};

template<typename _FermionOperatorTypeD>
class A2ASchurOriginalOperatorImpl: public A2ASchurOperatorImpl<_FermionOperatorTypeD>{
public:
  typedef _FermionOperatorTypeD FermionOperatorTypeD; //Double precision operator
  typedef typename _FermionOperatorTypeD::FermionField GridFermionFieldD;

private:
  Grid::SchurDiagMooeeOperator<FermionOperatorTypeD,GridFermionFieldD> linop;
  Grid::ConjugateGradient<GridFermionFieldD> CG_dummy; //not used but required for constructor
  Grid::SchurRedBlackDiagMooeeSolve<GridFermionFieldD> solver;
public:
  A2ASchurOriginalOperatorImpl(FermionOperatorTypeD &OpD): A2ASchurOperatorImpl<_FermionOperatorTypeD>(OpD), linop(OpD), CG_dummy(1e-8,10000), solver(CG_dummy){}

  Grid::LinearOperatorBase<GridFermionFieldD> & getLinOp() const override{ return const_cast<Grid::SchurDiagMooeeOperator<FermionOperatorTypeD,GridFermionFieldD>&>(linop); }
  Grid::SchurOperatorBase<GridFermionFieldD> & getSchurOp() const override{ return const_cast<Grid::SchurDiagMooeeOperator<FermionOperatorTypeD,GridFermionFieldD>&>(linop); }
  Grid::SchurRedBlackBase<GridFermionFieldD> & getSolver() const override{ return const_cast<Grid::SchurRedBlackDiagMooeeSolve<GridFermionFieldD> &>(solver); }

  //Get a new Solver instance with the operator function provided
  Grid::SchurRedBlackBase<GridFermionFieldD>* getNewSolver(Grid::OperatorFunction<GridFermionFieldD> &herm_solver) const override{ return new Grid::SchurRedBlackDiagMooeeSolve<GridFermionFieldD>(herm_solver); }


  void SchurEvecMulVl(GridFermionFieldD &out, const GridFermionFieldD &in) const{
    out = in;
  }
};

template<typename _FermionOperatorTypeD>
class A2ASchurDiagTwoOperatorImpl: public A2ASchurOperatorImpl<_FermionOperatorTypeD>{
public:
  typedef _FermionOperatorTypeD FermionOperatorTypeD; //Double precision operator
  typedef typename _FermionOperatorTypeD::FermionField GridFermionFieldD;

private:
  Grid::SchurDiagTwoOperator<FermionOperatorTypeD,GridFermionFieldD> linop;
  Grid::ConjugateGradient<GridFermionFieldD> CG_dummy; //not used but required for constructor
  Grid::SchurRedBlackDiagTwoSolve<GridFermionFieldD> solver;
public:
  A2ASchurDiagTwoOperatorImpl(FermionOperatorTypeD &OpD): A2ASchurOperatorImpl<_FermionOperatorTypeD>(OpD), linop(OpD), CG_dummy(1e-8,10000), solver(CG_dummy){}

  Grid::LinearOperatorBase<GridFermionFieldD> & getLinOp() const override{ return const_cast<Grid::SchurDiagTwoOperator<FermionOperatorTypeD,GridFermionFieldD>&>(linop); }
  Grid::SchurOperatorBase<GridFermionFieldD> & getSchurOp() const override{ return const_cast<Grid::SchurDiagTwoOperator<FermionOperatorTypeD,GridFermionFieldD>&>(linop); }
  Grid::SchurRedBlackBase<GridFermionFieldD> & getSolver() const override{ return const_cast<Grid::SchurRedBlackDiagTwoSolve<GridFermionFieldD> &>(solver); }

  //Get a new Solver instance with the operator function provided
  Grid::SchurRedBlackBase<GridFermionFieldD>* getNewSolver(Grid::OperatorFunction<GridFermionFieldD> &herm_solver) const override{ return new Grid::SchurRedBlackDiagTwoSolve<GridFermionFieldD>(herm_solver); }

  void SchurEvecMulVl(GridFermionFieldD &out, const GridFermionFieldD &in) const{
    //Only difference between SchurOriginal and SchurDiagTwo for Vl is the multiplication of the evec by M_oo^{-1}
    this->OpD.MooeeInv(in,out);
  }
};


CPS_END_NAMESPACE
#endif
