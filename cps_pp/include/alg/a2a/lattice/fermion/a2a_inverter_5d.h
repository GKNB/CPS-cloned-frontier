#pragma once
#include<config.h>

#if defined(USE_GRID) && defined(USE_GRID_A2A)
#include<Grid/Grid.h>
#include "evec_interface.h"

CPS_START_NAMESPACE

//Base class for 5D multi-src inverters
template<typename _GridFermionFieldD>
class A2Ainverter5dBase{
public:
  typedef _GridFermionFieldD GridFermionFieldD;

  //Invert 5D source -> 5D solution
  virtual void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const{ assert(0); }

  virtual ~A2Ainverter5dBase(){}
};

template<typename _GridFermionFieldD>
class A2Ainverter5dCG: public A2Ainverter5dBase<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
private:
  Grid::LinearOperatorBase<GridFermionFieldD> & LinOp;
  const EvecInterface<GridFermionFieldD> &evecs;
  Grid::ConjugateGradient<GridFermionFieldD> cg;
public:
  
  A2Ainverter5dCG(Grid::LinearOperatorBase<GridFermionFieldD> &LinOp, const EvecInterface<GridFermionFieldD> &evecs, double tol, int maxits): LinOp(LinOp), evecs(evecs), cg(tol,maxits){}

  //Invert 5D source -> 5D solution
  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{
    assert(out.size() == in.size());
    evecs.deflatedGuessD(out, in);
    Grid::ConjugateGradient<GridFermionFieldD> &cg_ = const_cast<Grid::ConjugateGradient<GridFermionFieldD> &>(cg); //grr
    Grid::LinearOperatorBase<GridFermionFieldD> &LinOp_ = const_cast<Grid::LinearOperatorBase<GridFermionFieldD> &>(LinOp); //grr

    for(int s=0;s<in.size();s++)
      cg_(LinOp_,in[s],out[s]);
  }
};

template<typename _GridFermionFieldD, typename _GridFermionFieldF>
class A2Ainverter5dReliableUpdateCG: public A2Ainverter5dBase<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridFermionFieldF GridFermionFieldF;
private:
  Grid::LinearOperatorBase<GridFermionFieldD> & LinOpD;
  Grid::LinearOperatorBase<GridFermionFieldF> & LinOpF;
  const EvecInterface<GridFermionFieldD> &evecs;
  Grid::GridBase* singlePrecGrid;
  Grid::ConjugateGradientReliableUpdate<GridFermionFieldD,GridFermionFieldF> cg;
public:
  A2Ainverter5dReliableUpdateCG(Grid::LinearOperatorBase<GridFermionFieldD> &LinOpD, Grid::LinearOperatorBase<GridFermionFieldF> &LinOpF,
				const EvecInterface<GridFermionFieldD> &evecs, Grid::GridBase* singlePrecGrid,
				double tol, int maxits, double delta):
    LinOpD(LinOpD), LinOpF(LinOpF), evecs(evecs), singlePrecGrid(singlePrecGrid), cg(tol,maxits,delta,singlePrecGrid,LinOpF,LinOpD){}

  //Invert 5D source -> 5D solution
  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{
    assert(out.size() == in.size());
    evecs.deflatedGuessD(out, in);
    Grid::ConjugateGradientReliableUpdate<GridFermionFieldD,GridFermionFieldF> &cg_ = const_cast<Grid::ConjugateGradientReliableUpdate<GridFermionFieldD,GridFermionFieldF> &>(cg); //grr
    for(int i=0;i<in.size();i++)
      cg_(in[i], out[i]);
  }
};


template<typename _GridFermionFieldD, typename _GridFermionFieldF>
class A2Ainverter5dMixedPrecCG: public A2Ainverter5dBase<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridFermionFieldF GridFermionFieldF;
private:
  Grid::LinearOperatorBase<GridFermionFieldD> & LinOpD;
  Grid::LinearOperatorBase<GridFermionFieldF> & LinOpF;
  const EvecInterface<GridFermionFieldD> &evecs;
  Grid::GridBase* singlePrecGrid;
  Grid::MixedPrecisionConjugateGradient<GridFermionFieldD,GridFermionFieldF> mCG;
  EvecInterfaceSinglePrecGuesser<GridFermionFieldF,GridFermionFieldD> guesser;
public:
  A2Ainverter5dMixedPrecCG(Grid::LinearOperatorBase<GridFermionFieldD> &LinOpD, Grid::LinearOperatorBase<GridFermionFieldF> &LinOpF,
			   const EvecInterface<GridFermionFieldD> &evecs, Grid::GridBase* singlePrecGrid,
			   double tol, int maxits, double inner_tol):
    LinOpD(LinOpD), LinOpF(LinOpF), evecs(evecs), singlePrecGrid(singlePrecGrid), 
    mCG(tol, maxits, 50, singlePrecGrid, LinOpF, LinOpD),
    guesser(evecs){

    mCG.useGuesser(guesser);
    mCG.InnerTolerance = inner_tol;
  }

  //Invert 5D source -> 5D solution
  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{
    std::cout << "Tianle: DEBUG: inside A2Ainverter5dMixedPrecCG::invert5Dto5D, in.size = " << in.size() << std::endl;
    assert(out.size() == in.size());
    std::cout << "Tianle: DEBUG: Will call evecs.deflatedGuessD(out, in) inside A2Ainverter5dMixedPrecCG::invert5Dto5D" << std::endl;
    evecs.deflatedGuessD(out, in);
    std::cout << "Tianle: DEBUG: Will create Grid::MixedPrecisionConjugateGradient inst inside A2Ainverter5dMixedPrecCG::invert5Dto5D" << std::endl;
    Grid::MixedPrecisionConjugateGradient<GridFermionFieldD,GridFermionFieldF> &mCG_ = const_cast<Grid::MixedPrecisionConjugateGradient<GridFermionFieldD,GridFermionFieldF> &>(mCG); //grr
    for(int i=0;i<in.size();i++)
    {
      std::cout << "Tianle: DEBUG: Will call mCG_(in[i], out[i]) inside A2Ainverter5dMixedPrecCG::invert5Dto5D with i = " << i << std::endl;
      mCG_(in[i], out[i]);
    }
    std::cout << "Tianle: DEBUG: A2Ainverter5dMixedPrecCG::invert5Dto5D finish" << std::endl;
  }
};



template<typename _GridFermionFieldD, typename _GridFermionFieldF>
class A2Ainverter5dReliableUpdateSplitCG: public A2Ainverter5dBase<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridFermionFieldF GridFermionFieldF;
private:
  Grid::LinearOperatorBase<GridFermionFieldD> & LinOpD_subgrid;
  Grid::LinearOperatorBase<GridFermionFieldF> & LinOpF_subgrid;
  const EvecInterface<GridFermionFieldD> &evecs_fullgrid;

  Grid::GridBase* doublePrecGrid_subgrid;
  Grid::GridBase* singlePrecGrid_subgrid;

  Grid::ConjugateGradientReliableUpdate<GridFermionFieldD,GridFermionFieldF> cg_subgrid;
  int Nsplit; //number of solves that can be performed in parallel
public:
  A2Ainverter5dReliableUpdateSplitCG(Grid::LinearOperatorBase<GridFermionFieldD> &LinOpD_subgrid, Grid::LinearOperatorBase<GridFermionFieldF> &LinOpF_subgrid,
				     const EvecInterface<GridFermionFieldD> &evecs_fullgrid, 
				     Grid::GridBase* doublePrecGrid_subgrid, Grid::GridBase* singlePrecGrid_subgrid, int Nsplit,
				     double tol, int maxits, double delta):
    LinOpD_subgrid(LinOpD_subgrid), LinOpF_subgrid(LinOpF_subgrid), evecs_fullgrid(evecs_fullgrid),
    doublePrecGrid_subgrid(doublePrecGrid_subgrid), singlePrecGrid_subgrid(singlePrecGrid_subgrid), Nsplit(Nsplit),
    cg_subgrid(tol,maxits,delta,singlePrecGrid_subgrid,LinOpF_subgrid,LinOpD_subgrid){}

  //Invert 5D source -> 5D solution
  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{
    assert(out.size() == in.size());
    assert(in.size() >= 1);
    evecs_fullgrid.deflatedGuessD(out, in);

    std::cout << Grid::GridLogMessage << "Doing split Grid solve with " << in.size() << " sources and " << Nsplit << " split grids" << std::endl;

    Grid::ConjugateGradientReliableUpdate<GridFermionFieldD,GridFermionFieldF> &cg_subgrid_ = const_cast<Grid::ConjugateGradientReliableUpdate<GridFermionFieldD,GridFermionFieldF> &>(cg_subgrid); //grr
    
    Grid::GridBase* doublePrecGrid_fullgrid = in[0].Grid();

    int Nsrc = in.size();

    std::vector<GridFermionFieldD> sol_fullgrid(Nsplit,doublePrecGrid_fullgrid);
    std::vector<GridFermionFieldD> src_fullgrid(Nsplit,doublePrecGrid_fullgrid);
    GridFermionFieldD sol_subgrid(doublePrecGrid_subgrid);
    GridFermionFieldD src_subgrid(doublePrecGrid_subgrid);

    int Nsplit_solves = (Nsrc + Nsplit - 1) / Nsplit; //number of split solves, round up
    std::cout << Grid::GridLogMessage << "Requires " << Nsplit_solves << " concurrent solves" << std::endl;
    for(int solve=0;solve<Nsplit_solves;solve++){
      int StartSrc = solve*Nsplit;
      int Nactual = std::min(Nsplit, Nsrc - StartSrc);
      std::cout << "Solving sources " << StartSrc << "-" << StartSrc+Nactual-1 << std::endl;

      for(int i=0;i<Nactual;i++){
	src_fullgrid[i] = in[StartSrc + i];
	sol_fullgrid[i] = out[StartSrc + i];
      }
      for(int i=Nactual;i<Nsplit;i++){
	src_fullgrid[i] = in[StartSrc]; //do dummy solves
	sol_fullgrid[i] = out[StartSrc];
      }

      Grid_split(src_fullgrid, src_subgrid);
      Grid_split(sol_fullgrid, sol_subgrid);
      
      cg_subgrid_(src_subgrid, sol_subgrid);

      Grid_unsplit(sol_fullgrid, sol_subgrid);
      for(int i=0;i<Nactual;i++){
	out[StartSrc + i] = sol_fullgrid[i];
      }
    }
  }
};

CPS_END_NAMESPACE

#endif
