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

  //Invert 5D source -> 5D solution with optional deflation through the evec interface
  virtual void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const{ assert(0); }

  virtual ~A2Ainverter5dBase(){}
};

//Base class for 5D deflated multi-src inverters
template<typename _GridFermionFieldD>
class A2AdeflatedInverter5dBase: public A2Ainverter5dBase<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
private:
  const EvecInterface<GridFermionFieldD> &evecs;
  bool do_deflate; //control whether the initial deflation (to produce the guess) is performed or not
protected:
  void deflate(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const{
    if(do_deflate) evecs.deflatedGuessD(out, in);
  }
public:
  A2AdeflatedInverter5dBase(const EvecInterface<GridFermionFieldD> &evecs): evecs(evecs), do_deflate(true){}

  void enableInitialDeflation(bool val){ do_deflate = val; }

  bool doInitialDeflation() const{ return do_deflate; }
};

template<typename _GridFermionFieldD>
class A2Ainverter5dCG: public A2AdeflatedInverter5dBase<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
private:
  Grid::LinearOperatorBase<GridFermionFieldD> & LinOp;
  Grid::ConjugateGradient<GridFermionFieldD> cg;
public:
  
  A2Ainverter5dCG(Grid::LinearOperatorBase<GridFermionFieldD> &LinOp, const EvecInterface<GridFermionFieldD> &evecs, double tol, int maxits): LinOp(LinOp), cg(tol,maxits), A2AdeflatedInverter5dBase<_GridFermionFieldD>(evecs){}

  //Invert 5D source -> 5D solution
  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{
    assert(out.size() == in.size());
    this->deflate(out,in);
    Grid::ConjugateGradient<GridFermionFieldD> &cg_ = const_cast<Grid::ConjugateGradient<GridFermionFieldD> &>(cg); //grr
    Grid::LinearOperatorBase<GridFermionFieldD> &LinOp_ = const_cast<Grid::LinearOperatorBase<GridFermionFieldD> &>(LinOp); //grr

    for(int s=0;s<in.size();s++)
      cg_(LinOp_,in[s],out[s]);
  }
};

template<typename _GridFermionFieldD, typename _GridFermionFieldF>
class A2Ainverter5dReliableUpdateCG: public A2AdeflatedInverter5dBase<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridFermionFieldF GridFermionFieldF;
private:
  Grid::LinearOperatorBase<GridFermionFieldD> & LinOpD;
  Grid::LinearOperatorBase<GridFermionFieldF> & LinOpF;
  Grid::GridBase* singlePrecGrid;
  Grid::ConjugateGradientReliableUpdate<GridFermionFieldD,GridFermionFieldF> cg;
public:
  A2Ainverter5dReliableUpdateCG(Grid::LinearOperatorBase<GridFermionFieldD> &LinOpD, Grid::LinearOperatorBase<GridFermionFieldF> &LinOpF,
				const EvecInterface<GridFermionFieldD> &evecs, Grid::GridBase* singlePrecGrid,
				double tol, int maxits, double delta):
    LinOpD(LinOpD), LinOpF(LinOpF), singlePrecGrid(singlePrecGrid), cg(tol,maxits,delta,singlePrecGrid,LinOpF,LinOpD), A2AdeflatedInverter5dBase<_GridFermionFieldD>(evecs){}

  //Invert 5D source -> 5D solution
  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{
    assert(out.size() == in.size());
    this->deflate(out,in);
    Grid::ConjugateGradientReliableUpdate<GridFermionFieldD,GridFermionFieldF> &cg_ = const_cast<Grid::ConjugateGradientReliableUpdate<GridFermionFieldD,GridFermionFieldF> &>(cg); //grr
    for(int i=0;i<in.size();i++)
      cg_(in[i], out[i]);
  }
};


template<typename _GridFermionFieldD, typename _GridFermionFieldF>
class A2Ainverter5dMixedPrecCG: public A2AdeflatedInverter5dBase<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridFermionFieldF GridFermionFieldF;
private:
  Grid::LinearOperatorBase<GridFermionFieldD> & LinOpD;
  Grid::LinearOperatorBase<GridFermionFieldF> & LinOpF;
  Grid::GridBase* singlePrecGrid;
  Grid::MixedPrecisionConjugateGradient<GridFermionFieldD,GridFermionFieldF> mCG;
  EvecInterfaceSinglePrecGuesser<GridFermionFieldF,GridFermionFieldD> guesser;
public:
  A2Ainverter5dMixedPrecCG(Grid::LinearOperatorBase<GridFermionFieldD> &LinOpD, Grid::LinearOperatorBase<GridFermionFieldF> &LinOpF,
			   const EvecInterface<GridFermionFieldD> &evecs, Grid::GridBase* singlePrecGrid,
			   double tol, int maxits, double inner_tol):
    LinOpD(LinOpD), LinOpF(LinOpF), singlePrecGrid(singlePrecGrid), 
    mCG(tol, maxits, 50, singlePrecGrid, LinOpF, LinOpD),
    guesser(evecs, singlePrecGrid), A2AdeflatedInverter5dBase<_GridFermionFieldD>(evecs){

    mCG.useGuesser(guesser);
    mCG.InnerTolerance = inner_tol;
  }

  //Invert 5D source -> 5D solution
  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{
    assert(out.size() == in.size());
    this->deflate(out,in);
    Grid::MixedPrecisionConjugateGradient<GridFermionFieldD,GridFermionFieldF> &mCG_ = const_cast<Grid::MixedPrecisionConjugateGradient<GridFermionFieldD,GridFermionFieldF> &>(mCG); //grr
    for(int i=0;i<in.size();i++)
      mCG_(in[i], out[i]);
  }
};



template<typename _GridFermionFieldD, typename _GridFermionFieldF>
class A2Ainverter5dReliableUpdateSplitCG: public A2AdeflatedInverter5dBase<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef _GridFermionFieldF GridFermionFieldF;
private:
  Grid::LinearOperatorBase<GridFermionFieldD> & LinOpD_subgrid;
  Grid::LinearOperatorBase<GridFermionFieldF> & LinOpF_subgrid;

  Grid::GridBase* doublePrecGrid_subgrid;
  Grid::GridBase* singlePrecGrid_subgrid;

  Grid::ConjugateGradientReliableUpdate<GridFermionFieldD,GridFermionFieldF> cg_subgrid;
  int Nsplit; //number of solves that can be performed in parallel
public:
  A2Ainverter5dReliableUpdateSplitCG(Grid::LinearOperatorBase<GridFermionFieldD> &LinOpD_subgrid, Grid::LinearOperatorBase<GridFermionFieldF> &LinOpF_subgrid,
				     const EvecInterface<GridFermionFieldD> &evecs_fullgrid, 
				     Grid::GridBase* doublePrecGrid_subgrid, Grid::GridBase* singlePrecGrid_subgrid, int Nsplit,
				     double tol, int maxits, double delta):
    LinOpD_subgrid(LinOpD_subgrid), LinOpF_subgrid(LinOpF_subgrid),
    doublePrecGrid_subgrid(doublePrecGrid_subgrid), singlePrecGrid_subgrid(singlePrecGrid_subgrid), Nsplit(Nsplit),
    cg_subgrid(tol,maxits,delta,singlePrecGrid_subgrid,LinOpF_subgrid,LinOpD_subgrid), A2AdeflatedInverter5dBase<_GridFermionFieldD>(evecs_fullgrid){}

  //Invert 5D source -> 5D solution
  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{
    assert(out.size() == in.size());
    assert(in.size() >= 1);
    this->deflate(out,in);

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

//This wrapper takes a 2f G-parity field and converts back and forth to an X-conjugate field
//It requires an X-conjugate solver
//NOTE: the deflation (if enabled) will be performed prior to converting the src/sol to an X-conj field
//so the inner solver should not do the deflation
template<typename _GridFermionFieldD>
class A2Ainverter5dXconjWrapper: public A2AdeflatedInverter5dBase<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef decltype( Grid::PeekIndex<GparityFlavourIndex>( *( (GridFermionFieldD*)nullptr), 0) ) GridXconjFermionFieldD;

private:
  const A2Ainverter5dBase<GridXconjFermionFieldD> &inner_inv;
  bool check_Xconj;
  
public:
  //if check_Xconj is enabled, the sources will be checked as being in X-conjugate form
  //NOTE: We do *not* by default check the input fields are X-conjugate! Please use wisely
  A2Ainverter5dXconjWrapper(const A2Ainverter5dBase<GridXconjFermionFieldD> &inner_inv, const EvecInterface<GridFermionFieldD> &evecs, bool check_Xconj = false): inner_inv(inner_inv), A2AdeflatedInverter5dBase<_GridFermionFieldD>(evecs), check_Xconj(check_Xconj){}

  const Grid::Gamma & Xmatrix() const{
    static Grid::Gamma C = Grid::Gamma(Grid::Gamma::Algebra::MinusGammaY) * Grid::Gamma(Grid::Gamma::Algebra::GammaT);
    static Grid::Gamma g5 = Grid::Gamma(Grid::Gamma::Algebra::Gamma5);
    static Grid::Gamma X = C*g5;
    return X;
  }

  //Invert 5D source -> 5D solution with optional deflation through the evec interface
  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{
    assert(in.size() == out.size());
    if(in.size() == 0) return;
    int N = in.size();
    Grid::GridBase* grid = in[0].Grid();

    this->deflate(out,in);
    
    std::vector<GridXconjFermionFieldD> outX(N,grid);
    std::vector<GridXconjFermionFieldD> inX(N,grid);
    for(int i=0;i<N;i++){
      outX[i] = Grid::PeekIndex<GparityFlavourIndex>(out[i],0);
      inX[i] = Grid::PeekIndex<GparityFlavourIndex>(in[i],0);
      
      if(check_Xconj){
	if(!XconjugateCheck(in[i],1e-8,true)) ERR.General("A2Ainverter5dXconjWrapper","invert5Dto5D","Source %d is not X-conjugate",i);
	if(!XconjugateCheck(out[i],1e-8,true)) ERR.General("A2Ainverter5dXconjWrapper","invert5Dto5D","Guess %d is not X-conjugate",i);
      }

    }
    inner_inv.invert5Dto5D(outX,inX);
    GridXconjFermionFieldD tmp(grid);
    for(int i=0;i<N;i++){
      Grid::PokeIndex<GparityFlavourIndex>(out[i], outX[i], 0);
      tmp = -(Xmatrix()*conjugate(outX[i]));
      Grid::PokeIndex<GparityFlavourIndex>(out[i], tmp, 1);
    }        
  }
};

CPS_END_NAMESPACE

#endif
