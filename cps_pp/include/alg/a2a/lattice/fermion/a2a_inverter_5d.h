#pragma once
#include<config.h>

#if defined(USE_GRID) && defined(USE_GRID_A2A)
#include<Grid/Grid.h>
#include "evec_interface.h"
#include "schur_operator.h"

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

//A wrapper implementation that performs an initial deflation given an EvecInterface
template<typename _GridFermionFieldD>
class A2AdeflatedInverter5dWrapper: public A2Ainverter5dBase<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
private:
  const A2Ainverter5dBase<GridFermionFieldD> &inverter;
  const EvecInterface<GridFermionFieldD> &evecs;

public:
  A2AdeflatedInverter5dWrapper(const EvecInterface<GridFermionFieldD> &evecs, const A2Ainverter5dBase<GridFermionFieldD> &inverter): evecs(evecs), inverter(inverter){}

  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{ 
    LOGA2A << "A2AdeflatedInverter5dWrapper deflating " << in.size() << " fields" << std::endl;
    evecs.deflatedGuessD(out, in); //note this discards the input value of 'out'
    inverter.invert5Dto5D(out,in);
  }
  
};

template<typename _GridFermionFieldD>
class A2Ainverter5dCG: public A2Ainverter5dBase<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
private:
  Grid::LinearOperatorBase<GridFermionFieldD> & LinOp;
  Grid::ConjugateGradient<GridFermionFieldD> cg;
public:
  
  A2Ainverter5dCG(Grid::LinearOperatorBase<GridFermionFieldD> &LinOp, double tol, int maxits): LinOp(LinOp), cg(tol,maxits){}

  //Invert 5D source -> 5D solution
  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{
    assert(out.size() == in.size());
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
  Grid::GridBase* singlePrecGrid;
  Grid::ConjugateGradientReliableUpdate<GridFermionFieldD,GridFermionFieldF> cg;
public:
  A2Ainverter5dReliableUpdateCG(Grid::LinearOperatorBase<GridFermionFieldD> &LinOpD, Grid::LinearOperatorBase<GridFermionFieldF> &LinOpF,
				Grid::GridBase* singlePrecGrid,
				double tol, int maxits, double delta):
    LinOpD(LinOpD), LinOpF(LinOpF), singlePrecGrid(singlePrecGrid), cg(tol,maxits,delta,singlePrecGrid,LinOpF,LinOpD){}

  //Invert 5D source -> 5D solution
  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{
    assert(out.size() == in.size());
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
  Grid::GridBase* singlePrecGrid;
  Grid::MixedPrecisionConjugateGradient<GridFermionFieldD,GridFermionFieldF> mCG;
  EvecInterfaceSinglePrecGuesser<GridFermionFieldF,GridFermionFieldD> *guesser;
public:
  //Version with no internal deflation upon restart
  A2Ainverter5dMixedPrecCG(Grid::LinearOperatorBase<GridFermionFieldD> &LinOpD, Grid::LinearOperatorBase<GridFermionFieldF> &LinOpF,
			   Grid::GridBase* singlePrecGrid,
			   double tol, int maxits, double inner_tol):
    LinOpD(LinOpD), LinOpF(LinOpF), singlePrecGrid(singlePrecGrid), 
    mCG(tol, maxits, 50, singlePrecGrid, LinOpF, LinOpD),  guesser(nullptr){
    mCG.InnerTolerance = inner_tol;
  }
  //This version uses the evecs to deflate upon restart
  A2Ainverter5dMixedPrecCG(Grid::LinearOperatorBase<GridFermionFieldD> &LinOpD, Grid::LinearOperatorBase<GridFermionFieldF> &LinOpF,
			   const EvecInterface<GridFermionFieldD> &evecs, Grid::GridBase* singlePrecGrid,
			   double tol, int maxits, double inner_tol):
    A2Ainverter5dMixedPrecCG(LinOpD,LinOpF,singlePrecGrid,tol,maxits,inner_tol)
  {
    guesser = new EvecInterfaceSinglePrecGuesser<GridFermionFieldF,GridFermionFieldD>(evecs, singlePrecGrid);
    mCG.useGuesser(*guesser);
  }

  //Invert 5D source -> 5D solution
  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{
    assert(out.size() == in.size());
    Grid::MixedPrecisionConjugateGradient<GridFermionFieldD,GridFermionFieldF> &mCG_ = const_cast<Grid::MixedPrecisionConjugateGradient<GridFermionFieldD,GridFermionFieldF> &>(mCG); //grr
    for(int i=0;i<in.size();i++)
    {
      mCG_(in[i], out[i]);
    }
  }

  ~A2Ainverter5dMixedPrecCG(){
    if(guesser) delete guesser;
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

  Grid::GridBase* doublePrecGrid_subgrid;
  Grid::GridBase* singlePrecGrid_subgrid;

  Grid::ConjugateGradientReliableUpdate<GridFermionFieldD,GridFermionFieldF> cg_subgrid;
  int Nsplit; //number of solves that can be performed in parallel
public:
  A2Ainverter5dReliableUpdateSplitCG(Grid::LinearOperatorBase<GridFermionFieldD> &LinOpD_subgrid, Grid::LinearOperatorBase<GridFermionFieldF> &LinOpF_subgrid,
				     Grid::GridBase* doublePrecGrid_subgrid, Grid::GridBase* singlePrecGrid_subgrid, int Nsplit,
				     double tol, int maxits, double delta):
    LinOpD_subgrid(LinOpD_subgrid), LinOpF_subgrid(LinOpF_subgrid),
    doublePrecGrid_subgrid(doublePrecGrid_subgrid), singlePrecGrid_subgrid(singlePrecGrid_subgrid), Nsplit(Nsplit),
    cg_subgrid(tol,maxits,delta,singlePrecGrid_subgrid,LinOpF_subgrid,LinOpD_subgrid){}

  //Invert 5D source -> 5D solution
  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{
    assert(out.size() == in.size());
    assert(in.size() >= 1);
    LOGA2A << "Doing split Grid solve with " << in.size() << " sources and " << Nsplit << " split grids" << std::endl;

    Grid::ConjugateGradientReliableUpdate<GridFermionFieldD,GridFermionFieldF> &cg_subgrid_ = const_cast<Grid::ConjugateGradientReliableUpdate<GridFermionFieldD,GridFermionFieldF> &>(cg_subgrid); //grr
    
    Grid::GridBase* doublePrecGrid_fullgrid = in[0].Grid();

    int Nsrc = in.size();

    std::vector<GridFermionFieldD> sol_fullgrid(Nsplit,doublePrecGrid_fullgrid);
    std::vector<GridFermionFieldD> src_fullgrid(Nsplit,doublePrecGrid_fullgrid);
    GridFermionFieldD sol_subgrid(doublePrecGrid_subgrid);
    GridFermionFieldD src_subgrid(doublePrecGrid_subgrid);

    int Nsplit_solves = (Nsrc + Nsplit - 1) / Nsplit; //number of split solves, round up
    LOGA2A << "Requires " << Nsplit_solves << " concurrent solves" << std::endl;
    for(int solve=0;solve<Nsplit_solves;solve++){
      int StartSrc = solve*Nsplit;
      int Nactual = std::min(Nsplit, Nsrc - StartSrc);
      LOGA2A << "Solving sources " << StartSrc << "-" << StartSrc+Nactual-1 << std::endl;

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
class A2Ainverter5dXconjWrapper: public A2Ainverter5dBase<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
  typedef decltype( Grid::PeekIndex<GparityFlavourIndex>( *( (GridFermionFieldD*)nullptr), 0) ) GridXconjFermionFieldD;

private:
  std::unique_ptr< A2Ainverter5dBase<GridXconjFermionFieldD> > inner_inv_up; //can take ownership of the inner inverter if necessary
  const A2Ainverter5dBase<GridXconjFermionFieldD> &inner_inv;
  bool check_Xconj;
public:
  //if check_Xconj is enabled, the sources will be checked as being in X-conjugate form
  //NOTE: We do *not* by default check the input fields are X-conjugate! Please use wisely
  A2Ainverter5dXconjWrapper(const A2Ainverter5dBase<GridXconjFermionFieldD> &inner_inv,  bool check_Xconj = false): inner_inv(inner_inv), check_Xconj(check_Xconj){}

  //This version takes ownership of the underlying inverter
  A2Ainverter5dXconjWrapper(std::unique_ptr<A2Ainverter5dBase<GridXconjFermionFieldD> > &&inner_inv,  bool check_Xconj = false): inner_inv_up(std::move(inner_inv)), inner_inv(*inner_inv_up), check_Xconj(check_Xconj){}

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

    std::vector<GridXconjFermionFieldD> outX(N,grid);
    std::vector<GridXconjFermionFieldD> inX(N,grid);
    for(int i=0;i<N;i++){
      outX[i] = Grid::PeekIndex<GparityFlavourIndex>(out[i],0);
      inX[i] = Grid::PeekIndex<GparityFlavourIndex>(in[i],0);
      
      if(check_Xconj){
	if(!XconjugateCheck(in[i],1e-8,true,"src["+std::to_string(i)+"]" )) ERR.General("A2Ainverter5dXconjWrapper","invert5Dto5D","Source %d is not X-conjugate",i);
	if(!XconjugateCheck(out[i],1e-8,true,"guess["+std::to_string(i)+"]")) ERR.General("A2Ainverter5dXconjWrapper","invert5Dto5D","Guess %d is not X-conjugate",i);
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

template<typename _GridFermionFieldD>
class A2Ainverter5dCheckpointWrapper: public A2Ainverter5dBase<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
private:
  A2Ainverter5dBase<GridFermionFieldD> &solver;
  static int idx(){
    static int i = 0;
    return i++;
  }

  //Maintain a regular CG to check solutions!
  Grid::LinearOperatorBase<GridFermionFieldD> & LinOp;
  Grid::ConjugateGradient<GridFermionFieldD> cg;

public:


  A2Ainverter5dCheckpointWrapper(A2Ainverter5dBase<GridFermionFieldD> &_solver, 
				 Grid::LinearOperatorBase<GridFermionFieldD> &_LinOp, double _tol): solver(_solver), LinOp(_LinOp), cg(_tol,10000){}
  
  void invert5Dto5D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in) const override{
    assert(out.size() == in.size());
    std::string filename = "cgwrapper_ckpoint_" + std::to_string(idx());    
    Grid::GridBase* grid = in[0].Grid();
    Grid::emptyUserRecord record;
    if(checkFileExists(filename)){
      std::cout << "A2Ainverter5dCheckpointWrapper reloading checkpoint " << filename << std::endl;
      Grid::ScidacReader RD;
      RD.open(filename);
      for(int i=0; i<out.size();i++)
	RD.readScidacFieldRecord(out[i],record);
      RD.close();

      //Check; should converge immediately
      for(int i=0; i<out.size();i++)
	const_cast<Grid::ConjugateGradient<GridFermionFieldD> &>(cg)(LinOp, in[i], out[i]);

    }else{
      solver.invert5Dto5D(out,in);
      cps::sync();      
      std::cout << "A2Ainverter5dCheckpointWrapper writing checkpoint " << filename << std::endl;
      Grid::ScidacWriter WR(grid->IsBoss());
      WR.open(filename);
      for(int i=0; i<in.size();i++)
	WR.writeScidacFieldRecord(out[i],record);
      WR.close();
    }
  }
};


//Data structure for holding objects used by the inverters
template<typename Policies>
struct A2AinverterData{
  typedef typename Policies::GridDirac GridDiracD;
  typedef typename Policies::GridDiracF GridDiracF;
  typedef typename Policies::GridDiracXconj GridDiracXconjD;
  typedef typename Policies::GridDiracXconjF GridDiracXconjF;
  typedef typename Policies::GridDiracZMobius GridDiracZMobiusD;
  typedef typename Policies::GridDiracFZMobiusInner GridDiracZMobiusF;

  typedef typename Policies::GridFermionField GridFermionFieldD;
  typedef typename Policies::GridFermionFieldF GridFermionFieldF;

  template<typename T>
  using up = std::unique_ptr<T>;

  typename Policies::FgridGFclass &lattice;

  //Quark mass
  double mass;

  //Grids
  Grid::GridCartesian *FGridD, *FGridF, *UGridD, *UGridF;
  Grid::GridRedBlackCartesian *FrbGridD, *FrbGridF, *UrbGridD, *UrbGridF;

  //Gauge fields
  Grid::LatticeGaugeFieldD *UmuD;
  up<Grid::LatticeGaugeFieldF> UmuF;

  //Operators
  up<GridDiracD> OpD;
  up<GridDiracF> OpF;
  
  //Preconditioned operator wrappers
  up<A2ASchurOriginalOperatorImpl<GridDiracD> > SchurOpD;
  up<A2ASchurOriginalOperatorImpl<GridDiracF> > SchurOpF;

  //Split Grids
  int Nsplit;
  up<Grid::GridCartesian> SUGridD, SUGridF, SFGridD, SFGridF;
  up<Grid::GridRedBlackCartesian> SUrbGridD, SUrbGridF, SFrbGridD, SFrbGridF;

  //Split gauge fields
  up<Grid::LatticeGaugeFieldD> SUmuD;
  up<Grid::LatticeGaugeFieldF> SUmuF;

  //Split operators
  up<GridDiracD> SOpD;
  up<GridDiracF> SOpF;

  //Split Gparity preconditioned operator wrappers
  up<A2ASchurOriginalOperatorImpl<GridDiracD> > SSchurOpD;
  up<A2ASchurOriginalOperatorImpl<GridDiracF> > SSchurOpF;

  //Split Xconj operators
  up<GridDiracXconjD> SOpXD;
  up<GridDiracXconjF> SOpXF;

  //Split Xconj precondioned operator wrappers
  up<A2ASchurOriginalOperatorImpl<GridDiracXconjD> > SSchurOpXD;
  up<A2ASchurOriginalOperatorImpl<GridDiracXconjF> > SSchurOpXF;

  //ZMobius Grids
  up<Grid::GridCartesian> FGridInnerD, FGridInnerF;
  Grid::GridRedBlackCartesian *FrbGridInnerD;
  up<Grid::GridRedBlackCartesian> FrbGridInnerF;

  //ZMobius Gparity Dirac operators
  up<GridDiracZMobiusD> ZopD;
  up<GridDiracZMobiusF> ZopF;
  
  //ZMobius Gparity preconditioned operator wrappers
  up<A2ASchurOperatorImpl<GridDiracZMobiusD> > SchurOpD_inner;
  up<A2ASchurOperatorImpl<GridDiracZMobiusF> > SchurOpF_inner;

  A2AinverterData(Lattice &lat, double mass):
    lattice(dynamic_cast<typename Policies::FgridGFclass &>(lat)),
    FrbGridD(lattice.getFrbGrid()), FrbGridF(lattice.getFrbGridF()), FGridD(lattice.getFGrid()), FGridF(lattice.getFGridF()), 
    UGridD(lattice.getUGrid()), UGridF(lattice.getUGridF()), UrbGridD(lattice.getUrbGrid()), UrbGridF(lattice.getUrbGridF()),
    UmuD(lattice.getUmu()), mass(mass){
    UmuF.reset(new Grid::LatticeGaugeFieldF(UGridF));
    precisionChange(*UmuF, *UmuD);
  }
  
  template<typename Operator>
  static void benchmark(Operator &op, int is_gparity, const std::string &descr){
    LOGA2A << "Benchmarking operator " << descr << std::endl;
    std::vector<int> seeds5({5,6,7,8});    
    Grid::GridCartesian* grid = op.FermionGrid();
    Grid::GridParallelRNG RNG5(grid);  RNG5.SeedFixedIntegers(seeds5);
  
    typedef typename Operator::FermionField FermionField;

    FermionField src(grid); random(RNG5,src);
    Grid::RealD N2 = 1.0/::sqrt(norm2(src));
    src = src*N2;

    FermionField result(grid); result = Grid::Zero();

    int ncall =1000;
    double volume = grid->gSites();
    double NP = grid->ProcessorCount();
    double flops=(is_gparity + 1)*1320*volume*ncall;

    grid->Barrier();
    op.Dhop(src,result,0);
    double t0=Grid::usecond();
    for(int i=0;i<ncall;i++){
      op.Dhop(src,result,0);
    }
    double t1=Grid::usecond();
    grid->Barrier();
    
    LOGA2A << "Communicator comprises " << NP << " ranks" << std::endl;
    LOGA2A << "Called Dw "<<ncall<<" times in "<<t1-t0<<" us"<<std::endl;
    LOGA2A << "mflop/s =   "<< flops/(t1-t0)<<std::endl;
    LOGA2A << "mflop/s per rank =  "<< flops/(t1-t0)/NP<<std::endl;
  }  

  void setupGparityOperators(){
    if(!OpD){
      const double mob_b = lattice.get_mob_b();
      const double mob_c = lattice.get_mob_c();
      const double M5 = GJP.DwfHeight();
      const int Ls = GJP.Snodes()*GJP.SnodeSites(); 
      
      typename GridDiracD::ImplParams params;
      lattice.SetParams(params);
      
      OpD.reset(new GridDiracD(*UmuD,*FGridD,*FrbGridD,*UGridD,*UrbGridD,mass,M5,mob_b,mob_c, params));
      OpF.reset(new GridDiracF(*UmuF,*FGridF,*FrbGridF,*UGridF,*UrbGridF,mass,M5,mob_b,mob_c, params));

      SchurOpD.reset(new A2ASchurOriginalOperatorImpl<GridDiracD>(*OpD));
      SchurOpF.reset(new A2ASchurOriginalOperatorImpl<GridDiracF>(*OpF));

      benchmark(*OpD,true,"Gparity Dirac op (double)");
      benchmark(*OpF,true,"Gparity Dirac op (single)");
    }
  }


  void setupSplitGrids(const CGcontrols &cg){
    if(!SUGridD){
      LOGA2A << "Setting up split grids for VW calculation" << std::endl;
      printMem("Prior to setting up split grids");
      const int Ls = GJP.Snodes()*GJP.SnodeSites(); 

      Nsplit = 1;
    
      std::vector<int> split_grid_proc(4);
      for(int i=0;i<4;i++){
	split_grid_proc[i] = cg.split_grid_geometry.split_grid_geometry_val[i];
	if(UGridD->_processors[i] % split_grid_proc[i] != 0) ERR.General("A2AinverterData","setupSplitGrids","Split size %d in direction %d not a divisor of node geometry size %d",split_grid_proc[i],i,UGridD->_processors[i]);
	Nsplit *= UGridD->_processors[i]/split_grid_proc[i];
      }
      
      LOGA2A << Nsplit << " split Grids" << std::endl;
      LOGA2A << "Setting up double precision split grids" << std::endl;
      printMem("Prior to setting up double-precision split grids");
      
      SUGridD.reset(new Grid::GridCartesian(UGridD->_fdimensions,
					    UGridD->_simd_layout,
					    split_grid_proc,
					    *UGridD)); 
    
      SFGridD.reset(Grid::SpaceTimeGrid::makeFiveDimGrid(Ls,SUGridD.get()));
      SUrbGridD.reset(Grid::SpaceTimeGrid::makeFourDimRedBlackGrid(SUGridD.get()));
      SFrbGridD.reset(Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,SUGridD.get()));
      
      LOGA2A << "Setting up single precision split grids" << std::endl;
      printMem("Prior to setting up single-precision split grids");
      
      SUGridF.reset(new Grid::GridCartesian(UGridF->_fdimensions,
					    UGridF->_simd_layout,
					    split_grid_proc,
					    *UGridF)); 
   
      SFGridF.reset(Grid::SpaceTimeGrid::makeFiveDimGrid(Ls,SUGridF.get()));
      SUrbGridF.reset(Grid::SpaceTimeGrid::makeFourDimRedBlackGrid(SUGridF.get()));
      SFrbGridF.reset(Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,SUGridF.get()));
      
      LOGA2A << "Splitting double-precision gauge field" << std::endl;
      printMem("Prior to splitting double-precision gauge field");

      SUmuD.reset(new Grid::LatticeGaugeFieldD(SUGridD.get()));
      Grid::Grid_split(*UmuD,*SUmuD);
    
      LOGA2A << "Performing split gauge field precision change" << std::endl;
      printMem("Prior to split gauge field precision change");
      SUmuF.reset(new Grid::LatticeGaugeFieldF(SUGridF.get()));
      Grid::precisionChange(*SUmuF,*SUmuD);
      LOGA2A << "Finished setting up split grids" << std::endl;
    }
  }

  void setupSplitGparityOperators(const CGcontrols &cg){
    if(!SOpD){
      setupSplitGrids(cg);
      
      LOGA2A << "Creating split Gparity Dirac operators" << std::endl;
      printMem("Prior to creating split Dirac operators");
      const double mob_b = lattice.get_mob_b();
      const double mob_c = lattice.get_mob_c();
      const double M5 = GJP.DwfHeight();
      const int Ls = GJP.Snodes()*GJP.SnodeSites(); 
      
      typename GridDiracD::ImplParams params;
      lattice.SetParams(params);
      
      SOpD.reset(new GridDiracD(*SUmuD,*SFGridD,*SFrbGridD,*SUGridD,*SUrbGridD,mass,M5,mob_b,mob_c, params));
      SOpF.reset(new GridDiracF(*SUmuF,*SFGridF,*SFrbGridF,*SUGridF,*SUrbGridF,mass,M5,mob_b,mob_c, params));

      SSchurOpD.reset(new A2ASchurOriginalOperatorImpl<GridDiracD>(*SOpD));
      SSchurOpF.reset(new A2ASchurOriginalOperatorImpl<GridDiracF>(*SOpF));

      benchmark(*SOpD,true,"Split Gparity Dirac op (double)");
      benchmark(*SOpF,true,"Split Gparity Dirac op (single)");

      std::cout << "Finished setting up split Gparity Dirac operators" << std::endl;
    }
  }
  void setupSplitXconjOperators(const CGcontrols &cg){
    if(!SOpD){
      setupSplitGrids(cg);
      
      LOGA2A << "Creating split Xconj Dirac operators" << std::endl;
      printMem("Prior to creating split Dirac operators");
      const double mob_b = lattice.get_mob_b();
      const double mob_c = lattice.get_mob_c();
      const double M5 = GJP.DwfHeight();
      const int Ls = GJP.Snodes()*GJP.SnodeSites(); 
      
      typename GridDiracD::ImplParams params;
      lattice.SetParams(params);
      
      typename GridDiracXconjD::ImplParams xparams;
#define CP(A) xparams.A = params.A
      CP(twists); CP(dirichlet); CP(partialDirichlet);
#undef CP
      xparams.boundary_phase = 1.0; //X-conj op

      SOpXD.reset(new GridDiracXconjD(*SUmuD,*SFGridD,*SFrbGridD,*SUGridD,*SUrbGridD,mass,M5,mob_b,mob_c, xparams));
      SOpXF.reset(new GridDiracXconjF(*SUmuF,*SFGridF,*SFrbGridF,*SUGridF,*SUrbGridF,mass,M5,mob_b,mob_c, xparams));

      SSchurOpXD.reset(new A2ASchurOriginalOperatorImpl<GridDiracXconjD>(*SOpXD));
      SSchurOpXF.reset(new A2ASchurOriginalOperatorImpl<GridDiracXconjF>(*SOpXF));

      benchmark(*SOpXD,false,"Split Xconj Dirac op (double)");
      benchmark(*SOpXF,false,"Split Xconj Dirac op (single)");

      std::cout << "Finished setting up split Xconj Dirac operators" << std::endl;
    }
  }

  void setupZMobiusGrids(Grid::GridRedBlackCartesian *_FrbGridInnerD, const CGcontrols &cg){
    if(!FGridInnerD){
      const int Ls_inner = cg.madwf_params.Ls_inner;
      FGridInnerD.reset(Grid::SpaceTimeGrid::makeFiveDimGrid(Ls_inner,UGridD));
      FrbGridInnerD = _FrbGridInnerD; //needs to match the one used to generate the evecs
      FGridInnerF.reset(Grid::SpaceTimeGrid::makeFiveDimGrid(Ls_inner,UGridF));
      FrbGridInnerF.reset(Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls_inner, UGridF));
    }
  }

  void setupGparityZMobiusOperators(Grid::GridRedBlackCartesian *_FrbGridInnerD, const CGcontrols &cg){
    if(!ZopD){
      setupZMobiusGrids(_FrbGridInnerD, cg);

      const double M5 = GJP.DwfHeight();
      const int Ls_inner = cg.madwf_params.Ls_inner;
      const double bpc_inner = cg.madwf_params.b_plus_c_inner;
      double bmc_inner = 1.0; //Shamir kernel assumed
      double b_inner = (bpc_inner + bmc_inner)/2.;
      double c_inner = (bpc_inner - bmc_inner)/2.;
      typename GridDiracD::ImplParams params;
      lattice.SetParams(params);

      std::vector<Grid::ComplexD> gamma_inner = getZMobiusGamma(bpc_inner, Ls_inner, cg.madwf_params);
      ZopD.reset(new GridDiracZMobiusD(*UmuD, *FGridInnerD, *FrbGridInnerD, *UGridD, *UrbGridD, mass, M5, gamma_inner, b_inner, c_inner, params));
      ZopF.reset(new GridDiracZMobiusF(*UmuF, *FGridInnerF, *FrbGridInnerF, *UGridF, *UrbGridF, mass, M5, gamma_inner, b_inner, c_inner, params));

      //Low mode part computed using inner operator, 5' space
      if(cg.madwf_params.precond == SchurOriginal){
	SchurOpD_inner.reset(new A2ASchurOriginalOperatorImpl<GridDiracZMobiusD>(*ZopD));
	SchurOpF_inner.reset(new A2ASchurOriginalOperatorImpl<GridDiracZMobiusF>(*ZopF));
      }
      else if(cg.madwf_params.precond == SchurDiagTwo){
	SchurOpD_inner.reset(new A2ASchurDiagTwoOperatorImpl<GridDiracZMobiusD>(*ZopD));
	SchurOpF_inner.reset(new A2ASchurDiagTwoOperatorImpl<GridDiracZMobiusF>(*ZopF));
      }else{
	assert(0);
      }
    }
  }

};

  template<typename Policies>
  std::unique_ptr<A2Ainverter5dBase<typename Policies::GridFermionField> > 
  A2Ainverter5dFactory(A2AinverterData<Policies> &data, A2ACGalgorithm alg, const CGcontrols &cg){
    typedef typename Policies::GridFermionField GridFermionFieldD;
    typedef typename Policies::GridFermionFieldF GridFermionFieldF;
    
    std::unique_ptr<A2Ainverter5dBase<GridFermionFieldD> > inv5d;
    if(alg == AlgorithmCG){
      LOGA2A << "Using double precision CG solver" << std::endl;
      data.setupGparityOperators();
      inv5d.reset(new A2Ainverter5dCG<GridFermionFieldD>(data.SchurOpD->getLinOp(),cg.CG_tolerance,cg.CG_max_iters));
    }else if(alg == AlgorithmMixedPrecisionReliableUpdateCG){
      LOGA2A << "Using mixed precision reliable update CG solver" << std::endl;
      assert(cg.reliable_update_transition_tol == 0);
      data.setupGparityOperators();
      inv5d.reset(new A2Ainverter5dReliableUpdateCG<GridFermionFieldD,GridFermionFieldF>(data.SchurOpD->getLinOp(),data.SchurOpF->getLinOp(),data.FrbGridF,
											 cg.CG_tolerance,cg.CG_max_iters,cg.reliable_update_delta));
    }else if(alg == AlgorithmMixedPrecisionReliableUpdateSplitCG){
      LOGA2A << "Using mixed precision reliable update split CG solver" << std::endl;
      data.setupSplitGparityOperators(cg);

      inv5d.reset(new A2Ainverter5dReliableUpdateSplitCG<GridFermionFieldD,GridFermionFieldF>(data.SSchurOpD->getLinOp(),data.SSchurOpF->getLinOp(), data.SFrbGridD.get(), data.SFrbGridF.get(), 
											      data.Nsplit, cg.CG_tolerance,cg.CG_max_iters,cg.reliable_update_delta));
    }else if(alg == AlgorithmXconjMixedPrecisionReliableUpdateSplitCG){
      LOGA2A << "Using mixed precision reliable update split Xconj CG solver" << std::endl;
      data.setupSplitXconjOperators(cg);
      std::unique_ptr<A2Ainverter5dBase<typename Policies::GridXconjFermionField> > inv_base(
									   new A2Ainverter5dReliableUpdateSplitCG<typename Policies::GridXconjFermionField,typename Policies::GridXconjFermionFieldF>
									   (data.SSchurOpXD->getLinOp(),data.SSchurOpXF->getLinOp(), data.SFrbGridD.get(), data.SFrbGridF.get(), data.Nsplit, cg.CG_tolerance,cg.CG_max_iters,cg.reliable_update_delta));			   
      inv5d.reset(new A2Ainverter5dXconjWrapper<GridFermionFieldD>(std::move(inv_base), true)); //takes ownership
    }else{
      ERR.General("","A2Ainverter5dFactory","Unknown inverter");
    }
    return inv5d;
  }







CPS_END_NAMESPACE

#endif
