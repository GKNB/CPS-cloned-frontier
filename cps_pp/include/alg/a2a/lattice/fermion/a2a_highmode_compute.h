#pragma once
#include<config.h>

#if defined(USE_GRID) && defined(USE_GRID_A2A)
#include<Grid/Grid.h>
#include "evec_interface.h"
#include "schur_operator.h"
#include "grid_4d5d.h"

CPS_START_NAMESPACE

//Class for abstracting computational elements required for the low-mode contributions
template<typename _GridFermionFieldD>
class A2AhighModeCompute{
public:
  typedef _GridFermionFieldD GridFermionFieldD;

  virtual Grid::GridBase* get4Dgrid() const = 0;

  //The calculation of the low mode contribution to be subtracted from the high mode component is also dependent on the preconditioning scheme
  //Input and output are 4D fields
  virtual void highModeContribution4D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in, 
				     const EvecInterface<GridFermionFieldD> &evecs, const int nl) const = 0;
  
  virtual ~A2AhighModeCompute(){}
};

//A generic version which takes a separate 4D solver and an instance of A2AlowModeCompute to construct the low-mode contributioo
template<typename _GridFermionFieldD>
class A2AhighModeComputeGeneric: public A2AhighModeCompute<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
private:
  const A2AlowModeCompute<GridFermionFieldD> &low_mode_impl;
  const A2Ainverter4dBase<GridFermionFieldD> &inverter;

public:
  A2AhighModeComputeGeneric( const A2AlowModeCompute<GridFermionFieldD> &low_mode_impl,
			     const A2Ainverter4dBase<GridFermionFieldD> &inverter ): low_mode_impl(low_mode_impl), inverter(inverter){}

  Grid::GridBase* get4Dgrid() const override{ return low_mode_impl.get4Dgrid(); }

  //The calculation of the low mode contribution to be subtracted from the high mode component is also dependent on the preconditioning scheme
  //Input and output are 4D fields
  void highModeContribution4D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in, 
			      const EvecInterface<GridFermionFieldD> &evecs, const int nl) const override{
    assert(out.size() == in.size());
    int N = out.size();
    // Perform inversion from *4D->4D* (use a deflated solver to speed this up)
    inverter.invert4Dto4D(out, in);

    // Compute the low mode contribution
    std::vector<GridFermionFieldD> low(N, low_mode_impl.get4Dgrid());
    low_mode_impl.lowModeContribution4D(low, in, evecs, nl);

    // Subtract low mode contribution from solution
    for(int i=0;i<N;i++) out[i] = out[i] - low[i];
  }
    

};


//Common code for Schur preconditioned operators. The guess for the CG and the determination of the low-mode contribution are generated simultaneously
//to save on a deflation call
//NOTE: If using a 5D solver that does an initial deflation, this can be disabled to save on an expensive deflate call, as the code here already performs the deflation
template<typename _FermionOperatorTypeD>
class A2AhighModeComputeSchurPreconditioned: public A2AhighModeCompute<typename _FermionOperatorTypeD::FermionField>{
public:
  typedef _FermionOperatorTypeD FermionOperatorTypeD; //Double precision operator
  typedef typename _FermionOperatorTypeD::FermionField GridFermionFieldD;

protected:
  A2ASchurOperatorImpl<FermionOperatorTypeD> & OpD;
  const A2Ainverter5dBase<GridFermionFieldD> &inv5D;
public:
  A2AhighModeComputeSchurPreconditioned(A2ASchurOperatorImpl<FermionOperatorTypeD> &OpD, const A2Ainverter5dBase<GridFermionFieldD> &inv5D): OpD(OpD), inv5D(inv5D){}

  Grid::GridBase* get4Dgrid() const override{ return OpD.getOp().GaugeGrid(); }

  void highModeContribution4D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in, 
			     const EvecInterface<GridFermionFieldD> &evecs, const int nl) const override{
    LOGA2A << "Computing high mode contribution with for " << out.size() << " sources using shared low-mode approximation" << std::endl;
    FermionOperatorTypeD &fermop = OpD.getOp();
    Grid::SchurRedBlackBase<GridFermionFieldD> &solver = OpD.getSolver();
    Grid::GridBase* FrbGrid = fermop.FermionRedBlackGrid();
    Grid::GridBase* FGrid = fermop.FermionGrid();
    Grid::GridBase* UGrid = in[0].Grid();
    
    int N = in.size();
    assert(out.size() == N);
    GridFermionFieldD zero_e(FrbGrid); zero_e.Checkerboard() = Grid::Even; zero_e = Grid::Zero();
    GridFermionFieldD source_5d(FGrid);
    std::vector<GridFermionFieldD> source_5d_e(N,FrbGrid);
    std::vector<GridFermionFieldD> source_5d_o(N,FrbGrid);
    for(int s=0;s<N;s++){      
      fermop.ImportPhysicalFermionSource(in[s], source_5d); //applies D- appropriately
      solver.RedBlackSource(fermop, source_5d, source_5d_e[s], source_5d_o[s]);
    }

    LOGA2A << "Computing low-mode contribution / guess" << std::endl;
    std::vector<GridFermionFieldD> lowmode_contrib_5d_o(N,FrbGrid);    
    evecs.deflatedGuessD(lowmode_contrib_5d_o, source_5d_o, nl);
    
    LOGA2A << "Performing deflated solve" << std::endl;
    std::vector<GridFermionFieldD> sol_5d_o = lowmode_contrib_5d_o; //use lowmode approx as guess to speed up inversion
    inv5D.invert5Dto5D(sol_5d_o, source_5d_o);

    LOGA2A << "Reconstructing 4D high mode solutions" << std::endl;
    GridFermionFieldD lowmode_contrib_5d(FGrid);
    GridFermionFieldD full_sol_5d(FGrid);

    for(int s=0;s<N;s++){
      //Because the low-mode part is computed only through the odd-checkerboard eigenvectors, the even component of the solution comes only through -Mee^-1 Meo applied to the odd solution.
      //i.e. there is no M_ee^-1 src_e in the even part of the solution
      //We can spoof that behavior by zeroing the even part of the source in RedBlackSolution such that M_ee^-1 src_e = 0
      solver.RedBlackSolution(fermop, lowmode_contrib_5d_o[s], zero_e, lowmode_contrib_5d);
      solver.RedBlackSolution(fermop, sol_5d_o[s], source_5d_e[s], full_sol_5d);

      //Get the high mode contribution (subtract in 5D space to avoid having to export both)
      full_sol_5d = full_sol_5d - lowmode_contrib_5d;

      //Convert to 4D
      fermop.ExportPhysicalFermionSolution(full_sol_5d, out[s]);
    }
  }

};


template<typename _GridFermionFieldD>
class A2AhighModeComputeCheckpointWrapper: public A2AhighModeCompute<_GridFermionFieldD>{
public:
  typedef _GridFermionFieldD GridFermionFieldD;
private:
  const A2AhighModeCompute<GridFermionFieldD> &solver;
  static int idx(){
    static int i = 0;
    return i++;
  }
public:
  A2AhighModeComputeCheckpointWrapper(const A2AhighModeCompute<GridFermionFieldD> &solver): solver(solver){}

  Grid::GridBase* get4Dgrid() const override{ return solver.get4Dgrid(); }

  //The calculation of the low mode contribution to be subtracted from the high mode component is also dependent on the preconditioning scheme
  //Input and output are 4D fields
  void highModeContribution4D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in, 
			      const EvecInterface<GridFermionFieldD> &evecs, const int nl) const override{
    assert(out.size() == in.size());
    std::string filename = "cgwrapper4d_ckpoint_" + std::to_string(idx());    
    Grid::GridBase* grid = in[0].Grid();
    Grid::emptyUserRecord record;
    if(checkFileExists(filename)){
      std::cout << "A2AhighModeComputeCheckpointWrapper reloading checkpoint " << filename << std::endl;
      Grid::ScidacReader RD;
      RD.open(filename);
      for(int i=0; i<out.size();i++)
	RD.readScidacFieldRecord(out[i],record);
      RD.close();
    }else{
      solver.highModeContribution4D(out,in,evecs,nl);
      cps::sync();      
      std::cout << "A2AhighModeComputeCheckpointWrapper writing checkpoint " << filename << std::endl;
      Grid::ScidacWriter WR(grid->IsBoss());
      WR.open(filename);
      for(int i=0; i<in.size();i++)
	WR.writeScidacFieldRecord(out[i],record);
      WR.close();
    }
  }
 
};


CPS_END_NAMESPACE

#endif
