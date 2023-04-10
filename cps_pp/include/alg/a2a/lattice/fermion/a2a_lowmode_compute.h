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
class A2AlowModeCompute{
public:
  typedef _GridFermionFieldD GridFermionFieldD;

  virtual Grid::GridBase* get4Dgrid() const = 0;

  //Out is a *4D* field defined on the full, unpreconditioned Grid. evec resides on either a checkerboarded or full Grid in whatever space appropriate for the implementation
  virtual void computeVl(GridFermionFieldD &out, const GridFermionFieldD &evec, const Grid::RealD eval) const = 0;
  
  virtual void computeWl(GridFermionFieldD &out, const GridFermionFieldD &evec) const = 0;

  //The calculation of the low mode contribution to be subtracted from the high mode component is also dependent on the preconditioning scheme
  //Input and output are 4D fields
  virtual void lowModeContribution4D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in, 
				     const EvecInterface<GridFermionFieldD> &evecs, const int nl) const = 0;
  
  virtual ~A2AlowModeCompute(){}
};

//Common code for Schur preconditioned operators
template<typename _FermionOperatorTypeD>
class A2AlowModeComputeSchurPreconditioned: public A2AlowModeCompute<typename _FermionOperatorTypeD::FermionField>{
public:
  typedef _FermionOperatorTypeD FermionOperatorTypeD; //Double precision operator
  typedef typename _FermionOperatorTypeD::FermionField GridFermionFieldD;

protected:
  A2ASchurOperatorImpl<FermionOperatorTypeD> & OpD;

public:
  A2AlowModeComputeSchurPreconditioned(A2ASchurOperatorImpl<FermionOperatorTypeD> &OpD): OpD(OpD){}

  Grid::GridBase* get4Dgrid() const override{ return OpD.getOp().GaugeGrid(); }

  //Apply operations in Eqs 21 and 27 of https://rbc.phys.columbia.edu/rbc_ukqcd/individual_postings/ckelly/Gparity/note_a2a_v5.pdf
  void computeVl(GridFermionFieldD &out, const GridFermionFieldD &evec, const Grid::RealD eval) const override{
    assert(evec.Checkerboard() == Grid::Odd);
    GridFermionFieldD tmp(evec.Grid()), tmp2(evec.Grid()), evecmul(evec.Grid());
    OpD.SchurEvecMulVl(evecmul, evec);
    
    FermionOperatorTypeD &fermop = OpD.getOp();

    //Compute even part [ -(Mee)^-1 Meo bq_tmp, bq_tmp ]
    fermop.Meooe(evecmul,tmp2);	//tmp2 = Meo bq_tmp 
    fermop.MooeeInv(tmp2,tmp);   //tmp = (Mee)^-1 Meo bq_tmp
    tmp = -tmp; //even checkerboard
    
    assert(tmp.Checkerboard() == Grid::Even);
    
    GridFermionFieldD tmp_full(fermop.FermionGrid());
    setCheckerboard(tmp_full, tmp); //even checkerboard
    setCheckerboard(tmp_full, evecmul); //odd checkerboard

    assert(fermop.FermionGrid()->Nd() == 5);
    int Ls = fermop.FermionGrid()->GlobalDimensions()[0]; //5th dim is dimension 0!
    
    //Get 4D part and poke into a
    //Recall that D^{-1} = <v w^\dagger> = <q \bar q>.  v therefore transforms like a spinor. For spinors \psi(x) = P_R \psi(x,Ls-1) + P_L \psi(x,0),  i.e. s_u=Ls-1 and s_l=0 for CPS gamma5
    conformable(out.Grid(), fermop.GaugeGrid());

    DomainWallFiveToFour(out, tmp_full, Ls-1,0);
    out = Grid::RealD(1./eval) * out;
  }

  //No difference between SchurOriginal and SchurDiagTwo apart from the linop
  void computeWl(GridFermionFieldD &out, const GridFermionFieldD &evec) const override{
    assert(evec.Checkerboard() == Grid::Odd);
    GridFermionFieldD tmp(evec.Grid()), tmp2(evec.Grid()), tmp3(evec.Grid());

    //Do tmp = [ -[Mee^-1]^dag [Meo]^dag Doo bq_tmp,  Doo bq_tmp ]    (Note that for the Moe^dag in Daiqian's thesis, the dagger also implies a transpose of the spatial indices, hence the Meo^dag in the code)
    OpD.getSchurOp().Mpc(evec,tmp2);  //tmp2 = Doo bq_tmp
        
    FermionOperatorTypeD &fermop = OpD.getOp();

    fermop.MeooeDag(tmp2,tmp3); //tmp3 = Meo^dag Doo bq_tmp
    fermop.MooeeInvDag(tmp3,tmp); //tmp = [Mee^-1]^dag Meo^dag Doo bq_tmp
    tmp = -tmp;
    
    assert(tmp.Checkerboard() == Grid::Even);
    assert(tmp2.Checkerboard() == Grid::Odd);

    GridFermionFieldD tmp_full(fermop.FermionGrid()), tmp_full2(fermop.FermionGrid());
    setCheckerboard(tmp_full, tmp);
    setCheckerboard(tmp_full, tmp2);

    //Left-multiply by D-^dag
    fermop.DminusDag(tmp_full, tmp_full2);

    int Ls = fermop.FermionGrid()->GlobalDimensions()[0]; //5th dim is dimension 0!   
    conformable(out.Grid(), fermop.GaugeGrid());    

    //Get 4D part, poke onto a then copy into wl
    //Recall that D^{-1} = <v w^\dagger> = <q \bar q>.  w (and w^\dagger) therefore transforms like a conjugate spinor. For spinors \bar\psi(x) =  \bar\psi(x,0) P_R +  \bar\psi(x,Ls-1) P_L,  i.e. s_u=0 and s_l=Ls-1 for CPS gamma5
    DomainWallFiveToFour(out, tmp_full2, 0, Ls-1);
  }

  void lowModeContribution4D(std::vector<GridFermionFieldD> &out, const std::vector<GridFermionFieldD> &in, 
			     const EvecInterface<GridFermionFieldD> &evecs, const int nl) const override{
    FermionOperatorTypeD &fermop = OpD.getOp();
    Grid::SchurRedBlackBase<GridFermionFieldD> &solver = OpD.getSolver();
    Grid::GridBase* FrbGrid = fermop.FermionRedBlackGrid();
    Grid::GridBase* FGrid = fermop.FermionGrid();

    int N = in.size();
    assert(out.size() == N);
    GridFermionFieldD source_5d(FGrid);
    std::vector<GridFermionFieldD> source_5d_e(N,FrbGrid);
    std::vector<GridFermionFieldD> source_5d_o(N,FrbGrid);
    for(int s=0;s<N;s++){      
      fermop.ImportPhysicalFermionSource(in[s], source_5d); //applies D- appropriately
      solver.RedBlackSource(fermop, source_5d, source_5d_e[s], source_5d_o[s]);
    }

    std::vector<GridFermionFieldD> lowmode_contrib_5d_o(N,FrbGrid);    
    evecs.deflatedGuessD(lowmode_contrib_5d_o, source_5d_o, nl);
    
    //Because the low-mode part is computed only through the odd-checkerboard eigenvectors, the even component of the solution comes only through -Mee^-1 Meo applied to the odd solution.
    //i.e. there is no M_ee^-1 src_e in the even part of the solution
    //We can spoof that behavior by zeroing the even part of the source in RedBlackSolution such that M_ee^-1 src_e = 0
    source_5d_e[0] = Grid::Zero();

    GridFermionFieldD lowmode_contrib_5d(FGrid);
    for(int s=0;s<N;s++){
      solver.RedBlackSolution(fermop, lowmode_contrib_5d_o[s], source_5d_e[0], lowmode_contrib_5d);
      fermop.ExportPhysicalFermionSolution(lowmode_contrib_5d, out[s]);
    }
  }

};



CPS_END_NAMESPACE

#endif
