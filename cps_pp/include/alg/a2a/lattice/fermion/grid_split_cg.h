#ifndef _GRID_SPLIT_CG_H
#define _GRID_SPLIT_CG_H
#if defined(USE_GRID) && defined(USE_GRID_A2A)

#include<Grid/Grid.h>
#include<comms/sysfunc_cps.h>

namespace Grid{
  
  struct SplitGrids5D{
    GridCartesian* SUGrid;
    GridCartesian* SFGrid;
    GridRedBlackCartesian * SUrbGrid;
    GridRedBlackCartesian * SFrbGrid;
    int NsubGrids;
    
    void Setup(const std::vector<int> &subgrid_geometry, GridCartesian* UGrid, const int Ls){
      SUGrid = new GridCartesian(UGrid->_fdimensions,
				 UGrid->_simd_layout,
				 subgrid_geometry,
				 *UGrid); 

      SFGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,SUGrid);
      SUrbGrid  = SpaceTimeGrid::makeFourDimRedBlackGrid(SUGrid);
      SFrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,SUGrid);

      NsubGrids = 1;
      for(int i=0;i<Nd;i++)
	NsubGrids *= UGrid->_processors[i]/subgrid_geometry[i];
    }
    
    SplitGrids5D(const std::vector<int> &subgrid_geometry, GridCartesian* UGrid, const int Ls){
      Setup(subgrid_geometry,UGrid,Ls);
    }
    SplitGrids5D(){}

    ~SplitGrids5D(){
      delete SUGrid;
      delete SFGrid;
      delete SUrbGrid;
      delete SFrbGrid;
    }
  };

  template<typename ImplParams>
  struct MobiusLinopPolicyInputType{
    double mass;
    double b;
    double c;
    double M5;
    ImplParams params;
    
    MobiusLinopPolicyInputType(){}
    MobiusLinopPolicyInputType(const double _m, const double _b, const double _c, const double _M5, const ImplParams &p): mass(_m), b(_b), c(_c), M5(_M5), params(p){}
  };

    
  template<typename DiracOp, template<typename,typename> class LinopType>
  class MobiusLinopPolicy{
  public:
    typedef typename DiracOp::ImplParams ImplParams;
    typedef typename DiracOp::FermionField FermionField;
    typedef typename DiracOp::GaugeField GaugeField;
    typedef MobiusLinopPolicyInputType<typename DiracOp::ImplParams> InputType;

  private:
    InputType inputs;
    LinopType<DiracOp,FermionField> *linop;
    DiracOp* diracop;
  public:
    MobiusLinopPolicy(const InputType &_inputs): inputs(_inputs), linop(NULL), diracop(NULL){ }

    void Setup(const GaugeField &Umu_s, const SplitGrids5D &sgrids){
      diracop = new DiracOp(const_cast<GaugeField&>(Umu_s), *sgrids.SFGrid, *sgrids.SFrbGrid, *sgrids.SUGrid, *sgrids.SUrbGrid, inputs.mass, inputs.M5, inputs.b, inputs.c, inputs.params);
      linop = new LinopType<DiracOp,FermionField>(*diracop);
    }

    ~MobiusLinopPolicy(){
      if(linop!=NULL) delete linop;
      if(diracop!=NULL) delete diracop;
    }

    LinopType<DiracOp,FermionField> &getLinop(){ return *linop; }    
  };

  
  template<class LinopPolicyD, class LinopPolicyF,
	   typename std::enable_if<
	     getPrecision<typename LinopPolicyD::FermionField>::value == 2 &&
	     getPrecision<typename LinopPolicyF::FermionField>::value == 1
	     , int>::type = 0>
  class SplitConjugateGradientReliableUpdate{
  public:
    typedef typename LinopPolicyD::FermionField FermionFieldD;
    typedef typename LinopPolicyF::FermionField FermionFieldF;
    typedef typename LinopPolicyD::GaugeField GaugeFieldD;
    typedef typename LinopPolicyF::GaugeField GaugeFieldF;
    
    static_assert(std::is_same<typename LinopPolicyD::InputType, typename LinopPolicyF::InputType>::value, "InputTypes must be the same");
    typedef typename LinopPolicyD::InputType LinopParameters;

  private:
    LinopPolicyD linop_d;
    LinopPolicyF linop_f;
    SplitGrids5D sgrids_d;
    SplitGrids5D sgrids_f;
    GaugeFieldD *Umu_sd;
    GaugeFieldF *Umu_sf;
    ConjugateGradientReliableUpdate<FermionFieldD,FermionFieldF> *CG;
    bool use_rbgrid;
    int NsubGrids;
    LinearOperatorBase<FermionFieldD> &linop_check;
  public:
    SplitConjugateGradientReliableUpdate(RealD tol, Integer maxit, RealD _delta,
					 LinearOperatorBase<FermionFieldD> &_linop_check, //while inner workings are performed with internally-generated linops, the result is checked with this operator
					 const LinopParameters &linop_params,
					 const std::vector<int> &subgrid_geometry,
					 const GaugeFieldD &Umu_d, const GaugeFieldF &Umu_f,
					 const int Ls, bool _use_rbgrid = true, bool err_on_no_conv = true):
      linop_check(_linop_check),
      linop_d(linop_params), linop_f(linop_params),
      sgrids_d(subgrid_geometry, dynamic_cast<GridCartesian*>(Umu_d.Grid()), Ls), sgrids_f(subgrid_geometry, dynamic_cast<GridCartesian*>(Umu_f.Grid()), Ls),
      use_rbgrid(_use_rbgrid){
      
      assert(sgrids_d.NsubGrids == sgrids_f.NsubGrids);
      NsubGrids = sgrids_d.NsubGrids;

      Umu_sd = new GaugeFieldD(sgrids_d.SUGrid);
      Umu_sf = new GaugeFieldF(sgrids_f.SUGrid);

      double tsplit_Umu_d = -usecond();      
      Grid_split(const_cast<GaugeFieldD&>(Umu_d),*Umu_sd);
      tsplit_Umu_d += usecond();
      
      double tsplit_Umu_f = -usecond(); 
      Grid_split(const_cast<GaugeFieldF&>(Umu_f),*Umu_sf);
      tsplit_Umu_f += usecond();

      LOGA2A << "SplitConjugateGradientReliableUpdate constructor timings: split Umu_d " << tsplit_Umu_d/1e3 << " ms, split Umu_f " << tsplit_Umu_f/1e3 << " ms\n";
      linop_d.Setup(*Umu_sd, sgrids_d);
      linop_f.Setup(*Umu_sf, sgrids_f);

      LinearOperatorBase<FermionFieldF> &lop_f = linop_f.getLinop();
      LinearOperatorBase<FermionFieldD> &lop_d = linop_d.getLinop();
      
      CG = new ConjugateGradientReliableUpdate<FermionFieldD,FermionFieldF>(tol, maxit, _delta, use_rbgrid ? (GridBase*)sgrids_f.SFrbGrid : (GridBase*)sgrids_f.SFGrid,
									    lop_f, lop_d, err_on_no_conv);
    }

    //Functions require for setting up an external linop for fallback
    const GaugeFieldF & getSinglePrecSplitGaugeField() const{ return *Umu_sf; }
    const SplitGrids5D & getSinglePrecSplitGrids() const{ return sgrids_f; } 

    void setFallbackLinop(LinearOperatorBase<FermionFieldF> &_Linop_fallback, const RealD _fallback_transition_tol){
      CG->setFallbackLinop(_Linop_fallback, _fallback_transition_tol);
    }

    void operator()(const std::vector<FermionFieldD> &src, std::vector<FermionFieldD> &sol) {
      assert(sol.size() == src.size());
      const int Nfield = src.size();
      if(Nfield != NsubGrids){
	(std::cout << GridLogError << "SplitConjugateGradientReliableUpdate with " << NsubGrids << " split Grids received " << Nfield << " sources: these must match!\n").flush();
	assert(0);
      }
      GridBase *fgrid_s = use_rbgrid ? (GridBase*)sgrids_d.SFrbGrid : (GridBase*)sgrids_d.SFGrid;
      FermionFieldD src_split(fgrid_s);
      FermionFieldD sol_split(fgrid_s);

      double tsplit_src = -usecond();
      Grid_split(const_cast<std::vector<FermionFieldD>& >(src), src_split);
      tsplit_src += usecond();

      double tsplit_sol = -usecond();
      Grid_split(sol, sol_split);
      tsplit_sol += usecond();

      double tinvert = -usecond();
      (*CG)(src_split,sol_split);
      cps::sync(); //make sure all CGs are finished for timings
      tinvert += usecond();

      double tunsplit_sol = -usecond();
      Grid_unsplit(sol, sol_split);
      tunsplit_sol += usecond();

      //Check the solutions
      FermionFieldD mmp(src[0].Grid());
      FermionFieldD p(src[0].Grid());
      for(int i=0; i<Nfield; i++){
	linop_check.HermOp(sol[i], mmp);
        p = mmp - src[i];
	RealD srcnorm = sqrt(norm2(src[i]));
        RealD resnorm = sqrt(norm2(p));
        RealD true_residual = resnorm / srcnorm;
	LOGA2A << "\tSolution " << i << " true residual " << true_residual<<std::endl;
	LOGA2A << "\tTarget " << CG->Tolerance << std::endl;
	if (CG->ErrorOnNoConverge) assert(true_residual / CG->Tolerance < 10000.0);
      }
      LOGA2A << "SplitConjugateGradientReliableUpdate operator timings: "
      << "split sources " << tsplit_src/1e3 << " ms, "
      << "split guesses " << tsplit_sol/1e3 << " ms, "
      << "CG " << tinvert/1e3 << " ms, "
      << "unsplit solutions " << tunsplit_sol/1e3 << " ms\n";
    }

    ~SplitConjugateGradientReliableUpdate(){
      delete Umu_sd;
      delete Umu_sf;
      delete CG;
    }
  };
};

#endif
#endif


