template<typename GridPolicies>
void EvecInterface<GridPolicies>::CGNE_MdagM(Grid::SchurDiagMooeeOperator<GridDirac,GridFermionField> &linop,
					     typename GridPolicies::GridFermionField &solution, const typename GridPolicies::GridFermionField &source,
					     const CGcontrols &cg_controls){
  if(cg_controls.CGalgorithm != AlgorithmCG) ERR.General("EvecInterface","CGNE_MdagM","Only regular CG is presently supported");
  Grid::ConjugateGradient<GridFermionField> CG(cg_controls.CG_tolerance, cg_controls.CG_max_iters);
  CG(linop, source, solution);
}

//BFM evecs
#ifdef USE_BFM_LANCZOS

template<typename GridPolicies>
class EvecInterfaceBFM: public EvecInterface<GridPolicies>{
  typedef typename GridPolicies::GridFermionField GridFermionField;
  typename GridPolicies::FgridFclass FgridFclass;
  
  BFM_Krylov::Lanczos_5d<double> &eig;
  bfm_evo<double> &dwf;
  FgridFclass *latg;
  double *cps_tmp_d;
  Fermion_t bq_tmp_bfm;
  bool singleprec_evecs;
  int len;
  GridFermionField *tmp_full;
public:
  EvecInterfaceBFM(BFM_Krylov::Lanczos_5d<double> &_eig, bfm_evo<double> &_dwf, Lattice &lat, const bool _singleprec_evecs): eig(_eig), dwf(_dwf), singleprec_evecs(_singleprec_evecs){
    
    len = 24 * eig.dop.node_cbvol * (1 + dwf.gparity) * eig.dop.cbLs;
    cps_tmp_d = (double*)malloc(len * sizeof(double));
    bq_tmp_bfm = dwf.allocCompactFermion(); 

    assert(lat.Fclass() == GridPolicies::FGRID_CLASS_NAME);
    assert(dwf.precon_5d == 0);
    latg = dynamic_cast<FgridFclass*>(&lat);

    Grid::GridCartesian *FGrid = latg->getFGrid();
    tmp_full = new GridFermionField(FGrid);

    const int gparity = GJP.Gparity();
    if(eig.dop.gparity != gparity){ ERR.General("EvecInterfaceBFM","EvecInterfaceBFM","Gparity must be disabled/enabled for *both* CPS and the eigenvectors"); }
  }
  Float getEvec(GridFermionField &into, const int idx){
    omp_set_num_threads(bfmarg::threads);
    
    //Copy bq[i][1] into bq_tmp
    if(singleprec_evecs){ // eig->bq is in single precision
      //Upcast the float type to double
#pragma omp parallel for 
      for(int j = 0; j < len; j++) {
	((double*)bq_tmp_bfm)[j] = ((float*)(eig.bq[idx][1]))[j];
      }
      //Use bfm_evo to convert to a CPS field
      dwf.cps_impexcbFermion<double>(cps_tmp_d, bq_tmp_bfm, 0, Odd);

    }else{ // eig.bq is in double precision
      //Use bfm_evo to convert to a CPS field
      dwf.cps_impexcbFermion<double>(cps_tmp_d, eig.bq[idx][1], 0, Odd);     
    }
    //Use Fgrid to convert to a Grid field
    *tmp_full = Grid::zero;
    latg->ImportFermion(*tmp_full, (Vector*)cps_tmp_d, FgridBase::Odd);
    pickCheckerboard(Odd,into,*tmp_full);

    return eig.evals[idx];
  }
  int nEvecs() const{
    return eig.get;
  }

  ~EvecInterfaceBFM(){
    free(cps_tmp_d);
    dwf.freeFermion(bq_tmp_bfm);
    delete tmp_full;
  }

};

#endif





#ifdef USE_GRID_LANCZOS

template<typename GridPolicies>
class EvecInterfaceGrid: public EvecInterface<GridPolicies>{
  typedef typename GridPolicies::GridFermionField GridFermionField;
  const std::vector<Grid::RealD> &eval; 
  const std::vector<GridFermionField> &evec;

public:
  EvecInterfaceGrid(const std::vector<GridFermionField> &_evec, const std::vector<Grid::RealD> &_eval): evec(_evec), eval(_eval){}

  Float getEvec(GridFermionField &into, const int idx){
    into = evec[idx];
    return eval[idx];
  }
  int nEvecs() const{
    return eval.size();
  }
};


//Fed to mixed precision solver to improve inner solve guesses using single prec eigenvectors
template<typename GridFermionField>
class deflateGuess: public Grid::LinearFunction<GridFermionField>{
  const std::vector<Grid::RealD> &eval; 
  const std::vector<GridFermionField> &evec;
public:
  
  deflateGuess(const std::vector<GridFermionField> &_evec, const std::vector<Grid::RealD> &_eval): evec(_evec), eval(_eval){}

  void operator() (const GridFermionField &src, GridFermionField &sol){
    for(int i=0;i<eval.size();i++){
      Grid::ComplexD cn = innerProduct(evec[i], src);	
      axpy(sol, cn / eval[i], evec[i], sol);
    }
  }  
};

template<typename GridPolicies>
class EvecInterfaceGridSinglePrec: public EvecInterface<GridPolicies>{
  typedef typename GridPolicies::GridFermionField GridFermionField;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDirac GridDirac;
  typedef typename GridDirac::GaugeField GridGaugeField;

  typedef typename GridPolicies::GridDiracF GridDiracF; //single/single
  typedef typename GridPolicies::GridDiracFMixedCGInner GridDiracFMixedCGInner;
  typedef typename GridPolicies::GridFermionFieldF GridFermionFieldF;
  typedef typename GridDiracF::GaugeField GridGaugeFieldF;
  
  const std::vector<Grid::RealD> &eval; 
  const std::vector<GridFermionFieldF> &evec;

  Grid::GridRedBlackCartesian * FrbGrid_f;
  GridGaugeFieldF *Umu_f;
  GridDiracFMixedCGInner* Ddwf_f;
  Grid::SchurDiagMooeeOperator<GridDiracFMixedCGInner,GridFermionFieldF> *Linop_f; //single/single *or* single/half depending on policies

  GridDiracF* Ddwf_f_ss;
  Grid::SchurDiagMooeeOperator<GridDiracF,GridFermionFieldF> *Linop_f_ss; //single/single

  bool delete_FrbGrid_f; //if this object news the grid rather than imports it, it must be deleted

public:
  EvecInterfaceGridSinglePrec(const std::vector<GridFermionFieldF> &_evec, const std::vector<Grid::RealD> &_eval, Lattice &lat, const double mass): evec(_evec), eval(_eval), delete_FrbGrid_f(false){
    FgridFclass &latg = dynamic_cast<FgridFclass&>(lat);
    const GridGaugeField & Umu = *latg.getUmu();    
    
    //Make a single precision Grid (used by the Mixed prec solver also even if no evecs)
    std::vector<int> nodes(4);
    std::vector<int> vol(4);
    for(int i=0;i<4;i++){
      vol[i]= GJP.NodeSites(i)*GJP.Nodes(i);;
      nodes[i]= GJP.Nodes(i);
    }
    Grid::GridCartesian *UGrid_f = Grid::QCD::SpaceTimeGrid::makeFourDimGrid(vol,Grid::GridDefaultSimd(Grid::QCD::Nd,Grid::vComplexF::Nsimd()),nodes);
    Grid::GridCartesian *FGrid_f = Grid::QCD::SpaceTimeGrid::makeFiveDimGrid(GJP.SnodeSites()*GJP.Snodes(),UGrid_f);
    Grid::GridRedBlackCartesian *UrbGrid_f = Grid::QCD::SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_f);

    if(_evec.size() > 0) FrbGrid_f = dynamic_cast<Grid::GridRedBlackCartesian*>(_evec[0]._grid);
    else{
      FrbGrid_f = Grid::QCD::SpaceTimeGrid::makeFiveDimRedBlackGrid(GJP.SnodeSites()*GJP.Snodes(),UGrid_f);
      delete_FrbGrid_f = true;
    }
    
    Umu_f = new GridGaugeFieldF(UGrid_f);
    precisionChange(*Umu_f, Umu);

    const double mob_b = latg.get_mob_b();
    const double mob_c = mob_b - 1.;   //b-c = 1
    const double M5 = GJP.DwfHeight();

    typename GridDiracFMixedCGInner::ImplParams params;
    latg.SetParams(params);
    
    Ddwf_f = new GridDiracFMixedCGInner(*Umu_f,*FGrid_f,*FrbGrid_f,*UGrid_f,*UrbGrid_f,mass,M5,mob_b,mob_c, params);
    Linop_f = new Grid::SchurDiagMooeeOperator<GridDiracFMixedCGInner,GridFermionFieldF>(*Ddwf_f);

    Ddwf_f_ss = new GridDiracF(*Umu_f,*FGrid_f,*FrbGrid_f,*UGrid_f,*UrbGrid_f,mass,M5,mob_b,mob_c, params);
    Linop_f_ss = new Grid::SchurDiagMooeeOperator<GridDiracF,GridFermionFieldF>(*Ddwf_f_ss);
  }
  ~EvecInterfaceGridSinglePrec(){
    delete Umu_f;
    delete Ddwf_f;
    delete Linop_f;
    delete Ddwf_f_ss;
    delete Linop_f_ss;
    if(delete_FrbGrid_f) delete FrbGrid_f;
  }
  
  Float getEvec(GridFermionField &into, const int idx){ //get *double precision* eigenvector
    precisionChange(into,evec[idx]);
    return eval[idx];
  }
  
  int nEvecs() const{
    return eval.size();
  }

  const std::vector<Grid::RealD> getEvals() const{ return eval; }
  const std::vector<GridFermionFieldF> getEvecs() const{ return evec; }
  
  //Overload high-mode solve to call mixed precision CG with single prec evecs
  void CGNE_MdagM(Grid::SchurDiagMooeeOperator<GridDirac, GridFermionField> &linop,
		  GridFermionField &solution, const GridFermionField &source,
		  const CGcontrols &cg_controls){


    
    if(cg_controls.CGalgorithm == AlgorithmMixedPrecisionRestartedCG){
      Grid::MixedPrecisionConjugateGradient<GridFermionField,GridFermionFieldF> mCG(cg_controls.CG_tolerance, cg_controls.CG_max_iters, 50, FrbGrid_f, *Linop_f, linop);
      deflateGuess<GridFermionFieldF> guesser(evec,eval);
      mCG.useGuesser(guesser);
# ifndef DISABLE_GRID_MCG_INNERTOL //Temporary catch for old branches of Grid that do not have the new mixed CG inner tol option
      mCG.InnerTolerance = cg_controls.mixedCG_init_inner_tolerance;
# endif
      mCG(source,solution);
    }
#ifndef DISABLE_GRID_RELIABLE_UPDATE_CG //Old versions of Grid don't have it      
    else if(cg_controls.CGalgorithm == AlgorithmMixedPrecisionReliableUpdateCG){
      Grid::ConjugateGradientReliableUpdate<GridFermionField,GridFermionFieldF> rlCG(cg_controls.CG_tolerance, cg_controls.CG_max_iters, cg_controls.reliable_update_delta, FrbGrid_f, *Linop_f, linop);
      if(cg_controls.reliable_update_transition_tol > 0.) rlCG.setFallbackLinop(*Linop_f_ss, cg_controls.reliable_update_transition_tol);
      rlCG(source, solution);
    }      
#endif
    else if(cg_controls.CGalgorithm == AlgorithmCG){
      this->EvecInterface<GridPolicies>::CGNE_MdagM(linop, solution, source, cg_controls);
    }
    else ERR.General("EvecInterfaceGridSinglePrec","CGNE_MdagM","Unknown CG algorithm");
  }

  void Report() const{
    Ddwf_f->Report();
  }
  
};

#endif
