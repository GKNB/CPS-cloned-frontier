template<typename GridPolicies>
void EvecInterface<GridPolicies>::CGNE_MdagM(Grid::SchurDiagMooeeOperator<GridDirac,GridFermionField> &linop,
					     typename GridPolicies::GridFermionField &solution, const typename GridPolicies::GridFermionField &source,
					     const CGcontrols &cg_controls){
  if(cg_controls.CGalgorithm != AlgorithmCG) ERR.General("EvecInterface","CGNE_MdagM","Only regular CG is presently supported");
  Grid::ConjugateGradient<GridFermionField> CG(cg_controls.CG_tolerance, cg_controls.CG_max_iters);
  CG(linop, source, solution);
}
template<typename GridPolicies>
void EvecInterface<GridPolicies>::CGNE_MdagM_multi(Grid::SchurDiagMooeeOperator<GridDirac,GridFermionField> &linop,
						   std::vector<typename GridPolicies::GridFermionField> &solution, const std::vector<typename GridPolicies::GridFermionField> &source,
						   const CGcontrols &cg_controls){
  ERR.General("EvecInterface","CGNE_MdagM_multi","Not presently supported");
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
    cps_tmp_d = (double*)malloc_check(len * sizeof(double));
    bq_tmp_bfm = dwf.allocCompactFermion(); 

    assert(lat.Fclass() == GridPolicies::FGRID_CLASS_NAME);
    assert(dwf.precon_5d == 0);
    latg = dynamic_cast<FgridFclass*>(&lat);

    Grid::GridCartesian *FGrid = latg->getFGrid();
    tmp_full = new GridFermionField(FGrid);

    const int gparity = GJP.Gparity();
    if(eig.dop.gparity != gparity){ ERR.General("EvecInterfaceBFM","EvecInterfaceBFM","Gparity must be disabled/enabled for *both* CPS and the eigenvectors"); }
  }

  Grid::GridBase* getEvecGrid() const{ return latg->getFrbGrid(); }

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
  int evecPrecision() const{ return 2; }

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

  Grid::GridBase* getEvecGrid() const{ assert(evec.size()>0); return evec[0].Grid(); }

  Float getEvec(GridFermionField &into, const int idx) const{
    into = evec[idx];
    return eval[idx];
  }
  int nEvecs() const{
    return eval.size();
  }
  int evecPrecision() const{ return 2; }
};


//Fed to mixed precision solver to improve inner solve guesses using single prec eigenvectors
template<typename GridFermionField>
class deflateGuess: public Grid::LinearFunction<GridFermionField>{
  const std::vector<Grid::RealD> &eval; 
  const std::vector<GridFermionField> &evec;
public:
  
  deflateGuess(const std::vector<GridFermionField> &_evec, const std::vector<Grid::RealD> &_eval): evec(_evec), eval(_eval){}

  void operator() (const GridFermionField &src, GridFermionField &sol){
    sol = Grid::Zero();
    for(int i=0;i<eval.size();i++){
      Grid::ComplexD cn = innerProduct(evec[i], src);	
      axpy(sol, cn / eval[i], evec[i], sol);
    }
  }  
};

//Note, for using this interface for the high modes, make sure to use the constructor that passes in cg_controls, or else call
//method setupCG(const CGcontrols &cg_controls). If this is not performed, some algorithms that require more complex setup will not work (eg MADWF)

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
  
  typedef typename GridPolicies::GridDiracZMobius GridDiracZMobius;
  typedef typename GridPolicies::GridDiracFZMobiusInner GridDiracFZMobiusInner;

  const std::vector<Grid::RealD> &eval; 
  const std::vector<GridFermionFieldF> &evec;

  Grid::GridCartesian *FGrid_f;
  Grid::GridRedBlackCartesian * FrbGrid_f;
  GridGaugeFieldF *Umu_f;
  GridDiracFMixedCGInner* Ddwf_f;  
  Grid::SchurDiagMooeeOperator<GridDiracFMixedCGInner,GridFermionFieldF> *Linop_f; //single/single *or* single/half depending on policies
  GridDiracF* Ddwf_f_ss;
  Grid::SchurDiagMooeeOperator<GridDiracF,GridFermionFieldF> *Linop_f_ss; //single/single

  //Operators required for MADWF
  GridDiracFZMobiusInner* DZmob_f;
  Grid::GridCartesian * FGrid_Zmob_f;
  Grid::GridRedBlackCartesian * FrbGrid_Zmob_f;

  bool delete_FrbGrid_Zmob_f; //if this object news the grid rather than imports it, it must be deleted

  //Other parameters
  Grid::GridRedBlackCartesian *UrbGrid;
  Grid::GridCartesian *UGrid;
  Grid::GridCartesian *UGrid_f;
  Grid::GridRedBlackCartesian *UrbGrid_f;
  const GridGaugeField & Umu;
  double mass;
  double mob_b;
  double mob_c;
  double M5;
  typename GridDiracFMixedCGInner::ImplParams params;
public: 

  //Note this version doesn't setup all the linear operators required for the high mode CG as some require the cg_controls
  EvecInterfaceGridSinglePrec(const std::vector<GridFermionFieldF> &_evec, 
			      const std::vector<Grid::RealD> &_eval, 
			      Lattice &lat, const double _mass): evec(_evec), eval(_eval),
								 mass(_mass),
								 Umu(*dynamic_cast<FgridFclass&>(lat).getUmu()),								 
								 DZmob_f(NULL), FGrid_Zmob_f(NULL), FrbGrid_Zmob_f(NULL), delete_FrbGrid_Zmob_f(false)
  {
    //Copy the Grid pointers
    FgridFclass &latg = dynamic_cast<FgridFclass&>(lat);
    UGrid = latg.getUGrid();
    UrbGrid = latg.getUrbGrid();
    UGrid_f = latg.getUGridF();
    FGrid_f = latg.getFGridF();
    UrbGrid_f = latg.getUrbGridF();
    FrbGrid_f = latg.getFrbGridF();
   
    //Setup gauge field and Dirac operators
    Umu_f = new GridGaugeFieldF(UGrid_f);
    precisionChange(*Umu_f, Umu);

    mob_b = latg.get_mob_b();
    mob_c = mob_b - 1.;   //b-c = 1
    M5 = GJP.DwfHeight();

    latg.SetParams(params);
    
    //single-single/single-half
    Ddwf_f = new GridDiracFMixedCGInner(*Umu_f,*FGrid_f,*FrbGrid_f,*UGrid_f,*UrbGrid_f,mass,M5,mob_b,mob_c, params);
    Linop_f = new Grid::SchurDiagMooeeOperator<GridDiracFMixedCGInner,GridFermionFieldF>(*Ddwf_f);

    //single-single
    Ddwf_f_ss = new GridDiracF(*Umu_f,*FGrid_f,*FrbGrid_f,*UGrid_f,*UrbGrid_f,mass,M5,mob_b,mob_c, params);
    Linop_f_ss = new Grid::SchurDiagMooeeOperator<GridDiracF,GridFermionFieldF>(*Ddwf_f_ss);
  }
  
  void setupCG(const CGcontrols &cg_controls){
    if(cg_controls.CGalgorithm == AlgorithmMixedPrecisionMADWF){
      if(!UniqueID()) printf("EvecInterfaceGridSinglePrec::setupCG setting up inner Dirac operator\n");
      std::vector<Grid::ComplexD> gamma_inner = computeZmobiusGammaWithCache(cg_controls.MADWF_b_plus_c_inner, 
									     cg_controls.MADWF_Ls_inner, 
									     mob_b+mob_c, GJP.SnodeSites()*GJP.Snodes(),
									     cg_controls.MADWF_ZMobius_lambda_max, cg_controls.MADWF_use_ZMobius);

      Grid::GridCartesian * FGrid_Zmob_f = Grid::SpaceTimeGrid::makeFiveDimGrid(cg_controls.MADWF_Ls_inner,UGrid_f);

      //If we have eigenvectors, the single prec rb grid should be imported from those
      if(evec.size() > 0) FrbGrid_Zmob_f = dynamic_cast<Grid::GridRedBlackCartesian*>(evec[0].Grid());
      else{
	FrbGrid_Zmob_f = Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid(cg_controls.MADWF_Ls_inner,UGrid_f);
	delete_FrbGrid_Zmob_f = true;
      }

      double bmc = 1.0;//Shamir kernel
      double bpc = cg_controls.MADWF_b_plus_c_inner;
      double b_inner = 0.5*(bpc + bmc);
      double c_inner = 0.5*(bpc - bmc);
      if(!UniqueID()) printf("EvecInterfaceGridSinglePrec::setupCG Creating single-prec ZMobius inner Dirac operator with b=%g c=%g b+c=%g Ls=%d\n",b_inner,c_inner, bpc,cg_controls.MADWF_Ls_inner);
      DZmob_f = new GridDiracFZMobiusInner(*Umu_f, *FGrid_Zmob_f, *FrbGrid_Zmob_f, *UGrid_f, *UrbGrid_f, mass, M5, gamma_inner, b_inner, c_inner, params);
    }
  }


  EvecInterfaceGridSinglePrec(const std::vector<GridFermionFieldF> &_evec, 
			      const std::vector<Grid::RealD> &_eval, 
			      Lattice &lat, const double _mass, const CGcontrols &cg_controls):
    EvecInterfaceGridSinglePrec(_evec, _eval, lat, _mass){
    this->setupCG(cg_controls);
  }
  

  ~EvecInterfaceGridSinglePrec(){
    delete Umu_f;
    delete Ddwf_f;
    delete Linop_f;
    delete Ddwf_f_ss;
    delete Linop_f_ss;

    if(DZmob_f) delete DZmob_f;
    if(FGrid_Zmob_f) delete FGrid_Zmob_f;
    if(delete_FrbGrid_Zmob_f && FrbGrid_Zmob_f) delete FrbGrid_Zmob_f;
  }
  
  Grid::GridBase* getEvecGrid() const{ assert(evec.size()>0); return evec[0].Grid(); }

  Float getEvec(GridFermionField &into, const int idx) const{ //get *double precision* eigenvector
    precisionChange(into,evec[idx]);
    return eval[idx];
  }
  
  int nEvecs() const{
    return eval.size();
  }

  const std::vector<Grid::RealD> getEvals() const{ return eval; }
  const std::vector<GridFermionFieldF> &getEvecs() const{ return evec; }
  
  //Overload high-mode solve to call mixed precision CG with single prec evecs
  //Note that the solution vector guess provided for this function typically is computed from the low-mode approximation and so
  //further use of the eigenvectors is not strictly necessary. Here we use them to speed up the inner solves of the restarted algorithms.
  //The above does not apply when using MADWF as the eigenvectors have a different Ls than the source/solution; instead they are used to form the low-mode guess internally
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
    else if(cg_controls.CGalgorithm == AlgorithmMixedPrecisionMADWF){
      if(DZmob_f == NULL) 
	ERR.General("EvecInterfaceGridSinglePrec","CGNE_MdagM", 
		    "Mobius inner operator has not been initialized. Make sure you called either setupCG or used the constructor that passes in cg_controls");

      //*NOTE* : this assumes the eigenvectors are for the inner Mobius operator and preconditioned using the SchurRedBlackDiagTwoSolve preconditioner
      deflateGuess<GridFermionFieldF> guesser(evec,eval);
      Grid_MADWF_mixedprec_invert<GridPolicies, deflateGuess<GridFermionFieldF> >(solution, source, cg_controls, Umu, linop._Mat, *DZmob_f, guesser, cg_controls.MADWF_precond);

    }else if(cg_controls.CGalgorithm == AlgorithmCG){
      this->EvecInterface<GridPolicies>::CGNE_MdagM(linop, solution, source, cg_controls); //converts evecs to double precision on-the-fly
    }
    else ERR.General("EvecInterfaceGridSinglePrec","CGNE_MdagM","Unknown CG algorithm");
  }

  void CGNE_MdagM_multi(Grid::SchurDiagMooeeOperator<GridDirac,GridFermionField> &linop,
			std::vector<typename GridPolicies::GridFermionField> &solution, const std::vector<typename GridPolicies::GridFermionField> &source,
			const CGcontrols &cg_controls){
    if(cg_controls.CGalgorithm == AlgorithmMixedPrecisionReliableUpdateSplitCG){
      std::vector<int> split_grid_geometry(cg_controls.split_grid_geometry.split_grid_geometry_val, cg_controls.split_grid_geometry.split_grid_geometry_val + cg_controls.split_grid_geometry.split_grid_geometry_len);
      typedef Grid::MobiusLinopPolicy<GridDirac,Grid::SchurDiagMooeeOperator> LinopPolicyD;
      typedef Grid::MobiusLinopPolicy<GridDiracFMixedCGInner,Grid::SchurDiagMooeeOperator> LinopPolicyF;
      typedef Grid::MobiusLinopPolicy<GridDiracF,Grid::SchurDiagMooeeOperator> LinopPolicyFFallback;

      typename LinopPolicyD::InputType linop_inputs(mass,mob_b,mob_c,M5,params);

      //We ignore the input Linop and make our own here using the above policies
      Grid::SplitConjugateGradientReliableUpdate<LinopPolicyD,LinopPolicyF> CG(cg_controls.CG_tolerance, cg_controls.CG_max_iters, cg_controls.reliable_update_delta,linop,
									       linop_inputs, split_grid_geometry, Umu, *Umu_f, GJP.Snodes()*GJP.SnodeSites(), true, true);

      
      LinopPolicyFFallback fallback(linop_inputs);
      fallback.Setup(CG.getSinglePrecSplitGaugeField(), CG.getSinglePrecSplitGrids());

      if(cg_controls.reliable_update_transition_tol > 0.) CG.setFallbackLinop(fallback.getLinop(), cg_controls.reliable_update_transition_tol);

      CG(source,solution);
    }else{
      ERR.General("EvecInterfaceGridSinglePrec","CGNE_MdagM_multi","Unknown multi-CG algorithm");
    }
  }
  
  void Report() const{
    Ddwf_f->Report();
  }
  
  int evecPrecision() const{ return 1; }
};

#endif
