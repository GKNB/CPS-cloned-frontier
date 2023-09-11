//Use a class to instantiate and run all the components
//The components can also be instantiated manually in the traditional Grid fashion if desired
template<typename Vtype, typename Wtype>
struct computeVW_impl{
  typedef typename Vtype::Policies Policies;
  typedef typename Policies::GridDirac GridDiracD;
  typedef typename Policies::GridDiracF GridDiracF;
  typedef typename Policies::GridFermionField GridFermionFieldD;
  typedef typename Policies::GridFermionFieldF GridFermionFieldF;

  typedef std::unique_ptr<Grid::GridCartesian> GridCartesianUq;
  typedef std::unique_ptr<Grid::GridRedBlackCartesian> GridRedBlackCartesianUq;
  typedef std::unique_ptr<Grid::LatticeGaugeFieldD> LatticeGaugeFieldDUq;
  typedef std::unique_ptr<Grid::LatticeGaugeFieldF> LatticeGaugeFieldFUq;
  typedef std::unique_ptr<GridDiracD> GridDiracDUq;
  typedef std::unique_ptr<GridDiracF> GridDiracFUq;

  typename Policies::FgridGFclass &lattice;
  
  bool use_split_grid;
  int Nsplit;

  //Grids
  Grid::GridCartesian *FGridD, *FGridF, *UGridD, *UGridF;
  Grid::GridRedBlackCartesian *FrbGridD, *FrbGridF, *UrbGridD, *UrbGridF;

  //Split grids
  GridCartesianUq SUGridD, SUGridF, SFGridD, SFGridF;
  GridRedBlackCartesianUq SUrbGridD, SUrbGridF, SFrbGridD, SFrbGridF;

  //Gauge fields
  Grid::LatticeGaugeFieldD *UmuD;
  LatticeGaugeFieldFUq UmuF;

  //Split gauge fields
  LatticeGaugeFieldDUq SUmuD;
  LatticeGaugeFieldFUq SUmuF;
  
  //Operators
  GridDiracDUq OpD;
  GridDiracFUq OpF;

  //Split operators
  GridDiracDUq SOpD;
  GridDiracFUq SOpF;

  ~computeVW_impl(){}

  void setupSplitGrid(const double mass, const double mob_b, const double mob_c, const double M5, const int Ls, const typename GridDiracD::ImplParams &params, const CGcontrols &cg){
    LOGA2A << "Setting up split grids for VW calculation" << std::endl;
    printMem("Prior to setting up split grids");
    use_split_grid = true;

    Nsplit = 1;
    
    std::vector<int> split_grid_proc(4);
    for(int i=0;i<4;i++){
      split_grid_proc[i] = cg.split_grid_geometry.split_grid_geometry_val[i];
      Nsplit *= UGridD->_processors[i]/split_grid_proc[i];
    }

    LOGA2A << Nsplit << " split Grids" << std::endl;
    LOGA2A << "Setting up double precision split grids" << std::endl;
   
    SUGridD.reset(new Grid::GridCartesian(UGridD->_fdimensions,
					  UGridD->_simd_layout,
					  split_grid_proc,
					  *UGridD)); 
    
    SFGridD.reset(Grid::SpaceTimeGrid::makeFiveDimGrid(Ls,SUGridD.get()));
    SUrbGridD.reset(Grid::SpaceTimeGrid::makeFourDimRedBlackGrid(SUGridD.get()));
    SFrbGridD.reset(Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,SUGridD.get()));

     LOGA2A << "Setting up single precision split grids" << std::endl;
    
    SUGridF.reset(new Grid::GridCartesian(UGridF->_fdimensions,
					  UGridF->_simd_layout,
					  split_grid_proc,
					  *UGridF)); 
   
    SFGridF.reset(Grid::SpaceTimeGrid::makeFiveDimGrid(Ls,SUGridF.get()));
    SUrbGridF.reset(Grid::SpaceTimeGrid::makeFourDimRedBlackGrid(SUGridF.get()));
    SFrbGridF.reset(Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,SUGridF.get()));

    LOGA2A << "Splitting double-precision gauge field" << std::endl;
    SUmuD.reset(new Grid::LatticeGaugeFieldD(SUGridD.get()));
    Grid::Grid_split(*UmuD,*SUmuD);
    
    LOGA2A << "Performing split gauge field precision change" << std::endl;    
    SUmuF.reset(new Grid::LatticeGaugeFieldF(SUGridF.get()));
    Grid::precisionChange(*SUmuF,*SUmuD);

    LOGA2A << "Creating split Dirac operators" << std::endl;    
    SOpD.reset(new GridDiracD(*SUmuD,*SFGridD,*SFrbGridD,*SUGridD,*SUrbGridD,mass,M5,mob_b,mob_c, params));
    SOpF.reset(new GridDiracF(*SUmuF,*SFGridF,*SFrbGridF,*SUGridF,*SUrbGridF,mass,M5,mob_b,mob_c, params));
    LOGA2A << "Finished setting up split grids" << std::endl;
    printMem("After setting up split grids");
  }

  computeVW_impl(Vtype &V, Wtype &W, Lattice &lat, const EvecInterfaceMixedPrec<typename Policies::GridFermionField, typename Policies::GridFermionFieldF> &evecs,
		 const double mass, const CGcontrols &cg):
    lattice(dynamic_cast<typename Policies::FgridGFclass &>(lat)),
    FrbGridD(lattice.getFrbGrid()), FrbGridF(lattice.getFrbGridF()), FGridD(lattice.getFGrid()), FGridF(lattice.getFGridF()), 
    UGridD(lattice.getUGrid()), UGridF(lattice.getUGridF()), UrbGridD(lattice.getUrbGrid()), UrbGridF(lattice.getUrbGridF()),
    UmuD(lattice.getUmu()),
    use_split_grid(false){

    //Setup the W src policy
    std::unique_ptr<A2AhighModeSource<Policies> > Wsrc_impl;
    switch(cg.highmode_source){
    case A2AhighModeSourceTypeOrig:
      Wsrc_impl.reset(new A2AhighModeSourceOriginal<Policies>()); break;
    case A2AhighModeSourceTypeXconj:
      Wsrc_impl.reset(new A2AhighModeSourceXconj<Policies>()); break;
    case A2AhighModeSourceTypeFlavorUnit:
      Wsrc_impl.reset(new A2AhighModeSourceFlavorUnit<Policies>()); break;
    case A2AhighModeSourceTypeFlavorCConj:
      Wsrc_impl.reset(new A2AhighModeSourceFlavorCConj<Policies>()); break;
    case A2AhighModeSourceTypeFlavorUnitary:
      Wsrc_impl.reset(new A2AhighModeSourceFlavorUnitary<Policies>()); break;
    case A2AhighModeSourceTypeFlavorRotY:
      Wsrc_impl.reset(new A2AhighModeSourceFlavorRotY<Policies>()); break;
    default:
      assert(0);
    }
    
    UmuF.reset(new Grid::LatticeGaugeFieldF(UGridF));
    precisionChange(*UmuF, *UmuD);

    //Set up operators  
    const double mob_b = lattice.get_mob_b();
    const double mob_c = lattice.get_mob_c();
    const double M5 = GJP.DwfHeight();
    const int Ls = GJP.Snodes()*GJP.SnodeSites(); 

    typename GridDiracD::ImplParams params;
    lattice.SetParams(params);
    
    OpD.reset(new GridDiracD(*UmuD,*FGridD,*FrbGridD,*UGridD,*UrbGridD,mass,M5,mob_b,mob_c, params));
    OpF.reset(new GridDiracF(*UmuF,*FGridF,*FrbGridF,*UGridF,*UrbGridF,mass,M5,mob_b,mob_c, params));

    bool use_split_grid = cg.CGalgorithm == AlgorithmMixedPrecisionReliableUpdateSplitCG;
    if(use_split_grid) setupSplitGrid(mass, mob_b, mob_c, M5, Ls, params, cg);

    if(cg.CGalgorithm == AlgorithmMixedPrecisionMADWF){
      typedef typename Policies::GridDiracZMobius GridDiracZMobiusD;
      typedef typename Policies::GridDiracFZMobiusInner GridDiracZMobiusF;
      
      const int Ls_inner = cg.madwf_params.Ls_inner;
      const double bpc_inner = cg.madwf_params.b_plus_c_inner;
      double bmc_inner = 1.0; //Shamir kernel assumed
      double b_inner = (bpc_inner + bmc_inner)/2.;
      double c_inner = (bpc_inner - bmc_inner)/2.;

      Grid::GridCartesian* FGridInnerD = Grid::SpaceTimeGrid::makeFiveDimGrid(Ls_inner,UGridD);
      Grid::GridRedBlackCartesian* FrbGridInnerD = dynamic_cast<Grid::GridRedBlackCartesian*>(evecs.getEvecGridD());
      Grid::GridCartesian* FGridInnerF = Grid::SpaceTimeGrid::makeFiveDimGrid(Ls_inner,UGridF);
      Grid::GridRedBlackCartesian* FrbGridInnerF = Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls_inner, UGridF);
      
      std::vector<Grid::ComplexD> gamma_inner = getZMobiusGamma(bpc_inner, Ls_inner, cg.madwf_params);
      GridDiracZMobiusD ZopD(*UmuD, *FGridInnerD, *FrbGridInnerD, *UGridD, *UrbGridD, mass, M5, gamma_inner, b_inner, c_inner, params);
      GridDiracZMobiusF ZopF(*UmuF, *FGridInnerF, *FrbGridInnerF, *UGridF, *UrbGridF, mass, M5, gamma_inner, b_inner, c_inner, params);

      //Low mode part computed using inner operator, 5' space
      std::unique_ptr<A2ASchurOperatorImpl<GridDiracZMobiusD> > SchurOpD_inner;
      std::unique_ptr<A2ASchurOperatorImpl<GridDiracZMobiusF> > SchurOpF_inner;
      if(cg.madwf_params.precond == SchurOriginal){
	SchurOpD_inner.reset(new A2ASchurOriginalOperatorImpl<GridDiracZMobiusD>(ZopD));
	SchurOpF_inner.reset(new A2ASchurOriginalOperatorImpl<GridDiracZMobiusF>(ZopF));
      }
      else if(cg.madwf_params.precond == SchurDiagTwo){
	SchurOpD_inner.reset(new A2ASchurDiagTwoOperatorImpl<GridDiracZMobiusD>(ZopD));
	SchurOpF_inner.reset(new A2ASchurDiagTwoOperatorImpl<GridDiracZMobiusF>(ZopF));
      }else{
	assert(0);
      }

      A2AlowModeComputeSchurPreconditioned<GridDiracZMobiusD> vwlowimpl_inner(*SchurOpD_inner);
      computeVWlow(V,W,evecs,vwlowimpl_inner);

      //Currently fix to use single precision inner Dirac operator. This requires an EvecInterface object that supports single-prec deflation
      //EvecInterfaceMixedPrec<GridFermionFieldD, GridFermionFieldF> const& evecs_mix = dynamic_cast<EvecInterfaceMixedPrec<GridFermionFieldD, GridFermionFieldF> const&>(evecs);

      //For the ZMADWF implementation, the 4d inversions must be done into the space of the outer Dirac operator (5 space), using the ZMobius operator with smaller Ls internally
      A2Ainverter4dSchurPreconditionedMADWF<GridDiracD, GridDiracZMobiusF, GridFermionFieldD, GridFermionFieldF> inv4d(*OpD, *SchurOpF_inner, evecs, *UmuD,  
														       cg.CG_tolerance, cg.mixedCG_init_inner_tolerance, cg.CG_max_iters);

      A2AhighModeComputeGeneric<GridFermionFieldD> vwhighimpl(vwlowimpl_inner, inv4d);

      computeVWhigh(V,W,*Wsrc_impl,evecs,vwhighimpl, cg.multiCG_block_size);
    }else{
      A2ASchurOriginalOperatorImpl<GridDiracD> SchurOpD(*OpD);
      A2ASchurOriginalOperatorImpl<GridDiracF> SchurOpF(*OpF);

      A2AlowModeComputeSchurPreconditioned<GridDiracD> vwlowimpl(SchurOpD);
      computeVWlow(V,W,evecs,vwlowimpl);

      std::unique_ptr<A2Ainverter5dBase<GridFermionFieldD> > inv5d;

      /////Required only for split CG
      std::unique_ptr<A2ASchurOriginalOperatorImpl<GridDiracD> > SSchurOpD;
      std::unique_ptr<A2ASchurOriginalOperatorImpl<GridDiracF> > SSchurOpF;
      /////

      if(cg.CGalgorithm == AlgorithmCG){
	LOGA2A << "Using double precision CG solver" << std::endl;
	inv5d.reset(new A2Ainverter5dCG<GridFermionFieldD>(SchurOpD.getLinOp(),cg.CG_tolerance,cg.CG_max_iters));
      }else if(cg.CGalgorithm == AlgorithmMixedPrecisionReliableUpdateCG){
	LOGA2A << "Using mixed precision reliable update CG solver" << std::endl;
	assert(cg.reliable_update_transition_tol == 0);
	inv5d.reset(new A2Ainverter5dReliableUpdateCG<GridFermionFieldD,GridFermionFieldF>(SchurOpD.getLinOp(),SchurOpF.getLinOp(),FrbGridF,
											   cg.CG_tolerance,cg.CG_max_iters,cg.reliable_update_delta));
      }else if(cg.CGalgorithm == AlgorithmMixedPrecisionRestartedCG){
	//note, we use the evecs to deflate again on each restart
	LOGA2A << "Using mixed precision restarted CG solver" << std::endl;
	inv5d.reset(new A2Ainverter5dMixedPrecCG<GridFermionFieldD,GridFermionFieldF>(SchurOpD.getLinOp(),SchurOpF.getLinOp(),evecs,FrbGridF,
										      cg.CG_tolerance,cg.CG_max_iters,cg.mixedCG_init_inner_tolerance));
      }else if(cg.CGalgorithm == AlgorithmMixedPrecisionReliableUpdateSplitCG){
	LOGA2A << "Using mixed precision reliable update split CG solver" << std::endl;
	assert(use_split_grid);

	SSchurOpD.reset(new A2ASchurOriginalOperatorImpl<GridDiracD>(*SOpD));
	SSchurOpF.reset(new A2ASchurOriginalOperatorImpl<GridDiracF>(*SOpF));
	inv5d.reset(new A2Ainverter5dReliableUpdateSplitCG<GridFermionFieldD,GridFermionFieldF>(SSchurOpD->getLinOp(),SSchurOpF->getLinOp(), SFrbGridD.get(), SFrbGridF.get(), 
												Nsplit, cg.CG_tolerance,cg.CG_max_iters,cg.reliable_update_delta));
      }else{
	assert(0);
      }

// #ifdef A2A_CHECKPOINT_INVERSIONS
//COMMENTED OUT BECAUSE FILES ARE WAY TOO LARGE!
//       //This compile option will checkpoint around every block of inversions performed, allowing much finer grained checkpointing
//       //WARNING: these are only reusable between runs if the sources and eigenvectors are identical to the run in which they were generated
//       std::unique_ptr<A2Ainverter5dBase<GridFermionFieldD> > inv5d_int(std::move(inv5d));
//       inv5d.reset(new A2Ainverter5dCheckpointWrapper<GridFermionFieldD>(*inv5d_int, SchurOpD.getLinOp(), cg.CG_tolerance));
// #endif
      
      std::unique_ptr< A2AhighModeCompute<GridFermionFieldD> > vwhighimpl;     
#if 0
      //Slower, original version
      A2AdeflatedInverter5dWrapper<GridFermionFieldD> inv5d_defl_wrap(evecs, *inv5d);
      A2Ainverter4dSchurPreconditioned<GridDiracD> inv4d(SchurOpD, inv5d_defl_wrap);  
      vwhighimpl.reset(new A2AhighModeComputeGeneric<GridFermionFieldD>(vwlowimpl, inv4d));
#else
      //Use the high mode implementation that combines the low mode part calculation with computing the guess
      //No need for initial deflation
      vwhighimpl.reset(new A2AhighModeComputeSchurPreconditioned<GridDiracD>(SchurOpD, *inv5d));
#endif

#ifdef A2A_CHECKPOINT_INVERSIONS
      //WARNING: these are only reusable between runs if the sources and eigenvectors are identical to the run in which they were generated
      //Unfortunately, because these are 4D solutions there is no way to check this!
      std::unique_ptr< A2AhighModeCompute<GridFermionFieldD> > vwhighimpl_int(std::move(vwhighimpl));
      vwhighimpl.reset(new A2AhighModeComputeCheckpointWrapper<GridFermionFieldD>(*vwhighimpl_int));
#endif

      computeVWhigh(V,W,*Wsrc_impl,evecs,*vwhighimpl, cg.multiCG_block_size);
    }
  }
};

template<typename Vtype, typename Wtype>
void computeVW(Vtype &V, Wtype &W, Lattice &lat, 
	       const EvecInterfaceMixedPrec<typename Vtype::Policies::GridFermionField, typename Vtype::Policies::GridFermionFieldF> &evecs,
	       const double mass, const CGcontrols &cg){
  computeVW_impl<Vtype,Wtype> c(V,W,lat,evecs,mass,cg);
}	       


template<typename Vtype, typename Wtype>
void computeVW(Vtype &V, Wtype &W, Lattice &lat, 
	       const std::vector<typename Vtype::Policies::GridFermionFieldF> &evec, const std::vector<Grid::RealD> &eval, 
	       const double mass, const CGcontrols &cg){
  typedef typename Vtype::Policies Policies;
  typename Policies::FgridGFclass &lat_ =  dynamic_cast<typename Policies::FgridGFclass &>(lat);
  Grid::GridRedBlackCartesian *FrbGridD = lat_.getFrbGrid();
  Grid::GridRedBlackCartesian *FrbGridF = lat_.getFrbGridF();

  //Note we currently assume the evecs are generated on the standard checkerboarded grids; this will not be true for ZMADWF or for non-checkerboarded evecs
  //In those cases use the version that takes an EvecInterface object
  if(evec.size()>0)  assert(evec[0].Grid() == FrbGridF);

  EvecInterfaceSinglePrec<typename Policies::GridFermionField, typename Policies::GridFermionFieldF> eveci(evec, eval, FrbGridD, FrbGridF);
  computeVW<Vtype,Wtype>(V,W,lat,eveci,mass,cg);
}

template<typename Vtype, typename Wtype>
void computeVW(Vtype &V, Wtype &W, Lattice &lat, 
	       const std::vector<typename Vtype::Policies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, 
	       const double mass, const CGcontrols &cg){
  typedef typename Vtype::Policies Policies;
  typename Policies::FgridGFclass &lat_ =  dynamic_cast<typename Policies::FgridGFclass &>(lat);
  Grid::GridRedBlackCartesian *FrbGridD = lat_.getFrbGrid();
  Grid::GridRedBlackCartesian *FrbGridF = lat_.getFrbGridF();

  //Note we currently assume the evecs are generated on the standard checkerboarded grids; this will not be true for ZMADWF or for non-checkerboarded evecs
  //In those cases use the version that takes an EvecInterface object
  if(evec.size()>0)  assert(evec[0].Grid() == FrbGridD);

  EvecInterfaceMixedDoublePrec<typename Policies::GridFermionField, typename Policies::GridFermionFieldF> eveci(evec, eval, FrbGridD, FrbGridF);
  computeVW<Vtype,Wtype>(V,W,lat,eveci,mass,cg);
}
