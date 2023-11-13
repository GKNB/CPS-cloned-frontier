//Use a class to instantiate and run all the components
//The components can also be instantiated manually in the traditional Grid fashion if desired
template<typename Vtype, typename Wtype>
struct computeVW_impl{
  typedef typename Vtype::Policies Policies;
  typedef typename Policies::GridDirac GridDiracD;
  typedef typename Policies::GridDiracF GridDiracF;
  typedef typename Policies::GridDiracZMobius GridDiracZMobiusD;
  typedef typename Policies::GridDiracFZMobiusInner GridDiracZMobiusF;
  typedef typename Policies::GridFermionField GridFermionFieldD;
  typedef typename Policies::GridFermionFieldF GridFermionFieldF;

  A2AinverterData<Policies> data; //common data structure for holding Grid operators, Grids etc

  ~computeVW_impl(){}

  computeVW_impl(Vtype &V, Wtype &W, Lattice &lat, const EvecInterfaceMixedPrec<typename Policies::GridFermionField, typename Policies::GridFermionFieldF> &evecs,
		 const double mass, const CGcontrols &cg): data(lat,mass){

    //Setup the W src policy
    std::unique_ptr<A2AhighModeSource<Policies> > Wsrc_impl(highModeSourceFactory<Policies>(cg.highmode_source));
    
    data.setupGparityOperators(); //needed for everything

    if(cg.CGalgorithm == AlgorithmMixedPrecisionMADWF){
      data.setupGparityZMobiusOperators(evecs.getEvecGridD(), cg);

      A2AlowModeComputeSchurPreconditioned<GridDiracZMobiusD> vwlowimpl_inner(*data.SchurOpD_inner);
      computeVWlow(V,W,evecs,vwlowimpl_inner);

      //Currently fix to use single precision inner Dirac operator. This requires an EvecInterface object that supports single-prec deflation
      //EvecInterfaceMixedPrec<GridFermionFieldD, GridFermionFieldF> const& evecs_mix = dynamic_cast<EvecInterfaceMixedPrec<GridFermionFieldD, GridFermionFieldF> const&>(evecs);

      //For the ZMADWF implementation, the 4d inversions must be done into the space of the outer Dirac operator (5 space), using the ZMobius operator with smaller Ls internally
      A2Ainverter4dSchurPreconditionedMADWF<GridDiracD, GridDiracZMobiusF, GridFermionFieldD, GridFermionFieldF> inv4d(*data.OpD, *data.SchurOpF_inner, evecs, *data.UmuD,  
														       cg.CG_tolerance, cg.mixedCG_init_inner_tolerance, cg.CG_max_iters);

      A2AhighModeComputeGeneric<GridFermionFieldD> vwhighimpl(vwlowimpl_inner, inv4d);

      computeVWhigh(V,W,*Wsrc_impl,evecs,vwhighimpl, cg.multiCG_block_size);
    }else{
      A2AlowModeComputeSchurPreconditioned<GridDiracD> vwlowimpl(*data.SchurOpD);
      computeVWlow(V,W,evecs,vwlowimpl);

      std::unique_ptr<A2Ainverter5dBase<GridFermionFieldD> > inv5d = A2Ainverter5dFactory(data, cg.CGalgorithm, cg);

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
      vwhighimpl.reset(new A2AhighModeComputeSchurPreconditioned<GridDiracD>(*data.SchurOpD, *inv5d));
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
