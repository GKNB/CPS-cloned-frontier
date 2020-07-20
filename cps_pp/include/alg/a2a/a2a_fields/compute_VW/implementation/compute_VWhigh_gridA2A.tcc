//Compute the high mode parts of V and W.   "Single" means this version is designed for single-RHS inverters
template< typename Policies>
void computeVWhighSingle(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass, const CGcontrols &cg_controls){
  assert(!isMultiCG(cg_controls.CGalgorithm));
  assert(cg_controls.CGalgorithm != AlgorithmMixedPrecisionMADWF);

  typedef typename Policies::GridFermionField GridFermionField;
  typedef typename Policies::FgridFclass FgridFclass;
  typedef typename Policies::GridDirac GridDirac;
  
  const char *fname = "computeVWhighSingle(....)";

  int ngp = 0;
  for(int i=0;i<3;i++) if(GJP.Bc(i) == BND_CND_GPARITY) ++ngp;

  if(Policies::GPARITY == 1 && ngp == 0) ERR.General("A2AvectorW","computeVWhighSingle","A2Apolicy is for G-parity\n");
  if(Policies::GPARITY == 0 && ngp != 0) ERR.General("A2AvectorW","computeVWhighSingle","A2Apolicy is not for G-parity\n");

  assert(lat.Fclass() == Policies::FGRID_CLASS_NAME);
  FgridFclass &latg = dynamic_cast<FgridFclass&>(lat);

  //Grids and gauge field
  Grid::GridCartesian *UGrid = latg.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = latg.getUrbGrid();
  Grid::GridCartesian *FGrid = latg.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = latg.getFrbGrid();
  Grid::LatticeGaugeFieldD *Umu = latg.getUmu();
  
  //Mobius parameters
  const double mob_b = latg.get_mob_b();
  const double mob_c = mob_b - 1.;   //b-c = 1
  const double M5 = GJP.DwfHeight();
  printf("Grid b=%g c=%g b+c=%g\n",mob_b,mob_c,mob_b+mob_c);

  const int gparity = GJP.Gparity();

  //Setup Grid Dirac operator
  typename GridDirac::ImplParams params;
  latg.SetParams(params);

  GridDirac Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c, params);
  Grid::SchurDiagMooeeOperator<GridDirac, GridFermionField> linop(Ddwf);

  VRB.Result("A2AvectorW", fname, "Start computing high modes using Grid.\n");
    
  //Generate the compact random sources for the high modes if not yet set
#ifndef MEMTEST_MODE
  W.setWhRandom();
#endif
  
  //Allocate temp *double precision* storage for fermions
  CPSfermion4D<typename Policies::ComplexTypeD,typename Policies::FermionFieldType::FieldMappingPolicy, typename Policies::FermionFieldType::FieldAllocPolicy> v4dfield(V.getFieldInputParams());
  
  const int glb_ls = GJP.SnodeSites() * GJP.Snodes();

  GridFermionField gtmp(FrbGrid);
  GridFermionField gtmp2(FrbGrid);
  GridFermionField gtmp3(FrbGrid);

  GridFermionField gsrc(FGrid);
  GridFermionField gtmp_full(FGrid);
  GridFermionField gtmp_full2(FGrid);

  GridFermionField tmp_full_4d(UGrid);

  size_t nh = W.getNh();
  size_t nl = W.getNl();
  size_t nhits = W.getNhits();

  //Details of this process can be found in Daiqian's thesis, page 60
#ifndef MEMTEST_MODE
  for(size_t i=0; i<nh; i++){
    //Step 1) Get the diluted W vector to invert upon
    W.getDilutedSource(v4dfield, i);

    //Step 2) Solve V
    v4dfield.exportGridField(tmp_full_4d);
    DomainWallFourToFive(gsrc, tmp_full_4d, 0, glb_ls-1);

    //Left-multiply by D-.  D- = (1-c*DW)
    Ddwf.DW(gsrc, gtmp_full, Grid::DaggerNo);
    axpy(gsrc, -mob_c, gtmp_full, gsrc); 

    //We can re-use previously computed solutions to speed up the calculation if rerunning for a second mass by using them as a guess
    //If no previously computed solutions this wastes a few flops, but not enough to care about
    //V vectors default to zero, so this is a zero guess if not reusing existing solutions
    V.getVh(i).exportGridField(tmp_full_4d);
    DomainWallFourToFive(gtmp_full, tmp_full_4d, 0, glb_ls-1);

    Ddwf.DW(gtmp_full, gtmp_full2, Grid::DaggerNo);
    axpy(gtmp_full, -mob_c, gtmp_full2, gtmp_full); 

    //Do the CG
    Grid_CGNE_M_high<Policies>(gtmp_full, gsrc, cg_controls, evecs, nl, latg, Ddwf, FGrid, FrbGrid);
    
    //CPSify the solution, including 1/nhit for the hit average
    DomainWallFiveToFour(tmp_full_4d, gtmp_full, glb_ls-1,0);
    tmp_full_4d = Grid::RealD(1. / nhits) * tmp_full_4d;
    V.getVh(i).importGridField(tmp_full_4d);
  }
#endif
}




//Compute the high mode parts of V and W using MADWF. "Single" means this version is designed for single-RHS inverters
//It is expected that the eigenvectors are for the inner Mobius operator
template< typename Policies>
void computeVWhighSingleMADWF(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass, const CGcontrols &cg_controls){
  assert(cg_controls.CGalgorithm == AlgorithmMixedPrecisionMADWF);

  typedef typename Policies::GridFermionField GridFermionField;
  typedef typename Policies::FgridFclass FgridFclass;
  typedef typename Policies::GridDirac GridDiracOuter;
  typedef typename Policies::GridDiracZMobius GridDiracInner;

  const char *fname = "computeVWhighSingleMADWF(....)";

  int ngp = 0;
  for(int i=0;i<3;i++) if(GJP.Bc(i) == BND_CND_GPARITY) ++ngp;

  if(Policies::GPARITY == 1 && ngp == 0) ERR.General("A2AvectorW","computeVWhighSingle","A2Apolicy is for G-parity\n");
  if(Policies::GPARITY == 0 && ngp != 0) ERR.General("A2AvectorW","computeVWhighSingle","A2Apolicy is not for G-parity\n");

  assert(lat.Fclass() == Policies::FGRID_CLASS_NAME);
  FgridFclass &latg = dynamic_cast<FgridFclass&>(lat);

  //Grids and gauge field
  Grid::GridCartesian *UGrid = latg.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = latg.getUrbGrid();
  Grid::GridCartesian *FGrid = latg.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = latg.getFrbGrid();
  Grid::LatticeGaugeFieldD *Umu = latg.getUmu();
  
  //Mobius parameters
  const double b_outer = latg.get_mob_b();
  const double c_outer = b_outer - 1.;   //b-c = 1
  const int Ls_outer = GJP.Snodes()*GJP.SnodeSites();
  const double M5 = GJP.DwfHeight();
  if(!UniqueID()) printf("computeVWhighSingleMADWF outer Dirac op b=%g c=%g b+c=%g Ls=%d\n",b_outer,c_outer,b_outer+c_outer,GJP.SnodeSites()*GJP.Snodes());

  const int gparity = GJP.Gparity();

  //Setup Dirac operator for outer solver
  typename GridDiracOuter::ImplParams params;
  latg.SetParams(params);

  GridDiracOuter DopOuter(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b_outer,c_outer, params);
  Grid::SchurDiagMooeeOperator<GridDiracOuter, GridFermionField> linopOuter(DopOuter);

  //Setup Dirac operator for inner solver
  int Ls_inner = cg_controls.madwf_params.Ls_inner;
  Grid::GridCartesian * FGrid_inner = Grid::SpaceTimeGrid::makeFiveDimGrid(Ls_inner,UGrid);
  Grid::GridRedBlackCartesian * FrbGrid_inner = Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls_inner,UGrid);

  std::vector<Grid::ComplexD> gamma_inner = getZMobiusGamma(b_outer+c_outer, Ls_outer, cg_controls.madwf_params);

  double bmc_inner = 1.0;//Shamir kernel
  double bpc_inner = cg_controls.madwf_params.b_plus_c_inner;
  double b_inner = 0.5*(bpc_inner + bmc_inner);
  double c_inner = 0.5*(bpc_inner - bmc_inner);

  if(!UniqueID()) printf("computeVWhighSingleMADWF double-precision inner Dirac op b=%g c=%g b+c=%g Ls=%d\n",b_inner,c_inner, bpc_inner, Ls_inner);

  GridDiracInner DopInner(*Umu, *FGrid_inner, *FrbGrid_inner, *UGrid, *UrbGrid, mass, M5, gamma_inner, b_inner, c_inner, params);

  VRB.Result("A2AvectorW", fname, "Start computing high modes using Grid.\n");
    
  //Generate the compact random sources for the high modes if not yet set
#ifndef MEMTEST_MODE
  W.setWhRandom();
#endif
  
  //Allocate temp *double precision* storage for fermions
  CPSfermion4D<typename Policies::ComplexTypeD,typename Policies::FermionFieldType::FieldMappingPolicy, typename Policies::FermionFieldType::FieldAllocPolicy> v4dfield(V.getFieldInputParams());
  
  const int glb_ls = GJP.SnodeSites() * GJP.Snodes();

  GridFermionField grid_src(UGrid);
  GridFermionField grid_sol(UGrid);

  //cf https://rbc.phys.columbia.edu/rbc_ukqcd/individual_postings/ckelly/Gparity/note_a2a_v4.pdf  section D
  size_t nh = W.getNh();
  size_t nl = W.getNl();
  size_t nhits = W.getNhits();

#ifndef MEMTEST_MODE
  for(size_t i=0; i<nh; i++){
    //Step 1) Get the diluted W vector to invert upon
    W.getDilutedSource(v4dfield, i);

    //Step 2) Export to Grid field
    v4dfield.exportGridField(grid_src);

    //Step 3) Perform deflated inversion from *4D->4D*
    grid_sol = Grid::Zero();

    evecs.CGNE_MdagM(linopOuter, grid_sol, grid_src, cg_controls);

    //TESTING
    GridFermionField sol_4d_e(UrbGrid);
    GridFermionField sol_4d_o(UrbGrid);
    pickCheckerboard(Odd, sol_4d_o, grid_sol);
    pickCheckerboard(Even, sol_4d_e, grid_sol);
    std::cout << "4D solution odd:" << norm2(sol_4d_o) << " even:" << norm2(sol_4d_e) << std::endl;
    //TESTING


    //Step 4) Subtract low-mode part in *4D* space
    GridFermionField lowmode_contrib(UGrid);
    computeMADWF_lowmode_contrib_4D(lowmode_contrib, grid_src, nl, evecs, DopInner, cg_controls.madwf_params.precond);
    std::cout << "4D lowmode contribution " << norm2(lowmode_contrib) << std::endl;
    
    grid_sol = grid_sol - lowmode_contrib;
      
    grid_sol = Grid::RealD(1. / nhits) * grid_sol; //include factor of 1/Nhits into V such that it doesn't need to be explicitly included

    V.getVh(i).importGridField(grid_sol);
  }
#endif

  delete FGrid_inner;
  delete FrbGrid_inner;
}







//Version of the above that calls into CG with multiple src/solution pairs so we can use split-CG or multi-RHS CG
template< typename Policies>
void computeVWhighMulti(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass, const CGcontrols &cg_controls){
  assert(isMultiCG(cg_controls.CGalgorithm));
  
  size_t nh = W.getNh();
  size_t nl = W.getNl();
  size_t nhits = W.getNhits();

  typedef typename Policies::GridFermionField GridFermionField;
  typedef typename Policies::FgridFclass FgridFclass;
  typedef typename Policies::GridDirac GridDirac;
  
  const char *fname = "computeVWhighMulti(....)";

  int ngp = 0;
  for(int i=0;i<3;i++) if(GJP.Bc(i) == BND_CND_GPARITY) ++ngp;

  if(Policies::GPARITY == 1 && ngp == 0) ERR.General("A2AvectorW","computeVWhighMulti","A2Apolicy is for G-parity\n");
  if(Policies::GPARITY == 0 && ngp != 0) ERR.General("A2AvectorW","computeVWhighMulti","A2Apolicy is not for G-parity\n");

  assert(lat.Fclass() == Policies::FGRID_CLASS_NAME);
  FgridFclass &latg = dynamic_cast<FgridFclass&>(lat);

  //Grids and gauge field
  Grid::GridCartesian *UGrid = latg.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = latg.getUrbGrid();
  Grid::GridCartesian *FGrid = latg.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = latg.getFrbGrid();
  Grid::LatticeGaugeFieldD *Umu = latg.getUmu();
  
  //Mobius parameters
  const double mob_b = latg.get_mob_b();
  const double mob_c = mob_b - 1.;   //b-c = 1
  const double M5 = GJP.DwfHeight();
  printf("Grid b=%g c=%g b+c=%g\n",mob_b,mob_c,mob_b+mob_c);

  const int gparity = GJP.Gparity();

  //Setup Grid Dirac operator
  typename GridDirac::ImplParams params;
  latg.SetParams(params);

  GridDirac Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c, params);
  Grid::SchurDiagMooeeOperator<GridDirac, GridFermionField> linop(Ddwf);

  VRB.Result("A2AvectorW", fname, "Start computing high modes using Grid multi-RHS solver type.\n");
    
  //Generate the compact random sources for the high modes if not yet set
#ifndef MEMTEST_MODE
  W.setWhRandom();
#endif
  
  //Allocate temp *double precision* storage for fermions
  CPSfermion4D<typename Policies::ComplexTypeD,typename Policies::FermionFieldType::FieldMappingPolicy, typename Policies::FermionFieldType::FieldAllocPolicy> v4dfield(V.getFieldInputParams());
  
  const int glb_ls = GJP.SnodeSites() * GJP.Snodes();

  GridFermionField gtmp(FrbGrid);
  GridFermionField gtmp2(FrbGrid);
  GridFermionField gtmp3(FrbGrid);

  GridFermionField gtmp_full2(FGrid);

  GridFermionField tmp_full_4d(UGrid);

  if(nh % cg_controls.multiCG_block_size != 0)
    ERR.General("A2AvectorW",fname,"Block size %d must be a divisor of the number of high modes %d\n",cg_controls.multiCG_block_size, nh);
  
  const int nblocks = nh / cg_controls.multiCG_block_size;

  std::vector<GridFermionField> gsrc(cg_controls.multiCG_block_size, GridFermionField(FGrid));
  std::vector<GridFermionField> gtmp_full(cg_controls.multiCG_block_size, GridFermionField(FGrid));
    
  //Details of this process can be found in Daiqian's thesis, page 60
#ifndef MEMTEST_MODE
  for(int b=0;b<nblocks;b++){
    for(int s=0;s<cg_controls.multiCG_block_size;s++){
      int i = s + b*cg_controls.multiCG_block_size;
      
      //Step 1) Get the diluted W vector to invert upon
      W.getDilutedSource(v4dfield, i);

      //Step 2) Solve V
      v4dfield.exportGridField(tmp_full_4d);
      DomainWallFourToFive(gsrc[s], tmp_full_4d, 0, glb_ls-1);

      //Left-multiply by D-.  D- = (1-c*DW)
      Ddwf.DW(gsrc[s], gtmp_full[s], Grid::DaggerNo);
      axpy(gsrc[s], -mob_c, gtmp_full[s], gsrc[s]); 
      
      //We can re-use previously computed solutions to speed up the calculation if rerunning for a second mass by using them as a guess
      //If no previously computed solutions this wastes a few flops, but not enough to care about
      //V vectors default to zero, so this is a zero guess if not reusing existing solutions
      V.getVh(i).exportGridField(tmp_full_4d);
      DomainWallFourToFive(gtmp_full[s], tmp_full_4d, 0, glb_ls-1);

      Ddwf.DW(gtmp_full[s], gtmp_full2, Grid::DaggerNo);
      axpy(gtmp_full[s], -mob_c, gtmp_full2, gtmp_full[s]); 
    }
      
    //Do the CG
    Grid_CGNE_M_high_multi<Policies>(gtmp_full, gsrc, cg_controls, evecs, nl, latg, Ddwf, FGrid, FrbGrid);

    for(int s=0;s<cg_controls.multiCG_block_size;s++){
      int i = s + b*cg_controls.multiCG_block_size;
      
      //CPSify the solution, including 1/nhit for the hit average
      DomainWallFiveToFour(tmp_full_4d, gtmp_full[s], glb_ls-1,0);
      tmp_full_4d = Grid::RealD(1. / nhits) * tmp_full_4d;
      V.getVh(i).importGridField(tmp_full_4d);
    }
  }
#endif
}



template< typename Policies>
void computeVWhigh(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass, const CGcontrols &cg_controls){
  if(isMultiCG(cg_controls.CGalgorithm)){
    return computeVWhighMulti(V,W,lat,evecs,mass,cg_controls);
  }else if(cg_controls.CGalgorithm == AlgorithmMixedPrecisionMADWF){
    return computeVWhighSingleMADWF(V,W,lat,evecs,mass,cg_controls);
  }else{
    return computeVWhighSingle(V,W,lat,evecs,mass,cg_controls);
  }
}

#ifdef USE_GRID_LANCZOS
template< typename Policies>
void computeVWhigh(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, const std::vector<typename Policies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls){
  if(!UniqueID()) printf("computeVWhigh with EvecInterfaceGrid\n");
  EvecInterfaceGrid<Policies> ev(evec,eval);
  return computeVWhigh(V,W,lat,ev,mass,cg_controls);
}
template< typename Policies>
void computeVWhigh(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, const std::vector<typename Policies::GridFermionFieldF> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls){
  if(!UniqueID()) printf("computeVWhigh with EvecInterfaceGridSinglePrec\n");
  EvecInterfaceGridSinglePrec<Policies> ev(evec,eval,lat,mass,cg_controls);
  return computeVWhigh(V,W,lat,ev,mass,cg_controls);
}
#endif

#ifdef USE_BFM_LANCZOS
template< typename Policies>
void computeVWhigh(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, BFM_Krylov::Lanczos_5d<double> &eig, bool singleprec_evecs, Lattice &lat, const CGcontrols &cg_controls, bfm_evo<double> &dwf_d, bfm_evo<float> *dwf_fp){
  bool mixed_prec_cg = dwf_fp != NULL; 
  if(mixed_prec_cg){
    //NOT IMPLEMENTED YET
    ERR.General("A2AvectorW","computeVWhigh","No grid implementation of mixed precision CG with BFM evecs\n");
  }

  if(mixed_prec_cg && !singleprec_evecs){ ERR.General("A2AvectorW","computeVWhigh","If using mixed precision CG, input eigenvectors must be stored in single precision"); }

  EvecInterfaceBFM<Policies> ev(eig,dwf_d,lat,singleprec_evecs);
  return computeVWhigh(V,W,lat,ev,dwf_d.mass,cg_controls);
}
#endif
