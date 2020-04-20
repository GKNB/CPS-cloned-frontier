//Apply 1/2(1+-g5) to field. In Grid conventions this just zeroes the lower/upper spin components
template<typename FermionField>
void chiralProject(FermionField &out, const FermionField &in, const char sgn){
  int base; //where to start zeroing
  switch(sgn){
  case '+':
    base = 2;
    break;
  case '-':
    base = 0;
    break;
  default:
    assert(0);
  }
  
  out.Checkerboard() = in.Checkerboard();
  conformable(in,out);

  const int Ns = 4;
  Grid::GridBase *grid=in.Grid();

  //decltype(Grid::peekSpin(static_cast<const Grid::Lattice<typename FermionField::vector_object>&>(in),0)) zero_spn(in._grid);
  decltype(Grid::PeekIndex<SpinIndex>(in,0)) zero_spn(in.Grid());
  Grid::zeroit(zero_spn);

  out = in;
  Grid::PokeIndex<SpinIndex>(out, zero_spn, base);
  Grid::PokeIndex<SpinIndex>(out, zero_spn, base+1);
}

//Convert a 5D field to a 4D field, with the upper 2 spin components taken from s-slice 's_u' and the lower 2 from 's_l'
template<typename FermionField>
void DomainWallFiveToFour(FermionField &out, const FermionField &in, int s_u, int s_l){
  assert(out.Grid()->Nd() == 4 && in.Grid()->Nd() == 5);

  FermionField tmp1_4d(out.Grid());
  FermionField tmp2_4d(out.Grid());
  FermionField tmp3_4d(out.Grid());
  ExtractSlice(tmp1_4d,const_cast<FermionField&>(in),s_u, 0); //Note Grid conventions, s-dimension is index 0!
  chiralProject(tmp2_4d, tmp1_4d, '+'); // 1/2(1+g5)  zeroes lower spin components
  
  ExtractSlice(tmp1_4d,const_cast<FermionField&>(in),s_l, 0); 
  chiralProject(tmp3_4d, tmp1_4d, '-'); // 1/2(1-g5)  zeroes upper spin components

  out = tmp2_4d + tmp3_4d;
}
template<typename FermionField>
void DomainWallFourToFive(FermionField &out, const FermionField &in, int s_u, int s_l){
  assert(out.Grid()->Nd() == 5 && in.Grid()->Nd() == 4);

  zeroit(out);
  FermionField tmp1_4d(in.Grid());
  chiralProject(tmp1_4d, in, '+'); // 1/2(1+g5)  zeroes lower spin components
  InsertSlice(tmp1_4d, out,s_u, 0);

  chiralProject(tmp1_4d, in, '-'); // 1/2(1-g5)  zeroes upper spin components
  InsertSlice(tmp1_4d, out,s_l, 0);
}

//Main implementations with generic interface
template< typename mf_Policies>
void A2AvectorW<mf_Policies>::computeVWlow(A2AvectorV<mf_Policies> &V, Lattice &lat, EvecInterface<mf_Policies> &evecs, const Float mass){
  if(!UniqueID()) printf("Computing VWlow using Grid\n");
  typedef typename mf_Policies::GridFermionField GridFermionField;
  typedef typename mf_Policies::FgridFclass FgridFclass;
  typedef typename mf_Policies::GridDirac GridDirac;
  
  if(evecs.nEvecs() < nl) 
    ERR.General("A2AvectorW","computeVWlow","Number of low modes %d is larger than the number of provided eigenvectors %d\n",nl,evecs.nEvecs());

  int ngp = 0;
  for(int i=0;i<3;i++) if(GJP.Bc(i) == BND_CND_GPARITY) ++ngp;

  if(mf_Policies::GPARITY == 1 && ngp == 0) ERR.General("A2AvectorW","computeVWlow","A2Apolicy is for G-parity\n");
  if(mf_Policies::GPARITY == 0 && ngp != 0) ERR.General("A2AvectorW","computeVWlow","A2Apolicy is not for G-parity\n");

  assert(lat.Fclass() == mf_Policies::FGRID_CLASS_NAME);
  FgridFclass &latg = dynamic_cast<FgridFclass&>(lat);

  //Grids and gauge field
  Grid::GridCartesian *UGrid = latg.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = latg.getUrbGrid();
  Grid::GridCartesian *FGrid = latg.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = latg.getFrbGrid();
  Grid::LatticeGaugeFieldD *Umu = latg.getUmu();
  
  //Mobius parameters
  const double mob_b = latg.get_mob_b();
  const double mob_c = latg.get_mob_c();
  const double M5 = GJP.DwfHeight();
  printf("Grid b=%g c=%g b+c=%g\n",mob_b,mob_c,mob_b+mob_c);

  const int gparity = GJP.Gparity();

  //Double precision temp fields
  CPSfermion4D<ComplexD> afield;
  CPSfermion5D<ComplexD> bfield;

  const int glb_ls = GJP.SnodeSites() * GJP.Snodes();

  //Setup Grid Dirac operator
  typename GridDirac::ImplParams params;
  latg.SetParams(params);

  GridDirac Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c, params);
  Grid::SchurDiagMooeeOperator<GridDirac,GridFermionField> linop(Ddwf);

  //Eigenvectors exist on odd checkerboard
  GridFermionField bq_tmp(FrbGrid);
  GridFermionField tmp(FrbGrid);
  GridFermionField tmp2(FrbGrid);
  GridFermionField tmp3(FrbGrid);

  GridFermionField tmp_full(FGrid);
  GridFermionField tmp_full2(FGrid);

  GridFermionField tmp_full_4d(UGrid);
  
  //The general method is described by page 60 of Daiqian's thesis
#ifndef MEMTEST_MODE
  for(int i = 0; i < nl; i++) {
    //Step 1) Compute V
    Float eval = evecs.getEvec(bq_tmp,i);
    assert(bq_tmp.Checkerboard() == Grid::Odd);

    //Compute  [ -(Mee)^-1 Meo bq_tmp, bg_tmp ]
    Ddwf.Meooe(bq_tmp,tmp2);	//tmp2 = Meo bq_tmp 
    Ddwf.MooeeInv(tmp2,tmp);   //tmp = (Mee)^-1 Meo bq_tmp
    tmp = -tmp; //even checkerboard
    
    assert(tmp.Checkerboard() == Grid::Even);
    
    setCheckerboard(tmp_full, tmp); //even checkerboard
    setCheckerboard(tmp_full, bq_tmp); //odd checkerboard

    //Get 4D part and poke into a
    //Recall that D^{-1} = <v w^\dagger> = <q \bar q>.  v therefore transforms like a spinor. For spinors \psi(x) = P_R \psi(x,Ls-1) + P_L \psi(x,0),  i.e. s_u=Ls-1 and s_l=0 for CPS gamma5

    DomainWallFiveToFour(tmp_full_4d, tmp_full, glb_ls-1,0);
    tmp_full_4d = Grid::RealD(1./eval) * tmp_full_4d;
    V.getVl(i).importGridField(tmp_full_4d); //Multiply by 1/lambda[i] and copy into v (with precision change if necessary)
    
    
    //Step 2) Compute Wl

    //Do tmp = [ -[Mee^-1]^dag [Meo]^dag Doo bq_tmp,  Doo bq_tmp ]    (Note that for the Moe^dag in Daiqian's thesis, the dagger also implies a transpose of the spatial indices, hence the Meo^dag in the code)
    linop.Mpc(bq_tmp,tmp2);  //tmp2 = Doo bq_tmp
    
    Ddwf.MeooeDag(tmp2,tmp3); //tmp3 = Meo^dag Doo bq_tmp
    Ddwf.MooeeInvDag(tmp3,tmp); //tmp = [Mee^-1]^dag Meo^dag Doo bq_tmp
    tmp = -tmp;
    
    assert(tmp.Checkerboard() == Grid::Even);
    assert(tmp2.Checkerboard() == Grid::Odd);

    setCheckerboard(tmp_full, tmp);
    setCheckerboard(tmp_full, tmp2);

    //Left-multiply by D-^dag.  D- = (1-c*DW)
    Ddwf.DW(tmp_full, tmp_full2, 1);
    axpy(tmp_full, -mob_c, tmp_full2, tmp_full); 

    //Get 4D part, poke onto a then copy into wl
    //Recall that D^{-1} = <v w^\dagger> = <q \bar q>.  w (and w^\dagger) therefore transforms like a conjugate spinor. For spinors \bar\psi(x) =  \bar\psi(x,0) P_R +  \bar\psi(x,Ls-1) P_L,  i.e. s_u=0 and s_l=Ls-1 for CPS gamma5
    DomainWallFiveToFour(tmp_full_4d, tmp_full, 0, glb_ls-1);
    wl[i]->importGridField(tmp_full_4d);
  }
#endif
}


inline bool isMultiCG(const A2ACGalgorithm al){
  if(al == AlgorithmMixedPrecisionReliableUpdateSplitCG) return true;
  return false;
}




//Compute the high mode parts of V and W.   "Single" means this version is designed for single-RHS inverters
template< typename mf_Policies>
void A2AvectorW<mf_Policies>::computeVWhighSingle(A2AvectorV<mf_Policies> &V, Lattice &lat, EvecInterface<mf_Policies> &evecs, const Float mass, const CGcontrols &cg_controls){
  assert(!isMultiCG(cg_controls.CGalgorithm));
  assert(cg_controls.CGalgorithm != AlgorithmMixedPrecisionMADWF);

  typedef typename mf_Policies::GridFermionField GridFermionField;
  typedef typename mf_Policies::FgridFclass FgridFclass;
  typedef typename mf_Policies::GridDirac GridDirac;
  
  const char *fname = "computeVWhighSingle(....)";

  int ngp = 0;
  for(int i=0;i<3;i++) if(GJP.Bc(i) == BND_CND_GPARITY) ++ngp;

  if(mf_Policies::GPARITY == 1 && ngp == 0) ERR.General("A2AvectorW","computeVWhighSingle","A2Apolicy is for G-parity\n");
  if(mf_Policies::GPARITY == 0 && ngp != 0) ERR.General("A2AvectorW","computeVWhighSingle","A2Apolicy is not for G-parity\n");

  assert(lat.Fclass() == mf_Policies::FGRID_CLASS_NAME);
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
  setWhRandom();
#endif
  
  //Allocate temp *double precision* storage for fermions
  CPSfermion4D<typename mf_Policies::ComplexTypeD,typename mf_Policies::FermionFieldType::FieldMappingPolicy, typename mf_Policies::FermionFieldType::FieldAllocPolicy> v4dfield(V.getFieldInputParams());
  
  const int glb_ls = GJP.SnodeSites() * GJP.Snodes();

  GridFermionField gtmp(FrbGrid);
  GridFermionField gtmp2(FrbGrid);
  GridFermionField gtmp3(FrbGrid);

  GridFermionField gsrc(FGrid);
  GridFermionField gtmp_full(FGrid);
  GridFermionField gtmp_full2(FGrid);

  GridFermionField tmp_full_4d(UGrid);

  //Details of this process can be found in Daiqian's thesis, page 60
#ifndef MEMTEST_MODE
  for(int i=0; i<nh; i++){
    //Step 1) Get the diluted W vector to invert upon
    getDilutedSource(v4dfield, i);

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
    Grid_CGNE_M_high<mf_Policies>(gtmp_full, gsrc, cg_controls, evecs, nl, latg, Ddwf, FGrid, FrbGrid);
    
    //CPSify the solution, including 1/nhit for the hit average
    DomainWallFiveToFour(tmp_full_4d, gtmp_full, glb_ls-1,0);
    tmp_full_4d = Grid::RealD(1. / nhits) * tmp_full_4d;
    V.getVh(i).importGridField(tmp_full_4d);
  }
#endif
}




//Compute the high mode parts of V and W using MADWF. "Single" means this version is designed for single-RHS inverters
//It is expected that the eigenvectors are for the inner Mobius operator
template< typename mf_Policies>
void A2AvectorW<mf_Policies>::computeVWhighSingleMADWF(A2AvectorV<mf_Policies> &V, Lattice &lat, EvecInterface<mf_Policies> &evecs, const Float mass, const CGcontrols &cg_controls){
  assert(cg_controls.CGalgorithm == AlgorithmMixedPrecisionMADWF);

  typedef typename mf_Policies::GridFermionField GridFermionField;
  typedef typename mf_Policies::FgridFclass FgridFclass;
  typedef typename mf_Policies::GridDirac GridDiracOuter;
  typedef typename mf_Policies::GridDiracZMobius GridDiracInner;

  const char *fname = "computeVWhighSingleMADWF(....)";

  int ngp = 0;
  for(int i=0;i<3;i++) if(GJP.Bc(i) == BND_CND_GPARITY) ++ngp;

  if(mf_Policies::GPARITY == 1 && ngp == 0) ERR.General("A2AvectorW","computeVWhighSingle","A2Apolicy is for G-parity\n");
  if(mf_Policies::GPARITY == 0 && ngp != 0) ERR.General("A2AvectorW","computeVWhighSingle","A2Apolicy is not for G-parity\n");

  assert(lat.Fclass() == mf_Policies::FGRID_CLASS_NAME);
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
  if(!UniqueID()) printf("computeVWhighSingleMADWF outer Dirac op b=%g c=%g b+c=%g Ls=%d\n",mob_b,mob_c,mob_b+mob_c,GJP.SnodeSites()*GJP.Snodes());

  const int gparity = GJP.Gparity();

  //Setup Dirac operator for outer solver
  typename GridDiracOuter::ImplParams params;
  latg.SetParams(params);

  GridDiracOuter DopOuter(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c, params);
  Grid::SchurDiagMooeeOperator<GridDiracOuter, GridFermionField> linopOuter(DopOuter);

  //Setup Dirac operator for inner solver
  Grid::GridCartesian * FGrid_inner = Grid::SpaceTimeGrid::makeFiveDimGrid(cg_controls.MADWF_Ls_inner,UGrid);
  Grid::GridRedBlackCartesian * FrbGrid_inner = Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid(cg_controls.MADWF_Ls_inner,UGrid);

  std::vector<ComplexD> gamma_inner = computeZmobiusGammaWithCache(cg_controls.MADWF_b_plus_c_inner, 
								   cg_controls.MADWF_Ls_inner, 
								   mob_b+mob_c, GJP.SnodeSites()*GJP.Snodes(),
								   cg_controls.MADWF_ZMobius_lambda_max, cg_controls.MADWF_use_ZMobius);
  double bmc = 1.0;//Shamir kernel
  double bpc = cg_controls.MADWF_b_plus_c_inner;
  double b_inner = 0.5*(bpc + bmc);
  double c_inner = 0.5*(bpc - bmc);

  if(!UniqueID()) printf("computeVWhighSingleMADWF double-precision inner Dirac op b=%g c=%g b+c=%g Ls=%d\n",b_inner,c_inner, bpc,cg_controls.MADWF_Ls_inner);

  GridDiracInner DopInner(*Umu, *FGrid_inner, *FrbGrid_inner, *UGrid, *UrbGrid, mass, M5, gamma_inner, b_inner, c_inner, params);

  VRB.Result("A2AvectorW", fname, "Start computing high modes using Grid.\n");
    
  //Generate the compact random sources for the high modes if not yet set
#ifndef MEMTEST_MODE
  setWhRandom();
#endif
  
  //Allocate temp *double precision* storage for fermions
  CPSfermion4D<typename mf_Policies::ComplexTypeD,typename mf_Policies::FermionFieldType::FieldMappingPolicy, typename mf_Policies::FermionFieldType::FieldAllocPolicy> v4dfield(V.getFieldInputParams());
  
  const int glb_ls = GJP.SnodeSites() * GJP.Snodes();

  GridFermionField grid_src(UGrid);
  GridFermionField grid_sol(UGrid);

  //cf https://rbc.phys.columbia.edu/rbc_ukqcd/individual_postings/ckelly/Gparity/note_a2a_v4.pdf  section D

#ifndef MEMTEST_MODE
  for(int i=0; i<nh; i++){
    //Step 1) Get the diluted W vector to invert upon
    getDilutedSource(v4dfield, i);

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
    computeMADWF_lowmode_contrib_4D(lowmode_contrib, grid_src, nl, evecs, DopInner, cg_controls.MADWF_precond);
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
template< typename mf_Policies>
void A2AvectorW<mf_Policies>::computeVWhighMulti(A2AvectorV<mf_Policies> &V, Lattice &lat, EvecInterface<mf_Policies> &evecs, const Float mass, const CGcontrols &cg_controls){
  assert(isMultiCG(cg_controls.CGalgorithm));

  typedef typename mf_Policies::GridFermionField GridFermionField;
  typedef typename mf_Policies::FgridFclass FgridFclass;
  typedef typename mf_Policies::GridDirac GridDirac;
  
  const char *fname = "computeVWhighMulti(....)";

  int ngp = 0;
  for(int i=0;i<3;i++) if(GJP.Bc(i) == BND_CND_GPARITY) ++ngp;

  if(mf_Policies::GPARITY == 1 && ngp == 0) ERR.General("A2AvectorW","computeVWhighMulti","A2Apolicy is for G-parity\n");
  if(mf_Policies::GPARITY == 0 && ngp != 0) ERR.General("A2AvectorW","computeVWhighMulti","A2Apolicy is not for G-parity\n");

  assert(lat.Fclass() == mf_Policies::FGRID_CLASS_NAME);
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
  setWhRandom();
#endif
  
  //Allocate temp *double precision* storage for fermions
  CPSfermion4D<typename mf_Policies::ComplexTypeD,typename mf_Policies::FermionFieldType::FieldMappingPolicy, typename mf_Policies::FermionFieldType::FieldAllocPolicy> v4dfield(V.getFieldInputParams());
  
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
      getDilutedSource(v4dfield, i);

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
    Grid_CGNE_M_high_multi<mf_Policies>(gtmp_full, gsrc, cg_controls, evecs, nl, latg, Ddwf, FGrid, FrbGrid);

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



template< typename mf_Policies>
void A2AvectorW<mf_Policies>::computeVWhigh(A2AvectorV<mf_Policies> &V, Lattice &lat, EvecInterface<mf_Policies> &evecs, const Float mass, const CGcontrols &cg_controls){
  if(isMultiCG(cg_controls.CGalgorithm)){
    return computeVWhighMulti(V,lat,evecs,mass,cg_controls);
  }else if(cg_controls.CGalgorithm == AlgorithmMixedPrecisionMADWF){
    return computeVWhighSingleMADWF(V,lat,evecs,mass,cg_controls);
  }else{
    return computeVWhighSingle(V,lat,evecs,mass,cg_controls);
  }
}





//Wrappers for generic interface

//BFM evecs
#ifdef USE_BFM_LANCZOS

//Compute the low mode part of the W and V vectors. In the Lanczos class you can choose to store the vectors in single precision (despite the overall precision, which is fixed to double here)
//Set 'singleprec_evecs' if this has been done
template< typename mf_Policies>
void A2AvectorW<mf_Policies>::computeVWlow(A2AvectorV<mf_Policies> &V, Lattice &lat, BFM_Krylov::Lanczos_5d<double> &eig, bfm_evo<double> &dwf, bool singleprec_evecs){
  EvecInterfaceBFM<mf_Policies> ev(eig,dwf,lat,singleprec_evecs);
  return computeVWlow(V,lat,ev,dwf.mass);
}

template< typename mf_Policies>
void A2AvectorW<mf_Policies>::computeVWhigh(A2AvectorV<mf_Policies> &V, BFM_Krylov::Lanczos_5d<double> &eig, bool singleprec_evecs, Lattice &lat, const CGcontrols &cg_controls, bfm_evo<double> &dwf_d, bfm_evo<float> *dwf_fp){
  bool mixed_prec_cg = dwf_fp != NULL; 
  if(mixed_prec_cg){
    //NOT IMPLEMENTED YET
    ERR.General("A2AvectorW","computeVWhigh","No grid implementation of mixed precision CG with BFM evecs\n");
  }

  if(mixed_prec_cg && !singleprec_evecs){ ERR.General("A2AvectorW","computeVWhigh","If using mixed precision CG, input eigenvectors must be stored in single precision"); }

  EvecInterfaceBFM<mf_Policies> ev(eig,dwf_d,lat,singleprec_evecs);
  return computeVWhigh(V,lat,ev,dwf_d.mass,cg_controls);
}

#endif



//Grid evecs
#ifdef USE_GRID_LANCZOS

template< typename mf_Policies>
void A2AvectorW<mf_Policies>::computeVWlow(A2AvectorV<mf_Policies> &V, Lattice &lat, const std::vector<typename mf_Policies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, const double mass){
  EvecInterfaceGrid<mf_Policies> ev(evec,eval);
  return computeVWlow(V,lat,ev,mass);
}

template< typename mf_Policies>
void A2AvectorW<mf_Policies>::computeVWhigh(A2AvectorV<mf_Policies> &V, Lattice &lat, const std::vector<typename mf_Policies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls){
  if(!UniqueID()) printf("computeVWhigh with EvecInterfaceGrid\n");
  EvecInterfaceGrid<mf_Policies> ev(evec,eval);
  return computeVWhigh(V,lat,ev,mass,cg_controls);
}

template< typename mf_Policies>
void A2AvectorW<mf_Policies>::computeVWlow(A2AvectorV<mf_Policies> &V, Lattice &lat, const std::vector<typename mf_Policies::GridFermionFieldF> &evec, const std::vector<Grid::RealD> &eval, const double mass){
  EvecInterfaceGridSinglePrec<mf_Policies> ev(evec,eval,lat,mass);
  return computeVWlow(V,lat,ev,mass);
}

template< typename mf_Policies>
void A2AvectorW<mf_Policies>::computeVWhigh(A2AvectorV<mf_Policies> &V, Lattice &lat, const std::vector<typename mf_Policies::GridFermionFieldF> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls){
  if(!UniqueID()) printf("computeVWhigh with EvecInterfaceGridSinglePrec\n");
  EvecInterfaceGridSinglePrec<mf_Policies> ev(evec,eval,lat,mass,cg_controls);
  return computeVWhigh(V,lat,ev,mass,cg_controls);
}


#endif
