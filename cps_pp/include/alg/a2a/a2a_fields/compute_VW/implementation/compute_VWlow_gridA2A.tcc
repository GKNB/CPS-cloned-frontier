//Implementation of VW low where the Dirac operator is the same as that used for the gauge fields (and FGrid)
template< typename Policies>
void computeVWlowStandard(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass, const CGcontrols &cg_controls){
  if(!UniqueID()) printf("Computing VWlow using Grid\n");
  typedef typename Policies::GridFermionField GridFermionField;
  typedef typename Policies::FgridFclass FgridFclass;
  typedef typename Policies::GridDirac GridDirac;
  
  size_t nl = W.getNl();
  if(evecs.nEvecs() < nl) 
    ERR.General("A2AvectorW","computeVWlow","Number of low modes %d is larger than the number of provided eigenvectors %d\n",nl,evecs.nEvecs());

  int ngp = 0;
  for(int i=0;i<3;i++) if(GJP.Bc(i) == BND_CND_GPARITY) ++ngp;

  if(Policies::GPARITY == 1 && ngp == 0) ERR.General("","computeVWlowStandard","A2Apolicy is for G-parity\n");
  if(Policies::GPARITY == 0 && ngp != 0) ERR.General("","computeVWlowStandard","A2Apolicy is not for G-parity\n");

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
  const double mob_c = latg.get_mob_c();
  const double M5 = GJP.DwfHeight();
  printf("Grid b=%g c=%g b+c=%g\n",mob_b,mob_c,mob_b+mob_c);

  const int gparity = GJP.Gparity();

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
  for(size_t i = 0; i < nl; i++) {
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
    W.getWl(i).importGridField(tmp_full_4d);
  }
#endif
}



//When using MADWF the Dirac operator used for the eigenvectors is different from the Dirac operator associated with the gauge fields (and with FGrid)
//A different preconditioning scheme is also typically used
//We therefore use a different function to compute the low modes
template< typename Policies>
void computeVWlowMADWF(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass, const CGcontrols &cg_controls){
  if(W.getNl() == 0) return;

  if(!UniqueID()) printf("Computing VWlow using Grid with (Z)Mobius Dirac operator of different Ls\n");
  typedef typename Policies::GridFermionField GridFermionField;
  typedef typename Policies::FgridFclass FgridFclass;
  typedef typename Policies::GridDiracZMobius GridDirac;
  
  size_t nl = W.getNl();
  if(evecs.nEvecs() < nl) 
    ERR.General("A2AvectorW","computeVWlow","Number of low modes %d is larger than the number of provided eigenvectors %d\n",nl,evecs.nEvecs());

  int ngp = 0;
  for(int i=0;i<3;i++) if(GJP.Bc(i) == BND_CND_GPARITY) ++ngp;

  if(Policies::GPARITY == 1 && ngp == 0) ERR.General("","computeVWlowMADWF","A2Apolicy is for G-parity\n");
  if(Policies::GPARITY == 0 && ngp != 0) ERR.General("","computeVWlowMADWF","A2Apolicy is not for G-parity\n");

  assert(lat.Fclass() == Policies::FGRID_CLASS_NAME);
  FgridFclass &latg = dynamic_cast<FgridFclass&>(lat);

  //Grids and gauge field
  int Ls = cg_controls.madwf_params.Ls_inner;
  Grid::GridCartesian *UGrid = latg.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = latg.getUrbGrid();
  Grid::LatticeGaugeFieldD *Umu = latg.getUmu();
  Grid::GridCartesian * FGrid = Grid::SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  Grid::GridRedBlackCartesian * FrbGrid;
  if(evecs.evecPrecision() == 1){ //Make a double precision rb Grid
    FrbGrid = Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);
  }else{ //have to use the native GridCartesian instance
    FrbGrid = (Grid::GridRedBlackCartesian*)evecs.getEvecGrid();
  }

  const int gparity = GJP.Gparity();

  //Setup Grid Dirac operator
  typename GridDirac::ImplParams params;
  latg.SetParams(params);

  
  double mob_b_outer = latg.get_mob_b();
  double mob_c_outer = mob_b_outer - 1. ; //b-c=1
  int Ls_outer = GJP.SnodeSites()*GJP.Snodes();

  //Setup (Z)Mobius inner Dirac operator
  std::vector<Grid::ComplexD> gamma_inner = getZMobiusGamma(mob_b_outer+mob_c_outer, Ls_outer, cg_controls.madwf_params);

  double bmc = 1.0;//Shamir kernel
  double bpc = cg_controls.madwf_params.b_plus_c_inner;
  double mob_b = 0.5*(bpc + bmc);
  double mob_c = 0.5*(bpc - bmc);
  const double M5 = GJP.DwfHeight();
  if(!UniqueID()) printf("computeVWlowMADWF double-precision (Z)Mobius Dirac op b=%g c=%g b+c=%g Ls=%d\n",mob_b,mob_c, bpc, Ls);

  GridDirac DZmob(*Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass, M5, gamma_inner, mob_b, mob_c, params);
  A2Apreconditioning precond = cg_controls.madwf_params.precond;
  Grid::SchurOperatorBase<GridFermionField> *linop;
  if(precond == SchurOriginal) linop = new Grid::SchurDiagMooeeOperator<GridDirac,GridFermionField>(DZmob);
  else                         linop = new Grid::SchurDiagTwoOperator<GridDirac,GridFermionField>(DZmob);

  //Eigenvectors exist on odd checkerboard
  GridFermionField bq_tmp(FrbGrid);
  GridFermionField tmp(FrbGrid);
  GridFermionField tmp2(FrbGrid);
  GridFermionField tmp3(FrbGrid);

  GridFermionField tmp_full(FGrid);
  GridFermionField tmp_full2(FGrid);

  GridFermionField tmp_full_4d(UGrid);

  //See section 1.3 of 
  //https://rbc.phys.columbia.edu/rbc_ukqcd/individual_postings/ckelly/Gparity/note_a2a_v5.pdf
#ifndef MEMTEST_MODE
  for(size_t i = 0; i < nl; i++) {
    //Step 1) Compute V
    Float eval = evecs.getEvec(bq_tmp,i);
    assert(bq_tmp.Checkerboard() == Grid::Odd);

    //Only difference between SchurOriginal and SchurDiagTwo for Vl is the multiplication of the evec by M_oo^{-1}
    GridFermionField *which = &bq_tmp;
    if(precond == SchurDiagTwo){
      DZmob.MooeeInv(bq_tmp,tmp3);
      which = &tmp3;
    }

    //Compute even part [ -(Mee)^-1 Meo bq_tmp, bg_tmp ]
    DZmob.Meooe(*which,tmp2);	//tmp2 = Meo bq_tmp 
    DZmob.MooeeInv(tmp2,tmp);   //tmp = (Mee)^-1 Meo bq_tmp
    tmp = -tmp; //even checkerboard
    
    assert(tmp.Checkerboard() == Grid::Even);
    
    setCheckerboard(tmp_full, tmp); //even checkerboard
    setCheckerboard(tmp_full, *which); //odd checkerboard

    //Get 4D part and poke into a
    //Recall that D^{-1} = <v w^\dagger> = <q \bar q>.  v therefore transforms like a spinor. For spinors \psi(x) = P_R \psi(x,Ls-1) + P_L \psi(x,0),  i.e. s_u=Ls-1 and s_l=0 for CPS gamma5

    DomainWallFiveToFour(tmp_full_4d, tmp_full, Ls-1,0);
    tmp_full_4d = Grid::RealD(1./eval) * tmp_full_4d;
    V.getVl(i).importGridField(tmp_full_4d); //Multiply by 1/lambda[i] and copy into v (with precision change if necessary)
    
    
    //Step 2) Compute Wl. As the Dirac preconditioned Dirac operators is abstracted out, there are no differences between the two preconditioning schemes here

    //Do tmp = [ -[Mee^-1]^dag [Meo]^dag Doo bq_tmp,  Doo bq_tmp ]    (Note that for the Moe^dag in Daiqian's thesis, the dagger also implies a transpose of the spatial indices, hence the Meo^dag in the code)
    linop->Mpc(bq_tmp,tmp2);  //tmp2 = Doo bq_tmp
    
    DZmob.MeooeDag(tmp2,tmp3); //tmp3 = Meo^dag Doo bq_tmp
    DZmob.MooeeInvDag(tmp3,tmp); //tmp = [Mee^-1]^dag Meo^dag Doo bq_tmp
    tmp = -tmp;
    
    assert(tmp.Checkerboard() == Grid::Even);
    assert(tmp2.Checkerboard() == Grid::Odd);

    setCheckerboard(tmp_full, tmp);
    setCheckerboard(tmp_full, tmp2);

    //Left-multiply by D-^dag.  D- = (1-c*DW)
    DZmob.DW(tmp_full, tmp_full2, 1);
    axpy(tmp_full, -mob_c, tmp_full2, tmp_full); 

    //Get 4D part, poke onto a then copy into wl
    //Recall that D^{-1} = <v w^\dagger> = <q \bar q>.  w (and w^\dagger) therefore transforms like a conjugate spinor. For spinors \bar\psi(x) =  \bar\psi(x,0) P_R +  \bar\psi(x,Ls-1) P_L,  i.e. s_u=0 and s_l=Ls-1 for CPS gamma5
    DomainWallFiveToFour(tmp_full_4d, tmp_full, 0, Ls-1);
    W.getWl(i).importGridField(tmp_full_4d);
  }
#endif

  delete linop;
}



template< typename Policies>
void computeVWlow(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, EvecInterface<Policies> &evecs, const Float mass, const CGcontrols &cg_controls){
  if(cg_controls.CGalgorithm == AlgorithmMixedPrecisionMADWF){
    computeVWlowMADWF(V,W,lat,evecs,mass,cg_controls);
  }else{
    computeVWlowStandard(V,W,lat,evecs,mass,cg_controls);
  }
}


#ifdef USE_GRID_LANCZOS
template< typename Policies>
void computeVWlow(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, const std::vector<typename Policies::GridFermionField> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls){
  EvecInterfaceGrid<Policies> ev(evec,eval);
  return computeVWlow(V,W,lat,ev,mass,cg_controls);
}
template< typename Policies>
void computeVWlow(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, const std::vector<typename Policies::GridFermionFieldF> &evec, const std::vector<Grid::RealD> &eval, const double mass, const CGcontrols &cg_controls){
  EvecInterfaceGridSinglePrec<Policies> ev(evec,eval,lat,mass);
  return computeVWlow(V,W,lat,ev,mass,cg_controls);
}
#endif


#ifdef USE_BFM_LANCZOS
//Compute the low mode part of the W and V vectors. In the Lanczos class you can choose to store the vectors in single precision (despite the overall precision, which is fixed to double here)
//Set 'singleprec_evecs' if this has been done
template< typename Policies>
void computeVWlow(A2AvectorV<Policies> &V, A2AvectorW<Policies> &W, Lattice &lat, BFM_Krylov::Lanczos_5d<double> &eig, bfm_evo<double> &dwf, bool singleprec_evecs, const CGcontrols &cg_controls){
  EvecInterfaceBFM<Policies> ev(eig,dwf,lat,singleprec_evecs);
  return computeVWlow(V,W,lat,ev,dwf.mass,cg_controls);
}
#endif
