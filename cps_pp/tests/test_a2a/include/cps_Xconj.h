template<typename A2Apolicies>
void testXconjAction(typename A2Apolicies::FgridGFclass &lattice, const double tol){
  using namespace Grid;

  typedef typename A2Apolicies::GridXconjFermionField FermionField1f;
  typedef typename A2Apolicies::GridFermionField FermionField2f;

  typedef typename A2Apolicies::FgridFclass FgridFclass;

  typedef typename A2Apolicies::GridDiracXconj GridDiracXconj;
  typedef typename A2Apolicies::GridDirac GridDiracGP;

  if(!lattice.getGridFullyInitted()) ERR.General("","gridLanczosXconj","Grid/Grids are not initialized!");
  
  Grid::GridCartesian *UGrid = lattice.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();
  Grid::GridCartesian *FGrid = lattice.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();
  Grid::LatticeGaugeFieldD *Umu = lattice.getUmu();

  double b = lattice.get_mob_b();
  double c = b - 1.;   //b-c = 1
  double M5 = GJP.DwfHeight();

  //double bpc=2.0;
  //double bmc=1.0;
  //double b=0.5*(bpc+bmc);
  // double c=bpc-b;
  //double M5=1.8;
  double mass = 0.01;

  // LatticeGaugeFieldD Umu(UGrid);
  // GridParallelRNG RNG4(UGrid);  
  // RNG4.SeedFixedIntegers({1,2,3,4});
  // gaussian(RNG4,Umu);

  typename GridDiracGP::ImplParams params_gp;
  //params_gp.twists = Coordinate({1,1,1,0});
  lattice.SetParams(params_gp);

  typename GridDiracXconj::ImplParams params_x;
  params_x.twists = params_gp.twists;
  params_x.boundary_phase = 1.0;

  GridDiracXconj actionX(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c, params_x);
  GridDiracGP actionGP(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c, params_gp);
  
  GridParallelRNG RNG(FGrid);  
  RNG.SeedFixedIntegers({1,2,3,4});
  FermionField1f rnd_1f(FGrid), tmp1f(FGrid), M1f(FGrid);
  gaussian(RNG, rnd_1f);

  FermionField2f rnd_2f(FGrid), M2f(FGrid);
  XconjugateBoost(rnd_2f, rnd_1f);
  
  actionX.M(rnd_1f, M1f);
  actionGP.M(rnd_2f, M2f);

  FermionField1f M2f_f0 = PeekIndex<GparityFlavourIndex>(M2f,0); 
  
  FermionField1f diff = M2f_f0 - M1f;
  std::cout << "GP vs Xconj action test on Xconj src : GP " << norm2(M2f_f0) << " Xconj: " << norm2(M1f) << " diff ( expect 0): " << norm2(diff) << std::endl;
  assert(norm2(diff)/norm2(rnd_1f) < 1e-4);

  //Check 2f result is X-conjugate
  FermionField1f M2f_f1 = PeekIndex<GparityFlavourIndex>(M2f,1); 
  tmp1f = -(Xmatrix()*conjugate(M2f_f0));

  diff = tmp1f - M2f_f1;
  std::cout << "GP action on Xconj src produces X-conj vector (expect 0): " << norm2(diff) << std::endl;
  assert(norm2(diff)/norm2(rnd_1f) < 1e-4);
}
