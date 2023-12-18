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

template<typename GridPropagatorField2f>
void unitGridPropagatorField2f(GridPropagatorField2f &field){
  using namespace Grid;
  typename GridPropagatorField2f::vector_object one;
  one = Zero();
  for(int s=0;s<Ns;s++)
    for(int c=0;c<Nc;c++)
      for(int f=0;f<Ngp;f++)
	one(f,f)(s,s)(c,c) = typename GridPropagatorField2f::vector_type(1.);
  
  field = one;
}

template<typename A2Apolicies>
void testXconjDiagRecon(typename A2Apolicies::FgridGFclass &lattice){
  using namespace Grid;

  typedef typename A2Apolicies::GridXconjFermionField FermionField1f;
  typedef typename A2Apolicies::GridFermionField FermionField2f;

  typedef typename A2Apolicies::FgridFclass FgridFclass;

  typedef typename A2Apolicies::GridDiracXconj GridDiracXconj;
  typedef typename A2Apolicies::GridDirac GridDiracGP;

  typedef iMatrix<iMatrix<iMatrix<vComplexD, Nc>, Ns>, Ngp> vSCFmatrixD;
  typedef iMatrix<iMatrix<iMatrix<Grid::ComplexD, Nc>, Ns>, Ngp> SCFmatrixD;
  typedef Grid::Lattice<vSCFmatrixD> PropagatorField2f;

  if(!lattice.getGridFullyInitted()) ERR.General("","gridLanczosXconj","Grid/Grids are not initialized!");
  
  Grid::GridCartesian *UGrid = lattice.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();
  Grid::GridCartesian *FGrid = lattice.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();
  Grid::LatticeGaugeFieldD *Umu = lattice.getUmu();

  double b = lattice.get_mob_b();
  double c = b - 1.;   //b-c = 1
  double M5 = GJP.DwfHeight();
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

  //GridDiracXconj actionX(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c, params_x);
  GridDiracGP actionGP(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c, params_gp);
  
  Grid::ConjugateGradient<FermionField2f> CG(1e-8,100000);
  Grid::SchurRedBlackDiagTwoSolve<FermionField2f> SchurSolver(CG);

  GridParallelRNG RNG(FGrid);  
  RNG.SeedFixedIntegers({1,2,3,4});

  {
    //Test U U^dag f = U^dag U f = f
    FermionField2f f(FGrid),tmp1(FGrid),tmp2(FGrid),diff(FGrid);
    gaussian(RNG, f);
    
    multUdag(tmp1,f);
    multU(tmp2,tmp1);
    diff = tmp2 - f;
    double nd = norm2(diff);
    std::cout << "Test UUdag f = f (expect 0): " << nd << std::endl;

    multU(tmp1,f);
    multUdag(tmp2,tmp1);
    diff = tmp2 - f;
    nd = norm2(diff);
    std::cout << "Test UdagU f = f (expect 0): " << nd << std::endl;
  }
  {
    //Demonstrate   U^dag M^-1 U f   for real f is pure-real
    FermionField2f f(FGrid),tmp1(FGrid),tmp2(FGrid),tmp3(FGrid);
    gaussian(RNG, f);
    f = real(f);
    multU(tmp1,f);

    tmp2 = Zero();
    SchurSolver(actionGP,tmp1,tmp2); 
    
    multUdag(tmp1,tmp2);
    
    tmp2 = real(tmp1);
    tmp3 = imag(tmp1);
    std::cout << "Test  U M^-1 U^dag f  for real f is real:   real part " << norm2(tmp2) << " imag part " << norm2(tmp3) << std::endl;
  }
  {
    //Test  U U^dag f = f
    PropagatorField2f f(FGrid),tmp2(FGrid),tmp3(FGrid),diff(FGrid);
    gaussian(RNG, f);

    multUleft(tmp2,f);
    multUdagLeft(tmp3,tmp2);
    diff = tmp3 - f;
    double nd = norm2(diff);
    std::cout << "Test UdagU f = f for matrix f (expect 0): " << nd << std::endl;

    multUdagLeft(tmp2,f);
    multUleft(tmp3,tmp2);
    diff = tmp3 - f;
    nd = norm2(diff);
    std::cout << "Test UUdag f = f for matrix f (expect 0): " << nd << std::endl;
  }
  {
    //Test  f U U^dag  = f
    PropagatorField2f f(FGrid),tmp2(FGrid),tmp3(FGrid),diff(FGrid);
    gaussian(RNG, f);

    multUright(tmp2,f);
    multUdagRight(tmp3,tmp2);
    diff = tmp3 - f;
    double nd = norm2(diff);
    std::cout << "Test f UUdag = f for matrix f (expect 0): " << nd << std::endl;

    multUdagRight(tmp2,f);
    multUright(tmp3,tmp2);
    diff = tmp3 - f;
    nd = norm2(diff);
    std::cout << "Test f UdagU= f for matrix f (expect 0): " << nd << std::endl;
  }
  {
    //Test  U f U^dag = f  for unit f
    PropagatorField2f f(FGrid),tmp2(FGrid),tmp3(FGrid),diff(FGrid);
    unitGridPropagatorField2f(f);
  
    multUleft(tmp2,f);
    multUdagRight(tmp3,tmp2);
    diff = tmp3 - f;
    double nd = norm2(diff);
    std::cout << "Test U f dagU = f for unit matrix f (expect 0): " << nd << std::endl;

    multUdagLeft(tmp2,f);
    multUright(tmp3,tmp2);
    diff = tmp3 - f;
    nd = norm2(diff);
    std::cout << "Test Udag f U = f for unit matrix f (expect 0): " << nd << std::endl;
  }

  // {
  //  NEGATIVE:  U does not commute with a general diagonal matrix
  //   //Test  U f U^dag f = f  for diagonal f
  //   PropagatorField2f f(FGrid),tmp2(FGrid),tmp3(FGrid),diff(FGrid);

  //   FermionField2f z(FGrid);
  //   gaussian(RNG, z);
    
  //   {
  //     autoView(f_v,f,AcceleratorWrite);
  //     autoView(z_v,z,AcceleratorRead);
  //     accelerator_for(x, f.Grid()->oSites(), PropagatorField2f::vector_object::Nsimd(), {
  // 	  decltype(f_v(x)) v = Zero();
  // 	  auto zx = z_v(x);
  // 	  for(int f=0;f<Ngp;f++)
  // 	    for(int s=0;s<Ns;s++)
  // 	      for(int c=0;c<Nc;c++)
  // 		v(f,f)(s,s)(c,c) = zx(f)(s)(c);
  // 	  coalescedWrite(f_v[x],v);
  // 	});
  //   }

  //   multUleft(tmp2,f);
  //   multUdagRight(tmp3,tmp2);
  //   diff = tmp3 - f;
  //   double nd = norm2(diff);
  //   std::cout << "Test U f dagU = f for diagonal matrix f (expect 0): " << nd << std::endl;

  //   multUdagLeft(tmp2,f);
  //   multUright(tmp3,tmp2);
  //   diff = tmp3 - f;
  //   nd = norm2(diff);
  //   std::cout << "Test Udag f U = f for diagonal matrix f (expect 0): " << nd << std::endl;
  // }

}
