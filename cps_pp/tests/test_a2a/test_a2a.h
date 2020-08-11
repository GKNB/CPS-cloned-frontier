#ifndef _TEST_A2A_H_
#define _TEST_A2A_H_

void testCPSsquareMatrix(){
  std::cout << "Testing CPSsquareMatrix types" << std::endl; 

  {
    CPSflavorMatrix<cps::Complex> f1, f2, f3;
    f1.unit(); f2.unit();
    f1.pr(sigma1);
    f2.pr(sigma2);

    f3 = f1 * f2;
    std::cout << "sigma1=" << f1 << std::endl;
    std::cout << "sigma2=" << f2 << std::endl;
    std::cout << "sigma1*sigma2 = " << f3 << std::endl;
    CPSflavorMatrix<cps::Complex> f3_expect; 
    f3_expect.unit(); f3_expect.pr(sigma3); f3_expect.timesI();
    assert( f3_expect == f3 );



    CPSspinMatrix<CPSflavorMatrix<cps::Complex> > sf1;
    sf1.unit();
    sf1.gr(-5);
    std::cout << "gamma5(spin,flavor) =\n" << sf1 << std::endl;

    typedef typename CPSspinMatrix<CPSflavorMatrix<cps::Complex> >::scalar_type scalar_type;
    static_assert( _equal<scalar_type, cps::Complex>::value, "scalar_type deduction");
    scalar_type tr = sf1.Trace();

    std::cout << "Trace: ";
    CPSprintT(std::cout, tr);
    std::cout << std::endl;

    assert( tr == scalar_type(0.) );


    cps::Complex dbl_trace = sf1.TraceIndex<0>().TraceIndex<0>();
    std::cout << "Trace(flavor)Trace(spin): ";
    CPSprintT(std::cout, dbl_trace);
    std::cout << std::endl;
    
    assert( tr == cps::Complex(0.) );

    
    std::cout << "Trace product g5*g5=I_8x8: ";
    scalar_type tr_prod = Trace(sf1,sf1);
    CPSprintT(std::cout, tr_prod);
    std::cout << std::endl;

    assert( tr_prod == scalar_type(8.) );

    typedef CPSspinMatrix<CPSflavorMatrix<cps::Complex> > SFmat;
    typedef CPSflavorMatrix<cps::Complex> Fmat;
    typedef CPSspinMatrix<cps::Complex> Smat;
    
    static_assert( _equal< typename _PartialTraceFindReducedType<SFmat,0>::type, Fmat>::value, "Trace reduce 1");
    static_assert( _equal< typename _PartialTraceFindReducedType<SFmat,1>::type, Smat>::value, "Trace reduce 2");

    SFmat sf2;
    sf2.unit();
    Fmat tridx = sf2.TraceIndex<0>();

    std::cout << "Spin trace of spin-flavor unit matrix:\n" << tridx << std::endl;
    
    assert( tridx(0,0) == cps::Complex(4.0) & tridx(1,1) == cps::Complex(4.0) &&
	    tridx(0,1) == cps::Complex(0.0) & tridx(1,0) == cps::Complex(0.0) );


    sf2.unit();
    for(int i=0;i<4;i++)
      for(int j=0;j<4;j++)
	sf2(i,j).pr(sigma3);
    sf2.gr(-5);
    
    std::cout << "Spin-flavor matrix g5*sigma3\n" << sf2 << std::endl;


    typedef CPSspinMatrix<CPSflavorMatrix<Grid::vComplexD> > vSFmat;
    vSFmat vsf;
    vsf.unit();
    std::cout << "Vectorized sf unit matrix\n" << vsf << std::endl;

    static_assert( _equal<typename _PartialTraceFindReducedType<Fmat,0>::type, cps::Complex>::value, "Foutertracetest");
  }

  //Test scalar size deduction
  {
    typedef CPSsquareMatrix<double, 5> M1_t;
    M1_t M1;
    size_t NN_M1 = M1.nScalarType();
    assert(NN_M1 = 25);

    typedef CPSsquareMatrix<CPSsquareMatrix<double,2>, 3> M2_t;
    M2_t M2;
    size_t NN_M2 = M2.nScalarType();
    assert(NN_M2 = 9*4);
  }
#ifdef USE_GRID
  //Test Grid reduction of matrix
  {
    typedef CPSsquareMatrix<Grid::vComplexD,2> m_t;
    m_t m;
    m.unit();
    constexpr size_t nsimd = Grid::vComplexD::Nsimd();
    typedef CPSsquareMatrix<Grid::ComplexD,2> ms_t;
    ms_t m_r = Reduce(m);
    ms_t ms;
    ms.unit();
    ms *= nsimd;
    
    assert( ms == m_r );
  }
#endif

  std::cout << "Passed CPSsquareMatrix tests" << std::endl;
}


template<typename A2Apolicies_grid>
void checkCPSfieldGridImpex5Dcb(typename A2Apolicies_grid::FgridGFclass &lattice){
  std::cout << "Checking CPSfield 5D Grid impex with and without checkerboarding" << std::endl;

  Grid::GridCartesian* grid5d_full = lattice.getFGrid();
  Grid::GridCartesian* grid4d_full = lattice.getUGrid();
  Grid::GridRedBlackCartesian* grid5d_cb = lattice.getFrbGrid();
  Grid::GridRedBlackCartesian* grid4d_cb = lattice.getUrbGrid();
  typedef typename A2Apolicies_grid::GridFermionField GridFermionField;
    
  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  Grid::GridParallelRNG          RNG5(grid5d_full);  RNG5.SeedFixedIntegers(seeds5);
  Grid::GridParallelRNG          RNG4(grid4d_full);  RNG4.SeedFixedIntegers(seeds4);

  {//5D non-cb impex
    GridFermionField fivedin(grid5d_full); random(RNG5,fivedin);

    CPSfermion5D<cps::ComplexD> cpscp1;
    cpscp1.importGridField(fivedin);

    CPSfermion5D<cps::ComplexD> cpscp2;
    lattice.ImportFermion((Vector*)cpscp2.ptr(), fivedin);

    assert(cpscp1.equals(cpscp2));

    double nrm_cps = cpscp1.norm2();
    double nrm_grid = Grid::norm2(fivedin);
      
    std::cout << "5D import pass norms " << nrm_cps << " " << nrm_grid << std::endl;
    
    assert(fabs(nrm_cps - nrm_grid) < 1e-8 );

    GridFermionField fivedout(grid5d_full);
    cpscp1.exportGridField(fivedout);
    double nrm_fivedout = Grid::norm2(fivedout);
    std::cout << "Export to grid: " << nrm_fivedout << std::endl;

    assert( fabs( nrm_fivedout - nrm_cps ) < 1e-8 );
  }
  { //5D checkerboarded impex
    GridFermionField fivedin(grid5d_full); random(RNG5,fivedin);
    GridFermionField fivedcb(grid5d_cb);
    Grid::pickCheckerboard(Grid::Odd, fivedcb, fivedin);

    Grid::Coordinate test_site(5,0);
    test_site[1] = 3;

    typedef typename Grid::GridTypeMapper<typename GridFermionField::vector_object>::scalar_object sobj;
    sobj v1, v2;
    Grid::peekLocalSite(v1,fivedin,test_site);

    Grid::peekLocalSite(v2,fivedcb,test_site);
      
    std::cout << "v1:\n" << v1 << std::endl;
    std::cout << "v2:\n" << v2 << std::endl;
      

    CPSfermion5Dcb4Dodd<cps::ComplexD> cpscp1;
    std::cout << "From Grid CB\n";
    cpscp1.importGridField(fivedcb);

    double nrm_cps = cpscp1.norm2();
    double nrm_grid = Grid::norm2(fivedcb);

    GridFermionField tmp(grid5d_full);
    zeroit(tmp);
    Grid::setCheckerboard(tmp, fivedcb);

    double nrm2_grid = Grid::norm2(tmp);

      
    CPSfermion5Dcb4Dodd<cps::ComplexD> cpscp3;
    std::cout << "From Grid full\n";
    cpscp3.importGridField(fivedin);
    double nrm_cps2 = cpscp3.norm2();
      
    std::cout << "5D CB odd import norms CPS " << nrm_cps << " CPS direct " << nrm_cps2 << " Grid "  << nrm_grid << " Grid putback " << nrm2_grid << std::endl;

    assert( fabs(nrm_cps -nrm_cps2) < 1e-8 );
    assert( fabs(nrm_cps - nrm_grid) < 1e-8 );
    assert( fabs(nrm_cps - nrm2_grid) < 1e-8 );    
  }
}

template<typename A2Apolicies_std, typename A2Apolicies_grid>
void compareVgridstd(A2AvectorV<A2Apolicies_std> &V_std,
		     A2AvectorV<A2Apolicies_grid> &V_grid){

  int nl = V_std.getNl();
  int nh = V_std.getNh();
  
  for(int i=0;i<nl;i++){
    double nrm_grid = V_grid.getVl(i).norm2();
    double nrm_std = V_std.getVl(i).norm2();
    double diff = nrm_grid - nrm_std;
    std::cout << "vl " << i << " grid " << nrm_grid << " std " << nrm_std << " diff " << diff << std::endl;
    if(fabs(diff) > 1e-10){
      assert(false);
    }
  }
  for(int i=0;i<nh;i++){
    double nrm_grid = V_grid.getVh(i).norm2();
    double nrm_std = V_std.getVh(i).norm2();
    double diff = nrm_grid - nrm_std;
    std::cout << "vh " << i << " grid " << nrm_grid << " std " << nrm_std << " diff " << diff << std::endl;
    if(fabs(diff) > 1e-10){
      assert(false);
    }
  }
}


template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testGaugeFixAndPhasingGridStd(typename SIMDpolicyBase<4>::ParamType &simd_dims,
			    typename A2Apolicies_grid::FgridGFclass &lattice){
  ThreeMomentum p_plus( GJP.Bc(0)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(1)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(2)==BND_CND_GPARITY? 1 : 0 );
  ThreeMomentum p_minus = -p_plus;

  typename A2Apolicies_std::FermionFieldType field_std;
  field_std.testRandom();
  typename A2Apolicies_grid::FermionFieldType field_grid(simd_dims);
  field_grid.importField(field_std);

  std::cout << "Import CPS->CPS/Grid " << field_std.norm2() << " " << field_grid.norm2() << std::endl;

  field_std.gaugeFix(lattice,true);
  field_grid.gaugeFix(lattice,true);

  std::cout << "After gauge fix CPS->CPS/Grid " << field_std.norm2() << " " << field_grid.norm2() << std::endl;

  typename A2Apolicies_std::FermionFieldType field_std_tmp;
  field_std_tmp.importField(field_grid);

  compareField(field_std, field_std_tmp, "Gauge fix test", 1e-10);
    
  std::cout << "Phasing with " << p_plus.str() << std::endl;
  field_std.applyPhase(p_plus.ptr(),true);
  field_grid.applyPhase(p_plus.ptr(),true);

  field_std_tmp.importField(field_grid);
  compareField(field_std, field_std_tmp, "Phase test", 1e-10);

  CPSfermion4DglobalInOneDir<typename A2Apolicies_grid::ScalarComplexType> dbl_grid(0);
  CPSfermion4DglobalInOneDir<typename A2Apolicies_std::ComplexType> dbl_std(0);

  dbl_std.gather(field_std);
  dbl_std.fft();
    
  dbl_grid.gather(field_grid);
  dbl_grid.fft();
    
  compareField(dbl_std, dbl_grid, "Gather test", 1e-10);

  dbl_grid.scatter(field_grid);
  dbl_std.scatter(field_std);

  field_std_tmp.importField(field_grid);
  compareField(field_std, field_std_tmp, "FFT/scatter test", 1e-10);
    
}


template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testPionContractionGridStd(A2AvectorV<A2Apolicies_std> &V_std, A2AvectorW<A2Apolicies_std> &W_std,
			 A2AvectorV<A2Apolicies_grid> &V_grid, A2AvectorW<A2Apolicies_grid> &W_grid,
			 typename A2Apolicies_grid::FgridGFclass &lattice,
			 typename SIMDpolicyBase<3>::ParamType simd_dims_3d,
			 double tol){
  StandardPionMomentaPolicy momenta;
  MesonFieldMomentumContainer<A2Apolicies_std> mf_ll_con_std;
  MesonFieldMomentumContainer<A2Apolicies_grid> mf_ll_con_grid;
  
  computeGparityLLmesonFields1s<A2Apolicies_std,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_std,momenta,W_std,V_std,2.0,lattice);
  computeGparityLLmesonFields1s<A2Apolicies_grid,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_grid,momenta,W_grid,V_grid,2.0,lattice,simd_dims_3d);

  fMatrix<typename A2Apolicies_std::ScalarComplexType> fmat_std;
  ComputePion<A2Apolicies_std>::compute(fmat_std, mf_ll_con_std, momenta, 0);

  fMatrix<typename A2Apolicies_grid::ScalarComplexType> fmat_grid;
  ComputePion<A2Apolicies_grid>::compute(fmat_grid, mf_ll_con_grid, momenta, 0);

  bool fail = false;
  for(int r=0;r<fmat_std.nRows();r++){
    for(int c=0;c<fmat_std.nCols();c++){
      double rdiff = fmat_std(r,c).real() - fmat_grid(r,c).real();
      double idiff = fmat_std(r,c).imag() - fmat_grid(r,c).imag();
      if(rdiff > tol|| idiff > tol){
	printf("Fail Pion %d %d : (%f,%f) (%f,%f) diff (%g,%g)\n",r,c, fmat_std(r,c).real(),  fmat_std(r,c).imag(), fmat_grid(r,c).real(), fmat_grid(r,c).imag(), rdiff, idiff);	
	fail = true;
      }
    }
  }
  if(fail)ERR.General("","","Standard vs Grid implementation pion test failed\n");
  printf("Pion pass\n");
}


template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testKaonContractionGridStd(A2AvectorV<A2Apolicies_std> &V_std, A2AvectorW<A2Apolicies_std> &W_std,
			 A2AvectorV<A2Apolicies_grid> &V_grid, A2AvectorW<A2Apolicies_grid> &W_grid,
			 typename A2Apolicies_grid::FgridGFclass &lattice,
			 typename SIMDpolicyBase<3>::ParamType simd_dims_3d,
			 double tol){


  StationaryKaonMomentaPolicy kaon_mom;

  std::vector<A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorVfftw> > mf_ls_std;
  std::vector<A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorVfftw> > mf_sl_std;
  ComputeKaon<A2Apolicies_std>::computeMesonFields(mf_ls_std, mf_sl_std,
						   W_std, V_std,
						   W_std, V_std,
						   kaon_mom,
						   2.0, lattice);

  std::vector<A2AmesonField<A2Apolicies_grid,A2AvectorWfftw,A2AvectorVfftw> > mf_ls_grid;
  std::vector<A2AmesonField<A2Apolicies_grid,A2AvectorWfftw,A2AvectorVfftw> > mf_sl_grid;
  ComputeKaon<A2Apolicies_grid>::computeMesonFields(mf_ls_grid, mf_sl_grid,
						    W_grid, V_grid,
						    W_grid, V_grid,
						    kaon_mom,
						    2.0, lattice, simd_dims_3d);

  fMatrix<typename A2Apolicies_std::ScalarComplexType> fmat_std;
  ComputeKaon<A2Apolicies_std>::compute(fmat_std, mf_ls_std, mf_sl_std);

  fMatrix<typename A2Apolicies_grid::ScalarComplexType> fmat_grid;
  ComputeKaon<A2Apolicies_grid>::compute(fmat_grid, mf_ls_grid, mf_sl_grid);
  
  bool fail = false;
  for(int r=0;r<fmat_std.nRows();r++){
    for(int c=0;c<fmat_std.nCols();c++){
      double rdiff = fmat_std(r,c).real() - fmat_grid(r,c).real();
      double idiff = fmat_std(r,c).imag() - fmat_grid(r,c).imag();
      if(rdiff > tol|| idiff > tol){
	printf("Fail Kaon %d %d : (%f,%f) (%f,%f) diff (%g,%g)\n",r,c, fmat_std(r,c).real(),  fmat_std(r,c).imag(), fmat_grid(r,c).real(), fmat_grid(r,c).imag(), rdiff, idiff);
	fail = true;
      }
    }
  }
  if(fail)ERR.General("","","Standard vs Grid implementation kaon test failed\n");
  printf("Kaon pass\n");
}



template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testPiPiContractionGridStd(A2AvectorV<A2Apolicies_std> &V_std, A2AvectorW<A2Apolicies_std> &W_std,
				A2AvectorV<A2Apolicies_grid> &V_grid, A2AvectorW<A2Apolicies_grid> &W_grid,
				typename A2Apolicies_grid::FgridGFclass &lattice,
				typename SIMDpolicyBase<3>::ParamType simd_dims_3d,
				double tol){
  ThreeMomentum p_plus( GJP.Bc(0)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(1)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(2)==BND_CND_GPARITY? 1 : 0 );
  ThreeMomentum p_minus = -p_plus;

  ThreeMomentum p_pi_plus = p_plus * 2;
  
  StandardPionMomentaPolicy momenta;
  MesonFieldMomentumContainer<A2Apolicies_std> mf_ll_con_std;
  MesonFieldMomentumContainer<A2Apolicies_grid> mf_ll_con_grid;
  
  computeGparityLLmesonFields1s<A2Apolicies_std,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_std,momenta,W_std,V_std,2.0,lattice);
  computeGparityLLmesonFields1s<A2Apolicies_grid,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_grid,momenta,W_grid,V_grid,2.0,lattice,simd_dims_3d);

  char diags[] = {'C','D','R'};
  for(int d=0;d<3;d++){
    fMatrix<typename A2Apolicies_std::ScalarComplexType> fmat_std;
    MesonFieldProductStore<A2Apolicies_std> products_std;
    ComputePiPiGparity<A2Apolicies_std>::compute(fmat_std, diags[d], p_pi_plus, p_pi_plus, 2, 1, mf_ll_con_std, products_std);

    fMatrix<typename A2Apolicies_grid::ScalarComplexType> fmat_grid;
    MesonFieldProductStore<A2Apolicies_grid> products_grid;
    ComputePiPiGparity<A2Apolicies_grid>::compute(fmat_grid, diags[d], p_pi_plus, p_pi_plus, 2, 1, mf_ll_con_grid, products_grid);

    bool fail = false;
    for(int r=0;r<fmat_std.nRows();r++){
      for(int c=0;c<fmat_std.nCols();c++){
	double rdiff = fmat_std(r,c).real() - fmat_grid(r,c).real();
	double idiff = fmat_std(r,c).imag() - fmat_grid(r,c).imag();
	if(rdiff > tol|| idiff > tol){
	  printf("Fail Pipi fig %c elem %d %d : (%f,%f) (%f,%f) diff (%g,%g)\n",diags[d],r,c, fmat_std(r,c).real(),  fmat_std(r,c).imag(), fmat_grid(r,c).real(), fmat_grid(r,c).imag(), rdiff, idiff);
	  fail = true;
	}
      }
    }
    if(fail)ERR.General("","","Standard vs Grid implementation pipi fig %c test failed\n",diags[d]);
    printf("Pipi fig %c pass\n",diags[d]);    
  }
    
    
  {
    fVector<typename A2Apolicies_std::ScalarComplexType> pipi_figV_std;
    fVector<typename A2Apolicies_grid::ScalarComplexType> pipi_figV_grid;
    
    ComputePiPiGparity<A2Apolicies_std>::computeFigureVdis(pipi_figV_std, p_pi_plus, 1, mf_ll_con_std);
    ComputePiPiGparity<A2Apolicies_grid>::computeFigureVdis(pipi_figV_grid, p_pi_plus, 1, mf_ll_con_grid);

    bool fail = false;
    for(int r=0;r<pipi_figV_std.size();r++){
      double rdiff = pipi_figV_std(r).real() - pipi_figV_grid(r).real();
      double idiff = pipi_figV_std(r).imag() - pipi_figV_grid(r).imag();
      if(rdiff > tol|| idiff > tol){
	printf("Fail Pipi fig V elem %d : (%f,%f) (%f,%f) diff (%g,%g)\n",r, pipi_figV_std(r).real(),  pipi_figV_std(r).imag(), pipi_figV_grid(r).real(), pipi_figV_grid(r).imag(), rdiff, idiff);
	fail = true;
      }      
    }
    if(fail)ERR.General("","","Standard vs Grid implementation pipi fig V test failed\n");
    printf("Pipi fig V pass\n");    
  }
}





template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testKtoPiPiContractionGridStd(A2AvectorV<A2Apolicies_std> &V_std, A2AvectorW<A2Apolicies_std> &W_std,
				   A2AvectorV<A2Apolicies_grid> &V_grid, A2AvectorW<A2Apolicies_grid> &W_grid,
				   typename A2Apolicies_grid::FgridGFclass &lattice,
				   typename SIMDpolicyBase<3>::ParamType simd_dims_3d,
				   double tol){
  ThreeMomentum p_plus( GJP.Bc(0)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(1)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(2)==BND_CND_GPARITY? 1 : 0 );
  ThreeMomentum p_minus = -p_plus;

  ThreeMomentum p_pi_plus = p_plus * 2;
  
  StandardPionMomentaPolicy momenta;
  MesonFieldMomentumContainer<A2Apolicies_std> mf_ll_con_std;
  MesonFieldMomentumContainer<A2Apolicies_grid> mf_ll_con_grid;
  
  computeGparityLLmesonFields1s<A2Apolicies_std,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_std,momenta,W_std,V_std,2.0,lattice);
  computeGparityLLmesonFields1s<A2Apolicies_grid,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_grid,momenta,W_grid,V_grid,2.0,lattice,simd_dims_3d);

  StandardLSWWmomentaPolicy ww_mom;  
  std::vector<A2AmesonField<A2Apolicies_grid,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww_grid;
  ComputeKtoPiPiGparity<A2Apolicies_grid>::generatelsWWmesonfields(mf_ls_ww_grid,W_grid,W_grid, ww_mom, 2.0,lattice, simd_dims_3d);

  std::vector<A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww_std;
  ComputeKtoPiPiGparity<A2Apolicies_std>::generatelsWWmesonfields(mf_ls_ww_std,W_std,W_std, ww_mom, 2.0,lattice);

  mf_ll_con_grid.printMomenta(std::cout);

  if(1){
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type1_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type1(type1_grid, 4, 2, 1, 1, p_pi_plus, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);

    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type1_std;
    ComputeKtoPiPiGparity<A2Apolicies_std>::type1(type1_std, 4, 2, 1, 1, p_pi_plus, mf_ls_ww_std, mf_ll_con_std, V_std, V_std, W_std, W_std);

  
    bool fail = false;
    for(int i=0;i<type1_std.nElementsTotal();i++){
      std::complex<double> val_std = convertComplexD(type1_std[i]);
      std::complex<double> val_grid = convertComplexD(type1_grid[i]);
    
      double rdiff = fabs(val_grid.real()-val_std.real());
      double idiff = fabs(val_grid.imag()-val_std.imag());
      if(rdiff > tol|| idiff > tol){
	printf("!!!Fail: Type1 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
	fail = true;
      }//else printf("Pass: Type1 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
    }
    if(fail) ERR.General("","","Standard vs Grid implementation type1 test failed\n");
    printf("Type 1 pass\n");
  }
  if(1){
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type2_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type2(type2_grid, 4, 2, 1, p_pi_plus, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);

    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type2_std;
    ComputeKtoPiPiGparity<A2Apolicies_std>::type2(type2_std, 4, 2, 1, p_pi_plus, mf_ls_ww_std, mf_ll_con_std, V_std, V_std, W_std, W_std);

  
    bool fail = false;
    for(int i=0;i<type2_std.nElementsTotal();i++){
      std::complex<double> val_std = convertComplexD(type2_std[i]);
      std::complex<double> val_grid = convertComplexD(type2_grid[i]);
    
      double rdiff = fabs(val_grid.real()-val_std.real());
      double idiff = fabs(val_grid.imag()-val_std.imag());
      if(rdiff > tol|| idiff > tol){
	printf("!!!Fail: Type2 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
	fail = true;
      }//else printf("Pass: Type2 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
    }
    if(fail) ERR.General("","","Standard vs Grid implementation type2 test failed\n");
    printf("Type 2 pass\n");
  }
  if(1){
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type3_grid;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::MixDiagResultsContainerType type3_mix_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type3(type3_grid, type3_mix_grid, 4, 2, 1, p_pi_plus, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);

    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type3_std;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::MixDiagResultsContainerType type3_mix_std;
    ComputeKtoPiPiGparity<A2Apolicies_std>::type3(type3_std, type3_mix_std, 4, 2, 1, p_pi_plus, mf_ls_ww_std, mf_ll_con_std, V_std, V_std, W_std, W_std);

  
    bool fail = false;
    for(int i=0;i<type3_std.nElementsTotal();i++){
      std::complex<double> val_std = convertComplexD(type3_std[i]);
      std::complex<double> val_grid = convertComplexD(type3_grid[i]);
    
      double rdiff = fabs(val_grid.real()-val_std.real());
      double idiff = fabs(val_grid.imag()-val_std.imag());
      if(rdiff > tol|| idiff > tol){
	printf("!!!Fail: Type3 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
	fail = true;
      }//else printf("Pass: Type3 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
    }
    if(fail) ERR.General("","","Standard vs Grid implementation type3 test failed\n");
    printf("Type 3 pass\n");
  }
  if(1){
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type4_grid;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::MixDiagResultsContainerType type4_mix_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type4(type4_grid, type4_mix_grid, 1, mf_ls_ww_grid, V_grid, V_grid, W_grid, W_grid);

    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type4_std;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::MixDiagResultsContainerType type4_mix_std;
    ComputeKtoPiPiGparity<A2Apolicies_std>::type4(type4_std, type4_mix_std, 1, mf_ls_ww_std, V_std, V_std, W_std, W_std);
  
    bool fail = false;
    for(int i=0;i<type4_std.nElementsTotal();i++){
      std::complex<double> val_std = convertComplexD(type4_std[i]);
      std::complex<double> val_grid = convertComplexD(type4_grid[i]);
    
      double rdiff = fabs(val_grid.real()-val_std.real());
      double idiff = fabs(val_grid.imag()-val_std.imag());
      if(rdiff > tol|| idiff > tol){
	printf("!!!Fail: Type4 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
	fail = true;
      }//else printf("Pass: Type4 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
    }
    if(fail) ERR.General("","","Standard vs Grid implementation type4 test failed\n");
    printf("Type 4 pass\n");
  }
}




template<typename A2Apolicies>
void testMFmult(const A2AArg &a2a_args, const double tol){
  typedef A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_WV; 
  typedef typename mf_WV::ScalarComplexType ScalarComplexType;

  mf_WV l;
  l.setup(a2a_args,a2a_args,0,0);
  l.testRandom();  

  if(!UniqueID()) printf("mf_WV sizes %d %d\n",l.getNrows(),l.getNcols());

  mf_WV r;
  r.setup(a2a_args,a2a_args,1,1);
  r.testRandom();  

  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> c_base;
  c_base.setup(a2a_args,a2a_args,0,1);

  A2Aparams a2a_params(a2a_args);
  int nfull = a2a_params.getNv();
  if(!UniqueID()){ printf("Total modes %d\n", nfull); fflush(stdout); }

  for(int i=0;i<nfull;i++){
    for(int k=0;k<nfull;k++){
      ScalarComplexType *oe = c_base.elem_ptr(i,k);
      if(oe == NULL) continue; //zero by definition

      *oe = 0.;
      for(int j=0;j<nfull;j++)
      	*oe += l.elem(i,j) * r.elem(j,k);
    }
  }
 
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> c;
  mult(c, l, r, true); //node local

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node local mult failed!\n");
  
  mult(c, l, r, false); //node distributed

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node distributed mult failed!\n");

  if(!UniqueID()) printf("Passed MF mult tests\n");


#ifdef MULT_IMPL_GSL
  //Test other GSL implementations
  
  /////////////////////////////////////
  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_orig(c, l, r, true); //node local

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node local mult_orig failed!\n");
  
  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_orig(c, l, r, false); //node distributed

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node distributed mult_orig failed!\n");

  if(!UniqueID()) printf("Passed MF mult_orig tests\n");

  /////////////////////////////////////
  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt1(c,l,r, true);

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node local mult_opt1 failed!\n");

  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt1(c,l,r, false);

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node distributed mult_opt1 failed!\n");

  if(!UniqueID()) printf("Passed MF mult_opt1 tests\n");

  /////////////////////////////////////
  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt2(c,l,r, true);

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node local mult_opt2 failed!\n");

  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt2(c,l,r, false);

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node distributed mult_opt2 failed!\n");

  if(!UniqueID()) printf("Passed MF mult_opt2 tests\n");
#endif //MULT_IMPL_GSL
}

template<typename mf_Complex, typename grid_Complex>
bool compare(const CPSspinColorFlavorMatrix<mf_Complex> &orig, const CPSspinColorFlavorMatrix<grid_Complex> &grid, const double tol){
  bool fail = false;
  
  mf_Complex gd;
  for(int sl=0;sl<4;sl++)
    for(int cl=0;cl<3;cl++)
      for(int fl=0;fl<2;fl++)
	for(int sr=0;sr<4;sr++)
	  for(int cr=0;cr<3;cr++)
	    for(int fr=0;fr<2;fr++){
	      gd = Reduce( grid(sl,sr)(cl,cr)(fl,fr) );
	      const mf_Complex &cp = orig(sl,sr)(cl,cr)(fl,fr);
	      
	      double rdiff = fabs(gd.real()-cp.real());
	      double idiff = fabs(gd.imag()-cp.imag());
	      if(rdiff > tol|| idiff > tol){
		printf("Fail: Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		fail = true;
	      }
	    }
  return !fail;
}

template<typename mf_Complex>
bool compare(const CPSspinColorFlavorMatrix<mf_Complex> &orig, const CPSspinColorFlavorMatrix<mf_Complex> &newimpl, const double tol){
  bool fail = false;
  
  for(int sl=0;sl<4;sl++)
    for(int cl=0;cl<3;cl++)
      for(int fl=0;fl<2;fl++)
	for(int sr=0;sr<4;sr++)
	  for(int cr=0;cr<3;cr++)
	    for(int fr=0;fr<2;fr++){
	      const mf_Complex &gd = newimpl(sl,sr)(cl,cr)(fl,fr);
	      const mf_Complex &cp = orig(sl,sr)(cl,cr)(fl,fr);
	      
	      double rdiff = fabs(gd.real()-cp.real());
	      double idiff = fabs(gd.imag()-cp.imag());
	      if(rdiff > tol|| idiff > tol){
		printf("Fail: Newimpl (%g,%g) Orig (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		fail = true;
	      }
	    }
  return !fail;
}



template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void testvMvGridOrig(const A2AArg &a2a_args, const int ntests, const int nthreads, const double tol){
#ifdef USE_GRID
#define CPS_VMV
  //#define GRID_VMV
  //#define CPS_SPLIT_VMV
#define GRID_SPLIT_VMV
  //#define CPS_SPLIT_VMV_XALL
#define GRID_SPLIT_VMV_XALL
#define GRID_SPLIT_LITE_VMV;

  std::cout << "Starting vMv benchmark\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  typename FourDSIMDPolicy<typename ScalarA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<typename ScalarA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorWfftw<ScalarA2Apolicies> W(a2a_args);
  A2AvectorVfftw<ScalarA2Apolicies> V(a2a_args);
    
  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);

  W.testRandom();
  V.testRandom();
  Wgrid.importFields(W);
  Vgrid.importFields(V);
  
  A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf;
  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_grid;
  mf.setup(W,V,0,0);
  mf_grid.setup(Wgrid,Vgrid,0,0);     
  mf.testRandom();
  for(int i=0;i<mf.getNrows();i++)
    for(int j=0;j<mf.getNcols();j++)
      mf_grid(i,j) = mf(i,j); //both are scalar complex
  
  typedef typename GridA2Apolicies::ComplexType grid_Complex;
  typedef typename ScalarA2Apolicies::ComplexType mf_Complex;

      
  Float total_time = 0.;
  Float total_time_orig = 0.;
  Float total_time_split_orig = 0.;
  Float total_time_split_grid = 0.;
  Float total_time_split_orig_xall = 0.;
  Float total_time_split_grid_xall = 0.;
  Float total_time_split_lite_grid = 0.;
  Float total_time_field_offload = 0.;
  mult_vMv_field_offload_timers::get().reset();

  CPSspinColorFlavorMatrix<mf_Complex> orig_sum[nthreads];
  CPSspinColorFlavorMatrix<grid_Complex> grid_sum[nthreads];

  CPSspinColorFlavorMatrix<mf_Complex> orig_tmp[nthreads];
  CPSspinColorFlavorMatrix<grid_Complex> grid_tmp[nthreads];

  CPSspinColorFlavorMatrix<mf_Complex> orig_sum_split[nthreads];
  CPSspinColorFlavorMatrix<grid_Complex> grid_sum_split[nthreads];

  CPSspinColorFlavorMatrix<mf_Complex> orig_sum_split_xall[nthreads];
  CPSspinColorFlavorMatrix<grid_Complex> grid_sum_split_xall[nthreads];

  CPSspinColorFlavorMatrix<grid_Complex> grid_sum_split_lite[nthreads];      

  int orig_3vol = GJP.VolNodeSites()/GJP.TnodeSites();
  int grid_3vol = Vgrid.getMode(0).nodeSites(0) * Vgrid.getMode(0).nodeSites(1) *Vgrid.getMode(0).nodeSites(2);

  mult_vMv_split<ScalarA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw> vmv_split_orig;
  mult_vMv_split<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw> vmv_split_grid;
  mult_vMv_split_lite<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw> vmv_split_lite_grid;

  std::vector<CPSspinColorFlavorMatrix<mf_Complex> > orig_split_xall_tmp(orig_3vol);
  typename AlignedVector<CPSspinColorFlavorMatrix<grid_Complex> >::type grid_split_xall_tmp(grid_3vol);
      
  for(int iter=0;iter<ntests;iter++){
    for(int i=0;i<nthreads;i++){
      orig_sum[i].zero(); grid_sum[i].zero();
      orig_sum_split[i].zero(); grid_sum_split[i].zero();
      grid_sum_split_lite[i].zero();
      orig_sum_split_xall[i].zero(); grid_sum_split_xall[i].zero();
    }
	
    for(int top = 0; top < GJP.TnodeSites(); top++){
#ifdef CPS_VMV
      //ORIG VMV
      total_time_orig -= dclock();	  
#pragma omp parallel for
      for(int xop=0;xop<orig_3vol;xop++){
	int me = omp_get_thread_num();
	mult(orig_tmp[me], V, mf, W, xop, top, false, true);
	orig_sum[me] += orig_tmp[me];
      }
      total_time_orig += dclock();
#endif
#ifdef GRID_VMV
      //GRID VMV
      total_time -= dclock();
#pragma omp parallel for
      for(int xop=0;xop<grid_3vol;xop++){
	int me = omp_get_thread_num();
	mult(grid_tmp[me], Vgrid, mf_grid, Wgrid, xop, top, false, true);
	grid_sum[me] += grid_tmp[me];
      }
      total_time += dclock();
#endif

#ifdef CPS_SPLIT_VMV
      //SPLIT VMV
      total_time_split_orig -= dclock();	  
      vmv_split_orig.setup(V, mf, W, top);

#pragma omp parallel for
      for(int xop=0;xop<orig_3vol;xop++){
	int me = omp_get_thread_num();
	vmv_split_orig.contract(orig_tmp[me], xop, false, true);
	orig_sum_split[me] += orig_tmp[me];
      }
      total_time_split_orig += dclock();
#endif

#ifdef GRID_SPLIT_VMV
      //SPLIT VMV GRID
      total_time_split_grid -= dclock();	  
      vmv_split_grid.setup(Vgrid, mf_grid, Wgrid, top);

#pragma omp parallel for
      for(int xop=0;xop<grid_3vol;xop++){
	int me = omp_get_thread_num();
	vmv_split_grid.contract(grid_tmp[me], xop, false, true);
	grid_sum_split[me] += grid_tmp[me];
      }
      total_time_split_grid += dclock();
#endif

#ifdef CPS_SPLIT_VMV_XALL	  	 
      //SPLIT VMV THAT DOES IT FOR ALL SITES
      total_time_split_orig_xall -= dclock();	  
      vmv_split_orig.setup(V, mf, W, top);
      vmv_split_orig.contract(orig_split_xall_tmp, false, true);
#pragma omp parallel for
      for(int xop=0;xop<orig_3vol;xop++){
	int me = omp_get_thread_num();
	orig_sum_split_xall[me] += orig_split_xall_tmp[xop];
      }
      total_time_split_orig_xall += dclock();
#endif

#ifdef GRID_SPLIT_VMV_XALL
      //SPLIT VMV GRID THAT DOES IT FOR ALL SITES
      total_time_split_grid_xall -= dclock();	  
      vmv_split_grid.setup(Vgrid, mf_grid, Wgrid, top);
      vmv_split_grid.contract(grid_split_xall_tmp, false, true);
#pragma omp parallel for
      for(int xop=0;xop<grid_3vol;xop++){
	int me = omp_get_thread_num();	    
	grid_sum_split_xall[me] += grid_split_xall_tmp[xop];
      }
      total_time_split_grid_xall += dclock();
#endif	  

#ifdef GRID_SPLIT_LITE_VMV
      //SPLIT LITE VMV GRID
      total_time_split_lite_grid -= dclock();	  
      vmv_split_lite_grid.setup(Vgrid, mf_grid, Wgrid, top);

#pragma omp parallel for
      for(int xop=0;xop<grid_3vol;xop++){
	int me = omp_get_thread_num();
	vmv_split_lite_grid.contract(grid_tmp[me], xop, false, true);
	grid_sum_split_lite[me] += grid_tmp[me];
      }
      total_time_split_lite_grid += dclock();
#endif
    }//end top loop
    for(int i=1;i<nthreads;i++){
      orig_sum[0] += orig_sum[i];
      grid_sum[0] += grid_sum[i];
      orig_sum_split[0] += orig_sum_split[i];
      grid_sum_split[0] += grid_sum_split[i];
      orig_sum_split_xall[0] += orig_sum_split_xall[i];
      grid_sum_split_xall[0] += grid_sum_split_xall[i];
      grid_sum_split_lite[0] += grid_sum_split_lite[i];  
    }

    //Offload version computes all x,t, so we just have to sum over 4 volume afterwards
    total_time_field_offload -= dclock();
    typedef typename mult_vMv_field<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw>::PropagatorField PropagatorField;
    PropagatorField pfield(simd_dims);
    
    mult(pfield, Vgrid, mf_grid, Wgrid, false, true);
    total_time_field_offload += dclock();

    CPSspinColorFlavorMatrix<grid_Complex> vmv_offload_sum4;
    vmv_offload_sum4.zero();
    for(size_t i=0;i<pfield.size();i++){
      vmv_offload_sum4 += *pfield.fsite_ptr(i);
    }


#ifdef CPS_VMV
    if(iter == 0){
#  ifdef GRID_VMV
      if(!compare(orig_sum[0],grid_sum[0],tol)) ERR.General("","","Standard vs Grid implementation test failed\n");
      else if(!UniqueID()) printf("Standard vs Grid implementation test pass\n");
#  endif
#  ifdef CPS_SPLIT_VMV
      if(!compare(orig_sum[0],orig_sum_split[0],tol)) ERR.General("","","Standard vs Split implementation test failed\n");
      else if(!UniqueID()) printf("Standard vs Split implementation test pass\n");
#  endif
#  ifdef GRID_SPLIT_VMV
      if(!compare(orig_sum[0],grid_sum_split[0],tol)) ERR.General("","","Standard vs Grid Split implementation test failed\n");
      else if(!UniqueID()) printf("Standard vs Grid Split implementation test pass\n");
#  endif
#  ifdef CPS_SPLIT_VMV_XALL
      if(!compare(orig_sum[0],orig_sum_split_xall[0],tol)) ERR.General("","","Standard vs Split xall implementation test failed\n");
      else if(!UniqueID()) printf("Standard vs Split xall implementation test pass\n");
#  endif
#  ifdef GRID_SPLIT_VMV_XALL
      if(!compare(orig_sum[0],grid_sum_split_xall[0],tol)) ERR.General("","","Standard vs Grid split xall implementation test failed\n");
      else if(!UniqueID()) printf("Standard vs Grid split xall implementation test pass\n");
#  endif
#  ifdef GRID_SPLIT_LITE_VMV
      if(!compare(orig_sum[0],grid_sum_split_lite[0],tol)) ERR.General("","","Standard vs Grid Split Lite implementation test failed\n");
      else if(!UniqueID()) printf("Standard vs Grid Split Lite implementation test pass\n");
#  endif

      if(!compare(orig_sum[0],vmv_offload_sum4,tol)) ERR.General("","","Standard vs Grid field offload implementation test failed\n");
      else if(!UniqueID()) printf("Standard vs Grid field offload implementation test pass\n");
    }
#endif
  } //tests loop
#ifdef CPS_VMV
  printf("vMv: Avg time old code %d iters: %g secs\n",ntests,total_time_orig/ntests);
#endif
#ifdef GRID_VMV
  printf("vMv: Avg time new code %d iters: %g secs\n",ntests,total_time/ntests);
#endif
#ifdef CPS_SPLIT_VMV
  printf("vMv: Avg time old code split %d iters: %g secs\n",ntests,total_time_split_orig/ntests);
#endif
#ifdef GRID_SPLIT_VMV
  printf("vMv: Avg time new code split %d iters: %g secs\n",ntests,total_time_split_grid/ntests);
#endif
#ifdef CPS_SPLIT_VMV_XALL
  printf("vMv: Avg time old code split xall %d iters: %g secs\n",ntests,total_time_split_orig_xall/ntests);
#endif
#ifdef GRID_SPLIT_VMV_XALL
  printf("vMv: Avg time new code split xall %d iters: %g secs\n",ntests,total_time_split_grid_xall/ntests);
#endif
#ifdef GRID_SPLIT_LITE_VMV
  printf("vMv: Avg time new code split lite %d iters: %g secs\n",ntests,total_time_split_lite_grid/ntests);
#endif
  printf("vMv: Avg time offload %d iters: %g secs\n",ntests,total_time_field_offload/ntests);

  if(!UniqueID()){
    printf("vMv offload timings:\n");
    mult_vMv_field_offload_timers::get().print();
  }

#endif
}

std::ostream & operator<<(std::ostream &os, const std::pair<int,int> &p){
  os << "(" << p.first << ", " << p.second << ")";
  return os;
}

template<typename T>
void _expect_eq(const T &a, const T &b, const char* file, const int line){
  if(!UniqueID()) std::cout << file << ":" << line << " : Expected equal " << a << " " << b << std::endl;
  if(a!=b) exit(1);
}
#define EXPECT_EQ(A,B) _expect_eq<typename std::decay<decltype(A)>::type>(A,B, __FILE__, __LINE__)
  
void testModeMappingTranspose(const A2AArg &a2a_arg){
  if(!UniqueID()) printf("Starting testModeMappingTranspose\n");
  //FullyPackedIndexDilution dilA(a2a_arg);
  //TimeFlavorPackedIndexDilution dilB(a2a_arg);
  typedef ModeMapping<FullyPackedIndexDilution, TimeFlavorPackedIndexDilution> mapAB;
  typedef ModeMapping<TimeFlavorPackedIndexDilution, FullyPackedIndexDilution> mapBA;

  typename mapAB::TensorType mapAB_v;
  mapAB::compute(mapAB_v, a2a_arg);

  typename mapBA::TensorType mapBA_v;
  mapBA::compute(mapBA_v, a2a_arg);

  //FullyPackedIndexDilution  packed sc, f, t
  //TimeFlavorPackedIndexDilution   packed f,t

  int nf = GJP.Gparity() ? 2:1;
  int nt = GJP.Tnodes()*GJP.TnodeSites();

  int sizes_expect_AB[] = {12, nf, nt, nf, nt};
  int sizes_expect_BA[] = {nf, nt, 12, nf, nt};

  EXPECT_EQ(mapAB_v.size(), sizes_expect_AB[0]);
  EXPECT_EQ(mapAB_v[0].size(), sizes_expect_AB[1]);
  EXPECT_EQ(mapAB_v[0][0].size(), sizes_expect_AB[2]);
  EXPECT_EQ(mapAB_v[0][0][0].size(), sizes_expect_AB[3]);
  EXPECT_EQ(mapAB_v[0][0][0][0].size(), sizes_expect_AB[4]);

  EXPECT_EQ(mapBA_v.size(), sizes_expect_BA[0]);
  EXPECT_EQ(mapBA_v[0].size(), sizes_expect_BA[1]);
  EXPECT_EQ(mapBA_v[0][0].size(), sizes_expect_BA[2]);
  EXPECT_EQ(mapBA_v[0][0][0].size(), sizes_expect_BA[3]);
  EXPECT_EQ(mapBA_v[0][0][0][0].size(), sizes_expect_BA[4]);

  for(int sc1=0;sc1<12;sc1++){
    for(int f1=0;f1<nf;f1++){
      for(int t1=0;t1<nt;t1++){
	for(int f2=0;f2<nf;f2++){
	  for(int t2=0;t2<nt;t2++){	    
	    EXPECT_EQ(mapAB_v[sc1][f1][t1][f2][t2].size(), mapBA_v[f2][t2][sc1][f1][t1].size());
	    for(int i=0;i<mapAB_v[sc1][f1][t1][f2][t2].size();i++){
	      const std::pair<int,int> &lv = mapAB_v[sc1][f1][t1][f2][t2][i];
	      const std::pair<int,int> &rv = mapBA_v[f2][t2][sc1][f1][t1][i];
	      std::pair<int,int> rvt = {rv.second, rv.first}; //of course it will transpose the indices
	      EXPECT_EQ(lv, rvt);
	    }
	  }
	}
      }
    }
  }

 
  if(!UniqueID()) printf("Finished testModeMappingTranspose\n");
}



#ifdef USE_GRID

template<typename GridA2Apolicies>
void testComputeLowModeMADWF(const A2AArg &a2a_args, const LancArg &lanc_arg,
			     typename GridA2Apolicies::FgridGFclass &lattice, const typename SIMDpolicyBase<4>::ParamType &simd_dims, 
			     const double tol){
  //If we use the same eigenvectors and the same Dirac operator we should get the same result
  GridLanczosWrapper<GridA2Apolicies> evecs_rand;
  evecs_rand.randomizeEvecs(lanc_arg, lattice);
  EvecInterfaceGrid<GridA2Apolicies> eveci_rand(evecs_rand.evec, evecs_rand.eval);

  A2AvectorV<GridA2Apolicies> V_orig(a2a_args,simd_dims);
  A2AvectorW<GridA2Apolicies> W_orig(a2a_args,simd_dims);

  A2AvectorV<GridA2Apolicies> V_test(a2a_args,simd_dims);
  A2AvectorW<GridA2Apolicies> W_test(a2a_args,simd_dims);

  int Ls = GJP.Snodes() * GJP.SnodeSites();
  double mob_b = lattice.get_mob_b();
  double mob_c = mob_b - 1.;
  
  CGcontrols cg_con_orig;

  CGcontrols cg_con_test;
  cg_con_test.madwf_params.Ls_inner = Ls;
  cg_con_test.madwf_params.b_plus_c_inner = mob_b + mob_c;
  cg_con_test.madwf_params.use_ZMobius = false;
  cg_con_test.madwf_params.precond = SchurOriginal;
  
  computeVWlowStandard(V_orig, W_orig, lattice, eveci_rand, evecs_rand.mass, cg_con_orig);
  computeVWlowMADWF(V_test, W_test, lattice, eveci_rand, evecs_rand.mass, cg_con_test);

  int nl = a2a_args.nl;
  for(int i=0;i<nl;i++){
    if( ! V_orig.getVl(i).equals(  V_test.getVl(i), tol, true ) ){ std::cout << "FAIL" << std::endl; exit(1); }
  }
  if(!UniqueID()) printf("Passed Vl test\n");

  for(int i=0;i<nl;i++){
    if( ! W_orig.getWl(i).equals(  W_test.getWl(i), tol, true ) ){ std::cout << "FAIL" << std::endl; exit(1); }
  }

}

template<typename GridA2Apolicies>
void testCPSfieldDeviceCopy(){
  typedef typename GridA2Apolicies::ComplexType ComplexType;
  int nsimd = ComplexType::Nsimd();
  typename SIMDpolicyBase<4>::ParamType simd_dims;
  SIMDpolicyBase<4>::SIMDdefaultLayout(simd_dims,nsimd,2); //only divide over spatial directions

  typedef typename GridA2Apolicies::FermionFieldType FermionFieldType;
  
  //Test a host-allocated CPSfield
  FermionFieldType field(simd_dims);
  field.testRandom();

  ComplexType* into = (ComplexType*)managed_alloc_check(sizeof(ComplexType));
  typedef SIMT<ComplexType> ACC;

  ComplexType expect = *field.site_ptr(size_t(0));

  copyControl::shallow() = true;
  accelerator_for(x, 1, nsimd,
		  {
		    auto v = ACC::read(*field.site_ptr(x));
		    ACC::write(*into, v);
		  });
  copyControl::shallow() = false;


  std::cout << "Got " << *into << " expect " << expect << std::endl;
  
  assert( Reduce(expect == *into) );
  
  //Test a CPSfield allocated directly into managed memory such that it's contents are directly accessible to the device
  ManagedPtrWrapper<FermionFieldType> wrp(field);
  //wrp.emplace(field);
  
  memset(into, 0, sizeof(ComplexType));

  copyControl::shallow() = true;
  accelerator_for(x, 1, nsimd,
		  {
		    auto v = ACC::read(*wrp->site_ptr(x));
		    ACC::write(*into, v);
		  });
  copyControl::shallow() = false;


  std::cout << "Got " << *into << " expect " << expect << std::endl;
  
  assert( Reduce(expect == *into) );  

  //Test a ManagedVector<ManagedPtrWrapper>
  ManagedVector<ManagedPtrWrapper<FermionFieldType> > vect(1);
  vect[0] = field;

  memset(into, 0, sizeof(ComplexType));

  copyControl::shallow() = true;
  accelerator_for(x, 1, nsimd,
		  {
		    auto v = ACC::read(* vect[0]->site_ptr(x));
		    ACC::write(*into, v);
		  });
  copyControl::shallow() = false;

  std::cout << "Got " << *into << " expect " << expect << std::endl;
  
  assert( Reduce(expect == *into) );  

  A2AArg a2a_arg;
  a2a_arg.nl = 1;
  a2a_arg.nhits = 1;
  a2a_arg.rand_type = UONE;
  a2a_arg.src_width = 1;

  A2AvectorV<GridA2Apolicies> v(a2a_arg,simd_dims);
  v.getMode(0) = field;
  
  memset(into, 0, sizeof(ComplexType));

  copyControl::shallow() = true;
  accelerator_for(x, 1, nsimd,
		  {
		    const ComplexType &vv = *v.getMode(0).site_ptr(x);
		    auto v = ACC::read(vv);
		    ACC::write(*into, v);
		  });
  copyControl::shallow() = false;

  std::cout << "Got " << *into << " expect " << expect << std::endl;
  
  assert( Reduce(expect == *into) );  





  managed_free(into);
}



template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void testVVgridOrig(const A2AArg &a2a_args, const int ntests, const int nthreads, const double tol){
  std::cout << "Starting vv test/timing\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorWfftw<ScalarA2Apolicies> W(a2a_args);
  A2AvectorVfftw<ScalarA2Apolicies> V(a2a_args);
    
  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);

  W.testRandom();
  V.testRandom();
  Wgrid.importFields(W);
  Vgrid.importFields(V);
  
  typedef typename GridA2Apolicies::ComplexType grid_Complex;
  typedef typename ScalarA2Apolicies::ComplexType mf_Complex;
      
  Float total_time = 0.;
  Float total_time_orig = 0.;
  Float total_time_field_offload = 0;
  CPSspinColorFlavorMatrix<mf_Complex> orig_sum[nthreads];
  CPSspinColorFlavorMatrix<grid_Complex> grid_sum[nthreads];

  CPSspinColorFlavorMatrix<mf_Complex> orig_tmp[nthreads];
  CPSspinColorFlavorMatrix<grid_Complex> grid_tmp[nthreads];

  int orig_3vol = GJP.VolNodeSites()/GJP.TnodeSites();
  int grid_3vol = Vgrid.getMode(0).nodeSites(0) * Vgrid.getMode(0).nodeSites(1) *Vgrid.getMode(0).nodeSites(2);
      
  for(int iter=0;iter<ntests;iter++){
    for(int i=0;i<nthreads;i++){
      orig_sum[i].zero(); grid_sum[i].zero();
    }
	
    for(int top = 0; top < GJP.TnodeSites(); top++){
      //std::cout << "top " << top << std::endl;
      //std::cout << "Starting orig\n";
      total_time_orig -= dclock();	  
#pragma omp parallel for
      for(int xop=0;xop<orig_3vol;xop++){
	int me = omp_get_thread_num();
	mult(orig_tmp[me], V, W, xop, top, false, true);
	orig_sum[me] += orig_tmp[me];
      }
      total_time_orig += dclock();
      //std::cout << "Starting Grid\n";
      total_time -= dclock();
#pragma omp parallel for
      for(int xop=0;xop<grid_3vol;xop++){
	int me = omp_get_thread_num();
	mult(grid_tmp[me], Vgrid, Wgrid, xop, top, false, true);
	grid_sum[me] += grid_tmp[me];
      }
      total_time += dclock();	  
    }
    for(int i=1;i<nthreads;i++){
      orig_sum[0] += orig_sum[i];
      grid_sum[0] += grid_sum[i];
    }

    //Offload version computes all x,t, so we just have to sum over 4 volume afterwards
    total_time_field_offload -= dclock();
    typedef typename mult_vv_field<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw>::PropagatorField PropagatorField;
    PropagatorField pfield(simd_dims);
    
    mult(pfield, Vgrid, Wgrid, false, true);
    total_time_field_offload += dclock();

    CPSspinColorFlavorMatrix<grid_Complex> vmv_offload_sum4;
    vmv_offload_sum4.zero();
    for(size_t i=0;i<pfield.size();i++){
      vmv_offload_sum4 += *pfield.fsite_ptr(i);
    }
	
    bool fail = false;
	
    typename GridA2Apolicies::ScalarComplexType gd;
    for(int sl=0;sl<4;sl++)
      for(int cl=0;cl<3;cl++)
	for(int fl=0;fl<2;fl++)
	  for(int sr=0;sr<4;sr++)
	    for(int cr=0;cr<3;cr++)
	      for(int fr=0;fr<2;fr++){
		gd = Reduce( grid_sum[0](sl,sr)(cl,cr)(fl,fr) );
		const mf_Complex &cp = orig_sum[0](sl,sr)(cl,cr)(fl,fr);

		double rdiff = fabs(gd.real()-cp.real());
		double idiff = fabs(gd.imag()-cp.imag());
		if(rdiff > tol|| idiff > tol){
		  printf("Fail: Iter %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",iter, gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		  fail = true;
		}
	      }

    if(fail) ERR.General("","","Standard vs Grid implementation test failed\n");


    for(int sl=0;sl<4;sl++)
      for(int cl=0;cl<3;cl++)
	for(int fl=0;fl<2;fl++)
	  for(int sr=0;sr<4;sr++)
	    for(int cr=0;cr<3;cr++)
	      for(int fr=0;fr<2;fr++){
		gd = Reduce( vmv_offload_sum4(sl,sr)(cl,cr)(fl,fr) );
		const mf_Complex &cp = orig_sum[0](sl,sr)(cl,cr)(fl,fr);

		double rdiff = fabs(gd.real()-cp.real());
		double idiff = fabs(gd.imag()-cp.imag());
		if(rdiff > tol|| idiff > tol){
		  printf("Fail: Iter %d Grid field offload (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",iter, gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		  fail = true;
		}
	      }

    if(fail) ERR.General("","","Standard vs Grid field offload implementation test failed\n");
  }

  printf("vv: Avg time new code %d iters: %g secs\n",ntests,total_time/ntests);
  printf("vv: Avg time old code %d iters: %g secs\n",ntests,total_time_orig/ntests);
  printf("vv: Avg time field offload code %d iters: %g secs\n",ntests,total_time_field_offload/ntests);

  if(!UniqueID()){
    printf("vv offload timings:\n");
    mult_vv_field_offload_timers::get().print();
  }

}



template<typename GridA2Apolicies>
void testCPSmatrixField(const double tol){
  //Test type conversion
  {
      typedef CPSspinColorFlavorMatrix<typename GridA2Apolicies::ComplexType> VectorMatrixType;
      typedef CPSspinColorFlavorMatrix<typename GridA2Apolicies::ScalarComplexType> ScalarMatrixType;
      typedef typename VectorMatrixType::template RebaseScalarType<typename GridA2Apolicies::ScalarComplexType>::type ScalarMatrixTypeTest;
      static_assert( std::is_same<ScalarMatrixType, ScalarMatrixTypeTest>::value );
      static_assert( VectorMatrixType::isDerivedFromCPSsquareMatrix != -1 );
  }
  {
      typedef CPSspinMatrix<typename GridA2Apolicies::ComplexType> VectorMatrixType;
      typedef CPSspinMatrix<typename GridA2Apolicies::ScalarComplexType> ScalarMatrixType;
      typedef typename VectorMatrixType::template RebaseScalarType<typename GridA2Apolicies::ScalarComplexType>::type ScalarMatrixTypeTest;
      static_assert( std::is_same<ScalarMatrixType, ScalarMatrixTypeTest>::value );
      static_assert( VectorMatrixType::isDerivedFromCPSsquareMatrix != -1 );
  }

  typedef typename GridA2Apolicies::ComplexType ComplexType;
  typedef typename GridA2Apolicies::ScalarComplexType ScalarComplexType;
  typedef CPSspinColorFlavorMatrix<ComplexType> VectorMatrixType;
  typedef CPSmatrixField<VectorMatrixType> PropagatorField;

  static const int nsimd = GridA2Apolicies::ComplexType::Nsimd();
  typename PropagatorField::InputParamType simd_dims;
  PropagatorField::SIMDdefaultLayout(simd_dims,nsimd,2);
  
  PropagatorField a(simd_dims), b(simd_dims);
  for(size_t x4d=0; x4d< a.size(); x4d++){
    for(int s1=0;s1<4;s1++){
      for(int c1=0;c1<3;c1++){
	for(int f1=0;f1<2;f1++){
	  for(int s2=0;s2<4;s2++){
	    for(int c2=0;c2<3;c2++){
	      for(int f2=0;f2<2;f2++){
		ComplexType &v = (*a.site_ptr(x4d))(s1,s2)(c1,c2)(f1,f2);
		for(int s=0;s<nsimd;s++) v.putlane( ScalarComplexType( LRG.Urand(FOUR_D), LRG.Urand(FOUR_D) ), s );
		
		//ScalarComplexType to[nsimd];
		//for(int s=0;s<nsimd;s++) to[s] = ScalarComplexType( LRG.Urand(FOUR_D), LRG.Urand(FOUR_D) );
		//Grid::vset(v,to);

		ComplexType &u = (*b.site_ptr(x4d))(s1,s2)(c1,c2)(f1,f2);
		for(int s=0;s<nsimd;s++) u.putlane( ScalarComplexType( LRG.Urand(FOUR_D), LRG.Urand(FOUR_D) ), s );

		//for(int s=0;s<nsimd;s++) to[s] = ScalarComplexType( LRG.Urand(FOUR_D), LRG.Urand(FOUR_D) );
		//Grid::vset(u,to);
	      }
	    }
	  }
	}
      }
    }
  }

  //Test operator*
  PropagatorField c = a * b;

  bool fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=*a.site_ptr(x4d);
    auto bb=*b.site_ptr(x4d);
    auto cc = aa*bb;
    for(int s1=0;s1<4;s1++){
    for(int c1=0;c1<3;c1++){
    for(int f1=0;f1<2;f1++){
    for(int s2=0;s2<4;s2++){
    for(int c2=0;c2<3;c2++){
    for(int f2=0;f2<2;f2++){
      auto got = Reduce( (*c.site_ptr(x4d))(s1,s2)(c1,c2)(f1,f2) );
      auto expect = Reduce( cc(s1,s2)(c1,c2)(f1,f2) );
      
      double rdiff = fabs(got.real()-expect.real());
      double idiff = fabs(got.imag()-expect.imag());
      if(rdiff > tol|| idiff > tol){
	printf("Fail: operator* (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	fail = true;
      }
    }
    }
    }
    }
    }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField operator* failed\n");

  //Test Trace
  typedef CPSmatrixField<ComplexType> ComplexField;
 
  ComplexField d = Trace(a);

  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=*a.site_ptr(x4d);
    ComplexType aat = aa.Trace();
    auto got = Reduce( *d.site_ptr(x4d) );
    auto expect = Reduce( aat );
      
    double rdiff = fabs(got.real()-expect.real());
    double idiff = fabs(got.imag()-expect.imag());
    if(rdiff > tol|| idiff > tol){
      printf("Fail: Trace (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
      fail = true;
    } 
  }
  if(fail) ERR.General("","","CPSmatrixField Trace failed\n");

  //Test SpinFlavorTrace
  typedef CPSmatrixField<CPScolorMatrix<ComplexType> > ColorMatrixField;
  ColorMatrixField ac = SpinFlavorTrace(a);
  
  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=a.site_ptr(x4d)->SpinFlavorTrace();
    for(int c1=0;c1<3;c1++){
    for(int c2=0;c2<3;c2++){
      auto got = Reduce( (*ac.site_ptr(x4d))(c1,c2) );
      auto expect = Reduce( aa(c1,c2) );
      
      double rdiff = fabs(got.real()-expect.real());
      double idiff = fabs(got.imag()-expect.imag());
      if(rdiff > tol|| idiff > tol){
	printf("Fail: SpinFlavorTrace (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	fail = true;
      }
    }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField SpinFlavorTrace failed\n");



  //Test TransposeOnIndex
  typedef CPSmatrixField< CPSsquareMatrix<CPSsquareMatrix<ComplexType,2> ,2>  > Matrix2Field;
  Matrix2Field e(simd_dims);
  for(size_t x4d=0; x4d< e.size(); x4d++){
    for(int s1=0;s1<2;s1++){
      for(int c1=0;c1<2;c1++){
	for(int s2=0;s2<2;s2++){
	  for(int c2=0;c2<2;c2++){
	    ComplexType &v = (*e.site_ptr(x4d))(s1,s2)(c1,c2);
	    for(int s=0;s<nsimd;s++) v.putlane( ScalarComplexType( LRG.Urand(FOUR_D), LRG.Urand(FOUR_D) ), s );
	  }
	}
      }
    }
  }

  Matrix2Field f = TransposeOnIndex<1>(e);

  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto ee=*e.site_ptr(x4d);
    auto eet = ee.template TransposeOnIndex<1>();
    for(int s1=0;s1<2;s1++){
    for(int c1=0;c1<2;c1++){
    for(int s2=0;s2<2;s2++){
    for(int c2=0;c2<2;c2++){
      auto got = Reduce( (*f.site_ptr(x4d))(s1,s2)(c1,c2) );
      auto expect = Reduce( eet(s1,s2)(c1,c2) );
      
      double rdiff = fabs(got.real()-expect.real());
      double idiff = fabs(got.imag()-expect.imag());
      if(rdiff > tol|| idiff > tol){
	printf("Fail: TranposeOnIndex (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	fail = true;
      }
    }
    }
    }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField TransposeOnIndex failed\n");


  
  //Test TimesMinusI
  PropagatorField tmIa(a);
  timesMinusI(tmIa);

  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=*a.site_ptr(x4d);
    aa.timesMinusI();
    for(int s1=0;s1<4;s1++){
    for(int c1=0;c1<3;c1++){
    for(int f1=0;f1<2;f1++){
    for(int s2=0;s2<4;s2++){
    for(int c2=0;c2<3;c2++){
    for(int f2=0;f2<2;f2++){
      auto got = Reduce( (*tmIa.site_ptr(x4d))(s1,s2)(c1,c2)(f1,f2) );
      auto expect = Reduce( aa(s1,s2)(c1,c2)(f1,f2) );
      
      double rdiff = fabs(got.real()-expect.real());
      double idiff = fabs(got.imag()-expect.imag());
      if(rdiff > tol|| idiff > tol){
	printf("Fail: timesMinusI (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	fail = true;
      }
    }
    }
    }
    }
    }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField timesMinusI failed\n");




  //Test pl
  PropagatorField pla(a);
  pl(pla, sigma2);

  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=*a.site_ptr(x4d);
    aa.pl(sigma2);
    for(int s1=0;s1<4;s1++){
    for(int c1=0;c1<3;c1++){
    for(int f1=0;f1<2;f1++){
    for(int s2=0;s2<4;s2++){
    for(int c2=0;c2<3;c2++){
    for(int f2=0;f2<2;f2++){
      auto got = Reduce( (*pla.site_ptr(x4d))(s1,s2)(c1,c2)(f1,f2) );
      auto expect = Reduce( aa(s1,s2)(c1,c2)(f1,f2) );
      
      double rdiff = fabs(got.real()-expect.real());
      double idiff = fabs(got.imag()-expect.imag());
      if(rdiff > tol|| idiff > tol){
	printf("Fail: pl (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	fail = true;
      }
    }
    }
    }
    }
    }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField pl failed\n");


  //Test local reduction
  {
    if(!UniqueID()){ printf("Testing local reduction\n"); fflush(stdout); }
    VectorMatrixType sum_expect = localNodeSumSimple(a);
    VectorMatrixType sum_got = localNodeSum(a);

    fail = false;
    for(int s1=0;s1<4;s1++){
      for(int c1=0;c1<3;c1++){
	for(int f1=0;f1<2;f1++){
	  for(int s2=0;s2<4;s2++){
	    for(int c2=0;c2<3;c2++){
	      for(int f2=0;f2<2;f2++){
		auto got = Reduce(sum_got(s1,s2)(c1,c2)(f1,f2) );
		auto expect = Reduce( sum_expect(s1,s2)(c1,c2)(f1,f2) );
      
		double rdiff = fabs(got.real()-expect.real());
		double idiff = fabs(got.imag()-expect.imag());
		if(rdiff > tol|| idiff > tol){
		  printf("Fail: local node reduce (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
		  fail = true;
		}
	      }
	    }
	  }
	}
      }  
    }
    if(fail) ERR.General("","","CPSmatrixField local node reduction failed\n");
  }



  //Test 3d local reduction
  {
    if(!UniqueID()){ printf("Testing local 3d reduction\n"); fflush(stdout); }
    ManagedVector<VectorMatrixType> sum_expect = localNodeSpatialSumSimple(a);
    ManagedVector<VectorMatrixType> sum_got = localNodeSpatialSum(a);

    assert(sum_expect.size() == GJP.TnodeSites());
    assert(sum_got.size() == GJP.TnodeSites());

    fail = false;
    for(int t=0;t<GJP.TnodeSites();t++){
      for(int s1=0;s1<4;s1++){
	for(int c1=0;c1<3;c1++){
	  for(int f1=0;f1<2;f1++){
	    for(int s2=0;s2<4;s2++){
	      for(int c2=0;c2<3;c2++){
		for(int f2=0;f2<2;f2++){
		  auto got = Reduce(sum_got[t](s1,s2)(c1,c2)(f1,f2) );
		  auto expect = Reduce( sum_expect[t](s1,s2)(c1,c2)(f1,f2) );
      
		  double rdiff = fabs(got.real()-expect.real());
		  double idiff = fabs(got.imag()-expect.imag());
		  if(rdiff > tol|| idiff > tol){
		    printf("Fail: local node 3d reduce (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
		    fail = true;
		  }
		}
	      }
	    }
	  }
	}  
      }
    }
    if(fail) ERR.General("","","CPSmatrixField local node 3d reduction failed\n");
  }

  //Test global sum-reduce
  {
    if(!UniqueID()){ printf("Testing global 4d sum/SIMD reduce\n"); fflush(stdout); }
    PropagatorField unit_4d(simd_dims);
    for(size_t x4d=0; x4d< unit_4d.size(); x4d++)
      unit_4d.site_ptr(x4d)->unit();
    
    typedef typename GridA2Apolicies::ScalarComplexType ScalarComplexType;
    typedef CPSspinColorFlavorMatrix<ScalarComplexType> ScalarMatrixType;
    ScalarMatrixType sum_got = globalSumReduce(unit_4d);
    ScalarMatrixType sum_expect;
    sum_expect.unit();
    sum_expect = sum_expect * GJP.VolNodeSites() * GJP.TotalNodes();

    fail = false;

    for(int s1=0;s1<4;s1++){
      for(int c1=0;c1<3;c1++){
	for(int f1=0;f1<2;f1++){
	  for(int s2=0;s2<4;s2++){
	    for(int c2=0;c2<3;c2++){
	      for(int f2=0;f2<2;f2++){
		auto got = sum_got(s1,s2)(c1,c2)(f1,f2);
		auto expect = sum_expect(s1,s2)(c1,c2)(f1,f2);
      
		double rdiff = fabs(got.real()-expect.real());
		double idiff = fabs(got.imag()-expect.imag());
		if(rdiff > tol|| idiff > tol){
		  printf("Fail: global 4d reduce (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
		  fail = true;
		}
	      }
	    }
	  }
	}
      }  
    }

    if(fail) ERR.General("","","CPSmatrixField global 4d reduction failed\n");
  }

  //Test global sum-reduce with SIMD scalar data
  {
    if(!UniqueID()){ printf("Testing global 4d sum/SIMD reduce with SIMD scalar data\n"); fflush(stdout); }
    typedef typename GridA2Apolicies::ScalarComplexType ScalarComplexType;
    ComplexField one_4d(simd_dims);
    for(size_t x4d=0; x4d< one_4d.size(); x4d++)
      vsplat( *one_4d.site_ptr(x4d), ScalarComplexType(1.0, 0.0) );
    
    ScalarComplexType got = globalSumReduce(one_4d);
    ScalarComplexType expect(1.0, 0.0);
    expect = expect * double(GJP.VolNodeSites() * GJP.TotalNodes());

    fail = false;
    
    double rdiff = fabs(got.real()-expect.real());
    double idiff = fabs(got.imag()-expect.imag());
    if(rdiff > tol|| idiff > tol){
      printf("Fail: global 4d reduce (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
      fail = true;
    }

    if(fail) ERR.General("","","CPSmatrixField global 4d reduction failed\n");
  }

  


  if(!UniqueID()){ printf("testCPSmatrixField tests passed\n"); fflush(stdout); }
}




#endif //USE_GRID



template<typename A2Apolicies>
void testFFTopt(){
  typedef typename A2Apolicies::FermionFieldType::FieldSiteType mf_Complex;
  typedef typename A2Apolicies::FermionFieldType::FieldMappingPolicy MappingPolicy;
  typedef typename A2Apolicies::FermionFieldType::FieldAllocPolicy AllocPolicy;

  typedef typename MappingPolicy::template Rebase<OneFlavorPolicy>::type OneFlavorMap;
  
  typedef CPSfield<mf_Complex,12,OneFlavorMap, AllocPolicy> FieldType;
  typedef typename FieldType::InputParamType FieldInputParamType;
  FieldInputParamType fp; setupFieldParams<FieldType>(fp);

  bool do_dirs[4] = {1,1,0,0};
  
  FieldType in(fp);
  in.testRandom();

  FieldType out1(fp);
  if(!UniqueID()) printf("FFT orig\n");
  fft(out1,in,do_dirs);

  FieldType out2(fp);
  if(!UniqueID()) printf("FFT opt\n");
  fft_opt(out2,in,do_dirs);

  assert( out1.equals(out2, 1e-8, true ) );
  printf("Passed FFT test\n");

  //Test inverse
  FieldType inv(fp);
  if(!UniqueID()) printf("FFT opt inverse\n");
  fft_opt(inv,out2,do_dirs,true);

  assert( inv.equals(in, 1e-8, true ) );
  printf("Passed FFT inverse test\n");  
}


template<typename mf_Policies>
class ComputeKtoPiPiGparityTest: public ComputeKtoPiPiGparity<mf_Policies>{
public:
  typedef typename ComputeKtoPiPiGparity<mf_Policies>::ResultsContainerType ResultsContainerType;
  typedef typename ComputeKtoPiPiGparity<mf_Policies>::SCFmat SCFmat;
  typedef typename ComputeKtoPiPiGparity<mf_Policies>::SCFmatrixField SCFmatrixField;

  static void type4_contract_test(ResultsContainerType &result, const int t_K, const int t_dis, const int thread_id, 
				  const SCFmat &part1, const SCFmat &part2_L, const SCFmat &part2_H){
    ComputeKtoPiPiGparity<mf_Policies>::type4_contract(result, t_K, t_dis, thread_id, part1, part2_L, part2_H);
  }
  static void type4_contract_test(ResultsContainerType &result, const int t_K, 
				  const SCFmatrixField &part1, const SCFmatrixField &part2_L, const SCFmatrixField &part2_H){
    ComputeKtoPiPiGparity<mf_Policies>::type4_contract(result, t_K, part1, part2_L, part2_H);
  }
};

template<typename GridA2Apolicies>
void testKtoPiPiType4FieldContraction(const double tol){
  typedef typename GridA2Apolicies::ComplexType ComplexType;
  typedef typename GridA2Apolicies::ScalarComplexType ScalarComplexType;
  typedef CPSspinColorFlavorMatrix<ComplexType> VectorMatrixType;
  typedef CPSmatrixField<VectorMatrixType> PropagatorField;

  static const int nsimd = GridA2Apolicies::ComplexType::Nsimd();
  typename PropagatorField::InputParamType simd_dims;
  PropagatorField::SIMDdefaultLayout(simd_dims,nsimd,2);
  
  PropagatorField part1(simd_dims), part2_L(simd_dims), part2_H(simd_dims);
  for(size_t x4d=0; x4d< part1.size(); x4d++){
    for(int s1=0;s1<4;s1++){
      for(int c1=0;c1<3;c1++){
	for(int f1=0;f1<2;f1++){
	  for(int s2=0;s2<4;s2++){
	    for(int c2=0;c2<3;c2++){
	      for(int f2=0;f2<2;f2++){
		{
		  ComplexType &v = (*part1.site_ptr(x4d))(s1,s2)(c1,c2)(f1,f2);
		  for(int s=0;s<nsimd;s++) v.putlane( ScalarComplexType( LRG.Urand(FOUR_D), LRG.Urand(FOUR_D) ), s );
		}		

		{
		  ComplexType &v = (*part2_L.site_ptr(x4d))(s1,s2)(c1,c2)(f1,f2);
		  for(int s=0;s<nsimd;s++) v.putlane( ScalarComplexType( LRG.Urand(FOUR_D), LRG.Urand(FOUR_D) ), s );
		}		

		{
		  ComplexType &v = (*part2_H.site_ptr(x4d))(s1,s2)(c1,c2)(f1,f2);
		  for(int s=0;s<nsimd;s++) v.putlane( ScalarComplexType( LRG.Urand(FOUR_D), LRG.Urand(FOUR_D) ), s );
		}		

	      }
	    }
	  }
	}
      }
    }
  }

  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType;

  static const int n_contract = 10; //ten type4 diagrams
  static const int con_off = 23; //index of first contraction in set
  int nthread = omp_get_max_threads();

  ResultsContainerType expect_r(n_contract, nthread);
  ResultsContainerType got_r(n_contract);

  int t_K = 1;

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  for(int t_loc=0;t_loc<GJP.TnodeSites();t_loc++){
    int t_glob = t_loc + GJP.TnodeSites()*GJP.TnodeCoor();
    int t_dis =  ComputeKtoPiPiGparityBase::modLt(t_glob - t_K, Lt);
    
    size_t vol3d = part1.size()/GJP.TnodeSites();
#pragma omp parallel for
    for(size_t x3d=0;x3d<vol3d;x3d++){
      int me = omp_get_thread_num();
      ComputeKtoPiPiGparityTest<GridA2Apolicies>::type4_contract_test(expect_r, t_K, t_dis, me,
								      *part1.site_ptr(part1.threeToFour(x3d,t_loc)),
								      *part2_L.site_ptr(part1.threeToFour(x3d,t_loc)),
								      *part2_H.site_ptr(part1.threeToFour(x3d,t_loc)));
    }
  }

  ComputeKtoPiPiGparityTest<GridA2Apolicies>::type4_contract_test(got_r, t_K, part1, part2_L, part2_H);

  got_r.nodeSum();
  expect_r.threadSum();
  expect_r.nodeSum();
  
  bool fail = false;
  for(int tdis=0;tdis<Lt;tdis++){
    for(int cidx=0; cidx<n_contract; cidx++){
      for(int gcombidx=0;gcombidx<8;gcombidx++){
	std::cout << "tdis " << tdis << " C" << cidx+con_off << " gcombidx " << gcombidx << std::endl;
	ComplexD expect = convertComplexD(expect_r(t_K,tdis,cidx,gcombidx));
	ComplexD got = convertComplexD(got_r(t_K,tdis,cidx,gcombidx));

	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: KtoPiPi type4 contract (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoPiPi type4 contract failed\n");
    

}




template<typename GridA2Apolicies>
void testKtoPiPiType4FieldFull(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting type4 full test\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorW<GridA2Apolicies> Wgrid(a2a_args, simd_dims), Whgrid(a2a_args, simd_dims);
  A2AvectorV<GridA2Apolicies> Vgrid(a2a_args, simd_dims), Vhgrid(a2a_args, simd_dims);

  Wgrid.testRandom();
  Vgrid.testRandom();

  Whgrid.testRandom();
  Vhgrid.testRandom();

  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::mf_WW mf_WW;
  std::vector<mf_WW> mf_kaon(Lt);
  for(int t=0;t<Lt;t++){
    mf_kaon[t].setup(Wgrid,Whgrid,t,t);
    mf_kaon[t].testRandom();
  }
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType;
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::MixDiagResultsContainerType MixDiagResultsContainerType;

  ResultsContainerType expect_r;
  ResultsContainerType got_r;
  MixDiagResultsContainerType expect_mix_r;
  MixDiagResultsContainerType got_mix_r;

  int tstep = 2;
  ComputeKtoPiPiGparity<GridA2Apolicies>::type4_omp(expect_r, expect_mix_r, tstep, mf_kaon, Vgrid, Vhgrid, Wgrid, Whgrid);
  ComputeKtoPiPiGparity<GridA2Apolicies>::type4_field_SIMD(got_r, got_mix_r, tstep, mf_kaon, Vgrid, Vhgrid, Wgrid, Whgrid);  

  static const int n_contract = 10; //ten type4 diagrams
  static const int con_off = 23; //index of first contraction in set
  
  bool fail = false;
  for(int t_K=0;t_K<Lt;t_K++){
    for(int tdis=0;tdis<Lt;tdis++){
      for(int cidx=0; cidx<n_contract; cidx++){
	for(int gcombidx=0;gcombidx<8;gcombidx++){
	  std::cout << "tK " << t_K << " tdis " << tdis << " C" << cidx+con_off << " gcombidx " << gcombidx << std::endl;
	  ComplexD expect = convertComplexD(expect_r(t_K,tdis,cidx,gcombidx));
	  ComplexD got = convertComplexD(got_r(t_K,tdis,cidx,gcombidx));
	  
	  double rdiff = fabs(got.real()-expect.real());
	  double idiff = fabs(got.imag()-expect.imag());
	  if(rdiff > tol|| idiff > tol){
	    printf("Fail: KtoPiPi type4 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	    fail = true;
	  }else
	    printf("Pass: KtoPiPi type4 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoPiPi type4 contract full failed\n");
    
  for(int t_K=0;t_K<Lt;t_K++){
    for(int tdis=0;tdis<Lt;tdis++){
      for(int cidx=0; cidx<2; cidx++){
	std::cout << "tK " << t_K << " tdis " << tdis << " mix4(" << cidx << ")" << std::endl;
	ComplexD expect = convertComplexD(expect_mix_r(t_K,tdis,cidx));
	ComplexD got = convertComplexD(got_mix_r(t_K,tdis,cidx));
	  
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	    printf("Fail: KtoPiPi mix4 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	    fail = true;
	}else
	  printf("Pass: KtoPiPi mix4 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
      }
    }
  }

  if(fail) ERR.General("","","KtoPiPi mix4 contract full failed\n");
}





template<typename GridA2Apolicies>
void testKtoPiPiType1FieldFull(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting type1 full test\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorW<GridA2Apolicies> Wgrid(a2a_args, simd_dims), Whgrid(a2a_args, simd_dims);
  A2AvectorV<GridA2Apolicies> Vgrid(a2a_args, simd_dims), Vhgrid(a2a_args, simd_dims);

  Wgrid.testRandom();
  Vgrid.testRandom();

  Whgrid.testRandom();
  Vhgrid.testRandom();

  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::mf_WW mf_WW;
  std::vector<mf_WW> mf_kaon(Lt);
  for(int t=0;t<Lt;t++){
    mf_kaon[t].setup(Wgrid,Whgrid,t,t);
    mf_kaon[t].testRandom();
  }
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType;
  
  std::vector<int> tsep_k_pi = {3,4};
  std::vector<ResultsContainerType> expect_r(2);
  std::vector<ResultsContainerType> got_r(2);

  int tstep = 2;
  int tsep_pion = 1;
  ThreeMomentum p_pi1(1,1,1);
  ThreeMomentum p_pi2 = -p_pi1;

  MesonFieldMomentumContainer<GridA2Apolicies> mf_pion;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_tmp(Lt);
  for(int t=0;t<Lt;t++){
    mf_pion_tmp[t].setup(Wgrid,Vgrid,t,t);
    mf_pion_tmp[t].testRandom();
  }
  mf_pion.copyAdd(p_pi1, mf_pion_tmp);
  for(int t=0;t<Lt;t++){
    mf_pion_tmp[t].testRandom();
  }
  mf_pion.copyAdd(p_pi2, mf_pion_tmp);

  ComputeKtoPiPiGparity<GridA2Apolicies>::type1_omp(expect_r.data(), tsep_k_pi, tsep_pion, tstep, 1,  p_pi1, mf_kaon, mf_pion, Vgrid, Vhgrid, Wgrid, Whgrid);  
  ComputeKtoPiPiGparity<GridA2Apolicies>::type1_field_SIMD(got_r.data(), tsep_k_pi, tsep_pion, tstep, p_pi1, mf_kaon, mf_pion, Vgrid, Vhgrid, Wgrid, Whgrid);

  static const int n_contract = 6; //ten type4 diagrams
  static const int con_off = 1; //index of first contraction in set
  
  bool fail = false;
  for(int tsep_k_pi_idx=0; tsep_k_pi_idx<2; tsep_k_pi_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<n_contract; cidx++){
	  for(int gcombidx=0;gcombidx<8;gcombidx++){
	    std::cout << "tsep_k_pi=" << tsep_k_pi[tsep_k_pi_idx] << " tK " << t_K << " tdis " << tdis << " C" << cidx+con_off << " gcombidx " << gcombidx << std::endl;
	    ComplexD expect = convertComplexD(expect_r[tsep_k_pi_idx](t_K,tdis,cidx,gcombidx));
	    ComplexD got = convertComplexD(got_r[tsep_k_pi_idx](t_K,tdis,cidx,gcombidx));
	    
	    double rdiff = fabs(got.real()-expect.real());
	    double idiff = fabs(got.imag()-expect.imag());
	    if(rdiff > tol|| idiff > tol){
	      printf("Fail: KtoPiPi type1 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      printf("Pass: KtoPiPi type1 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoPiPi type1 contract full failed\n");

}



template<typename GridA2Apolicies>
void testKtoPiPiType2FieldFull(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting type2 full test\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorW<GridA2Apolicies> Wgrid(a2a_args, simd_dims), Whgrid(a2a_args, simd_dims);
  A2AvectorV<GridA2Apolicies> Vgrid(a2a_args, simd_dims), Vhgrid(a2a_args, simd_dims);

  Wgrid.testRandom();
  Vgrid.testRandom();

  Whgrid.testRandom();
  Vhgrid.testRandom();

  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::mf_WW mf_WW;
  std::vector<mf_WW> mf_kaon(Lt);
  for(int t=0;t<Lt;t++){
    mf_kaon[t].setup(Wgrid,Whgrid,t,t);
    mf_kaon[t].testRandom();
  }
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType;
  
  std::vector<int> tsep_k_pi = {3,4};
  std::vector<ResultsContainerType> expect_r(2);
  std::vector<ResultsContainerType> got_r(2);

  int tstep = 2;
  int tsep_pion = 1;
  ThreeMomentum p_pi1(1,1,1);
  ThreeMomentum p_pi2 = -p_pi1;

  MesonFieldMomentumContainer<GridA2Apolicies> mf_pion;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_tmp(Lt);
  for(int t=0;t<Lt;t++){
    mf_pion_tmp[t].setup(Wgrid,Vgrid,t,t);
    mf_pion_tmp[t].testRandom();
  }
  mf_pion.copyAdd(p_pi1, mf_pion_tmp);
  for(int t=0;t<Lt;t++){
    mf_pion_tmp[t].testRandom();
  }
  mf_pion.copyAdd(p_pi2, mf_pion_tmp);

  std::vector<ThreeMomentum> p_pi1_all(1, p_pi1);

  ComputeKtoPiPiGparity<GridA2Apolicies>::type2_omp_v2(expect_r.data(), tsep_k_pi, tsep_pion, tstep,  p_pi1_all, mf_kaon, mf_pion, Vgrid, Vhgrid, Wgrid, Whgrid);  
  ComputeKtoPiPiGparity<GridA2Apolicies>::type2_field_SIMD(got_r.data(), tsep_k_pi, tsep_pion, tstep, p_pi1_all, mf_kaon, mf_pion, Vgrid, Vhgrid, Wgrid, Whgrid);

  static const int n_contract = 6; //ten type4 diagrams
  static const int con_off = 7; //index of first contraction in set
  
  bool fail = false;
  for(int tsep_k_pi_idx=0; tsep_k_pi_idx<2; tsep_k_pi_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<n_contract; cidx++){
	  for(int gcombidx=0;gcombidx<8;gcombidx++){
	    std::cout << "tsep_k_pi=" << tsep_k_pi[tsep_k_pi_idx] << " tK " << t_K << " tdis " << tdis << " C" << cidx+con_off << " gcombidx " << gcombidx << std::endl;
	    ComplexD expect = convertComplexD(expect_r[tsep_k_pi_idx](t_K,tdis,cidx,gcombidx));
	    ComplexD got = convertComplexD(got_r[tsep_k_pi_idx](t_K,tdis,cidx,gcombidx));
	    
	    double rdiff = fabs(got.real()-expect.real());
	    double idiff = fabs(got.imag()-expect.imag());
	    if(rdiff > tol|| idiff > tol){
	      printf("Fail: KtoPiPi type2 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      printf("Pass: KtoPiPi type2 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoPiPi type2 contract full failed\n");

}




template<typename GridA2Apolicies>
void testKtoPiPiType3FieldFull(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting type3 full test\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorW<GridA2Apolicies> Wgrid(a2a_args, simd_dims), Whgrid(a2a_args, simd_dims);
  A2AvectorV<GridA2Apolicies> Vgrid(a2a_args, simd_dims), Vhgrid(a2a_args, simd_dims);

  Wgrid.testRandom();
  Vgrid.testRandom();

  Whgrid.testRandom();
  Vhgrid.testRandom();

  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::mf_WW mf_WW;
  std::vector<mf_WW> mf_kaon(Lt);
  for(int t=0;t<Lt;t++){
    mf_kaon[t].setup(Wgrid,Whgrid,t,t);
    mf_kaon[t].testRandom();
  }
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType;
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::MixDiagResultsContainerType MixDiagResultsContainerType;  

  std::vector<int> tsep_k_pi = {3,4};
  std::vector<ResultsContainerType> expect_r(2);
  std::vector<ResultsContainerType> got_r(2);

  std::vector<MixDiagResultsContainerType> expect_mix_r(2);
  std::vector<MixDiagResultsContainerType> got_mix_r(2);

  int tstep = 2;
  int tsep_pion = 1;
  ThreeMomentum p_pi1(1,1,1);
  ThreeMomentum p_pi2 = -p_pi1;

  MesonFieldMomentumContainer<GridA2Apolicies> mf_pion;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_tmp(Lt);
  for(int t=0;t<Lt;t++){
    mf_pion_tmp[t].setup(Wgrid,Vgrid,t,t);
    mf_pion_tmp[t].testRandom();
  }
  mf_pion.copyAdd(p_pi1, mf_pion_tmp);
  for(int t=0;t<Lt;t++){
    mf_pion_tmp[t].testRandom();
  }
  mf_pion.copyAdd(p_pi2, mf_pion_tmp);

  std::vector<ThreeMomentum> p_pi1_all(1, p_pi1);

  ComputeKtoPiPiGparity<GridA2Apolicies>::type3_omp_v2(expect_r.data(), expect_mix_r.data(), tsep_k_pi, tsep_pion, tstep,  p_pi1_all, mf_kaon, mf_pion, Vgrid, Vhgrid, Wgrid, Whgrid);  
  ComputeKtoPiPiGparity<GridA2Apolicies>::type3_field_SIMD(got_r.data(), got_mix_r.data(), tsep_k_pi, tsep_pion, tstep, p_pi1_all, mf_kaon, mf_pion, Vgrid, Vhgrid, Wgrid, Whgrid);

  static const int n_contract = 10; //ten type4 diagrams
  static const int con_off = 13; //index of first contraction in set
  
  bool fail = false;
  for(int tsep_k_pi_idx=0; tsep_k_pi_idx<2; tsep_k_pi_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<n_contract; cidx++){
	  for(int gcombidx=0;gcombidx<8;gcombidx++){
	    std::cout << "tsep_k_pi=" << tsep_k_pi[tsep_k_pi_idx] << " tK " << t_K << " tdis " << tdis << " C" << cidx+con_off << " gcombidx " << gcombidx << std::endl;
	    ComplexD expect = convertComplexD(expect_r[tsep_k_pi_idx](t_K,tdis,cidx,gcombidx));
	    ComplexD got = convertComplexD(got_r[tsep_k_pi_idx](t_K,tdis,cidx,gcombidx));
	    
	    double rdiff = fabs(got.real()-expect.real());
	    double idiff = fabs(got.imag()-expect.imag());
	    if(rdiff > tol|| idiff > tol){
	      printf("Fail: KtoPiPi type3 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      printf("Pass: KtoPiPi type3 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoPiPi type3 contract full failed\n");

  for(int tsep_k_pi_idx=0; tsep_k_pi_idx<2; tsep_k_pi_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<2; cidx++){
	  std::cout << "tsep_k_pi=" << tsep_k_pi[tsep_k_pi_idx] << " tK " << t_K << " tdis " << tdis << " mix3(" << cidx << ")" << std::endl;
	  ComplexD expect = convertComplexD(expect_mix_r[tsep_k_pi_idx](t_K,tdis,cidx));
	  ComplexD got = convertComplexD(got_mix_r[tsep_k_pi_idx](t_K,tdis,cidx));
	  
	  double rdiff = fabs(got.real()-expect.real());
	  double idiff = fabs(got.imag()-expect.imag());
	  if(rdiff > tol|| idiff > tol){
	    printf("Fail: KtoPiPi mix3 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	    fail = true;
	  }else
	    printf("Pass: KtoPiPi mix3 contract full (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	}
      }
    }
  }

  if(fail) ERR.General("","","KtoPiPi mix3 contract full failed\n");
}


template<typename GridA2Apolicies>
void testKtoSigmaType12FieldFull(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting K->sigma type1/2 full test\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorW<GridA2Apolicies> Wgrid(a2a_args, simd_dims), Whgrid(a2a_args, simd_dims);
  A2AvectorV<GridA2Apolicies> Vgrid(a2a_args, simd_dims), Vhgrid(a2a_args, simd_dims);

  Wgrid.testRandom();
  Vgrid.testRandom();

  Whgrid.testRandom();
  Vhgrid.testRandom();

  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::mf_WW mf_WW;
  std::vector<mf_WW> mf_kaon(Lt);
  for(int t=0;t<Lt;t++){
    mf_kaon[t].setup(Wgrid,Whgrid,t,t);
    mf_kaon[t].testRandom();
  }
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType;
  
  std::vector<int> tsep_k_sigma = {3,4};
  std::vector<ResultsContainerType> expect_r(2);
  std::vector<ResultsContainerType> got_r(2);

  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_sigma(Lt);
  for(int t=0;t<Lt;t++){
    mf_sigma[t].setup(Wgrid,Vgrid,t,t);
    mf_sigma[t].testRandom();
  }

  ComputeKtoSigma<GridA2Apolicies> compute(Vgrid, Wgrid, Vhgrid, Whgrid, mf_kaon, tsep_k_sigma);

  compute.type12_omp(expect_r, mf_sigma);
  compute.type12_field_SIMD(got_r, mf_sigma);

  static const int n_contract = 5;  
  
  bool fail = false;
  for(int tsep_k_sigma_idx=0; tsep_k_sigma_idx<2; tsep_k_sigma_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<n_contract; cidx++){
	  for(int gcombidx=0;gcombidx<8;gcombidx++){
	    std::cout << "tsep_k_sigma=" << tsep_k_sigma[tsep_k_sigma_idx] << " tK " << t_K << " tdis " << tdis << " C" << cidx << " gcombidx " << gcombidx << std::endl;
	    ComplexD expect = convertComplexD(expect_r[tsep_k_sigma_idx](t_K,tdis,cidx,gcombidx));
	    ComplexD got = convertComplexD(got_r[tsep_k_sigma_idx](t_K,tdis,cidx,gcombidx));
	    
	    double rdiff = fabs(got.real()-expect.real());
	    double idiff = fabs(got.imag()-expect.imag());
	    if(rdiff > tol|| idiff > tol){
	      printf("Fail: KtoSigma type1/2 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      printf("Pass: KtoSigma type1/2 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoSigma type1/2 contract full failed\n");

}



template<typename GridA2Apolicies>
void testKtoSigmaType3FieldFull(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting K->sigma type3 full test\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorW<GridA2Apolicies> Wgrid(a2a_args, simd_dims), Whgrid(a2a_args, simd_dims);
  A2AvectorV<GridA2Apolicies> Vgrid(a2a_args, simd_dims), Vhgrid(a2a_args, simd_dims);

  Wgrid.testRandom();
  Vgrid.testRandom();

  Whgrid.testRandom();
  Vhgrid.testRandom();

  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::mf_WW mf_WW;
  std::vector<mf_WW> mf_kaon(Lt);
  for(int t=0;t<Lt;t++){
    mf_kaon[t].setup(Wgrid,Whgrid,t,t);
    mf_kaon[t].testRandom();
  }
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType;
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::MixDiagResultsContainerType MixDiagResultsContainerType;
  
  std::vector<int> tsep_k_sigma = {3,4};
  std::vector<ResultsContainerType> expect_r(2);
  std::vector<ResultsContainerType> got_r(2);
  std::vector<MixDiagResultsContainerType> expect_mix_r(2);
  std::vector<MixDiagResultsContainerType> got_mix_r(2);

  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_sigma(Lt);
  for(int t=0;t<Lt;t++){
    mf_sigma[t].setup(Wgrid,Vgrid,t,t);
    mf_sigma[t].testRandom();
  }

  ComputeKtoSigma<GridA2Apolicies> compute(Vgrid, Wgrid, Vhgrid, Whgrid, mf_kaon, tsep_k_sigma);

  compute.type3_omp(expect_r, expect_mix_r, mf_sigma);
  compute.type3_field_SIMD(got_r, got_mix_r, mf_sigma);

  static const int n_contract = 9;  
  
  bool fail = false;
  for(int tsep_k_sigma_idx=0; tsep_k_sigma_idx<2; tsep_k_sigma_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<n_contract; cidx++){
	  for(int gcombidx=0;gcombidx<8;gcombidx++){
	    std::cout << "tsep_k_sigma=" << tsep_k_sigma[tsep_k_sigma_idx] << " tK " << t_K << " tdis " << tdis << " C" << cidx << " gcombidx " << gcombidx << std::endl;
	    ComplexD expect = convertComplexD(expect_r[tsep_k_sigma_idx](t_K,tdis,cidx,gcombidx));
	    ComplexD got = convertComplexD(got_r[tsep_k_sigma_idx](t_K,tdis,cidx,gcombidx));
	    
	    double rdiff = fabs(got.real()-expect.real());
	    double idiff = fabs(got.imag()-expect.imag());
	    if(rdiff > tol|| idiff > tol){
	      printf("Fail: KtoSigma type3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      printf("Pass: KtoSigma type3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoSigma type3 contract full failed\n");


  for(int tsep_k_sigma_idx=0; tsep_k_sigma_idx<2; tsep_k_sigma_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<2; cidx++){
	  std::cout << "tsep_k_sigma=" << tsep_k_sigma[tsep_k_sigma_idx] << " tK " << t_K << " tdis " << tdis << " mix3(" << cidx << ")" << std::endl;
	  ComplexD expect = convertComplexD(expect_mix_r[tsep_k_sigma_idx](t_K,tdis,cidx));
	  ComplexD got = convertComplexD(got_mix_r[tsep_k_sigma_idx](t_K,tdis,cidx));
	  
	  double rdiff = fabs(got.real()-expect.real());
	  double idiff = fabs(got.imag()-expect.imag());
	  if(rdiff > tol|| idiff > tol){
	    printf("Fail: KtoSigma mix3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	    fail = true;
	  }else
	    printf("Pass: KtoSigma mix3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	}
      }
    }
  }

  if(fail) ERR.General("","","KtoSigma mix3 contract full failed\n");
}




template<typename GridA2Apolicies>
void testKtoSigmaType4FieldFull(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting K->sigma type4 full test\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorW<GridA2Apolicies> Wgrid(a2a_args, simd_dims), Whgrid(a2a_args, simd_dims);
  A2AvectorV<GridA2Apolicies> Vgrid(a2a_args, simd_dims), Vhgrid(a2a_args, simd_dims);

  Wgrid.testRandom();
  Vgrid.testRandom();

  Whgrid.testRandom();
  Vhgrid.testRandom();

  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::mf_WW mf_WW;
  std::vector<mf_WW> mf_kaon(Lt);
  for(int t=0;t<Lt;t++){
    mf_kaon[t].setup(Wgrid,Whgrid,t,t);
    mf_kaon[t].testRandom();
  }
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType;
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::MixDiagResultsContainerType MixDiagResultsContainerType;
  
  std::vector<int> tsep_k_sigma = {3,4};
  ResultsContainerType expect_r;
  ResultsContainerType got_r;
  MixDiagResultsContainerType expect_mix_r;
  MixDiagResultsContainerType got_mix_r;

  ComputeKtoSigma<GridA2Apolicies> compute(Vgrid, Wgrid, Vhgrid, Whgrid, mf_kaon, tsep_k_sigma);

  compute.type4_omp(expect_r, expect_mix_r);
  compute.type4_field_SIMD(got_r, got_mix_r);

  static const int n_contract = 9;  
  
  bool fail = false;
  for(int t_K=0;t_K<Lt;t_K++){
    for(int tdis=0;tdis<Lt;tdis++){
      for(int cidx=0; cidx<n_contract; cidx++){
	for(int gcombidx=0;gcombidx<8;gcombidx++){
	  std::cout << "tK " << t_K << " tdis " << tdis << " C" << cidx << " gcombidx " << gcombidx << std::endl;
	  ComplexD expect = convertComplexD(expect_r(t_K,tdis,cidx,gcombidx));
	  ComplexD got = convertComplexD(got_r(t_K,tdis,cidx,gcombidx));
	    
	  double rdiff = fabs(got.real()-expect.real());
	  double idiff = fabs(got.imag()-expect.imag());
	  if(rdiff > tol|| idiff > tol){
	    printf("Fail: KtoSigma type4 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	    fail = true;
	  }else
	    printf("Pass: KtoSigma type4 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	}
      }
    }
  }
  
  if(fail) ERR.General("","","KtoSigma type4 contract full failed\n");

  
  for(int t_K=0;t_K<Lt;t_K++){
    for(int tdis=0;tdis<Lt;tdis++){
      for(int cidx=0; cidx<2; cidx++){
	std::cout << "tK " << t_K << " tdis " << tdis << " mix3(" << cidx << ")" << std::endl;
	ComplexD expect = convertComplexD(expect_mix_r(t_K,tdis,cidx));
	ComplexD got = convertComplexD(got_mix_r(t_K,tdis,cidx));
	  
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: KtoSigma mix4 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}else
	  printf("Pass: KtoSigma mix4 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
      }
    }
  }

  if(fail) ERR.General("","","KtoSigma mix4 contract full failed\n");
}




#ifdef USE_GRID

//Test that the "MADWF" codepath is the same for the two different supported preconditionings
//Requires multiple hits
template<typename GridA2Apolicies>
void testMADWFprecon(const A2AArg &a2a_args, const LancArg &lanc_arg,
			     typename GridA2Apolicies::FgridGFclass &lattice, const typename SIMDpolicyBase<4>::ParamType &simd_dims, 
			     const double tol){
  if(!UniqueID()){ printf("Computing SchurOriginal evecs"); fflush(stdout); }
  GridLanczosWrapper<GridA2Apolicies> evecs_orig;
  evecs_orig.compute(lanc_arg, lattice, SchurOriginal);
  evecs_orig.toSingle();

    
  if(!UniqueID()){ printf("Computing SchurDiagTwo evecs"); fflush(stdout); }
  GridLanczosWrapper<GridA2Apolicies> evecs_diagtwo;
  evecs_diagtwo.compute(lanc_arg, lattice, SchurDiagTwo);
  evecs_diagtwo.toSingle();

  int Ls = GJP.Snodes() * GJP.SnodeSites();
  double mob_b = lattice.get_mob_b();
  double mob_c = mob_b - 1.;
  
  CGcontrols cg_con_orig;  
  cg_con_orig.CGalgorithm = AlgorithmMixedPrecisionMADWF;
  cg_con_orig.CG_tolerance = 1e-8;
  cg_con_orig.CG_max_iters = 10000;
  cg_con_orig.mixedCG_init_inner_tolerance =1e-4;
  cg_con_orig.madwf_params.Ls_inner = Ls;
  cg_con_orig.madwf_params.b_plus_c_inner = mob_b + mob_c;
  cg_con_orig.madwf_params.use_ZMobius = false;
  cg_con_orig.madwf_params.precond = SchurOriginal;

  CGcontrols cg_con_diagtwo(cg_con_orig);
  cg_con_diagtwo.madwf_params.precond = SchurDiagTwo;


  EvecInterfaceGridSinglePrec<GridA2Apolicies> eveci_orig(evecs_orig.evec_f, evecs_orig.eval, lattice, lanc_arg.mass, cg_con_orig);
  EvecInterfaceGridSinglePrec<GridA2Apolicies> eveci_diagtwo(evecs_diagtwo.evec_f, evecs_diagtwo.eval, lattice, lanc_arg.mass, cg_con_diagtwo);

  typename GridA2Apolicies::SourceFieldType::InputParamType simd_dims_3d;
  setupFieldParams<typename GridA2Apolicies::SourceFieldType>(simd_dims_3d);

  Grid::ComplexD sum_tr_orig(0), sum_tr_diagtwo(0);
  fVector<Grid::ComplexD> sum_pion2pt_orig_srcavg, sum_pion2pt_diagtwo_srcavg;

  int hits = 100;
  for(int h=0;h<hits;h++){
    A2AvectorV<GridA2Apolicies> V_orig(a2a_args,simd_dims);
    A2AvectorW<GridA2Apolicies> W_orig(a2a_args,simd_dims);
    computeVW(V_orig, W_orig, lattice, eveci_orig, evecs_orig.mass, cg_con_orig);

    A2AvectorV<GridA2Apolicies> V_diagtwo(a2a_args,simd_dims);
    A2AvectorW<GridA2Apolicies> W_diagtwo(a2a_args,simd_dims);
    computeVW(V_diagtwo, W_diagtwo, lattice, eveci_diagtwo, evecs_diagtwo.mass, cg_con_diagtwo);

    //This one doesn't seem to care much about the low modes
    {
      typedef typename mult_vv_field<GridA2Apolicies, A2AvectorV, A2AvectorW>::PropagatorField PropagatorField;
      PropagatorField prop_orig(simd_dims), prop_diagtwo(simd_dims);
      mult(prop_orig, V_orig, W_orig, false, true);
      mult(prop_diagtwo, V_diagtwo, W_diagtwo, false, true);
      
      Grid::ComplexD tr_orig = globalSumReduce(Trace(prop_orig));
      Grid::ComplexD tr_diagtwo = globalSumReduce(Trace(prop_diagtwo));
      
      sum_tr_orig = sum_tr_orig + tr_orig;
      sum_tr_diagtwo = sum_tr_diagtwo + tr_diagtwo;
      
      Grid::ComplexD avg_tr_orig = sum_tr_orig/double(h+1);
      Grid::ComplexD avg_tr_diagtwo = sum_tr_diagtwo/double(h+1);
      double reldiff_r = 2. * (avg_tr_orig.real() - avg_tr_diagtwo.real())/(avg_tr_orig.real() + avg_tr_diagtwo.real());
      double reldiff_i = 2. * (avg_tr_orig.imag() - avg_tr_diagtwo.imag())/(avg_tr_orig.imag() + avg_tr_diagtwo.imag());

      
      if(!UniqueID()){ printf("Hits %d Tr(VW^dag) Orig (%g,%g) DiagTwo (%g,%g) Rel.Diff (%g,%g)\n", h+1,
			      avg_tr_orig.real(), avg_tr_orig.imag(),
			      avg_tr_diagtwo.real(), avg_tr_diagtwo.imag(),
			      reldiff_r, reldiff_i); fflush(stdout); }
    }
    {
      //Do a pion two point function comparison
      assert(GridA2Apolicies::GPARITY == 1);
      ThreeMomentum p_quark_plus(0,0,0), p_quark_minus(0,0,0);
      for(int i=0;i<3;i++)
	if(GJP.Bc(i) == BND_CND_GPARITY){ //sum to +2
	  p_quark_plus(i) = 3;
	  p_quark_minus(i) = -1;
	}
      RequiredMomentum quark_mom;
      quark_mom.addPandMinusP({p_quark_plus,p_quark_minus});
      
      MesonFieldMomentumContainer<GridA2Apolicies> mf_orig, mf_diagtwo;
      typedef computeGparityLLmesonFields1sSumOnTheFly<GridA2Apolicies, RequiredMomentum, 15, sigma2> mfCompute;
      mfCompute::computeMesonFields(mf_orig, quark_mom, W_orig, V_orig, 1., lattice, simd_dims_3d);
      mfCompute::computeMesonFields(mf_diagtwo, quark_mom, W_diagtwo, V_diagtwo, 1., lattice, simd_dims_3d);

      fMatrix<Grid::ComplexD> pion2pt_orig, pion2pt_diagtwo;
      ComputePion<GridA2Apolicies>::compute(pion2pt_orig, mf_orig, quark_mom, 0);
      ComputePion<GridA2Apolicies>::compute(pion2pt_diagtwo, mf_diagtwo, quark_mom, 0);

      fVector<Grid::ComplexD> pion2pt_orig_srcavg = rowAverage(pion2pt_orig);
      fVector<Grid::ComplexD> pion2pt_diagtwo_srcavg = rowAverage(pion2pt_diagtwo);

      if(h==0){
	sum_pion2pt_orig_srcavg = pion2pt_orig_srcavg;
	sum_pion2pt_diagtwo_srcavg = pion2pt_diagtwo_srcavg;
      }else{
	sum_pion2pt_orig_srcavg += pion2pt_orig_srcavg;
	sum_pion2pt_diagtwo_srcavg += pion2pt_diagtwo_srcavg;
      }

      fVector<Grid::ComplexD> avg_pion2pt_orig_srcavg = sum_pion2pt_orig_srcavg;  
      avg_pion2pt_orig_srcavg /= double(h+1);
      fVector<Grid::ComplexD> avg_pion2pt_diagtwo_srcavg = sum_pion2pt_diagtwo_srcavg; 
      avg_pion2pt_diagtwo_srcavg /= double(h+1);

      if(!UniqueID()){
	for(int t=0;t<pion2pt_orig_srcavg.size();t++){
	  double reldiff_r = 2. * (avg_pion2pt_orig_srcavg(t).real() - avg_pion2pt_diagtwo_srcavg(t).real())/(avg_pion2pt_orig_srcavg(t).real() + avg_pion2pt_diagtwo_srcavg(t).real());
	  double reldiff_i = 2. * (avg_pion2pt_orig_srcavg(t).imag() - avg_pion2pt_diagtwo_srcavg(t).imag())/(avg_pion2pt_orig_srcavg(t).imag() + avg_pion2pt_diagtwo_srcavg(t).imag());
	  
	  printf("Hits %d pion2pt[%d] Orig (%g,%g) DiagTwo (%g,%g) Rel.Diff (%g,%g)\n", h+1, t,
		 avg_pion2pt_orig_srcavg(t).real(), avg_pion2pt_orig_srcavg(t).imag(),
		 avg_pion2pt_diagtwo_srcavg(t).real(), avg_pion2pt_diagtwo_srcavg(t).imag(),
		 reldiff_r, reldiff_i); 
	  fflush(stdout);
	}
      }
      
    }


  }


}

#endif //USE_GRID




#endif
