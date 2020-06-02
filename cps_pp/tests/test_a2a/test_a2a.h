#ifndef _TEST_A2A_H_
#define _TEST_A2A_H_

void testSpinFlavorMatrices(){
  std::cout << "Testing CPSsquareMatrix types" << std::endl; 
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
    typedef _mult_vMv_field_offload_v<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw, grid_vector_complex_mark> offload;
    typename offload::PropagatorField pfield(simd_dims);
    
    offload::v5(pfield, Vgrid, mf_grid, Wgrid, false, true);
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

// std::ostream & operator<<(std::ostream &os, const std::pair<int,int> &p){
//   os << "(" << p.first << ", " << p.second << ")";
//   return os;
// }

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
  cg_con_test.MADWF_Ls_inner = Ls;
  cg_con_test.MADWF_b_plus_c_inner = mob_b + mob_c;
  cg_con_test.MADWF_use_ZMobius = false;
  cg_con_test.MADWF_precond = SchurOriginal;
  
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


#endif //USE_GRID

#endif
