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


#endif
