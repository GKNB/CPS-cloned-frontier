#ifndef _TEST_A2A_H_
#define _TEST_A2A_H_

void testSpinFlavorMatrices(){
//CPSspinColorFlavorMatrix<cps::Complex> scf1;

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




#endif
