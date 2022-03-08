#pragma once

CPS_START_NAMESPACE

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
  
    //Check transpose of nested matrix
    {
      SFmat nmtr_test;
      size_t mm=0;
      for(int s1=0;s1<4;s1++)
	for(int s2=0;s2<4;s2++)
	  for(int f1=0;f1<2;f1++)
	    for(int f2=0;f2<2;f2++)
	      nmtr_test(s1,s2)(f1,f2) = cps::Complex(mm++,mm++);
      SFmat tr = nmtr_test.Transpose();
      
      for(int s1=0;s1<4;s1++)
	for(int s2=0;s2<4;s2++)
	  for(int f1=0;f1<2;f1++)
	    for(int f2=0;f2<2;f2++)
	      assert( tr(s1,s2)(f1,f2) == nmtr_test(s2,s1)(f2,f1) );
    }

    //Test scalar size deduction
    {
      typedef CPSsquareMatrix<double, 5> M1_t;
      M1_t M1;
      size_t NN_M1 = M1.nScalarType();
      assert(NN_M1 == 25);

      typedef CPSsquareMatrix<CPSsquareMatrix<double,2>, 3> M2_t;
      M2_t M2;
      size_t NN_M2 = M2.nScalarType();
      assert(NN_M2 == 9*4);
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
  }
  std::cout << "Passed CPSsquareMatrix tests" << std::endl;
}

void testCPSspinColorMatrix(){
  std::cout << "Testing CPSspinColorMatrix types" << std::endl; 

  CPSspinColorMatrix<cps::ComplexD> m1;
  m1.unit();
  for(int s1=0;s1<4;s1++){
    for(int s2=0;s2<4;s2++){
      for(int c1=0;c1<3;c1++){
	for(int c2=0;c2<3;c2++){
	  if(s1 == s2 && c1 == c2) assert(m1(s1,s2)(c1,c2) == cps::ComplexD(1.0) );
	  else assert(m1(s1,s2)(c1,c2) == cps::ComplexD(0.0) );
	}
      }
    }
  }
  
  CPSspinMatrix<cps::ComplexD> ms1 = m1.ColorTrace();
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      if(s1==s2) assert( ms1(s1,s2) == cps::ComplexD(3.0));
      else assert( ms1(s1,s2) == cps::ComplexD(0.0));


  CPScolorMatrix<cps::ComplexD> mc1 = m1.SpinTrace();
  for(int c1=0;c1<3;c1++)
    for(int c2=0;c2<3;c2++)
      if(c1==c2) assert( mc1(c1,c2) == cps::ComplexD(4.0));
      else assert( mc1(c1,c2) == cps::ComplexD(0.0));
  
  
  CPSspinColorMatrix<cps::ComplexD> m2;
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  m2(s1,s2)(c1,c2) = cps::ComplexD(s1+4*(s2+4*(c1+3*c2)));
	  
  CPSspinColorMatrix<cps::ComplexD> m2t = m2.TransposeColor();
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  assert(m2t(s1,s2)(c1,c2) == m2(s1,s2)(c2,c1));
  
  CPSspinColorMatrix<cps::ComplexD> m2t2;
  m2t2.equalsColorTranspose(m2);
   for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  assert(m2t2(s1,s2)(c1,c2) == m2(s1,s2)(c2,c1));
   
  std::cout << "Passed CPSspinColorMatrix tests" << std::endl;
}


//Test the spin-color contraction with specific spin structure producting a flavor matrix for MF computation
template<typename A2Apolicies_std>
void testFlavorMatrixSCcontractStd(double tol){
  std::cout << "Testing flavormatrix spin color contraction" << std::endl;
  typedef typename A2Apolicies_std::ScalarComplexType T;
  typedef flavorMatrixSpinColorContract<15,true,false> conj_g5;
  FlavorMatrixGeneral<T> out, expect;
  T lf0[12], lf1[12], rf0[12], rf1[12];
  for(int i=0;i<12;i++){
    _testRandom<T>::rand(lf0,12,0.5,-0.5);
    _testRandom<T>::rand(lf1,12,0.5,-0.5);
    _testRandom<T>::rand(rf0,12,0.5,-0.5);
    _testRandom<T>::rand(rf1,12,0.5,-0.5);
  }
  SCFvectorPtr<T> l(lf0,lf1,false,false), r(rf0,rf1,false,false);
  assert(l.getPtr(0)==lf0);
  assert(l.getPtr(1)==lf1);
  assert(l.isZero(0)==false);
  assert(l.isZero(1)==false);
  
  conj_g5::spinColorContract(out, l, r);

  //out(f1,f2) = \sum_c \sum_{s1,s2} l*(s1,c,f1) g5(s1,s2) r(s2,c,f2)
  CPSspinMatrix<T> g5; g5.unit(); g5.gl(-5);
  for(int f1=0;f1<2;f1++){
    for(int f2=0;f2<2;f2++){
      expect(f1,f2) = 0;
      for(int c=0;c<3;c++){
	for(int s1=0;s1<4;s1++){
	  for(int s2=0;s2<4;s2++){
	    expect(f1,f2) += conj(l(s1,c,f1))*g5(s1,s2)*r(s2,c,f2);
	  }
	}
      }
    }
  }

  assert(equals(out,expect,tol));
  
  std::cout << "Flavormatrix spin color contraction test passed" << std::endl;
}
	  
template<typename A2Apolicies_std>
void testGparityInnerProduct(double tol){
  std::cout << "Testing Gparity inner product" << std::endl;
  typedef typename A2Apolicies_std::ComplexType T;
  typedef flavorMatrixSpinColorContract<15,true,false> conj_g5;
  typedef GparityNoSourceInnerProduct<T, conj_g5> InnerProductType;
  InnerProductType inner(sigma3);

  T expect=0, out=0;
  
  T lf0[12], lf1[12], rf0[12], rf1[12];
  for(int i=0;i<12;i++){
    _testRandom<T>::rand(lf0,12,0.5,-0.5);
    _testRandom<T>::rand(lf1,12,0.5,-0.5);
    _testRandom<T>::rand(rf0,12,0.5,-0.5);
    _testRandom<T>::rand(rf1,12,0.5,-0.5);
  }
  SCFvectorPtr<T> l(lf0,lf1,false,false), r(rf0,rf1,false,false);

  //Compute   (l g5 r)[f1,f3] sigma[f1,f3]  =   (l g5 r)^T[f3,f1] sigma[f1,f3]
  //(l g5 r)(f1,f2) = \sum_c \sum_{s1,s2} l*(s1,c,f1) g5(s1,s2) r(s2,c,f2)
  FlavorMatrixGeneral<T> lg5r;
  CPSspinMatrix<T> g5; g5.unit(); g5.gl(-5);
  for(int f1=0;f1<2;f1++){
    for(int f2=0;f2<2;f2++){
      lg5r(f1,f2) = 0;
      for(int c=0;c<3;c++){
	for(int s1=0;s1<4;s1++){
	  for(int s2=0;s2<4;s2++){
	    lg5r(f1,f2) += conj(l(s1,c,f1))*g5(s1,s2)*r(s2,c,f2);
	  }
	}
      }
    }
  }

  lg5r = lg5r.transpose();
  FlavorMatrixGeneral<T> s3; s3.unit(); s3.pl(sigma3);
  
  FlavorMatrixGeneral<T> prod = lg5r*s3;
  expect = prod.Trace();

  inner(out,l,r,0,0);

  std::cout << "Got (" << out.real() <<"," << out.imag() << ") expect (" << expect.real() <<"," << expect.imag() << ")  diff (" << out.real()-expect.real() << "," << out.imag()-expect.imag() << ")" << std::endl;
  
  assert( fabs(out.real() - expect.real()) < tol );
  assert( fabs(out.imag() - expect.imag()) < tol );
  std::cout << "Gparity inner product test passed" << std::endl;
}



void testSCFmat(){
  typedef std::complex<double> ComplexType;
  typedef CPSspinMatrix<ComplexType> SpinMat;

  SpinMat one; one.unit();
  SpinMat minusone(one); minusone *= -1.;
  SpinMat zero; zero.zero();

  //Test 1.gr(i) == 1.gl(i)
  {
    std::cout << "Test 1.gr(i) == 1.gl(i)\n";
    for(int i=0;i<5;i++){
      int mu = i<4 ? i : -5;
      SpinMat a(one); a.gr(mu);
      SpinMat b(one); b.gl(mu);
      std::cout << mu << " " << a << "\n" << b << std::endl;
      assert(a==b);
    }
  }

  SpinMat gamma[6] = {one,one,one,one,one,one};
  for(int i=0;i<4;i++) gamma[i].gr(i);
  gamma[5].gr(-5);

  //Test anticommutation reln
  {
    SpinMat two(one); two *= 2.; 
    std::cout << "Test anticommutation reln\n";

    for(int mu=0;mu<4;mu++){
      for(int nu=0; nu<4; nu++){
	SpinMat c = gamma[mu]*gamma[nu] + gamma[nu]*gamma[mu];
	std::cout << mu << " " << nu << " " << c << std::endl;
	if(mu == nu) assert(c == two);
	else assert(c == zero);
      }
    }
  }

  //Test glAx
  {
    std::cout << "Testing glAx\n";
    for(int mu=0;mu<4;mu++){
      SpinMat a(one); a.glAx(mu);
      SpinMat b(one); b.gl(-5).gl(mu);
      std::cout << mu << " " << a << "\n" << b << std::endl;
      assert(a==b);
    }
  }

  //Test grAx
  {
    std::cout << "Testing grAx\n";
    for(int mu=0;mu<4;mu++){
      SpinMat a(one); a.grAx(mu);
      SpinMat b(one); b.gr(mu).gr(-5);
      std::cout << mu << " " << a << "\n" << b << std::endl;
      assert(a==b);

      SpinMat c(one); c.glAx(mu); c.grAx(mu); 
      
      std::cout << mu << " pow2 " << c << std::endl;
      assert(c == minusone);
    }
  }
}



CPS_END_NAMESPACE
