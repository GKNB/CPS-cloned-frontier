#ifndef _TEST_A2A_H_
#define _TEST_A2A_H_

#ifdef USE_GRID
#ifdef GRID_SYCL


bool equals(const fMatrix<cps::ComplexD> &l, const oneMKLmatrix<cps::ComplexD> &r, double tol = 1e-10){
  if(l.nRows() != r.rows()) return false;
  if(l.nCols() != r.cols()) return false;
  for(int i=0;i<l.nRows();i++){
    for(int j=0;j<l.nCols();j++){
      const cps::ComplexD &aa = l(i,j);
      const cps::ComplexD &bb = r(i,j);	
      if(fabs(aa.real() - bb.real()) > tol)
	return false;
      if(fabs(aa.imag() - bb.imag()) > tol)
	return false;
    }
  }
  return true;
}

void testOneMKLwrapper(){
  std::cout << "Testing oneMKLwrapper" << std::endl; 
   
  std::default_random_engine gen(1234);
  std::normal_distribution<double> dist(5.0,2.0);
  
  fMatrix<cps::ComplexD> A(100,50);
  fMatrix<cps::ComplexD> B(50,100);

  oneMKLmatrix<cps::ComplexD> A2(100,50);
  oneMKLmatrix<cps::ComplexD> B2(50,100);
  
  for(int i=0;i<100;i++){
    for(int j=0;j<50;j++){
      A(i,j) = cps::ComplexD(dist(gen), dist(gen));
      B(j,i) = cps::ComplexD(dist(gen), dist(gen));

      A2(i,j) = A(i,j);
      B2(j,i) = B(j,i);
    }
  }
  assert(!A.equals(B));

  std::cout << "Test ZGEMM" << std::endl;
  fMatrix<cps::ComplexD> C = A*B;

  oneMKLmatrix<cps::ComplexD> C2(100,100);
  mult_offload_oneMKL(C2, A2, B2);

  assert(equals(C,C2,1e-10));

  std::cout << "oneMKLwrapper tests passed" << std::endl;
}

#endif
#endif



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


template<typename T, typename ComplexClass>
struct _fmatequals{};

template<typename T>
struct _fmatequals<T,complex_double_or_float_mark>{
  static bool equals(const FlavorMatrixGeneral<T> &l, const FlavorMatrixGeneral<T> &r, double tol = 1e-10){
    for(int i=0;i<2;i++){
      for(int j=0;j<2;j++){
	const T &aa = l(i,j);
	const T &bb = r(i,j);	
	if(fabs(aa.real() - bb.real()) > tol)
	  return false;
	if(fabs(aa.imag() - bb.imag()) > tol)
	  return false;
      }
    }
    return true;
  }
};

template<typename T>
struct _fmatequals<T,grid_vector_complex_mark>{
  static bool equals(const FlavorMatrixGeneral<T> &l, const FlavorMatrixGeneral<T> &r, double tol = 1e-10){
    for(int i=0;i<2;i++){
      for(int j=0;j<2;j++){
	auto aa = Reduce(l(i,j));
	auto bb = Reduce(r(i,j));	
	if(fabs(aa.real() - bb.real()) > tol)
	  return false;
	if(fabs(aa.imag() - bb.imag()) > tol)
	  return false;
      }
    }
    return true;
  }
};

template<typename T>
inline bool equals(const FlavorMatrixGeneral<T> &l, const FlavorMatrixGeneral<T> &r, double tol = 1e-10){
  return _fmatequals<T,typename ComplexClassify<T>::type>::equals(l,r,tol);
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


//Test the getFlavorDilutedVect methods
template<typename A2Apolicies_std>
void testA2AfieldGetFlavorDilutedVect(const A2AArg &a2a_args, double tol){
  assert(a2a_args.src_width == 1);
  std::cout << "Running testA2AfieldGetFlavorDilutedVect" << std::endl;
  assert(GJP.Gparity());
  A2AvectorWfftw<A2Apolicies_std> Wf_p(a2a_args);
  Wf_p.testRandom();
  typedef typename A2Apolicies_std::FermionFieldType::FieldSiteType FieldSiteType;
  typedef typename A2Apolicies_std::ComplexType ComplexType;

  //wFFT^(i)_{sc,f}(p,t) = \rho^(i_h,i_f,i_t,i_sc)_sc(p,i_t) \delta_{i_f,f}\delta_{i_t, t}
  //wFFTP^(i_h,i_sc)_{sc,f} (p,t) = \rho^(i_h,f,t,i_sc)_sc(p,t)
  
  //For high modes getFlavorDilutedVect returns   wFFTP^(j_h,j_sc)_{sc',f'}(p,t) \delta_{f',j_f}   [cf a2a_dilutions.h]
  //i is only used for low mode indices. If i>=nl  the appropriate data is picked using i_high_unmapped
  //t is the local time

  int p3d=0;

  TimePackedIndexDilution dil_tp(a2a_args);
  StandardIndexDilution dil_full(a2a_args);
  
  int nl = Wf_p.getNl();
  int nh = dil_tp.getNmodes() - nl;
    
  for(int tpmode=nl+1;tpmode<nl+nh;tpmode++){
    std::cout << "Checking timepacked mode " << tpmode << std::endl;
    //Unmap mode index
    modeIndexSet u; dil_tp.indexUnmap(tpmode-nl,u);

    //u should contain all indices but time
    assert(u.spin_color != -1 && u.flavor != -1 && u.hit != -1 && u.time == -1);

    for(int tlcl=0;tlcl<GJP.TnodeSites();tlcl++){
      int tglb = tlcl+GJP.TnodeCoor()*GJP.TnodeSites();
      std::cout << "Mode flavor is " << u.flavor << std::endl;
      //Check it is a delta function flavor
      //note the first arg, the mode index, is ignored for mode>=nl
      SCFvectorPtr<FieldSiteType> ptr = Wf_p.getFlavorDilutedVect(nl, u, p3d, tlcl);

      //Should be non-zero for the f = u.flavor
      ComplexType const *dfa = ptr.getPtr(u.flavor);
      ComplexType const *dfb = ptr.getPtr(!u.flavor);
      
      for(int sc=0;sc<12;sc++){
	
	std::cout << "For sc=" << sc << " t=" << tglb << "  f=" << u.flavor << " : (" << dfa->real() << "," << dfa->imag() << ") f=" << !u.flavor << " (" << dfb->real() << "," << dfb->imag() << ")" << std::endl; 
    
	assert(!ptr.isZero(u.flavor));
	assert(dfa->real() != 0.0 && dfa->imag() != 0.0);    
	assert(ptr.isZero(!u.flavor));
	assert(dfb->real() == 0.0 && dfb->imag() == 0.0);    
     
	//Get elem directly
	//We have  wFFTP^(j_h,j_sc)_{sc,f}(p,t) \delta_{f,j_f}
	//         = \rho^(j_h,f,t,j_sc)_sc(p,t) \delta_{f,j_f}
	//         = wFFT(j_h,f,t,j_sc)_{sc,f}(p,t) \delta_{f,j_f}
	//Full mode (j_h,j_f,t,j_sc):
	int full_mode = nl + dil_full.indexMap(u.hit, tglb, u.spin_color, u.flavor);

	FieldSiteType ea = Wf_p.elem(full_mode, p3d, tlcl, sc, u.flavor);

	std::cout << "Using elem function : (" << ea.real() << "," << ea.imag() << ")" << std::endl; 
	
	assert(*dfa == ea);
	++dfa;
	++dfb;
      }
    }
    
  }
  std::cout << "testA2AfieldGetFlavorDilutedVect passed" << std::endl;
}

  


//This test will test the reference implementation for packed data against expectation
template<typename A2Apolicies_std>
void testMesonFieldComputeReference(const A2AArg &a2a_args, double tol){

  std::cout << "Starting test of reference implementation" << std::endl;
  typedef flavorMatrixSpinColorContract<15,true,false> SCconPol;
  typedef GparityNoSourceInnerProduct<typename A2Apolicies_std::ComplexType, SCconPol> InnerProductType;
  InnerProductType inner(sigma3);

  A2AvectorWfftw<A2Apolicies_std> Wf_p(a2a_args);
  A2AvectorVfftw<A2Apolicies_std> Vf_p(a2a_args);
  Wf_p.testRandom();
  Vf_p.testRandom();

  //Build Wf and Vf as unpacked fields
  typedef typename A2Apolicies_std::FermionFieldType FermionFieldType;
  int nv = Wf_p.getNv();
  std::vector<FermionFieldType> Wf_u(nv), Vf_u(nv);
  for(int i=0;i<nv;i++){
    Wf_p.unpackMode(Wf_u[i],i);
    Vf_p.unpackMode(Vf_u[i],i);
  }

  typedef typename A2Apolicies_std::ScalarComplexType ScalarComplexType;

  std::cout << "Computing MF using unpacked reference implementation" << std::endl;
  fMatrix<ScalarComplexType> mf_u;
  compute_simple(mf_u,Wf_u,inner,Vf_u,0);

  typedef A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorVfftw> MFtype;    
  MFtype mf_p;
  std::cout << "Computing MF using packed reference implementation" << std::endl;
  compute_simple(mf_p,Wf_p,inner,Vf_p,0);

  bool err = false;
  for(int i=0;i<nv;i++){
    for(int j=0;j<nv;j++){
      const ScalarComplexType &elem_u = mf_u(i,j);
      const ScalarComplexType &elem_p = mf_p.elem(i,j);
      if( fabs(elem_u.real() - elem_p.real()) > tol || fabs(elem_u.imag() - elem_p.imag()) > tol){
	std::cout << "Fail " << i << " " << j << " unpacked (" << elem_u.real() << "," << elem_u.imag() << ") packed (" << elem_p.real() << "," << elem_p.imag() << ") diff ("
		  << elem_u.real()-elem_p.real() << "," << elem_u.imag()-elem_p.imag() << ")" << std::endl;
	err = true;
      }else{
	//std::cout << "Success " << i << " " << j << " unpacked (" << elem_u.real() << "," << elem_u.imag() << ") packed (" << elem_p.real() << "," << elem_p.imag() << ") diff ("
	//	  << elem_u.real()-elem_p.real() << "," << elem_u.imag()-elem_p.imag() << ")" << std::endl;
      }	
    }
  }  
  assert(err == false);

  std::cout << "Passed testMesonFieldComputeSingleReference tests" << std::endl;
}



template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void compute_test_g5s3(A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &into, const A2AfieldL<mf_Policies> &l, const A2AfieldR<mf_Policies> &r, const int t){
  into.setup(l,r,t,t);
  
  if(!UniqueID()) printf("Starting TEST compute timeslice %d with %d threads\n",t, omp_get_max_threads());

  const typename mf_Policies::FermionFieldType &mode0 = l.getMode(0);
  const int size_3d = mode0.nodeSites(0)*mode0.nodeSites(1)*mode0.nodeSites(2);
  if(mode0.nodeSites(3) != GJP.TnodeSites()) ERR.General("A2AmesonField","compute","Not implemented for fields where node time dimension != GJP.TnodeSites()\n");

  int nl_l = into.getRowParams().getNl();
  int nl_r = into.getColParams().getNl();

  int nmodes_l = into.getNrows();
  int nmodes_r = into.getNcols();

  typedef typename mf_Policies::ComplexType T;
  CPSspinMatrix<T> g5; g5.unit(); g5.gl(-5);
  FlavorMatrixGeneral<T> s3; s3.unit(); s3.pl(sigma3);

  typedef flavorMatrixSpinColorContract<15,true,false> SCconPol;
  typedef GparityNoSourceInnerProduct<T, SCconPol> InnerProductType;
  InnerProductType inner(sigma3);
  
  int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();
  if(t_lcl >= 0 && t_lcl < GJP.TnodeSites()){ //if timeslice is on-node

#pragma omp parallel for
    for(int i = 0; i < nmodes_l; i++){
      T mf_accum;

      modeIndexSet i_high_unmapped; if(i>=nl_l) into.getRowParams().indexUnmap(i-nl_l,i_high_unmapped);

      for(int j = 0; j < nmodes_r; j++) {
	modeIndexSet j_high_unmapped; if(j>=nl_r) into.getColParams().indexUnmap(j-nl_r,j_high_unmapped);

	mf_accum = 0.;

	for(int p_3d = 0; p_3d < size_3d; p_3d++) {
	  SCFvectorPtr<T> lscf = l.getFlavorDilutedVect(i,i_high_unmapped,p_3d,t_lcl); //dilute flavor in-place if it hasn't been already
	  SCFvectorPtr<T> rscf = r.getFlavorDilutedVect(j,j_high_unmapped,p_3d,t_lcl);

	  FlavorMatrixGeneral<T> lg5r; lg5r.zero();
	  inner.spinColorContract(lg5r,lscf,rscf);
	  doAccum(mf_accum, TransLeftTrace(lg5r, s3) ); //still agrees
	}
	into(i,j) = mf_accum; //downcast after accumulate      
      }
    }
  }
  cps::sync();
  into.nodeSum();
}





//This test will ensure the implementation above gives the same result as the reference implementation
template<typename A2Apolicies_std>
void testMesonFieldComputePackedReference(const A2AArg &a2a_args, double tol){

  std::cout << "Starting testMesonFieldComputePackedReference: test of test_a2a implementation vs reference implementation" << std::endl;
  typedef flavorMatrixSpinColorContract<15,true,false> SCconPol;
  typedef GparityNoSourceInnerProduct<typename A2Apolicies_std::ComplexType, SCconPol> InnerProductType;
  InnerProductType inner(sigma3);

  A2AvectorWfftw<A2Apolicies_std> Wfftw_pp(a2a_args);
  A2AvectorVfftw<A2Apolicies_std> Vfftw_pp(a2a_args);
  Wfftw_pp.testRandom();
  Vfftw_pp.testRandom();
   
  typedef A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorVfftw> MFtype;

  MFtype mf_test;
  std::cout << "Computing MF using packed test implementation" << std::endl;
  compute_test_g5s3(mf_test,Wfftw_pp, Vfftw_pp, 0);

  MFtype mf_lib; 
  std::cout << "Computing MF using library implementation" << std::endl;
  mf_lib.compute(Wfftw_pp, inner, Vfftw_pp, 0);

  std::cout << "Test whether test implementation agrees with library implementation" << std::endl;
  bool val = mf_test.equals(mf_lib,tol,true);
  std::cout << "Result: " << val << std::endl;
  
  MFtype mf_ref;
  std::cout << "Computing MF using reference implementation" << std::endl;
  compute_simple(mf_ref,Wfftw_pp,inner,Vfftw_pp,0);

  std::cout << "Test whether test implementation agrees with reference implementation" << std::endl;
  assert(mf_test.equals(mf_ref, tol, true));

  std::cout << "Passed testMesonFieldComputePackedReference tests" << std::endl;
}





//This test will ensure the basic, single timeslice CPU implementation gives the same result as the reference implementation
template<typename A2Apolicies_std>
void testMesonFieldComputeSingleReference(const A2AArg &a2a_args, double tol){

  std::cout << "Starting test of single timeslice CPU implementation vs reference implementation" << std::endl;
  typedef flavorMatrixSpinColorContract<15,true,false> SCconPol;
  typedef GparityNoSourceInnerProduct<typename A2Apolicies_std::ComplexType, SCconPol> InnerProductType;
  InnerProductType inner(sigma3);

  A2AvectorWfftw<A2Apolicies_std> Wfftw_pp(a2a_args);
  A2AvectorVfftw<A2Apolicies_std> Vfftw_pp(a2a_args);
  Wfftw_pp.testRandom();
  Vfftw_pp.testRandom();
   
  typedef A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorVfftw> MFtype;
  MFtype mf; //(W^dag V) mesonfield
  std::cout << "Computing MF using single-timeslice implementation" << std::endl;
  mf.compute(Wfftw_pp, inner, Vfftw_pp, 0);

  MFtype mf_ref;
  std::cout << "Computing MF using reference implementation" << std::endl;
  compute_simple(mf_ref,Wfftw_pp,inner,Vfftw_pp,0);
 
  assert(mf.equals(mf_ref, tol, true));

  std::cout << "Passed testMesonFieldComputeSingleReference tests" << std::endl;
}

//This test will ensure the scalar version of the general (multi-timeslice) optimized MF compute gives the same result as the basic, single timeslice CPU implementation
template<typename A2Apolicies_std>
void testMesonFieldComputeSingleMulti(const A2AArg &a2a_args, double tol){

  std::cout << "Starting test of multi-timeslice optimized MF compute vs basic single timeslice CPU implementation" << std::endl;
  typedef flavorMatrixSpinColorContract<15,true,false> SCconPol;
  typedef GparityNoSourceInnerProduct<typename A2Apolicies_std::ComplexType, SCconPol> InnerProductType;
  InnerProductType inner(sigma3);

  A2AvectorWfftw<A2Apolicies_std> Wfftw_pp(a2a_args);
  A2AvectorVfftw<A2Apolicies_std> Vfftw_pp(a2a_args);
  Wfftw_pp.testRandom();
  Vfftw_pp.testRandom();
   
  typedef A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorVfftw> MFtype;
  MFtype mf; //(W^dag V) mesonfield
  std::cout << "Computing MF using single-timeslice implementation" << std::endl;
  mf.compute(Wfftw_pp, inner, Vfftw_pp, 0);

  std::vector<MFtype > mf_t;
  std::cout << "Computing MF using multi-timeslice implementation" << std::endl;
  MFtype::compute(mf_t, Wfftw_pp, inner, Vfftw_pp);
  
  assert(mf_t[0].equals(mf, tol, true));

  std::cout << "Passed testMesonFieldComputeSingleMulti tests" << std::endl;
}

//This test checks the Grid (SIMD) general MF compute against the non-SIMD
template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void testGridMesonFieldCompute(const A2AArg &a2a_args, const int nthreads, const double tol){
 
#ifdef USE_GRID
  std::cout << "Starting MF contraction test comparing Grid SIMD vs non-SIMD multi-timeslice implementations\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorWfftw<ScalarA2Apolicies> W(a2a_args);
  A2AvectorVfftw<ScalarA2Apolicies> V(a2a_args);
    
  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);
  
  std::vector<A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_grid;
  
  typedef typename GridA2Apolicies::ComplexType grid_Complex;
  typedef typename ScalarA2Apolicies::ComplexType mf_Complex;

  typedef typename GridA2Apolicies::ScalarComplexType Ctype;
  typedef typename Ctype::value_type Ftype;
  
  ThreeDSIMDPolicy<OneFlavorPolicy>::ParamType simd_dims_3d;
  ThreeDSIMDPolicy<OneFlavorPolicy>::SIMDdefaultLayout(simd_dims_3d,nsimd);

  typedef typename GridA2Apolicies::SourcePolicies GridSrcPolicy;    
  int p[3] = {1,1,1};
  if(!UniqueID()){ printf("Generating Grid source\n"); fflush(stdout); }
  A2AflavorProjectedExpSource<GridSrcPolicy> src_grid(2.0,p,simd_dims_3d);
  typedef SCFspinflavorInnerProduct<15,typename GridA2Apolicies::ComplexType,A2AflavorProjectedExpSource<GridSrcPolicy> > GridInnerProduct;
  if(!UniqueID()){ printf("Generating Grid inner product\n"); fflush(stdout); }
  GridInnerProduct mf_struct_grid(sigma3,src_grid);


  //typedef GparityNoSourceInnerProduct<typename GridA2Apolicies::ComplexType, flavorMatrixSpinColorContract<15,true,false> > GridInnerProduct;
  //GridInnerProduct mf_struct_grid(sigma3);
  
  if(!UniqueID()){ printf("Generating std. source\n"); fflush(stdout); }
  A2AflavorProjectedExpSource<> src(2.0,p);
  typedef SCFspinflavorInnerProduct<15,typename ScalarA2Apolicies::ComplexType,A2AflavorProjectedExpSource<> > StdInnerProduct;
  if(!UniqueID()){ printf("Generating std. inner product\n"); fflush(stdout); }
  StdInnerProduct mf_struct(sigma3,src);

  if(!UniqueID()){ printf("Generating random fields\n"); fflush(stdout); }
  W.testRandom();
  V.testRandom();
  if(!UniqueID()){ printf("Importing fields to Grid\n"); fflush(stdout); }
  Wgrid.importFields(W);
  Vgrid.importFields(V);
  
#ifndef GPU_VEC
  //Original Grid implementation
  {
    if(!UniqueID()){ printf("Grid non-GPU version\n"); fflush(stdout); }
    typedef typename std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> >::allocator_type Allocator;
    typedef SingleSrcVectorPoliciesSIMD<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,GridInnerProduct> VectorPolicies;
    mfComputeGeneral<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw, GridInnerProduct, VectorPolicies> cg;
    cg.compute(mf_grid,Wgrid,mf_struct_grid,Vgrid, true);
  }
#else
  {
    if(!UniqueID()){ printf("Grid GPU version\n"); fflush(stdout); }
    typedef typename std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> >::allocator_type Allocator;
    typedef SingleSrcVectorPoliciesSIMDoffload<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,GridInnerProduct> VectorPolicies;
    mfComputeGeneralOffload<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw, GridInnerProduct, VectorPolicies> cg;
    cg.compute(mf_grid,Wgrid,mf_struct_grid,Vgrid, true);
  }
#endif

  if(!UniqueID()){ printf("Starting scalar version\n"); fflush(stdout); }
  A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf,W,mf_struct,V);


  if(!UniqueID()){ printf("Comparing\n"); fflush(stdout); }
  bool fail = false;
  for(int t=0;t<mf.size();t++){
    for(int i=0;i<mf[t].size();i++){
      const Ctype& gd = mf_grid[t].ptr()[i];
      const Ctype& cp = mf[t].ptr()[i];
      Ftype rdiff = fabs(gd.real()-cp.real());
      Ftype idiff = fabs(gd.imag()-cp.imag());
      if(rdiff > tol|| idiff > tol){
	printf("Fail: t %d idx %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",t, i,gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
	fail = true;
      }
    }
  }
  if(fail) ERR.General("","","Standard vs Grid implementation test failed\n");
  else if(!UniqueID()){ printf("Passed MF contraction test\n"); fflush(stdout); }
#endif
}




template<typename GridA2Apolicies>
void testGridMultiSourceMesonFieldCompute(const A2AArg &a2a_args, const int nthreads, const double tol){
 #ifdef USE_GRID
  std::cout << "Starting multi-source MF contraction test\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorWfftw<GridA2Apolicies> W(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> V(a2a_args, simd_dims);

  W.testRandom();
  V.testRandom();
  
  typedef typename GridA2Apolicies::ComplexType ComplexType;
  typedef typename GridA2Apolicies::ScalarComplexType ScalarComplexType;
  
  ThreeDSIMDPolicy<OneFlavorPolicy>::ParamType simd_dims_3d;
  ThreeDSIMDPolicy<OneFlavorPolicy>::SIMDdefaultLayout(simd_dims_3d,nsimd);

  typedef typename GridA2Apolicies::SourcePolicies SrcPolicy;    
  int p[3] = {1,1,1};
  A2AflavorProjectedExpSource<SrcPolicy> exp_src(2.0,p,simd_dims_3d);
  A2AflavorProjectedPointSource<SrcPolicy> pnt_src(simd_dims_3d);

  typedef Elem<A2AflavorProjectedExpSource<SrcPolicy>, Elem<A2AflavorProjectedPointSource<SrcPolicy>, ListEnd> > Sources;
  A2AmultiSource<Sources> multi_src;
  multi_src.template getSource<0>().setup(2.0,p,simd_dims_3d);
  multi_src.template getSource<1>().setup(simd_dims_3d);

  typedef SCFspinflavorInnerProduct<15, ComplexType, A2AflavorProjectedExpSource<SrcPolicy> > ExpSrcInnerProduct;
  typedef SCFspinflavorInnerProduct<15, ComplexType, A2AflavorProjectedPointSource<SrcPolicy> > PointSrcInnerProduct;
  typedef SCFspinflavorInnerProduct<15, ComplexType, A2AmultiSource<Sources>  > MultiSrcInnerProduct;
  
  if(!UniqueID()){ printf("Generating inner products\n"); fflush(stdout); }
  ExpSrcInnerProduct inner_product_exp(sigma3, exp_src);
  PointSrcInnerProduct inner_product_pnt(sigma3, pnt_src);
  MultiSrcInnerProduct inner_product_multi(sigma3, multi_src);
    
  typedef A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>  MFtype;
  std::vector<MFtype> mf_exp;
  std::vector<MFtype> mf_pnt;

  std::vector<MFtype> mf_exp_m;
  std::vector<MFtype> mf_pnt_m;
  std::vector< std::vector<MFtype>* > mf_m = { &mf_exp_m, &mf_pnt_m };
    
  if(!UniqueID()){ printf("Exp source computation\n"); fflush(stdout); }
  MFtype::compute(mf_exp, W, inner_product_exp, V);
  if(!UniqueID()){ printf("Point source computation\n"); fflush(stdout); }
  MFtype::compute(mf_pnt, W, inner_product_pnt, V);
  if(!UniqueID()){ printf("Multi source computation\n"); fflush(stdout); }    
  MFtype::compute(mf_m, W, inner_product_multi, V);

  int Lt=GJP.Tnodes()*GJP.TnodeSites();
  for(int t=0;t<Lt;t++){
    std::cout << "Checking exp source t=" << t << std::endl;
    assert(mf_exp[t].equals(mf_exp_m[t], tol, true));
    std::cout << "Checking point source t=" << t << std::endl;
    assert(mf_pnt[t].equals(mf_pnt_m[t], tol, true));
  }
  if(!UniqueID()){ printf("Passed multi-source MF contraction test\n"); fflush(stdout); }
#endif
}




template<typename GridA2Apolicies>
void testGridShiftMultiSourceMesonFieldCompute(const A2AArg &a2a_args, const int nthreads, const double tol){
 #ifdef USE_GRID
  std::cout << "Starting shifted multi-source MF contraction test\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorWfftw<GridA2Apolicies> W(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> V(a2a_args, simd_dims);

  W.testRandom();
  V.testRandom();
  
  typedef typename GridA2Apolicies::ComplexType ComplexType;
  typedef typename GridA2Apolicies::ScalarComplexType ScalarComplexType;
  
  ThreeDSIMDPolicy<OneFlavorPolicy>::ParamType simd_dims_3d;
  ThreeDSIMDPolicy<OneFlavorPolicy>::SIMDdefaultLayout(simd_dims_3d,nsimd);

  typedef typename GridA2Apolicies::SourcePolicies SrcPolicy;    
  int p[3] = {1,1,1};
  A2AflavorProjectedExpSource<SrcPolicy> exp_src(2.0,p,simd_dims_3d);
  A2AflavorProjectedPointSource<SrcPolicy> pnt_src(simd_dims_3d);

  typedef Elem<A2AflavorProjectedExpSource<SrcPolicy>, Elem<A2AflavorProjectedPointSource<SrcPolicy>, ListEnd> > Sources;
  A2AmultiSource<Sources> multi_src;
  multi_src.template getSource<0>().setup(2.0,p,simd_dims_3d);
  multi_src.template getSource<1>().setup(simd_dims_3d);

  typedef flavorMatrixSpinColorContract<15,true,false> ContractPolicy;
  typedef GparitySourceShiftInnerProduct<ComplexType, A2AflavorProjectedExpSource<SrcPolicy>, ContractPolicy> ExpSrcInnerProduct;
  typedef GparitySourceShiftInnerProduct<ComplexType, A2AflavorProjectedPointSource<SrcPolicy>, ContractPolicy> PointSrcInnerProduct;
  typedef GparitySourceShiftInnerProduct<ComplexType, A2AmultiSource<Sources>, ContractPolicy> MultiSrcInnerProduct;


  std::vector< std::vector<int> > shifts = {  {1,1,1}  };
  std::vector< std::vector<int> > shifts_multi = {  {1,1,1} };
  
  if(!UniqueID()){ printf("Generating inner products\n"); fflush(stdout); }
  ExpSrcInnerProduct inner_product_exp(sigma3, exp_src);
  PointSrcInnerProduct inner_product_pnt(sigma3, pnt_src);
  MultiSrcInnerProduct inner_product_multi(sigma3, multi_src);

  inner_product_pnt.setShifts(shifts);
  inner_product_exp.setShifts(shifts);
  inner_product_multi.setShifts(shifts_multi);
  
  typedef A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>  MFtype;
  std::vector<MFtype> mf_exp;
  std::vector< std::vector<MFtype>* > mf_exp_a = { &mf_exp };
  
  std::vector<MFtype> mf_pnt;
  std::vector< std::vector<MFtype>* > mf_pnt_a = { &mf_pnt };

  std::vector<MFtype> mf_exp_m;
  std::vector<MFtype> mf_pnt_m;
  std::vector< std::vector<MFtype>* > mf_m = { &mf_exp_m, &mf_pnt_m };
    
  if(!UniqueID()){ printf("Exp source computation\n"); fflush(stdout); }
  MFtype::compute(mf_exp_a, W, inner_product_exp, V); 
  if(!UniqueID()){ printf("Point source computation\n"); fflush(stdout); }
  MFtype::compute(mf_pnt_a, W, inner_product_pnt, V);
  if(!UniqueID()){ printf("Multi source computation\n"); fflush(stdout); }    
  MFtype::compute(mf_m, W, inner_product_multi, V);

  int Lt=GJP.Tnodes()*GJP.TnodeSites();
  for(int t=0;t<Lt;t++){
    std::cout << "Checking exp source t=" << t << std::endl;
    assert(mf_exp[t].equals(mf_exp_m[t], tol, true));
    std::cout << "Checking point source t=" << t << std::endl;
    assert(mf_pnt[t].equals(mf_pnt_m[t], tol, true));
  }
  if(!UniqueID()){ printf("Passed shifted multi-source MF contraction test\n"); fflush(stdout); }
#endif
}


template<typename GridA2Apolicies>
void testGridGetTwistedFFT(const A2AArg &a2a_args, const int nthreads, const double tol){
#ifdef USE_GRID
  std::cout << "Starting testGridGetTwistedFFT\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorWfftw<GridA2Apolicies> W1(a2a_args, simd_dims);
  A2AvectorWfftw<GridA2Apolicies> W2(a2a_args, simd_dims);
  
  W1.testRandom();
  W2.testRandom();
  
  int pbase[3];
  for(int i=0;i<3;i++) pbase[i] = GJP.Bc(i) == BND_CND_GPARITY ? 1 : 0;

  ThreeMomentum pp(pbase);
  ThreeMomentum pm = -pp;

  A2AvectorWfftw<GridA2Apolicies> Wtmp(a2a_args, simd_dims);

  //Does copy work?
  std::cout << "Testing mode copy" << std::endl;
  for(int i=0;i<W1.getNmodes();i++)
    Wtmp.getMode(i) = W1.getMode(i);
      
  for(int i=0;i<W1.getNmodes();i++){
    assert(Wtmp.getMode(i).equals(W1.getMode(i),1e-12,true));
  }
 
  std::cout << "Testing copy" << std::endl;
  Wtmp = W1;
  for(int i=0;i<W1.getNmodes();i++){
    assert(Wtmp.getMode(i).equals(W1.getMode(i),1e-12,true));
  }

  std::cout << "Testing getTwistedFFT" << std::endl;
  
  Wtmp.getTwistedFFT(pp.ptr(), &W1, &W2);  //base_p, base_m

  //Should be same as W1 without a shift
  for(int i=0;i<W1.getNmodes();i++){
    assert(Wtmp.getMode(i).equals(W1.getMode(i),1e-12,true));
  }
  std::cout << "testGridGetTwistedFFT passed" << std::endl;
#endif
}
  

  

template<typename GridA2Apolicies>
void testGridMesonFieldComputeManySimple(A2AvectorV<GridA2Apolicies> &V, A2AvectorW<GridA2Apolicies> &W,
					 const A2AArg &a2a_args,
					 typename GridA2Apolicies::FgridGFclass &lattice,
					 typename SIMDpolicyBase<3>::ParamType simd_dims_3d,
					 typename SIMDpolicyBase<4>::ParamType simd_dims,
					 const double tol){
#ifdef USE_GRID
  std::cout << "Starting testGridMesonFieldComputeManySimple" << std::endl;

  //Define the inner product
  int p[3] = {1,1,1};
  typedef typename GridA2Apolicies::SourcePolicies SrcPolicy;
  typedef A2AflavorProjectedExpSource<SrcPolicy> SrcType;
  SrcType src(2.0,p,simd_dims_3d); //note at present the value of p is unimportant
  typedef SCFspinflavorInnerProduct<15,typename GridA2Apolicies::ComplexType,SrcType> InnerProductType;
  InnerProductType inner(sigma3, src);
  
  //Define the storage type for ComputeMany
  typedef GparityFlavorProjectedBasicSourceStorage<GridA2Apolicies, InnerProductType> StorageType;
  StorageType storage(inner);
  
  //We want a pion with momentum +2 in each G-parity direction
  int pbase[3];
  for(int i=0;i<3;i++) pbase[i] = GJP.Bc(i) == BND_CND_GPARITY ? 1 : 0;
  
  ThreeMomentum p_v(pbase), p_wdag(pbase);
  ThreeMomentum p_w = -p_wdag; //computemany takes the momentum of the W, not Wdag
  
  storage.addCompute(0,0,p_w,p_v);

  //Compute externally
  std::cout << "Starting hand computation" << std::endl;
  A2AvectorWfftw<GridA2Apolicies> Wfft(a2a_args,simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vfft(a2a_args,simd_dims);
  
  Wfft.gaugeFixTwistFFT(W,p_w.ptr(),lattice);
  Vfft.gaugeFixTwistFFT(V,p_v.ptr(),lattice);
  inner.getSource()->setMomentum(p_v.ptr());

  typedef A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> MFType;
  typedef std::vector<MFType> MFVec;
  MFVec mf_ref;
  MFType::compute(mf_ref, Wfft, inner, Vfft);
 
  //Use computemany
  std::cout << "Starting ComputeMany computation" << std::endl;
  typedef ComputeMesonFields<GridA2Apolicies,StorageType> ComputeType;
  typename ComputeType::WspeciesVector Wv = {&W};
  typename ComputeType::VspeciesVector Vv = {&V};
  ComputeType::compute(storage, Wv,Vv, lattice);
  const MFVec &mf = storage.getMf(0);  


  int Lt=GJP.Tnodes()*GJP.TnodeSites();
  for(int t=0;t<Lt;t++){
    std::cout << "Checking t=" << t << std::endl;
    assert(mf[t].equals(mf_ref[t], tol, true));
  }

  std::cout << "testGridMesonFieldComputeManySimple passeed" << std::endl;
#endif
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

    auto fivedin_view = fivedin.View(Grid::CpuRead);
    Grid::peekLocalSite(v1,fivedin_view,test_site);
    fivedin_view.ViewClose();

    auto fivedcb_view = fivedcb.View(Grid::CpuRead);
    Grid::peekLocalSite(v2,fivedcb_view,test_site);
    fivedcb_view.ViewClose();      

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
		     A2AvectorV<A2Apolicies_grid> &V_grid,
		     double tol){

  int nl = V_std.getNl();
  int nh = V_std.getNh();
  
  for(int i=0;i<nl;i++){
    double nrm_grid = V_grid.getVl(i).norm2();
    double nrm_std = V_std.getVl(i).norm2();
    double diff = nrm_grid - nrm_std;
    std::cout << "vl " << i << " grid " << nrm_grid << " std " << nrm_std << " diff " << diff << std::endl;
    assert(fabs(diff) < tol);
  }
  for(int i=0;i<nh;i++){
    double nrm_grid = V_grid.getVh(i).norm2();
    double nrm_std = V_std.getVh(i).norm2();
    double diff = nrm_grid - nrm_std;
    std::cout << "vh " << i << " grid " << nrm_grid << " std " << nrm_std << " diff " << diff << std::endl;
    assert(fabs(diff) < tol);
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
  std::cout << "Starting testPionContractionGridStd" << std::endl;
  StandardPionMomentaPolicy momenta;
  MesonFieldMomentumContainer<A2Apolicies_std> mf_ll_con_std;
  MesonFieldMomentumContainer<A2Apolicies_grid> mf_ll_con_grid;

  std::cout << "Computing non-SIMD meson fields" << std::endl;
  computeGparityLLmesonFields1s<A2Apolicies_std,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_std,momenta,W_std,V_std,2.0,lattice);
  std::cout << "Computing SIMD meson fields" << std::endl;
  computeGparityLLmesonFields1s<A2Apolicies_grid,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_grid,momenta,W_grid,V_grid,2.0,lattice,simd_dims_3d);

  std::cout << "Computing non-SIMD pion 2pt" << std::endl;
  fMatrix<typename A2Apolicies_std::ScalarComplexType> fmat_std;
  ComputePion<A2Apolicies_std>::compute(fmat_std, mf_ll_con_std, momenta, 0);

  std::cout << "Computing SIMD pion 2pt" << std::endl;
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
  std::cout << "testPionContractionGridStd passed" << std::endl;
}


template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testKaonContractionGridStd(A2AvectorV<A2Apolicies_std> &V_std, A2AvectorW<A2Apolicies_std> &W_std,
				A2AvectorV<A2Apolicies_grid> &V_grid, A2AvectorW<A2Apolicies_grid> &W_grid,
				typename A2Apolicies_grid::FgridGFclass &lattice,
				typename SIMDpolicyBase<3>::ParamType simd_dims_3d,
				double tol){
  std::cout << "Starting testKaonContractionGridStd" << std::endl;

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
  std::cout << "testKaonContractionGridStd passed" << std::endl;
}



template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testPiPiContractionGridStd(A2AvectorV<A2Apolicies_std> &V_std, A2AvectorW<A2Apolicies_std> &W_std,
				A2AvectorV<A2Apolicies_grid> &V_grid, A2AvectorW<A2Apolicies_grid> &W_grid,
				typename A2Apolicies_grid::FgridGFclass &lattice,
				typename SIMDpolicyBase<3>::ParamType simd_dims_3d,
				double tol){
  std::cout << "Starting testPiPiContractionGridStd" << std::endl;
  
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

    std::cout << "testPiPiContractionGridStd passed" << std::endl;
  }
}


//Both should be scalar matrices
template<typename A2Apolicies_1, typename A2Apolicies_2, template<typename> class L, template<typename> class R>
bool compare(const std::vector<A2AmesonField<A2Apolicies_1,L,R> > &M1, const std::vector<A2AmesonField<A2Apolicies_2,L,R> > &M2, double tol){
  if(M1.size() != M2.size()){
    std::cout << "Fail: time vector size mismatch" << std::endl;
    return false;
  }
  for(int t=0;t<M1.size();t++){
    if(M1[t].getNrows() != M2[t].getNrows() ||
       M1[t].getNcols() != M2[t].getNcols() ){
      std::cout << "Fail: matrix size mismatch" << std::endl;
      return false;
    }
    if(M1[t].getRowTimeslice() != M2[t].getRowTimeslice() || 
       M1[t].getColTimeslice() != M2[t].getColTimeslice() ){
      std::cout << "Fail: matrix timeslice mismatch" << std::endl;
      return false;      
    }
      
    for(int i=0;i<M1[t].getNrows();i++){
      for(int j=0;j<M1[t].getNcols();j++){
	auto v1 = M1[t](i,j);
	auto v2 = M2[t](i,j);
	if(fabs(v1.real() - v2.real()) > tol ||
	   fabs(v1.imag() - v2.imag()) > tol){
	  std::cout << "Fail " << i << " " << j << " :  (" << v1.real() << "," << v1.imag() << ")  (" << v2.real() << "," << v2.imag() << ")  diff (" << v1.real()-v2.real() << "," << v1.imag()-v2.imag() << ")" << std::endl;
	  return false;
	}
      }
    }
  }
  return true;
}

//Both should be scalar matrices
template<typename A2Apolicies_1, typename A2Apolicies_2, template<typename> class L, template<typename> class R>
void copy(std::vector<A2AmesonField<A2Apolicies_1,L,R> > &Mout, const std::vector<A2AmesonField<A2Apolicies_2,L,R> > &Min){
  assert(Mout.size() == Min.size());
  for(int t=0;t<Min.size();t++){
    assert(Mout[t].getNrows() == Min[t].getNrows() && Min[t].getNcols() == Min[t].getNcols());
    assert(Mout[t].getRowTimeslice() == Min[t].getRowTimeslice() && Mout[t].getColTimeslice() == Min[t].getColTimeslice());
    for(int i=0;i<Min[t].getNrows();i++){
      for(int j=0;j<Min[t].getNcols();j++){
	Mout[t](i,j) = Min[t](i,j);
      }
    }
  }
}

template<typename ResultsTypeA, typename ResultsTypeB>
bool compareKtoPiPi(const ResultsTypeA &result_A, const std::string &Adescr,
		    const ResultsTypeB &result_B, const std::string &Bdescr,
		    const std::string &descr, double tol){
  if(result_A.nElementsTotal() != result_B.nElementsTotal()){
    std::cout << descr << " fail: size mismatch " << result_A.nElementsTotal() << " " << result_B.nElementsTotal() << std::endl;
    return false;
  }
    
  bool fail = false;
  for(int i=0;i<result_A.nElementsTotal();i++){
    std::complex<double> val_A = convertComplexD(result_A[i]);
    std::complex<double> val_B = convertComplexD(result_B[i]);
    
    double rdiff = fabs(val_A.real()-val_B.real());
    double idiff = fabs(val_A.imag()-val_B.imag());
    if(rdiff > tol|| idiff > tol){
      printf("!!!Fail: %s elem %d %s (%g,%g) %s (%g,%g) Diff (%g,%g)\n", descr.c_str(), i,
	     Adescr.c_str(), val_A.real(),val_A.imag(),
	     Bdescr.c_str(), val_B.real(),val_B.imag(),
	     val_A.real()-val_B.real(), val_A.imag()-val_B.imag());
      fail = true;
    }//else printf("Pass: Type1 elem %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",i, val_grid.real(),val_grid.imag(), val_std.real(),val_std.imag(), val_std.real()-val_grid.real(), val_std.imag()-val_grid.imag());
  }
  return !fail;
}


template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testKtoPiPiContractionGridStd(A2AvectorV<A2Apolicies_std> &V_std, A2AvectorW<A2Apolicies_std> &W_std,
				   A2AvectorV<A2Apolicies_grid> &V_grid, A2AvectorW<A2Apolicies_grid> &W_grid,
				   typename A2Apolicies_grid::FgridGFclass &lattice,
				   typename SIMDpolicyBase<3>::ParamType simd_dims_3d,
				   double tol){
  std::cout << "Starting testKtoPiPiContractionGridStd" << std::endl;

#if 0
  ThreeMomentum p_plus( GJP.Bc(0)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(1)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(2)==BND_CND_GPARITY? 1 : 0 );
  ThreeMomentum p_minus = -p_plus;

  ThreeMomentum p_pi_plus = p_plus * 2;
  
  StandardPionMomentaPolicy momenta;
  MesonFieldMomentumContainer<A2Apolicies_std> mf_ll_con_std;
  MesonFieldMomentumContainer<A2Apolicies_grid> mf_ll_con_grid;

  std::cout << "testKtoPiPiContractionGridStd computing LL meson fields standard calc" << std::endl;
  computeGparityLLmesonFields1s<A2Apolicies_std,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_std,momenta,W_std,V_std,2.0,lattice);

  std::cout << "testKtoPiPiContractionGridStd computing LL meson fields Grid calc" << std::endl;
  computeGparityLLmesonFields1s<A2Apolicies_grid,StandardPionMomentaPolicy,15,sigma3>::computeMesonFields(mf_ll_con_grid,momenta,W_grid,V_grid,2.0,lattice,simd_dims_3d);

  std::cout << "testKtoPiPiContractionGridStd comparing LL MF momenta" << std::endl;
  std::vector<ThreeMomentum> mom_std; mf_ll_con_std.getMomenta(mom_std);
  std::vector<ThreeMomentum> mom_grid; mf_ll_con_grid.getMomenta(mom_grid);

  assert(mom_std.size() == mom_grid.size());
  for(int pp=0;pp<mom_std.size();pp++)
    assert(mom_std[pp] == mom_grid[pp]);

  std::cout << "testKtoPiPiContractionGridStd comparison of LL MF momenta passed" << std::endl;

  std::cout << "testKtoPiPiContractionGridStd comparing LL meson fields between standard and Grid implementations" << std::endl;
  for(int pp=0;pp<mom_std.size();pp++){
    const auto &MFvec_std = mf_ll_con_std.get(mom_std[pp]);
    const auto &MFvec_grid = mf_ll_con_grid.get(mom_grid[pp]);
    std::cout << "Comparing meson fields for momentum " << mom_std[pp] << std::endl;
    assert(compare(MFvec_std, MFvec_grid, tol));
  }
  std::cout << "testKtoPiPiContractionGridStd comparison of LL meson fields between standard and Grid implementations passed" << std::endl;
 
  StandardLSWWmomentaPolicy ww_mom;

  std::cout << "testKtoPiPiContractionGridStd computing WW fields standard calc" << std::endl;
  std::vector<A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww_std;
  ComputeKtoPiPiGparity<A2Apolicies_std>::generatelsWWmesonfields(mf_ls_ww_std,W_std,W_std, ww_mom, 2.0,lattice);
  
  std::cout << "testKtoPiPiContractionGridStd computing WW fields Grid calc" << std::endl;
  std::vector<A2AmesonField<A2Apolicies_grid,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww_grid;
  ComputeKtoPiPiGparity<A2Apolicies_grid>::generatelsWWmesonfields(mf_ls_ww_grid,W_grid,W_grid, ww_mom, 2.0,lattice, simd_dims_3d);

  assert(mf_ls_ww_std.size() == mf_ls_ww_grid.size());
  
  std::cout << "testKtoPiPiContractionGridStd comparing WW fields" << std::endl;
  assert(compare(mf_ls_ww_std, mf_ls_ww_grid, tol));
  
  std::cout << "testKtoPiPiContractionGridStd comparison of WW fields passed" << std::endl;
  
  mf_ll_con_grid.printMomenta(std::cout);
#else

  const int nsimd = A2Apolicies_grid::ComplexType::Nsimd();      
 
  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  std::vector<A2AmesonField<A2Apolicies_grid,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww_grid(Lt);
  std::vector<A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww_std(Lt);
  
  for(int t=0;t<Lt;t++){
    mf_ls_ww_grid[t].setup(W_grid,W_grid,t,t);
    mf_ls_ww_grid[t].testRandom();

    mf_ls_ww_std[t].setup(W_std,W_std,t,t);
  }
  copy(mf_ls_ww_std, mf_ls_ww_grid);
  assert(compare(mf_ls_ww_std, mf_ls_ww_grid, 1e-12));
  
  ThreeMomentum p_pi_plus(2,2,2);
  ThreeMomentum p_pi_minus = -p_pi_plus;

  MesonFieldMomentumContainer<A2Apolicies_grid> mf_ll_con_grid;
  std::vector<A2AmesonField<A2Apolicies_grid,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_grid_tmp(Lt);

  MesonFieldMomentumContainer<A2Apolicies_std> mf_ll_con_std;
  std::vector<A2AmesonField<A2Apolicies_std,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_std_tmp(Lt);

  for(int t=0;t<Lt;t++){
    mf_pion_grid_tmp[t].setup(W_grid,V_grid,t,t);
    mf_pion_grid_tmp[t].testRandom();

    mf_pion_std_tmp[t].setup(W_std,V_std,t,t);
  }
  copy(mf_pion_std_tmp, mf_pion_grid_tmp);
  
  mf_ll_con_grid.copyAdd(p_pi_plus, mf_pion_grid_tmp);
  mf_ll_con_std.copyAdd(p_pi_plus, mf_pion_std_tmp);
  
  for(int t=0;t<Lt;t++){
    mf_pion_grid_tmp[t].testRandom();
  }
  copy(mf_pion_std_tmp, mf_pion_grid_tmp);
  
  mf_ll_con_grid.copyAdd(p_pi_minus, mf_pion_grid_tmp);
  mf_ll_con_std.copyAdd(p_pi_minus, mf_pion_std_tmp);

  assert(mf_ll_con_grid.get(p_pi_plus).size() ==  Lt);
  assert(mf_ll_con_grid.get(p_pi_minus).size() ==  Lt);
  assert(mf_ll_con_std.get(p_pi_plus).size() ==  Lt);
  assert(mf_ll_con_std.get(p_pi_minus).size() ==  Lt);
    
  assert(compare(mf_ll_con_grid.get(p_pi_plus), mf_ll_con_std.get(p_pi_plus),1e-12));
  assert(compare(mf_ll_con_grid.get(p_pi_minus), mf_ll_con_std.get(p_pi_minus),1e-12));
#endif
 
  if(1){
    std::cout << "testKtoPiPiContractionGridStd computing type1 standard calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type1_std;
    //const int tsep_k_pi, const int tsep_pion, const int tstep, const int xyzStep,
    ComputeKtoPiPiGparity<A2Apolicies_std>::type1(type1_std, 4, 2, 1, 1, p_pi_plus, mf_ls_ww_std, mf_ll_con_std, V_std, V_std, W_std, W_std);

#ifdef GPU_VEC
    //Test the SIMD CPU version agrees with the non-SIMD CPU version first
    std::cout << "testKtoPiPiContractionGridStd computing type1 standard calc with SIMD" << std::endl;    
    std::vector<int> tsep_k_pi(1,4);
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type1_grid_cpu;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type1_omp(&type1_grid_cpu, tsep_k_pi, 2, 1, 1, p_pi_plus, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);

    if(!compareKtoPiPi(type1_std, "non-SIMD", type1_grid_cpu, "SIMD",  "type1", tol))
      ERR.General("","testKtoPiPiContractionGridStd","non-SIMD vs Grid SIMD implementation type1 test failed\n");
    std::cout << "testKtoPiPiContractionGridStd non-SIMD vs Grid SIMD type1 comparison passed" << std::endl;
#endif
       
    std::cout << "testKtoPiPiContractionGridStd computing type1 Grid calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type1_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type1(type1_grid, 4, 2, 1, 1, p_pi_plus, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);
  
    std::cout << "testKtoPiPiContractionGridStd comparing type1" << std::endl;
    if(!compareKtoPiPi(type1_std, "CPS", type1_grid, "Grid",  "type1", tol))
      ERR.General("","testKtoPiPiContractionGridStd","Standard vs Grid implementation type1 test failed\n");
    std::cout << "testKtoPiPiContractionGridStd type1 comparison passed" << std::endl;
  }
  if(1){
    std::cout << "testKtoPiPiContractionGridStd computing type2 standard calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type2_std;
    ComputeKtoPiPiGparity<A2Apolicies_std>::type2(type2_std, 4, 2, 1, p_pi_plus, mf_ls_ww_std, mf_ll_con_std, V_std, V_std, W_std, W_std);

#ifdef GPU_VEC
    //Test the SIMD CPU version agrees with the non-SIMD CPU version first
    std::cout << "testKtoPiPiContractionGridStd computing type2 standard calc with SIMD" << std::endl;    
    std::vector<int> tsep_k_pi(1,4);
    std::vector<ThreeMomentum> p(1, p_pi_plus);
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type2_grid_cpu;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type2_omp_v2(&type2_grid_cpu, tsep_k_pi, 2, 1, p, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);

    if(!compareKtoPiPi(type2_std, "non-SIMD", type2_grid_cpu, "SIMD",  "type2", tol))
      ERR.General("","testKtoPiPiContractionGridStd","non-SIMD vs Grid SIMD implementation type2 test failed\n");
    std::cout << "testKtoPiPiContractionGridStd non-SIMD vs Grid SIMD type2 comparison passed" << std::endl;
#endif
    
    std::cout << "testKtoPiPiContractionGridStd computing type2 Grid calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type2_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type2(type2_grid, 4, 2, 1, p_pi_plus, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);

    std::cout << "testKtoPiPiContractionGridStd comparing type2" << std::endl;
    if(!compareKtoPiPi(type2_std, "CPS", type2_grid, "Grid",  "type2", tol))
      ERR.General("","testKtoPiPiContractionGridStd","Standard vs Grid implementation type2 test failed\n");

    std::cout << "testKtoPiPiContractionGridStd type2 comparison passed" << std::endl;
  }
  if(1){
    std::cout << "testKtoPiPiContractionGridStd computing type3 standard calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type3_std;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::MixDiagResultsContainerType type3_mix_std;
    ComputeKtoPiPiGparity<A2Apolicies_std>::type3(type3_std, type3_mix_std, 4, 2, 1, p_pi_plus, mf_ls_ww_std, mf_ll_con_std, V_std, V_std, W_std, W_std);

    std::cout << "testKtoPiPiContractionGridStd computing type3 Grid calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type3_grid;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::MixDiagResultsContainerType type3_mix_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type3(type3_grid, type3_mix_grid, 4, 2, 1, p_pi_plus, mf_ls_ww_grid, mf_ll_con_grid, V_grid, V_grid, W_grid, W_grid);

    std::cout << "testKtoPiPiContractionGridStd comparing type3" << std::endl;
    if(!compareKtoPiPi(type3_std, "CPS", type3_grid, "Grid",  "type3", tol))
      ERR.General("","testKtoPiPiContractionGridStd","Standard vs Grid implementation type3 test failed\n");
    
    std::cout << "testKtoPiPiContractionGridStd type3 comparison passed" << std::endl;
  }
  if(1){
    std::cout << "testKtoPiPiContractionGridStd computing type4 Grid calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::ResultsContainerType type4_grid;
    typename ComputeKtoPiPiGparity<A2Apolicies_grid>::MixDiagResultsContainerType type4_mix_grid;
    ComputeKtoPiPiGparity<A2Apolicies_grid>::type4(type4_grid, type4_mix_grid, 1, mf_ls_ww_grid, V_grid, V_grid, W_grid, W_grid);

    std::cout << "testKtoPiPiContractionGridStd computing type4 standard calc" << std::endl;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::ResultsContainerType type4_std;
    typename ComputeKtoPiPiGparity<A2Apolicies_std>::MixDiagResultsContainerType type4_mix_std;
    ComputeKtoPiPiGparity<A2Apolicies_std>::type4(type4_std, type4_mix_std, 1, mf_ls_ww_std, V_std, V_std, W_std, W_std);

    std::cout << "testKtoPiPiContractionGridStd comparing type4" << std::endl;
    if(!compareKtoPiPi(type4_std, "CPS", type4_grid, "Grid",  "type4", tol))
      ERR.General("","testKtoPiPiContractionGridStd","Standard vs Grid implementation type4 test failed\n");
    std::cout << "testKtoPiPiContractionGridStd type4 comparison passed" << std::endl;
  }
  std::cout << "Passed testKtoPiPiContractionGridStd" << std::endl;
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

  std::cout << "norm2(l)=" << l.norm2() << " norm2(r)=" << r.norm2() << " norm2(c_base)=" << c_base.norm2() << std::endl; 
  
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> c;
  mult(c, l, r, true); //node local

  std::cout << "norm2(c)=" << c.norm2() << std::endl; 

  bool test = c.equals(c_base, tol, true);
  std::cout << "Node local mult result: " << test << std::endl;
  if(!test) ERR.General("","testMFmult","Node local mult failed!\n");
  
  mult(c, l, r, false); //node distributed

  test = c.equals(c_base, tol, true);

  std::cout << "norm2(c)=" << c.norm2() << std::endl; 
  
  std::cout << "Node distributed mult result: " << test << std::endl;
  if(!test) ERR.General("","testMFmult","Node distributed mult failed!\n");

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


template<typename mf_Complex, typename grid_Complex>
bool compare(const CPSspinColorMatrix<mf_Complex> &orig, const CPSspinColorMatrix<grid_Complex> &grid, const double tol){
  bool fail = false;
  
  mf_Complex gd;
  for(int sl=0;sl<4;sl++)
    for(int cl=0;cl<3;cl++)
      for(int sr=0;sr<4;sr++)
	for(int cr=0;cr<3;cr++){
	  gd = Reduce( grid(sl,sr)(cl,cr) );
	  const mf_Complex &cp = orig(sl,sr)(cl,cr);
  
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
bool compare(const CPSspinColorMatrix<mf_Complex> &orig, const CPSspinColorMatrix<mf_Complex> &newimpl, const double tol){
  bool fail = false;
  
  for(int sl=0;sl<4;sl++)
    for(int cl=0;cl<3;cl++)
      for(int sr=0;sr<4;sr++)
	for(int cr=0;cr<3;cr++){
	  const mf_Complex &gd = newimpl(sl,sr)(cl,cr);
	  const mf_Complex &cp = orig(sl,sr)(cl,cr);
	  
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
void testvMvGridOrigGparity(const A2AArg &a2a_args, const int nthreads, const double tol){
#define BASIC_VMV
#define BASIC_GRID_VMV
#define GRID_VMV
#define GRID_SPLIT_LITE_VMV;

  std::cout << "Starting testvMvGridOrigGparity : vMv tests\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  typename FourDSIMDPolicy<typename ScalarA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<typename ScalarA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorWfftw<ScalarA2Apolicies> W(a2a_args);
  A2AvectorVfftw<ScalarA2Apolicies> V(a2a_args);
  W.testRandom();
  V.testRandom();

  A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf;
  mf.setup(W,V,0,0);
  mf.testRandom();
  typedef typename ScalarA2Apolicies::ComplexType mf_Complex;

  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);
  Wgrid.importFields(W);
  Vgrid.importFields(V);

  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_grid;
  mf_grid.setup(Wgrid,Vgrid,0,0);     
  for(int i=0;i<mf.getNrows();i++)
    for(int j=0;j<mf.getNcols();j++)
      mf_grid(i,j) = mf(i,j); //both are scalar complex
  
  typedef typename GridA2Apolicies::ComplexType grid_Complex;
      
  CPSspinColorFlavorMatrix<mf_Complex> 
    basic_sum[nthreads], orig_sum[nthreads], orig_tmp[nthreads];
  int orig_3vol = GJP.VolNodeSites()/GJP.TnodeSites();

  CPSspinColorFlavorMatrix<grid_Complex> 
    basic_grid_sum[nthreads], grid_sum[nthreads], grid_tmp[nthreads], grid_sum_split_lite[nthreads];      
  int grid_3vol = Vgrid.getMode(0).nodeSites(0) * Vgrid.getMode(0).nodeSites(1) *Vgrid.getMode(0).nodeSites(2);

  mult_vMv_split_lite<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw> vmv_split_lite_grid;
      
  for(int i=0;i<nthreads;i++){
    basic_sum[i].zero(); basic_grid_sum[i].zero();
    orig_sum[i].zero(); grid_sum[i].zero();
    grid_sum_split_lite[i].zero();
  }
  
  if(!UniqueID()){ printf("Starting vMv tests\n"); fflush(stdout); }
  for(int top = 0; top < GJP.TnodeSites(); top++){
    //ORIG VMV
#pragma omp parallel for
    for(int xop=0;xop<orig_3vol;xop++){
      int me = omp_get_thread_num();
      mult(orig_tmp[me], V, mf, W, xop, top, false, true);
      orig_sum[me] += orig_tmp[me];
    }
    
#ifdef BASIC_VMV
    //BASIC VMV FOR TESTING
#pragma omp parallel for
    for(int xop=0;xop<orig_3vol;xop++){
      int me = omp_get_thread_num();
      mult_slow(orig_tmp[me], V, mf, W, xop, top, false, true);
      basic_sum[me] += orig_tmp[me];
    }
#endif

#ifdef GRID_VMV
    //GRID VMV
#pragma omp parallel for
    for(int xop=0;xop<grid_3vol;xop++){
      int me = omp_get_thread_num();
      mult(grid_tmp[me], Vgrid, mf_grid, Wgrid, xop, top, false, true);
      grid_sum[me] += grid_tmp[me];
    }
#endif

#ifdef BASIC_GRID_VMV
    //BASIC GRID VMV FOR TESTING
#pragma omp parallel for
    for(int xop=0;xop<grid_3vol;xop++){
      int me = omp_get_thread_num();
      mult_slow(grid_tmp[me], Vgrid, mf_grid, Wgrid, xop, top, false, true);
      basic_grid_sum[me] += grid_tmp[me];
    }
#endif

#ifdef GRID_SPLIT_LITE_VMV
    //SPLIT LITE VMV GRID
    int top_glb = top + GJP.TnodeCoor() * GJP.TnodeSites();
    vmv_split_lite_grid.setup(Vgrid, mf_grid, Wgrid, top_glb);
#pragma omp parallel for
    for(int xop=0;xop<grid_3vol;xop++){
      int me = omp_get_thread_num();
      vmv_split_lite_grid.contract(grid_tmp[me], xop, false, true);
      grid_sum_split_lite[me] += grid_tmp[me];
    }
#endif
  }//end top loop

  for(int i=1;i<nthreads;i++){
    basic_sum[0] += basic_sum[i];
    orig_sum[0] += orig_sum[i];
    basic_grid_sum[0] += basic_grid_sum[i];
    grid_sum[0] += grid_sum[i];
    grid_sum_split_lite[0] += grid_sum_split_lite[i];  
  }
  
  //Offload version computes all x,t, so we just have to sum over 4 volume afterwards
  typedef mult_vMv_field<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw> vMvFieldImpl;
  typedef typename vMvFieldImpl::PropagatorField PropagatorField;
  PropagatorField pfield(simd_dims);

  //mult(pfield, Vgrid, mf_grid, Wgrid, false, true);
  vMvFieldImpl::optimized(pfield, Vgrid, mf_grid, Wgrid, false, true);

  CPSspinColorFlavorMatrix<grid_Complex> vmv_offload_sum4;
  vmv_offload_sum4.zero();
  for(size_t i=0;i<pfield.size();i++){
    vmv_offload_sum4 += *pfield.fsite_ptr(i);
  }

  //Same for simple field version
  vMvFieldImpl::simple(pfield, Vgrid, mf_grid, Wgrid, false, true);

  CPSspinColorFlavorMatrix<grid_Complex> vmv_offload_simple_sum4;
  vmv_offload_simple_sum4.zero();
  for(size_t i=0;i<pfield.size();i++){
    vmv_offload_simple_sum4 += *pfield.fsite_ptr(i);
  }
  
  
#ifdef BASIC_VMV
  if(!compare(orig_sum[0],basic_sum[0],tol)) ERR.General("","","Standard vs Basic implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Basic implementation test pass\n");
#endif

#ifdef GRID_VMV
  if(!compare(orig_sum[0],grid_sum[0],tol)) ERR.General("","","Standard vs Grid implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Grid implementation test pass\n");
#endif

#ifdef BASIC_GRID_VMV
  if(!compare(orig_sum[0],basic_grid_sum[0],tol)) ERR.General("","","Standard vs Basic Grid implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Basic Grid implementation test pass\n");
#endif

#ifdef GRID_SPLIT_LITE_VMV
  if(!compare(orig_sum[0],grid_sum_split_lite[0],tol)) ERR.General("","","Standard vs Grid Split Lite implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Grid Split Lite implementation test pass\n");
#endif

  if(!compare(orig_sum[0],vmv_offload_sum4,tol)) ERR.General("","","Standard vs Grid field offload optimized implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Grid field offload optimized implementation test pass\n");

  if(!compare(orig_sum[0],vmv_offload_simple_sum4,tol)) ERR.General("","","Standard vs Grid field offload simple implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Grid field offload simple implementation test pass\n");
  
  std::cout << "testvMvGridOrigGparity passed" << std::endl;
}





template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void testvMvGridOrigPeriodic(const A2AArg &a2a_args, const int nthreads, const double tol){
  std::cout << "Starting testvMvGridOrigPeriodic" << std::endl;
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

  //#define BASIC_VMV
  //#define BASIC_GRID_VMV
#define GRID_VMV
#define GRID_SPLIT_VMV_LITE
#define ORIG_SPLIT_VMV_LITE

  std::cout << "Starting vMv tests\n";
      
  CPSspinColorMatrix<mf_Complex> 
    basic_sum[nthreads], orig_sum[nthreads], orig_tmp[nthreads],
    orig_sum_split_xall[nthreads], orig_sum_split[nthreads], orig_sum_split_lite[nthreads];
  int orig_3vol = GJP.VolNodeSites()/GJP.TnodeSites();

  CPSspinColorMatrix<grid_Complex> 
    basic_grid_sum[nthreads], grid_sum[nthreads], grid_tmp[nthreads], 
    grid_sum_split[nthreads], grid_sum_split_xall[nthreads], grid_sum_split_lite[nthreads];      
  int grid_3vol = Vgrid.getMode(0).nodeSites(0) * Vgrid.getMode(0).nodeSites(1) *Vgrid.getMode(0).nodeSites(2);

      
  for(int i=0;i<nthreads;i++){
    basic_sum[i].zero(); basic_grid_sum[i].zero();
    orig_sum[i].zero(); orig_sum_split_lite[i].zero();
    grid_sum[i].zero(); grid_sum_split_lite[i].zero();
  }

  mult_vMv_split_lite<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw> vmv_split_lite_grid;
  mult_vMv_split_lite<ScalarA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw> vmv_split_lite_orig;
  
  if(!UniqueID()){ printf("Starting vMv tests\n"); fflush(stdout); }
  for(int top = 0; top < GJP.TnodeSites(); top++){
    //ORIG VMV
#pragma omp parallel for
    for(int xop=0;xop<orig_3vol;xop++){
      int me = omp_get_thread_num();
      mult(orig_tmp[me], V, mf, W, xop, top, false, true);
      orig_sum[me] += orig_tmp[me];
    }
    
#ifdef BASIC_VMV
    //BASIC VMV FOR TESTING
#pragma omp parallel for
    for(int xop=0;xop<orig_3vol;xop++){
      int me = omp_get_thread_num();
      mult_slow(orig_tmp[me], V, mf, W, xop, top, false, true);
      basic_sum[me] += orig_tmp[me];
    }
#endif

#ifdef GRID_VMV
    //GRID VMV
#pragma omp parallel for
    for(int xop=0;xop<grid_3vol;xop++){
      int me = omp_get_thread_num();
      mult(grid_tmp[me], Vgrid, mf_grid, Wgrid, xop, top, false, true);
      grid_sum[me] += grid_tmp[me];
    }
#endif

#ifdef BASIC_GRID_VMV
    //BASIC GRID VMV FOR TESTING
#pragma omp parallel for
    for(int xop=0;xop<orig_3vol;xop++){
      int me = omp_get_thread_num();
      mult_slow(grid_tmp[me], Vgrid, mf_grid, Wgrid, xop, top, false, true);
      basic_grid_sum[me] += grid_tmp[me];
    }
#endif

#ifdef GRID_SPLIT_VMV_LITE
    //GRID SPLIT VMV LITE
    vmv_split_lite_grid.setup(Vgrid, mf_grid, Wgrid, top);
#pragma omp parallel for
    for(int xop=0;xop<grid_3vol;xop++){
    int me = omp_get_thread_num();
    vmv_split_lite_grid.contract(grid_tmp[me], xop, false, true);
    grid_sum_split_lite[me] += grid_tmp[me];
  }
#endif

#ifdef ORIG_SPLIT_VMV_LITE
    //ORIG SPLIT VMV LITE
    vmv_split_lite_orig.setup(V, mf, W, top);
#pragma omp parallel for
    for(int xop=0;xop<orig_3vol;xop++){
    int me = omp_get_thread_num();
    vmv_split_lite_orig.contract(orig_tmp[me], xop, false, true);
    orig_sum_split_lite[me] += orig_tmp[me];
  }
#endif



  }//end top loop

  for(int i=1;i<nthreads;i++){
    basic_sum[0] += basic_sum[i];
    orig_sum[0] += orig_sum[i];
    basic_grid_sum[0] += basic_grid_sum[i];
    grid_sum[0] += grid_sum[i];
    grid_sum_split_lite[0] += grid_sum_split_lite[i];
    orig_sum_split_lite[0] += orig_sum_split_lite[i];
  }
  
#ifdef BASIC_VMV
  if(!compare(orig_sum[0],basic_sum[0],tol)) ERR.General("","","Standard vs Basic implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Basic implementation test pass\n");
#endif

#ifdef GRID_VMV
  if(!compare(orig_sum[0],grid_sum[0],tol)) ERR.General("","","Standard vs Grid implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Grid implementation test pass\n");
#endif

#ifdef GRID_SPLIT_VMV_LITE
  if(!compare(orig_sum[0],grid_sum_split_lite[0],tol)) ERR.General("","","Standard vs Grid split lite implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Grid split lite implementation test pass\n");
#endif

#ifdef ORIG_SPLIT_VMV_LITE
  if(!compare(orig_sum[0],orig_sum_split_lite[0],tol)) ERR.General("","","Standard vs Scalar split lite implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Scalar split lite implementation test pass\n");
#endif


#ifdef BASIC_GRID_VMV
  if(!compare(orig_sum[0],basic_grid_sum[0],tol)) ERR.General("","","Standard vs Basic Grid implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Basic Grid implementation test pass\n");
#endif

  std::cout << "testvMvGridOrigPeriodic passed" << std::endl;
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


#ifdef __SYCL_DEVICE_ONLY__
  #define CONSTANT __attribute__((opencl_constant))
#else
  #define CONSTANT
#endif


template<typename T>
class ArrayArray{
public:
  typedef T FermionFieldType;
  typedef typename FermionFieldType::FieldSiteType FieldSiteType;
  typedef typename FermionFieldType::InputParamType FieldInputParamType;

  CPSfieldArray<FermionFieldType> a;
  CPSfieldArray<FermionFieldType> b; //these have been diluted in spin/color but not the other indices, hence there are nhit * 12 fields here (spin/color index changes fastest in mapping)

  ArrayArray(int na, int nb, const FieldInputParamType &simd_dims){
#if 0
    a.resize(na); for(int i=0;i<na;i++) a[i].emplace(simd_dims);
    b.resize(nb); for(int i=0;i<nb;i++) b[i].emplace(simd_dims);
#else
    a.resize(na); for(int i=0;i<na;i++) a[i].set(new FermionFieldType(simd_dims)); //  emplace(simd_dims);
    b.resize(nb); for(int i=0;i<nb;i++) b[i].set(new FermionFieldType(simd_dims)); //  emplace(simd_dims);
#endif
    
  }

  inline PtrWrapper<FermionFieldType> & getA(size_t i){ return a[i]; }
  inline PtrWrapper<FermionFieldType> & getB(size_t i){ return b[i]; }

};


  


template<typename GridA2Apolicies>
void testCPSfieldArray(){
  std::cout << "Starting testCPSfieldArray" << std::endl;
  typedef typename GridA2Apolicies::ComplexType ComplexType;
  size_t nsimd = ComplexType::Nsimd();
  typename SIMDpolicyBase<4>::ParamType simd_dims;
  SIMDpolicyBase<4>::SIMDdefaultLayout(simd_dims,nsimd,2); //only divide over spatial directions

  typedef typename GridA2Apolicies::FermionFieldType FermionFieldType;

  std::cout << "Generating random field" << std::endl;
  FermionFieldType field(simd_dims);
  field.testRandom();

  std::cout  << "Seting up field array" << std::endl;
  CPSfieldArray<FermionFieldType> farray(1);
  farray[0].set(new FermionFieldType(field));

  ComplexType* into = (ComplexType*)managed_alloc_check(sizeof(ComplexType));
  ComplexType expect = *field.site_ptr(size_t(0));

  std::cout << "Getting view" << std::endl;
  CPSautoView(av, farray); //auto destruct memory alloced
   
  using Grid::acceleratorThreads;

  typedef SIMT<ComplexType> ACC;

#ifdef GRID_CUDA  
  using Grid::LambdaApply;
  using Grid::acceleratorAbortOnGpuError;  
#elif defined(GRID_SYCL)
  using Grid::theGridAccelerator;
#endif

  std::cout << "Starting kernel" << std::endl;

  ComplexType* expect_p = farray[0]->fsite_ptr(size_t(0));
  std::cout << "Site 0 ptr " << expect_p << std::endl;
  
  accelerator_for(x, farray[0]->size(), nsimd,
		  {
		    if(x == 0){
		      ComplexType* site_ptr = av[0].fsite_ptr(x);
		      auto v = ACC::read(*site_ptr);
		      ACC::write(*into, v);
		    }
		  });

  std::cout << "Got " << *into << " expect " << expect << std::endl;
  
  assert( Reduce(expect == *into) );
  
  managed_free(into);

  //Test copy
  std::cout << "Testing copy assignment" << std::endl;
  CPSfieldArray<FermionFieldType> farray2= farray;
  assert(farray2.size() == farray.size());
  for(int i=0;i<farray2.size();i++){
    assert( farray2[i]->equals( *farray[i] ) );
  }

  std::cout << "Testing copy" << std::endl;
  CPSfieldArray<FermionFieldType> farray3;
  farray3 = farray;
  assert(farray2.size() == farray.size());
  for(int i=0;i<farray3.size();i++){
    assert( farray3[i]->equals( *farray[i] ) );
  }

  //Test ptrwrapper
  PtrWrapper<FermionFieldType> p; p.set(new FermionFieldType(field));
  PtrWrapper<FermionFieldType> p2; p2.set(new FermionFieldType(field));

  std::cout << "Testing PtrWrapper copy" << std::endl;
  p2 = p;
  assert(p->equals(*p2));
  
  //Test array of arrays   
  std::cout << "Testing array of arrays mode copy" << std::endl;
  ArrayArray<FermionFieldType> a2(1,1,simd_dims);
 
  a2.getA(0) = p; //.set(new FermionFieldType(field)); //farray[0];
  a2.getB(0) = p; //.set(new FermionFieldType(field)); //farray[0]; 

  std::cout << "Testing array of arrays copy" << std::endl;
  ArrayArray<FermionFieldType> a2_cp(0,0,simd_dims);
  a2_cp = a2;

  assert( a2_cp.getA(0)->equals(field) );
  assert( a2_cp.getB(0)->equals(field) );

  
  std::cout << "Passed testCPSfieldArray" << std::endl;
}


template<typename GridA2Apolicies>
void testA2AfieldAccess(){
  std::cout << "Starting testA2AfieldAccess" << std::endl;
  typedef typename GridA2Apolicies::ComplexType ComplexType;
  size_t nsimd = ComplexType::Nsimd();
  typename SIMDpolicyBase<4>::ParamType simd_dims;
  SIMDpolicyBase<4>::SIMDdefaultLayout(simd_dims,nsimd,2); //only divide over spatial directions

  typedef typename GridA2Apolicies::FermionFieldType FermionFieldType;

  std::cout << "Generating random field" << std::endl;
  FermionFieldType field(simd_dims);
  field.testRandom();

  ComplexType* into = (ComplexType*)managed_alloc_check(sizeof(ComplexType));
  ComplexType expect = *field.site_ptr(size_t(0));
  
  A2AArg a2a_arg;
  a2a_arg.nl = 1;
  a2a_arg.nhits = 1;
  a2a_arg.rand_type = UONE;
  a2a_arg.src_width = 1;

  A2AvectorV<GridA2Apolicies> v(a2a_arg,simd_dims);
  v.getMode(0) = field;
  
  memset(into, 0, sizeof(ComplexType));

  using Grid::acceleratorThreads;

  typedef SIMT<ComplexType> ACC;

#ifdef GRID_CUDA  
  using Grid::LambdaApply;
  using Grid::acceleratorAbortOnGpuError;
#elif defined(GRID_SYCL)
  using Grid::theGridAccelerator;
#endif

  std::cout << "Generating views" << std::endl;
  CPSautoView(vv, v);
   
  size_t fsize = field.size();
 
  std::cout << "Starting kernel, fsize " << fsize << std::endl;
  accelerator_for(x, fsize, nsimd,
  		  {
  		    if(x==0){
		      const ComplexType &val_vec = *vv.getMode(0).fsite_ptr(x);
		      auto val_lane = ACC::read(val_vec);
 		      ACC::write(*into, val_lane);
		    }
  		  });

  std::cout << "Got " << *into << " expect " << expect << std::endl;
  
  assert( Reduce(expect == *into) );  

  managed_free(into);
 
  std::cout << "Passed testA2AfieldAccess" << std::endl;
}


struct autoViewTest1{
  double v;
  bool free_called;
  autoViewTest1(): free_called(false){}
  
  struct View{    
    double v;
    View(const autoViewTest1 &p): v(p.v){}
  };
  View view() const{ return View(*this); }
};

struct autoViewTest2{
  double v;
  bool free_called;    
  
  struct View{
    double v;
    bool* free_called;
    
    View(autoViewTest2 &p): v(p.v), free_called(&p.free_called){}
    void free(){ *free_called = true; }
  };
  View view(){ return View(*this); }
};

void testAutoView(){ 
  autoViewTest1 t1;
  t1.v = 3.14;
  
  {  
    CPSautoView(t1_v, t1);

    assert(t1_v.v == t1.v);
  }
  assert( t1.free_called == false );

  autoViewTest2 t2;
  t1.v = 6.28;
  
  {  
    CPSautoView(t2_v, t2);

    assert(t2_v.v == t2.v);
  }
  assert( t2.free_called == true );
  
}

void testViewArray(){
  //Test for a type that doesn't have a free method in its view
  std::vector<autoViewTest1> t1(2);
  t1[0].v = 3.14;
  t1[1].v = 6.28;
  
  std::vector<autoViewTest1*> t1_p = { &t1[0], &t1[1] };
  ViewArray<typename autoViewTest1::View> t1_v(t1_p);

  double* into = (double*)managed_alloc_check(2*sizeof(double));

  using Grid::acceleratorThreads;

#ifdef GRID_CUDA  
  using Grid::LambdaApply;
  using Grid::acceleratorAbortOnGpuError;
#elif defined(GRID_SYCL)
  using Grid::theGridAccelerator;
#endif
  
  
  accelerator_for(x, 100, 1,
		  {
		    if(x==0 || x==1){
		      into[x] = t1_v[x].v;
		    }
		  });
  assert(into[0] == 3.14);
  assert(into[1] == 6.28);


  //Test for a type that does have a free method in its view
  std::vector<autoViewTest2> t2(2);
  t2[0].v = 31.4;
  t2[1].v = 62.8;
  
  std::vector<autoViewTest2*> t2_p = { &t2[0], &t2[1] };
  ViewArray<typename autoViewTest2::View> t2_v(t2_p);

  accelerator_for(x, 100, 1,
		  {
		    if(x==0 || x==1){
		      into[x] = t2_v[x].v;
		    }
		  });
  assert(into[0] == 31.4);
  assert(into[1] == 62.8);

  t2_v.free();
  
  assert(t2[0].free_called);
  assert(t2[1].free_called); 

  
  managed_free(into);
  std::cout << "testViewArray passed" << std::endl;
}





template<typename GridA2Apolicies>
void testCPSfieldDeviceCopy(){
#ifdef GPU_VEC

  typedef typename GridA2Apolicies::ComplexType ComplexType;
  size_t nsimd = ComplexType::Nsimd();
  typename SIMDpolicyBase<4>::ParamType simd_dims;
  SIMDpolicyBase<4>::SIMDdefaultLayout(simd_dims,nsimd,2); //only divide over spatial directions

  typedef typename GridA2Apolicies::FermionFieldType FermionFieldType;
  
  //Test a host-allocated CPSfield
  FermionFieldType field(simd_dims);
  field.testRandom();

  ComplexType* into = (ComplexType*)managed_alloc_check(sizeof(ComplexType));
  typedef SIMT<ComplexType> ACC;

  ComplexType expect = *field.site_ptr(size_t(0));

  using Grid::acceleratorThreads;

#ifdef GRID_CUDA  
  using Grid::LambdaApply;
  using Grid::acceleratorAbortOnGpuError;
#elif defined(GRID_SYCL)
  using Grid::theGridAccelerator;
#endif
  auto field_v = field.view();
  
  accelerator_for(x, 1, nsimd,
		  {
		    auto v = ACC::read(*field_v.site_ptr(x));
		    ACC::write(*into, v);
		  });

  std::cout << "Got " << *into << " expect " << expect << std::endl;
  
  assert( Reduce(expect == *into) );

  //Test a view that is allocated in shared memory; basically a 1-element array
  ManagedPtrWrapper<typename FermionFieldType::View> wrp(field.view());
  
  memset(into, 0, sizeof(ComplexType));

  auto wrp_v = wrp.view();
  accelerator_for(x, 1, nsimd,
  		  {
  		    auto v = ACC::read(*wrp_v->site_ptr(x));
  		    ACC::write(*into, v);
  		  });

  std::cout << "Got " << *into << " expect " << expect << std::endl;
  
  assert( Reduce(expect == *into) );  

  managed_free(into);

#endif
}





template<typename GridA2Apolicies>
void testMultiSourceDeviceCopy(){
#ifdef GPU_VEC

  typedef typename GridA2Apolicies::ComplexType ComplexType;
  size_t nsimd = ComplexType::Nsimd();

  typedef typename GridA2Apolicies::SourceFieldType SourceFieldType;
  typename SourceFieldType::InputParamType simd_dims_3d;
  setupFieldParams<SourceFieldType>(simd_dims_3d);

  typedef typename GridA2Apolicies::SourcePolicies SourcePolicies;

  typedef A2AflavorProjectedExpSource<SourcePolicies> ExpSrcType;
  typedef A2AflavorProjectedHydrogenSource<SourcePolicies> HydSrcType;
  typedef Elem<ExpSrcType, Elem<HydSrcType,ListEnd > > SrcList;
  typedef A2AmultiSource<SrcList> MultiSrcType;

  MultiSrcType src;

  double rad = 2.0;
  int pbase[3] = {1,1,1};
  src.template getSource<0>().setup(rad,pbase, simd_dims_3d); //1s
  src.template getSource<1>().setup(2,0,0,rad,pbase, simd_dims_3d); //2s

  auto src_v = src.view();

  ComplexType expect1 = src.template getSource<0>().siteComplex(size_t(0));
  ComplexType expect2 = src.template getSource<1>().siteComplex(size_t(0));

  using Grid::acceleratorThreads;

#ifdef GRID_CUDA  
  using Grid::LambdaApply;
  using Grid::acceleratorAbortOnGpuError;
#elif defined(GRID_SYCL)
  using Grid::theGridAccelerator;
#endif

  ComplexType* into = (ComplexType*)managed_alloc_check(sizeof(ComplexType));
  typedef SIMT<ComplexType> ACC;
 
  
  accelerator_for(x, 100, nsimd,
		  {
		    if(x==0){
		      auto v1 = ACC::read(src_v.template getSource<0>().siteComplex(x));
		      ACC::write(into[0], v1);
		      
		      auto v2 = ACC::read(src_v.template getSource<1>().siteComplex(x));
		      ACC::write(into[1], v2);
		    }
		  });

  std::cout << "Got " << into[0] << " expect " << expect1 << std::endl;
  assert( Reduce(expect1 == into[0]) );
  std::cout << "Got " << into[1] << " expect " << expect2 << std::endl;
  assert( Reduce(expect2 == into[1]) );
		  
  managed_free(into);
  src_v.free();
  std::cout << "Passed testMultiSourceDeviceCopy" << std::endl;
#endif
}



template<typename GridA2Apolicies>
void testFlavorProjectedSourceView(){
#ifdef GPU_VEC  
  std::cout << "Starting testFlavorProjectedSourceView" << std::endl;
  
  typedef typename GridA2Apolicies::ComplexType ComplexType;
  size_t nsimd = ComplexType::Nsimd();

  typedef typename GridA2Apolicies::SourceFieldType SourceFieldType;
  typename SourceFieldType::InputParamType simd_dims_3d;
  setupFieldParams<SourceFieldType>(simd_dims_3d);

  typedef typename GridA2Apolicies::SourcePolicies SourcePolicies;

  int pbase[3] = {1,1,1};
  
  typedef A2AflavorProjectedExpSource<SourcePolicies> SrcType;
  SrcType src(2.0, pbase, simd_dims_3d);

  typedef typename ComplexType::scalar_type ScalarComplexType;
  typedef FlavorMatrixGeneral<ComplexType> vFmatType;
  typedef FlavorMatrixGeneral<ScalarComplexType> sFmatType;
  
  vFmatType* into = (vFmatType*)managed_alloc_check(sizeof(vFmatType));

  auto src_v = src.view();
  
  typedef SIMT<ComplexType> ACC;

  //Check we can access the source sites
  {
    std::cout << "Checking site access" << std::endl;
    using namespace Grid;
    ComplexType *into_c = (ComplexType*)into;
    
    accelerator_for(x, 100, nsimd,
		  {
		    if(x==0){
		      typename ACC::value_type tmp = ACC::read(src_v.siteComplex(0));
		      ACC::write(*into_c,tmp);  
		    }
		  });
    ComplexType expect = src.siteComplex(0);

    ScalarComplexType got_c = Reduce(*into_c);
    ScalarComplexType expect_c = Reduce(expect);
    assert(got_c == expect_c);
    std::cout << "Checked site access" << std::endl;
  }
    
   
  {
    std::cout << "Checking Fmat access" << std::endl;
    using namespace Grid;
 
    accelerator_for(x, 100, nsimd,
		  {
		    if(x==0){
		      FlavorMatrixGeneral<typename ACC::value_type> tmp;
		      //sFmatType tmp;
		      src_v.siteFmat(tmp, 0);
		      for(int f1=0;f1<2;f1++)
			for(int f2=0;f2<2;f2++)
			  ACC::write( (*into)(f1,f2), tmp(f1,f2) );
		    }
		  });

    
    vFmatType expect;
    src.siteFmat(expect,0);
    
    assert(equals(*into,expect,1e-12));
  }

  managed_free(into);
  std::cout << "Passed testFlavorProjectedSourceView" << std::endl;
#endif
}
  


template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void testVVgridOrigGparity(const A2AArg &a2a_args, const int nthreads, const double tol){
  std::cout << "Starting testVVgridOrigGparity\n";

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
      
  //Temporaries
  CPSspinColorFlavorMatrix<mf_Complex> orig_tmp[nthreads];
  CPSspinColorFlavorMatrix<grid_Complex> grid_tmp[nthreads];
  
  //Accumulation output
  CPSspinColorFlavorMatrix<mf_Complex> orig_slow_sum[nthreads], orig_sum[nthreads];
  CPSspinColorFlavorMatrix<grid_Complex> grid_sum[nthreads];
  for(int i=0;i<nthreads;i++){
    orig_sum[i].zero(); orig_slow_sum[i].zero(); grid_sum[i].zero();
  }
  
  int orig_3vol = GJP.VolNodeSites()/GJP.TnodeSites();
  int grid_3vol = Vgrid.getMode(0).nodeSites(0) * Vgrid.getMode(0).nodeSites(1) *Vgrid.getMode(0).nodeSites(2);
      
  for(int top = 0; top < GJP.TnodeSites(); top++){
#pragma omp parallel for
    for(int xop=0;xop<orig_3vol;xop++){
      int me = omp_get_thread_num();
      //Slow
      mult_slow(orig_tmp[me], V, W, xop, top, false, true);
      orig_slow_sum[me] += orig_tmp[me];

      //Non-SIMD
      mult(orig_tmp[me], V, W, xop, top, false, true);
      orig_sum[me] += orig_tmp[me];
    }

#pragma omp parallel for   
    for(int xop=0;xop<grid_3vol;xop++){
      int me = omp_get_thread_num();

      //SIMD
      mult(grid_tmp[me], Vgrid, Wgrid, xop, top, false, true);
      grid_sum[me] += grid_tmp[me];
    }
  }

  //Combine sums from threads > 1 into thread 0 output
  for(int i=1;i<nthreads;i++){
    orig_sum[0] += orig_sum[i];
    orig_slow_sum[0] += orig_slow_sum[i];
    grid_sum[0] += grid_sum[i];
  }
  
  //Offload version computes all x,t, so we just have to sum over 4 volume afterwards
  typedef typename mult_vv_field<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw>::PropagatorField PropagatorField;
  PropagatorField pfield(simd_dims);
  
  mult(pfield, Vgrid, Wgrid, false, true);

  CPSspinColorFlavorMatrix<grid_Complex> vmv_offload_sum4;
  vmv_offload_sum4.zero();
  for(size_t i=0;i<pfield.size();i++){
    vmv_offload_sum4 += *pfield.fsite_ptr(i);
  }

  //Do the comparison
  if(!compare(orig_sum[0],orig_slow_sum[0],tol)) ERR.General("","","Standard vs Slow implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Slow implementation test pass\n");
  
  if(!compare(orig_sum[0], grid_sum[0],tol)) ERR.General("","","Standard vs Grid implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Grid implementation test pass\n");
  
  if(!compare(orig_sum[0], vmv_offload_sum4,tol)) ERR.General("","","Standard vs Field Offload implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Field Offload implementation test pass\n");

  std::cout << "testVVgridOrigGparity passed" << std::endl;
}


template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void testVVgridOrigPeriodic(const A2AArg &a2a_args, const int ntests, const int nthreads, const double tol){
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
  CPSspinColorMatrix<mf_Complex> orig_slow_sum[nthreads], orig_sum[nthreads], orig_tmp[nthreads];
  CPSspinColorMatrix<grid_Complex> grid_sum[nthreads], grid_tmp[nthreads];

  int orig_3vol = GJP.VolNodeSites()/GJP.TnodeSites();
  int grid_3vol = Vgrid.getMode(0).nodeSites(0) * Vgrid.getMode(0).nodeSites(1) *Vgrid.getMode(0).nodeSites(2);
      
  for(int iter=0;iter<ntests;iter++){
    for(int i=0;i<nthreads;i++){
      orig_sum[i].zero(); orig_slow_sum[i].zero(); grid_sum[i].zero();
    }
	
    for(int top = 0; top < GJP.TnodeSites(); top++){
#pragma omp parallel for
      for(int xop=0;xop<orig_3vol;xop++){
	int me = omp_get_thread_num();
	mult_slow(orig_tmp[me], V, W, xop, top, false, true);
	orig_slow_sum[me] += orig_tmp[me];
      }

      total_time_orig -= dclock();	  
#pragma omp parallel for
      for(int xop=0;xop<orig_3vol;xop++){
	int me = omp_get_thread_num();
	mult(orig_tmp[me], V, W, xop, top, false, true);
	orig_sum[me] += orig_tmp[me];
      }
      total_time_orig += dclock();

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
      orig_slow_sum[0] += orig_slow_sum[i];
      grid_sum[0] += grid_sum[i];
    }

    //Offload version computes all x,t, so we just have to sum over 4 volume afterwards
    total_time_field_offload -= dclock();
    typedef typename mult_vv_field<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw>::PropagatorField PropagatorField;
    PropagatorField pfield(simd_dims);
    
    mult(pfield, Vgrid, Wgrid, false, true);
    total_time_field_offload += dclock();

    if(iter == 0){
      CPSspinColorMatrix<grid_Complex> vmv_offload_sum4;
      vmv_offload_sum4.zero();
      for(size_t i=0;i<pfield.size();i++){
	vmv_offload_sum4 += *pfield.fsite_ptr(i);
      }
      
      if(!compare(orig_sum[0],orig_slow_sum[0],tol)) ERR.General("","","Standard vs Slow implementation test failed\n");
      else if(!UniqueID()) printf("Standard vs Slow implementation test pass\n");
      
      if(!compare(orig_sum[0], grid_sum[0],tol)) ERR.General("","","Standard vs Grid implementation test failed\n");
      else if(!UniqueID()) printf("Standard vs Grid implementation test pass\n");
      
      if(!compare(orig_sum[0], vmv_offload_sum4,tol)) ERR.General("","","Standard vs Field Offload implementation test failed\n");
      else if(!UniqueID()) printf("Standard vs Field Offload implementation test pass\n");
    }
  }

  printf("vv: Avg time new code %d iters: %g secs\n",ntests,total_time/ntests);
  printf("vv: Avg time old code %d iters: %g secs\n",ntests,total_time_orig/ntests);
  printf("vv: Avg time field offload code %d iters: %g secs\n",ntests,total_time_field_offload/ntests);

  if(!UniqueID()){
    printf("vv offload timings:\n");
    mult_vv_field_offload_timers::get().print();
  }

}







struct _tr{
  template<typename MatrixType>
  accelerator_inline auto operator()(const MatrixType &matrix) const ->decltype(matrix.Trace()){ return matrix.Trace(); }  
};

struct _times{
  template<typename MatrixType>
  accelerator_inline auto operator()(const MatrixType &a, const MatrixType &b) const ->decltype(a * b){ return a * b; }  
};

template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
struct _timesIV_unop{
  typedef VectorMatrixType OutputType;
  accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &in, const int lane) const{ 
    timesI(out, in, lane);
  }
};

template<typename VectorMatrixType>
struct _trtrV{
  typedef typename VectorMatrixType::scalar_type OutputType;
  accelerator_inline void operator()(OutputType &out, const VectorMatrixType &a, const VectorMatrixType &b, const int lane) const{ 
    typename VectorMatrixType::scalar_type tmp, tmp2; //each thread will have one of these but will only write to a single thread    
    Trace(tmp, a, lane);
    Trace(tmp2, b, lane);
    mult(out, tmp, tmp2, lane);
  }
};


template<typename GridA2Apolicies>
void testCPSmatrixField(const double tol){
  std::cout << "Starting testCPSmatrixField" << std::endl;
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


  //Test binop using operator*
  c = binop(a,b, _times());
  
  fail = false;
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
	printf("Fail: binop (operator*) (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
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



  //Test binop_v using operator*
  c = binop_v(a,b, _timesV<VectorMatrixType>());
  
  fail = false;
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
	printf("Fail: binop_v (operator*) (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	fail = true;
      }
    }
    }
    }
    }
    }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField binop_v operator* failed\n");




  //Test trace * trace using binop_v
  auto trtr = binop_v(a,b, _trtrV<VectorMatrixType>());
  
  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=*a.site_ptr(x4d);
    auto bb=*b.site_ptr(x4d);
    auto cc = aa.Trace()*bb.Trace();
    auto got = Reduce( *trtr.site_ptr(x4d) );
    auto expect = Reduce( cc );
      
    double rdiff = fabs(got.real()-expect.real());
    double idiff = fabs(got.imag()-expect.imag());
    if(rdiff > tol|| idiff > tol){
      printf("Fail: trace*trace binop_v (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
      fail = true;
    }
  }
  if(fail) ERR.General("","","CPSmatrixField trace*trace binop_v failed\n");





  //Test binop_v using operator+
  c = binop_v(a,b, _addV<VectorMatrixType>());
  
  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=*a.site_ptr(x4d);
    auto bb=*b.site_ptr(x4d);
    auto cc = aa + bb;
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
	printf("Fail: binop_v (operator+) (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	fail = true;
      }
    }
    }
    }
    }
    }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField binop_v operator+ failed\n");



  //Test binop_v using operator-
  c = binop_v(a,b, _subV<VectorMatrixType>());
  
  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=*a.site_ptr(x4d);
    auto bb=*b.site_ptr(x4d);
    auto cc = aa - bb;
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
	printf("Fail: binop_v (operator-) (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	fail = true;
      }
    }
    }
    }
    }
    }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField binop_v operator- failed\n");



  //Test unop_self_v using unit
  unop_self_v(c, _unitV<VectorMatrixType>());
  
  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    VectorMatrixType v;
    v.unit();
    for(int s1=0;s1<4;s1++){
    for(int c1=0;c1<3;c1++){
    for(int f1=0;f1<2;f1++){
    for(int s2=0;s2<4;s2++){
    for(int c2=0;c2<3;c2++){
    for(int f2=0;f2<2;f2++){
      auto got = Reduce( (*c.site_ptr(x4d))(s1,s2)(c1,c2)(f1,f2) );
      auto expect = Reduce( v(s1,s2)(c1,c2)(f1,f2) );
      
      double rdiff = fabs(got.real()-expect.real());
      double idiff = fabs(got.imag()-expect.imag());
      if(rdiff > tol|| idiff > tol){
	printf("Fail: unop_self_v (unit) (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	fail = true;
      }
    }
    }
    }
    }
    }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField unop_self_v unit failed\n");



  //Test unop_v using timesI
  c = unop_v(a, _timesIV_unop<VectorMatrixType>());
  
  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=*a.site_ptr(x4d);
    aa.timesI();

    for(int s1=0;s1<4;s1++){
    for(int c1=0;c1<3;c1++){
    for(int f1=0;f1<2;f1++){
    for(int s2=0;s2<4;s2++){
    for(int c2=0;c2<3;c2++){
    for(int f2=0;f2<2;f2++){
      auto got = Reduce( (*c.site_ptr(x4d))(s1,s2)(c1,c2)(f1,f2) );
      auto expect = Reduce( aa(s1,s2)(c1,c2)(f1,f2) );
      
      double rdiff = fabs(got.real()-expect.real());
      double idiff = fabs(got.imag()-expect.imag());
      if(rdiff > tol|| idiff > tol){
	printf("Fail: unop_v (timesI) (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	fail = true;
      }
    }
    }
    }
    }
    }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField unop_v timesI failed\n");


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



  //Test Trace-product
  typedef CPSmatrixField<ComplexType> ComplexField;
 
  d = Trace(a,b);

  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=*a.site_ptr(x4d);
    auto bb=*b.site_ptr(x4d);
    ComplexType aat = Trace(aa,bb);
    auto got = Reduce( *d.site_ptr(x4d) );
    auto expect = Reduce( aat );
      
    double rdiff = fabs(got.real()-expect.real());
    double idiff = fabs(got.imag()-expect.imag());
    if(rdiff > tol|| idiff > tol){
      printf("Fail: Trace-product (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
      fail = true;
    } 
  }
  if(fail) ERR.General("","","CPSmatrixField Trace-product failed\n");



  //Test Trace-product of non-field
  typedef CPSmatrixField<ComplexType> ComplexField;
  VectorMatrixType bs = *b.site_ptr(size_t(0));
  d = Trace(a,bs);

  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=*a.site_ptr(x4d);
    auto bb=*b.site_ptr(x4d);
    ComplexType aat = Trace(aa,bs);
    auto got = Reduce( *d.site_ptr(x4d) );
    auto expect = Reduce( aat );
      
    double rdiff = fabs(got.real()-expect.real());
    double idiff = fabs(got.imag()-expect.imag());
    if(rdiff > tol|| idiff > tol){
      printf("Fail: Trace-product non-field (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
      fail = true;
    } 
  }
  if(fail) ERR.General("","","CPSmatrixField Trace-product non-field failed\n");




  //Test unop via trace
  d = unop(a, _tr());

  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=*a.site_ptr(x4d);
    ComplexType aat = aa.Trace();
    auto got = Reduce( *d.site_ptr(x4d) );
    auto expect = Reduce( aat );
      
    double rdiff = fabs(got.real()-expect.real());
    double idiff = fabs(got.imag()-expect.imag());
    if(rdiff > tol|| idiff > tol){
      printf("Fail: Unop (Trace) (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
      fail = true;
    } 
  }
  if(fail) ERR.General("","","CPSmatrixField Unop (Trace) failed\n");


  //Test unop_v via trace
  d = unop_v(a, _trV<VectorMatrixType>());

  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=*a.site_ptr(x4d);
    ComplexType aat = aa.Trace();
    auto got = Reduce( *d.site_ptr(x4d) );
    auto expect = Reduce( aat );
      
    double rdiff = fabs(got.real()-expect.real());
    double idiff = fabs(got.imag()-expect.imag());
    if(rdiff > tol|| idiff > tol){
      printf("Fail: Unop_v (Trace) (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
      fail = true;
    } 
  }
  if(fail) ERR.General("","","CPSmatrixField Unop_v (Trace) failed\n");


  //Test partial trace using unop_v
  auto a_tridx1 = unop_v(a, _trIndexV<1,VectorMatrixType>());

  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=*a.site_ptr(x4d);
    auto aat = aa.template TraceIndex<1>();
    for(size_t i=0;i<aat.nScalarType();i++){
      auto got = Reduce( a_tridx1.site_ptr(x4d)->scalarTypePtr()[i] );
      auto expect = Reduce( aat.scalarTypePtr()[i] );

      double rdiff = fabs(got.real()-expect.real());
      double idiff = fabs(got.imag()-expect.imag());
      if(rdiff > tol|| idiff > tol){
	printf("Fail: Unop_v (TraceIdx<1>) (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	fail = true;
      } 
    }
  }
  if(fail) ERR.General("","","CPSmatrixField Unop_v (TraceIdx<1>) failed\n");


  //Test partial double trace using unop_v
  auto a_tridx0_2 = unop_v(a, _trTwoIndicesV<0,2,VectorMatrixType>());

  fail = false;
  for(size_t x4d=0; x4d< a.size(); x4d++){
    auto aa=*a.site_ptr(x4d);
    auto aat = aa.template TraceTwoIndices<0,2>();
    for(size_t i=0;i<aat.nScalarType();i++){
      auto got = Reduce( a_tridx0_2.site_ptr(x4d)->scalarTypePtr()[i] );
      auto expect = Reduce( aat.scalarTypePtr()[i] );

      double rdiff = fabs(got.real()-expect.real());
      double idiff = fabs(got.imag()-expect.imag());
      if(rdiff > tol|| idiff > tol){
	printf("Fail: Unop_v (TraceTwoIndices<0,2>) (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	fail = true;
      } 
    }
  }
  if(fail) ERR.General("","","CPSmatrixField Unop_v (TraceTwoIndices<0,2>) failed\n");



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



  //Test unop_v TransposeOnIndex
  f = unop_v(e, _transIdx<1,typename Matrix2Field::FieldSiteType>());

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
	printf("Fail: unop_v TranposeOnIndex (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	fail = true;
      }
    }
    }
    }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField unop_v TransposeOnIndex failed\n");



  
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


  //Test gl
  int gl_dirs[5] = {0,1,2,3,-5};
  for(int gg=0;gg<5;gg++){
    int dir = gl_dirs[gg];

    PropagatorField pla(a);
    gl(pla, dir);

    fail = false;
    for(size_t x4d=0; x4d< a.size(); x4d++){
      auto aa=*a.site_ptr(x4d);
      aa.gl(dir);
      for(int i=0;i<aa.nScalarType();i++){
	auto got = Reduce( pla.site_ptr(x4d)->scalarTypePtr()[i] );
	auto expect = Reduce( aa.scalarTypePtr()[i] );
	
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: gl[%d] (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",dir,got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}
      }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField gl failed\n");


  //Test gr
  for(int gg=0;gg<5;gg++){
    int dir = gl_dirs[gg];

    PropagatorField pla(a);
    gr(pla, dir);

    fail = false;
    for(size_t x4d=0; x4d< a.size(); x4d++){
      auto aa=*a.site_ptr(x4d);
      aa.gr(dir);
      for(int i=0;i<aa.nScalarType();i++){
	auto got = Reduce( pla.site_ptr(x4d)->scalarTypePtr()[i] );
	auto expect = Reduce( aa.scalarTypePtr()[i] );
	
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: gr[%d] (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",dir,got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}
      }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField gr failed\n");


  //Test glAx
  int glAx_dirs[4] = {0,1,2,3};
  for(int gg=0;gg<4;gg++){
    int dir = glAx_dirs[gg];

    PropagatorField pla(a);
    glAx(pla, dir);

    fail = false;
    for(size_t x4d=0; x4d< a.size(); x4d++){
      auto aa=*a.site_ptr(x4d);
      aa.glAx(dir);
      for(int i=0;i<aa.nScalarType();i++){
	auto got = Reduce( pla.site_ptr(x4d)->scalarTypePtr()[i] );
	auto expect = Reduce( aa.scalarTypePtr()[i] );
	
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: glAx[%d] (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",dir,got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}
      }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField glAx failed\n");


  //Test grAx
  for(int gg=0;gg<4;gg++){
    int dir = glAx_dirs[gg];

    PropagatorField pla(a);
    grAx(pla, dir);

    fail = false;
    for(size_t x4d=0; x4d< a.size(); x4d++){
      auto aa=*a.site_ptr(x4d);
      aa.grAx(dir);
      for(int i=0;i<aa.nScalarType();i++){
	auto got = Reduce( pla.site_ptr(x4d)->scalarTypePtr()[i] );
	auto expect = Reduce( aa.scalarTypePtr()[i] );
	
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: grAx[%d] (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",dir,got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}
      }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField grAx failed\n");



  //Test gl_r
  for(int gg=0;gg<5;gg++){
    int dir = gl_dirs[gg];

    PropagatorField pla = gl_r(a, dir);

    fail = false;
    for(size_t x4d=0; x4d< a.size(); x4d++){
      auto aa=*a.site_ptr(x4d);
      aa.gl(dir);
      for(int i=0;i<aa.nScalarType();i++){
	auto got = Reduce( pla.site_ptr(x4d)->scalarTypePtr()[i] );
	auto expect = Reduce( aa.scalarTypePtr()[i] );
	
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: gl_r[%d] (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",dir,got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}
      }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField gl_r failed\n");


  //Test gr_r
  for(int gg=0;gg<5;gg++){
    int dir = gl_dirs[gg];

    PropagatorField pla = gr_r(a, dir);

    fail = false;
    for(size_t x4d=0; x4d< a.size(); x4d++){
      auto aa=*a.site_ptr(x4d);
      aa.gr(dir);
      for(int i=0;i<aa.nScalarType();i++){
	auto got = Reduce( pla.site_ptr(x4d)->scalarTypePtr()[i] );
	auto expect = Reduce( aa.scalarTypePtr()[i] );
	
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: gr_r[%d] (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",dir,got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}
      }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField gr_r failed\n");


  //Test glAx_r
  for(int gg=0;gg<4;gg++){
    int dir = glAx_dirs[gg];

    PropagatorField pla = glAx_r(a, dir);

    fail = false;
    for(size_t x4d=0; x4d< a.size(); x4d++){
      auto aa=*a.site_ptr(x4d);
      aa.glAx(dir);
      for(int i=0;i<aa.nScalarType();i++){
	auto got = Reduce( pla.site_ptr(x4d)->scalarTypePtr()[i] );
	auto expect = Reduce( aa.scalarTypePtr()[i] );
	
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: glAx_r[%d] (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",dir,got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}
      }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField glAx_r failed\n");


  //Test grAx_r
  for(int gg=0;gg<4;gg++){
    int dir = glAx_dirs[gg];

    PropagatorField pla = grAx_r(a, dir);

    fail = false;
    for(size_t x4d=0; x4d< a.size(); x4d++){
      auto aa=*a.site_ptr(x4d);
      aa.grAx(dir);
      for(int i=0;i<aa.nScalarType();i++){
	auto got = Reduce( pla.site_ptr(x4d)->scalarTypePtr()[i] );
	auto expect = Reduce( aa.scalarTypePtr()[i] );
	
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: grAx_r[%d] (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",dir,got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}
      }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField grAx_r failed\n");


  //Test pl
  FlavorMatrixType ftypes[7] = {F0, F1, Fud, sigma0, sigma1, sigma2, sigma3};
  for(int tt=0;tt<7;tt++){
    FlavorMatrixType type = ftypes[tt];

    PropagatorField pla(a);
    pl(pla, type);

    fail = false;
    for(size_t x4d=0; x4d< a.size(); x4d++){
      auto aa=*a.site_ptr(x4d);
      aa.pl(type);
      for(int i=0;i<aa.nScalarType();i++){
	auto got = Reduce( pla.site_ptr(x4d)->scalarTypePtr()[i] );
	auto expect = Reduce( aa.scalarTypePtr()[i] );
	
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: pl[%d] (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",tt,got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}
      }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField pl failed\n");


  //Test pr
  for(int tt=0;tt<7;tt++){
    FlavorMatrixType type = ftypes[tt];

    PropagatorField pla(a);
    pr(pla, type);

    fail = false;
    for(size_t x4d=0; x4d< a.size(); x4d++){
      auto aa=*a.site_ptr(x4d);
      aa.pr(type);
      for(int i=0;i<aa.nScalarType();i++){
	auto got = Reduce( pla.site_ptr(x4d)->scalarTypePtr()[i] );
	auto expect = Reduce( aa.scalarTypePtr()[i] );
	
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: pr[%d] (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",tt,got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}
      }
    }
  }
  if(fail) ERR.General("","","CPSmatrixField pr failed\n");


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

  {
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

  { //test it works a second time! (plans are persistent)
    bool do_dirs[4] = {0,1,1,0};
    
    FieldType in(fp);
    in.testRandom();

    FieldType out1(fp);
    if(!UniqueID()) printf("FFT orig (2)\n");
    fft(out1,in,do_dirs);

    FieldType out2(fp);
    if(!UniqueID()) printf("FFT opt (2)\n");
    fft_opt(out2,in,do_dirs);

    assert( out1.equals(out2, 1e-8, true ) );
    printf("Passed FFT test\n");

    //Test inverse
    FieldType inv(fp);
    if(!UniqueID()) printf("FFT opt inverse (2)\n");
    fft_opt(inv,out2,do_dirs,true);

    assert( inv.equals(in, 1e-8, true ) );
    printf("Passed FFT inverse test (2)\n");  
  }

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

template<typename StandardA2Apolicies, typename GridA2Apolicies>
void testMesonFieldNormGridStd(const A2AArg &a2a_args, const double tol){
  std::cout << "Running testMesonFieldNorm" << std::endl;
  A2Aparams params(a2a_args);
  std::vector<A2AmesonField<StandardA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_std(1);
  mf_std[0].setup(params,params,0,0);
  mf_std[0].testRandom();

  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>  > mf_grid(1);
  mf_grid[0].setup(params,params,0,0);
  copy(mf_grid,mf_std);

  assert(compare(mf_grid,mf_std,tol));

  double n1 = mf_grid[0].norm2(), n2 = mf_std[0].norm2();
  if(fabs(n1-n2) > tol){
    std::cout << "testMesonFieldNorm failed: " << n1 << " " << n2 << " diff " << n1 - n2 << std::endl;
    exit(1);
  }
  

  std::cout << "testMesonFieldNorm passed" << std::endl;
}

void testConvertComplexD(){
  std::cout << "Starting testConvertComplexD" << std::endl;
  std::complex<double> std(3.14, 2.22);
  Grid::ComplexD grid(3.14,2.22);

  std::complex<double> grid_conv = convertComplexD(grid);

  std::cout << "Std (" << std.real() << "," << std.imag() << ")  Grid (" << grid_conv.real() << "," << grid_conv.imag() << ")" << std::endl;
  
  assert( fabs( grid_conv.real() - std.real() ) < 1e-12 && fabs( grid_conv.imag() - std.imag() ) < 1e-12 );
  
  std::cout << "testConvertComplexD passed" << std::endl;
}



//Test the openmp Grid implementation vs the non-Grid implementation of type 1
template<typename StandardA2Apolicies, typename GridA2Apolicies>
void testKtoPiPiType1GridOmpStd(const A2AArg &a2a_args,
				const A2AvectorW<GridA2Apolicies> &Wgrid, A2AvectorV<GridA2Apolicies> &Vgrid,
				const A2AvectorW<GridA2Apolicies> &Whgrid, A2AvectorV<GridA2Apolicies> &Vhgrid,
				const A2AvectorW<StandardA2Apolicies> &Wstd, A2AvectorV<StandardA2Apolicies> &Vstd,
				const A2AvectorW<StandardA2Apolicies> &Whstd, A2AvectorV<StandardA2Apolicies> &Vhstd,
				const double tol){
  std::cout << "Starting testKtoPiPiType1GridOmpStd type1 full test\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      
 
  int Lt = GJP.TnodeSites()*GJP.Tnodes();
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::mf_WW mf_WW_grid;
  typedef typename ComputeKtoPiPiGparity<StandardA2Apolicies>::mf_WW mf_WW_std;
  std::vector<mf_WW_grid> mf_kaon_grid(Lt);
  std::vector<mf_WW_std> mf_kaon_std(Lt);
  
  for(int t=0;t<Lt;t++){
    mf_kaon_grid[t].setup(Wgrid,Whgrid,t,t);
    mf_kaon_grid[t].testRandom();

    mf_kaon_std[t].setup(Wstd,Whstd,t,t);
  }
  copy(mf_kaon_std, mf_kaon_grid);
  assert(compare(mf_kaon_std, mf_kaon_grid, 1e-12));

  
  typedef typename ComputeKtoPiPiGparity<GridA2Apolicies>::ResultsContainerType ResultsContainerType_grid;
  typedef typename ComputeKtoPiPiGparity<StandardA2Apolicies>::ResultsContainerType ResultsContainerType_std;
  
  std::vector<int> tsep_k_pi = {3,4};
  std::vector<ResultsContainerType_std> std_r(2);
  std::vector<ResultsContainerType_grid> grid_r(2);

  //int tstep = 2;
  int tstep = 1;
  int tsep_pion = 1;
  ThreeMomentum p_pi1(1,1,1);
  ThreeMomentum p_pi2 = -p_pi1;

  MesonFieldMomentumContainer<GridA2Apolicies> mf_pion_grid;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_grid_tmp(Lt);

  MesonFieldMomentumContainer<StandardA2Apolicies> mf_pion_std;
  std::vector<A2AmesonField<StandardA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_std_tmp(Lt);

  for(int t=0;t<Lt;t++){
    mf_pion_grid_tmp[t].setup(Wgrid,Vgrid,t,t);
    mf_pion_grid_tmp[t].testRandom();

    mf_pion_std_tmp[t].setup(Wstd,Vstd,t,t);
  }
  copy(mf_pion_std_tmp, mf_pion_grid_tmp);
  
  mf_pion_grid.copyAdd(p_pi1, mf_pion_grid_tmp);
  mf_pion_std.copyAdd(p_pi1, mf_pion_std_tmp);
  
  for(int t=0;t<Lt;t++){
    mf_pion_grid_tmp[t].testRandom();
  }
  copy(mf_pion_std_tmp, mf_pion_grid_tmp);
  
  mf_pion_grid.copyAdd(p_pi2, mf_pion_grid_tmp);
  mf_pion_std.copyAdd(p_pi2, mf_pion_std_tmp);

  assert(mf_pion_grid.get(p_pi1).size() ==  Lt);
  assert(mf_pion_grid.get(p_pi2).size() ==  Lt);
  assert(mf_pion_std.get(p_pi1).size() ==  Lt);
  assert(mf_pion_std.get(p_pi2).size() ==  Lt);
    
  assert(compare(mf_pion_grid.get(p_pi1), mf_pion_std.get(p_pi1),1e-12));
  assert(compare(mf_pion_grid.get(p_pi2), mf_pion_std.get(p_pi2),1e-12));

  for(int t=0;t<Lt;t++)
    std::cout << "TEST pi1 " << mf_pion_grid.get(p_pi1)[t].norm2() << " " << mf_pion_std.get(p_pi1)[t].norm2() << std::endl;
  
  for(int t=0;t<Lt;t++)
    std::cout << "TEST pi2 " << mf_pion_grid.get(p_pi2)[t].norm2() << " " << mf_pion_std.get(p_pi2)[t].norm2() << std::endl;

  
  std::cout << "testKtoPiPiType1GridOmpStd computing using SIMD implementation" << std::endl;
  ComputeKtoPiPiGparity<GridA2Apolicies>::type1_omp(grid_r.data(), tsep_k_pi, tsep_pion, tstep, 1,  p_pi1, mf_kaon_grid, mf_pion_grid, Vgrid, Vhgrid, Wgrid, Whgrid);

  std::cout << "testKtoPiPiType1GridOmpStd computing using non-SIMD implementation" << std::endl;
  ComputeKtoPiPiGparity<StandardA2Apolicies>::type1_omp(std_r.data(), tsep_k_pi, tsep_pion, tstep, 1,  p_pi1, mf_kaon_std, mf_pion_std, Vstd, Vhstd, Wstd, Whstd);

  for(int tsep_k_pi_idx=0; tsep_k_pi_idx<2; tsep_k_pi_idx++){
    std::cout << "testKtoPiPiType1GridOmpStd comparing results for tsep_k_pi idx " << tsep_k_pi_idx << std::endl;
      
    if(!compareKtoPiPi(std_r[tsep_k_pi_idx], "std",
  		       grid_r[tsep_k_pi_idx], "grid",
  		       "KtoPiPi type1", tol)){
      ERR.General("","testKtoPiPiType1GridOmpStd","KtoPiPi type1 contract full failed\n");
    }
  }	  
  std::cout << "testKtoPiPiType1GridOmpStd passed" << std::endl;
}




template<typename GridA2Apolicies>
void testKtoPiPiType1FieldFull(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting testKtoPiPiType1FieldFull type1 full test\n";

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
	      printf("Fail rank %d: KtoPiPi type1 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",UniqueID(),got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      printf("Pass rank %d: KtoPiPi type1 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",UniqueID(),got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","testKtoPiPiType1FieldFull","KtoPiPi type1 contract full failed\n");
  std::cout << "testKtoPiPiType1FieldFull passed" << std::endl;
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

  //For type2 we want >1 p_pi1 to test the average over momenta
  ThreeMomentum p_pi1(1,1,1);
  ThreeMomentum p_pi2 = -p_pi1;

  ThreeMomentum p_pi1_2(-1,1,1);
  ThreeMomentum p_pi2_2 = -p_pi1_2;


  MesonFieldMomentumContainer<GridA2Apolicies> mf_pion;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_pion_tmp(Lt);
  for(int t=0;t<Lt;t++){
    mf_pion_tmp[t].setup(Wgrid,Vgrid,t,t);
    mf_pion_tmp[t].testRandom();
  }
  mf_pion.copyAdd(p_pi1, mf_pion_tmp);
  for(int t=0;t<Lt;t++)
    mf_pion_tmp[t].testRandom();
  mf_pion.copyAdd(p_pi2, mf_pion_tmp);

  for(int t=0;t<Lt;t++)
    mf_pion_tmp[t].testRandom();  
  mf_pion.copyAdd(p_pi1_2, mf_pion_tmp);

  for(int t=0;t<Lt;t++)
    mf_pion_tmp[t].testRandom();
  mf_pion.copyAdd(p_pi2_2, mf_pion_tmp);

  std::vector<ThreeMomentum> p_pi1_all(2);
  p_pi1_all[0] = p_pi1;
  p_pi1_all[1] = p_pi1_2;


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
	      printf("Fail: KtoPiPi type2 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      printf("Pass: KtoPiPi type2 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
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
	      printf("Fail: KtoPiPi type3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      printf("Pass: KtoPiPi type3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
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
  if(!UniqueID()) std::cout << "Starting K->sigma type1/2 full test\n";

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
	    if(!UniqueID()) std::cout << "tsep_k_sigma=" << tsep_k_sigma[tsep_k_sigma_idx] << " tK " << t_K << " tdis " << tdis << " C" << cidx << " gcombidx " << gcombidx << std::endl;
	    ComplexD expect = convertComplexD(expect_r[tsep_k_sigma_idx](t_K,tdis,cidx,gcombidx));
	    ComplexD got = convertComplexD(got_r[tsep_k_sigma_idx](t_K,tdis,cidx,gcombidx));
	    
	    double rdiff = fabs(got.real()-expect.real());
	    double idiff = fabs(got.imag()-expect.imag());
	    if(rdiff > tol|| idiff > tol){
	      if(!UniqueID()) printf("Fail: KtoSigma type1/2 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      if(!UniqueID()) printf("Pass: KtoSigma type1/2 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoSigma type1/2 contract full failed on node %d\n", UniqueID());

}



template<typename GridA2Apolicies>
void testKtoSigmaType3FieldFull(const A2AArg &a2a_args, const double tol){
  if(!UniqueID()) std::cout << "Starting K->sigma type3 full test\n";

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
	    if(!UniqueID()) std::cout << "tsep_k_sigma=" << tsep_k_sigma[tsep_k_sigma_idx] << " tK " << t_K << " tdis " << tdis << " C" << cidx << " gcombidx " << gcombidx << std::endl;
	    ComplexD expect = convertComplexD(expect_r[tsep_k_sigma_idx](t_K,tdis,cidx,gcombidx));
	    ComplexD got = convertComplexD(got_r[tsep_k_sigma_idx](t_K,tdis,cidx,gcombidx));
	    
	    double rdiff = fabs(got.real()-expect.real());
	    double idiff = fabs(got.imag()-expect.imag());
	    if(rdiff > tol|| idiff > tol){
	      if(!UniqueID()) printf("Fail: KtoSigma type3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	      fail = true;
	    }else
	      if(!UniqueID()) printf("Pass: KtoSigma type3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  }
	}
      }
    }
  }
  if(fail) ERR.General("","","KtoSigma type3 contract full failed node %d\n", UniqueID());


  for(int tsep_k_sigma_idx=0; tsep_k_sigma_idx<2; tsep_k_sigma_idx++){
    for(int t_K=0;t_K<Lt;t_K++){
      for(int tdis=0;tdis<Lt;tdis++){
	for(int cidx=0; cidx<2; cidx++){
	  if(!UniqueID()) std::cout << "tsep_k_sigma=" << tsep_k_sigma[tsep_k_sigma_idx] << " tK " << t_K << " tdis " << tdis << " mix3(" << cidx << ")" << std::endl;
	  ComplexD expect = convertComplexD(expect_mix_r[tsep_k_sigma_idx](t_K,tdis,cidx));
	  ComplexD got = convertComplexD(got_mix_r[tsep_k_sigma_idx](t_K,tdis,cidx));
	  
	  double rdiff = fabs(got.real()-expect.real());
	  double idiff = fabs(got.imag()-expect.imag());
	  if(rdiff > tol|| idiff > tol){
	    if(!UniqueID()) printf("Fail: KtoSigma mix3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	    fail = true;
	  }else
	    if(!UniqueID()) printf("Pass: KtoSigma mix3 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	}
      }
    }
  }

  if(fail) ERR.General("","","KtoSigma mix3 contract full failed node %d\n", UniqueID());
}




template<typename GridA2Apolicies>
void testKtoSigmaType4FieldFull(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting testKtoSigmaType4FieldFull: K->sigma type4 full test\n";

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
	  if(!UniqueID()) std::cout << "tK " << t_K << " tdis " << tdis << " C" << cidx << " gcombidx " << gcombidx << std::endl;
	  ComplexD expect = convertComplexD(expect_r(t_K,tdis,cidx,gcombidx));
	  ComplexD got = convertComplexD(got_r(t_K,tdis,cidx,gcombidx));
	    
	  double rdiff = fabs(got.real()-expect.real());
	  double idiff = fabs(got.imag()-expect.imag());
	  if(rdiff > tol|| idiff > tol){
	    if(!UniqueID()) printf("Fail: KtoSigma type4 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	    fail = true;
	  }else
	    if(!UniqueID()) printf("Pass: KtoSigma type4 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	}
      }
    }
  }
  
  if(fail) ERR.General("","","KtoSigma type4 contract full failed node %d\n", UniqueID());

  
  for(int t_K=0;t_K<Lt;t_K++){
    for(int tdis=0;tdis<Lt;tdis++){
      for(int cidx=0; cidx<2; cidx++){
	if(!UniqueID()) std::cout << "tK " << t_K << " tdis " << tdis << " mix3(" << cidx << ")" << std::endl;
	ComplexD expect = convertComplexD(expect_mix_r(t_K,tdis,cidx));
	ComplexD got = convertComplexD(got_mix_r(t_K,tdis,cidx));
	  
	double rdiff = fabs(got.real()-expect.real());
	double idiff = fabs(got.imag()-expect.imag());
	if(rdiff > tol|| idiff > tol){
	  if(!UniqueID()) printf("Fail: KtoSigma mix4 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
	  fail = true;
	}else
	  if(!UniqueID()) printf("Pass: KtoSigma mix4 contract got (%g,%g) expect (%g,%g) Diff (%g,%g)\n",got.real(),got.imag(), expect.real(),expect.imag(), expect.real()-got.real(), expect.imag()-got.imag());
      }
    }
  }

  if(fail) ERR.General("","","KtoSigma mix4 contract full failed node %d\n", UniqueID());
  std::cout << "testKtoSigmaType4FieldFull passed" << std::endl;
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
