#pragma once

CPS_START_NAMESPACE

#ifdef USE_GRID

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


CPS_END_NAMESPACE