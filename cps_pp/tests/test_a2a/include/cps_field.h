#pragma once

CPS_START_NAMESPACE

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
    {
      CPSautoView(v,cpscp2,HostWrite);
      lattice.ImportFermion((Vector*)v.ptr(), fivedin);
    }

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


void testCPSfieldImpex(){
  { //4D fields
    typedef CPSfermion4D<cps::ComplexD> CPSfermion4DBasic;
    CPSfermion4DBasic a;
    a.testRandom();
    
    {
      CPSfermion4DBasic b;
      a.exportField(b);
      CPSfermion4DBasic c;
      c.importField(b);
      assert( a.equals(c) );
    }

    {
      CPSfermion4DBasic b_odd, b_even;
      IncludeCBsite<4> odd_mask(1);
      IncludeCBsite<4> even_mask(0);
      a.exportField(b_odd, &odd_mask);
      a.exportField(b_even, &even_mask);
          
      CPSfermion4DBasic c;
      c.importField(b_odd, &odd_mask);
      c.importField(b_even, &even_mask);

      assert( a.equals(c) );
    }

#ifdef USE_GRID
    typedef CPSfermion4D<Grid::vComplexD, FourDSIMDPolicy<DynamicFlavorPolicy>,UVMallocPolicy> CPSfermion4DGrid;
    typedef typename CPSfermion4DGrid::InputParamType CPSfermion4DGridParams;
    CPSfermion4DGridParams gp;
    setupFieldParams<CPSfermion4DGrid>(gp);
    
    {
      CPSfermion4DGrid b(gp);
      a.exportField(b);
      
      CPSfermion4DBasic c;
      c.importField(b);
      
      assert( a.equals(c) );
    }
    {
      CPSfermion4DGrid b(gp);
      b.importField(a);
      
      CPSfermion4DBasic c;
      b.exportField(c);
      
      assert( a.equals(c) );
    }

    {
      CPSfermion4DGrid b_odd(gp), b_even(gp);
      IncludeCBsite<4> odd_mask(1);
      IncludeCBsite<4> even_mask(0);
      a.exportField(b_odd, &odd_mask);
      a.exportField(b_even, &even_mask);
          
      CPSfermion4DBasic c;
      c.importField(b_odd, &odd_mask);
      c.importField(b_even, &even_mask);

      assert( a.equals(c) );
    }
#endif //USE_GRID
    
  }

  

  { //5D fields
    typedef CPSfermion5D<cps::ComplexD> CPSfermion5DBasic;
    CPSfermion5DBasic a;
    a.testRandom();

    {
      CPSfermion5DBasic b;
      a.exportField(b);
      CPSfermion5DBasic c;
      c.importField(b);
      assert( a.equals(c) );
    }

    {
      CPSfermion5DBasic b_odd, b_even;
      IncludeCBsite<5> odd_mask(1); //4d prec
      IncludeCBsite<5> even_mask(0);
      a.exportField(b_odd, &odd_mask);
      a.exportField(b_even, &even_mask);
          
      CPSfermion5DBasic c;
      c.importField(b_odd, &odd_mask);
      c.importField(b_even, &even_mask);

      assert( a.equals(c) );
    }

    {//The reduced size checkerboarded fields
      CPSfermion5Dcb4Dodd<cps::ComplexD> b_odd;
      CPSfermion5Dcb4Deven<cps::ComplexD> b_even;

      IncludeCBsite<5> odd_mask(1); //4d prec
      IncludeCBsite<5> even_mask(0);
      a.exportField(b_odd, &odd_mask);
      a.exportField(b_even, &even_mask);
          
      CPSfermion5DBasic c;
      c.importField(b_odd, &odd_mask);
      c.importField(b_even, &even_mask); //shouldn't need mask because only the cb sites are contained in the imported field but it disables the site number check

      assert( a.equals(c) );
    }    
  }//end of 5d field testing
}


#ifdef USE_GRID

template<typename GridA2Apolicies>
void testGridFieldImpex(typename GridA2Apolicies::FgridGFclass &lattice){

  { //test my peek poke
    typedef Grid::iVector<Grid::iScalar<Grid::vRealD>, 3> vtype;
    typedef typename Grid::GridTypeMapper<vtype>::scalar_object stype;
    typedef typename Grid::GridTypeMapper<vtype>::scalar_type rtype;

    const int Nsimd = vtype::vector_type::Nsimd();
    
    vtype* vp = (vtype*)memalign(128,sizeof(vtype));
    stype* sp = (stype*)memalign(128,Nsimd*sizeof(stype));
    stype* sp2 = (stype*)memalign(128,Nsimd*sizeof(stype));

    for(int i=0;i<Nsimd;i++)
      for(int j=0;j<sizeof(stype)/sizeof(rtype);j++)
	(  (rtype*)(sp+i) )[j] = rtype(j+Nsimd*i);

    std::cout << "Poking:\n";
    for(int i=0;i<Nsimd;i++) std::cout << sp[i] << std::endl;

    
    for(int lane=0;lane<Nsimd;lane++)
      pokeLane(*vp, sp[lane], lane);

    
    std::cout << "\nAfter poke: " << *vp << std::endl;


    std::cout << "Peeked:\n";    
    for(int lane=0;lane<Nsimd;lane++){
      peekLane(sp[lane], *vp, lane);
      std::cout << sp[lane] << std::endl;
    }

    free(vp);
    free(sp);
    free(sp2);
  }
  
  Grid::GridCartesian *FGrid = lattice.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();
  Grid::GridCartesian *UGrid = lattice.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();

  typedef typename GridA2Apolicies::GridFermionField GridFermionField;

  typedef CPSfermion5D<cps::ComplexD> CPSfermion5DBasic;
  CPSfermion5DBasic a;
  a.testRandom();

  GridFermionField a_grid(FGrid);
  a.exportGridField(a_grid);

  {
    CPSfermion5DBasic b;
    b.importGridField(a_grid);
    assert(b.equals(a));    
  }

  {
    CPSfermion5DBasic b_odd, b_even;
    IncludeCBsite<5> odd_mask(1); //4d prec
    IncludeCBsite<5> even_mask(0);
    b_odd.importGridField(a_grid, &odd_mask);
    b_even.importGridField(a_grid, &even_mask);
          
    CPSfermion5DBasic c;
    c.importField(b_odd, &odd_mask);
    c.importField(b_even, &even_mask);

    assert( a.equals(c) );
  }

  
  {//The reduced size checkerboarded fields
    CPSfermion5Dcb4Dodd<cps::ComplexD> b_odd;
    CPSfermion5Dcb4Deven<cps::ComplexD> b_even;

    IncludeCBsite<5> odd_mask(1); //4d prec
    IncludeCBsite<5> even_mask(0);
    b_odd.importGridField(a_grid, &odd_mask);
    b_even.importGridField(a_grid, &even_mask);
          
    CPSfermion5DBasic c;
    c.importField(b_odd, &odd_mask);
    c.importField(b_even, &even_mask); //shouldn't need mask because only the cb sites are contained in the imported field but it disables the site number check

    assert( a.equals(c) );
  }    
}

#endif //USE_GRID



void testCPSfieldIO(){
  if(!UniqueID()) printf("testCPSfieldIO called\n");

  CPSfield_checksumType cksumtype[2] = { checksumBasic, checksumCRC32 };
  FP_FORMAT fileformat[2] = { FP_IEEE64BIG, FP_IEEE64LITTLE };
  
  for(int i=0;i<2;i++){
    for(int j=0;j<2;j++){
      {
	CPSfermion4D<cps::ComplexD> a;
	a.testRandom();
    
	a.writeParallel("field", fileformat[j], cksumtype[i]);
    
	CPSfermion4D<cps::ComplexD> b;
	b.readParallel("field");
    
	assert( a.equals(b) );
      }
#ifdef USE_GRID
      {
	//Native write with SIMD intact
	typedef CPSfield<Grid::vComplexD,12,FourDSIMDPolicy<DynamicFlavorPolicy>,UVMallocPolicy> GridFieldType;
	typedef CPSfield<cps::ComplexD,12,FourDpolicy<DynamicFlavorPolicy> > ScalarFieldType;
	typedef GridFieldType::InputParamType ParamType;

	ParamType params;
	GridFieldType::SIMDdefaultLayout(params, Grid::vComplexD::Nsimd());
	
	GridFieldType a(params);
	a.testRandom();

	a.writeParallel("field_simd", fileformat[j], cksumtype[i]);
    
	GridFieldType b(params);
	b.readParallel("field_simd");
    
	assert( a.equals(b) );

	//Impex to non-SIMD
	NullObject null;
	ScalarFieldType c(null);
	c.importField(a);

	c.writeParallel("field_scalar", fileformat[j], cksumtype[i]);

	ScalarFieldType d(null);
	d.readParallel("field_scalar");
	b.importField(d);
	
	assert( a.equals(b) );
      }
#endif
      
      {
	CPScomplex4D<cps::ComplexD> a;
	a.testRandom();
    
	a.writeParallel("field", fileformat[j], cksumtype[i]);
    
	CPScomplex4D<cps::ComplexD> b;
	b.readParallel("field");
    
	assert( a.equals(b) );
      }

      {
	typedef CPSfermion5D<cps::ComplexD> CPSfermion5DBasic;
	CPSfermion5DBasic a;
	a.testRandom();

	CPSfermion5Dcb4Dodd<cps::ComplexD> b_odd;
	CPSfermion5Dcb4Deven<cps::ComplexD> b_even;
    
	IncludeCBsite<5> odd_mask(1); //4d prec
	IncludeCBsite<5> even_mask(0);
	a.exportField(b_odd, &odd_mask);
	a.exportField(b_even, &even_mask);

	b_odd.writeParallel("field_odd", fileformat[j], cksumtype[i]);
	b_even.writeParallel("field_even", fileformat[j], cksumtype[i]);
    
	CPSfermion5Dcb4Dodd<cps::ComplexD> c_odd;
	CPSfermion5Dcb4Deven<cps::ComplexD> c_even;
	c_odd.readParallel("field_odd");
	c_even.readParallel("field_even");

	CPSfermion5DBasic d;
	d.importField(c_odd, &odd_mask);
	d.importField(c_even, &even_mask); 
    
	assert( a.equals(d) );
      }
    }
  }

  //Test parallel write with separate metadata
  
  {
    FP_FORMAT fileformat[2] = { FP_IEEE64BIG, FP_IEEE64LITTLE };
    for(int i=0;i<2;i++){
      {
	CPSfermion4D<cps::ComplexD> a;
	a.testRandom();
	
	a.writeParallelSeparateMetadata("field_split", fileformat[i]);
	
	CPSfermion4D<cps::ComplexD> b;
	b.readParallelSeparateMetadata("field_split");
	assert(a.equals(b));
      }
      {
    	CPSfermion4D<cps::ComplexD> a;
    	CPSfermion4D<cps::ComplexD> b;
    	a.testRandom();
    	b.testRandom();

    	typedef typename baseCPSfieldType<CPSfermion4D<cps::ComplexD> >::type baseField;
	
    	std::vector<baseField const*> ptrs_wr(2);
    	ptrs_wr[0] = &a;
    	ptrs_wr[1] = &b;

    	writeParallelSeparateMetadata<typename baseField::FieldSiteType,baseField::FieldSiteSize,
				      typename baseField::FieldMappingPolicy> wr(fileformat[i]);

	wr.writeManyFields("field_split_multi", ptrs_wr);
	
	CPSfermion4D<cps::ComplexD> c;
    	CPSfermion4D<cps::ComplexD> d;

	std::vector<baseField*> ptrs_rd(2);
    	ptrs_rd[0] = &c;
    	ptrs_rd[1] = &d;

	readParallelSeparateMetadata<typename baseField::FieldSiteType,baseField::FieldSiteSize,
				     typename baseField::FieldMappingPolicy> rd;
	
	rd.readManyFields(ptrs_rd, "field_split_multi");

	assert(a.equals(c));
	assert(b.equals(d));

#ifdef USE_GRID
	//Test for SIMD types too
	typedef CPSfield<Grid::vComplexD,12,FourDSIMDPolicy<DynamicFlavorPolicy>,UVMallocPolicy> GridFieldType;
	typedef GridFieldType::InputParamType ParamType;

	ParamType params;
	GridFieldType::SIMDdefaultLayout(params, Grid::vComplexD::Nsimd());
	
	GridFieldType asimd(params);
	asimd.importField(a);
	
	GridFieldType bsimd(params);
	bsimd.importField(b);
	
	//First save in SIMD format and re-read in SIMD format
	std::vector<GridFieldType const*> ptrs_wrsimd(2);
	ptrs_wrsimd[0] = &asimd;
	ptrs_wrsimd[1] = &bsimd;
	
	wr.writeManyFields("field_split_multi_simd", ptrs_wrsimd);

	GridFieldType csimd(params);
	GridFieldType dsimd(params);

	std::vector<GridFieldType*> ptrs_rdsimd(2);
	ptrs_rdsimd[0] = &csimd;
	ptrs_rdsimd[1] = &dsimd;

	rd.readManyFields(ptrs_rdsimd, "field_split_multi_simd");
	
	assert(asimd.equals(csimd));
	assert(bsimd.equals(dsimd));

	//Also try loading SIMD field as non-SIMD
	rd.readManyFields(ptrs_rd, "field_split_multi_simd");
	assert(a.equals(c));
	assert(b.equals(d));

	//Finally try loading non-SIMD field as SIMD
	rd.readManyFields(ptrs_rdsimd, "field_split_multi");

	assert(asimd.equals(csimd));
	assert(bsimd.equals(dsimd));	
#endif
      }
      
    }
  }
    
  
}



CPS_END_NAMESPACE
