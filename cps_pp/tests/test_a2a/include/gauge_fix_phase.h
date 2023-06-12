#pragma once

CPS_START_NAMESPACE



//Original implementation
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy, typename ComplexClass>
struct _gauge_fix_site_op_impl_test;

template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
struct _gauge_fix_site_op_impl_test<mf_Complex,MappingPolicy,AllocPolicy,complex_double_or_float_mark>{
  ViewAutoDestructWrapper<typename CPSfermion<mf_Complex,MappingPolicy,AllocPolicy>::View> field_v;

  _gauge_fix_site_op_impl_test(CPSfermion<mf_Complex,MappingPolicy,AllocPolicy> &f, const int num_threads){
    { CPSautoView(f_v,f,HostRead); }    
    field_v.reset(f.view(HostWrite));
  }

  void gauge_fix_site_op(const int x4d[], const int f, Lattice &lat, const bool dagger, const int thread){
    typedef typename mf_Complex::value_type mf_Float;
    int i = x4d[0] + GJP.XnodeSites()*( x4d[1] + GJP.YnodeSites()* ( x4d[2] + GJP.ZnodeSites()*x4d[3] ) );
    mf_Complex tmp[3];
    const Matrix* gfmat = lat.FixGaugeMatrix(i,f);
    mf_Complex* sc_base = (mf_Complex*)field_v->site_ptr(x4d,f); //if Dimension < 4 the site_ptr method will ignore the remaining indices. Make sure this is what you want
    for(int s=0;s<4;s++){
      memcpy(tmp, sc_base + 3 * s, 3 * sizeof(mf_Complex));
      if(!dagger)
	colorMatrixMultiplyVector<mf_Float,Float>( (mf_Float*)(sc_base + 3*s), (Float*)gfmat, (mf_Float*)tmp);
      else
	colorMatrixDaggerMultiplyVector<mf_Float,Float>( (mf_Float*)(sc_base + 3*s), (Float*)gfmat, (mf_Float*)tmp);      
    }
  }
};


#ifdef USE_GRID
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
struct _gauge_fix_site_op_impl_test<mf_Complex,MappingPolicy,AllocPolicy,grid_vector_complex_mark>{
  typedef typename mf_Complex::scalar_type stype;
  int nsimd;
  ViewAutoDestructWrapper<typename CPSfermion<mf_Complex,MappingPolicy,AllocPolicy>::View> field_v;

  _gauge_fix_site_op_impl_test(CPSfermion<mf_Complex,MappingPolicy,AllocPolicy> &f, const int num_threads){
    { CPSautoView(f_v,f,HostRead); }    
    field_v.reset(f.view(HostWrite));
    nsimd = f.Nsimd();
  }
  
  void gauge_fix_site_op(const int x4d[], const int &f, Lattice &lat, const bool dagger, const int thread){
    //x4d is an outer site index
    int nsimd = mf_Complex::Nsimd();
    int ndim = MappingPolicy::EuclideanDimension;
    assert(ndim == 4);

    //Assemble pointers to the GF matrices for each lane
    std::vector<cps::Complex*> gf_base_ptrs(nsimd);
    int x4d_lane[4];
    int lane_off[4];
    
    for(int lane=0;lane<nsimd;lane++){
      field_v->SIMDunmap(lane, lane_off);		      
      for(int xx=0;xx<4;xx++) x4d_lane[xx] = x4d[xx] + lane_off[xx];
      int gf_off = x4d_lane[0] + GJP.XnodeSites()*( x4d_lane[1] + GJP.YnodeSites()* ( x4d_lane[2] + GJP.ZnodeSites()*x4d_lane[3] ) );
      gf_base_ptrs[lane] = (cps::Complex*)lat.FixGaugeMatrix(gf_off,f);
    }


    //Poke the GFmatrix elements into SIMD vector objects
    stype buf[nsimd];
    mf_Complex gfmat[3][3];
    for(int i=0;i<3;i++){
      for(int j=0;j<3;j++){
	for(int lane=0;lane<nsimd;lane++)
	  buf[lane] = *(gf_base_ptrs[lane] + j + 3*i);
	vset(gfmat[i][j], buf);
      }
    }

    //Do the matrix multiplication
    mf_Complex* sc_base = field_v->site_ptr(x4d,f);
    mf_Complex tmp[3];
    
    for(int s=0;s<4;s++){
      mf_Complex* s_base = sc_base + 3 * s;
      tmp[0] = *(s_base);
      tmp[1] = *(s_base+1);
      tmp[2] = *(s_base+2);
      
      if(!dagger)
	for(int i=0;i<3;i++)
	  s_base[i] = gfmat[i][0]*tmp[0] + gfmat[i][1]*tmp[1] + gfmat[i][2]*tmp[2];
      else
	for(int i=0;i<3;i++)
	  s_base[i] = conjugate(gfmat[0][i])*tmp[0] + conjugate(gfmat[1][i])*tmp[1] + conjugate(gfmat[2][i])*tmp[2];
    }
  }
};
#endif

template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void gaugeFixOrig(CPSfermion4D<mf_Complex,MappingPolicy,AllocPolicy> &field,  Lattice &lat, const bool parallel, const bool dagger){
  _gauge_fix_site_op_impl_test<mf_Complex,MappingPolicy,AllocPolicy,typename ComplexClassify<mf_Complex>::type> op(field, parallel ? omp_get_max_threads() : 1);
  
  if(parallel){
#pragma omp parallel for
    for(size_t fi=0;fi<field.nfsites();fi++){
      int x4d[4]; int f; field.fsiteUnmap(fi,x4d,f);
      op.gauge_fix_site_op(x4d, f, lat,dagger, omp_get_thread_num());
    }
  }else{
    int x4d[4]; int f;
    for(size_t fi=0;fi<field.nfsites();fi++){
      field.fsiteUnmap(fi,x4d,f);
      op.gauge_fix_site_op(x4d, f, lat,dagger, 0);
    }
  }
}




template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
void gaugeFixNew(CPSfermion4D<mf_Complex,MappingPolicy,AllocPolicy> &field,  Lattice &lat, const bool dagger){
  field.gaugeFix(lat,dagger);
}


template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testGaugeFixOrigNew(typename SIMDpolicyBase<4>::ParamType &simd_dims,
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
  
  //Test non-SIMD old vs new
  typename A2Apolicies_std::FermionFieldType field_std_gforig(field_std), field_std_gfnew(field_std);  
  //non-dagger
  gaugeFixOrig(field_std_gforig, lattice, true, false);
  gaugeFixNew(field_std_gfnew, lattice, false);
  compareField(field_std_gforig, field_std_gfnew, "Gauge fix test std non-dagger", 1e-10);
  
  field_std_gforig = field_std; field_std_gfnew = field_std;
  gaugeFixOrig(field_std_gforig, lattice, true, true);
  gaugeFixNew(field_std_gfnew, lattice, true);
  compareField(field_std_gforig, field_std_gfnew, "Gauge fix test std dagger", 1e-10);

  
  //Test SIMD old vs new
  typename A2Apolicies_grid::FermionFieldType field_grid_gforig(field_grid), field_grid_gfnew(field_grid);  
  typename A2Apolicies_std::FermionFieldType field_std_tmp1, field_std_tmp2;

  //non-dagger
  gaugeFixOrig(field_grid_gforig, lattice, true, false);
  gaugeFixNew(field_grid_gfnew, lattice, false);
  field_std_tmp1.importField(field_grid_gforig); field_std_tmp2.importField(field_grid_gfnew);   
  compareField(field_std_tmp1, field_std_tmp2, "Gauge fix test grid non-dagger", 1e-10);
  
  field_grid_gforig = field_grid; field_grid_gfnew = field_grid;
  gaugeFixOrig(field_grid_gforig, lattice, true, true);
  gaugeFixNew(field_grid_gfnew, lattice, true);
  field_std_tmp1.importField(field_grid_gforig); field_std_tmp2.importField(field_grid_gfnew);   
  compareField(field_std_tmp1, field_std_tmp2, "Gauge fix test grid dagger", 1e-10);
}




template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testGaugeFixAndPhasingGridStd(typename SIMDpolicyBase<4>::ParamType &simd_dims,
			    typename A2Apolicies_grid::FgridGFclass &lattice){
  ThreeMomentum p_plus( GJP.Bc(0)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(1)==BND_CND_GPARITY? 1 : 0,
			GJP.Bc(2)==BND_CND_GPARITY? 1 : 0 );
  ThreeMomentum p_minus = -p_plus;

  typedef typename A2Apolicies_std::FermionFieldType StdFermionField;
  typedef typename A2Apolicies_grid::FermionFieldType GridFermionField;
  
  StdFermionField field_std;
  field_std.testRandom();
  GridFermionField field_grid(simd_dims);
  field_grid.importField(field_std);

  std::cout << "Import CPS->CPS/Grid " << field_std.norm2() << " " << field_grid.norm2() << std::endl;

  field_std.gaugeFix(lattice,true);
  field_grid.gaugeFix(lattice,true);

  std::cout << "After gauge fix CPS->CPS/Grid " << field_std.norm2() << " " << field_grid.norm2() << std::endl;

  StdFermionField field_std_tmp;
  field_std_tmp.importField(field_grid);

  compareField(field_std, field_std_tmp, "Gauge fix test", 1e-10);
    
  std::cout << "Phasing with " << p_plus.str() << std::endl;
  field_std.applyPhase(p_plus.ptr(),true);
  field_grid.applyPhase(p_plus.ptr(),true);

  field_std_tmp.importField(field_grid);
  compareField(field_std, field_std_tmp, "Phase test", 1e-10);

  CPSfermion4DglobalInOneDir<typename A2Apolicies_grid::ScalarComplexType, typename GridFermionField::FieldFlavorPolicy, typename GridFermionField::FieldAllocPolicy > dbl_grid(0);
  CPSfermion4DglobalInOneDir<typename A2Apolicies_std::ComplexType, typename StdFermionField::FieldFlavorPolicy, typename StdFermionField::FieldAllocPolicy> dbl_std(0);

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



template<typename A2Apolicies>
void testGaugeFixInvertible(Lattice &lat){
//Test Gauge fixing is reversible
  typedef typename A2Apolicies::FermionFieldType::FieldSiteType mf_Complex;
  typedef typename A2Apolicies::FermionFieldType::FieldMappingPolicy MappingPolicy;
  typedef typename A2Apolicies::FermionFieldType::FieldAllocPolicy AllocPolicy;

  typedef typename MappingPolicy::template Rebase<OneFlavorPolicy>::type OneFlavorMap;
  
  typedef CPSfermion4D<mf_Complex,OneFlavorMap, AllocPolicy> FieldType;
  
  typedef typename FieldType::InputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);
  
  if(lat.FixGaugeKind() == FIX_GAUGE_NONE){
    FixGaugeArg fix_gauge_arg;
    fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
    fix_gauge_arg.hyperplane_start = 0;
    fix_gauge_arg.hyperplane_step = 1;
    fix_gauge_arg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites();
    fix_gauge_arg.stop_cond = 1e-08;
    fix_gauge_arg.max_iter_num = 10000;

    CommonArg common_arg;
  
    AlgFixGauge fix_gauge(lat,&common_arg,&fix_gauge_arg);
    fix_gauge.run();
  }

  FieldType a(fp);
  a.testRandom();
  
  FieldType Va(a);
  Va.gaugeFix(lat,false); //no dagger

  printRow(a,0,"a");
  printRow(Va,0,"Va");
  
  FieldType VdagVa(Va);
  VdagVa.gaugeFix(lat,true); //dagger

  printRow(VdagVa,0,"VdagVa");

  assert( VdagVa.equals(a, 1e-8, true) );

  FieldType diff = VdagVa - a;
  printRow(diff,0,"diff");

  double n2 = diff.norm2();
  printf("Norm diff = %g\n",n2);

  FieldType zro(fp); zro.zero();

  assert( diff.equals(zro,1e-12,true));
}




//Test the application of a gauge rotation using CPSmatrixField vs CPSField
void testGfixCPSmatrixField(Lattice &lat, const SIMDdims<4> &simd_dims){
  std::cout << "Starting testGfixCPSmatrixField" << std::endl;
  //Generate a random gauge fixing matrix
  typedef CPSfield<cps::ComplexD,9,FourDpolicy<OneFlavorPolicy> > GaugeRotLinField;
  typedef CPSfield<cps::ComplexD,4*9,FourDpolicy<OneFlavorPolicy> > GaugeLinField;

  NullObject null_obj;
  GaugeRotLinField gfmat_s(null_obj); 
  gfmat_s.testRandom();

  std::vector<GaugeRotLinField> gfmat_plus_s(4, null_obj);
  for(int i=0;i<4;i++)
    gfmat_plus_s[i] = CshiftCconjBc(gfmat_s, i, -1); //data motion leftward
 
  GaugeLinField gauge_s((cps::ComplexD*)lat.GaugeField(),null_obj);
  std::vector<GaugeRotLinField> gauge_rotated_s(4,null_obj);

  for(int mu=0;mu<4;mu++){
    CPSautoView(gauge_rotated_s_v, gauge_rotated_s[mu],HostWrite);
    CPSautoView(gfmat_s_v, gfmat_s, HostRead);
    CPSautoView(gfmat_plus_mu_v, gfmat_plus_s[mu], HostRead);
    CPSautoView(gauge_s_v, gauge_s,HostWrite);
#pragma omp parallel for
    for(size_t i=0;i<GJP.VolNodeSites();i++){
      Matrix g_plus_dag;  g_plus_dag.Dagger( *((Matrix*)gfmat_plus_mu_v.site_ptr(i)) );
      Matrix g = *((Matrix*)gfmat_s_v.site_ptr(i) );
      Matrix U = *( ((Matrix*)gauge_s_v.site_ptr(i)) + mu );
      *((Matrix*)gauge_rotated_s_v.site_ptr(i) ) = g*U*g_plus_dag;
    }
  }

  //SIMD-ize gauge field
  CPSfield<Grid::vComplexD,4*9,FourDSIMDPolicy<OneFlavorPolicy> > gauge_v(simd_dims);
  gauge_v.importField(gauge_s);
  CPSfield<Grid::vComplexD,9,FourDSIMDPolicy<OneFlavorPolicy> > gfmat_v(simd_dims); 
  gfmat_v.importField(gfmat_s);

  typedef CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > CPSgfMatrixField;
  std::vector< CPSgfMatrixField > Umu(4, simd_dims);
  for(int i=0;i<4;i++) Umu[i] = gaugeFixTest::getUmu(gauge_v,i);
  
  CPSgfMatrixField gfmat = linearRepack<CPScolorMatrix<Grid::vComplexD> >(gfmat_v);
  std::vector< CPSgfMatrixField > gauge_rotated(4, simd_dims);

  for(int mu=0;mu<4;mu++){
    CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > g_plus_mu = CshiftCconjBc(gfmat, mu, -1);
    gauge_rotated[mu] = gaugeFixTest::computeLj(Umu[mu],gfmat,g_plus_mu,mu);
  }

  for(int mu=0;mu<4;mu++){
    auto gauge_rotated_up_v = linearUnpack(gauge_rotated[mu]);
    GaugeRotLinField gauge_rotated_up_s(null_obj); gauge_rotated_up_s.importField(gauge_rotated_up_v);

    GaugeRotLinField diff = gauge_rotated_up_s - gauge_rotated_s[mu];
    double n2 = diff.norm2();
    std::cout << mu << " " << n2 << std::endl;
    if(n2 > 1e-10) ERR.General("","testGfixCPSmatrixField","Test failed");
  }
  std::cout << "testGfixCPSmatrixField passed" << std::endl;
}

//Test the Grid implementation of gauge fixing vs CPS
void testGridGaugeFix(Lattice &lat, double gfix_alpha, const SIMDdims<4> &simd_dims){
  int Lt=GJP.Tnodes()*GJP.TnodeSites();

  FixGaugeType types[2] = {FIX_GAUGE_LANDAU,FIX_GAUGE_COULOMB_T};
  std::string descr[2] = {"Landau","Coulomb-T"};
  int nhyperplane[2] = {1,Lt};
  int orthog_dir[2] = {-1,3};

  for(int type=0;type<2;type++){
    std::cout << "Checking " << descr[type] << " gauge fixing" << std::endl;
    FixGaugeArg farg_orig;
    farg_orig.fix_gauge_kind = types[type];
    farg_orig.hyperplane_start = 0;
    farg_orig.hyperplane_step = 1;
    farg_orig.hyperplane_num = nhyperplane[type];
    farg_orig.stop_cond = 1e-8;
    farg_orig.max_iter_num = 100000;

    FixGaugeArgGrid farg_grid;
    farg_grid.fix_gauge_kind = types[type];
    farg_grid.stop_cond = 1e-8;
    farg_grid.max_iter_num = 100000;
    farg_grid.alpha = gfix_alpha;

    typedef CPSfield<cps::ComplexD,9,FourDpolicy<DynamicFlavorPolicy> > CPSvField;
    typedef CPSfield<cps::ComplexD,9,FourDpolicy<OneFlavorPolicy> > CPS1fvField;

    NullObject null_obj;

    doGaugeFix(lat,false,farg_orig);
    CPSvField orig_gfmat = getGaugeFixingMatrix(lat);
    CPS1fvField orig_gfmat_1f = getGaugeFixingMatrixFlavor0(lat);

    std::cout << "Check CPS gauge fixing condition " << gaugeFixTest::delta(lat, simd_dims, orthog_dir[type]) << std::endl;
    
    //Gauge fix with Grid
    doGaugeFix(lat,false,farg_grid);
   
    CPSvField grid_gfmat = getGaugeFixingMatrix(lat);
    CPS1fvField grid_gfmat_1f = getGaugeFixingMatrixFlavor0(lat);

    //Check
    std::cout << "Check Grid gauge fixing condition " << gaugeFixTest::delta(lat, simd_dims, orthog_dir[type]) << std::endl;

    CPSvField diff = grid_gfmat - orig_gfmat;
    double n2 = diff.norm2();
    
    std::cout << "Test CPS vs Grid gauge fix: " << n2 << " (expect 0)" << std::endl;    

    CPS1fvField diff1f = grid_gfmat_1f - orig_gfmat_1f;
    n2 = diff1f.norm2();
    std::cout << "Test CPS vs Grid gauge fix (f=0): " << n2 << " (expect 0)" << std::endl;

    //Try applying CPS gauge fixing to the config then applying Grid's. It should converge quickly with a unit matrix transform
    CPSfield<cps::ComplexD,4*9,FourDpolicy<DynamicFlavorPolicy> > cps_gauge_s((cps::ComplexD*)lat.GaugeField(),null_obj); //backup original cfg
    
    doGaugeFix(lat,false,farg_orig); //CPS gfix
    std::cout << "Recheck CPS gauge fixing condition " << gaugeFixTest::delta(lat, simd_dims, orthog_dir[type]) << std::endl;
    gaugeFixCPSlattice(lat);
    doGaugeFix(lat,false,farg_orig); //CPS gfix
    std::cout << "Recheck CPS gauge fixing condition 2 " << gaugeFixTest::delta(lat, simd_dims, orthog_dir[type]) << std::endl;

    doGaugeFix(lat,false,farg_grid);
    std::cout << "Check Grid gauge fixing condition on lattice gauge fixed under CPS " << gaugeFixTest::delta(lat, simd_dims, orthog_dir[type]) << std::endl;
    
    //Expect GFmat to be unit matrix
    grid_gfmat_1f = getGaugeFixingMatrixFlavor0(lat);
    CPS1fvField expect(null_obj);
    {
      CPScolorMatrix<cps::Complex> one_c;
      one_c.unit();

      CPSautoView(expect_v,expect,HostWrite);
#pragma omp parallel for
      for(size_t i=0;i<GJP.VolNodeSites();i++){
	memcpy(expect_v.site_ptr(i), &one_c, 9*sizeof(cps::ComplexD));
      }
    }
    CPS1fvField diff_1f = grid_gfmat_1f - expect;
    std::cout << "Check resulting Grid GF matrices are unit matrix: " << diff_1f.norm2() << " (expect 0)" << std::endl;
    
    //Put the gauge field back as it was!
    {
      CPSautoView(cps_gauge_s_v,cps_gauge_s,HostRead);
      memcpy(lat.GaugeField(), cps_gauge_s_v.ptr(), cps_gauge_s_v.size()*sizeof(cps::ComplexD));
    }

  }



}


CPS_END_NAMESPACE
