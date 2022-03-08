#pragma once

CPS_START_NAMESPACE



//Original implementation
template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy, typename ComplexClass>
struct _gauge_fix_site_op_impl_test;

template< typename mf_Complex, typename MappingPolicy, typename AllocPolicy>
struct _gauge_fix_site_op_impl_test<mf_Complex,MappingPolicy,AllocPolicy,complex_double_or_float_mark>{
  CPSfermion<mf_Complex,MappingPolicy,AllocPolicy> &field;

  _gauge_fix_site_op_impl_test(CPSfermion<mf_Complex,MappingPolicy,AllocPolicy> &f, const int num_threads): field(f){}

  void gauge_fix_site_op(const int x4d[], const int f, Lattice &lat, const bool dagger, const int thread){
    typedef typename mf_Complex::value_type mf_Float;
    int i = x4d[0] + GJP.XnodeSites()*( x4d[1] + GJP.YnodeSites()* ( x4d[2] + GJP.ZnodeSites()*x4d[3] ) );
    mf_Complex tmp[3];
    const Matrix* gfmat = lat.FixGaugeMatrix(i,f);
    mf_Complex* sc_base = (mf_Complex*)field.site_ptr(x4d,f); //if Dimension < 4 the site_ptr method will ignore the remaining indices. Make sure this is what you want
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
  CPSfermion<mf_Complex,MappingPolicy,AllocPolicy> &field;

  _gauge_fix_site_op_impl_test(CPSfermion<mf_Complex,MappingPolicy,AllocPolicy> &f, const int num_threads): field(f){
    nsimd = field.Nsimd();
  }
  
  void gauge_fix_site_op(const int x4d[], const int &f, Lattice &lat, const bool dagger, const int thread){
    //x4d is an outer site index
    int nsimd = field.Nsimd();
    int ndim = MappingPolicy::EuclideanDimension;
    assert(ndim == 4);

    //Assemble pointers to the GF matrices for each lane
    std::vector<cps::Complex*> gf_base_ptrs(nsimd);
    int x4d_lane[4];
    int lane_off[4];
    
    for(int lane=0;lane<nsimd;lane++){
      field.SIMDunmap(lane, lane_off);		      
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
    mf_Complex* sc_base = field.site_ptr(x4d,f);
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


CPS_END_NAMESPACE
