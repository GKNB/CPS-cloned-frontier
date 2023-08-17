#pragma once

CPS_START_NAMESPACE

//This test will test the reference implementation for packed data against expectation
template<typename A2Apolicies_std>
void testMesonFieldComputeReference(const A2AArg &a2a_args, double tol){

  std::cout << "Starting testMesonFieldComputeReference test of reference implementation " << std::endl;
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
  CPSautoView(mf_p_v,mf_p,HostRead);
  
  bool err = false;
  for(int i=0;i<nv;i++){
    for(int j=0;j<nv;j++){
      const ScalarComplexType &elem_u = mf_u(i,j);
      const ScalarComplexType &elem_p = mf_p_v.elem(i,j);
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

  std::cout << "Passed testMesonFieldComputeReference tests" << std::endl;
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

    CPSautoView(l_v,l,HostRead);
    CPSautoView(r_v,r,HostRead);
    CPSautoView(into_v,into,HostWrite);
    
#pragma omp parallel for
    for(int i = 0; i < nmodes_l; i++){
      T mf_accum;

      modeIndexSet i_high_unmapped; if(i>=nl_l) into.getRowParams().indexUnmap(i-nl_l,i_high_unmapped);

      for(int j = 0; j < nmodes_r; j++) {
	modeIndexSet j_high_unmapped; if(j>=nl_r) into.getColParams().indexUnmap(j-nl_r,j_high_unmapped);

	mf_accum = 0.;

	for(int p_3d = 0; p_3d < size_3d; p_3d++) {
	  SCFvectorPtr<T> lscf = l_v.getFlavorDilutedVect(i,i_high_unmapped,p_3d,t_lcl); //dilute flavor in-place if it hasn't been already
	  SCFvectorPtr<T> rscf = r_v.getFlavorDilutedVect(j,j_high_unmapped,p_3d,t_lcl);

	  FlavorMatrixGeneral<T> lg5r; lg5r.zero();
	  inner.spinColorContract(lg5r,lscf,rscf);
	  doAccum(mf_accum, TransLeftTrace(lg5r, s3) ); //still agrees
	}
	into_v(i,j) = mf_accum; //downcast after accumulate      
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
//This test will ensure the SIMD implementation gives the same result as the reference implementation
template<typename A2Apolicies_std, typename A2Apolicies_grid>
void testMesonFieldComputePackedReferenceSIMD(const A2AArg &a2a_args, double tol, const typename SIMDpolicyBase<4>::ParamType &simd_dims){

  std::cout << "Starting testMesonFieldComputeReferenceSIMD test of reference implementation " << std::endl;
  typedef flavorMatrixSpinColorContract<15,true,false> SCconPol;
  typedef GparityNoSourceInnerProduct<typename A2Apolicies_std::ComplexType, SCconPol> InnerProductTypeU;
  InnerProductTypeU inner_u(sigma3);

  typedef GparityNoSourceInnerProduct<typename A2Apolicies_grid::ComplexType, SCconPol> InnerProductTypeS;
  InnerProductTypeS inner_s(sigma3);

  A2AvectorWfftw<A2Apolicies_std> Wf_p(a2a_args);
  A2AvectorVfftw<A2Apolicies_std> Vf_p(a2a_args);
  Wf_p.testRandom();
  Vf_p.testRandom();

  //Get SIMD versions
  A2AvectorWfftw<A2Apolicies_grid> Wf_s(a2a_args,simd_dims);
  A2AvectorVfftw<A2Apolicies_grid> Vf_s(a2a_args,simd_dims);
  for(int i=0;i<Wf_s.getNmodes();i++) Wf_s.getMode(i).importField(Wf_p.getMode(i));
  for(int i=0;i<Vf_s.getNmodes();i++) Vf_s.getMode(i).importField(Vf_p.getMode(i));
  
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
  compute_simple(mf_u,Wf_u,inner_u,Vf_u,0);

  typedef A2AmesonField<A2Apolicies_grid,A2AvectorWfftw,A2AvectorVfftw> MFtype;    
  MFtype mf_s;
  std::cout << "Computing MF using packed reference implementation" << std::endl;
  mf_s.compute(Wf_s,inner_s,Vf_s,0);
  CPSautoView(mf_s_v,mf_s,HostRead);
  
  bool err = false;
  for(int i=0;i<nv;i++){
    for(int j=0;j<nv;j++){
      const ScalarComplexType &elem_u = mf_u(i,j);
      const ScalarComplexType &elem_p = mf_s_v.elem(i,j);
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

  std::cout << "Passed testMesonFieldComputePackedReferenceSIMD tests" << std::endl;
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
  std::cout << "USE_GRID is enabled!" << std::endl;
#endif  

#ifdef GRID_CUDA
  std::cout << "GRID_CUDA is enabled!" << std::endl;
#endif  

#ifdef GRID_HIP
  std::cout << "GRID_HIP is enabled!" << std::endl;
#endif  
 
#ifdef GPU_VEC
  std::cout << "GPU_VEC is enabled!" << std::endl;
#endif  
    	
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
    CPSautoView(mf_grid_t_v,mf_grid[t],HostRead);
    CPSautoView(mf_t_v,mf[t],HostRead);
    
    for(int i=0;i<mf[t].size();i++){
      const Ctype& gd = mf_grid_t_v.ptr()[i];
      const Ctype& cp = mf_t_v.ptr()[i];
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
  typedef A2AvectorV<GridA2Apolicies> Vtype;
  typedef A2AvectorW<GridA2Apolicies> Wtype;
  typedef GparityFlavorProjectedBasicSourceStorage<Vtype,Wtype, InnerProductType> StorageType;
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
  typedef ComputeMesonFields<Vtype,Wtype,StorageType> ComputeType;
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



template<typename A2Apolicies>
void testMultiSource(const A2AArg &a2a_args,Lattice &lat){
  assert(GJP.Gparity());
  
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
  
  //Demonstrate relation between FFTW fields
  typedef typename A2Apolicies::ComplexType mf_Complex;
  typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);

  typedef A2AvectorV<A2Apolicies> Vtype;
  typedef A2AvectorW<A2Apolicies> Wtype;
  
  Wtype W(a2a_args,fp);
  Vtype V(a2a_args,fp);

  W.testRandom();
  V.testRandom();

  int p[3];
  GparityBaseMomentum(p,+1);
  ThreeMomentum pp(p);

  GparityBaseMomentum(p,-1);
  ThreeMomentum pm(p);

  ThreeMomentum pp3 = pp * 3;
  ThreeMomentum pm3 = pm * 3;
  if(1){ //1s + 2s source
    typedef typename A2AflavorProjectedExpSource<typename A2Apolicies::SourcePolicies>::FieldParamType SrcFieldParamType;
    typedef typename A2AflavorProjectedExpSource<typename A2Apolicies::SourcePolicies>::ComplexType SrcComplexType;
    SrcFieldParamType sfp; defaultFieldParams<SrcFieldParamType, SrcComplexType>::get(sfp);

    typedef A2AflavorProjectedExpSource<typename A2Apolicies::SourcePolicies> ExpSrcType;
    typedef A2AflavorProjectedHydrogenSource<typename A2Apolicies::SourcePolicies> HydSrcType;
  
    ExpSrcType _1s_src(2.0, pp.ptr(), sfp);
    HydSrcType _2s_src(2,0,0, 2.0, pp.ptr(), sfp);

    typedef SCFspinflavorInnerProduct<15,mf_Complex,ExpSrcType,true,false> ExpInnerType;
    typedef SCFspinflavorInnerProduct<15,mf_Complex,HydSrcType,true,false> HydInnerType;
  
    ExpInnerType _1s_inner(sigma3, _1s_src);
    HydInnerType _2s_inner(sigma3, _2s_src);

    A2AvectorWfftw<A2Apolicies> Wfftw_pp(a2a_args,fp);
    Wfftw_pp.gaugeFixTwistFFT(W,pp.ptr(),lat);

    A2AvectorVfftw<A2Apolicies> Vfftw_pp(a2a_args,fp);
    Vfftw_pp.gaugeFixTwistFFT(V,pp.ptr(),lat);
  
    std::vector< A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_std_1s_pp_pp;
    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_std_1s_pp_pp, Wfftw_pp, _1s_inner, Vfftw_pp);

    typedef GparityFlavorProjectedBasicSourceStorage<Vtype,Wtype, ExpInnerType> ExpStorageType;
  
    ExpStorageType exp_store_1s_pp_pp(_1s_inner);
    exp_store_1s_pp_pp.addCompute(0,0,pp,pp);

    typename ComputeMesonFields<Vtype,Wtype,ExpStorageType>::WspeciesVector Wspecies(1, &W);
    typename ComputeMesonFields<Vtype,Wtype,ExpStorageType>::VspeciesVector Vspecies(1, &V);

    std::cout << "Start 1s ExpStorage compute\n";
    ComputeMesonFields<Vtype,Wtype,ExpStorageType>::compute(exp_store_1s_pp_pp,Wspecies,Vspecies,lat);

    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    for(int t=0;t<Lt;t++){
      if(!UniqueID()) printf("Comparing test 1 t=%d\n",t);
      assert( exp_store_1s_pp_pp[0][t].equals(mf_std_1s_pp_pp[t],1e-10,true) );
    }
    if(!UniqueID()) printf("Passed equivalence test 1\n");

    typedef Elem<ExpSrcType,Elem<HydSrcType,ListEnd> > SrcList;
    typedef A2AmultiSource<SrcList> MultiSrcType;
    typedef SCFspinflavorInnerProduct<15,mf_Complex,MultiSrcType,true,false> ExpHydMultiInnerType;

    MultiSrcType exp_hyd_multi_src;
    exp_hyd_multi_src.template getSource<0>().setup(2.0,pp.ptr(),sfp);
    exp_hyd_multi_src.template getSource<1>().setup(2,0,0, 2.0, pp.ptr(), sfp);
  
    ExpHydMultiInnerType exp_hyd_multi_inner(sigma3,exp_hyd_multi_src);

    typedef GparityFlavorProjectedBasicSourceStorage<Vtype,Wtype, HydInnerType> HydStorageType;
    HydStorageType exp_store_2s_pp_pp(_2s_inner);
    exp_store_2s_pp_pp.addCompute(0,0,pp,pp);
    exp_store_2s_pp_pp.addCompute(0,0,pm,pp);
    exp_store_2s_pp_pp.addCompute(0,0,pp3,pp);

  
    ComputeMesonFields<Vtype,Wtype,HydStorageType>::compute(exp_store_2s_pp_pp,Wspecies,Vspecies,lat);

  
    typedef GparityFlavorProjectedMultiSourceStorage<Vtype,Wtype, ExpHydMultiInnerType> ExpHydMultiStorageType;
    ExpHydMultiStorageType exp_store_1s_2s_pp_pp(exp_hyd_multi_inner, exp_hyd_multi_src);
    exp_store_1s_2s_pp_pp.addCompute(0,0,pp,pp);

    std::cout << "Start 1s/2s ExpHydMultiStorage compute\n";
    ComputeMesonFields<Vtype,Wtype,ExpHydMultiStorageType>::compute(exp_store_1s_2s_pp_pp,Wspecies,Vspecies,lat);
  
    for(int t=0;t<Lt;t++){
      if(!UniqueID()) printf("Comparing test 2 t=%d\n",t);
      assert( exp_store_1s_2s_pp_pp(0,0)[t].equals(mf_std_1s_pp_pp[t],1e-10,true) );
    }
    if(!UniqueID()) printf("Passed equivalence test 2\n");
    for(int t=0;t<Lt;t++){
      if(!UniqueID()) printf("Comparing test 3 t=%d\n",t);
      assert( exp_store_1s_2s_pp_pp(1,0)[t].equals(exp_store_2s_pp_pp[0][t],1e-10,true) );
    }
    if(!UniqueID()) printf("Passed equivalence test 3\n");
  }

  if(1){ //1s + point source
    if(!UniqueID()) printf("Doing 1s+point source\n");
    typedef typename A2AflavorProjectedExpSource<typename A2Apolicies::SourcePolicies>::FieldParamType SrcFieldParamType;
    typedef typename A2AflavorProjectedExpSource<typename A2Apolicies::SourcePolicies>::ComplexType SrcComplexType;
    SrcFieldParamType sfp; defaultFieldParams<SrcFieldParamType, SrcComplexType>::get(sfp);

    typedef A2AflavorProjectedExpSource<typename A2Apolicies::SourcePolicies> ExpSrcType;
    typedef A2AflavorProjectedPointSource<typename A2Apolicies::SourcePolicies> PointSrcType;
    typedef A2ApointSource<typename A2Apolicies::SourcePolicies> PointSrcBasicType;

    ExpSrcType _1s_src(2.0, pp3.ptr(), sfp);
    PointSrcType _pt_src(sfp);
    PointSrcBasicType _pt_basic_src(sfp);
    
    typedef SCFspinflavorInnerProduct<15,mf_Complex,ExpSrcType,true,false> ExpInnerType;
    typedef SCFspinflavorInnerProduct<15,mf_Complex,PointSrcType,true,false> PointInnerType;
    typedef SCFspinflavorInnerProduct<15,mf_Complex,PointSrcBasicType,true,false> PointBasicInnerType;
  
    ExpInnerType _1s_inner(sigma3, _1s_src);
    PointInnerType _pt_inner(sigma3, _pt_src);
    PointBasicInnerType _pt_basic_inner(sigma3, _pt_basic_src);

    A2AvectorWfftw<A2Apolicies> Wfftw_pp(a2a_args,fp);
    Wfftw_pp.gaugeFixTwistFFT(W,pp.ptr(),lat);

    A2AvectorVfftw<A2Apolicies> Vfftw_pp(a2a_args,fp);
    Vfftw_pp.gaugeFixTwistFFT(V,pp3.ptr(),lat);
  
    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    //Do the point and 1s by regular means
    if(!UniqueID()){ printf("Computing with point source\n"); fflush(stdout); }       
    std::vector< A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_pt_std;
    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_pt_std, Wfftw_pp, _pt_inner, Vfftw_pp);

    if(!UniqueID()){ printf("Computing with 1s source\n"); fflush(stdout); }
    std::vector< A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_1s_std;
    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_1s_std, Wfftw_pp, _1s_inner, Vfftw_pp);

    //1) Check flavor projected point and basic point give the same result (no projector for point)
    {
      if(!UniqueID()){ printf("Computing with non-flavor projected point source\n"); fflush(stdout); }
      std::vector< A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_basic;
      A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_basic, Wfftw_pp, _pt_basic_inner, Vfftw_pp);
      
      for(int t=0;t<Lt;t++){
	if(!UniqueID()) printf("1) Comparing flavor projected point src to basic point src t=%d\n",t);
	assert( mf_pt_std[t].equals(mf_basic[t],1e-10,true) );
      }
      if(!UniqueID()) printf("Passed point check test\n");
    }
    
    //Prepare the compound src
    typedef Elem<ExpSrcType,Elem<PointSrcType,ListEnd> > SrcList;
    typedef A2AmultiSource<SrcList> MultiSrcType;

    if(1){
      typedef SCFspinflavorInnerProduct<15,mf_Complex,MultiSrcType,true,false> MultiInnerType;
      
      MultiSrcType multi_src;    
      multi_src.template getSource<0>().setup(2.0,pp3.ptr(),sfp);
      multi_src.template getSource<1>().setup(sfp);      
      
      MultiInnerType multi_inner(sigma3,multi_src);
      
      typedef GparityFlavorProjectedMultiSourceStorage<Vtype,Wtype, MultiInnerType> MultiStorageType;
      MultiStorageType store(multi_inner, multi_src);
      store.addCompute(0,0,pp,pp3);

      std::cout << "Start 1s/point MultiStorage compute\n";
      typename ComputeMesonFields<Vtype,Wtype,MultiStorageType>::WspeciesVector Wspecies(1, &W);
      typename ComputeMesonFields<Vtype,Wtype,MultiStorageType>::VspeciesVector Vspecies(1, &V);

      ComputeMesonFields<Vtype,Wtype,MultiStorageType>::compute(store,Wspecies,Vspecies,lat);
  
      //Test 1s
      for(int t=0;t<Lt;t++){
	if(!UniqueID()) printf("Comparing 1s t=%d\n",t);
	assert( store(0,0)[t].equals(mf_1s_std[t],1e-6,true) );
      }
      if(!UniqueID()) printf("Passed 1s multisrc equivalence test\n");
      
      //Test point
      for(int t=0;t<Lt;t++){
	if(!UniqueID()) printf("Comparing point t=%d\n",t);
	assert( store(1,0)[t].equals(mf_pt_std[t],1e-6,true) );
      }
      if(!UniqueID()) printf("Passed point multisrc equivalence test\n");
    }

    //Test the compound shift source also
    {
      typedef GparitySourceShiftInnerProduct<mf_Complex,MultiSrcType, flavorMatrixSpinColorContract<15,true,false> > MultiInnerType;
      
      MultiSrcType multi_src;    
      multi_src.template getSource<0>().setup(2.0,pp3.ptr(),sfp);
      multi_src.template getSource<1>().setup(sfp);      
      
      MultiInnerType multi_inner(sigma3,multi_src);
      
      typedef GparityFlavorProjectedShiftSourceStorage<Vtype,Wtype, MultiInnerType> MultiStorageType;
      MultiStorageType store(multi_inner, multi_src);
      store.addCompute(0,0,pp,pp3);

      if(!UniqueID()){ printf("Start 1s/point shift multiStorage compute\n"); fflush(stdout); }
      typename ComputeMesonFields<Vtype,Wtype,MultiStorageType>::WspeciesVector Wspecies(1, &W);
      typename ComputeMesonFields<Vtype,Wtype,MultiStorageType>::VspeciesVector Vspecies(1, &V);

      ComputeMesonFields<Vtype,Wtype,MultiStorageType>::compute(store,Wspecies,Vspecies,lat);
  
      //Test 1s
      for(int t=0;t<Lt;t++){
	if(!UniqueID()) printf("Comparing shift 1s t=%d\n",t);
	assert( store(0,0)[t].equals(mf_1s_std[t],1e-6,true) );
      }
      if(!UniqueID()) printf("Passed 1s shift multisrc equivalence test\n");
      
      //Test point
      for(int t=0;t<Lt;t++){
	if(!UniqueID()) printf("Comparing shift point t=%d\n",t);
	assert( store(1,0)[t].equals(mf_pt_std[t],1e-6,true) );
      }
      if(!UniqueID()) printf("Passed point shift multisrc equivalence test\n");
    }
  }
  
}



//Test the compute-many storage that sums meson fields on the fly
template<typename A2Apolicies>
void testSumSource(const A2AArg &a2a_args,Lattice &lat){
  assert(GJP.Gparity());
  
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
  
  int Lt = GJP.TnodeSites() * GJP.Tnodes();

  typedef typename A2Apolicies::ComplexType mf_Complex;
  typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);
  
  A2AvectorW<A2Apolicies> W(a2a_args,fp);
  A2AvectorV<A2Apolicies> V(a2a_args,fp);
  W.testRandom();
  V.testRandom();


  std::vector<ThreeMomentum> p_wdag;
  std::vector<ThreeMomentum> p_v;



  //Total mom (-2,*,*)

  //Base
  p_wdag.push_back(ThreeMomentum(-3,0,0));
  p_v.push_back(ThreeMomentum(1,0,0));
  
  //Symmetrized (lives in same momentum set as base)
  p_wdag.push_back(ThreeMomentum(1,0,0));
  p_v.push_back(ThreeMomentum(-3,0,0));
  
  //Alt (lives in other momentum set)
  p_wdag.push_back(ThreeMomentum(3,0,0));
  p_v.push_back(ThreeMomentum(-5,0,0));
  
  //Alt symmetrized
  p_wdag.push_back(ThreeMomentum(-5,0,0));
  p_v.push_back(ThreeMomentum(3,0,0));

  int nmom = p_v.size();

  for(int i=1;i<3;i++){
    if(GJP.Bc(i) == BND_CND_GPARITY){
      for(int p=0;p<nmom;p++){
	p_wdag[p](i) = p_wdag[p](0);
	p_v[p](i) = p_v[p](0);
      }
    }
  }
  typedef A2AvectorV<A2Apolicies> Vtype;
  typedef A2AvectorW<A2Apolicies> Wtype;
  
  typedef A2AflavorProjectedExpSource<typename A2Apolicies::SourcePolicies> ExpSrcType;
  typedef typename ExpSrcType::FieldParamType SrcFieldParamType;
  typedef typename ExpSrcType::ComplexType SrcComplexType;
  SrcFieldParamType sfp; defaultFieldParams<SrcFieldParamType, SrcComplexType>::get(sfp);

  typedef SCFspinflavorInnerProduct<15,mf_Complex,ExpSrcType,true,false> ExpInnerType;
  
  ExpSrcType src(2.0, p_v[0].ptr(), sfp); //momentum is not relevant as it is shifted internally
  ExpInnerType inner(sigma3, src);

  typedef GparityFlavorProjectedBasicSourceStorage<Vtype,Wtype, ExpInnerType> BasicStorageType;

  typename ComputeMesonFields<Vtype,Wtype,BasicStorageType>::WspeciesVector Wspecies(1, &W);
  typename ComputeMesonFields<Vtype,Wtype,BasicStorageType>::VspeciesVector Vspecies(1, &V);

  BasicStorageType store_basic(inner);
  for(int p=0;p<nmom;p++){
    store_basic.addCompute(0,0,-p_wdag[p],p_v[p]);
  }

  ComputeMesonFields<Vtype,Wtype,BasicStorageType>::compute(store_basic,Wspecies,Vspecies,lat);

  typedef std::vector< A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > MFvectorType;
  
  for(int p=0;p<nmom;p++)
    nodeGetMany(1, &store_basic(p));

  MFvectorType mf_basic = store_basic(0);
  for(int t=0;t<Lt;t++){
    for(int p=1;p<nmom;p++){
      mf_basic[t].plus_equals(store_basic(p)[t]);
    }
    mf_basic[t].times_equals(1./nmom);
  }
  
  std::vector< std::pair<ThreeMomentum,ThreeMomentum> > set_mom;
  for(int p=0;p<nmom;p++)
    set_mom.push_back( std::pair<ThreeMomentum,ThreeMomentum>(-p_wdag[p],p_v[p]) );
  
  typedef GparityFlavorProjectedSumSourceStorage<Vtype,Wtype, ExpInnerType> SumStorageType;

  SumStorageType store_sum(inner);
  store_sum.addComputeSet(0,0, set_mom);
  
  ComputeMesonFields<Vtype,Wtype,SumStorageType>::compute(store_sum,Wspecies,Vspecies,lat);

  store_sum.sumToAverage();

  nodeGetMany(1, &store_sum(0) );

  for(int t=0;t<Lt;t++){
    if(!UniqueID()) printf("Comparing mf avg t=%d\n",t);
    assert( mf_basic[t].equals( store_sum(0)[t],1e-6,true) );
  }
  if(!UniqueID()) printf("Passed mf avg sum source equivalence test\n");

  typedef typename A2Apolicies::ComplexType VectorComplexType;
  
  typedef GparitySourceShiftInnerProduct<VectorComplexType,ExpSrcType, flavorMatrixSpinColorContract<15,true,false> > ShiftInnerType;
  typedef GparityFlavorProjectedShiftSourceSumStorage<Vtype,Wtype, ShiftInnerType> ShiftSumStorageType;
  
  ShiftInnerType shift_inner(sigma3, src);

  ShiftSumStorageType shift_store_sum(shift_inner,src);
  shift_store_sum.addComputeSet(0,0, set_mom, true);
  
  ComputeMesonFields<Vtype,Wtype,ShiftSumStorageType>::compute(shift_store_sum,Wspecies,Vspecies,lat);

  typedef std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mfVector;
  const mfVector &avgd = shift_store_sum(0);
  printf("Index 0 points to %p\n", &avgd); fflush(stdout);
  
  nodeGetMany(1, &shift_store_sum(0) );

  for(int t=0;t<Lt;t++){
    if(!UniqueID()) printf("Comparing mf avg t=%d\n",t);
    assert( mf_basic[t].equals( shift_store_sum(0)[t],1e-6,true) );
  }
  if(!UniqueID()) printf("Passed mf avg sum source equivalence test\n");
}



#ifdef USE_GRID
template<typename vComplexType, bool conj_left, bool conj_right>
class GridVectorizedSpinColorContractBasic{
public:
  inline static vComplexType g5(const vComplexType *const l, const vComplexType *const r){
    const static int sc_size =12;
    const static int half_sc = 6;

    vComplexType v3; zeroit(v3);

    for(int i = half_sc; i < sc_size; i++){ 
      v3 -= MconjGrid<vComplexType,conj_left,conj_right>::doit(l+i,r+i);
    }
    for(int i = 0; i < half_sc; i ++){ 
      v3 += MconjGrid<vComplexType,conj_left,conj_right>::doit(l+i,r+i);
    }
    return v3;
  }
};


#endif

template<typename mf_Complex>
void testGridg5Contract(){
#ifdef USE_GRID
  Grid::Vector<mf_Complex> vec1(12);
  Grid::Vector<mf_Complex> vec2(12);
  for(int i=0;i<12;i++){
    vec1[i] = randomvType<mf_Complex>();
    vec2[i] = randomvType<mf_Complex>();
  }

  mf_Complex a = GridVectorizedSpinColorContractBasic<mf_Complex,true,false>::g5(vec1.data(),vec2.data());
  mf_Complex b = GridVectorizedSpinColorContract<mf_Complex,true,false>::g5(vec1.data(),vec2.data());
  assert(vTypeEquals(a,b,1e-6,true) == true);
  if(!UniqueID()){ printf("Passed g5 contract repro\n"); fflush(stdout); }
#endif
}


CPS_END_NAMESPACE
