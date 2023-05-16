#pragma once

CPS_START_NAMESPACE

template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void testvMvGridOrigGparity(const A2AArg &a2a_args, const int nthreads, const double tol){
#define BASIC_VMV
  //#define BASIC_GRID_VMV
#define GRID_VMV
#define GRID_SPLIT_LITE_VMV;
  //#define GRID_FIELD_SIMPLE
  
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
  {
    CPSautoView(mf_grid_v,mf_grid,HostWrite);
    CPSautoView(mf_v,mf,HostRead);
    
    for(int i=0;i<mf.getNrows();i++)
      for(int j=0;j<mf.getNcols();j++)
	mf_grid_v(i,j) = mf_v(i,j); //both are scalar complex
  }
  
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

  if(!UniqueID()){ printf("Orig vMv\n"); fflush(stdout); }
  {
    CPSautoView(V_v,V,HostRead);
    CPSautoView(W_v,W,HostRead);
    CPSautoView(mf_v,mf,HostRead);

    CPSautoView(Vgrid_v,Vgrid,HostRead);
    CPSautoView(Wgrid_v,Wgrid,HostRead);
    CPSautoView(mf_grid_v,mf_grid,HostRead);
    
    for(int top = 0; top < GJP.TnodeSites(); top++){
      if(!UniqueID()){ printf("Timeslice %d\n", top); fflush(stdout); }


      //ORIG VMV
#pragma omp parallel for
      for(int xop=0;xop<orig_3vol;xop++){
	int me = omp_get_thread_num();
	mult(orig_tmp[me], V_v, mf_v, W_v, xop, top, false, true);
	orig_sum[me] += orig_tmp[me];
      }
    
#ifdef BASIC_VMV
      if(!UniqueID()){ printf("Basic vMv\n"); fflush(stdout); }
      //BASIC VMV FOR TESTING
#pragma omp parallel for
      for(int xop=0;xop<orig_3vol;xop++){
	int me = omp_get_thread_num();
	mult_slow(orig_tmp[me], V_v, mf_v, W_v, xop, top, false, true);
	basic_sum[me] += orig_tmp[me];
      }
#endif

#ifdef GRID_VMV
      //GRID VMV
      if(!UniqueID()){ printf("Grid vMv\n"); fflush(stdout); }
#pragma omp parallel for
      for(int xop=0;xop<grid_3vol;xop++){
	int me = omp_get_thread_num();
	mult(grid_tmp[me], Vgrid_v, mf_grid_v, Wgrid_v, xop, top, false, true);
	grid_sum[me] += grid_tmp[me];
      }
#endif

#ifdef BASIC_GRID_VMV
      if(!UniqueID()){ printf("Grid basic vMv\n"); fflush(stdout); }
      //BASIC GRID VMV FOR TESTING
#pragma omp parallel for
      for(int xop=0;xop<grid_3vol;xop++){
	int me = omp_get_thread_num();
	mult_slow(grid_tmp[me], Vgrid_v, mf_grid_v, Wgrid_v, xop, top, false, true);
	basic_grid_sum[me] += grid_tmp[me];
      }
#endif

#ifdef GRID_SPLIT_LITE_VMV
      //SPLIT LITE VMV GRID
      if(!UniqueID()){ printf("Grid splite vMv\n"); fflush(stdout); }

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
  }
    
  for(int i=1;i<nthreads;i++){
    basic_sum[0] += basic_sum[i];
    orig_sum[0] += orig_sum[i];
    basic_grid_sum[0] += basic_grid_sum[i];
    grid_sum[0] += grid_sum[i];
    grid_sum_split_lite[0] += grid_sum_split_lite[i];  
  }
  
  //Offload version computes all x,t, so we just have to sum over 4 volume afterwards
  typedef typename getPropagatorFieldType<GridA2Apolicies, ExplicitCopyPoolAllocPolicy>::type PropagatorField; //works with UVM policy!!!  FIXMEFIXME
  //typedef typename getPropagatorFieldType<GridA2Apolicies, ExplicitCopyAllocPolicy>::type PropagatorField; //works with UVM policy!!!  FIXMEFIXME
  typedef mult_vMv_field<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw, PropagatorField> vMvFieldImpl;
  PropagatorField pfield(simd_dims);

  //mult(pfield, Vgrid, mf_grid, Wgrid, false, true);
  if(!UniqueID()){ printf("Field vMv\n"); fflush(stdout); }
  vMvFieldImpl::optimized(pfield, Vgrid, mf_grid, Wgrid, false, true, 0, GJP.Tnodes()*GJP.TnodeSites()-1);

  std::cout << Grid::GridLogMessage << " vMv field optimized timings" << std::endl;
  mult_vMv_field_offload_timers::get().print();

  
  CPSspinColorFlavorMatrix<grid_Complex> vmv_offload_sum4;
  {
    CPSautoView(pfield_v,pfield,HostRead);
    vmv_offload_sum4.zero();
    for(size_t i=0;i<pfield.size();i++){
      vmv_offload_sum4 += *pfield_v.fsite_ptr(i);
    }
  }

#ifdef GRID_FIELD_SIMPLE
  //Same for simple field version
  if(!UniqueID()){ printf("Field vMv (simple)\n"); fflush(stdout); }
  vMvFieldImpl::simple(pfield, Vgrid, mf_grid, Wgrid, false, true);

  CPSspinColorFlavorMatrix<grid_Complex> vmv_offload_simple_sum4;
  {
    CPSautoView(pfield_v,pfield,HostRead);
    vmv_offload_simple_sum4.zero();
    for(size_t i=0;i<pfield.size();i++){
      vmv_offload_simple_sum4 += *pfield_v.fsite_ptr(i);
    }
  }
#endif
  
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

#ifdef GRID_FIELD_SIMPLE
  if(!compare(orig_sum[0],vmv_offload_simple_sum4,tol)) ERR.General("","","Standard vs Grid field offload simple implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Grid field offload simple implementation test pass\n");
#endif
  
  if(!compare(orig_sum[0],vmv_offload_sum4,tol)) ERR.General("","","Standard vs Grid field offload optimized implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Grid field offload optimized implementation test pass\n");
  
  std::cout << "testvMvGridOrigGparity passed" << std::endl;
}






template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void testvMvGridOrigGparityTblock(A2AArg a2a_args, const int nthreads, const double tol){  
  std::cout << "Starting testvMvGridOrigGparityTblock : vMv tests\n";
  a2a_args.src_width = 2;

  //#define BASIC_VMV
  //#define BASIC_GRID_VMV
  #define GRID_VMV
  #define GRID_SPLIT_LITE_VMV;
  #define FIELD
  //#define GRID_FIELD_SIMPLE
    
  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  typename FourDSIMDPolicy<typename ScalarA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<typename ScalarA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorWfftw<ScalarA2Apolicies> W(a2a_args);
  A2AvectorVfftw<ScalarA2Apolicies> V(a2a_args);
  W.testRandom();
  V.testRandom();

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  assert(V.getNhighModes() == 24*Lt/2);

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
  {
    CPSautoView(mf_grid_v,mf_grid,HostWrite);
    CPSautoView(mf_v,mf,HostRead);
    
    for(int i=0;i<mf.getNrows();i++)
      for(int j=0;j<mf.getNcols();j++)
	mf_grid_v(i,j) = mf_v(i,j); //both are scalar complex
  }
    
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

  {
    CPSautoView(V_v,V,HostRead);
    CPSautoView(W_v,W,HostRead);
    CPSautoView(mf_v,mf,HostRead);

    CPSautoView(Vgrid_v,Vgrid,HostRead);
    CPSautoView(Wgrid_v,Wgrid,HostRead);
    CPSautoView(mf_grid_v,mf_grid,HostRead);
   
    for(int top = 0; top < GJP.TnodeSites(); top++){
      if(!UniqueID()){ printf("Timeslice %d\n", top); fflush(stdout); }

      if(!UniqueID()){ printf("Orig vMv\n"); fflush(stdout); }
      //ORIG VMV
#pragma omp parallel for
      for(int xop=0;xop<orig_3vol;xop++){
	int me = omp_get_thread_num();
	mult(orig_tmp[me], V_v, mf_v, W_v, xop, top, false, true);
	orig_sum[me] += orig_tmp[me];
      }
    
#ifdef BASIC_VMV
      if(!UniqueID()){ printf("Basic vMv\n"); fflush(stdout); }
      //BASIC VMV FOR TESTING
#pragma omp parallel for
      for(int xop=0;xop<orig_3vol;xop++){
	int me = omp_get_thread_num();
	mult_slow(orig_tmp[me], V_v, mf_v, W_v, xop, top, false, true);
	basic_sum[me] += orig_tmp[me];
      }
#endif

#ifdef GRID_VMV
      //GRID VMV
      if(!UniqueID()){ printf("Grid vMv\n"); fflush(stdout); }
#pragma omp parallel for
      for(int xop=0;xop<grid_3vol;xop++){
	int me = omp_get_thread_num();
	mult(grid_tmp[me], Vgrid_v, mf_grid_v, Wgrid_v, xop, top, false, true);
	grid_sum[me] += grid_tmp[me];
      }
#endif

#ifdef BASIC_GRID_VMV
      if(!UniqueID()){ printf("Grid basic vMv\n"); fflush(stdout); }
      //BASIC GRID VMV FOR TESTING
#pragma omp parallel for
      for(int xop=0;xop<grid_3vol;xop++){
	int me = omp_get_thread_num();
	mult_slow(grid_tmp[me], Vgrid_v, mf_grid_v, Wgrid_v, xop, top, false, true);
	basic_grid_sum[me] += grid_tmp[me];
      }
#endif

#ifdef GRID_SPLIT_LITE_VMV
      //SPLIT LITE VMV GRID
      if(!UniqueID()){ printf("Grid splite vMv\n"); fflush(stdout); }

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
  }
  
  for(int i=1;i<nthreads;i++){
    basic_sum[0] += basic_sum[i];
    orig_sum[0] += orig_sum[i];
    basic_grid_sum[0] += basic_grid_sum[i];
    grid_sum[0] += grid_sum[i];
    grid_sum_split_lite[0] += grid_sum_split_lite[i];  
  }
  
#ifdef FIELD
  //Offload version computes all x,t, so we just have to sum over 4 volume afterwards
  if(!UniqueID()){ printf("Field vMv\n"); fflush(stdout); }

  typedef typename getPropagatorFieldType<GridA2Apolicies>::type PropagatorField;
  typedef mult_vMv_field<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw, PropagatorField> vMvFieldImpl;
  PropagatorField pfield(simd_dims);

  //mult(pfield, Vgrid, mf_grid, Wgrid, false, true);
  vMvFieldImpl::optimized(pfield, Vgrid, mf_grid, Wgrid, false, true, 0, GJP.Tnodes()*GJP.TnodeSites()-1);

  CPSspinColorFlavorMatrix<grid_Complex> vmv_offload_sum4;
  {
    vmv_offload_sum4.zero();
    CPSautoView(pfield_v,pfield,HostRead);
    for(size_t i=0;i<pfield.size();i++){
      vmv_offload_sum4 += *pfield_v.fsite_ptr(i);
    }
  }

#ifdef GRID_FIELD_SIMPLE
  //Same for simple field version
  vMvFieldImpl::simple(pfield, Vgrid, mf_grid, Wgrid, false, true);

  CPSspinColorFlavorMatrix<grid_Complex> vmv_offload_simple_sum4;
  {
    vmv_offload_simple_sum4.zero();
    CPSautoView(pfield_v,pfield,HostRead);
    for(size_t i=0;i<pfield.size();i++){
      vmv_offload_simple_sum4 += *pfield_v.fsite_ptr(i);
    }
  }
#endif

  
#endif
  
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

#ifdef FIELD
#ifdef GRID_FIELD_SIMPLE
  if(!compare(orig_sum[0],vmv_offload_simple_sum4,tol)) ERR.General("","","Standard vs Grid field offload simple implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Grid field offload simple implementation test pass\n");
#endif
  
  if(!compare(orig_sum[0],vmv_offload_sum4,tol)) ERR.General("","","Standard vs Grid field offload optimized implementation test failed\n");
  else if(!UniqueID()) printf("Standard vs Grid field offload optimized implementation test pass\n");
#endif
  std::cout << "testvMvGridOrigGparityTblock passed" << std::endl;
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


template<typename CPSfieldType>
bool compare(const CPSfieldType &a, const CPSfieldType &b, double tolerance, bool verbose = false){
  if(a.size() != b.size()) return false;
  size_t fsize = a.size();
  typedef typename CPSfieldType::FieldSiteType T;
  CPSautoView(a_v,a,HostRead);
  CPSautoView(b_v,b,HostRead);
  
  T const* ap = a_v.ptr();
  T const* bp = b_v.ptr();
  constexpr int SiteSize = CPSfieldType::FieldSiteSize;
  typedef typename CPSfieldType::FieldMappingPolicy MappingPolicy;
  
  for(size_t i=0;i<fsize;i++){
    if(verbose && !UniqueID()){
      size_t rem = i;
      size_t s = rem % SiteSize; rem /= SiteSize;
      size_t x = rem % a.nsites(); rem /= a.nsites();
      int flav = rem;
      int coor[MappingPolicy::EuclideanDimension]; a.siteUnmap(x,coor);
      std::ostringstream os; for(int a=0;a<MappingPolicy::EuclideanDimension;a++) os << coor[a] << " ";
      std::string coor_str = os.str();
      
      printf("Off %d  [s=%d coor=(%s) f=%d] this[%g,%g] vs that[%g,%g] : diff [%g,%g]\n",i, s,coor_str.c_str(),flav,
	     ap[i].real(),ap[i].imag(),bp[i].real(),bp[i].imag(),fabs(ap[i].real()-bp[i].real()), fabs(ap[i].imag()-bp[i].imag()) );
    }
    if( fabs(ap[i].real() - bp[i].real()) > tolerance || fabs(ap[i].imag() - bp[i].imag()) > tolerance ){
      printf("ERROR\n"); fflush(stdout); return false;
    }
  }
  return true;
}


  

template<typename GridA2Apolicies>
void testvMvFieldTimesliceRange(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting testvMvFieldTimesliceRange : vMv tests\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  typename FourDSIMDPolicy<typename GridA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<typename GridA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorW<GridA2Apolicies> W(a2a_args, simd_dims);
  A2AvectorV<GridA2Apolicies> V(a2a_args, simd_dims);
  W.testRandom();
  V.testRandom();

  std::cout << "W fields alloc policy " << printType<typename A2AvectorW<GridA2Apolicies>::FermionFieldType::FieldAllocPolicy>() << std::endl;
  std::cout << "W fields alloc policy " << printType<typename A2AvectorV<GridA2Apolicies>::FermionFieldType::FieldAllocPolicy>() << std::endl;
      
  A2Aparams par(a2a_args);
  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf;
  mf.setup(par,par,0,0);
  mf.testRandom();
  typedef typename GridA2Apolicies::ComplexType vComplex;

  typedef typename getPropagatorFieldType<GridA2Apolicies>::type PropagatorField;
  typedef mult_vMv_field<GridA2Apolicies, A2AvectorV, A2AvectorWfftw, A2AvectorVfftw, A2AvectorW, PropagatorField> vMvFieldImpl;
  PropagatorField pfield_expect(simd_dims), pfield_got(simd_dims);
  std::cout << "Allocated pfield_expect " << &pfield_expect << " pfield_got " << &pfield_got << std::endl;
  
  //vMvFieldImpl::optimized(pfield_expect, V, mf, W, false, true);

  //Compute the full result using the CPU implementation
  {
    CPSautoView(pfield_expect_v,pfield_expect,HostWrite);
    CPSautoView(V_v,V,HostRead);
    CPSautoView(W_v,W,HostRead);
    CPSautoView(mf_v,mf,HostRead);
    
    for(int top=0;top<GJP.TnodeSites();top++){
#pragma omp parallel for
      for(size_t xop=0;xop<pfield_expect.nsites()/GJP.TnodeSites();xop++){
	size_t off = pfield_expect.threeToFour(xop,top);
	mult(*pfield_expect_v.site_ptr(off), V_v, mf_v, W_v, xop, top, false, true);
      }
    }
  } 

  //Check we get the same field if we span the entire lattice
  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  vMvFieldImpl::optimized(pfield_got, V, mf, W, false, true, 0, Lt-1);

  constexpr int N = PropagatorField::FieldSiteType::nScalarType();
  typedef CPSfield<cps::ComplexD, N, FourDpolicy<OneFlavorPolicy> > ScalarComplexField;
  typedef CPSfield<vComplex, N, FourDSIMDPolicy<OneFlavorPolicy>, typename PropagatorField::FieldAllocPolicy > VectorComplexField;   //NOTE: The cast from PropagatorField to this is only legitimate if we have the same allocpolicy!
  NullObject null;
  ScalarComplexField tmp1(null), tmp2(null), tmp3(null);

  std::cout << "Allocated 3 temp fields " << &tmp1 << " " << &tmp2 << " " << &tmp3 << std::endl;
  
  //Check propagator field conversion working
  PropagatorField test(simd_dims);
  std::cout << "Allocated test field " << &test << " and alloc policy " << printType<typename PropagatorField::FieldAllocPolicy>() <<  std::endl;
  test.testRandom();
  std::cout << "Testrandom complete on test field " << &test << std::endl;
  tmp1.zero();
  std::cout << "zero complete on field " << &tmp1 << std::endl;

  std::cout << &tmp2 << " with allocpolicy " << printType<typename ScalarComplexField::FieldAllocPolicy>() << " importing " << &test << " with allocpolicy " << printType<typename PropagatorField::FieldAllocPolicy>() << std::endl;
  tmp2.importField( (VectorComplexField const &)test );
  assert( !tmp1.equals(tmp2, 1e-12, true) );

  tmp1.importField( (VectorComplexField const &)test );
  assert( tmp1.equals(tmp2, 1e-12, true) );
  std::cout << "Passed propagatorField conversion check (the first reported error above was intentional!)" << std::endl;
  
  tmp1.importField( (VectorComplexField const &)pfield_got );
  tmp2.importField( (VectorComplexField const &)pfield_expect );
  assert( tmp1.equals(tmp2, tol, false) );
  std::cout << "Full Lt test passed" << std::endl;

  int tdis = Lt/2 - 1;
  vMvFieldImpl::optimized(pfield_got, V, mf, W, false, true, 0, tdis);
  tmp1.importField( (VectorComplexField const &)pfield_got );

  //Zero timeslices in comparison data
  tmp3 = tmp2;
  {
    CPSautoView(tmp3_v,tmp3,HostWrite);
    for(size_t xx = 0; xx < tmp3.nfsites(); xx++){
      int f; int x[4];
      tmp3.fsiteUnmap(xx,x,f);
      int t = x[3] + GJP.TnodeCoor()*GJP.TnodeSites();
      if(t>tdis){
	cps::ComplexD *p = tmp3_v.fsite_ptr(xx);
	for(int i=0;i<N;i++)
	  p[i] = 0;
      }
    }
  }
  assert( compare(tmp1,tmp3,tol,false) ); 
  std::cout << "Partial Lt test passed 0 <= t <= Lt/2-1" << std::endl;


  //Do the same but start from second half of lattice to ensure periodic logic is correct
  tdis = 2;
  vMvFieldImpl::optimized(pfield_got, V, mf, W, false, true, Lt-1, tdis);
  tmp1.importField( (VectorComplexField const &)pfield_got );
 
  //Zero timeslices in comparison data
  tmp3 = tmp2;
  {
    CPSautoView(tmp3_v,tmp3,HostWrite);
    
    for(size_t xx = 0; xx < tmp3.nfsites(); xx++){
      int f; int x[4];
      tmp3.fsiteUnmap(xx,x,f);
      int t = x[3] + GJP.TnodeCoor()*GJP.TnodeSites();
      
      if(!(t == Lt-1 || t == 0 || t==1)){    
	cps::ComplexD *p = tmp3_v.fsite_ptr(xx);
	for(int i=0;i<N;i++)
	  p[i] = 0;
      }
    }
  }
  
  assert( compare(tmp1,tmp3,tol,false) ); 
  std::cout << "Partial Lt test passed Lt-1 <= t <= 1" << std::endl;

  
  std::cout << "testvMvFieldTimesliceRange passed" << std::endl;
}


class A2AparamsOverride: public A2Aparams{
 public:
  A2AparamsOverride(): A2Aparams(){}
  A2AparamsOverride(const A2AArg &_args): A2Aparams(_args){}

  //Set the number to tblocks to 'to', overriding setting from A2Aargs input. This is intended for benchmarking estimates for large jobs using single nodes
  //and may have unexpected consequences!
  void setNtBlocks(const int to){
    ntblocks = to;
    ndilute =  ntblocks * nspincolor* nflavors;      
    nhits = args.nhits;
    nh = nhits * ndilute;
    nv = nl + nh;
  }
  void setLt(const int to){
    Lt = to;
  }
};



//Test that we can correctly evaluate the vMv operation for a number of temporal blocks unrelated
//to the size of the job lattice
template<typename GridA2Apolicies>
void testvMvFieldArbitraryNtblock(const A2AArg &a2a_args, const DoArg &do_arg, const double tol){
  std::cout << "Starting testvMvFieldArbitraryNtblock\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  typename FourDSIMDPolicy<typename GridA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<typename GridA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AparamsOverride params(a2a_args);
  
  int nl = a2a_args.nl;
  int orig_Lt = GJP.Tnodes()*GJP.TnodeSites();
  int new_Lt = 2*orig_Lt;
  int ntblocks = new_Lt;
  params.setLt(new_Lt);
  params.setNtBlocks(ntblocks);

  std::cout << "Orig Lt=" << orig_Lt << " new Lt=" << new_Lt << " params tblocks " << params.getNtBlocks() << std::endl;

  A2AvectorW<GridA2Apolicies> W(params, simd_dims);
  A2AvectorV<GridA2Apolicies> V(params, simd_dims);

  int nf = GJP.Gparity()+1;
  std::cout << "V high modes: " << V.getNhighModes() << " expect " << 12*nf*ntblocks << std::endl;

  assert(V.getNhighModes() == 12*nf*ntblocks);

  //We want the W, V to have a predictable structure that can be replicated on the doubled lattice
  int node_sites[4];
  int node_off[4];
  int glb_size[4];
  for(int i=0;i<4;i++){
    node_sites[i] = GJP.NodeSites(i);
    node_off[i] = GJP.NodeSites(i)*GJP.NodeCoor(i);
    glb_size[i] = GJP.NodeSites(i)*GJP.Nodes(i);
  }
  glb_size[3] = new_Lt;

  for(int t=0;t<node_sites[3];t++){
    int tg = t + node_off[3];
    for(int z=0;z<node_sites[2];z++){
      int zg = z + node_off[2];

      for(int y=0;y<node_sites[1];y++){
	int yg = y + node_off[1];

	for(int x=0;x<node_sites[0];x++){
	  int xg = x + node_off[0];

	  int c[4] = {x,y,z,t};

	  for(int f=0;f<nf;f++){

	    for(int i=0;i<nl;i++){
	      auto &wl = W.getWl(i); //fermion type
	      auto &vl = V.getVl(i);

	      CPSautoView(wl_v,wl,HostWrite);
	      CPSautoView(vl_v,vl,HostWrite);
	      
	      auto* wsite = wl_v.site_ptr(c,f);
	      auto* vsite = vl_v.site_ptr(c,f);

	      size_t val_base = 12*f + 24*(xg + glb_size[0]*(yg + glb_size[1]*(zg + glb_size[2]*(tg + glb_size[3]*i))));
	      for(int sc=0;sc<12;sc++){
		*(wsite++) = sc + val_base;
		*(vsite++) = sc + val_base;
	      }
	    }

	    for(int i=0;i<V.getNhighModes();i++){
	      auto &vh = V.getVh(i);
	      CPSautoView(vh_v,vh,HostWrite);
		      
	      auto* vsite = vh_v.site_ptr(c,f);

	      size_t val_base = 12*f + 24*(xg + glb_size[0]*(yg + glb_size[1]*(zg + glb_size[2]*(tg + glb_size[3]*i))));
	      for(int sc=0;sc<12;sc++){
		*(vsite++) = sc + val_base;
	      }
	    }		

	    for(int i=0;i<W.getNhighModes();i++){
	      auto &wh = W.getWh(i);
	      CPSautoView(wh_v,wh,HostWrite);		      
	      auto* wsite = wh_v.site_ptr(c,f);

	      size_t val_base = f + 2*(xg + glb_size[0]*(yg + glb_size[1]*(zg + glb_size[2]*(tg + glb_size[3]*i))));
	      *wsite = val_base;
	    }		

	  }
	}
      }
    }
  }
  
  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf;
  mf.setup(params,params,0,0);
  mf.testRandom();
  typedef typename GridA2Apolicies::ComplexType vComplex;
  
  std::cout << "Meson field rows = " << mf.getNrows() << " expect " << nl + 12*nf << std::endl;
  std::cout << "Meson field cols = " << mf.getNcols() << " expect " << nl + 12*nf*ntblocks << std::endl;
  assert(mf.getNcols() == nl + 12*nf*ntblocks);

  typedef typename getPropagatorFieldType<GridA2Apolicies>::type PropagatorField;
  typedef mult_vMv_field<GridA2Apolicies, A2AvectorV, A2AvectorWfftw, A2AvectorVfftw, A2AvectorW, PropagatorField> vMvFieldImpl;
  PropagatorField pfield_got(simd_dims);

  mult(pfield_got, V, mf, W, false, true);
  
  cps::ComplexD tline_got[new_Lt];
  memset(tline_got, 0, new_Lt*sizeof(cps::ComplexD));

  {
    CPSautoView(pfield_got_v,pfield_got,HostRead);
    for(int t=0;t<node_sites[3];t++){
      int tg = t + node_off[3];  
      for(int z=0;z<node_sites[2];z++){
	for(int y=0;y<node_sites[1];y++){
	  for(int x=0;x<node_sites[0];x++){
	    int c[4] = {x,y,z,t};
	    auto* sp = pfield_got_v.site_ptr(c);

	    for(int s1=0;s1<4;s1++){
	      for(int c1=0;c1<3;c1++){
		for(int f1=0;f1<nf;f1++){
		  for(int s2=0;s2<4;s2++){
		    for(int c2=0;c2<3;c2++){
		      for(int f2=0;f2<nf;f2++){
			tline_got[tg] += convertComplexD(Reduce((*sp)(s1,s2)(c1,c2)(f1,f2)));
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  globalSum(tline_got, new_Lt);
  

  DoArg do_arg_dbl(do_arg);
  do_arg_dbl.t_sites *= 2;

  GJP.Initialize(do_arg_dbl);

  assert(GJP.Tnodes()*GJP.TnodeSites() == new_Lt);  

  A2AvectorW<GridA2Apolicies> W2(a2a_args, simd_dims);
  A2AvectorV<GridA2Apolicies> V2(a2a_args, simd_dims);

  for(int i=0;i<4;i++){
    node_sites[i] = GJP.NodeSites(i);
    node_off[i] = GJP.NodeSites(i)*GJP.NodeCoor(i);
    glb_size[i] = GJP.NodeSites(i)*GJP.Nodes(i);
  }

  for(int t=0;t<node_sites[3];t++){
    int tg = t + node_off[3];
    for(int z=0;z<node_sites[2];z++){
      int zg = z + node_off[2];

      for(int y=0;y<node_sites[1];y++){
	int yg = y + node_off[1];

	for(int x=0;x<node_sites[0];x++){
	  int xg = x + node_off[0];

	  int c[4] = {x,y,z,t};

	  for(int f=0;f<nf;f++){

	    for(int i=0;i<nl;i++){
	      auto &wl = W2.getWl(i); //fermion type
	      auto &vl = V2.getVl(i);

	      CPSautoView(wl_v,wl,HostWrite);
	      CPSautoView(vl_v,vl,HostWrite);
	      
	      auto* wsite = wl_v.site_ptr(c,f);
	      auto* vsite = vl_v.site_ptr(c,f);

	      size_t val_base = 12*f + 24*(xg + glb_size[0]*(yg + glb_size[1]*(zg + glb_size[2]*(tg + glb_size[3]*i))));
	      for(int sc=0;sc<12;sc++){
		*(wsite++) = sc + val_base;
		*(vsite++) = sc + val_base;
	      }
	    }

	    for(int i=0;i<V2.getNhighModes();i++){
	      auto &vh = V2.getVh(i);
	      CPSautoView(vh_v,vh,HostWrite);	      
	      auto* vsite = vh_v.site_ptr(c,f);

	      size_t val_base = 12*f + 24*(xg + glb_size[0]*(yg + glb_size[1]*(zg + glb_size[2]*(tg + glb_size[3]*i))));
	      for(int sc=0;sc<12;sc++){
		*(vsite++) = sc + val_base;
	      }
	    }		

	    for(int i=0;i<W2.getNhighModes();i++){
	      auto &wh = W2.getWh(i);
	      CPSautoView(wh_v,wh,HostWrite);	      		      
	      auto* wsite = wh_v.site_ptr(c,f);

	      size_t val_base = f + 2*(xg + glb_size[0]*(yg + glb_size[1]*(zg + glb_size[2]*(tg + glb_size[3]*i))));
	      *wsite = val_base;
	      //*wsite = 2*val_base;
	    }		

	  }
	}
      }
    }
  }

  PropagatorField pfield_expect(simd_dims);

  mult(pfield_expect, V2, mf, W2, false, true); //use same meson field


  cps::ComplexD tline_expect[new_Lt];

  memset(tline_expect, 0, new_Lt*sizeof(cps::ComplexD));

  CPSautoView(pfield_expect_v, pfield_expect, HostRead);
  for(int t=0;t<node_sites[3];t++){
    int tg = t + node_off[3];  
    for(int z=0;z<node_sites[2];z++){
      for(int y=0;y<node_sites[1];y++){
	for(int x=0;x<node_sites[0];x++){
	  int c[4] = {x,y,z,t};
	  auto* sp = pfield_expect_v.site_ptr(c);

	  for(int s1=0;s1<4;s1++){
	    for(int c1=0;c1<3;c1++){
	      for(int f1=0;f1<nf;f1++){
		for(int s2=0;s2<4;s2++){
		  for(int c2=0;c2<3;c2++){
		    for(int f2=0;f2<nf;f2++){
		      tline_expect[tg] += convertComplexD(Reduce((*sp)(s1,s2)(c1,c2)(f1,f2)));
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  globalSum(tline_expect, new_Lt);

  std::cout << "Got   Expect   Diff" << std::endl;
  for(int i=0;i<new_Lt;i++){
    cps::ComplexD diff = tline_got[i]-tline_expect[i];
    std::cout << i << " " << tline_got[i] << " " << tline_expect[i] << " " << diff << " " << (i>=orig_Lt ? "(expect diff)" : "(expect same)") <<  std::endl;
    if(i < orig_Lt && ( fabs(diff.real()) > tol || fabs(diff.imag()) > tol) ) ERR.General("","","Comparison failed");      
  }
  GJP.Initialize(do_arg);

  std::cout << "testvMvFieldArbitraryNtblock passed" << std::endl;
}





CPS_END_NAMESPACE
