#pragma once

CPS_START_NAMESPACE

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

CPS_END_NAMESPACE
