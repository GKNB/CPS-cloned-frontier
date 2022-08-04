#pragma once

CPS_START_NAMESPACE

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

  std::cout << "Running CPU threaded versions" << std::endl;
  
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

  std::cout << "Running GPU version" << std::endl;
  
  //Offload version computes all x,t, so we just have to sum over 4 volume afterwards
  typedef typename mult_vv_field<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw>::PropagatorField PropagatorField;
  PropagatorField pfield(simd_dims);
  
  mult(pfield, Vgrid, Wgrid, false, true);

  CPSspinColorFlavorMatrix<grid_Complex> vmv_offload_sum4;
  vmv_offload_sum4.zero();
  for(size_t i=0;i<pfield.size();i++){
    vmv_offload_sum4 += *pfield.fsite_ptr(i);
  }

  std::cout << "Comparing results" << std::endl;
  
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
void testVVgridOrigGparityTblock(A2AArg a2a_args, const int nthreads, const double tol){
  std::cout << "Starting testVVgridOrigGparityTblock: vv tests\n";
  a2a_args.src_width = 2;

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


CPS_END_NAMESPACE
