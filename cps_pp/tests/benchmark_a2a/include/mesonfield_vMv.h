#pragma once

CPS_START_NAMESPACE

#ifdef USE_GRID

template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void benchmarkvMvGridOrig(const A2AArg &a2a_args, const int ntests, const int nthreads){
#ifdef USE_GRID
#define CPS_VMV
  //#define GRID_VMV
#define GRID_SPLIT_LITE_VMV;

  std::cout << "Starting vMv benchmark\n";
  std::cout << "nl=" << a2a_args.nl << "\n";
  

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
  {
    CPSautoView(mf_grid_v,mf_grid,HostWrite);
    CPSautoView(mf_v,mf,HostRead);
    
    for(int i=0;i<mf.getNrows();i++)
      for(int j=0;j<mf.getNcols();j++)
	mf_grid_v(i,j) = mf_v(i,j); //both are scalar complex
  }
  
  typedef typename GridA2Apolicies::ComplexType grid_Complex;
  typedef typename ScalarA2Apolicies::ComplexType mf_Complex;

  int nf = GJP.Gparity()+1;

  //Compute Flops
  size_t Flops = 0;

  {
    typedef typename A2AvectorVfftw<ScalarA2Apolicies>::DilutionType iLeftDilutionType;
    typedef typename A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::LeftDilutionType iRightDilutionType;

    typedef typename A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::RightDilutionType jLeftDilutionType;    
    typedef typename A2AvectorWfftw<ScalarA2Apolicies>::DilutionType jRightDilutionType;

    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(V);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(W);
    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    size_t vol3d = GJP.XnodeSites()*GJP.YnodeSites()*GJP.ZnodeSites();

    //Count Flops on node
    for(int t=GJP.TnodeCoor()*GJP.TnodeSites();t<(GJP.TnodeCoor()+1)*GJP.TnodeSites();t++){
      //Count elements of Mr actually used
      std::vector<bool> ir_used(mf.getNrows(),false);
      for(int scl=0;scl<12;scl++){
	for(int fl=0;fl<nf;fl++){
	  modeIndexSet ilp, irp;
	  ilp.time = t;
	  irp.time = mf.getRowTimeslice();
	      	      
	  ilp.spin_color = scl;
	  ilp.flavor = fl;

	  int ni = i_ind.getNindices(ilp,irp);
	  for(int i=0;i<ni;i++){
	    int il = i_ind.getLeftIndex(i, ilp,irp);
	    ir_used[il] = true;
	  }
	}
      }
      int nir_used = 0;
      for(int i=0;i<ir_used.size();i++)
	if(ir_used[i]) nir_used++;
      
      //Mr[scr][fr]
      for(int scr=0;scr<12;scr++){
	for(int fr=0;fr<nf;fr++){

	  modeIndexSet jlp, jrp;
	      
	  jlp.time = mf.getColTimeslice();
	  jrp.time = t;
	  
	  jrp.spin_color = scr;
	  jrp.flavor = fr;
	  
	  int nj = j_ind.getNindices(jlp,jrp);
	      	      
	  //Mr =  nir_used * nj * (cmul + cadd)  per site
	  Flops += nir_used * nj * 8 * vol3d;
	}
      }

      //l[scl][fl](Mr[scr][fr])
      for(int scl=0;scl<12;scl++){
	for(int fl=0;fl<nf;fl++){
	  for(int scr=0;scr<12;scr++){
	    for(int fr=0;fr<nf;fr++){

	      modeIndexSet ilp, irp, jlp, jrp;
	      ilp.time = t;
	      irp.time = mf.getRowTimeslice();
	            
	      ilp.spin_color = scl;
	      ilp.flavor = fl;

	      int ni = i_ind.getNindices(ilp,irp);
	      	      
	      //l( Mr) = ni  * (cmul + cadd)  per site
	      Flops += ni * 8 * vol3d;
	    }
	  }
	}
      } 
    }//t

  }//Flops count
  double MFlops = double(Flops)/1e6;
  
      
  Float total_time = 0.;
  Float total_time_orig = 0.;
  Float total_time_split_lite_grid = 0.;
  Float total_time_field_offload = 0.;
  mult_vMv_field_offload_timers::get().reset();

  typedef typename AlignedVector<CPSspinColorFlavorMatrix<mf_Complex> >::type BasicVector;
  typedef typename AlignedVector<CPSspinColorFlavorMatrix<grid_Complex> >::type SIMDvector;

  BasicVector orig_sum(nthreads);
  SIMDvector grid_sum(nthreads);

  BasicVector orig_tmp(nthreads);
  SIMDvector grid_tmp(nthreads);

  SIMDvector grid_sum_split_lite(nthreads);      

  int orig_3vol = GJP.VolNodeSites()/GJP.TnodeSites();
  int grid_3vol = Vgrid.getMode(0).nodeSites(0) * Vgrid.getMode(0).nodeSites(1) *Vgrid.getMode(0).nodeSites(2);

  mult_vMv_split_lite<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw> vmv_split_lite_grid;

  for(int iter=0;iter<ntests;iter++){
    for(int i=0;i<nthreads;i++){
      orig_sum[i].zero(); grid_sum[i].zero();
      grid_sum_split_lite[i].zero();
    }

    {
      CPSautoView(V_v,V,HostRead);
      CPSautoView(W_v,W,HostRead);    
      CPSautoView(mf_v,mf,HostRead);
      CPSautoView(Vgrid_v,Vgrid,HostRead);
      CPSautoView(Wgrid_v,Wgrid,HostRead);    
      CPSautoView(mf_grid_v,mf_grid,HostRead);

      for(int top = 0; top < GJP.TnodeSites(); top++){
#ifdef CPS_VMV
	//ORIG VMV
	total_time_orig -= dclock();	  
#pragma omp parallel for
	for(int xop=0;xop<orig_3vol;xop++){
	  int me = omp_get_thread_num();
	  mult(orig_tmp[me], V_v, mf_v, W_v, xop, top, false, true);
	  orig_sum[me] += orig_tmp[me];
	}
	total_time_orig += dclock();
#endif
#ifdef GRID_VMV
	//GRID VMV
	total_time -= dclock();
#pragma omp parallel for
	for(int xop=0;xop<grid_3vol;xop++){
	  int me = omp_get_thread_num();
	  mult(grid_tmp[me], Vgrid_v, mf_grid_v, Wgrid_v, xop, top, false, true);
	  grid_sum[me] += grid_tmp[me];
	}
	total_time += dclock();
#endif
      }//end top loop
      for(int i=1;i<nthreads;i++){
	orig_sum[0] += orig_sum[i];
	grid_sum[0] += grid_sum[i];
	grid_sum_split_lite[0] += grid_sum_split_lite[i];  
      }
    }

    for(int top = 0; top < GJP.TnodeSites(); top++){
#ifdef GRID_SPLIT_LITE_VMV
      //SPLIT LITE VMV GRID
      total_time_split_lite_grid -= dclock();	  
      vmv_split_lite_grid.setup(Vgrid, mf_grid, Wgrid, top);

#pragma omp parallel for
      for(int xop=0;xop<grid_3vol;xop++){
	int me = omp_get_thread_num();
	vmv_split_lite_grid.contract(grid_tmp[me], xop, false, true);
	grid_sum_split_lite[me] += grid_tmp[me];
      }
      total_time_split_lite_grid += dclock();
#endif
    }
    for(int i=1;i<nthreads;i++){
      grid_sum_split_lite[0] += grid_sum_split_lite[i];  
    }


    //Offload version computes all x,t, so we just have to sum over 4 volume afterwards
    total_time_field_offload -= dclock();
    typedef typename getPropagatorFieldType<GridA2Apolicies>::type PropagatorField;
    PropagatorField pfield(simd_dims);
    
    mult(pfield, Vgrid, mf_grid, Wgrid, false, true);
    total_time_field_offload += dclock();

    CPSspinColorFlavorMatrix<grid_Complex> vmv_offload_sum4;
    vmv_offload_sum4.zero();
    {
      CPSautoView(pfield_v,pfield,HostRead);
      for(size_t i=0;i<pfield.size();i++){
        vmv_offload_sum4 += *pfield_v.fsite_ptr(i);
      }
    }
    
  } //tests loop
#ifdef CPS_VMV
  printf("vMv: Avg time vMv (non-SIMD) %d iters: %g secs/iter  %g Mflops\n",ntests,total_time_orig/ntests,  MFlops/(total_time_orig/ntests) );
#endif
#ifdef GRID_VMV
  printf("vMv: Avg time vMv (SIMD) code %d iters: %g secs/iter  %g Mflops\n",ntests,total_time/ntests, MFlops/(total_time/ntests) );
#endif
#ifdef GRID_SPLIT_LITE_VMV
  printf("vMv: Avg time split vMv lite (SIMD) %d iters: %g secs/iter  %g Mflops\n",ntests,total_time_split_lite_grid/ntests, MFlops/(total_time_split_lite_grid/ntests) );
#endif
  printf("vMv: Avg time vMv field offload %d iters: %g secs/iter  %g Mflops\n",ntests,total_time_field_offload/ntests, MFlops/(total_time_field_offload/ntests) );

  if(!UniqueID()){
    printf("vMv offload timings:\n");
    mult_vMv_field_offload_timers::get().print();
  }

#endif
}




template<typename GridA2Apolicies, template<typename> class A2AfieldL, template<typename> class A2AfieldR>
void benchmarkvMvGridOffload(const A2AArg &a2a_args, const int ntests, const int nthreads){
  std::cout << "Starting vMv offload benchmark with policies " << printType<GridA2Apolicies>() << std::endl;
  //DeviceMemoryPoolManager::globalPool().setVerbose(true);
  
  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      
  A2Aparams a2a_params(a2a_args);
  
  typename FourDSIMDPolicy<typename GridA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<typename GridA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);

  std::cout << "Constructing A2A vectors" << std::endl;
  A2AfieldL<GridA2Apolicies> fieldL(a2a_args, simd_dims);
  A2AfieldR<GridA2Apolicies> fieldR(a2a_args, simd_dims);

  fieldL.zero();
  fieldR.zero();

  std::cout << "Constructing meson field" << std::endl;
  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_grid;
  mf_grid.setup(a2a_params,a2a_params,0,0);     
  mf_grid.testRandom();

  std::cout << "Constructing propagator field" << std::endl;
  typedef typename getPropagatorFieldType<GridA2Apolicies>::type PropagatorField;
  PropagatorField pfield(simd_dims);

  std::cout << "Running benchmark" << std::endl;
  
  Float total_time_field_offload = 0.;
  mult_vMv_field_offload_timers::get().reset();

  for(int i=0;i<ntests;i++){
    if(!UniqueID()){ printf("."); fflush(stdout); }
    total_time_field_offload -= dclock();    
    mult(pfield, fieldL, mf_grid, fieldR, false, true);
    total_time_field_offload += dclock();
  }
  if(!UniqueID()){ printf("\n"); fflush(stdout); }

  int nf = GJP.Gparity() + 1;

  //Count flops (over all nodes)
  //\sum_i\sum_j v(il)_{scl,fl}(x)  M(ir, jl) * v(jr)_{scr,fr}(x)  for all t, x3d

  //Simple method
  //for(int x4d=0;x4d<vol4d;x4d++)
  // for(int scl=0;scl<12;scl++)
  //  for(int fl=0;fl<nf;fl++)
  //   for(int scr=0;scl<12;scl++)
  //    for(int fr=0;fr<nf;fr++)
  //     for(int i=0;i<ni;i++)
  //      for(int j=0;j<nj;j++)
  //        out(fl,scl; fr, scr)(x) += v(il[i])_{scl,fl}(x) * M(ir[i], jl[j]) * v(jr[j])_{scr,fr}(x)     
  //
  //vol4d * 12*nf * 12*nf * ni * nj * 14 flops
  


  //Split method
  //for(int x4d=0;x4d<vol4d;x4d++)
  // for(int scr=0;scr<12;scr++)
  //  for(int fr=0;fr<nf;fr++)
  //    for(int i=0;i<ni;i++)
  //     for(int j=0;j<nj;j++)
  //        Mr(ir[i])_{scr,fr}(x)   +=   M(ir[i], jl[j]) * v(jr[j])_{scr,fr}(x) 
  //
  //vol4d * 12 * nf *  ni * nj * 8 flops

  //+

  //for(int x4d=0;x4d<vol4d;x4d++)
  // for(int scl=0;scl<12;scl++)
  //  for(int fl=0;fl<nf;fl++)
  //   for(int scr=0;scl<12;scl++)
  //    for(int fr=0;fr<nf;fr++)
  //     for(int i=0;i<ni;i++)
  //        out(fl,scl; fr, scr)(x) += v(il[i])_{scl,fl}(x) * Mr(ir[i])_{scr,fr}(x)    
  //vol4d * 12 * nf * 12 * nf *  ni * 8 flops   


  //vol4d * 12 * nf * ni * ( nj * 8 + 12*nf*ni *8)

  typedef mult_vMv_field<GridA2Apolicies, A2AfieldL, A2AvectorWfftw, A2AvectorVfftw, A2AfieldR, PropagatorField> offload;
  ModeContractionIndices<typename offload::iLeftDilutionType, typename offload::iRightDilutionType> i_ind(a2a_params);
  ModeContractionIndices<typename offload::jLeftDilutionType, typename offload::jRightDilutionType> j_ind(a2a_params);
  size_t Flops = 0;
  for(int t_glob=0;t_glob<GJP.TnodeSites()*GJP.Tnodes();t_glob++){
    modeIndexSet ilp, irp, jlp, jrp;
    ilp.time = jrp.time = t_glob;
    irp.time = mf_grid.getRowTimeslice();
    jlp.time = mf_grid.getColTimeslice();
    
    //ni is actually a function of scl, fl, but we can work out exactly which ir are used for any of the scl,fl
    std::set<int> ir_used;
    for(int fl=0;fl<nf;fl++){
      ilp.flavor = irp.flavor = fl;
      for(int scl=0;scl<12;scl++){
	ilp.spin_color = irp.spin_color = scl;
	auto const &ivec = i_ind.getIndexVector(ilp,irp);
	for(int i=0;i<ivec.size();i++)
	  ir_used.insert(ivec[i].second);
      }
    }
    for(int fr=0;fr<nf;fr++){
      jlp.flavor = jrp.flavor = fr;
      for(int scr=0;scr<12;scr++){
	jlp.spin_color = jrp.spin_color = scr;
	size_t nj = j_ind.getIndexVector(jlp,jrp).size();
	
	Flops += ir_used.size() * nj * 8;
      }
    }

    for(int fr=0;fr<nf;fr++){
      for(int scr=0;scr<12;scr++){
	
	for(int fl=0;fl<nf;fl++){
	  ilp.flavor = irp.flavor = fl;
	  for(int scl=0;scl<12;scl++){
	    ilp.spin_color = irp.spin_color = scl;
	    size_t ni = i_ind.getIndexVector(ilp,irp).size();
	    
	    Flops += ni * 8;
	  }
	}
      }
    }
  }
  Flops *= GJP.TotalNodes()*GJP.VolNodeSites()/GJP.TnodeSites(); //the above is done for every 3d site

  double tavg = total_time_field_offload/ntests;
  double Mflops = double(Flops)/tavg/1e6;

  if(!UniqueID()){
    printf("vMv: Avg time offload %d iters: %g secs  perf %f Mflops\n",ntests,tavg,Mflops);
    printf("vMv offload timings:\n");
    mult_vMv_field_offload_timers::get().print();
  }
  //DeviceMemoryPoolManager::globalPool().setVerbose(false);    
}



template<typename GridA2Apolicies>
void benchmarkvMvPartialTimeGridOffload(const A2AArg &a2a_args, const int ntests, const int tstart, const int tend, bool compare_full){
  std::cout << "Starting vMv partial-Lt offload benchmark\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  typename FourDSIMDPolicy<typename GridA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<typename GridA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);

  Wgrid.zero();
  Vgrid.zero();
  
  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_grid;
  mf_grid.setup(Wgrid,Vgrid,0,0);     
  mf_grid.testRandom();
  
  typedef typename getPropagatorFieldType<GridA2Apolicies>::type PropagatorField;
  PropagatorField pfield(simd_dims);

  Float total_time_orig = 0;
  if(compare_full){
    std::cout << "Running full-Lt implementation" << std::endl;
    for(int i=0;i<ntests;i++){
      if(!UniqueID()){ printf("."); fflush(stdout); }
      total_time_orig -= dclock();    
      mult(pfield, Vgrid, mf_grid, Wgrid, false, true);
      total_time_orig += dclock();
    }
  }

  Float total_time_partial = 0.;
  mult_vMv_field_offload_timers::get().reset();

  std::cout << "Running partial-Lt implementation for t in range " << tstart << " " <<  tend << std::endl;
  for(int i=0;i<ntests;i++){
    if(!UniqueID()){ printf("."); fflush(stdout); }
    total_time_partial -= dclock();    
    mult(pfield, Vgrid, mf_grid, Wgrid, false, true, tstart, tend);
    total_time_partial += dclock();
  }
  if(!UniqueID()){ printf("\n"); fflush(stdout); }

  int nf = GJP.Gparity() + 1;

  typedef mult_vMv_field<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw, PropagatorField> offload;
  ModeContractionIndices<typename offload::iLeftDilutionType, typename offload::iRightDilutionType> i_ind(Vgrid);
  ModeContractionIndices<typename offload::jLeftDilutionType, typename offload::jRightDilutionType> j_ind(Vgrid);
  size_t Flops = 0;
  for(int t_glob=0;t_glob<GJP.TnodeSites()*GJP.Tnodes();t_glob++){
    if(t_glob < tstart || t_glob > tend) continue;
    
    modeIndexSet ilp, irp, jlp, jrp;
    ilp.time = jrp.time = t_glob;
    irp.time = mf_grid.getRowTimeslice();
    jlp.time = mf_grid.getColTimeslice();
    
    //ni is actually a function of scl, fl, but we can work out exactly which ir are used for any of the scl,fl
    std::set<int> ir_used;
    for(int fl=0;fl<nf;fl++){
      ilp.flavor = irp.flavor = fl;
      for(int scl=0;scl<12;scl++){
	ilp.spin_color = irp.spin_color = scl;
	auto const &ivec = i_ind.getIndexVector(ilp,irp);
	for(int i=0;i<ivec.size();i++)
	  ir_used.insert(ivec[i].second);
      }
    }
    for(int fr=0;fr<nf;fr++){
      jlp.flavor = jrp.flavor = fr;
      for(int scr=0;scr<12;scr++){
	jlp.spin_color = jrp.spin_color = scr;
	size_t nj = j_ind.getIndexVector(jlp,jrp).size();
	
	Flops += ir_used.size() * nj * 8;
      }
    }

    for(int fr=0;fr<nf;fr++){
      for(int scr=0;scr<12;scr++){
	
	for(int fl=0;fl<nf;fl++){
	  ilp.flavor = irp.flavor = fl;
	  for(int scl=0;scl<12;scl++){
	    ilp.spin_color = irp.spin_color = scl;
	    size_t ni = i_ind.getIndexVector(ilp,irp).size();
	    
	    Flops += ni * 8;
	  }
	}
      }
    }
  }
  Flops *= GJP.TotalNodes()*GJP.VolNodeSites()/GJP.TnodeSites(); //the above is done for every 3d site

  double tavg = total_time_partial/ntests;
  double Mflops = double(Flops)/tavg/1e6;

  if(!UniqueID()){
    printf("vMv: Avg time offload %d iters: %g secs  perf %f Mflops\n",ntests,tavg,Mflops);
    printf("vMv offload timings:\n");
    mult_vMv_field_offload_timers::get().print();

    if(compare_full){
      double tavg_full = total_time_orig / ntests;
      printf("vMv: Avg time full-Lt %d iters: %g secs,  ratio full/partial: %g\n",ntests,tavg_full, tavg_full/tavg);
    }
    
  }
}





#endif //USE_GRID

CPS_END_NAMESPACE
