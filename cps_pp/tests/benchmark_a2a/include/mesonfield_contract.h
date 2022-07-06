#pragma once

CPS_START_NAMESPACE

template<typename GridA2Apolicies>
void benchmarkMFcontractKernel(const int ntests, const int nthreads){
#ifdef USE_GRID
  // GridVectorizedSpinColorContract benchmark
  typedef typename GridA2Apolicies::ComplexType GVtype;
  typedef typename GridA2Apolicies::ScalarComplexType GCtype;
  const int nsimd = GVtype::Nsimd();      

  FourDSIMDPolicy<OneFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<OneFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
  
  NullObject n;
  CPSfield<GCtype,12,FourDpolicy<OneFlavorPolicy> > a(n); a.testRandom();
  CPSfield<GCtype,12,FourDpolicy<OneFlavorPolicy> > b(n); b.testRandom();
  CPSfield<GVtype,12,FourDSIMDPolicy<OneFlavorPolicy>,Aligned128AllocPolicy> aa(simd_dims); aa.importField(a);
  CPSfield<GVtype,12,FourDSIMDPolicy<OneFlavorPolicy>,Aligned128AllocPolicy> bb(simd_dims); bb.importField(b);
  CPSfield<GVtype,1,FourDSIMDPolicy<OneFlavorPolicy>,Aligned128AllocPolicy> cc(simd_dims);

#ifdef TIMERS_OFF
  printf("Timers are OFF\n"); fflush(stdout);
#else
  printf("Timers are ON\n"); fflush(stdout);
#endif

  double t0;

#ifndef GPU_VEC
  printf("Max threads %d\n",omp_get_max_threads());
  
#pragma omp parallel //avoid thread creation overheads
  {
    int me = omp_get_thread_num();
    size_t work, off;
    thread_work(work, off, aa.nfsites(), me, omp_get_num_threads());
	
    GVtype *abase = aa.fsite_ptr(off);
    GVtype *bbase = bb.fsite_ptr(off);
    GVtype *cbase = cc.fsite_ptr(off);

    for(int test=0;test<ntests+1;test++){
      if(test == 1) t0 = Grid::usecond(); //ignore first iteration
      GVtype *ai = abase;
      GVtype *bi = bbase;
      GVtype *ci = cbase;
      __SSC_MARK(0x1);
      for(size_t i=0;i<work;i++){
	*ci = GridVectorizedSpinColorContract<GVtype,true,false>::g5(ai,bi);
	ai += 12;
	bi += 12;
	ci += 1;
      }
      __SSC_MARK(0x2);
    }
  }


#else
  //Should operate entirely out of GPU memory
  size_t work = aa.nfsites();
  static const int Nsimd = GVtype::Nsimd();

  size_t site_size_ab = aa.siteSize();
  size_t site_size_c = cc.siteSize();
    
  GVtype const* adata = aa.ptr();
  GVtype const* bdata = bb.ptr();
  GVtype * cdata = cc.ptr();

  for(int test=0;test<ntests+1;test++){   
   {
      using namespace Grid;
      if(test == 1) t0 = Grid::usecond(); //ignore first iteration

      if(test == ntests -1) cudaProfilerStart();

      accelerator_for(item, work, Nsimd, 
		      {
			size_t x = item;
			GVtype const* ax = adata + site_size_ab*x;
			GVtype const* bx = bdata + site_size_ab*x;
			GVtype *cx = cdata + site_size_c*x;
			
			typename SIMT<GVtype>::value_type v = GridVectorizedSpinColorContract<GVtype,true,false>::g5(ax,bx);

			SIMT<GVtype>::write(*cx, v);			  
		      });
      if(test == ntests -1) cudaProfilerStop();
    }   
  }    

#endif

  double t1 = Grid::usecond();
  double dt = t1 - t0;
      
  int FLOPs = 12*6*nsimd //12 vectorized conj(a)*b
    + 12*2*nsimd; //12 vectorized += or -=
  double call_FLOPs = double(FLOPs) * double(aa.nfsites());

  double call_time_us = dt/ntests;
  double call_time_s = dt/ntests /1e6;
    
  double flops = call_FLOPs/call_time_us; //dt in us   dt/(1e-6 s) in Mflops

  double bytes_read = 2* 12 * 2*8 * nsimd; 
  double bytes_store = 2 * nsimd;

  double call_bytes = (bytes_read + bytes_store) * double(aa.nfsites());
        
  double bandwidth = call_bytes/call_time_s / 1024./1024.; // in MB/s

  double FLOPS_per_byte = FLOPs/(bytes_read + bytes_store);
  double theor_perf = FLOPS_per_byte * bandwidth; //in Mflops (assuming bandwidth bound)
    
  std::cout << "GridVectorizedSpinColorContract( conj(a)*b ): New code " << ntests << " tests over " << nthreads << " threads: Time " << dt << " usecs  flops " << flops/1e3 << " Gflops (hot)\n";
  std::cout << "Time per call " << call_time_us << " usecs" << std::endl;
  std::cout << "Memory bandwidth " << bandwidth << " MB/s" << std::endl;
  std::cout << "FLOPS/byte " << FLOPS_per_byte << std::endl;
  std::cout << "Theoretical performance " << theor_perf/1e3 << " Gflops\n";    
  std::cout << "Total work is " << aa.nfsites() << " and Nsimd = " << nsimd << std::endl;
#endif
}

template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void benchmarkMFcontract(const A2Aparams &a2a_params, const int ntests, const int nthreads){
#ifdef USE_GRID
  typedef typename GridA2Apolicies::SourcePolicies GridSrcPolicy;
  typedef typename ScalarA2Apolicies::ScalarComplexType Ctype;
  typedef typename Ctype::value_type Ftype;

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);

  A2AvectorWfftw<ScalarA2Apolicies> W(a2a_params);
  A2AvectorVfftw<ScalarA2Apolicies> V(a2a_params);
  
  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_params, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_params, simd_dims);
  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_grid;

  ThreeDSIMDPolicy<OneFlavorPolicy>::ParamType simd_dims_3d;
  ThreeDSIMDPolicy<OneFlavorPolicy>::SIMDdefaultLayout(simd_dims_3d,nsimd);

  printf("Nsimd = %d, SIMD dimensions:\n", nsimd);
  for(int i=0;i<4;i++)
    printf("%d ", simd_dims[i]);
  printf("\n");
  
  int p[3] = {1,1,1};
  A2AflavorProjectedExpSource<GridSrcPolicy> src_grid(2.0,p,simd_dims_3d);
  //typedef SCFspinflavorInnerProductCT<15,sigma3,typename GridA2Apolicies::ComplexType,A2AflavorProjectedExpSource<GridSrcPolicy> > GridInnerProduct;
  //GridInnerProduct mf_struct_grid(src_grid);

  typedef SCFspinflavorInnerProduct<15,typename GridA2Apolicies::ComplexType,A2AflavorProjectedExpSource<GridSrcPolicy> > GridInnerProduct;
  GridInnerProduct mf_struct_grid(sigma3,src_grid);
  
  std::cout << "Starting all-time mesonfield contract benchmark\n";
  if(!UniqueID()){ printf("Using outer blocking bi %d bj %d bp %d\n",BlockedMesonFieldArgs::bi,BlockedMesonFieldArgs::bj,BlockedMesonFieldArgs::bp); fflush(stdout); }
  if(!UniqueID()){ printf("Using inner blocking bi %d bj %d bp %d\n",BlockedMesonFieldArgs::bii,BlockedMesonFieldArgs::bjj,BlockedMesonFieldArgs::bpp); fflush(stdout); }

  Float total_time = 0.;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_grid_t;

#if 0
  std::cout << "Generating random fields" << std::endl;
  W.testRandom();
  V.testRandom();
  std::cout << "Importing random fields into Grid A2A vectors" << std::endl;
  Wgrid.importFields(W);
  Vgrid.importFields(V);
#else
  //Just zero the data, makes no difference
  Wgrid.zero();
  Vgrid.zero();
#endif
  
  typedef typename std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> >::allocator_type Allocator;

#ifndef GPU_VEC
  std::cout << "Using CPU implementation" << std::endl;
  typedef SingleSrcVectorPoliciesSIMD<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,GridInnerProduct> VectorPolicies;
  //typedef SingleSrcVectorPolicies<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,GridInnerProduct> VectorPolicies;
  mfComputeGeneral<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw, GridInnerProduct, VectorPolicies> cg;
#else
  std::cout << "Using Grid offloaded implementation" << std::endl;
  typedef SingleSrcVectorPoliciesSIMDoffload<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,GridInnerProduct> VectorPolicies;
  mfComputeGeneralOffload<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw, GridInnerProduct, VectorPolicies> cg;
#endif

  BlockedMesonFieldArgs::enable_profiling = false; 

  std::cout << "Starting benchmark test loop" << std::endl; std::cout.flush();
  //ProfilerStart("SingleSrcProfile.prof");  
  for(int iter=0;iter<ntests+1;iter++){
    if(iter > 0) total_time -= dclock();

    //__itt_resume();
    if(iter == ntests) BlockedMesonFieldArgs::enable_profiling = true;
    cg.compute(mf_grid_t,Wgrid,mf_struct_grid,Vgrid, true);
    if(iter == ntests) BlockedMesonFieldArgs::enable_profiling = false;
    //__itt_pause();

    if(iter > 0) total_time += dclock();
  }
  //  __itt_detach();
//ProfilerStop();  

  const typename GridA2Apolicies::FermionFieldType &mode0 = Wgrid.getMode(0);
  size_t g5_FLOPs = 12*6*nsimd + 12*2*nsimd;//12 vectorized conj(a)*b  + 12 vectorized += or -=         
  size_t siteFmat_FLOPs = 3*nsimd;  //1 vectorized z.im*-1, 1 vectorized -1*z
  size_t s3_FLOPs = 4*nsimd; //2 vectorized -1*z
  size_t TransLeftTrace_FLOPs = nsimd*4*6 + nsimd*3*2; //4 vcmul + 3vcadd
  size_t reduce_FLOPs = nsimd*2; //nsimd cadd  (reduce over lanes and sites)

  size_t size_3d = mode0.nodeSites(0)*mode0.nodeSites(1)*mode0.nodeSites(2);
  size_t field4d_bytes = size_3d * GJP.TnodeSites() * 24 * 2*8*nsimd;

  size_t rd_bytes = 
    Wgrid.getNmodes() * field4d_bytes 
    +
    Vgrid.getNmodes() * field4d_bytes 
    +
    size_3d * 2*8*nsimd; //the source

  size_t wr_bytes = 0;
    
  size_t FLOPs_per_site = 0.;
  for(int t=GJP.TnodeCoor()*GJP.TnodeSites(); t<(GJP.TnodeCoor()+1)*GJP.TnodeSites(); t++){
    const int nl_l = mf_grid_t[t].getRowParams().getNl();
    const int nl_r = mf_grid_t[t].getColParams().getNl();

    wr_bytes += nl_l * nl_r * 2*8; //non-SIMD complex matrix

    int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();

    for(int i = 0; i < mf_grid_t[t].getNrows(); i++){
      modeIndexSet i_high_unmapped; if(i>=nl_l) mf_grid_t[t].getRowParams().indexUnmap(i-nl_l,i_high_unmapped);
      SCFvectorPtr<typename GridA2Apolicies::FermionFieldType::FieldSiteType> lscf = Wgrid.getFlavorDilutedVect(i,i_high_unmapped,0,t_lcl); //dilute flavor in-place if it hasn't been already
                                                                                                                                                                       
      for(int j = 0; j < mf_grid_t[t].getNcols(); j++) {
	modeIndexSet j_high_unmapped; if(j>=nl_r) mf_grid_t[t].getColParams().indexUnmap(j-nl_r,j_high_unmapped);
	SCFvectorPtr<typename GridA2Apolicies::FermionFieldType::FieldSiteType> rscf = Vgrid.getFlavorDilutedVect(j,j_high_unmapped,0,t_lcl);

	for(int a=0;a<2;a++)
	  for(int b=0;b<2;b++)
	    if(!lscf.isZero(a) && !rscf.isZero(b))
	      FLOPs_per_site += g5_FLOPs;
	FLOPs_per_site += siteFmat_FLOPs + s3_FLOPs + TransLeftTrace_FLOPs + reduce_FLOPs;
      }
    }
  }
  size_t total_FLOPs = FLOPs_per_site * size_3d;
  double t_avg = total_time / ntests;
  double mem_bandwidth_GBps = double(rd_bytes + wr_bytes)/t_avg / 1024/1024/1024;
  double Gflops = double(total_FLOPs)/t_avg/1e9;

  printf("MF contract all t: Avg time new code %d iters: %g secs. Avg flops %g Gflops. Avg bandwidth %g GB/s\n",ntests,t_avg, Gflops, mem_bandwidth_GBps);
#endif
}


template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void benchmarkMultiSrcMFcontract(const A2AArg &a2a_args, const int ntests, const int nthreads){
#ifdef USE_GRID
  typedef typename GridA2Apolicies::SourcePolicies GridSrcPolicy;
  typedef typename ScalarA2Apolicies::ScalarComplexType Ctype;
  typedef typename Ctype::value_type Ftype;

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);

  A2AvectorWfftw<ScalarA2Apolicies> W(a2a_args);
  A2AvectorVfftw<ScalarA2Apolicies> V(a2a_args);
  
  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);
  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_grid;

  ThreeDSIMDPolicy<OneFlavorPolicy>::ParamType simd_dims_3d;
  ThreeDSIMDPolicy<OneFlavorPolicy>::SIMDdefaultLayout(simd_dims_3d,nsimd);

  printf("Nsimd = %d, SIMD dimensions:\n", nsimd);
  for(int i=0;i<4;i++)
    printf("%d ", simd_dims[i]);
  printf("\n");
  
  int p[3] = {1,1,1};

  typedef typename GridA2Apolicies::ComplexType ComplexType;  
  typedef A2AflavorProjectedExpSource<GridSrcPolicy> ExpSrcType;
  typedef A2AflavorProjectedHydrogenSource<GridSrcPolicy> HydSrcType;
  typedef Elem<ExpSrcType, Elem<HydSrcType,ListEnd > > SrcList;
  typedef A2AmultiSource<SrcList> MultiSrcType;
  typedef GparitySourceShiftInnerProduct<ComplexType,MultiSrcType, flavorMatrixSpinColorContract<15,true,false> > MultiInnerType;

  const double rad = 2.0;
  MultiSrcType src;
  src.template getSource<0>().setup(rad,p,simd_dims_3d); //1s
  src.template getSource<1>().setup(2,0,0,rad,p,simd_dims_3d); //2s

  MultiInnerType g5_s3_inner(sigma3, src);
  std::vector<std::vector<int> > shifts(1, std::vector<int>(3,0));
  g5_s3_inner.setShifts(shifts);
  
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();
  
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_exp;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_hyd;

  std::vector< std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> >* > mf_st(2);
  mf_st[0] = &mf_exp;
  mf_st[1] = &mf_hyd;
  
  std::cout << "Starting all-time mesonfield contract benchmark with multi-src (1s, 2s hyd)\n";
  if(!UniqueID()) printf("Using outer blocking bi %d bj %d bp %d\n",BlockedMesonFieldArgs::bi,BlockedMesonFieldArgs::bj,BlockedMesonFieldArgs::bp);
  if(!UniqueID()) printf("Using inner blocking bi %d bj %d bp %d\n",BlockedMesonFieldArgs::bii,BlockedMesonFieldArgs::bjj,BlockedMesonFieldArgs::bpp);

  Float total_time = 0.;

  W.testRandom();
  V.testRandom();
  Wgrid.importFields(W);
  Vgrid.importFields(V);
      
  CALLGRIND_START_INSTRUMENTATION ;
  CALLGRIND_TOGGLE_COLLECT ;
  
  typedef typename std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> >::allocator_type Allocator;

#ifndef GPU_VEC
  typedef MultiSrcVectorPoliciesSIMD<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,MultiInnerType> VectorPolicies;
  //typedef MultiSrcVectorPolicies<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,MultiInnerType> VectorPolicies;
  mfComputeGeneral<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw, MultiInnerType, VectorPolicies> cg;
#else
  typedef MultiSrcVectorPoliciesSIMDoffload<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,MultiInnerType> VectorPolicies;
  mfComputeGeneralOffload<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw, MultiInnerType, VectorPolicies> cg;
#endif  

  ProfilerStart("MultiSrcProfile.prof");
  for(int iter=0;iter<ntests;iter++){
    total_time -= dclock();

    __itt_resume();
    cg.compute(mf_st,Wgrid,g5_s3_inner,Vgrid, true);
    __itt_pause();
    
    //A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_grid_t,Wgrid,mf_struct_grid,Vgrid);
    total_time += dclock();
  }
  ProfilerStop();
  __itt_detach();


  CALLGRIND_TOGGLE_COLLECT ;
  CALLGRIND_STOP_INSTRUMENTATION ;

  int nsrc = 2;
  
  int g5_FLOPs = 12*6*nsimd + 12*2*nsimd;//4 flav * 12 vectorized conj(a)*b  + 12 vectorized += or -=
  int siteFmat_FLOPs = nsrc*3*nsimd;  //1 vectorized z.im*-1, 1 vectorized -1*z
  int s3_FLOPs = nsrc*4*nsimd; //2 vectorized -1*z
  int TransLeftTrace_FLOPs = nsrc*nsimd*4*6 + nsrc*nsimd*3*2; //4 vcmul + 3vcadd
  int reduce_FLOPs = 0; // (nsimd - 1)*2; //nsimd-1 cadd

  double FLOPs_per_site = 0.;
  for(int t=GJP.TnodeCoor()*GJP.TnodeSites(); t<(GJP.TnodeCoor()+1)*GJP.TnodeSites(); t++){
    const int nl_l = mf_exp[t].getRowParams().getNl();
    const int nl_r = mf_exp[t].getColParams().getNl();

    int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();

    for(int i = 0; i < mf_exp[t].getNrows(); i++){
      modeIndexSet i_high_unmapped; if(i>=nl_l) mf_exp[t].getRowParams().indexUnmap(i-nl_l,i_high_unmapped);
      SCFvectorPtr<typename GridA2Apolicies::FermionFieldType::FieldSiteType> lscf = Wgrid.getFlavorDilutedVect(i,i_high_unmapped,0,t_lcl); //dilute flavor in-place if it hasn't been already \
                                                                                                                                                                                                           
      for(int j = 0; j < mf_exp[t].getNcols(); j++) {
	modeIndexSet j_high_unmapped; if(j>=nl_r) mf_exp[t].getColParams().indexUnmap(j-nl_r,j_high_unmapped);
	SCFvectorPtr<typename GridA2Apolicies::FermionFieldType::FieldSiteType> rscf = Vgrid.getFlavorDilutedVect(j,j_high_unmapped,0,t_lcl);

	for(int a=0;a<2;a++)
	  for(int b=0;b<2;b++)
	    if(!lscf.isZero(a) && !rscf.isZero(b))
	      FLOPs_per_site += g5_FLOPs;
	FLOPs_per_site += siteFmat_FLOPs + s3_FLOPs + TransLeftTrace_FLOPs + reduce_FLOPs;
      }
    }
  }
  const typename GridA2Apolicies::FermionFieldType &mode0 = Wgrid.getMode(0);
  const int size_3d = mode0.nodeSites(0)*mode0.nodeSites(1)*mode0.nodeSites(2);
  double total_FLOPs = double(FLOPs_per_site) * double(size_3d) * double(ntests);

  printf("MF contract all t multi-src: Avg time new code %d iters: %g secs. Avg flops %g Gflops\n",ntests,total_time/ntests, total_FLOPs/total_time/1e9);
#endif
}





template<typename GridA2Apolicies>
void benchmarkMultiShiftMFcontract(const A2AArg &a2a_args, const int nshift){
#ifdef USE_GRID
  typedef typename GridA2Apolicies::ComplexType ComplexType;  

  const int nsimd = ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);

  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);

  //Just zero the data, makes no difference
  Wgrid.zero();
  Vgrid.zero();


  ThreeDSIMDPolicy<OneFlavorPolicy>::ParamType simd_dims_3d;
  ThreeDSIMDPolicy<OneFlavorPolicy>::SIMDdefaultLayout(simd_dims_3d,nsimd);

  printf("Nsimd = %d, SIMD dimensions:\n", nsimd);
  for(int i=0;i<4;i++)
    printf("%d ", simd_dims[i]);
  printf("\n");

  const typename GridA2Apolicies::FermionFieldType &mode0 = Wgrid.getMode(0);
  const int size_3d = mode0.nodeSites(0)*mode0.nodeSites(1)*mode0.nodeSites(2);

  typedef typename GridA2Apolicies::SourcePolicies SourcePolicies;
  typedef A2AflavorProjectedExpSource<SourcePolicies> ExpSrcType;
  typedef GparitySourceShiftInnerProduct<ComplexType,ExpSrcType, flavorMatrixSpinColorContract<0,true,false> > InnerType;

  double rad_1s = 2.;
  int pbase[3];
  GparityBaseMomentum(pbase,+1);    
  ExpSrcType src(rad_1s,pbase,simd_dims_3d);
  InnerType inner(sigma0, src);    

  std::vector<int> shift_base = {1,1,1};
  std::vector<std::vector<int> > shifts(nshift, shift_base);
  
  inner.setShifts(shifts);

  typedef std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > MfVectorType;
  std::vector<MfVectorType> mf(nshift);
  std::vector<MfVectorType*> mf_p(nshift);
  for(int s=0;s<nshift;s++) mf_p[s] = &mf[s];
  
  std::cout << "Timing contraction with nshift=" << nshift << std::endl;
  double time = -dclock();
  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_p,Wgrid, inner, Vgrid);
  time += dclock();

  int nsrc = nshift;
  
  int g5_FLOPs = 12*6*nsimd + 12*2*nsimd;//4 flav * 12 vectorized conj(a)*b  + 12 vectorized += or -=
  int siteFmat_FLOPs = nsrc*3*nsimd;  //1 vectorized z.im*-1, 1 vectorized -1*z
  int s3_FLOPs = nsrc*4*nsimd; //2 vectorized -1*z
  int TransLeftTrace_FLOPs = nsrc*nsimd*4*6 + nsrc*nsimd*3*2; //4 vcmul + 3vcadd
  int reduce_FLOPs = 0; // (nsimd - 1)*2; //nsimd-1 cadd
  
  double FLOPs_per_site = 0.;
  for(int t=GJP.TnodeCoor()*GJP.TnodeSites(); t<(GJP.TnodeCoor()+1)*GJP.TnodeSites(); t++){
    const A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> &mf_shift0 = mf[0][t];
    const int nl_l = mf_shift0.getRowParams().getNl();
    const int nl_r = mf_shift0.getColParams().getNl();

    int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();
    
    for(int i = 0; i < mf_shift0.getNrows(); i++){
      modeIndexSet i_high_unmapped; if(i>=nl_l) mf_shift0.getRowParams().indexUnmap(i-nl_l,i_high_unmapped);
      SCFvectorPtr<typename GridA2Apolicies::FermionFieldType::FieldSiteType> lscf = Wgrid.getFlavorDilutedVect(i,i_high_unmapped,0,t_lcl); //dilute flavor in-place if it hasn't been already \
																	    
      for(int j = 0; j < mf_shift0.getNcols(); j++) {
	modeIndexSet j_high_unmapped; if(j>=nl_r) mf_shift0.getColParams().indexUnmap(j-nl_r,j_high_unmapped);
	SCFvectorPtr<typename GridA2Apolicies::FermionFieldType::FieldSiteType> rscf = Vgrid.getFlavorDilutedVect(j,j_high_unmapped,0,t_lcl);
	
	for(int a=0;a<2;a++)
	  for(int b=0;b<2;b++)
	    if(!lscf.isZero(a) && !rscf.isZero(b))
	      FLOPs_per_site += g5_FLOPs;
	FLOPs_per_site += siteFmat_FLOPs + s3_FLOPs + TransLeftTrace_FLOPs + reduce_FLOPs;
      }
    }
  }
  double total_FLOPs = double(FLOPs_per_site) * double(size_3d);

  double Mflops = total_FLOPs / time / 1e6;
  std::cout << "Nshift=" << nshift << " Time per=" << time << "s Mflops=" << Mflops << std::endl;
 
#endif
}



CPS_END_NAMESPACE
