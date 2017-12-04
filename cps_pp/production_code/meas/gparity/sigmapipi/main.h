#ifndef _MAIN_H
#define _MAIN_H

//Header for main program

using namespace cps;


//Setup the A2A policy
#ifdef USE_DESTRUCTIVE_FFT

#ifdef A2A_PREC_DOUBLE
typedef A2ApoliciesDoubleManualAlloc A2Apolicies;
#elif defined(A2A_PREC_SINGLE)
typedef A2ApoliciesSingleManualAlloc A2Apolicies;
#elif defined(A2A_PREC_SIMD_DOUBLE)
typedef A2ApoliciesSIMDdoubleManualAlloc A2Apolicies;
#elif defined(A2A_PREC_SIMD_SINGLE)
typedef A2ApoliciesSIMDsingleManualAlloc A2Apolicies;
#else
#error "Must provide an A2A precision"
#endif

#else

#ifdef A2A_PREC_DOUBLE
typedef A2ApoliciesDoubleAutoAlloc A2Apolicies;
#elif defined(A2A_PREC_SINGLE)
typedef A2ApoliciesSingleAutoAlloc A2Apolicies;
#elif defined(A2A_PREC_SIMD_DOUBLE)
typedef A2ApoliciesSIMDdoubleAutoAlloc A2Apolicies;
#elif defined(A2A_PREC_SIMD_SINGLE)
typedef A2ApoliciesSIMDsingleAutoAlloc A2Apolicies;
#else
#error "Must provide an A2A precision"
#endif

#endif

//Command line argument store/parse
struct CommandLineArgs{
  int nthreads;
  bool randomize_vw; //rather than doing the Lanczos and inverting the propagators, etc, just use random vectors for V and W
  bool randomize_evecs; //skip Lanczos and just use random evecs for testing.
  bool force_evec_compute; //randomize_evecs causes Lanczos to be skipped unless this option is used
  bool tune_lanczos; //just run the light lanczos on first config then exit
  bool skip_gauge_fix;
  bool double_latt; //most ancient 8^4 quenched lattices stored both U and U*. Enable this to read those configs

#ifdef USE_GRID
  bool run_initial_grid_benchmarks;
#endif

  bool load_sigma_mf;
  std::string load_sigma_mf_dir;
  bool load_pion_mf;
  std::string load_pion_mf_dir;
  
  CommandLineArgs(int argc, char **argv, int begin){
    nthreads = 1;
#if TARGET == BGQ
    nthreads = 64;
#endif
    randomize_vw = false;
    randomize_evecs = false;
    force_evec_compute = false; //randomize_evecs causes Lanczos to be skipped unless this option is used
    tune_lanczos = false; //just run the light lanczos on first config then exit
    skip_gauge_fix = false;
    double_latt = false; //most ancient 8^4 quenched lattices stored both U and U*. Enable this to read those configs

#ifdef USE_GRID
    run_initial_grid_benchmarks = false;
#endif

    load_sigma_mf = false;
    load_pion_mf = false;
     
    parse(argc,argv,begin);
  }

  void parse(int argc, char **argv, int begin){
    if(!UniqueID()){ printf("Arguments:\n"); fflush(stdout); }
    for(int i=0;i<argc;i++){
      if(!UniqueID()){ printf("%d \"%s\"\n",i,argv[i]); fflush(stdout); }
    }
    
    const int ngrid_arg = 13;
    const std::string grid_args[ngrid_arg] = { "--debug-signals", "--dslash-generic", "--dslash-unroll",
					       "--dslash-asm", "--shm", "--lebesgue",
					       "--cacheblocking", "--comms-concurrent", "--comms-sequential",
					       "--comms-overlap", "--log", "--comms-threads",
					       "--shm-hugepages" };
    const int grid_args_skip[ngrid_arg] =    { 1  , 1 , 1,
					       1  , 2 , 1,
					       2  , 1 , 1,
					       1  , 2 , 2,
					       1  	  };

    int arg = begin;
    while(arg < argc){
      char* cmd = argv[arg];
      if( strncmp(cmd,"-nthread",8) == 0){
	if(arg == argc-1){ if(!UniqueID()){ printf("-nthread must be followed by a number!\n"); fflush(stdout); } exit(-1); }
	nthreads = strToAny<int>(argv[arg+1]);
	if(!UniqueID()){ printf("Setting number of threads to %d\n",nthreads); }
	arg+=2;
      }else if( strncmp(cmd,"-randomize_vw",15) == 0){
	randomize_vw = true;
	if(!UniqueID()){ printf("Using random vectors for V and W, skipping Lanczos and inversion stages\n"); fflush(stdout); }
	arg++;
      }else if( strncmp(cmd,"-randomize_evecs",15) == 0){
	randomize_evecs = true;
	if(!UniqueID()){ printf("Using random eigenvectors\n"); fflush(stdout); }
	arg++;      
      }else if( strncmp(cmd,"-force_evec_compute",15) == 0){
	force_evec_compute = true;
	if(!UniqueID()){ printf("Forcing evec compute despite randomize_vw\n"); fflush(stdout); }
	arg++;      
      }else if( strncmp(cmd,"-tune_lanczos",15) == 0){
	tune_lanczos = true;
	if(!UniqueID()){ printf("Just tuning light lanczos on first config\n"); fflush(stdout); }
	arg++;
      }else if( strncmp(cmd,"-double_latt",15) == 0){
	double_latt = true;
	if(!UniqueID()){ printf("Loading doubled lattices\n"); fflush(stdout); }
	arg++;
      }else if( strncmp(cmd,"-skip_gauge_fix",20) == 0){
	skip_gauge_fix = true;
	if(!UniqueID()){ printf("Skipping gauge fixing\n"); fflush(stdout); }
	arg++;
      }else if( strncmp(cmd,"-mf_outerblocking",15) == 0){
	int* b[3] = { &BlockedMesonFieldArgs::bi, &BlockedMesonFieldArgs::bj, &BlockedMesonFieldArgs::bp };
	for(int a=0;a<3;a++) *b[a] = strToAny<int>(argv[arg+1+a]);
	arg+=4;
      }else if( strncmp(cmd,"-mf_innerblocking",15) == 0){
	int* b[3] = { &BlockedMesonFieldArgs::bii, &BlockedMesonFieldArgs::bjj, &BlockedMesonFieldArgs::bpp };
	for(int a=0;a<3;a++) *b[a] = strToAny<int>(argv[arg+1+a]);
	arg+=4;    
      }else if( strncmp(cmd,"--comms-isend",30) == 0){
	ERR.General("","main","Grid option --comms-isend is deprecated: use --comms-concurrent instead");
      }else if( strncmp(cmd,"--comms-sendrecv",30) == 0){
	ERR.General("","main","Grid option --comms-sendrecv is deprecated: use --comms-sequential instead");
#ifdef BNL_KNL_PERFORMANCE_CHECK
      }else if( strncmp(cmd,"-bnl_knl_minperf",30) == 0){
	bnl_knl_minperf = strToAny<double>(argv[arg+1]);
	if(!UniqueID()) printf("Set BNL KNL min performance to %f Mflops/node\n",bnl_knl_minperf);
	arg+=2;
#endif
#ifdef MESONFIELD_USE_BURSTBUFFER
      }else if( strncmp(cmd,"-mesonfield_scratch_stub",50) == 0){
	BurstBufferMemoryStorage::filestub() = argv[arg+1];
	if(!UniqueID()) printf("Set mesonfield scratch stub to %s\n",BurstBufferMemoryStorage::filestub().c_str());
	arg+=2;
#endif
      }else if( strncmp(cmd,"-load_sigma_mf",30) == 0){
	load_sigma_mf = true;
	load_sigma_mf_dir = argv[arg+1];
	if(!UniqueID()) printf("Loading sigma meson fields from %s\n",load_sigma_mf_dir.c_str());
	arg +=2;
      }else if( strncmp(cmd,"-load_pion_mf",30) == 0){
	load_pion_mf = true;
	load_pion_mf_dir = argv[arg+1];
	if(!UniqueID()) printf("Loading pion meson fields from %s\n",load_pion_mf_dir.c_str());
	arg +=2;
      }else{
	bool is_grid_arg = false;
	for(int i=0;i<ngrid_arg;i++){
	  if( std::string(cmd) == grid_args[i] ){
	    if(!UniqueID()){ printf("main.C: Ignoring Grid argument %s\n",cmd); fflush(stdout); }
	    arg += grid_args_skip[i];
	    is_grid_arg = true;
	    break;
	  }
	}
	if(!is_grid_arg){
	  if(UniqueID()==0) printf("Unrecognised argument: %s\n",cmd);
	  exit(-1);
	}
      }
    }
  }

};

//Store/read job parameters
struct Parameters{
  CommonArg common_arg;
  CommonArg common_arg2;
  DoArg do_arg;
  JobParams jp;
  MeasArg meas_arg;
  FixGaugeArg fix_gauge_arg;
  A2AArg a2a_arg;
  LancArg lanc_arg;

  Parameters(const char* directory): common_arg("",""), common_arg2("",""){
    if(chdir(directory)!=0) ERR.General("Parameters","Parameters","Unable to switch to directory '%s'\n",directory);

    if(!do_arg.Decode("do_arg.vml","do_arg")){
      do_arg.Encode("do_arg.templ","do_arg");
      VRB.Result("Parameters","Parameters","Can't open do_arg.vml!\n");exit(1);
    }
    if(!jp.Decode("job_params.vml","job_params")){
      jp.Encode("job_params.templ","job_params");
      VRB.Result("Parameters","Parameters","Can't open job_params.vml!\n");exit(1);
    }
    if(!meas_arg.Decode("meas_arg.vml","meas_arg")){
      meas_arg.Encode("meas_arg.templ","meas_arg");
      std::cout<<"Can't open meas_arg!"<<std::endl;exit(1);
    }
    if(!a2a_arg.Decode("a2a_arg.vml","a2a_arg")){
      a2a_arg.Encode("a2a_arg.templ","a2a_arg");
      VRB.Result("Parameters","Parameters","Can't open a2a_arg.vml!\n");exit(1);
    }
    if(!lanc_arg.Decode("lanc_arg.vml","lanc_arg")){
      lanc_arg.Encode("lanc_arg.templ","lanc_arg");
      VRB.Result("Parameters","Parameters","Can't open lanc_arg.vml!\n");exit(1);
    }
    if(!fix_gauge_arg.Decode("fix_gauge_arg.vml","fix_gauge_arg")){
      fix_gauge_arg.Encode("fix_gauge_arg.templ","fix_gauge_arg");
      VRB.Result("Parameters","Parameters","Can't open fix_gauge_arg.vml!\n");exit(1);
    }

    common_arg.set_filename(meas_arg.WorkDirectory);
  }

};

void setupJob(int argc, char **argv, const Parameters &params, const CommandLineArgs &cmdline){
  initCPS(argc, argv, params.do_arg, cmdline.nthreads);
  
#ifdef NODE_DISTRIBUTE_MESONFIELDS
  if(!UniqueID()) printf("Using node distribution of meson fields\n");
#endif
#ifdef MEMTEST_MODE
  if(!UniqueID()) printf("Running in MEMTEST MODE (so don't expect useful results)\n");
#endif
  
#ifdef A2A_LANCZOS_SINGLE
  if(!cmdline.evecs_single_prec) ERR.General("",fname,"Must use single-prec eigenvectors when doing Lanczos in single precision\n");
#endif
  
  if(cmdline.double_latt) SerialIO::dbl_latt_storemode = true;

  if(!cmdline.tune_lanczos){ 
    assert(params.a2a_arg.nl <= params.lanc_arg.N_true_get);
  }

  printMem("Initial memory post-initialize");
}

void runInitialGridBenchmarks(const CommandLineArgs &cmdline, const Parameters &params){
#if defined(USE_GRID) && defined(USE_GRID_A2A)
  if(cmdline.run_initial_grid_benchmarks){
    typedef typename A2Apolicies::FgridGFclass A2ALattice;
    A2ALattice* lat = createLattice<A2ALattice,isGridtype>::doit(params.jp);
    gridBenchmark<A2Apolicies>(*lat);
    gridBenchmarkSinglePrec<A2Apolicies>(*lat);
    delete lat;
  }
#endif
}


//Tune the Lanczos and exit
void LanczosTune(const Parameters &params, BFMGridSolverWrapper &solvers){
  BFMGridLanczosWrapper<A2Apolicies> eig(solvers, params.jp);
  if(!UniqueID()) printf("Tuning lanczos light with mass %f\n", params.lanc_arg.mass);
    
  double time = -dclock();
  eig.compute(params.lanc_arg);
  time += dclock();
  print_time("main","Lanczos light",time);
    
  exit(0);
}

void computeVW(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, const Parameters &params,
	       const BFMGridLanczosWrapper<A2Apolicies> &eig, const BFMGridA2ALatticeWrapper<A2Apolicies> &a2a_lat,
	       const bool randomize_vw){
  typedef typename A2Apolicies::FermionFieldType::InputParamType Field4DparamType;
  Field4DparamType field4dparams = V.getFieldInputParams();
  
  if(!UniqueID()){ printf("V vector requires %f MB, W vector %f MB of memory\n", 
			  A2AvectorV<A2Apolicies>::Mbyte_size(params.a2a_arg,field4dparams), A2AvectorW<A2Apolicies>::Mbyte_size(params.a2a_arg,field4dparams) );
    fflush(stdout);
  }
  
  if(!UniqueID()) printf("Computing light quark A2A vectors\n");
  double time = -dclock();

  a2a_lat.computeVW(V,W,eig,params.jp.cg_controls,randomize_vw);
  
  printMem("Memory after light A2A vector computation");

  time += dclock();
  std::ostringstream os; os << "Light quark A2A vectors";
  print_time("main",os.str().c_str(),time);
}

void doGaugeFix(Lattice &lat, const bool skip_gauge_fix, const Parameters &params){
  doGaugeFix(lat,skip_gauge_fix,params.fix_gauge_arg);
}

void computePionMesonFields(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con_2s,
			  typename ComputePion<A2Apolicies>::Vtype &V, typename ComputePion<A2Apolicies>::Wtype &W,
			  const StandardPionMomentaPolicy &pion_mom,
			  const int conf, Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  if(!UniqueID()) printf("Computing pion meson fields\n");
  double time = -dclock();
  if(!GJP.Gparity()) ComputePion<A2Apolicies>::computeMesonFields(mf_ll_con, params.meas_arg.WorkDirectory,conf, pion_mom, W, V, params.jp.pion_rad, lat, field3dparams);
  else ComputePion<A2Apolicies>::computeGparityMesonFields(mf_ll_con, mf_ll_con_2s, params.meas_arg.WorkDirectory,conf, pion_mom, W, V, params.jp.pion_rad, lat, field3dparams);
  time += dclock();
  print_time("main","Pion meson fields",time);

  printMem("Memory after pion meson field computation");
}

void computeSigmaMesonFields(MesonFieldMomentumPairContainer<A2Apolicies> &store_1s,
			     MesonFieldMomentumPairContainer<A2Apolicies> &store_2s,
			     typename ComputeSigma<A2Apolicies>::Vtype &V, typename ComputeSigma<A2Apolicies>::Wtype &W,
			     const  StationarySigmaMomentaPolicy &sigma_mom,
			     const int conf, Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  double time = -dclock();
  if(!UniqueID()) printf("Computing sigma mesonfields\n");
  assert(GJP.Gparity());
  ComputeSigma<A2Apolicies>::computeGparityMesonFields(store_1s,store_2s,sigma_mom,W,V, params.jp.pion_rad, lat, field3dparams);
  time += dclock();
  print_time("main","Sigma meson fields ",time);
}

void readSigmaMesonFields(MesonFieldMomentumPairContainer<A2Apolicies> &store_1s,
			  MesonFieldMomentumPairContainer<A2Apolicies> &store_2s,
			  const StationarySigmaMomentaPolicy &sigma_mom,
			  const int traj, const Parameters &params,
			  const std::string &mf_dir){
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > tmp(Lt);

  for(int p=0;p<sigma_mom.nMom();p++){
    {
      std::ostringstream os;
      os << mf_dir << "/traj_" << traj << "_sigma_mfwv_mom" << sigma_mom.getWdagMom(p).file_str() << "_plus" << sigma_mom.getVmom(p).file_str() << "_hyd1s_rad" << params.jp.pion_rad << ".dat";
      A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::read(os.str(), tmp);    
      std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > &stored = store_1s.copyAdd(sigma_mom.getWdagMom(p), sigma_mom.getVmom(p), tmp);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
      nodeDistributeMany(1,&stored);
#endif
    }
    {
      std::ostringstream os;
      os << mf_dir << "/traj_" << traj << "_sigma_mfwv_mom" << sigma_mom.getWdagMom(p).file_str() << "_plus" << sigma_mom.getVmom(p).file_str() << "_hyd2s_rad" << params.jp.pion_rad << ".dat";
      A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::read(os.str(), tmp);    
      std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > &stored = store_2s.copyAdd(sigma_mom.getWdagMom(p), sigma_mom.getVmom(p), tmp);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
      nodeDistributeMany(1,&stored);
#endif
    }
  }
}  


void readPionMesonFields(MesonFieldMomentumContainer<A2Apolicies> &store_1s,
			 MesonFieldMomentumContainer<A2Apolicies> &store_2s,
			 const StandardPionMomentaPolicy &pion_mom,
			 const int traj, const Parameters &params,
			 const std::string &mf_dir){
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > tmp(Lt);

  for(int p=0;p<pion_mom.nMom();p++){
    {
      std::ostringstream os;
      os << mf_dir << "/traj_" << traj << "_pion_mf_mom" << pion_mom.getMesonMomentum(p).file_str() << "_hyd1s_rad" << params.jp.pion_rad << ".dat";
      A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::read(os.str(), tmp);    
      std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > &stored = store_1s.copyAdd(pion_mom.getMesonMomentum(p), tmp);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
      nodeDistributeMany(1,&stored);
#endif
    }
    {
      std::ostringstream os;
      os << mf_dir << "/traj_" << traj << "_pion_mf_mom" << pion_mom.getMesonMomentum(p).file_str() << "_hyd2s_rad" << params.jp.pion_rad << ".dat";
      A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::read(os.str(), tmp);    
      std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > &stored = store_2s.copyAdd(pion_mom.getMesonMomentum(p), tmp);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
      nodeDistributeMany(1,&stored);
#endif
    }
  }
}  




void readGaugeRNG(const Parameters &params, const CommandLineArgs &cmdline){
  readGaugeRNG(params.do_arg, params.meas_arg, cmdline.double_latt);
}

#define USE_TIANLES_CONVENTIONS

template<typename mf_Policies>
struct ComputeSigmaContractions{
  //Sigma has 2 terms in it's Wick contraction:    
  //0.5 * tr( G(x1,tsnk; x2, tsnk) ) * tr( G(y1, tsrc; y2, tsrc) )
  //-0.5 tr( G(y2, tsrc; x1, tsnk) * G(x2, tsnk; y1, tsrc) )
  //where x are sink locations and y src locations
  
  //We perform the Fourier transform   \Theta_x\Theta_y = \sum_{x1,x2,y1,y2} exp(-i p1_snk x1) exp(-i p2_snk x2) exp(-i p1_src y1) exp(-i p2_src y2) 
  
  
  //The first term has a vacuum subtraction so we just compute tr( G(x1,x2) ) and do the rest offline
  //\Theta_x tr( G(x1,tsnk; x2, tsnk) ) = \Theta_x tr( V(x1,tsnk) W^dag(x2,tsnk) ) =  tr( [\Theta_x W^dag(x2,tsnk) V(x1,tsnk)]  ) = tr( M(tsnk,tsnk) )

  template<typename SigmaMomentumPolicy>
  static void computeDisconnectedBubble(fVector<typename mf_Policies::ScalarComplexType> &into,
					MesonFieldMomentumPairContainer<A2Apolicies> &mf_sigma_con, const SigmaMomentumPolicy &sigma_mom, const int pidx
		      ){
    typedef typename mf_Policies::ComplexType ComplexType;
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt);

    if(sigma_mom.nAltMom(pidx) != 1) ERR.General("ComputeSigmaContractions","computeDisconnectedBubble","Sigma with alternate momenta not implemented. Idx %d with momenta %s %s has %d alternative momenta", pidx, sigma_mom.getWdagMom(pidx).str().c_str(),  sigma_mom.getVmom(pidx).str().c_str(), sigma_mom.nAltMom(pidx) );
    
    ThreeMomentum p1 = sigma_mom.getWdagMom(pidx);
    ThreeMomentum p2 = sigma_mom.getVmom(pidx);

    assert(mf_sigma_con.contains(p1,p2));

    //Construct the meson fields
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf = mf_sigma_con.get(p1,p2);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    if(!UniqueID()){ printf("Gathering meson fields\n");  fflush(stdout); }
    nodeGetMany(1,&mf);
    cps::sync();
#endif
    
    //Distribute load over all nodes
    int work = Lt; //consider a better parallelization
    int node_work, node_off; bool do_work;
    getNodeWork(work,node_work,node_off,do_work);
    
    if(do_work){
      for(int t=node_off; t<node_off + node_work; t++){
	into(t) = trace(mf[t]);
      }
    }
    into.nodeSum();
  }



  //Tianle computes the product of the traces online and adds it to his answer. For consistency I do the same here
  static void computeDisconnectedDiagram(fMatrix<typename mf_Policies::ScalarComplexType> &into,
					 const fVector<typename mf_Policies::ScalarComplexType> &sigma_bubble_snk,
					 const fVector<typename mf_Policies::ScalarComplexType> &sigma_bubble_src){
    //0.5 * tr( G(x1,tsnk; x2, tsnk) ) * tr( G(y1, tsrc; y2, tsrc) )
    typedef typename mf_Policies::ScalarComplexType Complex;
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt,Lt);
    for(int tsrc=0; tsrc<Lt; tsrc++){
      for(int tsep=0; tsep<Lt; tsep++){
	int tsnk = (tsrc + tsep) % Lt;
	into(tsrc, tsep) = Complex(0.5) * sigma_bubble_snk(tsnk) * sigma_bubble_src(tsrc); 
      }
    }
  }
  
  //The second term we compute in full
  //  \Theta_x \Theta_y -0.5 tr( G(y2, tsrc; x1, tsnk) * G(x2, tsnk; y1, tsrc) )
  //= \Theta_x \Theta_y -0.5 tr( V(y2, tsrc) * W^dag(x1, tsnk) * V(x2, tsnk)*W^dag(y1, tsrc) )
  //= -0.5 tr(  [ \Theta_x W^dag(x1, tsnk) * V(x2, tsnk)] * [ \Theta_y W^dag(y1, tsrc) V(y2, tsrc) ] )
  //= -0.5 tr( M(tsnk,tsnk) * M(tsrc, tsrc) )
  
  template<typename SigmaMomentumPolicy>
  static void computeConnected(fMatrix<typename mf_Policies::ScalarComplexType> &into,
		      MesonFieldMomentumPairContainer<A2Apolicies> &mf_sigma_con, const SigmaMomentumPolicy &sigma_mom, const int pidx_src, const int pidx_snk
		      ){
    typedef typename mf_Policies::ComplexType ComplexType;
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt,Lt);

    if(sigma_mom.nAltMom(pidx_snk) != 1) ERR.General("ComputeSigmaContractions","computeConnected","Sigma (sink) with alternate momenta not implemented. Idx %d with momenta %s %s has %d alternative momenta", pidx_snk, sigma_mom.getWdagMom(pidx_snk).str().c_str(),  sigma_mom.getVmom(pidx_snk).str().c_str(), sigma_mom.nAltMom(pidx_snk) );

    if(sigma_mom.nAltMom(pidx_src) != 1) ERR.General("ComputeSigmaContractions","computeConnected","Sigma (source) with alternate momenta not implemented. Idx %d with momenta %s %s has %d alternative momenta", pidx_src, sigma_mom.getWdagMom(pidx_src).str().c_str(),  sigma_mom.getVmom(pidx_src).str().c_str(), sigma_mom.nAltMom(pidx_src) );
        
    ThreeMomentum p1_src = sigma_mom.getWdagMom(pidx_src);
    ThreeMomentum p2_src = sigma_mom.getVmom(pidx_src);

    ThreeMomentum p1_snk = sigma_mom.getWdagMom(pidx_snk);
    ThreeMomentum p2_snk = sigma_mom.getVmom(pidx_snk);

    assert(mf_sigma_con.contains(p1_src,p2_src));
    assert(mf_sigma_con.contains(p1_snk,p2_snk));
	   
    //Construct the meson fields
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_src = mf_sigma_con.get(p1_src,p2_src);
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_snk = mf_sigma_con.get(p1_snk,p2_snk);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    if(!UniqueID()){ printf("Gathering meson fields\n");  fflush(stdout); }
    nodeGetMany(2,&mf_src,&mf_snk);
    cps::sync();
#endif

    if(!UniqueID()){ printf("Starting trace\n");  fflush(stdout); }
    trace(into,mf_snk,mf_src);
    into *= ScalarComplexType(-0.5,0);
    rearrangeTsrcTsep(into); //rearrange temporal ordering
    
    cps::sync();
    if(!UniqueID()){ printf("Finished trace\n");  fflush(stdout); }

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(2,&mf_src,&mf_snk);
#endif
  }
};




template<typename mf_Policies>
struct ComputePiPiToSigmaContractions{
  //This is identical to the sigma->pipi up to an overall - sign, a sign change of the momenta and swapping tsrc<->tsnk. However we compute it explicitly here rather than reusing it because we want to compute only on a limited number of source timeslices
  
  //This has 2 Wick contractions. The first is disconnected, and we are able to re-use the sigma bubble from the sigma->sigma contractions and the pipi bubble from the pipi->pipi contractions
  //The second is connected, and has to be computed explicitly
  //  +sqrt(6)/2 \Theta_y \Theta_x tr( g5 s3 G(y1,tsrc; x1,tsnk) G(x2,tsnk; y3,tsrc) g5 s3 G(y4,tsrc; y2, tsrc) )
  //  +sqrt(6)/2 \Theta_y \Theta_x tr( g5 s3 V(y1,tsrc) W^dag(x1,tsnk) V(x2,tsnk) W^dag(y3,tsrc) g5 s3 V(y4,tsrc) W^dag(y2, tsrc) )
  //  +sqrt(6)/2 tr( [[\Theta_y12 W^dag(y2, tsrc) g5 s3 V(y1,tsrc)]] [[\Theta_x W^dag(x1,tsnk) V(x2,tsnk)]] [[\Theta_y34 W^dag(y3,tsrc) g5 s3 V(y4,tsrc)]]  )
  //  +sqrt(6)/2 tr( mf_piA(tsrc,tsrc) mf_sigma(tsnk,tsnk) mf_piB(tsrc,tsrc)  )

  //\Theta_x = \sum_x1,x2 exp(+i psnk2.y2) exp(+i psnk1.y1 )
  //\Theta_y = \sum_y1,y2,y3,y4  exp(-i psrc1.y1 ) exp(-i psrc2.y2) exp(-i psrc3.y3 ) exp(-i psrc4.y4)     

  //When split sources
  //  +sqrt(6)/4 tr( [mf_piB(tsrc,tsrc) mf_piA(tsrc-delta,tsrc-delta) + mf_piB(tsrc-delta,tsrc-delta) mf_piA(tsrc,tsrc) ] mf_sigma(tsnk,tsnk)   )

  //We label the inner pion as pi1, and it is this pion for which we specify the momentum

  //  +sqrt(6)/4 tr( [mf_pi1(tsrc,tsrc) mf_pi2(tsrc-delta,tsrc-delta) + mf_pi2(tsrc-delta,tsrc-delta) mf_pi1(tsrc,tsrc) ] mf_sigma(tsnk,tsnk)   )
  
  //We provide the momentum index of the second (inner) sink pion
  template<typename SigmaMomentumPolicy, typename PionMomentumPolicy>
  static void computeConnected(fMatrix<typename mf_Policies::ScalarComplexType> &into,
			       MesonFieldMomentumPairContainer<A2Apolicies> &mf_sigma_con, const SigmaMomentumPolicy &sigma_mom, const int pidx_sigma,
			       MesonFieldMomentumContainer<A2Apolicies> &mf_pion_con, const PionMomentumPolicy &pion_mom, const int pidx_pi1,
			       const int tsep_pipi, const int tstep_src
		      ){
    typedef typename mf_Policies::ComplexType ComplexType;
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt,Lt); into.zero();

    if(Lt % tstep_src != 0) ERR.General("ComputeSigmaToPipiContractions","computeConnected(..)","tstep_src must divide the time range exactly\n"); 

    if(sigma_mom.nAltMom(pidx_sigma) != 1) ERR.General("ComputeSigmaContractions","compute","Sigma with alternate momenta not implemented");

    //Work out the momenta we need
    ThreeMomentum p_pi1_src = pion_mom.getMesonMomentum(pidx_pi1);
    ThreeMomentum p_pi2_src = -p_pi1_src; //total zero momentum


    ThreeMomentum pWdag_sigma_snk = sigma_mom.getWdagMom(pidx_sigma);
    ThreeMomentum pV_sigma_snk = sigma_mom.getVmom(pidx_sigma);
    
    assert(mf_sigma_con.contains(pWdag_sigma_snk,pV_sigma_snk));
    assert(mf_pion_con.contains(p_pi1_src));
    assert(mf_pion_con.contains(p_pi2_src));

    //Distribute load over all nodes
    int work = Lt*Lt/tstep_src;
    int node_work, node_off; bool do_work;
    getNodeWork(work,node_work,node_off,do_work);
    
    //Gather the meson fields
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_sigma = mf_sigma_con.get(pWdag_sigma_snk, pV_sigma_snk);
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_pi1 = mf_pion_con.get(p_pi1_src); //meson field of the inner pion
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_pi2 = mf_pion_con.get(p_pi2_src);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    if(!UniqueID()){ printf("Gathering meson fields\n");  fflush(stdout); }

    //Only get the meson fields we actually need
    std::vector<bool> tslice_sigma_mask(Lt, false);
    std::vector<bool> tslice_pi1_mask(Lt, false);
    std::vector<bool> tslice_pi2_mask(Lt, false);

    if(do_work){
      for(int tt=node_off; tt<node_off + node_work; tt++){
	int rem = tt;
	int tsnk = rem % Lt; rem /= Lt; //sink time
	int tsrc = rem * tstep_src; //source time
	int tsrc2 = (tsrc-tsep_pipi+Lt) % Lt;

	tslice_sigma_mask[tsnk] = true;
	tslice_pi1_mask[tsrc] = true;
	tslice_pi2_mask[tsrc2] = true;
      }
    }
    
    nodeGetMany(3,
		&mf_sigma, &tslice_sigma_mask,
		&mf_pi1, &tslice_pi1_mask,
		&mf_pi2, &tslice_pi2_mask);
    cps::sync();
#endif

    //Do the contraction  +sqrt(6)/4 tr( [mf_pi1(tsrc,tsrc) mf_pi2(tsrc-delta,tsrc-delta) + mf_pi2(tsrc-delta,tsrc-delta) mf_pi1(tsrc,tsrc) ] mf_sigma(tsnk,tsnk)   )
    //We can relate the two terms by g5-hermiticity
#define PIPI_SIGMA_USE_G5_HERM

    if(do_work){
      for(int tt=node_off; tt<node_off + node_work; tt++){
	int rem = tt;
	int tsnk = rem % Lt; rem /= Lt; //sink time
	int tsrc = rem * tstep_src; //source time
	int tsrc2 = (tsrc-tsep_pipi+Lt) % Lt;
	int tdis = (tsnk - tsrc + Lt) % Lt;
	
#ifdef PIPI_SIGMA_USE_G5_HERM
	A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> pi_prod;
	//mult(pi_prod, mf_pi2[tsrc2], mf_pi1[tsrc],NODE_LOCAL);
	mult(pi_prod, mf_pi1[tsrc], mf_pi2[tsrc2],NODE_LOCAL);

	
	ScalarComplexType incr(0,0);
	incr += trace(pi_prod, mf_sigma[tsnk]);

	into(tsrc,tdis) +=  sqrt(6.)/2. * incr; 	
#else
	A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> pi_prod_1, pi_prod_2;
	mult(pi_prod_1, mf_pi2[tsrc2], mf_pi1[tsrc],NODE_LOCAL);
	mult(pi_prod_2, mf_pi1[tsrc], mf_pi2[tsrc2],NODE_LOCAL);

	ScalarComplexType incr(0,0);
	incr += trace(pi_prod_1, mf_sigma[tsnk]);
	incr += trace(pi_prod_2, mf_sigma[tsnk]);

	into(tsrc,tdis) +=  sqrt(6.)/4. * incr; //extra 1/2 from average of 2 topologies
#endif

#ifdef USE_TIANLES_CONVENTIONS
	into(tsrc,tdis) *= -1.;
#endif
      }
    }
    into.nodeSum();

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(3,&mf_sigma,&mf_pi1,&mf_pi2);
#endif
  }


  //Tianle computes the product of the traces online and adds it to his answer. For consistency I do the same here
  static void computeDisconnectedDiagram(fMatrix<typename mf_Policies::ScalarComplexType> &into,
  					 const fVector<typename mf_Policies::ScalarComplexType> &sigma_bubble,
  					 const fVector<typename mf_Policies::ScalarComplexType> &pipi_bubble,  
  					 const int tstep_src){
    //-sqrt(6)/4 \Theta_y \Theta_x \tr( G(x2, tsnk; x1, tsnk) ) \tr( g5 G(y1, tsrc; y3, tsrc) g5 s3 G(y4, tsrc; y2, tsrc) s3 ) 
    //-sqrt(6)/4 \Theta_y \Theta_x \tr( V(x2, tsnk)W^dag(x1, tsnk) ) \tr( g5 V(y1, tsrc)W^dag(y3, tsrc) g5 s3 V(y4, tsrc)W^dag(y2, tsrc) s3 ) 
    //-sqrt(6)/4 \tr( [[\Theta_x W^dag(x1, tsnk) V(x2, tsnk)]] ) \tr( [[\Theta_y12 W^dag(y2, tsrc) s3 g5 V(y1, tsrc)]] [[\Theta_y34 W^dag(y3, tsrc) g5 s3 V(y4, tsrc)]] ) 
    //-sqrt(6)/4 \tr( M_sigma(tsnk,tsnk) ) \tr( M_piA(tsrc,tsrc) M_piB(tsrc,tsrc)) 

    //When separated
    //-sqrt(6)/8 \tr( M_sigma(tsnk,tsnk) ) [  \tr( M_piA(tsrc-delta,tsrc-delta) M_piB(tsrc,tsrc)) + \tr( M_piA(tsrc,tsrc) M_piB(tsrc-delta,tsrc-delta))   ] 
    //-sqrt(6)/8 \tr( M_sigma(tsnk,tsnk) ) [  \tr( M_pi2(tsrc-delta,tsrc-delta) M_pi1(tsrc,tsrc)) + \tr( M_pi1(tsrc,tsrc) M_pi2(tsrc-delta,tsrc-delta))   ] 
    //-sqrt(6)/4 \tr( M_sigma(tsnk,tsnk) ) [  M_pi1(tsrc,tsrc)) \tr( M_pi2(tsrc-delta,tsrc-delta)  ] 
    
    //\Theta_x = \sum_x1,x2 exp(+i psnk2.y2) exp(+i psnk1.y1 )
    //\Theta_y = \sum_y1,y2,y3,y4  exp(-i psrc1.y1 ) exp(-i psrc2.y2) exp(-i psrc3.y3 ) exp(-i psrc4.y4)     
    
    //As pipi total momentum = 0,    psrc1+psrc2 = -psrc3-psrc4
    
    //We have pipi_bubble(t, p) = 0.5 * tr( M_pi1(p, t) M_pi2(-p, t-delta) )  which is created by ComputePiPiGparity::computeFigureVdis
    //Thus we compute
    //-sqrt(6)/2 \tr( M_sigma(tsnk,tsnk) ) B(tsrc)
        
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt,Lt); into.zero();
    double coeff = -sqrt(6.)/2;
#ifdef USE_TIANLES_CONVENTIONS
    coeff *= -1.;
#endif
    
    for(int tsrc=0; tsrc<Lt; tsrc+=tstep_src){
      for(int tsep=0; tsep<Lt; tsep++){
	int tsnk = (tsrc + tsep) % Lt;
  	into(tsrc, tsep) = coeff * sigma_bubble(tsnk) * pipi_bubble(tsrc);
      }
    }
  }
};














void computeSigma2ptTianle(std::vector< fVector<typename A2Apolicies::ScalarComplexType> > &sigma_bub, //output bubble
			   MesonFieldMomentumPairContainer<A2Apolicies> &mf_sigma_con, const StationarySigmaMomentaPolicy &sigma_mom, const int conf, const Parameters &params){
  const int nmom = sigma_mom.nMom();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();

  //All momentum combinations have total momentum 0 at source and sink
  if(!UniqueID()) printf("Computing sigma 2pt function\n");
  double time = -dclock();

  sigma_bub.resize(nmom);
  for(int pidx=0;pidx<nmom;pidx++){
    //Compute the disconnected bubble
    if(!UniqueID()) printf("Sigma disconnected bubble pidx=%d\n",pidx);
    fVector<typename A2Apolicies::ScalarComplexType> &into = sigma_bub[pidx]; into.resize(Lt);
    ComputeSigmaContractions<A2Apolicies>::computeDisconnectedBubble(into, mf_sigma_con, sigma_mom, pidx);

    std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf;
    os  << "_sigmaself_mom" << sigma_mom.getWmom(pidx).file_str(1) << "_v2"; //note Vmom == -WdagMom = Wmom for sigma as momentum 0
    into.write(os.str());
# ifdef WRITE_HEX_OUTPUT
    os << ".hexfloat";
    into.write(os.str(),true);
# endif
  }

  for(int psnkidx=0;psnkidx<nmom;psnkidx++){
    for(int psrcidx=0;psrcidx<nmom;psrcidx++){
      if(!UniqueID()) printf("Sigma connected psrcidx=%d psnkidx=%d\n",psrcidx,psnkidx);
      fMatrix<typename A2Apolicies::ScalarComplexType> into(Lt,Lt);
      ComputeSigmaContractions<A2Apolicies>::computeConnected(into, mf_sigma_con, sigma_mom, psrcidx, psnkidx);

      fMatrix<typename A2Apolicies::ScalarComplexType> disconn(Lt,Lt);
      ComputeSigmaContractions<A2Apolicies>::computeDisconnectedDiagram(disconn, sigma_bub[psnkidx], sigma_bub[psrcidx]);

      into += disconn;

      std::ostringstream os; //traj_0_sigmacorr_mompsrc_1_1_1psnk_1_1_1_v2
      os
	<< params.meas_arg.WorkDirectory << "/traj_" << conf << "_sigmacorr_mom"
	<< "psrc" << sigma_mom.getWmom(psrcidx).file_str() << "psnk" << sigma_mom.getWmom(psnkidx).file_str() << "_v2";

      into.write(os.str());
# ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      into.write(os.str(),true);
# endif
    }
  }
        
  time += dclock();
  print_time("main","Sigma 2pt function",time);

  printMem("Memory after Sigma 2pt function computation");
}





void computeSigma2pt(MesonFieldMomentumPairContainer<A2Apolicies> &mf_sigma_con, const StationarySigmaMomentaPolicy &sigma_mom, const int conf, const Parameters &params){
  const int nmom = sigma_mom.nMom();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();

  //All momentum combinations have total momentum 0 at source and sink
  if(!UniqueID()) printf("Computing sigma 2pt function\n");
  double time = -dclock();

  for(int psnkidx=0;psnkidx<nmom;psnkidx++){
    {
      //Compute the disconnected bubble
      if(!UniqueID()) printf("Sigma disconnected bubble pidx=%d\n",psnkidx);
      fVector<typename A2Apolicies::ScalarComplexType> into(Lt);
      ComputeSigmaContractions<A2Apolicies>::computeDisconnectedBubble(into, mf_sigma_con, sigma_mom, psnkidx);

      std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf;
      os  << "_sigmabubble_mom" << sigma_mom.getWdagMom(psnkidx).file_str(1) << "_plus" << sigma_mom.getVmom(psnkidx).file_str(1);    
      into.write(os.str());
# ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      into.write(os.str(),true);
# endif
    }
    
    for(int psrcidx=0;psrcidx<nmom;psrcidx++){
      if(!UniqueID()) printf("Sigma connected psrcidx=%d psnkidx=%d\n",psrcidx,psnkidx);
      fMatrix<typename A2Apolicies::ScalarComplexType> into(Lt,Lt);
      ComputeSigmaContractions<A2Apolicies>::computeConnected(into, mf_sigma_con, sigma_mom, psrcidx, psnkidx);
      
      std::ostringstream os;
      os
	<< params.meas_arg.WorkDirectory << "/traj_" << conf << "_sigmaconnected"
	<< "_momsnk" << sigma_mom.getWdagMom(psnkidx).file_str(1) << "_plus" << sigma_mom.getVmom(psnkidx).file_str(1)
	<< "_momsrc" << sigma_mom.getWdagMom(psrcidx).file_str(1) << "_plus" << sigma_mom.getVmom(psrcidx).file_str(1);

      into.write(os.str());
# ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      into.write(os.str(),true);
# endif
    }
  }

  time += dclock();
  print_time("main","Sigma 2pt function",time);

  printMem("Memory after Sigma 2pt function computation");
}









void computePiPiToSigmaTianle(const std::vector< fVector<typename A2Apolicies::ScalarComplexType> > &sigma_bub,
			      MesonFieldMomentumPairContainer<A2Apolicies> &mf_sigma_con, const StationarySigmaMomentaPolicy &sigma_mom, 
			      MesonFieldMomentumContainer<A2Apolicies> &mf_pion_con, const StandardPionMomentaPolicy &pion_mom,
			      const int conf, const Parameters &params){
  const int nmom_sigma = sigma_mom.nMom();
  const int nmom_pi = pion_mom.nMom();
  
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();

  std::vector<fVector<typename A2Apolicies::ScalarComplexType> > pipi_bub(nmom_pi);
  for(int pidx=0;pidx<nmom_pi;pidx++){
    ComputePiPiGparity<A2Apolicies>::computeFigureVdis(pipi_bub[pidx], pion_mom.getMesonMomentum(pidx), params.jp.pipi_separation, mf_pion_con);
  }
  
  //All momentum combinations have total momentum 0 at source and sink
  if(!UniqueID()) printf("Computing Pipi->sigma\n");
  double time = -dclock();
  for(int ppi1_idx=0;ppi1_idx<nmom_pi;ppi1_idx++){
    for(int psigma_idx=0;psigma_idx<nmom_sigma;psigma_idx++){
      if(!UniqueID()) printf("Pipi->sigma connected psigma_idx=%d ppi1_idx=%d\n",psigma_idx,ppi1_idx);
      fMatrix<typename A2Apolicies::ScalarComplexType> into(Lt,Lt);

      ComputePiPiToSigmaContractions<A2Apolicies>::computeConnected(into,mf_sigma_con,sigma_mom,psigma_idx,
								    mf_pion_con,pion_mom,ppi1_idx,
								    params.jp.pipi_separation, params.jp.tstep_pipi); //reuse same tstep currently

      //Tianle also computes the disconnected part
      fMatrix<typename A2Apolicies::ScalarComplexType> disconn(Lt,Lt);
      ComputePiPiToSigmaContractions<A2Apolicies>::computeDisconnectedDiagram(disconn, sigma_bub[psigma_idx], pipi_bub[ppi1_idx], params.jp.tstep_pipi);

      into += disconn;
      
      std::ostringstream os;
      os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_pipitosigma_sigmawdagmom";
      os << sigma_mom.getWmom(psigma_idx).file_str() << "_pionmom" << (-pion_mom.getMesonMomentum(ppi1_idx)).file_str() << "_v2";
      
      into.write(os.str());
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      into.write(os.str(),true);
#endif
    }
  }
  time += dclock();
  print_time("main","Pipi->sigma function",time);

  printMem("Memory after Pipi->sigma function computation");
}

  
void doConfiguration(const int conf, Parameters &params, const CommandLineArgs &cmdline,
		     const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
		     const typename A2Apolicies::FermionFieldType::InputParamType &field4dparams, BFMGridSolverWrapper &solvers){

  params.meas_arg.TrajCur = conf;

  std::string dir(params.meas_arg.WorkDirectory);

  //-------------------- Read gauge field --------------------//
  readGaugeRNG(params,cmdline);
    
  printMem("Memory after gauge and RNG read");

  runInitialGridBenchmarks(cmdline,params);
  
  if(cmdline.tune_lanczos) LanczosTune(params, solvers);    

  bool need_lanczos = true;
  if(cmdline.load_pion_mf && cmdline.load_sigma_mf) need_lanczos = false;
  if(cmdline.randomize_vw) need_lanczos = false;
  if(cmdline.force_evec_compute) need_lanczos = true;

  bool need_vw = true;
  if(cmdline.load_pion_mf && cmdline.load_sigma_mf) need_vw = false;
  
  //-------------------- Light quark Lanczos ---------------------//
  BFMGridLanczosWrapper<A2Apolicies> eig(solvers, params.jp);
  if(need_lanczos) computeEvecs(eig, params.lanc_arg, params.jp, "light", cmdline.randomize_evecs);
  
  //-------------------- Light quark v and w --------------------//
  A2AvectorV<A2Apolicies> V(params.a2a_arg, field4dparams);
  A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
  
  BFMGridA2ALatticeWrapper<A2Apolicies> latwrp(solvers, params.jp); //lattice created temporarily
  if(need_vw) computeVW(V, W, params, eig, latwrp, cmdline.randomize_vw);

  if(!UniqueID()){ printf("Freeing light evecs\n"); fflush(stdout); }
  eig.freeEvecs();
  printMem("Memory after light evec free");
    
  //From now one we just need a generic lattice instance, so use a2a_lat
  Lattice& lat = (Lattice&)(*latwrp.a2a_lat);
    
  //-------------------Fix gauge----------------------------
  doGaugeFix(lat, cmdline.skip_gauge_fix, params);

  //----------------------------Compute the sigma meson fields--------------------------------
  StationarySigmaMomentaPolicy sigma_mom;
  
  MesonFieldMomentumPairContainer<A2Apolicies> mf_sigma_con_1s;
  MesonFieldMomentumPairContainer<A2Apolicies> mf_sigma_con_2s;
  if(cmdline.load_sigma_mf)
    readSigmaMesonFields(mf_sigma_con_1s, mf_sigma_con_2s, sigma_mom, conf, params, cmdline.load_sigma_mf_dir);
  else
    computeSigmaMesonFields(mf_sigma_con_1s, mf_sigma_con_2s, V,W, sigma_mom, conf,lat,params,field3dparams);

  //----------------------------Compute the pion meson fields---------------------------------
  StandardPionMomentaPolicy pion_mom; //these are the W and V momentum combinations

  MesonFieldMomentumContainer<A2Apolicies> mf_pion_con_1s; //stores light-light meson fields, accessible by momentum
  MesonFieldMomentumContainer<A2Apolicies> mf_pion_con_2s; //Gparity only
  if(cmdline.load_pion_mf)
    readPionMesonFields(mf_pion_con_1s, mf_pion_con_2s, pion_mom, conf, params, cmdline.load_pion_mf_dir);
  else   
    computePionMesonFields(mf_pion_con_1s, mf_pion_con_2s, V, W, pion_mom, conf, lat, params, field3dparams);


  
  //1s-1s
#ifdef USE_TIANLES_CONVENTIONS
  std::vector< fVector<typename A2Apolicies::ScalarComplexType> > sigma_bub;
  computeSigma2ptTianle(sigma_bub, mf_sigma_con_1s, sigma_mom, conf, params);
  computePiPiToSigmaTianle(sigma_bub,mf_sigma_con_1s, sigma_mom, 
			  mf_pion_con_1s, pion_mom,
			  conf, params);
#else
  computeSigma2pt(mf_sigma_con_1s, sigma_mom, conf, params);
  /* computeSigmaToPipi(mf_sigma_con_1s, sigma_mom,  */
  /* 		     mf_pion_con_1s, pion_mom, */
  /* 		     conf, params); */
#endif
  

}

#endif
