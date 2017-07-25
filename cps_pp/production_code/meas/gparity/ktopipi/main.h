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

//Defines for Grid/BFM wrapping
#ifdef USE_GRID_LANCZOS
  typedef GridLanczosWrapper<A2Apolicies> LanczosWrapper;
  typedef typename A2Apolicies::FgridGFclass LanczosLattice;
# define LANCZOS_LATARGS params.jp
# define COMPUTE_EVECS_LANCZOS_LATARGS jp
# define LANCZOS_LATMARK isGridtype
# define LANCZOS_EXTRA_ARG *lanczos_lat
# define COMPUTE_EVECS_EXTRA_ARG_PASS NULL
# define COMPUTE_EVECS_EXTRA_ARG_GRAB void*
#else //USE_BFM_LANCZOS
  typedef BFMLanczosWrapper LanczosWrapper;
  typedef GwilsonFdwf LanczosLattice;
# define LANCZOS_LATARGS bfm_solvers
# define COMPUTE_EVECS_LANCZOS_LATARGS bfm_solvers
# define LANCZOS_LATMARK isBFMtype
# define LANCZOS_EXTRA_ARG bfm_solvers
# define COMPUTE_EVECS_EXTRA_ARG_PASS bfm_solvers
# define COMPUTE_EVECS_EXTRA_ARG_GRAB BFMsolvers &bfm_solvers
#endif

#ifdef USE_GRID_A2A
  typedef A2Apolicies::FgridGFclass A2ALattice;
# define A2A_LATARGS params.jp
# define A2A_LATMARK isGridtype
#else
  typedef GwilsonFdwf A2ALattice;
# define A2A_LATARGS bfm_solvers
# define A2A_LATMARK isBFMtype
#endif

#ifdef USE_GRID
//Returns per-node performance in Mflops of double/double Dirac op
template<typename GridPolicies>
double gridBenchmark(Lattice &lat){
  typedef typename GridPolicies::GridFermionField GridFermionField;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDirac GridDirac;
  typedef typename GridDirac::GaugeField GridGaugeField;
  
  FgridFclass & lgrid = dynamic_cast<FgridFclass &>(lat);
  
  Grid::GridCartesian *FGrid = lgrid.getFGrid();
  Grid::GridCartesian *UGrid = lgrid.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lgrid.getUrbGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lgrid.getFrbGrid();

  GridGaugeField & Umu = *lgrid.getUmu();    

  //Setup Dirac operator
  const double mass = 0.1;
  const double mob_b = lgrid.get_mob_b();
  const double mob_c = mob_b - 1.;   //b-c = 1
  const double M5 = GJP.DwfHeight();
  
  typename GridDirac::ImplParams params;
  lgrid.SetParams(params);
    
  GridDirac Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c, params);

  //Run benchmark
  std::cout << "gridBenchmark: Running Grid Dhop benchmark\n";
  std::cout << "Dirac operator type is : " << printType<GridDirac>() << std::endl;
  
  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  
  std::cout << Grid::GridLogMessage << "Initialising 4d RNG" << std::endl;
  Grid::GridParallelRNG          RNG4(UGrid);  RNG4.SeedFixedIntegers(seeds4);
  std::cout << Grid::GridLogMessage << "Initialising 5d RNG" << std::endl;
  Grid::GridParallelRNG          RNG5(FGrid);  RNG5.SeedFixedIntegers(seeds5);
  std::cout << Grid::GridLogMessage << "Initialised RNGs" << std::endl;
  
  GridFermionField src   (FGrid); random(RNG5,src);
  Grid::RealD N2 = 1.0/::sqrt(norm2(src));
  src = src*N2;
  
  GridFermionField result(FGrid); result=Grid::zero;
  GridFermionField    ref(FGrid);    ref=Grid::zero;
  GridFermionField    tmp(FGrid);
  GridFermionField    err(FGrid);
  
  Grid::RealD NP = UGrid->_Nprocessors;
  Grid::RealD NN = UGrid->NodeCount();
  
  std::cout << Grid::GridLogMessage<< "* Vectorising space-time by "<<Grid::vComplexF::Nsimd()<<std::endl;
#ifdef GRID_OMP
  if ( Grid::QCD::WilsonKernelsStatic::Comms == Grid::QCD::WilsonKernelsStatic::CommsAndCompute ) std::cout << Grid::GridLogMessage<< "* Using Overlapped Comms/Compute" <<std::endl;
  if ( Grid::QCD::WilsonKernelsStatic::Comms == Grid::QCD::WilsonKernelsStatic::CommsThenCompute) std::cout << Grid::GridLogMessage<< "* Using sequential comms compute" <<std::endl;
#endif
  if ( Grid::QCD::WilsonKernelsStatic::Opt == Grid::QCD::WilsonKernelsStatic::OptGeneric   ) std::cout << Grid::GridLogMessage<< "* Using GENERIC Nc WilsonKernels" <<std::endl;
  if ( Grid::QCD::WilsonKernelsStatic::Opt == Grid::QCD::WilsonKernelsStatic::OptHandUnroll) std::cout << Grid::GridLogMessage<< "* Using Nc=3       WilsonKernels" <<std::endl;
  if ( Grid::QCD::WilsonKernelsStatic::Opt == Grid::QCD::WilsonKernelsStatic::OptInlineAsm ) std::cout << Grid::GridLogMessage<< "* Using Asm Nc=3   WilsonKernels" <<std::endl;
  std::cout << Grid::GridLogMessage<< "*****************************************************************" <<std::endl;
  
  int ncall =1000;

  FGrid->Barrier();
  Ddwf.ZeroCounters();
  Ddwf.Dhop(src,result,0);
  std::cout<<Grid::GridLogMessage<<"Called warmup"<<std::endl;
  double t0=Grid::usecond();
  for(int i=0;i<ncall;i++){
    Ddwf.Dhop(src,result,0);
  }
  double t1=Grid::usecond();
  FGrid->Barrier();

  double volume=GJP.Snodes()*GJP.SnodeSites();  for(int mu=0;mu<4;mu++) volume=volume*GJP.NodeSites(mu)*GJP.Nodes(mu);
  double flops=(GJP.Gparity() + 1)*1344*volume*ncall;

  std::cout<<Grid::GridLogMessage << "Called Dw "<<ncall<<" times in "<<t1-t0<<" us"<<std::endl;
  std::cout<<Grid::GridLogMessage << "mflop/s =   "<< flops/(t1-t0)<<std::endl;
  std::cout<<Grid::GridLogMessage << "mflop/s per rank =  "<< flops/(t1-t0)/NP<<std::endl;
  std::cout<<Grid::GridLogMessage << "mflop/s per node =  "<< flops/(t1-t0)/NN<<std::endl;
  Ddwf.Report();
  return flops/(t1-t0)/NN; //node performance in Mflops
}
#endif




//Command line argument store/parse
struct CommandLineArgs{
  int nthreads;
  bool randomize_vw; //rather than doing the Lanczos and inverting the propagators, etc, just use random vectors for V and W
  bool randomize_evecs; //skip Lanczos and just use random evecs for testing.
  bool force_evec_compute; //randomize_evecs causes Lanczos to be skipped unless this option is used
  bool tune_lanczos_light; //just run the light lanczos on first config then exit
  bool tune_lanczos_heavy; //just run the heavy lanczos on first config then exit
  bool skip_gauge_fix;
  bool double_latt; //most ancient 8^4 quenched lattices stored both U and U*. Enable this to read those configs
  bool mixed_solve; //do high mode inversions using mixed precision solves. Is disabled if we turn off the single-precision conversion of eigenvectors (because internal single-prec inversion needs singleprec eigenvectors)
  bool evecs_single_prec; //convert the eigenvectors to single precision to save memory
  bool do_kaon2pt;
  bool do_pion2pt;
  bool do_pipi;
  bool do_ktopipi;
  bool do_sigma;

  Float inner_cg_resid;
  Float *inner_cg_resid_p;

  bool do_split_job;
  int split_job_part;
  std::string checkpoint_dir; //directory the checkpointed data is stored in (could be scratch for example)

#ifdef BNL_KNL_PERFORMANCE_CHECK
  double bnl_knl_minperf; //in Mflops per node (not per rank, eg MPI3!)
#endif
  
  CommandLineArgs(int argc, char **argv, int begin){
    nthreads = 1;
#if TARGET == BGQ
    nthreads = 64;
#endif
    randomize_vw = false;
    randomize_evecs = false;
    force_evec_compute = false; //randomize_evecs causes Lanczos to be skipped unless this option is used
    tune_lanczos_light = false; //just run the light lanczos on first config then exit
    tune_lanczos_heavy = false; //just run the heavy lanczos on first config then exit
    skip_gauge_fix = false;
    double_latt = false; //most ancient 8^4 quenched lattices stored both U and U*. Enable this to read those configs
    mixed_solve = true; //do high mode inversions using mixed precision solves. Is disabled if we turn off the single-precision conversion of eigenvectors (because internal single-prec inversion needs singleprec eigenvectors)
    evecs_single_prec = true; //convert the eigenvectors to single precision to save memory
    do_kaon2pt = true;
    do_pion2pt = true;
    do_pipi = true;
    do_ktopipi = true;
    do_sigma = true;

    inner_cg_resid;
    inner_cg_resid_p = NULL;

    do_split_job = false;

#ifdef BNL_KNL_PERFORMANCE_CHECK
    bnl_knl_minperf = 50000;
#endif
    parse(argc,argv,begin);
  }

  void parse(int argc, char **argv, int begin){
    if(!UniqueID()){ printf("Arguments:\n"); fflush(stdout); }
    for(int i=0;i<argc;i++){
      if(!UniqueID()){ printf("%d \"%s\"\n",i,argv[i]); fflush(stdout); }
    }
    
    const int ngrid_arg = 10;
    const std::string grid_args[ngrid_arg] = { "--debug-signals", "--dslash-generic", "--dslash-unroll", "--dslash-asm", "--shm", "--lebesgue", "--cacheblocking", "--comms-concurrent", "--comms-sequential", "--comms-overlap" };
    const int grid_args_skip[ngrid_arg] =    { 1                , 1                 , 1                , 1             , 2      , 1           , 2                , 1              , 1                 , 1 };

    int arg = begin;
    while(arg < argc){
      char* cmd = argv[arg];
      if( strncmp(cmd,"-nthread",8) == 0){
	if(arg == argc-1){ if(!UniqueID()){ printf("-nthread must be followed by a number!\n"); fflush(stdout); } exit(-1); }
	nthreads = strToAny<int>(argv[arg+1]);
	if(!UniqueID()){ printf("Setting number of threads to %d\n",nthreads); }
	arg+=2;
      }else if( strncmp(cmd,"-set_inner_resid",16) == 0){ //only for mixed CG
	if(arg == argc-1){ if(!UniqueID()){ printf("-set_inner_resid must be followed by a number!\n"); fflush(stdout); } exit(-1); }
	inner_cg_resid = strToAny<Float>(argv[arg+1]);
	inner_cg_resid_p = &inner_cg_resid;
	if(!UniqueID()){ printf("Setting inner CG initial residual to %g\n",inner_cg_resid); }
#ifdef USE_BFM_A2A
	ERR.General("","main","Changing initial inner CG residual not implemented for BFM version\n");
#endif      
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
      }else if( strncmp(cmd,"-tune_lanczos_light",15) == 0){
	tune_lanczos_light = true;
	if(!UniqueID()){ printf("Just tuning light lanczos on first config\n"); fflush(stdout); }
	arg++;
      }else if( strncmp(cmd,"-tune_lanczos_heavy",15) == 0){
	tune_lanczos_heavy = true;
	if(!UniqueID()){ printf("Just tuning heavy lanczos on first config\n"); fflush(stdout); }
	arg++;
      }else if( strncmp(cmd,"-double_latt",15) == 0){
	double_latt = true;
	if(!UniqueID()){ printf("Loading doubled lattices\n"); fflush(stdout); }
	arg++;
      }else if( strncmp(cmd,"-skip_gauge_fix",20) == 0){
	skip_gauge_fix = true;
	if(!UniqueID()){ printf("Skipping gauge fixing\n"); fflush(stdout); }
	arg++;
      }else if( strncmp(cmd,"-disable_evec_singleprec_convert",30) == 0){
	evecs_single_prec = false;
	mixed_solve = false;
	if(!UniqueID()){ printf("Disabling single precision conversion of evecs\n"); fflush(stdout); }
	arg++;
      }else if( strncmp(cmd,"-disable_mixed_prec_CG",30) == 0){
	mixed_solve = false;
	if(!UniqueID()){ printf("Disabling mixed-precision CG\n"); fflush(stdout); }
	arg++;
      }else if( strncmp(cmd,"-mf_outerblocking",15) == 0){
	int* b[3] = { &BlockedMesonFieldArgs::bi, &BlockedMesonFieldArgs::bj, &BlockedMesonFieldArgs::bp };
	for(int a=0;a<3;a++) *b[a] = strToAny<int>(argv[arg+1+a]);
	arg+=4;
      }else if( strncmp(cmd,"-mf_innerblocking",15) == 0){
	int* b[3] = { &BlockedMesonFieldArgs::bii, &BlockedMesonFieldArgs::bjj, &BlockedMesonFieldArgs::bpp };
	for(int a=0;a<3;a++) *b[a] = strToAny<int>(argv[arg+1+a]);
	arg+=4;
      }else if( strncmp(cmd,"-do_split_job",30) == 0){
	do_split_job = true;
	split_job_part = strToAny<int>(argv[arg+1]);
	checkpoint_dir = argv[arg+2];
	if(!UniqueID()) printf("Doing split job part %d with checkpoint directory %s\n",split_job_part,checkpoint_dir.c_str());
	arg+=3;       
      }else if( strncmp(cmd,"-skip_kaon2pt",30) == 0){
	do_kaon2pt = false;
	arg++;
      }else if( strncmp(cmd,"-skip_pion2pt",30) == 0){
	do_pion2pt = false;
	arg++;
      }else if( strncmp(cmd,"-skip_sigma",30) == 0){
	do_sigma = false;
	arg++;
      }else if( strncmp(cmd,"-skip_pipi",30) == 0){
	do_pipi = false;
	arg++;
      }else if( strncmp(cmd,"-skip_ktopipi",30) == 0){
	do_ktopipi = false;
	arg++;
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
  A2AArg a2a_arg_s;
  LancArg lanc_arg;
  LancArg lanc_arg_s;

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
    if(!a2a_arg_s.Decode("a2a_arg_s.vml","a2a_arg_s")){
      a2a_arg_s.Encode("a2a_arg_s.templ","a2a_arg_s");
      VRB.Result("Parameters","Parameters","Can't open a2a_arg_s.vml!\n");exit(1);
    }
    if(!lanc_arg.Decode("lanc_arg.vml","lanc_arg")){
      lanc_arg.Encode("lanc_arg.templ","lanc_arg");
      VRB.Result("Parameters","Parameters","Can't open lanc_arg.vml!\n");exit(1);
    }
    if(!lanc_arg_s.Decode("lanc_arg_s.vml","lanc_arg_s")){
      lanc_arg_s.Encode("lanc_arg_s.templ","lanc_arg_s");
      VRB.Result("Parameters","Parameters","Can't open lanc_arg_s.vml!\n");exit(1);
    }
    if(!fix_gauge_arg.Decode("fix_gauge_arg.vml","fix_gauge_arg")){
      fix_gauge_arg.Encode("fix_gauge_arg.templ","fix_gauge_arg");
      VRB.Result("Parameters","Parameters","Can't open fix_gauge_arg.vml!\n");exit(1);
    }

    common_arg.set_filename(meas_arg.WorkDirectory);
  }

};


void setupJob(int argc, char **argv, const Parameters &params, const CommandLineArgs &cmdline){
#ifdef NODE_DISTRIBUTE_MESONFIELDS
  if(!UniqueID()) printf("Using node distribution of meson fields\n");
#endif
#ifdef MEMTEST_MODE
  if(!UniqueID()) printf("Running in MEMTEST MODE (so don't expect useful results)\n");
#endif
  
#ifdef A2A_LANCZOS_SINGLE
  if(!cmdline.evecs_single_prec) ERR.General("",fname,"Must use single-prec eigenvectors when doing Lanczos in single precision\n");
#endif

  GJP.Initialize(params.do_arg);
  LRG.Initialize();

#if defined(USE_GRID_A2A) || defined(USE_GRID_LANCZOS)
  if(GJP.Gparity()){
#ifndef USE_GRID_GPARITY
    ERR.General("","","Must compile main program with flag USE_GRID_GPARITY to enable G-parity\n");
#endif
  }else{
#ifdef USE_GRID_GPARITY
    ERR.General("","","Must compile main program with flag USE_GRID_GPARITY off to disable G-parity\n");
#endif
  }      
#endif
  
  if(cmdline.double_latt) SerialIO::dbl_latt_storemode = true;

  if(!cmdline.tune_lanczos_light && !cmdline.tune_lanczos_heavy){ 
    assert(params.a2a_arg.nl <= params.lanc_arg.N_true_get);
    assert(params.a2a_arg_s.nl <= params.lanc_arg_s.N_true_get);
  }
#ifdef USE_BFM
  cps_qdp_init(&argc,&argv);
  //Chroma::initialize(&argc,&argv);
#endif
  omp_set_num_threads(cmdline.nthreads);

  if(!UniqueID()) printf("Initial memory post-initialize:\n");
  printMem();
}

#ifdef BNL_KNL_PERFORMANCE_CHECK
void bnl_knl_performance_check(const CommandLineArgs &args,const Parameters &params){
  A2ALattice* lat = createLattice<A2ALattice,A2A_LATMARK>::doit(A2A_LATARGS);
  lat->SetGfieldOrd(); //so we don't interfere with the RNG state
  double node_perf = gridBenchmark<A2Apolicies>(*lat);  
  delete lat;
  if(node_perf < args.bnl_knl_minperf && !UniqueID()){ printf("BAD PERFORMANCE\n"); fflush(stdout); exit(-1); }
}
#endif


//Tune the Lanczos and exit
void LanczosTune(bool tune_lanczos_light, bool tune_lanczos_heavy, const Parameters &params, COMPUTE_EVECS_EXTRA_ARG_GRAB){
  LanczosLattice* lanczos_lat = createLattice<LanczosLattice,LANCZOS_LATMARK>::doit(LANCZOS_LATARGS);

  if(tune_lanczos_light){
    LanczosWrapper eig;
    if(!UniqueID()) printf("Tuning lanczos light with mass %f\n", params.lanc_arg.mass);

    double time = -dclock();
    eig.compute(params.lanc_arg, LANCZOS_EXTRA_ARG);
    time += dclock();
    print_time("main","Lanczos light",time);
  }
  if(tune_lanczos_heavy){
    LanczosWrapper eig;
    if(!UniqueID()) printf("Tuning lanczos heavy with mass %f\n", params.lanc_arg_s.mass);

    double time = -dclock();
    eig.compute(params.lanc_arg_s, LANCZOS_EXTRA_ARG);
    time += dclock();
    print_time("main","Lanczos heavy",time);
  }
  
  delete lanczos_lat;
  exit(0);
}

enum LightHeavy { Light, Heavy };

void computeEvecs(LanczosWrapper &eig, const LancArg &lanc_arg, const JobParams &jp, const char* name, const bool evecs_single_prec, const bool randomize_evecs, COMPUTE_EVECS_EXTRA_ARG_GRAB){
  if(!UniqueID()) printf("Running %s quark Lanczos\n",name);
  LanczosLattice* lanczos_lat = createLattice<LanczosLattice,LANCZOS_LATMARK>::doit(COMPUTE_EVECS_LANCZOS_LATARGS);
  double time = -dclock();
  if(randomize_evecs) eig.randomizeEvecs(lanc_arg, LANCZOS_EXTRA_ARG);
  else eig.compute(lanc_arg, LANCZOS_EXTRA_ARG);
  time += dclock();

  std::ostringstream os; os << name << " quark Lanczos";
      
  print_time("main",os.str().c_str(),time);

  if(!UniqueID()) printf("Memory after %s quark Lanczos:\n",name);
  printMem();      

#ifndef A2A_LANCZOS_SINGLE
  if(evecs_single_prec){ 
    eig.toSingle();
    if(!UniqueID()) printf("Memory after single-prec conversion of %s quark evecs:\n",name);
    printMem();
  }
#endif
#ifdef USE_BFM_LANCZOS
  eig.checkEvecMemGuards();
#endif
  delete lanczos_lat;
}
void computeEvecs(LanczosWrapper &eig, const LightHeavy lh, const Parameters &params, const bool evecs_single_prec, const bool randomize_evecs, COMPUTE_EVECS_EXTRA_ARG_GRAB){
  const char* name = (lh ==  Light ? "light" : "heavy");
  const LancArg &lanc_arg = (lh == Light ? params.lanc_arg : params.lanc_arg_s);
  return computeEvecs(eig, lanc_arg, params.jp, name, evecs_single_prec, randomize_evecs, COMPUTE_EVECS_EXTRA_ARG_PASS);
}

A2ALattice* computeVW(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, const LightHeavy lh, const Parameters &params, const LanczosWrapper &eig,
		      const bool evecs_single_prec, const bool randomize_vw, const bool mixed_solve, Float const* inner_cg_resid_p, const bool delete_lattice, COMPUTE_EVECS_EXTRA_ARG_GRAB){
  const A2AArg &a2a_arg = lh == Light ? params.a2a_arg : params.a2a_arg_s;  
  const char* name = (lh ==  Light ? "light" : "heavy");
  A2ALattice* a2a_lat = createLattice<A2ALattice,A2A_LATMARK>::doit(A2A_LATARGS); //the lattice class used to perform the CG and whatnot
  
  if(!UniqueID()) printf("Computing %s quark A2A vectors\n",name);
  double time = -dclock();

#ifdef USE_DESTRUCTIVE_FFT
  V.allocModes(); W.allocModes();
#endif

  typedef typename A2Apolicies::FermionFieldType::InputParamType Field4DparamType;
  Field4DparamType field4dparams = V.getFieldInputParams();
  
  if(!UniqueID()){ printf("V vector requires %f MB, W vector %f MB of memory\n", 
			  A2AvectorV<A2Apolicies>::Mbyte_size(a2a_arg,field4dparams), A2AvectorW<A2Apolicies>::Mbyte_size(a2a_arg,field4dparams) );
    fflush(stdout);
  }
    
  if(!randomize_vw){
#ifdef USE_BFM_LANCZOS
    W.computeVW(V, *a2a_lat, *eig.eig, evecs_single_prec, bfm_solvers.dwf_d, mixed_solve ? & bfm_solvers.dwf_f : NULL);
#else
    if(evecs_single_prec){
      W.computeVW(V, *a2a_lat, eig.evec_f, eig.eval, eig.mass, eig.resid, 10000, inner_cg_resid_p);
    }else{
      W.computeVW(V, *a2a_lat, eig.evec, eig.eval, eig.mass, eig.resid, 10000);
    }
#endif     
  }else randomizeVW<A2Apolicies>(V,W);    

  if(!UniqueID()) printf("Memory after %s A2A vector computation:\n", name);
  printMem();

  time += dclock();
  std::ostringstream os; os << name << " quark A2A vectors";
  print_time("main",os.str().c_str(),time);
  
  if(delete_lattice){
    delete a2a_lat;
    return NULL;
  }else return a2a_lat;
}


void doGaugeFix(Lattice &lat, const bool skip_gauge_fix, const Parameters &params){
  AlgFixGauge fix_gauge(lat, const_cast<CommonArg *>(&params.common_arg), const_cast<FixGaugeArg *>(&params.fix_gauge_arg) );
  if( (lat.FixGaugeKind() != FIX_GAUGE_NONE) || (lat.FixGaugePtr() != NULL) )
    lat.FixGaugeFree(); //in case it has previously been allocated
  if(skip_gauge_fix){
    if(!UniqueID()) printf("Skipping gauge fix -> Setting all GF matrices to unity\n");
    gaugeFixUnity(lat,params.fix_gauge_arg);      
  }else{
    if(!UniqueID()){ printf("Gauge fixing\n"); fflush(stdout); }
    double time = -dclock();
#ifndef MEMTEST_MODE
    fix_gauge.run();
#else
    gaugeFixUnity(lat,params.fix_gauge_arg);
#endif      
    time += dclock();
    print_time("main","Gauge fix",time);
  }

  if(!UniqueID()) printf("Memory after gauge fix:\n");
  printMem();
}


void computeKaon2pt(typename ComputeKaon<A2Apolicies>::Vtype &V, typename ComputeKaon<A2Apolicies>::Wtype &W, 
		    typename ComputeKaon<A2Apolicies>::Vtype &V_s, typename ComputeKaon<A2Apolicies>::Wtype &W_s,
		 const int conf, Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  if(!UniqueID()) printf("Computing kaon 2pt function\n");
  double time = -dclock();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  fMatrix<typename A2Apolicies::ScalarComplexType> kaon(Lt,Lt);
  ComputeKaon<A2Apolicies>::compute(kaon,
				    W, V, W_s, V_s,
				    params.jp.kaon_rad, lat, field3dparams);
  std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_kaoncorr";
  kaon.write(os.str());
#ifdef WRITE_HEX_OUTPUT
  os << ".hexfloat";
  kaon.write(os.str(),true);
#endif
  time += dclock();
  print_time("main","Kaon 2pt function",time);

  if(!UniqueID()) printf("Memory after kaon 2pt function computation:\n");
  printMem();
}

void computeLLmesonFields(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con_2s,
			  typename ComputePion<A2Apolicies>::Vtype &V, typename ComputePion<A2Apolicies>::Wtype &W,
			  const StandardPionMomentaPolicy &pion_mom,
			  const int conf, Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  if(!UniqueID()) printf("Computing light-light meson fields\n");
  double time = -dclock();
  if(!GJP.Gparity()) ComputePion<A2Apolicies>::computeMesonFields(mf_ll_con, params.meas_arg.WorkDirectory,conf, pion_mom, W, V, params.jp.pion_rad, lat, field3dparams);
  else ComputePion<A2Apolicies>::computeGparityMesonFields(mf_ll_con, mf_ll_con_2s, params.meas_arg.WorkDirectory,conf, pion_mom, W, V, params.jp.pion_rad, lat, field3dparams);
  time += dclock();
  print_time("main","Light-light meson fields",time);

  if(!UniqueID()) printf("Memory after light-light meson field computation:\n");
  printMem();
}

void computePion2pt(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, const StandardPionMomentaPolicy &pion_mom, const int conf, const Parameters &params){
  const int nmom = pion_mom.nMom();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  
  if(!UniqueID()) printf("Computing pion 2pt function\n");
  double time = -dclock();
  for(int p=0;p<nmom;p+=2){ //note odd indices 1,3,5 etc have equal and opposite momenta to 0,2,4... 
    if(!UniqueID()) printf("Starting pidx %d\n",p);
    fMatrix<typename A2Apolicies::ScalarComplexType> pion(Lt,Lt);
    ComputePion<A2Apolicies>::compute(pion, mf_ll_con, pion_mom, p);
    //Note it seems Daiqian's pion momenta are opposite what they should be for 'conventional' Fourier transform phase conventions:
    //f'(p) = \sum_{x,y}e^{ip(x-y)}f(x,y)  [conventional]
    //f'(p) = \sum_{x,y}e^{-ip(x-y)}f(x,y) [Daiqian]
    //This may have been a mistake as it only manifests in the difference between the labelling of the pion momenta and the sign of 
    //the individual quark momenta.
    //However it doesn't really make any difference. If you enable DAIQIAN_PION_PHASE_CONVENTION
    //the output files will be labelled in Daiqian's convention
#define DAIQIAN_PION_PHASE_CONVENTION

    std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_pioncorr_mom";
#ifndef DAIQIAN_PION_PHASE_CONVENTION
    os << pion_mom.getMesonMomentum(p).file_str(2);  //note the divisor of 2 is to put the momenta in units of pi/L and not pi/2L
#else
    os << (-pion_mom.getMesonMomentum(p)).file_str(2);
#endif
    pion.write(os.str());
#ifdef WRITE_HEX_OUTPUT
    os << ".hexfloat";
    pion.write(os.str(),true);
#endif
  }
  time += dclock();
  print_time("main","Pion 2pt function",time);

  if(!UniqueID()) printf("Memory after pion 2pt function computation:\n");
  printMem();
}

void computeSigmaMesonFields(typename ComputeSigma<A2Apolicies>::Vtype &V, typename ComputeSigma<A2Apolicies>::Wtype &W,
			     const int conf, Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  double time = -dclock();
  if(!UniqueID()) printf("Computing sigma mesonfield computation\n");
  ComputeSigma<A2Apolicies>::computeAndWrite(params.meas_arg.WorkDirectory,conf,W,V, params.jp.pion_rad, lat, field3dparams);
  time += dclock();
  print_time("main","Sigma meson fields ",time);
}

void computePiPi2pt(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, const StandardPionMomentaPolicy &pion_mom, const int conf, const Parameters &params){
  const int nmom = pion_mom.nMom();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();

  if(!UniqueID()) printf("Computing pi-pi 2pt function\n");
  double timeC(0), timeD(0), timeR(0), timeV(0);
  double* timeCDR[3] = {&timeC, &timeD, &timeR};

  for(int psrcidx=0; psrcidx < nmom; psrcidx++){
    ThreeMomentum p_pi1_src = pion_mom.getMesonMomentum(psrcidx);

    for(int psnkidx=0; psnkidx < nmom; psnkidx++){	
      fMatrix<typename A2Apolicies::ScalarComplexType> pipi(Lt,Lt);
      ThreeMomentum p_pi1_snk = pion_mom.getMesonMomentum(psnkidx);

#ifndef DISABLE_PIPI_PRODUCTSTORE
      MesonFieldProductStore<A2Apolicies> products; //try to reuse products of meson fields wherever possible
#endif
      
      char diag[3] = {'C','D','R'};
      for(int d = 0; d < 3; d++){
	if(!UniqueID()){ printf("Doing pipi figure %c, psrcidx=%d psnkidx=%d\n",diag[d],psrcidx,psnkidx); fflush(stdout); }
	printMem(0);

	double time = -dclock();
	ComputePiPiGparity<A2Apolicies>::compute(pipi, diag[d], p_pi1_src, p_pi1_snk, params.jp.pipi_separation, params.jp.tstep_pipi, mf_ll_con
#ifndef DISABLE_PIPI_PRODUCTSTORE
						 , products
#endif
#ifdef NODE_DISTRIBUTE_MESONFIELDS
						 , (d == 2 ? true : false)
#endif
						 );
	std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_Figure" << diag[d] << "_sep" << params.jp.pipi_separation;
#ifndef DAIQIAN_PION_PHASE_CONVENTION
	os << "_mom" << p_pi1_src.file_str(2) << "_mom" << p_pi1_snk.file_str(2);
#else
	os << "_mom" << (-p_pi1_src).file_str(2) << "_mom" << (-p_pi1_snk).file_str(2);
#endif
	pipi.write(os.str());
#ifdef WRITE_HEX_OUTPUT
	os << ".hexfloat";
	pipi.write(os.str(),true);
#endif	  
	time += dclock();
	*timeCDR[d] += time;
      }
    }

    { //V diagram
      if(!UniqueID()){ printf("Doing pipi figure V, pidx=%d\n",psrcidx); fflush(stdout); }
      printMem(0);
      double time = -dclock();
      fVector<typename A2Apolicies::ScalarComplexType> figVdis(Lt);
      ComputePiPiGparity<A2Apolicies>::computeFigureVdis(figVdis,p_pi1_src,params.jp.pipi_separation,mf_ll_con);
      std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_FigureVdis_sep" << params.jp.pipi_separation;
#ifndef DAIQIAN_PION_PHASE_CONVENTION
      os << "_mom" << p_pi1_src.file_str(2);
#else
      os << "_mom" << (-p_pi1_src).file_str(2);
#endif
      figVdis.write(os.str());
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      figVdis.write(os.str(),true);
#endif	
      time += dclock();
      timeV += time;
    }
  }//end of psrcidx loop

  print_time("main","Pi-pi figure C",timeC);
  print_time("main","Pi-pi figure D",timeD);
  print_time("main","Pi-pi figure R",timeR);
  print_time("main","Pi-pi figure V",timeV);

  if(!UniqueID()) printf("Memory after pi-pi 2pt function computation:\n");
  printMem();
}


void computeKtoPiPi(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con_2s,
		    const A2AvectorV<A2Apolicies> &V, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W,
		    const A2AvectorV<A2Apolicies> &V_s, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W_s,
		    Lattice &lat, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
		    const StandardPionMomentaPolicy &pion_mom, const int conf, const Parameters &params){
  const int nmom = pion_mom.nMom();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  
  //We first need to generate the light-strange W*W contraction
  std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww;
  ComputeKtoPiPiGparity<A2Apolicies>::generatelsWWmesonfields(mf_ls_ww,W,W_s,params.jp.kaon_rad,lat, field3dparams);

  std::vector<int> k_pi_separation(params.jp.k_pi_separation.k_pi_separation_len);
  for(int i=0;i<params.jp.k_pi_separation.k_pi_separation_len;i++) k_pi_separation[i] = params.jp.k_pi_separation.k_pi_separation_val[i];

  if(!UniqueID()) printf("Memory after computing W*W meson fields:\n");
  printMem();

  typedef ComputeKtoPiPiGparity<A2Apolicies>::ResultsContainerType ResultsContainerType;
  typedef ComputeKtoPiPiGparity<A2Apolicies>::MixDiagResultsContainerType MixDiagResultsContainerType;

  MesonFieldMomentumContainer<A2Apolicies>* ll_meson_field_ptrs[2] = { &mf_ll_con, &mf_ll_con_2s };
  const int nsource = GJP.Gparity() ? 2 : 1;
  const std::string src_str[2] = { "", "_src2s" };
    
  //For type1 loop over momentum of pi1 (conventionally the pion closest to the kaon)
  int ngp = 0; for(int i=0;i<3;i++) if(GJP.Bc(i)==BND_CND_GPARITY) ngp++;
#define TYPE1_DO_ASSUME_ROTINVAR_GP3  //For GPBC in 3 directions we can assume rotational invariance around the G-parity diagonal vector (1,1,1) and therefore calculate only one off-diagonal momentum

  if(!UniqueID()) printf("Starting type 1 contractions, nmom = %d\n",nmom);
  double time = -dclock();
    
  for(int pidx=0; pidx < nmom; pidx++){
#ifdef TYPE1_DO_ASSUME_ROTINVAR_GP3
    if(ngp == 3 && pidx >= 4) continue; // p_pi1 = (-1,-1,-1), (1,1,1) [diag] (1,-1,-1), (-1,1,1) [orth] only
#endif
    for(int sidx=0; sidx<nsource;sidx++){
      
      if(!UniqueID()) printf("Starting type 1 contractions with pidx=%d and source idx %d\n",pidx,sidx);
      if(!UniqueID()) printf("Memory status before type1 K->pipi:\n");
      printMem();

      ThreeMomentum p_pi1 = pion_mom.getMesonMomentum(pidx);
      std::vector<ResultsContainerType> type1;
      ComputeKtoPiPiGparity<A2Apolicies>::type1(type1,
						k_pi_separation, params.jp.pipi_separation, params.jp.tstep_type12, params.jp.xyzstep_type1, p_pi1,
						mf_ls_ww, *ll_meson_field_ptrs[sidx],
						V, V_s,
						W, W_s);
      for(int kpi_idx=0;kpi_idx<k_pi_separation.size();kpi_idx++){
	std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_type1_deltat_" << k_pi_separation[kpi_idx] << src_str[sidx] << "_sep_" << params.jp.pipi_separation;
#ifndef DAIQIAN_PION_PHASE_CONVENTION
	os << "_mom" << p_pi1.file_str(2);
#else
	os << "_mom" << (-p_pi1).file_str(2);
#endif
	type1[kpi_idx].write(os.str());
#ifdef WRITE_HEX_OUTPUT
	os << ".hexfloat";
	type1[kpi_idx].write(os.str(),true);
#endif
      }
      if(!UniqueID()) printf("Memory status after type1 K->pipi:\n");
      printMem();
    }
  }

    
  time += dclock();
  print_time("main","K->pipi type 1",time);

  if(!UniqueID()) printf("Memory after type1 K->pipi:\n");
  printMem();

  //Type 2 and 3 are optimized by performing the sum over pipi momentum orientations within the contraction
  time = -dclock();    
  for(int sidx=0; sidx< nsource; sidx++){
    if(!UniqueID()) printf("Starting type 2 contractions with source idx %d\n", sidx);
    std::vector<ResultsContainerType> type2;
    ComputeKtoPiPiGparity<A2Apolicies>::type2(type2,
					      k_pi_separation, params.jp.pipi_separation, params.jp.tstep_type12, pion_mom,
					      mf_ls_ww, *ll_meson_field_ptrs[sidx],
					      V, V_s,
					      W, W_s);
    for(int kpi_idx=0;kpi_idx<k_pi_separation.size();kpi_idx++){
      std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_type2_deltat_" << k_pi_separation[kpi_idx] << src_str[sidx] << "_sep_" << params.jp.pipi_separation;
      type2[kpi_idx].write(os.str());
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      type2[kpi_idx].write(os.str(),true);
#endif
    }
  }
  time += dclock();
  print_time("main","K->pipi type 2",time);
    
  if(!UniqueID()) printf("Memory after type2 K->pipi:\n");
  printMem();
    

  time = -dclock();
  for(int sidx=0; sidx< nsource; sidx++){
    if(!UniqueID()) printf("Starting type 3 contractions with source idx %d\n", sidx);
    std::vector<ResultsContainerType> type3;
    std::vector<MixDiagResultsContainerType> mix3;
    ComputeKtoPiPiGparity<A2Apolicies>::type3(type3,mix3,
					      k_pi_separation, params.jp.pipi_separation, 1, pion_mom,
					      mf_ls_ww, *ll_meson_field_ptrs[sidx],
					      V, V_s,
					      W, W_s);
    for(int kpi_idx=0;kpi_idx<k_pi_separation.size();kpi_idx++){
      std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_type3_deltat_" << k_pi_separation[kpi_idx] << src_str[sidx] << "_sep_" << params.jp.pipi_separation;
      write(os.str(),type3[kpi_idx],mix3[kpi_idx]);
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      write(os.str(),type3[kpi_idx],mix3[kpi_idx],true);
#endif
    }
  }
  time += dclock();
  print_time("main","K->pipi type 3",time);
    
  if(!UniqueID()) printf("Memory after type3 K->pipi:\n");
  printMem();
    

  {
    //Type 4 has no momentum loop as the pion disconnected part is computed as part of the pipi 2pt function calculation
    time = -dclock();
    if(!UniqueID()) printf("Starting type 4 contractions\n");
    ResultsContainerType type4;
    MixDiagResultsContainerType mix4;
      
    ComputeKtoPiPiGparity<A2Apolicies>::type4(type4, mix4,
					      1,
					      mf_ls_ww,
					      V, V_s,
					      W, W_s);
      
    {
      std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_type4";
      write(os.str(),type4,mix4);
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      write(os.str(),type4,mix4,true);
#endif
    }
    time += dclock();
    print_time("main","K->pipi type 4",time);
    
    if(!UniqueID()) printf("Memory after type4 K->pipi and end of config loop:\n");
    printMem();
  }
}//do_ktopipi



void doConfiguration(const int conf, Parameters &params, const CommandLineArgs &cmdline,
		     const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
		     const typename A2Apolicies::FermionFieldType::InputParamType &field4dparams, COMPUTE_EVECS_EXTRA_ARG_GRAB){

  params.meas_arg.TrajCur = conf;

  std::string dir(params.meas_arg.WorkDirectory);

  //-------------------- Read gauge field --------------------//
  ReadGaugeField(params.meas_arg,cmdline.double_latt); 
  ReadRngFile(params.meas_arg,cmdline.double_latt); 
    
  if(!UniqueID()) printf("Memory after gauge and RNG read:\n");
  printMem();

  if(cmdline.tune_lanczos_light || cmdline.tune_lanczos_heavy) LanczosTune(cmdline.tune_lanczos_light, cmdline.tune_lanczos_heavy, params, COMPUTE_EVECS_EXTRA_ARG_PASS);    

  //-------------------- Light quark Lanczos ---------------------//
  LanczosWrapper eig;
  if(!cmdline.randomize_vw || cmdline.force_evec_compute) computeEvecs(eig, Light, params, cmdline.evecs_single_prec, cmdline.randomize_evecs, COMPUTE_EVECS_EXTRA_ARG_PASS);

  //-------------------- Light quark v and w --------------------//
  A2AvectorV<A2Apolicies> V(params.a2a_arg, field4dparams);
  A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
  computeVW(V, W, Light, params, eig, cmdline.evecs_single_prec, cmdline.randomize_vw, cmdline.mixed_solve, cmdline.inner_cg_resid_p, true, COMPUTE_EVECS_EXTRA_ARG_PASS);

  if(!UniqueID()){ printf("Freeing light evecs\n"); fflush(stdout); }
  eig.freeEvecs();
  if(!UniqueID()) printf("Memory after light evec free:\n");
  printMem();
    
  //-------------------- Strange quark Lanczos ---------------------//
  LanczosWrapper eig_s;
  if(!cmdline.randomize_vw || cmdline.force_evec_compute) computeEvecs(eig_s, Heavy, params, cmdline.evecs_single_prec, cmdline.randomize_evecs, COMPUTE_EVECS_EXTRA_ARG_PASS);

  //-------------------- Strange quark v and w --------------------//
  A2AvectorV<A2Apolicies> V_s(params.a2a_arg_s,field4dparams);
  A2AvectorW<A2Apolicies> W_s(params.a2a_arg_s,field4dparams);
  A2ALattice* a2a_lat = computeVW(V_s, W_s, Heavy, params, eig_s, cmdline.evecs_single_prec, cmdline.randomize_vw, cmdline.mixed_solve, cmdline.inner_cg_resid_p, false, COMPUTE_EVECS_EXTRA_ARG_PASS);

  eig_s.freeEvecs();
  if(!UniqueID()) printf("Memory after heavy evec free:\n");
  printMem();

  //From now one we just need a generic lattice instance, so use a2a_lat
  Lattice& lat = (Lattice&)(*a2a_lat);
    
  //-------------------Fix gauge----------------------------
  doGaugeFix(lat, cmdline.skip_gauge_fix, params);

  //-------------------------Compute the kaon two-point function---------------------------------
  if(cmdline.do_kaon2pt) computeKaon2pt(V,W,V_s,W_s,conf,lat,params,field3dparams);

  //----------------------------Compute the sigma meson fields---------------------------------
  if(cmdline.do_sigma) computeSigmaMesonFields(V,W,conf,lat,params,field3dparams);

  //The pion two-point function and pipi/k->pipi all utilize the same meson fields. Generate those here
  //For convenience pointers to the meson fields are collected into a single object that is passed to the compute methods
  StandardPionMomentaPolicy pion_mom; //these are the W and V momentum combinations

  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con; //stores light-light meson fields, accessible by momentum
  MesonFieldMomentumContainer<A2Apolicies> mf_ll_con_2s; //Gparity only

  computeLLmesonFields(mf_ll_con, mf_ll_con_2s, V, W, pion_mom, conf, lat, params, field3dparams);

  //----------------------------Compute the pion two-point function---------------------------------
  if(cmdline.do_pion2pt) computePion2pt(mf_ll_con, pion_mom, conf, params);
    
  //------------------------------I=0 and I=2 PiPi two-point function---------------------------------
  if(cmdline.do_pipi) computePiPi2pt(mf_ll_con, pion_mom, conf, params);

  //--------------------------------------K->pipi contractions--------------------------------------------------------
  if(cmdline.do_ktopipi) computeKtoPiPi(mf_ll_con,mf_ll_con_2s,V,W,V_s,W_s,lat,field3dparams,pion_mom,conf,params);

  delete a2a_lat;
}

void checkWriteable(const std::string &dir,const int conf){
  std::string file;
  {
    std::ostringstream os; os << dir << "/writeTest.node" << UniqueID() << ".conf" << conf;
    file = os.str();
  }
  std::ofstream of(file);
  double fail = 0;
  if(!of.good()){ std::cout << "checkWriteable failed to open file for write: " << file << std::endl; std::cout.flush(); fail = 1; }

  of << "Test\n";
  if(!of.good()){ std::cout << "checkWriteable failed to write to file: " << file << std::endl; std::cout.flush(); fail = 1; }

  glb_sum_five(&fail);

  if(fail != 0.){
    if(!UniqueID()){ printf("Disk write check failed\n");  fflush(stdout); }
    exit(-1);
  }else{
    if(!UniqueID()){ printf("Disk write check passed\n"); fflush(stdout); }
  }
}


void doConfigurationSplit(const int conf, Parameters &params, const CommandLineArgs &cmdline,
			  const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
			  const typename A2Apolicies::FermionFieldType::InputParamType &field4dparams, COMPUTE_EVECS_EXTRA_ARG_GRAB){
  checkWriteable(cmdline.checkpoint_dir,conf);
  params.meas_arg.TrajCur = conf;

  std::string dir(params.meas_arg.WorkDirectory);

  //-------------------- Read gauge field --------------------//
  ReadGaugeField(params.meas_arg,cmdline.double_latt); 
  ReadRngFile(params.meas_arg,cmdline.double_latt); 
    
  if(!UniqueID()) printf("Memory after gauge and RNG read:\n");
  printMem();

  if(cmdline.split_job_part == 0){
    //Do light Lanczos, strange Lanczos and strange CG, store results
    
    //-------------------- Light quark Lanczos ---------------------//
    {
      LanczosWrapper eig;
      computeEvecs(eig, Light, params, cmdline.evecs_single_prec, cmdline.randomize_evecs, COMPUTE_EVECS_EXTRA_ARG_PASS);
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.lanczos_l.cfg" << conf;
      if(!UniqueID()){ printf("Writing light Lanczos to %s\n",os.str().c_str()); fflush(stdout); }
      double time = -dclock();
      eig.writeParallel(os.str());
      time+=dclock();
      print_time("main","Light Lanczos write",time);
    }

    {//Do the light A2A vector random fields to ensure same ordering as unsplit job
      A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
      W.setWhRandom();
    }
    
    //-------------------- Strange quark Lanczos ---------------------//
    LanczosWrapper eig_s;
    computeEvecs(eig_s, Heavy, params, cmdline.evecs_single_prec, cmdline.randomize_evecs, COMPUTE_EVECS_EXTRA_ARG_PASS);

    //-------------------- Strange quark v and w --------------------//
    A2AvectorV<A2Apolicies> V_s(params.a2a_arg_s,field4dparams);
    A2AvectorW<A2Apolicies> W_s(params.a2a_arg_s,field4dparams);
    computeVW(V_s, W_s, Heavy, params, eig_s, cmdline.evecs_single_prec, cmdline.randomize_vw, cmdline.mixed_solve, cmdline.inner_cg_resid_p, true, COMPUTE_EVECS_EXTRA_ARG_PASS);
    
    {
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.V_s.cfg" << conf;
      if(!UniqueID()){ printf("Writing V_s to %s\n",os.str().c_str()); fflush(stdout); }
      double time = -dclock();
      V_s.writeParallel(os.str());
      time+=dclock();
      print_time("main","V_s write",time);
    }
    {
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.W_s.cfg" << conf;
      if(!UniqueID()){ printf("Writing W_s to %s\n",os.str().c_str()); fflush(stdout); }
      double time = -dclock();
      W_s.writeParallel(os.str());
      time+=dclock();
      print_time("main","W_s write",time);
    }
    
  }else if(cmdline.split_job_part == 1){
    //Do light CG and contractions

    LanczosWrapper eig;
    {
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.lanczos_l.cfg" << conf;
      if(!UniqueID()) printf("Reading light Lanczos from %s\n",os.str().c_str());
      double time = -dclock();
#ifdef USE_GRID_LANCZOS
      LanczosLattice* lanczos_lat = createLattice<LanczosLattice,LANCZOS_LATMARK>::doit(LANCZOS_LATARGS);
      eig.readParallel(os.str(),*lanczos_lat);
      delete lanczos_lat;
#else
      eig.readParallel(os.str());      
#endif
      time+=dclock();
      print_time("main","Light Lanczos read",time);
    }

    //-------------------- Light quark v and w --------------------//
    A2AvectorV<A2Apolicies> V(params.a2a_arg, field4dparams);
    A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
    A2ALattice* a2a_lat = computeVW(V, W, Light, params, eig, cmdline.evecs_single_prec, cmdline.randomize_vw, cmdline.mixed_solve, cmdline.inner_cg_resid_p, false, COMPUTE_EVECS_EXTRA_ARG_PASS);
    
    eig.freeEvecs();
    if(!UniqueID()) printf("Memory after light evec free:\n");
    printMem();    
    
    //-------------------- Strange quark v and w read --------------------//
    A2AvectorV<A2Apolicies> V_s(params.a2a_arg_s,field4dparams);
    A2AvectorW<A2Apolicies> W_s(params.a2a_arg_s,field4dparams);
    
    {
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.V_s.cfg" << conf;
      if(!UniqueID()) printf("Reading V_s from %s\n",os.str().c_str());
      double time = -dclock();
      V_s.readParallel(os.str());
      time+=dclock();
      print_time("main","V_s read",time);
    }
    {
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.W_s.cfg" << conf;
      if(!UniqueID()) printf("Reading W_s from %s\n",os.str().c_str());
      double time = -dclock();
      W_s.readParallel(os.str());
      time+=dclock();
      print_time("main","W_s read",time);
    }

    //From now one we just need a generic lattice instance, so use a2a_lat
    Lattice& lat = (Lattice&)(*a2a_lat);
    
    //-------------------Fix gauge----------------------------
    doGaugeFix(lat, cmdline.skip_gauge_fix, params);
  
    //-------------------------Compute the kaon two-point function---------------------------------
    if(cmdline.do_kaon2pt) computeKaon2pt(V,W,V_s,W_s,conf,lat,params,field3dparams);
  
    //The pion two-point function and pipi/k->pipi all utilize the same meson fields. Generate those here
    //For convenience pointers to the meson fields are collected into a single object that is passed to the compute methods
    StandardPionMomentaPolicy pion_mom; //these are the W and V momentum combinations

    MesonFieldMomentumContainer<A2Apolicies> mf_ll_con; //stores light-light meson fields, accessible by momentum
    MesonFieldMomentumContainer<A2Apolicies> mf_ll_con_2s; //Gparity only

    computeLLmesonFields(mf_ll_con, mf_ll_con_2s, V, W, pion_mom, conf, lat, params, field3dparams);

    //----------------------------Compute the pion two-point function---------------------------------
    if(cmdline.do_pion2pt) computePion2pt(mf_ll_con, pion_mom, conf, params);

    //----------------------------Compute the sigma meson fields---------------------------------
    if(cmdline.do_sigma) computeSigmaMesonFields(V,W,conf,lat,params,field3dparams);
    
    //------------------------------I=0 and I=2 PiPi two-point function---------------------------------
    if(cmdline.do_pipi) computePiPi2pt(mf_ll_con, pion_mom, conf, params);

    //--------------------------------------K->pipi contractions--------------------------------------------------------
    if(cmdline.do_ktopipi) computeKtoPiPi(mf_ll_con,mf_ll_con_2s,V,W,V_s,W_s,lat,field3dparams,pion_mom,conf,params);

    delete a2a_lat;
  }else{ //part 1
    ERR.General("","doConfigurationSplit","Invalid part index %d\n", cmdline.split_job_part);
  }

    
}







#endif
