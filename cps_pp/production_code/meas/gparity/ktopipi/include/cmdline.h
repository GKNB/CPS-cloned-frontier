#ifndef _KTOPIPI_MAIN_A2A_CMDLINE_H_
#define _KTOPIPI_MAIN_A2A_CMDLINE_H_

//Command line argument store/parse
struct CommandLineArgs{
  int nthreads;
  int nthread_contractions;
  bool randomize_vw; //rather than doing the Lanczos and inverting the propagators, etc, just use random vectors for V and W
  bool randomize_evecs; //skip Lanczos and just use random evecs for testing.
  bool randomize_mf; //use random meson fields
  bool force_evec_compute; //randomize_evecs causes Lanczos to be skipped unless this option is used
  bool tune_lanczos_light; //just run the light lanczos on first config then exit
  bool tune_lanczos_heavy; //just run the heavy lanczos on first config then exit
  bool skip_gauge_fix;
  bool double_latt; //most ancient 8^4 quenched lattices stored both U and U*. Enable this to read those configs
  bool do_kaon2pt;
  bool do_pion2pt;
  bool do_pipi;
  bool do_ktopipi;
  bool do_sigma;

  bool do_split_job;
  int split_job_part;
  std::string checkpoint_dir; //directory the checkpointed data is stored in (could be scratch for example)

#ifdef BNL_KNL_PERFORMANCE_CHECK
  double bnl_knl_minperf; //in Mflops per node (not per rank, eg MPI3!)
#endif

#ifdef USE_GRID
  bool run_initial_grid_benchmarks;
#endif
  
  CommandLineArgs(int argc, char **argv, int begin){
    nthreads = 1;
#if TARGET == BGQ
    nthreads = 64;
#endif
    nthread_contractions = -1;
    randomize_vw = false;
    randomize_evecs = false;
    randomize_mf = false;
    force_evec_compute = false; //randomize_evecs causes Lanczos to be skipped unless this option is used
    tune_lanczos_light = false; //just run the light lanczos on first config then exit
    tune_lanczos_heavy = false; //just run the heavy lanczos on first config then exit
    skip_gauge_fix = false;
    double_latt = false; //most ancient 8^4 quenched lattices stored both U and U*. Enable this to read those configs
    do_kaon2pt = true;
    do_pion2pt = true;
    do_pipi = true;
    do_ktopipi = true;
    do_sigma = true;

    do_split_job = false;

#ifdef BNL_KNL_PERFORMANCE_CHECK
    bnl_knl_minperf = 50000;
#endif
#ifdef USE_GRID
    run_initial_grid_benchmarks = false;
#endif
    
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
      std::string cmdstr(cmd);
      if( cmdstr == "-nthread" ){
	if(arg == argc-1){ if(!UniqueID()){ printf("-nthread must be followed by a number!\n"); fflush(stdout); } exit(-1); }
	nthreads = strToAny<int>(argv[arg+1]);
	if(!UniqueID()){ printf("Setting number of threads to %d\n",nthreads); }
	arg+=2;
      }else if( cmdstr == "-nthread_contractions" ){ //optional - use if you want more/less threads in the contraction part
	if(arg == argc-1){ if(!UniqueID()){ printf("-nthread_contractions must be followed by a number!\n"); fflush(stdout); } exit(-1); }
	nthread_contractions = strToAny<int>(argv[arg+1]);
	if(!UniqueID()){ printf("Setting number of threads in contractions to %d\n",nthread_contractions); }
	arg+=2;
      }else if( strncmp(cmd,"-randomize_vw",15) == 0){
	randomize_vw = true;
	if(!UniqueID()){ printf("Using random vectors for V and W, skipping Lanczos and inversion stages\n"); fflush(stdout); }
	arg++;
      }else if( strncmp(cmd,"-randomize_evecs",15) == 0){
	randomize_evecs = true;
	if(!UniqueID()){ printf("Using random eigenvectors\n"); fflush(stdout); }
	arg++;      
      }else if( strncmp(cmd,"-randomize_mf",15) == 0){
	randomize_mf = true;
	if(!UniqueID()){ printf("Using random meson fields\n"); fflush(stdout); }
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
      }else if( strncmp(cmd,"-mmap_threshold_and_max",40) == 0){ //Using these options can reduce memory fragmentation but may impact performance
	size_t threshold = strToAny<size_t>(argv[arg+1]);
	size_t mmap_max = strToAny<size_t>(argv[arg+2]);
	if(!UniqueID()){
	  std::ostringstream os; os << "Set mmap_threshold to " << threshold << " and mmap_max to " << mmap_max << std::endl;  
	  printf(os.str().c_str());
	}
	assert(mallopt(M_MMAP_THRESHOLD, threshold)==1);
	assert(mallopt(M_MMAP_MAX, mmap_max)==1);
	arg+=3;
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
#ifdef USE_GRID
      }else if( strncmp(cmd,"-run_initial_grid_benchmarks",50) == 0){
	run_initial_grid_benchmarks = true;
	if(!UniqueID()) printf("Running initial Grid benchmarks\n");
	arg++;
#endif
#if defined(MESONFIELD_USE_BURSTBUFFER) || defined(MESONFIELD_USE_NODE_SCRATCH)
      }else if( strncmp(cmd,"-mesonfield_scratch_stub",50) == 0){
	BurstBufferMemoryStorage::filestub() = argv[arg+1];
	if(!UniqueID()) printf("Set mesonfield scratch stub to %s\n",BurstBufferMemoryStorage::filestub().c_str());
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

    if(nthread_contractions == -1) nthread_contractions = nthreads; //default equal
  }

};


#endif
