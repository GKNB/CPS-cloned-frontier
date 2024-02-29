#ifndef _KTOPIPI_MAIN_A2A_CMDLINE_H_
#define _KTOPIPI_MAIN_A2A_CMDLINE_H_

struct computeEvecsOpts{
  bool randomize_evecs; //skip Lanczos and just use random evecs for testing.

  bool save_evecs;
  std::string save_evecs_stub;

  bool load_evecs;
  std::string load_evecs_stub;

  computeEvecsOpts(): randomize_evecs(false), save_evecs(false), load_evecs(false)
  {}
};

struct computeVWopts{
  bool randomize_vw; //rather than doing the Lanczos and inverting the propagators, etc, just use random vectors for V and W

  bool save_vw;
  std::string save_vw_stub;

  bool load_vw;
  std::string load_vw_stub;

  computeVWopts(): randomize_vw(false), save_vw(false), load_vw(false)
  {}
};


//Command line argument store/parse
struct CommandLineArgs{
  int nthreads;
  int nthread_contractions;
	
  computeEvecsOpts evec_opts_l;
  computeEvecsOpts evec_opts_h;

  computeVWopts vw_opts_l;
  computeVWopts vw_opts_h;

  bool randomize_evecs; //skip Lanczos and just use random evecs for testing.
  bool randomize_mf; //use random meson fields
  bool force_evec_compute; //randomize_evecs causes Lanczos to be skipped unless this option is used
  bool tune_lanczos_light; //just run the light lanczos on first config then exit
  bool tune_lanczos_heavy; //just run the heavy lanczos on first config then exit
  bool skip_gauge_fix;
  bool tune_gauge_fix;
  bool double_latt; //most ancient 8^4 quenched lattices stored both U and U*. Enable this to read those configs
  bool do_kaon2pt;
  bool do_pion2pt;
  bool do_pipi;
  bool do_ktopipi;
  bool do_sigma;
  bool do_ktosigma;
  bool do_sigma2pt;
  bool do_pipitosigma;
  bool do_crusher_benchmark;	//Only for doing performance benchmark on frontier for lanc, gfix and cg for light/heavy

  bool old_gparity_cfg; //for old G-parity configs the plaquette in the header was 2x too small. Now this is fixed those configs won't load unless this arg is set

  //For split job version
  bool do_split_job;
  int split_job_part;
  std::string checkpoint_dir; //directory the checkpointed data is stored in (could be scratch for example)
  
  //For version that just computes light quark props (also sets checkpoint_dir as above)
  bool do_LL_props_only;

  //For sigma->pipi, choose to save meson fields or load and not recompute
  bool ktosigma_load_sigma_mf;
  bool ktosigma_save_sigma_mf;
  std::string ktosigma_sigma_mf_dir;

  //
  bool save_all_a2a_inputs; //save all meson fields and a2a vectors (not all jobs have this set up)
  std::string save_all_a2a_inputs_dir;

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
    randomize_mf = false;
    force_evec_compute = false; //randomize_evecs causes Lanczos to be skipped unless this option is used
    tune_lanczos_light = false; //just run the light lanczos on first config then exit
    tune_lanczos_heavy = false; //just run the heavy lanczos on first config then exit
    skip_gauge_fix = false;
    tune_gauge_fix = false;
    double_latt = false; //most ancient 8^4 quenched lattices stored both U and U*. Enable this to read those configs
    do_kaon2pt = true;
    do_pion2pt = true;
    do_pipi = true;
    do_ktopipi = true;
    do_sigma = true;
    do_ktosigma = true;
    do_sigma2pt = true;
    do_pipitosigma = true;
    do_crusher_benchmark = false;

    old_gparity_cfg = false;

    do_split_job = false;

    do_LL_props_only = false;

    ktosigma_load_sigma_mf = false;
    ktosigma_save_sigma_mf = false;

    save_all_a2a_inputs = false;

#ifdef BNL_KNL_PERFORMANCE_CHECK
    bnl_knl_minperf = 50000;
#endif
#ifdef USE_GRID
    run_initial_grid_benchmarks = false;
#endif
    
    parse(argc,argv,begin);
  }

  void parse(int argc, char **argv, int begin){
    LOGA2A << "Arguments:" << std::endl;
    for(int i=0;i<argc;i++){
      a2a_printfnt("%d \"%s\"\n",i,argv[i]);
    }
    
    const int ngrid_arg = 16;
    const std::string grid_args[ngrid_arg] = { "--debug-signals", "--dslash-generic", "--dslash-unroll",
					       "--dslash-asm", "--shm", "--lebesgue",
					       "--cacheblocking", "--comms-concurrent", "--comms-sequential",
					       "--comms-overlap", "--log", "--comms-threads",
					       "--shm-hugepages", "--accelerator-threads",
   					       "--device-mem", "--shm-mpi" };
    const int grid_args_skip[ngrid_arg] =    { 1  , 1 , 1,
					       1  , 2 , 1,
					       2  , 1 , 1,
					       1  , 2 , 2,
					       1  , 2 ,
					       2  , 2    };

    int arg = begin;
    while(arg < argc){
      char* cmd = argv[arg];
      std::string cmdstr(cmd);
      if( cmdstr == "-nthread" ){
	if(arg == argc-1){ LOGA2A << "-nthread must be followed by a number!" << std::endl; exit(-1); }
	nthreads = strToAny<int>(argv[arg+1]);
	LOGA2A << "Setting number of threads to " << nthreads << std::endl;
	arg+=2;
      }else if( cmdstr == "-nthread_contractions" ){ //optional - use if you want more/less threads in the contraction part
	if(arg == argc-1){ LOGA2A << "-nthread_contractions must be followed by a number!" << std::endl; exit(-1); }
	nthread_contractions = strToAny<int>(argv[arg+1]);
	LOGA2A << "Setting number of threads in contractions to " << nthread_contractions << std::endl;
	arg+=2;
      }else if( strncmp(cmd,"-randomize_vw",15) == 0){
	vw_opts_l.randomize_vw = vw_opts_h.randomize_vw = true;
	LOGA2A << "Using random vectors for V and W, skipping Lanczos and inversion stages" << std::endl;
	arg++;
      }else if( strncmp(cmd,"-randomize_evecs",15) == 0){
	evec_opts_l.randomize_evecs = evec_opts_h.randomize_evecs = true;
	LOGA2A << "Using random eigenvectors" << std::endl;
	arg++;      
      }else if( cmdstr == "-load_light_evecs"){
	assert(arg < argc-1);
	evec_opts_l.load_evecs = true;
	evec_opts_l.load_evecs_stub = argv[arg+1];
	LOGA2A << "Loading light eigenvectors with stub " << evec_opts_l.load_evecs_stub << std::endl;
	arg += 2;
      }else if( cmdstr == "-save_light_evecs"){
	assert(arg < argc-1);
	evec_opts_l.save_evecs = true;
	evec_opts_l.save_evecs_stub = argv[arg+1];
	LOGA2A << "Saving light eigenvectors with stub " << evec_opts_l.save_evecs_stub << std::endl;
	arg += 2;
      }else if( cmdstr == "-load_heavy_evecs"){
	assert(arg < argc-1);
	evec_opts_h.load_evecs = true;
	evec_opts_h.load_evecs_stub = argv[arg+1];
	LOGA2A << "Loading heavy eigenvectors with stub " << evec_opts_h.load_evecs_stub << std::endl;
	arg += 2;
      }else if( cmdstr == "-save_heavy_evecs"){
	assert(arg < argc-1);
	evec_opts_h.save_evecs = true;
	evec_opts_h.save_evecs_stub = argv[arg+1];
	LOGA2A << "Saving heavy eigenvectors with stub " << evec_opts_h.save_evecs_stub << std::endl;
	arg += 2;
      }else if( cmdstr == "-load_light_vw"){
	assert(arg < argc-1);
	vw_opts_l.load_vw = true;
	vw_opts_l.load_vw_stub = argv[arg+1];
	LOGA2A << "Loading light VW fields with stub " << vw_opts_l.load_vw_stub << std::endl;
	arg += 2;
      }else if( cmdstr == "-save_light_vw"){
	assert(arg < argc-1);
	vw_opts_l.save_vw = true;
	vw_opts_l.save_vw_stub = argv[arg+1];
	LOGA2A << "Saving light VW fields with stub " << vw_opts_l.save_vw_stub << std::endl;
	arg += 2;
      }else if( cmdstr == "-load_heavy_vw"){
	assert(arg < argc-1);
	vw_opts_h.load_vw = true;
	vw_opts_h.load_vw_stub = argv[arg+1];
	LOGA2A << "Loading heavy VW fields with stub " << vw_opts_h.load_vw_stub << std::endl;
	arg += 2;
      }else if( cmdstr == "-save_heavy_vw"){
	assert(arg < argc-1);
	vw_opts_h.save_vw = true;
	vw_opts_h.save_vw_stub = argv[arg+1];
	LOGA2A << "Saving heavy VW fields with stub " << vw_opts_h.save_vw_stub << std::endl;
	arg += 2;
      }else if( strncmp(cmd,"-randomize_mf",15) == 0){
	randomize_mf = true;
	LOGA2A << "Using random meson fields" << std::endl;
	arg++; 
      }else if( strncmp(cmd,"-force_evec_compute",15) == 0){
	force_evec_compute = true;
	LOGA2A << "Forcing evec compute despite randomize_vw" << std::endl;
	arg++;      
      }else if( strncmp(cmd,"-tune_lanczos_light",15) == 0){
	tune_lanczos_light = true;
	LOGA2A << "Just tuning light lanczos on first config" << std::endl;
	arg++;
      }else if( strncmp(cmd,"-tune_lanczos_heavy",15) == 0){
	tune_lanczos_heavy = true;
	LOGA2A << "Just tuning heavy lanczos on first config" << std::endl;
	arg++;
      }else if( strncmp(cmd,"-double_latt",15) == 0){
	double_latt = true;
	LOGA2A << "Loading doubled lattices" << std::endl;
	arg++;
      }else if( strncmp(cmd,"-skip_gauge_fix",20) == 0){
	skip_gauge_fix = true;
	LOGA2A << "Skipping gauge fixing" << std::endl;
	arg++;
      }else if( strncmp(cmd,"-do_crusher_benchmark", 20) == 0){
	do_crusher_benchmark = true;
	LOGA2A << "Doing benchmark for lanc/gfix/CG on crusher/frontier" << std::endl;
	arg++;
      }else if( cmdstr == "-tune_gauge_fix"){
	tune_gauge_fix = true;
	LOGA2A << "Tuning gauge fixing" << std::endl;
	arg++;
      }else if( strncmp(cmd,"-mf_outerblocking",15) == 0){
	int* b[3] = { &BlockedMesonFieldArgs::bi, &BlockedMesonFieldArgs::bj, &BlockedMesonFieldArgs::bp };
	for(int a=0;a<3;a++) *b[a] = strToAny<int>(argv[arg+1+a]);
	arg+=4;
      }else if( strncmp(cmd,"-mf_innerblocking",15) == 0){
	int* b[3] = { &BlockedMesonFieldArgs::bii, &BlockedMesonFieldArgs::bjj, &BlockedMesonFieldArgs::bpp };
	for(int a=0;a<3;a++) *b[a] = strToAny<int>(argv[arg+1+a]);
	arg+=4;
      }else if(cmdstr == "-vMv_blocking"){
	BlockedvMvOffloadArgs::b = strToAny<int>(argv[arg+1]);
	BlockedSplitvMvArgs::b = BlockedvMvOffloadArgs::b;
	arg+=2;
      }else if(cmdstr == "-vMv_inner_blocking"){
	BlockedvMvOffloadArgs::bb = strToAny<int>(argv[arg+1]);
	arg+=2;
      }else if( strncmp(cmd,"-do_split_job",30) == 0){
	do_split_job = true;
	split_job_part = strToAny<int>(argv[arg+1]);
	checkpoint_dir = argv[arg+2];
	a2a_printf("Doing split job part %d with checkpoint directory %s\n",split_job_part,checkpoint_dir.c_str());
	arg+=3;
      }else if( strncmp(cmd,"-do_LL_props_only",30) == 0){
	do_LL_props_only = true;
	checkpoint_dir = argv[arg+1];
	LOGA2A << "Doing LL props only with checkpoint directory "<< checkpoint_dir << std::endl;
	arg+=2;
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
      }else if( cmdstr == "-skip_ktosigma"){
	do_ktosigma = false;
	arg++;
      }else if( cmdstr == "-skip_sigma2pt"){
	do_sigma2pt = false;
	arg++;
      }else if( cmdstr == "-skip_pipitosigma"){
	do_pipitosigma = false;
	arg++;
      }else if( cmdstr == "-ktosigma_load_sigma_mf"){
	ktosigma_load_sigma_mf = true;
	ktosigma_sigma_mf_dir = argv[arg+1];
	arg+=2;
      }else if( cmdstr == "-ktosigma_save_sigma_mf"){
	ktosigma_save_sigma_mf = true;
	ktosigma_sigma_mf_dir = argv[arg+1];
	arg+=2;
      }else if( cmdstr == "-save_all_a2a_inputs"){
	save_all_a2a_inputs = true;
	save_all_a2a_inputs_dir = argv[arg+1];

	ktosigma_save_sigma_mf = true;
	ktosigma_sigma_mf_dir = argv[arg+1];

	arg+=2;
	
      }else if( strncmp(cmd,"-mmap_threshold_and_max",40) == 0){ //Using these options can reduce memory fragmentation but may impact performance
	size_t threshold = strToAny<size_t>(argv[arg+1]);
	size_t mmap_max = strToAny<size_t>(argv[arg+2]);
	LOGA2A << "Set mmap_threshold to " << threshold << " and mmap_max to " << mmap_max << std::endl;  
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
	LOGA2A << "Set BNL KNL min performance to " << bnl_knl_minperf << " Mflops/node" << std::endl;
	arg+=2;
#endif

#ifdef USE_GRID
      }else if( strncmp(cmd,"-run_initial_grid_benchmarks",50) == 0){
	run_initial_grid_benchmarks = true;
	LOGA2A << "Running initial Grid benchmarks" << std::endl;
	arg++;
#endif

      }else if( cmdstr == "-mesonfield_scratch_stub" ){
	BurstBufferMemoryStorage::filestub() = argv[arg+1];
	LOGA2A << "Set mesonfield scratch stub to " << BurstBufferMemoryStorage::filestub() << std::endl;
	arg+=2;

#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
      }else if( cmdstr == "-max_memblocks" ){
	ReuseBlockAllocatorOptions::maxBlocks() = strToAny<int>(argv[arg+1]);
	LOGA2A << "Set max memory blocks in block allocator to " << ReuseBlockAllocatorOptions::maxBlocks() << std::endl;
	arg+=2;
#endif

      }else if( cmdstr == "-gpu_pool_max_mem" ){
	std::stringstream ss; ss << argv[arg+1];
	size_t v; ss >> v;
	DeviceMemoryPoolManager::globalPool().setPoolMaxSize(v);
	HolisticMemoryPoolManager::globalPool().setPoolMaxSize(v, HolisticMemoryPoolManager::DevicePool);	
	arg+=2;
      }else if( cmdstr == "-host_pool_max_mem" ){
	std::stringstream ss; ss << argv[arg+1];
	size_t v; ss >> v;
	HolisticMemoryPoolManager::globalPool().setPoolMaxSize(v, HolisticMemoryPoolManager::HostPool);	
	arg+=2;
      }else if( cmdstr == "-mempool_verbose" ){
	DeviceMemoryPoolManager::globalPool().setVerbose(true);
	HolisticMemoryPoolManager::globalPool().setVerbose(true);
	arg+=1;
      }else if( cmdstr == "-mempool_scratchdir" ){
	HolisticMemoryPoolManager::globalPool().setDiskRoot(argv[arg+1]);
	arg+=2;	
      }else if( cmdstr == "-old_gparity_cfg"){
	old_gparity_cfg = true;
	arg++;
      }else{
	bool is_grid_arg = false;
	for(int i=0;i<ngrid_arg;i++){
	  if( std::string(cmd) == grid_args[i] ){
	    LOGA2A << "main.C: Ignoring Grid argument " << cmd << std::endl;
	    arg += grid_args_skip[i];
	    is_grid_arg = true;
	    break;
	  }
	}
	if(!is_grid_arg){
	  LOGA2A << "Unrecognised argument: " << cmd << std::endl;
	  exit(-1);
	}
      }
    }

    if(nthread_contractions == -1) nthread_contractions = nthreads; //default equal

    if(do_pipitosigma && !do_sigma2pt) ERR.General("CommandLineArgs","parse","pipi->sigma calculation also requires sigma 2pt calculation\n");
  }

};


#endif