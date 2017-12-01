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
    randomize_vw = false;
    randomize_evecs = false;
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
#ifdef MESONFIELD_USE_BURSTBUFFER
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

  if(!cmdline.tune_lanczos_light && !cmdline.tune_lanczos_heavy){ 
    assert(params.a2a_arg.nl <= params.lanc_arg.N_true_get);
    assert(params.a2a_arg_s.nl <= params.lanc_arg_s.N_true_get);
  }

  printMem("Initial memory post-initialize");
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
void LanczosTune(bool tune_lanczos_light, bool tune_lanczos_heavy, const Parameters &params, BFMGridSolverWrapper &solvers){
  if(tune_lanczos_light){
    BFMGridLanczosWrapper<A2Apolicies> eig(solvers, params.jp);
    if(!UniqueID()) printf("Tuning lanczos light with mass %f\n", params.lanc_arg.mass);

    double time = -dclock();
    eig.compute(params.lanc_arg);
    time += dclock();
    print_time("main","Lanczos light",time);
  }
  if(tune_lanczos_heavy){
    BFMGridLanczosWrapper<A2Apolicies> eig(solvers, params.jp);
    if(!UniqueID()) printf("Tuning lanczos heavy with mass %f\n", params.lanc_arg_s.mass);

    double time = -dclock();
    eig.compute(params.lanc_arg_s);
    time += dclock();
    print_time("main","Lanczos heavy",time);
  }
  exit(0);
}


enum LightHeavy { Light, Heavy };

void computeEvecs(BFMGridLanczosWrapper<A2Apolicies> &eig, const LightHeavy lh, const Parameters &params, const bool randomize_evecs){
  const char* name = (lh ==  Light ? "light" : "heavy");
  const LancArg &lanc_arg = (lh == Light ? params.lanc_arg : params.lanc_arg_s);
  return computeEvecs(eig, lanc_arg, params.jp, name, randomize_evecs);
}

void computeVW(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, const LightHeavy lh, const Parameters &params,
	       const BFMGridLanczosWrapper<A2Apolicies> &eig, const BFMGridA2ALatticeWrapper<A2Apolicies> &a2a_lat,
	       const bool randomize_vw){
  const A2AArg &a2a_arg = lh == Light ? params.a2a_arg : params.a2a_arg_s;  
  const char* name = (lh ==  Light ? "light" : "heavy");

  typedef typename A2Apolicies::FermionFieldType::InputParamType Field4DparamType;
  Field4DparamType field4dparams = V.getFieldInputParams();
  
  if(!UniqueID()){ printf("V vector requires %f MB, W vector %f MB of memory\n", 
			  A2AvectorV<A2Apolicies>::Mbyte_size(a2a_arg,field4dparams), A2AvectorW<A2Apolicies>::Mbyte_size(a2a_arg,field4dparams) );
    fflush(stdout);
  }
  
  if(!UniqueID()) printf("Computing %s quark A2A vectors\n",name);
  double time = -dclock();

  a2a_lat.computeVW(V,W,eig,params.jp.cg_controls,randomize_vw);
  
  printMem(stringize("Memory after %s A2A vector computation", name));

  time += dclock();
  std::ostringstream os; os << name << " quark A2A vectors";
  print_time("main",os.str().c_str(),time);
}

void doGaugeFix(Lattice &lat, const bool skip_gauge_fix, const Parameters &params){
  doGaugeFix(lat,skip_gauge_fix,params.fix_gauge_arg);
}

template<typename KaonMomentumPolicy>
void computeKaon2pt(typename ComputeKaon<A2Apolicies>::Vtype &V, typename ComputeKaon<A2Apolicies>::Wtype &W, 
		    typename ComputeKaon<A2Apolicies>::Vtype &V_s, typename ComputeKaon<A2Apolicies>::Wtype &W_s,
		    const KaonMomentumPolicy &kaon_mom,
		    const int conf, Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  if(!UniqueID()) printf("Computing kaon 2pt function\n");
  double time = -dclock();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  fMatrix<typename A2Apolicies::ScalarComplexType> kaon(Lt,Lt);
  ComputeKaon<A2Apolicies>::compute(kaon,
				    W, V, W_s, V_s, kaon_mom,
				    params.jp.kaon_rad, lat, field3dparams);
  std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_kaoncorr";
  kaon.write(os.str());
#ifdef WRITE_HEX_OUTPUT
  os << ".hexfloat";
  kaon.write(os.str(),true);
#endif
  time += dclock();
  print_time("main","Kaon 2pt function",time);

  printMem("Memory after kaon 2pt function computation");
}

template<typename PionMomentumPolicy>
void computeLLmesonFields(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con_2s,
			  typename ComputePion<A2Apolicies>::Vtype &V, typename ComputePion<A2Apolicies>::Wtype &W,
			  const PionMomentumPolicy &pion_mom,
			  const int conf, Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  if(!UniqueID()) printf("Computing light-light meson fields\n");
  double time = -dclock();
  if(!GJP.Gparity()) ComputePion<A2Apolicies>::computeMesonFields(mf_ll_con, params.meas_arg.WorkDirectory,conf, pion_mom, W, V, params.jp.pion_rad, lat, field3dparams);
  else ComputePion<A2Apolicies>::computeGparityMesonFields(mf_ll_con, mf_ll_con_2s, params.meas_arg.WorkDirectory,conf, pion_mom, W, V, params.jp.pion_rad, lat, field3dparams);
  time += dclock();
  print_time("main","Light-light meson fields",time);

  printMem("Memory after light-light meson field computation");
}
template<typename PionMomentumPolicy>
void computePion2pt(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, const PionMomentumPolicy &pion_mom, const int conf, const Parameters &params){
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

  printMem("Memory after pion 2pt function computation");
}

template<typename SigmaMomentumPolicy>
void computeSigmaMesonFields(typename ComputeSigma<A2Apolicies>::Vtype &V, typename ComputeSigma<A2Apolicies>::Wtype &W, const SigmaMomentumPolicy &sigma_mom,
			     const int conf, Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  double time = -dclock();
  if(!UniqueID()) printf("Computing sigma mesonfield computation\n");
  ComputeSigma<A2Apolicies>::computeAndWrite(params.meas_arg.WorkDirectory,conf,sigma_mom,W,V, params.jp.pion_rad, lat, field3dparams);
  time += dclock();
  print_time("main","Sigma meson fields ",time);
}
template<typename PionMomentumPolicy>
void computePiPi2pt(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, const PionMomentumPolicy &pion_mom, const int conf, const Parameters &params){
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
	printMem(stringize("Doing pipi figure %c, psrcidx=%d psnkidx=%d",diag[d],psrcidx,psnkidx),0);

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
      printMem(stringize("Doing pipi figure V, pidx=%d",psrcidx),0);
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

  printMem("Memory after pi-pi 2pt function computation");
}

template<typename PionMomentumPolicy>
void computeKtoPiPi(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con_2s,
		    const A2AvectorV<A2Apolicies> &V, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W,
		    const A2AvectorV<A2Apolicies> &V_s, typename ComputeKtoPiPiGparity<A2Apolicies>::Wtype &W_s,
		    Lattice &lat, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
		    const PionMomentumPolicy &pion_mom, const int conf, const Parameters &params){
  const int nmom = pion_mom.nMom();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  
  //We first need to generate the light-strange W*W contraction
  std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww;
  ComputeKtoPiPiGparity<A2Apolicies>::generatelsWWmesonfields(mf_ls_ww,W,W_s,params.jp.kaon_rad,lat, field3dparams);

  std::vector<int> k_pi_separation(params.jp.k_pi_separation.k_pi_separation_len);
  for(int i=0;i<params.jp.k_pi_separation.k_pi_separation_len;i++) k_pi_separation[i] = params.jp.k_pi_separation.k_pi_separation_val[i];

  printMem("Memory after computing W*W meson fields");

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
      printMem("Memory status before type1 K->pipi");

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
      printMem("Memory status after type1 K->pipi");
    }
  }

    
  time += dclock();
  print_time("main","K->pipi type 1",time);

  printMem("Memory after type1 K->pipi");

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
    
  printMem("Memory after type2 K->pipi");
    

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
    
  printMem("Memory after type3 K->pipi");
    

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
    
    printMem("Memory after type4 K->pipi and end of config loop");
  }
}//do_ktopipi


void readGaugeRNG(const Parameters &params, const CommandLineArgs &cmdline){
  readGaugeRNG(params.do_arg, params.meas_arg, cmdline.double_latt);
}



void doContractions(const int conf, Parameters &params, const CommandLineArgs &cmdline, Lattice& lat,
		    A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W,
		    A2AvectorV<A2Apolicies> &V_s, A2AvectorW<A2Apolicies> &W_s,
		    const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  //-------------------Fix gauge----------------------------
  doGaugeFix(lat, cmdline.skip_gauge_fix, params);

  //-------------------------Compute the kaon two-point function---------------------------------
  StationaryKaonMomentaPolicy kaon_mom;
  if(cmdline.do_kaon2pt) computeKaon2pt(V,W,V_s,W_s,kaon_mom,conf,lat,params,field3dparams);

  //----------------------------Compute the sigma meson fields---------------------------------
  StationarySigmaMomentaPolicy sigma_mom;
  if(cmdline.do_sigma) computeSigmaMesonFields(V,W,sigma_mom,conf,lat,params,field3dparams);

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
  
  if(cmdline.tune_lanczos_light || cmdline.tune_lanczos_heavy) LanczosTune(cmdline.tune_lanczos_light, cmdline.tune_lanczos_heavy, params, solvers);    

  //-------------------- Light quark Lanczos ---------------------//
  BFMGridLanczosWrapper<A2Apolicies> eig(solvers, params.jp);
  if(!cmdline.randomize_vw || cmdline.force_evec_compute) computeEvecs(eig, Light, params, cmdline.randomize_evecs);

  //-------------------- Light quark v and w --------------------//
  A2AvectorV<A2Apolicies> V(params.a2a_arg, field4dparams);
  A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
  {
    BFMGridA2ALatticeWrapper<A2Apolicies> latwrp(solvers, params.jp); //lattice created temporarily
    computeVW(V, W, Light, params, eig, latwrp, cmdline.randomize_vw);
  }
  if(!UniqueID()){ printf("Freeing light evecs\n"); fflush(stdout); }
  eig.freeEvecs();
  printMem("Memory after light evec free");
    
  //-------------------- Strange quark Lanczos ---------------------//
  BFMGridLanczosWrapper<A2Apolicies> eig_s(solvers, params.jp);
  if(!cmdline.randomize_vw || cmdline.force_evec_compute) computeEvecs(eig_s, Heavy, params, cmdline.randomize_evecs);

  //-------------------- Strange quark v and w --------------------//
  A2AvectorV<A2Apolicies> V_s(params.a2a_arg_s,field4dparams);
  A2AvectorW<A2Apolicies> W_s(params.a2a_arg_s,field4dparams);
  BFMGridA2ALatticeWrapper<A2Apolicies> latwrp(solvers, params.jp);
  computeVW(V_s, W_s, Heavy, params, eig_s, latwrp, cmdline.randomize_vw);

  eig_s.freeEvecs();
  printMem("Memory after heavy evec free");

  //From now one we just need a generic lattice instance, so use a2a_lat
  Lattice& lat = (Lattice&)(*latwrp.a2a_lat);
  
  doContractions(conf,params,cmdline,lat,V,W,V_s,W_s,field3dparams);
}

void doConfigurationSplit(const int conf, Parameters &params, const CommandLineArgs &cmdline,
			  const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
			  const typename A2Apolicies::FermionFieldType::InputParamType &field4dparams, BFMGridSolverWrapper &solvers){
  checkWriteable(cmdline.checkpoint_dir,conf);
  params.meas_arg.TrajCur = conf;

  std::string dir(params.meas_arg.WorkDirectory);

  //-------------------- Read gauge field --------------------//
  readGaugeRNG(params,cmdline);
    
  printMem("Memory after gauge and RNG read");

  runInitialGridBenchmarks(cmdline,params);
  
  if(cmdline.split_job_part == 0){
    //Do light Lanczos, strange Lanczos and strange CG, store results
    
    //-------------------- Light quark Lanczos ---------------------//
    {
      BFMGridLanczosWrapper<A2Apolicies> eig(solvers, params.jp);
      computeEvecs(eig, Light, params, cmdline.randomize_evecs);
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.lanczos_l.cfg" << conf;
      if(!UniqueID()){ printf("Writing light Lanczos to %s\n",os.str().c_str()); fflush(stdout); }
      double time = -dclock();
      if(params.lanc_arg.N_true_get > 0) eig.writeParallel(os.str());
      time+=dclock();
      print_time("main","Light Lanczos write",time);
    }

    {//Do the light A2A vector random fields to ensure same ordering as unsplit job
      A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
#ifdef USE_DESTRUCTIVE_FFT
      W.allocModes();
#endif
      W.setWhRandom();
    }
    
    //-------------------- Strange quark Lanczos ---------------------//
    BFMGridLanczosWrapper<A2Apolicies> eig_s(solvers, params.jp);
    computeEvecs(eig_s, Heavy, params, cmdline.randomize_evecs);

    //-------------------- Strange quark v and w --------------------//
    A2AvectorV<A2Apolicies> V_s(params.a2a_arg_s,field4dparams);
    A2AvectorW<A2Apolicies> W_s(params.a2a_arg_s,field4dparams);

    BFMGridA2ALatticeWrapper<A2Apolicies> latwrp(solvers, params.jp);
    computeVW(V_s, W_s, Heavy, params, eig_s, latwrp, cmdline.randomize_vw);
    
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

    BFMGridLanczosWrapper<A2Apolicies> eig(solvers, params.jp);
    {
      std::ostringstream os; os << cmdline.checkpoint_dir << "/checkpoint.lanczos_l.cfg" << conf;
      if(!UniqueID()) printf("Reading light Lanczos from %s\n",os.str().c_str());
      double time = -dclock();
      eig.readParallel(os.str());      
      time+=dclock();
      print_time("main","Light Lanczos read",time);
    }

    //-------------------- Light quark v and w --------------------//
    A2AvectorV<A2Apolicies> V(params.a2a_arg, field4dparams);
    A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
    BFMGridA2ALatticeWrapper<A2Apolicies> latwrp(solvers, params.jp);
    computeVW(V, W, Light, params, eig, latwrp, cmdline.randomize_vw);
    
    eig.freeEvecs();
    printMem("Memory after light evec free");    
    
    //-------------------- Strange quark v and w read --------------------//
    A2AvectorV<A2Apolicies> V_s(params.a2a_arg_s,field4dparams);
    A2AvectorW<A2Apolicies> W_s(params.a2a_arg_s,field4dparams);
#ifdef USE_DESTRUCTIVE_FFT
    V_s.allocModes(); W_s.allocModes();
#endif
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
    Lattice& lat = (Lattice&)(*latwrp.a2a_lat);
    
    doContractions(conf,params,cmdline,lat,V,W,V_s,W_s,field3dparams);
  }else{ //part 1
    ERR.General("","doConfigurationSplit","Invalid part index %d\n", cmdline.split_job_part);
  }

    
}







#endif
