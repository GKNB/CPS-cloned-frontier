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
  if(!UniqueID()) printf("Computing light quark A2A vectors\n");
  double time = -dclock();

  a2a_lat.computeVW(V,W,eig,params.jp.cg_controls,randomize_vw);
  
  printMem("Memory after light A2A vector computation");

  time += dclock();
  std::ostringstream os; os << "Light quark A2A vectors";
  print_time("main",os.str().c_str(),time);

  V.writeParallel("a2aVectorV.dat");
  W.writeParallel("a2aVectorW.dat");
}

void doGaugeFix(Lattice &lat, const bool skip_gauge_fix, const Parameters &params){
  doGaugeFix(lat,skip_gauge_fix,params.fix_gauge_arg);
}

template<typename MomentumPolicy>
void computeSigmaMesonFields(MesonFieldMomentumPairContainer<A2Apolicies> &mf_ll_con,
			    typename computeMesonFieldsBase<A2Apolicies>::Vtype &V, typename computeMesonFieldsBase<A2Apolicies>::Wtype &W,
			    const MomentumPolicy &pion_mom,
			    Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  double time = -dclock();
  if(!UniqueID()) printf("Computing 1s sigma meson fields\n");
  computeSigmaMesonFields1s<A2Apolicies,MomentumPolicy>::computeMesonFields(mf_ll_con, pion_mom, W, V, params.jp.pion_rad, lat, field3dparams);

  time += dclock();
  print_time("main","Sigma meson fields",time);

  printMem("Memory after pion meson field computation");
}

void readGaugeRNG(const Parameters &params, const CommandLineArgs &cmdline){
  readGaugeRNG(params.do_arg, params.meas_arg, cmdline.double_latt);
}
  
//(4,0,0)pi/2L + perms. Use parity to avoid computing (4,0,0)
//We symmetrize but do not do alternative momenta because the meson fields contain both psi_+ and psi_- quark fields naturally
class MovingSigmaMomentaPolicy400: public RequiredMomentum{
public:
  MovingSigmaMomentaPolicy400(): RequiredMomentum() {
    const int ngp = this->nGparityDirs();
    assert(ngp==3);
   
    std::pair<ThreeMomentum,ThreeMomentum> mom_pair = ThreeMomentum::parse_str_two_mom("(1,1,1) + (3,-1,-1)");
    for(int perm=0; perm<3; perm++){
      addPandMinusP(mom_pair);
      mom_pair.first.cyclicPermute();
      mom_pair.second.cyclicPermute();
    }

    this->symmetrizeABmomentumAssignments();
  }
};

class MovingSigmaMomentaPolicy400Alt: public RequiredMomentum{
public:
  MovingSigmaMomentaPolicy400Alt(): RequiredMomentum() {
    const int ngp = this->nGparityDirs();
    assert(ngp==3);

    std::pair<ThreeMomentum,ThreeMomentum> mom_pair = ThreeMomentum::parse_str_two_mom("(-1,-1,-1) + (5,1,1)");
    for(int perm=0; perm<3; perm++){
      addPandMinusP(mom_pair);
      mom_pair.first.cyclicPermute();
      mom_pair.second.cyclicPermute();
    }

    this->symmetrizeABmomentumAssignments();
  }
};

//(4,4,0),  (4,-4,0)  + perms. Use parity to avoid computing negatives
//We symmetrize but do not do alternative momenta because the meson fields contain both psi_+ and psi_- quark fields naturally
//Choose p1 in psi_+  (1,1,1) + 4(n1,n2,n3)  momentum set
class MovingSigmaMomentaPolicy440: public RequiredMomentum{
public:
  MovingSigmaMomentaPolicy440(): RequiredMomentum() {
    const int ngp = this->nGparityDirs();

    assert(ngp==3);

    std::pair<ThreeMomentum,ThreeMomentum> mom_pair_same = ThreeMomentum::parse_str_two_mom("(1,1,1) + (3,3,-1)");
    std::pair<ThreeMomentum,ThreeMomentum> mom_pair_diff = ThreeMomentum::parse_str_two_mom("(3,-1,-1) + (1,-3,1)");
    for(int perm=0; perm<3; perm++){
      addPandMinusP(mom_pair_same);
      mom_pair_same.first.cyclicPermute();
      mom_pair_same.second.cyclicPermute();
      addPandMinusP(mom_pair_diff);
      mom_pair_diff.first.cyclicPermute();
      mom_pair_diff.second.cyclicPermute();
    } 
    
    this->symmetrizeABmomentumAssignments();
  }
};

class MovingSigmaMomentaPolicy440Alt: public RequiredMomentum{
public:
  MovingSigmaMomentaPolicy440Alt(): RequiredMomentum() {
    const int ngp = this->nGparityDirs();
    assert(ngp==3);

    std::pair<ThreeMomentum,ThreeMomentum> mom_pair_same = ThreeMomentum::parse_str_two_mom("(3,-1,-1) + (1,5,1)");
    std::pair<ThreeMomentum,ThreeMomentum> mom_pair_diff = ThreeMomentum::parse_str_two_mom("(3,-5,-1) + (1,1,1)");
    for(int perm=0; perm<3; perm++){
      addPandMinusP(mom_pair_same);
      mom_pair_same.first.cyclicPermute();
      mom_pair_same.second.cyclicPermute();
      addPandMinusP(mom_pair_diff);
      mom_pair_diff.first.cyclicPermute();
      mom_pair_diff.second.cyclicPermute();
    }

    this->symmetrizeABmomentumAssignments();
  }
};


//(4,4,4),  (4,4,-4),  (4,-4,4),   (-4,4,4). Use parity to avoid computing negatives
//We symmetrize but do not do alternative momenta because the meson fields contain both psi_+ and psi_- quark fields naturally
//Choose p1 in psi_+  (1,1,1) + 4(n1,n2,n3)  momentum set
class MovingSigmaMomentaPolicy444: public RequiredMomentum{
public:
  MovingSigmaMomentaPolicy444(): RequiredMomentum() {
    const int ngp = this->nGparityDirs();
    assert(ngp==3);

    std::pair<ThreeMomentum,ThreeMomentum> mom_pair_diff = ThreeMomentum::parse_str_two_mom("(3,3,-1) + (1,1,-3)");
    for(int perm=0; perm<3; perm++){
      addPandMinusP(mom_pair_diff);
      mom_pair_diff.first.cyclicPermute();
      mom_pair_diff.second.cyclicPermute();
    }
    std::pair<ThreeMomentum,ThreeMomentum> mom_pair_same = ThreeMomentum::parse_str_two_mom("(3,3,3) + (1,1,1)");
    addPandMinusP(mom_pair_same);

    this->symmetrizeABmomentumAssignments();
  }
};

class MovingSigmaMomentaPolicy444Alt: public RequiredMomentum{
public:
  MovingSigmaMomentaPolicy444Alt(): RequiredMomentum() {
    const int ngp = this->nGparityDirs();
    assert(ngp==3);

    std::pair<ThreeMomentum,ThreeMomentum> mom_pair_diff = ThreeMomentum::parse_str_two_mom("(3,-1,-1) + (1,5,-3)");
    for(int perm=0; perm<3; perm++){
      addPandMinusP(mom_pair_diff);
      mom_pair_diff.first.cyclicPermute();
      mom_pair_diff.second.cyclicPermute();
    }
    std::pair<ThreeMomentum,ThreeMomentum> mom_pair_same = ThreeMomentum::parse_str_two_mom("(3,3,-1) + (1,1,5)");
    addPandMinusP(mom_pair_same);

    this->symmetrizeABmomentumAssignments();
  }
};

template<typename Base, typename Alt>
class MovingSigmaMomentaPolicyCombine: public RequiredMomentum{
public:
  MovingSigmaMomentaPolicyCombine(): RequiredMomentum() {
    this->combineSameTotalMomentum(true);
    Base base;
    Alt alt;
    this->addAll(base);
    this->addAll(alt);
  }
};



//Moving sigma meson fields can be uniquely indexed by their total momentum
template<typename MomentumPolicy>
void computeMovingSigmaMesonFields1s(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con,
				     typename computeMesonFieldsBase<A2Apolicies>::Vtype &V, typename computeMesonFieldsBase<A2Apolicies>::Wtype &W,
				     const MomentumPolicy &mom,
				     Lattice &lat, const Parameters &params, 
				     const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  if(!UniqueID()) printf("Computing 1s moving sigma meson fields\n");
  double time = -dclock();

  typedef computeGparityLLmesonFields1sSumOnTheFly<A2Apolicies, MomentumPolicy, 0, sigma0> computeType;
  assert(GJP.Gparity());
  computeType::computeMesonFields(mf_ll_con, mom, W, V, params.jp.pion_rad, lat, field3dparams);

  time += dclock();
  print_time("main","1s moving sigma meson fields",time);

  printMem("Memory after 1s moving sigma meson field computation");
}

//Compute sigma 2pt function with file in Tianle's format for sigma meson fields indexed by their total momentum
template<typename SigmaMomentumPolicy>
void computeMovingSigma2pt(std::vector< fVector<typename A2Apolicies::ScalarComplexType> > &sigma_bub, //output bubble
			   MesonFieldMomentumContainer<A2Apolicies> &mf_sigma_con, const SigmaMomentumPolicy &sigma_mom, const int conf, const Parameters &params){
  const int nmom = sigma_mom.nMom();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();

  if(!UniqueID()) printf("Computing movingsigma 2pt function\n");
  double time = -dclock();

  sigma_bub.resize(nmom);
  for(int pidx=0;pidx<nmom;pidx++){
    //Compute the disconnected bubble
    if(!UniqueID()) printf("Sigma disconnected bubble pidx=%d\n",pidx);
    fVector<typename A2Apolicies::ScalarComplexType> &into = sigma_bub[pidx]; into.resize(Lt);
    ComputeSigmaContractions<A2Apolicies>::computeDisconnectedBubble(into, mf_sigma_con, sigma_mom, pidx);

    std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf;
    os  << "_sigmaself_ptot" << sigma_mom.getTotalMomentum(pidx).file_str(1); //MOVE TO DAIQIAN'S CONVENTIONS
    into.write(os.str());
# ifdef WRITE_HEX_OUTPUT
    os << ".hexfloat";
    into.write(os.str(),true);
# endif
  }

  for(int pidx=0;pidx<nmom;pidx+=2){ //odd indices have equal and opposite total momentum
    assert(sigma_mom.getTotalMomentum(pidx+1) == -sigma_mom.getTotalMomentum(pidx));

    //Connected part
    if(!UniqueID()) printf("Sigma connected pidx=%d\n",pidx);
    fMatrix<typename A2Apolicies::ScalarComplexType> into(Lt,Lt);
    ComputeSigmaContractions<A2Apolicies>::computeConnected(into, mf_sigma_con, sigma_mom, pidx);

    fMatrix<typename A2Apolicies::ScalarComplexType> disconn(Lt,Lt);
    ComputeSigmaContractions<A2Apolicies>::computeDisconnectedDiagram(disconn, sigma_bub[pidx], sigma_bub[pidx+1]);

    into += disconn;

    std::ostringstream os;
    os
      << params.meas_arg.WorkDirectory << "/traj_" << conf << "_sigmacorr_ptot"
      << sigma_mom.getTotalMomentum(pidx).file_str(2);

    into.write(os.str());
# ifdef WRITE_HEX_OUTPUT
    os << ".hexfloat";
    into.write(os.str(),true);
# endif
  }
        
  time += dclock();
  print_time("main","Moving Sigma 2pt function",time);

  printMem("Memory after Moving Sigma 2pt function computation");
}


template<typename MomentumPolicy>
void computePionMesonFields1s(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con,
			      typename computeMesonFieldsBase<A2Apolicies>::Vtype &V, typename computeMesonFieldsBase<A2Apolicies>::Wtype &W,
			      const MomentumPolicy &pion_mom,
			      Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  double time = -dclock();
  if(!UniqueID()) printf("Computing 1s pion meson fields\n");
  computeGparityLLmesonFields1sSumOnTheFly<A2Apolicies,MomentumPolicy,15,sigma3>::computeMesonFields(mf_ll_con, pion_mom, W, V, params.jp.pion_rad, lat, field3dparams);
  time += dclock();
  print_time("main","Pion meson fields",time);

  printMem("Memory after pion meson field computation");
}

template<typename PionMomentumPolicy>
void computePion2pt(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, const PionMomentumPolicy &pion_mom, const int conf, const Parameters &params, const std::string &postpend = ""){
  const int nmom = pion_mom.nMom();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();
  
  if(!UniqueID()) printf("Computing pion 2pt function\n");
  double time = -dclock();
  for(int p=0;p<nmom;p+=2){ //note odd indices 1,3,5 etc have equal and opposite momenta to 0,2,4... 
    if(!UniqueID()) printf("Starting pidx %d\n",p);
    fMatrix<typename A2Apolicies::ScalarComplexType> pion(Lt,Lt);
    ComputePion<A2Apolicies>::compute(pion, mf_ll_con, pion_mom, p);

    std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_pioncorr_mom";
    os << (-pion_mom.getMesonMomentum(p)).file_str(2);
    os << postpend;
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

template<typename SigmaMomentumPolicy, typename PionMomentumPolicy>
void computeMovingPipiToSigma(MesonFieldMomentumContainer<A2Apolicies> &mf_sigma_con, const SigmaMomentumPolicy &sigma_mom, const int pidx_sigma,
			      MesonFieldMomentumContainer<A2Apolicies> &mf_pion_con, const PionMomentumPolicy &pion_mom, 
			      const int conf, const Parameters &params,
			      const std::vector< fVector<typename A2Apolicies::ScalarComplexType> > &sigma_bub,
			      const std::string &postpend = ""){
  const int nmom_sigma = sigma_mom.nMom();
  const int nmom_pion = pion_mom.nMom();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();

  if(!UniqueID()) printf("Computing moving pipi->sigma\n");
  double time = -dclock();

  //Get total source momentum
  ThreeMomentum p_sigma = sigma_mom.getMesonMomentum(pidx_sigma);
  ThreeMomentum ptot_src = -p_sigma;

  //Figure out what combinations of pion momenta give this total momentum
  std::vector< std::pair<int, int> > pipi_mompairs;
  for(int i=0;i<nmom_pion;i++){
    for(int j=0;j<nmom_pion;j++){ //want both pi1=A pi2=B  and pi1=B pi2=A
      if(pion_mom.getMesonMomentum(i) + pion_mom.getMesonMomentum(j) == ptot_src) pipi_mompairs.push_back({i,j});      
    }
  }
  
  //Get pipi bubbles
  std::vector< fVector<typename A2Apolicies::ScalarComplexType> > pipi_bub(pipi_mompairs.size());
  for(int i=0;i<pipi_mompairs.size();i++){
    ThreeMomentum p1 = pion_mom.getMesonMomentum(pipi_mompairs[i].first), p2 = pion_mom.getMesonMomentum(pipi_mompairs[i].second);
    printMem(stringize("Doing pipi figure V, p1=%s p2=%s",p1.str().c_str(), p2.str().c_str() ),0);
    ComputePiPiGparity<A2Apolicies>::computeFigureVdis(pipi_bub[i], p1, p2, params.jp.pipi_separation, mf_pion_con);
    std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_FigureVdis_sep" << params.jp.pipi_separation
			      << "_pi1mom" << (-p1).file_str(2) << "_pi2mom" << (-p2).file_str(2)
			      << postpend;  //Daiqian's annoying phase convention

    pipi_bub[i].write(os.str());
#ifdef WRITE_HEX_OUTPUT
    os << ".hexfloat";
    pipi_bub[i].write(os.str(),true);
#endif	
  }

  for(int ppipair_idx=0; ppipair_idx<pipi_mompairs.size(); ppipair_idx++){
    int p_pi1_idx = pipi_mompairs[ppipair_idx].first,  p_pi2_idx = pipi_mompairs[ppipair_idx].second;
    ThreeMomentum p_pi1 = pion_mom.getMesonMomentum(p_pi1_idx), p_pi2 = pion_mom.getMesonMomentum(p_pi2_idx);

    //We save the connected part separately from the disconnected part because it is only computed on timeslices that are multiples of tstep
    //Connected
    {
      if(!UniqueID()) printf("Pipi->sigma connected p_tot=%s    p_pi1=%s p_pi2=%s\n", ptot_src.str().c_str(), p_pi1.str().c_str(), p_pi2.str().c_str());
      fMatrix<typename A2Apolicies::ScalarComplexType> into(Lt,Lt);
      
      ComputePiPiToSigmaContractions<A2Apolicies>::computeConnected(into, mf_sigma_con, sigma_mom, mf_pion_con, pion_mom, 
								    p_pi1_idx, p_pi2_idx,
								    params.jp.pipi_separation, params.jp.tstep_pipi); //reuse same tstep currently
      std::ostringstream os;
      os << params.meas_arg.WorkDirectory << 
	"/traj_" << conf << "_pipitosigma_ptot" << ptot_src.file_str() <<
	"_pi1mom" << (-p_pi1).file_str(2) << "_pi2mom" << (-p_pi2).file_str(2) << "_conn"; //Daiqian's annoying phase convention
      
      into.write(os.str());
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      into.write(os.str(),true);
#endif
    }
    //Disconnected part
    {
      if(!UniqueID()) printf("Pipi->sigma disconnected p_tot=%s    p_pi1=%s p_pi2=%s\n", ptot_src.str().c_str(), p_pi1.str().c_str(), p_pi2.str().c_str());
      
      fMatrix<typename A2Apolicies::ScalarComplexType> into(Lt,Lt);
      ComputePiPiToSigmaContractions<A2Apolicies>::computeDisconnectedDiagram(into, sigma_bub[pidx_sigma], pipi_bub[ppipair_idx], 1); //compute on every timeslice

/*       { */
/* 	int Lt = GJP.Tnodes()*GJP.TnodeSites(); */
/* 	into.resize(Lt,Lt); into.zero(); */
/* 	double coeff = -sqrt(6.)/2; */
/* #ifdef USE_TIANLES_CONVENTIONS */
/* 	coeff *= -1.; */
/* #endif */
/* 	auto const &sigma_bubble = sigma_bub[pidx_sigma]; */
/* 	auto const &pipi_bubble = pipi_bub[ppipair_idx]; */

/* 	assert(sigma_bubble.size() == Lt); */
/* 	assert(pipi_bubble.size() == Lt); */
	
/* 	for(int tsrc=0; tsrc<Lt; tsrc++){ */
/* 	  for(int tsep=0; tsep<Lt; tsep++){ */
/* 	    int tsnk = (tsrc + tsep) % Lt; */
/* 	    std::cout << tsrc << " " << tsep << " " << &sigma_bubble(tsnk) << " " << &pipi_bubble(tsrc) << std::endl; */

/* 	    into(tsrc, tsep) = coeff * sigma_bubble(tsnk) * pipi_bubble(tsrc); */
/* 	  } */
/* 	} */
/*       } */




      std::ostringstream os;
      os << params.meas_arg.WorkDirectory << 
	"/traj_" << conf << "_pipitosigma_ptot" << ptot_src.file_str() <<
	"_pi1mom" << (-p_pi1).file_str(2) << "_pi2mom" << (-p_pi2).file_str(2) << "_disconn"; 
      
      into.write(os.str());
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      into.write(os.str(),true);
#endif
    }
  }
        
  time += dclock();
  print_time("main","Moving PiPi->Sigma",time);

  printMem("Memory after Moving PiPi->Sigma computation");
}



void doConfiguration(const int conf, Parameters &params, const CommandLineArgs &cmdline,
		     const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams,
		     const typename A2Apolicies::FermionFieldType::InputParamType &field4dparams, BFMGridSolverWrapper &solvers){

  params.meas_arg.TrajCur = conf;

  std::string dir(params.meas_arg.WorkDirectory);

  //-------------------- Read gauge field --------------------//
  readGaugeRNG(params,cmdline);
    
  printMem("Memory after gauge and RNG read");

  if(cmdline.tune_lanczos) LanczosTune(params, solvers);    

  bool need_lanczos = true;
  if(cmdline.randomize_vw) need_lanczos = false;
  if(cmdline.force_evec_compute) need_lanczos = true;
 
  //-------------------- Light quark Lanczos ---------------------//
  BFMGridLanczosWrapper<A2Apolicies> eig(solvers, params.jp);
  if(need_lanczos) computeEvecs(eig, params.lanc_arg, params.jp, "light", cmdline.randomize_evecs);
  
  //-------------------- Light quark v and w --------------------//
  A2AvectorV<A2Apolicies> V(params.a2a_arg, field4dparams);
  A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
  
  BFMGridA2ALatticeWrapper<A2Apolicies> latwrp(solvers, params.jp); //lattice created temporarily
  computeVW(V, W, params, eig, latwrp, cmdline.randomize_vw);

  if(!UniqueID()){ printf("Freeing light evecs\n"); fflush(stdout); }
  eig.freeEvecs();
  printMem("Memory after light evec free");
    
  //From now one we just need a generic lattice instance, so use a2a_lat
  Lattice& lat = (Lattice&)(*latwrp.a2a_lat);
    
  //-------------------Fix gauge----------------------------
  doGaugeFix(lat, cmdline.skip_gauge_fix, params);
  
  //-------------------Get sigma momentum policies and meson fields-----------
  MovingSigmaMomentaPolicyCombine<MovingSigmaMomentaPolicy400, MovingSigmaMomentaPolicy400Alt> sigma_mom_400;
  MovingSigmaMomentaPolicyCombine<MovingSigmaMomentaPolicy440, MovingSigmaMomentaPolicy440Alt> sigma_mom_440;
  MovingSigmaMomentaPolicyCombine<MovingSigmaMomentaPolicy444, MovingSigmaMomentaPolicy444Alt> sigma_mom_444;
  
  RequiredMomentum sigma_mom_all; 
  {
    sigma_mom_all.combineSameTotalMomentum(true);
    sigma_mom_all.addAll(sigma_mom_400);
    sigma_mom_all.addAll(sigma_mom_440);
    sigma_mom_all.addAll(sigma_mom_444);
  }
  sigma_mom_all.print("All sigma momenta");

  MesonFieldMomentumContainer<A2Apolicies> mf_sigma;
  computeMovingSigmaMesonFields1s(mf_sigma, V, W, sigma_mom_all, lat, params, field3dparams);

  //------------------Sigma 2pt functions--------------------------------------
  std::vector< fVector<typename A2Apolicies::ScalarComplexType> > sigma_bub_400, sigma_bub_440, sigma_bub_444;
  computeMovingSigma2pt(sigma_bub_400, mf_sigma, sigma_mom_400, conf, params);
  computeMovingSigma2pt(sigma_bub_440, mf_sigma, sigma_mom_440, conf, params);
  computeMovingSigma2pt(sigma_bub_444, mf_sigma, sigma_mom_444, conf, params);

  SymmetricPionMomentaPolicy pion_mom_111;
  AltExtendedPionMomentaPolicy pion_mom_311;
  
  RequiredMomentum pion_mom_all; 
  {
    pion_mom_all.combineSameTotalMomentum(true);
    pion_mom_all.addAll(pion_mom_111);
    pion_mom_all.addAll(pion_mom_311);
  }
  //-----------------Pion meson fields----------------------------------------
  MesonFieldMomentumContainer<A2Apolicies> mf_pion;
  computePionMesonFields1s(mf_pion,  V, W, pion_mom_all, lat, params, field3dparams);
  
  //-----------------Pion 2pt------------------------------------------------
  computePion2pt(mf_pion, pion_mom_all, conf, params);

  //------------------PiPi->Sigma---------------------------------------------
  for(int i=0;i<sigma_mom_400.nMom();i++)
    computeMovingPipiToSigma(mf_sigma, sigma_mom_400, i, mf_pion, pion_mom_all, conf, params, sigma_bub_400);
  for(int i=0;i<sigma_mom_440.nMom();i++)
    computeMovingPipiToSigma(mf_sigma, sigma_mom_440, i, mf_pion, pion_mom_all, conf, params, sigma_bub_440);
  for(int i=0;i<sigma_mom_444.nMom();i++)
    computeMovingPipiToSigma(mf_sigma, sigma_mom_444, i, mf_pion, pion_mom_all, conf, params, sigma_bub_444);


}

#endif
