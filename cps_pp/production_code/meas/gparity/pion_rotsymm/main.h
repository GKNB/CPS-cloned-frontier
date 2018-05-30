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
}

void doGaugeFix(Lattice &lat, const bool skip_gauge_fix, const Parameters &params){
  doGaugeFix(lat,skip_gauge_fix,params.fix_gauge_arg);
}

template<typename MomentumPolicy>
void computePionMesonFields(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con,
			    typename computeMesonFieldsBase<A2Apolicies>::Vtype &V, typename computeMesonFieldsBase<A2Apolicies>::Wtype &W,
			    const MomentumPolicy &pion_mom,
			    Lattice &lat, const Parameters &params, const typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType &field3dparams){
  double time = -dclock();
#ifdef USE_POINT_SOURCES
  if(!UniqueID()) printf("Computing point pion meson fields\n");
  computeGparityLLmesonFieldsPoint<A2Apolicies,MomentumPolicy>::computeMesonFields(mf_ll_con, pion_mom, W, V, lat, field3dparams);
#else
  if(!UniqueID()) printf("Computing 1s pion meson fields\n");
  computeGparityLLmesonFields1s<A2Apolicies,MomentumPolicy>::computeMesonFields(mf_ll_con, pion_mom, W, V, params.jp.pion_rad, lat, field3dparams);
#endif

  time += dclock();
  print_time("main","Pion meson fields",time);

  printMem("Memory after pion meson field computation");
}

void readGaugeRNG(const Parameters &params, const CommandLineArgs &cmdline){
  readGaugeRNG(params.do_arg, params.meas_arg, cmdline.double_latt);
}
  

class NaivePionMomentaPolicy: public RequiredMomentum{
public:
  NaivePionMomentaPolicy(): RequiredMomentum() {
    this->combineSameTotalMomentum(true); //momentum pairs with same total momentum will be added to same entry and treated as 'alternates' which we average together below
    assert(this->nGparityDirs() == 3);

    //p_pi = (-2,-2,-2)     (units of pi/2L)
    addPandMinusP("(-1,-1,-1) + (-1,-1,-1)");

    //p_pi = (2,-2,-2)
    addPandMinusP("(-1,-1,-1) + (3,-1,-1)");
    
    //p_pi = (-2,2,-2)
    addPandMinusP("(-1,-1,-1) + (-1,3,-1)");
    
    //p_pi = (-2,-2,2)
    addPandMinusP("(-1,-1,-1) + (-1,-1,3)");
    
    assert(nMom() == 8);
    for(int i=0;i<8;i++) assert(nAltMom(i) == 1);
  }
};

class NaiveAltPionMomentaPolicy: public RequiredMomentum{
public:
  NaiveAltPionMomentaPolicy(): RequiredMomentum() {
    this->combineSameTotalMomentum(true); //momentum pairs with same total momentum will be added to same entry and treated as 'alternates' which we average together below

    assert(this->nGparityDirs() == 3);

    //p_pi = (-2,-2,-2)     (units of pi/2L)
    addPandMinusP("(1,1,1) + (-3,-3,-3)");

    //p_pi = (2,-2,-2)
    addPandMinusP("(1,1,1) + (1,-3,-3)");
    
    //p_pi = (-2,2,-2)
    addPandMinusP("(1,1,1) + (-3,1,-3)");
    
    //p_pi = (-2,-2,2)
    addPandMinusP("(1,1,1) + (-3,-3,1)");
    
    assert(nMom() == 8);
    for(int i=0;i<8;i++) assert(nAltMom(i) == 1);
  };
};


class NaiveReversePionMomentaPolicy: public NaivePionMomentaPolicy{
public:
  NaiveReversePionMomentaPolicy(): NaivePionMomentaPolicy() {
    this->reverseABmomentumAssignments();
  }
};



//Have a base + alt momentum, symmetrized. These satisfy the conditions p1+p2=p3+p4=ptot  and p1-p2 + p3-p4 = n*ptot  with n=-2
class NaiveExtendedPionMomentaPolicy: public RequiredMomentum{
public:
  NaiveExtendedPionMomentaPolicy(): RequiredMomentum() {
    this->combineSameTotalMomentum(true); //momentum pairs with same total momentum will be added to same entry and treated as 'alternates' which we average together below
    const int ngp = this->nGparityDirs();
    assert(ngp == 3);

    //For the (+-6,+-2,+-2) define the 8 orientations of (-6.-2,-2) obtained by giving each component a minus sign respectively, and then cyclically permute to move the -6 around
    std::vector<std::pair<ThreeMomentum, ThreeMomentum> > base(4);
    
    //(-6, -2, -2) (-1, -1, -1)+(-5, -1, -1) (1, 1, 1)+(-7, -3, -3)
    base[0] = ThreeMomentum::parse_str_two_mom("(-1, -1, -1)+(-5, -1, -1)");
    
    //(6, -2, -2) (-1, -1, -1)+(7, -1, -1) (1, 1, 1)+(5, -3, -3)
    base[1] = ThreeMomentum::parse_str_two_mom("(-1, -1, -1)+(7, -1, -1)");
    
    //(-6, 2, -2) (-1, -1, -1)+(-5, 3, -1) (1, 1, 1)+(-7, 1, -3)
    base[2] = ThreeMomentum::parse_str_two_mom("(-1, -1, -1)+(-5, 3, -1)");
    
    //(-6, -2, 2) (-1, -1, -1)+(-5, -1, 3) (1, 1, 1)+(-7, -3, 1)
    base[3] = ThreeMomentum::parse_str_two_mom("(-1, -1, -1)+(-5, -1, 3)");

    for(int perm=0;perm<3;perm++){
      for(int o=0;o<4;o++){ 
	addPandMinusP(base[o]);
	base[o].first.cyclicPermute();
	base[o].second.cyclicPermute();
      }
    }	
    assert(nMom() == 24);
  };
};

class NaiveAltExtendedPionMomentaPolicy: public RequiredMomentum{
public:
  NaiveAltExtendedPionMomentaPolicy(): RequiredMomentum() {
    this->combineSameTotalMomentum(true); //momentum pairs with same total momentum will be added to same entry and treated as 'alternates' which we average together below
    const int ngp = this->nGparityDirs();
    assert(ngp == 3);

    //For the (+-6,+-2,+-2) define the 8 orientations of (-6.-2,-2) obtained by giving each component a minus sign respectively, and then cyclically permute to move the -6 around
    std::vector<std::pair<ThreeMomentum, ThreeMomentum> > alt(4);
    
    //(-6, -2, -2) (-1, -1, -1)+(-5, -1, -1) (1, 1, 1)+(-7, -3, -3)
    alt[0] = ThreeMomentum::parse_str_two_mom("(1, 1, 1)+(-7, -3, -3)");
    
    //(6, -2, -2) (-1, -1, -1)+(7, -1, -1) (1, 1, 1)+(5, -3, -3)
    alt[1] = ThreeMomentum::parse_str_two_mom("(1, 1, 1)+(5, -3, -3)");
    
    //(-6, 2, -2) (-1, -1, -1)+(-5, 3, -1) (1, 1, 1)+(-7, 1, -3)
    alt[2] = ThreeMomentum::parse_str_two_mom("(1, 1, 1)+(-7, 1, -3)");
    
    //(-6, -2, 2) (-1, -1, -1)+(-5, -1, 3) (1, 1, 1)+(-7, -3, 1)
    alt[3] = ThreeMomentum::parse_str_two_mom("(1, 1, 1)+(-7, -3, 1)");

    for(int perm=0;perm<3;perm++){
      for(int o=0;o<4;o++){ 
	addPandMinusP(alt[o]);
	alt[o].first.cyclicPermute();
	alt[o].second.cyclicPermute();
      }
    }
    assert(nMom() == 24);
  };
};

class NaiveReverseExtendedPionMomentaPolicy: public NaiveExtendedPionMomentaPolicy{
public:
  NaiveReverseExtendedPionMomentaPolicy(): NaiveExtendedPionMomentaPolicy() {
    this->reverseABmomentumAssignments();
  }
};


//This experiment chooses alt momentum combinations that have the other quark boundary eigenstate but do not satisfy p1-p2 + p3-p4 = n*ptot
class NaiveAltExperimentalExtendedPionMomentaPolicy: public RequiredMomentum{
public:
  NaiveAltExperimentalExtendedPionMomentaPolicy(): RequiredMomentum() {
    this->combineSameTotalMomentum(true); //momentum pairs with same total momentum will be added to same entry and treated as 'alternates' which we average together below
    const int ngp = this->nGparityDirs();
    assert(ngp == 3);

    //For the (+-6,+-2,+-2) define the 8 orientations of (-6.-2,-2) obtained by giving each component a minus sign respectively, and then cyclically permute to move the -6 around
    std::vector<std::pair<ThreeMomentum, ThreeMomentum> > alt(4);
    
    //(-6, -2, -2) base (-1, -1, -1)+(-5, -1, -1)
    alt[0] = ThreeMomentum::parse_str_two_mom("(1, 1, -3)+(-7, -3, 1)");  //(4,0,0) + (8,4,-4) = (12,4,-4)
    
    //(6, -2, -2) base (-1, -1, -1)+(7, -1, -1)
    alt[1] = ThreeMomentum::parse_str_two_mom("(1, 1, -3)+(5, -3, 1)"); //(-8,0,0) + (-4,4,-4) = (-12,4,-4)
    
    //(-6, 2, -2) base (-1, -1, -1)+(-5, 3, -1)
    alt[2] = ThreeMomentum::parse_str_two_mom("(1, 1, -3)+(-7, 1, 1)"); //(4,-4,0) + (8,0,-4) = (12,-4,-4)
    
    //(-6, -2, 2) base (-1, -1, -1)+(-5, -1, 3)
    alt[3] = ThreeMomentum::parse_str_two_mom("(1, -3, 1)+(-7, 1, 1)"); //(4,0,-4) + (8,-4,0) = (12,-4,-4)

    for(int perm=0;perm<3;perm++){
      for(int o=0;o<4;o++){ 
	addPandMinusP(alt[o]);
	alt[o].first.cyclicPermute();
	alt[o].second.cyclicPermute();
      }
    }
    assert(nMom() == 24);
  };
};





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

#define DAIQIAN_PION_PHASE_CONVENTION

    std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_pioncorr_mom";
#ifndef DAIQIAN_PION_PHASE_CONVENTION
    os << pion_mom.getMesonMomentum(p).file_str(2);  //note the divisor of 2 is to put the momenta in units of pi/L and not pi/2L
#else
    os << (-pion_mom.getMesonMomentum(p)).file_str(2);
#endif
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

#ifdef USE_EXTENDED_MOMENTA
  NaiveExtendedPionMomentaPolicy naive_pion_mom;
  NaiveAltExtendedPionMomentaPolicy alt_pion_mom;
  NaiveReverseExtendedPionMomentaPolicy reverse_pion_mom;
#else
  NaivePionMomentaPolicy naive_pion_mom;
  NaiveAltPionMomentaPolicy alt_pion_mom;
  NaiveReversePionMomentaPolicy reverse_pion_mom;
#endif

  //Naive pion meson fields (no alt mom or symm)
  MesonFieldMomentumContainer<A2Apolicies> mf_pion_naive;
  computePionMesonFields(mf_pion_naive,  V, W, naive_pion_mom, lat, params, field3dparams);
  
  //Alt pion mom
  MesonFieldMomentumContainer<A2Apolicies> mf_pion_alt;
  computePionMesonFields(mf_pion_alt,  V, W, alt_pion_mom, lat, params, field3dparams);
  
  //Reverse pion mom
  MesonFieldMomentumContainer<A2Apolicies> mf_pion_rev;
  computePionMesonFields(mf_pion_rev,  V, W, reverse_pion_mom, lat, params, field3dparams);
  

  //Combine naive with alt and naive with reverse
  MesonFieldMomentumContainer<A2Apolicies> &mf_pion_naive_plus_alt = mf_pion_alt;
  mf_pion_naive_plus_alt.average(mf_pion_naive);

  MesonFieldMomentumContainer<A2Apolicies> &mf_pion_naive_plus_rev = mf_pion_rev;
  mf_pion_naive_plus_rev.average(mf_pion_naive);

  //Compute pion 2pt func
  computePion2pt(mf_pion_naive, naive_pion_mom, conf, params, "_naive");
  computePion2pt(mf_pion_naive_plus_alt, naive_pion_mom, conf, params, "_naive_plus_alt");
  computePion2pt(mf_pion_naive_plus_rev, naive_pion_mom, conf, params, "_naive_plus_rev");

#ifdef DO_EXTENDED_MOMENTUM_EXPT
  NaiveAltExperimentalExtendedPionMomentaPolicy expt_pion_mom;
  MesonFieldMomentumContainer<A2Apolicies> mf_pion_expt;
  computePionMesonFields(mf_pion_expt,  V, W, expt_pion_mom, lat, params, field3dparams);

  MesonFieldMomentumContainer<A2Apolicies> &mf_pion_naive_plus_expt = mf_pion_expt;
  mf_pion_naive_plus_expt.average(mf_pion_naive);
  computePion2pt(mf_pion_naive_plus_expt, naive_pion_mom, conf, params, "_naive_plus_expt");
#endif

}

#endif
