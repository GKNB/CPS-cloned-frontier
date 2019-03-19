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
  A2AArg a2a_arg_s;
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
    if(!lanc_arg.Decode("lanc_arg.vml","lanc_arg")){
      lanc_arg.Encode("lanc_arg.templ","lanc_arg");
      VRB.Result("Parameters","Parameters","Can't open lanc_arg.vml!\n");exit(1);
    }
    if(!a2a_arg.Decode("a2a_arg.vml","a2a_arg")){
      a2a_arg.Encode("a2a_arg.templ","a2a_arg");
      VRB.Result("Parameters","Parameters","Can't open a2a_arg.vml!\n");exit(1);
    }
    if(!lanc_arg_s.Decode("lanc_arg_s.vml","lanc_arg")){
      lanc_arg_s.Encode("lanc_arg_s.templ","lanc_arg");
      VRB.Result("Parameters","Parameters","Can't open lanc_arg_s.vml!\n");exit(1);
    }
    if(!a2a_arg_s.Decode("a2a_arg_s.vml","a2a_arg")){
      a2a_arg_s.Encode("a2a_arg_s.templ","a2a_arg");
      VRB.Result("Parameters","Parameters","Can't open a2a_arg_s.vml!\n");exit(1);
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
}\


enum LightHeavy { Light, Heavy };

void computeVW(A2AvectorV<A2Apolicies> &V, A2AvectorW<A2Apolicies> &W, const LightHeavy lh, const Parameters &params,
	       const BFMGridLanczosWrapper<A2Apolicies> &eig, const BFMGridA2ALatticeWrapper<A2Apolicies> &a2a_lat,
	       const bool randomize_vw){
  const A2AArg &a2a_arg = lh == Light ? params.a2a_arg : params.a2a_arg_s;  
  const char* name = (lh ==  Light ? "light" : "heavy");
  
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

void readGaugeRNG(const Parameters &params, const CommandLineArgs &cmdline){
  readGaugeRNG(params.do_arg, params.meas_arg, cmdline.double_latt);
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
  
  {
    BFMGridA2ALatticeWrapper<A2Apolicies> latwrp(solvers, params.jp); //lattice created temporarily
    computeVW(V, W, Light, params, eig, latwrp, cmdline.randomize_vw);
  }
  eig.freeEvecs();
  printMem("Memory after light evec free");

  //-------------------- Strange quark Lanczos ---------------------//
  BFMGridLanczosWrapper<A2Apolicies> eig_s(solvers, params.jp);
  if(need_lanczos) computeEvecs(eig_s, params.lanc_arg_s, params.jp, "heavy",cmdline.randomize_evecs);

  //--------------------- Strange quark V and W
  A2AvectorV<A2Apolicies> V_s(params.a2a_arg_s,field4dparams);
  A2AvectorW<A2Apolicies> W_s(params.a2a_arg_s,field4dparams);

  BFMGridA2ALatticeWrapper<A2Apolicies> latwrp(solvers, params.jp);
  computeVW(V_s, W_s, Heavy, params, eig_s, latwrp, cmdline.randomize_vw);
  
  eig_s.freeEvecs();
  printMem("Memory after heavy evec free");
   
  //From now one we just need a generic lattice instance, so use a2a_lat
  Lattice& lat = (Lattice&)(*latwrp.a2a_lat);
    
  //-------------------Fix gauge----------------------------
  doGaugeFix(lat, cmdline.skip_gauge_fix, params);

  //----------------------------Compute the sigma meson fields--------------------------------
  StationarySigmaMomentaPolicy sigma_mom;
  MesonFieldMomentumPairContainer<A2Apolicies> mf_sigma;
  computeSigmaMesonFields1s<A2Apolicies, StationarySigmaMomentaPolicy>::computeMesonFields(mf_sigma, sigma_mom, W, V, params.jp.pion_rad, lat, field3dparams);
    
  //----------------------------Compute the pion meson fields
  SymmetricPionMomentaPolicy pion_mom;
  MesonFieldMomentumContainer<A2Apolicies> mf_pion;
  computeGparityLLmesonFields1s<A2Apolicies, SymmetricPionMomentaPolicy>::computeMesonFields(mf_pion, pion_mom, W, V, params.jp.pion_rad, lat, field3dparams);

  //--------------------------- Compute the WW meson fields ----------------------------------
  StandardLSWWmomentaPolicy ww_mom;
  std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww;
  ComputeKtoPiPiGparity<A2Apolicies>::generatelsWWmesonfields(mf_ls_ww, W,W_s, ww_mom, params.jp.pion_rad, lat, field3dparams);


  //----------------------------Pre-average over sigma meson fields---------------------------
  const int Lt=GJP.Tnodes()*GJP.TnodeSites();
  std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_sigma_avg(Lt);
  
  for(int s = 0; s< sigma_mom.nMom(); s++){
    ThreeMomentum pwdag = sigma_mom.getWdagMom(s);
    ThreeMomentum pv = sigma_mom.getVmom(s);

    std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > &mf_sigma_s = mf_sigma.get(pwdag,pv);

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeGetMany(1, &mf_sigma_s);
#endif
    
    if(s==0)
      for(int t=0;t<Lt;t++) mf_sigma_avg[t] = mf_sigma_s[t];
    else
      for(int t=0;t<Lt;t++) mf_sigma_avg[t].plus_equals(mf_sigma_s[t]);


#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(1,&mf_sigma_s);
#endif
  }

  //-------------------------- Compute K->sigma using my methods ----------------------------
  if(!UniqueID()){ printf("Starting K->sigma (CK version)\n"); fflush(stdout); }

  std::vector<int> k_sigma_separation(params.jp.k_pi_separation.k_pi_separation_len);
  for(int i=0;i<params.jp.k_pi_separation.k_pi_separation_len;i++) k_sigma_separation[i] = params.jp.k_pi_separation.k_pi_separation_val[i];

  ComputeKtoSigma<A2Apolicies> ktosigma_ck(V,W,V_s,W_s,mf_ls_ww,k_sigma_separation);

  typedef ComputeKtoPiPiGparity<A2Apolicies>::ResultsContainerType ResultsContainerTypeCK;
  typedef ComputeKtoPiPiGparity<A2Apolicies>::MixDiagResultsContainerType MixDiagResultsContainerTypeCK;
  
  std::vector<ResultsContainerTypeCK> type12_ck;
  ktosigma_ck.type12(type12_ck, mf_sigma_avg);

  std::vector<ResultsContainerTypeCK> type3_ck;
  std::vector<MixDiagResultsContainerTypeCK> mix3_ck;
  ktosigma_ck.type3(type3_ck, mix3_ck, mf_sigma_avg);
  
  ResultsContainerTypeCK type4_ck;
  MixDiagResultsContainerTypeCK mix4_ck;
  ktosigma_ck.type4(type4_ck, mix4_ck);

  //---------------------------Compute K->sigma using Tianle's methods
  if(!UniqueID()){ printf("Starting K->sigma (TW version)\n"); fflush(stdout); }

  typedef ComputeKtoSigmaGparityTianle<A2Apolicies>::ResultsContainerType ResultsContainerTypeTW;
  typedef ComputeKtoSigmaGparityTianle<A2Apolicies>::MixResultsContainerType MixResultsContainerTypeTW;

  std::vector<ResultsContainerTypeTW> type_1234_tw;
  std::vector<MixResultsContainerTypeTW> mix_34_tw;
  
  //int tsep_max = -1; for(int i=0;i<k_sigma_separation.size();i++) if(k_sigma_separation[i] > tsep_max) tsep_max = k_sigma_separation[i];

  // std::vector<int> tsep(tsep_max); 
  // if(!UniqueID()){ printf("tsep values: "); fflush(stdout); }
  // for(int t=0;t<tsep_max;t++){
  //   if(!UniqueID()){ printf(" %d", t); }
  //   tsep[t] = t;
  // }
  if(!UniqueID()){ printf("\n"); fflush(stdout); }

  ComputeKtoSigmaGparityTianle<A2Apolicies>::compute(type_1234_tw, mix_34_tw, k_sigma_separation, 1, mf_ls_ww, mf_sigma_avg, V,V_s,W,W_s);

  int type12_idx[5] = {1,6,8,11,19};

  //-------------------------Compare---------------------------------------------
  if(!UniqueID()){ printf("Comparing type1/2\n"); fflush(stdout); }
  for(int ks_sep_idx=0; ks_sep_idx<k_sigma_separation.size(); ks_sep_idx++){
    for(int d=0;d<5;d++){
      for(int gidx=0;gidx<8;gidx++){
	for(int tk=0;tk<Lt;tk++){
	  for(int tdis=1; tdis< k_sigma_separation[ks_sep_idx]; tdis++){
	    std::complex<double> ck = convertComplexD(type12_ck[ks_sep_idx](tk,tdis,d,gidx));
	    std::complex<double> tw = convertComplexD(type_1234_tw[ks_sep_idx](tk,tdis,type12_idx[d],gidx));
	    double rdr = 2.*(tw.real()-ck.real())/(tw.real()+ck.real());
	    double rdi = 2.*(tw.imag()-ck.imag())/(tw.imag()+ck.imag());
	    std::string msg = "";
	    if(fabs(rdr) > 1e-12 || fabs(rdi) > 1e-12) msg = " ERROR";
	    if(!UniqueID()) printf("tsep %d D %d G %d tK %d tdis %d CK (%g %g) TW (%g %g) reldiff (%g,%g)%s\n",k_sigma_separation[ks_sep_idx],type12_idx[d],gidx,tk,tdis,ck.real(),ck.imag(),tw.real(),tw.imag(),rdr,rdi,msg.c_str());
	  }
	}
      }      
    }
  }

  if(!UniqueID()){ printf("Comparing type3\n"); fflush(stdout); }
  int type3_idx[9] = {2, 3, 7, 10, 14, 16, 18, 21, 23};

  for(int ks_sep_idx=0; ks_sep_idx<k_sigma_separation.size(); ks_sep_idx++){
    for(int d=0;d<9;d++){
      for(int gidx=0;gidx<8;gidx++){
	for(int tk=0;tk<Lt;tk++){
	  for(int tdis=1; tdis< k_sigma_separation[ks_sep_idx]; tdis++){
	    std::complex<double> ck = convertComplexD(type3_ck[ks_sep_idx](tk,tdis,d,gidx));
	    std::complex<double> tw = convertComplexD(type_1234_tw[ks_sep_idx](tk,tdis,type3_idx[d],gidx));
	    double rdr = 2.*(tw.real()-ck.real())/(tw.real()+ck.real());
	    double rdi = 2.*(tw.imag()-ck.imag())/(tw.imag()+ck.imag());
	    std::string msg = "";
	    if(fabs(rdr) > 1e-12 || fabs(rdi) > 1e-12) msg = " ERROR";
	    if(!UniqueID()) printf("tsep %d D %d G %d tK %d tdis %d CK (%g %g) TW (%g %g) reldiff (%g,%g)%s\n",k_sigma_separation[ks_sep_idx],type3_idx[d],gidx,tk,tdis,ck.real(),ck.imag(),tw.real(),tw.imag(),rdr,rdi,msg.c_str());
	  }
	}
      }      
    }
  }


  if(!UniqueID()){ printf("Comparing type4\n"); fflush(stdout); }
  int type4_idx[9] = {4,5,9,12,13,15,17,20,22};

  int twksidx = k_sigma_separation.size()-1;

  for(int d=0;d<9;d++){
    for(int gidx=0;gidx<8;gidx++){
      for(int tk=0;tk<Lt;tk++){
	for(int tdis=1; tdis< k_sigma_separation.back(); tdis++){
	  std::complex<double> ck = convertComplexD(type4_ck(tk,tdis,d,gidx));
	  std::complex<double> tw = convertComplexD(type_1234_tw[twksidx](tk,tdis,type4_idx[d],gidx));
	  double rdr = 2.*(tw.real()-ck.real())/(tw.real()+ck.real());
	  double rdi = 2.*(tw.imag()-ck.imag())/(tw.imag()+ck.imag());
	  std::string msg = "";
	  if(fabs(rdr) > 1e-12 || fabs(rdi) > 1e-12) msg = " ERROR";
	  if(!UniqueID()) printf("D %d G %d tK %d tdis %d CK (%g %g) TW (%g %g) reldiff (%g,%g)%s\n",type4_idx[d],gidx,tk,tdis,ck.real(),ck.imag(),tw.real(),tw.imag(),rdr,rdi,msg.c_str());
	}
      }
    }      
  }


  if(!UniqueID()){ printf("Comparing mix3\n"); fflush(stdout); }

  for(int ks_sep_idx=0; ks_sep_idx<k_sigma_separation.size(); ks_sep_idx++){
    for(int f=0;f<2;f++){
      for(int tk=0;tk<Lt;tk++){
	for(int tdis=1; tdis< k_sigma_separation[ks_sep_idx]; tdis++){
	  std::complex<double> ck = convertComplexD(mix3_ck[ks_sep_idx](tk,tdis,f));
	  std::complex<double> tw = convertComplexD(mix_34_tw[ks_sep_idx](tk,tdis,3,f));
	  double rdr = 2.*(tw.real()-ck.real())/(tw.real()+ck.real());
	  double rdi = 2.*(tw.imag()-ck.imag())/(tw.imag()+ck.imag());
	  std::string msg = "";
	  if(fabs(rdr) > 1e-12 || fabs(rdi) > 1e-12) msg = " ERROR";
	  if(!UniqueID()) printf("tsep %d F %d tK %d tdis %d CK (%g %g) TW (%g %g) reldiff (%g,%g)%s\n",k_sigma_separation[ks_sep_idx],f,tk,tdis,ck.real(),ck.imag(),tw.real(),tw.imag(),rdr,rdi,msg.c_str());
	}
      }
    }
  }

  if(!UniqueID()){ printf("Comparing mix4\n"); fflush(stdout); }

  for(int f=0;f<2;f++){
    for(int tk=0;tk<Lt;tk++){
      for(int tdis=1; tdis< k_sigma_separation.back(); tdis++){
	std::complex<double> ck = convertComplexD(mix4_ck(tk,tdis,f));
	std::complex<double> tw = convertComplexD(mix_34_tw[twksidx](tk,tdis,4,f));
	double rdr = 2.*(tw.real()-ck.real())/(tw.real()+ck.real());
	double rdi = 2.*(tw.imag()-ck.imag())/(tw.imag()+ck.imag());
	std::string msg = "";
	if(fabs(rdr) > 1e-12 || fabs(rdi) > 1e-12) msg = " ERROR";
	if(!UniqueID()) printf("F %d tK %d tdis %d CK (%g %g) TW (%g %g) reldiff (%g,%g)%s\n",f,tk,tdis,ck.real(),ck.imag(),tw.real(),tw.imag(),rdr,rdi,msg.c_str());
      }
    }
  }

}

#endif
