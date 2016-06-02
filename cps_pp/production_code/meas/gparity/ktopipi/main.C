//We can save some time by efficiently precomputing SpinColorFlavorMatrix parts for the K->pipi before the inner site loop. However storing these takes a lot of memory and, at least for the DD1 with a 16^3 job
//causes the machine to run out of memory. Use the below options to disable the precompute.
//#define DISABLE_TYPE1_PRECOMPUTE
//#define DISABLE_TYPE2_PRECOMPUTE
//#define DISABLE_TYPE3_PRECOMPUTE
//#define DISABLE_TYPE3_SPLIT_VMV //also disables precompute
//#define DISABLE_TYPE4_PRECOMPUTE

#define NODE_DISTRIBUTE_MESONFIELDS //Save memory by keeping meson fields only on single node until needed

#include<chroma.h>

//bfm headers
#ifdef USE_BFM
#include<bfm.h>
#include<util/lattice/bfm_eigcg.h> // This is for the Krylov.h function "matrix_dgemm"
#include<util/lattice/bfm_evo.h>
#endif

//cps headers
#include<alg/common_arg.h>
#include<alg/fix_gauge_arg.h>
#include<alg/do_arg.h>
#include<alg/meas_arg.h>
#include<alg/a2a_arg.h>
#include<alg/lanc_arg.h>
#include<alg/ktopipi_jobparams.h>
#include<util/qioarg.h>
#include<util/ReadLatticePar.h>
#include<alg/alg_fix_gauge.h>
#include<util/flavormatrix.h>
#include<alg/wilson_matrix.h>
#include<util/spincolorflavormatrix.h>

#if defined(USE_GRID) && !defined(DISABLE_GRID_A2A)
#include<util/lattice/fgrid.h>
#endif

#ifdef USE_MPI
//mpi headers
#warning "WARNING : USING MPI"
#include<mpi.h>
#endif

//c++ classes
#include<sys/stat.h>
#include<unistd.h>

using namespace Chroma;
using namespace cps;

#include <alg/a2a/a2a.h>
#include <alg/a2a/mesonfield.h>
#include <alg/a2a/compute_kaon.h>
#include <alg/a2a/compute_pion.h>
#include <alg/a2a/compute_pipi.h>
#include <alg/a2a/compute_ktopipi.h>
#include <alg/a2a/main.h>

#ifdef A2A_PREC_DOUBLE
typedef double mf_Float;
#elif defined(A2A_PREC_SINGLE)
typedef float mf_Float;
#else
#error "Must provide an A2A precision"
#endif

typedef std::complex<mf_Float> mf_Complex;

int main (int argc,char **argv )
{
  Start(&argc, &argv);
  if(!UniqueID()){ printf("Arguments:\n"); fflush(stdout); }
  for(int i=0;i<argc;i++){
    if(!UniqueID()){ printf("%d \"%s\"\n",i,argv[i]); fflush(stdout); }
  }
#ifdef NODE_DISTRIBUTE_MESONFIELDS
  if(!UniqueID()) printf("Using node distribution of meson fields\n");
#endif

  const char *cname=argv[0];
  const int TrajStart = atoi(argv[2]);
  const int LessThanLimit = atoi(argv[3]);

  int nthreads = 1;
#if TARGET == BGQ
  nthreads = 64;
#endif
  bool randomize_vw = false; //rather than doing the Lanczos and inverting the propagators, etc, just use random vectors for V and W. This should only be used after you have tested
  bool tune_lanczos_light = false; //just run the light lanczos on first config then exit
  bool tune_lanczos_heavy = false; //just run the heavy lanczos on first config then exit
  bool skip_gauge_fix = false;
  bool double_latt = false; //most ancient 8^4 quenched lattices stored both U and U*. Enable this to read those configs
  bool mixed_solve = true; //do high mode inversions using mixed precision solves. Is disabled if we turn off the single-precision conversion of eigenvectors (because internal single-prec inversion needs singleprec eigenvectors)
  bool evecs_single_prec = true; //convert the eigenvectors to single precision to save memory

  int arg = 4;
  while(arg < argc){
    char* cmd = argv[arg];
    if( strncmp(cmd,"-nthread",8) == 0){
      if(arg == argc-1){ 
	if(!UniqueID()){ printf("-nthread must be followed by a number!\n"); fflush(stdout); }
	exit(-1);
      }
      std::stringstream ss; ss << argv[arg+1];
      ss >> nthreads;
      if(!UniqueID()){ printf("Setting number of threads to %d\n",nthreads); }
      arg+=2;
    }else if( strncmp(cmd,"-randomize_vw",15) == 0){
      randomize_vw = true;
      if(!UniqueID()){ printf("Using random vectors for V and W, skipping Lanczos and inversion stages\n"); fflush(stdout); }
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
    }else{
      if(UniqueID()==0) printf("Unrecognised argument: %s\n",cmd);
      exit(-1);
    }
  }

  const char *fname="main(int,char**)";
  if(chdir(argv[1])!=0) ERR.General("",fname,"Unable to switch to directory '%s'\n",argv[1]);
  CommonArg common_arg("",""), common_arg2("","");
  DoArg do_arg;
  JobParams jp;
  MeasArg meas_arg;
  FixGaugeArg fix_gauge_arg;
  A2AArg a2a_arg, a2a_arg_s;
  LancArg lanc_arg, lanc_arg_s;

  if(!do_arg.Decode("do_arg.vml","do_arg")){
    do_arg.Encode("do_arg.templ","do_arg");
    VRB.Result(cname,fname,"Can't open do_arg.vml!\n");exit(1);
  }
  if(!jp.Decode("job_params.vml","job_params")){
    jp.Encode("job_params.templ","job_params");
    VRB.Result(cname,fname,"Can't open job_params.vml!\n");exit(1);
  }
  if(!meas_arg.Decode("meas_arg.vml","meas_arg")){
    meas_arg.Encode("meas_arg.templ","meas_arg");
    std::cout<<"Can't open meas_arg!"<<std::endl;exit(1);
  }
  if(!a2a_arg.Decode("a2a_arg.vml","a2a_arg")){
    a2a_arg.Encode("a2a_arg.templ","a2a_arg");
    VRB.Result(cname,fname,"Can't open a2a_arg.vml!\n");exit(1);
  }
  if(!a2a_arg_s.Decode("a2a_arg_s.vml","a2a_arg_s")){
    a2a_arg_s.Encode("a2a_arg_s.templ","a2a_arg_s");
    VRB.Result(cname,fname,"Can't open a2a_arg_s.vml!\n");exit(1);
  }
  if(!lanc_arg.Decode("lanc_arg.vml","lanc_arg")){
    lanc_arg.Encode("lanc_arg.templ","lanc_arg");
    VRB.Result(cname,fname,"Can't open lanc_arg.vml!\n");exit(1);
  }
  if(!lanc_arg_s.Decode("lanc_arg_s.vml","lanc_arg_s")){
    lanc_arg_s.Encode("lanc_arg_s.templ","lanc_arg_s");
    VRB.Result(cname,fname,"Can't open lanc_arg_s.vml!\n");exit(1);
  }
  if(!fix_gauge_arg.Decode("fix_gauge_arg.vml","fix_gauge_arg")){
    fix_gauge_arg.Encode("fix_gauge_arg.templ","fix_gauge_arg");
    VRB.Result(cname,fname,"Can't open fix_gauge_arg.vml!\n");exit(1);
  }

  //Chroma::initialize(&argc,&argv);
  GJP.Initialize(do_arg);
  LRG.Initialize();

  if(double_latt) SerialIO::dbl_latt_storemode = true;

  if(!UniqueID()) printf("Initial memory post-initialize:\n");
  printMem();


  if(!tune_lanczos_light && !tune_lanczos_heavy){ 
    assert(a2a_arg.nl <= lanc_arg.N_true_get);
    assert(a2a_arg_s.nl <= lanc_arg_s.N_true_get);
  }
#ifdef USE_BFM
  cps_qdp_init(&argc,&argv);
  Chroma::initialize(&argc,&argv);
#endif

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  
  LatticeSolvers solvers(jp,nthreads); //for BFM holds a double and single precision bfm instance, nothing for Grid
  
  if(chdir(meas_arg.WorkDirectory)!=0) ERR.General("",fname,"Unable to switch to work directory '%s'\n",meas_arg.WorkDirectory);
  double time;

  if(!UniqueID()) printf("Memory prior to config loop:\n");
  printMem();

  //-------------------- Main Loop Begin! -------------------- //
  for(int conf = TrajStart; conf < LessThanLimit; conf += meas_arg.TrajIncrement) {
    double conf_time = -dclock();
    if(!UniqueID()) std::cout<<"Starting configuration "<<conf<< std::endl;

    meas_arg.TrajCur = conf;

    std::string dir(meas_arg.WorkDirectory);
    common_arg.set_filename(dir.c_str());

    //-------------------- Read gauge field --------------------//
    ReadGaugeField(meas_arg,double_latt); 
    ReadRngFile(meas_arg,double_latt); 

    if(!UniqueID()) printf("Memory after gauge and RNG read:\n");
    printMem();

    LatticeSetup lattice_setup(jp,solvers); //for BFM this creates a lattice object and imports the gauge field into the bfm instances, for Grid a lattice object and import of the gauge field
    LatticeSetup::LatticeType &lat = lattice_setup.getLattice();

    if(tune_lanczos_light || tune_lanczos_heavy){
      if(!UniqueID()) printf("Tuning lanczos %s with mass %f\n", tune_lanczos_light ? "light": "heavy", tune_lanczos_light ? lanc_arg.mass : lanc_arg_s.mass);
      time = -dclock();
      Lanczos eig;
      eig.compute(tune_lanczos_light ? lanc_arg : lanc_arg_s, solvers, lat);
      time += dclock();
      print_time("main","Lanczos",time);
      exit(0);
    }

    //-------------------- Light quark v and w --------------------//
    Lanczos eig;

    if(!randomize_vw){
      if(lanc_arg.N_true_get > 0){
	if(!UniqueID()) printf("Running light quark Lanczos\n");
	time = -dclock();
	eig.compute(lanc_arg, solvers, lat);
	time += dclock();
	print_time("main","Light quark Lanczos",time);

	if(!UniqueID()) printf("Memory after light quark Lanczos:\n");
	printMem();
      }

      if(evecs_single_prec){
	eig.toSingle();
	if(!UniqueID()) printf("Memory after single-prec conversion of light quark evecs:\n");
	printMem();
      }
    }

    if(!UniqueID()) printf("Computing light quark A2A vectors\n");
    time = -dclock();
    
    A2AvectorV<mf_Complex> V(a2a_arg);
    A2AvectorW<mf_Complex> W(a2a_arg);

    if(!randomize_vw){
      computeA2Avectors<mf_Complex>::compute(V,W,mixed_solve,evecs_single_prec, lat, eig, solvers);
      //W.computeVW(V, lat, *eig.eig, evecs_single_prec, solvers.dwf_d, mixed_solve ? & solvers.dwf_f : NULL);
    }else randomizeVW<mf_Complex>(V,W);    

    if(!UniqueID()) printf("Memory after light A2A vector computation:\n");
    printMem();

    eig.freeEvecs();

    if(!UniqueID()) printf("Memory after light evec free:\n");
    printMem();

    time += dclock();
    print_time("main","Light quark A2A vectors",time);

    //-------------------- Strange quark v and w --------------------//
    Lanczos eig_s;

    if(!randomize_vw){
      if(lanc_arg_s.N_true_get > 0){
	if(!UniqueID()) printf("Running strange quark Lanczos\n");
	time = -dclock();
	eig_s.compute(lanc_arg_s,solvers,lat);
	time += dclock();
	print_time("main","Strange quark Lanczos",time);

	if(!UniqueID()) printf("Memory after heavy quark Lanczos:\n");
	printMem();
      }
      if(evecs_single_prec){
	eig_s.toSingle();
	if(!UniqueID()) printf("Memory after single-prec conversion of heavy quark evecs:\n");
	printMem();
      }
    }

    if(!UniqueID()) printf("Computing strange quark A2A vectors\n");
    time = -dclock();

    A2AvectorV<mf_Complex> V_s(a2a_arg_s);
    A2AvectorW<mf_Complex> W_s(a2a_arg_s);

    if(!randomize_vw){
      computeA2Avectors<mf_Complex>::compute(V_s,W_s,mixed_solve,evecs_single_prec, lat, eig_s, solvers);
      //W_s.computeVW(V_s, lat, *eig_s.eig, evecs_single_prec, solvers.dwf_d, mixed_solve ? & solvers.dwf_f : NULL);
    }else randomizeVW<mf_Complex>(V_s,W_s);      

    if(!UniqueID()) printf("Memory after heavy A2A vector computation:\n");
    printMem();

    eig_s.freeEvecs();

    if(!UniqueID()) printf("Memory after heavy evec free:\n");
    printMem();

    time += dclock();
    print_time("main","Strange quark A2A vectors",time);

    //-------------------Fix gauge----------------------------
    AlgFixGauge fix_gauge(lat,&common_arg,&fix_gauge_arg);
    if(skip_gauge_fix){
      if(!UniqueID()) printf("Skipping gauge fix -> Setting all GF matrices to unity\n");
      gaugeFixUnity(lat,fix_gauge_arg);      
    }else{
      if(!UniqueID()) printf("Gauge fixing\n");
      time = -dclock();
      fix_gauge.run();
      time += dclock();
      print_time("main","Gauge fix",time);
    }

    if(!UniqueID()) printf("Memory after gauge fix:\n");
    printMem();

    //-------------------------Compute the kaon two-point function---------------------------------
    {
      if(!UniqueID()) printf("Computing kaon 2pt function\n");
      time = -dclock();
      fMatrix<mf_Complex> kaon(Lt,Lt);
      ComputeKaon<mf_Complex>::compute(kaon,
				     W, V, W_s, V_s,
				     jp.kaon_rad, lat);
      std::ostringstream os; os << meas_arg.WorkDirectory << "/traj_" << conf << "_kaoncorr";
      kaon.write(os.str());
      time += dclock();
      print_time("main","Kaon 2pt function",time);
    }

    if(!UniqueID()) printf("Memory after kaon 2pt function computation:\n");
    printMem();

    //The pion two-point function and pipi/k->pipi all utilize the same meson fields. Generate those here
    //For convenience pointers to the meson fields are collected into a single object that is passed to the compute methods
    RequiredMomentum<StandardPionMomentaPolicy> pion_mom; //these are the W and V momentum combinations

    std::vector< std::vector<A2AmesonField<mf_Complex,A2AvectorWfftw,A2AvectorVfftw> > > mf_ll; //[pidx][t]   stores the meson fields
    MesonFieldMomentumContainer<mf_Complex> mf_ll_con; //manager for pointers to the above
    
    if(!UniqueID()) printf("Computing light-light meson fields\n");
    time = -dclock();
    ComputePion<mf_Complex>::computeMesonFields(mf_ll, mf_ll_con, pion_mom, W, V, jp.pion_rad, lat);
    time += dclock();
    print_time("main","Light-light meson fields",time);

    if(!UniqueID()) printf("Memory after light-light meson field computation:\n");
    printMem();

    //----------------------------Compute the pion two-point function---------------------------------
    int nmom = pion_mom.nMom();

    if(!UniqueID()) printf("Computing pion 2pt function\n");
    time = -dclock();
    for(int p=0;p<nmom;p+=2){ //note odd indices 1,3,5 etc have equal and opposite momenta to 0,2,4... 
      if(!UniqueID()) printf("Starting pidx %d\n",p);
      fMatrix<mf_Complex> pion(Lt,Lt);
      ComputePion<mf_Complex>::compute(pion, mf_ll_con, pion_mom, p);
      //Note it seems Daiqian's pion momenta are opposite what they should be for 'conventional' Fourier transform phase conventions:
      //f'(p) = \sum_{x,y}e^{ip(x-y)}f(x,y)  [conventional]
      //f'(p) = \sum_{x,y}e^{-ip(x-y)}f(x,y) [Daiqian]
      //This may have been a mistake as it only manifests in the difference between the labelling of the pion momenta and the sign of 
      //the individual quark momenta.
      //However it doesn't really make any difference. If you enable DAIQIAN_PION_PHASE_CONVENTION
      //the output files will be labelled in Daiqian's convention
#define DAIQIAN_PION_PHASE_CONVENTION

      std::ostringstream os; os << meas_arg.WorkDirectory << "/traj_" << conf << "_pioncorr_mom";
#ifndef DAIQIAN_PION_PHASE_CONVENTION
      os << pion_mom.getMesonMomentum(p).file_str(2);  //note the divisor of 2 is to put the momenta in units of pi/L and not pi/2L
#else
      os << (-pion_mom.getMesonMomentum(p)).file_str(2);
#endif
      pion.write(os.str());
    }
    time += dclock();
    print_time("main","Pion 2pt function",time);

    if(!UniqueID()) printf("Memory after pion 2pt function computation:\n");
    printMem();

    //------------------------------I=0 and I=2 PiPi two-point function---------------------------------
    if(!UniqueID()) printf("Computing pi-pi 2pt function\n");
    double timeC(0), timeD(0), timeR(0), timeV(0);
    double* timeCDR[3] = {&timeC, &timeD, &timeR};

    for(int psrcidx=0; psrcidx < nmom; psrcidx++){
      ThreeMomentum p_pi1_src = pion_mom.getMesonMomentum(psrcidx);

      for(int psnkidx=0; psnkidx < nmom; psnkidx++){	
	fMatrix<mf_Complex> pipi(Lt,Lt);
	ThreeMomentum p_pi1_snk = pion_mom.getMesonMomentum(psnkidx);
	
	MesonFieldProductStore<mf_Complex> products; //try to reuse products of meson fields wherever possible

	char diag[3] = {'C','D','R'};
	for(int d = 0; d < 3; d++){
	  if(!UniqueID()){ printf("Doing pipi figure %c, psrcidx=%d psnkidx=%d\n",diag[d],psrcidx,psnkidx); fflush(stdout); }

	  time = -dclock();
	  ComputePiPiGparity<mf_Complex>::compute(pipi, diag[d], p_pi1_src, p_pi1_snk, jp.pipi_separation, jp.tstep_pipi, mf_ll_con, products);
	  std::ostringstream os; os << meas_arg.WorkDirectory << "/traj_" << conf << "_Figure" << diag[d] << "_sep" << jp.pipi_separation;
#ifndef DAIQIAN_PION_PHASE_CONVENTION
	  os << "_mom" << p_pi1_src.file_str(2) << "_mom" << p_pi1_snk.file_str(2);
#else
	  os << "_mom" << (-p_pi1_src).file_str(2) << "_mom" << (-p_pi1_snk).file_str(2);
#endif
	  pipi.write(os.str());
	  time += dclock();
	  *timeCDR[d] += time;
	}
      }

      { //V diagram
	if(!UniqueID()){ printf("Doing pipi figure V, pidx=%d\n",psrcidx); fflush(stdout); }
	time = -dclock();
	fVector<mf_Complex> figVdis(Lt);
	ComputePiPiGparity<mf_Complex>::computeFigureVdis(figVdis,p_pi1_src,jp.pipi_separation,mf_ll_con);
	std::ostringstream os; os << meas_arg.WorkDirectory << "/traj_" << conf << "_FigureVdis_sep" << jp.pipi_separation;
#ifndef DAIQIAN_PION_PHASE_CONVENTION
	os << "_mom" << p_pi1_src.file_str(2);
#else
	os << "_mom" << (-p_pi1_src).file_str(2);
#endif
	figVdis.write(os.str());
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

    //--------------------------------------K->pipi contractions--------------------------------------------------------
    //We first need to generate the light-strange W*W contraction
    std::vector<A2AmesonField<mf_Complex,A2AvectorWfftw,A2AvectorWfftw> > mf_ls_ww;
    ComputeKtoPiPiGparity<mf_Complex>::generatelsWWmesonfields(mf_ls_ww,W,W_s,jp.kaon_rad,lat);

    std::vector<int> k_pi_separation(jp.k_pi_separation.k_pi_separation_len);
    for(int i=0;i<jp.k_pi_separation.k_pi_separation_len;i++) k_pi_separation[i] = jp.k_pi_separation.k_pi_separation_val[i];

    if(!UniqueID()) printf("Memory after computing W*W meson fields:\n");
    printMem();

    //For type1 loop over momentum of pi1 (conventionally the pion closest to the kaon)
    int ngp = 0; for(int i=0;i<3;i++) if(GJP.Bc(i)==BND_CND_GPARITY) ngp++;
#define TYPE1_DO_ASSUME_ROTINVAR_GP3  //For GPBC in 3 directions we can assume rotational invariance around the G-parity diagonal vector (1,1,1) and therefore calculate only one off-diagonal momentum

    if(!UniqueID()) printf("Starting type 1 contractions, nmom = %d\n",nmom);
    time = -dclock();
    for(int pidx=0; pidx < nmom; pidx++){
#ifdef TYPE1_DO_ASSUME_ROTINVAR_GP3
      if(ngp == 3 && pidx >= 4) continue; // p_pi1 = (-1,-1,-1), (1,1,1) [diag] (1,-1,-1), (-1,1,1) [orth] only
#endif
      if(!UniqueID()) printf("Starting type 1 contractions with pidx=%d\n",pidx);
      if(!UniqueID()) printf("Memory status before type1 K->pipi:\n");
      printMem();

      ThreeMomentum p_pi1 = pion_mom.getMesonMomentum(pidx);
      std::vector<KtoPiPiGparityResultsContainer> type1;
      ComputeKtoPiPiGparity<mf_Complex>::type1(type1,
					     k_pi_separation, jp.pipi_separation, jp.tstep_type12, jp.xyzstep_type1, p_pi1,
					     mf_ls_ww, mf_ll_con,
					     V, V_s,
					     W, W_s);
      for(int kpi_idx=0;kpi_idx<k_pi_separation.size();kpi_idx++){
	std::ostringstream os; os << meas_arg.WorkDirectory << "/traj_" << conf << "_type1_deltat_" << k_pi_separation[kpi_idx] << "_sep_" << jp.pipi_separation;
#ifndef DAIQIAN_PION_PHASE_CONVENTION
	os << "_mom" << p_pi1.file_str(2);
#else
	os << "_mom" << (-p_pi1).file_str(2);
#endif
	type1[kpi_idx].write(os.str());
      }
      if(!UniqueID()) printf("Memory status after type1 K->pipi:\n");
      printMem();

    }
    time += dclock();
    print_time("main","K->pipi type 1",time);

    if(!UniqueID()) printf("Memory after type1 K->pipi:\n");
    printMem();

    //Type 2 and 3 are optimized by performing the sum over pipi momentum orientations within the contraction
    {
      time = -dclock();
      if(!UniqueID()) printf("Starting type 2 contractions\n");
      std::vector<KtoPiPiGparityResultsContainer> type2;
      ComputeKtoPiPiGparity<mf_Complex>::type2(type2,
					     k_pi_separation, jp.pipi_separation, jp.tstep_type12, pion_mom,
					     mf_ls_ww, mf_ll_con,
					     V, V_s,
					     W, W_s);
      for(int kpi_idx=0;kpi_idx<k_pi_separation.size();kpi_idx++){
	std::ostringstream os; os << meas_arg.WorkDirectory << "/traj_" << conf << "_type2_deltat_" << k_pi_separation[kpi_idx] << "_sep_" << jp.pipi_separation;
	type2[kpi_idx].write(os.str());
      }
      time += dclock();
      print_time("main","K->pipi type 2",time);

      if(!UniqueID()) printf("Memory after type2 K->pipi:\n");
      printMem();
    }

    {
      time = -dclock();
      if(!UniqueID()) printf("Starting type 3 contractions\n");
      std::vector<KtoPiPiGparityResultsContainer> type3;
      std::vector<KtoPiPiGparityMixDiagResultsContainer> mix3;
      ComputeKtoPiPiGparity<mf_Complex>::type3(type3,mix3,
					     k_pi_separation, jp.pipi_separation, 1, pion_mom,
					     mf_ls_ww, mf_ll_con,
					     V, V_s,
					     W, W_s);
      for(int kpi_idx=0;kpi_idx<k_pi_separation.size();kpi_idx++){
	std::ostringstream os; os << meas_arg.WorkDirectory << "/traj_" << conf << "_type3_deltat_" << k_pi_separation[kpi_idx] << "_sep_" << jp.pipi_separation;
	write(os.str(),type3[kpi_idx],mix3[kpi_idx]);
      }
      time += dclock();
      print_time("main","K->pipi type 3",time);
      
      if(!UniqueID()) printf("Memory after type3 K->pipi:\n");
      printMem();
    }

    {
      //Type 4 has no momentum loop as the pion disconnected part is computed as part of the pipi 2pt function calculation
      time = -dclock();
      if(!UniqueID()) printf("Starting type 4 contractions\n");
      KtoPiPiGparityResultsContainer type4;
      KtoPiPiGparityMixDiagResultsContainer mix4;
      
      ComputeKtoPiPiGparity<mf_Complex>::type4(type4, mix4,
					     1,
					     mf_ls_ww,
					     V, V_s,
					     W, W_s);
      
      {
	std::ostringstream os; os << meas_arg.WorkDirectory << "/traj_" << conf << "_type4";
	write(os.str(),type4,mix4);
      }
      time += dclock();
      print_time("main","K->pipi type 4",time);
    
      if(!UniqueID()) printf("Memory after type4 K->pipi and end of config loop:\n");
      printMem();
    }

    conf_time += dclock();
    print_time("main","Configuration total",conf_time);
  }//end of config loop

  QDPIO::cout<<"Done!"<<std::endl;
  End();
}

