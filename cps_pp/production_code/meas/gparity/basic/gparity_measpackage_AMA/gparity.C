#if defined(USE_TBC_INPUT)
#warning "Using TBC specified in do_arg"
#elif defined(USE_TBC_FB)
#warning "Using F=P+A and B=P-A temporal boundary condition combinations"
#elif defined(DOUBLE_TLATT)
#warning "Doubling time direction and using specified in do_arg"
#define USE_TBC_INPUT
#else
#error "Must specify TBC flag USE_TBC_INPUT or USE_TBC_FB"
#endif

#ifdef TESTING
#warning "Compiling for TEST mode"
#endif

#include <alg/alg_fix_gauge.h>
#include "cmdline_args.h"
#include "cshift.h"
#include "gparity.h"
#include "meas.h"
#include "tests.h"

#include "main_fb.h"
#include "main_std.h"
USING_NAMESPACE_CPS

//#define TESTING

int main(int argc,char *argv[])
{
#ifdef TESTING
  return run_tests(argc,argv);
#endif

  Start(&argc,&argv);
  if(argc<2) ERR.General("","main()","Arguments: Require a directory containing the vml files");

  BfmArg bfm_arg; //note, only the solver and mobius scale are used here
  DoArg do_arg;
  LancArg lanc_arg_l;
  LancArg lanc_arg_h;
  GparityAMAarg2 ama_arg;

  decode_vml_all(do_arg, bfm_arg, lanc_arg_l, lanc_arg_h, ama_arg, argv[argc-1]);
  if(ama_arg.conf_start >= ama_arg.conf_lessthan || ama_arg.conf_incr == 0) ERR.General("","main()","Invalid configuration args");
  if(lanc_arg_l.mass != ama_arg.ml) ERR.General("","main()","Light lanczos mass differs from value in AMA args");
  if(lanc_arg_h.mass != ama_arg.mh) ERR.General("","main()","Heavy lanczos mass differs from value in AMA args");

  CmdLine cmdline(argc,argv);

  if(UniqueID()==0) printf("Configuration format is '%s'\n",ama_arg.config_fmt);
  if(!contains_pctd(ama_arg.config_fmt)) ERR.General("","main()","GparityAMAarg config format '%s' does not contain a %%d",ama_arg.config_fmt);

  if(UniqueID()==0) printf("RNG format is '%s'\n",ama_arg.rng_fmt);
  if(!contains_pctd(ama_arg.rng_fmt)) ERR.General("","main()","GparityAMAarg rng format '%s' does not contain a %%d",ama_arg.rng_fmt);
  
  GJP.Initialize(do_arg);
  GJP.StartConfKind(START_CONF_MEM); //we will handle the gauge field read/write thankyou!
  GJP.StartSeedKind(START_SEED_INPUT); //and the LRG load!

  if(do_arg.start_seed_kind != START_SEED_FILE) printf("WARNING: Using fixed input start seed. You might want to use START_SEED_FILE to aid reproducibility!\n");

#if TARGET == BGQ
  LRG.setSerial();
#endif

  LRG.Initialize();
  
  init_fbfm(&argc,&argv,bfm_arg);

  if(UniqueID()==0)
    if(Fbfm::use_mixed_solver) printf("Using Fbfm mixed precision solver\n");
    else printf("Using Fbfm double precision solver\n");

  check_bk_tsources(ama_arg); //Check the time seps for BK before we have to do any work

  GnoneFbfm lattice;
  CommonArg carg("label","filename");

  //Decide on the meson and quark momenta we wish to compute
  MesonMomenta pion_momenta;
  PionMomenta::setup(pion_momenta,cmdline.do_alternative_mom);
  pion_momenta.printAllCombs("pion_momenta");
    
  MesonMomenta su2_singlet_momenta;
  LightFlavorSingletMomenta::setup(su2_singlet_momenta);
  su2_singlet_momenta.printAllCombs("su2_singlet_momenta");

  MesonMomenta kaon_momenta;
  KaonMomenta::setup(kaon_momenta);
  kaon_momenta.printAllCombs("kaon_momenta");

  //Determine the quark momenta we will need
  QuarkMomenta light_quark_momenta;
  QuarkMomenta heavy_quark_momenta;
    
  pion_momenta.appendQuarkMomenta(Light, light_quark_momenta); //adds the quark momenta it needs
  su2_singlet_momenta.appendQuarkMomenta(Light, light_quark_momenta);
  kaon_momenta.appendQuarkMomenta(Light, light_quark_momenta); //each momentum is unique
  kaon_momenta.appendQuarkMomenta(Heavy, heavy_quark_momenta);

  if(!UniqueID()){
    printf("Light quark momenta to be computed:\n");
    for(int i=0;i<light_quark_momenta.nMom();i++)
      std::cout << light_quark_momenta.getMom(i).str() << '\n';
    printf("Heavy quark momenta to be computed:\n");
    for(int i=0;i<heavy_quark_momenta.nMom();i++)
      std::cout << heavy_quark_momenta.getMom(i).str() << '\n';
  }

  for(int conf=ama_arg.conf_start; conf < ama_arg.conf_lessthan; conf += ama_arg.conf_incr){
    Float conf_start_time = dclock();

    readLatticeAndRNG(lattice, cmdline, do_arg, ama_arg, conf);
  
    if(cmdline.rng_test){ //just load config, generate some uniform random numbers then move onto next config
      if(!UniqueID()) printf("Random offsets in range 0..Lt\n");
      const int Lt = GJP.Tnodes()*GJP.TnodeSites();
      for(int i=0;i<50;i++){
	int offset = int(floor( LRG.Lrand(Lt,0) )) % Lt;
	if(!UniqueID()) printf("%d\n",offset);
      }
      continue;
    }

    //Gauge fix lattice if required. Do this before the fermion time BCs are applied to the gauge field
    if(ama_arg.fix_gauge.fix_gauge_kind != FIX_GAUGE_NONE){
      Float time = -dclock();
      AlgFixGauge fix_gauge(lattice,&carg,&ama_arg.fix_gauge);
      if(cmdline.skip_gauge_fix){
	if(!UniqueID()) printf("Skipping gauge fix -> Setting all GF matrices to unity\n");
	gaugeFixUnity(lattice,ama_arg.fix_gauge);      
      }else{
	fix_gauge.run();
      }
      print_time("main","Gauge fix",time + dclock());
    }

    if(cmdline.lanczos_tune_l || cmdline.lanczos_tune_h){
      Float time = -dclock();
      doLanczos(lattice, cmdline.lanczos_tune_l ? lanc_arg_l : lanc_arg_h, GJP.Tbc());
      print_time("main","Lanczos tune",time + dclock());
      if(UniqueID()==0){
	printf("Main job complete\n"); 
	fflush(stdout);
      }
      return 0;
    }

#ifdef USE_TBC_FB
    runFB(cmdline, lattice,
	  do_arg,
	  lanc_arg_l,
	  lanc_arg_h,
	  ama_arg,
	  light_quark_momenta,
	  heavy_quark_momenta,
	  pion_momenta,
	  kaon_momenta,
	  su2_singlet_momenta,
	  conf);
#elif defined(USE_TBC_INPUT)
    runFB(cmdline, lattice,
	  do_arg,
	  lanc_arg_l,
	  lanc_arg_h,
	  ama_arg,
	  light_quark_momenta,
	  heavy_quark_momenta,
	  pion_momenta,
	  kaon_momenta,
	  su2_singlet_momenta,
	  conf);
#endif
    
    lattice.FixGaugeFree();
    
    print_time("main","Configuration time",dclock()-conf_start_time);
  }//end of conf loop

  if(UniqueID()==0){
    printf("Main job complete\n"); 
    fflush(stdout);
  }  
  End();
  return 0;
}
