#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <util/qcdio.h>
#ifdef PARALLEL
#include <comms/sysfunc_cps.h>
#endif
#include <comms/scu.h>
#include <comms/glb.h>

#include <util/lattice.h>
#include<util/lattice/fbfm.h>
#include <util/time_cps.h>
#include <util/smalloc.h>

#include <util/ReadLatticePar.h>
#include <util/WriteLatticePar.h>

#include <util/command_line.h>

#include<unistd.h>
#include<config.h>

#include <util/data_shift.h>

#include <alg/propmanager.h>
#include <alg/alg_fix_gauge.h>
#include <alg/gparity_contract_arg.h>
#include <alg/alg_gparitycontract.h>
#include <alg/prop_dft.h>
#include <alg/eigen/Krylov_5d.h>

#include <chroma.h>
#include <omp.h>
#include <pthread.h>

#include <alg/bfm_arg.h>

#include <alg/a2a/threemomentum.h>
#include <alg/a2a/utils.h>
#include <alg/a2a/fmatrix.h>

#include "propmomcontainer.h"
#include "mesonmomenta.h"
#include "wallsinkprop.h"
#include "meas.h"
#include "gparity.h"
#include "tests.h"

USING_NAMESPACE_CPS

#define TESTING

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
  GparityAMAarg ama_arg;

  decode_vml_all(do_arg, bfm_arg, lanc_arg_l, lanc_arg_h, ama_arg, argv[argc-1]);
  if(ama_arg.conf_start >= ama_arg.conf_lessthan || ama_arg.conf_incr == 0) ERR.General("","main()","Invalid configuration args");

  bool lanczos_tune = false;
  bool dbl_latt_storemode = false;
  Fbfm::use_mixed_solver = false;
  {
    int i = 1;
    while(i<argc-1){
      if( std::string(argv[i]) == "-lanczos_tune" ){
	lanczos_tune = true;
	i++;
      }else if( std::string(argv[i]) == "-load_dbl_latt" ){
	if(!UniqueID()) printf("Loading double latt\n");
	dbl_latt_storemode = true;
	i++;
      }else if( std::string(argv[i]) == "-use_mixed_solver" ){
	Fbfm::use_mixed_solver = true;
	i++;
      }else{
	ERR.General("","main","Unknown argument: %s",argv[i]);
      }
    }
  }

  const char *c = ama_arg.config_fmt;
  if(UniqueID()==0) printf("Configuration format is '%s'\n",ama_arg.config_fmt);
  bool found(false);
  while(*c!='\0'){
    if(*c=='\%' && *(c+1)=='d'){ found=true; break; }
    c++;
  }
  if(!found) ERR.General("","main()","GparityAMAarg config format '%s' does not contain a %%d",ama_arg.config_fmt);

  GJP.Initialize(do_arg);
  GJP.StartConfKind(START_CONF_MEM); //we will handle the gauge field read/write thankyou!

#if TARGET == BGQ
  LRG.setSerial();
#endif

  LRG.Initialize();
  
  init_fbfm(&argc,&argv,bfm_arg);

  if(UniqueID()==0)
    if(Fbfm::use_mixed_solver) printf("Using Fbfm mixed precision solver\n");
    else printf("Using Fbfm double precision solver\n");

  GnoneFbfm lattice;
  CommonArg carg("label","filename");
  char load_config_file[1000];

  //Double and single precision bfm instances
  bfm_evo<double> &dwf_d = static_cast<Fbfm&>(lattice).bd;
  bfm_evo<float> &dwf_f = static_cast<Fbfm&>(lattice).bf;

  for(int conf=ama_arg.conf_start; conf < ama_arg.conf_lessthan; conf += ama_arg.conf_incr){

    //Read/generate the gauge configuration 
    if(do_arg.start_conf_kind == START_CONF_FILE){
    
      if(sprintf(load_config_file,ama_arg.config_fmt,conf) < 0){
	ERR.General("","main()","Congfiguration filename creation problem : %s | %s",load_config_file,ama_arg.config_fmt);
      }
      //load the configuration
      ReadLatticeParallel readLat;
      if(UniqueID()==0) printf("Reading: %s (NERSC-format)\n",load_config_file);
      if(dbl_latt_storemode){
	if(!UniqueID()) printf("Disabling U* field reconstruction\n");
	readLat.disableGparityReconstructUstarField();
      }
      readLat.read(lattice,load_config_file);
      if(UniqueID()==0) printf("Config read.\n");
    }else if(do_arg.start_conf_kind == START_CONF_ORD){
      if(!UniqueID()) printf("Using unit gauge links\n");
      lattice.SetGfieldOrd();
    }else if(do_arg.start_conf_kind == START_CONF_DISORD){
      if(!UniqueID()) printf("Using random gauge links\n");
      lattice.SetGfieldDisOrd();
      printf("Gauge checksum = %d\n", lattice.CheckSum());
    }else{
      ERR.General("","main()","Invalid do_arg.start_conf_kind\n");
    }
    lattice.BondCond(); //apply BC and import to internal bfm instances

    //Gauge fix lattice if required
    if(ama_arg.fix_gauge.fix_gauge_kind != FIX_GAUGE_NONE){
      AlgFixGauge fix_gauge(lattice,&carg,&ama_arg.fix_gauge);
      fix_gauge.run();
    }

    //Generate eigenvectors
    Float time = -dclock();
    BFM_Krylov::Lanczos_5d<double> lanc_l(dwf_d, lanc_arg_l);
    lanc_l.Run();
    if(Fbfm::use_mixed_solver){
      //Convert eigenvectors to single precision
      lanc_l.toSingle();
    }
    time += dclock();    
    print_time("main","Light quark Lanczos",time);

    time = -dclock();
    BFM_Krylov::Lanczos_5d<double> lanc_h(dwf_d, lanc_arg_h);
    lanc_h.Run();
    if(Fbfm::use_mixed_solver){
      //Convert eigenvectors to single precision
      lanc_h.toSingle();
    }
    time += dclock();    
    print_time("main","Heavy quark Lanczos",time);
 
    //We want stationary mesons and moving mesons. For GPBC there are two inequivalent directions: along the G-parity axis and perpendicular to it. 
    PropMomContainer props; //stores generated propagators by tag

    bool do_alternative_mom = true;

    //Decide on the meson momenta we wish to compute
    MesonMomenta pion_momenta;
    PionMomenta::setup(pion_momenta,do_alternative_mom);
    
    MesonMomenta su2_singlet_momenta;
    LightFlavorSingletMomenta::setup(su2_singlet_momenta);

    MesonMomenta kaon_momenta;
    KaonMomenta::setup(kaon_momenta);

    //Determine the quark momenta we will need
    QuarkMomenta light_quark_momenta;
    QuarkMomenta heavy_quark_momenta;
    
    pion_momenta.appendQuarkMomenta(Light, light_quark_momenta); //adds the quark momenta it needs
    su2_singlet_momenta.appendQuarkMomenta(Light, light_quark_momenta);
    kaon_momenta.appendQuarkMomenta(Light, light_quark_momenta); //each momentum is unique
    kaon_momenta.appendQuarkMomenta(Heavy, heavy_quark_momenta);

    const int Lt = GJP.Tnodes()*GJP.TnodeSites();

    double sloppy_prec, exact_prec;
    double ml, mh;
    std::string results_dir;

    std::vector<int> tslice_sloppy;
    std::vector<int> tslice_exact;

    for(int status = 0; status < 2; status++){ //sloppy, exact
      PropPrecision pp = status == 0 ? Sloppy : Exact;
      const std::vector<int> &tslices = status == 0 ? tslice_sloppy : tslice_exact;
      double prec = status == 0 ? sloppy_prec : exact_prec;
      
      //Light-quark inversions
      lightQuarkInvert(props, pp, prec,ml,tslices,light_quark_momenta,lattice,lanc_l);

      //Pion 2pt LW functions pseudoscalar and axial sinks	     
      measurePion2ptLW(props,pp,tslices,pion_momenta,results_dir,conf);

      //Pion 2pt WW function pseudoscalar sink
      measurePion2ptPPWW(props,pp,tslices,pion_momenta,lattice,results_dir,conf);

      props.clear(); //delete all propagators thus far computed
    }
  }//end of conf loop

  if(UniqueID()==0){
    printf("Main job complete\n"); 
    fflush(stdout);
  }
  
  return 0;
}
