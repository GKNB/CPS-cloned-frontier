//We can save some time by efficiently precomputing SpinColorFlavorMatrix parts for the K->pipi before the inner site loop. However storing these takes a lot of memory and, at least for the DD1 with a 16^3 job
//causes the machine to run out of memory. Use the below options to disable the precompute.
//#define DISABLE_TYPE1_PRECOMPUTE
//#define DISABLE_TYPE2_PRECOMPUTE
//#define DISABLE_TYPE3_PRECOMPUTE
//#define DISABLE_TYPE3_SPLIT_VMV //also disables precompute
//#define DISABLE_TYPE4_PRECOMPUTE

//This option disables the majority of the compute but keeps everything else intact allowing you to test the memory usage without doing a full run
//#define MEMTEST_MODE

#define NODE_DISTRIBUTE_MESONFIELDS //Save memory by keeping meson fields only on single node until needed

#include <alg/alg_fix_gauge.h>
#include <alg/a2a/utils_main.h>
#include <alg/a2a/grid_wrappers.h>
#include <alg/a2a/bfm_wrappers.h>
#include <alg/a2a/compute_kaon.h>
#include <alg/a2a/compute_pion.h>
#include <alg/a2a/compute_sigma.h>
#include <alg/a2a/compute_pipi.h>
#include <alg/a2a/compute_ktopipi.h>

#include "main.h"

int main (int argc,char **argv )
{
  const char *fname="main(int,char**)";
  Start(&argc, &argv);

  const char *cname=argv[0];
  const int TrajStart = atoi(argv[2]);
  const int LessThanLimit = atoi(argv[3]);

  CommandLineArgs cmdline(argc,argv,4); //control the functioning of the program from the command line    
  Parameters params(argv[1]);

  setupJob(argc, argv, params, cmdline);

  const int Lt = GJP.Tnodes()*GJP.TnodeSites();

#if defined(USE_BFM_A2A) || defined(USE_BFM_LANCZOS)
  BFMsolvers bfm_solvers(cmdline.nthreads, 0.01, 1e-08, 20000, params.jp.solver, params.jp.mobius_scale); //for BFM holds a double and single precision bfm instance. Mass is not important as it is changed when necessary
#endif
  
  if(chdir(params.meas_arg.WorkDirectory)!=0) ERR.General("",fname,"Unable to switch to work directory '%s'\n",params.meas_arg.WorkDirectory);
  double time;

  if(!UniqueID()) printf("Memory prior to config loop:\n");
  printMem();

  //Setup parameters of fields
  typedef typename A2Apolicies::ComplexType mf_Complex;
  typedef typename A2Apolicies::SourcePolicies::DimensionPolicy::ParamType Field3DparamType;
  typedef typename A2Apolicies::FermionFieldType::InputParamType Field4DparamType;
  Field4DparamType field4dparams; setupFieldParams<mf_Complex>(field4dparams);
  Field3DparamType field3dparams; setupFieldParams<mf_Complex>(field3dparams);
  
  //-------------------- Main Loop Begin! -------------------- //
  for(int conf = TrajStart; conf < LessThanLimit; conf += params.meas_arg.TrajIncrement) {
    double conf_time = -dclock();
    if(!UniqueID()) std::cout<<"Starting configuration "<<conf<< std::endl;

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
    if(!cmdline.randomize_vw) computeEvecs(eig, Light, params, cmdline.evecs_single_prec, cmdline.randomize_evecs, COMPUTE_EVECS_EXTRA_ARG_PASS);

    //-------------------- Light quark v and w --------------------//
    A2AvectorV<A2Apolicies> V(params.a2a_arg, field4dparams);
    A2AvectorW<A2Apolicies> W(params.a2a_arg, field4dparams);
    computeVW(V, W, Light, params, eig, cmdline.evecs_single_prec, cmdline.randomize_vw, cmdline.mixed_solve, cmdline.inner_cg_resid_p, true, COMPUTE_EVECS_EXTRA_ARG_PASS);

    eig.freeEvecs();
    if(!UniqueID()) printf("Memory after light evec free:\n");
    printMem();
    
    //-------------------- Strange quark Lanczos ---------------------//
    LanczosWrapper eig_s;
    if(!cmdline.randomize_vw) computeEvecs(eig_s, Heavy, params, cmdline.evecs_single_prec, cmdline.randomize_evecs, COMPUTE_EVECS_EXTRA_ARG_PASS);

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

    //The pion two-point function and pipi/k->pipi all utilize the same meson fields. Generate those here
    //For convenience pointers to the meson fields are collected into a single object that is passed to the compute methods
    RequiredMomentum<StandardPionMomentaPolicy> pion_mom; //these are the W and V momentum combinations

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
    conf_time += dclock();
    print_time("main","Configuration total",conf_time);
  }//end of config loop

  if(!UniqueID()) printf("Done\n");
  End();
}

