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

#ifdef BNL_KNL_PERFORMANCE_CHECK
  bnl_knl_performance_check(cmdline,params);
#endif
  
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();

#if defined(USE_BFM_A2A) || defined(USE_BFM_LANCZOS)
  BFMsolvers bfm_solvers(cmdline.nthreads, 0.01, 1e-08, 20000, params.jp.solver, params.jp.mobius_scale); //for BFM holds a double and single precision bfm instance. Mass is not important as it is changed when necessary
#endif
  
  if(chdir(params.meas_arg.WorkDirectory)!=0) ERR.General("",fname,"Unable to switch to work directory '%s'\n",params.meas_arg.WorkDirectory);
  double time;

  if(!UniqueID()) printf("Memory prior to config loop:\n");
  printMem();

  //Setup parameters of fields
  typedef typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType Field3DparamType;
  typedef typename A2Apolicies::FermionFieldType::InputParamType Field4DparamType;
  typedef typename A2Asource<typename A2Apolicies::SourcePolicies::ComplexType, typename A2Apolicies::SourcePolicies::MappingPolicy, typename A2Apolicies::SourcePolicies::AllocPolicy>::FieldType SourceFieldType;
  Field4DparamType field4dparams; setupFieldParams<typename A2Apolicies::FermionFieldType>(field4dparams);
  Field3DparamType field3dparams; setupFieldParams<SourceFieldType>(field3dparams);
  
  //-------------------- Main Loop Begin! -------------------- //
  for(int conf = TrajStart; conf < LessThanLimit; conf += params.meas_arg.TrajIncrement) {
    double conf_time = -dclock();
    if(!UniqueID()) std::cout<<"Starting configuration "<<conf<< std::endl;

    if(cmdline.do_split_job)
      doConfigurationSplit(conf,params,cmdline,field3dparams,field4dparams, COMPUTE_EVECS_EXTRA_ARG_PASS);
    else
      doConfiguration(conf,params,cmdline,field3dparams,field4dparams, COMPUTE_EVECS_EXTRA_ARG_PASS);
    
    conf_time += dclock();
    print_time("main","Configuration total",conf_time);
  }//end of config loop

  if(!UniqueID()) printf("Done\n");
  End();
}

