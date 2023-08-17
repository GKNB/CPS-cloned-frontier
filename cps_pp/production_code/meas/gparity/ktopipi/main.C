#ifndef DISABLE_NODE_DISTRIBUTE_MESONFIELDS
#define NODE_DISTRIBUTE_MESONFIELDS //Save memory by keeping meson fields only on single node until needed
#endif
#include <alg/a2a/a2a_fields.h>
#include <alg/a2a/ktopipi_gparity.h>

#include "include/main.h"

int main (int argc,char **argv )
{
  HeapProfilerStart("distribute");
  
  const char *fname="main(int,char**)";
  Start(&argc, &argv);

  MemoryMonitor mmon; mmon.Start();
  
  const char *cname=argv[0];
  const int TrajStart = atoi(argv[2]);
  const int LessThanLimit = atoi(argv[3]);

  CommandLineArgs cmdline(argc,argv,4); //control the functioning of the program from the command line    
  Parameters params(argv[1]);

  setupJob(argc, argv, params, cmdline);
  
  assert(GJP.Gparity());

#ifdef BNL_KNL_PERFORMANCE_CHECK
  bnl_knl_performance_check(cmdline,params);
#endif
  
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();
 
  if(chdir(params.meas_arg.WorkDirectory)!=0) ERR.General("",fname,"Unable to switch to work directory '%s'\n",params.meas_arg.WorkDirectory);
  double time;

  LOGA2A << "Memory prior to config loop:" << std::endl;
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
    LOGA2A <<"Starting configuration "<<conf<< std::endl;

    if(cmdline.do_LL_props_only){
      if(cmdline.do_split_job)
	doConfigurationLLpropsSplit(conf,params,cmdline,field3dparams,field4dparams);
      else
	doConfigurationLLprops(conf,params,cmdline,field3dparams,field4dparams);
    }else{
      if(cmdline.do_split_job)
	doConfigurationSplit(conf,params,cmdline,field3dparams,field4dparams);
      else
	doConfiguration(conf,params,cmdline,field3dparams,field4dparams);
    }
    
    conf_time += dclock();
    a2a_print_time("main","Configuration total",conf_time);
    
    //Write a file to work directory indicate that the config has been complete
    {
      std::ostringstream fn; fn << "done." << conf;
      std::ofstream of(fn.str().c_str());
      of << "Config " << conf << " complete in " << conf_time << "s\n";
    }

    LOGA2A << "Completed configuration " << conf << std::endl;
    
  }//end of config loop

  mmon.Stop();
  
  LOGA2A << "Done" << std::endl;
  End();

  HeapProfilerStop();
}

