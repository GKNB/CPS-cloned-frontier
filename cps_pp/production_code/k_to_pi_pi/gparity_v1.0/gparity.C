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

USING_NAMESPACE_CPS

int main(int argc,char *argv[])
{
  Start(&argc,&argv);
  
  if(argc!=4){
    ERR.General("","main()","Not enough arguments. Require DoArg, JobPropagatorArgs and GparityContractArg");
  }

  DoArg do_arg;
  if(!do_arg.Decode(argv[1],"do_arg")){
    ERR.General("","main()","Failed to decode %s\n",argv[1]); exit(-1);
  }
  JobPropagatorArgs prop_args;
  if(!prop_args.Decode(argv[2],"prop_arg")){
    ERR.General("","main()","Failed to decode %s\n",argv[2]); exit(-1);
  }
  GparityContractArg contract_args;
  if(!contract_args.Decode(argv[3],"contract_arg")){
    ERR.General("","main()","Failed to decode %s\n",argv[3]); exit(-1);
  }  
  if(contract_args.conf_start >= contract_args.conf_lessthan || contract_args.conf_incr == 0){
    ERR.General("","main()","Invalid configuration args");
  }

  if(!UniqueID()){
    printf("contract_args contains %d measurements:\n",contract_args.meas.meas_len);
    for(int m=0;m<contract_args.meas.meas_len;m++) contract_args.meas.meas_val[m].print();
  }

  char *c = contract_args.config_fmt;

  if(UniqueID()==0) printf("Configuration format is '%s'\n",contract_args.config_fmt);
  bool found(false);
  while(*c!='\0'){
    if(*c=='\%' && *(c+1)=='d'){ found=true; break; }
    c++;
  }
  if(!found) ERR.General("","main()","GparityContractArg config format '%s' does not contain a %%d",contract_args.config_fmt);

  GJP.Initialize(do_arg);
  GJP.StartConfKind(START_CONF_MEM); //we will handle the gauge field read/write thankyou!
  LRG.Initialize();

  PropManager::setup(prop_args);
  if(UniqueID()==0){
    printf("prop_args contains %d propagators\n", prop_args.props.props_len);
    prop_args.print();
  }
  
  GwilsonFdwf lattice;
  CommonArg carg("label","filename");
  char load_config_file[1000];
  
  for(int conf=contract_args.conf_start; conf < contract_args.conf_lessthan; conf += contract_args.conf_incr){
    PropManager::startNewTraj();

    //Read/generate the gauge configuration 
    if(do_arg.start_conf_kind == START_CONF_FILE){
    
      if(sprintf(load_config_file,contract_args.config_fmt,conf) < 0){
	ERR.General("","main()","Congfiguration filename creation problem : %s | %s",load_config_file,contract_args.config_fmt);
      }
      //load the configuration
      ReadLatticeParallel readLat;
      if(UniqueID()==0) printf("Reading: %s (NERSC-format)\n",load_config_file);
      readLat.read(lattice,load_config_file);
      if(UniqueID()==0) printf("Config read.\n");
    }else if(do_arg.start_conf_kind == START_CONF_ORD){
      if(!UniqueID()) printf("Using unit gauge links\n");
      lattice.SetGfieldOrd();
    }else if(do_arg.start_conf_kind == START_CONF_DISORD){
      if(!UniqueID()) printf("Using random gauge links\n");
      lattice.SetGfieldDisOrd();
    }else{
      ERR.General("","main()","Invalid do_arg.start_conf_kind\n");
    }

    //Checksum the lattice
    {
      unsigned int gcsum = lattice.CheckSum();
      QioControl qc;
      gcsum = qc.globalSumUint(gcsum);
      
      if(UniqueID()==0) printf("Gauge field checksum %u\n",gcsum);
    }

    //Gauge fix lattice if required
    if(contract_args.fix_gauge.fix_gauge_kind != FIX_GAUGE_NONE){
      AlgFixGauge fix_gauge(lattice,&carg,&contract_args.fix_gauge);
      fix_gauge.run();
    }

    //Perform the inversions/contractions
    AlgGparityContract contract(lattice,carg,contract_args);
    contract.run(conf);

    //Free the gauge fixing matrices and reset for next config
    if(contract_args.fix_gauge.fix_gauge_kind != FIX_GAUGE_NONE) lattice.FixGaugeFree();
  }

  if(UniqueID()==0){
    printf("Main job complete\n"); 
    fflush(stdout);
  }
  
  return 0;
}
