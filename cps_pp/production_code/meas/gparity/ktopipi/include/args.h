#ifndef _KTOPIPI_MAIN_A2A_ARGS_H_
#define _KTOPIPI_MAIN_A2A_ARGS_H_

//Store/read job parameters
struct Parameters{
  CommonArg common_arg;
  CommonArg common_arg2;
  DoArg do_arg;
  JobParams jp;
  MeasArg meas_arg;
  FixGaugeArg fix_gauge_arg;
  A2AArg a2a_arg;
  A2AArg a2a_arg_s;
  LancArg lanc_arg;
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
    if(!a2a_arg.Decode("a2a_arg.vml","a2a_arg")){
      a2a_arg.Encode("a2a_arg.templ","a2a_arg");
      VRB.Result("Parameters","Parameters","Can't open a2a_arg.vml!\n");exit(1);
    }
    if(!a2a_arg_s.Decode("a2a_arg_s.vml","a2a_arg_s")){
      a2a_arg_s.Encode("a2a_arg_s.templ","a2a_arg_s");
      VRB.Result("Parameters","Parameters","Can't open a2a_arg_s.vml!\n");exit(1);
    }
    if(!lanc_arg.Decode("lanc_arg.vml","lanc_arg")){
      lanc_arg.Encode("lanc_arg.templ","lanc_arg");
      VRB.Result("Parameters","Parameters","Can't open lanc_arg.vml!\n");exit(1);
    }
    if(!lanc_arg_s.Decode("lanc_arg_s.vml","lanc_arg_s")){
      lanc_arg_s.Encode("lanc_arg_s.templ","lanc_arg_s");
      VRB.Result("Parameters","Parameters","Can't open lanc_arg_s.vml!\n");exit(1);
    }
    if(!fix_gauge_arg.Decode("fix_gauge_arg.vml","fix_gauge_arg")){
      fix_gauge_arg.Encode("fix_gauge_arg.templ","fix_gauge_arg");
      VRB.Result("Parameters","Parameters","Can't open fix_gauge_arg.vml!\n");exit(1);
    }

    common_arg.set_filename(meas_arg.WorkDirectory);

#ifdef DO_EXTENDED_CALC_V1
    //Check for existence and readability of input file for pipi    
    std::vector<CorrelatorMomenta> correlators;
    parsePiPiMomFile(correlators, "pipi_correlators.in");
#endif
  }

};

#endif

