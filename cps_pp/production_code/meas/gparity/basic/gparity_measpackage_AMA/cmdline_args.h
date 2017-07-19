#ifndef _CMDLINE_ARGS_H
#define _CMDLINE_ARGS_H

#include <util/lattice/fbfm.h>

CPS_START_NAMESPACE

struct CmdLine{
  bool lanczos_tune_l;
  bool lanczos_tune_h;
  bool dbl_latt_storemode;
  bool mres_do_flavor_project;
  bool bk_do_flavor_project;
  bool do_alternative_mom;
  bool disable_lanczos;
  bool random_prop_solns; //don't invert, just make the solutions random spin-color-flavor matrices
  bool skip_gauge_fix;
  bool random_exact_tsrc_offset; //randomly shift the full set of exact src timeslices
  bool rng_test; //just load config, generate some uniform random numbers then move onto next config
  int tshift;

  CmdLine(int argc,char *argv[]){
    lanczos_tune_l = false;
    lanczos_tune_h = false;
    dbl_latt_storemode = false;
    Fbfm::use_mixed_solver = true;
    mres_do_flavor_project = true;
    bk_do_flavor_project = true;
    do_alternative_mom = true;
    disable_lanczos = false;
    random_prop_solns = false; //don't invert, just make the solutions random spin-color-flavor matrices
    skip_gauge_fix = false;
    random_exact_tsrc_offset = true; //randomly shift the full set of exact src timeslices
    rng_test = false; //just load config, generate some uniform random numbers then move onto next config
    tshift = 0;
    
    int i = 1;
    while(i<argc-1){
      if( std::string(argv[i]) == "-lanczos_tune_l" ){
	lanczos_tune_l = true;
	i++;
      }else if( std::string(argv[i]) == "-lanczos_tune_h" ){
	lanczos_tune_h = true;
	i++;
      }else if( std::string(argv[i]) == "-load_dbl_latt" ){
	if(!UniqueID()) printf("Loading double latt\n");
	dbl_latt_storemode = true;
	i++;
      }else if( std::string(argv[i]) == "-disable_random_exact_tsrc_offset" ){
	if(!UniqueID()) printf("Disabling random offset of exact tsrc\n");
	random_exact_tsrc_offset = false;
	i++;
      }else if( std::string(argv[i]) == "-disable_mixed_solver" ){
	if(!UniqueID()) printf("Disabling mixed solver\n");
	Fbfm::use_mixed_solver = false;
	i++;
      }else if( std::string(argv[i]) == "-disable_lanczos" ){
	if(!UniqueID()) printf("Not computing or using low-modes\n");
	disable_lanczos = true;
	i++;
      }else if( std::string(argv[i]) == "-disable_mres_flavor_project" ){ //for comparison with old code
	if(!UniqueID()) printf("Disabling mres flavor project\n");
	mres_do_flavor_project = false;
	i++;
      }else if( std::string(argv[i]) == "-disable_bk_flavor_project" ){ //for comparison with old code
	if(!UniqueID()) printf("Disabling BK flavor project\n");
	bk_do_flavor_project = false;
	i++;
      }else if( std::string(argv[i]) == "-disable_use_alternate_mom" ){ 
	if(!UniqueID()) printf("Disabling use of alternative momentum combinations\n");
	do_alternative_mom = false;
	i++;
      }else if( std::string(argv[i]) == "-random_prop_solns"){
	if(!UniqueID()) printf("Not inverting, just using random propagator solutions\n");
	random_prop_solns = true;
	i++;
      }else if( std::string(argv[i]) == "-skip_gauge_fix"){
	skip_gauge_fix = true;
	if(!UniqueID()){ printf("Skipping gauge fixing\n"); fflush(stdout); }
	i++;
      }else if( std::string(argv[i]) == "-rng_test"){
	rng_test = true;
	if(!UniqueID()){ printf("Doing RNG test\n"); fflush(stdout); }
	i++;
      }else if( std::string(argv[i]) == "-tshift_gauge"){
	std::stringstream ss; ss << argv[i+1]; ss >> tshift;
	if(!UniqueID()){ printf("Shifting gauge field by %d in time direction\n",tshift); fflush(stdout); }
	i+=2;
      }else{
	ERR.General("","main","Unknown argument: %s",argv[i]);
      }
    }
  }
};
  
CPS_END_NAMESPACE

#endif
