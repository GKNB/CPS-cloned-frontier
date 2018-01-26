#ifndef _AMA_ MAIN_FB_H
#define _AMA_ MAIN_FB_H

CPS_START_NAMESPACE

void measFB(const Props &props_l_F, const Props &props_l_B,
	    const Props &props_h_F, const Props &props_h_B,
	    const Props &props_l_A,
	    const std::vector<int> &tslices_se,
	    const std::string &se_str,
	    const MesonMomenta &pion_momenta,
	    const MesonMomenta &kaon_momenta,
	    const MesonMomenta &su2_singlet_momenta,
	    const CmdLine &cmdline,
	    const GparityAMAarg2 &ama_arg,
	    const std::vector<int> &bk_tseps,
	    const int conf,
	    Lattice &lattice
	    ){
  std::string results_dir(ama_arg.results_dir);
  
  for(int fb=0; fb<2; fb++){
    std::string fb_str = fb == 0 ? "_F" : "_B";

    const Props &props_l_base = fb == 0 ? props_l_F : props_l_B;
    const Props &props_l_shift = fb == 0 ? props_l_B : props_l_F;

    const Props &props_h_base = fb == 0 ? props_h_F : props_h_B;
    const Props &props_h_shift = fb == 0 ? props_h_B : props_h_F;
	
    PropGetterFB sites_l(props_l_base,props_l_shift);
    PropGetterFB sites_h(props_h_base,props_h_shift);
		
    //Pion 2pt LW functions pseudoscalar and axial sinks      
    measurePion2ptLW(sites_l, tslices_se, pion_momenta, results_dir, conf, se_str + fb_str);

    //Kaon 2pt LW functions pseudoscalar and axial sinks
    measureKaon2ptLW(sites_l, sites_h,tslices_se,kaon_momenta,results_dir,conf, se_str + fb_str);
	
    if(GJP.Gparity()){
      WallSinkPropGetterFB<SpinColorFlavorMatrix> wallsites_l(props_l_base, props_l_shift, lattice);
      WallSinkPropGetterFB<SpinColorFlavorMatrix> wallsites_h(props_h_base, props_h_shift, lattice);

      //SU(2) flavor singlet
      measureLightFlavorSingletLW(sites_l,tslices_se,su2_singlet_momenta,results_dir,conf, se_str + fb_str);
	  	  
      //Pion 2pt WW function pseudoscalar sink
      measurePion2ptPPWWGparity(wallsites_l, tslices_se, pion_momenta, results_dir, conf, se_str + fb_str);
	  
      //Kaon 2pt WW function pseudoscalar sink
      measureKaon2ptPPWWGparity(wallsites_l,wallsites_h,tslices_se,kaon_momenta,results_dir,conf, se_str + fb_str);
    }else{
      WallSinkPropGetterFB<WilsonMatrix> wallsites_l(props_l_base, props_l_shift, lattice);
      WallSinkPropGetterFB<WilsonMatrix> wallsites_h(props_h_base, props_h_shift, lattice);
	  
      //Pion 2pt WW function pseudoscalar sink
      measurePion2ptPPWWStandard(wallsites_l, tslices_se, pion_momenta, results_dir, conf, se_str + fb_str);

      //Kaon 2pt WW function pseudoscalar sink
      measureKaon2ptPPWWStandard(wallsites_l,wallsites_h,tslices_se,kaon_momenta,results_dir,conf, se_str + fb_str);
    }

    //BK O_{VV+AA} 3pt contractions
    //Need to ensure that props exist on t0 and t0+tsep for all tseps
    measureBK(sites_l,sites_h,tslices_se,bk_tseps,kaon_momenta,results_dir,conf,se_str+fb_str,cmdline.bk_do_flavor_project);   
  }//fb

  PropGetterStd sites_l_A(props_l_A, BND_CND_APRD);
  //J5 and J5q for mres
  measureMres(sites_l_A,tslices_se,pion_momenta,results_dir,conf, se_str+"_A",cmdline.mres_do_flavor_project);
}
    

void runFB(const CmdLine &cmdline, GnoneFbfm &lattice,
	   const DoArg &do_arg,
	   const LancArg &lanc_arg_l,
	   const LancArg &lanc_arg_h,
	   const GparityAMAarg2 &ama_arg,
	   const QuarkMomenta &light_quark_momenta,
	   const QuarkMomenta &heavy_quark_momenta,
	   const MesonMomenta &pion_momenta,
	   const MesonMomenta &kaon_momenta,
	   const MesonMomenta &su2_singlet_momenta,
	   const int conf){
  std::vector<int> tslice_sloppy, tslice_exact, bk_tseps;
  getTimeslices(tslice_sloppy, tslice_exact, bk_tseps,ama_arg,cmdline.random_exact_tsrc_offset);
  
  LanczosPtrType lanc_l_P(NULL), lanc_l_A(NULL), lanc_h_P(NULL), lanc_h_A(NULL);

  //Light quark Lanczos
  Float time = -dclock();
  if(!cmdline.disable_lanczos) lanc_l_P = doLanczos(lattice,lanc_arg_l,BND_CND_PRD);
  time += dclock();    
  print_time("main","Light quark Lanczos PRD",time);

  time = -dclock();
  if(!cmdline.disable_lanczos) lanc_l_A = doLanczos(lattice,lanc_arg_l,BND_CND_APRD);
  time += dclock();    
  print_time("main","Light quark Lanczos APRD",time);

  //Heavy quark Lanczos
  time = -dclock();
  if(!cmdline.disable_lanczos) lanc_h_P = doLanczos(lattice,lanc_arg_h,BND_CND_PRD);
  time += dclock();    
  print_time("main","Heavy quark Lanczos PRD",time);

  time = -dclock();
  if(!cmdline.disable_lanczos) lanc_h_A = doLanczos(lattice,lanc_arg_h,BND_CND_APRD);
  time += dclock();    
  print_time("main","Heavy quark Lanczos APRD",time);

    
  for(int se=0; se<2; se++){
    const std::vector<int> &tslices_se = se == 0 ? tslice_sloppy : tslice_exact;
    const double precision_se = se == 0 ? ama_arg.sloppy_precision : ama_arg.exact_precision;
    
#ifdef DO_MOM_SOURCE
    {
      std::string se_str = se == 0 ? "_momsrc_sloppy" : "_momsrc_exact";
      
      //Light quark props
      Props props_l_P, props_l_A;
      computeMomSourcePropagators(props_l_P, ama_arg.ml, precision_se, tslices_se, light_quark_momenta, BND_CND_PRD, true, lattice, lanc_l_P.get(), cmdline.random_prop_solns);
      computeMomSourcePropagators(props_l_A, ama_arg.ml, precision_se, tslices_se, light_quark_momenta, BND_CND_APRD, true, lattice, lanc_l_A.get(), cmdline.random_prop_solns);

      Props props_l_F, props_l_B;
      combinePA(props_l_F, props_l_B, props_l_P, props_l_A);

      //Heavy quark props
      Props props_h_P, props_h_A;
      computeMomSourcePropagators(props_h_P, ama_arg.mh, precision_se, tslices_se, heavy_quark_momenta, BND_CND_PRD, true, lattice, lanc_h_P.get(), cmdline.random_prop_solns);
      computeMomSourcePropagators(props_h_A, ama_arg.mh, precision_se, tslices_se, heavy_quark_momenta, BND_CND_APRD, true, lattice, lanc_h_A.get(), cmdline.random_prop_solns);

      Props props_h_F, props_h_B;
      combinePA(props_h_F, props_h_B, props_h_P, props_h_A);

      measFB(props_l_F, props_l_B, props_h_F, props_h_B, props_l_A, tslices_se, se_str, pion_momenta, kaon_momenta, su2_singlet_momenta, cmdline, ama_arg, bk_tseps, conf, lattice);
    }
#endif

#ifdef DO_Z2_MOM_SOURCE
    {
      std::string se_str = se == 0 ? "_z2momsrc_sloppy" : "_z2momsrc_exact";

       //Light quark props
      Props props_l_P, props_l_A;
      computeRandMomSourcePropagators(props_l_P, ZTWO, ama_arg.ml, precision_se, tslices_se, light_quark_momenta, BND_CND_PRD, true, lattice, lanc_l_P.get(), cmdline.random_prop_solns);
      computeRandMomSourcePropagators(props_l_A, ZTWO, ama_arg.ml, precision_se, tslices_se, light_quark_momenta, BND_CND_APRD, true, lattice, lanc_l_A.get(), cmdline.random_prop_solns);

      Props props_l_F, props_l_B;
      combinePA(props_l_F, props_l_B, props_l_P, props_l_A);

      //Heavy quark props
      Props props_h_P, props_h_A;
      computeRandMomSourcePropagators(props_h_P, ZTWO, ama_arg.mh, precision_se, tslices_se, heavy_quark_momenta, BND_CND_PRD, true, lattice, lanc_h_P.get(), cmdline.random_prop_solns);
      computeRandMomSourcePropagators(props_h_A, ZTWO, ama_arg.mh, precision_se, tslices_se, heavy_quark_momenta, BND_CND_APRD, true, lattice, lanc_h_A.get(), cmdline.random_prop_solns);

      Props props_h_F, props_h_B;
      combinePA(props_h_F, props_h_B, props_h_P, props_h_A);

      measFB(props_l_F, props_l_B, props_h_F, props_h_B, props_l_A, tslices_se, se_str, pion_momenta, kaon_momenta, su2_singlet_momenta, cmdline, ama_arg, bk_tseps, conf, lattice);
    }
#endif
      
  }//se
  cps::sync();
  freeLanczos(lanc_l_P);
  freeLanczos(lanc_l_A);
  freeLanczos(lanc_h_P);
  freeLanczos(lanc_h_A);
}

CPS_END_NAMESPACE

#endif
