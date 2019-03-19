#ifndef _AMA_ MAIN_FB_H
#define _AMA_ MAIN_FB_H

CPS_START_NAMESPACE

void runStd(const CmdLine &cmdline, GnoneFbfm &lattice,
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
  std::string results_dir(ama_arg.results_dir);

  std::vector<int> tslice_sloppy, tslice_exact, bk_tseps;
  getTimeslices(tslice_sloppy, tslice_exact, bk_tseps,ama_arg,cmdline.random_exact_tsrc_offset);
  
  LanczosPtrType lanc_l(NULL), lanc_h(NULL);
  Float time = -dclock();
  if(!disable_lanczos) lanc_l = doLanczos(lattice,lanc_arg_l,GJP.Tbc());
  time += dclock();    
  print_time("main","Light quark Lanczos",time);

  time = -dclock();
  if(!disable_lanczos) lanc_h = doLanczos(lattice,lanc_arg_h,GJP.Tbc());
  time += dclock();    
  print_time("main","Heavy quark Lanczos",time);

  for(int se=0; se<2; se++){
    std::string se_str = se == 0 ? "_sloppy" : "_exact";
      
    //Light quark props
    Props props_l;
    computeMomSourcePropagators(props_l, ama_arg.ml,
				se == 0 ? ama_arg.sloppy_precision : ama_arg.exact_precision,
				se == 0 ? tslice_sloppy : tslice_exact,
				light_quark_momenta, GJP.Tbc(), true, lattice, lanc_l.get(), random_prop_solns);
    //Heavy quark props
    Props props_h;
    computeMomSourcePropagators(props_h, ama_arg.mh,
				se == 0 ? ama_arg.sloppy_precision : ama_arg.exact_precision,
				se == 0 ? tslice_sloppy : tslice_exact,
				heavy_quark_momenta, GJP.Tbc(), true, lattice, lanc_h.get(), random_prop_solns);
      
    const std::vector<int> &tslices_se = se == 0 ? tslice_sloppy : tslice_exact;

    PropGetterStd sites_l(props_l, GJP.Tbc());
    PropGetterStd sites_h(props_h, GJP.Tbc());
		
    //Pion 2pt LW functions pseudoscalar and axial sinks      
    measurePion2ptLW(sites_l, tslices_se, pion_momenta, results_dir, conf, se_str);

    //Kaon 2pt LW functions pseudoscalar and axial sinks
    measureKaon2ptLW(sites_l, sites_h,tslices_se,kaon_momenta,results_dir,conf, se_str);
	
    if(GJP.Gparity()){
      WallSinkPropGetterStd<SpinColorFlavorMatrix> wallsites_l(props_l, GJP.Tbc(), lattice);
      WallSinkPropGetterStd<SpinColorFlavorMatrix> wallsites_h(props_h, GJP.Tbc(), lattice);

      //SU(2) flavor singlet
      measureLightFlavorSingletLW(sites_l,tslices_se,su2_singlet_momenta,results_dir,conf, se_str);
	
      //Pion 2pt WW function pseudoscalar sink
      measurePion2ptPPWWGparity(wallsites_l, tslices_se, pion_momenta, results_dir, conf, se_str);
	
      //Kaon 2pt WW function pseudoscalar sink
      measureKaon2ptPPWWGparity(wallsites_l,wallsites_h,tslices_se,kaon_momenta,results_dir,conf, se_str);
    }else{
      WallSinkPropGetterStd<WilsonMatrix> wallsites_l(props_l, GJP.Tbc(), lattice);
      WallSinkPropGetterStd<WilsonMatrix> wallsites_h(props_h, GJP.Tbc(), lattice);
	  
      //Pion 2pt WW function pseudoscalar sink
      measurePion2ptPPWWStandard(wallsites_l, tslices_se, pion_momenta, results_dir, conf, se_str);
	
      //Kaon 2pt WW function pseudoscalar sink
      measureKaon2ptPPWWStandard(wallsites_l,wallsites_h,tslices_se,kaon_momenta,results_dir,conf, se_str);
    }
      
    //BK O_{VV+AA} 3pt contractions
    //Need to ensure that props exist on t0 and t0+tsep for all tseps
    measureBK(sites_l,sites_h,tslices_se,bk_tseps,kaon_momenta,results_dir,conf,se_str,bk_do_flavor_project);   
      
    //J5 and J5q for mres
    measureMres(sites_l,tslices_se,pion_momenta,results_dir,conf, se_str,mres_do_flavor_project);
  }
}

CPS_END_NAMESPACE

#endif
