#ifndef _KTOPIPI_MAIN_A2A_MISC_H_
#define _KTOPIPI_MAIN_A2A_MISC_H_

void setupJob(int argc, char **argv, const Parameters &params, const CommandLineArgs &cmdline){
  initCPS(argc, argv, params.do_arg, cmdline.nthreads);
  
#ifdef NODE_DISTRIBUTE_MESONFIELDS
  LOGA2A << "Using node distribution of meson fields" << std::endl;
#endif
#ifdef MEMTEST_MODE
  LOGA2A << "Running in MEMTEST MODE (so don't expect useful results)" << std::endl;
#endif
  
#ifdef A2A_LANCZOS_SINGLE
  if(!cmdline.evecs_single_prec) ERR.General("",fname,"Must use single-prec eigenvectors when doing Lanczos in single precision\n");
#endif
  
  if(cmdline.double_latt) SerialIO::dbl_latt_storemode = true;

  if(!cmdline.tune_lanczos_light && !cmdline.tune_lanczos_heavy){ 
    assert(params.a2a_arg.nl <= params.lanc_arg.N_true_get);
    assert(params.a2a_arg_s.nl <= params.lanc_arg_s.N_true_get);
  }

  if(cmdline.old_gparity_cfg){
    LOGA2A << "Correcting for incorrect factor of 2 in G-parity config header plaquette" << std::endl;
    LatticeHeader::GparityMultPlaqByTwo() = true;
  }
  printMem("Initial memory post-initialize");
}

#ifdef BNL_KNL_PERFORMANCE_CHECK
void bnl_knl_performance_check(const CommandLineArgs &args,const Parameters &params){
  auto lat = createFgridLattice<typename A2Apolicies::FgridGFclass>(params.jp);
  lat->SetGfieldOrd(); //so we don't interfere with the RNG state
  double node_perf = gridBenchmark<A2Apolicies>(*lat);  
  delete lat;
  if(node_perf < args.bnl_knl_minperf){ LOGA2A << "BAD PERFORMANCE" << std::endl; exit(-1); }
}
#endif

void runInitialGridBenchmarks(const CommandLineArgs &cmdline, const Parameters &params){
#if defined(USE_GRID) && defined(USE_GRID_A2A)
  if(cmdline.run_initial_grid_benchmarks){
    auto lat = createFgridLattice<typename A2Apolicies::FgridGFclass>(params.jp);
    gridBenchmark<A2Apolicies>(*lat);
    gridBenchmarkSinglePrec<A2Apolicies>(*lat);
    delete lat;
  }
#endif
}

void doGaugeFix(Lattice &lat, const bool skip_gauge_fix, const Parameters &params){
  doGaugeFix(lat,skip_gauge_fix,params.fix_gauge_arg);
}

void readGaugeRNG(const Parameters &params, const CommandLineArgs &cmdline){
  readGaugeRNG(params.do_arg, params.meas_arg, cmdline.double_latt);
}

#endif
