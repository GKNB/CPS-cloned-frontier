#pragma once

CPS_START_NAMESPACE

template<typename A2Apolicies>
void benchmarkPiPiContractions(const A2AArg &a2a_args){
  printMem("Benchmark start");
  
  StandardPionMomentaPolicy mom_policy;

  A2Aparams params(a2a_args);

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  
  typedef A2AvectorW<A2Apolicies> Wtype;
  typedef A2AvectorV<A2Apolicies> Vtype;
  typedef A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> MfType; 
  typedef ComputePiPiGparity<Vtype,Wtype> Compute;

  MesonFieldMomentumContainer<MfType> mf_con;

  //Only do the first mom combination with total mom 0
  int psrcidx(0), psnkidx(0);
  ThreeMomentum p_pi1_src = mom_policy.getMesonMomentum(psrcidx);
  ThreeMomentum p_pi1_snk = mom_policy.getMesonMomentum(psnkidx);
  ThreeMomentum p_pi2_src = -p_pi1_src;
  ThreeMomentum p_pi2_snk = -p_pi1_snk;
  std::set<ThreeMomentum> allp;
  allp.insert(p_pi1_src);   allp.insert(p_pi1_snk);   allp.insert(p_pi2_src);   allp.insert(p_pi2_snk); 

  for(int i=0;i<allp.size();i++){
    ThreeMomentum p = *std::next(allp.begin(),i);
    std::cout << "Adding meson field of momentum " << p.str() << std::endl;
    std::vector<MfType> mf(Lt);
    for(int t=0;t<Lt;t++){
      mf[t].setup(params,params,t,t);
      mf[t].testRandom();
    }

    //Do one call to mult to cover setup costs
    if(i==0){
      double mfMB =  double(mf[0].byte_size())/1024./1024.;
      std::cout << "Meson field memory size per timeslice " << mfMB << "MB, all timeslices " << mfMB * Lt << std::endl;
      
      double time = -dclock();
      MfType tmp;
      mult(tmp, mf[0],mf[0], true);
      time += dclock();
    
      std::cout << "Initial mult call to amortize overheads "<< time << "s" << std::endl;	
#ifdef MULT_IMPL_CUBLASXT
      _mult_impl_base::getTimers().reset();
#endif
							       
    }
    
    mf_con.moveAdd(p, mf);
  }

  printMem("After mesonfield creation");
  
  double timeC(0), timeD(0), timeR(0), timeV(0);
  double* timeCDR[3] = {&timeC, &timeD, &timeR};

  int tsep_pipi = 2;
  int tstep_src = 1;

  fMatrix<typename A2Apolicies::ScalarComplexType> pipi(Lt,Lt);
  
  MesonFieldProductStore<MfType> products; //try to reuse products of meson fields wherever possible (not used ifdef DISABLE_PIPI_PRODUCTSTORE)

  //Predetermine which products we are going to reuse in order to save memory
  MesonFieldProductStoreComputeReuse<MfType> product_usage;
  char diag[3] = {'C','D','R'};
  for(int d = 0; d < 3; d++)
    Compute::setupProductStore(product_usage, diag[d], p_pi1_src, p_pi1_snk, tsep_pipi, tstep_src, mf_con);

  product_usage.addAllowedStores(products); //restrict storage only to products we know we are going to reuse

  printMem("Prior to diagram loop");
  
  for(int d = 0; d < 3; d++){
    printMem(stringize("Doing pipi figure %c",diag[d]),0);

    bool redistribute_src = d == 2;
    bool redistribute_snk = d == 2;

    typename Compute::Options opt;
    opt.redistribute_pi1_src = opt.redistribute_pi2_src = redistribute_src;
    opt.redistribute_pi1_snk = opt.redistribute_pi2_snk = redistribute_snk;
	
    double time = -dclock();
    Compute::compute(pipi, diag[d], p_pi1_src, p_pi1_snk, tsep_pipi, tstep_src, mf_con, products, opt);
    time += dclock();
    *timeCDR[d] += time;

#ifdef GPU_VEC
    if(!UniqueID()) _mult_impl_base::getTimers().print();	
#endif
  }

  std::cout << "Product container contains " << products.size() << " products consuming " << double(products.byte_size())/1024/1024 << " MB. Saved " << products.productsReused() << " products" << std::endl;


  { //V diagram
    printMem(stringize("Doing pipi figure V, pidx=%d",psrcidx),0);
    double time = -dclock();
    fVector<typename A2Apolicies::ScalarComplexType> figVdis(Lt);
    Compute::computeFigureVdis(figVdis,p_pi1_src,tsep_pipi,mf_con);
    time += dclock();
    timeV += time;
  }

  print_time("main","Pi-pi figure C",timeC);
  print_time("main","Pi-pi figure D",timeD);
  print_time("main","Pi-pi figure R",timeR);
  print_time("main","Pi-pi figure V",timeV);

  Compute::timingsC().report();
  Compute::timingsD().report();
  Compute::timingsR().report();
  Compute::timingsV().report();
  
#ifdef MULT_IMPL_CUBLASXT
  if(!UniqueID()) _mult_impl_base::getTimers().print();
  _mult_impl_base::getTimers().reset();
#endif

  printMem("Benchmark end");
}

CPS_END_NAMESPACE
