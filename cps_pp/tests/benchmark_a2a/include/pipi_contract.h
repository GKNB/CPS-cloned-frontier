#pragma once

CPS_START_NAMESPACE

template<typename A2Apolicies>
void benchmarkPiPiContractions(const A2AArg &a2a_args){
  StandardPionMomentaPolicy mom_policy;
  MesonFieldMomentumContainer<A2Apolicies> mf_con;

  A2Aparams params(a2a_args);

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  
  typedef A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> MfType; 

  int nmom = mom_policy.nMom();
  
  for(int i=0;i<nmom;i++){      
    ThreeMomentum p = mom_policy.getMesonMomentum(i);
    std::cout << "Adding meson field of momentum " << p.str() << std::endl;
    std::vector<MfType> mf(Lt);
    for(int t=0;t<Lt;t++){
      mf[t].setup(params,params,t,t);
      mf[t].testRandom();
    }

    //Do one call to mult to cover setup costs
    if(i==0){
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

  double timeC(0), timeD(0), timeR(0), timeV(0);
  double* timeCDR[3] = {&timeC, &timeD, &timeR};

  int tsep_pipi = 2;
  int tstep_src = 1;
  
  for(int psrcidx=0; psrcidx < 1; psrcidx++){ //only do one src momentum otherwise it takes too long
    ThreeMomentum p_pi1_src = mom_policy.getMesonMomentum(psrcidx);

    for(int psnkidx=0; psnkidx < 1; psnkidx++){	 //only do one sink momentum also
      fMatrix<typename A2Apolicies::ScalarComplexType> pipi(Lt,Lt);
      ThreeMomentum p_pi1_snk = mom_policy.getMesonMomentum(psnkidx);

      MesonFieldProductStore<A2Apolicies> products; //try to reuse products of meson fields wherever possible (not used ifdef DISABLE_PIPI_PRODUCTSTORE)

      //Predetermine which products we are going to reuse in order to save memory
      MesonFieldProductStoreComputeReuse<A2Apolicies> product_usage;
      char diag[3] = {'C','D','R'};
      for(int d = 0; d < 3; d++)
	ComputePiPiGparity<A2Apolicies>::setupProductStore(product_usage, diag[d], p_pi1_src, p_pi1_snk, tsep_pipi, tstep_src, mf_con);

      product_usage.addAllowedStores(products); //restrict storage only to products we know we are going to reuse
      
      for(int d = 0; d < 3; d++){
	printMem(stringize("Doing pipi figure %c, psrcidx=%d psnkidx=%d",diag[d],psrcidx,psnkidx),0);

	bool redistribute_src = d == 2 && psnkidx == nmom - 1;
	bool redistribute_snk = d == 2;

	double time = -dclock();
	ComputePiPiGparity<A2Apolicies>::compute(pipi, diag[d], p_pi1_src, p_pi1_snk, tsep_pipi, tstep_src, mf_con, products
#ifdef NODE_DISTRIBUTE_MESONFIELDS
						 , redistribute_src, redistribute_snk
#endif
						 );
	time += dclock();
	*timeCDR[d] += time;

	if(!UniqueID()) _mult_impl_base::getTimers().print();	
      }

      std::cout << "Product container contains " << products.size() << " products consuming " << double(products.byte_size())/1024/1024 << " MB. Saved " << products.productsReused() << " products" << std::endl;
    }

    { //V diagram
      printMem(stringize("Doing pipi figure V, pidx=%d",psrcidx),0);
      double time = -dclock();
      fVector<typename A2Apolicies::ScalarComplexType> figVdis(Lt);
      ComputePiPiGparity<A2Apolicies>::computeFigureVdis(figVdis,p_pi1_src,tsep_pipi,mf_con);
      time += dclock();
      timeV += time;
    }
  }//end of psrcidx loop

  print_time("main","Pi-pi figure C",timeC);
  print_time("main","Pi-pi figure D",timeD);
  print_time("main","Pi-pi figure R",timeR);
  print_time("main","Pi-pi figure V",timeV);

  ComputePiPiGparity<A2Apolicies>::timingsC().report();
  ComputePiPiGparity<A2Apolicies>::timingsD().report();
  ComputePiPiGparity<A2Apolicies>::timingsR().report();
  ComputePiPiGparity<A2Apolicies>::timingsV().report();
  
#ifdef MULT_IMPL_CUBLASXT
  if(!UniqueID()) _mult_impl_base::getTimers().print();
  _mult_impl_base::getTimers().reset();
#endif

}

CPS_END_NAMESPACE
