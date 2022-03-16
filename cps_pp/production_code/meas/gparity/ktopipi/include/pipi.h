#ifndef _KTOPIPI_MAIN_A2A_PIPI_H_
#define _KTOPIPI_MAIN_A2A_PIPI_H_

template<typename PionMomentumPolicy>
void computePiPi2pt(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, const PionMomentumPolicy &pion_mom, const int conf, const Parameters &params, const std::string &postpend = ""){
  const int nmom = pion_mom.nMom();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();

  if(!UniqueID()) printf("Computing pi-pi 2pt function\n");
  double timeC(0), timeD(0), timeR(0), timeV(0);
  double* timeCDR[3] = {&timeC, &timeD, &timeR};

  for(int psrcidx=0; psrcidx < nmom; psrcidx++){
    ThreeMomentum p_pi1_src = pion_mom.getMesonMomentum(psrcidx);

    for(int psnkidx=0; psnkidx < nmom; psnkidx++){	
      fMatrix<typename A2Apolicies::ScalarComplexType> pipi(Lt,Lt);
      ThreeMomentum p_pi1_snk = pion_mom.getMesonMomentum(psnkidx);

      MesonFieldProductStore<A2Apolicies> products; //try to reuse products of meson fields wherever possible (not used ifdef DISABLE_PIPI_PRODUCTSTORE)
      
      char diag[3] = {'C','D','R'};
      for(int d = 0; d < 3; d++){
	printMem(stringize("Doing pipi figure %c, psrcidx=%d psnkidx=%d",diag[d],psrcidx,psnkidx),0);

#ifdef PIPI_FORCE_REDISTRIBUTE
	bool redistribute_src = true;
	bool redistribute_snk = true;
#else
	bool redistribute_src = d == 2 && psnkidx == nmom - 1;
	bool redistribute_snk = d == 2;
#endif	

	double time = -dclock();
	ComputePiPiGparity<A2Apolicies>::compute(pipi, diag[d], p_pi1_src, p_pi1_snk, params.jp.pipi_separation, params.jp.tstep_pipi, mf_ll_con, products
#ifdef NODE_DISTRIBUTE_MESONFIELDS
						 , redistribute_src, redistribute_snk
#endif
						 );
	std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_Figure" << diag[d] << "_sep" << params.jp.pipi_separation;
#ifndef DAIQIAN_PION_PHASE_CONVENTION
	os << "_mom" << p_pi1_src.file_str(2) << "_mom" << p_pi1_snk.file_str(2);
#else
	os << "_mom" << (-p_pi1_src).file_str(2) << "_mom" << (-p_pi1_snk).file_str(2);
#endif
	os << postpend;
	pipi.write(os.str());
#ifdef WRITE_HEX_OUTPUT
	os << ".hexfloat";
	pipi.write(os.str(),true);
#endif	  
	time += dclock();
	*timeCDR[d] += time;
      }
    }

    { //V diagram
      printMem(stringize("Doing pipi figure V, pidx=%d",psrcidx),0);
      double time = -dclock();
      fVector<typename A2Apolicies::ScalarComplexType> figVdis(Lt);
      ComputePiPiGparity<A2Apolicies>::computeFigureVdis(figVdis,p_pi1_src,params.jp.pipi_separation,mf_ll_con);
      std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_FigureVdis_sep" << params.jp.pipi_separation;
#ifndef DAIQIAN_PION_PHASE_CONVENTION
      os << "_mom" << p_pi1_src.file_str(2);
#else
      os << "_mom" << (-p_pi1_src).file_str(2);
#endif
      os << postpend;
      figVdis.write(os.str());
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      figVdis.write(os.str(),true);
#endif	
      time += dclock();
      timeV += time;
    }
  }//end of psrcidx loop

  print_time("main","Pi-pi figure C",timeC);
  print_time("main","Pi-pi figure D",timeD);
  print_time("main","Pi-pi figure R",timeR);
  print_time("main","Pi-pi figure V",timeV);

  printMem("Memory after pi-pi 2pt function computation");
}

inline std::string momPrint(const ThreeMomentum &p){ 
#ifndef DAIQIAN_PION_PHASE_CONVENTION
  return p.file_str(2);
#else
  return (-p).file_str(2);
#endif
}

template<typename PionMomentumPolicy>
void computePiPi2ptFromFile(MesonFieldMomentumContainer<A2Apolicies> &mf_ll_con, const std::string &pipi_corr_file, const PionMomentumPolicy &pion_mom, const int conf, const Parameters &params, const std::string &postpend = ""){
  const int nmom = pion_mom.nMom();
  const int Lt = GJP.Tnodes() * GJP.TnodeSites();

  if(!UniqueID()) printf("Computing pi-pi 2pt function\n");
  double timeC(0), timeD(0), timeR(0), timeV(0);
  double* timeCDR[3] = {&timeC, &timeD, &timeR};

  char diag[3] = {'C','D','R'};

  std::vector<CorrelatorMomenta> correlators;
  parsePiPiMomFile(correlators, pipi_corr_file);
  
  for(int c=0;c<correlators.size();c++){
    fMatrix<typename A2Apolicies::ScalarComplexType> pipi(Lt,Lt);
    MesonFieldProductStore<A2Apolicies> products;

    //Predetermine which products we are going to reuse in order to save memory
    MesonFieldProductStoreComputeReuse<A2Apolicies> product_usage;
    char diag[3] = {'C','D','R'};
    for(int d = 0; d < 3; d++)
      ComputePiPiGparity<A2Apolicies>::setupProductStore(product_usage, diag[d],
							 correlators[c].pi1_src, correlators[c].pi2_src,
							 correlators[c].pi1_snk, correlators[c].pi2_snk,
							 params.jp.pipi_separation, params.jp.tstep_pipi, mf_ll_con);
    
    product_usage.addAllowedStores(products); //restrict storage only to products we know we are going to reuse
    
    for(int d = 0; d < 3; d++){
      printMem(stringize("Doing pipi figure %c, pi1_src=%s pi2_src=%s pi1_snk=%s pi2_snk=%s",diag[d],
			 correlators[c].pi1_src.str().c_str(), correlators[c].pi2_src.str().c_str(),
			 correlators[c].pi1_snk.str().c_str(), correlators[c].pi2_snk.str().c_str()), 0);
      
      bool redistribute_src = true;
      bool redistribute_snk = true;
      
      double time = -dclock();
      ComputePiPiGparity<A2Apolicies>::compute(pipi, diag[d], 
					       correlators[c].pi1_src, correlators[c].pi2_src,
					       correlators[c].pi1_snk, correlators[c].pi2_snk,
					       params.jp.pipi_separation, params.jp.tstep_pipi, mf_ll_con, products
#ifdef NODE_DISTRIBUTE_MESONFIELDS
					       , redistribute_src, redistribute_snk
#endif
					       );
      
      std::ostringstream os; 
      os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_Figure" << diag[d] << "_sep" << params.jp.pipi_separation
	 << "_p1src" << momPrint(correlators[c].pi1_src) << "_p2src" << momPrint(correlators[c].pi2_src) 
	 << "_p1snk" << momPrint(correlators[c].pi1_snk) << "_p2snk" << momPrint(correlators[c].pi2_snk) 
	 << postpend;
    
      pipi.write(os.str());
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      pipi.write(os.str(),true);
#endif	  
      time += dclock();
      *timeCDR[d] += time;
    }
  }
  
  //Do the V diagram for all momentum combinations as it's cheap
  for(int p1idx=0; p1idx < nmom; p1idx++){
    for(int p2idx=0; p2idx < nmom; p2idx++){
      ThreeMomentum p_pi1= pion_mom.getMesonMomentum(p1idx);
      ThreeMomentum p_pi2= pion_mom.getMesonMomentum(p2idx); //pion at earlier timeslice!

      printMem(stringize("Doing pipi figure V, p1=%s p2=%s",p_pi1.str().c_str(), p_pi2.str().c_str()),0);

      double time = -dclock();
      fVector<typename A2Apolicies::ScalarComplexType> figVdis(Lt);
      ComputePiPiGparity<A2Apolicies>::computeFigureVdis(figVdis,p_pi1,p_pi2,params.jp.pipi_separation,mf_ll_con);
      std::ostringstream os; os << params.meas_arg.WorkDirectory << "/traj_" << conf << "_FigureVdis_sep" << params.jp.pipi_separation
				<< "_pi1mom" << momPrint(p_pi1) << "_pi2mom" << momPrint(p_pi2)
				<< postpend;

      figVdis.write(os.str());
#ifdef WRITE_HEX_OUTPUT
      os << ".hexfloat";
      figVdis.write(os.str(),true);
#endif	
      time += dclock();
      timeV += time;
    }
  }

  print_time("main","Pi-pi figure C",timeC);
  print_time("main","Pi-pi figure D",timeD);
  print_time("main","Pi-pi figure R",timeR);
  print_time("main","Pi-pi figure V",timeV);

  printMem("Memory after pi-pi 2pt function computation");
}



#endif
