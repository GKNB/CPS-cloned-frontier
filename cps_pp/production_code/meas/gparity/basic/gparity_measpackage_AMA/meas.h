#ifndef _MEAS_GP_H  
#define _MEAS_GP_H  

#include <algorithm>
#include <memory>
#include "pion_twopoint.h"
#include "kaon_twopoint.h"
#include "compute_bk.h"
#include "enums.h"
#include "mesonmomenta.h"
#include "propagators.h"

CPS_START_NAMESPACE

void writeBasic2ptLW(fMatrix<Rcomplex> &results, const std::string &results_dir, const std::string &corr_descr, const ThreeMomentum &p_psibar, const ThreeMomentum &p_psi, 
		     const int conf, const std::string &extra_descr = ""){
  std::ostringstream os; 
  os << results_dir << '/' << corr_descr << "_mom" << p_psibar.file_str() << "_plus" << p_psi.file_str() << extra_descr << '.' << conf;
  results.write(os.str());
}


void writePion2ptLW(fMatrix<Rcomplex> &results, const std::string &results_dir, const std::string &snk_op, const ThreeMomentum &p_psibar, const ThreeMomentum &p_psi, 
		    const int conf, const std::string &extra_descr){
  std::ostringstream os; 
  os << results_dir << "/pion_" << snk_op << "_P_LW_mom" << p_psibar.file_str() << "_plus" << p_psi.file_str() << extra_descr << '.' << conf;
  results.write(os.str());
}

#include "meas_standard.tcc"
#include "meas_gparity.tcc"


void measurePion2ptLW(const PropGetter &props, const std::vector<int> &tslices, const MesonMomenta &ll_meson_momenta,
		      const std::string &results_dir, const int conf, const std::string &extra_descr){
  if(!UniqueID()) printf("Computing pion 2pt LW with description %s\n", extra_descr.c_str());
  double time = -dclock();

  if(GJP.Gparity()) measurePion2ptLWGparity(props, tslices,ll_meson_momenta,results_dir, conf, extra_descr);
  else measurePion2ptLWStandard(props,tslices,ll_meson_momenta,results_dir, conf, extra_descr);

  print_time("main","Pion 2pt LW",time + dclock());
}

void measureKaon2ptLW(const PropGetter &props_l, const PropGetter &props_h, const std::vector<int> &tslices, const MesonMomenta &meson_momenta,
		      const std::string results_dir, const int conf, const std::string &extra_descr){
  if(!UniqueID()) printf("Computing kaon 2pt LW with description %s\n", extra_descr.c_str());
  double time = -dclock();

  if(GJP.Gparity()) measureKaon2ptLWGparity(props_l,props_h,tslices,meson_momenta,results_dir,conf,extra_descr);
  else measureKaon2ptLWStandard(props_l,props_h,tslices,meson_momenta,results_dir,conf,extra_descr);

  print_time("main","Kaon 2pt LW",time + dclock());
}

//Measure BK with source kaons on each of the timeslices t0 in prop_tsources and K->K time separations tseps
//Can use standard P or A time BCs but you will need to use closer-together kaon sources to avoid round-the-world effects. These can be eliminated by using the F=P+A and B=P-A combinations
//Either can be specified using the appropriate time_bc parameter below
//For G-parity can optionally choose to disable the source/sink flavor projection (ignored for standard BCs)
void measureBK(const PropGetter &props_l, const PropGetter &props_h, const std::vector<int> &prop_tsources, const std::vector<int> &tseps, const MesonMomenta &meson_momenta,
	       const std::string &results_dir, const int conf, const std::string &extra_descr, const bool do_flavor_project = true){
  if(!UniqueID()) printf("Computing BK with description %s\n", extra_descr.c_str() );
  double time = -dclock();

  if(GJP.Gparity()) measureBKGparity(props_l,props_h,prop_tsources,tseps,meson_momenta,results_dir,conf,extra_descr,do_flavor_project);
  else measureBKStandard(props_l,props_h,prop_tsources,tseps,meson_momenta,results_dir,conf,extra_descr);

  print_time("main","BK",time + dclock());
}


//Note: Mres is only properly defined with APRD time BCs. A runtime check is *not* performed
//For G-parity can optionally choose to disable the source/sink flavor projection (ignored for standard BCs)
void measureMres(const PropGetter &props, const std::vector<int> &tslices, const MesonMomenta &meson_momenta,
		 const std::string &results_dir, const int conf, const std::string &extra_descr, const bool do_flavor_project = true){
  if(!UniqueID()) printf("Computing J5 and J5q (mres) with description %s\n", extra_descr.c_str());
  double time = -dclock();

  if(GJP.Gparity()) measureMresGparity(props,tslices,meson_momenta,results_dir,conf,extra_descr,do_flavor_project);
  else measureMresStandard(props,tslices,meson_momenta,results_dir,conf,extra_descr);

  print_time("main","J5 and J5q",time + dclock());
}








CPS_END_NAMESPACE


#endif
