#ifndef _MEAS_GP_H  
#define _MEAS_GP_H  

#include <algorithm>
#include <memory>
#include "pion_twopoint.h"
#include "kaon_twopoint.h"
#include "compute_bk.h"
#include "enums.h"
#include "mesonmomenta.h"
#include <alg/eigen/Krylov_5d.h>
#include <util/lattice/fbfm.h>

CPS_START_NAMESPACE

QPropWMomSrc* randomSolutionPropagator(const bool store_midprop, Lattice &latt){
  if(!UniqueID()) printf("Generating random propagator solution vector\n");
  CommonArg c_arg;
  QPropWMomSrc* ret = new QPropWMomSrc(latt,&c_arg); //this constructor does nothing
  const int msize = 12*12*2;
  ret->Allocate(PROP);
  for(int f=0;f<GJP.Gparity()+1;f++){
    for(int x=0;x<GJP.VolNodeSites();x++){
      LRG.AssignGenerator(x,f);
      Float* off = (Float*)&(ret->SiteMatrix(x,f));
      for(int i=0;i<msize;i++) off[i] = LRG.Urand(0.5,-0.5,FOUR_D);
    }
  }
  if(store_midprop){
    ret->Allocate(MIDPROP);
    for(int f=0;f<GJP.Gparity()+1;f++){
      for(int x=0;x<GJP.VolNodeSites();x++){
	LRG.AssignGenerator(x,f);
	Float* off = (Float*)&(ret->MidPlaneSiteMatrix(x,f));
	for(int i=0;i<msize;i++) off[i] = LRG.Urand(0.5,-0.5,FOUR_D);
      }
    }
  }
  return ret;
}


//Generate a flavor 'f' gauge fixed wall momentum propagator from given timeslice. Momenta are in units of pi/2L
//Eigenvectors must be those appropriate to the choice of temporal BC, 'time_bc'

//Note: If using Fbfm or FGrid, the current temporal BC listed in GJP.Tbc() must be applied to the bfm/Grid internal gauge field (i.e. minuses on t-links at boundard for APRD) prior to using this method. Internally
//it changes the bc to 'time_bc' but it changes it back at the end.
QPropWMomSrc* computeMomSourcePropagator(const double mass, const double stop_prec, const int t, const int flav, const ThreeMomentum &mom, const BndCndType time_bc, const bool store_midprop, 
					  Lattice &latt,  BFM_Krylov::Lanczos_5d<double> *deflate = NULL, const bool random_solution = false){ 
  if(random_solution) return randomSolutionPropagator(store_midprop,latt);
  int const* p = mom.ptr();
  multi1d<float> *eval_conv = NULL;

  if(deflate != NULL){
    if(latt.Fclass() != F_CLASS_BFM && latt.Fclass() != F_CLASS_BFM_TYPE2)
      ERR.General("","computePropagator","Deflation only implemented for Fbfm\n");
    if(Fbfm::use_mixed_solver){
      //Have to convert evals to single prec
      eval_conv = new multi1d<float>(deflate->bl.size());
      for(int i=0;i<eval_conv->size();i++) eval_conv->operator[](i) = deflate->bl[i];
      dynamic_cast<Fbfm&>(latt).set_deflation<float>(&deflate->bq,eval_conv,0); //last argument is really obscure - it's the number of eigenvectors subtracted from the solution to produce a high-mode inverse - we want zero here
    }else dynamic_cast<Fbfm&>(latt).set_deflation(&deflate->bq,&deflate->bl,0);
  }

  CommonArg c_arg;
  
  CgArg cg;
  cg.mass = mass;
  cg.max_num_iter = 10000;
  cg.stop_rsd = stop_prec;
  cg.true_rsd = stop_prec;
  cg.RitzMatOper = NONE;
  cg.Inverter = CG;
  cg.bicgstab_n = 0;

  QPropWArg qpropw_arg;
  qpropw_arg.cg = cg;
  qpropw_arg.x = 0;
  qpropw_arg.y = 0;
  qpropw_arg.z = 0;
  qpropw_arg.t = t;
  qpropw_arg.flavor = flav; 
  qpropw_arg.ensemble_label = "ens";
  qpropw_arg.ensemble_id = "ens_id";
  qpropw_arg.StartSrcSpin = 0;
  qpropw_arg.EndSrcSpin = 4;
  qpropw_arg.StartSrcColor = 0;
  qpropw_arg.EndSrcColor = 3;
  qpropw_arg.gauge_fix_src = 1;
  qpropw_arg.gauge_fix_snk = 0;
  qpropw_arg.store_midprop = store_midprop ? 1 : 0; //for mres

  //Switching boundary conditions is poorly implemented in Fbfm (and probably FGrid)
  //For traditional lattice types the BC is applied by modififying the gauge field when the Dirac operator is created and reverting when destroyed. This only ever happens internally - no global instance of the Dirac operator exists
  //On the other hand, Fbfm does all its inversion internally and doesn't instantiate a CPS Dirac operator. We therefore have to manually force Fbfm to change its internal gauge field by applying BondCond
  bool is_wrapper_type = ( latt.Fclass() == F_CLASS_BFM || latt.Fclass() == F_CLASS_BFM_TYPE2 ); //I hate this!

  BndCndType init_tbc = GJP.Tbc();
  BndCndType target_tbc = time_bc;

  GJP.Bc(3,target_tbc);
  if(is_wrapper_type) latt.BondCond();  //Apply new BC to internal gauge fields

  QPropWMomSrc* ret = new QPropWMomSrc(latt,&qpropw_arg,const_cast<int*>(p),&c_arg);

  //Restore the BCs
  if(is_wrapper_type) latt.BondCond();  //unapply existing BC
  GJP.Bc(3,init_tbc);
  
  if(deflate != NULL) dynamic_cast<Fbfm&>(latt).unset_deflation();
  if(eval_conv !=NULL) delete eval_conv;

  return ret;
}


PropWrapper computeMomSourcePropagator(const double mass, const double stop_prec, const int t, const ThreeMomentum &mom, const BndCndType time_bc, const bool store_midprop, 
					Lattice &latt,  BFM_Krylov::Lanczos_5d<double> *deflate = NULL, const bool random_solution = false){ 
  QPropWMomSrc* prop_f0 = computeMomSourcePropagator(mass,stop_prec,t,0,mom,time_bc,store_midprop,latt,deflate,random_solution);
  QPropWMomSrc* prop_f1 = GJP.Gparity() ? computeMomSourcePropagator(mass,stop_prec,t,1,mom,time_bc,store_midprop,latt,deflate,random_solution) : NULL;
  return PropWrapper(prop_f0, prop_f1);
}



void computeMomSourcePropagators(Props &props, const double mass, const double stop_prec, const std::vector<int> &tslices, const QuarkMomenta &quark_momenta, const BndCndType time_bc, const bool store_midprop, 
				  Lattice &latt,  BFM_Krylov::Lanczos_5d<double> *deflate = NULL, const bool random_solution = false){
  for(int tt=0;tt<tslices.size();tt++){
    const int t = tslices[tt];
    for(int pp=0;pp<quark_momenta.nMom();pp++){
      const ThreeMomentum &p = quark_momenta.getMom(pp);

      props(t,p) = computeMomSourcePropagator(mass,stop_prec,t,p,time_bc,store_midprop,latt,deflate,random_solution);

      if(GJP.Gparity()){
	//Free to add - momentum
	(props(t,-p) = props(t,p)).setFlip(true);
      }	      
    }
  }
}

//Combine quarks with P and A Tbcs into F=P+A and B=P-A types which are added to the PropMomContainer with appropriate tags
void combinePA(Props &props_F, Props &props_B, const Props &props_P, const Props &props_A){
  Props::const_iterator itP = props_P.begin();
  Props::const_iterator itA = props_A.begin();

  while(itP != props_P.end()){
    const int t = itP->first.first;
    const ThreeMomentum &p = itP->first.second;

    assert(itA->first.first == t);
    assert(itA->first.second == p);

    PropWrapper combF = PropWrapper::combinePA(itP->second,itA->second,CombinationF);
    PropWrapper combB = PropWrapper::combinePA(itP->second,itA->second,CombinationB);

    props_F(t,p) = combF;
    props_B(t,p) = combB;

    itP++; itA++;
  }
}

typedef std::auto_ptr<BFM_Krylov::Lanczos_5d<double> > LanczosPtrType;

inline std::auto_ptr<BFM_Krylov::Lanczos_5d<double> > doLanczos(GnoneFbfm &lattice, const LancArg lanc_arg, const BndCndType time_bc){
  if(lanc_arg.N_get == 0) return std::auto_ptr<BFM_Krylov::Lanczos_5d<double> >(NULL);

  BndCndType init_tbc = GJP.Tbc();
  BndCndType target_tbc = time_bc;

  GJP.Bc(3,target_tbc);
  lattice.BondCond();  //Apply BC to internal gauge fields

  bfm_evo<double> &dwf_d = static_cast<Fbfm&>(lattice).bd;
  std::auto_ptr<BFM_Krylov::Lanczos_5d<double> > ret(new BFM_Krylov::Lanczos_5d<double>(dwf_d, const_cast<LancArg&>(lanc_arg)));
  ret->Run();
  if(Fbfm::use_mixed_solver){
    //Convert eigenvectors to single precision
    ret->toSingle();
  }

  //Restore the BCs
  lattice.BondCond();  //unapply BC
  GJP.Bc(3,init_tbc);

  return ret;
}


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
