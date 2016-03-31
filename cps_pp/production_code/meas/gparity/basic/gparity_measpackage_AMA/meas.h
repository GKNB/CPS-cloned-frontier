#ifndef _MEAS_GP_H  
#define _MEAS_GP_H  

#include <algorithm>
#include <memory>
#include "pion_twopoint.h"
#include "kaon_twopoint.h"
#include "compute_bk.h"
#include "enums.h"
#include <alg/eigen/Krylov_5d.h>

CPS_START_NAMESPACE

class TbcStatus{
  BndCndType basic_bc;
  TbcCombination bc_comb;
public:
  TbcStatus(const BndCndType bbc): basic_bc(bbc), bc_comb(Single){}
  TbcStatus(const TbcCombination bcomb): bc_comb(bcomb){}
  
  inline std::string getTag() const{
    if(bc_comb == Single){
      switch(basic_bc){
      case BND_CND_PRD:
	return std::string("P");
	break;
      case BND_CND_APRD:
	return std::string("A");
	break;
      default:
	ERR.General("TbcStatus","getTag","Unknown TBC\n");
	break;
      }
    }else return std::string(bc_comb == CombinationF ? "F" : "B");
  }
  inline bool isCombinedType() const{
    return bc_comb != Single;
  }
  inline BndCndType getSingleBc() const{
    assert(!isCombinedType());
    return basic_bc;
  }

};

inline static std::string propTag(const QuarkType lh, const PropPrecision prec, const int tsrc, const ThreeMomentum &p, const TbcStatus &tbc){
  std::ostringstream tag;
  tag << (lh == Light ? "light" : "heavy");
  tag << (prec == Sloppy ? "_sloppy" : "_exact");
  tag << "_t" << tsrc;
  tag << "_mom" << p.file_str();
  tag << "_tbc_" << tbc.getTag();
  return tag.str();
}
inline static std::string propTag(const QuarkType lh, const PropPrecision prec, const int tsrc, const ThreeMomentum &p, const BndCndType tbc){
  return propTag(lh,prec,tsrc,p,TbcStatus(tbc));
}
inline static std::string propTag(const QuarkType lh, const PropPrecision prec, const int tsrc, const ThreeMomentum &p, const TbcCombination bc_comb){
  return propTag(lh,prec,tsrc,p,TbcStatus(bc_comb));
}

//Generate a flavor 'f' gauge fixed wall momentum propagator from given timeslice. Momenta are in units of pi/2L
//Eigenvectors must be those appropriate to the choice of temporal BC, 'time_bc'
QPropWMomSrc* computePropagator(const double mass, const double stop_prec, const int t, const int flav, const int p[3], const BndCndType time_bc, Lattice &latt, BFM_Krylov::Lanczos_5d<double> *deflate = NULL){ 
  if(deflate != NULL)
    dynamic_cast<Fbfm&>(latt).set_deflation(&deflate->bq,&deflate->bl,0); //last argument is really obscure - it's the number of eigenvectors subtracted from the solution to produce a high-mode inverse - we want zero here

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
  qpropw_arg.store_midprop = 1; //for mres

  //Switching boundary conditions is poorly implemented in Fbfm (and probably FGrid)
  //For traditional lattice types the BC is applied by modififying the gauge field when the Dirac operator is created and reverting when destroyed. This only ever happens internally - no global instance of the Dirac operator exists
  //On the other hand, Fbfm does all its inversion internally and doesn't instantiate a CPS Dirac operator. We therefore have to manually force Fbfm to change its internal gauge field by applying BondCond
  bool is_wrapper_type = ( latt.Fclass() == F_CLASS_BFM || latt.Fclass() == F_CLASS_BFM_TYPE2 ); //I hate this!

  BndCndType init_tbc = GJP.Tbc();
  BndCndType target_tbc = time_bc;
  bool change_bc = (init_tbc != target_tbc);

  if(change_bc){
    if(is_wrapper_type) latt.BondCond();  //CPS Lattice currently has the BC applied. We first un-apply it before changing things
    GJP.Bc(3,target_tbc);
    if(is_wrapper_type) latt.BondCond();  //Apply new BC to internal gauge fields
  }
  QPropW* ret = new QPropWMomSrc(latt,&qpropw_arg,p,&c_arg);

  if(change_bc){
    //Restore the BCs
    if(is_wrapper_type) latt.BondCond();  //unapply existing BC
    GJP.Bc(3,init_tbc);
    if(is_wrapper_type) latt.BondCond();  //Reapply original BC to internal gauge fields
  }

  return ret;
}

QPropWMomSrc* computePropagator(const double mass, const double stop_prec, const int t, const int flav, const ThreeMomentum &p, const BndCndType time_bc, Lattice &latt, BFM_Krylov::Lanczos_5d<double> *deflate = NULL){ 
  return computePropagator(mass,stop_prec,t,flav,p.ptr(),time_bc,latt,deflate);
}

void quarkInvert(PropMomContainer &props, const QuarkType qtype, const PropPrecision pp, const double stop_prec, const double mass, const BndCndType time_bc,
		 const std::vector<int> &tslices, const QuarkMomenta &quark_momenta,
		 Lattice &lattice, BFM_Krylov::Lanczos_5d<double> &lanc){
  if(!UniqueID()) printf("Computing %s %s quark propagators\n", pp == Sloppy ? "sloppy":"exact", qtype==Light ? "light" : "heavy");
  double time = -dclock();
  
  for(int s=0;s<tslices.size();s++){
    const int tsrc = tslices[s];
    
    for(int pidx=0;pidx<quark_momenta.nMom();pidx++){
      const ThreeMomentum &p = quark_momenta.getMom(pidx);
      if(!UniqueID()) std::cout << "Starting inversion for prop on timeslice " << tsrc << " with momentum phase " << p.str() << '\n';  

      if(GJP.Gparity()){
	QPropWMomSrc* prop_f0 = computePropagator(mass,stop_prec,tsrc,0,p.ptr(),time_bc,lattice,&lanc);
	QPropWMomSrc* prop_f1 = computePropagator(mass,stop_prec,tsrc,1,p.ptr(),time_bc,lattice,&lanc);
	
	//Add both + and - source momentum  (PropMomContainer manages prop memory)
	PropWrapper prop_pplus(prop_f0,prop_f1,false);
	props.insert(prop_pplus, propTag(qtype,pp,tsrc,p,time_bc));
	
	PropWrapper prop_pminus(prop_f0,prop_f1,true);
	props.insert(prop_pminus, propTag(qtype,pp,tsrc,-p,time_bc));
      }else{
	QPropWMomSrc* prop = computePropagator(mass,stop_prec,tsrc,0,p.ptr(),time_bc,lattice,&lanc);
	PropWrapper propw(prop);
	props.insert(propw, propTag(qtype,pp,tsrc,p,time_bc));
      }
    }
  }
  print_time("main","Inversions",time + dclock());
}

//Combine quarks with P and A Tbcs into F=P+A and B=P-A types which are added to the PropMomContainer with appropriate tags
static void quarkCombine(PropMomContainer &props, const QuarkType qtype, const PropPrecision pp, const std::vector<int> &tslices, const QuarkMomenta &quark_momenta){
  if(!UniqueID()) printf("Combining %s %s quark propagators with different Tbcs\n", pp == Sloppy ? "sloppy":"exact", qtype==Light ? "light" : "heavy");
  double time = -dclock();
  
  for(int s=0;s<tslices.size();s++){
    const int tsrc = tslices[s];
    
    for(int pidx=0;pidx<quark_momenta.nMom();pidx++){
      const ThreeMomentum &p = quark_momenta.getMom(pidx);
      if(!UniqueID()) std::cout << "Starting combination of props on timeslice " << tsrc << " with momentum phase " << p.str() << '\n';  

      PropWrapper prop_P = props.get(propTag(qtype,pp,tsrc,p,BND_CND_PRD));
      PropWrapper prop_A = props.get(propTag(qtype,pp,tsrc,p,BND_CND_APRD));

      PropWrapper combF = PropWrapper::combinePA(prop_P,prop_A,CombinationF);
      props.insert(combF, propTag(qtype,pp,tsrc,p,CombinationF));

      PropWrapper combB = PropWrapper::combinePA(prop_P,prop_A,CombinationB);
      props.insert(combB, propTag(qtype,pp,tsrc,p,CombinationB));

      if(GJP.Gparity()){
	//We can just change the flip flag and the momentum for the -ve mom counterparts without using up extra memory
	combF.setFlip(true);
	props.insert(combF, propTag(qtype,pp,tsrc,-p,CombinationF));
	
	combB.setFlip(true);
	props.insert(combB, propTag(qtype,pp,tsrc,-p,CombinationB));
      }
    }
  }
  print_time("main","Combinations",time + dclock());
}

inline std::auto_ptr<BFM_Krylov::Lanczos_5d<double> > doLanczos(GnoneFbfm &lattice, const LancArg lanc_arg, const BndCndType time_bc){
  BndCndType init_tbc = GJP.Tbc();
  BndCndType target_tbc = time_bc;
  bool change_bc = (init_tbc != target_tbc);

  if(change_bc){
    lattice.BondCond();
    GJP.Bc(3,target_tbc);
    lattice.BondCond();  //Apply new BC to internal gauge fields
  }

  bfm_evo<double> &dwf_d = static_cast<Fbfm&>(lattice).bd;
  std::auto_ptr<BFM_Krylov::Lanczos_5d<double> > ret(new BFM_Krylov::Lanczos_5d<double>(dwf_d, const_cast<LancArg&>(lanc_arg)));
  ret->Run();
  if(Fbfm::use_mixed_solver){
    //Convert eigenvectors to single precision
    ret->toSingle();
  }

  if(change_bc){
    //Restore the BCs
    lattice.BondCond();  //unapply existing BC
    GJP.Bc(3,init_tbc);
    lattice.BondCond();  //Reapply original BC to internal gauge fields
  }
  return ret;
}




void writeBasic2ptLW(fMatrix<double> &results, const std::string &results_dir, const std::string &descr, const ThreeMomentum &p1, const ThreeMomentum &p2, 
		     const PropPrecision status, const TbcStatus &time_bc, const int conf, const std::string &extra_descr = ""){
  std::ostringstream os; 
  os << results_dir << '/' << descr << "_mom" << (-p1).file_str() << "_plus" << p2.file_str() << (status == Sloppy ? "_sloppy" : "_exact") << "_tbc" << time_bc.getTag() << time_bc.getTag() << extra_descr << '.' << conf;
  results.write(os.str());
}


void writePion2ptLW(fMatrix<double> &results, const std::string &results_dir, const std::string &snk_op, const ThreeMomentum &p1, const ThreeMomentum &p2, 
		    const PropPrecision status, const TbcStatus &time_bc, const int conf, const std::string &extra_descr){
  std::ostringstream os; 
  os << results_dir << "/pion_" << snk_op << "_P_LW_mom" << (-p1).file_str() << "_plus" << p2.file_str() << (status == Sloppy ? "_sloppy" : "_exact") << "_tbc" << time_bc.getTag() << time_bc.getTag() << extra_descr << '.' << conf;
  results.write(os.str());
}

//Pion 2pt LW functions pseudoscalar and axial sinks
void measurePion2ptLWStandard(const PropMomContainer &props, const PropPrecision status, const TbcStatus &time_bc, const std::vector<int> &tslices, const MesonMomenta &ll_meson_momenta,
		      const std::string &results_dir, const int conf){
  //Loop over light-light meson momenta
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();

  for(int pidx=0;pidx<ll_meson_momenta.nMom();pidx++){
    ThreeMomentum p1 = ll_meson_momenta.getQuarkMom(0,pidx); //note the total meson momentum is p2 - p1 because the Hermitian conjugate of the first propagator swaps the momentum
    ThreeMomentum p2 = ll_meson_momenta.getQuarkMom(1,pidx);
	  
    Pion2PtSinkOp sink_ops[5] = { AX, AY, AZ, AT, P };//Generate a flavor 'f' gauge fixed wall momentum propagator from given timeslice. Momenta are in units of pi/2L
    std::string sink_op_stub[5] = { "AX", "AY", "AZ", "AT", "P" };
	
    for(int op=0;op<5;op++){

      fMatrix<double> results(Lt,Lt); //[tsrc][tsnk-tsrc]
      for(int s=0;s<tslices.size();s++){
	const int tsrc = tslices[s];
	    
	PropWrapper &prop1 = props.get(propTag(Light,status,tsrc,p1,time_bc));
	PropWrapper &prop2 = props.get(propTag(Light,status,tsrc,p2,time_bc));

	pionTwoPointLWStandard(results,tsrc,sink_ops[op],p1,p2,prop1,prop2);
      }
      writePion2ptLW(results, results_dir, sink_op_stub[op], p1, p2, status, time_bc, conf, "");
    }
  }
}

void measurePion2ptLWGparity(const PropMomContainer &props, const PropPrecision status, const TbcStatus &time_bc, const std::vector<int> &tslices, const MesonMomenta &ll_meson_momenta,
		      const std::string &results_dir, const int conf){
  //Loop over light-light meson momenta
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();

  for(int pidx=0;pidx<ll_meson_momenta.nMom();pidx++){
    ThreeMomentum p1 = ll_meson_momenta.getQuarkMom(0,pidx); //note the total meson momentum is p2 - p1 because the Hermitian conjugate of the first propagator swaps the momentum
    ThreeMomentum p2 = ll_meson_momenta.getQuarkMom(1,pidx);
	  
    Pion2PtSinkOp sink_ops[5] = { AX, AY, AZ, AT, P };//Generate a flavor 'f' gauge fixed wall momentum propagator from given timeslice. Momenta are in units of pi/2L
    std::string sink_op_stub[5] = { "AX", "AY", "AZ", "AT", "P" };
	
    for(int op=0;op<5;op++){

      fMatrix<double> results(Lt,Lt); //[tsrc][tsnk-tsrc]
      fMatrix<double> results_wrongsinkmom(Lt,Lt); //[tsrc][tsnk-tsrc]  wrong sink momentum (should give zero within statistics)
      fMatrix<double> results_wrongproj(Lt,Lt); //[tsrc][tsnk-tsrc]  opposite projection op (optional, used for paper)
      fMatrix<double> results_wrongproj_wrongsinkmom(Lt,Lt); //wrong sink mom and wrong projector - non-zero within statistics as discussed in paper

      for(int s=0;s<tslices.size();s++){
	const int tsrc = tslices[s];
	    
	PropWrapper &prop1 = props.get(propTag(Light,status,tsrc,p1,time_bc));
	PropWrapper &prop2 = props.get(propTag(Light,status,tsrc,p2,time_bc));

	pionTwoPointLWGparity(results,tsrc,sink_ops[op],p1,p2,prop1,prop2,SPLANE_BOUNDARY,false,false); //right proj op, right sink mom

	if(op == 4){ //only pseudoscalar sink op
	  pionTwoPointLWGparity(results_wrongsinkmom,tsrc,sink_ops[op],p1,p2,prop1,prop2,SPLANE_BOUNDARY,false,true); //right proj op, right sink mom
	  
	  pionTwoPointLWGparity(results_wrongproj,tsrc,sink_ops[op],p1,p2,prop1,prop2,SPLANE_BOUNDARY,true,false); //wrong proj op, right sink mom
	  pionTwoPointLWGparity(results_wrongproj_wrongsinkmom,tsrc,sink_ops[op],p1,p2,prop1,prop2,SPLANE_BOUNDARY,true,true); //wrong proj op, wrong sink mom
	}
      }
      writePion2ptLW(results, results_dir, sink_op_stub[op], p1, p2, status, time_bc, conf, "");
      if(op == 4){
	writePion2ptLW(results_wrongsinkmom, results_dir, sink_op_stub[op], p1, p2, status, time_bc, conf, "_wrongsinkmom");
	writePion2ptLW(results_wrongproj, results_dir, sink_op_stub[op], p1, p2, status, time_bc, conf, "_wrongproj");
	writePion2ptLW(results_wrongproj_wrongsinkmom, results_dir, sink_op_stub[op], p1, p2, status, time_bc, conf, "_wrongproj_wrongsinkmom");
      }
    }

    //Also do A4 A4 
    fMatrix<double> results(Lt,Lt);
    for(int s=0;s<tslices.size();s++){
      const int tsrc = tslices[s];
      
      PropWrapper &prop1 = props.get(propTag(Light,status,tsrc,p1,time_bc));
      PropWrapper &prop2 = props.get(propTag(Light,status,tsrc,p2,time_bc));
      pionTwoPointA4A4LWGparity(results,tsrc,p1,p2,prop1,prop2);
    }
    writeBasic2ptLW(results,results_dir,"pion_AT_AT_LW",p1,p2,status,time_bc,conf);
  }
}

void measurePion2ptLW(const PropMomContainer &props, const PropPrecision status, const TbcStatus &time_bc, const std::vector<int> &tslices, const MesonMomenta &ll_meson_momenta,
		      const std::string &results_dir, const int conf){
  if(!UniqueID()) printf("Computing pion 2pt LW with %s props\n", status == Sloppy ? "sloppy" : "exact");
  double time = -dclock();

  if(GJP.Gparity()) measurePion2ptLWGparity(props,status,time_bc, tslices,ll_meson_momenta,results_dir, conf);
  else measurePion2ptLWStandard(props,status,time_bc,tslices,ll_meson_momenta,results_dir, conf);

  print_time("main","Pion 2pt LW",time + dclock());
}



//Pion 2pt LW functions pseudoscalar sink
void measurePion2ptPPWW(const PropMomContainer &props, const PropPrecision status, const TbcStatus &time_bc, const std::vector<int> &tslices, const MesonMomenta &ll_meson_momenta, Lattice &lat,
			const std::string results_dir, const int conf){
  if(!UniqueID()) printf("Computing pion 2pt WW with %s props\n", status == Sloppy ? "sloppy" : "exact");
  double time = -dclock();

  const int Lt = GJP.Tnodes()*GJP.TnodeSites();

  //Loop over light-light meson momenta
  for(int pidx=0;pidx<ll_meson_momenta.nMom();pidx++){
    ThreeMomentum p1 = ll_meson_momenta.getQuarkMom(0,pidx); //note the total meson momentum is p2 - p1 because the Hermitian conjugate of the first propagator swaps the momentum
    ThreeMomentum p2 = ll_meson_momenta.getQuarkMom(1,pidx);
	  
    //Consider two scenarios with the same total sink momentum
    //1) The quarks each have the same sink momenta as they do at the source
    //2) The quarks have their sink momenta exchanged

    ThreeMomentum p1_snk_keep = -p1; //exp(+ip.x)
    ThreeMomentum p2_snk_keep = -p2;

    ThreeMomentum p1_snk_exch = -p2; //exp(+ip.x)
    ThreeMomentum p2_snk_exch = -p1;
    
    fMatrix<double> results_momkeep(Lt,Lt); //[tsrc][tsnk-tsrc]
    fMatrix<double> results_momexch(Lt,Lt);

    for(int s=0;s<tslices.size();s++){
      const int tsrc = tslices[s];
      
      //Prop1
      PropWrapper &prop1 = props.get(propTag(Light,status,tsrc,p1,time_bc));
     
      WallSinkProp<SpinColorFlavorMatrix> prop1_FT_keep; 
      prop1_FT_keep.setProp(prop1);
      prop1_FT_keep.compute(lat, p1_snk_keep.ptr());

      WallSinkProp<SpinColorFlavorMatrix> prop1_FT_exch; 
      prop1_FT_exch.setProp(prop1);
      prop1_FT_exch.compute(lat, p1_snk_exch.ptr());      

      //Prop2
      PropWrapper &prop2 = props.get(propTag(Light,status,tsrc,p2,time_bc));

      WallSinkProp<SpinColorFlavorMatrix> prop2_FT_keep; 
      prop2_FT_keep.setProp(prop2);
      prop2_FT_keep.compute(lat, p2_snk_keep.ptr());

      WallSinkProp<SpinColorFlavorMatrix> prop2_FT_exch; 
      prop2_FT_exch.setProp(prop2);
      prop2_FT_exch.compute(lat, p2_snk_exch.ptr());   
 
      pionTwoPointPPWWGparity(results_momkeep, tsrc, p1, p2_snk_keep, prop1_FT_keep, prop2_FT_keep);
      pionTwoPointPPWWGparity(results_momexch, tsrc, p1, p2_snk_exch, prop1_FT_exch, prop2_FT_exch);
    }
    writeBasic2ptLW(results_momkeep,results_dir,"pion_P_P_WW_momkeep",p1,p2,status,time_bc,conf);
    writeBasic2ptLW(results_momexch,results_dir,"pion_P_P_WW_momexch",p1,p2,status,time_bc,conf);
  }
  print_time("main","Pion 2pt WW",time + dclock());
}


//SU(2) singlet 2pt LW functions
void measureLightFlavorSingletLW(const PropMomContainer &props, const PropPrecision status, const TbcStatus &time_bc, const std::vector<int> &tslices, const MesonMomenta &meson_momenta,
				 const std::string results_dir, const int conf){

  const int Lt = GJP.Tnodes()*GJP.TnodeSites();

  //Loop over light-light meson momenta
  for(int pidx=0;pidx<meson_momenta.nMom();pidx++){
    ThreeMomentum p1 = meson_momenta.getQuarkMom(0,pidx); //note the total meson momentum is p2 - p1 because the Hermitian conjugate of the first propagator swaps the momentum
    ThreeMomentum p2 = meson_momenta.getQuarkMom(1,pidx);
	  
    fMatrix<double> results(Lt,Lt); //[tsrc][tsnk-tsrc]
    for(int s=0;s<tslices.size();s++){
      const int tsrc = tslices[s];
	    
      PropWrapper &prop1 = props.get(propTag(Light,status,tsrc,p1,time_bc));
      PropWrapper &prop2 = props.get(propTag(Light,status,tsrc,p2,time_bc));

      lightFlavorSingletLWGparity(results,tsrc,p1,p2,prop1,prop2);
    }
    writeBasic2ptLW(results,results_dir,"light_pseudo_flav_singlet_LW",p1,p2,status,time_bc,conf);
  }
}


void measureKaon2ptLW(const PropMomContainer &props, const PropPrecision status, const TbcStatus &time_bc, const std::vector<int> &tslices, const MesonMomenta &meson_momenta,
		      const std::string results_dir, const int conf){

  const int Lt = GJP.Tnodes()*GJP.TnodeSites();

  for(int pidx=0;pidx<meson_momenta.nMom();pidx++){
    assert(meson_momenta.getQuarkType(0,pidx) == Heavy);   //The strange quark should be on index 0 of the meson_momenta
    assert(meson_momenta.getQuarkType(1,pidx) == Light);

    ThreeMomentum p_h = meson_momenta.getQuarkMom(0,pidx); //note the total meson momentum is p2 - p1 because the Hermitian conjugate of the first propagator swaps the momentum
    ThreeMomentum p_l = meson_momenta.getQuarkMom(1,pidx);
	  
    fMatrix<double> results_PP(Lt,Lt); //[tsrc][tsnk-tsrc]
    fMatrix<double> results_A4physP(Lt,Lt);
    fMatrix<double> results_A4unphysP(Lt,Lt);
    fMatrix<double> results_A4combA4comb(Lt,Lt);

    for(int s=0;s<tslices.size();s++){
      const int tsrc = tslices[s];
	    
      PropWrapper &prop_h = props.get(propTag(Heavy,status,tsrc,p_h,time_bc));
      PropWrapper &prop_l = props.get(propTag(Light,status,tsrc,p_l,time_bc));

      kaonTwoPointPPLWGparity(results_PP, tsrc, p_h, p_l, prop_h, prop_l);
      kaonTwoPointA4PhysPLWGparity(results_A4physP, tsrc, p_h, p_l, prop_h, prop_l);
      kaonTwoPointA4UnphysPLWGparity(results_A4unphysP, tsrc, p_h, p_l, prop_h, prop_l);
      kaonTwoPointA4combA4combLWGparity(results_A4combA4comb, tsrc, p_h, p_l, prop_h, prop_l);
    }
    writeBasic2ptLW(results_PP,results_dir,"kaon_P_P_LW",p_h,p_l,status,time_bc,conf);
    writeBasic2ptLW(results_A4physP,results_dir,"kaon_ATphys_P_LW",p_h,p_l,status,time_bc,conf);
    writeBasic2ptLW(results_A4unphysP,results_dir,"kaon_ATunphys_P_LW",p_h,p_l,status,time_bc,conf);
    writeBasic2ptLW(results_A4combA4comb,results_dir,"kaon_ATcomb_ATcomb_LW",p_h,p_l,status,time_bc,conf);
  }
}


//Kaon 2pt LW functions pseudoscalar sink
void measureKaon2ptPPWW(const PropMomContainer &props, const PropPrecision status, const TbcStatus &time_bc, const std::vector<int> &tslices, const MesonMomenta &meson_momenta, Lattice &lat,
			const std::string results_dir, const int conf){
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();

  //Loop over light-light meson momenta
  for(int pidx=0;pidx<meson_momenta.nMom();pidx++){
    assert(meson_momenta.getQuarkType(0,pidx) == Heavy);   //The strange quark should be on index 0 of the meson_momenta
    assert(meson_momenta.getQuarkType(1,pidx) == Light);

    ThreeMomentum ph = meson_momenta.getQuarkMom(0,pidx); //note the total meson momentum is pl - ph because the Hermitian conjugate of the first propagator swaps the momentum
    ThreeMomentum pl = meson_momenta.getQuarkMom(1,pidx);
	  
    //Consider two scenarios with the same total sink momentum
    //1) The quarks each have the same sink momenta as they do at the source
    //2) The quarks have their sink momenta exchanged

    ThreeMomentum ph_snk_keep = -ph; //exp(+ip.x)
    ThreeMomentum pl_snk_keep = -pl;

    ThreeMomentum ph_snk_exch = -pl; //exp(+ip.x)
    ThreeMomentum pl_snk_exch = -ph;
    
    fMatrix<double> results_momkeep(Lt,Lt); //[tsrc][tsnk-tsrc]
    fMatrix<double> results_momexch(Lt,Lt);

    for(int s=0;s<tslices.size();s++){
      const int tsrc = tslices[s];
      
      //Strange prop
      PropWrapper &proph = props.get(propTag(Heavy,status,tsrc,ph,time_bc));
     
      WallSinkProp<SpinColorFlavorMatrix> proph_FT_keep; 
      proph_FT_keep.setProp(proph);
      proph_FT_keep.compute(lat, ph_snk_keep.ptr());

      WallSinkProp<SpinColorFlavorMatrix> proph_FT_exch; 
      proph_FT_exch.setProp(proph);
      proph_FT_exch.compute(lat, ph_snk_exch.ptr());      

      //Light prop
      PropWrapper &propl = props.get(propTag(Light,status,tsrc,pl,time_bc));

      WallSinkProp<SpinColorFlavorMatrix> propl_FT_keep; 
      propl_FT_keep.setProp(propl);
      propl_FT_keep.compute(lat, pl_snk_keep.ptr());

      WallSinkProp<SpinColorFlavorMatrix> propl_FT_exch; 
      propl_FT_exch.setProp(propl);
      propl_FT_exch.compute(lat, pl_snk_exch.ptr());   
 
      kaonTwoPointPPWWGparity(results_momkeep, tsrc, ph, pl_snk_keep, proph_FT_keep, propl_FT_keep);
      kaonTwoPointPPWWGparity(results_momexch, tsrc, ph, pl_snk_exch, proph_FT_exch, propl_FT_exch);
    }
    writeBasic2ptLW(results_momkeep,results_dir,"kaon_P_P_WW_momkeep",ph,pl,status,time_bc,conf);
    writeBasic2ptLW(results_momexch,results_dir,"kaon_P_P_WW_momexch",ph,pl,status,time_bc,conf);
  }
}



//Measure BK with source kaons on each of the timeslices t0 in prop_tsources and K->K time separations tseps
//Can use standard P or A time BCs but you will need to use closer-together kaon sources to avoid round-the-world effects. These can be eliminated by using the F=P+A and B=P-A combinations
//Either can be specified using the appropriate time_bc parameter below
void measureBK(const PropMomContainer &props, const PropPrecision status, const std::vector<int> &prop_tsources, const std::vector<int> &tseps, const MesonMomenta &meson_momenta,
	       const TbcStatus &time_bc_t0, const TbcStatus &time_bc_t1,
	       const std::string results_dir, const int conf, const bool do_flavor_project = true){
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();
  
  //Do all combinations of source and sink kaon momenta that have the same total momentum. This allows us to look at alternate quark momentum combinations
  //In the meson_momenta, prop index 0 is the strange quark (as in the standard 2pt function case), and is the propagator that is daggered. Prop index 1 is the light quark.
  for(int p0idx=0;p0idx<meson_momenta.nMom();p0idx++){
    assert(meson_momenta.getQuarkType(0,p0idx) == Heavy);   //The strange quark should be on index 0 of the meson_momenta
    assert(meson_momenta.getQuarkType(1,p0idx) == Light);

    ThreeMomentum prop_h_t0_srcmom = meson_momenta.getQuarkMom(0,p0idx);
    ThreeMomentum prop_l_t0_srcmom = meson_momenta.getQuarkMom(1,p0idx);

    for(int p1idx=p0idx;p1idx<meson_momenta.nMom();p1idx++){
      if(meson_momenta.getMesonMom(p1idx) != meson_momenta.getMesonMom(p0idx)) continue;

      ThreeMomentum prop_h_t1_srcmom = meson_momenta.getQuarkMom(0,p1idx);
      ThreeMomentum prop_l_t1_srcmom = meson_momenta.getQuarkMom(1,p1idx);
      
      for(int tspi=0;tspi<tseps.size();tspi++){
	fMatrix<double> results(Lt,Lt);

	for(int t0i=0;t0i<prop_tsources.size();t0i++){
	  int t0 = prop_tsources[t0i];
	  int t1 = (t0 + tseps[tspi]) % Lt;
	  
	  //check t1 is in the vector, if not skip
	  if(std::find(prop_tsources.begin(), prop_tsources.end(), t1) == prop_tsources.end()) continue;

	  PropWrapper &prop_h_t0 = props.get(propTag(Heavy,status,t0,prop_h_t0_srcmom,time_bc_t0));
	  PropWrapper &prop_l_t0 = props.get(propTag(Light,status,t0,prop_l_t0_srcmom,time_bc_t0));

	  PropWrapper &prop_h_t1 = props.get(propTag(Heavy,status,t1,prop_h_t1_srcmom,time_bc_t1));
	  PropWrapper &prop_l_t1 = props.get(propTag(Light,status,t1,prop_l_t1_srcmom,time_bc_t1));

	  GparityBK(results, t0, 
		    prop_h_t0, prop_l_t0, prop_h_t0_srcmom,
		    prop_h_t1, prop_l_t1, prop_h_t1_srcmom,
		    do_flavor_project);
	}
	{
	  std::ostringstream os;
	  os << results_dir << "/BK_srcMom" << (-prop_h_t0_srcmom).file_str() << "_plus" << prop_l_t0_srcmom.file_str() 
	     << "snkMom" << (-prop_h_t1_srcmom).file_str() << "_plus" << prop_l_t1_srcmom.file_str()
	     << "_srcTbc" << time_bc_t0.getTag()
	     << "_snkTbc" << time_bc_t1.getTag()
	     << "_tsep" << tseps[tspi]
	     << (status == Sloppy ? "_sloppy" : "_exact") 	    
	     << '.' << conf;
	  results.write(os.str());
	}
      }
    }
  }
}


//Note: Mres is only properly defined with APRD time BCs. A runtime check is performed
void measureMres(const PropMomContainer &props, const PropPrecision status, const TbcStatus &time_bc, const std::vector<int> &tslices, const MesonMomenta &meson_momenta,
		 const std::string results_dir, const int conf, const bool do_flavor_project = true){

  const int Lt = GJP.Tnodes()*GJP.TnodeSites();

  for(int pidx=0;pidx<meson_momenta.nMom();pidx++){
    ThreeMomentum p1 = meson_momenta.getQuarkMom(0,pidx); //note the total meson momentum is p2 - p1 because the Hermitian conjugate of the first propagator swaps the momentum
    ThreeMomentum p2 = meson_momenta.getQuarkMom(1,pidx);
	  
    fMatrix<double> pion(Lt,Lt); //[tsrc][tsnk-tsrc]
    fMatrix<double> j5q(Lt,Lt);
    
    for(int s=0;s<tslices.size();s++){
      const int tsrc = tslices[s];
	    
      PropWrapper &prop1 = props.get(propTag(Light,status,tsrc,p1,time_bc));
      PropWrapper &prop2 = props.get(propTag(Light,status,tsrc,p2,time_bc));

      J5Gparity(pion,tsrc,p1,p2,prop1,prop2,SPLANE_BOUNDARY,do_flavor_project); 
      J5Gparity(j5q,tsrc,p1,p2,prop1,prop2,SPLANE_MIDPOINT,do_flavor_project);
    }
    writeBasic2ptLW(pion,results_dir,"J5_LW",p1,p2,status,time_bc,conf);
    writeBasic2ptLW(j5q,results_dir,"J5q_LW",p1,p2,status,time_bc,conf);
  }
}





CPS_END_NAMESPACE


#endif
