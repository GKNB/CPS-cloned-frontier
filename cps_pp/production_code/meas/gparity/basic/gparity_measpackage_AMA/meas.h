#ifndef _MEAS_GP_H  
#define _MEAS_GP_H  

#include "pion_twopoint.h"
#include "kaon_twopoint.h"
#include "compute_bk.h"
#include <alg/eigen/Krylov_5d.h>

CPS_START_NAMESPACE

enum PropPrecision { Sloppy, Exact };

static std::string propTag(const QuarkType lh, const PropPrecision prec, const int tsrc, const ThreeMomentum &p){
  std::ostringstream tag;
  tag << (lh == Light ? "light" : "heavy");
  tag << (prec == Sloppy ? "_sloppy" : "_exact");
  tag << '_t' << tsrc;
  tag << '_' << p.file_str();
  return tag.str();
}

//Generate a flavor 'f' gauge fixed wall momentum propagator from given timeslice. Momenta are in units of pi/2L
QPropWMomSrc* computePropagator(const double mass, const double stop_prec, const int t, const int flav, const int p[3], Lattice &latt, BFM_Krylov::Lanczos_5d<double> *deflate = NULL){ 
  if(deflate != NULL)
    dynamic_cast<Fbfm&>(latt).set_deflation(&deflate->bq,&deflate->bl,deflate->get);

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

  return new QPropWMomSrc(latt,&qpropw_arg,p,&c_arg);
}

QPropWMomSrc* computePropagator(const double mass, const double stop_prec, const int t, const int flav, const ThreeMomentum &p, Lattice &latt, BFM_Krylov::Lanczos_5d<double> *deflate = NULL){ 
  return computePropagator(mass,stop_prec,t,flav,p.ptr(),latt,deflate);
}


//Light-quark inversions
void quarkInvert(PropMomContainer &props, const QuarkType qtype, const PropPrecision pp, const double prec, const double mass,
		 const std::vector<int> &tslices, const QuarkMomenta &quark_momenta,
		 Lattice &lattice, BFM_Krylov::Lanczos_5d<double> &lanc){
  for(int s=0;s<tslices.size();s++){
    const int tsrc = tslices[s];
    
    for(int pidx=0;pidx<quark_momenta.nMom();pidx++){
      const ThreeMomentum &p = quark_momenta.getMom(pidx);
	  
      if(GJP.Gparity()){
	QPropWMomSrc* prop_f0 = computePropagator(mass,prec,tsrc,0,p.ptr(),lattice,&lanc);
	QPropWMomSrc* prop_f1 = computePropagator(mass,prec,tsrc,1,p.ptr(),lattice,&lanc);
	
	//Add both + and - source momentum  (PropMomContainer manages prop memory)
	PropWrapper prop_pplus(prop_f0,prop_f1,false);
	props.insert(prop_pplus, propTag(qtype,pp,tsrc,p));
	
	PropWrapper prop_pminus(prop_f0,prop_f1,true);
	props.insert(prop_pminus, propTag(qtype,pp,tsrc,-p));
      }else{
	QPropWMomSrc* prop = computePropagator(mass,prec,tsrc,0,p.ptr(),lattice,&lanc);
	PropWrapper propw(prop);
	props.insert(propw, propTag(qtype,pp,tsrc,p));
      }
    }
  }
}


void writePion2ptLW(fMatrix<double> &results, const std::string &results_dir, const std::string &snk_op, const ThreeMomentum &p1, const ThreeMomentum &p2, const PropPrecision status, const int conf, const std::string &extra_descr){
  std::ostringstream os; 
  os << results_dir << "/pion_" << snk_op << "_P_LW_mom" << (-p1).file_str() << "_plus" << p2.file_str() << (status == Sloppy ? "_sloppy" : "_exact") << extra_descr << '.' << conf;
  results.write(os.str());
}

//Pion 2pt LW functions pseudoscalar and axial sinks
void measurePion2ptLWStandard(const PropMomContainer &props, const PropPrecision status, const std::vector<int> &tslices, const MesonMomenta &ll_meson_momenta,
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
	    
	PropWrapper &prop1 = props.get(propTag(Light,status,tsrc,p1));
	PropWrapper &prop2 = props.get(propTag(Light,status,tsrc,p2));

	pionTwoPointLWStandard(results,tsrc,sink_ops[op],p1,p2,prop1,prop2);
      }
      writePion2ptLW(results, results_dir, sink_op_stub[op], p1, p2, status, conf, "");
    }
  }
}

void measurePion2ptLWGparity(const PropMomContainer &props, const PropPrecision status, const std::vector<int> &tslices, const MesonMomenta &ll_meson_momenta,
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
	    
	PropWrapper &prop1 = props.get(propTag(Light,status,tsrc,p1));
	PropWrapper &prop2 = props.get(propTag(Light,status,tsrc,p2));

	pionTwoPointLWGparity(results,tsrc,sink_ops[op],p1,p2,prop1,prop2,SPLANE_BOUNDARY,false,false); //right proj op, right sink mom

	if(op == 4){ //only pseudoscalar sink op
	  pionTwoPointLWGparity(results_wrongsinkmom,tsrc,sink_ops[op],p1,p2,prop1,prop2,SPLANE_BOUNDARY,false,true); //right proj op, right sink mom
	  
	  pionTwoPointLWGparity(results_wrongproj,tsrc,sink_ops[op],p1,p2,prop1,prop2,SPLANE_BOUNDARY,true,false); //wrong proj op, right sink mom
	  pionTwoPointLWGparity(results_wrongproj_wrongsinkmom,tsrc,sink_ops[op],p1,p2,prop1,prop2,SPLANE_BOUNDARY,true,true); //wrong proj op, wrong sink mom
	}
      }
      writePion2ptLW(results, results_dir, sink_op_stub[op], p1, p2, status, conf, "");
      if(op == 4){
	writePion2ptLW(results_wrongsinkmom, results_dir, sink_op_stub[op], p1, p2, status, conf, "_wrongsinkmom");
	writePion2ptLW(results_wrongproj, results_dir, sink_op_stub[op], p1, p2, status, conf, "_wrongproj");
	writePion2ptLW(results_wrongproj_wrongsinkmom, results_dir, sink_op_stub[op], p1, p2, status, conf, "_wrongproj_wrongsinkmom");
      }
    }
  }
}

void measurePion2ptLW(const PropMomContainer &props, const PropPrecision status, const std::vector<int> &tslices, const MesonMomenta &ll_meson_momenta,
		      const std::string &results_dir, const int conf){
  if(GJP.Gparity()) return measurePion2ptLWGparity(props,status,tslices,ll_meson_momenta,results_dir, conf);
  else return measurePion2ptLWStandard(props,status,tslices,ll_meson_momenta,results_dir, conf);
}



//Pion 2pt LW functions pseudoscalar sink
void measurePion2ptPPWW(const PropMomContainer &props, const PropPrecision status, const std::vector<int> &tslices, const MesonMomenta &ll_meson_momenta, Lattice &lat,
			const std::string results_dir, const int conf){
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
    fMatrix<double> results_momexch(Lt,Lt); //[tsrc][tsnk-tsrc]

    for(int s=0;s<tslices.size();s++){
      const int tsrc = tslices[s];
      
      //Prop1
      PropWrapper &prop1 = props.get(propTag(Light,status,tsrc,p1));
     
      WallSinkProp<SpinColorFlavorMatrix> prop1_FT_keep; 
      prop1_FT_keep.setProp(prop1);
      prop1_FT_keep.compute(lat, p1_snk_keep.ptr());

      WallSinkProp<SpinColorFlavorMatrix> prop1_FT_exch; 
      prop1_FT_exch.setProp(prop1);
      prop1_FT_exch.compute(lat, p1_snk_exch.ptr());      

      //Prop2
      PropWrapper &prop2 = props.get(propTag(Light,status,tsrc,p2));

      WallSinkProp<SpinColorFlavorMatrix> prop2_FT_keep; 
      prop2_FT_keep.setProp(prop2);
      prop2_FT_keep.compute(lat, p2_snk_keep.ptr());

      WallSinkProp<SpinColorFlavorMatrix> prop2_FT_exch; 
      prop2_FT_exch.setProp(prop2);
      prop2_FT_exch.compute(lat, p2_snk_exch.ptr());   
 
      pionTwoPointPPWWGparity(results_momkeep, tsrc, p1, p2_snk_keep, prop1_FT_keep, prop2_FT_keep);
      pionTwoPointPPWWGparity(results_momexch, tsrc, p1, p2_snk_exch, prop1_FT_exch, prop2_FT_exch);
    }
    {
      std::ostringstream os; 
      os << results_dir << "/pion_P_P_WW_mom" << (-p1).file_str() << "_plus" << p2.file_str() << "_keep_" << (status == Sloppy ? "sloppy" : "exact") << '.' << conf;
      results_momkeep.write(os.str());
    }
    {
      std::ostringstream os; 
      os << results_dir << "/pion_P_P_LW_mom" << (-p1).file_str() << "_plus" << p2.file_str() << "_exch_" << (status == Sloppy ? "sloppy" : "exact") << '.' << conf;
      results_momexch.write(os.str());
    }
  }
}


//SU(2) singlet 2pt LW functions
void measureLightFlavorSingletLW(const PropMomContainer &props, const PropPrecision pp, const std::vector<int> &tslices, const MesonMomenta &meson_momenta,
				 const std::string results_dir, const PropPrecision status, const int conf){

  const int Lt = GJP.Tnodes()*GJP.TnodeSites();

  //Loop over light-light meson momenta
  for(int pidx=0;pidx<meson_momenta.nMom();pidx++){
    ThreeMomentum p1 = meson_momenta.getQuarkMom(0,pidx); //note the total meson momentum is p2 - p1 because the Hermitian conjugate of the first propagator swaps the momentum
    ThreeMomentum p2 = meson_momenta.getQuarkMom(1,pidx);
	  
    fMatrix<double> results(Lt,Lt); //[tsrc][tsnk-tsrc]
    for(int s=0;s<tslices.size();s++){
      const int tsrc = tslices[s];
	    
      PropWrapper &prop1 = props.get(propTag(Light,pp,tsrc,p1));
      PropWrapper &prop2 = props.get(propTag(Light,pp,tsrc,p2));

      lightFlavorSingletLWGparity(results,tsrc,p1,p2,prop1,prop2);

      {
	std::ostringstream os;
	os << results_dir << "/light_pseudo_flav_singlet_LW_mom" << (-p1).file_str() << "_plus" << p2.file_str() << (status == Sloppy ? "_sloppy" : "_exact") << '.' << conf;
	results.write(os.str());
      }
    }
  }
}

//Measure BK with source kaons on each of the timeslices t0 in t0_vals and K->K time separations tseps
void measureBK(const PropMomContainer &props, const PropPrecision pp, const std::vector<int> &t0_vals, const std::vector<int> &tseps, const MesonMomenta &meson_momenta,
	       const std::string results_dir, const PropPrecision status, const int conf, const bool do_flavor_project = true){
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();
  
  //Do all combinations of source and sink kaon momenta that have the same total momentum. This allows us to look at alternate quark momentum combinations
  //In the meson_momenta, prop index 0 is the strange quark (as in the standard 2pt function case), and is the propagator that is daggered. Prop index 1 is the light quark.
  for(int p0idx=0;p0idx<meson_momenta.nMom();p0idx++){
    ThreeMomentum prop_h_t0_srcmom = meson_momenta.getQuarkMom(0,p0idx);
    ThreeMomentum prop_l_t0_srcmom = meson_momenta.getQuarkMom(1,p0idx);

    for(int p1idx=p0idx;p1idx<meson_momenta.nMom();p1idx++){
      if(meson_momenta.getMesonMom(p1idx) != meson_momenta.getMesonMom(p0idx)) continue;

      ThreeMomentum prop_h_t1_srcmom = meson_momenta.getQuarkMom(0,p1idx);
      ThreeMomentum prop_l_t1_srcmom = meson_momenta.getQuarkMom(1,p1idx);
      
      for(int tspi=0;tspi<tseps.size();tspi++){
	fMatrix<double> results(Lt,Lt);

	for(int t0i=0;t0i<t0_vals.size();t0i++){
	  int t0 = t0_vals[t0i];
	  int t1 = (t0 + tseps[tspi]) % Lt;

	  PropWrapper &prop_h_t0 = props.get(propTag(Heavy,pp,t0,prop_h_t0_srcmom));
	  PropWrapper &prop_l_t0 = props.get(propTag(Light,pp,t0,prop_l_t0_srcmom));

	  PropWrapper &prop_h_t1 = props.get(propTag(Heavy,pp,t1,prop_h_t1_srcmom));
	  PropWrapper &prop_l_t1 = props.get(propTag(Light,pp,t1,prop_l_t1_srcmom));

	  GparityBK(results, t0, 
		    prop_h_t0, prop_l_t0, prop_h_t0_srcmom,
		    prop_h_t1, prop_l_t1, prop_h_t1_srcmom,
		    do_flavor_project);
	}
	{
	  std::ostringstream os;
	  os << results_dir << "/BK_srcK_mom" << (-prop_h_t0_srcmom).file_str() << "_plus" << prop_l_t0_srcmom.file_str() 
	     << "snkK_mom" << (-prop_h_t1_srcmom).file_str() << "_plus" << prop_l_t1_srcmom.file_str()
	     << "_tsep" << tseps[tspi]
	     << (status == Sloppy ? "_sloppy" : "_exact") << '.' << conf;
	  results.write(os.str());
	}
      }
    }
  }
}


CPS_END_NAMESPACE


#endif
