#ifndef _MEAS_GP_H  
#define _MEAS_GP_H  

#include "pion_twopoint.h"
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
void lightQuarkInvert(PropMomContainer &props, const PropPrecision pp, const double prec, const double ml,
		      const std::vector<int> &tslices, const QuarkMomenta &light_quark_momenta,
		      Lattice &lattice, BFM_Krylov::Lanczos_5d<double> &lanc_l){
  for(int s=0;s<tslices.size();s++){
    const int tsrc = tslices[s];
    
    for(int pidx=0;pidx<light_quark_momenta.nMom();pidx++){
      const ThreeMomentum &p = light_quark_momenta.getMom(pidx);
	  
      QPropWMomSrc* prop_f0 = computePropagator(ml,prec,tsrc,0,p.ptr(),lattice,&lanc_l);
      QPropWMomSrc* prop_f1 = computePropagator(ml,prec,tsrc,1,p.ptr(),lattice,&lanc_l);

      //Add both + and - source momentum  (PropMomContainer manages prop memory)
      PropWrapper prop_pplus(prop_f0,prop_f1,false);
      props.insert(prop_pplus, propTag(Light,pp,tsrc,p));
      
      PropWrapper prop_pminus(prop_f0,prop_f1,true);
      props.insert(prop_pminus, propTag(Light,pp,tsrc,-p));
    }
  }
}

//Pion 2pt LW functions pseudoscalar and axial sinks	     
void measurePion2ptLW(const PropMomContainer &props, const PropPrecision status, const std::vector<int> &tslices, const MesonMomenta &ll_meson_momenta,
		      const std::string results_dir, const int conf){
  //Loop over light-light meson momenta
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();

  for(int pidx=0;pidx<ll_meson_momenta.nMom();pidx++){
    ThreeMomentum p1 = ll_meson_momenta.getQuarkMom(0,pidx); //note the total meson momentum is p2 - p1 because the Hermitian conjugate of the first propagator swaps the momentum
    ThreeMomentum p2 = ll_meson_momenta.getQuarkMom(1,pidx);
	  
    Pion2PtSinkOp sink_ops[5] = { AX, AY, AZ, AT, P };//Generate a flavor 'f' gauge fixed wall momentum propagator from given timeslice. Momenta are in units of pi/2L
    std::string sink_op_stub[5] = { "AX", "AY", "AZ", "AT", "P" };
	
    for(int op=0;op<5;op++){

      fMatrix<double> results(Lt,Lt); //[tsrc][tsnk-tsrc]
      fMatrix<double> results_wrongproj(Lt,Lt); //[tsrc][tsnk-tsrc]  opposite projection op (optional, used for paper)
	  
      for(int s=0;s<tslices.size();s++){
	const int tsrc = tslices[s];
	    
	PropWrapper &prop1 = props.get(propTag(Light,status,tsrc,p1));
	PropWrapper &prop2 = props.get(propTag(Light,status,tsrc,p2));

	pionTwoPointLWGparity(results,tsrc,sink_ops[op],p1,p2,prop1,prop2);
	pionTwoPointLWGparity(results_wrongproj,tsrc,sink_ops[op],p1,p2,prop1,prop2,true); //wrong proj op
      }
      {
	std::ostringstream os; //pmeson.file_str(2) in units of pi/L
	os << results_dir << "/pion_" << sink_op_stub[op] << "_P_LW_mom" << (-p1).file_str() << "_plus" << p2.file_str() << (status == Sloppy ? "_sloppy" : "_exact") << '.' << conf;
	results.write(os.str());
      }
      {
	std::ostringstream os; //pmeson.file_str(2) in units of pi/L
	os << results_dir << "/pion_" << sink_op_stub[op] << "_P_LW_mom" << (-p1).file_str() << "_plus" << p2.file_str() << (status == Sloppy ? "_sloppy" : "_exact") << "_wrongproj." << conf;
	results_wrongproj.write(os.str());
      }
    }
  }
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
      std::ostringstream os; //pmeson.file_str(2) in units of pi/L
      os << results_dir << "/pion_P_P_WW_mom" << (-p1).file_str() << "_plus" << p2.file_str() << "_keep_" << (status == Sloppy ? "sloppy" : "exact") << '.' << conf;
      results_momkeep.write(os.str());
    }
    {
      std::ostringstream os; //pmeson.file_str(2) in units of pi/L
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





CPS_END_NAMESPACE


#endif
