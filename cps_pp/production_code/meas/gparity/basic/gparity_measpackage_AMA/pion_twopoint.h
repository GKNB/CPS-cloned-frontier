#ifndef _PION_TWOPOINT_H
#define _PION_TWOPOINT_H

#include "twopoint_function_generic.h"

CPS_START_NAMESPACE

//cf Eq 160 of GP paper. prop1 is the one that is daggered
//tsrc is used as the row index of the fmatrix. col index is (tsnk - tsrc + Lt) % Lt
//p1 is the source momentum of the first propagator (the one that will be daggered) and p2 is that of the second
//Option to use the wrong projection sign and wrong sink momentum (used for discussion in paper)
enum Pion2PtSinkOp { AX, AY, AZ, AT, P }; //Axial and pseudoscalar operator

inline SpinMatrixType snkOpMap(const Pion2PtSinkOp sink_op){
  SpinMatrixType snk_spn = spin_unit;
  switch(sink_op){
  case AX:
    snk_spn = gamma1; break;
  case AY:
    snk_spn = gamma2; break;
  case AZ:
    snk_spn = gamma3; break;
  case AT:
    snk_spn = gamma4; break;
  default:
    break;
  }
  return snk_spn;
}

void pionTwoPointLWStandard(fMatrix<double> &into, const int tsrc, const Pion2PtSinkOp sink_op, const ThreeMomentum &p1, const ThreeMomentum &p2,
			   const PropWrapper &prop1, const PropWrapper &prop2){

  BasicOp src_op(spin_unit);
  BasicOp snk_op(snkOpMap(sink_op));

  Complex coeff(1.0);
  twoPointFunctionGeneric(into,tsrc,coeff,snk_op,src_op,p1,p2,prop1,prop2);
}


void pionTwoPointLWGparity(fMatrix<double> &into, const int tsrc, const Pion2PtSinkOp sink_op, const ThreeMomentum &p1, const ThreeMomentum &p2,
			   const PropWrapper &prop1, const PropWrapper &prop2,
			   const PropSplane splane = SPLANE_BOUNDARY,
			   const bool use_wrong_proj_sign = false, const bool use_wrong_sink_mom = false){

  ThreeMomentum p_psi_src = -p1; //The source operator is \bar\psi(p2) \gamma^5 \sigma_3 proj(-p1) psi(-p1)  
  if(use_wrong_proj_sign) p_psi_src = -p_psi_src;

  GparityOpWithFlavorProject src_op(spin_unit,sigma3,p_psi_src);
  BasicGparityOp snk_op(snkOpMap(sink_op),sigma3);

  Complex coeff(0.5);
  twoPointFunctionGeneric(into,tsrc,coeff,snk_op,src_op,p1,p2,prop1,prop2,splane,use_wrong_sink_mom);
}

void pionTwoPointA4A4LWGparity(fMatrix<double> &into, const int tsrc, const ThreeMomentum &p1, const ThreeMomentum &p2,
			   const PropWrapper &prop1, const PropWrapper &prop2){

  ThreeMomentum p_psi_src = -p1; //The source operator is \bar\psi(p2) \gamma^4\gamma^5 \sigma_3 proj(-p1) psi(-p1)  

  GparityOpWithFlavorProject src_op(gamma4,sigma3,p_psi_src);
  BasicGparityOp snk_op(gamma4,sigma3);

  Complex coeff(0.5); //Hermitian conjugate of A_4 at source
  twoPointFunctionGeneric(into,tsrc,coeff,snk_op,src_op,p1,p2,prop1,prop2);
}




//WW 2pt function
//cf Eq 162 of GP paper. prop1 is the one that is daggered
//Here, as the propagators have already been Fourier transformed, we need only know the sink momentum of the undaggered prop (prop2) and the source momentum of the daggered prop (prop1) to determine the projectors
void pionTwoPointPPWWGparity(fMatrix<double> &into, const int tsrc, const ThreeMomentum &prop1_srcmom, const ThreeMomentum &prop2_snkmom,
			     const WallSinkProp<SpinColorFlavorMatrix> &prop1W, const WallSinkProp<SpinColorFlavorMatrix> &prop2W){
  ThreeMomentum p_psi_src = -prop1_srcmom; //momentum of psi field at source is opposite the source momentum of prop1 (cf Eq. 162)
  ThreeMomentum p_psi_snk = prop2_snkmom; //momentum of psi field at sink is the same as the momentum sink phase of prop2

  GparityOpWithFlavorProject src_op(spin_unit,sigma3,p_psi_src);
  GparityOpWithFlavorProject snk_op(spin_unit,sigma3,p_psi_snk);

  if(!UniqueID()) std::cout << "PPWW with source proj " << src_op.printProj() << " and sink proj " << snk_op.printProj() << '\n';

  Complex coeff(0.5);
  twoPointFunctionWallSinkGeneric(into, tsrc, coeff, snk_op, src_op, prop1W, prop2W);
}

//Pseudoscalar flavor singlet \bar\psi \gamma^5 \psi
//tsrc is used as the row index of the fmatrix. col index is (tsnk - tsrc + Lt) % Lt
void lightFlavorSingletLWGparity(fMatrix<double> &into, const int tsrc, const ThreeMomentum &p1, const ThreeMomentum &p2,
				 const PropWrapper &prop1, const PropWrapper &prop2){
  ThreeMomentum p_psi_src = -p1; //The source operator is \bar\psi(p2) \gamma^5 proj(-p1) psi(-p1)
  GparityOpWithFlavorProject src_op(spin_unit,sigma0,p_psi_src);
  BasicGparityOp snk_op(spin_unit,sigma0);

  Complex coeff(0.5,0);
  twoPointFunctionGeneric(into,tsrc,coeff,snk_op,src_op,p1,p2,prop1,prop2);
}

//J5 or J5q
void J5Gparity(fMatrix<double> &into, const int tsrc, const ThreeMomentum &p1, const ThreeMomentum &p2,
	       const PropWrapper &prop1, const PropWrapper &prop2, const PropSplane splane = SPLANE_BOUNDARY, bool do_source_project = true){
  ThreeMomentum p_psi_src = -p1;

  SrcSnkOp<SpinColorFlavorMatrix>* src_op;
  if(do_source_project) src_op = new GparityOpWithFlavorProject(spin_unit,sigma3,p_psi_src);
  else src_op = new BasicGparityOp(spin_unit,sigma3);

  BasicGparityOp snk_op(spin_unit,sigma3);

  Complex coeff(0.5);
  twoPointFunctionGeneric(into,tsrc,coeff,snk_op,*src_op,p1,p2,prop1,prop2,splane);

  delete src_op;
}








CPS_END_NAMESPACE
#endif
