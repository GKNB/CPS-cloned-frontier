#ifndef _PION_TWOPOINT_H
#define _PION_TWOPOINT_H

#include "twopoint_function_generic.h"

CPS_START_NAMESPACE


//Kaon two-point. prop_h is the heavy quark propagator (the one to which g5-hermiticity is applied), and prop_l the light-quark prop
//tsrc is used as the row index of the fmatrix. col index is (tsnk - tsrc + Lt) % Lt
void kaonTwoPointPPLWGparity(fMatrix<double> &into, const int tsrc, const ThreeMomentum &p_s, const ThreeMomentum &p_l,
			      const PropWrapper &prop_s, const PropWrapper &prop_l){
  ThreeMomentum p_psi_src = -p_s; //The source operator is \bar\psi_l(p2) \gamma^5 proj(-p1) psi_h(-p1)
  GparityOpWithFlavorProject src_op(spin_unit,sigma0,p_psi_src);
  BasicGparityOp snk_op(spin_unit,sigma0);

  Complex coeff(0.5,0); //note positive sign because we define the creation/annihilation operator with a factor of i
  twoPointFunctionGeneric(into,tsrc,coeff,snk_op,src_op,p_s,p_l,prop_h,prop_l);
}

//*Physical* time-component axial operator sink   \sqrt(2) F0 g4 g5    (g5 is removed by g5-hermiticity)
void kaonTwoPointA4PhysPLWGparity(fMatrix<double> &into, const int tsrc, const ThreeMomentum &p_s, const ThreeMomentum &p_l,
			      const PropWrapper &prop_s, const PropWrapper &prop_l){
  ThreeMomentum p_psi_src = -p_s; //The source operator is \bar\psi_l(p2) \gamma^5 proj(-p1) psi_h(-p1)
  GparityOpWithFlavorProject src_op(spin_unit,sigma0,p_psi_src);
  BasicGparityOp snk_op(gamma4,F0);

  Complex coeff(-1.0,0);
  twoPointFunctionGeneric(into,tsrc,coeff,snk_op,src_op,p_s,p_l,prop_h,prop_l);
}
//*Unphysical* time-component axial operator sink   \sqrt(2) F1 g4 g5, connects to unphysical kaon component    (g5 is removed by g5-hermiticity)
void kaonTwoPointA4UnphysPLWGparity(fMatrix<double> &into, const int tsrc, const ThreeMomentum &p_s, const ThreeMomentum &p_l,
			      const PropWrapper &prop_s, const PropWrapper &prop_l){
  ThreeMomentum p_psi_src = -p_s; //The source operator is \bar\psi_l(p2) \gamma^5 proj(-p1) psi_h(-p1)
  GparityOpWithFlavorProject src_op(spin_unit,sigma0,p_psi_src);
  BasicGparityOp snk_op(gamma4,F1);

  Complex coeff(-1.0,0);
  twoPointFunctionGeneric(into,tsrc,coeff,snk_op,src_op,p_s,p_l,prop_h,prop_l);
}
//Time-component axial source and sink that connects to both the physical and unphysical components
void kaonTwoPointA4combA4combLWGparity(fMatrix<double> &into, const int tsrc, const ThreeMomentum &p_s, const ThreeMomentum &p_l,
			      const PropWrapper &prop_s, const PropWrapper &prop_l){
  ThreeMomentum p_psi_src = -p_s; //The source operator is \bar\psi_l(p2) \gamma^4\gamma^5 proj(-p1) psi_h(-p1)
  GparityOpWithFlavorProject src_op(gamma4,sigma0,p_psi_src);
  BasicGparityOp snk_op(gamma4,sigma0);

  Complex coeff(-0.5,0);
  twoPointFunctionGeneric(into,tsrc,coeff,snk_op,src_op,p_s,p_l,prop_h,prop_l);
}

void kaonTwoPointPPWWGparity(fMatrix<double> &into, const int tsrc, const ThreeMomentum &prop_s_srcmom, const ThreeMomentum &prop_l_snkmom,
			     const WallSinkProp<SpinColorFlavorMatrix> &prop_s_W, const WallSinkProp<SpinColorFlavorMatrix> &prop_l_W){
  ThreeMomentum p_psi_src = -prop_s_srcmom; //momentum of psi field at source is opposite the source momentum of prop_s (cf Eq. 162)
  ThreeMomentum p_psi_snk = prop_l_snkmom; //momentum of psi field at sink is the same as the momentum sink phase of prop_l

  GparityOpWithFlavorProject src_op(spin_unit,sigma0,p_psi_src);
  GparityOpWithFlavorProject snk_op(spin_unit,sigma0,p_psi_snk);

  if(!UniqueID()) std::cout << "Kaon PPWW with source proj " << src_op.printProj() << " and sink proj " << snk_op.printProj() << '\n';

  Complex coeff(0.5,0);
  TwoPointFunctionWallSinkGeneric(into, tsrc, coeff, snk_op, src_op, prop_s_W, prop_l_W);
}




CPS_END_NAMESPACE
#endif
