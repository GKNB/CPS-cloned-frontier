#ifndef _PION_TWOPOINT_H
#define _PION_TWOPOINT_H

CPS_START_NAMESPACE

//Assumes momenta are in units of \pi/2L, and must be *odd integer* (checked)
inline static int getProjSign(const int p[3]){
  if(!GJP.Gparity()){ ERR.General("","getProjSign","Requires GPBC in at least one direction\n"); }

  //Sign is exp(i\pi n_p)
  //where n_p is the solution to  p_j = \pi/2L( 1 + 2n_p )
  //Must be consistent for all j

  int np;
  for(int j=0;j<3;j++){
    if(GJP.Bc(j)!=BND_CND_GPARITY) continue;

    if(abs(p[j]) %2 != 1){ ERR.General("","getProjSign","Component %d of G-parity momentum (%d,%d,%d) is invalid as it is not an odd integer!\n",j,p[0],p[1],p[2]); }
    int npj = (p[j] - 1)/2;
    if(j == 0) np = npj;
    else if(abs(npj)%2 != abs(np)%2){ 
      ERR.General("","getProjSign","Momentum component %d of G-parity momentum (%d,%d,%d) is invalid because it doesn't differ from component 0 by multiple of 2pi (4 in these units). Got np(0)=%d, np(j)=%d\n",j,p[0],p[1],p[2],np,npj); 
    }
  }
  int sgn = (abs(np) % 2 == 0 ? 1 : -1); //exp(i\pi n_p) = exp(-i\pi n_p)  for all integer n_p
  //if(!UniqueID()){ printf("getProjSign got sign %d (np = %d) for p=(%d,%d,%d)pi/2L\n",sgn,np,p[0],p[1],p[2]); fflush(stdout); }

  return sgn;
}

inline FlavorMatrix getProjector(const ThreeMomentum &p){
  const int proj_sign = getProjSign(p.ptr());
  //1/2(1 +/- sigma2)
  FlavorMatrix proj(0.0); 
  proj(0,0) = proj(0.5,0.5) = 1.0;
  proj(0,1) = Complex(0,-0.5*proj_sign);
  proj(1,0) = Complex(0,0.5*proj_sign);
  return proj;
}

class SrcSnkOp{
public:
  virtual void rightMultiply(SpinColorFlavorMatrix &prop) = 0;
};

class BasicOp : public SrcSnkOp{
  SpinMatrixType smat;
  FlavorMatrixType fmat;
public:
  BasicOp(const SpinMatrixType _smat, const FlavorMatrixType _fmat): smat(_smat), fmat(_fmat){}
  void rightMultiply(SpinColorFlavorMatrix &prop){
    switch(smat){
    case(gamma1):
      prop.gr(0); break;
    case(gamma2):
      prop.gr(1); break;
    case(gamma3):
      prop.gr(2); break;
    case(gamma4):
      prop.gr(3); break;
    case(gamma5):
      prop.gr(-5); break;
    case(spin_unit):
    default:
      break;
    }
    prop.pr(fmat);
  }
};

class OpWithFlavorProject : public BasicOp{
  ThreeMomentum p_psi;
public:
  OpWithFlavorProject(const SpinMatrixType _smat, const FlavorMatrixType _fmat, const ThreeMomentum &_p_psi, bool _use_wrong_proj_sign = false): BasicOp(_smat,_fmat), p_psi(_p_psi){}

  void rightMultiply(SpinColorFlavorMatrix &prop){
    this->BasicOp::rightMultiply(prop);
    FlavorMatrix proj = getProjector(p_psi);    
    prop *= proj;
  }
};



//Assume form
// coeff * \sum_y e^{i(-p1 + p2)y} Tr{  ( prop1(y,p1;tsrc) )^dag SinkOp prop2(y,p2;tsrc) SrcOp }
void twoPointFunctionGeneric(fMatrix<double> &into, const int tsrc, const Complex &coeff,
			     const SrcSnkOp &sink_op, const SrcSnkOp &src_op,
			     const ThreeMomentum &p1, const ThreeMomentum &p2,
			     const PropWrapper &prop1, const PropWrapper &prop2){
  ThreeMomentum p_tot_src = p2 - p1;
  ThreeMomentum p_tot_snk = -p_tot_src; //mom_phase computes exp(-p.x)

  const int Lt = GJP.TnodeSites()*GJP.Tnodes();

  const int nthread = omp_get_max_threads();
  basicComplexArray<double> tmp(Lt,nthread); //defaults to zero for all elements

  //Coefficient
#pragma omp_parallel for
  for(int x=0;x<GJP.VolNodeSites();x++){
    int pos[4];
    int rem = x;
    for(int i=0;i<4;i++){ pos[i] = rem % GJP.NodeSites(i); rem /= GJP.NodeSites(i); }

    int t_glb = pos[3] + GJP.TnodeCoor() * GJP.TnodeSites();
    int tdis_glb = (t_glb - tsrc + Lt) % Lt;
    
    SpinColorFlavorMatrix prop1_site;
    prop1.siteMatrix(prop1_site,x);
    prop1_site.hconj();
    sink_op.rightMultiply(prop1_site);
    
    SpinColorFlavorMatrix prop2_site;
    prop2.siteMatrix(prop2_site,x);
    src_op.rightMultiply(prop2_site);

    std::complex<double> phase = coeff * mom_phase(p_tot_snk, pos);

    tmp(tdis_glb, omp_get_thread_num()) += phase * Trace(prop1_site, prop2_site);
  }
  tmp.threadSum();
  tmp.nodeSum();

  for(int tdis=0;tdis<Lt;tdis++)
    into(tsrc, tdis) = tmp[tdis];
}

void TwoPointFunctionWallSinkGeneric(fMatrix<double> &into, const int tsrc, const Complex &coeff,
				     const SrcSnkOp &sink_op, const SrcSnkOp &src_op,
				     const WallSinkProp<SpinColorFlavorMatrix> &prop1W, const WallSinkProp<SpinColorFlavorMatrix> &prop2W){
  const int Lt = GJP.TnodeSites()*GJP.Tnodes();
  basicComplexArray<double> tmp(Lt,1); 

  //WallSinkProp are available for all times on every node, so no need to nodeSum

#pragma omp_parallel for
  for(int t_dis=0;t_dis<Lt;t_dis++){
    int tsnk = (tsrc + t_dis) % Lt;

    SpinColorFlavorMatrix prop1_t = prop1W(tsnk);
    prop1_t.hconj();
    sink_op.rightMultiply(prop1_t);

    SpinColorFlavorMatrix prop2_t = prop2W(tsnk);
    src_op.rightMultiply(prop2_t);

    tmp[t_dis] = coeff * Trace(prop1_t, prop2_t);
  }

  for(int tdis=0;tdis<Lt;tdis++)
    into(tsrc, tdis) = tmp[tdis];
}






//cf Eq 160 of GP paper. prop1 is the one that is daggered
//tsrc is used as the row index of the fmatrix. col index is (tsnk - tsrc + Lt) % Lt
//p1 is the source momentum of the first propagator (the one that will be daggered) and p2 is that of the second
//Option to use the wrong projection sign (used for discussion in paper)
enum Pion2PtSinkOp { AX, AY, AZ, AT, P }; //Axial and pseudoscalar operator

void pionTwoPointLWGparity(fMatrix<double> &into, const int tsrc, const Pion2PtSinkOp sink_op, const ThreeMomentum &p1, const ThreeMomentum &p2,
			   const PropWrapper &prop1, const PropWrapper &prop2,
			   const bool use_wrong_proj_sign = false){

  ThreeMomentum p_psi_src = -p1; //The source operator is \bar\psi(p2) \gamma^5 \sigma_3 proj(-p1) psi(-p1)  
  if(use_wrong_proj_sign) p_psi_src = -p_psi_src;

  OpWithFlavorProject src_op(spin_unit,sigma3,p_psi_src);

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
  BasicOp snk_op(snk_spn,sigma3);

  Complex coeff(-0.5);
  if(sink_op != P) coeff = -coeff; //Sink operator is \gamma^\mu \gamma^5, additional - sign for commuting

  twoPointFunctionGeneric(into,tsrc,coeff,snk_op,src_op,p1,p2,prop1,prop2);
}

//WW 2pt function
//cf Eq 162 of GP paper. prop1 is the one that is daggered
//Here, as the propagators have already been Fourier transformed, we need only know the sink momentum of the undaggered prop (prop2) and the source momentum of the daggered prop (prop1) to determine the projectors
void pionTwoPointPPWWGparity(fMatrix<double> &into, const int tsrc, const ThreeMomentum &prop1_srcmom, const ThreeMomentum &prop2_snkmom,
			     const WallSinkProp<SpinColorFlavorMatrix> &prop1W, const WallSinkProp<SpinColorFlavorMatrix> &prop2W){
  ThreeMomentum p_psi_src = -prop1_srcmom; //momentum of psi field at source is opposite the source momentum of prop1 (cf Eq. 162)
  ThreeMomentum p_psi_snk = prop2_snkmom; //momentum of psi field at sink is the same as the momentum sink phase of prop2

  OpWithFlavorProject src_op(spin_unit,sigma3,p_psi_src);
  OpWithFlavorProject snk_op(spin_unit,sigma3,p_psi_snk);

  Complex coeff(-0.5);
  TwoPointFunctionWallSinkGeneric(into, tsrc, coeff, snk_op, src_op, prop1W, prop2W);
}

//Pseudoscalar flavor singlet \bar\psi \gamma^5 \psi
//tsrc is used as the row index of the fmatrix. col index is (tsnk - tsrc + Lt) % Lt
void lightFlavorSingletLWGparity(fMatrix<double> &into, const int tsrc, const ThreeMomentum &p1, const ThreeMomentum &p2,
				 const PropWrapper &prop1, const PropWrapper &prop2){
  ThreeMomentum p_psi_src = -p1; //The source operator is \bar\psi(p2) \gamma^5 proj(-p1) psi(-p1)
  OpWithFlavorProject src_op(spin_unit,sigma0,p_psi_src);
  BasicOp snk_op(snk_spn,sigma0);

  Complex coeff(-1,0);
  twoPointFunctionGeneric(into,tsrc,coeff,snk_op,src_op,p1,p2,prop1,prop2);
}










CPS_END_NAMESPACE
#endif
