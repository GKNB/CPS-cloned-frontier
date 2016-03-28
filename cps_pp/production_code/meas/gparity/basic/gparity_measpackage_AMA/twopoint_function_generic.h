#ifndef _TWOPOINT_FUNCTION_GENERIC_H
#define _TWOPOINT_FUNCTION_GENERIC_H

#include "spin_flav_op.h"

CPS_START_NAMESPACE

//Assume form
// coeff * \sum_y e^{i(-p1 + p2)y} Tr{  ( prop1(y,p1;tsrc) )^dag SinkOp prop2(y,p2;tsrc) SrcOp }
//use_opposite_sink_mom optionally flips the sign of the sink momentum to the 'wrong' value - used in testing the flavor projection in the paper
template<typename MatrixType>
void twoPointFunctionGeneric(fMatrix<double> &into, const int tsrc, const Complex &coeff,
			     const SrcSnkOp<MatrixType> &sink_op, const SrcSnkOp<MatrixType> &src_op,
			     const ThreeMomentum &p1, const ThreeMomentum &p2,
			     const PropWrapper &prop1, const PropWrapper &prop2,
			     const PropSplane splane = SPLANE_BOUNDARY,
			     bool use_opposite_sink_mom = false){
  ThreeMomentum p_tot_src = p2 - p1;
  ThreeMomentum p_tot_snk = -p_tot_src; //mom_phase computes exp(-p.x)
  if(use_opposite_sink_mom) p_tot_snk = -p_tot_snk;
  
  //if(!UniqueID()) printf("Computing 2pt LW with src momentum %s and snk momentum %s\n",p_tot_src.str().c_str(),p_tot_snk.str().c_str());

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
    
    MatrixType prop1_site;
    prop1.siteMatrix(prop1_site,x,splane);
    prop1_site.hconj();
    sink_op.rightMultiply(prop1_site);
    
    MatrixType prop2_site;
    prop2.siteMatrix(prop2_site,x,splane);
    src_op.rightMultiply(prop2_site);

    std::complex<double> phase = coeff * mom_phase(p_tot_snk, pos);

    tmp(tdis_glb, omp_get_thread_num()) += phase * Trace(prop1_site, prop2_site);
  }
  tmp.threadSum();
  tmp.nodeSum();

  for(int tdis=0;tdis<Lt;tdis++)
    into(tsrc, tdis) = tmp[tdis];
}

template<typename MatrixType>
void twoPointFunctionWallSinkGeneric(fMatrix<double> &into, const int tsrc, const Complex &coeff,
				     const SrcSnkOp<MatrixType> &sink_op, const SrcSnkOp<MatrixType> &src_op,
				     const WallSinkProp<MatrixType> &prop1W, const WallSinkProp<MatrixType> &prop2W){
  const int Lt = GJP.TnodeSites()*GJP.Tnodes();
  basicComplexArray<double> tmp(Lt,1); 

  //WallSinkProp are available for all times on every node, so no need to nodeSum

#pragma omp_parallel for
  for(int t_dis=0;t_dis<Lt;t_dis++){
    int tsnk = (tsrc + t_dis) % Lt;

    MatrixType prop1_t = prop1W(tsnk);
    prop1_t.hconj();
    sink_op.rightMultiply(prop1_t);

    MatrixType prop2_t = prop2W(tsnk);
    src_op.rightMultiply(prop2_t);

    tmp[t_dis] = coeff * Trace(prop1_t, prop2_t);
  }

  for(int tdis=0;tdis<Lt;tdis++)
    into(tsrc, tdis) = tmp[tdis];
}


CPS_END_NAMESPACE

#endif
