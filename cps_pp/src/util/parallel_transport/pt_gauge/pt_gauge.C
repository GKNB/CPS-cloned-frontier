#include <config.h>
CPS_START_NAMESPACE
/*! \file
  \brief  Definition of ParTransStagTypes class constructor and destructor.

  $Id: pt_gauge.C,v 1.4.48.1 2012-11-15 18:17:09 ckelly Exp $
*/
//--------------------------------------------------------------------
//  CVS keywords
//
//  $Author: ckelly $
//  $Date: 2012-11-15 18:17:09 $
//  $Header: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/src/util/parallel_transport/pt_gauge/pt_gauge.C,v 1.4.48.1 2012-11-15 18:17:09 ckelly Exp $
//  $Id: pt_gauge.C,v 1.4.48.1 2012-11-15 18:17:09 ckelly Exp $
//  $Name: not supported by cvs2svn $
//  $Locker:  $
//  $RCSfile: pt_gauge.C,v $
//  $Revision: 1.4.48.1 $
//  $Source: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/src/util/parallel_transport/pt_gauge/pt_gauge.C,v $
//  $State: Exp $
//
//--------------------------------------------------------------------

CPS_END_NAMESPACE
#include <util/pt.h>
#include <util/lattice.h>
#include <util/verbose.h>
#include <util/dirac_op.h>
CPS_START_NAMESPACE

//------------------------------------------------------------------
/*!
  \param latt The lattice on which this operation is defined
 */
//------------------------------------------------------------------

static StrOrdType old_str_ord;
ParTransGauge::ParTransGauge(Lattice & latt) :
                                   ParTrans(latt)
{
  cname = "ParTransGauge";
  char *fname = "ParTransGauge(Lattice&)";
  VRB.Func(cname,fname);
//  if (lat.StrOrd() != WILSON && lat.StrOrd() != CANONICAL ){
  if (lat.StrOrd() != CANONICAL ){
    old_str_ord = lat.StrOrd();
    lat.Convert(CANONICAL);
  }
  if(GJP.Gparity() && DiracOp::scope_lock ==0)
    BondCond(latt, gauge_field); //remove fermionic boundary conditions on gauge links (they are applied in the ParTrans constructor)

  pt_init(lat);
  pt_init_g();
}


//------------------------------------------------------------------
ParTransGauge::~ParTransGauge() {
  char *fname = "~ParTransGauge()";
  VRB.Func(cname,fname);
  pt_delete_g();
  pt_delete();

  if(GJP.Gparity() && DiracOp::scope_lock ==0)
    BondCond(lat, gauge_field); //remove fermionic boundary conditions on gauge links (they are applied in the ParTrans destructor)

//  if (old_str_ord != WILSON && old_str_ord != CANONICAL ){
  if (old_str_ord != CANONICAL ){
    lat.Convert(old_str_ord);
  }
}

CPS_END_NAMESPACE
