#include <config.h>
CPS_START_NAMESPACE
/*! \file
  \brief  Definition of ParTransStagTypes class constructor and destructor.

  $Id: pt_stag_types.C,v 1.6 2004/08/18 11:58:07 zs Exp $
*/

CPS_END_NAMESPACE
#include <util/pt.h>
#include <util/lattice.h>
#include <util/verbose.h>
CPS_START_NAMESPACE

//------------------------------------------------------------------
/*!
  \param latt The lattice on which this operation is defined
 */
//------------------------------------------------------------------

static StrOrdType old_str_ord;
ParTransStagTypes::ParTransStagTypes(Lattice & latt) :
                                   ParTrans(latt)
{
  cname = "ParTransStagTypes";
  char *fname = "ParTransStagTypes(Lattice&)";
  VRB.Func(cname,fname);
  old_str_ord = lat.StrOrd();
  if (old_str_ord != STAG){
    lat.Convert(STAG);
  }
  pt_init(lat);
  pt_init_g();
}


//------------------------------------------------------------------
ParTransStagTypes::~ParTransStagTypes() {
  char *fname = "~ParTransStagTypes()";
  VRB.Func(cname,fname);
  if ( old_str_ord !=STAG){
    lat.Convert(old_str_ord);
  }
  pt_delete_g();
  pt_delete();
}

CPS_END_NAMESPACE
