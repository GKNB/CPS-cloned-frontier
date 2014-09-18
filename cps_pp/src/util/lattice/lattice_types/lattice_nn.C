#include<config.h>
CPS_START_NAMESPACE
/*!\file
  \brief  Implementation of GnoneFnone class.

  $Id: lattice_nn.C,v 1.4.450.1 2012/07/22 14:11:57 yinnht Exp $
*/
//--------------------------------------------------------------------
//  CVS keywords
//
//  $Author: yinnht $
//  $Date: 2012/07/22 14:11:57 $
//  $Header: /space/cvs/cps/cps++/src/util/lattice/lattice_types/lattice_nn.C,v 1.4.450.1 2012/07/22 14:11:57 yinnht Exp $
//  $Id: lattice_nn.C,v 1.4.450.1 2012/07/22 14:11:57 yinnht Exp $
//  $Name: v5_0_16_hantao_io_test_v7 $
//  $Locker:  $
//  $Revision: 1.4.450.1 $
//  $Source: /space/cvs/cps/cps++/src/util/lattice/lattice_types/lattice_nn.C,v $
//  $State: Exp $
//
//--------------------------------------------------------------------
//------------------------------------------------------------------
//
// lattice_nn.C
//
// This class has double inheritance. The virtual
// base class is Lattice with two types of derived classes. One type
// is relevant to the gauge part and has a name that begins with 
// "G". The other type is relevant to the fermion part and has
// a name that begins with "F". The class below inherits from one 
// Gnone and Fnone.
//
//------------------------------------------------------------------

CPS_END_NAMESPACE
#include <util/lattice.h>
#include <util/verbose.h>
CPS_START_NAMESPACE


//------------------------------------------------------------------
// No gauge action -- no fermion action
//------------------------------------------------------------------
GnoneFnone::GnoneFnone()
{
  cname = "GnoneFnone";
  char *fname = "GnoneFnone()";
  VRB.Func(cname,fname);

  //???
}

GnoneFnone::~GnoneFnone()
{
  char *fname = "~GnoneFnone()";
  VRB.Func(cname,fname);

  //???
}



CPS_END_NAMESPACE
