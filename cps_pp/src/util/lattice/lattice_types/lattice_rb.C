#include<config.h>

#ifdef USE_BFM

CPS_START_NAMESPACE
//--------------------------------------------------------------------
/*!\file
  \brief  Implementation of GnoneFasqtad class.

  $Id: lattice_rb.C,v 1.1.2.2 2012/07/30 21:22:09 yinnht Exp $
*/
//--------------------------------------------------------------------

CPS_END_NAMESPACE
#include <util/lattice.h>
#include <util/lattice/fbfm.h>
#include <util/verbose.h>
CPS_START_NAMESPACE

//------------------------------------------------------------------
// Iwasaki gauge action + BFM fermion action
//------------------------------------------------------------------
GimprRectFbfm::GimprRectFbfm():cname("GimprRectFbfm")
{
}

GimprRectFbfm::~GimprRectFbfm()
{
}

CPS_END_NAMESPACE

#endif
