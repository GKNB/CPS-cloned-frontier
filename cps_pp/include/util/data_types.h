#ifndef INCLUDED_DATA_TYPES_H
#define INCLUDED_DATA_TYPES_H

#include<config.h>
CPS_START_NAMESPACE
/*!\file
  \brief  Definitions of basic data types.

  $Id: data_types.h,v 1.5.386.3 2012/07/09 16:29:19 yinnht Exp $
*/
//--------------------------------------------------------------------
//  CVS keywords
//
//  $Author: yinnht $
//  $Date: 2012/07/09 16:29:19 $
//  $Header: /space/cvs/cps/cps++/include/util/data_types.h,v 1.5.386.3 2012/07/09 16:29:19 yinnht Exp $
//  $Id: data_types.h,v 1.5.386.3 2012/07/09 16:29:19 yinnht Exp $
//  $Name: v5_0_16_hantao_io_test_v7 $
//  $Locker:  $
//  $RCSfile: data_types.h,v $
//  $Revision: 1.5.386.3 $
//  $Source: /space/cvs/cps/cps++/include/util/data_types.h,v $
//  $State: Exp $
//
//--------------------------------------------------------------------

//------------------------------------------------------------------
// Global definitions:
//------------------------------------------------------------------

//------------------------------------------------------------------
//! Definition of 'Internal' floating point representation.
//------------------------------------------------------------------
//typedef INTERNAL_LOCALCALC_TYPE IFloat; 

//------------------------------------------------------------------
// Definition of rfloat and rcomplex classes:
//------------------------------------------------------------------
CPS_END_NAMESPACE
#include <util/enum.h>
#include <util/rcomplex.h>
#include <complex>
CPS_START_NAMESPACE

//------------------------------------------------------------------
//! Definition of general local floating point type.
//------------------------------------------------------------------
//typedef LOCALCALC_TYPE Float;

//------------------------------------------------------------------
//! Definition of Complex type.
//------------------------------------------------------------------
//typedef Rcomplex Complex;
typedef std::complex<IFloat> Complex;

CPS_END_NAMESPACE
#endif
