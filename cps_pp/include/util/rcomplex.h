#ifndef INCLUDED_RCOMPLEX_H
#define INCLUDED_RCOMPLEX_H

#include<config.h>
CPS_START_NAMESPACE
/*!\file
  \brief  Definitions of complex floating point data type.

  $Id: rcomplex.h,v 1.5.30.2 2012/07/07 20:03:31 yinnht Exp $
*/
//--------------------------------------------------------------------
//  CVS keywords
//
//  $Author: yinnht $
//  $Date: 2012/07/07 20:03:31 $
//  $Header: /space/cvs/cps/cps++/include/util/rcomplex.h,v 1.5.30.2 2012/07/07 20:03:31 yinnht Exp $
//  $Id: rcomplex.h,v 1.5.30.2 2012/07/07 20:03:31 yinnht Exp $
//  $Name: v5_0_16_hantao_io_test_v7 $
//  $Locker:  $
//  $RCSfile: rcomplex.h,v $
//  $Revision: 1.5.30.2 $
//  $Source: /space/cvs/cps/cps++/include/util/rcomplex.h,v $
//  $State: Exp $
//
//--------------------------------------------------------------------
// rcomplex.h

CPS_END_NAMESPACE
#include <util/data_types.h>
#include <math.h>
#include <complex>
CPS_START_NAMESPACE

// We use standard library for Rcomplex
typedef std::complex<IFloat> Rcomplex;

static inline Rcomplex operator/(const Rcomplex &a, IFloat b)
{
    Rcomplex tmp = a;
    tmp /= b;
    return tmp;
}

static inline Rcomplex operator*(const Rcomplex &a, int b) {return a * Float(b);}
static inline Rcomplex operator*(int b, const Rcomplex &a) {return a * Float(b);}

CPS_END_NAMESPACE

#endif
