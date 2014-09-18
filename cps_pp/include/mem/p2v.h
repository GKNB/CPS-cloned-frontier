#include<config.h>
CPS_START_NAMESPACE
//--------------------------------------------------------------------
//  CVS keywords
//
//  $Author: zs $
//  $Date: 2004/08/18 11:57:37 $
//  $Header: /space/cvs/cps/cps++/include/mem/Attic/p2v.h,v 1.4 2004/08/18 11:57:37 zs Exp $
//  $Id: p2v.h,v 1.4 2004/08/18 11:57:37 zs Exp $
//  $Name: v5_0_16_hantao_io_test_v7 $
//  $Locker:  $
//  $Revision: 1.4 $
//  $Source: /space/cvs/cps/cps++/include/mem/Attic/p2v.h,v $
//  $State: Exp $
//
//--------------------------------------------------------------------
//------------------------------------------------------------------//
//
// p2v.h
//
// Physical to virtual memory copy routines.
//
//------------------------------------------------------------------//


#ifndef INCLUDED_P2V_H
#define INCLUDED_P2V_H


//------------------------------------------------------------------//
// Copies into CRAM the Vector utilities assembly code
//------------------------------------------------------------------//
void p2vVector();


//------------------------------------------------------------------//
// Copies into CRAM the Wilson library 
//------------------------------------------------------------------//
void p2vWilsonLib();


//------------------------------------------------------------------//
// Copies into CRAM the Staggered dirac_serial assembly code
//------------------------------------------------------------------//
void p2vStagDs();


//------------------------------------------------------------------//
// Copies into CRAM the Gauge heat bath (ghb) assembly code
//------------------------------------------------------------------//
void p2vGhb();


//------------------------------------------------------------------//
// Copies into CRAM the Matrix Multiplication assembly code
//------------------------------------------------------------------//
void p2vCloverLib();

#endif


CPS_END_NAMESPACE
