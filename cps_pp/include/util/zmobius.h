#include<config.h>
CPS_START_NAMESPACE
//--------------------------------------------------------------------
//  CVS keywords
//
//  $Author: chulwoo $
//  $Date: 2013-04-05 17:46:30 $
//  $Header: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/include/util/mobius.h,v 1.2 2013-04-05 17:46:30 chulwoo Exp $
//  $Id: mobius.h,v 1.2 2013-04-05 17:46:30 chulwoo Exp $
//  $Name: not supported by cvs2svn $
//  $Locker:  $
//  $RCSfile: mobius.h,v $
//  $Revision: 1.2 $
//  $Source: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/include/util/mobius.h,v $
//  $State: Exp $
//
//--------------------------------------------------------------------
//------------------------------------------------------------------
//
// mobius.h
//
// C header file for the complex mobius fermion library
//
//------------------------------------------------------------------

#ifndef INCLUDED_ZMOBIUS_H
#define INCLUDED_ZMOBIUS_H

CPS_END_NAMESPACE
#include <util/wilson.h>
#include <util/vector.h>
#include <util/dwf.h>
CPS_START_NAMESPACE

//------------------------------------------------------------------
// External declarations
//------------------------------------------------------------------

extern int zmobiusso_wire_map[];
// Set in mobius_int. For a given index 0-1 corresponding to
// S+, S-  it gives the corresponding wire.


//------------------------------------------------------------------
// Type definitions
//------------------------------------------------------------------


//------------------------------------------------------------------
// Function declarations
//------------------------------------------------------------------


//------------------------------------------------------------------
// This routine performs all initializations needed before mobius 
// library funcs are called. It sets the addressing related arrays 
// and reserves memory for the needed temporary buffers. It only 
// needs to be called only once at the begining of the program 
// (or after a mobius_end call)before any number of calls to mobius 
// functions are made.
//------------------------------------------------------------------
//void mobius_init(Dwf *mobius_p);             // pointer to Dwf struct.


//------------------------------------------------------------------
// This routine frees any memory reserved by mobius_init
//------------------------------------------------------------------
//void mobius_end(Dwf *mobius_p);              // pointer to Dwf struct.


//------------------------------------------------------------------
// mobius_mdagm M^dag M where M is the fermion matrix.
// The in, out fields are defined on the checkerboard lattice.
// <out, in> = <mobius_mdagm*in, in> is returned in dot_prd.
//------------------------------------------------------------------
void  zmobius_mdagm(Vector *out, 
		   Matrix *gauge_field, 
		   Vector *in, 
		   Float *dot_prd,
		   Float mass,
		   Dwf *mobius_lib_arg);


// spectrum shifted version : (H-mu)(H-mu)
void    zmobius_mdagm_shift(Vector *out, 
			   Matrix *gauge_field, 
			   Vector *in, 
			   Float *dot_prd,
			   Float mass,
			   Dwf *mobius_lib_arg,
			   Float shift);

//------------------------------------------------------------------
// mobius_dslash is the derivative part of the fermion matrix. 
// The in, out fields are defined on the checkerboard lattice
// cb = 0/1 <--> even/odd checkerboard of in field.
// dag = 0/1 <--> Dslash/Dslash^dagger is calculated.
//------------------------------------------------------------------
void zmobius_dslash(Vector *out, 
		   Matrix *gauge_field, 
		   Vector *in,
		   Float mass,
		   int cb,
		   int dag,
		   Dwf *mobius_lib_arg);


//------------------------------------------------------------------
// mobius_m is the fermion matrix.  
// The in, out fields are defined on the checkerboard lattice.
//------------------------------------------------------------------
void  zmobius_m(Vector *out, 
	       Matrix *gauge_field, 
	       Vector *in,
	       Float mass,
	       Dwf *mobius_lib_arg);
void  zmobius_mdagm(Vector *out, 
		   Matrix *gauge_field, 
		   Vector *in,
		   Float mass,
		   Dwf *mobius_lib_arg);


//------------------------------------------------------------------
// mobius_mdag is the dagger of the fermion matrix. 
// The in, out fields are defined on the checkerboard lattice
//------------------------------------------------------------------
void zmobius_mdag(Vector *out, 
		 Matrix *gauge_field, 
		 Vector *in,
		 Float mass,
		 Dwf *mobius_lib_arg);


//------------------------------------------------------------------
// mobius_dslash_4 is the derivative part of the space-time part of
// the fermion matrix. 
// The in, out fields are defined on the checkerboard lattice
// cb = 0/1 <--> even/odd checkerboard of in field.
// dag = 0/1 <--> Dslash/Dslash^dagger is calculated.
//------------------------------------------------------------------
void zmobius_dslash_4(Vector *out, 
		     Matrix *gauge_field, 
		     Vector *in,
		     int cb,
		     int dag,
		     Dwf *mobius_lib_arg, Float mass);


void zmobius_m5inv(Vector *out,
	       Vector *in, 
	       Float mass,
	       int dag,
	       Dwf *mobius_lib_arg);

// in-place version
void zmobius_m5inv(Vector *inout, 
	       Float mass,
	       int dag,
	       Dwf *mobius_lib_arg);

#if 0
void zmobius_kappa_dslash_5_plus(Vector *out, 
				Vector *in, 
				Float mass,
				int dag, 
				Dwf *mobius_lib_arg,
				Float fact);
#endif

void zmobius_kappa_dslash_5_plus_cmplx(Vector *out, 
				Vector *in, 
				Float mass,
				int dag, 
				Dwf *mobius_lib_arg,
				Complex* fact);

#if 0
void zmobius_dslash_5_plus(Vector *out, 
			  Vector *in, 
			  Float mass,
			  int dag, 
			  Dwf *mobius_lib_arg);


void zmobius_kappa_dslash_5_plus_dag0(Vector *out,
                       Vector *in,
                       Float mass,
                       Dwf *mobius_lib_arg,
                       Float a_five_inv );
void zmobius_kappa_dslash_5_plus_dag1(Vector *out,
                       Vector *in,
                       Float mass,
                       Dwf *mobius_lib_arg,
                       Float a_five_inv );
#endif

#if 0
void ReflectAndMultGamma5( Vector *out, const Vector *in,  int nodevol, int ls);

void cPRL_plus(Vector *out, Vector *in, int dag, Float mass, Dwf *mobius_lib_arg);
#endif

void zmobius_dminus(Vector *out, 
		   Matrix *gauge_field, 
		   Vector *in, 
		   int cb, 
		   int dag, 
		   Dwf *mobius_lib_arg);




// TIZB
void vecEqualsVecTimesEquComplex(Complex *a, Complex *b, Complex c, int len);
void zTimesV1PlusV2(Complex *a, Complex b, const Complex *c,
			 const Complex *d, int len);
void vecTimesEquComplex(Complex *a, Complex b, int len);

void DebugPrintVec(Vector* vp, int len);




#endif

CPS_END_NAMESPACE
