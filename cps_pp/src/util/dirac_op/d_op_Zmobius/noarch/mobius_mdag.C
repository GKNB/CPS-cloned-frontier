#include<config.h>
CPS_START_NAMESPACE
//--------------------------------------------------------------------
//  CVS keywords
//
//  $Source: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/src/util/dirac_op/d_op_mobius/noarch/mobius_m.C,v $
//  $State: Exp $
//
//--------------------------------------------------------------------
//------------------------------------------------------------------
// mobius_m.C
//
// mobius_m is the fermion matrix.  
// The in, out fields are defined on the checkerboard lattice
//
//------------------------------------------------------------------

CPS_END_NAMESPACE
#include<util/dwf.h>
#include<util/zmobius.h>
#include<util/gjp.h>
#include<util/vector.h>
#include<util/verbose.h>
#include<util/error.h>
#include<util/dirac_op.h>
#include<util/time_cps.h>

#include "blas-subs.h"

CPS_START_NAMESPACE


#include "mobius_mdag-orig.h"
//#include "mobius_mdag-sym1.h"
//#include "mobius_mdag-sym1-MIT.h"
#include "mobius_mdag-sym2.h"
//#include "mobius_mdag-sym2-MIT.h"


void  zmobius_mdag(Vector *out, 
		   Matrix *gauge_field, 
		   Vector *in, 
		   Float mass, 
		   Dwf *mobius_lib_arg)
{
  if(global_zmobius_pc==0) 
    zmobius_mdag_orig(out, 
		 gauge_field, 
		 in, 
		 mass, 
		 mobius_lib_arg);
  if(global_zmobius_pc==2) 
    zmobius_mdag_sym2(out, 
		 gauge_field, 
		 in, 
		 mass, 
		 mobius_lib_arg);

}




CPS_END_NAMESPACE
