/*!\file
  \brief   Global options for compiling the Colombia code:

  Generated automatically from config.h.in by configure procedure.

  $Id: config.h.in,v 1.29.28.1 2012-11-15 18:17:08 ckelly Exp $
*/
/* Global options for compiling the Columbia code:  
 * config.h.  Generated from config.h.in by configure.
 * 
 */

#ifndef INCLUDED_CONFIG_H_
#define INCLUDED_CONFIG_H_                  //!< Prevent multiple inclusion 

#include <conf.h>

#define NOARCH 0
#define QCDOC  1
#define QCDSP  2
#define BGL    3
#define BGP    4
#define BGQ    5


#define VERSION_MAJOR 5
#define VERSION_MINOR 3
#define VERSION_SUB 0
#define VERSION_STR "CPS_V5.3.0"

#define TARGET BGQ
#define PARALLEL 1

#define HAVE_BFM 1

// The configure procedure should make this unnecessary, but just in case...
#ifndef TARGET
#define TARGET NOARCH
#endif

#if TARGET == BGL
#define CPS_FLOAT_ALIGN __attribute__((aligned(16)))
#else
#define CPS_FLOAT_ALIGN
#endif

#define CWDPREFIX(A) A



/*! Explicit casting away of the const-ness  */
#define CAST_AWAY_CONST(x) ( const_cast<char*>(x) )

/*!  Precision in the global sum (undefined gives QCDSP behaviour). */
#define GLOBALSUM_TYPE double

#define CPS_END_NAMESPACE    }  
#define CPS_START_NAMESPACE  namespace cps {
#define USING_NAMESPACE_CPS  using namespace cps;
#define CPS_NAMESPACE	     cps


#if TARGET == BGL
/*! Data size for the MPI comms layer: */
#define COMMS_DATASIZE (sizeof(double))
/* Override printf to only print from only one processor */
#else
#define COMMS_DATASIZE (sizeof(double))
#endif

#undef UNIFORM_SEED_TESTING
#undef UNIFORM_SEED_NO_COMMS

/* ------------------------------------------------------------------*/

#endif /* INCLUDED_CONFIG_H_ */





