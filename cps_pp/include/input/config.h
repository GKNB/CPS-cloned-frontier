/*!\file
  \brief   Global options for compiling the Colombia code:

  Generated automatically from config.h.in by configure procedure.

<<<<<<< HEAD
  $Id: config.h.in,v 1.29 2012/03/26 13:50:11 chulwoo Exp $
=======
  $Id: config.h.in,v 1.29.28.1 2012-11-15 18:17:08 ckelly Exp $
>>>>>>> f48da4751672751e97d244b8e3242b5a1fdb923d
*/
/* Global options for compiling the Columbia code:  
 * config.h.  Generated from config.h.in by configure.
 * 
 *--------------------------------------------------------------------
 *  CVS keywords
 *
<<<<<<< HEAD
 *  $Author: chulwoo $
 *  $Date: 2012/03/26 13:50:11 $
 *  $Header: /space/cvs/cps/cps++/config.h.in,v 1.29 2012/03/26 13:50:11 chulwoo Exp $
 *  $Id: config.h.in,v 1.29 2012/03/26 13:50:11 chulwoo Exp $
 *  $Name: v5_0_16_hantao_io_test_v7 $
 *  $Locker:  $
 *  $RCSfile: config.h.in,v $
 *  $Revision: 1.29 $
 *  $Source: /space/cvs/cps/cps++/config.h.in,v $
=======
 *  $Author: ckelly $
 *  $Date: 2012-11-15 18:17:08 $
 *  $Header: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/config.h.in,v 1.29.28.1 2012-11-15 18:17:08 ckelly Exp $
 *  $Id: config.h.in,v 1.29.28.1 2012-11-15 18:17:08 ckelly Exp $
 *  $Name: not supported by cvs2svn $
 *  $Locker:  $
 *  $RCSfile: config.h.in,v $
 *  $Revision: 1.29.28.1 $
 *  $Source: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/config.h.in,v $
>>>>>>> f48da4751672751e97d244b8e3242b5a1fdb923d
 *  $State: Exp $
 */
/* ------------------------------------------------------------------*/

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
#define VERSION_MINOR 2
#define VERSION_SUB 1
#define VERSION_STR "CPS_V5.2.1"

#define TARGET NOARCH
#define PARALLEL 1

#undef HAVE_BFM

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


#if TARGET == cpsMPI
/*! Data size for the MPI comms layer: */
#define COMMS_DATASIZE (sizeof(float))
/* Override printf to only print from only one processor */
#include<util/qcdio_qprintf.h>
#elif TARGET == BGL
/*! Data size for the MPI comms layer: */
#define COMMS_DATASIZE (sizeof(double))
/* Override printf to only print from only one processor */
#include<util/qcdio_qprintf.h>
#else
#define COMMS_DATASIZE (sizeof(double))
#endif

#undef UNIFORM_SEED_TESTING
#undef UNIFORM_SEED_NO_COMMS

/* ------------------------------------------------------------------*/

#endif /* INCLUDED_CONFIG_H_ */





