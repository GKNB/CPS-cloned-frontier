#include<config.h>
CPS_START_NAMESPACE
//-------------------------------------------------------------------
/*!\file
  \brief  Definition of glb_sum_multi_dir routine.

  $Id: glb_sum_multi_dir.C,v 1.2 2004-01-13 20:39:06 chulwoo Exp $
*/
//--------------------------------------------------------------------
//  CVS keywords
//
//  $Author: chulwoo $
//  $Date: 2004-01-13 20:39:06 $
//  $Header: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/src/comms/qcdoc/glb_cpp/glb_sum_multi_dir.C,v 1.2 2004-01-13 20:39:06 chulwoo Exp $
//  $Id: glb_sum_multi_dir.C,v 1.2 2004-01-13 20:39:06 chulwoo Exp $
//  $Name: not supported by cvs2svn $
//  $Locker:  $
//  $Log: not supported by cvs2svn $
//  Revision 1.1.2.1.2.1  2003/11/06 20:22:20  cwj
//  *** empty log message ***
//
//  Revision 1.1.1.1  2003/11/04 05:05:03  chulwoo
//
//  starting again
//
//
//  Revision 1.1  2003/10/08 18:38:29  chulwoo
//  start from vanilla_comms
//  start from QCDSP comms
//
//  Revision 1.1.1.1  2003/09/18 22:30:43  chulwoo
//  Mike's files for single node QCDOC + Parallel transport
//  I added some hacks for PARALLEL without MPI_SCU
//  PARALLEL=2 set PARALLEL without MPI_SCU
//
//
//  Revision 1.2  2003/07/24 16:53:54  zs
//  Addition of documentation via doxygen:
//  doxygen-parsable comment blocks added to many source files;
//  New target in makefile and consequent alterations to configure.in;
//  New directories and files under the doc directory.
//
//  Revision 1.1.1.1  2003/06/22 13:34:47  mcneile
//  This is the cleaned up version of the Columbia Physics System.
//  The directory structure has been changed.
//  The include paths have been updated.
//
//
//  Revision 1.6  2001/08/16 12:54:02  anj
//  Some fixes follosin the float-> IFloat change, mostly of the (variable
//  anme) IFloat_p -> float_p type.  A few fixes to ensure the test
//  programs use the same level of verbosity throughout, and an update of
//  the regression.pl script to make it more useful. Anj
//
//  Revision 1.5  2001/08/16 10:49:48  anj
//  The float->Float changes in the previous version were unworkable on QCDSP.
//  To allow type-flexibility, all references to "float" have been
//  replaced with "IFloat".  This can be undone via a typedef for QCDSP
//  (where Float=rfloat), and on all other machines allows the use of
//  double or float in all cases (i.e. for both Float and IFloat).  The I
//  stands for Internal, as in "for internal use only". Anj
//
//  Revision 1.3  2001/07/03 17:00:48  anj
//
//  Multiple minor alterations to change some #include's from referring to
//  files relative to the top-level source directory to referring to files
//  relative to the source-file positions.  This alteration makes the code
//  backwards compatable with the make structure of QCDSP, although this
//  may have to be changed to a more usual form in the future. Anj.
//
//  Revision 1.2  2001/06/19 18:11:51  anj
//  Serious ANSIfication.  Plus, degenerate double64.h files removed.
//  Next version will contain the new nga/include/double64.h.  Also,
//  Makefile.gnutests has been modified to work properly, propagating the
//  choice of C++ compiler and flags all the way down the directory tree.
//  The mpi_scu code has been added under phys/nga, and partially
//  plumbed in.
//
//  Everything has newer dates, due to the way in which this first alteration was handled.
//
//  Anj.
//
//  Revision 1.2  2001/05/25 06:16:01  cvs
//  Added CVS keywords to phys_v4_0_0_preCVS
//
//  $RCSfile: glb_sum_multi_dir.C,v $
//  $Revision: 1.2 $
//  $Source: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/src/comms/qcdoc/glb_cpp/glb_sum_multi_dir.C,v $
//  $State: Exp $
//
//--------------------------------------------------------------------
//--------------------------------------------------------------
// glb_sum_dir
// sum over multiple IFloating point data with single precision
// Sum over all nodes along a direction
// (0,1,2,3,4) <-> (x,y,z,t,s)
//--------------------------------------------------------------
CPS_END_NAMESPACE
#include<comms/glb.h>
#include<comms/scu.h>
#include<util/gjp.h>
#include<comms/double64.h>
#include <comms/sysfunc.h>
CPS_START_NAMESPACE

#define MAX_NUM_WORDS 40  //set to larger than size of double precision 3x3 matrix
static Double64 transmit_buf[MAX_NUM_WORDS];
static Double64 receive_buf[MAX_NUM_WORDS];
static Double64 gsum_buf[MAX_NUM_WORDS];


//----------------------------------------------------------------------
/*!
  \param float_p The number to be summed.
  \param dir The direction in which to sum; one of {0, 1, 2, 3, 4},
  corresponding to {x, y, z, t, s}.
  \param len The number of floating point numbers in the vector.
  \post The vector pointed to by \a float_p is summed over all nodes along the
  \a dir direction, \e i.e. over each strip of nodes where the grid
  coordinates in all other directions are constant.
  That sum vector is written back to \a float_p, which is identical on all
  nodes in this strip.

  \ingroup comms
*/
//---------------------------------------------------------------------- 
void glb_sum_multi_dir(Float * float_p, int dir, int len)
{
  int NP[5] = {GJP.Xnodes(), 
	       GJP.Ynodes(), 
	       GJP.Znodes(), 
	       GJP.Tnodes(), 
	       GJP.Snodes()};

  int COOR[5] = {GJP.XnodeCoor(), 
		 GJP.YnodeCoor(), 
		 GJP.ZnodeCoor(), 
		 GJP.TnodeCoor(), 
		 GJP.SnodeCoor()}; 

  int j;
  
  int blocksize=2*len*sizeof(Float);

  // Sum along dir
  //--------------------------------------------------------------
  for(j=0; j<len; j++){
    *(gsum_buf+j) = *(float_p+j);
    *(transmit_buf+j) = *(gsum_buf+j);
  }


  int itmp;
  for (itmp = 1; itmp < NP[dir]; itmp++) {
    SCUDirArg send(transmit_buf, gjp_scu_dir[2*dir], SCU_SEND, blocksize);
    SCUDirArg rcv(receive_buf, gjp_scu_dir[2*dir+1], SCU_REC, blocksize);

    SCUTrans(&send);
    SCUTrans(&rcv);

    SCUTransComplete();
    for(j=0;j<len;j++){
      *(gsum_buf+j) += *(receive_buf+j);
      *(transmit_buf+j) = *(receive_buf+j);
    }
  }


  // Broadcast the result of node with dir coordinate == 0
  //--------------------------------------------------------------

  if(COOR[dir] != 0) {
    for(j=0;j<len;j++){
      *(gsum_buf+j) = 0;
    }
  }
  
  for(j=0; j<len; j++){
    *(transmit_buf+j) = *(gsum_buf+j);
  }

  for (itmp = 1; itmp < NP[dir]; itmp++) {
    SCUDirArg send(transmit_buf, gjp_scu_dir[2*dir], SCU_SEND, blocksize);
    SCUDirArg rcv(receive_buf, gjp_scu_dir[2*dir+1], SCU_REC, blocksize);

    SCUTrans(&send);
    SCUTrans(&rcv);

    SCUTransComplete();
    for(j=0;j<len;j++){
      *(gsum_buf+j) += *(receive_buf+j);
      *(transmit_buf+j) = *(receive_buf+j);
    }
  }
 
  for(j=0; j<len; j++){ 
   *(float_p+j) = *(gsum_buf+j);
  }
 
}


CPS_END_NAMESPACE