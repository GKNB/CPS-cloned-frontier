#include<config.h>
CPS_START_NAMESPACE
//--------------------------------------------------------------------
//  CVS keywords
//
//  $Author: mcneile $
//  $Date: 2003-06-22 13:34:52 $
//  $Header: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/include/alg/quark_prop_s.h,v 1.1.1.1 2003-06-22 13:34:52 mcneile Exp $
//  $Id: quark_prop_s.h,v 1.1.1.1 2003-06-22 13:34:52 mcneile Exp $
//  $Name: not supported by cvs2svn $
//  $Locker:  $
//  $Log: not supported by cvs2svn $
//  Revision 1.5  2002/03/07 11:18:25  anj
//  Some mistaken and lazy friend declarations have been fixed.  We now use the
//  proper full syntax 'friend class Fred;', not just 'friend Fred;'. Anj.
//
//  Revision 1.4  2001/08/16 10:49:41  anj
//  The float->Float changes in the previous version were unworkable on QCDSP.
//  To allow type-flexibility, all references to "float" have been
//  replaced with "IFloat".  This can be undone via a typedef for QCDSP
//  (where Float=rfloat), and on all other machines allows the use of
//  double or float in all cases (i.e. for both Float and IFloat).  The I
//  stands for Internal, as in "for internal use only". Anj
//
//  Revision 1.2  2001/06/19 18:11:31  anj
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
//  Revision 1.2  2001/05/25 06:16:00  cvs
//  Added CVS keywords to phys_v4_0_0_preCVS
//
//  $RCSfile: quark_prop_s.h,v $
//  $Revision: 1.1.1.1 $
//  $Source: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/include/alg/quark_prop_s.h,v $
//  $State: Exp $
//
//--------------------------------------------------------------------
#ifndef INCLUDED_QUARK_PROP_S_H
#define INCLUDED_QUARK_PROP_S_H

CPS_END_NAMESPACE
#include<util/lattice.h>
#include<util/vector.h>
#include<alg/cg_arg.h>
#include<alg/s_spect_arg.h>
CPS_START_NAMESPACE

//====================================================================
// quark propagators, ie. green function of fermions of 
// in F.R. of SU(3), obtained by solving the following equation:
//
//		(D+m)G(x,t;x',t') = delta(x-x') delta(t-t')
//
// where (x',t') is the source location. G(x,t; x',t') is a 
// 3x3 color matrix on each lattice site. As a result one has to 
// use CG three times to solve for each column of G(x,t; x',t'), 
// with different color components of the source set to 1 every time.  
// We store three columns of the propagators on even sites as three 
// Float arrays, and another three columns for odd sites 
//====================================================================
//
// FORWARD CLASS DECLARATION
//
class MesonPropS;
class NucleonPropS;
class QuarkPropS;
class QuarkPropSMng;


class QuarkPropS {

   static char cname[];

   static Float *qSrc;
	// source vector: |even sites + odd sites|

   static int prop_count;    
        // count number of quark propagators in current scope

   // NON-STATICS -- for each instance

   int   qid;
	// quark ID, from StagQuarkArg
	
   Float **prop;	
	// pointers to 3 columns of G(x,t;x',t') 
	
   StagQuarkArg& qarg;
  	// all arguments for quark propagator

   Lattice& lat;

private:

   int X_OFFSET(const int *x);

   int g_offset(int *s, int *size)
   { return s[0] + s[1] * size[0] + s[2] * size[0] * size[1]
            + s[3] * size[0] * size[1] * size[2]; }

   void coulomb(IFloat *x, Matrix **g, int dir);

public:

   //---------------------------------------------
   // CTOR
   //---------------------------------------------
   QuarkPropS(Lattice &lattice, StagQuarkArg& arg);

   //---------------------------------------------
   // deallocate  source if prop_count == 0
   //---------------------------------------------
  ~QuarkPropS();

   //---------------------------------------------
   // allocate memory for source, propagator
   //---------------------------------------------
   void setupQuarkPropS();

   //-----------------------------------------------------------
   // deallocate memory for propagator 
   //-----------------------------------------------------------
   static void destroyQuarkPropS(int id = 0);

   //---------------------------------------------
   // set POINT source for a COLOR
   //---------------------------------------------
   void setPntSrc(const int site[], int color);		

   //-----------------------------------------------------------
   // set Z or 2Z sources for a Color on the Wall 
   //-----------------------------------------------------------
   void setWallSrc( Matrix **gm, StagQuarkSrc& qs, int color); 

   //-----------------------------------------------------------
   // compute quark propagator: all info is in StagQuarkArg 
   //-----------------------------------------------------------
   void getQuarkPropS(char *results);

   //-----------------------------------------------------------
   // ACCESSORS: for debugging, not implemented
   //-----------------------------------------------------------
   Float *AccessQuarkPropS(int color, int *site ) const;

   friend class QuarkPropSMng;
};


//==============================================================
// This class manages the quark propagator memory globally
//==============================================================
enum {MAXNUMQP = 100};
	// quark propagator ID must be 0 < ID < MAXNUMQP

class QuarkPropSMng {

   static char cname[];

   static int isInit;
	// Initialization flag

   static Float ***qtab;
	// record the pointer to propagators
    	// 0 if not set up or destroyed

   static int* slice;

private: 
   static void release(int id);

public:

   QuarkPropSMng();
   ~QuarkPropSMng(); 
   
   //-----------------------------------------------------------
   // register a quark propagator in this manager class
   //-----------------------------------------------------------
   static void qadd(QuarkPropS* qp);

   //-----------------------------------------------------------
   // destroy all quark propagators by default;
   // destroy a specific quark propagator given an qid
   //-----------------------------------------------------------
   static void destroyQuarkPropS(int qid = -1);

   //-----------------------------------------------------------
   // ACCESSOR
   //-----------------------------------------------------------
   static Float **prop(int id) { return qtab[id]; }
   static int srcSlice(int id) { return slice[id]; }

};


#endif
CPS_END_NAMESPACE
