#include<config.h>
CPS_START_NAMESPACE
/*------------------------------------------------------------------*/
/*!\file
  \brief  Definitions of the HmdArg structure.

  $Id: hmd_arg.x,v 1.2 2004-12-11 20:57:34 chulwoo Exp $
*/
/*------------------------------------------------------------------*/


#ifndef INCLUDED_HMD_ARG_H
#define INCLUDED_HMD_ARG_H

CPS_END_NAMESPACE
#include <util/vector.h>
CPS_START_NAMESPACE


enum ReunitarizeType { REUNITARIZE_NO    = 0,
		       REUNITARIZE_YES   = 1 };

enum MetropolisType { METROPOLIS_NO    = 0,
		      METROPOLIS_YES   = 1 };



#define MAX_HMD_MASSES  4       /*!< The maximum number of dynamical masses.*/

#define MAX_RAT_DEGREE  20          /* The maximum degree of the rational
				       approximation. */

/*! A structure holding the parameters relevant to the HMD algorithms.*/
/*!
  Not all parameters are relevant to all algorithms. An algorithm should
  safely ignore  parameters irrelevant to it.
 \ingroup algargs 
*/

struct HmdArg {


    int n_frm_masses;                /*!< The number of dynamical fermion
				      masses. This is not necessarily the
				      number of flavours: At each mass the
				      number of flavours is given by
				      Lattice::ExactFlavors. */
				
    Float frm_mass[MAX_HMD_MASSES];   /*!< The array of dynamical fermions
					masses . */

    int frm_flavors[MAX_HMD_MASSES];  /*!<
					The flavour number of each dynamical
					mass:
					This is relevant to the R
					algorithm only. Bosonic fields
                                    are simulated by setting 
                                    a negative flavor number. */

  int n_bsn_masses;                 /*!< The number of dynamical boson 
				    masses. This is not necessarily the
				      number of flavours: At each mass the
				      number of flavours is given by
				      Lattice::ExactFlavors. This is relevant
				      to the Phi algorithm only.*/
				
  Float bsn_mass[MAX_HMD_MASSES];   /* The array of dynamical bosons masses.
				       This is relevant to the Phi algorithm
				       only.*/

  int max_num_iter[MAX_HMD_MASSES]; /*!< The maximum number of iterations
				    of the solver for each mass 
				    of dynamical fermions/bosons */
                                   

  Float stop_rsd[MAX_HMD_MASSES];   /*!< The target residual for the solver
				      stopping condition for each mass 
				      of dynamical fermions/bosons */

  int steps_per_traj;		    /*!<
				      For the R algorithm, this is
				      the number of steps per trajectory. For
				      the HMC/phi algorithm this is
				      <em> one less than </em> the number of
				      steps per trajectory.
				    */
  
  Float step_size;		    /*!< The molecular dynamics step size. */
  
  MetropolisType metropolis;        /*!< Whether to do the metropolis
				      accept/reject step.
				      This is relevant to the Phi
				      algorithm only. */ 
  
  ReunitarizeType reunitarize;      /*!< Whether to reunitarize at the end of
				      the trajectory before the metropolis
				      step. */
  
  /*!< The parameters for the force rational approximations*/
  int FRatDeg[MAX_HMD_MASSES];
  Float FRatNorm[MAX_HMD_MASSES];
  Float FRatRes[MAX_HMD_MASSES][MAX_RAT_DEGREE];
  Float FRatPole[MAX_HMD_MASSES][MAX_RAT_DEGREE];

  /*!< The parameters for the action rational approximations*/
  int SRatDeg[MAX_HMD_MASSES];
  Float SRatNorm[MAX_HMD_MASSES];
  Float SRatRes[MAX_HMD_MASSES][MAX_RAT_DEGREE];
  Float SRatPole[MAX_HMD_MASSES][MAX_RAT_DEGREE];

  /*!< The parameters for the inverse action rational approximations*/
  Float SIRatNorm[MAX_HMD_MASSES];
  Float SIRatRes[MAX_HMD_MASSES][MAX_RAT_DEGREE];
  Float SIRatPole[MAX_HMD_MASSES][MAX_RAT_DEGREE];

  /*!< The location of the smallest polar shift in the rational approximations*/
  int isz;

  /*!< The Sexton-Weigngarten factor*/
  int sw;

};


#endif /* !INCLUDED_HMD_ARG_H */



CPS_END_NAMESPACE