#include<config.h>
CPS_START_NAMESPACE 
//------------------------------------------------------------------
//
// alg_int_omelyan.C
//
// AlgIntOmelyan is derived from AlgIntAB, it is an implementation of
// the omelyan integrator using abstract operators.
// 
// To construct a QPQPQ integrator, the update to the coordinate
// must be the first argument, the momentum the second.
//
// To construct a PQPQP integrator, the update to the momentum
// must be the second argument, the coordinate the second.
//------------------------------------------------------------------

CPS_END_NAMESPACE
#include<math.h>
#include<alg/alg_hmd.h>
#include<util/lattice.h>
#include<util/vector.h>
#include<util/gjp.h>
#include<util/smalloc.h>
#include<util/verbose.h>
#include<util/error.h>
#include<util/time_cps.h>
#include<alg/alg_int.h>
#include<util/checksum.h>
CPS_START_NAMESPACE


AlgIntOmelyan::AlgIntOmelyan(AlgInt &A, AlgInt &B, 
			     IntABArg &arg_ab) 
  : AlgIntAB(A,B,arg_ab)
{

  int_type = INT_OMELYAN;
  A_calls = 3;
  B_calls = 2;
  lambda = ab_arg->lambda;
  
}

AlgIntOmelyan::~AlgIntOmelyan() {

}

void AlgIntOmelyan::evolve(Float dt, int steps) 
{
  char * fname = "evolve(Float, int)";

  TimeStamp::stamp("AlgIntOmelyan evolving with dt = %f for %d steps",dt,steps);
  
  step_cnt = 0;
  if (level == TOP_LEVEL_INTEGRATOR) CSM.SaveComment(step_cnt);

  TimeStamp::stamp_incr("AlgIntOmelyan evolving integrator A with dt = %f for %d steps",lambda*dt/(Float)A_steps, A_steps);
  A->evolve(lambda*dt/(Float)A_steps, A_steps);
  TimeStamp::decr_depth();

  for (int i=0; i<steps; i++) {
    if (level == TOP_LEVEL_INTEGRATOR) CSM.SaveComment(++step_cnt);
    
    TimeStamp::stamp_incr("AlgIntOmelyan step %d evolving integrator B with dt = %f for %d steps",i,dt/(2.0*(Float)B_steps), B_steps);
    B->evolve(dt/(2.0*(Float)B_steps), B_steps);
    TimeStamp::decr_depth();

    TimeStamp::stamp_incr("AlgIntOmelyan step %d evolving integrator A with dt = %f for %d steps",i,(1-2*lambda)*dt/(Float)A_steps, A_steps);
    A->evolve((1-2*lambda)*dt/(Float)A_steps, A_steps);
    TimeStamp::decr_depth();

    TimeStamp::stamp_incr("AlgIntOmelyan step %d evolving integrator B with dt = %f for %d steps",i,dt/(2.0*(Float)B_steps), B_steps);
    B->evolve(dt/(2.0*(Float)B_steps), B_steps);
    TimeStamp::decr_depth();

    if (i < steps-1){
      TimeStamp::stamp_incr("AlgIntOmelyan step %d evolving integrator A with dt = %f for %d steps",i,2.0*lambda*dt/(Float)A_steps, A_steps);
      A->evolve(2.0*lambda*dt/(Float)A_steps, A_steps);
      TimeStamp::decr_depth();
    }
    else{
      TimeStamp::stamp_incr("AlgIntOmelyan step %d evolving integrator A with dt = %f for %d steps",i,lambda*dt/(Float)A_steps, A_steps);
      A->evolve(lambda*dt/(Float)A_steps, A_steps);
      TimeStamp::decr_depth();
    }
  }
  if(GJP.Gparity()){
    TimeStamp::stamp_incr("AlgIntOmelyan copy-conjugating 2f G-parity lattice");
    /*C.Kelly 09/11:
     *For lowest level integrator, during evolution of momentum and gauge fields we can be more efficient by only updating the links and not their conjugate copies
     *(the links we pull across the boundary can be conjugated in place for very few, if any, extra flops by using alternate functions, eg. Trans rather than Dagger.)
     *However for higher level integrators involving fermion fields we need both the links and their conjugates stored, so do the copy-conjugation here.
     *Do this by calling a function which does nothing apart from on the alg_action_gauge instance, where it performs the copy-conjugation.
     */
    A->copyConjLattice();
    B->copyConjLattice();
    TimeStamp::decr_stamp("AlgIntOmelyan finished copy-conjugating 2f G-parity lattice");
  }

  if (level == TOP_LEVEL_INTEGRATOR) CSM.SaveComment(++step_cnt);
  TimeStamp::stamp("AlgIntOmelyan finished evolving with dt = %f for %d steps",dt,steps);
}

CPS_END_NAMESPACE
