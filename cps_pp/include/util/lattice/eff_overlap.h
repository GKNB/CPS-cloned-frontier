#ifndef INCLUDED_EFF_OVERLAP_H__
#define INCLUDED_EFF_OVERLAP_H__

#ifdef USE_BFM

#include <util/lattice/bfm_evo.h>
#include <util/vector.h>

CPS_START_NAMESPACE

int ApplyOverlap(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, Float pv_stop_rsd);

int ApplyOverlapInverse(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, Float stop_rsd);

int ApplyOverlapDag(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, Float pv_stop_rsd);

int ApplyOverlapDagInverse(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, Float stop_rsd);


int ApplyOverlapInverseGuess(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, Float stop_rsd);

int InvertOverlapDefectCorrection(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, bfmarg cheap_approx, Matrix *gauge_field, 
    int num_iters, Float cheap_solve_stop_rsd, Float exact_solve_stop_rsd);

CPS_END_NAMESPACE

#endif 

#endif