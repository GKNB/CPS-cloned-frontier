#ifdef USE_BFM

#include <util/lattice/eff_overlap.h>
#include <util/lattice/bfm_mixed_solver.h>
#include <util/timer.h>
#include <util/gjp.h>
#include <util/smalloc.h>

CPS_START_NAMESPACE

// Applies either Dov or Dov^-1
//
// How to apply the 4D overlap operator:
//
// 1. Promote the 4D input vector to a 5D vector, putting the left-handed part
//    at s=0 and the right-handed part at s=Ls-1
//
// 3. Apply D_DW(m)    
//
// 4. Apply D_DW(1)^-1          
//
// 5. Reduce the 5D vector to a 4D vector, taking the left-handed part from s=0
//    and the right-handed part from s=Ls-1
//
// We don't have to worry about issues with D_- here because 
//
// D_DW(1)^-1 D_DW(m) = (D_-^-1 D_DW(1))^-1 (D_-^-1 D_DW(m))
//
// (the D_-'s cancel in the product we compute).
int ApplyOverlap(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, Float pv_stop_rsd)
{
    const char* fname = "ApplyOverlap()";

    static Timer timer(fname);
    static std::map<Float, Timer*> timers;
    if (timers.count(mass) == 0) {
	char timer_mass_name[512];
	sprintf(timer_mass_name, "ApplyOverlap(mass=%0.4f)", mass);
	timers[mass] = new Timer(timer_mass_name);
    }
    timer.start(true);
    timers[mass]->start(true);

    Fermion_t vec_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    Fermion_t tmp_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };

    // vec_5d = 5D version of input vector
    bfm_d.cps_impexFermion_4d((Float *)in, vec_5d, Import);

    bfm_d.set_mass(mass);
    if (use_mixed_solver) bfm_f.set_mass(mass);
#pragma omp parallel
    {
	// tmp_5d = D_DW(m) vec_5d
	bfm_d.G5D_Munprec(vec_5d, tmp_5d, DaggerNo);
    }

    int iters;
    bfm_d.set_mass(1.0);
    if (use_mixed_solver) bfm_f.set_mass(1.0);
    bfm_d.residual = pv_stop_rsd;
    if (use_mixed_solver) bfm_f.residual = 1e-5;
#pragma omp parallel
    {
	// vec_5d = D_DW(1)^-1 tmp_5d = D_DW(1)^-1 D_DW(m) in
	bfm_d.set_zero(vec_5d[Even]);
	bfm_d.set_zero(vec_5d[Odd]);
	iters = use_mixed_solver ?
	    mixed_cg::threaded_cg_mixed_M(vec_5d, tmp_5d, bfm_d, bfm_f, 5) :
	    bfm_d.CGNE_M(vec_5d, tmp_5d);
    }

    // out = 4D version of vec_5d
    bfm_d.cps_impexFermion_4d((Float *)out, vec_5d, Export);

    bfm_d.freeFermion(vec_5d[Even]);
    bfm_d.freeFermion(vec_5d[Odd]);
    bfm_d.freeFermion(tmp_5d[Even]);
    bfm_d.freeFermion(tmp_5d[Odd]);

    VRB.Result("", fname, "Applied overlap operator for mass = %e, Ls = %d: PV iterations = %d\n", mass, bfm_d.Ls, iters);

    timers[mass]->stop(true);
    timer.stop(true);
    
    return iters;
}

int ApplyOverlapInverse(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, Float stop_rsd)
{
    const char* fname = "ApplyOverlapInverse()";

    static Timer timer(fname);
    static std::map<Float, Timer*> timers;
    if (timers.count(mass) == 0) {
	char timer_mass_name[512];
	sprintf(timer_mass_name, "ApplyOverlapInverse(mass=%0.4f)", mass);
	timers[mass] = new Timer(timer_mass_name);
    }
    timer.start(true);
    timers[mass]->start(true);

    Fermion_t vec_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    Fermion_t tmp_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };

    bfm_d.cps_impexFermion_4d((Float *)in, vec_5d, Import);

    bfm_d.set_mass(1.0);
    if (use_mixed_solver) bfm_f.set_mass(1.0);
#pragma omp parallel
    {
	bfm_d.G5D_Munprec(vec_5d, tmp_5d, DaggerNo);
    }

    int iters;
    bfm_d.set_mass(mass);
    if (use_mixed_solver) bfm_f.set_mass(mass);
    bfm_d.residual = stop_rsd;
    if (use_mixed_solver) bfm_f.residual = 1e-5;
#pragma omp parallel
    {
	// TODO: optionally make use of initial guess
	bfm_d.set_zero(vec_5d[Even]);
	bfm_d.set_zero(vec_5d[Odd]);
	iters = use_mixed_solver ?
	    mixed_cg::threaded_cg_mixed_M(vec_5d, tmp_5d, bfm_d, bfm_f, 5) :
	    bfm_d.CGNE_M(vec_5d, tmp_5d);
    }

    bfm_d.cps_impexFermion_4d((Float *)out, vec_5d, Export);

    bfm_d.freeFermion(vec_5d[Even]);
    bfm_d.freeFermion(vec_5d[Odd]);
    bfm_d.freeFermion(tmp_5d[Even]);
    bfm_d.freeFermion(tmp_5d[Odd]);

    timers[mass]->stop(true);
    timer.stop(true);

    return iters;
}

int ApplyOverlapDag(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, Float pv_stop_rsd)
{
    const char* fname = "ApplyOverlapDag()";

    static Timer timer(fname);
    static std::map<Float, Timer*> timers;
    if (timers.count(mass) == 0) {
	char timer_mass_name[512];
	sprintf(timer_mass_name, "ApplyOverlapDag(mass=%0.4f)", mass);
	timers[mass] = new Timer(timer_mass_name);
    }
    timer.start(true);
    timers[mass]->start(true);


    Fermion_t vec_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    Fermion_t tmp_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };

    // vec_5d = 5D version of input vector
    bfm_d.cps_impexFermion_4d((Float *)in, vec_5d, Import);

    bfm_d.set_mass(1.0);
    if (use_mixed_solver) bfm_f.set_mass(1.0);
    bfm_d.residual = pv_stop_rsd;
    if (use_mixed_solver) bfm_f.residual = 1e-5;
    int iters;
#pragma omp parallel
    {
	// tmp_5d = D_DW(1)^dag^-1 vec_5d
	bfm_d.set_zero(tmp_5d[Even]);
	bfm_d.set_zero(tmp_5d[Odd]);
	iters = use_mixed_solver ?
	    mixed_cg::threaded_cg_mixed_Mdag(tmp_5d, vec_5d, bfm_d, bfm_f, 5) :
	    bfm_d.CGNE_Mdag(tmp_5d, vec_5d);
    }

    bfm_d.set_mass(mass);
    if (use_mixed_solver) bfm_f.set_mass(mass);
#pragma omp parallel
    {
	// vec_5d = D_DW(m)^-1^dag tmp_5d = D_DW(m)^-1^dag D_DW(1)^dag in
	bfm_d.G5D_Munprec(tmp_5d, vec_5d, DaggerYes);
    }

    // out = 4D version of vec_5d
    bfm_d.cps_impexFermion_4d((Float *)out, vec_5d, Export);

    bfm_d.freeFermion(vec_5d[Even]);
    bfm_d.freeFermion(vec_5d[Odd]);
    bfm_d.freeFermion(tmp_5d[Even]);
    bfm_d.freeFermion(tmp_5d[Odd]);

    VRB.Result("", fname, "Applied overlap-dagger operator for mass = %e, Ls = %d: PV iterations = %d\n", mass, bfm_d.Ls, iters);

    timers[mass]->stop(true);
    timer.stop(true);

    return iters;
}

int ApplyOverlapDagInverse(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, Float stop_rsd)
{
    const char* fname = "ApplyOverlapDagInverse()";

    static Timer timer(fname);
    static std::map<Float, Timer*> timers;
    if (timers.count(mass) == 0) {
	char timer_mass_name[512];
	sprintf(timer_mass_name, "ApplyOverlapDagInverse(mass=%0.4f)", mass);
	timers[mass] = new Timer(timer_mass_name);
    }
    timer.start(true);
    timers[mass]->start(true);

    Fermion_t vec_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    Fermion_t tmp_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };

    bfm_d.cps_impexFermion_4d((Float *)in, vec_5d, Import);

    bfm_d.set_mass(mass);
    if (use_mixed_solver) bfm_f.set_mass(mass);
    bfm_d.residual = stop_rsd;
    if (use_mixed_solver) bfm_f.residual = 1e-5;
    int iters;
#pragma omp parallel
    {
	// TODO: optionally make use of initial guess
	bfm_d.set_zero(tmp_5d[Even]);
	bfm_d.set_zero(tmp_5d[Odd]);
	iters = use_mixed_solver ?
	    mixed_cg::threaded_cg_mixed_Mdag(tmp_5d, vec_5d, bfm_d, bfm_f, 5) :
	    bfm_d.CGNE_Mdag(tmp_5d, vec_5d);
    }

    bfm_d.set_mass(1.0);
    if (use_mixed_solver) bfm_f.set_mass(1.0);
#pragma omp parallel
    {
	bfm_d.G5D_Munprec(tmp_5d, vec_5d, DaggerYes);
    }

    bfm_d.cps_impexFermion_4d((Float *)out, vec_5d, Export);

    bfm_d.freeFermion(vec_5d[Even]);
    bfm_d.freeFermion(vec_5d[Odd]);
    bfm_d.freeFermion(tmp_5d[Even]);
    bfm_d.freeFermion(tmp_5d[Odd]);

    timers[mass]->stop(true);
    timer.stop(true);

    return iters;
}


// Like ApplyOverlapInverse, but uses the initial value of out
// as a guess. This requires an extra inversion of D_DW(1)
int ApplyOverlapInverseGuess(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, Float stop_rsd)
{
    int iters = 0;

    Fermion_t out_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    Fermion_t Dm_Pout[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    Fermion_t in_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    Fermion_t rhs_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };

    out->VecTimesEquFloat()

    // out_5d = P out
    bfm_d.cps_impexFermion_5d((Float *)out, out_5d, Import);

    // in_5d = P in
    bfm_d.cps_impexFermion_4d((Float *)in, in_5d, Import);

    // Contruct initial guess for 5D solve

    // Dm_Pout = -D_DW(m) P out
    bfm_d.set_mass(mass);
    if (use_mixed_solver) bfm_f.set_mass(mass);
#pragma omp parallel
    {
	bfm_d.scale(out_5d, -1.0);
	bfm_d.G5D_Munprec(out_5d, Dm_Pout, DaggerNo);
    }
    
    bfm_d.set_mass(1.0);
    if (use_mixed_solver) bfm_f.set_mass(1.0);
#pragma omp parallel 
    {
	// out_5d = -D_DW(1)^{-1} D_DW(m) P out
	bfm_d.set_zero(out_5d[Even]);
	bfm_d.set_zero(out_5d[Odd]);
	iters += use_mixed_solver ?
	    mixed_cg::threaded_cg_mixed_M(Dm_Pout, out_5d, bfm_d, bfm_f, 5) :
	    bfm_d.CGNE_M(Dm_Pout, out_5d);

	// Construct RHS for 5D solve
	// rhs_5d = D_DW(1) P in
	bfm_d.G5D_Munprec(in_5d, rhs_5d, DaggerNo);
    }

    // get the zero component of the initial guess.
    // We do this by again importing out into out_5d. out_5d now has all
    // the other components of the initial guess, so we set the "prezero"
    // argument to false and just import to the zero component.
    bfm_d.cps_impexFermion_4d((Float *)out, out_5d, Import, false); 

    // Do the main 5D solve
    bfm_d.set_mass(mass);
    if (use_mixed_solver) bfm_f.set_mass(mass);
#pragma omp parallel
    {
	iters += use_mixed_solver ?
	    mixed_cg::threaded_cg_mixed_M(out_5d, rhs_5d, bfm_d, bfm_f, 5) :
	    bfm_d.CGNE_M(out_5d, rhs_5d);
    }

    bfm_d.cps_impexFermion_4d((Float *)out, out_5d, Export);

    bfm_d.freeFermion(out_5d[Even]);
    bfm_d.freeFermion(out_5d[Odd]);
    bfm_d.freeFermion(Dm_Pout[Even]);
    bfm_d.freeFermion(Dm_Pout[Odd]);
    bfm_d.freeFermion(in_5d[Even]);
    bfm_d.freeFermion(in_5d[Odd]);
    bfm_d.freeFermion(rhs_5d[Even]);
    bfm_d.freeFermion(rhs_5d[Odd]);

    return iters;
}



int InvertOverlapDefectCorrection(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, bfmarg cheap_approx, Matrix *gauge_field, int num_iters,
    Float cheap_solve_stop_rsd, Float exact_solve_stop_rsd)
{
    const char* fname = "InvertOverlapDefectCorrection()";

    VRB.Result("", fname, "Start!\n");

    bfm_d.comm_end();

    bfm_evo<double> bfm_cheap_d;
    bfm_evo<float> bfm_cheap_f;

    bfm_cheap_d.init(cheap_approx);
    bfm_cheap_d.cps_importGauge((Float *)gauge_field);
    if (use_mixed_solver) {
	bfm_cheap_d.comm_end();
	bfm_cheap_f.init(cheap_approx);
	bfm_cheap_f.cps_importGauge((Float *)gauge_field);
	bfm_cheap_f.comm_end();
	bfm_cheap_d.comm_init();
    }

    int iters = 0;

    int f_size = 24 * GJP.VolNodeSites();
    Vector *approx_sol = (Vector*)smalloc(f_size * sizeof(Float), "approx_sol", fname, "");
    Vector *residual = (Vector*)smalloc(f_size * sizeof(Float), "residual", fname, "");
    Vector *tmp = (Vector*)smalloc(f_size * sizeof(Float), "tmp", fname, "");

    // TODO: try to use initial guess
    out->VecZero(f_size);
    residual->CopyVec(in, f_size);
    Float norm2_residual = residual->NormSqGlbSum(f_size);

    VRB.Result("", fname, "Starting norm2_residual = %e\n", norm2_residual);

    for (int iter = 0; iter < num_iters; iter++) {
	// Approximately invert Dov on residual
	// approx_sol = Dov'^{-1} res
	VRB.Result("", fname, "Doing cheap solve (iter = %d of %d)\n", iter, num_iters);
	iters += ApplyOverlapInverse(bfm_cheap_d, bfm_cheap_f, use_mixed_solver,
	    approx_sol, residual, cheap_approx.mass, cheap_solve_stop_rsd);

	bfm_cheap_d.comm_end();
	bfm_d.comm_init();

	out->VecAddEquVec(approx_sol, f_size);

	// compute new true residual

	// tmp = Dov out
	iters += ApplyOverlap(bfm_d, bfm_f, use_mixed_solver,
	    tmp, out, mass, exact_solve_stop_rsd);

	bfm_d.comm_end();
	bfm_cheap_d.comm_init();

	// residual = in - Dov out
	residual->FTimesV1MinusV2(1.0, in, tmp, f_size);

	norm2_residual = residual->NormSqGlbSum(f_size);
	VRB.Result("", fname, "After iter #%d, norm2_residual = %e\n", iter, norm2_residual);
    }

    bfm_cheap_d.end();
    if (use_mixed_solver) {
	bfm_cheap_f.end();
    }
    bfm_d.comm_init();

    // Final cleanup solve. We use the approximate solution
    // we have computed as the initial guess.
    VRB.Result("", fname, "Doing final cleanup solve\n");
    iters += ApplyOverlapInverseGuess(bfm_d, bfm_f, use_mixed_solver,
	out, in, mass, exact_solve_stop_rsd);

    sfree(approx_sol, "approx_sol", fname, "");
    sfree(residual, "residual", fname, "");
    sfree(tmp, "tmp", fname, "");

    VRB.Result("", fname, "Done!\n");
    return iters;
}


CPS_END_NAMESPACE

#endif