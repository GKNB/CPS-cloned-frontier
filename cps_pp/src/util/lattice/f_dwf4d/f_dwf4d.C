#ifdef USE_BFM

#include <util/lattice/f_dwf4d.h>
#include <util/lattice/fforce_wilson_type.h>
#include <util/timer.h>

#include <util/lattice/bfm_mixed_solver.h>

#include <omp.h>

CPS_START_NAMESPACE

std::map<Float, DWFParams> Fdwf4d::paramMap;

bool Fdwf4d::use_mixed_solver = true;

Float Fdwf4d::pauli_villars_resid = 1e-12;

Timer Fdwf4d::inv_m_timer("mass=m inversions");
Timer Fdwf4d::inv_pv_timer("mass=pv inversions");

template<class T>
static void SetMass(bfm_evo<T> &bfm, double mass)
{
    VRB.Result("", "SetMass", "bfm.mass = %e, mass = %e\n", bfm.mass, mass);
    if (bfm.mass != mass) {
	bfm.mass = mass;
	bfm.GeneralisedFiveDimEnd();
	bfm.GeneralisedFiveDimInit();
    }
}

// Initialize QDP++ before using this class.
Fdwf4d::Fdwf4d(void)
    :cname("Fdwf4d")
{
    const char *fname = "Fdwf4d()";
    VRB.Func(cname, fname);

    if (GJP.Snodes() != 1) ERR.NotImplemented(cname, fname);
    if (sizeof(Float) == sizeof(float)) ERR.NotImplemented(cname, fname);

    dwfParams.mass = 1.0;
    dwfParams.M5 = 1.8;
    dwfParams.Ls = 8;
    dwfParams.mobius_scale = 1.0;
    InitBfmFromDWFParams(bfm_d, dwfParams, this->GaugeField());
    if (use_mixed_solver) {
	bfm_d.comm_end();
	InitBfmFromDWFParams(bfm_f, dwfParams, this->GaugeField());
	bfm_f.comm_end();
	bfm_d.comm_init();
    }
}

Fdwf4d::~Fdwf4d(void)
{
    const char *fname = "~Fdwf4d()";
    VRB.Func(cname,fname);

    bfm_d.end();
    if (use_mixed_solver) {
	bfm_f.end();
    }
}

void Fdwf4d::SetDWFParams(Float mass)
{
    const char* fname = "SetDwfParams(F)";

    if (paramMap.count(mass) == 0) ERR.General(cname, fname, "No DWFParams mapped for mass = %e\n", mass);
    dwfParams = paramMap[mass];

    VRB.Result(cname, fname, "Set DWF params from key mass = %e: dwfParams: mass = %e, M5 = %e, Ls = %d, mobius_scale = %e\n",
	mass, dwfParams.mass, dwfParams.M5, dwfParams.Ls, dwfParams.mobius_scale);

    bfm_d.end();
    InitBfmFromDWFParams(bfm_d, dwfParams, this->GaugeField());
    if (use_mixed_solver) {
	bfm_d.comm_end();
	bfm_f.end();
	InitBfmFromDWFParams(bfm_f, dwfParams, this->GaugeField());
	bfm_f.comm_end();
	bfm_d.comm_init();
    }
}

// Does phi = Dov * frm1  or  phi = Dov^dag * frm1
// Returns (frm1, frm1)
Float Fdwf4d::SetPhi(Vector *phi, Vector *frm1, Vector *frm2, Float mass, DagType dag)
{
    const char *fname = "SetPhi(V*,V*,V*,F)";
    static Timer time(cname, fname);
    time.start(true);

    if (phi == 0) ERR.Pointer(cname, fname, "phi");
    if (frm1 == 0) ERR.Pointer(cname, fname, "frm1");

    SetDWFParams(mass);

    if (dag == DAG_NO) ApplyOverlap(bfm_d, bfm_f, use_mixed_solver, phi, frm1, dwfParams.mass, false, pauli_villars_resid); // Use tight stopping condition since we are only inverting D_DW(1)
    else ApplyOverlapDag(bfm_d, bfm_f, use_mixed_solver, phi, frm1, dwfParams.mass, false, pauli_villars_resid);

    Float ret = FhamiltonNode(frm1, frm1);
    time.stop(true);
    return ret;
}

// Does f_out = (M^dag M)^-1 f_in. Returns iteration count.
int Fdwf4d::FmatEvlInv(Vector *f_out, Vector *f_in, CgArg *cg_arg, Float *true_res, CnvFrmType cnv_frm)
{
    const char* fname = "FmatEvlInv()";

    static Timer timer(cname, fname);
    static std::map<Float, Timer*> timers;
    if (timers.count(cg_arg->mass) == 0) {
	char timer_mass_name[512];
	sprintf(timer_mass_name, "FmatEvlInv(mass=%0.4f)", cg_arg->mass);
	timers[cg_arg->mass] = new Timer(cname, timer_mass_name);
    }
    timer.start(true);
    timers[cg_arg->mass]->start(true);

    SetDWFParams(cg_arg->mass);

    Vector *tmp = (Vector*)smalloc(this->FvecSize() * sizeof(Float), "tmp", fname, cname);

    int iters = 0;

    //tmp = Dov^dag^-1 f_in
    iters += ApplyOverlapDag(bfm_d, bfm_f, use_mixed_solver, tmp, f_in, dwfParams.mass, true, cg_arg->stop_rsd);

    // f_out = Dov^-1 tmp = Dov^-1 Dov^dag^-1 f_in
    iters += ApplyOverlap(bfm_d, bfm_f, use_mixed_solver, f_out, tmp, dwfParams.mass, true, cg_arg->stop_rsd);

    sfree(tmp, "tmp", fname, cname);

    // TODO: calculate true residual
    if (true_res != NULL) *true_res = -1.0;

    timers[cg_arg->mass]->stop(true); 
    timer.stop(true);
    return iters;
}

int Fdwf4d::FmatEvlInv(Vector *f_out, Vector *f_in, CgArg *cg_arg, CnvFrmType cnv_frm)
{
    FmatEvlInv(f_out, f_in, cg_arg, NULL, cnv_frm);
}

int Fdwf4d::FeigSolv(Vector **f_eigenv, Float *lambda, Float *chirality, int *valid_eig,
    Float **hsum, EigArg *eig_arg, CnvFrmType cnv_frm)
{
    const char* fname = "FeigSolv()";
    ERR.NotImplemented(cname, fname);
}

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
    Vector *out, Vector *in, Float mass, bool invert, Float residual)
{
    const char* fname = "ApplyOverlap()";

    static Timer timer(fname);
    static std::map<Float, Timer*> timers;
    Float key = invert ? -mass : mass;
    if (timers.count(key) == 0) {
	char timer_mass_name[512];
	sprintf(timer_mass_name, "ApplyOverlap(mass=%0.4f)", key);
	timers[key] = new Timer(timer_mass_name);
    }
    timer.start(true);
    timers[key]->start(true);

    Fermion_t vec_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    Fermion_t tmp_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    
    // vec_5d = 5D version of input vector
    bfm_d.cps_impexFermion_4d((Float *)in, vec_5d, Import);

    SetMass(bfm_d, invert ? 1.0 : mass);
    if(use_mixed_solver) SetMass(bfm_f, invert ? 1.0 : mass);
#pragma omp parallel
    {
	// tmp_5d = D_DW(m) vec_5d
	bfm_d.G5D_Munprec(vec_5d, tmp_5d, DaggerNo);
    }

    int iters;
    SetMass(bfm_d, invert ? mass : 1.0);
    if (use_mixed_solver) SetMass(bfm_f, invert ? mass : 1.0);
    bfm_d.residual = residual;
    if (use_mixed_solver) bfm_f.residual = 1e-5;
#pragma omp parallel
    {
	// vec_5d = D_DW(1)^-1 tmp_5d = D_DW(1)^-1 D_DW(m) in
	bfm_d.set_zero(vec_5d[Even]);
	bfm_d.set_zero(vec_5d[Odd]);
	// TODO: MADWF inversion
	if(!omp_get_thread_num()) (bfm_d.mass == 1.0 ? Fdwf4d::inv_pv_timer : Fdwf4d::inv_m_timer).start(true);
	//iters = bfm.CGNE_M(vec_5d, tmp_5d);
	iters = use_mixed_solver ?
	    mixed_cg::threaded_cg_mixed_M(vec_5d, tmp_5d, bfm_d, bfm_f, 5) :
	    bfm_d.CGNE_M(vec_5d, tmp_5d);
	if (!omp_get_thread_num()) (bfm_d.mass == 1.0 ? Fdwf4d::inv_pv_timer : Fdwf4d::inv_m_timer).stop(true);
    }

    // out = 4D version of vec_5d
    bfm_d.cps_impexFermion_4d((Float *)out, vec_5d, Export);

    bfm_d.freeFermion(vec_5d[Even]);
    bfm_d.freeFermion(vec_5d[Odd]);
    bfm_d.freeFermion(tmp_5d[Even]);
    bfm_d.freeFermion(tmp_5d[Odd]);

    timers[key]->stop(true);
    timer.stop(true);

    if (!invert) {
	VRB.Result("", fname, "Applied overlap operator for mass = %e, Ls = %e: PV iterations = %d\n", iters);
    }

    return iters;
}

// Applies either Dov^dag or Dov^dag^-1
int ApplyOverlapDag(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, bool invert, Float residual)
{
    const char* fname = "ApplyOverlapDag()";

    static Timer timer(fname);
    static std::map<Float, Timer*> timers;
    Float key = invert ? -mass : mass;
    if (timers.count(key) == 0) {
	char timer_mass_name[512];
	sprintf(timer_mass_name, "ApplyOverlapDag(mass=%0.4f)", key);
	timers[key] = new Timer(timer_mass_name);
    }
    timer.start(true);
    timers[key]->start(true);


    Fermion_t vec_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    Fermion_t tmp_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };

    // vec_5d = 5D version of input vector
    bfm_d.cps_impexFermion_4d((Float *)in, vec_5d, Import);
    SetMass(bfm_d, invert ? mass : 1.0);
    if (use_mixed_solver) SetMass(bfm_f, invert ? mass : 1.0);
    bfm_d.residual = residual;
    if (use_mixed_solver) bfm_f.residual = 1e-5;
    int iters;
#pragma omp parallel
    {
	// tmp_5d = D_DW(1)^dag^-1 vec_5d
	bfm_d.set_zero(tmp_5d[Even]);
	bfm_d.set_zero(tmp_5d[Odd]);
	// TODO: MADWF inversion
	if (!omp_get_thread_num()) (bfm_d.mass == 1.0 ? Fdwf4d::inv_pv_timer : Fdwf4d::inv_m_timer).start(true);
	//iters = bfm.CGNE_Mdag(tmp_5d, vec_5d);
	iters = use_mixed_solver ?
	    mixed_cg::threaded_cg_mixed_Mdag(tmp_5d, vec_5d, bfm_d, bfm_f, 5) :
	    bfm_d.CGNE_Mdag(tmp_5d, vec_5d);
	if (!omp_get_thread_num()) (bfm_d.mass == 1.0 ? Fdwf4d::inv_pv_timer : Fdwf4d::inv_m_timer).stop(true);
    }

    SetMass(bfm_d, invert ? 1.0 : mass);
    if (use_mixed_solver) SetMass(bfm_f, invert ? 1.0 : mass);
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

    timers[key]->stop(true);
    timer.stop(true);

    if (!invert) {
	VRB.Result("", fname, "Applied overlap-dagger operator for mass = %e, Ls = %e: PV iterations = %d\n", iters);
    }

    return iters;
}

void Fdwf4d::FminResExt(Vector *sol, Vector *source, Vector **sol_old, Vector **vm, int degree, CgArg *cg_arg, CnvFrmType cnv_frm)
{
    sol->VecZero(this->FvecSize());
}

// Takes the inner product of two fermion vectors
Float Fdwf4d::FhamiltonNode(Vector *phi, Vector *chi)
{
    const char *fname = "FhamiltonNode(V*, V*)";
    if (phi == 0) ERR.Pointer(cname, fname, "phi");
    if (chi == 0) ERR.Pointer(cname, fname, "chi");

    return phi->ReDotProductNode(chi, this->FvecSize());
}

ForceArg Fdwf4d::EvolveMomFforce(Matrix *mom, Vector *phi, Vector *eta, Float mass, Float step_size)
{
    const char* fname = "EvolveMomFforce(M*,V*,V*,F,F)";
    static Timer time(cname, fname);
    time.start(true);

    SetDWFParams(mass);
    ForceArg ret = Dwf4d_EvolveMomFforceBase(bfm_d, bfm_f, use_mixed_solver, 
	this->GaugeField(), mom, phi, eta, dwfParams.mass, -step_size, pauli_villars_resid); // note minus sign

    time.stop(true);
    return ret;
}

ForceArg Fdwf4d::EvolveMomFforce(Matrix *mom, Vector *frm, Float mass, Float step_size)
{
    const char* fname = "EvolveMomFforce(M*,V*,F,F)";
    static Timer time(cname, fname);
    time.start(true);

    SetDWFParams(mass);

    // Compute Dfrm = Dov * frm
    Vector *Dfrm = (Vector*)smalloc(this->FvecSize() * sizeof(Float), "Dfrm", fname, cname);
    ApplyOverlap(bfm_d, bfm_f, use_mixed_solver, Dfrm, frm, dwfParams.mass, false, pauli_villars_resid); // tight residual since we only have to invert D_DW(1)

    ForceArg force_arg = Dwf4d_EvolveMomFforceBase(bfm_d, bfm_f, use_mixed_solver, 
	this->GaugeField(), mom, Dfrm, frm, dwfParams.mass, step_size, pauli_villars_resid);

    sfree(Dfrm, "Dfrm", fname, cname);
    time.stop(true);
    return force_arg;
}


ForceArg Dwf4d_EvolveMomFforceBase(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver, 
    Matrix *gauge_field, Matrix *mom, Vector *phi1, Vector *phi2, Float mass, Float coef, Float pauli_villars_resid)
{
    const char* fname = "Dwf4d_EvolveMomFforceBase()";
    static Timer time(fname);
    time.start(true);

    long f_size_5d = (long)SPINOR_SIZE * GJP.VolNodeSites() * bfm_d.Ls;
    Float *v1 = (Float *)smalloc("", fname, "v1", sizeof(Float) * f_size_5d);
    Float *v2 = (Float *)smalloc("", fname, "v2", sizeof(Float) * f_size_5d);

    Dwf4d_CalcHmdForceVecsBilinear(bfm_d, bfm_f, use_mixed_solver, v1, v2, phi1, phi2, mass, pauli_villars_resid);

    FforceWilsonType cal_force(mom, gauge_field, v1, v2, bfm_d.Ls, -coef); // note minus sign
    ForceArg ret = cal_force.run();

    sfree("", fname, "v1", v1);
    sfree("", fname, "v2", v2);
    time.stop(true);
    return ret;
}

// Calculates two 5D vectors 
//  v1 = D_DW(1)^dag^-1 P phi1
//  v2 = [B(m) - B(1) D_DW(1)^-1 D_DW(m)] P phi2
// which are used in the force calculation.
//
// v1 and v2 are placed in (color, spin, s, x, y, z, t) order
// because this is what is needed by the force calculation
void Dwf4d_CalcHmdForceVecsBilinear(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver, 
    Float *v1, Float *v2, Vector *phi1, Vector *phi2, Float mass, Float pauli_villars_resid)
{
    const char* fname = "Dwf4d_CalcHmdForceVecsBilinear()";
    static Timer time(fname);
    time.start(true);

    // TODO: reduce number of temporary vectors allocated

    Fermion_t v1_bfm[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    Fermion_t v2_bfm[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };

    Fermion_t phi1_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    Fermion_t phi2_5d[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };

    Fermion_t Dm_Pphi2[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    Fermion_t D1inv_Dm_Pphi2[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    Fermion_t B1_D1inv_Dm_Pphi2[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };
    Fermion_t Bm_Pphi2[] = { bfm_d.allocFermion(), bfm_d.allocFermion() };

    // convert phi_1 and phi_2 to 5D and apply the P matrix
    bfm_d.cps_impexFermion_4d((Float *)phi1, phi1_5d, Import);
    bfm_d.cps_impexFermion_4d((Float *)phi2, phi2_5d, Import);

    SetMass(bfm_d, mass);
    if (use_mixed_solver) SetMass(bfm_f, mass);
#pragma omp parallel
    {
	// Dm_Pphi2 = D_DW(m) P phi2
	bfm_d.G5D_Munprec(phi2_5d, Dm_Pphi2, DaggerNo);

	// Bm_Pphi2 = B(m) P phi2
	bfm_d.Booee(phi2_5d[Even], Bm_Pphi2[Even], DaggerNo);
	bfm_d.Booee(phi2_5d[Odd], Bm_Pphi2[Odd], DaggerNo);
    }

    SetMass(bfm_d, 1.0);
    if (use_mixed_solver) SetMass(bfm_f, 1.0);
    bfm_d.residual = pauli_villars_resid; // solve tightly since we only have to invert D_DW(1)
    if (use_mixed_solver) bfm_f.residual = 1e-5; // solve tightly since we only have to invert D_DW(1)
    int iters = 0;
#pragma omp parallel
    {
	// D1inv_Dm_Pphi2 = D_DW(1)^-1 D_DW(m) P phi2
	bfm_d.set_zero(D1inv_Dm_Pphi2[Even]);
	bfm_d.set_zero(D1inv_Dm_Pphi2[Odd]);
	if (!omp_get_thread_num()) Fdwf4d::inv_pv_timer.start(true);
	//bfm.CGNE_M(D1inv_Dm_Pphi2, Dm_Pphi2);
	iters += use_mixed_solver ?
	    mixed_cg::threaded_cg_mixed_M(D1inv_Dm_Pphi2, Dm_Pphi2, bfm_d, bfm_f, 5) :
	    bfm_d.CGNE_M(D1inv_Dm_Pphi2, Dm_Pphi2);
	if (!omp_get_thread_num()) Fdwf4d::inv_pv_timer.stop(true);

	// B1_D1inv_Dm_Pphi2 = B(1) D_DW(1)^-1 D_DW(M) P phi
	bfm_d.Booee(D1inv_Dm_Pphi2[Even], B1_D1inv_Dm_Pphi2[Even], DaggerNo);
	bfm_d.Booee(D1inv_Dm_Pphi2[Odd], B1_D1inv_Dm_Pphi2[Odd], DaggerNo);

	// v2 = B(m) P phi2 - B(1) D_DW(1)^-1 D_DW(m) P phi2
	bfm_d.axpy(v2_bfm, B1_D1inv_Dm_Pphi2, Bm_Pphi2, -1.0);

	bfm_d.set_zero(v1_bfm[Even]);
	bfm_d.set_zero(v1_bfm[Odd]);
	if (!omp_get_thread_num()) Fdwf4d::inv_pv_timer.start(true);
	//bfm.CGNE_Mdag(v1_bfm, phi1_5d); // v1_bfm = D_DW(1)^dag^-1 P phi1
	iters += use_mixed_solver ?
	    mixed_cg::threaded_cg_mixed_Mdag(v1_bfm, phi1_5d, bfm_d, bfm_f, 5) :
	    bfm_d.CGNE_Mdag(v1_bfm, phi1_5d);
	if (!omp_get_thread_num()) Fdwf4d::inv_pv_timer.stop(true);
    }

    bfm_d.cps_impexFermion_s(v1, v1_bfm, Export);
    bfm_d.cps_impexFermion_s(v2, v2_bfm, Export);

    bfm_d.freeFermion(v1_bfm[Even]);
    bfm_d.freeFermion(v1_bfm[Odd]);
    bfm_d.freeFermion(v2_bfm[Even]);
    bfm_d.freeFermion(v2_bfm[Odd]);
    bfm_d.freeFermion(phi1_5d[Even]);
    bfm_d.freeFermion(phi1_5d[Odd]);
    bfm_d.freeFermion(phi2_5d[Even]);
    bfm_d.freeFermion(phi2_5d[Odd]);
    bfm_d.freeFermion(Dm_Pphi2[Even]);
    bfm_d.freeFermion(Dm_Pphi2[Odd]);
    bfm_d.freeFermion(D1inv_Dm_Pphi2[Even]);
    bfm_d.freeFermion(D1inv_Dm_Pphi2[Odd]);
    bfm_d.freeFermion(B1_D1inv_Dm_Pphi2[Even]);
    bfm_d.freeFermion(B1_D1inv_Dm_Pphi2[Odd]);
    bfm_d.freeFermion(Bm_Pphi2[Even]);
    bfm_d.freeFermion(Bm_Pphi2[Odd]);

    VRB.Result("", fname, "mass = %e, Ls = %d, two PV inversions totaled %d iterations\n", mass, bfm_d.Ls, iters);

    time.stop(true);
}


int Fdwf4d::FmatInv(Vector *f_out, Vector *f_in, CgArg *cg_arg, Float *true_res, CnvFrmType cnv_frm, PreserveType prs_f_in)
{
    const char* fname = "FmatInv()";
    ERR.NotImplemented(cname, fname);
}

int Fdwf4d::FmatInv(Vector *f_out, Vector *f_in, CgArg *cg_arg, CnvFrmType cnv_frm, PreserveType prs_f_in)
{
    FmatInv(f_out, f_in, cg_arg, NULL, cnv_frm, prs_f_in);
}


int Fdwf4d::FmatEvlMInv(Vector **f_out, Vector *f_in, Float *shift, int Nshift, int isz, CgArg **cg_arg,
    CnvFrmType cnv_frm, MultiShiftSolveType type, Float *alpha, Vector **f_out_d)
{
    const char* fname = "FmatEvlMInv()";
    ERR.NotImplemented(cname, fname);
}

ForceArg Fdwf4d::RHMC_EvolveMomFforce(Matrix *mom, Vector **sol, int degree,
    int isz, Float *alpha, Float mass, Float dt,
    Vector **sol_d, ForceMeasure measure)
{
    const char* fname = "RHMC_EvolveMomFforce()";
    ERR.NotImplemented(cname, fname);
}


int Fdwf4d::FsiteOffsetChkb(const int* x) const
{
    const char* fname = "FsiteOffsetChkb()";
    ERR.NotImplemented(cname, fname);
}

int Fdwf4d::FsiteOffset(const int* x) const
{
    const char* fname = "FsiteOffset()";
    ERR.NotImplemented(cname, fname);
}

void Fdwf4d::Fconvert(Vector *f_field, StrOrdType to, StrOrdType from)
{
    const char* fname = "Fconvert()";
    ERR.NotImplemented(cname, fname);
}

Float Fdwf4d::BhamiltonNode(Vector *boson, Float mass)
{
    const char* fname = "BhamiltonNode()";
    ERR.NotImplemented(cname, fname);
}

void Fdwf4d::BforceVector(Vector *in, CgArg *cg_arg)
{
    const char* fname = "BforceVector()";
    ERR.NotImplemented(cname, fname);
}



CPS_END_NAMESPACE

#endif
