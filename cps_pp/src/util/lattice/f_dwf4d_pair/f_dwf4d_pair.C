#ifdef USE_BFM

#include <util/lattice/f_dwf4d_pair.h>
#include <util/lattice/f_dwf4d.h>
#include <util/lattice/eff_overlap.h>
#include <util/timer.h>

#include <util/lattice/bfm_mixed_solver.h>

CPS_START_NAMESPACE

std::map<Float, DWFParams*> Fdwf4dPair::paramMap;

bool Fdwf4dPair::use_mixed_solver = true;

Float Fdwf4dPair::pauli_villars_resid = 1e-12;


void Fdwf4dPair::SetActiveBfm(int idx)
{
    const char* fname = "SetActiveBfm(int)";

    if (idx != active_bfm_idx ||
	bfm_d[active_bfm_idx].mass != dwfParams[active_bfm_idx].mass ||
	bfm_d[active_bfm_idx].M5 != dwfParams[active_bfm_idx].M5 ||
	bfm_d[active_bfm_idx].Ls != dwfParams[active_bfm_idx].Ls ||
	bfm_d[active_bfm_idx].mobius_scale != dwfParams[active_bfm_idx].mobius_scale) {

	VRB.Result(cname, fname, "changing active bfm from %d to %d\n", active_bfm_idx, idx);

	if (active_bfm_idx != -1) {
	    bfm_d[active_bfm_idx].end();
	    if (use_mixed_solver) {
		bfm_f[active_bfm_idx].end();
	    }
	}
	InitBfmFromDWFParams(bfm_d[idx], dwfParams[idx], this->GaugeField());
	if (use_mixed_solver) {
	    bfm_d[idx].comm_end();
	    InitBfmFromDWFParams(bfm_f[idx], dwfParams[idx], this->GaugeField());
	    bfm_f[idx].comm_end();
	    bfm_d[idx].comm_init();
	}
	active_bfm_idx = idx;
    } else {
	VRB.Result(cname, fname, "active bfm is already %d\n", active_bfm_idx);
    }
}

void Fdwf4dPair::SetDWFParams(Float mass)
{
    const char* fname = "SetDwfParams(F)";

    if (paramMap.count(mass) == 0) ERR.General(cname, fname, "No DWFParams mapped for mass = %e\n", mass);
    dwfParams[0] = paramMap[mass][0];
    dwfParams[1] = paramMap[mass][1];

    VRB.Result(cname, fname, "Set DWF params from key mass = %e: dwfParams[0]: mass = %e, M5 = %e, Ls = %d, mobius_scale = %e\n",
	mass, dwfParams[0].mass, dwfParams[0].M5, dwfParams[0].Ls, dwfParams[0].mobius_scale);
    VRB.Result(cname, fname, "Set DWF params from key mass = %e: dwfParams[1]: mass = %e, M5 = %e, Ls = %d, mobius_scale = %e\n",
	mass, dwfParams[1].mass, dwfParams[1].M5, dwfParams[1].Ls, dwfParams[1].mobius_scale);
}

// Initialize QDP++ before using this class.
Fdwf4dPair::Fdwf4dPair(void)
    :cname("Fdwf4dPair")
{
    const char *fname = "Fdwf4dPair()";
    VRB.Func(cname, fname);

    if (GJP.Snodes() != 1) ERR.NotImplemented(cname, fname);
    if (sizeof(Float) == sizeof(float)) ERR.NotImplemented(cname, fname);

//    init_bfm(bfm[0], this->GaugeField());
//    bfm[0].comm_end();
//    init_bfm(bfm[1], this->GaugeField());    
//    active_bfm_idx = 1;
    active_bfm_idx = -1;
}

Fdwf4dPair::~Fdwf4dPair(void)
{
    const char *fname = "~Fdwf4dPair()";
    VRB.Func(cname,fname);

    if (active_bfm_idx != -1) {
	bfm_d[active_bfm_idx].end();
	if (use_mixed_solver) {
	    bfm_f[active_bfm_idx].end();
	}
    }
}

// Does phi = Dov[0] * Dov[1] * frm1  or  phi = Dov[1]^dag * Dov[0]^dag * frm1
// Returns (frm1, frm1)
Float Fdwf4dPair::SetPhi(Vector *phi, Vector *frm1, Vector *frm2, Float mass, DagType dag)
{
    const char *fname = "SetPhi(V*,V*,V*,F)";
    static Timer time(cname, fname);
    time.start(true);

    if (phi == 0) ERR.Pointer(cname, fname, "phi");
    if (frm1 == 0) ERR.Pointer(cname, fname, "frm1");

    SetDWFParams(mass);

    Vector *tmp = (Vector*)smalloc(this->FvecSize() * sizeof(Float), "tmp", fname, cname);

    if (dag == DAG_NO) {
	SetActiveBfm(1);
	ApplyOverlap(bfm_d[1], bfm_f[1], use_mixed_solver, tmp, frm1, dwfParams[1].mass, pauli_villars_resid);
	SetActiveBfm(0);
	ApplyOverlap(bfm_d[0], bfm_f[0], use_mixed_solver, phi, tmp, dwfParams[0].mass, pauli_villars_resid);
    } else {
	SetActiveBfm(0);
	ApplyOverlapDag(bfm_d[0], bfm_f[0], use_mixed_solver, tmp, frm1, dwfParams[0].mass, pauli_villars_resid);
	SetActiveBfm(1);
	ApplyOverlapDag(bfm_d[1], bfm_f[1], use_mixed_solver, phi, tmp, dwfParams[1].mass, pauli_villars_resid);
    }

    sfree(tmp, "tmp", fname, cname);

    Float ret = FhamiltonNode(frm1, frm1);
    time.stop(true);
    return ret;
}

// Does f_out = (M^dag M)^-1 f_in. Returns iteration count.
// Here M = Dov[0] Dov[1]
// So M^dag = Dov[1]^dag Dov[0]^dag
// and M^dag M = Dov[1]^dag Dov[0]^dag Dov[0] Dov[1]
// and (M^dag M)^-1 = Dov[1]^-1 Dov[0]^-1 Dov[0]^dag^-1 Dov[1]^dag^-1
int Fdwf4dPair::FmatEvlInv(Vector *f_out, Vector *f_in, CgArg *cg_arg, Float *true_res, CnvFrmType cnv_frm)
{
    const char* fname = "FmatEvlInv()";
    static Timer time(cname, fname);
    time.start(true);
    
    SetDWFParams(cg_arg->mass);

    Vector *tmp1 = (Vector*)smalloc(this->FvecSize() * sizeof(Float), "tmp1", fname, cname);
    Vector *tmp2 = (Vector*)smalloc(this->FvecSize() * sizeof(Float), "tmp2", fname, cname);

    int iters = 0;

    // tmp1 = Dov[1]^dag^-1 f_in
    SetActiveBfm(1);
    iters += ApplyOverlapDagInverse(bfm_d[1], bfm_f[1], use_mixed_solver, tmp1, f_in, dwfParams[1].mass, cg_arg->stop_rsd);

    // tmp2 = Dov[0]^dag^-1 Dov[1]^dag^-1 f_in              
    //   then
    // tmp1 = Dov[0]^-1 Dov[0]^dag^-1 Dov[1]^dag^-1 f_in
    SetActiveBfm(0);
    iters += ApplyOverlapDagInverse(bfm_d[0], bfm_f[0], use_mixed_solver, tmp2, tmp1, dwfParams[0].mass, cg_arg->stop_rsd);
    iters += ApplyOverlapInverse(bfm_d[0], bfm_f[0], use_mixed_solver, tmp1, tmp2, dwfParams[0].mass, cg_arg->stop_rsd);

    // f_out = Dov[1]^-1 Dov[0]^-1 Dov[0]^dag^-1 Dov[1]^dag^-1 f_in 
    SetActiveBfm(1);
    iters += ApplyOverlapInverse(bfm_d[1], bfm_f[1], use_mixed_solver, f_out, tmp1, dwfParams[1].mass, cg_arg->stop_rsd);

    sfree(tmp1, "tmp1", fname, cname);
    sfree(tmp2, "tmp2", fname, cname);

    // TODO: calculate true residual
    if (true_res != NULL) *true_res = -1.0;

    time.stop(true);
    return iters;
}

int Fdwf4dPair::FmatEvlInv(Vector *f_out, Vector *f_in, CgArg *cg_arg, CnvFrmType cnv_frm)
{
    FmatEvlInv(f_out, f_in, cg_arg, NULL, cnv_frm);
}

int Fdwf4dPair::FeigSolv(Vector **f_eigenv, Float *lambda, Float *chirality, int *valid_eig,
    Float **hsum, EigArg *eig_arg, CnvFrmType cnv_frm)
{
    const char* fname = "FeigSolv()";
    ERR.NotImplemented(cname, fname);
}

void Fdwf4dPair::FminResExt(Vector *sol, Vector *source, Vector **sol_old, Vector **vm, int degree, CgArg *cg_arg, CnvFrmType cnv_frm)
{
    sol->VecZero(this->FvecSize());
}

// Takes the inner product of two fermion vectors
Float Fdwf4dPair::FhamiltonNode(Vector *phi, Vector *chi)
{
    const char *fname = "FhamiltonNode(V*, V*)";
    if (phi == 0) ERR.Pointer(cname, fname, "phi");
    if (chi == 0) ERR.Pointer(cname, fname, "chi");

    return phi->ReDotProductNode(chi, this->FvecSize());
}

ForceArg Fdwf4dPair::EvolveMomFforce(Matrix *mom, Vector *phi, Vector *eta, Float mass, Float step_size)
{
    const char* fname = "EvolveMomFforce(M*,V*,V*,F,F)";
    static Timer time(cname, fname);
    time.start(true);

    SetDWFParams(mass);

    ForceArg force_arg = EvolveMomFforceBase(mom, phi, eta, mass, -step_size); // note minus sign

    time.stop(true);
    return force_arg;
}

ForceArg Fdwf4dPair::EvolveMomFforce(Matrix *mom, Vector *frm, Float mass, Float step_size)
{
    const char* fname = "EvolveMomFforce(M*,V*,F,F)";
    static Timer time(cname, fname);
    time.start(true);

    SetDWFParams(mass);

    // Compute Mfrm = M * frm = Dov[0] * Dov[1] * frm
    Vector *Mfrm = (Vector*)smalloc(this->FvecSize() * sizeof(Float), "Dfrm", fname, cname);
    Vector *tmp = (Vector*)smalloc(this->FvecSize() * sizeof(Float), "tmp", fname, cname);
    
    SetActiveBfm(1);
    ApplyOverlap(bfm_d[1], bfm_f[1], use_mixed_solver, tmp, frm, dwfParams[1].mass, pauli_villars_resid); // tight residual since we only have to invert D_DW(1)
    SetActiveBfm(0);
    ApplyOverlap(bfm_d[0], bfm_f[0], use_mixed_solver, Mfrm, tmp, dwfParams[0].mass, pauli_villars_resid); // tight residual since we only have to invert D_DW(1)

    ForceArg force_stats = EvolveMomFforceBase(mom, Mfrm, frm, mass, step_size);

    sfree(Mfrm, "Mfrm", fname, cname);
    sfree(tmp, "tmp", fname, cname);

    time.stop(true);
    return force_stats;
}

// For each link U_x,u
//
// mom_x,u += coef * T^a [phi_1^dag (d^a_x,u M) phi_2 + phi_2^dag (d^a_x,u M^dag) phi_1]
//
// where d^a_x,u is the derivative with respect to U_x,u 
// in the direction of the su(2) generator T^a
//
// Here M = Dov[0] Dov[1]. Using the product rule we can write the momentum update as
// two separate updates:
//
// alpha = Dov[0]^dag phi_1
// beta = Dov[1] phi_2
//
// mom_x,u += coef * T^a [phi_1^dag (d^a_x,u Dov[0]) beta + beta^dag (d^a_x,u Dov[0]^dag) phi_1]
// mom_x,u += coef * T^a [alpha^dag (d^a_x,u Dov[1]) phi_2 + phi_2^dag (d^a_x,u Dov[1]^dag) alpha]
//
// These two updates are performed by Dwf4d_EvolveMomFforceBase()
ForceArg Fdwf4dPair::EvolveMomFforceBase(Matrix *mom, Vector *phi1, Vector *phi2, Float mass, Float coef)
{
    const char* fname = "EvolveMomFforceBase()";
    static Timer time(cname, fname);
    time.start(true);

    int num_links = 4 * GJP.VolNodeSites();
    Vector *alpha = (Vector*)smalloc(this->FvecSize() * sizeof(Float), "alpha", fname, cname);
    Vector *beta = (Vector*)smalloc(this->FvecSize() * sizeof(Float), "beta", fname, cname);

    SetActiveBfm(0);
    ApplyOverlapDag(bfm_d[0], bfm_f[0], use_mixed_solver, alpha, phi1, dwfParams[0].mass, pauli_villars_resid); // tight residual since we only have to invert D_DW(1)
    SetActiveBfm(1);
    ApplyOverlap(bfm_d[1], bfm_f[1], use_mixed_solver, beta, phi2, dwfParams[1].mass, pauli_villars_resid); // tight residual since we only have to invert D_DW(1)

    // We collect the total momentum update into delta_mom so that we can compute
    // force stats
    Matrix *delta_mom = (Matrix*)smalloc(num_links * sizeof(Matrix), "delta_mom", fname, cname);
#pragma omp parallel for
    for (int i = 0; i < num_links; i++) {
	delta_mom[i] = 0.0;
    }

    SetActiveBfm(0);
    Dwf4d_EvolveMomFforceBase(bfm_d[0], bfm_f[0], use_mixed_solver, 
	this->GaugeField(), delta_mom, phi1, beta, dwfParams[0].mass, coef, pauli_villars_resid);
    SetActiveBfm(1);
    Dwf4d_EvolveMomFforceBase(bfm_d[1], bfm_f[1], use_mixed_solver, 
	this->GaugeField(), delta_mom, alpha, phi2, dwfParams[1].mass, coef, pauli_villars_resid);

#pragma omp parallel for
    for (int i = 0; i < num_links; i++) {
	mom[i] += delta_mom[i];
    }

    ForceArg force_stats;
    force_stats.measure(delta_mom);
    force_stats.glb_reduce();

    sfree(alpha, "alpha", fname, cname);
    sfree(beta, "beta", fname, cname);
    sfree(delta_mom, "delta_mom", fname, cname);

    time.stop(true);
    return force_stats;
}

int Fdwf4dPair::FmatInv(Vector *f_out, Vector *f_in, CgArg *cg_arg, Float *true_res, CnvFrmType cnv_frm, PreserveType prs_f_in)
{
    const char* fname = "FmatInv()";
    ERR.NotImplemented(cname, fname);
}

int Fdwf4dPair::FmatInv(Vector *f_out, Vector *f_in, CgArg *cg_arg, CnvFrmType cnv_frm, PreserveType prs_f_in)
{
    FmatInv(f_out, f_in, cg_arg, NULL, cnv_frm, prs_f_in);
}


int Fdwf4dPair::FmatEvlMInv(Vector **f_out, Vector *f_in, Float *shift, int Nshift, int isz, CgArg **cg_arg,
    CnvFrmType cnv_frm, MultiShiftSolveType type, Float *alpha, Vector **f_out_d)
{
    const char* fname = "FmatEvlMInv()";
    ERR.NotImplemented(cname, fname);
}

ForceArg Fdwf4dPair::RHMC_EvolveMomFforce(Matrix *mom, Vector **sol, int degree,
    int isz, Float *alpha, Float mass, Float dt,
    Vector **sol_d, ForceMeasure measure)
{
    const char* fname = "RHMC_EvolveMomFforce()";
    ERR.NotImplemented(cname, fname);
}


int Fdwf4dPair::FsiteOffsetChkb(const int* x) const
{
    const char* fname = "FsiteOffsetChkb()";
    ERR.NotImplemented(cname, fname);
}

int Fdwf4dPair::FsiteOffset(const int* x) const
{
    const char* fname = "FsiteOffset()";
    ERR.NotImplemented(cname, fname);
}

void Fdwf4dPair::Fconvert(Vector *f_field, StrOrdType to, StrOrdType from)
{
    const char* fname = "Fconvert()";
    ERR.NotImplemented(cname, fname);
}

Float Fdwf4dPair::BhamiltonNode(Vector *boson, Float mass)
{
    const char* fname = "BhamiltonNode()";
    ERR.NotImplemented(cname, fname);
}

void Fdwf4dPair::BforceVector(Vector *in, CgArg *cg_arg)
{
    const char* fname = "BforceVector()";
    ERR.NotImplemented(cname, fname);
}



CPS_END_NAMESPACE

#endif
