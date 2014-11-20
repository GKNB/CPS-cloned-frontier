#ifndef INCLUDED_FDWF4D_H__
#define INCLUDED_FDWF4D_H__

#ifdef USE_BFM

#include <config.h>
#include <util/lattice.h>
#include <util/lattice/bfm_evo.h>
#include <util/timer.h>

#include <bfm.h>

CPS_START_NAMESPACE

// A DWFParams instance specifies a 5D domain wall operator.
struct DWFParams
{
    Float mass;
    Float M5;
    int Ls;
    Float mobius_scale;
};

// Some functions are defined outside Fdwf4d so that they can also be used
// by Fdwf4dPair

template<class T>
void InitBfmFromDWFParams(bfm_evo<T> &bfm, DWFParams dwfParams, Matrix *gauge_field)
{
    bfmarg ba;
    ba.ScaledShamirCayleyTanh(dwfParams.mass, dwfParams.M5, dwfParams.Ls, dwfParams.mobius_scale);

    ba.max_iter = 1000000;
    ba.residual = 1e-12; // default, should always be overwritten before a solve

    multi1d<int> lattSize = QDP::Layout::subgridLattSize();
    ba.node_latt[0] = lattSize[0];
    ba.node_latt[1] = lattSize[1];
    ba.node_latt[2] = lattSize[2];
    ba.node_latt[3] = lattSize[3];

    multi1d<int> procs = QDP::Layout::logicalSize();
    ba.local_comm[0] = procs[0] > 1 ? 0 : 1;
    ba.local_comm[1] = procs[1] > 1 ? 0 : 1;
    ba.local_comm[2] = procs[2] > 1 ? 0 : 1;
    ba.local_comm[3] = procs[3] > 1 ? 0 : 1;

    ba.ncoor[0] = 0;
    ba.ncoor[1] = 0;
    ba.ncoor[2] = 0;
    ba.ncoor[3] = 0;

    ba.verbose = BfmMessage | BfmError;

    bfm.init(ba);
    bfm.cps_importGauge((Float *)gauge_field);
}

// out = Dov * in
//   or
// out = Dov^-1 * in
int ApplyOverlap(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, bool invert, Float residual);

// out = Dov^dag * in
//   or
// out = Dov^dag^-1 * in
int ApplyOverlapDag(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Vector *out, Vector *in, Float mass, bool invert, Float residual);

// For each link U_x,u
//
// mom_x,u += coef * T^a [phi_1^dag (d^a_x,u Dov) phi_2 + phi_2^dag (d^a_x,u Dov^dag) phi_1]
//
// where d^a_x,u is the derivative with respect to U_x,u 
// in the direction of the su(2) generator T^a
ForceArg Dwf4d_EvolveMomFforceBase(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Matrix *gauge_field, Matrix *mom, Vector *phi1, Vector *phi2, Float mass, Float coef, Float pauli_villars_resid);

// Used by Dwf4d_EvolveMomFforceBase
void Dwf4d_CalcHmdForceVecsBilinear(bfm_evo<double> &bfm_d, bfm_evo<float> &bfm_f, bool use_mixed_solver,
    Float *v1, Float *v2, Vector *phi1, Vector *phi2, Float mass, Float pauli_villars_resid);



// Implements a fermion action based on the approximate 4D overlap Dirac operator
// corresponding to a 5D domain wall fermion operator.
class Fdwf4d : public virtual Lattice {
public:
    static std::map<Float, DWFParams> paramMap;

    static bool use_mixed_solver;

    static Float pauli_villars_resid;

    static Timer inv_m_timer;
    static Timer inv_pv_timer;

private:
    const char *cname;

    bfm_evo<double> bfm_d;
    bfm_evo<float> bfm_f;
    DWFParams dwfParams;

    void SetDWFParams(Float mass);

public:

    Fdwf4d(void);
    virtual ~Fdwf4d(void);

    // Does phi = M frm1 or M^dag frm1. In our case M will
    // be Dov. Returns FhamiltonNode(frm1, frm1)
    Float SetPhi(Vector *phi, Vector *frm1, Vector *frm2, Float mass, DagType dag);

    // Does f_out = M^-1 f_in
    int FmatInv(Vector *f_out, Vector *f_in, CgArg *cg_arg, Float *true_res, CnvFrmType cnv_frm = CNV_FRM_YES, PreserveType prs_f_in = PRESERVE_YES);
    int FmatInv(Vector *f_out, Vector *f_in, CgArg *cg_arg, CnvFrmType cnv_frm = CNV_FRM_YES, PreserveType prs_f_in = PRESERVE_YES);

    // Does f_out = (M^dag M)^-1 f_in. Returns iteration count.
    int FmatEvlInv(Vector *f_out, Vector *f_in, CgArg *cg_arg, Float *true_res, CnvFrmType cnv_frm = CNV_FRM_YES);
    int FmatEvlInv(Vector *f_out, Vector *f_in, CgArg *cg_arg, CnvFrmType cnv_frm = CNV_FRM_YES);

    int FeigSolv(Vector **f_eigenv, Float *lambda, Float *chirality, int *valid_eig,
	Float **hsum, EigArg *eig_arg, CnvFrmType cnv_frm = CNV_FRM_YES);

    // Takes the inner product of two fermion vectors
    Float FhamiltonNode(Vector *phi, Vector *chi);

    // Sets the guess sol. We just set sol to all zeros
    void FminResExt(Vector *sol, Vector *source, Vector **sol_old, Vector **vm, int degree, CgArg *cg_arg, CnvFrmType cnv_frm);

    // It evolves the canonical momentum mom by step_size
    // using the fermion force.
    ForceArg EvolveMomFforce(Matrix *mom, Vector *frm, Float mass, Float step_size);

    // It evolve the canonical momentum mom  by step_size
    // using the bosonic quotient force.
    ForceArg EvolveMomFforce(Matrix *mom, Vector *phi, Vector *eta, Float mass, Float step_size);    

    int FmatEvlMInv(Vector **f_out, Vector *f_in, Float *shift, int Nshift, int isz, CgArg **cg_arg,
	CnvFrmType cnv_frm, MultiShiftSolveType type, Float *alpha, Vector **f_out_d);

    ForceArg RHMC_EvolveMomFforce(Matrix *mom, Vector **sol, int degree,
	int isz, Float *alpha, Float mass, Float dt,
	Vector **sol_d, ForceMeasure measure);

    FclassType Fclass() const { return F_CLASS_DWF4D; }

    // Do NOT use checkerboarding in the evolution for overlap
    int FchkbEvl() const { return 0; }

    // Number of Floats in a spin-color vector: 2 complex components * 3 colors * 4 spins.
    // Overlap uses 4D fermions.
    int FsiteSize() const { return 2 * Colors() * SpinComponents(); }

    int FvecSize() const { return GJP.VolNodeSites() * this->FsiteSize(); }

    int ExactFlavors() const { return 2; }

    int SpinComponents() const { return 4; }

    int FsiteOffsetChkb(const int* x) const;
    int FsiteOffset(const int* x) const;

    void Fconvert(Vector *f_field, StrOrdType to, StrOrdType from);

    Float BhamiltonNode(Vector *boson, Float mass);

    void BforceVector(Vector *in, CgArg *cg_arg);


};

class GnoneFdwf4d
    : public virtual Lattice,
      public virtual Gnone,
      public virtual Fdwf4d
{
private:
    const char *cname;
public:
    GnoneFdwf4d(void);
    virtual ~GnoneFdwf4d();
};

CPS_END_NAMESPACE

#endif

#endif
