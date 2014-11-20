#ifndef INCLUDED_FDWF4DPAIR_H__
#define INCLUDED_FDWF4DPAIR_H__

#ifdef USE_BFM

#include <config.h>
#include <util/lattice.h>
#include <util/lattice/f_dwf4d.h>

#include <map>

CPS_START_NAMESPACE

// Implements a fermion action based on Dov(m_1) Dov'(m_2) where
// Dov and Dov' are the approximate 4D overlap Dirac operators corresponding
// to two different 5D domain wall operators. 
class Fdwf4dPair : public virtual Lattice {
public:

    static std::map<Float, DWFParams*> paramMap;

    static bool use_mixed_solver;

    static Float pauli_villars_resid;

private:
    const char *cname;

    int active_bfm_idx;
    bfm_evo<double> bfm_d[2];
    bfm_evo<float> bfm_f[2];
    DWFParams dwfParams[2];

    void SetActiveBfm(int idx);
    void SetDWFParams(Float mass);

public:

    Fdwf4dPair(void);
    virtual ~Fdwf4dPair(void);

    // Does phi = M frm1 or M^dag frm1. In our case M will
    // be Dov(m_1) Dov'(m_2). Returns FhamiltonNode(frm1, frm1)
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

    // For each link U_x,u
    //
    // mom_x,u += coef * T^a [phi_1^dag (d^a_x,u M) phi_2 + phi_2^dag (d^a_x,u M^dag) phi_1]
    //
    // where d^a_x,u is the derivative with respect to U_x,u 
    // in the direction of the su(2) generator T^a
    ForceArg EvolveMomFforceBase(Matrix *mom, Vector *phi1, Vector *phi2, Float mass, Float coef);

    int FmatEvlMInv(Vector **f_out, Vector *f_in, Float *shift, int Nshift, int isz, CgArg **cg_arg,
	CnvFrmType cnv_frm, MultiShiftSolveType type, Float *alpha, Vector **f_out_d);

    ForceArg RHMC_EvolveMomFforce(Matrix *mom, Vector **sol, int degree,
	int isz, Float *alpha, Float mass, Float dt,
	Vector **sol_d, ForceMeasure measure);

    FclassType Fclass() const { return F_CLASS_DWF4D_PAIR; }

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

class GnoneFdwf4dPair
    : public virtual Lattice,
      public virtual Gnone,
      public virtual Fdwf4dPair
{
private:
    const char *cname;
public:
    GnoneFdwf4dPair(void);
    virtual ~GnoneFdwf4dPair();
};

CPS_END_NAMESPACE

#endif

#endif
