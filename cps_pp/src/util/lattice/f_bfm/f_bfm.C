// -*- c-basic-offset: 4 -*-
#include<config.h>
#include<math.h>

#include<util/multi_cg_controller.h>
CPS_START_NAMESPACE
MultiShiftCGcontroller MultiShiftController;
CPS_END_NAMESPACE
#ifdef USE_BFM

<<<<<<< HEAD
#include <util/lattice/bfm_evo.h>
#include <util/lattice/bfm_eigcg.h>
//#include <util/lattice/bfm_hdcg.h>
||||||| merged common ancestors
#include <util/lattice/bfm_evo.h>
=======
>>>>>>> ckelly_latest
#include <util/lattice/fbfm.h>
#include <util/lattice/bfm_eigcg.h> //Is included in BFM

#include <util/wilson.h>
#include <util/verbose.h>
#include <util/gjp.h>
#include <util/error.h>
#include <util/pt.h>
#include <comms/scu.h>
#include <comms/glb.h>
#include <util/enum_func.h>
#include <util/sproj_tr.h>
<<<<<<< HEAD
#include <util/time_cps.h>
#include <util/lattice/fforce_wilson_type.h>
#include <util/timer.h>
#include <util/lattice/hdcg_controller.h>
||||||| merged common ancestors
#include<util/time_cps.h>
=======
#include <util/time_cps.h>
#include <util/lattice/fforce_wilson_type.h>
>>>>>>> ckelly_latest

#include<omp.h>

#include <util/qioarg.h>
#include <util/WriteLatticePar.h>
#include <util/ReadLatticePar.h>

// These have to be defined somewhere. Why not here?
BfmHDCGParams HDCGInstance::Params;
HDCG_wrapper  *HDCGInstance::_instance = NULL;

CPS_START_NAMESPACE
MultiShiftCGcontroller MultiShiftController;
int Fbfm::current_arg_idx(0);
bfmarg Fbfm::bfm_args[2] = {};
bool Fbfm::use_mixed_solver = false;

<<<<<<< HEAD
std::map<Float, bfmarg> Fbfm::arg_map;
Float Fbfm::current_key_mass = -1789.8;

std::map<Float, MADWFParams> Fbfm::madwf_arg_map;

bool Fbfm::use_mixed_solver = false;


// NOTE:
//
// 1. Initialize QDP++ and the static Fbfm::arg_map before
// using this class.
//
// 2. This class acts like a DiracOp class, while it is in scope the
// gauge field has the boundary condition on.
||||||| merged common ancestors
inline void compute_coord(int x[4], const int hl[4], const int low[4], int i)
{
    x[0] = i % hl[0] + low[0]; i /= hl[0];
    x[1] = i % hl[1] + low[1]; i /= hl[1];
    x[2] = i % hl[2] + low[2]; i /= hl[2];
    x[3] = i % hl[3] + low[3];
}

// ----------------------------------------------------------------
// static void BondCond: toggle boundary condition on/off for any
// gauge-like field. Based on code from
// src/util/dirac_op/d_op_base/comsrc/dirac_op_base.C
//
// u_base must be in CANONICAL order.
// ----------------------------------------------------------------
template<typename Float>
static void BondCond(Float *u_base)
{
    for(int mu = 0; mu < 4; ++mu) {
        if(GJP.NodeBc(mu) != BND_CND_APRD) continue;

        int low[4] = { 0, 0, 0, 0 };
        int high[4] = { GJP.XnodeSites(), GJP.YnodeSites(),
                        GJP.ZnodeSites(), GJP.TnodeSites() };
        low[mu] = high[mu] - 1;

        int hl[4] = { high[0] - low[0], high[1] - low[1],
                      high[2] - low[2], high[3] - low[3] };

        const int hl_sites = hl[0] * hl[1] * hl[2] * hl[3];

#pragma omp parallel for
        for(int i = 0; i < hl_sites; ++i) {
            int x[4];
            compute_coord(x, hl, low, i);

            int off = mu + 4 * (x[0] + high[0] *
                                (x[1] + high[1] *
                                 (x[2] + high[2] * x[3])));
            Float *m = u_base + off * 18;
            for(int j = 0; j < 18; ++j) {
                m[j] = -m[j];
            }
        }
    }
}

bfmarg Fbfm::bfm_arg;
bool Fbfm::use_mixed_solver = 0;

// NOTE: Initialize QDP++ before using this class!
=======
// NOTE:
//
// 1. Initialize QDP++ and the static copy Fbfm::bfm_arg before
// using this class.
//
// 2. This class acts like a DiracOp class, while it is in scope the
// gauge field has the boundary condition on.
>>>>>>> ckelly_latest
Fbfm::Fbfm(void):cname("Fbfm")
{
    const char *fname = "Fbfm()";
    VRB.Func(cname,fname);

    if(GJP.Snodes() != 1) {
        ERR.NotImplemented(cname, fname);
    }
    if(sizeof(Float) == sizeof(float)) {
        ERR.NotImplemented(cname, fname);
    }

<<<<<<< HEAD
    Lattice::BondCond();
||||||| merged common ancestors
    bevo.init(bfm_arg);

    Float *gauge = (Float *)(this->GaugeField());
    BondCond(gauge);
    bevo.cps_importGauge(gauge);
    BondCond(gauge);

    // Fill in the array of sproj_tr functions, used for evolution.
    sproj_tr[SPROJ_XM] = sprojTrXm;
    sproj_tr[SPROJ_YM] = sprojTrYm;
    sproj_tr[SPROJ_ZM] = sprojTrZm;
    sproj_tr[SPROJ_TM] = sprojTrTm;
    sproj_tr[SPROJ_XP] = sprojTrXp;
    sproj_tr[SPROJ_YP] = sprojTrYp;
    sproj_tr[SPROJ_ZP] = sprojTrZp;
    sproj_tr[SPROJ_TP] = sprojTrTp;

    lclx[0] = GJP.XnodeSites();
    lclx[1] = GJP.YnodeSites();
    lclx[2] = GJP.ZnodeSites();
    lclx[3] = GJP.TnodeSites();
    lclx[4] = GJP.SnodeSites();
=======
    bd.init(bfm_args[current_arg_idx]);
>>>>>>> ckelly_latest

<<<<<<< HEAD
    bfm_inited = false;
    Float current_key_mass = -1789.8;
||||||| merged common ancestors
    int vol_5d = lclx[0] * lclx[1] * lclx[2] * lclx[3] * lclx[4];
    surf_size_all = 0;
    for(int i = 0; i < 4; ++i) {
        surf_size[i] = SPINOR_SIZE * (vol_5d / lclx[i]);
        surf_size_all += surf_size[i];
    }
=======
    if(use_mixed_solver) {
        bd.comm_end();
        bf.init(bfm_args[current_arg_idx]);
        bf.comm_end();
        bd.comm_init();
    }
>>>>>>> ckelly_latest

<<<<<<< HEAD
    evec = NULL;
    evald = NULL;
    evalf = NULL;
    ecnt = 0;
||||||| merged common ancestors
    // calculate offset of surface vectors v1 and v2
    surf_v1[0] = 0;
    surf_v2[0] = surf_size[0];
    for(int i = 1; i < 4; ++i) {
        surf_v1[i] = surf_v1[i-1] + surf_size[i-1] * 2;
        surf_v2[i] = surf_v1[i] + surf_size[i];
    }
=======
    // call our own version to import gauge field.
    Fbfm::BondCond();

    evec = NULL;
    evald = NULL;
    evalf = NULL;
    ecnt = 0;
>>>>>>> ckelly_latest
}

Fbfm::~Fbfm(void)
{
<<<<<<< HEAD
    const char *fname = "~Fbfm()";
    VRB.Result(cname, fname,"start");
    // we call base version just to revert the change, no need to
    // import to BFM in a destructor.
    Lattice::BondCond();
    VRB.Result(cname, fname,"BondCond");

    if (bfm_inited) {
	bd.end();
    VRB.Result(cname, fname,"bd.end()");
#if 0
	kernel.end();
    VRB.Result(cname, fname,"kernel.end()");
#endif
	if (use_mixed_solver) {
	    bf.end();
    VRB.Result(cname, fname,"bf.end()");
	}
    }
}

// automatically fills in some bfmarg fields
void AutofillBfmarg(bfmarg &arg)
{
    // Make sure some fields are filled in properly
    multi1d<int> sub_latt_size = QDP::Layout::subgridLattSize();
    arg.node_latt[0] = sub_latt_size[0];
    arg.node_latt[1] = sub_latt_size[1];
    arg.node_latt[2] = sub_latt_size[2];
    arg.node_latt[3] = sub_latt_size[3];

    multi1d<int> procs = QDP::Layout::logicalSize();
    arg.local_comm[0] = procs[0] > 1 ? 0 : 1;
    arg.local_comm[1] = procs[1] > 1 ? 0 : 1;
    arg.local_comm[2] = procs[2] > 1 ? 0 : 1;
    arg.local_comm[3] = procs[3] > 1 ? 0 : 1;

    arg.ncoor[0] = 0;
    arg.ncoor[1] = 0;
    arg.ncoor[2] = 0;
    arg.ncoor[3] = 0;

    arg.max_iter = 100000;
    arg.verbose = BfmMessage | BfmError;
}

void Fbfm::SetBfmArg(Float key_mass)
{
    const char* fname = "SetBfmArg(F)";

    if (arg_map.count(key_mass) == 0) {
	ERR.General(cname, fname, "No entry for key mass %e in arg_map!\n", key_mass);
||||||| merged common ancestors
    bevo.end();
}

// copy 3d surface data from v4d to v3d in mu direction, use this
// function to fill the buffer v4d before communication.
//
// If send_neg == true, then it copies data on x[mu] == 0 surface to
// v3d (sends data in negative direction), otherwise it copies data on
// x[mu] == size - 1 surface to v3d.
//
// !!!NOTE: v4d is assumed to be in sxyzt order, i.e., the s index
// changes fastest.
void Fbfm::CopySendFrmData(Float *v3d, Float *v4d, int mu, bool send_neg)
{
    int low[4] = { 0, 0, 0, 0 };
    int high[4] = { lclx[0], lclx[1], lclx[2], lclx[3] };
    low[mu] = send_neg ? 0 : lclx[mu] - 1;
    high[mu] = low[mu] + 1;

    int block_size = SPINOR_SIZE * lclx[4]; // s inner most

    const int hl[4] = {high[0] - low[0],
                       high[1] - low[1],
                       high[2] - low[2],
                       high[3] - low[3] };
    const int hl_sites = hl[0] * hl[1] * hl[2] * hl[3];

#pragma omp parallel for 
    for(int i = 0; i < hl_sites; ++i) {
        int x[4];
        compute_coord(x, hl, low, i);
        int off_4d = idx_4d(x, lclx);
        int off_3d = idx_4d_surf(x, lclx, mu);
        
        memcpy(v3d + off_3d * block_size,
               v4d + off_4d * block_size,
               sizeof(Float) * block_size);
    }
}

// just TrLessAntiHermMatrix() ...
static inline void trless_am(Float *p, Float coef)
{
    p[0] = p[8] = p[16] = 0.;

    Float tmp = 0.5*(p[2] - p[6]) * coef;
    p[2]=tmp; p[6] = -tmp;

    tmp = 0.5*(p[3] + p[7]) * coef;
    p[3]=tmp; p[7] = tmp;

    tmp = 0.5*(p[4] - p[12]) * coef;
    p[4]=tmp; p[12] = -tmp;

    tmp = 0.5*(p[5] + p[13]) * coef;
    p[5]=tmp; p[13] = tmp;

    tmp = 0.5*(p[10] - p[14]) * coef;
    p[10]=tmp; p[14] = -tmp;

    tmp = 0.5*(p[11] + p[15]) * coef;
    p[11]=tmp; p[15] = tmp;

    IFloat c = 1./3. * (p[1] + p[9] + p[17]);

    p[1] = (p[1] - c) * coef;
    p[9] = (p[9] - c) * coef;
    p[17] = (p[17] - c) * coef;
}

static void thread_work_partial(int nwork, int me, int nthreads,
                                int &mywork, int &myoff)
{
    int basework = nwork / nthreads;
    int backfill = nthreads - (nwork % nthreads);
    mywork = (nwork + me) / nthreads;
    myoff  = basework * me;
    if ( me > backfill ) 
        myoff += (me-backfill);
}

ForceArg Fbfm::EvolveMomFforceInternal(Matrix *mom,
                                       Float *v1, Float *v2, // only internal data will be used
                                       Float coef, int mu,
                                       int nthreads)
{
    int low[4] = { 0, 0, 0, 0 };
    int high[4] = { lclx[0], lclx[1], lclx[2], lclx[3] };
    --high[mu];
    const int hl[4] = {high[0] - low[0],
                       high[1] - low[1],
                       high[2] - low[2],
                       high[3] - low[3] };
    const int hl_sites = hl[0] * hl[1] * hl[2] * hl[3];

    Matrix *gauge = GaugeField();

    int block_size = SPINOR_SIZE * lclx[4];

    int me = omp_get_thread_num();
    int mywork, myoff;
    // some threads are used in communication
    thread_work_partial(hl_sites, me, nthreads, mywork, myoff);

    ForceArg f_arg(0, 0, 0);
    for(int i = 0; i < mywork; ++i) {
        int x[4];
        compute_coord(x, hl, low, i + myoff);
        int off_4d = idx_4d(x, lclx);
        int gid = mu + 4 * off_4d;
        int fid = block_size * off_4d;

        int y[4] = {x[0], x[1], x[2], x[3]};
        ++y[mu];
        int fidp = block_size * idx_4d(y, lclx);

        Matrix force;
        FforceSiteS(force, gauge[gid],
                    v2 + fid, v2 + fidp,
                    v1 + fid, v1 + fidp, mu);
        trless_am((Float *)&force, -coef);
        // force.TrLessAntiHermMatrix();
        // force *= -coef;
        *(mom + gid) += force;
        updateForce(f_arg, force);
=======
    // we call base version just to revert the change, no need to
    // import to BFM in a destructor.

    Lattice::BondCond();

    bd.end();
    if(use_mixed_solver) {
	bf.comm_init(); //CK: As the comms have already been ended for bf, calling bf.end() causes a crash when it tries to deallocate the message handle that has already been deallocated. To prevent this I reinitialize the handle.
        bf.end();
>>>>>>> ckelly_latest
    }
<<<<<<< HEAD

    VRB.Result(cname, fname, "SetBfmArg: (Re)initing BFM objects from key mass %e)\n", key_mass);

    bfmarg new_arg = arg_map.at(key_mass);
    bfmarg kernel_arg = new_arg;
    kernel_arg.solver=DWFKernel;
    kernel_arg.Ls=1;

    if (!bfm_inited) {
	AutofillBfmarg(new_arg);
//	AutofillBfmarg(kernel_arg);
 
	bd.init(new_arg);
	if (use_mixed_solver) {
	    bd.comm_end();
	    bf.init(new_arg);
	    bf.comm_end();
	    bd.comm_init();
	}
#if 0
	bd.comm_end();
	kernel.init(kernel_arg);
	kernel.comm_end();
	bd.comm_init();
#endif

	ImportGauge();
	VRB.Result(cname, fname, "inited BFM objects with new BFM arg: solver = %d, mass = %e, Ls = %d, mobius_scale = %e\n", bd.solver, bd.mass, bd.Ls, bd.mobius_scale);
    } else {
	if (key_mass == current_key_mass) {
	    VRB.Result(cname, fname, "Already inited from desired key mass %e\n", key_mass);
	    return; // already inited with desired params
	}

        bool bad_change = false;
	if (bd.solver != new_arg.solver || bd.CGdiagonalMee != new_arg.CGdiagonalMee) {
          bad_change = true;
        } else if (bd.solver != WilsonTM) {
          if (bd.mobius_scale != new_arg.mobius_scale
	      || bd.Ls != new_arg.Ls
	      || bd.precon_5d != new_arg.precon_5d) {
            bad_change = true;
          }
        }
        if (bad_change) {
	    ERR.General(cname, fname, "Can't change solver, mobius_scale, Ls, precon_5d, or CGdiagonalMee "
                "during lifetime of Fbfm object: must destroy and recreate lattice object "
                "(solver=%d->%d, mobius_scale=%e->%e, Ls=%d->%d, precon_5d=%d->%d, CGdiagonalMee=%d->%d).\n", 
                bd.solver, new_arg.solver, bd.mobius_scale, new_arg.mobius_scale, bd.Ls, new_arg.Ls, 
                bd.precon_5d, new_arg.precon_5d, bd.CGdiagonalMee, new_arg.CGdiagonalMee);
	}

	if (new_arg.solver == WilsonTM) {
	    bd.mass = new_arg.mass;
	    bf.mass = new_arg.mass;
	    bd.twistedmass = new_arg.twistedmass;
	    bf.twistedmass = new_arg.twistedmass;
	} else {
	    SetMass(new_arg.mass);
	}
	VRB.Result(cname, fname, "Just set new mass %e for solver = %d, Ls = %d\n", bd.mass, bd.solver, bd.Ls);
    }

    bfm_inited = true;
    current_key_mass = key_mass;
||||||| merged common ancestors

    return f_arg;
}

ForceArg Fbfm::EvolveMomFforceSurface(Matrix *mom,
                                      Float *v1, Float *v2, // internal data
                                      Float *v1_s, Float *v2_s, // surface data
                                      Float coef, int mu)
{
    int low[4] = { 0, 0, 0, 0 };
    int high[4] = { lclx[0], lclx[1], lclx[2], lclx[3] };
    low[mu] = lclx[mu] - 1;
    high[mu] = lclx[mu];
    const int hl[4] = {high[0] - low[0],
                       high[1] - low[1],
                       high[2] - low[2],
                       high[3] - low[3] };
    const int hl_sites = hl[0] * hl[1] * hl[2] * hl[3];

    Matrix *gauge = GaugeField();

    int block_size = SPINOR_SIZE * lclx[4];
    int sign = GJP.NodeBc(mu) == BND_CND_APRD ? -1.0 : 1.0;

    int nthreads = omp_get_num_threads();
    int me = omp_get_thread_num();
    int mywork, myoff;

    // here all threads participate
    thread_work_partial(hl_sites, me, nthreads, mywork, myoff);

    ForceArg f_arg(0, 0, 0);
    for(int i = 0; i < mywork; ++i) {
        int x[4];
        compute_coord(x, hl, low, i + myoff);

        int off_4d = idx_4d(x, lclx);
        int gid = mu + 4 * off_4d;
        int fid = block_size * off_4d;
        int fid_s = block_size * idx_4d_surf(x, lclx, mu);

        Matrix force;
        FforceSiteS(force, gauge[gid],
                    v2 + fid, v2_s + fid_s,
                    v1 + fid, v1_s + fid_s, mu);
        trless_am((Float *)&force, -coef * sign);
        // force.TrLessAntiHermMatrix();
        // force *= -coef * sign;
        *(mom + gid) += force;
        updateForce(f_arg, force);
    }

    return f_arg;
}

// Calculate fermion force on a specific site, also do the
// summation over s direction.
void Fbfm::FforceSiteS(Matrix& force, Matrix &gauge,
                       Float *v1, Float *v1p,
                       Float *v2, Float *v2p, int mu)
{
    Matrix t1, t2;

    sproj_tr[mu](   (Float *)&t1, v1p, v2, lclx[4], 0, 0);
    sproj_tr[mu+4]( (Float *)&t2, v2p, v1, lclx[4], 0, 0);
    
    t1 += t2;

    force.DotMEqual(gauge, t1);
=======
>>>>>>> ckelly_latest
}

// This function differs from the original CalcHmdForceVecsBilinear()
// in that it stores v1 and v2 in (color, spin, s, x, y, z, t) order
// to facilitate force evaluation.
void Fbfm::CalcHmdForceVecsBilinear(Float *v1,
                                    Float *v2,
                                    Vector *phi1,
                                    Vector *phi2,
                                    Float mass, Float epsilon)
{
<<<<<<< HEAD
    SetBfmArg(mass);
||||||| merged common ancestors
    Fermion_t pi[2] = {bevo.allocFermion(), bevo.allocFermion()};
    Fermion_t po[4] = {bevo.allocFermion(), bevo.allocFermion(),
                       bevo.allocFermion(), bevo.allocFermion()};
=======
    Fermion_t pi[2] = {bd.allocFermion(), bd.allocFermion()};
    Fermion_t po[4] = {bd.allocFermion(), bd.allocFermion(),
                       bd.allocFermion(), bd.allocFermion()};
>>>>>>> ckelly_latest

<<<<<<< HEAD
    VRB.Result(cname, "CalcHmdForceVecsBilinear()", "bd.CGdiagonalMee = %d\n", bd.CGdiagonalMee);

    Fermion_t pi[2] = { bd.allocFermion(), bd.allocFermion() };
    Fermion_t po[4] = {bd.allocFermion(), bd.allocFermion(),
                       bd.allocFermion(), bd.allocFermion()};
    Fermion_t tmp = bd.allocFermion();

    bd.cps_impexcbFermion((Float *)phi1, pi[0], 1, 1);
    bd.cps_impexcbFermion((Float *)phi2, pi[1], 1, 1);
||||||| merged common ancestors
    bevo.mass = mass;
    // reinitialize since we are using a new mass.
    bevo.GeneralisedFiveDimEnd();
    bevo.GeneralisedFiveDimInit();

    bevo.cps_impexcbFermion((Float *)phi1, pi[0], 1, 1);
    bevo.cps_impexcbFermion((Float *)phi2, pi[1], 1, 1);
=======
    SetMass(mass, epsilon);
    bd.cps_impexcbFermion((Float *)phi1, pi[0], 1, 1);
    bd.cps_impexcbFermion((Float *)phi2, pi[1], 1, 1);
>>>>>>> ckelly_latest

#pragma omp parallel
    {
<<<<<<< HEAD
	// For CGdiagonalMee == 2 there is an extra factor of
	// Moo^{-1} in front of phi2_o
	if (bd.CGdiagonalMee == 2) {
	    bd.MooeeInv(pi[1], tmp, DaggerNo);
	    bd.copy(pi[1], tmp);
	}

	// For CGdiagonalMee == 1 there is an extra factor of
	// Moo^{\dag-1} in front of phi1_o
	if (bd.CGdiagonalMee == 1) {
	    bd.MooeeInv(pi[0], tmp, DaggerYes);
	    bd.copy(pi[0], tmp);
	}

        bd.calcMDForceVecs(po + 0, po + 2, pi[0], pi[1]);
||||||| merged common ancestors
        bevo.calcMDForceVecs(po + 0, po + 2, pi[0], pi[1]);
=======
        bd.calcMDForceVecs(po + 0, po + 2, pi[0], pi[1]);
>>>>>>> ckelly_latest
    }

    bd.cps_impexFermion_s(v1, po + 0, 0);
    bd.cps_impexFermion_s(v2, po + 2, 0);

<<<<<<< HEAD
    bd.freeFermion(pi[0]);
    bd.freeFermion(pi[1]);
    bd.freeFermion(po[0]);
    bd.freeFermion(po[1]);
    bd.freeFermion(po[2]);
    bd.freeFermion(po[3]);
    bd.freeFermion(tmp);
||||||| merged common ancestors
    bevo.freeFermion(pi[0]);
    bevo.freeFermion(pi[1]);
    bevo.freeFermion(po[0]);
    bevo.freeFermion(po[1]);
    bevo.freeFermion(po[2]);
    bevo.freeFermion(po[3]);
=======
    bd.freeFermion(pi[0]);
    bd.freeFermion(pi[1]);
    bd.freeFermion(po[0]);
    bd.freeFermion(po[1]);
    bd.freeFermion(po[2]);
    bd.freeFermion(po[3]);
>>>>>>> ckelly_latest
}

ForceArg Fbfm::EvolveMomFforceBaseThreaded(Matrix *mom,
                                           Vector *phi1, Vector *phi2,
                                           Float mass, Float coef)
{
    const char *fname = "EvolveMomFforceBaseThreaded()";

    Float dtime = -dclock();

<<<<<<< HEAD
    SetBfmArg(mass);
||||||| merged common ancestors
    Fermion_t in[2] = {bevo.allocFermion(), bevo.allocFermion()};
=======
    Fermion_t in[2] = {bd.allocFermion(), bd.allocFermion()};
>>>>>>> ckelly_latest

<<<<<<< HEAD
    Fermion_t in[2] = { bd.allocFermion(), bd.allocFermion() };

    bd.cps_impexcbFermion((Float *)phi1, in[0], 1, 1);
    bd.cps_impexcbFermion((Float *)phi2, in[1], 1, 1);
||||||| merged common ancestors
    bevo.mass = mass;
    // reinitialize since we are using a new mass.
    bevo.GeneralisedFiveDimEnd();
    bevo.GeneralisedFiveDimInit();
=======
    SetMass(mass,-12345);

    bd.cps_impexcbFermion((Float *)phi1, in[0], 1, 1);
    bd.cps_impexcbFermion((Float *)phi2, in[1], 1, 1);
>>>>>>> ckelly_latest

    Float *gauge = (Float *)(this->GaugeField());
#pragma omp parallel
    {
        bd.compute_force((Float *)mom, gauge, in[0], in[1], coef);
    }

    bd.freeFermion(in[0]);
    bd.freeFermion(in[1]);

    dtime += dclock();

    VRB.Result(cname, fname, "takes %17.10e seconds\n", dtime);
    return ForceArg();
}

// It evolves the canonical Momemtum mom:
// mom += coef * (phi1^\dag e_i(M) \phi2 + \phi2^\dag e_i(M^\dag) \phi1)
//
// NOTE:
//
// 1. This function does not exist in the base Lattice class.
//
// 2. The 2 auxiliary vectors v1 and v2 calculated by
// CalcHmdForceVecsBilinear must be in (reim, color, spin, s, x, y, z,
// t) order.
//
// 3. For BFM M is M = M_oo - M_oe M^{-1}_ee M_eo

//CK: when called below, phi1 = Mpc phi2
ForceArg Fbfm::EvolveMomFforceBase(Matrix *mom,
                                   Vector *phi1,
                                   Vector *phi2,
                                   Float mass, Float epsilon,
                                   Float coef)
{
    const char *fname = "EvolveMomFforceBase()";
<<<<<<< HEAD
    static Timer time(cname, fname);
    time.start(true);

    SetBfmArg(mass);
||||||| merged common ancestors
=======
    VRB.Result(cname,fname,"started\n");
>>>>>>> ckelly_latest

#if 0
    return EvolveMomFforceBaseThreaded(mom, phi1, phi2, mass, coef);
#endif

<<<<<<< HEAD
    long f_size = (long)SPINOR_SIZE * GJP.VolNodeSites() * bd.Ls;
||||||| merged common ancestors
    Float dtime1 = dclock();

    int f_size_4d = SPINOR_SIZE * GJP.VolNodeSites();
    int f_size = f_size_4d * GJP.SnodeSites();
  
=======
    long f_size = (long)SPINOR_SIZE * GJP.VolNodeSites() * Fbfm::bfm_args[current_arg_idx].Ls;
    if(GJP.Gparity()) f_size*=2;

>>>>>>> ckelly_latest
    Float *v1 = (Float *)smalloc(cname, fname, "v1", sizeof(Float) * f_size);
    Float *v2 = (Float *)smalloc(cname, fname, "v2", sizeof(Float) * f_size);

    CalcHmdForceVecsBilinear(v1, v2, phi1, phi2, mass, epsilon);

<<<<<<< HEAD
    FforceWilsonType cal_force(mom, this->GaugeField(),
	v1, v2, bd.Ls, coef);
    ForceArg ret = cal_force.run();
||||||| merged common ancestors
    Float dtime2 = dclock();

    for(int i = 0; i < 4; ++i) {
        CopySendFrmData(sndbuf + surf_v1[i], v1, i, true);
        CopySendFrmData(sndbuf + surf_v2[i], v2, i, true);
    }

    Float dtime3 = dclock();

    // single threaded comm
    for(int dir = 0; dir < 4; ++dir) {
        getPlusData(rcvbuf + surf_v1[dir], sndbuf + surf_v1[dir],
                    surf_size[dir] * 2, dir);
    }

    omp_set_num_threads(bfm_arg.threads);
    ForceArg ret;

#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        // int me = omp_get_thread_num();
        ForceArg f_arg; // threaded

        // internal forces
        for(int i = 0; i < 4; ++i) {
            ForceArg t = EvolveMomFforceInternal(mom, v1, v2, coef, i, nthreads);
            f_arg.combine(t);
        }

// #pragma omp barrier

        for(int i = 0; i < 4; ++i) {
            ForceArg t = EvolveMomFforceSurface(mom, v1, v2,
                                                rcvbuf + surf_v1[i], // v1 surface
                                                rcvbuf + surf_v2[i], // v2 surface
                                                coef, i);
            f_arg.combine(t);
        }

#pragma omp critical
        {
            ret.combine(f_arg);
        }
    }

    Float dtime4 = dclock();
=======
    FforceWilsonType cal_force(mom, this->GaugeField(),
                               v1, v2, Fbfm::bfm_args[current_arg_idx].Ls, coef);
    ForceArg ret = cal_force.run();
>>>>>>> ckelly_latest

    sfree(cname, fname, "v1", v1);
    sfree(cname, fname, "v2", v2);
<<<<<<< HEAD

    time.stop(true);
||||||| merged common ancestors
    sfree(cname, fname, "sndbuf", sndbuf);
    sfree(cname, fname, "rcvbuf", rcvbuf);

    VRB.Result(cname, fname, "cal aux vectors  : takes %17.10e seconds\n", dtime2 - dtime1);
    VRB.Result(cname, fname, "prepare for comm : takes %17.10e seconds\n", dtime3 - dtime2);
    VRB.Result(cname, fname, "comm/forces      : takes %17.10e seconds\n", dtime4 - dtime3);
    VRB.Result(cname, fname, "total            : takes %17.10e seconds\n", dtime4 - dtime1);

    glb_sum(&ret.L1);
    glb_sum(&ret.L2);
    glb_max(&ret.Linf);

    ret.unitarize(4 * GJP.VolSites());
=======
>>>>>>> ckelly_latest
    return ret;
}

//------------------------------------------------------------------
//! Multiplication of a lattice spin-colour vector by gamma_5.
//------------------------------------------------------------------
void Fbfm::Gamma5(Vector *v_out, Vector *v_in, int num_sites)
{
    Float *p_out = (Float *)v_out;
    Float *p_in  = (Float *)v_in;

    int half_site_size = 12 ;
    for (int site = 0; site < num_sites; ++site) {

        for(int comp = 0; comp < half_site_size; ++comp) {
            *p_out++ = *p_in++ ;
        }
        for(int comp = 0; comp < half_site_size; ++comp) {
            *p_out++ = -*p_in++ ;
        }
    }
}

// Sets the offsets for the fermion fields on a 
// checkerboard. The fermion field storage order
// is not the canonical one but it is particular
// to the Dwf fermion type. x[i] is the 
// ith coordinate where i = {0,1,2,3} = {x,y,z,t}.
int Fbfm::FsiteOffsetChkb(const int *x) const
{
    const char *fname = "FsiteOffsetChkb()";
    ERR.NotImplemented(cname, fname);
}

// Sets the offsets for the fermion fields on a 
// checkerboard. The fermion field storage order
// is the canonical one. X[I] is the
// ith coordinate where i = {0,1,2,3} = {x,y,z,t}.
#if 0
int Fbfm::FsiteOffset(const int *x) const
{
    const char *fname = "FsiteOffset()";
    ERR.NotImplemented(cname, fname);
}
<<<<<<< HEAD

||||||| merged common ancestors

// Returns the number of fermion field 
// components (including real/imaginary) on a
// site of the 4-D lattice.
int Fbfm::FsiteSize(void)const
{
    return 24 * GJP.SnodeSites();
}

// Returns 0 => If no checkerboard is used for the evolution
//      or the CG that inverts the evolution matrix.
int Fbfm::FchkbEvl(void)const
{
    return 1;
}

=======
#endif

>>>>>>> ckelly_latest
// It calculates f_out where A * f_out = f_in and
// A is the preconditioned fermion matrix that appears
// in the HMC evolution (even/odd preconditioning 
// of [Dirac^dag Dirac]). The inversion is done
// with the conjugate gradient. cg_arg is the structure
// that contains all the control parameters, f_in is the
// fermion field source vector, f_out should be set to be
// the initial guess and on return is the solution.
// f_in and f_out are defined on a checkerboard.
// If true_res !=0 the value of the true residual is returned
// in true_res.
// *true_res = |src - MatPcDagMatPc * sol| / |src|
// The function returns the total number of CG iterations.
int Fbfm::FmatEvlInv(Vector *f_out, Vector *f_in,
    CgArg *cg_arg,
    Float *true_res,
    CnvFrmType cnv_frm)
{
<<<<<<< HEAD
    const char *fname = "FmatEvlInv()";
||||||| merged common ancestors
    const char *fname = "FmatEvlInv(V*, V*, CgArg *, ...)";

    if(cg_arg == NULL)
        ERR.Pointer(cname, fname, "cg_arg");

    if(use_mixed_solver) {
        return FmatEvlInvMixed(f_out, f_in, cg_arg, 1e-5,
                               cg_arg->max_num_iter,
                               5);
    }

    Fermion_t in  = bevo.allocFermion();
    Fermion_t out = bevo.allocFermion();

    bevo.mass = cg_arg->mass;
    bevo.residual = cg_arg->stop_rsd;
    bevo.max_iter = cg_arg->max_num_iter;
    // reinitialize since we are using a new mass.
    bevo.GeneralisedFiveDimEnd();
    bevo.GeneralisedFiveDimInit();

    bevo.cps_impexcbFermion((Float *)f_in , in,  1, 1);
    bevo.cps_impexcbFermion((Float *)f_out, out, 1, 1);

    int iter;
#pragma omp parallel
    {
        iter = bevo.CGNE_prec_MdagM(out, in);
    }

    bevo.cps_impexcbFermion((Float *)f_out, out, 0, 1);

    bevo.freeFermion(in);
    bevo.freeFermion(out);

    return iter;
}

int Fbfm::FmatEvlInvMixed(Vector *f_out, Vector *f_in, 
                          CgArg *cg_arg,
                          Float single_rsd,
                          int max_iter,
                          int max_cycle)
{
    bfm_arg.mass = cg_arg->mass;

    bfm_evo<float> bfm_f;
    bfm_f.init(bfm_arg);
    bfm_f.residual = single_rsd;
    bfm_f.max_iter = max_iter;

    Float *gauge = (Float *)(this->GaugeField());
    BondCond(gauge);
    bfm_f.cps_importGauge(gauge);
    BondCond(gauge);

    bfm_f.comm_end();
    bevo.comm_init();

    bevo.mass = cg_arg->mass;
    bevo.residual = cg_arg->stop_rsd;
    bevo.max_iter = cg_arg->max_num_iter;
    // reinitialize since we are using a new mass.
    bevo.GeneralisedFiveDimEnd();
    bevo.GeneralisedFiveDimInit();

    Fermion_t src = bevo.allocFermion();
    Fermion_t sol = bevo.allocFermion();

    bevo.cps_impexcbFermion((Float *)f_in , src, 1, 1);
    bevo.cps_impexcbFermion((Float *)f_out, sol, 1, 1);

=======
    const char *fname = "FmatEvlInv()";

    if(cg_arg == NULL)
        ERR.Pointer(cname, fname, "cg_arg");

    Fermion_t in  = bd.allocFermion();
    Fermion_t out = bd.allocFermion();

    SetMass(cg_arg->mass, cg_arg->epsilon);
    bd.residual = cg_arg->stop_rsd;
    bd.max_iter = bf.max_iter = cg_arg->max_num_iter;
    // FIXME: pass single precision rsd in a reasonable way.
    bf.residual = 1e-5;

    bd.cps_impexcbFermion((Float *)f_in , in,  1, 1);
    bd.cps_impexcbFermion((Float *)f_out, out, 1, 1);

>>>>>>> ckelly_latest
    int iter = -1;

    static Timer timer(cname, fname);
    static std::map<Float, Timer*> timers;
    if (timers.count(cg_arg->mass) == 0) {
	char timer_mass_name[512];
	sprintf(timer_mass_name, "FmatEvlInv(mass=%0.4f)", cg_arg->mass);
	timers[cg_arg->mass] = new Timer(cname, timer_mass_name);
    }
    timer.start(true);
    timers[cg_arg->mass]->start(true);

    SetBfmArg(cg_arg->mass);

    VRB.Result(cname, fname, "target residual = %e\n", cg_arg->stop_rsd);

    if (cg_arg == NULL)
	ERR.Pointer(cname, fname, "cg_arg");

    Fermion_t in = bd.allocFermion();
    Fermion_t out = bd.allocFermion();

    bd.residual = cg_arg->stop_rsd;
    bd.max_iter = bf.max_iter = cg_arg->max_num_iter;
    // FIXME: pass single precision rsd in a reasonable way.
    bf.residual = 1e-5;

    bd.cps_impexcbFermion((Float *)f_in, in, 1, 1);
    bd.cps_impexcbFermion((Float *)f_out, out, 1, 1);

#pragma omp parallel
    {
<<<<<<< HEAD
	iter = use_mixed_solver ?
	    mixed_cg::threaded_cg_mixed_MdagM(out, in, bd, bf, 5) :
	    bd.CGNE_prec_MdagM(out, in);
||||||| merged common ancestors
        iter = mixed_cg::threaded_cg_mixed_MdagM(sol, src, bevo, bfm_f, max_cycle);

        // bevo.max_iter = 20;
        // iter = mixed_cg::cg_MdagM_single_precnd(sol, src, bevo, bfm_f);
        // bevo.max_iter = cg_arg->max_num_iter;
=======
        iter =
            use_mixed_solver 
            ? mixed_cg::threaded_cg_mixed_MdagM(out, in, bd, bf, 5)
            : bd.CGNE_prec_MdagM(out, in);
>>>>>>> ckelly_latest
    }

    bd.cps_impexcbFermion((Float *)f_out, out, 0, 1);

<<<<<<< HEAD
    bd.freeFermion(in);
    bd.freeFermion(out);

    timers[cg_arg->mass]->stop(true);
    timer.stop(true);
||||||| merged common ancestors
    bevo.cps_impexcbFermion((Float *)f_out, sol, 0, 1);

    bevo.freeFermion(src);
    bevo.freeFermion(sol);
=======
    bd.freeFermion(in);
    bd.freeFermion(out);
>>>>>>> ckelly_latest

    return iter;
}

<<<<<<< HEAD
int Fbfm::FmatEvlMInv(Vector **f_out, Vector *f_in, Float *shift,
    int Nshift, int isz, CgArg **cg_arg,
    CnvFrmType cnv_frm, MultiShiftSolveType type, Float *alpha,
    Vector **f_out_d)
||||||| merged common ancestors
int Fbfm::FmatEvlInv(Vector *f_out, Vector *f_in, 
                     CgArg *cg_arg, 
                     CnvFrmType cnv_frm)
{
    return FmatEvlInv(f_out, f_in, cg_arg, NULL, cnv_frm);
}

int Fbfm::FmatEvlMInv(Vector **f_out, Vector *f_in, Float *shift, 
                      int Nshift, int isz, CgArg **cg_arg, 
                      CnvFrmType cnv_frm, MultiShiftSolveType type, Float *alpha,
                      Vector **f_out_d)
=======
int Fbfm::FmatEvlMInv(Vector **f_out, Vector *f_in, Float *shift, 
                      int Nshift, int isz, CgArg **cg_arg, 
                      CnvFrmType cnv_frm, MultiShiftSolveType type, Float *alpha,
                      Vector **f_out_d)
>>>>>>> ckelly_latest
{
    const char *fname = "FmatEvlMInv(V*,V*,F*, ...)";

    if (isz != 0) {
	ERR.General(cname, fname, "Non-zero isz is not implemented.\n");
    }

    SetBfmArg(cg_arg[0]->mass);

    Fermion_t *sol_multi = new Fermion_t[Nshift];
    double *ones = new double[Nshift];
    double *mresidual = new double[Nshift];
<<<<<<< HEAD
    for (int i = 0; i < Nshift; ++i) {
	sol_multi[i] = bd.allocFermion();
	ones[i] = 1.0;
	mresidual[i] = cg_arg[i]->stop_rsd;
||||||| merged common ancestors
    for(int i = 0; i < Nshift; ++i) {
        sol_multi[i] = bevo.allocFermion();
        ones[i] = 1.0;
        mresidual[i] = cg_arg[i]->stop_rsd;
=======
    for(int i = 0; i < Nshift; ++i) {
        sol_multi[i] = bd.allocFermion();
        ones[i] = 1.0;
        mresidual[i] = cg_arg[i]->stop_rsd;
>>>>>>> ckelly_latest
    }

    // source
    Fermion_t src = bd.allocFermion();
    bd.cps_impexcbFermion((Float *)f_in, src, 1, 1);

<<<<<<< HEAD
    bd.residual = cg_arg[0]->stop_rsd;
    bd.max_iter = cg_arg[0]->max_num_iter;
||||||| merged common ancestors
    // reinitialize since we are using a new mass.
    bevo.mass = cg_arg[0]->mass;
    bevo.residual = cg_arg[0]->stop_rsd;
    bevo.max_iter = cg_arg[0]->max_num_iter;
    bevo.GeneralisedFiveDimEnd();
    bevo.GeneralisedFiveDimInit();
=======
    SetMass(cg_arg[0]->mass, cg_arg[0]->epsilon);
>>>>>>> ckelly_latest

    int iter;
<<<<<<< HEAD
    if (use_mixed_solver && bd.solver != WilsonTM) {
	MultiShiftController.MInv(sol_multi, src, shift, Nshift, mresidual, ones, 0, bd, bf);
    } else {
||||||| merged common ancestors
=======
    //If use_mixed_solver we use the MultiShiftController instance, otherwise just do it in double precision
    if(use_mixed_solver){
	bd.max_iter = cg_arg[0]->max_num_iter;
	bf.max_iter = cg_arg[0]->max_num_iter;
	iter = MultiShiftController.MInv(sol_multi, src, shift, Nshift, mresidual, ones, 0, bd, bf);
    }
    else{
	bd.residual = cg_arg[0]->stop_rsd;
	bd.max_iter = cg_arg[0]->max_num_iter;
	    
>>>>>>> ckelly_latest
#pragma omp parallel
	{
	    iter = bd.CGNE_prec_MdagM_multi_shift(sol_multi, src, shift, ones, Nshift, mresidual, 0);
	}
    }

<<<<<<< HEAD
    if (type == SINGLE) {
	// FIXME
	int f_size_cb = GJP.VolNodeSites() * SPINOR_SIZE * bd.Ls / 2;
	Vector *t = (Vector *)smalloc(cname, fname, "t", sizeof(Float) * f_size_cb);
||||||| merged common ancestors
    if(type == SINGLE) {
        if(1) {
            // FIXME: Never use this in production code!
            int f_size_cb = GJP.VolNodeSites() * SPINOR_SIZE * GJP.SnodeSites() / 2;
            Vector *t = (Vector *)smalloc(cname, fname, "t", sizeof(Float) * f_size_cb);
=======
    //Combine solutions if in SINGLE mode
    if(type == SINGLE) {
        // FIXME
        int f_size_cb = GJP.VolNodeSites() * SPINOR_SIZE * Fbfm::bfm_args[current_arg_idx].Ls / 2;
	if(GJP.Gparity()) f_size_cb *=2;
        Vector *t = (Vector *)smalloc(cname, fname, "t", sizeof(Float) * f_size_cb);
>>>>>>> ckelly_latest

<<<<<<< HEAD
	for (int i = 0; i < Nshift; ++i) {
	    bd.cps_impexcbFermion((Float *)t, sol_multi[i], 0, 1);
	    f_out[0]->FTimesV1PlusV2(alpha[i], t, f_out[0], f_size_cb);
	}
	sfree(cname, fname, "t", t);
||||||| merged common ancestors
            for(int i = 0; i < Nshift; ++i) {
                bevo.cps_impexcbFermion((Float *)t, sol_multi[i], 0, 1);
                f_out[0]->FTimesV1PlusV2(alpha[i], t, f_out[0], f_size_cb);
            }
            sfree(cname, fname, "t", t);
        } else {
            // Can't do this since axpy needs a threaded environment (and also
            // other bfm related problems, mainly how it uses alpha now).
            // bevo.cps_impexcbFermion((Float *)f_out[0], bevo.psi_i[0], 1, 1);
            // for(int i = 0; i < Nshift; ++i) {
            //     bevo.axpy(bevo.psi_i[0],
            //               bevo.psi_multi[i],
            //               bevo.psi_i[0],
            //               alpha[i]);
            // }
            // bevo.axpy(bevo.psi_i[0],
            //           bevo.psi_multi[0],
            //           bevo.psi_i[0],
            //           1.0);
            // bevo.cps_impexcbFermion((Float *)f_out[0], bevo.psi_i[0], 0, 1);
        }
=======
        for(int i = 0; i < Nshift; ++i) {
            bd.cps_impexcbFermion((Float *)t, sol_multi[i], 0, 1);
            f_out[0]->FTimesV1PlusV2(alpha[i], t, f_out[0], f_size_cb);
        }
        sfree(cname, fname, "t", t);
>>>>>>> ckelly_latest
    } else {
<<<<<<< HEAD
	for (int i = 0; i < Nshift; ++i) {
	    bd.cps_impexcbFermion((Float *)f_out[i], sol_multi[i], 0, 1);
	}
||||||| merged common ancestors
        for(int i = 0; i < Nshift; ++i) {
            bevo.cps_impexcbFermion((Float *)f_out[i], sol_multi[i], 0, 1);
        }
=======
        for(int i = 0; i < Nshift; ++i) {
            bd.cps_impexcbFermion((Float *)f_out[i], sol_multi[i], 0, 1);
        }
>>>>>>> ckelly_latest
    }

<<<<<<< HEAD
    bd.freeFermion(src);
    for (int i = 0; i < Nshift; ++i) {
	bd.freeFermion(sol_multi[i]);
||||||| merged common ancestors
    bevo.freeFermion(src);
    for(int i = 0; i < Nshift; ++i) {
        bevo.freeFermion(sol_multi[i]);
=======
    bd.freeFermion(src);
    for(int i = 0; i < Nshift; ++i) {
        bd.freeFermion(sol_multi[i]);
>>>>>>> ckelly_latest
    }
    
    delete[] sol_multi;
    delete[] ones;
    delete[] mresidual;

    return iter;
}

void Fbfm::FminResExt(Vector *sol, Vector *source, Vector **sol_old,
    Vector **vm, int degree, CgArg *cg_arg, CnvFrmType cnv_frm)
{
    const char *fname = "FminResExt(V*, V*, V**, ...)";

<<<<<<< HEAD
    SetBfmArg(cg_arg->mass);
||||||| merged common ancestors
    int f_size_cb = GJP.VolNodeSites() * SPINOR_SIZE * GJP.SnodeSites() / 2;
=======
    int f_size_cb = GJP.VolNodeSites() * SPINOR_SIZE * Fbfm::bfm_args[current_arg_idx].Ls / 2;
    if(GJP.Gparity()) f_size_cb *=2;
>>>>>>> ckelly_latest

    // does nothing other than setting sol to zero
    int f_size_cb = GJP.VolNodeSites() * SPINOR_SIZE * bd.Ls / 2;
    sol->VecZero(f_size_cb);
}
    
// It calculates f_out where A * f_out = f_in and
// A is the fermion matrix (Dirac operator). The inversion
// is done with the conjugate gradient. cg_arg is the 
// structure that contains all the control parameters, f_in 
// is the fermion field source vector, f_out should be set 
// to be the initial guess and on return is the solution.
// f_in and f_out are defined on the whole lattice.
// If true_res !=0 the value of the true residual is returned
// in true_res.
// *true_res = |src - MatPcDagMatPc * sol| / |src|
// cnv_frm is used to specify if f_in should be converted 
// from canonical to fermion order and f_out from fermion 
// to canonical. 
// prs_f_in is used to specify if the source
// f_in should be preserved or not. If not the memory usage
// is less by half the size of a fermion vector.
// The function returns the total number of CG iterations.
int Fbfm::FmatInv(Vector *f_out, Vector *f_in,
<<<<<<< HEAD
    CgArg *cg_arg,
    Float *true_res,
    CnvFrmType cnv_frm,
    PreserveType prs_f_in)
||||||| merged common ancestors
                  CgArg *cg_arg,
                  Float *true_res,
                  CnvFrmType cnv_frm,
                  PreserveType prs_f_in)
=======
                  CgArg *cg_arg,
                  Float *true_res,
                  CnvFrmType cnv_frm,
                  PreserveType prs_f_in, int if_dminus)
>>>>>>> ckelly_latest
{
    const char *fname = "FmatInv()";
    VRB.Func(cname, fname);

    if (cg_arg == NULL)
	ERR.Pointer(cname, fname, "cg_arg");
    int threads = omp_get_max_threads();

<<<<<<< HEAD
    SetBfmArg(cg_arg->mass);
||||||| merged common ancestors
    Fermion_t in[2]  = {bevo.allocFermion(), bevo.allocFermion()};
    Fermion_t out[2] = {bevo.allocFermion(), bevo.allocFermion()};
=======
    Fermion_t in[2]  = {bd.allocFermion(), bd.allocFermion()};
    Fermion_t out[2] = {bd.allocFermion(), bd.allocFermion()};
>>>>>>> ckelly_latest

<<<<<<< HEAD
    Fermion_t in[2] = { bd.allocFermion(), bd.allocFermion() };
    Fermion_t out[2] = { bd.allocFermion(), bd.allocFermion() };
||||||| merged common ancestors
    bevo.mass = cg_arg->mass;
    bevo.residual = cg_arg->stop_rsd;
    bevo.max_iter = cg_arg->max_num_iter;
    // reinitialize since we are using a new mass.
    bevo.GeneralisedFiveDimEnd();
    bevo.GeneralisedFiveDimInit();
=======
    SetMass(cg_arg->mass, cg_arg->epsilon);
    bd.residual = cg_arg->stop_rsd;
    bd.max_iter = bf.max_iter = cg_arg->max_num_iter;
    // FIXME: pass single precision rsd in a reasonable way.
    bf.residual = 1e-5;
>>>>>>> ckelly_latest

<<<<<<< HEAD
    bd.residual = cg_arg->stop_rsd;
    bd.max_iter = bf.max_iter = cg_arg->max_num_iter;
    // FIXME: pass single precision rsd in a reasonable way.
    bf.residual = 1e-5;
||||||| merged common ancestors
    bevo.cps_impexFermion((Float *)f_in , in,  1);
    bevo.cps_impexFermion((Float *)f_out, out, 1);
=======
    // deal with Mobius Dminus
    if(if_dminus && bd.solver == HmCayleyTanh) {
        bd.cps_impexFermion((Float *)f_in , out,  1);
#pragma omp parallel
        {
            bd.G5D_Dminus(out, in, 0);
        }
    } else {
        bd.cps_impexFermion((Float *)f_in , in,  1);
    }
>>>>>>> ckelly_latest

<<<<<<< HEAD
    // deal with Mobius Dminus
    if (bd.solver == HmCayleyTanh) {
	bd.cps_impexFermion((Float *)f_in, out, 1);
||||||| merged common ancestors
    int iter;
=======
    bd.cps_impexFermion((Float *)f_out, out, 1);

    int iter = -1;
>>>>>>> ckelly_latest
#pragma omp parallel
<<<<<<< HEAD
	{
	    bd.G5D_Dminus(out, in, 0);
	}
    } else {
	bd.cps_impexFermion((Float *)f_in, in, 1);
||||||| merged common ancestors
    {
        iter = bevo.prop_solve(out, in);
=======
    {
        if(use_mixed_solver) {
            iter = mixed_cg::threaded_cg_mixed_M(out, in, bd, bf, 5, cg_arg->Inverter, evec, evalf, ecnt);
        } else {
            switch(cg_arg->Inverter) {
            case CG:
                if(evec && evald && ecnt) {
                    iter = bd.CGNE_M(out, in, *evec, *evald);
                } else {
                    iter = bd.CGNE_M(out, in);
                }
                break;
            case EIGCG:
                iter = bd.EIG_CGNE_M(out, in);
                break;
            default:
                if(bd.isBoss()) {
                    printf("%s::%s: Not implemented\n", cname, fname);
                }
                exit(-1);
                break;
            }
        }
>>>>>>> ckelly_latest
    }

<<<<<<< HEAD
    bd.cps_impexFermion((Float *)f_out, out, 1);
||||||| merged common ancestors
    bevo.cps_impexFermion((Float *)f_out, out, 0);
=======
    bd.cps_impexFermion((Float *)f_out, out, 0);
>>>>>>> ckelly_latest

<<<<<<< HEAD
    int iter = -1;

    if (madwf_arg_map.count(cg_arg->mass) > 0) {
	// MADWF inversion
	VRB.Result(cname, fname, "Using MADWF: Main Ls = %d, cheap approx Ls = %d.\n", bd.Ls, madwf_arg_map[cg_arg->mass].cheap_approx.Ls);

	iter = MADWF_CG_M(bd, bf, use_mixed_solver,
	    out, in, bd.mass, this->GaugeField(), cg_arg->stop_rsd, madwf_arg_map[cg_arg->mass], cg_arg->Inverter);
    } else if (cg_arg->Inverter == HDCG) {
	HDCG_wrapper *control = HDCGInstance::getInstance();
	assert(control != NULL);
	control->HDCG_set_mass(cg_arg->mass);
	control->HDCG_invert(out, in, cg_arg->stop_rsd, cg_arg->max_num_iter);
    } else {
	// no MADWF:
#pragma omp parallel
	{
	    if (use_mixed_solver) {
		iter = mixed_cg::threaded_cg_mixed_M(out, in, bd, bf, 5, cg_arg->Inverter, evec, evalf, ecnt);
	    } else {
		switch (cg_arg->Inverter) {
		    case CG:
			if (evec && evald && ecnt) {
			    iter = bd.CGNE_M(out, in, *evec, *evald);
			} else {
			    iter = bd.CGNE_M(out, in);
			}
			break;
		    case EIGCG:
			iter = bd.EIG_CGNE_M(out, in);
			break;
		    default:
			if (bd.isBoss()) {
			    printf("%s::%s: Not implemented\n", cname, fname);
			}
			exit(-1);
			break;
		}
	    }
	}
    }


    bd.cps_impexFermion((Float *)f_out, out, 0);

    bd.freeFermion(in[0]);
    bd.freeFermion(in[1]);
    bd.freeFermion(out[0]);
    bd.freeFermion(out[1]);
||||||| merged common ancestors
    bevo.freeFermion(in[0]);
    bevo.freeFermion(in[1]);
    bevo.freeFermion(out[0]);
    bevo.freeFermion(out[1]);
=======
    bd.freeFermion(in[0]);
    bd.freeFermion(in[1]);
    bd.freeFermion(out[0]);
    bd.freeFermion(out[1]);
>>>>>>> ckelly_latest

    return iter;
}

//!< Transforms a 4-dimensional fermion field into a 5-dimensional field.
/* The 5d field is zero */
// The 5d field is zero
// except for the upper two components (right chirality)
// at s = s_u which are equal to the ones of the 4d field
// and the lower two components (left chirality) 
// at s_l, which are equal to the ones of the 4d field
// For spread-out DWF s_u, s_l refer to the global
// s coordinate i.e. their range is from 
// 0 to [GJP.Snodes() * GJP.SnodeSites() - 1]
void Fbfm::Ffour2five(Vector *five, Vector *four, int s_u, int s_l, int Ncb)
{
    const char *fname = "Ffour2five(V*, V*, ...)";

    // Note: we don't allow splitting in s direction
    if(GJP.Snodes() != 1) {
        ERR.NotImplemented(cname, fname);
    }
    // what does Ncb do?
    if(Ncb != 2) {
        ERR.NotImplemented(cname, fname);
    }

    Float *f5d = (Float *)five;
    Float *f4d = (Float *)four;

<<<<<<< HEAD
    const int size_4d = GJP.VolNodeSites() * SPINOR_SIZE;
    VRB.Result(cname, fname, "Taking Ls from current_key_mass = %e!\n", current_key_mass);
    const int size_5d = size_4d * arg_map.at(current_key_mass).Ls; // current_key_mass must be set correctly!!!
||||||| merged common ancestors
    //------------------------------------------------------------------
    // Set *five using the 4D field *four. 
    //------------------------------------------------------------------
=======
    int size_4d = GJP.VolNodeSites() * SPINOR_SIZE;
    if(GJP.Gparity()) size_4d *= 2;
    const int size_5d = size_4d * Fbfm::bfm_args[current_arg_idx].Ls;
>>>>>>> ckelly_latest

    // zero 5D vector
#pragma omp parallel for
    for(int i=0; i< size_5d; ++i) {
        f5d[i]  = 0.0;
    }

<<<<<<< HEAD
    Float *f4du = f4d;                      
    Float *f4dl = f4d + 12;                 
    Float *f5du = f5d + s_u * size_4d;
    Float *f5dl = f5d + s_l * size_4d + 12;
||||||| merged common ancestors
    // Do the two upper spin components if s_u is in the node
    //---------------------------------------------------------------
    if( s_u_node == GJP.SnodeCoor() ){
        field_4D  = (Float *) four;
        field_5D  = (Float *) five;
        field_5D  = field_5D  + s_u_local * ls_stride;
        for(x=0; x<vol_4d; x++){
            for(i=0; i<12; i++){
                field_5D[i]  = field_4D[i];
            }
            field_4D  = field_4D  + 24;
            field_5D  = field_5D  + 24;
        }
    }
=======
    Float *f4du = f4d;
    Float *f4dl = f4d + 12;
    Float *f5du = f5d + s_u * size_4d;
    Float *f5dl = f5d + s_l * size_4d + 12;
>>>>>>> ckelly_latest

#pragma omp parallel for
    for(int x = 0; x < size_4d; x += SPINOR_SIZE) {
        memcpy(f5du + x, f4du + x, sizeof(Float) * 12);
        memcpy(f5dl + x, f4dl + x, sizeof(Float) * 12);
    }
}

//!< Transforms a 5-dimensional fermion field into a 4-dimensional field.
//The 4d field has
// the upper two components (right chirality) equal to the
// ones of the 5d field at s = s_u and the lower two 
// components (left chirality) equal to the
// ones of the 5d field at s = s_l, where s is the 
// coordinate in the 5th direction.
// For spread-out DWF s_u, s_l refer to the global
// s coordinate i.e. their range is from 
// 0 to [GJP.Snodes() * GJP.SnodeSites() - 1]
// The same 4D field is generarted in all s node slices.
void Fbfm::Ffive2four(Vector *four, Vector *five, int s_u, int s_l, int Ncb)
{
    const char *fname = "Ffive2four(V*,V*,i,i)";

    // Note: we don't allow splitting in s direction
    if(GJP.Snodes() != 1) {
        ERR.NotImplemented(cname, fname);
    }
    // what does Ncb do?
    if(Ncb != 2) {
        ERR.NotImplemented(cname, fname);
    }

    Float *f5d = (Float *)five;
    Float *f4d = (Float *)four;

<<<<<<< HEAD
    const int size_4d = GJP.VolNodeSites() * SPINOR_SIZE;
||||||| merged common ancestors
    // Set all components of the 4D field to zero.
    //---------------------------------------------------------------
    field_4D  = (Float *) four;
    for(i=0; i<f_size; i++){
        field_4D[i]  = 0.0;
    }
=======
    int size_4d = GJP.VolNodeSites() * SPINOR_SIZE;
    if(GJP.Gparity()) size_4d *= 2;
>>>>>>> ckelly_latest

    // zero 4D vector
#pragma omp parallel for
    for(int i=0; i< size_4d; ++i) {
        f4d[i]  = 0.0;
    }

    Float *f4du = f4d;
    Float *f4dl = f4d + 12;
    Float *f5du = f5d + s_u * size_4d;
    Float *f5dl = f5d + s_l * size_4d + 12;

#pragma omp parallel for
    for(int x = 0; x < size_4d; x += SPINOR_SIZE) {
        memcpy(f4du + x, f5du + x, sizeof(Float) * 12);
        memcpy(f4dl + x, f5dl + x, sizeof(Float) * 12);
    }
}

// It finds the eigenvectors and eigenvalues of A where
// A is the fermion matrix (Dirac operator). The solution
// uses Ritz minimization. eig_arg is the 
// structure that contains all the control parameters, f_eigenv
// are the fermion field source vectors which should be
// defined initially, lambda are the eigenvalues returned 
// on solution. f_eigenv is defined on the whole lattice.
// The function returns the total number of Ritz iterations.
int Fbfm::FeigSolv(Vector **f_eigenv, Float *lambda,
                   Float *chirality, int *valid_eig,
                   Float **hsum,
                   EigArg *eig_arg, 
                   CnvFrmType cnv_frm)
{
    const char *fname = "FeigSolv(EigArg*,V*,F*,CnvFrmType)";

    // only 1 eigenvalue can be computed now.
    if(eig_arg->N_eig != 1) {
        ERR.NotImplemented(cname, fname);
    }
    if(eig_arg->RitzMatOper != MATPCDAG_MATPC &&
       eig_arg->RitzMatOper != NEG_MATPCDAG_MATPC) {
        ERR.NotImplemented(cname, fname);
    }
    
<<<<<<< HEAD
    SetBfmArg(eig_arg->mass);
    bd.residual = eig_arg->Rsdlam;
    bd.max_iter = eig_arg->MaxCG;
||||||| merged common ancestors
    bevo.residual = eig_arg->Rsdlam;
    bevo.max_iter = eig_arg->MaxCG;
    bevo.mass = eig_arg->mass;
    // reinitialize since we are using a new mass.
    bevo.GeneralisedFiveDimEnd();
    bevo.GeneralisedFiveDimInit();
=======
    SetMass(eig_arg->mass, eig_arg->epsilon);
    bd.residual = eig_arg->Rsdlam;
    bd.max_iter = eig_arg->MaxCG;
>>>>>>> ckelly_latest

    VRB.Result(cname, fname, "residual = %17.10e max_iter = %d mass = %17.10e\n",
<<<<<<< HEAD
               bd.residual, bd.max_iter, bd.mass);
#if 0
    if( eig_arg->RitzMatOper == MATPCDAG_MATPC) 
{
    Fermion_t x[2];
    x[0] = bd.allocFermion();
    x[1] = bd.allocFermion();
    LatVector random0(4*GJP.SnodeSites());
    LatVector random1(4*GJP.SnodeSites());
    RandGaussVector(random0.Vec(),0.5,1);
    RandGaussVector(random1.Vec(),0.5,1);
//    Float * f_tmp = (Float *)f_eigenv[0];
    bd.cps_impexcbFermion(random0.Field(), x[0], 1, 1);
//    f_tmp += (GJP.VolNodeSites()/2)*(4*3*2);// checkerboarded 4D volume
    bd.cps_impexcbFermion(random1.Field(), x[1], 1, 1);
||||||| merged common ancestors
               bevo.residual, bevo.max_iter, bevo.mass);
=======
               bd.residual, bd.max_iter, bd.mass);
>>>>>>> ckelly_latest

<<<<<<< HEAD
#pragma omp parallel
    {
        lambda[0] = bd.simple_lanczos(x);
    }

    bd.freeFermion(x[0]);
    bd.freeFermion(x[1]);
}
#endif

    Fermion_t in = bd.allocFermion();
    bd.cps_impexcbFermion((Float *)f_eigenv[0], in, 1, 1);
||||||| merged common ancestors
    Fermion_t in = bevo.allocFermion();
    bevo.cps_impexcbFermion((Float *)f_eigenv[0], in, 1, 1);
=======
    Fermion_t in = bd.allocFermion();
    bd.cps_impexcbFermion((Float *)f_eigenv[0], in, 1, 1);
>>>>>>> ckelly_latest

#pragma omp parallel
    {
        lambda[0] = bd.ritz(in, eig_arg->RitzMatOper == MATPCDAG_MATPC);
    }

<<<<<<< HEAD

    bd.cps_impexcbFermion((Float *)f_eigenv[0], in, 0, 1);
||||||| merged common ancestors
    bevo.cps_impexcbFermion((Float *)f_eigenv[0], in, 0, 1);
=======
    bd.cps_impexcbFermion((Float *)f_eigenv[0], in, 0, 1);
>>>>>>> ckelly_latest

    // correct the eigenvalue for a dumb convention problem.
    if(eig_arg->RitzMatOper == NEG_MATPCDAG_MATPC) lambda[0] = -lambda[0];

    valid_eig[0] = 1;
    bd.freeFermion(in);

    return 0;
}

// It sets the pseudofermion field phi from frm1, frm2.
Float Fbfm::SetPhi(Vector *phi, Vector *frm1, Vector *frm2,	       
                   Float mass, Float epsilon, DagType dag)
{
    const char *fname = "SetPhi(V*,V*,V*,F)";

    if (phi == 0)
        ERR.Pointer(cname,fname,"phi") ;

    if (frm1 == 0)
        ERR.Pointer(cname,fname,"frm1") ;

<<<<<<< HEAD
    SetBfmArg(mass);

    MatPc(phi, frm1, mass, dag);
    Float ret = FhamiltonNode(frm1, frm1);
    return ret;
||||||| merged common ancestors
    MatPc(phi, frm1, mass, dag);
    return FhamiltonNode(frm1, frm1);
=======
    MatPc(phi, frm1, mass, epsilon, dag);
    return FhamiltonNode(frm1, frm1);
>>>>>>> ckelly_latest
}

void Fbfm::MatPc(Vector *out, Vector *in, Float mass, Float epsilon, DagType dag)
{
    const char *fname = "MatPc()";

<<<<<<< HEAD
    VRB.Result(cname, fname, "start MatPc: mass = %e\n", mass);
||||||| merged common ancestors
    Fermion_t i = bevo.allocFermion();
    Fermion_t o = bevo.allocFermion();
    Fermion_t t = bevo.allocFermion();
=======
    Fermion_t i = bd.allocFermion();
    Fermion_t o = bd.allocFermion();
    Fermion_t t = bd.allocFermion();
>>>>>>> ckelly_latest

<<<<<<< HEAD
    SetBfmArg(mass);
||||||| merged common ancestors
    bevo.mass = mass;
    // reinitialize since we are using a new mass.
    bevo.GeneralisedFiveDimEnd();
    bevo.GeneralisedFiveDimInit();
=======
    SetMass(mass, epsilon);
>>>>>>> ckelly_latest

<<<<<<< HEAD
    Fermion_t i = bd.allocFermion();
    Fermion_t o = bd.allocFermion();
    Fermion_t t = bd.allocFermion();

    bd.cps_impexcbFermion((Float *)in , i, 1, 1);
||||||| merged common ancestors
    bevo.cps_impexcbFermion((Float *)in , i, 1, 1);
=======
    bd.cps_impexcbFermion((Float *)in , i, 1, 1);
>>>>>>> ckelly_latest
#pragma omp parallel
    {
        bd.Mprec(i, o, t, dag == DAG_YES, 0);
    }
    bd.cps_impexcbFermion((Float *)out, o, 0, 1);

<<<<<<< HEAD
    bd.freeFermion(i);
    bd.freeFermion(o);
    bd.freeFermion(t);

    VRB.Result(cname, fname, "end MatPc: mass = %e\n", mass);
||||||| merged common ancestors
    bevo.freeFermion(i);
    bevo.freeFermion(o);
    bevo.freeFermion(t);
=======
    bd.freeFermion(i);
    bd.freeFermion(o);
    bd.freeFermion(t);
>>>>>>> ckelly_latest
}

// It evolves the canonical momentum mom by step_size
// using the fermion force.
ForceArg Fbfm::EvolveMomFforce(Matrix *mom, Vector *frm, 
                               Float mass, Float epsilon, Float step_size)
{
    const char *fname = "EvolveMomFforce()";
  
<<<<<<< HEAD
    SetBfmArg(mass);

    const int f_size_4d = SPINOR_SIZE * GJP.VolNodeSites();
    const int f_size_cb = f_size_4d * bd.Ls / 2;
||||||| merged common ancestors
    const int f_size_4d = SPINOR_SIZE * GJP.VolNodeSites();
    const int f_size_cb = f_size_4d * GJP.SnodeSites() / 2;
=======
    int f_size_4d = SPINOR_SIZE * GJP.VolNodeSites();
    if(GJP.Gparity()) f_size_4d *= 2;

    const int f_size_cb = f_size_4d * Fbfm::bfm_args[current_arg_idx].Ls / 2;
>>>>>>> ckelly_latest
  
    Vector *tmp = (Vector *)smalloc(cname, fname, "tmp", sizeof(Float)*f_size_cb);
    MatPc(tmp, frm, mass, epsilon, DAG_NO);

    ForceArg f_arg = EvolveMomFforceBase(mom, tmp, frm, mass, epsilon, step_size);
    sfree(cname, fname, "tmp", tmp);

    return f_arg;
}

ForceArg Fbfm::RHMC_EvolveMomFforce(Matrix *mom, Vector **sol, int degree,
                                    int isz, Float *alpha, Float mass, Float epsilon, Float dt,
                                    Vector **sol_d, ForceMeasure force_measure)
{
    const char *fname = "RHMC_EvolveMomFforce()";
    static Timer time(cname, fname);
    time.start(true);

    char *force_label=NULL;

    int g_size = GJP.VolNodeSites() * GsiteSize();
    if(GJP.Gparity()) g_size *= 2;

    Matrix *mom_tmp;

    if (force_measure == FORCE_MEASURE_YES) {
        mom_tmp = (Matrix*)smalloc(g_size * sizeof(Float),cname, fname, "mom_tmp");
        ((Vector*)mom_tmp) -> VecZero(g_size);
        force_label = new char[100];
    } else {
        mom_tmp = mom;
    }

    if(!UniqueID()){
	Float pvals[4];
	for(int ii=0;ii<4;ii++){
	    int off = 18 * ii + 2;
	    pvals[ii] = ((Float*)mom)[off];
	}
	if(UniqueID()==0) printf("Fbfm::RHMC_EvolveMomFforce initial mom Px(0) = %e, Py(0) = %e, Pz(0) = %e, Pt(0) = %e\n",pvals[0],pvals[1],pvals[2],pvals[3]);
    }  

    for (int i=0; i<degree; i++) {
        ForceArg Fdt = EvolveMomFforce(mom_tmp, sol[i], mass, epsilon, alpha[i]*dt);

        if (force_measure == FORCE_MEASURE_YES) {
            sprintf(force_label, "Rational, mass = %e, pole = %d:", mass, i+isz);
            Fdt.print(dt, force_label);
        }

	if(!UniqueID()){
	    Float pvals[4];
	    for(int ii=0;ii<4;ii++){
		int off = 18 * ii + 2;
		pvals[ii] = ((Float*)mom_tmp)[off];
	    }
	    if(UniqueID()==0) printf("Fbfm::RHMC_EvolveMomFforce mom_tmp after pole %d:  Px(0) = %e, Py(0) = %e, Pz(0) = %e, Pt(0) = %e\n",i,pvals[0],pvals[1],pvals[2],pvals[3]);
	}  
    }

    ForceArg ret;

    // If measuring the force, need to measure and then sum to mom
    if (force_measure == FORCE_MEASURE_YES) {
        ret.measure(mom_tmp);
        ret.glb_reduce();

        fTimesV1PlusV2((IFloat*)mom, 1.0, (IFloat*)mom_tmp, (IFloat*)mom, g_size);

        delete[] force_label;
        sfree(mom_tmp, cname, fname, "mom_tmp");
    }

<<<<<<< HEAD
    time.stop(true);
    return ret;
||||||| merged common ancestors
    return ForceArg(L1, sqrt(L2), Linf);
=======
    return ret;
>>>>>>> ckelly_latest
}

// The fermion Hamiltonian of the node sublattice.
// chi must be the solution of Cg with source phi.
// copied from FdwfBase
Float Fbfm::FhamiltonNode(Vector *phi, Vector *chi)
{
    const char *fname = "FhamiltonNode(V*, V*)";

    if (phi == 0) ERR.Pointer(cname, fname, "phi");
    if (chi == 0) ERR.Pointer(cname, fname, "chi");

    int f_size = GJP.VolNodeSites() * FsiteSize() / 2;
    if(GJP.Gparity()) f_size *= 2;

    // Sum accross s nodes is not necessary for MDWF since the library
    // does not allow lattice splitting in s direction.
    return phi->ReDotProductNode(chi, f_size);
}

// Convert fermion field f_field from -> to
// Moved to fbfm.h by CJ
#if 0
                    StrOrdType from)
{
    const char *fname = "Fconvert()";

    // nothing needs to be done
    //ERR.NotImplemented(cname, fname);
}
#endif


// The boson Hamiltonian of the node sublattice
Float Fbfm::BhamiltonNode(Vector *boson, Float mass, Float epsilon)
{
    const char *fname = "BhamiltonNode()";
    ERR.NotImplemented(cname, fname);
}

// Reflexion in s operator, needed for the hermitian version 
// of the dirac operator in the Ritz solver.
void Fbfm::Freflex(Vector *out, Vector *in)
{
    const char *fname = "Freflex(V*,V*)";
    ERR.NotImplemented(cname, fname);
}

//!< Method to ensure bosonic force works (does nothing for Wilson
//!< theories.
void Fbfm::BforceVector(Vector *in, CgArg *cg_arg)
{
    return;
}

// !< Special for Mobius fermions, applies the D_- 5D matrix to an
// !< unpreconditioned fermion vector.
//
// !< The following gives an example of D_- with Ls = 4:
//       [ D_-^1 0      0      0     ]
//       [ 0     D_-^2  0      0     ]
// D_- = [ 0     0      D_-^3  0     ]
//       [ 0     0      0      D_-^4 ]
//
// !< where D_-^s = 1 - c[s] D_W, D_W is the 4D Wilson Dirac operator.
void Fbfm::Dminus(Vector *out, Vector *in)
{
    const char *fname = "Dminus(V*, V*)";

    // should be very easy to implement ...
    ERR.NotImplemented(cname, fname);
}

<<<<<<< HEAD
void Fbfm::BondCond()
{
    Lattice::BondCond();
    ImportGauge();
}

#if 1
void Fbfm::ImportGauge()
{
    const char *fname="ImportGauge()";
    VRB.Result(cname,fname,"NEW VERSION with CPS parallel transport\n");
    LatMatrix One;
    LatMatrix LatDir[8];
    Matrix *mout[8],*min[8];
#if 1
  for(int i=0;i<One.Vol();i++){
    *(One.Mat(i))= 1.;
  }
#endif
    int dirs[8]={0,2,4,6,1,3,5,7};
  for(int i=0;i<8;i++){
    min[i] = One.Mat();
    mout[i] = LatDir[i].Mat();
  }
{
    ParTransGauge pt_g(*this);
//    pt_g.run(8,mout,min,dirs);
    pt_g.run(4,mout,min,dirs); //positive Dirs
    pt_g.run(4,mout+4,min+4,dirs+4); //positive Dirs
}
    bd.cps_importGauge_dir(LatDir[0].Field(),1); //Plus X
    bd.cps_importGauge_dir(LatDir[1].Field(),3); //Plus Y
    bd.cps_importGauge_dir(LatDir[2].Field(),5); //Plus Z
    bd.cps_importGauge_dir(LatDir[3].Field(),7); //Plus T
    bd.cps_importGauge_dir(LatDir[4].Field(),0); //Minus X
    bd.cps_importGauge_dir(LatDir[5].Field(),2); //Minus Y
    bd.cps_importGauge_dir(LatDir[6].Field(),4); //Minus Z
    bd.cps_importGauge_dir(LatDir[7].Field(),6); //Minus T
    if(use_mixed_solver) {
        bd.comm_end();
        bf.comm_init();
    bf.cps_importGauge_dir(LatDir[0].Field(),1); //Plus X
    bf.cps_importGauge_dir(LatDir[1].Field(),3); //Plus Y
    bf.cps_importGauge_dir(LatDir[2].Field(),5); //Plus Z
    bf.cps_importGauge_dir(LatDir[3].Field(),7); //Plus T
    bf.cps_importGauge_dir(LatDir[4].Field(),0); //Minus X
    bf.cps_importGauge_dir(LatDir[5].Field(),2); //Minus Y
    bf.cps_importGauge_dir(LatDir[6].Field(),4); //Minus Z
    bf.cps_importGauge_dir(LatDir[7].Field(),6); //Minus T
        bf.comm_end();
        bd.comm_init();
    }
}
#else
void Fbfm::ImportGauge()
{
    const char *fname="ImportGauge()";
    VRB.Result(cname,fname,"OLD VERSION with qpd++ parallel transport\n");
    Float *gauge = (Float *)(this->GaugeField());
    bd.cps_importGauge(gauge);
#if 0
if (0){
        bd.comm_end();
        kernel.comm_init();
        kernel.cps_importGauge(gauge);
        kernel.comm_end();
        bd.comm_init();
}
#endif
    if(use_mixed_solver) {
        bd.comm_end();
        bf.comm_init();
        bf.cps_importGauge(gauge);
        bf.comm_end();
        bd.comm_init();
    }
}
#endif

||||||| merged common ancestors
=======
void Fbfm::BondCond()
{
    Lattice::BondCond();
    ImportGauge();
}

void Fbfm::ImportGauge()
{
    Float *gauge = (Float *)(this->GaugeField());
    bd.cps_importGauge(gauge);
    if(use_mixed_solver) {
        bd.comm_end();
        bf.comm_init();
        bf.cps_importGauge(gauge);
        bf.comm_end();
        bd.comm_init();
    }
}

//------------------------------------------------------------------
// Fdslash(Vector *f_out, Vector *f_in, CgArg *cg_arg, CnvFrmType cnv_frm,
//                    int dir_flag) :
// dir_flag is deprecated for Fbfm
// Fdslash calculates both odd-->even and even-->odd sites.
//------------------------------------------------------------------
void Fbfm::Fdslash(Vector *f_out, Vector *f_in, CgArg *cg_arg, 
		    CnvFrmType cnv_frm, int dir_flag)
{
  int offset;
  char *fname = "Fdslash(V*,V*,CgArg*,CnvFrmType,int)";
  VRB.Func(cname,fname);
  VRB.Result(cname,fname,"current_arg_idx=%d mobius_scale=%g\n",current_arg_idx,bfmarg::mobius_scale);
  if (dir_flag!=0) 
  ERR.General(cname,fname,"only implemented for dir_flag(%d)=0\n",dir_flag);

    Fermion_t in[2]  = {bd.allocFermion(), bd.allocFermion()};
    Fermion_t out[2] = {bd.allocFermion(), bd.allocFermion()};

    SetMass(cg_arg->mass,0.);

    bd.cps_impexFermion((Float *)f_in , in,  1);
#pragma omp parallel
    {
	bd.G5D_Munprec(in,out,DaggerNo);
    }

    bd.cps_impexFermion((Float *)f_out, out, 0);

    bd.freeFermion(in[0]);
    bd.freeFermion(in[1]);
    bd.freeFermion(out[0]);
    bd.freeFermion(out[1]);
  

}

>>>>>>> ckelly_latest
CPS_END_NAMESPACE

#endif
