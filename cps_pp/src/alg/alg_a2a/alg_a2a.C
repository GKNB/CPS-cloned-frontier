#include <alg/a2a/alg_a2a.h>
#include <alg/eigen/Krylov_5d.h>

#include<config.h>
#include<stdlib.h>
#include<string.h>
#include<alg/alg_base.h>
#include<alg/common_arg.h>
#include<util/lattice.h>
#include<util/gjp.h>
#include<util/verbose.h>
#include<util/error.h>
#include<util/wilson.h>

#include<util/qcdio.h>
//#include<util/qio_writeGenericFields.h>
//#include<util/qio_readGenericFields.h>
CPS_START_NAMESPACE

using namespace std;

A2APropbfm::A2APropbfm(Lattice &latt,
		const A2AArg &_a2a,
		CommonArg &common_arg,
		Lanczos_5d<double> *_eig) : Alg(latt, &common_arg)
{
	cname = "A2APropbfm";
	const char *fname = "A2AProp()";
	a2a = _a2a;
	eig=_eig;

	if(latt.Fclass() != F_CLASS_DWF) {
		ERR.General(cname, fname, "Only DWF is implemented.\n");
	}

	v = NULL;
	wl = NULL;
	wh = NULL;

	evec = NULL;
	eval = NULL;

	// make sure that the number of high modes specified is a multiple of a "base number",
	// the base number is equal to Nt * 12.
	if(a2a.src_width <= 0) {
		ERR.General(cname, fname, "Invalid number for source width (value = %d).\n", a2a.src_width);
	}
	if(GJP.TnodeSites() % a2a.src_width != 0) {
		ERR.General(cname, fname, "Local t size(%d) is not a multiple of source width(%d).\n", GJP.TnodeSites(), a2a.src_width);
	}

	nh_site = a2a.nhits;
	nh_base = GJP.Tnodes() * GJP.TnodeSites() * latt.Colors() * latt.SpinComponents() / a2a.src_width;
	nh = nh_site * nh_base;
	nvec = a2a.nl + nh;
}

A2APropbfm::~A2APropbfm()
{
	const char *fname = "~A2APropbfm()";

	free_vw();

	if(evec) {
		for(int i = 0; i < a2a.nl; ++i) {
			if(evec[i]) sfree(cname, fname, "evec[i]", evec[i]);
		}
		sfree(cname, fname, "evec", evec);
	}

	if(eval) {
		sfree(cname, fname, "eval", eval);
	}
}

void A2APropbfm::allocate_vw(void)
{
	const char *fname = "allocate_vw()";

	const int f_size_4d = GJP.VolNodeSites() * SPINOR_SIZE;

	v  = (Vector **)smalloc(cname, fname, "v" , sizeof(Vector *) * nvec);
	wl = (Vector **)smalloc(cname, fname, "wl", sizeof(Vector *) * a2a.nl); 

	for(int i = 0; i < nvec; ++i) {
		v[i]  = (Vector *)smalloc(cname, fname, "v[i]" , sizeof(Float) * f_size_4d);
	}
	for(int i = 0; i < a2a.nl; ++i) {
		wl[i] = (Vector *)smalloc(cname, fname, "wl[i]", sizeof(Float) * f_size_4d);
	}

	// allocate wh
	const int wh_size = 2 * GJP.VolNodeSites() * nh_site;
	wh = (Vector *)smalloc(cname, fname, "wh", sizeof(Float) * wh_size);
}

void A2APropbfm::gen_rand_4d_init(void)
{
	const char *fname = "gen_rand_4d_init()";

	if(!wh) {
		ERR.Pointer(cname, fname, "wh");
	}

	LRG.SetInterval(1, 0);
	const int sites = GJP.VolNodeSites();
	Float *f = (Float *)wh;

	const Float PI = 3.14159265358979323846;

	for(int i = 0; i < sites; ++i) {
		LRG.AssignGenerator(i);
		for(int j = 0; j < nh_site; ++j) {
			Float theta = LRG.Urand(FOUR_D);
			switch(a2a.rand_type) {
				case UONE:
					f[2 * (j * sites + i)    ] = cos(2. * PI * theta);
					f[2 * (j * sites + i) + 1] = sin(2. * PI * theta);
					break;
				case ZTWO:
					f[2 * (j * sites + i)    ] = theta > 0.5 ? 1 : -1;
					f[2 * (j * sites + i) + 1] = 0;
					break;
				case ZFOUR:
					if(theta > 0.75) {
						f[2 * (j * sites + i)    ] = 1;
						f[2 * (j * sites + i) + 1] = 0;
					}else if(theta > 0.5) {
						f[2 * (j * sites + i)    ] = -1;
						f[2 * (j * sites + i) + 1] = 0;
					}else if(theta > 0.25) {
						f[2 * (j * sites + i)    ] = 0;
						f[2 * (j * sites + i) + 1] = 1;
					}else {
						f[2 * (j * sites + i)    ] = 0;
						f[2 * (j * sites + i) + 1] = -1;
					}
					break;
				default:
					ERR.NotImplemented(cname, fname);
			}
		}
	}
}

bool A2APropbfm::compute_vw_low(bfm_evo<double> &dwf)
{
	const char *fname = "compute_vw_low()";
	Lattice &lat = AlgLattice();
	const int f_size = GJP.VolNodeSites() * lat.FsiteSize();
	const int f_size_4d = GJP.VolNodeSites() * SPINOR_SIZE;
	const int glb_ls = GJP.SnodeSites() * GJP.Snodes();

	Vector *a = (Vector *)smalloc(cname, fname, "a", sizeof(Float) * f_size);
	Vector *b = (Vector *)smalloc(cname, fname, "a", sizeof(Float) * f_size);

	Fermion_t tmp[2];
	Fermion_t Mtmp;
	tmp[0] = dwf.allocFermion();
	tmp[1] = dwf.allocFermion();
	Mtmp   = dwf.allocFermion();

	for(int i=0;i<a2a.nl;i++)
	{
		omp_set_num_threads(bfmarg::threads);
#pragma omp parallel
		{
			dwf.Meo(eig->bq[i][1],tmp[0],Even,0);
			dwf.MooeeInv(tmp[0],tmp[1],0);
			dwf.axpy(tmp[0],tmp[1],tmp[1],-2.0);
			dwf.axpy(tmp[1],eig->bq[i][1],eig->bq[i][1],0.0);
		}
		dwf.cps_impexFermion((Float *)b,tmp,0);
		lat.Ffive2four(v[i],b,glb_ls-1,0,2);
		v[i]->VecTimesEquFloat(1.0 / eig->evals[i],f_size_4d);
		// above is for v_low

		omp_set_num_threads(bfmarg::threads);
#pragma omp parallel
		{
			dwf.Mprec(eig->bq[i][1],tmp[1],Mtmp,DaggerNo);
			dwf.Meo(tmp[1],Mtmp,Even,1);
			dwf.MooeeInv(Mtmp,tmp[1],1);
			dwf.axpy(tmp[0],tmp[1],tmp[1],-2.0);
			dwf.Mprec(eig->bq[i][1],tmp[1],Mtmp,DaggerNo);
		}
		dwf.cps_impexFermion((Float *)b,tmp,0);
		lat.Ffive2four(wl[i],b,0,glb_ls-1, 2);
		//above is for w_low
	}


	sfree(cname, fname, "a", a);
	sfree(cname, fname, "b", b);
	dwf.freeFermion(tmp[0]);
	dwf.freeFermion(tmp[1]);
	dwf.freeFermion(Mtmp);
}

void A2APropbfm::gen_rand_4d(Vector *v4d, int id)
{
	const char *fname = "gen_rand_4d()";

	Lattice &lat = AlgLattice();
	const int t_glb = GJP.Tnodes() * GJP.TnodeSites();
	const int spin_color = lat.Colors() * lat.SpinComponents();
	const int size_4d = GJP.VolNodeSites() * 2 * spin_color;

	v4d->VecZero(size_4d);

	const int wh_id = id / nh_base;
	const int t_id = id % nh_base / spin_color * a2a.src_width;
	const int sc_id = id % nh_base % spin_color;
	VRB.Result(cname, fname, "Generating random wall source %d = (%d, %d, %d).\n    ", id, wh_id, t_id, sc_id);

	if(t_id / GJP.TnodeSites() != GJP.TnodeCoor()) return;

	const int t_lcl = t_id % GJP.TnodeSites();
	const int wall_size = size_4d / GJP.TnodeSites();

	Float *vf = (Float *)v4d;
	Float *whf = (Float *)wh;
	for(int i = sc_id; i < wall_size / 2 * a2a.src_width; i += spin_color) {
		int offset = t_lcl * wall_size / 2 + i;
		int wh_offset =
			wh_id * GJP.VolNodeSites()
			+ t_lcl * GJP.VolNodeSites() / GJP.TnodeSites()
			+ i / spin_color;

		vf[2 * offset    ] = whf[2 * wh_offset    ];
		vf[2 * offset + 1] = whf[2 * wh_offset + 1];
	}
}

bool A2APropbfm::compute_vw_high(bfm_evo<double> &dwf)
{
	const char *fname = "compute_vw_high()";
	VRB.Result(cname, fname, "Start computing high modes.\n");
	Lattice &lat = AlgLattice();

	// Set random number for all high modes.
	gen_rand_4d_init();

	const int f_size_4d = GJP.VolNodeSites() * 2 * lat.Colors() * lat.SpinComponents();
	const int f_size = GJP.VolNodeSites() * lat.FsiteSize();
	const int glb_ls = GJP.SnodeSites() * GJP.Snodes();

	Vector *a = (Vector *)smalloc(cname, fname, "a", sizeof(Float) * f_size);
	Vector *b = (Vector *)smalloc(cname, fname, "b", sizeof(Float) * f_size);

	Fermion_t src[2], V_tmp[2],tmp;
	src[0] = dwf.allocFermion();
	src[1] = dwf.allocFermion();
	V_tmp[0] = dwf.allocFermion();
	V_tmp[1] = dwf.allocFermion();
  tmp = dwf.allocFermion();

	multi1d<double> eval(a2a.nl);
	for(int i=0;i<a2a.nl;i++)
		eval[i] = eig->evals[i];

	for(int i = a2a.nl; i < nvec; i++) {
		// use v[i] as a temporary storage
		gen_rand_4d(v[i], i-a2a.nl);
		lat.Ffour2five(a, v[i], 0, glb_ls-1, 2);
		dwf.cps_impexFermion((Float *)a,src,1);

		omp_set_num_threads(bfmarg::threads);
#pragma omp parallel
		{
			dwf.CGNE_M_high(V_tmp,src,eig->bq,eval);
		}

		dwf.cps_impexFermion((Float *)b,V_tmp,0);
		lat.Ffive2four(v[i], b, glb_ls-1, 0, 2);
		v[i]->VecTimesEquFloat(1.0 / nh_site, f_size_4d);

	}
	sfree(cname, fname, "a", a);
	sfree(cname, fname, "b", b);
	dwf.freeFermion(src[0]); 
	dwf.freeFermion(src[1]);
	dwf.freeFermion(V_tmp[0]); 
	dwf.freeFermion(V_tmp[1]);
	dwf.freeFermion(tmp); 

	return true;
}

Vector *A2APropbfm::get_wl(int i)
{
	const char *fname = "get_wl()";

	if(wl == NULL) {
		ERR.Pointer(cname, fname, "wl");
	}
	if(i >= a2a.nl) {
		ERR.General(cname, fname, "Array index out of range %d >= %d.\n", i, a2a.nl);
	}
	return wl[i];
}

Vector *A2APropbfm::get_v(int i)
{
	const char *fname = "get_v()";
	if(v == NULL) {
		ERR.Pointer(cname, fname, "v");
	}
	if(i >= nvec) {
		ERR.General(cname, fname, "Array index out of range %d >= %d.\n", i, nvec);
	}
	return v[i];
}
void A2APropbfm::free_vw(void)
{
	const char *fname = "free_vw()";

	if(v) {
		for(int i=0;i<nvec;i++) {
			if(v[i]) 
				sfree(cname,fname,"v[i]",v[i]);
		}
		sfree(cname,fname,"v",v);
		v=NULL;
	}

	if(wl) {
		for(int i=0;i<a2a.nl;i++) {
			if(wl[i]) sfree(cname,fname,"wl[i]",wl[i]);
		}
		sfree(cname,fname,"wl",wl);
		wl=NULL;
	}

	if(wh) {
		sfree(cname,fname,"wh",wh);
		wh=NULL;
	}
}


CPS_END_NAMESPACE
