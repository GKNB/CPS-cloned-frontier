#ifndef _ALG_A2A_H
#define _ALG_A2A_H

#include<config.h>
#include <util/error.h>
#include <util/verbose.h>
#include <util/rcomplex.h>
#include <util/vector.h>
#include <util/lattice.h>
//#include <util/dirac_op.h>
#include <alg/alg_base.h>
#include <alg/cg_arg.h>
#include <alg/a2a/a2a_arg.h>
#include <util/lattice/bfm_evo.h>
#include <alg/a2a/lanc_arg.h>
#include <alg/eigen/Krylov_5d.h>

CPS_START_NAMESPACE

class A2APropbfm : public Alg {
	public:
		A2APropbfm(Lattice &latt, const A2AArg &_a2a,
				CommonArg &common_arg, Lanczos_5d<double> *lanczos);
		~A2APropbfm();

		bool load_evec5(void);
		bool load_eval5(void);

		void allocate_vw();
		void free_vw();

		bool compute_vw_low(bfm_evo<double> &dwf);
		bool compute_vw_high(bfm_evo<double> &dwf);
		bool compute_vw(void);

		void save_vw(void);
		void save_v(int i);
		void save_wl(int i);
		void save_wh(void);

		void load_vw(void);
		void load_v(Vector *v, int i);
		void load_wl(Vector *wl, int i);
		void load_wh(Vector *wh);

		void gen_rand_4d(Vector *v4d, int i);
		void gen_rand_4d_init(void);

		int get_nl()const { return a2a.nl; }
		int get_nh()const { return nh; }
		int get_nhits()const { return a2a.nhits; }
		int get_src_width()const { return a2a.src_width; }
		int get_nh_base()const { return nh_base; }
		int get_nvec()const { return nvec; }

		Vector *get_v(int i);
		Vector *get_wl(int i);
		Vector *get_wh(void) { return wh;};

		void test(bfm_evo<double> &dwf);
	private:
		const char *cname;
		A2AArg a2a;

		Vector **v; // v
		Vector **wl; // low modes w
		Vector *wh; // high modes w, compressed.

		int nvec;
		Lanczos_5d<double> *eig;

		// the number of wall sources used for any specific site (aka a2a.nhits)
		int nh_site;

		// minimum number of wall sources (with 1 hit per time slice)
		// equals GJP.Tnodes() * GJP.TnodeSites() * 12 / a2a.src_width
		//CK: presumably source_width is the number of time-slices spanned by a given stochastic source
		//Hmm, further code reading suggests the same stochastic source is used for both time slices
		int nh_base;

		// number of actual wall sources used (nh should always be nh_site * nh_base)
		int nh;

		// eigenvectors and eigenvalues of Hee
		Vector **evec;
		Float *eval;
};

CPS_END_NAMESPACE
#endif
