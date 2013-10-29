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
#include <fftw3.h>

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
		void allocate_vw_fftw();
		void free_vw_fftw();

		bool compute_vw_low(bfm_evo<double> &dwf);
		bool compute_vw_high(bfm_evo<double> &dwf);
		bool compute_vw(void);
		void fft_vw(); //Perform FFT of v and w

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
		
		Vector *get_v_fftw(int i);
		Vector *get_wl_fftw(int i);
		Vector *get_wh_fftw(void) { return wh_fftw;};

		void test(bfm_evo<double> &dwf);

		//Get FFT v for light/heavy quark at four-momentum index k in the range  0 -> four-vol. vec_idx is a mode index in the 
		//range 0 -> a2a_prop->get_nvec() = nl + nhits * Lt * sc_size / width. flavor is the G-parity flavor index (0,1)
		inline std::complex<double>* get_v_fftw(const int &vec_idx, const int &k, const int &flavor = 0){ return (complex<double> *)(v_fftw[vec_idx]) + k * SPINOR_SIZE/2 + flavor* GJP.VolNodeSites() * SPINOR_SIZE/2; }

		//Get FFT w for light/heavy quark at four-momentum index k in the range  0 -> four-vol. vec_idx is a mode index in the range 0 -> nl + sc_size * nhits. flavor is the G-parity flavor index (0,1)
		inline std::complex<double>* get_w_fftw(const int &vec_idx, const int &k, const int &flavor = 0){ 
		  if(vec_idx<a2a.nl) return (complex<double> *)(wl_fftw[vec_idx]) + k * SPINOR_SIZE/2 + flavor * GJP.VolNodeSites() * SPINOR_SIZE/2;
		  else return (complex<double> *)(wh_fftw) +  k * SPINOR_SIZE/2 + ( (GJP.Gparity() ? 2:1) * (vec_idx-a2a.nl) + flavor )* GJP.VolNodeSites() * SPINOR_SIZE/2;
		}

		inline int v_flavour_stride() const{ return (GJP.Gparity1fX() && GJP.Xnodes() == 1) ? GJP.XnodeSites()/2*SPINOR_SIZE/2 : GJP.VolNodeSites()*SPINOR_SIZE/2; }
		inline int w_flavour_stride() const{ return (GJP.Gparity1fX() && GJP.Xnodes() == 1) ? GJP.XnodeSites()/2*SPINOR_SIZE/2 : GJP.VolNodeSites()*SPINOR_SIZE/2; }

		//For 1-flavour G-parity (X-DIRECTION ONLY) with Xnodes>1, copy/communicate the fields such that copies of both flavours reside on the first and second halves of the lattice in the X direction
		//Their memory layout for each half is identical to the 2f case. Contractions of v and w should be confined to the first half of the lattice with the second half contributing zero to the global sum
		//(or global sum and divide by 2!)
		void gparity_1f_fftw_comm_flav();

	private:
		const char *cname;
		A2AArg a2a;

		Vector **v; // v
		Vector **wl; // low modes w
		Vector *wh; // high modes w, compressed.

		Vector **v_fftw;
		Vector **wl_fftw;
		Vector *wh_fftw; //FFT of wh
	       
		bool gparity_1f_fftw_comm_flav_performed;

		int nvec;
		Lanczos_5d<double> *eig;

		// the number of wall sources used for any specific site (aka a2a.nhits)
		int nh_site;

		// minimum number of wall sources (with 1 hit per time slice)
		// equals GJP.Tnodes() * GJP.TnodeSites() * 12 / a2a.src_width
		int nh_base;

		// number of actual wall sources used (nh should always be nh_site * nh_base)
		int nh;

		// eigenvectors and eigenvalues of Hee
		Vector **evec;
		Float *eval;

		void cnv_lcl_glb(fftw_complex *glb, fftw_complex *lcl, bool lcl_to_glb);
		void gf_vec(Vector *out, Vector *in);
};

CPS_END_NAMESPACE
#endif
