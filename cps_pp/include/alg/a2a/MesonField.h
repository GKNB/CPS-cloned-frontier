#ifndef _MESON_FIELD_H_
#define _MESON_FIELD_H_

#include <util/rcomplex.h>
#include <util/vector.h>
#include <util/error.h>
#include "alg_a2a.h"
#include <alg/alg_fix_gauge.h>
#include <alg/alg_base.h>
#include <alg/wilson_matrix.h>

#include <vector>
#include <fftw3.h>
enum QUARK {
	LIGHT=0,
	STRANGE=1
};
class MesonField {
	public:
		MesonField(cps::Lattice &, cps::A2APropbfm *, cps::AlgFixGauge *, cps::CommonArg *);
		MesonField(cps::Lattice &, cps::A2APropbfm *, cps::A2APropbfm *, cps::AlgFixGauge *, cps::CommonArg *);
		~MesonField();
		void cal_mf_ww(double,int,QUARK,QUARK);
		void cal_mf_ll(double,int);
		void cal_mf_sl(double,int);
		void cal_mf_ls(double,int);
		void save_mf(char *);
		void save_mf_sl(char *);
		void save_mf_ls(char *);
		void run_pipi(int);
		void run_kaon(double);
		void run_type1(int, int);
		void run_type1_three_sink_approach(int, int);
		void run_type1_four_sink_approach(int, int);
		void run_type2(int, int);
		void run_type2_three_sink_approach(int, int);
		void run_type2_four_sink_approach(int, int);
		void run_type3(int, int);
		void run_type3_three_sink_approach(int, int);
		void run_type4(int, int);
		void run_type4_three_sink_approach(int, int);
		void run_type3S5D(int, int);
		void run_type4S5D(int, int);
		void gf_vec(cps::Vector *, cps::Vector *);
		void prepare_vw();
		void allocate_vw_fftw();
		void free_vw_fftw();
	private:
		cps::Lattice &lat;
		char *cname;
		int nvec[2];
		int nl[2];
		int nhits[2];
		int src_width[2];
		int nbase[2];
		bool do_strange;//Whether calculate strange quark propagtor
		int t_size;
		int x_size;
		int y_size;
		int z_size;
		//CK: For G-parity, the *_size variables are the size of a *single flavour*, not both
		int size_4d;
		int size_3d;
		int size_4d_sc;
		const int sc_size;
		fftw_complex *src;
		void cnv_lcl_glb(fftw_complex *glb, fftw_complex *lcl, bool lcl_to_glb);
		void set_zero(fftw_complex *v, int n);
		void wh2w(cps::Vector *w, cps::Vector *wh, int hitid, int sc);
		void set_expsrc(fftw_complex *, cps::Float);
		void set_boxsrc(fftw_complex *, int);
		void prepare_src(double, int, const int &flavour = 0, const bool &zero_other_flavour = true);
		void src_glb2lcl(fftw_complex *, fftw_complex *);
		complex<double> Gamma5(complex<double> *, complex<double> *, complex<double>);
		complex<double> Unit(complex<double> *, complex<double> *, complex<double>);
		void mf_contraction(QUARK left, QUARK middle, QUARK right, complex<double> *left_mf, int t_sep, complex<double> *right_mf, complex<double> *result); 
		void mf_contraction_ww(QUARK left, QUARK middle, QUARK right, complex<double> *left_mf, int t_sep, complex<double> *right_mf, complex<double> *result); 
		// t_sep is t_left-t_right
		void show_wilson(const cps::WilsonMatrix &);
		void writeCorr(complex<double> *, char *, int, int);
		cps::WilsonMatrix Build_Wilson(cps::Vector *, int, int, QUARK, QUARK);
		cps::WilsonMatrix Build_Wilson_ww(cps::Vector *, int, int, int, QUARK, QUARK);
		cps::WilsonMatrix Build_Wilson_loop(int, QUARK);
		cps::A2APropbfm *a2a_prop;
		cps::A2APropbfm *a2a_prop_s;
		cps::AlgFixGauge *fix_gauge;
		cps::CommonArg *common_arg;
		cps::Vector **v_fftw;
		cps::Vector **wl_fftw[2];
		cps::Vector **v_s_fftw;
		cps::Vector *mf;
		cps::Vector *mf_ls;// light w dot product strange v
		cps::Vector *mf_sl;// strange w dot product light v
		cps::Vector *mf_ww[3]; // 0: light w light w; 1: light w strange w; 2: strange w light w;
		cps::Vector *wh_fftw[2];
		cps::Vector *wh_s_fftw;

		friend class MesonFieldTesting;
		friend void Gparity_1f_FFT(fftw_complex *fft_mem);
};
#endif
