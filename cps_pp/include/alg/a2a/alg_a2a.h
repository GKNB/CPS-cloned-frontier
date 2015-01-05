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
		void allocate_vw_fftw(); //Automatically called from fft_vw if not previously called
		void free_vw_fftw();

		bool compute_vw_low(bfm_evo<double> &dwf);
		bool compute_vw_high(bfm_evo<double> &dwf);
		bool compute_vw(void);
		void fft_vw(); //Perform FFT of v and w

		bool fft_vw_computed() const{ return fftw_computed; }

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

		const A2AArg & get_args() const{ return a2a; }
		bool do_gauge_fix() const{ return do_gfix; } //Are we gauge fixing? Defaults to true
		void do_gauge_fix(const bool &b){ do_gfix = b; }
		
		//Get v. Here i lies in the range [0, nvec) where nvec = nl + nhits * Lt * sc_size / width  (*2 with flavor dilution)
		Vector *get_v(int i);
		//Get wl. Here i lies in the range [0, nl)
		Vector *get_wl(int i);
		Vector *get_wh(void) { return wh;};
		
		inline std::complex<double>* get_v(const int &vec_idx, const int &x, const int &flavor = 0){ return (complex<double> *)(v[vec_idx]) + x * SPINOR_SIZE/2 + flavor* GJP.VolNodeSites() * SPINOR_SIZE/2; }

		//Here vec_idx lies in the range  [0, nl + sc_size * nhits)   (i.e.   (vec_idx - nl) = sc + sc_size * h    where sc is a spin-color index and h is a hit index)
		//With flavor dilution the range is  [0, nl + sc_size * nhits * 2)   (i.e.   (vec_idx - nl) = sc + sc_size * f + 2*sc_size * h    where f is a flavor index)
		//Note for the high modes this does not return w, but instead wh. When using this you need to remember it should be zero for all spin/color/flavor indices that are not equal to those within the mode index
		inline std::complex<double>* get_w(const int &vec_idx, const int &x, const int &flavor = 0){ 
		  if(vec_idx<a2a.nl) return (complex<double> *)(wl[vec_idx]) + x * SPINOR_SIZE/2 + flavor * GJP.VolNodeSites() * SPINOR_SIZE/2;
		  else if(!GJP.Gparity() || (GJP.Gparity() && a2a.dilute_flavor) ) return (complex<double> *)(wh) + x + (vec_idx-a2a.nl)/(SPINOR_SIZE/2*gparity_fac)*GJP.VolNodeSites(); //all spin/color/flavor indices use the same random field
		  else return (complex<double> *)(wh) + x + 2*(vec_idx-a2a.nl)/(SPINOR_SIZE/2)*GJP.VolNodeSites() + flavor*GJP.VolNodeSites();
		}

		Vector *get_v_fftw(int i);
		Vector *get_wl_fftw(int i);
		Vector *get_wh_fftw(void) { return wh_fftw;};

		void test(bfm_evo<double> &dwf);

		//Get FFT v for light/heavy quark at 3-momentum+time index k in the range  0 -> four-vol. vec_idx is a mode index in the 
		//range 0 -> a2a_prop->get_nvec() = nl + nhits * Lt * sc_size / width  (*2 with flavor dilution). flavor is the G-parity sink flavor index (0,1)
		inline std::complex<double>* get_v_fftw(const int &vec_idx, const int &k, const int &flavor = 0){ return (complex<double> *)(v_fftw[vec_idx]) + k * SPINOR_SIZE/2 + flavor* GJP.VolNodeSites() * SPINOR_SIZE/2; }

		//Get FFT w for light/heavy quark at 3-momentum+time index k in the range  0 -> four-vol. 
		//If flavor-dilution is switched off, vec_idx is a mode index in the range 0 -> nl + sc_size * nhits. flavor is the G-parity flavor index (0,1)
		//If flavor dilution is switched on,  vec_idx is a mode index in the range 0 -> nl + sc_size * nhits * 2
		inline std::complex<double>* get_w_fftw(const int &vec_idx, const int &k, const int &flavor = 0){ 
		  if(vec_idx<a2a.nl) return (complex<double> *)(wl_fftw[vec_idx]) + k * SPINOR_SIZE/2 + flavor * GJP.VolNodeSites() * SPINOR_SIZE/2;
		  else return (complex<double> *)(wh_fftw) +  k * SPINOR_SIZE/2 + ( gparity_fac * (vec_idx-a2a.nl) + flavor )* GJP.VolNodeSites() * SPINOR_SIZE/2;
		}

		//Total number of 'mode' indices for v
		inline const int & v_modes() const{ return nvec; }		
		//Total number of 'mode' indices for w. Note this is smaller than for v because we do not need to store separate fields for each temporal dilution
		inline int w_modes() const{ return a2a.nl + nh_site * SPINOR_SIZE/2 * fdilute_fac; }

		//Complex number offset to second G-parity flavor.
		inline int v_flavour_stride() const{ return (GJP.Gparity1fX() && GJP.Xnodes() == 1) ? GJP.XnodeSites()/2*SPINOR_SIZE/2 : GJP.VolNodeSites()*SPINOR_SIZE/2; }
		inline int w_flavour_stride() const{ return (GJP.Gparity1fX() && GJP.Xnodes() == 1) ? GJP.XnodeSites()/2*SPINOR_SIZE/2 : GJP.VolNodeSites()*SPINOR_SIZE/2; }

		//For 1-flavour G-parity (X-DIRECTION ONLY) with Xnodes>1, copy/communicate the fields such that copies of both flavours reside on the first and second halves of the lattice in the X direction
		//Their memory layout for each half is identical to the 2f case. Contractions of v and w should be confined to the first half of the lattice with the second half contributing zero to the global sum
		//(or global sum and divide by 2!)
		void gparity_1f_fftw_comm_flav();
		void gen_rand_4d_init_gp1f_flavdilute(void);
		
		//NOTE: For the momenta below, we use the conventions  \sum_x e^{-ipx} f(x) for a forwards Fourier transform with momentum p 
		//Set the momentum projection you wish to impose on the quark source in units of pi/L
		void set_w_momentum(const Float p[3]){ 
		  w_momentum[0] = p[0]; w_momentum[1] = p[1]; w_momentum[2] = p[2];
		}
		void set_wdag_momentum(const Float p[3]){ 
		  w_momentum[0] = -p[0]; w_momentum[1] = -p[1]; w_momentum[2] = -p[2];
		}

		//Set the momentum projection you wish to impose on the quark sink in units of pi/L
		void set_v_momentum(const Float p[3]){ 
		  v_momentum[0] = p[0]; v_momentum[1] = p[1]; v_momentum[2] = p[2];
		}
		
		const Float * get_v_momentum() const{ return v_momentum; }
		const Float * get_w_momentum() const{ return w_momentum; }

		//Multiplies v and w by the appropriate flavor matrices in the FFT stage such that the propagator is translationally covariant. Does not affect the position-space fields
		void gparity_make_fields_transconv(const bool &b = true){ gparity_transconv_fields = b; }
		
		//Convert from 0 <= i <  nl[0]+nhits[0]*dilute_size to mode index I of vector v with source timeslice t:  0 <= I < nl[0] + nhits[0] * Lt * dilute_size / width[0]
		//cf.  discussion in Mesonfield.h
		inline int idx_v(const int &i, const int &t) const{
		  int dilute_size = 12*fdilute_fac;
		  return (i < a2a.nl) ?   i  :  a2a.nl + (i-a2a.nl)/dilute_size*nh_base + t/a2a.src_width*dilute_size + (i-a2a.nl)%dilute_size ; 
		}

		
	private:
		const char *cname;
		A2AArg a2a;

		Vector **v; // v
		Vector **wl; // low modes w
		Vector *wh; // high modes w, compressed.

		Vector **v_fftw;
		Vector **wl_fftw;
		Vector *wh_fftw; //FFT of wh

		bool fftw_computed;
		bool do_gfix;

		bool gparity_1f_fftw_comm_flav_performed;

		int nvec;
		Lanczos_5d<double> *eig;

		// the number of wall sources used for any specific site (aka a2a.nhits)
		int nh_site;

		// minimum number of wall sources (with 1 hit per time slice)
		// equals GJP.Tnodes() * GJP.TnodeSites() * 12 / a2a.src_width  (*2 with flavor dilution)
		int nh_base;

		// number of actual wall sources used (nh should always be nh_site * nh_base)
		int nh;

		//Scale factor of 2 if 2f Gparity, 1 otherwise
		int gparity_fac;

		//Scale factor of 2 if Gparity *and* flavor dilution active, 1 otherwise
		int fdilute_fac;

		// eigenvectors and eigenvalues of Hee
		Vector **evec;
		Float *eval;

		void cnv_lcl_glb(fftw_complex *glb, fftw_complex *lcl, bool lcl_to_glb);
		void gf_vec(Vector *out, Vector *in);

		void gparity_make_field_transconv(Vector *v, const Float p[3]);

		//Perform the FFT of the vector 'vec'. fft_mem should be pre-allocated and is used as temporary memory for the FFT. 
		//fft_dim are the dimensions in the z,y and x directions respectively
		//Output into result
		//add_momentum is the momentum projection you wish to impose on the quark source/sink in units of pi/L
		//this must be applied before taking the FFT if the momenta are not integer multiples of 2pi/L, as is the case with G-parity BCs.

		void fft_vector(Vector* result, Vector* vec, const int fft_dim[3], const Float add_momentum[3], fftw_complex* fft_mem);
		
		//The momenta imposed on the quark fields v and w in units of pi/L; only affects the FFT vectors
		//Defaults to zero
		Float w_momentum[3];
		Float v_momentum[3];
		
		//The FFT fields are made translationally covariant by multiplying v and w by the appropriate flavor matrices
		//This occurs in fft_vw
		bool gparity_transconv_fields;

		friend class A2APropbfmTesting;
		friend class MesonFieldTesting;
};



inline static int sign(const Float &d){
  return d > 0 ? +1 : ( d < 0 ? -1 : 0 ) ;
}
int sign_p_gparity(const Float p[3]);


CPS_END_NAMESPACE
#endif
