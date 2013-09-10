#include <config.h>
#include <alg/a2a/MesonField.h>

#include <fstream>
#include <alg/wilson_matrix.h>
#include <util/rcomplex.h>
#include <util/wilson.h>
#include <util/qcdio.h>
//#include <qmp.h>

using namespace std;
using namespace cps;

int cout_time(char *info);

MesonField::MesonField(Lattice &_lat, A2APropbfm *_a2aprop, AlgFixGauge *_fixgauge, CommonArg *_common_arg):a2a_prop(_a2aprop),a2a_prop_s(NULL),fix_gauge(_fixgauge),lat(_lat),common_arg(_common_arg),sc_size(12)
{
	cname = "MesonField";
	char *fname = "Mesonfield(A2APropbfm &_a2aprop, AlgFixGauge &_fixgauge,  int src_type, int snk_type, int *pv, int *pw)";
	t_size = GJP.TnodeSites()*GJP.Tnodes();
	x_size = GJP.XnodeSites()*GJP.Xnodes();
	y_size = GJP.YnodeSites()*GJP.Ynodes();
	z_size = GJP.ZnodeSites()*GJP.Znodes();
	size_4d = GJP.VolNodeSites();
	size_4d_sc = size_4d * sc_size;
	size_3d = size_4d / GJP.TnodeSites();

	nvec[0] = a2a_prop->get_nvec();
	nl[0] = a2a_prop->get_nl();
	nhits[0] = a2a_prop->get_nhits();
	src_width[0] = a2a_prop->get_src_width();
	nbase[0] = t_size * 12 / src_width[0];

	do_strange = false;

	nvec[1] = 0;
	nl[1] = 0;
	nhits[1] = 0;
	src_width[1] = 0;
	nbase[1] = 0; 

	v_fftw = NULL;
	wl_fftw[0] = wl_fftw[1] = NULL;
	v_s_fftw = NULL;
	mf = (Vector *)smalloc(cname, fname, "mf", sizeof(Float)*nvec[0]*(nl[0]+nhits[0]*sc_size)*t_size*2);
	mf_sl = NULL;
	mf_ls = NULL;
	mf_ww[0] = mf_ww[1] = mf_ww[2] = NULL;
	wh_fftw[0] = wh_fftw[1] = NULL;
	src = fftw_alloc_complex(size_4d/GJP.TnodeSites());
}

MesonField::MesonField(Lattice &_lat, A2APropbfm *_a2aprop, A2APropbfm *_a2aprop_s, AlgFixGauge *_fixgauge, CommonArg *_common_arg):a2a_prop(_a2aprop),a2a_prop_s(_a2aprop_s),fix_gauge(_fixgauge),lat(_lat),common_arg(_common_arg),sc_size(12)
{
	cname = "MesonField";
	char *fname = "Mesonfield(A2APropbfm &_a2aprop, AlgFixGauge &_fixgauge,  int src_type, int snk_type, int *pv, int *pw)";
	t_size = GJP.TnodeSites()*GJP.Tnodes();
	x_size = GJP.XnodeSites()*GJP.Xnodes();
	y_size = GJP.YnodeSites()*GJP.Ynodes();
	z_size = GJP.ZnodeSites()*GJP.Znodes();
	size_4d = GJP.VolNodeSites();
	size_4d_sc = size_4d * sc_size;
	size_3d = size_4d/GJP.TnodeSites();

	nvec[0] = a2a_prop->get_nvec();
	nl[0] = a2a_prop->get_nl();
	nhits[0] = a2a_prop->get_nhits();
	src_width[0] = a2a_prop->get_src_width();
	nbase[0] = t_size * 12 / src_width[0];

	do_strange = true;

	nvec[1] = a2a_prop_s->get_nvec();
	nl[1] = a2a_prop_s->get_nl();
	nhits[1] = a2a_prop_s->get_nhits();
	src_width[1] = a2a_prop_s->get_src_width();
	nbase[1] = t_size * 12 / src_width[1];

	v_fftw = NULL;
	v_s_fftw = NULL;
	wl_fftw[0] = wl_fftw[1] = NULL;
	mf = (Vector *)smalloc(cname, fname, "mf", sizeof(Float)*nvec[0]*(nl[0]+sc_size*nhits[0])*t_size*2);
	mf_sl = (Vector *)smalloc(cname, fname, "mf_sl", sizeof(Float)*nvec[0]*(nl[1]+sc_size*nhits[1])*t_size*2);
	mf_ls = (Vector *)smalloc(cname, fname, "mf_ls", sizeof(Float)*nvec[1]*(nl[0]+sc_size*nhits[0])*t_size*2);
	mf_ww[0] = mf_ww[1] = mf_ww[2] = NULL;
	wh_fftw[0] = wh_fftw[1] = NULL;
	src = fftw_alloc_complex(size_4d/GJP.TnodeSites());
}

void MesonField::allocate_vw_fftw()
{
	const char *fname = "allocate_vw_fftw()";

	v_fftw = (Vector **)smalloc(cname, fname, "v_fftw", sizeof(Vector *) * nvec[0]);
	wl_fftw[0] = (Vector **)smalloc(cname, fname, "wl_fftw[0][i]", sizeof(Vector *) * nl[0]);
	for(int i = 0; i < nvec[0]; ++i) { 
		v_fftw[i]  = (Vector *)smalloc(cname, fname, "v[i]" , sizeof(Float) * size_4d * sc_size * 2);
	}
	for(int i = 0; i < nl[0]; ++i) {
		wl_fftw[0][i] = (Vector *)smalloc(cname, fname, "wl_fftw[0][i]", sizeof(Float) * size_4d * sc_size * 2);
	}
	wh_fftw[0] = (Vector *)smalloc(cname, fname, "wh_fftw[0]", sizeof(Float)*nhits[0]*sc_size*size_4d_sc*2);

	if(do_strange) {
		v_s_fftw = (Vector **)smalloc(cname, fname, "v_s_fftw", sizeof(Vector *) * nvec[1]);
		wl_fftw[1] = (Vector **)smalloc(cname, fname, "wl_fftw[1]", sizeof(Vector *) * nl[1]);
		for(int i = 0; i < nvec[1]; ++i) {
			v_s_fftw[i]  = (Vector *)smalloc(cname, fname, "v_s_fftw[i]" , sizeof(Float) * size_4d * sc_size * 2);
		}
		for(int i = 0; i < nl[1]; ++i) {
			wl_fftw[1][i] = (Vector *)smalloc(cname, fname, "wl_fftw[1][i]", sizeof(Float) * size_4d * sc_size * 2);
		}
	wh_fftw[1] = (Vector *)smalloc(cname, fname, "wh_fftw[1]", sizeof(Float)*nhits[1]*sc_size*size_4d_sc*2);
	}
}

void MesonField::free_vw_fftw()
{
	const char *fname = "free_vw_fftw()";

	if(v_fftw) {
		for(int i=0;i<nvec[0];i++) {
			if(v_fftw[i])
				sfree(cname,fname,"v_fftw[i]",v_fftw[i]);
		}
		sfree(cname,fname,"v_fftw",v_fftw);
		v_fftw=NULL;
	}
	if(wl_fftw[0]) {
		for(int i=0;i<nl[0];i++) {
			if(wl_fftw[0][i]) sfree(cname,fname,"wl_fftw[0][i]",wl_fftw[0][i]);
		}
		sfree(cname,fname,"wl_fftw[0]",wl_fftw[0]);
		wl_fftw[0]=NULL;
	}
	if(wh_fftw[0]) {
		sfree(cname,fname,"wh_fftw[0]",wh_fftw[0]);
		wh_fftw[0] = NULL;
	}
	
	if(v_s_fftw) {
		for(int i=0;i<nvec[1];i++) {
			if(v_s_fftw[i])
				sfree(cname,fname,"v_s_fftw[i]",v_s_fftw[i]);
		}
		sfree(cname,fname,"v_s_fftw",v_s_fftw);
		v_s_fftw=NULL;
	}
	if(wl_fftw[1]) {
		for(int i=0;i<nl[1];i++) {
			if(wl_fftw[1][i]) sfree(cname,fname,"wl_fftw[1][i]",wl_fftw[1][i]);
		}
		sfree(cname,fname,"wl_fftw[1]",wl_fftw[1]);
		wl_fftw[1]=NULL;
	}
	if(wh_fftw[1]) {
		sfree(cname,fname,"wh_fftw[1]",wh_fftw[1]);
		wh_fftw[1] = NULL;
	}
}

void MesonField::gf_vec(Vector *out, Vector *in)
{
	char *fname = "gf_vector(Vector *in, Vector *out)";

	cps::Matrix **gm = lat.FixGaugePtr();

	Vector tmp;

	int size_3d = GJP.VolNodeSites() / GJP.TnodeSites();

	Float *fi = (Float *)in;
	Float *fo = (Float *)out;

	for(int i = 0; i < size_4d; ++i) {
		const int t = i / size_3d;
		const int moff = i % size_3d;
		const int voff = sc_size * 2 * i;
		for(int s = 0; s < 4; ++s) {
			tmp.CopyVec((Vector *)(fi + voff + 6 * s), 6);
			uDotXEqual(fo + voff + 6 * s, (Float *)(&gm[t][moff]), (Float *)(&tmp));
		}
	}
}

void MesonField::cnv_lcl_glb(fftw_complex *glb, fftw_complex *lcl, bool lcl_to_glb)
{
	static const int glb_dim[4] = {
		GJP.XnodeSites() * GJP.Xnodes(),
		GJP.YnodeSites() * GJP.Ynodes(),
		GJP.ZnodeSites() * GJP.Znodes(),
		GJP.TnodeSites() * GJP.Tnodes(),
	};

	static const int lcl_dim[4] = {
		GJP.XnodeSites(),
		GJP.YnodeSites(),
		GJP.ZnodeSites(),
		GJP.TnodeSites(),
	};

	static const int shift[4] = {
		GJP.XnodeSites() * GJP.XnodeCoor(),
		GJP.YnodeSites() * GJP.YnodeCoor(),
		GJP.ZnodeSites() * GJP.ZnodeCoor(),
		GJP.TnodeSites() * GJP.TnodeCoor(),
	};

	int glb_vol = glb_dim[0] * glb_dim[1] * glb_dim[2] * glb_dim[3];
	if(lcl_to_glb) {
		set_zero(glb, glb_vol * sc_size);
	}

	int lcl_size_3d = lcl_dim[0] * lcl_dim[1] * lcl_dim[2];
	int glb_size_3d = glb_dim[0] * glb_dim[1] * glb_dim[2];

	for(int t = 0; t < lcl_dim[3]; ++t) {
		fftw_complex *lcl_slice = lcl + t * lcl_size_3d * sc_size;
		fftw_complex *glb_slice = glb + (t + shift[3]) * glb_size_3d * sc_size;

		for(int xyz = 0; xyz < lcl_size_3d; ++xyz) {
			int tmp = xyz;
			int x = tmp % lcl_dim[0] + shift[0]; tmp /= lcl_dim[0];
			int y = tmp % lcl_dim[1] + shift[1]; tmp /= lcl_dim[1];
			int z = tmp + shift[2];
			int xyz_glb = x + glb_dim[0] * (y + glb_dim[1] * z);

			for(int sc = 0; sc < sc_size; ++sc) {
				int lcl_off = sc + xyz * sc_size;
				int glb_off = sc + xyz_glb * sc_size;
				if(lcl_to_glb) {
					glb_slice[glb_off][0] = lcl_slice[lcl_off][0];
					glb_slice[glb_off][1] = lcl_slice[lcl_off][1];
				} else {
					lcl_slice[lcl_off][0] = glb_slice[glb_off][0];
					lcl_slice[lcl_off][1] = glb_slice[glb_off][1];
				}
			}
		}
	}

	if(lcl_to_glb) {
		QMP_status_t ret = QMP_sum_double_array((double *)glb, glb_vol * sc_size * 2);
	}
}

//load the v/w vector and compute their fourier transform
void MesonField::prepare_vw()
{
	const char *fname = "prepare_vw()";

	cout_time("Coulomb T gauge fixing BEGIN!");
	fix_gauge->run();
	cout_time("Coulomb T gauge fixing END!");
	
	fftw_init_threads();
	fftw_plan_with_nthreads(bfmarg::threads);
	const int fft_dim[3] = { GJP.ZnodeSites() * GJP.Znodes(),
		GJP.YnodeSites() * GJP.Ynodes(),
		GJP.XnodeSites() * GJP.Xnodes()};
	const int size_3d_glb = fft_dim[0] * fft_dim[1] * fft_dim[2];

	fftw_complex *fft_mem = fftw_alloc_complex(size_3d_glb * t_size * sc_size);
	Vector *t  = (Vector *)smalloc(cname, fname, "t" , sizeof(complex<double>) * size_4d_sc);
	fftw_plan plan;

	// load and process v
	for(int j = 0; j < nvec[0]; ++j) {
		gf_vec(t, a2a_prop->get_v(j));
		cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(t), true);
		for(int sc = 0; sc < sc_size; sc++)
		{
			plan = fftw_plan_many_dft(3, fft_dim, t_size,
					fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
					fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
					FFTW_FORWARD, FFTW_ESTIMATE);
			fftw_execute(plan);
			fftw_destroy_plan(plan);
		}
		cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(v_fftw[j]), false);
	}

	// load and process wl
	for(int j = 0; j < nl[0]; ++j) {
		gf_vec(t, a2a_prop->get_wl(j));
		cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(t), true);
		for(int sc = 0; sc < sc_size; sc++)
		{
			plan = fftw_plan_many_dft(3, fft_dim, t_size,
					fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
					fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
					FFTW_FORWARD, FFTW_ESTIMATE);
			fftw_execute(plan);
			fftw_destroy_plan(plan);
		}
		cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(wl_fftw[0][j]), false);
	}

	// load and process wh  
	for(int j = 0; j < nhits[0]; ++j) 
		for(int w_sc = 0; w_sc < sc_size; w_sc++) {
		Float *wh_fftw_offset = (Float *)(wh_fftw[0]) + size_4d_sc * 2 * (j * sc_size + w_sc);
		wh2w((Vector *)wh_fftw_offset, a2a_prop->get_wh(), j, w_sc);
		gf_vec((Vector *)wh_fftw_offset, (Vector *)wh_fftw_offset);
		cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(wh_fftw_offset), true);
		for(int sc = 0; sc < sc_size; sc++)
		{
			plan = fftw_plan_many_dft(3, fft_dim, t_size,
					fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
					fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
					FFTW_FORWARD, FFTW_ESTIMATE);

			fftw_execute(plan);
			fftw_destroy_plan(plan);
		}
		cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(wh_fftw_offset), false);
	}

	if(do_strange) {
		// load and process v
		for(int j = 0; j < nvec[1]; ++j) {
			gf_vec(t, a2a_prop_s->get_v(j));
			cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(t), true);
			for(int sc = 0; sc < sc_size; sc++)
			{
				plan = fftw_plan_many_dft(3, fft_dim, t_size,
						fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
						fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
						FFTW_FORWARD, FFTW_ESTIMATE);
				fftw_execute(plan);
				fftw_destroy_plan(plan);
			}
			cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(v_s_fftw[j]), false);
		}

		// load and process wl
		for(int j = 0; j < nl[1]; ++j) {
			gf_vec(t, a2a_prop_s->get_wl(j));
			cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(t), true);
			for(int sc = 0; sc < sc_size; sc++)
			{
				plan = fftw_plan_many_dft(3, fft_dim, t_size,
						fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
						fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
						FFTW_FORWARD, FFTW_ESTIMATE);
				fftw_execute(plan);
				fftw_destroy_plan(plan);
			}
			cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(wl_fftw[1][j]), false);
		}

		// load and process wh  
		for(int j = 0; j < nhits[1]; ++j) 
			for(int w_sc = 0; w_sc < sc_size; w_sc++) {
				Float *wh_fftw_offset = (Float *)(wh_fftw[1]) + size_4d_sc * 2 * (j * sc_size + w_sc);
				wh2w((Vector *)wh_fftw_offset, a2a_prop_s->get_wh(), j, w_sc);
				gf_vec((Vector *)wh_fftw_offset, (Vector *)wh_fftw_offset);
				cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(wh_fftw_offset), true);
				for(int sc = 0; sc < sc_size; sc++)
				{
					plan = fftw_plan_many_dft(3, fft_dim, t_size,
							fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
							fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
							FFTW_FORWARD, FFTW_ESTIMATE);

					fftw_execute(plan);
					fftw_destroy_plan(plan);
				}
				cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(wh_fftw_offset), false);
			}
	}

	sfree(cname, fname, "t", t);
	fftw_free(fft_mem);
	fftw_cleanup();
	fftw_cleanup_threads();
	fix_gauge->free();
}

void MesonField::set_zero(fftw_complex *v, int n)
{
	for(int i = 0; i < n; i++) {
		v[i][0] = 0;
		v[i][1] = 0;
	}
}

void MesonField::wh2w(Vector *w, Vector *wh, int hitid, int sc) // In order to fft the high modes scource
{
	char *fname = "wh2w()";
	
	complex<double> *whi = (complex<double> *)wh + hitid * size_4d;
	complex<double> *t = (complex<double> *)w;

	for(int i = 0; i < size_4d; i++) {
		memset(t + i * sc_size, 0, sizeof(complex<double>) * sc_size);
		t[sc + i * sc_size] = whi[i];
	}
}

void MesonField::cal_mf_ww(double rad, int src_kind, QUARK left, QUARK right)
{
	char *fname = "cal_mf_ww()";

	prepare_src(rad, src_kind);

	Vector* &mf_ww_this = mf_ww[left * 2 + right];
	if(mf_ww_this==NULL) {
		mf_ww_this = (Vector *)smalloc(cname, fname, "mf_ww[i]", sizeof(Float)*(nl[left]+nhits[left]*sc_size)*(nl[right]+nhits[right]*sc_size)*t_size*2);
	}
	mf_ww_this->VecZero((nl[left]+sc_size*nhits[left])*(nl[right]+sc_size*nhits[right])*t_size*2);

	cout_time("Inner product of w and w begins!");
	int nthreads = bfmarg::threads;
	omp_set_num_threads(nthreads);
#pragma omp parallel for
	for(int i = 0; i < nl[right] + nhits[right] * sc_size; i++)
		for(int j = 0; j < nl[left] + nhits[left] * sc_size; j++)
		{
			for(int x = 0; x < size_4d; x++)
			{
				int x_3d = x % size_3d;
				int t = x / size_3d;
				int glb_t = t + GJP.TnodeCoor() * GJP.TnodeSites();

				complex<double> *right_off;
				if(i<nl[right]) right_off = (complex<double> *)(wl_fftw[right][i]) + x * sc_size;
				else right_off = (complex<double> *)(wh_fftw[right]) + size_4d_sc * (i-nl[right]) + x * sc_size;
				complex<double> *left_off;
				if(j<nl[left]) left_off = (complex<double> *)(wl_fftw[left][j]) + x * sc_size;
				else left_off = (complex<double> *)(wh_fftw[left]) + size_4d_sc * (j-nl[left]) + x * sc_size;
				complex<double> *mf_off = (complex<double> *)(mf_ww_this) + (j * (nl[right]+nhits[right]*sc_size) + i) * t_size + glb_t;
				*mf_off+=Unit(left_off,right_off,((complex<double> *)src)[x_3d]);
			}
		}
	cout_time("Inner product of w and w ends!");

	QMP_sum_double_array((Float *)(mf_ww_this),(nl[left]+sc_size*nhits[left])*(nl[right]+sc_size*nhits[right])*t_size*2);
	QDPIO::cout<<"Global sum done!"<<endl;
}

void MesonField::cal_mf_ll(double rad, int src_kind)
{
	char *fname = "cal_mf_ll()";

	mf->VecZero(nvec[0]*(nl[0]+sc_size*nhits[0])*t_size*2);
	prepare_src(rad, src_kind);

	cout_time("Inner product of light v and light w begins!");
	int nthreads = 64;
	omp_set_num_threads(nthreads);
#pragma omp parallel for
	for(int i = 0; i < nvec[0]; i++)
		for(int j = 0; j < nl[0] + sc_size * nhits[0]; j++)
		{
			for(int x = 0; x < size_4d; x++)
			{
				int x_3d = x % size_3d;
				int t = x / size_3d;
				int glb_t = t + GJP.TnodeCoor() * GJP.TnodeSites();

				complex<double> *v_off = (complex<double> *)(v_fftw[i]) + x * sc_size;
				complex<double> *w_off;
				if(j<nl[0]) w_off = (complex<double> *)(wl_fftw[0][j]) + x * sc_size;
				else w_off = (complex<double> *)(wh_fftw[0]) + size_4d * sc_size * (j-nl[0]) + x * sc_size;

				complex<double> *mf_off = (complex<double> *)mf + (j*nvec[0]+i)*t_size+glb_t;
				*mf_off+=Gamma5(v_off,w_off,((complex<double> *)src)[x_3d]);
			}
		}
	cout_time("Inner product of light v and light w ends!");

	QMP_sum_double_array((Float *)mf,nvec[0]*(nl[0]+sc_size*nhits[0])*t_size*2);
	QDPIO::cout<<"Global sum done!"<<endl;
}

void MesonField::cal_mf_ls(double rad, int src_kind)
{
	char *fname = "cal_mf_ls()";

	mf_ls->VecZero(nvec[1]*(nl[0]+sc_size*nhits[0])*t_size*2);
	prepare_src(rad, src_kind);

	cout_time("Inner product of strange v and light w begins!");
	int nthreads = 64;
	omp_set_num_threads(nthreads);
#pragma omp parallel for
	for(int i = 0; i < nvec[1] ; i++)
		for(int j = 0; j < nl[0] + sc_size * nhits[0]; j++)
		{
			for(int x = 0; x < size_4d; x++)
			{
				int x_3d = x % size_3d;
				int t = x / size_3d;
				int glb_t = t + GJP.TnodeCoor()*GJP.TnodeSites();

				complex<double> *v_off = (complex<double> *)(v_s_fftw[i]) + x * sc_size;
				complex<double> *w_off;
				if(j<nl[0]) w_off = (complex<double> *)(wl_fftw[0][j]) + x * sc_size;
				else w_off = (complex<double> *)(wh_fftw[0]) + size_4d * sc_size * (j-nl[0]) + x * sc_size;

				complex<double> *mf_off = (complex<double> *)mf_ls + (j*nvec[1]+i)*t_size+glb_t;
				*mf_off+=Gamma5(v_off,w_off,((complex<double> *)src)[x_3d]);
			}
		}
	cout_time("Inner product of strange v and light w ends!");

	QMP_sum_double_array((Float *)mf_ls,nvec[1]*(nl[0]+sc_size*nhits[0])*t_size*2);
	QDPIO::cout<<"Global sum done!"<<endl;
}

void MesonField::cal_mf_sl(double rad, int src_kind)
{
	char *fname = "cal_mf_sl()";
	
	mf_sl->VecZero(nvec[0]*(nl[1]+sc_size*nhits[1])*t_size*2);
	prepare_src(rad, src_kind);

	cout_time("Inner product of light v and strange w begins!");
	int nthreads = 64;
	omp_set_num_threads(nthreads);
#pragma omp parallel for
	for(int i = 0; i < nvec[0] ; i++)
		for(int j = 0; j < nl[1] + sc_size * nhits[1]; j++)
		{
			for(int x = 0; x < size_4d; x++)
			{
				int x_3d = x % size_3d;
				int t = x / size_3d;
				int glb_t = t + GJP.TnodeCoor()*GJP.TnodeSites();

				complex<double> *v_off = (complex<double> *)(v_fftw[i]) + x * sc_size;
				complex<double> *w_off;
				if(j<nl[1]) w_off = (complex<double> *)(wl_fftw[1][j]) + x * sc_size;
				else w_off = (complex<double> *)(wh_fftw[1]) + size_4d * sc_size * (j-nl[1]) + x * sc_size;

				complex<double> *mf_off = (complex<double> *)mf_sl + (j*nvec[0]+i)*t_size+glb_t;
				*mf_off+=Gamma5(v_off,w_off,((complex<double> *)src)[x_3d]);
			}
		}
	cout_time("Inner product of light v and strange w ends!");

	QMP_sum_double_array((Float *)mf_sl,nvec[0]*(nl[1]+sc_size*nhits[1])*t_size*2);
	QDPIO::cout<<"Global sum done!"<<endl;
}

void MesonField::mf_contraction(QUARK left, QUARK middle, QUARK right, complex<double> *mf_left, int t_sep, complex<double> *mf_right, complex<double> *result)
//The contraction of \PI^ij(t+sep) = \pi_left^ik(t+sep) \cdot \pi_right^kj(t), where the \pi_left is meson field for [left QUARK w, right middle QUARK v], and \pi_right is meson field for [middle QUARK w, right QUARK v].
{
	char *fname = "mf_contraction(QUARK, QUARK, complex<double>, complex<double>, complex<double>)";

	const int nvec_compact[2] = {this->nl[0] + sc_size * this->nhits[0], this->nl[1] + sc_size * this->nhits[1]};

	omp_set_num_threads(bfmarg::threads);
#pragma omp parallel for
	for(int i = 0; i < nvec_compact[left]; i++) {
		complex<double> *left_off, *right_off, *result_off;
		int offset_k;
		for(int j = 0; j < nvec[right]; j++) {
			for(int k = 0; k < nvec_compact[middle]; k++) {
				for(int t = 0; t < t_size; t++) {
					offset_k = k<nl[middle]?k:(nl[middle]+(k-nl[middle])/sc_size*nbase[middle]+t/src_width[middle]*sc_size+(k-nl[middle])%sc_size);
					//left_off = mf_left + t_size * (offset_k + nvec[left] * i) + (t + t_sep) % t_size;
					left_off = mf_left + t_size * (offset_k + nvec[middle] * i) + (t + t_sep) % t_size;
					right_off = mf_right + t_size * (j + nvec[right] * k) + t;
					result_off = result + t_size * (j + nvec[right] * i) + (t + t_sep) % t_size;
					*result_off += (*left_off) * (*right_off);
				}
			}
		}
	}
}

void MesonField::mf_contraction_ww(QUARK left, QUARK middle, QUARK right, complex<double> *mf_left, int t_sep, complex<double> *mf_right, complex<double> *result)
//The contraction of \PI^ij(t+sep) = \pi_left^ik(t+sep) \cdot \pi_right^kj(t), where the \pi_left is meson field for [left QUARK w, right middle QUARK v], and \pi_right is meson field for [middle QUARK w, right QUARK w].
{
	char *fname = "mf_contraction(QUARK, QUARK, complex<double>, complex<double>, complex<double>)";

	const int nvec_compact[2] = {this->nl[0] + sc_size * this->nhits[0], this->nl[1] + sc_size * this->nhits[1]};

	omp_set_num_threads(bfmarg::threads);
#pragma omp parallel for
	for(int i = 0; i < nvec_compact[left]; i++) {
		complex<double> *left_off, *right_off, *result_off;
		int offset_k;
		for(int j = 0; j < nvec_compact[right]; j++) {
			for(int k = 0; k < nvec_compact[middle]; k++) {
				for(int t = 0; t < t_size; t++) {
					offset_k = k<nl[middle]?k:(nl[middle]+(k-nl[middle])/sc_size*nbase[middle]+t/src_width[middle]*sc_size+(k-nl[middle])%sc_size);
					left_off = mf_left + t_size * (offset_k + nvec[middle] * i) + (t + t_sep) % t_size;
					right_off = mf_right + t_size * (j + nvec_compact[right] * k) + t;
					result_off = result + t_size * (j + nvec_compact[right] * i) + (t + t_sep) % t_size;
					*result_off += (*left_off) * (*right_off);
				}
			}
		}
	}
}

void MesonField::set_expsrc(fftw_complex *src, Float radius)
{
	const char *fname = "set_expsrc()";

	const int X = x_size;
	const int Y = y_size;
	const int Z = z_size;

	for(int x = 0; x < X; ++x) {
		for(int y = 0; y < Y; ++y) {
			for(int z = 0; z < Z; ++z) {
				int off = x + X * (y + Y * z);
				int xr = (x + X / 2) % X - X / 2;
				int yr = (y + Y / 2) % Y - Y / 2;
				int zr = (z + Z / 2) % Z - Z / 2;
				Float v = sqrt(xr * xr + yr * yr + zr * zr) / radius;
				src[off][0] = exp(-v) / (X * Y * Z);
				src[off][1] = 0.;
			}
		}
	}
}

void MesonField::set_boxsrc(fftw_complex *src, int size)
{
	const int glb_dim[3] = {
		GJP.XnodeSites() * GJP.Xnodes(),
		GJP.YnodeSites() * GJP.Ynodes(),
		GJP.ZnodeSites() * GJP.Znodes(),
	};

	const int bound1[3] = { size / 2,
		size / 2,
		size / 2 };

	const int bound2[3] = { glb_dim[0] - size + size / 2 + 1,
		glb_dim[1] - size + size / 2 + 1,
		glb_dim[2] - size + size / 2 + 1 };
	int glb_size_3d = glb_dim[0] * glb_dim[1] * glb_dim[2];

	int x[3];
	for(x[0] = 0; x[0] < glb_dim[0]; ++x[0]) {
		for(x[1] = 0; x[1] < glb_dim[1]; ++x[1]) {
			for(x[2] = 0; x[2] < glb_dim[2]; ++x[2]) {
				int offset = x[0] + glb_dim[0] * (x[1] + glb_dim[1] * x[2]     );

				if(
						(x[0] <= bound1[0] || x[0] >= bound2[0])
						&& (x[1] <= bound1[1] || x[1] >= bound2[1])
						&& (x[2] <= bound1[2] || x[2] >= bound2[2])
					) {
					src[offset][0] = 1. / glb_size_3d;
				}else {
					src[offset][0] = 0.;
				}
				src[offset][1] = 0.;
			}
		}
	}
}

void MesonField::prepare_src(double rad, int kind) // kind1 exp; kind2 box
{
	char *fname = "prepare_src()";
	const int fft_dim[3] = {z_size, y_size, x_size};
	const int size_3d_glb = fft_dim[0] * fft_dim[1] * fft_dim[2];

	fftw_complex *fft_mem = fftw_alloc_complex(size_3d_glb);

	fftw_plan plan_src = fftw_plan_many_dft(3, fft_dim, 1,
																					fft_mem, NULL, 1, size_3d_glb,
																					fft_mem, NULL, 1, size_3d_glb,
																					FFTW_FORWARD, FFTW_ESTIMATE);
  switch(kind)
	{
		case 1: set_expsrc(fft_mem, rad); break;
		case 2: set_boxsrc(fft_mem, int(rad)); break;
		default: VRB.Result(cname, fname, "Src kind not implemented yet!"); exit(1);
	}
	fftw_execute(plan_src);

	src_glb2lcl(fft_mem,src);

	fftw_destroy_plan(plan_src);
	fftw_free(fft_mem);
}

void MesonField::src_glb2lcl(fftw_complex *glb, fftw_complex *lcl)
{
	char *fname = "src_glb2lcl(fftw_complex *, fftw_complex *)";

	const int shift[3] = {
		GJP.XnodeSites() * GJP.XnodeCoor(),
		GJP.YnodeSites() * GJP.YnodeCoor(),
		GJP.ZnodeSites() * GJP.ZnodeCoor(),
	};

	for(int xyz = 0; xyz < size_4d / GJP.TnodeSites(); xyz++) {
		int tmp = xyz;
		int x = tmp % GJP.XnodeSites() + shift[0]; tmp /= GJP.XnodeSites();
		int y = tmp % GJP.YnodeSites() + shift[1]; tmp /= GJP.YnodeSites();
		int z = tmp + shift[2];
		int xyz_glb = x + x_size * (y + y_size * z);

		lcl[xyz][0] = glb[xyz_glb][0];
		lcl[xyz][1] = glb[xyz_glb][1];
	}
}

void MesonField::run_pipi(int sep)
{
	char *fname = "run_pipi(int sep)";

	const int nvec_this = nvec[0];
	const int nl_this = nl[0];
	const int nhits_this = nhits[0];
	const int nbase_this = nbase[0];
	const int src_width_this = src_width[0];

	complex<double> *mf_pi_pi = (complex<double> *)smalloc(cname,fname,"mf_pi_pi",sizeof(complex<double>) * t_size * t_size * nvec_this * (nl_this + nhits_this * sc_size));
	((Vector *)mf_pi_pi)->VecZero(t_size * t_size * nvec_this * (nl_this + nhits_this * sc_size));
	complex<double> *pioncorr = (complex<double> *)smalloc(cname,fname,"pioncorr",sizeof(complex<double>) * t_size);
	((Vector *)pioncorr)->VecZero(t_size * 2);
	complex<double> *FigureC = (complex<double> *)smalloc(cname,fname,"FigureC",sizeof(complex<double>) * t_size);
	((Vector *)FigureC)->VecZero(t_size * 2);
	complex<double> *FigureD = (complex<double> *)smalloc(cname,fname,"FigureD",sizeof(complex<double>) * t_size);
	((Vector *)FigureD)->VecZero(t_size * 2);
	complex<double> *FigureR = (complex<double> *)smalloc(cname,fname,"FigureR",sizeof(complex<double>) * t_size);
	((Vector *)FigureR)->VecZero(t_size * 2);
	complex<double> *FigureVdis = (complex<double> *)smalloc(cname,fname,"FigureVdis",sizeof(complex<double>) * t_size);
	((Vector *)FigureVdis)->VecZero(t_size * 2);
	complex<double> Dtmp1, Dtmp2, Dtmp3, Dtmp4;

	FILE *p;
	char fn[1024];

	//##################################################
	// Contraction of two meson fields
	//##################################################

	// mf_pi_pi[tsrc + t_size * ( tsnk + t_size * (j + nvec * i))] = Pi_{i,k}(tsrc) * Pi_{k,j}(tsnk)
	omp_set_num_threads(bfmarg::threads);
#pragma omp parallel for
	for(int tsrc = 0; tsrc < t_size; tsrc++)
		for(int tsnk = 0; tsnk < t_size; tsnk++)
		{
			for(int j = 0; j < nvec_this; j++)
				for(int i = 0; i < nl_this + nhits_this * sc_size; i++)
				{
					int mf_pi_pi_off = tsrc + t_size * ( tsnk + t_size * (j + nvec_this * i));
					for(int k = 0; k < nl_this + nhits_this * sc_size; k++)
					{
						int offset_k = k<nl_this?k:(nl_this + (k-nl_this)/12 * nbase_this + tsnk/src_width_this * 12 + (k-nl_this)%12);
						mf_pi_pi[mf_pi_pi_off] += ((complex<double> *)mf)[tsrc + t_size * (offset_k + nvec_this * i)] * ((complex<double> *)mf)[tsnk + t_size * (j + nvec_this * k)];
					}
				}
		}

	//##################################################
	// Pion
	//##################################################
	for(int tsrc=0;tsrc<t_size;tsrc++)
		for(int tsnk=0;tsnk<t_size;tsnk++)
		{
			int t_dis = (tsnk-tsrc+t_size)%t_size;
			for(int i=0;i<nl_this+nhits_this*12;i++)
			{
				int offset_i = i<nl_this?i:(nl_this + (i-nl_this)/12 * nbase_this + tsrc/src_width_this *12 + (i-nl_this)%12);
				pioncorr[t_dis] += mf_pi_pi[tsrc + t_size * ( tsnk + t_size * (offset_i + nvec_this * i))];
			}
		}
	sprintf(fn,"%s_pioncorr",common_arg->filename);
	if((p = Fopen(fn,"w")) == NULL)
		ERR.FileA(cname,fname,fn);
	for(int i=0;i<t_size;i++)
		Fprintf(p,"%d\t%.16e\t%.16e\n",i,pioncorr[i].real(),pioncorr[i].imag());
	Fclose(p);
	sfree(cname,fname,"pioncorr",pioncorr);

	//##################################################
	// Figure C = mf_pi_pi(i,j,tsrc,tsnk) * mf_pi_pi(j,i,tsrc2,tsnk2) 
	// !!The direction of loop matters!!
	//##################################################
	// Figure R = 1/2 * ( mf_pi_pi(i,j,tsrc,tsrc2) * mf_pi_pi(j,i,tsnk,tsnk2) + mf_pi_pi(i,j,tsrc,tsnk2) * mf_pi_pi(j,i,tsnk2,tsnk) ) 
	// !!The direction of loop matters!!
	//##################################################
	// Figure D = 1/2 * ( mf_pi_pi(i,i,tsrc,tsnk) * mf_pi_pi(j,j,tsrc2,tsnk2) + mf_pi_pi(k,k,tsrc,tsnk2) * mf_pi_pi(l,l,tsrc2,tsnk) )
	//##################################################
	// Figure Vdis = mf_pi_pi(i,i,tsrc,tsrc2)
	//##################################################
	for(int tsrc = 0; tsrc < t_size; tsrc++)
		for(int tsnk3 = tsrc + sep; tsnk3 < tsrc - sep + t_size; tsnk3++)
		{
			int t_dis = tsnk3 - sep - tsrc;
			int tsnk = tsnk3 % t_size;
			int tsrc2 = (tsrc + sep) % t_size;
			int tsnk2 = (tsnk + sep) % t_size;
			for(int i = 0; i< nl_this + nhits_this * sc_size; i++)
				for(int j = 0; j < nl_this + nhits_this * sc_size; j++)
				{
					int offset_i = i<nl_this?i:(nl_this + (i-nl_this)/12 * nbase_this + tsrc/src_width_this *12 + (i-nl_this)%12);
					int offset_j = j<nl_this?j:(nl_this + (j-nl_this)/12 * nbase_this + tsrc2/src_width_this * 12 + (j-nl_this)%12);
					FigureC[t_dis] += mf_pi_pi[tsrc + t_size * ( tsnk + t_size * (offset_j + nvec_this * i))] * mf_pi_pi[tsrc2 + t_size * ( tsnk2 + t_size * (offset_i + nvec_this * j))] * 0.5;
					FigureC[t_dis] += mf_pi_pi[tsrc + t_size * ( tsnk2 + t_size * (offset_j + nvec_this * i))] * mf_pi_pi[tsrc2 + t_size * ( tsnk + t_size * (offset_i + nvec_this * j))] * 0.5;

					offset_i = i<nl_this?i:(nl_this + (i-nl_this)/12 * nbase_this + tsrc/src_width_this *12 + (i - nl_this) % 12);
					offset_j = j<nl_this?j:(nl_this + (j-nl_this)/12 * nbase_this + tsnk/src_width_this *12 + (j - nl_this) % 12);
					FigureR[t_dis] += mf_pi_pi[tsrc + t_size * ( tsrc2 + t_size * (offset_j + nvec_this * i))] * mf_pi_pi[tsnk + t_size * ( tsnk2 + t_size * (offset_i + nvec_this * j))] * 0.25;
					offset_i = i<nl_this?i:(nl_this + (i-nl_this)/12 * nbase_this + tsrc2/src_width_this *12 + (i-nl_this)%12);
					offset_j = j<nl_this?j:(nl_this + (j-nl_this)/12 * nbase_this + tsnk2/src_width_this *12 + (j-nl_this)%12);
					FigureR[t_dis] += mf_pi_pi[tsrc2 + t_size * ( tsrc + t_size * (offset_j + nvec_this * i))] * mf_pi_pi[tsnk2 + t_size * ( tsnk + t_size * (offset_i + nvec_this * j))] * 0.25;
					offset_i = i<nl_this?i:(nl_this + (i-nl_this)/12 * nbase_this + tsrc/src_width_this *12 + (i - nl_this) % 12);
					offset_j = j<nl_this?j:(nl_this + (j-nl_this)/12 * nbase_this + tsnk2/src_width_this *12 + (j - nl_this) % 12);
					FigureR[t_dis] += mf_pi_pi[tsrc + t_size * ( tsrc2 + t_size * (offset_j + nvec_this * i))] * mf_pi_pi[tsnk2 + t_size * ( tsnk + t_size * (offset_i + nvec_this * j))] * 0.25;
					offset_i = i<nl_this?i:(nl_this + (i-nl_this)/12 * nbase_this + tsrc2/src_width_this *12 + (i - nl_this) % 12);
					offset_j = j<nl_this?j:(nl_this + (j-nl_this)/12 * nbase_this + tsnk/src_width_this *12 + (j - nl_this) % 12);
					FigureR[t_dis] += mf_pi_pi[tsrc2 + t_size * ( tsrc + t_size * (offset_j + nvec_this * i))] * mf_pi_pi[tsnk + t_size * ( tsnk2 + t_size * (offset_i + nvec_this * j))] * 0.25;
				}

			Dtmp1 = 0;
			Dtmp2 = 0;
			Dtmp3 = 0;
			Dtmp4 = 0;
			for(int i = 0; i < nl_this + nhits_this * sc_size;i++)
			{
				int offset_i = i<nl_this?i:(nl_this + (i-nl_this)/12 * nbase_this + tsrc/src_width_this *12 + (i - nl_this) % 12);
				Dtmp1 += mf_pi_pi[tsrc + t_size * ( tsnk + t_size * (offset_i + nvec_this * i))];
				offset_i = i<nl_this?i:(nl_this + (i-nl_this)/12 * nbase_this + tsrc2/src_width_this *12 + (i - nl_this) % 12);
				Dtmp2 += mf_pi_pi[tsrc2 + t_size * ( tsnk2 + t_size * (offset_i + nvec_this * i))];
				offset_i = i<nl_this?i:(nl_this + (i-nl_this)/12 * nbase_this + tsrc/src_width_this *12 + (i - nl_this) % 12);
				Dtmp3 += mf_pi_pi[tsrc + t_size * ( tsnk2 + t_size * (offset_i + nvec_this * i))];
				offset_i = i<nl_this?i:(nl_this + (i-nl_this)/12 * nbase_this + tsrc2/src_width_this *12 + (i - nl_this) % 12);
				Dtmp4 += mf_pi_pi[tsrc2 + t_size * ( tsnk + t_size * (offset_i + nvec_this * i))];
			}
			FigureD[t_dis] += (Dtmp1 * Dtmp2 + Dtmp3 * Dtmp4) * 0.5;
		}

	for(int tsrc=0;tsrc<t_size;tsrc++)
	{
		int tsrc2=(tsrc+sep)%t_size;
		for(int i=0;i<nl_this+nhits_this*12;i++)
		{
			int offset_i = i<nl_this?i:(nl_this + (i-nl_this)/12 * nbase_this + tsrc/src_width_this *12 + (i - nl_this) % 12);
			FigureVdis[tsrc] += mf_pi_pi[tsrc + t_size * ( tsrc2 + t_size * (offset_i + nvec_this * i))];
		}
	}

	sprintf(fn,"%s_FigureC_sep%d",common_arg->filename,sep);
	if((p = Fopen(fn,"w")) == NULL)
		ERR.FileA(cname,fname,fn);
	for(int i=0;i<t_size-2*sep;i++)
		Fprintf(p,"%d\t%.16e\t%.16e\n",i,FigureC[i].real()/t_size,FigureC[i].imag()/t_size);
	Fclose(p);

	sprintf(fn,"%s_FigureD_sep%d",common_arg->filename,sep);
	if((p = Fopen(fn,"w")) == NULL)
		ERR.FileA(cname,fname,fn);
	for(int i=0;i<t_size-sep*2;i++)
		Fprintf(p,"%d\t%.16e\t%.16e\n",i,FigureD[i].real()/t_size,FigureD[i].imag()/t_size);
	Fclose(p);

	sprintf(fn,"%s_FigureR_sep%d",common_arg->filename,sep);
	if((p = Fopen(fn,"w")) == NULL)
		ERR.FileA(cname,fname,fn);
	for(int i=0;i<t_size-sep*2;i++)
		Fprintf(p,"%d\t%.16e\t%.16e\n",i,FigureR[i].real()/t_size,FigureR[i].imag()/t_size);
	Fclose(p);

	sprintf(fn,"%s_FigureVdis_sep%d",common_arg->filename,sep);
	if((p = Fopen(fn,"w")) == NULL)
		ERR.FileA(cname,fname,fn);
	for(int i=0;i<t_size;i++)
		Fprintf(p,"%d\t%.16e\t%.16e\n",i,FigureVdis[i].real(),FigureVdis[i].imag());
	Fclose(p);

	sfree(cname,fname,"FigureC",FigureC);
	sfree(cname,fname,"FigureD",FigureD);
	sfree(cname,fname,"FigureR",FigureR);
	sfree(cname,fname,"FigureVdis",FigureVdis);
	sfree(cname,fname,"mf_pi_pi",mf_pi_pi);
}


void MesonField::run_kaon(double rad) 
{
	char *fname = "run_kaon()";

	char fn[1024];
	sprintf(fn,"%s_kaoncorr_%1.2f",common_arg->filename,rad);
	complex<double> *kaoncorr = (complex<double> *)smalloc(cname,fname,"kaoncorr",sizeof(complex<double>)*t_size);
	((Vector *)kaoncorr)->VecZero(t_size*2);
	int t_dis;
	int offset_i;
	int offset_j;
	for(int tsrc=0;tsrc<t_size;tsrc++)
		for(int tsnk=0;tsnk<t_size;tsnk++) {
			t_dis = (tsnk-tsrc+t_size)%t_size;
			for(int i=0;i<nl[0]+nhits[0]*12;i++)
				for(int j=0;j<nl[1]+nhits[1]*12;j++) {
					offset_i = i<nl[0]?i:(nl[0]+(i-nl[0])/12*nbase[0]+tsnk/src_width[0]*12+(i-nl[0])%12);
					offset_j = j<nl[1]?j:(nl[1]+(j-nl[1])/12*nbase[1]+tsrc/src_width[1]*12+(j-nl[1])%12);
					kaoncorr[t_dis] += ((complex<double> *)mf_sl)[tsrc + t_size * (offset_i + nvec[0] * j)] * ((complex<double> *)mf_ls)[tsnk + t_size * (offset_j + nvec[1] * i)];
				}
		}

	FILE *p;
	if((p = Fopen(fn,"w")) == NULL)
		ERR.FileA(cname,fname,fn);
	for(int i=0;i<t_size;i++)
		Fprintf(p,"%d\t%.16e\t%.16e\n",i,kaoncorr[i].real()/t_size,kaoncorr[i].imag()/t_size);
	Fclose(p);
	sfree(cname,fname,"kaoncorr",kaoncorr);
}

void MesonField::run_type1(int t_delta, int sep)
{
	char *fname = "run_type1()";

	Vector *mf_pi_k_1 = (Vector *)smalloc(cname, fname, "mf_pi_k_1", sizeof(Float)*(nl[0]+sc_size*nhits[0])*nvec[1]*t_size*2);//This the contraction of pion and kaon mesonfields, t is position of pion
	Vector *mf_pi_k_2 = (Vector *)smalloc(cname, fname, "mf_pi_k_2", sizeof(Float)*(nl[0]+sc_size*nhits[0])*nvec[1]*t_size*2);//This the contraction of pion and kaon mesonfields, t is position of pion
	mf_pi_k_1->VecZero((nl[0]+sc_size*nhits[0])*nvec[1]*t_size*2);
	mf_pi_k_2->VecZero((nl[0]+sc_size*nhits[0])*nvec[1]*t_size*2);

	int n_threads = bfmarg::threads;

	// Below is the contraction of \Pi_pi_k_1^ij(t+t_delta+sep) = \pi^ik(t+t_delta+sep) * {\pi^\prime}^kj(t)
	//                         and \Pi_pi_k_2^ij(t+t_delta    ) = \pi^ik(t+t_delta    ) * {\pi^\prime}^kj(t)
	mf_contraction(LIGHT, LIGHT, STRANGE, (complex<double>*)mf, t_delta+sep, (complex<double>*)mf_ls, (complex<double>*)mf_pi_k_1);
	mf_contraction(LIGHT, LIGHT, STRANGE, (complex<double>*)mf, t_delta    , (complex<double>*)mf_ls, (complex<double>*)mf_pi_k_2);

	cout_time("Contraction of type1 begins!");
	complex<double> *type1_threaded = (complex<double> *)smalloc(cname,fname,"type1_threaded",sizeof(complex<double>)*t_size*8*n_threads);// 8 because 8 different contractions
	((Vector*)type1_threaded)->VecZero(t_size*8*n_threads*2);

	omp_set_num_threads(n_threads);
#pragma omp parallel for 
	for(int x = 0; x < size_4d; x++) {
		int id = omp_get_thread_num();
		int t_op = x / size_3d + GJP.TnodeSites() * GJP.TnodeCoor();
		complex<double> *offset_type1;
		for(int t_k = 0; t_k < t_size; t_k++) {
			int t_dis = (t_op - t_k + t_size) % t_size;
			WilsonMatrix Mtemp1 = Build_Wilson(mf, x, (t_k + t_delta) % t_size, LIGHT, LIGHT);
			WilsonMatrix Mtemp2 = Build_Wilson(mf_pi_k_1, x, (t_k + t_delta + sep) % t_size, LIGHT, STRANGE);


			for(int dir = 0; dir < 4; dir++) {
				//--------------------Fig 1 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 0) * n_threads + id;
				(*offset_type1)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type1)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 3 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 2) * n_threads + id;
				(*offset_type1)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type1)+=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 2 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 1) * n_threads + id;
				(*offset_type1)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type1)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 4 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 3) * n_threads + id;
				(*offset_type1)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type1)+=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 5 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 4) * n_threads + id;
				(*offset_type1)-=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
				(*offset_type1)-=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
				//--------------------Fig 7 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 6) * n_threads + id;
				(*offset_type1)+=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
				(*offset_type1)-=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
				//--------------------Fig 6 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 5) * n_threads + id;
				(*offset_type1)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type1)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 8 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 7) * n_threads + id;
				(*offset_type1)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type1)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			}

			//--------------------Now permute two pions--------------------//
			t_dis = (t_op - (t_k - sep) + t_size) % t_size;
			Mtemp2 = Build_Wilson(mf_pi_k_2, x, (t_k + t_delta - sep + t_size) % t_size, LIGHT, STRANGE);

			for(int dir = 0; dir < 4; dir++) {
				//--------------------Fig 1 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 0) * n_threads + id;
				(*offset_type1)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type1)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 3 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 2) * n_threads + id;
				(*offset_type1)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type1)+=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 2 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 1) * n_threads + id;
				(*offset_type1)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type1)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 4 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 3) * n_threads + id;
				(*offset_type1)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type1)+=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 5 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 4) * n_threads + id;
				(*offset_type1)-=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
				(*offset_type1)-=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
				//--------------------Fig 7 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 6) * n_threads + id;
				(*offset_type1)+=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
				(*offset_type1)-=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
				//--------------------Fig 6 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 5) * n_threads + id;
				(*offset_type1)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type1)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 8 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 7) * n_threads + id;
				(*offset_type1)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type1)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			}
		}
	}
	sfree(cname,fname,"mf_pi_k_1",mf_pi_k_1);
	sfree(cname,fname,"mf_pi_k_2",mf_pi_k_2);
	complex<double> *type1 = (complex<double> *)smalloc(cname,fname,"type1",sizeof(complex<double>)*t_size*8);
	((Vector*)type1)->VecZero(t_size*8*2);
	for(int i = 0; i < t_size*8; i++)
		for(int id = 0; id < n_threads; id++)
			type1[i] += type1_threaded[i*n_threads+id] * 0.5 / t_size; // 0.5 because doing permutation of two pions
	sfree(cname,fname,"type1_threaded",type1_threaded);
	QMP_sum_double_array((Float *)type1,t_size*8*2);
	cout_time("Contraction of type1 ends!");

	char fn[1024];
	sprintf(fn,"%s_type1",common_arg->filename);
	writeCorr(type1, fn, 8, t_size);

	sfree(cname,fname,"type1",type1);
}

void MesonField::run_type1_three_sink_approach(int t_delta, int sep) // This is treating three of the four quark lines at the operator as sinks
{
	char *fname = "run_type1_three_sink_approach(int t_delta, int sep)";

	const int nvec_compact_l = nl[0] + sc_size * nhits[0];
	const int nvec_compact_s = nl[1] + sc_size * nhits[1];

	Vector *mf_pi_k_1 = (Vector *)smalloc(cname, fname, "mf_pi_k_1", sizeof(Float)*nvec_compact_l*nvec_compact_s*t_size*2);//This the contraction of pion and kaon mesonfields, t is position of pion
	Vector *mf_pi_k_2 = (Vector *)smalloc(cname, fname, "mf_pi_k_2", sizeof(Float)*nvec_compact_l*nvec_compact_s*t_size*2);//This the contraction of pion and kaon mesonfields, t is position of pion
	mf_pi_k_1->VecZero(nvec_compact_l*nvec_compact_s*t_size*2);
	mf_pi_k_2->VecZero(nvec_compact_l*nvec_compact_s*t_size*2);

	int n_threads = bfmarg::threads;

	// Below is the contraction of \Pi_pi_k_1^ij(t+t_delta+sep) = \pi^ik(t+t_delta+sep) * {\pi^\prime}^kj(t)
	//                         and \Pi_pi_k_2^ij(t+t_delta    ) = \pi^ik(t+t_delta    ) * {\pi^\prime}^kj(t)
	mf_contraction_ww(LIGHT, LIGHT, STRANGE, (complex<double>*)mf, t_delta+sep, (complex<double>*)(mf_ww[1]), (complex<double>*)mf_pi_k_1);
	mf_contraction_ww(LIGHT, LIGHT, STRANGE, (complex<double>*)mf, t_delta    , (complex<double>*)(mf_ww[1]), (complex<double>*)mf_pi_k_2);

	cout_time("Contraction of type1 begins!");
	complex<double> *type1_threaded = (complex<double> *)smalloc(cname,fname,"type1_threaded",sizeof(complex<double>)*t_size*8*n_threads);// 8 because 8 different contractions
	((Vector*)type1_threaded)->VecZero(t_size*8*n_threads*2);

	omp_set_num_threads(n_threads);
#pragma omp parallel for 
	for(int x = 0; x < size_4d; x++) {
		int id = omp_get_thread_num();
		int t_op = x / size_3d + GJP.TnodeSites() * GJP.TnodeCoor();
		complex<double> *offset_type1;
		for(int t_k = 0; t_k < t_size; t_k++) {
			int t_dis = (t_op - t_k + t_size) % t_size;
			int t_pi_1 = (t_k + t_delta) % t_size;
			int t_pi_2 = (t_pi_1 + sep) % t_size;
			//WilsonMatrix Mtemp1 = Build_Wilson_ww(mf_ww[0], x, t_pi_1, t_pi_1, LIGHT, LIGHT);
			WilsonMatrix Mtemp1 = Build_Wilson(mf, x, t_pi_1, LIGHT, LIGHT);
			WilsonMatrix Mtemp2 = Build_Wilson_ww(mf_pi_k_1, x, t_pi_2, t_k, LIGHT, STRANGE);


			for(int dir = 0; dir < 4; dir++) {
				//--------------------Fig 1 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 0) * n_threads + id;
				(*offset_type1)+=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type1)+=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 3 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 2) * n_threads + id;
				(*offset_type1)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type1)+=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 2 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 1) * n_threads + id;
				(*offset_type1)+=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type1)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 4 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 3) * n_threads + id;
				(*offset_type1)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type1)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 5 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 4) * n_threads + id;
				(*offset_type1)+=Trace(Mtemp1.glA(dir) , Mtemp2.glA(dir));
				(*offset_type1)+=Trace(Mtemp1.glV(dir) , Mtemp2.glV(dir));
				//--------------------Fig 7 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 6) * n_threads + id;
				(*offset_type1)-=Trace(Mtemp1.glA(dir) , Mtemp2.glA(dir));
				(*offset_type1)+=Trace(Mtemp1.glV(dir) , Mtemp2.glV(dir));
				//--------------------Fig 6 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 5) * n_threads + id;
				(*offset_type1)+=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type1)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 8 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 7) * n_threads + id;
				(*offset_type1)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type1)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
			}

			//--------------------Now permute two pions--------------------//
			t_pi_2 = (t_pi_1 - sep + t_size) % t_size;
			t_k = (t_pi_2 - t_delta + t_size) % t_size;
			t_dis = (t_op - t_k + t_size) % t_size;
			Mtemp2 = Build_Wilson_ww(mf_pi_k_2, x, t_pi_2, t_k, LIGHT, STRANGE);

			for(int dir = 0; dir < 4; dir++) {
				//--------------------Fig 1 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 0) * n_threads + id;
				(*offset_type1)+=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type1)+=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 3 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 2) * n_threads + id;
				(*offset_type1)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type1)+=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 2 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 1) * n_threads + id;
				(*offset_type1)+=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type1)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 4 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 3) * n_threads + id;
				(*offset_type1)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type1)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 5 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 4) * n_threads + id;
				(*offset_type1)+=Trace(Mtemp1.glA(dir) , Mtemp2.glA(dir));
				(*offset_type1)+=Trace(Mtemp1.glV(dir) , Mtemp2.glV(dir));
				//--------------------Fig 7 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 6) * n_threads + id;
				(*offset_type1)-=Trace(Mtemp1.glA(dir) , Mtemp2.glA(dir));
				(*offset_type1)+=Trace(Mtemp1.glV(dir) , Mtemp2.glV(dir));
				//--------------------Fig 6 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 5) * n_threads + id;
				(*offset_type1)+=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type1)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 8 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 7) * n_threads + id;
				(*offset_type1)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type1)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
			}

		}
	}
	sfree(cname,fname,"mf_pi_k_1",mf_pi_k_1);
	sfree(cname,fname,"mf_pi_k_2",mf_pi_k_2);
	complex<double> *type1 = (complex<double> *)smalloc(cname,fname,"type1",sizeof(complex<double>)*t_size*8);
	((Vector*)type1)->VecZero(t_size*8*2);
	for(int i = 0; i < t_size*8; i++)
		for(int id = 0; id < n_threads; id++)
			type1[i] += type1_threaded[i*n_threads+id] * 0.5 / t_size; // 0.5 because doing permutation of two pions
	sfree(cname,fname,"type1_threaded",type1_threaded);
	QMP_sum_double_array((Float *)type1,t_size*8*2);
	cout_time("Contraction of type1 ends!");

	char fn[1024];
	sprintf(fn,"%s_type1_three_sink_approach",common_arg->filename);
	writeCorr(type1, fn, 8, t_size);

	sfree(cname,fname,"type1",type1);
}

void MesonField::run_type1_four_sink_approach(int t_delta, int sep) // This is treating all four quark lines at the operator as sinks
{
	char *fname = "run_type1_four_sink_approach(int t_delta, int sep)";

	const int nvec_compact_l = nl[0] + sc_size * nhits[0];
	const int nvec_compact_s = nl[1] + sc_size * nhits[1];

	Vector *mf_pi_k_1 = (Vector *)smalloc(cname, fname, "mf_pi_k_1", sizeof(Float)*nvec_compact_l*nvec_compact_s*t_size*2);//This the contraction of pion and kaon mesonfields, t is position of pion
	Vector *mf_pi_k_2 = (Vector *)smalloc(cname, fname, "mf_pi_k_2", sizeof(Float)*nvec_compact_l*nvec_compact_s*t_size*2);//This the contraction of pion and kaon mesonfields, t is position of pion
	mf_pi_k_1->VecZero(nvec_compact_l*nvec_compact_s*t_size*2);
	mf_pi_k_2->VecZero(nvec_compact_l*nvec_compact_s*t_size*2);

	int n_threads = bfmarg::threads;

	// Below is the contraction of \Pi_pi_k_1^ij(t+t_delta+sep) = \pi^ik(t+t_delta+sep) * {\pi^\prime}^kj(t)
	//                         and \Pi_pi_k_2^ij(t+t_delta    ) = \pi^ik(t+t_delta    ) * {\pi^\prime}^kj(t)
	mf_contraction_ww(LIGHT, LIGHT, STRANGE, (complex<double>*)mf, t_delta+sep, (complex<double>*)(mf_ww[1]), (complex<double>*)mf_pi_k_1);
	mf_contraction_ww(LIGHT, LIGHT, STRANGE, (complex<double>*)mf, t_delta    , (complex<double>*)(mf_ww[1]), (complex<double>*)mf_pi_k_2);

	cout_time("Contraction of type1 four sink approach begins!");
	complex<double> *type1_threaded = (complex<double> *)smalloc(cname,fname,"type1_threaded",sizeof(complex<double>)*t_size*8*n_threads);// 8 because 8 different contractions
	((Vector*)type1_threaded)->VecZero(t_size*8*n_threads*2);

	omp_set_num_threads(n_threads);
#pragma omp parallel for 
	for(int x = 0; x < size_4d; x++) {
		int id = omp_get_thread_num();
		int t_op = x / size_3d + GJP.TnodeSites() * GJP.TnodeCoor();
		complex<double> *offset_type1;
		for(int t_k = 0; t_k < t_size; t_k++) {
			int t_dis = (t_op - t_k + t_size) % t_size;
			int t_pi_1 = (t_k + t_delta) % t_size;
			int t_pi_2 = (t_pi_1 + sep) % t_size;
			WilsonMatrix Mtemp1 = Build_Wilson_ww(mf_ww[0], x, t_pi_1, t_pi_1, LIGHT, LIGHT);
			WilsonMatrix Mtemp2 = Build_Wilson_ww(mf_pi_k_1, x, t_pi_2, t_k, LIGHT, STRANGE);


			for(int dir = 0; dir < 4; dir++) {
				//--------------------Fig 1 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 0) * n_threads + id;
				(*offset_type1)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type1)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 3 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 2) * n_threads + id;
				(*offset_type1)+=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type1)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 2 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 1) * n_threads + id;
				(*offset_type1)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type1)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 4 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 3) * n_threads + id;
				(*offset_type1)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type1)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 5 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 4) * n_threads + id;
				(*offset_type1)-=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
				(*offset_type1)-=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
				//--------------------Fig 7 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 6) * n_threads + id;
				(*offset_type1)-=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
				(*offset_type1)+=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
				//--------------------Fig 6 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 5) * n_threads + id;
				(*offset_type1)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type1)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 8 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 7) * n_threads + id;
				(*offset_type1)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type1)+=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			}

			//--------------------Now permute two pions--------------------//
			t_pi_2 = (t_pi_1 - sep + t_size) % t_size;
			t_k = (t_pi_2 - t_delta + t_size) % t_size;
			t_dis = (t_op - t_k + t_size) % t_size;
			Mtemp2 = Build_Wilson_ww(mf_pi_k_2, x, t_pi_2, t_k, LIGHT, STRANGE);

			for(int dir = 0; dir < 4; dir++) {
				//--------------------Fig 1 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 0) * n_threads + id;
				(*offset_type1)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type1)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 3 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 2) * n_threads + id;
				(*offset_type1)+=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type1)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 2 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 1) * n_threads + id;
				(*offset_type1)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type1)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 4 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 3) * n_threads + id;
				(*offset_type1)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type1)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 5 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 4) * n_threads + id;
				(*offset_type1)-=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
				(*offset_type1)-=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
				//--------------------Fig 7 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 6) * n_threads + id;
				(*offset_type1)-=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
				(*offset_type1)+=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
				//--------------------Fig 6 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 5) * n_threads + id;
				(*offset_type1)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type1)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 8 below--------------------//
				offset_type1 = type1_threaded + (t_dis + t_size * 7) * n_threads + id;
				(*offset_type1)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type1)+=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			}
		}
	}
	sfree(cname,fname,"mf_pi_k_1",mf_pi_k_1);
	sfree(cname,fname,"mf_pi_k_2",mf_pi_k_2);
	complex<double> *type1 = (complex<double> *)smalloc(cname,fname,"type1",sizeof(complex<double>)*t_size*8);
	((Vector*)type1)->VecZero(t_size*8*2);
	for(int i = 0; i < t_size*8; i++)
		for(int id = 0; id < n_threads; id++)
			type1[i] += type1_threaded[i*n_threads+id] * 0.5 / t_size; // 0.5 because doing permutation of two pions
	sfree(cname,fname,"type1_threaded",type1_threaded);
	QMP_sum_double_array((Float *)type1,t_size*8*2);
	cout_time("Contraction of type1 four sink approach ends!");

	char fn[1024];
	sprintf(fn,"%s_type1_four_sink_approach",common_arg->filename);
	writeCorr(type1, fn, 8, t_size);

	sfree(cname,fname,"type1",type1);
}

void MesonField::run_type2_four_sink_approach(int t_delta, int sep) 
{
	char *fname = "run_type2_four_sink_approach()";

	const int nvec_compact_l = nl[0] + sc_size * nhits[0];
	const int nvec_compact_s = nl[1] + sc_size * nhits[1];

	Vector *mf_pi_pi_1 = (Vector *)smalloc(cname, fname, "mf_pi_pi_1", sizeof(Float)*nvec_compact_l*nvec_compact_l*t_size*2);//This the contraction of pion and kaon mesonfields, t is position of first pion
	Vector *mf_pi_pi_2 = (Vector *)smalloc(cname, fname, "mf_pi_pi_2", sizeof(Float)*nvec_compact_l*nvec_compact_l*t_size*2);//This the contraction of pion and kaon mesonfields, t is position of first pion
	mf_pi_pi_1->VecZero(nvec_compact_l*nvec_compact_l*t_size*2);
	mf_pi_pi_2->VecZero(nvec_compact_l*nvec_compact_l*t_size*2);

	int n_threads = bfmarg::threads;
	// Below is the contraction of \Pi_1^ij(t+sep) = \pi^ik(t+sep) * \pi^kj(t)
	mf_contraction_ww(LIGHT, LIGHT, LIGHT, (complex<double>*)mf,          sep, (complex<double>*)(mf_ww[0]), (complex<double>*)mf_pi_pi_1);
	// Below is the contraction of \Pi_2^ij(t-sep) = \pi^ik(t-sep) * \pi^kj(t)
	mf_contraction_ww(LIGHT, LIGHT, LIGHT, (complex<double>*)mf, t_size - sep, (complex<double>*)(mf_ww[0]), (complex<double>*)mf_pi_pi_2);

	cout_time("Contraction of type2 four sink approach begins!");
	complex<double> *type2_threaded = (complex<double> *)smalloc(cname,fname,"type2_threaded",sizeof(complex<double>)*t_size*8*n_threads);// 8 because 8 different contractions
	((Vector*)type2_threaded)->VecZero(t_size*8*n_threads*2);

	omp_set_num_threads(n_threads);
#pragma omp parallel for
	for(int x = 0; x < size_4d; x++) {
  	int id = omp_get_thread_num();
		int t_op = x / size_3d + GJP.TnodeSites()*GJP.TnodeCoor();
		complex<double> *offset_type2;
		for(int t_k = 0; t_k < t_size; t_k++) {
			int t_dis = (t_op - t_k + t_size) % t_size;
			int t_pi_1 = (t_k + t_delta) % t_size;
			int t_pi_2 = (t_pi_1 + sep) % t_size;
			WilsonMatrix Mtemp1 = Build_Wilson_ww(mf_ww[1], x, t_k, t_k, LIGHT, STRANGE);
			WilsonMatrix Mtemp2 = Build_Wilson_ww(mf_pi_pi_1, x, t_pi_2, t_pi_1, LIGHT, LIGHT);

			for(int dir = 0; dir < 4; dir++) {
			//--------------------Fig 9 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 0) * n_threads + id;
			(*offset_type2)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
			(*offset_type2)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
			//--------------------Fig 11 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 2) * n_threads + id;
			(*offset_type2)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
			(*offset_type2)+=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
			//--------------------Fig 10 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 1) * n_threads + id;
			(*offset_type2)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
			(*offset_type2)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
			//--------------------Fig 12 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 3) * n_threads + id;
			(*offset_type2)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
			(*offset_type2)+=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
			//--------------------Fig 13 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 4) * n_threads + id;
			(*offset_type2)-=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
			(*offset_type2)-=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
			//--------------------Fig 15 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 6) * n_threads + id;
			(*offset_type2)+=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
			(*offset_type2)-=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
			//--------------------Fig 14 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 5) * n_threads + id;
			(*offset_type2)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
			(*offset_type2)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			//--------------------Fig 16 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 7) * n_threads + id;
			(*offset_type2)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
			(*offset_type2)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			}

			//--------------------Now permute two pion--------------------//
			Mtemp2 = Build_Wilson_ww(mf_pi_pi_2, x, t_pi_1, t_pi_2, LIGHT, LIGHT);

			for(int dir = 0; dir < 4; dir++) {
			//--------------------Fig 9 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 0) * n_threads + id;
			(*offset_type2)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
			(*offset_type2)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
			//--------------------Fig 11 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 2) * n_threads + id;
			(*offset_type2)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
			(*offset_type2)+=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
			//--------------------Fig 10 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 1) * n_threads + id;
			(*offset_type2)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
			(*offset_type2)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
			//--------------------Fig 12 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 3) * n_threads + id;
			(*offset_type2)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
			(*offset_type2)+=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
			//--------------------Fig 13 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 4) * n_threads + id;
			(*offset_type2)-=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
			(*offset_type2)-=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
			//--------------------Fig 15 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 6) * n_threads + id;
			(*offset_type2)+=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
			(*offset_type2)-=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
			//--------------------Fig 14 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 5) * n_threads + id;
			(*offset_type2)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
			(*offset_type2)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			//--------------------Fig 16 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 7) * n_threads + id;
			(*offset_type2)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
			(*offset_type2)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			}
		}
	}
	sfree(cname,fname,"mf_pi_pi_1",mf_pi_pi_1);
	sfree(cname,fname,"mf_pi_pi_2",mf_pi_pi_2);
	complex<double> *type2 = (complex<double> *)smalloc(cname,fname,"type2",sizeof(complex<double>)*t_size*8);
  ((Vector*)type2)->VecZero(t_size*8*2);
	for(int i = 0; i < t_size*8; i++)
		for(int id = 0; id < n_threads; id++)
			type2[i] += type2_threaded[i*n_threads+id] * 0.5 / t_size; // 0.5 because doing permutation of two pions
	sfree(cname,fname,"type2_threaded",type2_threaded);
	QMP_sum_double_array((Float *)type2,t_size*8*2);
	cout_time("Contraction of type2 four sink approach ends!");

	char fn[1024];
	sprintf(fn,"%s_type2_four_sink_approach",common_arg->filename);
	writeCorr(type2, fn, 8, t_size);

	sfree(cname,fname,"type2",type2);
}

void MesonField::run_type2_three_sink_approach(int t_delta, int sep) 
{
	char *fname = "run_type2_three_sink_approach()";

	Vector *mf_pi_pi_1 = (Vector *)smalloc(cname, fname, "mf_pi_pi_1", sizeof(Float)*(nl[0]+sc_size*nhits[0])*nvec[0]*t_size*2);//This the contraction of pion and kaon mesonfields, t is position of first pion
	Vector *mf_pi_pi_2 = (Vector *)smalloc(cname, fname, "mf_pi_pi_2", sizeof(Float)*(nl[0]+sc_size*nhits[0])*nvec[0]*t_size*2);//This the contraction of pion and kaon mesonfields, t is position of first pion
	mf_pi_pi_1->VecZero((nl[0]+sc_size*nhits[0])*nvec[0]*t_size*2);
	mf_pi_pi_2->VecZero((nl[0]+sc_size*nhits[0])*nvec[0]*t_size*2);

	int n_threads = bfmarg::threads;
	// Below is the contraction of \Pi_1^ij(t+sep) = \pi^ik(t+sep) * \pi^kj(t)
	mf_contraction(LIGHT, LIGHT, LIGHT, (complex<double>*)mf,          sep, (complex<double>*)mf, (complex<double>*)mf_pi_pi_1);
	// Below is the contraction of \Pi_2^ij(t-sep) = \pi^ik(t-sep) * \pi^kj(t)
	mf_contraction(LIGHT, LIGHT, LIGHT, (complex<double>*)mf, t_size - sep, (complex<double>*)mf, (complex<double>*)mf_pi_pi_2);

	cout_time("Contraction of type2 begins!");
	complex<double> *type2_threaded = (complex<double> *)smalloc(cname,fname,"type2_threaded",sizeof(complex<double>)*t_size*8*n_threads);// 8 because 8 different contractions
	((Vector*)type2_threaded)->VecZero(t_size*8*n_threads*2);

	omp_set_num_threads(n_threads);
#pragma omp parallel for
	for(int x = 0; x < size_4d; x++) {
  	int id = omp_get_thread_num();
		int t_op = x / size_3d + GJP.TnodeSites()*GJP.TnodeCoor();
		complex<double> *offset_type2;
		for(int t_k = 0; t_k < t_size; t_k++) {
			int t_dis = (t_op - t_k + t_size) % t_size;
			//WilsonMatrix Mtemp1 = Build_Wilson(mf_ls, x, t_k, LIGHT, STRANGE);
			WilsonMatrix Mtemp1 = Build_Wilson_ww(mf_ww[1], x, t_k, t_k, LIGHT, STRANGE);
			WilsonMatrix Mtemp2 = Build_Wilson(mf_pi_pi_1, x, (t_k + t_delta + sep) % t_size, LIGHT, LIGHT);

			for(int dir = 0; dir < 4; dir++) {
			//--------------------Fig 9 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 0) * n_threads + id;
			(*offset_type2)+=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
			(*offset_type2)+=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
			//--------------------Fig 11 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 2) * n_threads + id;
			(*offset_type2)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
			(*offset_type2)+=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
			//--------------------Fig 10 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 1) * n_threads + id;
			(*offset_type2)+=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
			(*offset_type2)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
			//--------------------Fig 12 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 3) * n_threads + id;
			(*offset_type2)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
			(*offset_type2)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
			//--------------------Fig 13 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 4) * n_threads + id;
			(*offset_type2)+=Trace(Mtemp1.glA(dir) , Mtemp2.glA(dir));
			(*offset_type2)+=Trace(Mtemp1.glV(dir) , Mtemp2.glV(dir));
			//--------------------Fig 15 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 6) * n_threads + id;
			(*offset_type2)-=Trace(Mtemp1.glA(dir) , Mtemp2.glA(dir));
			(*offset_type2)+=Trace(Mtemp1.glV(dir) , Mtemp2.glV(dir));
			//--------------------Fig 14 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 5) * n_threads + id;
			(*offset_type2)+=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
			(*offset_type2)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
			//--------------------Fig 16 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 7) * n_threads + id;
			(*offset_type2)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
			(*offset_type2)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
			}

			//--------------------Now permute two pion--------------------//
			Mtemp2 = Build_Wilson(mf_pi_pi_2, x, (t_k + t_delta) % t_size, LIGHT, LIGHT);

			for(int dir = 0; dir < 4; dir++) {
			//--------------------Fig 9 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 0) * n_threads + id;
			(*offset_type2)+=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
			(*offset_type2)+=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
			//--------------------Fig 11 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 2) * n_threads + id;
			(*offset_type2)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
			(*offset_type2)+=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
			//--------------------Fig 10 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 1) * n_threads + id;
			(*offset_type2)+=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
			(*offset_type2)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
			//--------------------Fig 12 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 3) * n_threads + id;
			(*offset_type2)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
			(*offset_type2)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
			//--------------------Fig 13 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 4) * n_threads + id;
			(*offset_type2)+=Trace(Mtemp1.glA(dir) , Mtemp2.glA(dir));
			(*offset_type2)+=Trace(Mtemp1.glV(dir) , Mtemp2.glV(dir));
			//--------------------Fig 15 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 6) * n_threads + id;
			(*offset_type2)-=Trace(Mtemp1.glA(dir) , Mtemp2.glA(dir));
			(*offset_type2)+=Trace(Mtemp1.glV(dir) , Mtemp2.glV(dir));
			//--------------------Fig 14 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 5) * n_threads + id;
			(*offset_type2)+=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
			(*offset_type2)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
			//--------------------Fig 16 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 7) * n_threads + id;
			(*offset_type2)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
			(*offset_type2)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
			}
		}
	}
	sfree(cname,fname,"mf_pi_pi_1",mf_pi_pi_1);
	sfree(cname,fname,"mf_pi_pi_2",mf_pi_pi_2);
	complex<double> *type2 = (complex<double> *)smalloc(cname,fname,"type2",sizeof(complex<double>)*t_size*8);
  ((Vector*)type2)->VecZero(t_size*8*2);
	for(int i = 0; i < t_size*8; i++)
		for(int id = 0; id < n_threads; id++)
			type2[i] += type2_threaded[i*n_threads+id] * 0.5 / t_size; // 0.5 because doing permutation of two pions
	sfree(cname,fname,"type2_threaded",type2_threaded);
	QMP_sum_double_array((Float *)type2,t_size*8*2);
	cout_time("Contraction of type2 ends!");

	char fn[1024];
	sprintf(fn,"%s_type2_three_sink_approach",common_arg->filename);
	writeCorr(type2, fn, 8, t_size);

	sfree(cname,fname,"type2",type2);
}

void MesonField::run_type2(int t_delta, int sep) 
{
	char *fname = "run_type2()";

	Vector *mf_pi_pi_1 = (Vector *)smalloc(cname, fname, "mf_pi_pi_1", sizeof(Float)*(nl[0]+sc_size*nhits[0])*nvec[0]*t_size*2);//This the contraction of pion and kaon mesonfields, t is position of first pion
	Vector *mf_pi_pi_2 = (Vector *)smalloc(cname, fname, "mf_pi_pi_2", sizeof(Float)*(nl[0]+sc_size*nhits[0])*nvec[0]*t_size*2);//This the contraction of pion and kaon mesonfields, t is position of first pion
	mf_pi_pi_1->VecZero((nl[0]+sc_size*nhits[0])*nvec[0]*t_size*2);
	mf_pi_pi_2->VecZero((nl[0]+sc_size*nhits[0])*nvec[0]*t_size*2);

	int n_threads = bfmarg::threads;
	// Below is the contraction of \Pi_1^ij(t+sep) = \pi^ik(t+sep) * \pi^kj(t)
	mf_contraction(LIGHT, LIGHT, LIGHT, (complex<double>*)mf,          sep, (complex<double>*)mf, (complex<double>*)mf_pi_pi_1);
	// Below is the contraction of \Pi_2^ij(t-sep) = \pi^ik(t-sep) * \pi^kj(t)
	mf_contraction(LIGHT, LIGHT, LIGHT, (complex<double>*)mf, t_size - sep, (complex<double>*)mf, (complex<double>*)mf_pi_pi_2);

	cout_time("Contraction of type2 begins!");
	complex<double> *type2_threaded = (complex<double> *)smalloc(cname,fname,"type2_threaded",sizeof(complex<double>)*t_size*8*n_threads);// 8 because 8 different contractions
	((Vector*)type2_threaded)->VecZero(t_size*8*n_threads*2);

	omp_set_num_threads(n_threads);
#pragma omp parallel for
	for(int x = 0; x < size_4d; x++) {
  	int id = omp_get_thread_num();
		int t_op = x / size_3d + GJP.TnodeSites()*GJP.TnodeCoor();
		complex<double> *offset_type2;
		for(int t_k = 0; t_k < t_size; t_k++) {
			int t_dis = (t_op - t_k + t_size) % t_size;
			WilsonMatrix Mtemp1 = Build_Wilson(mf_ls, x, t_k, LIGHT, STRANGE);
			WilsonMatrix Mtemp2 = Build_Wilson(mf_pi_pi_1, x, (t_k + t_delta + sep) % t_size, LIGHT, LIGHT);

			for(int dir = 0; dir < 4; dir++) {
			//--------------------Fig 9 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 0) * n_threads + id;
			(*offset_type2)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
			(*offset_type2)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
			//--------------------Fig 11 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 2) * n_threads + id;
			(*offset_type2)+=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
			(*offset_type2)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
			//--------------------Fig 10 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 1) * n_threads + id;
			(*offset_type2)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
			(*offset_type2)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
			//--------------------Fig 12 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 3) * n_threads + id;
			(*offset_type2)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
			(*offset_type2)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
			//--------------------Fig 13 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 4) * n_threads + id;
			(*offset_type2)-=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
			(*offset_type2)-=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
			//--------------------Fig 15 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 6) * n_threads + id;
			(*offset_type2)-=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
			(*offset_type2)+=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
			//--------------------Fig 14 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 5) * n_threads + id;
			(*offset_type2)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
			(*offset_type2)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			//--------------------Fig 16 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 7) * n_threads + id;
			(*offset_type2)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
			(*offset_type2)+=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			}

			//--------------------Now permute two pion--------------------//
			Mtemp2 = Build_Wilson(mf_pi_pi_2, x, (t_k + t_delta) % t_size, LIGHT, LIGHT);

			for(int dir = 0; dir < 4; dir++) {
			//--------------------Fig 9 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 0) * n_threads + id;
			(*offset_type2)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
			(*offset_type2)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
			//--------------------Fig 11 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 2) * n_threads + id;
			(*offset_type2)+=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
			(*offset_type2)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
			//--------------------Fig 10 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 1) * n_threads + id;
			(*offset_type2)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
			(*offset_type2)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
			//--------------------Fig 12 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 3) * n_threads + id;
			(*offset_type2)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
			(*offset_type2)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
			//--------------------Fig 13 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 4) * n_threads + id;
			(*offset_type2)-=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
			(*offset_type2)-=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
			//--------------------Fig 15 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 6) * n_threads + id;
			(*offset_type2)-=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
			(*offset_type2)+=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
			//--------------------Fig 14 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 5) * n_threads + id;
			(*offset_type2)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
			(*offset_type2)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			//--------------------Fig 16 below--------------------//
			offset_type2 = type2_threaded + (t_dis + t_size * 7) * n_threads + id;
			(*offset_type2)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
			(*offset_type2)+=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			}
		}
	}
	sfree(cname,fname,"mf_pi_pi_1",mf_pi_pi_1);
	sfree(cname,fname,"mf_pi_pi_2",mf_pi_pi_2);
	complex<double> *type2 = (complex<double> *)smalloc(cname,fname,"type2",sizeof(complex<double>)*t_size*8);
  ((Vector*)type2)->VecZero(t_size*8*2);
	for(int i = 0; i < t_size*8; i++)
		for(int id = 0; id < n_threads; id++)
			type2[i] += type2_threaded[i*n_threads+id] * 0.5 / t_size; // 0.5 because doing permutation of two pions
	sfree(cname,fname,"type2_threaded",type2_threaded);
	QMP_sum_double_array((Float *)type2,t_size*8*2);
	cout_time("Contraction of type2 ends!");

	char fn[1024];
	sprintf(fn,"%s_type2",common_arg->filename);
	writeCorr(type2, fn, 8, t_size);

	sfree(cname,fname,"type2",type2);
}

void MesonField::run_type3_three_sink_approach(int t_delta, int sep) 
{
	char *fname = "run_type3_three_sink_approach()";

	Vector *mf_pi_pi = (Vector *)smalloc(cname, fname, "mf_pi_pi", sizeof(Float)*(nl[0]+sc_size*nhits[0])*nvec[0]*t_size*2);//This the contraction of pion and kaon mesonfields, t is position of first pion
	Vector *mf_pi_pi_k_1 = (Vector *)smalloc(cname, fname, "mf_pi_pi_k_1", sizeof(Float)*(nl[0]+sc_size*nhits[0])*nvec[1]*t_size*2);//This the contraction of pion1, pion2 and kaon mesonfields, t is position of first pion
	Vector *mf_pi_pi_k_2 = (Vector *)smalloc(cname, fname, "mf_pi_pi_k_2", sizeof(Float)*(nl[0]+sc_size*nhits[0])*nvec[1]*t_size*2);//This the contraction of pion1, pion2 and kaon mesonfields, t is position of first pion
	Vector *mf_k_pi_pi = (Vector *)smalloc(cname, fname, "mf_k_pi_pi", sizeof(Float)*(nl[1]+sc_size*nhits[1])*nvec[0]*t_size*2);//This the contraction of kaon, pion1 and pion2 mesonfields, t is position of first pion
	mf_pi_pi->VecZero((nl[0]+sc_size*nhits[0])*nvec[0]*t_size*2);
	mf_pi_pi_k_1->VecZero((nl[0]+sc_size*nhits[0])*nvec[1]*t_size*2);
	mf_pi_pi_k_2->VecZero((nl[0]+sc_size*nhits[0])*nvec[1]*t_size*2);
	mf_k_pi_pi->VecZero((nl[1]+sc_size*nhits[1])*nvec[0]*t_size*2);

	omp_set_num_threads(bfmarg::threads);

	mf_contraction(LIGHT, LIGHT, LIGHT, (complex<double>*)mf, sep, (complex<double>*)mf, (complex<double>*)mf_pi_pi);
	//mf_contraction(LIGHT, LIGHT, STRANGE, (complex<double>*)mf_pi_pi, t_delta + sep, (complex<double>*)mf_ls, (complex<double>*)mf_pi_pi_k_1);
	mf_contraction_ww(LIGHT, LIGHT, STRANGE, (complex<double>*)mf_pi_pi, t_delta + sep, (complex<double>*)mf_ww[1], (complex<double>*)mf_pi_pi_k_1);
	//mf_contraction(STRANGE, LIGHT, LIGHT, (complex<double>*)mf_sl, 2 * t_size - t_delta - sep, (complex<double>*)mf_pi_pi, (complex<double>*)mf_k_pi_pi);

	mf_pi_pi->VecZero((nl[0]+sc_size*nhits[0])*nvec[0]*t_size*2);

	mf_contraction(LIGHT, LIGHT, LIGHT, (complex<double>*)mf, t_size - sep, (complex<double>*)mf, (complex<double>*)mf_pi_pi);
	//mf_contraction(LIGHT, LIGHT, STRANGE, (complex<double>*)mf_pi_pi, t_delta, (complex<double>*)mf_ls, (complex<double>*)mf_pi_pi_k_2);
	mf_contraction_ww(LIGHT, LIGHT, STRANGE, (complex<double>*)mf_pi_pi, t_delta, (complex<double>*)mf_ww[1], (complex<double>*)mf_pi_pi_k_2);
	//mf_contraction(STRANGE, LIGHT, LIGHT, (complex<double>*)mf_sl, t_size - t_delta, (complex<double>*)mf_pi_pi, (complex<double>*)mf_k_pi_pi); // Now the mf_k_pi_pi already contain the permutation of two pions

	sfree(cname,fname,"mf_pi_pi",mf_pi_pi);

	cout_time("Contraction of type3 begins!");
	int n_threads = bfmarg::threads;
	complex<double> *type3_threaded = (complex<double> *)smalloc(cname,fname,"type3_threaded",sizeof(complex<double>)*t_size*16*n_threads);// 16 because 16 different contractions
	complex<double> *type3S5D_threaded = (complex<double> *)smalloc(cname,fname,"type3S5D_threaded",sizeof(complex<double>)*t_size*n_threads);
	((Vector*)type3_threaded)->VecZero(t_size*16*n_threads*2);
	((Vector*)type3S5D_threaded)->VecZero(t_size*n_threads*2);

	omp_set_num_threads(n_threads);
#pragma omp parallel for
	for(int x = 0; x < size_4d; x++) { 
		int id = omp_get_thread_num();
		int t_op = x / size_3d + GJP.TnodeSites()*GJP.TnodeCoor();
		complex<double> *offset_type3;
		complex<double> *offset_type3S5D;
		for(int t_k = 0; t_k < t_size; t_k++) {
			int t_dis = (t_op - t_k + t_size) % t_size;
			WilsonMatrix Mtemp0 = Build_Wilson_ww(mf_pi_pi_k_1, x, (t_k + t_delta + sep) % t_size, t_k, LIGHT, STRANGE);
			//WilsonMatrix Mtemp1 = Build_Wilson(mf_k_pi_pi, x, t_k, STRANGE, LIGHT);
			WilsonMatrix Mtemp2 = Build_Wilson_loop(x, LIGHT);
			WilsonMatrix Mtemp3 = Build_Wilson_loop(x, STRANGE);
			WilsonMatrix Mtemp5; // for Type3 S5D

			// 1. calculate the contraction with Light internal loop
			for(int dir = 0; dir < 4; dir++) {
				//--------------------Fig 17 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 0) * n_threads + id;
				(*offset_type3)+=(Mtemp0.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type3)+=(Mtemp0.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 19 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 2) * n_threads + id;
				(*offset_type3)-=(Mtemp0.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type3)+=(Mtemp0.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 18 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 1) * n_threads + id;
				(*offset_type3)+=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type3)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 20 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 3) * n_threads + id;
				(*offset_type3)-=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type3)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 21 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 4) * n_threads + id;
				(*offset_type3)+=(Mtemp0.glA(dir)).Trace() * (Mtemp3.glA(dir)).Trace();
				(*offset_type3)+=(Mtemp0.glV(dir)).Trace() * (Mtemp3.glV(dir)).Trace();
				//--------------------Fig 23 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 6) * n_threads + id;
				(*offset_type3)-=(Mtemp0.glA(dir)).Trace() * (Mtemp3.glA(dir)).Trace();
				(*offset_type3)+=(Mtemp0.glV(dir)).Trace() * (Mtemp3.glV(dir)).Trace();
				//--------------------Fig 22 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 5) * n_threads + id;
				(*offset_type3)+=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp3.glA(dir)));
				(*offset_type3)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp3.glV(dir)));
				//--------------------Fig 24 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 7) * n_threads + id;
				(*offset_type3)-=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp3.glA(dir)));
				(*offset_type3)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp3.glV(dir)));
				//--------------------Fig 25 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 8) * n_threads + id;
				(*offset_type3)+=Trace(Mtemp0.glA(dir) , Mtemp2.glA(dir));
				(*offset_type3)+=Trace(Mtemp0.glV(dir) , Mtemp2.glV(dir));
				//--------------------Fig 27 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 10) * n_threads + id;
				(*offset_type3)-=Trace(Mtemp0.glA(dir) , Mtemp2.glA(dir));
				(*offset_type3)+=Trace(Mtemp0.glV(dir) , Mtemp2.glV(dir));
				//--------------------Fig 26 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 9) * n_threads + id;
				(*offset_type3)+=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type3)+=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 28 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 11) * n_threads + id;
				(*offset_type3)-=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type3)+=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 29 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 12) * n_threads + id;
				(*offset_type3)+=Trace(Mtemp0.glA(dir) , Mtemp3.glA(dir));
				(*offset_type3)+=Trace(Mtemp0.glV(dir) , Mtemp3.glV(dir));
				//--------------------Fig 31 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 14) * n_threads + id;
				(*offset_type3)+=Trace(Mtemp0.glA(dir) , Mtemp3.glA(dir));
				(*offset_type3)-=Trace(Mtemp0.glV(dir) , Mtemp3.glV(dir));
				//--------------------Fig 30 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 13) * n_threads + id;
				(*offset_type3)+=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp3.glA(dir)));
				(*offset_type3)+=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp3.glV(dir)));
				//--------------------Fig 32 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 15) * n_threads + id;
				(*offset_type3)+=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp3.glA(dir)));
				(*offset_type3)-=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp3.glV(dir)));
			}
			//--------------------Type3 S5D--------------------//
			offset_type3S5D = type3S5D_threaded + t_dis * n_threads + id;
			Mtemp5 = Mtemp0;
			(*offset_type3S5D)+=Mtemp5.Trace();

			//--------------------Now permute two pions--------------------//
			Mtemp0 = Build_Wilson_ww(mf_pi_pi_k_2, x, (t_k + t_delta) % t_size, t_k, LIGHT, STRANGE);

			for(int dir = 0; dir < 4; dir++) {
				//--------------------Fig 17 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 0) * n_threads + id;
				(*offset_type3)+=(Mtemp0.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type3)+=(Mtemp0.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 19 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 2) * n_threads + id;
				(*offset_type3)-=(Mtemp0.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type3)+=(Mtemp0.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 18 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 1) * n_threads + id;
				(*offset_type3)+=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type3)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 20 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 3) * n_threads + id;
				(*offset_type3)-=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type3)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 21 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 4) * n_threads + id;
				(*offset_type3)+=(Mtemp0.glA(dir)).Trace() * (Mtemp3.glA(dir)).Trace();
				(*offset_type3)+=(Mtemp0.glV(dir)).Trace() * (Mtemp3.glV(dir)).Trace();
				//--------------------Fig 23 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 6) * n_threads + id;
				(*offset_type3)-=(Mtemp0.glA(dir)).Trace() * (Mtemp3.glA(dir)).Trace();
				(*offset_type3)+=(Mtemp0.glV(dir)).Trace() * (Mtemp3.glV(dir)).Trace();
				//--------------------Fig 22 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 5) * n_threads + id;
				(*offset_type3)+=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp3.glA(dir)));
				(*offset_type3)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp3.glV(dir)));
				//--------------------Fig 24 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 7) * n_threads + id;
				(*offset_type3)-=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp3.glA(dir)));
				(*offset_type3)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp3.glV(dir)));
				//--------------------Fig 25 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 8) * n_threads + id;
				(*offset_type3)+=Trace(Mtemp0.glA(dir) , Mtemp2.glA(dir));
				(*offset_type3)+=Trace(Mtemp0.glV(dir) , Mtemp2.glV(dir));
				//--------------------Fig 27 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 10) * n_threads + id;
				(*offset_type3)-=Trace(Mtemp0.glA(dir) , Mtemp2.glA(dir));
				(*offset_type3)+=Trace(Mtemp0.glV(dir) , Mtemp2.glV(dir));
				//--------------------Fig 26 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 9) * n_threads + id;
				(*offset_type3)+=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type3)+=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 28 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 11) * n_threads + id;
				(*offset_type3)-=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type3)+=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 29 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 12) * n_threads + id;
				(*offset_type3)+=Trace(Mtemp0.glA(dir) , Mtemp3.glA(dir));
				(*offset_type3)+=Trace(Mtemp0.glV(dir) , Mtemp3.glV(dir));
				//--------------------Fig 31 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 14) * n_threads + id;
				(*offset_type3)+=Trace(Mtemp0.glA(dir) , Mtemp3.glA(dir));
				(*offset_type3)-=Trace(Mtemp0.glV(dir) , Mtemp3.glV(dir));
				//--------------------Fig 30 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 13) * n_threads + id;
				(*offset_type3)+=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp3.glA(dir)));
				(*offset_type3)+=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp3.glV(dir)));
				//--------------------Fig 32 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 15) * n_threads + id;
				(*offset_type3)+=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp3.glA(dir)));
				(*offset_type3)-=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp3.glV(dir)));
			}

			//--------------------Type3 S5D--------------------//
			Mtemp5 = Mtemp0;
			(*offset_type3S5D) += Mtemp5.Trace();

			// 2. calculate the contraction with Strange inernal loop; already considered the permutation in Mtemp1.
			//for(int dir = 0; dir < 4; dir++) {
			//	//--------------------Fig 21 below--------------------//
			//	offset_type3 = type3_threaded + (t_dis + t_size * 4) * n_threads + id;
			//	(*offset_type3)-=(Mtemp1.glV(dir)).Trace() * (Mtemp3.glA(dir)).Trace();
			//	(*offset_type3)-=(Mtemp1.glA(dir)).Trace() * (Mtemp3.glV(dir)).Trace();
			//	//--------------------Fig 23 below--------------------//
			//	offset_type3 = type3_threaded + (t_dis + t_size * 6) * n_threads + id;
			//	(*offset_type3)+=(Mtemp1.glV(dir)).Trace() * (Mtemp3.glA(dir)).Trace();
			//	(*offset_type3)-=(Mtemp1.glA(dir)).Trace() * (Mtemp3.glV(dir)).Trace();
			//	//--------------------Fig 22 below--------------------//
			//	offset_type3 = type3_threaded + (t_dis + t_size * 5) * n_threads + id;
			//	(*offset_type3)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp3.glA(dir)));
			//	(*offset_type3)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp3.glV(dir)));
			//	//--------------------Fig 24 below--------------------//
			//	offset_type3 = type3_threaded + (t_dis + t_size * 7) * n_threads + id;
			//	(*offset_type3)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp3.glA(dir)));
			//	(*offset_type3)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp3.glV(dir)));
			//	//--------------------Fig 29 below--------------------//
			//	offset_type3 = type3_threaded + (t_dis + t_size * 12) * n_threads + id;
			//	(*offset_type3)-=Trace(Mtemp1.glV(dir) , Mtemp3.glA(dir));
			//	(*offset_type3)-=Trace(Mtemp1.glA(dir) , Mtemp3.glV(dir));
			//	//--------------------Fig 31 below--------------------//
			//	offset_type3 = type3_threaded + (t_dis + t_size * 14) * n_threads + id;
			//	(*offset_type3)+=Trace(Mtemp1.glV(dir) , Mtemp3.glA(dir));
			//	(*offset_type3)-=Trace(Mtemp1.glA(dir) , Mtemp3.glV(dir));
			//	//--------------------Fig 30 below--------------------//
			//	offset_type3 = type3_threaded + (t_dis + t_size * 13) * n_threads + id;
			//	(*offset_type3)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp3.glA(dir)));
			//	(*offset_type3)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp3.glV(dir)));
			//	//--------------------Fig 32 below--------------------//
			//	offset_type3 = type3_threaded + (t_dis + t_size * 15) * n_threads + id;
			//	(*offset_type3)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp3.glA(dir)));
			//	(*offset_type3)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp3.glV(dir)));
			//}
		}
	}
	sfree(cname,fname,"mf_pi_pi_k_1",mf_pi_pi_k_1);
	sfree(cname,fname,"mf_pi_pi_k_2",mf_pi_pi_k_2);
	sfree(cname,fname,"mf_k_pi_pi",mf_k_pi_pi);
	complex<double> *type3 = (complex<double> *)smalloc(cname,fname,"type3",sizeof(complex<double>)*t_size*16);
	complex<double> *type3S5D = (complex<double> *)smalloc(cname,fname,"type3S5D",sizeof(complex<double>)*t_size);
  ((Vector*)type3)->VecZero(t_size*16*2);
  ((Vector*)type3S5D)->VecZero(t_size*2);
	for(int i = 0; i < t_size*16; i++)
		for(int id = 0; id < n_threads; id++) {
			type3[i] += type3_threaded[i*n_threads+id] * 0.5 / t_size; // 0.5 because of permutation
		}
	for(int i = 0; i < t_size; i++)
		for(int id = 0; id < n_threads; id++) {
			type3S5D[i] += type3S5D_threaded[i*n_threads+id] * 0.5 / t_size; // 0.5 because of permutation
		}
	sfree(cname,fname,"type3_threaded",type3_threaded);
	sfree(cname,fname,"type3S5D_threaded",type3S5D_threaded);
	QMP_sum_double_array((Float *)type3,t_size*16*2);
	QMP_sum_double_array((Float *)type3S5D,t_size*2);
	cout_time("Contraction of type3 ends!");

	char fn[1024];
	sprintf(fn,"%s_type3_three_sink_approach",common_arg->filename);
	writeCorr(type3, fn, 16, t_size);
	sprintf(fn,"%s_type3S5D",common_arg->filename);
	writeCorr(type3S5D, fn, 1, t_size);

	sfree(cname,fname,"type3",type3);
	sfree(cname,fname,"type3S5D",type3S5D);
}
void MesonField::run_type3(int t_delta, int sep) 
{
	char *fname = "run_type3()";

	Vector *mf_pi_pi = (Vector *)smalloc(cname, fname, "mf_pi_pi", sizeof(Float)*(nl[0]+sc_size*nhits[0])*nvec[0]*t_size*2);//This the contraction of pion and kaon mesonfields, t is position of first pion
	Vector *mf_pi_pi_k_1 = (Vector *)smalloc(cname, fname, "mf_pi_pi_k_1", sizeof(Float)*(nl[0]+sc_size*nhits[0])*nvec[1]*t_size*2);//This the contraction of pion1, pion2 and kaon mesonfields, t is position of first pion
	Vector *mf_pi_pi_k_2 = (Vector *)smalloc(cname, fname, "mf_pi_pi_k_2", sizeof(Float)*(nl[0]+sc_size*nhits[0])*nvec[1]*t_size*2);//This the contraction of pion1, pion2 and kaon mesonfields, t is position of first pion
	Vector *mf_k_pi_pi = (Vector *)smalloc(cname, fname, "mf_k_pi_pi", sizeof(Float)*(nl[1]+sc_size*nhits[1])*nvec[0]*t_size*2);//This the contraction of kaon, pion1 and pion2 mesonfields, t is position of first pion
	mf_pi_pi->VecZero((nl[0]+sc_size*nhits[0])*nvec[0]*t_size*2);
	mf_pi_pi_k_1->VecZero((nl[0]+sc_size*nhits[0])*nvec[1]*t_size*2);
	mf_pi_pi_k_2->VecZero((nl[0]+sc_size*nhits[0])*nvec[1]*t_size*2);
	mf_k_pi_pi->VecZero((nl[1]+sc_size*nhits[1])*nvec[0]*t_size*2);

	omp_set_num_threads(bfmarg::threads);

	mf_contraction(LIGHT, LIGHT, LIGHT, (complex<double>*)mf, sep, (complex<double>*)mf, (complex<double>*)mf_pi_pi);
	mf_contraction(LIGHT, LIGHT, STRANGE, (complex<double>*)mf_pi_pi, t_delta + sep, (complex<double>*)mf_ls, (complex<double>*)mf_pi_pi_k_1);
	mf_contraction(STRANGE, LIGHT, LIGHT, (complex<double>*)mf_sl, 2 * t_size - t_delta - sep, (complex<double>*)mf_pi_pi, (complex<double>*)mf_k_pi_pi);

	mf_pi_pi->VecZero((nl[0]+sc_size*nhits[0])*nvec[0]*t_size*2);

	mf_contraction(LIGHT, LIGHT, LIGHT, (complex<double>*)mf, t_size - sep, (complex<double>*)mf, (complex<double>*)mf_pi_pi);
	mf_contraction(LIGHT, LIGHT, STRANGE, (complex<double>*)mf_pi_pi, t_delta, (complex<double>*)mf_ls, (complex<double>*)mf_pi_pi_k_2);
	mf_contraction(STRANGE, LIGHT, LIGHT, (complex<double>*)mf_sl, t_size - t_delta, (complex<double>*)mf_pi_pi, (complex<double>*)mf_k_pi_pi); // Now the mf_k_pi_pi already contain the permutation of two pions

	sfree(cname,fname,"mf_pi_pi",mf_pi_pi);

	cout_time("Contraction of type3 begins!");
	int n_threads = bfmarg::threads;
	complex<double> *type3_threaded = (complex<double> *)smalloc(cname,fname,"type3_threaded",sizeof(complex<double>)*t_size*16*n_threads);// 16 because 16 different contractions
	complex<double> *type3S5D_threaded = (complex<double> *)smalloc(cname,fname,"type3S5D_threaded",sizeof(complex<double>)*t_size*n_threads);
	((Vector*)type3_threaded)->VecZero(t_size*16*n_threads*2);
	((Vector*)type3S5D_threaded)->VecZero(t_size*n_threads*2);

	omp_set_num_threads(n_threads);
#pragma omp parallel for
	for(int x = 0; x < size_4d; x++) { 
		int id = omp_get_thread_num();
		int t_op = x / size_3d + GJP.TnodeSites()*GJP.TnodeCoor();
		complex<double> *offset_type3;
		complex<double> *offset_type3S5D;
		for(int t_k = 0; t_k < t_size; t_k++) {
			int t_dis = (t_op - t_k + t_size) % t_size;
			WilsonMatrix Mtemp0 = Build_Wilson(mf_pi_pi_k_1, x, (t_k + t_delta + sep) % t_size, LIGHT, STRANGE);
			WilsonMatrix Mtemp1 = Build_Wilson(mf_k_pi_pi, x, t_k, STRANGE, LIGHT);
			WilsonMatrix Mtemp2 = Build_Wilson_loop(x, LIGHT);
			WilsonMatrix Mtemp3 = Build_Wilson_loop(x, STRANGE);
			WilsonMatrix Mtemp5; // for Type3 S5D

			// 1. calculate the contraction with Light internal loop
			for(int dir = 0; dir < 4; dir++) {
				//--------------------Fig 17 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 0) * n_threads + id;
				(*offset_type3)-=(Mtemp0.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type3)-=(Mtemp0.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 19 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 2) * n_threads + id;
				(*offset_type3)+=(Mtemp0.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type3)-=(Mtemp0.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 18 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 1) * n_threads + id;
				(*offset_type3)-=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type3)-=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 20 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 3) * n_threads + id;
				(*offset_type3)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type3)-=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 25 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 8) * n_threads + id;
				(*offset_type3)-=Trace(Mtemp0.glV(dir) , Mtemp2.glA(dir));
				(*offset_type3)-=Trace(Mtemp0.glA(dir) , Mtemp2.glV(dir));
				//--------------------Fig 27 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 10) * n_threads + id;
				(*offset_type3)-=Trace(Mtemp0.glV(dir) , Mtemp2.glA(dir));
				(*offset_type3)+=Trace(Mtemp0.glA(dir) , Mtemp2.glV(dir));
				//--------------------Fig 26 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 9) * n_threads + id;
				(*offset_type3)-=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type3)-=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 28 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 11) * n_threads + id;
				(*offset_type3)-=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type3)+=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			}
			//--------------------Type3 S5D--------------------//
			offset_type3S5D = type3S5D_threaded + t_dis * n_threads + id;
			Mtemp5 = Mtemp0;
			(*offset_type3S5D)+=(Mtemp5.gl(-5)).Trace();

			//--------------------Now permute two pions--------------------//
			Mtemp0 = Build_Wilson(mf_pi_pi_k_2, x, (t_k + t_delta) % t_size, LIGHT, STRANGE);

			for(int dir = 0; dir < 4; dir++) {
				//--------------------Fig 17 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 0) * n_threads + id;
				(*offset_type3)-=(Mtemp0.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type3)-=(Mtemp0.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 19 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 2) * n_threads + id;
				(*offset_type3)+=(Mtemp0.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type3)-=(Mtemp0.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 18 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 1) * n_threads + id;
				(*offset_type3)-=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type3)-=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 20 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 3) * n_threads + id;
				(*offset_type3)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type3)-=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 25 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 8) * n_threads + id;
				(*offset_type3)-=Trace(Mtemp0.glV(dir) , Mtemp2.glA(dir));
				(*offset_type3)-=Trace(Mtemp0.glA(dir) , Mtemp2.glV(dir));
				//--------------------Fig 27 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 10) * n_threads + id;
				(*offset_type3)-=Trace(Mtemp0.glV(dir) , Mtemp2.glA(dir));
				(*offset_type3)+=Trace(Mtemp0.glA(dir) , Mtemp2.glV(dir));
				//--------------------Fig 26 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 9) * n_threads + id;
				(*offset_type3)-=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type3)-=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 28 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 11) * n_threads + id;
				(*offset_type3)-=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type3)+=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			}
			//--------------------Type3 S5D--------------------//
			Mtemp5 = Mtemp0;
			(*offset_type3S5D)+=(Mtemp5.gl(-5)).Trace();

			// 2. calculate the contraction with Strange inernal loop; already considered the permutation in Mtemp1.
			for(int dir = 0; dir < 4; dir++) {
				//--------------------Fig 21 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 4) * n_threads + id;
				(*offset_type3)-=(Mtemp1.glV(dir)).Trace() * (Mtemp3.glA(dir)).Trace();
				(*offset_type3)-=(Mtemp1.glA(dir)).Trace() * (Mtemp3.glV(dir)).Trace();
				//--------------------Fig 23 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 6) * n_threads + id;
				(*offset_type3)+=(Mtemp1.glV(dir)).Trace() * (Mtemp3.glA(dir)).Trace();
				(*offset_type3)-=(Mtemp1.glA(dir)).Trace() * (Mtemp3.glV(dir)).Trace();
				//--------------------Fig 22 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 5) * n_threads + id;
				(*offset_type3)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp3.glA(dir)));
				(*offset_type3)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp3.glV(dir)));
				//--------------------Fig 24 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 7) * n_threads + id;
				(*offset_type3)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp3.glA(dir)));
				(*offset_type3)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp3.glV(dir)));
				//--------------------Fig 29 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 12) * n_threads + id;
				(*offset_type3)-=Trace(Mtemp1.glV(dir) , Mtemp3.glA(dir));
				(*offset_type3)-=Trace(Mtemp1.glA(dir) , Mtemp3.glV(dir));
				//--------------------Fig 31 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 14) * n_threads + id;
				(*offset_type3)+=Trace(Mtemp1.glV(dir) , Mtemp3.glA(dir));
				(*offset_type3)-=Trace(Mtemp1.glA(dir) , Mtemp3.glV(dir));
				//--------------------Fig 30 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 13) * n_threads + id;
				(*offset_type3)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp3.glA(dir)));
				(*offset_type3)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp3.glV(dir)));
				//--------------------Fig 32 below--------------------//
				offset_type3 = type3_threaded + (t_dis + t_size * 15) * n_threads + id;
				(*offset_type3)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp3.glA(dir)));
				(*offset_type3)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp3.glV(dir)));
			}
		}
	}
	sfree(cname,fname,"mf_pi_pi_k_1",mf_pi_pi_k_1);
	sfree(cname,fname,"mf_pi_pi_k_2",mf_pi_pi_k_2);
	sfree(cname,fname,"mf_k_pi_pi",mf_k_pi_pi);
	complex<double> *type3 = (complex<double> *)smalloc(cname,fname,"type3",sizeof(complex<double>)*t_size*16);
	complex<double> *type3S5D = (complex<double> *)smalloc(cname,fname,"type3S5D",sizeof(complex<double>)*t_size);
  ((Vector*)type3)->VecZero(t_size*16*2);
  ((Vector*)type3S5D)->VecZero(t_size*2);
	for(int i = 0; i < t_size*16; i++)
		for(int id = 0; id < n_threads; id++) {
			type3[i] += type3_threaded[i*n_threads+id] * 0.5 / t_size; // 0.5 because of permutation
		}
	for(int i = 0; i < t_size; i++)
		for(int id = 0; id < n_threads; id++) {
			type3S5D[i] += type3S5D_threaded[i*n_threads+id] * 0.5 / t_size; // 0.5 because of permutation
		}
	sfree(cname,fname,"type3_threaded",type3_threaded);
	sfree(cname,fname,"type3S5D_threaded",type3S5D_threaded);
	QMP_sum_double_array((Float *)type3,t_size*16*2);
	QMP_sum_double_array((Float *)type3S5D,t_size*2);
	cout_time("Contraction of type3 ends!");

	char fn[1024];
	sprintf(fn,"%s_type3",common_arg->filename);
	writeCorr(type3, fn, 16, t_size);
	sprintf(fn,"%s_type3S5D",common_arg->filename);
	writeCorr(type3S5D, fn, 1, t_size);

	sfree(cname,fname,"type3",type3);
	sfree(cname,fname,"type3S5D",type3S5D);
}

void MesonField::run_type4_three_sink_approach(int t_delta, int sep)
{
	char *fname = "run_type4_three_sink_approach()";

	int n_threads = bfmarg::threads;

	cout_time("Contraction of type4 begins!");
	complex<double> *type4_threaded = (complex<double> *)smalloc(cname,fname,"type4_threaded",sizeof(complex<double>)*t_size*t_size*16*n_threads);// 16 because 16 different contractions
	complex<double> *type4_threaded_2 = (complex<double> *)smalloc(cname,fname,"type4_threaded_2",sizeof(complex<double>)*t_size*t_size*16*n_threads);// 16 because 16 different contractions
	complex<double> *type4S5D_threaded = (complex<double> *)smalloc(cname,fname,"type4S5D_threaded",sizeof(complex<double>)*t_size*t_size*n_threads);
	complex<double> *type4S5D_threaded_2 = (complex<double> *)smalloc(cname,fname,"type4S5D_threaded_2",sizeof(complex<double>)*t_size*t_size*n_threads);
	((Vector*)type4_threaded)->VecZero(t_size*t_size*16*n_threads*2);
	((Vector*)type4_threaded_2)->VecZero(t_size*t_size*16*n_threads*2);
	((Vector*)type4S5D_threaded)->VecZero(t_size*t_size*n_threads*2);
	((Vector*)type4S5D_threaded_2)->VecZero(t_size*t_size*n_threads*2);

	omp_set_num_threads(n_threads);
#pragma omp parallel for
	for(int x = 0; x < size_4d; x++) {
  	int id = omp_get_thread_num();
		int t_op = x / size_3d + GJP.TnodeSites()*GJP.TnodeCoor();
		complex<double> *offset_type4;
		complex<double> *offset_type4S5D;
		for(int t_k = 0; t_k < t_size; t_k++) {
			int t_dis = (t_op - t_k + t_size) % t_size;
			WilsonMatrix Mtemp0 = Build_Wilson_ww(mf_ww[1], x, t_k, t_k, LIGHT, STRANGE);
			WilsonMatrix Mtemp1 = Build_Wilson_ww(mf_ww[2], x, t_k, t_k, STRANGE, LIGHT);
			WilsonMatrix Mtemp2 = Build_Wilson_loop(x, LIGHT);
			WilsonMatrix Mtemp3 = Build_Wilson_loop(x, STRANGE);

			for(int dir = 0; dir < 4; dir++) {
				//--------------------Fig 33 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 0)) * n_threads + id;
				(*offset_type4)+=(Mtemp0.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type4)+=(Mtemp0.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 35 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 2)) * n_threads + id;
				(*offset_type4)-=(Mtemp0.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type4)+=(Mtemp0.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 34 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 1)) * n_threads + id;
				(*offset_type4)+=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type4)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 36 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 3)) * n_threads + id;
				(*offset_type4)-=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type4)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 37 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 4)) * n_threads + id;
				(*offset_type4)+=(Mtemp0.glA(dir)).Trace() * (Mtemp3.glA(dir)).Trace();
				(*offset_type4)+=(Mtemp0.glV(dir)).Trace() * (Mtemp3.glV(dir)).Trace();
				//--------------------Fig 39 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 6)) * n_threads + id;
				(*offset_type4)-=(Mtemp0.glA(dir)).Trace() * (Mtemp3.glA(dir)).Trace();
				(*offset_type4)+=(Mtemp0.glV(dir)).Trace() * (Mtemp3.glV(dir)).Trace();
				//--------------------Fig 38 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 5)) * n_threads + id;
				(*offset_type4)+=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp3.glA(dir)));
				(*offset_type4)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp3.glV(dir)));
				//--------------------Fig 40 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 7)) * n_threads + id;
				(*offset_type4)-=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp3.glA(dir)));
				(*offset_type4)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp3.glV(dir)));
				//--------------------Fig 41 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 8)) * n_threads + id;
				(*offset_type4)+=Trace(Mtemp0.glA(dir) , Mtemp2.glA(dir));
				(*offset_type4)+=Trace(Mtemp0.glV(dir) , Mtemp2.glV(dir));
				//--------------------Fig 43 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 10)) * n_threads + id;
				(*offset_type4)-=Trace(Mtemp0.glA(dir) , Mtemp2.glA(dir));
				(*offset_type4)+=Trace(Mtemp0.glV(dir) , Mtemp2.glV(dir));
				//--------------------Fig 42 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 9)) * n_threads + id;
				(*offset_type4)+=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type4)+=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 44 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 11)) * n_threads + id;
				(*offset_type4)-=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type4)+=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 45 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 12)) * n_threads + id;
				(*offset_type4)+=Trace(Mtemp0.glA(dir) , Mtemp3.glA(dir));
				(*offset_type4)+=Trace(Mtemp0.glV(dir) , Mtemp3.glV(dir));
				//--------------------Fig 47 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 14)) * n_threads + id;
				(*offset_type4)+=Trace(Mtemp0.glA(dir) , Mtemp3.glA(dir));
				(*offset_type4)-=Trace(Mtemp0.glV(dir) , Mtemp3.glV(dir));
				//--------------------Fig 46 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 13)) * n_threads + id;
				(*offset_type4)+=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp3.glA(dir)));
				(*offset_type4)+=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp3.glV(dir)));
				//--------------------Fig 48 below--------------------//
				offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 15)) * n_threads + id;
				(*offset_type4)+=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp3.glA(dir)));
				(*offset_type4)-=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp3.glV(dir)));
			}

			for(int dir = 0; dir < 4; dir++) {
				//--------------------Fig 33 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 0)) * n_threads + id;
				(*offset_type4)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type4)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 35 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 2)) * n_threads + id;
				(*offset_type4)+=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
				(*offset_type4)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
				//--------------------Fig 34 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 1)) * n_threads + id;
				(*offset_type4)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type4)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 36 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 3)) * n_threads + id;
				(*offset_type4)+=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glA(dir)));
				(*offset_type4)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glV(dir)));
				//--------------------Fig 37 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 4)) * n_threads + id;
				(*offset_type4)-=(Mtemp1.glA(dir)).Trace() * (Mtemp3.glA(dir)).Trace();
				(*offset_type4)-=(Mtemp1.glV(dir)).Trace() * (Mtemp3.glV(dir)).Trace();
				//--------------------Fig 39 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 6)) * n_threads + id;
				(*offset_type4)+=(Mtemp1.glA(dir)).Trace() * (Mtemp3.glA(dir)).Trace();
				(*offset_type4)-=(Mtemp1.glV(dir)).Trace() * (Mtemp3.glV(dir)).Trace();
				//--------------------Fig 38 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 5)) * n_threads + id;
				(*offset_type4)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp3.glA(dir)));
				(*offset_type4)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp3.glV(dir)));
				//--------------------Fig 40 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 7)) * n_threads + id;
				(*offset_type4)+=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp3.glA(dir)));
				(*offset_type4)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp3.glV(dir)));
				//--------------------Fig 41 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 8)) * n_threads + id;
				(*offset_type4)-=Trace(Mtemp1.glA(dir) , Mtemp2.glA(dir));
				(*offset_type4)-=Trace(Mtemp1.glV(dir) , Mtemp2.glV(dir));
				//--------------------Fig 43 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 10)) * n_threads + id;
				(*offset_type4)-=Trace(Mtemp1.glA(dir) , Mtemp2.glA(dir));
				(*offset_type4)+=Trace(Mtemp1.glV(dir) , Mtemp2.glV(dir));
				//--------------------Fig 42 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 9)) * n_threads + id;
				(*offset_type4)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type4)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 44 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 11)) * n_threads + id;
				(*offset_type4)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glA(dir)));
				(*offset_type4)+=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glV(dir)));
				//--------------------Fig 45 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 12)) * n_threads + id;
				(*offset_type4)-=Trace(Mtemp1.glA(dir) , Mtemp3.glA(dir));
				(*offset_type4)-=Trace(Mtemp1.glV(dir) , Mtemp3.glV(dir));
				//--------------------Fig 47 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 14)) * n_threads + id;
				(*offset_type4)+=Trace(Mtemp1.glA(dir) , Mtemp3.glA(dir));
				(*offset_type4)-=Trace(Mtemp1.glV(dir) , Mtemp3.glV(dir));
				//--------------------Fig 46 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 13)) * n_threads + id;
				(*offset_type4)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp3.glA(dir)));
				(*offset_type4)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp3.glV(dir)));
				//--------------------Fig 48 below--------------------//
				offset_type4 = type4_threaded_2 + (t_k + t_size * (t_dis + t_size * 15)) * n_threads + id;
				(*offset_type4)+=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp3.glA(dir)));
				(*offset_type4)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp3.glV(dir)));
			}
			//--------------------Type4 S5D--------------------//
			offset_type4S5D = type4S5D_threaded + (t_dis * t_size + t_k) * n_threads + id;
			(*offset_type4S5D) += Mtemp0.Trace();
			offset_type4S5D = type4S5D_threaded_2 + (t_dis * t_size + t_k) * n_threads + id;
			(*offset_type4S5D) += Mtemp1.Trace();
		}
	}
	complex<double> *type4 = (complex<double> *)smalloc(cname,fname,"type4",sizeof(complex<double>)*t_size*t_size*16);
	complex<double> *type4_2 = (complex<double> *)smalloc(cname,fname,"type4_2",sizeof(complex<double>)*t_size*t_size*16);
	complex<double> *type4S5D = (complex<double> *)smalloc(cname,fname,"type4S5D",sizeof(complex<double>)*t_size*t_size);
	complex<double> *type4S5D_2 = (complex<double> *)smalloc(cname,fname,"type4S5D_2",sizeof(complex<double>)*t_size*t_size);
  ((Vector*)type4)->VecZero(t_size*t_size*16*2);
  ((Vector*)type4_2)->VecZero(t_size*t_size*16*2);
  ((Vector*)type4S5D)->VecZero(t_size*t_size*2);
  ((Vector*)type4S5D_2)->VecZero(t_size*t_size*2);
	for(int i = 0; i < t_size*t_size*16; i++) {
		for(int id = 0; id < n_threads; id++) {
			type4[i] += type4_threaded[i*n_threads+id];
			type4_2[i] += type4_threaded_2[i*n_threads+id];
		}
	}
	for(int i = 0; i < t_size*t_size; i++) {
		for(int id = 0; id < n_threads; id++) {
			type4S5D[i] += type4S5D_threaded[i*n_threads+id];
			type4S5D_2[i] += type4S5D_threaded_2[i*n_threads+id];
		}
	}
	sfree(cname,fname,"type4_threaded",type4_threaded);
	sfree(cname,fname,"type4_threaded_2",type4_threaded_2);
	QMP_sum_double_array((Float *)type4,t_size*t_size*16*2);
	QMP_sum_double_array((Float *)type4_2,t_size*t_size*16*2);
	QMP_sum_double_array((Float *)type4S5D,t_size*t_size*2);
	QMP_sum_double_array((Float *)type4S5D_2,t_size*t_size*2);

	char fn[1024];
	sprintf(fn,"%s_type4_three_sink_approach",common_arg->filename);
	writeCorr(type4, fn, 16, t_size*t_size);
	sfree(cname,fname,"type4",type4);

	sprintf(fn,"%s_type4_three_sink_approach_2",common_arg->filename);
	writeCorr(type4_2, fn, 16, t_size*t_size);
	sfree(cname,fname,"type4_2",type4_2);

	sprintf(fn,"%s_type4S5D",common_arg->filename);
	writeCorr(type4S5D, fn, 1, t_size*t_size);
	sfree(cname,fname,"type4S5D",type4S5D);

	sprintf(fn,"%s_type4S5D_2",common_arg->filename);
	writeCorr(type4S5D_2, fn, 1, t_size*t_size);
	sfree(cname,fname,"type4S5D_2",type4S5D_2);

	cout_time("Contraction of type4 ends!");

}
void MesonField::run_type4(int t_delta, int sep) // FIXME this method should be deleted after merging it into type2
{
	char *fname = "run_type4()";

	int n_threads = bfmarg::threads;

	cout_time("Contraction of type4 begins!");
	complex<double> *type4_threaded = (complex<double> *)smalloc(cname,fname,"type4_threaded",sizeof(complex<double>)*t_size*t_size*16*n_threads);// 16 because 16 different contractions
	complex<double> *type4S5D_threaded = (complex<double> *)smalloc(cname,fname,"type4S5D_threaded",sizeof(complex<double>)*t_size*t_size*n_threads);
	((Vector*)type4_threaded)->VecZero(t_size*t_size*16*n_threads*2);
	((Vector*)type4S5D_threaded)->VecZero(t_size*t_size*n_threads*2);

	omp_set_num_threads(n_threads);
#pragma omp parallel for
	for(int x = 0; x < size_4d; x++) {
  	int id = omp_get_thread_num();
		int t_op = x / size_3d + GJP.TnodeSites()*GJP.TnodeCoor();
		complex<double> *offset_type4;
		complex<double> *offset_type4S5D;
		for(int t_k = 0; t_k < t_size; t_k++) {
			int t_dis = (t_op - t_k + t_size) % t_size;
			WilsonMatrix Mtemp0 = Build_Wilson(mf_sl, x, t_k, STRANGE, LIGHT);
			WilsonMatrix Mtemp1 = Build_Wilson(mf_ls, x, t_k, LIGHT, STRANGE);
			WilsonMatrix Mtemp2 = Build_Wilson_loop(x, LIGHT);
			WilsonMatrix Mtemp3 = Build_Wilson_loop(x, STRANGE);

			for(int dir = 0; dir < 4; dir++) {
			//--------------------Fig 33 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 0)) * n_threads + id;
			(*offset_type4)-=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
			(*offset_type4)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
			//--------------------Fig 35 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 2)) * n_threads + id;
			(*offset_type4)+=(Mtemp1.glV(dir)).Trace() * (Mtemp2.glA(dir)).Trace();
			(*offset_type4)-=(Mtemp1.glA(dir)).Trace() * (Mtemp2.glV(dir)).Trace();
			//--------------------Fig 34 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 1)) * n_threads + id;
			(*offset_type4)-=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
			(*offset_type4)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
			//--------------------Fig 36 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 3)) * n_threads + id;
			(*offset_type4)+=Tr(SpinTrace(Mtemp1.glV(dir)) , SpinTrace(Mtemp2.glA(dir)));
			(*offset_type4)-=Tr(SpinTrace(Mtemp1.glA(dir)) , SpinTrace(Mtemp2.glV(dir)));
			//--------------------Fig 37 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 4)) * n_threads + id;
			(*offset_type4)-=(Mtemp0.glV(dir)).Trace() * (Mtemp3.glA(dir)).Trace();
			(*offset_type4)-=(Mtemp0.glA(dir)).Trace() * (Mtemp3.glV(dir)).Trace();
			//--------------------Fig 39 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 6)) * n_threads + id;
			(*offset_type4)+=(Mtemp0.glV(dir)).Trace() * (Mtemp3.glA(dir)).Trace();
			(*offset_type4)-=(Mtemp0.glA(dir)).Trace() * (Mtemp3.glV(dir)).Trace();
			//--------------------Fig 38 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 5)) * n_threads + id;
			(*offset_type4)-=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp3.glA(dir)));
			(*offset_type4)-=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp3.glV(dir)));
			//--------------------Fig 40 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 7)) * n_threads + id;
			(*offset_type4)+=Tr(SpinTrace(Mtemp0.glV(dir)) , SpinTrace(Mtemp3.glA(dir)));
			(*offset_type4)-=Tr(SpinTrace(Mtemp0.glA(dir)) , SpinTrace(Mtemp3.glV(dir)));
			//--------------------Fig 41 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 8)) * n_threads + id;
			(*offset_type4)-=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
			(*offset_type4)-=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
			//--------------------Fig 43 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 10)) * n_threads + id;
			(*offset_type4)-=Trace(Mtemp1.glV(dir) , Mtemp2.glA(dir));
			(*offset_type4)+=Trace(Mtemp1.glA(dir) , Mtemp2.glV(dir));
			//--------------------Fig 42 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 9)) * n_threads + id;
			(*offset_type4)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
			(*offset_type4)-=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			//--------------------Fig 44 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 11)) * n_threads + id;
			(*offset_type4)-=Tr(ColorTrace(Mtemp1.glV(dir)) , ColorTrace(Mtemp2.glA(dir)));
			(*offset_type4)+=Tr(ColorTrace(Mtemp1.glA(dir)) , ColorTrace(Mtemp2.glV(dir)));
			//--------------------Fig 45 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 12)) * n_threads + id;
			(*offset_type4)-=Trace(Mtemp0.glV(dir) , Mtemp3.glA(dir));
			(*offset_type4)-=Trace(Mtemp0.glA(dir) , Mtemp3.glV(dir));
			//--------------------Fig 47 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 14)) * n_threads + id;
			(*offset_type4)+=Trace(Mtemp0.glV(dir) , Mtemp3.glA(dir));
			(*offset_type4)-=Trace(Mtemp0.glA(dir) , Mtemp3.glV(dir));
			//--------------------Fig 46 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 13)) * n_threads + id;
			(*offset_type4)-=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp3.glA(dir)));
			(*offset_type4)-=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp3.glV(dir)));
			//--------------------Fig 48 below--------------------//
			offset_type4 = type4_threaded + (t_k + t_size * (t_dis + t_size * 15)) * n_threads + id;
			(*offset_type4)+=Tr(ColorTrace(Mtemp0.glV(dir)) , ColorTrace(Mtemp3.glA(dir)));
			(*offset_type4)-=Tr(ColorTrace(Mtemp0.glA(dir)) , ColorTrace(Mtemp3.glV(dir)));
			}
			//--------------------Type4 S5D--------------------//
			offset_type4S5D = type4S5D_threaded + (t_dis * t_size + t_k) * n_threads + id;
			(*offset_type4S5D)+=(Mtemp0.gl(-5)).Trace();
		}
	}
	complex<double> *type4 = (complex<double> *)smalloc(cname,fname,"type4",sizeof(complex<double>)*t_size*t_size*16);
	complex<double> *type4S5D = (complex<double> *)smalloc(cname,fname,"type4S5D",sizeof(complex<double>)*t_size*t_size);
  ((Vector*)type4)->VecZero(t_size*t_size*16*2);
  ((Vector*)type4S5D)->VecZero(t_size*t_size*2);
	for(int i = 0; i < t_size*t_size*16; i++)
		for(int id = 0; id < n_threads; id++)
			type4[i] += type4_threaded[i*n_threads+id];
	for(int i = 0; i < t_size*t_size; i++)
		for(int id = 0; id < n_threads; id++)
			type4S5D[i] += type4S5D_threaded[i*n_threads+id];
	sfree(cname,fname,"type4_threaded",type4_threaded);
	QMP_sum_double_array((Float *)type4,t_size*t_size*16*2);
	QMP_sum_double_array((Float *)type4S5D,t_size*t_size*2);
	cout_time("Contraction of type4 ends!");

	char fn[1024];
	sprintf(fn,"%s_type4",common_arg->filename);
	writeCorr(type4, fn, 16, t_size*t_size);
	sprintf(fn,"%s_type4S5D",common_arg->filename);
	writeCorr(type4S5D, fn, 1, t_size*t_size);

	sfree(cname,fname,"type4",type4);
	sfree(cname,fname,"type4S5D",type4S5D);
}

WilsonMatrix MesonField::Build_Wilson(Vector *mf, int x_op, int t_mf, QUARK quark_v, QUARK quark_w)
//v^i(x) \cdot \pi^ij(t_mf) \cdot w^j(x)
{
  char *fname = "Build_Wilson(Vector *mf, int x_op, int t_mf, QUARK quark_v, QUARK quark_w)";
	const int nl_v = nl[quark_v];
	const int nl_w = nl[quark_w];
	const int nbase_v = nbase[quark_v];
	const int nbase_w = nbase[quark_w];
	const int nhits_v = nhits[quark_v];
	const int nhits_w = nhits[quark_w];
	const int nvec_compact_v = nl_v + sc_size * nhits_v;
	const int nvec_compact_w = nl_w + nhits_w;
	const int src_width_v = src_width[quark_v];
	const int src_width_w = src_width[quark_w];
	A2APropbfm *a2a_ptr[2] = {this->a2a_prop, this->a2a_prop_s};


	WilsonMatrix ret(0.);
	int t_op = x_op / size_3d + GJP.TnodeSites() * GJP.TnodeCoor();

	int offset_i;
	int offset_j;
	complex<double> *v_off;
	complex<double> *w_off;
	complex<double> *mf_off;

	for(int s1 = 0; s1 < 4; s1++)
		for(int c1 = 0; c1 < 3; c1++)
			for(int s2 = 0; s2 < 4; s2++)
				for(int c2 = 0; c2 < 3; c2++) 
				{
					int j_sc = s2 * 3 + c2; // Non-zero spin-color index of w 
					for(int i = 0; i < nvec_compact_v; i++)
						for(int j = 0; j < nvec_compact_w; j++) {
							offset_i = i<nl_v?i:(nl_v + (i-nl_v) / sc_size * nbase_v + t_mf / src_width_v * sc_size + (i-nl_v) % 12);
							offset_j = j<nl_w?j:(nl_w + (j-nl_w) * nbase_w + t_op / src_width_w * sc_size + j_sc);
							v_off = (complex<double> *)(a2a_ptr[quark_v]->get_v(offset_i)) + x_op * sc_size + (s1 * 3 + c1);
							if(j<nl_w) w_off = (complex<double> *)(a2a_ptr[quark_w]->get_wl(j)) + x_op * sc_size + (s2 * 3 + c2);
							else w_off = (complex<double> *)(a2a_ptr[quark_w]->get_wh()) + size_4d * (j - nl_w) + x_op;
							mf_off = (complex<double> *)mf + t_size * (offset_j + nvec[quark_w] * i) + t_mf;
							ret(s1,c1,s2,c2) += (*v_off) * (*mf_off) * conj(*w_off);
						}
				}
	return ret;
}

WilsonMatrix MesonField::Build_Wilson_ww(Vector *mf, int x_op, int t_mf_left, int t_mf_right, QUARK v_left, QUARK v_right)
//v^i(x) \cdot \pi^ij(t_mf) \cdot v^j(x)
{
  char *fname = "Build_Wilson(Vector *mf, int x_op, int t_mf_left, int t_mf_right, QUARK v_left, QUARK v_right)";
	const int nl_left = nl[v_left];
	const int nl_right = nl[v_right];
	const int nbase_left = nbase[v_left];
	const int nbase_right = nbase[v_right];
	const int nhits_left = nhits[v_left];
	const int nhits_right = nhits[v_right];
	const int nvec_compact_left = nl_left + sc_size * nhits_left;
	const int nvec_compact_right = nl_right + sc_size * nhits_right;
	const int src_width_left = src_width[v_left];
	const int src_width_right = src_width[v_right];
	A2APropbfm *a2a_ptr[2] = {this->a2a_prop, this->a2a_prop_s};


	WilsonMatrix ret(0);
	int t_op = x_op / size_3d + GJP.TnodeSites() * GJP.TnodeCoor();

	int offset_i;
	int offset_j;
	complex<double> *left_off;
	complex<double> *right_off;
	complex<double> *mf_off;

	for(int s1 = 0; s1 < 4; s1++) {
		for(int c1 = 0; c1 < 3; c1++) {
			for(int s2 = 0; s2 < 4; s2++) {
				for(int c2 = 0; c2 < 3; c2++) {
					for(int i = 0; i < nvec_compact_left; i++)
						for(int j = 0; j < nvec_compact_right; j++) {
							offset_i = i<nl_left?i:(nl_left + (i-nl_left) / sc_size * nbase_left + t_mf_left / src_width_left * sc_size + (i-nl_left) % 12);
							offset_j = j<nl_right?j:(nl_right + (j-nl_right) /sc_size * nbase_right + t_mf_right / src_width_right * sc_size + (j-nl_right) % 12);
							left_off = (complex<double> *)(a2a_ptr[v_left]->get_v(offset_i)) + x_op * sc_size + (s1 * 3 + c1);
							right_off = (complex<double> *)(a2a_ptr[v_right]->get_v(offset_j)) + x_op * sc_size + (s2 * 3 + c2);
							mf_off = (complex<double> *)mf + t_size * (j + nvec_compact_right * i) + t_mf_left;
							ret(s1,c1,s2,c2) += (*left_off) * (*mf_off) * conj(*right_off);
						}
				}
			}
		}
	}
	return ret;
}

WilsonMatrix MesonField::Build_Wilson_loop(int x_op, QUARK quark)
// \rho(x_op) = v^i(x_op) \cdot w^i(x_op)
{
  char *fname = "Build_Wilson_loop(int x_op)";

	const int nvec_compact[2] = {this->nl[0] + sc_size * this->nhits[0], this->nl[1] + sc_size * this->nhits[1]};
	A2APropbfm *a2aprop[2] = {this->a2a_prop, this->a2a_prop_s};

	const int nvec_this = nvec[quark];
	const int nhits_this = nhits[quark];
	const int nvec_compact_this = nl[quark] + nhits[quark];
	const int nl_this = nl[quark];
	const int nbase_this = nbase[quark];
	const int src_width_this = src_width[quark];
	A2APropbfm *a2a_prop_this = a2aprop[quark];


	WilsonMatrix ret(0);
	int t_op = x_op / size_3d + GJP.TnodeSites() * GJP.TnodeCoor();

	complex<double> *v_off;
	complex<double> *w_off;
	int offset_i;
	for(int s1 = 0; s1 < 4; s1++)
		for(int c1 = 0; c1 < 3; c1++)
			for(int s2 = 0; s2 < 4; s2++)
				for(int c2 = 0; c2 < 3; c2++) {
					int i_sc = s2 * 3 + c2; // Non-zero spin-color index of w 
					for(int i = 0; i < nvec_compact_this; i++) {
						offset_i = i<nl_this?i:(nl_this + (i - nl_this) * nbase_this + t_op / src_width_this * sc_size + i_sc);
						v_off = (complex<double>*)a2a_prop_this->get_v(offset_i) + x_op * sc_size + (s1*3+c1);
						if(i<nl_this) w_off = (complex<double> *)(a2a_prop_this->get_wl(i)) + x_op * sc_size + i_sc;
						else w_off = (complex<double> *)(a2a_prop_this->get_wh()) + size_4d * (i - nl_this) + x_op; 
						ret(s1,c1,s2,c2) += (*v_off) * conj(*w_off);
					}
				}
	return ret;
}

complex<double> MesonField::Gamma5(complex<double> *v, complex<double> *w, complex<double> psi)
{
	complex<double> ret(0,0);
	int half_sc = 6;
	for(int i=0;i<half_sc;i++)
		ret += v[i]*conj(w[i]);
	for(int i=half_sc;i<2*half_sc;i++)
		ret -= v[i]*conj(w[i]);
	ret *= psi;
	return ret;
}

complex<double> MesonField::Unit(complex<double> *left, complex<double> *right, complex<double> psi)
{
	complex<double> ret(0,0);
	for(int i=0;i<sc_size;i++)
		ret += right[i]*conj(left[i]);
	ret *= psi;
	return ret;
}

void MesonField::save_mf(char *mf_kind)
{
	char *fname = "save_mf()";
	
	char fn[200];
	sprintf(fn,"%s_mf_nl%d_nh%d_srcwidth%d_T%d_%s",common_arg->filename,nl[0],nhits[0],src_width[0],t_size,mf_kind);
	FILE *p;
	if((p = Fopen(fn,"w")) == NULL)
		ERR.FileA(cname,fname,fn);
	int offset_i;
	int offset_j;
	for(int t=0;t<t_size;t++)
		for(int i=0;i<nvec[0];i++)
			for(int j=0;j<nl[0]+nhits[0]*12;j++)
				Fwrite((Float *)mf+2 * (t + t_size * (i + nvec[0] * j)),8,2,p);
	Fclose(p);
}

void MesonField::save_mf_ls(char *mf_kind)
{
	char *fname = "save_mf_ls()";
	
	char fn[200];
	sprintf(fn,"%s_mf_nl%d_nh%d_srcwidth%d_T%d_%s",common_arg->filename,nl[0],nhits[0],src_width[0],t_size,mf_kind);
	FILE *p;
	if((p = Fopen(fn,"w")) == NULL)
		ERR.FileA(cname,fname,fn);
	for(int t=0;t<t_size;t++)
		for(int i=0;i<nvec[1];i++)
			for(int j=0;j<nl[0]+nhits[0]*12;j++)
				Fwrite((Float *)mf_ls+ 2 * (t + t_size * (i + nvec[1] * j)),8,2,p);
	Fclose(p);
}

void MesonField::save_mf_sl(char *mf_kind)
{
	char *fname = "save_mf_sl()";
	
	char fn[200];
	sprintf(fn,"%s_mf_nl%d_nh%d_srcwidth%d_T%d_%s",common_arg->filename,nl[0],nhits[0],src_width[1],t_size,mf_kind);
	FILE *p;
	if((p = Fopen(fn,"w")) == NULL)
		ERR.FileA(cname,fname,fn);
	for(int t=0;t<t_size;t++)
		for(int i=0;i<nvec[0];i++)
			for(int j=0;j<nl[1]+nhits[1]*12;j++)
				Fwrite((Float *)mf_sl+2 * (t + t_size * (i + nvec[0] * j)),8,2,p);
	Fclose(p);
}

void MesonField::writeCorr(complex<double> *corr, char *filename, int num_fig, int T)
{
	char *fname = "writeCorr(complex<double> *corr, char *filename, int num_fig, int T)";
	FILE *p;
	if((p = Fopen(filename,"w")) == NULL)
		ERR.FileA(cname,fname,filename);
	for(int i = 0; i < T; i++) {
		Fprintf(p,"%d",i);
		for(int fig = 0; fig < num_fig; fig++)
			Fprintf(p,"\t%.16e\t%.16e",corr[fig*T+i].real(), corr[fig*T+i].imag());
		Fprintf(p,"\n");
	}
	Fclose(p);
}

MesonField::~MesonField()
{
	char *fname = "~MesonField()";
  free_vw_fftw();
	if(mf) sfree(cname,fname,"mf",mf);
	if(mf_ls) sfree(cname,fname,"mf_ls",mf_ls);
	if(mf_sl) sfree(cname,fname,"mf_sl",mf_sl);
	for(int i = 0; i < 3; i++) {
		if(mf_ww[i]) {
			sfree(cname, fname, "mf_ww[i]", mf_ww[i]);
			mf_ww[i] = NULL;
		}
	}
	if(src)	fftw_free(src);
}

void MesonField::show_wilson(const WilsonMatrix &a)
{
	for(int s1 = 0; s1 < 4; s1++)
    for(int c1 = 0; c1 < 3; c1++)
		{
			for(int s2 = 0; s2 < 4; s2++)
				for(int c2 = 0; c2 < 3; c2++)
					QDPIO::cout<<"("<<a(s1,c1,s2,c2).real()<<","<<a(s1,c1,s2,c2).imag()<<")"<<"\t";
			QDPIO::cout<<endl;
		}
}
