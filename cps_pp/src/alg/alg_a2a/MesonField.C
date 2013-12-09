#include <config.h>
#include <alg/lanc_arg.h>
#include <alg/a2a/MesonField.h>

#include <fstream>
#include <alg/wilson_matrix.h>
#include <util/rcomplex.h>
#include <util/wilson.h>
#include <util/qcdio.h>
#include <assert.h>
#ifdef USE_QMP
#include <qmp.h>
#endif

using namespace std;

CPS_START_NAMESPACE

int cout_time(char *info)
{
	time_t t=time( 0 );
	char tmp[64];
	strftime(tmp, sizeof(tmp), "%Y/%m/%d %X %A %z",localtime(&t) );
	QDPIO::cout<<tmp<<"\t"<<info<<endl;
	return 0;
}


//CK: modified global sum routines used here such that QMP is not required
static void sum_double_array(Float* data, const int &size){
#ifdef USE_QMP
  QMP_sum_double_array((double *)data, size);  
#else
  slice_sum(data,size,99);
#endif
}

MesonField::MesonField(Lattice &_lat, A2APropbfm *_a2aprop, AlgFixGauge *_fixgauge, CommonArg *_common_arg):a2a_prop(_a2aprop),a2a_prop_s(NULL),fix_gauge(_fixgauge),lat(_lat),common_arg(_common_arg),sc_size(12)
{
  cname = "MesonField";
  const char *fname = "Mesonfield(A2APropbfm &_a2aprop, AlgFixGauge &_fixgauge,  int src_type, int snk_type, int *pv, int *pw)";
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

  int fftw_alloc_size = size_4d/GJP.TnodeSites();  //if(GJP.Gparity()) fftw_alloc_size *= 2; Use same source of each combination of flavour index of w and v
  src = fftw_alloc_complex(fftw_alloc_size);

  assert(sizeof(complex<double>[2]) == sizeof(double[4])); //check that we can cast complex<double>* to double*
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

  int fftw_alloc_size = size_4d/GJP.TnodeSites(); //(GJP.Gparity()?2:1) *     Use same source of each combination of flavour index of w and v
  src = fftw_alloc_complex(fftw_alloc_size);

  assert(sizeof(complex<double>[2]) == sizeof(double[4])); //check that we can cast complex<double>* to double*
}

void MesonField::allocate_vw_fftw()
{
  const char *fname = "allocate_vw_fftw()";

  int ferm_sz = size_4d * sc_size * 2 *sizeof(Float);
  if(GJP.Gparity()) ferm_sz *= 2;
  int v_ptr_sz = sizeof(Vector *);

  v_fftw = (Vector **)smalloc(cname, fname, "v_fftw", v_ptr_sz * nvec[0]);
  for(int i = 0; i < nvec[0]; ++i) { 
    v_fftw[i]  = (Vector *)smalloc(cname, fname, "v[i]" , ferm_sz);
  }

  wl_fftw[0] = (Vector **)smalloc(cname, fname, "wl_fftw[0][i]", v_ptr_sz * nl[0]);
  for(int i = 0; i < nl[0]; ++i) {
    wl_fftw[0][i] = (Vector *)smalloc(cname, fname, "wl_fftw[0][i]", ferm_sz);
  }

  wh_fftw[0] = (Vector *)smalloc(cname, fname, "wh_fftw[0]", nhits[0]*ferm_sz*sc_size); //separate FT of each spin-color component, hence 12 fermion vectors per hit

  printf("nl[0] %d nhits[0] %d nvec[0] %d\n",nl[0],nhits[0],nvec[0]);

  if(do_strange) {
    printf("MesonField::allocate_vw_fftw allocating for strange quark\n"); 
    v_s_fftw = (Vector **)smalloc(cname, fname, "v_s_fftw", v_ptr_sz * nvec[1]);
    wl_fftw[1] = (Vector **)smalloc(cname, fname, "wl_fftw[1]", v_ptr_sz * nl[1]);
    for(int i = 0; i < nvec[1]; ++i) {
      v_s_fftw[i]  = (Vector *)smalloc(cname, fname, "v_s_fftw[i]" , ferm_sz);
    }
    for(int i = 0; i < nl[1]; ++i) {
      wl_fftw[1][i] = (Vector *)smalloc(cname, fname, "wl_fftw[1][i]", ferm_sz);
    }
    wh_fftw[1] = (Vector *)smalloc(cname, fname, "wh_fftw[1]", nhits[1]*ferm_sz*sc_size);

    printf("nl[1] %d nhits[1] %d nvec[1] %d\n",nl[1],nhits[1],nvec[1]);
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
  //CK: This function originally assumes Coulomb gauge fixing. For Landau it will cause a SEGV. I have modified it to use 
  //    my generic function that picks out the correct gauge fixing matrix for a given site/flavour

  char *fname = "gf_vector(Vector *in, Vector *out)";

  //cps::Matrix **gm = lat.FixGaugePtr();

  Vector tmp;

  int size_3d = GJP.VolNodeSites() / GJP.TnodeSites();

  Float *fi = (Float *)in;
  Float *fo = (Float *)out;

  int nsites = size_4d; if(GJP.Gparity()) nsites *= 2;

  for(int i = 0; i < nsites; ++i) {
    //const int t = i / size_3d;
    //const int moff = i % size_3d;

    const int flav = i / size_4d; 
    const int site = i % size_4d;

    const cps::Matrix* gfmat = lat.FixGaugeMatrix(site,flav);

    //const int t = rem / size_3d;  rem = rem % size_3d;
    //const int moff = rem;
    const int voff = sc_size * 2 * i; //four-d ferm vector offset in units of float
    for(int s = 0; s < 4; ++s) {
      tmp.CopyVec((Vector *)(fi + voff + 6 * s), 6);
      //uDotXEqual(fo + voff + 6 * s, (Float *)(&gm[t][moff]), (Float *)(&tmp));
      uDotXEqual(fo + voff + 6 * s, (Float *)gfmat, (Float *)(&tmp));
    }
  }
}

void MesonField::cnv_lcl_glb(fftw_complex *glb, fftw_complex *lcl, bool lcl_to_glb)
{
  //CK: Removed static status because sometimes we change the lattice size during a calculation
  const int glb_dim[4] = {
    GJP.XnodeSites() * GJP.Xnodes(),
    GJP.YnodeSites() * GJP.Ynodes(),
    GJP.ZnodeSites() * GJP.Znodes(),
    GJP.TnodeSites() * GJP.Tnodes(),
  };

  const int lcl_dim[4] = {
    GJP.XnodeSites(),
    GJP.YnodeSites(),
    GJP.ZnodeSites(),
    GJP.TnodeSites(),
  };

  const int shift[4] = {
    GJP.XnodeSites() * GJP.XnodeCoor(),
    GJP.YnodeSites() * GJP.YnodeCoor(),
    GJP.ZnodeSites() * GJP.ZnodeCoor(),
    GJP.TnodeSites() * GJP.TnodeCoor(),
  };

  int glb_vol = (GJP.Gparity()?2:1)*glb_dim[0] * glb_dim[1] * glb_dim[2] * glb_dim[3];
  if(lcl_to_glb) {
    set_zero(glb, glb_vol * sc_size);
  }

  int lcl_size_3d = lcl_dim[0] * lcl_dim[1] * lcl_dim[2];
  int glb_size_3d = glb_dim[0] * glb_dim[1] * glb_dim[2];

  int nflav = GJP.Gparity() ? 2 : 1;
  
  for(int flav = 0; flav < nflav; ++flav){
  for(int t = 0; t < lcl_dim[3]; ++t) {
    fftw_complex *lcl_slice = lcl + flav * lcl_dim[3] * lcl_size_3d * sc_size + t * lcl_size_3d * sc_size; //CK: second G-parity flavour stacked after first four-volume as usual
    fftw_complex *glb_slice = glb + flav * glb_dim[3] * glb_size_3d * sc_size + (t + shift[3]) * glb_size_3d * sc_size; //CK: Similar but in a global sense

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
  }}

  if(lcl_to_glb) {
    sum_double_array((Float *)glb, glb_vol * sc_size * 2);
  }
}

static void Gparity_1f_FFT(fftw_complex *fft_mem, const int &sc, const int &sc_size, const int &t_size){
  if(GJP.Gparity1fX()){
    if(GJP.Gparity1fY() && !UniqueID()){ printf("MesonField.C - Gparity_1f_FFT: Not setup for 2 directions of G-parity\n"); exit(-1); }
    //We need to independently FFT over the first and second halves of the lattice, which represent the two independent flavours
    int fft_dim[3] = { GJP.ZnodeSites() * GJP.Znodes(),
		       GJP.YnodeSites() * GJP.Ynodes(),
		       GJP.XnodeSites() * GJP.Xnodes()};
    const int size_3d_glb = fft_dim[0] * fft_dim[1] * fft_dim[2];    
    
    //Allocate 2 half-volumes
    int fftw_alloc_sz = size_3d_glb * t_size * sc_size / 2;
    fftw_complex *fft_flavs[2] = { fftw_alloc_complex(fftw_alloc_sz), fftw_alloc_complex(fftw_alloc_sz) };

    //Copy from fft_mem into the two half-volumes
    for(int flav=0;flav<2;flav++){
      for(int x=0;x<fft_dim[2]/2;x++){
	for(int y=0;y<fft_dim[1];y++){
	  for(int z=0;z<fft_dim[0];z++){
	    for(int t=0;t<t_size;t++){
	      int dbl_site_off = (x + flav*fft_dim[2]/2) + fft_dim[2]*( y + fft_dim[1]* ( z + fft_dim[0]*t ));
	      int sngl_site_off = x + fft_dim[2]/2*( y + fft_dim[1]* ( z + fft_dim[0]*t ));
	      for(int scidx=0;scidx<sc_size;++scidx){
		fft_flavs[flav][sngl_site_off*sc_size + scidx][0] = fft_mem[dbl_site_off*sc_size + scidx][0]; 
		fft_flavs[flav][sngl_site_off*sc_size + scidx][1] = fft_mem[dbl_site_off*sc_size + scidx][1]; 
	      }
	    }
	  }
	}
      }
    }
    
    //FFT the two halves independently
    int n_fft = t_size;
    fft_dim[2]/=2;
    {
      fftw_plan plan = fftw_plan_many_dft(3, fft_dim, n_fft,
					  fft_flavs[0] + sc, NULL, sc_size, size_3d_glb/2 * sc_size,
					  fft_flavs[0] + sc, NULL, sc_size, size_3d_glb/2 * sc_size,
					  FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_execute(plan);
      fftw_destroy_plan(plan);
    }

    {
      fftw_plan plan = fftw_plan_many_dft(3, fft_dim, n_fft,
					  fft_flavs[1] + sc, NULL, sc_size, size_3d_glb/2 * sc_size,
					  fft_flavs[1] + sc, NULL, sc_size, size_3d_glb/2 * sc_size,
					  FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_execute(plan);
      fftw_destroy_plan(plan);
    }
    fft_dim[2]*=2;

    //Convert back to double lattice
    for(int flav=0;flav<2;flav++){
      for(int x=0;x<fft_dim[2]/2;x++){
	for(int y=0;y<fft_dim[1];y++){
	  for(int z=0;z<fft_dim[0];z++){
	    for(int t=0;t<t_size;t++){
	      int dbl_site_off = (x + flav*fft_dim[2]/2) + fft_dim[2]*( y + fft_dim[1]* ( z + fft_dim[0]*t ));
	      int sngl_site_off = x + fft_dim[2]/2*( y + fft_dim[1]* ( z + fft_dim[0]*t ));
	      for(int scidx=0;scidx<sc_size;++scidx){
		fft_mem[dbl_site_off*sc_size + scidx][0] = fft_flavs[flav][sngl_site_off*sc_size + scidx][0];
		fft_mem[dbl_site_off*sc_size + scidx][1] = fft_flavs[flav][sngl_site_off*sc_size + scidx][1];
	      }
	    }
	  }
	}
      }
    }

    //Deallocate
    fftw_free(fft_flavs[0]);
    fftw_free(fft_flavs[1]);
  }else{
    if(!UniqueID()) printf("MesonField.C - Gparity_1f_FFT: Must be used only when G-parity 1f is active\n"); 
    exit(-1); 
  }
}

//load the v/w vector and compute their fourier transform

//CK: FFT stuff should be methods belonging to the A2AProp class
void MesonField::prepare_vw()
{
  const char *fname = "prepare_vw()";

  if(lat.FixGaugeKind() != FIX_GAUGE_COULOMB_T){
    if(lat.FixGaugeKind() != FIX_GAUGE_NONE) lat.FixGaugeFree();

    cout_time("Coulomb T gauge fixing BEGIN!");
    fix_gauge->run();
    cout_time("Coulomb T gauge fixing END!");
  }	

  fftw_init_threads();
  fftw_plan_with_nthreads(bfmarg::threads);
  const int fft_dim[3] = { GJP.ZnodeSites() * GJP.Znodes(),
			   GJP.YnodeSites() * GJP.Ynodes(),
			   GJP.XnodeSites() * GJP.Xnodes()};
  const int size_3d_glb = fft_dim[0] * fft_dim[1] * fft_dim[2];

  int fftw_alloc_sz = size_3d_glb * t_size * sc_size; if(GJP.Gparity()) fftw_alloc_sz *= 2;
  fftw_complex *fft_mem = fftw_alloc_complex(fftw_alloc_sz);

  int t_size_alloc = sizeof(complex<double>) * size_4d_sc; if(GJP.Gparity()) t_size_alloc *=2;
  Vector *t  = (Vector *)smalloc(cname, fname, "t" , t_size_alloc);
  fftw_plan plan;

  // load and process v
  for(int j = 0; j < nvec[0]; ++j) {
    gf_vec(t, a2a_prop->get_v(j));
    cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(t), true); //convert local 't' to global 'fft_mem'
    for(int sc = 0; sc < sc_size; sc++) //loop over the 12 spin-colour indices
      {
	//CK: DFT on each timeslice. Arguments are 
	//1) tensor rank = 3,  
	//2) the dimension in each direction being fft_dim[dir]
	//3) how many FFTs are performed = Lt (aka t_size)
	//4_ base pointer of input = fft_mem + sc
	//5) arg for advanced feature not used = NULL
	//6) the offset between data included in the FFT. The spin-colour indices are transformed separately, hence stride = sc_size
	//7) the offset between data blocks used for each of the transforms (of which there are Lt) = sc_size * three-volume
	//8,9,10,11) same as previous four but for output. As we FFT in-place these are identical to the above
	//12,13) Other FFT args

	//For G-parity the vector for the second flavour is stacked after the four-volume associated with the first flavour, so we only need to modify the number of FFTs
	int n_fft = t_size; if(GJP.Gparity()) n_fft *= 2;

	if(GJP.Gparity1fX()) Gparity_1f_FFT(fft_mem, sc,sc_size,t_size); //1f G-parity testing
	else{
	  plan = fftw_plan_many_dft(3, fft_dim, n_fft,
				    fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
				    fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
				    FFTW_FORWARD, FFTW_ESTIMATE);
	  fftw_execute(plan);
	  fftw_destroy_plan(plan);
	}
      }
    cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(v_fftw[j]), false);
  }

  // load and process wl
  for(int j = 0; j < nl[0]; ++j) {
    gf_vec(t, a2a_prop->get_wl(j));
    cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(t), true);
    for(int sc = 0; sc < sc_size; sc++)
      {
	int n_fft = t_size; if(GJP.Gparity()) n_fft *= 2;
	
	if(GJP.Gparity1fX()) Gparity_1f_FFT(fft_mem, sc,sc_size,t_size); //1f G-parity testing
	else{
	  plan = fftw_plan_many_dft(3, fft_dim, n_fft,
				    fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
				    fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
				    FFTW_FORWARD, FFTW_ESTIMATE);
	  fftw_execute(plan);
	  fftw_destroy_plan(plan);
	}
      }
    cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(wl_fftw[0][j]), false);
  }
  // load and process wh  
  //In A2APropBfm, wh is 'nhits' field of c-numbers, and we generate independent v for each timeslice, spin-color and hit index
  //Here we perform the FFT for each spin-color index of w separately
  for(int j = 0; j < nhits[0]; ++j) 
    for(int w_sc = 0; w_sc < sc_size; w_sc++) {
       //Perform separate FFT for each spin-color component
      int off_fac = 1; if(GJP.Gparity()) off_fac = 2;
      Float *wh_fftw_offset = (Float *)(wh_fftw[0]) + off_fac * size_4d_sc * 2 * (j * sc_size + w_sc); //each hit is offset by  four_vol * 12*12 * 2
      wh2w((Vector *)wh_fftw_offset, a2a_prop->get_wh(), j, w_sc);  //wh_fftw_offset is set to zero apart from on spin-color index w_sc, where it is set to the value from hit j of wh
      gf_vec((Vector *)wh_fftw_offset, (Vector *)wh_fftw_offset);
      cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(wh_fftw_offset), true);
      for(int sc = 0; sc < sc_size; sc++)
	{
	  int n_fft = t_size; if(GJP.Gparity()) n_fft *= 2;

	  if(GJP.Gparity1fX()) Gparity_1f_FFT(fft_mem, sc,sc_size,t_size); //1f G-parity testing
	  else{
	    plan = fftw_plan_many_dft(3, fft_dim, n_fft,
				      fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
				      fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
				      FFTW_FORWARD, FFTW_ESTIMATE);

	    fftw_execute(plan);
	    fftw_destroy_plan(plan);
	  }
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
	  int n_fft = t_size; if(GJP.Gparity()) n_fft *= 2;

	  if(GJP.Gparity1fX()) Gparity_1f_FFT(fft_mem, sc,sc_size,t_size); //1f G-parity testing
	  else{
	    plan = fftw_plan_many_dft(3, fft_dim, n_fft,
				      fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
				      fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
				      FFTW_FORWARD, FFTW_ESTIMATE);
	    fftw_execute(plan);
	    fftw_destroy_plan(plan);
	  }
	}
      cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(v_s_fftw[j]), false);
    }

    // load and process wl
    for(int j = 0; j < nl[1]; ++j) {
      gf_vec(t, a2a_prop_s->get_wl(j));
      cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(t), true);
      for(int sc = 0; sc < sc_size; sc++)
	{
	  int n_fft = t_size; if(GJP.Gparity()) n_fft *= 2;

	  if(GJP.Gparity1fX()) Gparity_1f_FFT(fft_mem, sc,sc_size,t_size); //1f G-parity testing
	  else{
	    plan = fftw_plan_many_dft(3, fft_dim, n_fft,
				      fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
				      fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
				      FFTW_FORWARD, FFTW_ESTIMATE);
	    fftw_execute(plan);
	    fftw_destroy_plan(plan);
	  }
	}
      cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(wl_fftw[1][j]), false);
    }
    // load and process wh  
    for(int j = 0; j < nhits[1]; ++j) 
      for(int w_sc = 0; w_sc < sc_size; w_sc++) {
	int off_fac = 1; if(GJP.Gparity()) off_fac = 2;
	Float *wh_fftw_offset = (Float *)(wh_fftw[1]) + off_fac * size_4d_sc * 2 * (j * sc_size + w_sc);
	wh2w((Vector *)wh_fftw_offset, a2a_prop_s->get_wh(), j, w_sc);
	gf_vec((Vector *)wh_fftw_offset, (Vector *)wh_fftw_offset);
	cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(wh_fftw_offset), true);
	for(int sc = 0; sc < sc_size; sc++)
	  {
	    int n_fft = t_size; if(GJP.Gparity()) n_fft *= 2;

	    if(GJP.Gparity1fX()) Gparity_1f_FFT(fft_mem, sc,sc_size,t_size); //1f G-parity testing
	    else{
	      plan = fftw_plan_many_dft(3, fft_dim, n_fft,
					fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
					fft_mem + sc, NULL, sc_size, size_3d_glb * sc_size,
					FFTW_FORWARD, FFTW_ESTIMATE);
	      
	      fftw_execute(plan);
	      fftw_destroy_plan(plan);
	    }
	  }
	cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(wh_fftw_offset), false);
      }

    if(a2a_prop == a2a_prop_s){
      printf("Note: strange and light quark propagators are the same."); //Checking fftw vectors\n");
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


//CK: Method to poke onto a particular spin-color index the complex number from wh for a given hit
//    w is an output fermion field that is zero everywhere apart from on spin-color index sc
void MesonField::wh2w(Vector *w, Vector *wh, int hitid, int sc) // In order to fft the high modes scource
{
  const char *fname = "wh2w()";
	
  int hit_size = GJP.Gparity() ? 2*size_4d : size_4d;
  complex<double> *whi = (complex<double> *)wh + hitid * hit_size;
  complex<double> *t = (complex<double> *)w;

  for(int i = 0; i < hit_size; i++) {
    //memset(t + i * sc_size, 0, sizeof(complex<double>) * sc_size); //Fill the spin-color vector at position offset i with null stuff
    double *p = (double*)(t+i*sc_size);
    double *end = p + 2*sc_size; 
    while(p!=end) *(p++) = 0.0;
    t[sc + i * sc_size] = whi[i]; //At spin-colour index sc poke the complex number from wh
  }
}

void MesonField::cal_mf_ww(double rad, int src_kind, QUARK left, QUARK right)
{
  const char *fname = "cal_mf_ww()";

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

  sum_double_array((Float *)(mf_ww_this),(nl[left]+sc_size*nhits[left])*(nl[right]+sc_size*nhits[right])*t_size*2);
  QDPIO::cout<<"Global sum done!"<<endl;
}





//Calculate   \sum_k  v_i(k) \gamma^5 w_j (k)      where 'k' is a three-vector index in the range 0 -> spatial-volume. 
//'i' is the mode index of v, running from 0 -> nl + nhits * Lt * sc_size / width
//'j' is the mode index of w, running from 0 -> nl + nhits * sc_size
//The output is stored in 'mf', which has Lt complex numbers per combination of i,j. The mapping to a complex number array offset is:
//t + Lt*i + Lt*nvec*j   (global t)

//For G-parity the mesonfield is also a 2x2 matrix in flavour space:   M_{f, g}. The mapping to a complex number array offset is:
//t + Lt*f + 2*Lt*g + 4*Lt*i + 4*Lt*nvec*j

void MesonField::cal_mf_ll(double rad, int src_kind)
{
  //Unit matrix in flavour space  
  if(GJP.Gparity1fX()) return cal_mf_ll_gp1fx(rad, src_kind);

  const char *fname = "cal_mf_ll()";

  int n_flav = (GJP.Gparity() ? 2 : 1);

  mf->VecZero(nvec[0]*(nl[0]+sc_size*nhits[0]) * t_size * 2);
  prepare_src(rad, src_kind); //essentially a weighting factor for the sum over position

  printf("MesonField::cal_mf_ll rows = %d cols = %d. mf size %d\n",nl[0] + sc_size * nhits[0],nvec[0],nvec[0]*(nl[0]+sc_size*nhits[0]) * t_size * 2);

  cout_time("Inner product of light v and light w begins!");
  int nthreads = bfmarg::threads;
  omp_set_num_threads(nthreads);
#pragma omp parallel for
  for(int j = 0; j < nl[0] + sc_size * nhits[0]; j++)
    for(int i = 0; i < nvec[0]; i++){ // nvec = nl + nhits * Lt * sc_size / width   for v generated independently for each hit, source timeslice and spin-color index
      for(int x = 0; x < size_4d; x++){
	int x_3d = x % size_3d;
	int t = x / size_3d;
	int glb_t = t + GJP.TnodeCoor() * GJP.TnodeSites();
	complex<double> *mf_off = (complex<double> *)mf + (j*nvec[0]+i)*t_size +glb_t;

	complex<double> incr(0.0);

	for(int f=0;f<n_flav;++f){
	  int g = f;//unit matrix
	  
	  complex<double> *v_off = (complex<double> *)v_fftw[i] + f * size_4d * sc_size + x * sc_size;
	  complex<double> *w_off;
	  if(j<nl[0]) w_off = (complex<double> *)wl_fftw[0][j] + g * size_4d * sc_size + x * sc_size;
	  else w_off = (complex<double> *)wh_fftw[0] + size_4d * sc_size * ( n_flav*(j-nl[0]) + g ) + x * sc_size;
	    
	  incr += Gamma5(v_off,w_off,((complex<double> *)src)[x_3d]); //v g5 w*
	}
	*mf_off+=incr;

	//if(incr.real()!=0.0 || incr.imag()!=0.0) printf("i %d j %d x %d incr %f %f,  mf = %f %f\n",j,i,x,incr.real(),incr.imag(),mf_off->real(),mf_off->imag()); //yes j and i are swapped; this is because despite the way this code is written the i and j in the code are actually the *column* and *row* indices resp.
      }
    }

  cout_time("Inner product of light v and light w ends!");

  sum_double_array((Float *)mf,nvec[0]*(nl[0]+sc_size*nhits[0])*t_size*2);
  QDPIO::cout<<"Global sum done!"<<endl;
}



//FOR GPARITY 1F IN X DIRECTION ONLY
static void get_other_flavour(Float* of, Float* into, const int &site_size){ //site_size is in FLOAT UNITS
  if(GJP.Xnodes() == 1){
    //Both flavours are stored on the same node, so we just copy the second half of 'of' onto the first half of 'into' and the first half of 'of' onto the second half of 'into'. local_size is not used
    int halfX = GJP.XnodeSites()/2;
    for(int yzt = 0; yzt < GJP.YnodeSites()*GJP.ZnodeSites()*GJP.TnodeSites(); ++yzt){
      for(int x = 0; x < GJP.XnodeSites(); x++){
	int off_of = (x + GJP.XnodeSites()*yzt) * site_size;
	int off_into = ( (x+halfX)%GJP.XnodeSites() + GJP.XnodeSites()*yzt) * site_size;
	for(int s=0;s<site_size;++s) into[off_into + s ] = of[off_of + s ];
      }
    }
  }else{
    int local_size = GJP.VolNodeSites()*site_size;
    Float* buffer = (Float*)pmalloc(local_size * sizeof(Float) );
    memcpy( (void*)into, (void*)of, local_size * sizeof(Float) );
    
    Float* send = into;
    Float* recv = buffer;
    for(int shift = 0; shift < GJP.Xnodes()/2; shift++){
      getPlusData(recv,send,local_size,0); //cyclically permute anticlockwise around the torus
      Float* tmp = send;
      send = recv;
      recv = tmp;
    }
    if(recv != into) memcpy( (void*)into, (void*)recv, local_size * sizeof(Float) );
  }
}


void MesonField::cal_mf_ll_gp1fx(double rad, int src_kind)
{
  if(!GJP.Gparity1fX() || GJP.Gparity1fY() ) ERR.General("MesonField","cal_mf_ll_gp1fx(...)","Only knows how to do 1f G-parity for GPBC in X-direction\n");
  const char *fname = "cal_mf_ll_gp1fx()";
  
  int n_flav = 2;

  mf->VecZero(nvec[0]*(nl[0]+sc_size*nhits[0]) * t_size * 2);
  prepare_src(rad, src_kind); //essentially a weighting factor for the sum over position

  int nthreads = bfmarg::threads;
  omp_set_num_threads(nthreads);
  cout_time("Inner product of light v and light w begins!");


  if(GJP.Xnodes() == 1){
#pragma omp parallel for
    for(int i = 0; i < nvec[0]; i++) // nvec = nl + nhits * Lt * sc_size / width   for v generated independently for each hit, source timeslice and spin-color index
      for(int j = 0; j < nl[0] + sc_size * nhits[0]; j++){
	for(int f=0;f<n_flav;++f){
	  int g=f; //unit flavour matrix
	  for(int x = 0; x < size_4d; x++){
	    if(x % GJP.XnodeSites() >= GJP.XnodeSites()/2) continue; //only do first half (flavour 0)

	    int x_3d = x % size_3d;
	    int t = x / size_3d;
	    int glb_t = t + GJP.TnodeCoor() * GJP.TnodeSites();
	    
	    complex<double> *v_base = (f == 0 ? (complex<double> *)v_fftw[i]      :  (complex<double> *)v_fftw[i] + GJP.XnodeSites()/2*sc_size);
	    complex<double> *v_off = v_base + x * sc_size;
	      

	    complex<double> *wl_base = (g == 0 ? (complex<double> *)wl_fftw[0][j] :  (complex<double> *)wl_fftw[0][j] + GJP.XnodeSites()/2*sc_size);
	    complex<double> *wh_base = (g == 0 ? (complex<double> *)wh_fftw[0]    :  (complex<double> *)wh_fftw[0] +  GJP.XnodeSites()/2*sc_size);

	    complex<double> *w_off;
	    if(j<nl[0]) w_off = wl_base + x * sc_size;
	    else w_off = wh_base + size_4d * sc_size *(j-nl[0]) + x * sc_size;
	    
	    complex<double> *mf_off = (complex<double> *)mf + (j*nvec[0]+i)*t_size  +glb_t;

	    //complex<double> test(1.0,0.0);
	    //*mf_off+=Gamma5(v_off,w_off,test);	    

	    *mf_off+=Gamma5(v_off,w_off,((complex<double> *)src)[x_3d]);	          
	  }
	}
      }

  }else{
    int node_flav = (GJP.XnodeCoor() >= GJP.Xnodes()/2 ? 1 : 0);  //For Xnodes>1 only

    //We need to communicate and store a local copy of the v_fftw, wl_fftw and wh_fftw fields from the other flavour index - i.e. the other half of the lattice
    int ferm_sz = size_4d * sc_size * 2 *sizeof(Float);
    Vector** v_fftw_otherflav = (Vector **)pmalloc(sizeof(Vector*)* nvec[0]);
    for(int i = 0; i < nvec[0]; ++i){
      v_fftw_otherflav[i]  = (Vector *)pmalloc(ferm_sz);
      get_other_flavour( (Float*)v_fftw[i],  (Float*)v_fftw_otherflav[i], sc_size*2);
    }
    Vector** wl_fftw_otherflav = (Vector **)pmalloc(sizeof(Vector*)* nl[0]);
    for(int i = 0; i < nl[0]; ++i){
      wl_fftw_otherflav[i]  = (Vector *)pmalloc(ferm_sz);
      get_other_flavour( (Float*)wl_fftw[0][i],  (Float*)wl_fftw_otherflav[i], sc_size*2);
    }

    Vector* wh_fftw_otherflav = (Vector *)pmalloc(nhits[0]*ferm_sz*sc_size);
    for(int h=0;h<sc_size * nhits[0];h++) get_other_flavour( (Float*)wh_fftw[0] + h*size_4d * sc_size *2,  (Float*)wh_fftw_otherflav + h*size_4d * sc_size *2, sc_size*2);

#pragma omp parallel for
    for(int i = 0; i < nvec[0]; i++) // nvec = nl + nhits * Lt * sc_size / width   for v generated independently for each hit, source timeslice and spin-color index
      for(int j = 0; j < nl[0] + sc_size * nhits[0]; j++){
	for(int f=0;f<n_flav;++f){ 
	  int g=f; //unit flavour matrix
	  for(int x = 0; x < size_4d; x++){
	    int x_3d = x % size_3d;
	    int t = x / size_3d;
	    int glb_t = t + GJP.TnodeCoor() * GJP.TnodeSites();
	    
	    complex<double> *v_base = (f == node_flav ? (complex<double> *)v_fftw[i]      :  (complex<double> *)v_fftw_otherflav[i] );
	    complex<double> *v_off = v_base + x * sc_size;
	      
	    complex<double> *wl_base = (g == node_flav ? (complex<double> *)wl_fftw[0][j] :  (complex<double> *)wl_fftw_otherflav[j] );
	    complex<double> *wh_base = (g == node_flav ? (complex<double> *)wh_fftw[0]    :  (complex<double> *)wh_fftw_otherflav  );

	    complex<double> *w_off;
	    if(j<nl[0]) w_off = wl_base + x * sc_size;
	    else w_off = wh_base + size_4d * sc_size *(j-nl[0]) + x * sc_size;
	    
	    complex<double> *mf_off = (complex<double> *)mf + (j*nvec[0]+i)*t_size +glb_t;
	    *mf_off+=Gamma5(v_off,w_off,((complex<double> *)src)[x_3d]);	         
	  }
	}
      }
  }


  cout_time("Inner product of light v and light w ends!");
  
  sum_double_array((Float *)mf,nvec[0]*(nl[0]+sc_size*nhits[0])*t_size*2);
  QDPIO::cout<<"Global sum done!"<<endl;
}





void MesonField::cal_mf_ls(double rad, int src_kind)
{
  char *fname = "cal_mf_ls()";

  int n_flav = (GJP.Gparity() ? 2 : 1);

  mf_ls->VecZero(nvec[1]*(nl[0]+sc_size*nhits[0])*t_size*2);
  prepare_src(rad, src_kind);

  cout_time("Inner product of strange v and light w begins!");
  int nthreads = bfmarg::threads;
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

	    complex<double> *mf_off = (complex<double> *)mf_ls + (j*nvec[1]+i)*t_size+glb_t;

	    complex<double> incr(0.0);
	    for(int f=0;f<n_flav;++f){
	      int g=f; //unit matrix in flavour space

	      complex<double> *v_off = (complex<double> *)(v_s_fftw[i]) + f*size_4d*sc_size +  x * sc_size;
	      complex<double> *w_off;
	      if(j<nl[0]) w_off = (complex<double> *)(wl_fftw[0][j]) + g*size_4d*sc_size + x * sc_size;
	      else w_off = (complex<double> *)(wh_fftw[0]) + size_4d * sc_size * ( n_flav*(j-nl[0]) + g ) + x * sc_size;
	      
	      incr += Gamma5(v_off,w_off,((complex<double> *)src)[x_3d]);
	    }
	    *mf_off+=incr;
	    if(incr.real()!=0.0 || incr.imag()!=0.0) printf("i %d j %d x %d incr %f %f,  mf_ls = %f %f\n",j,i,x,incr.real(),incr.imag(),mf_off->real(),mf_off->imag()); //yes j and i are swapped; this is because despite the way this code is written the i and j in the code are actually the *column* and *row* indices resp.

	  }
      }
  cout_time("Inner product of strange v and light w ends!");
  sum_double_array((Float *)mf_ls,nvec[1]*(nl[0]+sc_size*nhits[0])*t_size*2);
  QDPIO::cout<<"Global sum done!"<<endl;
}

void MesonField::cal_mf_sl(double rad, int src_kind)
{
  char *fname = "cal_mf_sl()";
	
  printf("MesonField::cal_mf_sl rows = %d cols = %d. mf size %d\n",nl[1] + sc_size * nhits[1],nvec[0],nvec[0]*(nl[1]+sc_size*nhits[1]) * t_size * 2);

  mf_sl->VecZero(nvec[0]*(nl[1]+sc_size*nhits[1])*t_size*2);
  prepare_src(rad, src_kind);

  int n_flav = (GJP.Gparity() ? 2 : 1);
  
  cout_time("Inner product of light v and strange w begins!");
  int nthreads = bfmarg::threads;
  omp_set_num_threads(nthreads);
#pragma omp parallel for
  for(int j = 0; j < nl[1] + sc_size * nhits[1]; j++)
    for(int i = 0; i < nvec[0] ; i++)
      {
	for(int x = 0; x < size_4d; x++)
	  {
	    int x_3d = x % size_3d;
	    int t = x / size_3d;
	    int glb_t = t + GJP.TnodeCoor()*GJP.TnodeSites();

	    complex<double> *mf_off = (complex<double> *)mf_sl + (j*nvec[0]+i)*t_size+glb_t;

	    complex<double> incr(0.0);

	    for(int f=0;f<n_flav;++f){
	      int g=f; //unit matrix in flavour space (you could insert some other flavour matrix here if you wished)

	      complex<double> *v_off = (complex<double> *)(v_fftw[i]) + f*size_4d*sc_size + x * sc_size;
	      complex<double> *w_off;
	      if(j<nl[1]) w_off = (complex<double> *)(wl_fftw[1][j]) + g*size_4d*sc_size + x * sc_size;
	      else w_off = (complex<double> *)(wh_fftw[1]) + size_4d * sc_size * ( n_flav*(j-nl[1]) + g ) + x * sc_size;

	      incr+=Gamma5(v_off,w_off,((complex<double> *)src)[x_3d]);
	    }
	    *mf_off += incr;
	    if(incr.real()!=0.0 || incr.imag()!=0.0) printf("i %d j %d x %d incr %f %f,  mf_sl = %f %f\n",j,i,x,incr.real(),incr.imag(),mf_off->real(),mf_off->imag()); //yes j and i are swapped; this is because despite the way this code is written the i and j in the code are actually the *column* and *row* indices resp. For comparison to my code I call the row index i and the column index j
	  }
      }
  cout_time("Inner product of light v and strange w ends!");

  sum_double_array((Float *)mf_sl,nvec[0]*(nl[1]+sc_size*nhits[1])*t_size*2);
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

  const int X = (GJP.Gparity1fX() ? x_size/2 : x_size);
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
  int glb_dim[3] = {
    GJP.XnodeSites() * GJP.Xnodes(),
    GJP.YnodeSites() * GJP.Ynodes(),
    GJP.ZnodeSites() * GJP.Znodes(),
  };
  //For 1f G-parity it is assumed that the prepare_src method will place a second copy of this source on the flavour-1 side 
  if(GJP.Gparity1fX()) glb_dim[0]/=2;
  if(GJP.Gparity1fY()) glb_dim[1]/=2;


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
  int fft_dim[3] = {z_size, y_size, x_size};
  const int size_3d_glb = fft_dim[0] * fft_dim[1] * fft_dim[2];

  if(GJP.Gparity1fY()) ERR.General("MesonField","prepare_src(...)","Not implemented for Gparity 1f in Y-directionn");

  fftw_complex *fft_mem;

  if(!GJP.Gparity1fX()){
    fft_mem = fftw_alloc_complex(size_3d_glb);

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

    //printf("MesonField::prepare_src standard mode\n");
    //for(int i=0;i<size_3d_glb;i++) printf("Source %d: %f %f\n",i,fft_mem[i][0],fft_mem[i][1]);

    fftw_execute(plan_src);
    fftw_destroy_plan(plan_src);
  }else{
    //1-flavour (double lattice) G-parity
    //Place a copy of the same source on the first and second halves of the lattice
    fftw_complex* fft_mem_half = fftw_alloc_complex(size_3d_glb/2); //one half-volume field

    fft_dim[2]/=2;
    fftw_plan plan_src = fftw_plan_many_dft(3, fft_dim, 1,
					    fft_mem_half, NULL, 1, size_3d_glb/2,
					    fft_mem_half, NULL, 1, size_3d_glb/2,
					    FFTW_FORWARD, FFTW_ESTIMATE);
    switch(kind)
      {
      case 1: set_expsrc(fft_mem_half, rad); break;
      case 2: set_boxsrc(fft_mem_half, int(rad)); break;
      default: VRB.Result(cname, fname, "Src kind not implemented yet!"); exit(1);
      }

    fftw_execute(plan_src);
    fftw_destroy_plan(plan_src);
    fft_dim[2]*=2;

    //allocate a full volume and copy the result of the FFT onto the first and second halves in the x-direction
    fft_mem = fftw_alloc_complex(size_3d_glb); //one half-volume field
    for(int z=0;z<z_size;++z)
      for(int y=0;y<y_size;++y)
	for(int x=0;x<x_size/2;++x){
	  int off0 = x+x_size*(y+y_size*z);
	  int off1 = x+x_size/2 + x_size*(y+y_size*z);

	  int off_half = x + x_size/2*(y+y_size*z);
	  fft_mem[off0][0] = fft_mem_half[off_half][0];
	  fft_mem[off0][1] = fft_mem_half[off_half][1];

	  fft_mem[off1][0] = fft_mem_half[off_half][0];
	  fft_mem[off1][1] = fft_mem_half[off_half][1];
	}
    fftw_free(fft_mem_half);
  }

  src_glb2lcl(fft_mem,src);
  fftw_free(fft_mem);
}

void MesonField::src_glb2lcl(fftw_complex *glb, fftw_complex *lcl)
{
  //CK: Convert a time-slice local source to global or the inverse. Does not need modification for G-parity (cf. prepare_src)
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

    //printf("MesonField::src_glb2lcl  lcl %d  glb %d\n",xyz,xyz_glb);

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


void MesonField::run_kaon(complex<double> *kaoncorr){ //writes to array (size should be Lt complex numbers)
  ((Vector *)kaoncorr)->VecZero(t_size*2);
  int t_dis;
  int offset_i;
  int offset_j;

  //Note: mf_sl and mf_ls have the form   \sum_k  w*_i(k) \gamma^5 v_j (k)      where 'k' is a three-vector index in the range 0 -> spatial-volume. 
  //'j' is the mode index of v, running from 0 -> nl + nhits * Lt * sc_size / width
  //'i' is the mode index of w, running from 0 -> nl + nhits * sc_size
  //Each 'mf', which has Lt complex numbers per combination of i,j. The mapping to a complex number array offset is:
  //t + Lt*i + Lt*nvec*j   (global t)
  
  //For G-parity the mesonfield is also a 2x2 matrix in flavour space:   M_{f, g}. The mapping to a complex number array offset is:
  //t + Lt*f + 2*Lt*g + 4*Lt*i + 4*Lt*nvec*j

  for(int tsrc=0;tsrc<t_size;tsrc++)
    for(int tsnk=0;tsnk<t_size;tsnk++) {
      t_dis = (tsnk-tsrc+t_size)%t_size;
      for(int j=0;j<nl[1]+nhits[1]*12;j++)
	for(int i=0;i<nl[0]+nhits[0]*12;i++){	    
	  offset_i = i<nl[0]?i:(nl[0]+(i-nl[0])/12*nbase[0]+tsnk/src_width[0]*12+(i-nl[0])%12); //Convert  0 <= i <  nl[0]+nhits[0]*12   to   0 <= i' < nl[0] + nhits[0] * Lt * sc_size / width[0]
	  offset_j = j<nl[1]?j:(nl[1]+(j-nl[1])/12*nbase[1]+tsrc/src_width[1]*12+(j-nl[1])%12);

	  complex<double> l = ((complex<double> *)mf_sl)[tsrc + t_size * (j*nvec[0] + offset_i)];
	  complex<double> r = ((complex<double> *)mf_ls)[tsnk + t_size * (i*nvec[1] + offset_j)];
	  
	  printf("mf tsrc %d tsnk %d tdis %d i %d j %d  l = (%f %f)  r = (%f %f)  contr = (%f %f) -> ",tsrc,tsnk,t_dis,j,i, l.real(), l.imag(), r.real(), r.imag(), kaoncorr[t_dis].real(), kaoncorr[t_dis].imag());	  
	  kaoncorr[t_dis] += l*r;
	  printf("(%f %f)\n",kaoncorr[t_dis].real(), kaoncorr[t_dis].imag());

	  //kaoncorr[t_dis] += ((complex<double> *)mf_sl)[tsrc + t_size * (offset_i + nvec[0] * j)] * ((complex<double> *)mf_ls)[tsnk + t_size * (offset_j + nvec[1] * i)];
	}
    }
}


  // if(into.threadType() == CorrelationFunction::UNTHREADED){
  //   for(int tsrc=0;tsrc<t_size;++tsrc)
  //     for(int tsnk=0;tsnk<t_size;++tsnk){
  // 	int t_dis = (tsnk-tsrc+t_size)% t_size;
  // 	for(int i=0;i<left.nl[0]+left.nhits[0]*12;++i)
  // 	  for(int j=0;j<left.nl[1]+left.nhits[1]*12;++j){
  // 	    cnum l = left(i,j,tsnk);
  // 	    cnum r = right(j,i,tsrc);
  // 	    printf("mf2 tsrc %d tsnk %d tdis %d i %d j %d  l = (%f %f)  r = (%f %f)  contr = (%f %f) + (%f %f) = ",tsrc,tsnk,t_dis,i,j, l.real(), l.imag(), r.real(), r.imag(), into(contraction_idx,t_dis).real(), into(contraction_idx,t_dis).imag());

  // 	    into(contraction_idx,t_dis) += l*r;
  // 	    printf("(%f %f)\n",into(contraction_idx,t_dis).real(), into(contraction_idx,t_dis).imag());

  // 	    //into(contraction_idx,t_dis) += left(i,j,tsnk)*right(j,i,tsrc);
  // 	  }
  //     }


void MesonField::run_kaon(double rad) 
{
  char *fname = "run_kaon()";

  char fn[1024];
  sprintf(fn,"%s_kaoncorr_%1.2f",common_arg->filename,rad);
  complex<double> *kaoncorr = (complex<double> *)smalloc(cname,fname,"kaoncorr",sizeof(complex<double>)*t_size);
  
  run_kaon(kaoncorr);

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
  sum_double_array((Float *)type1,t_size*8*2);
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
  sum_double_array((Float *)type1,t_size*8*2);
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
  sum_double_array((Float *)type1,t_size*8*2);
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
  sum_double_array((Float *)type2,t_size*8*2);
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
  sum_double_array((Float *)type2,t_size*8*2);
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
  sum_double_array((Float *)type2,t_size*8*2);
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
  sum_double_array((Float *)type3,t_size*16*2);
  sum_double_array((Float *)type3S5D,t_size*2);
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
  sum_double_array((Float *)type3,t_size*16*2);
  sum_double_array((Float *)type3S5D,t_size*2);
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
  sum_double_array((Float *)type4,t_size*t_size*16*2);
  sum_double_array((Float *)type4_2,t_size*t_size*16*2);
  sum_double_array((Float *)type4S5D,t_size*t_size*2);
  sum_double_array((Float *)type4S5D_2,t_size*t_size*2);

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
  sum_double_array((Float *)type4,t_size*t_size*16*2);
  sum_double_array((Float *)type4S5D,t_size*t_size*2);
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





void MFBasicSource::set_expsrc(fftw_complex *tmpsrc, const Float &radius)
{
  int x_size = GJP.Xnodes()*GJP.XnodeSites();
  int y_size = GJP.Ynodes()*GJP.YnodeSites();
  int z_size = GJP.Znodes()*GJP.ZnodeSites();

  const int X = (GJP.Gparity1fX() ? x_size/2 : x_size);
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
	tmpsrc[off][0] = exp(-v) / (X * Y * Z);
	tmpsrc[off][1] = 0.;
      }
    }
  }
}

void MFBasicSource::set_boxsrc(fftw_complex *tmpsrc, const int &size)
{
  int glb_dim[3] = {
    GJP.XnodeSites() * GJP.Xnodes(),
    GJP.YnodeSites() * GJP.Ynodes(),
    GJP.ZnodeSites() * GJP.Znodes(),
  };
  //For 1f G-parity it is assumed that the prepare_src method will place a second copy of this source on the flavour-1 side 
  if(GJP.Gparity1fX()) glb_dim[0]/=2;
  if(GJP.Gparity1fY()) glb_dim[1]/=2;


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
	  tmpsrc[offset][0] = 1. / glb_size_3d;
	}else {
	  tmpsrc[offset][0] = 0.;
	}
	tmpsrc[offset][1] = 0.;
      }
    }
  }
}


void MFsource::src_glb2lcl(const fftw_complex *glb, fftw_complex *lcl)
{
  //CK: Convert a time-slice local source to global or the inverse. Does not need modification for G-parity (cf. prepare_src)
  int x_size = GJP.XnodeSites() * GJP.Xnodes(),  y_size = GJP.YnodeSites() * GJP.Ynodes(),  z_size =  GJP.ZnodeSites() * GJP.Znodes();

  const int shift[3] = { GJP.XnodeSites() * GJP.XnodeCoor(), GJP.YnodeSites() * GJP.YnodeCoor() , GJP.ZnodeSites() * GJP.ZnodeCoor() };
  int size_4d = GJP.VolNodeSites();

  for(int xyz = 0; xyz < size_4d / GJP.TnodeSites(); xyz++) {
    int tmp = xyz;
    int x = tmp % GJP.XnodeSites() + shift[0]; tmp /= GJP.XnodeSites();
    int y = tmp % GJP.YnodeSites() + shift[1]; tmp /= GJP.YnodeSites();
    int z = tmp + shift[2];
    int xyz_glb = x + x_size * (y + y_size * z);

    //printf("MFsource::src_glb2lcl  lcl %d  glb %d\n",xyz,xyz_glb);

    lcl[xyz][0] = glb[xyz_glb][0];
    lcl[xyz][1] = glb[xyz_glb][1];
  }
}

//Generate the source Fourier transform
void MFsource::fft_src(){
  int x_size = GJP.XnodeSites()*GJP.Xnodes(), y_size = GJP.YnodeSites()*GJP.Ynodes(), z_size = GJP.ZnodeSites()*GJP.Znodes();
  
  int fft_dim[3] = {z_size, y_size, x_size};
  const int size_3d_glb = fft_dim[0] * fft_dim[1] * fft_dim[2];

  if(GJP.Gparity1fY()) ERR.General("MFsource","fft_src()","Not implemented for Gparity 1f in Y-directionn");

  allocate(); //allocate the source field if not done so already

  fftw_complex *fft_mem;

  if(!GJP.Gparity1fX()){
    //Generate the source in position space
    fft_mem = fftw_alloc_complex(size_3d_glb);

    //FFT the source
    fftw_plan plan_src = fftw_plan_many_dft(3, fft_dim, 1,
					    fft_mem, NULL, 1, size_3d_glb,
					    fft_mem, NULL, 1, size_3d_glb,
					    FFTW_FORWARD, FFTW_ESTIMATE);
    set_source(fft_mem);

    //printf("MFsource::fft_src standard mode\n");
    //for(int i=0;i<size_3d_glb;i++) printf("Source %d: %f %f\n",i,fft_mem[i][0],fft_mem[i][1]);

    fftw_execute(plan_src);
    fftw_destroy_plan(plan_src);
  }else{
    //1-flavour (double lattice) G-parity
    //Place a copy of the same source on the first and second halves of the lattice
    fftw_complex* fft_mem_half = fftw_alloc_complex(size_3d_glb/2); //one half-volume field
    set_source(fft_mem_half);

    fft_dim[2]/=2;
    fftw_plan plan_src = fftw_plan_many_dft(3, fft_dim, 1,
					    fft_mem_half, NULL, 1, size_3d_glb/2,
					    fft_mem_half, NULL, 1, size_3d_glb/2,
					    FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(plan_src);
    fftw_destroy_plan(plan_src);
    fft_dim[2]*=2;

    //allocate a full volume and copy the result of the FFT onto the first and second halves in the x-direction
    fft_mem = fftw_alloc_complex(size_3d_glb); //one half-volume field
    for(int z=0;z<z_size;++z)
      for(int y=0;y<y_size;++y)
	for(int x=0;x<x_size/2;++x){
	  int off0 = x+x_size*(y+y_size*z);
	  int off1 = x+x_size/2 + x_size*(y+y_size*z);

	  int off_half = x + x_size/2*(y+y_size*z);
	  fft_mem[off0][0] = fft_mem_half[off_half][0];
	  fft_mem[off0][1] = fft_mem_half[off_half][1];

	  fft_mem[off1][0] = fft_mem_half[off_half][0];
	  fft_mem[off1][1] = fft_mem_half[off_half][1];
	}
    fftw_free(fft_mem_half);
  }

  src_glb2lcl(fft_mem,src);
  fftw_free(fft_mem);
}



void MFqdpMatrix::set_matrix(const int &qdp_spin_idx, const FlavorMatrixType &flav_mat){ 
  int n[4];
  int rem = qdp_spin_idx; 
  for(int i=0;i<4;i++){ n[i] = rem % 2; rem /= 2; }

  scf.unit();
  //Set the spin structure
  for(int i=0;i<4;i++) if(n[i]) scf.gr(i);

  //Set the flavour structure
  scf.pr(flav_mat);
}

//Any 4x4 complex matrix can be represented as a linear combination of the 16 Gamma matrices (here in QDP order cf. below), and likewise any 2x2 complex matrix is a linear combination of Pauli matrices and the unit matrix (index 0)
void MFqdpMatrix::set_matrix(const Float gamma_matrix_linear_comb[16], const Float pauli_matrix_linear_comb[4]){
  scf = 0.0;
  for(int g=0;g<16;g++){
    if(gamma_matrix_linear_comb[g]==0.0) continue;
    SpinColorFlavorMatrix gamma; 
    gamma.unit();
    gamma *= gamma_matrix_linear_comb[g];
    int rem = g; 
    for(int i=0;i<4;i++){ 
      int n = rem % 2; rem /= 2; 
      if(n) gamma.gr(i);
    }
    scf += gamma;
  }
  if(GJP.Gparity() || GJP.Gparity1fX()){
    SpinColorFlavorMatrix pcomb = 0.0;
    const static FlavorMatrixType pmats[4] = {sigma0, sigma1, sigma2, sigma3};
    
    for(int p=0;p<4;p++){
      if(pauli_matrix_linear_comb[p]==0.0) continue;
      SpinColorFlavorMatrix pauli; 
      pauli.unit();
      pauli *= pauli_matrix_linear_comb[p];
      if(p>0) pauli.pr(pmats[p]);
      pcomb += pauli;
    }
    scf *= pcomb;
  }
}


//\Gamma(n) = \gamma_1^n1 \gamma_2^n2  \gamma_3^n3 \gamma_4^n4    where ni are bit fields: n4 n3 n2 n1 
MFqdpMatrix::cnum MFqdpMatrix::contract_internal_indices(const cnum* left[2], const cnum* right[2]) const{
  cnum ret(0,0);
  for(int f=0;f<2;++f) 
    for(int g=0;g<2;++g) 
      for(int i=0;i<4;++i) 
	for(int j=0;j<4;++j) 
	  for(int a=0;a<3;++a) 
	    for(int b=0;b<3;++b)
	      ret += (this->conj_left ? conj(left[f][3*i+a]) : left[f][3*i+a]) * scf(i,a,f,j,b,g) * (this->conj_right ? conj(right[g][3*j+b]) : right[g][3*j+b]);  
  return ret;
}

#if 1
MFqdpMatrix::cnum MFqdpMatrix::contract_internal_indices(const cnum* left, const cnum* right) const{
  cnum ret(0,0);
  for(int i=0;i<4;++i) 
    for(int j=0;j<4;++j) 
      for(int a=0;a<3;++a) 
	for(int b=0;b<3;++b)
	  ret += (this->conj_left ? conj(left[3*i+a]) : left[3*i+a]) * scf(i,a,0,j,b,0) * (this->conj_right ? conj(right[3*j+b]) : right[3*j+b]);
  
  return ret;
}
#else
//TEST
std::complex<double> MFqdpMatrix::contract_internal_indices(const std::complex<double> *w, const std::complex<double> *v) const{
  printf("GAMMA 5!\n");
  std::complex<double> ret(0,0);

  int half_sc = 6;
  for(int i=0;i<half_sc;i++)
    ret += v[i]*conj(w[i]);
  for(int i=half_sc;i<2*half_sc;i++)
    ret -= v[i]*conj(w[i]);
  return ret;
}
#endif


void MesonField2::construct(A2APropbfm &left, A2APropbfm &right, const MFstructure &structure, const MFsource &source){
  const char *fname = "construct(...)";

  //Assign stored parameters etc
  n_flav = (GJP.Gparity() ? 2 : 1);
  int t_size = GJP.Tnodes()*GJP.TnodeSites();
  int sc_size = 12;
  int size_4d = GJP.VolNodeSites();
  int size_3d = size_4d / GJP.TnodeSites();

  if(left.get_args().dilute_flavor != right.get_args().dilute_flavor) ERR.General("MesonField2",fname,"One propagator is flavor diluted, the other isn't\n");

  //Ensure the FFT vectors have been computed, if not make it so
  if(!left.fft_vw_computed()) left.fft_vw();
  if(&right!=&left && !right.fft_vw_computed()) right.fft_vw();

  dilute_flavor = left.get_args().dilute_flavor;
  if( (GJP.Gparity()||GJP.Gparity1fX()) && dilute_flavor) dilute_size = 24;

  nvec[0] = left.get_nvec(); nvec[1] = right.get_nvec(); 
  nl[0] = left.get_nl(); nl[1] = right.get_nl();
  nhits[0] = left.get_nhits(); nhits[1] = right.get_nhits();
  src_width[0] = left.get_src_width();   src_width[1] = right.get_src_width(); 
  nbase[0] = t_size * dilute_size / src_width[0];    nbase[1] = t_size * dilute_size / src_width[1];
  form[0] = structure.left_vector(); form[1] = structure.right_vector();
  conj[0] = structure.cconj_left(); conj[1] = structure.cconj_right(); 

  size[0] = (form[0] == MFstructure::V) ? nvec[0] :  nl[0]+nhits[0]*dilute_size;
  size[1] = (form[1] == MFstructure::V) ? nvec[1] :  nl[1]+nhits[1]*dilute_size;

  //Allocate mf if not done so already
  int mf_size = size[0] * size[1] * t_size * 2;

  if(mf == NULL) mf = (Float*)pmalloc( mf_size * sizeof(Float) );

  //Zero mf
  ((Vector*)mf)->VecZero( mf_size ); //despite being a method of cps::Vector,   VecZero actually operates on the floats and its argument is the number of floats. Crazy huh?

  cout_time("Inner product begins!");
  int nthreads = bfmarg::threads;
  omp_set_num_threads(nthreads);

  //For G-parity 1f in the X-direction we perform comms such that both flavours reside on each node with the memory layout the same as for the 2f approach
  if(GJP.Gparity1fX() && GJP.Xnodes()>1){ left.gparity_1f_fftw_comm_flav(); right.gparity_1f_fftw_comm_flav(); }

  int left_flavour_stride = (form[0] == MFstructure::V) ? left.v_flavour_stride() : left.w_flavour_stride();
  int right_flavour_stride = (form[1] == MFstructure::V) ? right.v_flavour_stride() : right.w_flavour_stride();

  printf("MesonField2::construct rows = %d cols = %d. mf size %d\n",size[0],size[1],mf_size);

#pragma omp parallel for
  for(int i = 0; i < size[0]; i++) // nvec = nl + nhits * Lt * sc_size / width   for v generated independently for each hit, source timeslice and spin-color index
    for(int j = 0; j < size[1]; j++)
      for(int x = 0; x < size_4d; x++){	
	if(GJP.Gparity1fX())
	  if(GJP.Xnodes() == 1 && x % GJP.XnodeSites() >= GJP.XnodeSites()/2) continue; //only do first half (flavour 0)
	  else if(GJP.Xnodes() > 1 && GJP.XnodeCoor() >= GJP.Xnodes()/2) continue; //yeah I know this can be moved out but 1f code doesn't need to be efficient

	int x_3d = x % size_3d;
	int t = x / size_3d;
	int glb_t = t + GJP.TnodeCoor() * GJP.TnodeSites();
	
	complex<double> *left_vec = (form[0] == MFstructure::V) ? left.get_v_fftw(i,x,0) : left.get_w_fftw(i,x,0);
	complex<double> *right_vec = (form[1] == MFstructure::V) ? right.get_v_fftw(j,x,0) : right.get_w_fftw(j,x,0);

	complex<double> *mf_to = mf_val(i,j,glb_t);
	if(GJP.Gparity() || GJP.Gparity1fX()){
	  const complex<double> *flvec[2] = {left_vec, left_vec+left_flavour_stride};
	  const complex<double> *frvec[2] = {right_vec, right_vec+right_flavour_stride};

	  //Contract spin, colour and flavour
	  //*mf_to += source(x_3d)*structure.contract_internal_indices(flvec,frvec);

	  complex<double> incr = source(x_3d)*structure.contract_internal_indices(flvec,frvec);
	  *mf_to += incr;

	  if(incr.real()!=0.0 || incr.imag()!=0.0) printf("i %d j %d x %d incr %f %f,  mf2 = %f %f\n",i,j,x,incr.real(),incr.imag(),mf_to->real(),mf_to->imag());

	}else *mf_to += source(x_3d)*structure.contract_internal_indices(left_vec,right_vec);
      }

  cout_time("Inner product ends!");

  sum_double_array((Float *)mf,mf_size);
  if(!UniqueID()) printf("Global sum done!");
}

//Form the contraction of two mesonfields, summing over mode indices and flavour
//Expects MesonField2 objects of the form W* \Gamma V   or  V \Gamma W* 
void MesonField2::contract(const MesonField2 &left, const MesonField2 &right, const int &contraction_idx, CorrelationFunction &into){
  //Check the contraction makes sense
  if( !parameter_match(left,right,Right,Left) || !parameter_match(left,right,Left,Right) )
    ERR.General("MesonField2","contract(..)","Field parameters do not match\n");
  int cform_l = con_form(left),  cform_r = con_form(right);
  if( cform_l != cform_r  || cform_l == 0)
    ERR.General("MesonField2","contract(..)","MesonFields must be both of W*V or VW* form\n");

  const int &cform = cform_l; //1 if W*V form, -1 if VW* form

  into.setGlobalSumOnWrite(false);

  int t_size = GJP.Tnodes()*GJP.TnodeSites();

  static const int NA = -1; //non-applicable!

  if(into.threadType() == CorrelationFunction::UNTHREADED){
    for(int tsrc=0;tsrc<t_size;++tsrc)
      for(int tsnk=0;tsnk<t_size;++tsnk){
	int t_dis = (tsnk-tsrc+t_size)% t_size;
	for(int i=0;i<left.nl[0]+left.nhits[0]*left.dilute_size;++i)
	  for(int j=0;j<left.nl[1]+left.nhits[1]*left.dilute_size;++j){
	    //G(tsrc,tsnk) A G(tsnk,tsrc) B = A V_j(tsrc,tsnk)W_j*(tsnk) B V_i(tsnk,tsrc)W_i*(tsrc) =  [W_i*(tsrc) A V_j(tsrc,tsnk)] [W_j*(tsnk) B V_i(tsnk,tsrc)]
	    //                                                                              =  [V_j(tsrc,tsnk) A^T W_i*(tsrc)] [V_i(tsnk,tsrc) B^T W_j*(tsnk)]  (can now rename indices i<->j as done below)
	    const cnum &l = (cform == 1) ? left(i,j,tsrc,NA,tsnk) : left(i,j,tsrc,tsnk,NA);
	    const cnum &r = (cform == 1) ? right(j,i,tsnk,NA,tsrc) : right(j,i,tsnk,tsrc,NA);
	    into(contraction_idx,t_dis) += l*r;
	  }
      }
  }else{
    int n_threads = bfmarg::threads;
    omp_set_num_threads(n_threads);
    int nmodes[2] = {left.nl[0]+left.nhits[0]*left.dilute_size , left.nl[1]+left.nhits[1]*left.dilute_size };

#pragma omp parallel for 
    for(int r=0;r<t_size*t_size*nmodes[0]*nmodes[1];++r){
      //r = j + nmodes[1]*(i + nmodes[0]*(tsnk + t_size*tsrc))

      int rem = r;
      int j=rem % nmodes[1]; rem/=nmodes[1];
      int i=rem % nmodes[0]; rem/=nmodes[0];
      int tsnk = rem % t_size;
      int tsrc = rem / t_size;
      int t_dis = (tsnk-tsrc+t_size)% t_size;

      const cnum &l = (cform == 1) ? left(i,j,tsrc,NA,tsnk) : left(i,j,tsrc,tsnk,NA);
      const cnum &r = (cform == 1) ? right(j,i,tsnk,NA,tsrc) : right(j,i,tsnk,tsrc,NA);
      into(omp_get_thread_num(),contraction_idx,t_dis) += l*r;
    }
    into.sumThreads();
  }
}

void MesonField2::contract_specify_tsrc(const MesonField2 &left, const MesonField2 &right, const int &contraction_idx, const int &tsrc, CorrelationFunction &into){
  if( !parameter_match(left,right,Right,Left) || !parameter_match(left,right,Left,Right) )
    ERR.General("MesonField2","contract_specify_tsrc(..)","Field parameters do not match\n");
  int cform_l = con_form(left),  cform_r = con_form(right);
  if( cform_l != cform_r  || cform_l == 0)
    ERR.General("MesonField2","contract_specify_tsrc(..)","MesonFields must be both of W*V or VW* form\n");

  const int &cform = cform_l; //1 if W*V form, -1 if VW* form

  into.setGlobalSumOnWrite(false);

  int t_size = GJP.Tnodes()*GJP.TnodeSites();

  static const int NA = -1; //non-applicable!

  if(into.threadType() == CorrelationFunction::UNTHREADED){
    for(int tsnk=0;tsnk<t_size;++tsnk)
      for(int i=0;i<left.nl[0]+left.nhits[0]*left.dilute_size;++i)
	for(int j=0;j<left.nl[1]+left.nhits[1]*left.dilute_size;++j){
	  int t_dis = (tsnk-tsrc+t_size)% t_size;
	  const cnum &l = (cform == 1) ? left(i,j,tsrc,NA,tsnk) : left(i,j,tsrc,tsnk,NA);
	  const cnum &r = (cform == 1) ? right(j,i,tsnk,NA,tsrc) : right(j,i,tsnk,tsrc,NA);
	  into(contraction_idx,t_dis) += l*r;
	}
  }else{
    int n_threads = bfmarg::threads;
    omp_set_num_threads(n_threads);
    int nmodes[2] = {left.nl[0]+left.nhits[0]*left.dilute_size , left.nl[1]+left.nhits[1]*left.dilute_size };

#pragma omp parallel for 
    for(int r=0;r<nmodes[0]*nmodes[1]*t_size;++r){
      //r = j + nmodes[1]*(i+nmodes[0]*tsnk)
      int rem = r;
      int j=rem % nmodes[1]; rem/=nmodes[1];
      int i=rem % nmodes[0]; rem/=nmodes[0];
      int tsnk = rem;

      int t_dis = (tsnk-tsrc+t_size)% t_size;

      const cnum &l = (cform == 1) ? left(i,j,tsrc,NA,tsnk) : left(i,j,tsrc,tsnk,NA);
      const cnum &r = (cform == 1) ? right(j,i,tsnk,NA,tsrc) : right(j,i,tsnk,tsrc,NA);
      into(omp_get_thread_num(),contraction_idx,t_dis) += l*r;
    }
    into.sumThreads();
  }
}





CPS_END_NAMESPACE
