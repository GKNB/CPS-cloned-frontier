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

  v_fftw = NULL;
  wl_fftw = NULL;
  wh_fftw = NULL;

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

  gparity_1f_fftw_comm_flav_performed = false;
}

A2APropbfm::~A2APropbfm()
{
  const char *fname = "~A2APropbfm()";

  free_vw();
  free_vw_fftw();

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

  int f_size_4d = GJP.VolNodeSites() * SPINOR_SIZE; //Note: SPINOR_SIZE=24 seems to be hardcoded
  if(GJP.Gparity()) f_size_4d *= 2;

  v  = (Vector **)smalloc(cname, fname, "v" , sizeof(Vector *) * nvec);
  wl = (Vector **)smalloc(cname, fname, "wl", sizeof(Vector *) * a2a.nl); 

  for(int i = 0; i < nvec; ++i) {
    v[i]  = (Vector *)smalloc(cname, fname, "v[i]" , sizeof(Float) * f_size_4d);
  }
  for(int i = 0; i < a2a.nl; ++i) {
    wl[i] = (Vector *)smalloc(cname, fname, "wl[i]", sizeof(Float) * f_size_4d);
  }

  // allocate wh
  int wh_size = 2 * GJP.VolNodeSites() * nh_site; //nhits of fields of complex numbers
  if(GJP.Gparity()) wh_size *= 2;
  wh = (Vector *)smalloc(cname, fname, "wh", sizeof(Float) * wh_size);

  //CK: The mapping fo the wh vector is as follows:
  // wh [ off ]   where   off = re/im + 2 * wh_offset
  //wh_offset =
  //		wh_id * lx * ly * lz * lt
  //		+ t_lcl * lx * ly * lz
  //		+ i;
  //where :
  //'l.' are local lattice sizes
  //'wh_id' is the hit index (?) running from 0 to nh_site-1
  //'t_lcl' is a time coordinate on the local volume
  //'i' is an index running from 0 to Lx * Ly * Lz / src_width   -1,  
  //i.e. each hit occupies a four-volume of complex doubles

  //It seems that despite the src_width variable, the size of a given hit is fixed to the spatial volume (see memory allocation above)

  //For G-parity we add an extra flavour index, which we place between the time and hit indices in the mapping
  //wh_offset = wh_id * 2 * (four-vol) + flav * (four_vol) + t_lcl * (three-vol) + i
	
}


void A2APropbfm::allocate_vw_fftw(void)
{
  const char *fname = "allocate_vw_fftw()";

  int ferm_sz = ( GJP.Gparity() ? 2 : 1 ) * GJP.VolNodeSites() * SPINOR_SIZE *sizeof(Float);
  int v_ptr_sz = sizeof(Vector *);

  v_fftw = (Vector **)smalloc(cname, fname, "v_fftw", v_ptr_sz * nvec);
  for(int i = 0; i < nvec; ++i) { 
    v_fftw[i]  = (Vector *)smalloc(cname, fname, "v_fftw[i]" , ferm_sz);
  }

  wl_fftw = (Vector **)smalloc(cname, fname, "wl_fftw[i]", v_ptr_sz * a2a.nl);
  for(int i = 0; i < a2a.nl; ++i) {
    wl_fftw[i] = (Vector *)smalloc(cname, fname, "wl_fftw[i]", ferm_sz);
  }

  wh_fftw = (Vector *)smalloc(cname, fname, "wh_fftw", a2a.nhits *ferm_sz * SPINOR_SIZE/2); //separate FT of each spin-color component, hence 12 fermion vectors per hit
}

void A2APropbfm::free_vw_fftw(void)
{
  const char *fname = "free_vw_fftw()";

  if(v_fftw) {
    for(int i=0;i<nvec;i++) {
      if(v_fftw[i]) 
	sfree(cname,fname,"v_fftw[i]",v_fftw[i]);
    }
    sfree(cname,fname,"v_fftw",v_fftw);
    v_fftw=NULL;
  }
  if(wl_fftw) {
    for(int i=0;i<a2a.nl;i++) {
      if(wl_fftw[i]) sfree(cname,fname,"wl_fftw[i]",wl_fftw[i]);
    }
    sfree(cname,fname,"wl_fftw",wl_fftw);
    wl_fftw=NULL;
  }

  if(wh_fftw) {
    sfree(cname,fname,"wh_fftw",wh_fftw);
    wh_fftw=NULL;
  }
}


void A2APropbfm::gen_rand_4d_init(void)
{
  //CK: Fills wh (high mode source) with random numbers
  //For G-parity we do this for both flavours (LRG maintains different generators for the second flavour)

  //CK: wh mappings
  //Standard  (re/im=[0,1], site=[0..four_vol-1],hit=[0..nh_site]) ->    re/im + 2*( site + hit*four_vol )
  //G-parity  (re/im=[0,1], site=[0..four_vol-1],flav=[0,1],hit=[0..nh_site]) ->    re/im + 2*( site + flav*four_vol + hit*2*four_vol )
  
  const char *fname = "gen_rand_4d_init()";

  if(!wh) {
    ERR.Pointer(cname, fname, "wh");
  }

  LRG.SetInterval(1, 0);
  const int vol_node_sites = GJP.VolNodeSites();
  int sites = vol_node_sites;
  if(GJP.Gparity()) sites*=2;

  Float *f = (Float *)wh;

  const Float PI = 3.14159265358979323846;

  for(int i = 0; i < sites; ++i) {
    int flav = i / vol_node_sites;
    int st = i % vol_node_sites;

    LRG.AssignGenerator(st,flav);
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
  int f_size = GJP.VolNodeSites() * lat.FsiteSize();
  int f_size_4d = GJP.VolNodeSites() * SPINOR_SIZE;
  if(GJP.Gparity()){ f_size *= 2; f_size_4d *= 2; }

  const int glb_ls = GJP.SnodeSites() * GJP.Snodes();

  Vector *a = (Vector *)smalloc(cname, fname, "a", sizeof(Float) * f_size);
  Vector *b = (Vector *)smalloc(cname, fname, "a", sizeof(Float) * f_size);

  Fermion_t tmp[2];
  Fermion_t Mtmp;
  tmp[0] = dwf.allocFermion(); //CK: these are G-parity aware
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

//Using  wh (complex number field identical for each high mode), 
//and given a spin-color index (sc_id), a global timeslice (t_id) and a stochastic hit (wh_id) in the form of an integer:   id = (sc_id + 3*4/src_width * t_id +  3*4/src_width*Lt * wh_id)
//Generate a four-d complex field  v4d  that is zero apart from on time-slice t_id, and spin-color index sc_id, for which it takes the appropriate value from wh 
void A2APropbfm::gen_rand_4d(Vector *v4d, int id)
{
  const char *fname = "gen_rand_4d()";

  Lattice &lat = AlgLattice();
  const int t_glb = GJP.Tnodes() * GJP.TnodeSites();
  const int spin_color = lat.Colors() * lat.SpinComponents();

  if(!GJP.Gparity()){
    const int size_4d = GJP.VolNodeSites() * 2 * spin_color;
    v4d->VecZero(size_4d);

    const int wh_id = id / nh_base;
    const int t_id = id % nh_base / spin_color * a2a.src_width;
    const int sc_id = id % nh_base % spin_color;
	
    //CK: id is an integer between 0 and nhits*nh_base - 1:    sc_id + 3*4/src_width * t_id +  3*4/src_width*Lt * wh_id
    //Note: nh_base = Lt * 3 * 4 / src_width

    VRB.Result(cname, fname, "Generating random wall source %d = (%d, %d, %d).\n    ", id, wh_id, t_id, sc_id);

    if(t_id / GJP.TnodeSites() != GJP.TnodeCoor()) return;

    const int t_lcl = t_id % GJP.TnodeSites();
    const int wall_size = size_4d / GJP.TnodeSites();

    Float *vf = (Float *)v4d;
    Float *whf = (Float *)wh;
    for(int i = sc_id; i < wall_size / 2 * a2a.src_width; i += spin_color) {   //CK:  i < lx * ly * lz * 3 * 4 * src_width
      int offset = t_lcl * wall_size / 2 + i; //Offset of t_lcl wall in units of complex numbers 
      int wh_offset =
	wh_id * GJP.VolNodeSites()
	+ t_lcl * GJP.VolNodeSites() / GJP.TnodeSites()
	+ i / spin_color; //offset of hit 'wh_id', timeslice 't_lcl' and spin-color offset 'sc_id'

      vf[2 * offset    ] = whf[2 * wh_offset    ];
      vf[2 * offset + 1] = whf[2 * wh_offset + 1];
    }
  }else{
    //CK: G-parity.  Mapping of id is the same. Need to loop over flavours

    const int vol_node_sites = GJP.VolNodeSites();
    const int vec_size_4d = vol_node_sites * 2 * spin_color * 2; //2 flavours
    v4d->VecZero(vec_size_4d);

    const int wh_id = id / nh_base;
    const int t_id = id % nh_base / spin_color * a2a.src_width;
    const int sc_id = id % nh_base % spin_color;
	
    VRB.Result(cname, fname, "Generating random G-parity wall source %d = (%d, %d, %d).\n    ", id, wh_id, t_id, sc_id);

    if(t_id / GJP.TnodeSites() != GJP.TnodeCoor()) return;

    const int t_lcl = t_id % GJP.TnodeSites();
    const int wall_size = vec_size_4d / GJP.TnodeSites() / 2; // = (3-vol)*(3 col)*(4 spin)*(2 re/im)   want to fill in only a single flavour at a time

    Float *vf = (Float *)v4d;
    Float *whf = (Float *)wh;
    for(int flav = 0; flav < 2; ++flav){
      for(int i = sc_id; i < wall_size / 2 * a2a.src_width; i += spin_color) {   //CK:  i < lx * ly * lz * 3 * 4 * src_width
	//G-parity: wh_offset = wh_id * 2 * (four-vol) + flav * (four_vol) + t_lcl * (three-vol) + i/spin_color

	int offset = flav * vol_node_sites * spin_color + t_lcl * wall_size / 2 + i;   //second flavour stacked after first at offset 12*(four-vol) in units of complex number size
	int wh_offset =
	  wh_id * vol_node_sites * 2
	  + flav * vol_node_sites
	  + t_lcl * vol_node_sites / GJP.TnodeSites()
	  + i / spin_color;
	    
	//v size is  3 (col) *4 (spin) *vol_node_sites * 2 (flav)
	vf[2 * offset    ] = whf[2 * wh_offset    ];
	vf[2 * offset + 1] = whf[2 * wh_offset + 1];
      }
    }
  }
}


//The complex field w is allocated. For each hit, spin-color index and timeslice the high-mode Dirac matrix is inverted on a source that is zero apart from that spin-color and timeslice, 
//for which the values are taken from w. The result is placed in v
bool A2APropbfm::compute_vw_high(bfm_evo<double> &dwf)
{
  const char *fname = "compute_vw_high()";
  VRB.Result(cname, fname, "Start computing high modes.\n");
  Lattice &lat = AlgLattice();

  
  gen_rand_4d_init(); // Set random complex number for each site (used for every high mode)

  int f_size_4d = GJP.VolNodeSites() * 2 * lat.Colors() * lat.SpinComponents();
  int f_size = GJP.VolNodeSites() * lat.FsiteSize();
  if(GJP.Gparity()){ f_size_4d *= 2; f_size *= 2; }

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

    //Fill four-d complex field v[i] with zero apart from on time-slice t_id, and spin-color index sc_id, for which it takes the appropriate value from wh, 
    //where  i-a2a.nl = (sc_id + 3*4/src_width * t_id +  3*4/src_width*Lt * wh_id)
    gen_rand_4d(v[i], i-a2a.nl);  
    lat.Ffour2five(a, v[i], 0, glb_ls-1, 2); //convert v[i] to a 5d field a
    dwf.cps_impexFermion((Float *)a,src,1); //convert a to BFM-format 5d field 'src'

    omp_set_num_threads(bfmarg::threads);
#pragma omp parallel
    {
      dwf.CGNE_M_high(V_tmp,src,eig->bq,eval); //High mode inversion upon src
    }

    dwf.cps_impexFermion((Float *)b,V_tmp,0); //convert result to 5d field 'b'
    lat.Ffive2four(v[i], b, glb_ls-1, 0, 2); //convert back to 4d and overwrite v[i]
    v[i]->VecTimesEquFloat(1.0 / nh_site, f_size_4d); //normalise

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

Vector *A2APropbfm::get_wl_fftw(int i)
{
  const char *fname = "get_wl_fftw()";

  if(wl == NULL) {
    ERR.Pointer(cname, fname, "wl");
  }
  if(i >= a2a.nl) {
    ERR.General(cname, fname, "Array index out of range %d >= %d.\n", i, a2a.nl);
  }
  return wl_fftw[i];
}

Vector *A2APropbfm::get_v_fftw(int i)
{
  const char *fname = "get_v_fftw()";
  if(v == NULL) {
    ERR.Pointer(cname, fname, "v");
  }
  if(i >= nvec) {
    ERR.General(cname, fname, "Array index out of range %d >= %d.\n", i, nvec);
  }
  return v_fftw[i];
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

void A2APropbfm::gf_vec(Vector *out, Vector *in)
{
  const char *fname = "gf_vector(Vector *in, Vector *out)";
  Vector tmp;

  int size_4d = GJP.VolNodeSites();
  int size_3d = size_4d / GJP.TnodeSites();

  Float *fi = (Float *)in;
  Float *fo = (Float *)out;

  int nsites = (GJP.Gparity() ? 2 : 1) * size_4d;

  for(int i = 0; i < nsites; ++i) {
    const int flav = i / size_4d; 
    const int site = i % size_4d;

    const cps::Matrix* gfmat = AlgLattice().FixGaugeMatrix(site,flav);

    const int voff = SPINOR_SIZE * i; //four-d ferm vector offset in units of float
    for(int s = 0; s < 4; ++s) {
      tmp.CopyVec((Vector *)(fi + voff + 6 * s), 6);
      uDotXEqual(fo + voff + 6 * s, (Float *)gfmat, (Float *)(&tmp));
    }
  }
}

static void set_zero(fftw_complex *v, int n){
  for(int i = 0; i < n; i++) {
    v[i][0] = 0;
    v[i][1] = 0;
  }
}
static void sum_double_array(Float* data, const int &size){
#ifdef USE_QMP
  QMP_sum_double_array((double *)data, size);  
#else
  slice_sum(data,size,99);
#endif
}
//CK: Method to poke onto a particular spin-color index the complex number from wh for a given hit
//    w is an output fermion field that is zero everywhere apart from on spin-color index sc
static void wh2w(Vector *w, Vector *wh, int hitid, int sc) // In order to fft the high modes scource
{
  char *fname = "wh2w()";
	
  int hit_size = (GJP.Gparity()?2:1) * GJP.VolNodeSites();
  complex<double> *whi = (complex<double> *)wh + hitid * hit_size;
  complex<double> *t = (complex<double> *)w;

  for(int i = 0; i < hit_size; i++) {
    double *p = (double*)(t+i*SPINOR_SIZE/2);
    double *end = p + SPINOR_SIZE; 
    while(p!=end) *(p++) = 0.0;
    t[sc + i * SPINOR_SIZE/2] = whi[i]; //At spin-colour index sc poke the complex number from wh
  }
}

void A2APropbfm::cnv_lcl_glb(fftw_complex *glb, fftw_complex *lcl, bool lcl_to_glb)
{
  int glb_dim[4], lcl_dim[4], shift[4];
  for(int i=0;i<4;++i){ 
    glb_dim[i] = GJP.NodeSites(i)*GJP.Nodes(i); 
    lcl_dim[i] = GJP.NodeSites(i); 
    shift[i] = GJP.NodeSites(i) * GJP.NodeCoor(i);
  }
  int sc_size = SPINOR_SIZE/2;

  int glb_vol = ( GJP.Gparity() ? 2 : 1 ) * glb_dim[0] * glb_dim[1] * glb_dim[2] * glb_dim[3];
  if(lcl_to_glb) set_zero(glb, glb_vol * sc_size);

  int lcl_size_3d = lcl_dim[0] * lcl_dim[1] * lcl_dim[2];
  int glb_size_3d = glb_dim[0] * glb_dim[1] * glb_dim[2];

  int nflav = 1; if(GJP.Gparity()) nflav = 2;
  
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
  if(!GJP.Gparity1fX()){ if(!UniqueID()) printf("MesonField.C - Gparity_1f_FFT: Must be used only when G-parity 1f is active\n"); exit(-1); }
  else if(GJP.Gparity1fY() && !UniqueID()){ printf("MesonField.C - Gparity_1f_FFT: Not setup for 2 directions of G-parity\n"); exit(-1); }

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


  //Convert back to double lattice format
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
}

void A2APropbfm::fft_vw(){
  const char *fname = "fft_vw()";

  if(AlgLattice().FixGaugeKind() == FIX_GAUGE_NONE) ERR.General(cname,fname,"Lattice must be gauge fixed\n");
	
  fftw_init_threads();
  fftw_plan_with_nthreads(bfmarg::threads);
  const int fft_dim[3] = { GJP.ZnodeSites() * GJP.Znodes(),
			   GJP.YnodeSites() * GJP.Ynodes(),
			   GJP.XnodeSites() * GJP.Xnodes()};
  const int t_size = GJP.TnodeSites()*GJP.Tnodes();
  const int size_4d = GJP.VolNodeSites();
  const int size_3d_glb = fft_dim[0] * fft_dim[1] * fft_dim[2];
  const int sc_size = SPINOR_SIZE/2;

  int fftw_alloc_sz = ( GJP.Gparity() ? 2 : 1 ) * size_3d_glb * t_size * sc_size;
  fftw_complex *fft_mem = fftw_alloc_complex(fftw_alloc_sz);

  int t_size_alloc = ( GJP.Gparity() ? 2 : 1 ) * size_4d * SPINOR_SIZE / 2 * sizeof(complex<double>);
  Vector *t  = (Vector *)smalloc(cname, fname, "t" , t_size_alloc);
  fftw_plan plan;

  // load and process v
  for(int j = 0; j < nvec; ++j) {
    gf_vec(t, get_v(j)); 
    cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(t), true); //convert local 't' to global 'fft_mem'
    for(int sc = 0; sc < sc_size; sc++){ //loop over the 12 spin-colour indices
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
      int n_fft = ( GJP.Gparity() ? 2 : 1 ) * t_size;

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
  for(int j = 0; j < a2a.nl; ++j) {
    gf_vec(t, get_wl(j));
    cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(t), true);
    for(int sc = 0; sc < sc_size; sc++){
      int n_fft = ( GJP.Gparity() ? 2 : 1 ) * t_size;
	
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
    cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(wl_fftw[j]), false);
  }
  // load and process wh  
  //wh is 'nhits' field of c-numbers, and we generate independent v for each timeslice, spin-color and hit index
  //Here we perform the FFT for each spin-color index of w separately
  for(int j = 0; j < a2a.nhits; ++j) 
    for(int w_sc = 0; w_sc < sc_size; w_sc++) {
      //Perform separate FFT for each spin-color component
      int off_fac = ( GJP.Gparity() ? 2 : 1 );
      Float *wh_fftw_offset = (Float *)(wh_fftw) + off_fac * size_4d * sc_size * 2 * (j * sc_size + w_sc); //each hit is offset by  four_vol * 12*12 * 2
      wh2w((Vector *)wh_fftw_offset, get_wh(), j, w_sc);  //wh_fftw_offset is set to zero apart from on spin-color index w_sc, where it is set to the value from hit j of wh
      gf_vec((Vector *)wh_fftw_offset, (Vector *)wh_fftw_offset);
      cnv_lcl_glb(fft_mem, reinterpret_cast<fftw_complex *>(wh_fftw_offset), true);
      for(int sc = 0; sc < sc_size; sc++){
	int n_fft = ( GJP.Gparity() ? 2 : 1 ) * t_size;
	
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
  sfree(cname, fname, "t", t);
  fftw_free(fft_mem);
  fftw_cleanup();
  fftw_cleanup_threads();
}

static void get_other_flavour(Float* of, Float* into, const int &site_size){ //site_size is in FLOAT UNITS
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

//For 1-flavour G-parity (X-DIRECTION ONLY) with Xnodes>1, copy/communicate the fields such that copies of both flavours reside on the first and second halves of the lattice in the X direction
//Their memory layout for each half is identical to the 2f case. Contractions of v and w should be confined to the first half of the lattice with the second half contributing zero to the global sum
//(or global sum and divide by 2!)
void A2APropbfm::gparity_1f_fftw_comm_flav(){
  if(GJP.Xnodes() == 1 || !GJP.Gparity1fX() || GJP.Gparity1fY()) ERR.General("A2APropbfm","gparity_1f_fftw_comm_flav()","Should be called only for G-parity 1f in X-direction with Xnodes>1\n");
  
  if(gparity_1f_fftw_comm_flav_performed) return; //No need to do it twice!
  int size_4d = GJP.VolNodeSites();
  int sc_size = SPINOR_SIZE/2;

  int node_flav = (GJP.XnodeCoor() >= GJP.Xnodes()/2 ? 1 : 0);  //For Xnodes>1 only					     
  int ferm_sz = size_4d * sc_size * 2 * 2 * sizeof(Float); //Both flavours

  //Do v_fftw
  for(int i = 0; i < nvec; ++i){
    Vector* new_v_fftw  = (Vector *)pmalloc(ferm_sz);
    Float* local_off = node_flav == 0 ? (Float*)new_v_fftw : (Float*)new_v_fftw + size_4d * sc_size * 2;
    Float* remote_off = node_flav == 1 ? (Float*)new_v_fftw : (Float*)new_v_fftw + size_4d * sc_size * 2;
    memcpy( (void*)local_off, (void*)v_fftw[i], ferm_sz/2 );
    get_other_flavour( (Float*)v_fftw[i], remote_off, sc_size*2);

    sfree(v_fftw[i]);
    v_fftw[i] = new_v_fftw;
  }
  //Do wl_fftw
  for(int i = 0; i < a2a.nl; ++i){
    Vector* new_wl_fftw  = (Vector *)pmalloc(ferm_sz);
    Float* local_off = node_flav == 0 ? (Float*)new_wl_fftw : (Float*)new_wl_fftw + size_4d * sc_size * 2;
    Float* remote_off = node_flav == 1 ? (Float*)new_wl_fftw : (Float*)new_wl_fftw + size_4d * sc_size * 2;
    memcpy( (void*)local_off, (void*)wl_fftw[i], ferm_sz/2 );
    get_other_flavour( (Float*)wl_fftw[i], remote_off, sc_size*2);

    sfree(wl_fftw[i]);
    wl_fftw[i] = new_wl_fftw;
  }
  //Do wh_fftw
  Vector* new_wh_fftw = (Vector *)pmalloc(a2a.nhits*ferm_sz*sc_size);
  for(int h=0;h<sc_size * a2a.nhits;h++){
    Float* from = (Float*)wh_fftw + h*size_4d * sc_size *2;
    Float* base_off = (Float*)new_wh_fftw + h*size_4d * sc_size *2 * 2;
    Float* local_off = node_flav == 0 ? base_off : base_off + size_4d * sc_size *2;
    Float* remote_off = node_flav == 1 ? base_off : base_off + size_4d * sc_size *2;

    memcpy( (void*)local_off, (void*)from, ferm_sz/2 );
    get_other_flavour(from, remote_off, sc_size*2);
  }
  sfree(wh_fftw);
  wh_fftw = new_wh_fftw;

  gparity_1f_fftw_comm_flav_performed = true;
}



CPS_END_NAMESPACE
