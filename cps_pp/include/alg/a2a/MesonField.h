CPS_START_NAMESPACE
#ifndef _MESON_FIELD_H_
#define _MESON_FIELD_H_
CPS_END_NAMESPACE

#include <util/rcomplex.h>
#include <util/vector.h>
#include <util/error.h>
#include "alg_a2a.h"
#include <alg/alg_fix_gauge.h>
#include <alg/alg_base.h>
#include <alg/wilson_matrix.h>
#include <alg/correlationfunction.h>
#include <util/spincolorflavormatrix.h>

#include <vector>
#include <fftw3.h>

CPS_START_NAMESPACE

enum QUARK {
  LIGHT=0,
  STRANGE=1
};
class MesonField {
public:
  MesonField(Lattice &, A2APropbfm *, AlgFixGauge *, CommonArg *);
  MesonField(Lattice &, A2APropbfm *, A2APropbfm *, AlgFixGauge *, CommonArg *);
  ~MesonField();
  void cal_mf_ww(double,int,QUARK,QUARK);
  void cal_mf_ll(double,int);
  void cal_mf_sl(double,int);
  void cal_mf_ls(double,int);
  void save_mf(char *);
  void save_mf_sl(char *);
  void save_mf_ls(char *);
  void run_pipi(int);
  void run_kaon(double rad); //saves to a file
  void run_kaon(complex<double> *kaoncorr); //writes to array (size should be Lt complex numbers)

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
  void gf_vec(Vector *, Vector *);
  void prepare_vw();
  void allocate_vw_fftw();
  void free_vw_fftw();
private:
  Lattice &lat;
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
  void wh2w(Vector *w, Vector *wh, int hitid, int sc);
  void set_expsrc(fftw_complex *, Float);
  void set_boxsrc(fftw_complex *, int);
  void prepare_src(double, int);
  void src_glb2lcl(fftw_complex *, fftw_complex *);
  complex<double> Gamma5(complex<double> *, complex<double> *, complex<double>);
  complex<double> Unit(complex<double> *, complex<double> *, complex<double>);
  void mf_contraction(QUARK left, QUARK middle, QUARK right, complex<double> *left_mf, int t_sep, complex<double> *right_mf, complex<double> *result); 
  void mf_contraction_ww(QUARK left, QUARK middle, QUARK right, complex<double> *left_mf, int t_sep, complex<double> *right_mf, complex<double> *result); 
  // t_sep is t_left-t_right
  void show_wilson(const WilsonMatrix &);
  void writeCorr(complex<double> *, char *, int, int);
  WilsonMatrix Build_Wilson(Vector *, int, int, QUARK, QUARK);
  WilsonMatrix Build_Wilson_ww(Vector *, int, int, int, QUARK, QUARK);
  WilsonMatrix Build_Wilson_loop(int, QUARK);
  A2APropbfm *a2a_prop;
  A2APropbfm *a2a_prop_s;
  AlgFixGauge *fix_gauge;
  CommonArg *common_arg;
  Vector **v_fftw;
  Vector **wl_fftw[2];
  Vector **v_s_fftw;
  Vector *mf;
  Vector *mf_ls;// light w dot product strange v
  Vector *mf_sl;// strange w dot product light v
  Vector *mf_ww[3]; // 0: light w light w; 1: light w strange w; 2: strange w light w;
  Vector *wh_fftw[2];
  Vector *wh_s_fftw;

  //1-flavour G-parity (double lattice)
  void cal_mf_ll_gp1fx(double rad, int src_kind);

  friend class MesonFieldTesting;
  friend void Gparity_1f_FFT(fftw_complex *fft_mem);
};

//CK: A class that represents a meson-field source; a 3d spatial function Fourier transformed into momentum space. 
//This class can be used with one of the standard forms (box, exponential) or a derived class can be created to build the source explicitly
//Note: for G-parity boundary conditions the same source is used for the two flavours so no modifications to the source generation code are necessary
class MFsource{
  friend class MesonFieldTesting;

protected:
  fftw_complex *src;
  static void src_glb2lcl(const fftw_complex* glb, fftw_complex *lcl);

public:

  inline void allocate(){   //automatically called from fft_src if not previously called manually
    if(src == NULL) src = fftw_alloc_complex(GJP.XnodeSites()*GJP.YnodeSites()*GJP.ZnodeSites());
  }  
  inline void free(){   
    if(src != NULL){ fftw_free(src); src = NULL; }
  }  
  void fft_src(); //Generate the source Fourier transform. Unless called by a derived class constructor, this function must be called manually before using the source in a MesonField2 contraction

  //Overridable method to set the 3d source layout in position space. Default is to setup one of the basic types according to the previously set variables basic_src_radius and basic_src_type
  //THIS FUNCTION IS CALLED BY fft_src
  virtual void set_source(fftw_complex *src_3d) = 0;

  inline const std::complex<Float> & operator()(const int &x_3d) const{ return ((std::complex<Float> *)src)[x_3d]; }

  MFsource(): src(NULL){}

  virtual ~MFsource(){ free(); }
};
//Basic source types
class MFBasicSource: public MFsource{
public:
  enum SourceType { BoxSource=0, ExponentialSource=1 };

private:
  static void set_expsrc(fftw_complex *tmpsrc, const Float &radius);
  static void set_boxsrc(fftw_complex *tmpsrc, const int &size);

  double basic_src_radius;
  SourceType basic_src_type;

public:  
  MFBasicSource(): MFsource(){}
  MFBasicSource(const SourceType &type, const double &radius): MFsource(), basic_src_type(type), basic_src_radius(radius){ this->fft_src(); }  

  inline void set_source_type_and_radius(const SourceType &type, const double &radius){ basic_src_type = type; basic_src_radius = radius; }

  //Set the 3d source layout in position space to one of the basic types according to the previously set variables basic_src_radius and basic_src_type
  void set_source(fftw_complex *src_3d){
    if(basic_src_type == BoxSource) return set_boxsrc(src_3d,int(basic_src_radius));
    else if(basic_src_type == ExponentialSource) return set_expsrc(src_3d,basic_src_radius);
    else ERR.General("MFBasicsource","set_source(..)","Invalid source type\n");
  }
};



//CK: A class that specifies the left and right fields used as well as the internal contraction of spin,colour and flavour indices within the meson field
class MFstructure{
public:
  enum VorW { V, W };
private:
  VorW left_v;
  VorW right_v;
protected:
  //Specify whether the left/right vectors are complex conjugated during the contraction
  bool conj_left;
  bool conj_right;

public:
  typedef std::complex<Float> cnum;  
  
  //Set the form of the meson fields : v,w  w,v etc 
  void set_form(const VorW &leftv, const VorW &rightv, const bool &_conj_left, const bool &_conj_right){ left_v = leftv; right_v = rightv; conj_left = _conj_left; conj_right = _conj_right; }
  
  const VorW & left_vector() const{ return left_v; }
  const VorW & right_vector() const{ return right_v; }

  const bool & cconj_left() const{ return conj_left; }
  const bool & cconj_right() const{ return conj_right; }

  //Perform the internal contractions with standard (non-G-parity) BCs. conj_left and conj_right specify whether the cnums should be complex conjugated
  //when the product is formed.
  virtual cnum contract_internal_indices(const cnum* left, const cnum* right) const = 0;

  //Perform the internal contraction with G-parity BCs active. The two cnums provided for the left and right vectors are the two flavours.
  virtual cnum contract_internal_indices(const cnum* left[2], const cnum* right[2]) const = 0;

  virtual ~MFstructure(){}
};
//Contract the internal spin/flavour indices with spin-matrices in the QDP index format
class MFqdpMatrix: public MFstructure{
  SpinColorFlavorMatrix scf;

public:
  typedef std::complex<Float> cnum;

  MFqdpMatrix(): MFstructure(){}
  MFqdpMatrix(const VorW &leftv, const VorW &rightv, const bool &_conj_left, const bool &_conj_right, const int &qdp_spin_idx, const FlavorMatrixType &flav_mat = sigma0): MFstructure(){
    set_form(leftv,rightv,_conj_left,_conj_right); set_matrix(qdp_spin_idx,flav_mat);
  }

  void set_matrix(const int &qdp_spin_idx, const FlavorMatrixType &flav_mat = sigma0);

  //\Gamma(n) = \gamma_1^n1 \gamma_2^n2  \gamma_3^n3 \gamma_4^n4    where ni are bit fields: n4 n3 n2 n1 
  cnum contract_internal_indices(const cnum* left[2], const cnum* right[2]) const;
  cnum contract_internal_indices(const cnum* left, const cnum* right) const;
};

//CK: A class that represents a single meson field, with convenience functions for contraction and element access. Based off Daiqian's code
class MesonField2{
  Float *mf;
  int nvec[2];
  int nl[2];
  int nhits[2];
  int src_width[2];
  int nbase[2];

  enum LeftOrRight { Left=0, Right=1 };
  MFstructure::VorW form[2]; //Gives the types (V or W) of the left and right fields of the meson field comprises

  bool conj[2];

  int size[2]; //size of each vector: for V is it nvec, for W nl+nhits*sc_size

  int n_flav; //Number of flavours - 1 or 2 with G-parity

  //Convert from 0 <= i <  nl[0]+nhits[0]*12 to mode index of left or right field vector given a timeslice t: if field is v:  0 <= i' < nl[0] + nhits[0] * Lt * sc_size / width[0]   else if field is w: i' = i  
  //Note: nbase = Lt * sc_size / width,   sc_size=12
  inline int idx(const int &i, const int &t, const LeftOrRight &field) const{
    return (form[(int)field] == MFstructure::W || i < nl[(int)field]) ?   i  :  nl[(int)field] + (i-nl[(int)field])/12*nbase[(int)field] + t/src_width[(int)field]*12 + (i-nl[(int)field])%12 ; //hit_idx, t, sc_idx
    //Note: for high mode indices of v, the mapping for a mode index j (j>nl) is :    j-nl = (sc_idx + 12/src_width * t_idx +  12/src_width*Lt * hit_idx),
    //hence (i-nl)/12*nbase = hit_idx * (12/src_width*Lt)
  }

  //Check a2a propagator parameters for the fields V and W match between 2 MesonField2 instances
  inline static bool parameter_match(const MesonField2 &left, const MesonField2 &right, const LeftOrRight &field_left, const LeftOrRight &field_right){
    return left.nvec[(int)field_left] == right.nvec[(int)field_right] && left.nl[(int)field_left] == right.nl[(int)field_right] && left.nhits[(int)field_left] == right.nhits[(int)field_right] &&
      left.src_width[(int)field_left] == right.src_width[(int)field_right] && left.nbase[(int)field_left] == right.nbase[(int)field_right]; 
  }

  typedef std::complex<Float> cnum;

  //Get the pointer at index i,j where i,j run over the range 0...size[0] and 0..size[1] respectively (unlike operator() where the indices are modified to have equal sizes with an intermediate mapping)
  inline cnum* mf_val(const int &i, const int &j, const int &t){ 
    return (cnum*)mf + t + GJP.TnodeSites()*GJP.Tnodes()* (j + size[1] * i ); 
  }
  
  //return 1 if W*V form, -1 if VW* form or 0 otherwise
  static int con_form(const MesonField2 &mf){
    if(mf.form[0] == MFstructure::W && mf.form[1] == MFstructure::V && mf.conj[0] && !mf.conj[1]) return 1;
    else if(mf.form[0] == MFstructure::V && mf.form[1] == MFstructure::W && !mf.conj[0] && mf.conj[1]) return -1;
    else return 0;
  }

public:
  MesonField2(): mf(NULL){}
  MesonField2(A2APropbfm &left, A2APropbfm &right, const MFstructure &structure, const MFsource &source): mf(NULL){ construct(left,right,structure,source); }

  //Get the value of the real part of the meson field for left/right mode indices i and j (where  0 <= i,j < nl + nhits*12)
  //For a general contraction this requires 3 times specified (some of which may be the same). The first, t_mf is the time-slice
  //for the mesonfield. t_left and t_right are associated with the right and left vectors respectively; for a vector of type W this time is ignored
  //and for a vector of type V it is only relevant to the high-modes, for which it refers to the time-slice of the stochastic source

  inline const cnum& operator()(const int &i, const int &j, const int &t_mf, const int &t_left, const int &t_right) const{ 
    return *( (const cnum*)mf + t_mf + GJP.TnodeSites()*GJP.Tnodes()*( idx(j,t_right,Right) + size[1]* idx(i,t_left,Left) ) );  //column index is fastest-changing
  }
  void construct(A2APropbfm &left, A2APropbfm &right, const MFstructure &structure, const MFsource &source);

  //Form the contraction of two mesonfields, summing over mode indices:   left_ij right_ji.  For G-parity we do  left_ij,fg right_ji,gf
  static void contract(const MesonField2 &left, const MesonField2 &right, const int &contraction_idx, CorrelationFunction &into);
  static void contract(const MesonField2 &left, const MesonField2 &right, CorrelationFunction &into){ return contract(left,right,0,into); }

  friend class MesonFieldTesting;
};

#endif

CPS_END_NAMESPACE
