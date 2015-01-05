CPS_START_NAMESPACE
#ifndef _MESON_FIELD_H_
#define _MESON_FIELD_H_
CPS_END_NAMESPACE

#include <util/rcomplex.h>
#include <util/vector.h>
#include <util/error.h>
#include <alg/a2a/alg_a2a.h>
#include <alg/alg_fix_gauge.h>
#include <alg/alg_base.h>
#include <alg/wilson_matrix.h>
#include <alg/correlationfunction.h>
#include <util/spincolorflavormatrix.h>
#include <alg/gparity_contract_arg.h>
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

  //Overridable method to set the 3d source layout in position space.
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
  MFBasicSource(const SourceType &type, const double &radius): MFsource(), basic_src_type(type), basic_src_radius(radius){ this->fft_src();  }  

  inline void set_source_type_and_radius(const SourceType &type, const double &radius){ basic_src_type = type; basic_src_radius = radius; }

  //Set the 3d source layout in position space to one of the basic types according to the previously set variables basic_src_radius and basic_src_type
  void set_source(fftw_complex *src_3d){
    if(basic_src_type == BoxSource) return set_boxsrc(src_3d,int(basic_src_radius));
    else if(basic_src_type == ExponentialSource) return set_expsrc(src_3d,basic_src_radius);
    else ERR.General("MFBasicsource","set_source(..)","Invalid source type\n");
  }

  static void set_smearing(MFBasicSource &smearing, const A2ASmearing &from){
    if(from.type == BOX_3D_SMEARING) smearing.set_source_type_and_radius(MFBasicSource::BoxSource, (Float)from.A2ASmearing_u.box_3d_smearing.side_length);
    else if(from.type == EXPONENTIAL_3D_SMEARING) smearing.set_source_type_and_radius(MFBasicSource::ExponentialSource, from.A2ASmearing_u.exponential_3d_smearing.radius);
    else ERR.General("","set_smearing(MFBasicSource &smearing, const A2ASmearing &from)","Unknown smearing type\n");
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
  MFqdpMatrix(const VorW &leftv, const VorW &rightv, const bool &_conj_left, const bool &_conj_right, const Float gamma_matrix_linear_comb[16], const Float pauli_matrix_linear_comb[4]): MFstructure(){
    set_form(leftv,rightv,_conj_left,_conj_right); set_matrix(gamma_matrix_linear_comb,pauli_matrix_linear_comb);
  }
  MFqdpMatrix(const VorW &leftv, const VorW &rightv, const bool &_conj_left, const bool &_conj_right,const SpinColorFlavorMatrix &to): MFstructure(), scf(to){
    set_form(leftv,rightv,_conj_left,_conj_right);
  }

  //Matrix form must be manually specified in the version below
  MFqdpMatrix(const VorW &leftv, const VorW &rightv, const bool &_conj_left, const bool &_conj_right): MFstructure(){
    set_form(leftv,rightv,_conj_left,_conj_right);
  }

  void set_matrix(const int &qdp_spin_idx, const FlavorMatrixType &flav_mat = sigma0);
  
  //Any 4x4 complex matrix can be represented as a linear combination of the 16 Gamma matrices (here in QDP order cf. below), and likewise any 2x2 complex matrix is a linear combination of Pauli matrices and the unit matrix (index 0)
  void set_matrix(const Float gamma_matrix_linear_comb[16], const Float pauli_matrix_linear_comb[4]);
  void set_matrix(const std::complex<Float> gamma_matrix_linear_comb[16], const std::complex<Float> pauli_matrix_linear_comb[4]);

  void set_matrix(const SpinColorFlavorMatrix &to){ scf = to; }

  //\Gamma(n) = \gamma_1^n1 \gamma_2^n2  \gamma_3^n3 \gamma_4^n4    where ni are bit fields: n4 n3 n2 n1 
  cnum contract_internal_indices(const cnum* left[2], const cnum* right[2]) const;
  cnum contract_internal_indices(const cnum* left, const cnum* right) const;
};

//range is a generic function the returns true if a particular value of t2 lies within the chosen range given a value of t1 and t3
struct RangeFunc{
  virtual bool allow(const int &t1, const int &t2, const int &t3) const = 0;
};
struct RangeAll: public RangeFunc{ //full range sum
  bool allow(const int &t1, const int &t2, const int &t3) const{ return true; }
};
struct RangeSpecificT: public RangeFunc{ //just one t value
  int t;
  RangeSpecificT(const int &_t): t(_t){}
  bool allow(const int &t1, const int &t2, const int &t3) const{ return t2 == t; }
};

struct RangeT1plusDelta: public RangeFunc{ //for sums over pion timeslices separated by delta
  int delta;
  RangeT1plusDelta(const int &d): delta(d){}

  bool allow(const int &t1, const int &t2, const int &t3) const{ 
    const int T = GJP.Tnodes()*GJP.TnodeSites();
    return t2 == (t1 + delta + T) % T; 
  }
};


//CK: A class that represents a single meson field, with convenience functions for contraction and element access. Based off Daiqian's code above.
//    In Daiqian's implementation, the meson field can be a non-square matrix. This is because when we are time-slice diluting, the vector V requires the
//    diluted source timeslice to be specified. In order to save memory space, the field W is not broken up into Lt/width different fields with zeroes
//    on all but one timeslice. There is therefore only one W field instance for Lt/width V instances. As a result, when the contraction is performed,
//    it does not look like a correct matrix multiplication, and you have to be very careful coverting the mode indices for the appropriate field.

//    Here I set up the meson field to look like a square matrix, and treat the time-slice dilution as a separate index that must be specified when
//    accessing the data.

class MesonField2{
public:
  enum LeftOrRight { Left=0, Right=1 };
private:
  Float *mf;
  int nvec[2];
  int nl[2];
  int nhits[2];
  int src_width[2];
  int nbase[2];

  bool dilute_flavor;

  int dilute_size; //span of diluted indices (not including the time dilution, which is handled separately). This is 3 color * 4 spin, and with an extra factor of 2 if using flavor dilution

  MFstructure::VorW form[2]; //Gives the types (V or W) of the left and right fields of the meson field comprises

  bool conj[2];

  int size[2]; //actual size of each vector: for V is it nvec = nl+nhits*Lt*dilute_size/width, for W nl+nhits*dilute_size

  int n_flav; //Number of flavours - 1 or 2 with G-parity

  //Check a2a propagator parameters for the fields V and W match between 2 MesonField2 instances
  inline static bool parameter_match(const MesonField2 &left, const MesonField2 &right, const LeftOrRight &field_left, const LeftOrRight &field_right){
    return left.nvec[(int)field_left] == right.nvec[(int)field_right] && left.nl[(int)field_left] == right.nl[(int)field_right] && left.nhits[(int)field_left] == right.nhits[(int)field_right] &&
      left.src_width[(int)field_left] == right.src_width[(int)field_right] && left.nbase[(int)field_left] == right.nbase[(int)field_right] && left.dilute_flavor == right.dilute_flavor;      
  }

  typedef std::complex<Float> cnum;
  
  //return 1 if W*V form, -1 if VW* form or 0 otherwise
  static int con_form(const MesonField2 &mf){
    if(mf.form[0] == MFstructure::W && mf.form[1] == MFstructure::V && mf.conj[0] && !mf.conj[1]) return 1;
    else if(mf.form[0] == MFstructure::V && mf.form[1] == MFstructure::W && !mf.conj[0] && mf.conj[1]) return -1;
    else return 0;
  }

public:
 MesonField2(): mf(NULL), dilute_flavor(false), dilute_size(12){}
 MesonField2(A2APropbfm &left, A2APropbfm &right, const MFstructure &structure, const MFsource &source): mf(NULL), dilute_flavor(false), dilute_size(12){ construct(left,right,structure,source); }

  int rows() const { return nl[0] + nhits[0]*dilute_size; }
  int cols() const { return nl[1] + nhits[1]*dilute_size; }
  
  int get_size(const LeftOrRight &f) const{ return size[(int)f]; }
  
  const MFstructure::VorW & field_type(const LeftOrRight &f) const{ return form[(int)f]; }
  

  //Convert from 0 <= i <  nl[0]+nhits[0]*dilute_size to mode index I of left or right field vector given a timeslice t: if field is v:  0 <= I < nl[0] + nhits[0] * Lt * dilute_size / width[0]   else if field is w: I = i 

  //Daiqian's 'Mode index' for the high modes of V contains a spin, color (flavor) and source-timeslice index: Mapping (index I > nl)
  //I-nl = (sc_id + 3*4/src_width * t_id +  3*4/src_width*Lt * wh_id)  or for G-parity with flavor dilution  (sc_id + 3*4*flav_id + 2*3*4* t_id/src_width +  2*3*4*Lt/src_width * wh_id)

  //My reduced mapping keeps (index i > nl) the timeslice index separate
  //i-nl = (sc_id + 3*4*wh_id) or for G-parity with flavor dilution (sc_id + 3*4*flav_id + 3*4*2*wh_id)

  //Thus   (i-nl) % dilute_size =  sc_id  or   sc_id + 3*4*flav_id  with flav dilution
  //       (i-nl) / dilute_size =  wh_id

  //Hence I-nl = (i-nl) % dilute_size + dilute_size/src_width * t_id + (i-nl)/dilute_size*nbase
  //where  nbase = Lt * dilute_size / width  and dilute_size = 3*4 or 3*4*2 with flavor dilution
  inline int idx(const int &i, const int &t, const LeftOrRight &field) const{
    return (form[(int)field] == MFstructure::W || i < nl[(int)field]) ?   i  :  nl[(int)field] + (i-nl[(int)field])/dilute_size*nbase[(int)field] + t/src_width[(int)field]*dilute_size + (i-nl[(int)field])%dilute_size ; //hit_idx, t, sc_idx
  }

  //Note, for non-unit src_width the above mapping is not invertible. We cannot fully recover t as the index is identical for all t within the source width
  inline void inv_idx(const int &I, int &i, int &t, const LeftOrRight &field){
    const static int NA(-1);
    if(form[(int)field] == MFstructure::W || I < nl[(int)field]){ i = I; t = NA; }
    else{
      int r = I-nl[(int)field];
      int hit = r / nbase[(int)field];  r %= nbase[(int)field];
      int scf = r % dilute_size;
      t = r / dilute_size;
      i = nl[(int)field] + scf +  dilute_size * hit;
    }
  }


  //Get the pointer at index I,J where I,J run over the range 0...size[0] and 0..size[1] respectively (unlike operator() where the indices are modified to have equal sizes with an intermediate mapping)
  inline cnum* mf_val(const int &I, const int &J, const int &t){ 
    return (cnum*)mf + t + GJP.TnodeSites()*GJP.Tnodes()* (J + size[1] * I ); 
  }
  inline const cnum* mf_val(const int &I, const int &J, const int &t) const{ 
    return (const cnum*)mf + t + GJP.TnodeSites()*GJP.Tnodes()* (J + size[1] * I ); 
  }
  //Get the value of the real part of the meson field for left/right mode indices i and j (where  0 <= i,j < nl + nhits*dilute_size)  where dilute_size = 3*4*(2) where the final 2 is for when flavor dilution is active
  //The meson field is formed as M_ij(t) = \sum_{\vec p} V_i(\vec p,t) S(\vec p,t) W_j*(\vec p,t)
  //hence we need only specify the mode indices i and j and the timeslice.
  //However in practise we use timeslice dilution for the high modes, so as well as specifying a hit index and spin/color(/flavor) combined index we must also specify a source *timeslice* when we are referring to the high-modes
  //of the vector V.
  //Within the A2APropbfm class the source timeslice index is combined with the hit/spin/color(/flavor) index into a larger 'mode' index. However it is convenient here to work with a unified range 0 <= i,j < nl + nhits*dilute_size
  //that applies both to W and V, and consider the source timeslice separately.
  
  //Allowing for a general form, including VV, we must allow for 3 specified timeslices. The first is the timeslice of the meson field ('t' in the above), then the source timeslice of the left and right vectors
  //which are used appropriately depending on the form of the contraction

  inline const cnum& operator()(const int &i, const int &j, const int &t_mf, const int &t_left, const int &t_right) const{ 
    return *( (const cnum*)mf + t_mf + GJP.TnodeSites()*GJP.Tnodes()*( idx(j,t_right,Right) + size[1]* idx(i,t_left,Left) ) );  //column index is fastest-changing
  }
  inline cnum& operator()(const int &i, const int &j, const int &t_mf, const int &t_left, const int &t_right){ 
    return *( (cnum*)mf + t_mf + GJP.TnodeSites()*GJP.Tnodes()*( idx(j,t_right,Right) + size[1]* idx(i,t_left,Left) ) );  //column index is fastest-changing
  }

  bool diluting_flavor() const{ return dilute_flavor; }

  //Build the meson field from the A2APropbfm instances
  void construct(A2APropbfm &left, A2APropbfm &right, const MFstructure &structure, const MFsource &source);

  friend class MesonFieldTesting;
  //--------------- Contraction Routines -------------------------------

  //Calculate   \sum_i M_iI(t1)  where I=(i,t2)
  std::complex<double> trace_wv(const int &t1, const int &t2);

  //Two-point function contractions:
  //Form the contraction of two mesonfields, summing over mode indices:   left_ij right_ji.  For G-parity we do  left_ij,fg right_ji,gf
  static void contract(const MesonField2 &left, const MesonField2 &right, const int &contraction_idx, CorrelationFunction &into);
  static void contract(const MesonField2 &left, const MesonField2 &right, CorrelationFunction &into){ return contract(left,right,0,into); }
  //Same as above but the user specifies a particular source and time-slice rather than summing over all. Useful for testing

  // into[t] = [ left(t)_iJ ][ right(tsrc)_jI ]
  //where J=(j,tsrc) and I=(i,t)
  static void contract_specify_tsrc(const MesonField2 &left, const MesonField2 &right, const int &contraction_idx, const int &tsrc, CorrelationFunction &into);

  // into[t1] = \sum_{t2 in 't2range'} [ left(t1)_iJ ][ right(t2)_jI ]
  //where J=(j,t2) and I=(i,t).  Assumes both mesonfields of  w^dag v form
  static void contract_specify_t2range(const MesonField2 &left, const MesonField2 &right, const int &contraction_idx, const RangeFunc &t2range, CorrelationFunction &into);

  //Assumes both meson fields have w^dag v form
  // result = [ left(t1)_iJ ][ right(t2)_jI ]
  //where J=(j,t2) and I=(i,t1)
  static std::complex<double> contract_fixedt1t2(const MesonField2 &left, const MesonField2 &right, const int &t1, const int &t2, const bool &threaded);

  //Combining meson fields in other ways:
  //Note we use lower-case Roman letters for w-vector indices and upper-case for v-vector indices. 
  //A v-vector index contains a source timeslice. To make this specific we can write  I=(i,t)
  
  
  //Take the outer product of a meson field of the form  w^dag v  with v and w 
  //result_{a,b} = \sum_{i,j} v_{aI}(x) M_{iJ}(t) w^dag_{bj}(z)
  //where I=(i,t)  and J=(j,z_4)

  static void contract_vleft_wright(SpinColorFlavorMatrix &result, 
				    A2APropbfm &prop_left,  const int &x,
				    A2APropbfm &prop_right, const int &z,
				    MesonField2 & mf,  const int &t);

  //Take the outer product of a meson field of the form  w^dag w  with v and v^dag 
  //result_{a,b} = \sum_{i,j} v_{aI}(x) M_{ij}(t) v^dag_{bJ}(z)
  //where I=(i,t)  and J=(j,t_vright)

  static void contract_vleft_vright(SpinColorFlavorMatrix &result, 
				    A2APropbfm &prop_left,  const int &x,
				    A2APropbfm &prop_right, const int &z,
				    MesonField2 & mf,  const int &t, const int &t_vright);

  //result_{ab} = \sum_i v_{aI}(x) w^dag_{bi}(z)
  //where I = (i,z_4)

  static void contract_vw(SpinColorFlavorMatrix &result, 
			  A2APropbfm &prop_left,  const int &x,
			  A2APropbfm &prop_right, const int &z);

  //Do the following:
  //into_iK(t1) =  \sum_{ t2 in 'range' }   [[\sum_{\vec x} w_i^dag(\vec x,t1) v_J(\vec x,t1)]] [[\sum_\vec y} w_j^dag(\vec y,t2) v_K(\vec y,t2) ]]
  //where J=(j,t2)
  //Where  double-square brackets indicate meson fields
 
  static void combine_mf_wv_wv(MesonField2 &into, const MesonField2 &mf_wv_l, const MesonField2 &mf_wv_r, const RangeFunc &range);

  //Do the following:
  //into_ik(t1) = \sum_{ t2 in 'range' } [[\sum_{\vec x} w_i^dag(\vec x,t1) v_J(\vec x, t1)]] [[\sum_\vec y} w_j^dag(\vec y, t2) w_k(\vec y, t2) ]]
  //where J=(j,t2)

  static void combine_mf_wv_ww(MesonField2 &into, const MesonField2 &mf_wv_l, const MesonField2 &mf_ww_r, const RangeFunc &range);
};
















#endif

CPS_END_NAMESPACE
