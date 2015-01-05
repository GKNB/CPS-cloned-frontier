#ifndef PROP_DFT_H
#define PROP_DFT_H

#include<config.h>
#include <alg/alg_base.h>
#include <util/lattice.h>
#include <alg/correlationfunction.h>
#include <alg/gparity_contract_arg.h>
#include <alg/common_arg.h>
#include <util/spincolorflavormatrix.h>
#include <vector>
#include <utility>
#include <map>
#include <string>
#include <alg/alg_gparitycontract.h>
#include <algorithm>

#ifdef USE_OMP
#include <omp.h>
#endif

CPS_START_NAMESPACE

//SiteObjectType is the object that is Fourier transformed over space
template<typename SiteObjectType>
class DFT{
protected:
  void transform(std::vector<SiteObjectType> &result) const; //result will be a vector of size Lt

  virtual void setZero(SiteObjectType &what) = 0;
  virtual void accumulate(SiteObjectType &into, SiteObjectType &what) = 0;
  virtual void evaluate(SiteObjectType &into, const int &site, const int global_pos[], const int &local_t) = 0;
  virtual int threadAccumulateWork() = 0;
  virtual void threadAccumulate(std::vector<SiteObjectType> &result, const std::vector<std::vector<SiteObjectType> > &thread_result, const int &idx) = 0;
  virtual void latticeSum(SiteObjectType &what) = 0;

  virtual ~DFT(){}
};


class PropDFT{
public:
  enum Superscript { None, Transpose, Conj, Dagger }; 
protected:
  typedef std::map<std::vector<Float>, int> mom_idx_map_type;
  mom_idx_map_type mom_idx_map;
  int nmom;

  int momIdxMinusP(const int &pidx) const;

  template<typename MatrixType>
  void conjugateMomReorder(std::vector<std::vector<MatrixType> > &to, const std::vector<std::vector<MatrixType> > &from);

  void global_coord(const int &site, int *into_vec);

  template<typename MatrixType>
  void do_superscript(MatrixType &mat, const Superscript &ss);
  
  //bool mom_sort_pred(const std::pair<Float,int> &i, const std::pair<Float,int> &j);

  void find_p2sorted(std::vector< std::pair<Float,int> > &p2list, std::map<int,std::vector<Float> > &p2map);
public:
  PropDFT(): nmom(0){}

  void copy_momenta(const PropDFT &r){
    nmom = r.nmom;
    mom_idx_map = r.mom_idx_map;
  }

  void clear(){
    mom_idx_map.clear();
    nmom=0;
  }

  static Superscript trans_conj(const Superscript &what, const bool &trans, const bool &conj);

  //Add a sink momentum to the set of those generated by the Fourier transform
  //If not using cosine sinks, automatically adds minus the vector as this allows for optimization in retrieval 
  //when the bilinear desired is the complex or Hermitian conjugate of one already calculated
  void add_momentum(std::vector<Float> sink_mom);

  virtual ~PropDFT(){}
};


template<typename MatrixType>
struct _FourierProp_helper{};
template<>
struct _FourierProp_helper<SpinColorFlavorMatrix>{
  static void site_matrix(SpinColorFlavorMatrix &into, QPropWcontainer &prop, Lattice &lat, const int &site);
  static void lattice_sum(SpinColorFlavorMatrix &what);
  static void mult_gauge_fix_mat(SpinColorFlavorMatrix &what, const int &site, Lattice &lat);
  static void write(FILE *fp, const SpinColorFlavorMatrix &mat, const Float &p2, const std::vector<Float> &mom, const int &t);
};
template<>
struct _FourierProp_helper<WilsonMatrix>{
  static void site_matrix(WilsonMatrix &into, QPropWcontainer &prop, Lattice &lat, const int &site);
  static void lattice_sum(WilsonMatrix &what);
  static void mult_gauge_fix_mat(WilsonMatrix &what, const int &site, Lattice &lat);
  static void write(FILE *fp, const WilsonMatrix &mat, const Float &p2, const std::vector<Float> &mom, const int &t);
};




template<typename MatrixType>
class FourierProp:  public PropDFT
{
  typedef std::map<std::string, std::vector<std::vector<MatrixType> > > map_info_type;  //internal vector<vector<MatrixType> > indexed by momentum index then time
  map_info_type props;

  bool calculation_started;
  bool gauge_fix_sink; //defaults to true (only fixes if gauge fixing was actually performed ofc)
  bool cosine_sink;

  void calcProp(const std::string &tag, Lattice &lat);
public:
 FourierProp(): gauge_fix_sink(true), cosine_sink(false), calculation_started(false), PropDFT(){}

  //sink phase is cos(p_0 x_0)*cos(p_1 x_1)*cos(p_2 x_2) rather than e^{ip.x}
  void enableCosineSink();

  //modify gauge fixing status
  void gaugeFixSink(const bool &tf = true);
  
  //add sink_mom and (if not using cosine sinks) minus sink_mom to the list of momenta to be calculated
  void add_momentum(std::vector<Float> sink_mom);

  const std::vector<MatrixType> & getFTProp(Lattice &lat,
					    const std::vector<Float> &sink_momentum, 
					    char const* tag
					    );
  
  //write all momenta calculated
  void write(const std::string &tag, const char *file, Lattice &lat);
  void write(const std::string &tag, FILE *fp, Lattice &lat);

  //return to initial state
  void clear();
};




struct _PropagatorBilinear_generics{
  //internal vector<vector<MatrixType> > indexed by momentum index then time
  typedef std::pair<std::string,PropDFT::Superscript> prop_info;
  typedef std::pair<prop_info,prop_info> prop_info_pair;
  typedef std::map<prop_info_pair, std::vector<std::vector<SpinColorFlavorMatrix> > > map_info_scfmat;
  typedef std::map<prop_info_pair, std::vector<std::vector<WilsonMatrix> > > map_info_scmat;

  static prop_info_pair trans_conj(const prop_info_pair &what,  const bool &trans, const bool &conj){
    if(!conj && !trans) return what;

    prop_info_pair out;
    if(trans){
      out.first = what.second;
      out.second = what.first;
    }else{
      out = what;
    }
    out.first.second = PropDFT::trans_conj(out.first.second,trans,conj);
    out.second.second = PropDFT::trans_conj(out.second.second,trans,conj);
    return out;
  }

};


template<typename MatrixType>
struct _PropagatorBilinear_helper{};

template<>
struct _PropagatorBilinear_helper<SpinColorFlavorMatrix>{
  const static int nidx; //number of possible spin-flavour matrices 

  //currently only allow for spin and flavour matrices between the propagators
  static int scf_map(const int &spinidx, const int &flavidx);

  //unmap a spin-flavour index into separate spin and flavour indices
  static std::pair<int,int> unmap(const int &scf_idx);

  //Calculate the coefficient of the result of transposing and/or conjugating the spin-flavor matrix
  static Float coeff(const int &scf_idx, const bool &transpose, const bool &conj);

  typedef _PropagatorBilinear_generics::map_info_scfmat map_info_type; //the map type

  static void site_matrix(SpinColorFlavorMatrix &into, QPropWcontainer &prop, Lattice &lat, const int &site);

  //right-multiply with spin and flavour matrices
  static void rmult_matrix(SpinColorFlavorMatrix &into, const std::pair<int,int> &spin_flav);

  static void lattice_sum(SpinColorFlavorMatrix &what);
  
  static void unit_matrix(SpinColorFlavorMatrix &mat);

  static void write(FILE *fp, const SpinColorFlavorMatrix &mat, const int &idx, const Float &p2, const std::vector<Float> &mom, const int &t);
};
template<>
struct _PropagatorBilinear_helper<WilsonMatrix>{
  const static int nidx;
  
  static int scf_map(const int &spinidx, const int &flavidx);

  //put spin index on first element of output
  static std::pair<int,int> unmap(const int &sc_idx);

  //Calculate the coefficient of the result of transposing and/or conjugating the spin matrix
  static Float coeff(const int &sidx, const bool &transpose, const bool &conj);

  typedef _PropagatorBilinear_generics::map_info_scmat map_info_type; //the map type

  static void site_matrix(WilsonMatrix &into, QPropWcontainer &prop, Lattice &lat, const int &site);
  
  //right-multiply with spin matrices
  static void rmult_matrix(WilsonMatrix &into, const std::pair<int,int> &spin_flav);

  static void lattice_sum(WilsonMatrix &what);

  static void unit_matrix(WilsonMatrix &mat);

  static void write(FILE *fp, const WilsonMatrix &mat, const int &idx, const Float &p2, const std::vector<Float> &mom, const int &t);
};

template<typename MatrixType>
class PropagatorBilinear: public PropDFT{
private:
  typedef _PropagatorBilinear_generics::prop_info prop_info;
  typedef _PropagatorBilinear_generics::prop_info_pair prop_info_pair;
  typedef typename _PropagatorBilinear_helper<MatrixType>::map_info_type map_info_type;
  typedef PropDFT::mom_idx_map_type mom_idx_map_type;

  std::vector<map_info_type*> all_mats;
    
  static prop_info_pair trans_conj(const prop_info_pair &what,  const bool &trans, const bool &conj){ return _PropagatorBilinear_generics::trans_conj(what,trans,conj); }

  bool nmom_fixed;

  bool getBilinearFromExisting(std::vector<std::vector<MatrixType> > &mats, const int &idx, const prop_info_pair &props);

  void calcAllBilinears(const prop_info_pair &props, Lattice &lat);

  void calcBilinear(const int &idx, const prop_info_pair &props, Lattice &lat);

public:
  PropagatorBilinear(): all_mats(_PropagatorBilinear_helper<MatrixType>::nidx,NULL), nmom_fixed(false), PropDFT(){}

  //Get a (locally) Fourier transformed bilinear as a function of time
  //Sigma is ignored if MatrixType is WilsonMatrix
  const std::vector<MatrixType> & getBilinear(Lattice &lat,
					      const std::vector<Float> &sink_momentum, 
					      char const* tag_A, const Superscript &ss_A,  
					      char const* tag_B, const Superscript &ss_B, 
					      const int &Gamma, const int &Sigma = 0
					      );

  void calcAllBilinears(Lattice &lat,
			char const* tag_A, const Superscript &ss_A,  
			char const* tag_B, const Superscript &ss_B);
  
  void clear();

  void add_momentum(std::vector<Float> sink_mom);


  void write(char const* tag_A, const Superscript &ss_A,  
	     char const* tag_B, const Superscript &ss_B,
	     const int &Gamma, const int &Sigma,
	     const char *file, Lattice &lat);

  void write(char const* tag_A, const Superscript &ss_A,  
	     char const* tag_B, const Superscript &ss_B,
	     const int &Gamma, const int &Sigma,
	     FILE *fp, Lattice &lat);
    

  void write(char const* tag_A, const Superscript &ss_A,  
	     char const* tag_B, const Superscript &ss_B,
	     const char *file, Lattice &lat);

  void write(char const* tag_A, const Superscript &ss_A,  
	     char const* tag_B, const Superscript &ss_B,
	     FILE *fp, Lattice &lat);

};




template<typename MatrixType>
struct _ContractedBilinear_helper{};
template<>
struct _ContractedBilinear_helper<SpinColorFlavorMatrix>{
  static void write(FILE *fp, const Rcomplex &val, const int &idx1, const int &idx2, const Float &p2, const std::vector<Float> &mom, const int &t);
};
template<>
struct _ContractedBilinear_helper<WilsonMatrix>{
  static void write(FILE *fp, const Rcomplex &val, const int &idx1, const int &idx2, const Float &p2, const std::vector<Float> &mom, const int &t);
};


//Calculate  tr{  A G_1 B G_2  }
//where A, B are spin/flavour matrices and G_1 and G_2 are propagators with an arbitrary superscript (transpose, hermition conj, conj or nothing)

template<typename MatrixType>
class ContractedBilinear: public PropDFT{
private:
  typedef _PropagatorBilinear_generics::prop_info prop_info;
  typedef _PropagatorBilinear_generics::prop_info_pair prop_info_pair;
  typedef PropDFT::mom_idx_map_type mom_idx_map_type;

  typedef std::map<prop_info_pair, Rcomplex *> map_info_type; //elements of the Rcomplex* indexed by mapping from [mat1][mat2][mom][t]
  
  const int nmat;
  int array_size;
  map_info_type results;
  
  static prop_info_pair trans_conj(const prop_info_pair &what,  const bool &trans, const bool &conj){ return _PropagatorBilinear_generics::trans_conj(what,trans,conj); }

  int idx_map(const int &mat1, const int &mat2, const int & mom_idx, const int &t) const;
  void idx_unmap(const int &idx, int &mat1, int &mat2, int & mom_idx, int &t) const;

  void calcAllContractedBilinears1(const prop_info_pair &props, Lattice &lat);
  void calcAllContractedBilinears2(const prop_info_pair &props, Lattice &lat);

  static prop_info_pair inplacetrans_conj(const prop_info_pair &what,  const bool &trans, const bool &conj);

  bool getBilinearsFromExisting(const prop_info_pair &props);


  void calculateBilinears(Lattice &lat,
			  const prop_info_pair &props,
			  const int &version = 0
			  );


public:
 ContractedBilinear(): array_size(-1), nmat(_PropagatorBilinear_helper<MatrixType>::nidx), PropDFT(){}

  void calculateBilinears(Lattice &lat,
			  char const* tag_A, const Superscript &ss_A,  
			  char const* tag_B, const Superscript &ss_B, 
			  const int &version = 0
			  );


  //Get a Fourier transformed bilinear correlation function as a function of time
  //Sigma is ignored if MatrixType is WilsonMatrix
  std::vector<Rcomplex> getBilinear(Lattice &lat,
				    const std::vector<Float> &sink_momentum, 
				    char const* tag_A, const Superscript &ss_A,  
				    char const* tag_B, const Superscript &ss_B, 
				    const int &Gamma1, const int &Sigma1,
				    const int &Gamma2, const int &Sigma2,
				    const int &version = 0
				    );

  //for use with WilsonMatrix where sigma (flavour matrix idx) does not play a role
  std::vector<Rcomplex> getBilinear(Lattice &lat,
				    const std::vector<Float> &sink_momentum, 
				    char const* tag_A, const Superscript &ss_A,  
				    char const* tag_B, const Superscript &ss_B, 
				    const int &Gamma1, const int &Gamma2, const int &version = 0
				    );
    

  void add_momentum(const std::vector<Float> &sink_mom);

  void clear();

  ~ContractedBilinear(){ clear(); }


  void write(char const* tag_A, const Superscript &ss_A,  
	     char const* tag_B, const Superscript &ss_B, 
	     const int &Gamma1, const int &Sigma1,
	     const int &Gamma2, const int &Sigma2,
	     const char *file, Lattice &lat);

  void write(char const* tag_A, const Superscript &ss_A,  
	     char const* tag_B, const Superscript &ss_B, 
	     const int &Gamma1, const int &Sigma1,
	     const int &Gamma2, const int &Sigma2,
	     FILE *fp, Lattice &lat);
  
  //write all combinations
  void write(char const* tag_A, const Superscript &ss_A,  
	     char const* tag_B, const Superscript &ss_B,
	     const std::string &file, Lattice &lat);

  void write(char const* tag_A, const Superscript &ss_A,  
	     char const* tag_B, const Superscript &ss_B, 
	     FILE *fp, Lattice &lat);
};



//Calculate  tr{  A G_1 B G_2  }
//where A, B are spin/flavour matrices and G_1 and G_2 are propagators with an arbitrary superscript (transpose, hermition conj, conj or nothing)
//This is a simpler version of the above which just computes and stores the results for one pair of propagators. 
template<typename MatrixType>
class ContractedBilinearSimple: public PropDFT{
private:
  typedef PropDFT::mom_idx_map_type mom_idx_map_type;

  const int nmat; //number of matrices:  for spin matrices there are 16, for G-parity spin-flavor matrices there are 16*4
  int array_size;
  Rcomplex * results; //elements of the Rcomplex* indexed by mapping from [mat1][mat2][mom][t]

  inline int idx_map(const int &mat1, const int &mat2, const int & mom_idx, const int &t) const{ return mat1 + nmat * (mat2 + nmat*(mom_idx + nmom*t)); }
  inline void idx_unmap(const int &idx, int &mat1, int &mat2, int & mom_idx, int &t) const{
    int rem = idx;
    mat1 = rem % nmat;  rem /= nmat;
    mat2 = rem % nmat;  rem /= nmat;
    mom_idx = rem % nmom; rem /= nmom;
    t = rem;
  };

public:
 ContractedBilinearSimple(): array_size(-1), nmat(_PropagatorBilinear_helper<MatrixType>::nidx), PropDFT(){}

  void calculateBilinears(Lattice &lat,
			  char const* tag_A, const Superscript &ss_A,  
			  char const* tag_B, const Superscript &ss_B);

  //Get a Fourier transformed bilinear correlation function as a function of time
  //Sigma is ignored if MatrixType is WilsonMatrix
  std::vector<Rcomplex> getBilinear(const std::vector<Float> &sink_momentum, 
				    const int &Gamma1, const int &Sigma1,
				    const int &Gamma2, const int &Sigma2);

  //for use with WilsonMatrix where sigma (flavour matrix idx) does not play a role
  std::vector<Rcomplex> getBilinear(const std::vector<Float> &sink_momentum, 
				    const int &Gamma1, const int &Gamma2);
    
  //Can only be added before calculation has been performed
  void add_momentum(const std::vector<Float> &sink_mom);

  void clear();

  ~ContractedBilinearSimple(){ clear(); }
  
  //write all combinations
  void write(const std::string &file);
  void write(FILE *fp);

  //Contents become the sum of the contents of this object and r
  ContractedBilinearSimple<MatrixType> & operator+=(const ContractedBilinearSimple<MatrixType> &r);
  ContractedBilinearSimple<MatrixType> & operator-=(const ContractedBilinearSimple<MatrixType> &r);
  //Contents are divided by a float
  ContractedBilinearSimple<MatrixType> & operator/=(const Float &r); 

  //Shift the data:   data[t] -> data[t+dt]  (modulo lattice size)
  void Tshift(const int &dt);
};


//24^4 tensor in spin,color and flavor
class QuadrilinearSCFVertex{
 private:
  Rcomplex *tensor;
  int size;

  int map(const int &s1,const int &c1, const int &f1,
	  const int &s2,const int &c2, const int &f2,
	  const int &s3,const int &c3, const int &f3,
	  const int &s4,const int &c4, const int &f4) const;
  void unmap(int idx,
	     int &s1,int &c1, int &f1,
	     int &s2,int &c2, int &f2,
	     int &s3,int &c3, int &f3,
	     int &s4,int &c4, int &f4) const;

 public:
  QuadrilinearSCFVertex();
  QuadrilinearSCFVertex(const Rcomplex &val);

  //form tensor from outer product of 2 SpinColorFlavorMatrix
  QuadrilinearSCFVertex(const SpinColorFlavorMatrix &bilA, const SpinColorFlavorMatrix &bilB, const Rcomplex &phase = 1.0);

  //add tensor formed from outer product of 2 SpinColorFlavorMatrix
  QuadrilinearSCFVertex &add(const SpinColorFlavorMatrix &bilA, const SpinColorFlavorMatrix &bilB, const Rcomplex &phase = 1.0);

  QuadrilinearSCFVertex& operator=(const Rcomplex &val);
  QuadrilinearSCFVertex& operator=(const Float &val);
  QuadrilinearSCFVertex& operator+=(const QuadrilinearSCFVertex &rhs);

  const Rcomplex & operator()(const int &s1,const int &c1, const int &f1,
			      const int &s2,const int &c2, const int &f2,
			      const int &s3,const int &c3, const int &f3,
			      const int &s4,const int &c4, const int &f4) const;

  Rcomplex* ptr();

  ~QuadrilinearSCFVertex();
};


//16^4 tensor in spin and color
class QuadrilinearSCVertex{
 private:
  Rcomplex *tensor;
  int size;

  int map(const int &s1,const int &c1,
		 const int &s2,const int &c2,
		 const int &s3,const int &c3,
	  const int &s4,const int &c4) const;

  void unmap(int idx,
		    int &s1,int &c1,
		    int &s2,int &c2,
		    int &s3,int &c3,
	     int &s4,int &c4) const;


 public:
  QuadrilinearSCVertex();
  QuadrilinearSCVertex(const Rcomplex &val);

  //form tensor from outer product of 2 WilsonMatrix
  QuadrilinearSCVertex(const WilsonMatrix &bilA, const WilsonMatrix &bilB, const Rcomplex &phase = 1.0);

  //add tensor formed from outer product of 2 WilsonMatrix
  QuadrilinearSCVertex &add(const WilsonMatrix &bilA, const WilsonMatrix &bilB, const Rcomplex &phase = 1.0);

  QuadrilinearSCVertex& operator=(const Rcomplex &val);
  QuadrilinearSCVertex& operator=(const Float &val);

  QuadrilinearSCVertex& operator+=(const QuadrilinearSCVertex &rhs);

  const Rcomplex & operator()(const int &s1,const int &c1,
			      const int &s2,const int &c2,
			      const int &s3,const int &c3,
			      const int &s4,const int &c4) const;

  Rcomplex* ptr();

  ~QuadrilinearSCVertex();
};



template<typename MatrixType>
struct _PropagatorQuadrilinear_helper{};
template<>
struct _PropagatorQuadrilinear_helper<SpinColorFlavorMatrix>{
  typedef QuadrilinearSCFVertex TensorType;
  static void write(FILE *fp, const TensorType &tensor, const int &idx1, const int &idx2, const Float &p2, const std::vector<Float> &mom, const int &t);
  static void lattice_sum(TensorType &t);
};
template<>
struct _PropagatorQuadrilinear_helper<WilsonMatrix>{
  typedef QuadrilinearSCVertex TensorType;
  static void write(FILE *fp, const TensorType &tensor, const int &idx1, const int &idx2, const Float &p2, const std::vector<Float> &mom, const int &t);
  static void lattice_sum(TensorType &t);
};


template<typename MatrixType>
class PropagatorQuadrilinear: 
  public PropDFT
{
private:
  typedef _PropagatorBilinear_generics::prop_info prop_info;
  typedef _PropagatorBilinear_generics::prop_info_pair prop_info_pair;
  typedef std::pair<prop_info_pair,prop_info_pair> prop_info_quad;

  typedef typename _PropagatorQuadrilinear_helper<MatrixType>::TensorType TensorType;

  typedef typename std::map< std::pair<prop_info_quad,int>, std::vector<std::vector<TensorType > > >  map_info_type; 
  //key maps a prop_info_pair and an index idx_pair = idx1+nidx*idx2 to a vector<vector<TensorType >> indexed by [p][t]
  //the tensors represent an outer product of matrices Fourier transformed over position
  
  typedef PropDFT::mom_idx_map_type mom_idx_map_type;

  map_info_type all_mats;
  bool nmom_fixed;
  const int nmat;

  void unmap(int pair_idx, int &idx1, int &idx2) const;

  void calcQuadrilinear(const int &idx, const prop_info_quad &props, Lattice &lat);

  void calcAllQuadrilinears(const prop_info_quad &props, Lattice &lat);

public:
 PropagatorQuadrilinear(): PropDFT(), nmat(_PropagatorBilinear_helper<MatrixType>::nidx), nmom_fixed(false){}

  //Get a Fourier transformed quadrilinear correlation function as a function of time
  //Sigma is ignored if MatrixType is WilsonMatrix
  //quadrilinear form is    A Gamma1 Sigma1 B  \otimes  C Gamma2 Sigma2 D        where \otimes is an outer product and A,B,C,D are propagators
  const std::vector<TensorType> &getQuadrilinear(Lattice &lat,
						 const std::vector<Float> &sink_momentum, 
						 char const* tag_A, const Superscript &ss_A,  
						 char const* tag_B, const Superscript &ss_B, 
						 char const* tag_C, const Superscript &ss_C,  
						 char const* tag_D, const Superscript &ss_D, 
						 const int &Gamma1, const int &Sigma1,
						 const int &Gamma2, const int &Sigma2);

  //for use with WilsonMatrix where sigma (flavour matrix idx) does not play a role
  const std::vector<TensorType> & getQuadrilinear(Lattice &lat,
				    const std::vector<Float> &sink_momentum, 
				    char const* tag_A, const Superscript &ss_A,  
				    char const* tag_B, const Superscript &ss_B, 
				    char const* tag_C, const Superscript &ss_C,  
				    char const* tag_D, const Superscript &ss_D, 
				    const int &Gamma1, const int &Gamma2
						  );
    

  void add_momentum(std::vector<Float> sink_mom);
  void clear();

  void write(char const* tag_A, const Superscript &ss_A,  
	     char const* tag_B, const Superscript &ss_B,
	     char const* tag_C, const Superscript &ss_C,  
	     char const* tag_D, const Superscript &ss_D, 
	     const int &Gamma1, const int &Sigma1,
	     const int &Gamma2, const int &Sigma2,
	     const char *file, Lattice &lat);

  void write(char const* tag_A, const Superscript &ss_A,  
	     char const* tag_B, const Superscript &ss_B,
	     char const* tag_C, const Superscript &ss_C,  
	     char const* tag_D, const Superscript &ss_D,
	     const int &Gamma1, const int &Sigma1,
	     const int &Gamma2, const int &Sigma2,
	     FILE *fp, Lattice &lat);

  void write(char const* tag_A, const Superscript &ss_A,  
	     char const* tag_B, const Superscript &ss_B,
	     char const* tag_C, const Superscript &ss_C,  
	     char const* tag_D, const Superscript &ss_D, 
	     const char *file, Lattice &lat);

  void write(char const* tag_A, const Superscript &ss_A,  
	     char const* tag_B, const Superscript &ss_B,
	     char const* tag_C, const Superscript &ss_C,  
	     char const* tag_D, const Superscript &ss_D,
	     FILE *fp, Lattice &lat);
};



template<typename MatrixType>
class ContractedWallSinkBilinear: public FourierProp<MatrixType>{
private:
  typedef _PropagatorBilinear_generics::prop_info prop_info;
  typedef _PropagatorBilinear_generics::prop_info_pair prop_info_pair;
  typedef PropDFT::mom_idx_map_type mom_idx_map_type;

  typedef std::map<prop_info_pair, Rcomplex *> map_info_type; //elements of the Rcomplex* indexed by mapping from [mat1][mat2][mom][t]
  
  const int nmat;
  int array_size;
  map_info_type results;
  
  static prop_info_pair trans_conj(const prop_info_pair &what,  const bool &trans, const bool &conj){ return _PropagatorBilinear_generics::trans_conj(what,trans,conj); }

  int idx_map(const int &mat1, const int &mat2, const int & mom_idx, const int &t) const;
  void idx_unmap(const int &idx, int &mat1, int &mat2, int & mom_idx, int &t) const;

  void calcAllContractedBilinears(const prop_info_pair &props, Lattice &lat);

  static prop_info_pair inplacetrans_conj(const prop_info_pair &what,  const bool &trans, const bool &conj);

  bool getBilinearsFromExisting(const prop_info_pair &props);

  void calculateBilinears(Lattice &lat,
			  const prop_info_pair &props);


public:
 ContractedWallSinkBilinear(): array_size(-1), nmat(_PropagatorBilinear_helper<MatrixType>::nidx), FourierProp<MatrixType>(){}

  void calculateBilinears(Lattice &lat,
			  char const* tag_A, const PropDFT::Superscript &ss_A,  
			  char const* tag_B, const PropDFT::Superscript &ss_B
			  );

  //Get a Fourier transformed wall sink bilinear correlation function as a function of time
  //Sigma is ignored if MatrixType is WilsonMatrix
  std::vector<Rcomplex> getBilinear(Lattice &lat,
				    const std::vector<Float> &sink_momentum, 
				    char const* tag_A, const PropDFT::Superscript &ss_A,  
				    char const* tag_B, const PropDFT::Superscript &ss_B, 
				    const int &Gamma1, const int &Sigma1,
				    const int &Gamma2, const int &Sigma2
				    );

  //for use with WilsonMatrix where sigma (flavour matrix idx) does not play a role
  std::vector<Rcomplex> getBilinear(Lattice &lat,
				    const std::vector<Float> &sink_momentum, 
				    char const* tag_A, const PropDFT::Superscript &ss_A,  
				    char const* tag_B, const PropDFT::Superscript &ss_B, 
				    const int &Gamma1, const int &Gamma2
				    );
    

  void add_momentum(const std::vector<Float> &sink_mom);

  void clear();
  ~ContractedWallSinkBilinear(){ clear(); }

private:
  void _writeit(FILE *fp, Rcomplex* con,const int &scf_idx1, const int &scf_idx2, const std::vector< std::pair<Float,int> > &p2list, const std::map<int,std::vector<Float> > &p2map);
public:

  void write(char const* tag_A, const PropDFT::Superscript &ss_A,  
	     char const* tag_B, const PropDFT::Superscript &ss_B, 
	     const int &Gamma1, const int &Sigma1,
	     const int &Gamma2, const int &Sigma2,
	     const char *file, Lattice &lat);

  void write(char const* tag_A, const PropDFT::Superscript &ss_A,  
	     char const* tag_B, const PropDFT::Superscript &ss_B, 
	     const int &Gamma1, const int &Sigma1,
	     const int &Gamma2, const int &Sigma2,
	     FILE *fp, Lattice &lat);
  
  //write all combinations
  void write(char const* tag_A, const PropDFT::Superscript &ss_A,  
	     char const* tag_B, const PropDFT::Superscript &ss_B,
	     const std::string &file, Lattice &lat);

  void write(char const* tag_A, const PropDFT::Superscript &ss_A,  
	     char const* tag_B, const PropDFT::Superscript &ss_B, 
	     FILE *fp, Lattice &lat);
};




template<typename MatrixType>
struct _ContractedWallSinkBilinearSpecMomentum_helper{};
template<>
struct _ContractedWallSinkBilinearSpecMomentum_helper<SpinColorFlavorMatrix>{
  static void write(FILE *fp, const Rcomplex &val, const int &idx1, const int &idx2, const Float &p2, const std::vector<Float> &mom1, const std::vector<Float> &mom2, const int &t);
};
template<>
struct _ContractedWallSinkBilinearSpecMomentum_helper<WilsonMatrix>{
  static void write(FILE *fp, const Rcomplex &val, const int &idx1, const int &idx2, const Float &p2, const std::vector<Float> &mom1, const std::vector<Float> &mom2, const int &t);
};


//In this version you can manually specify the momenta of the fourier transformed quarks individually
//The total momentum at the sink is the sum of the two specified momenta 
//(if any of the props is complex conjugated or daggered, the momentum assigned to that prop will automatically be flipped for the sink fourier transform
// such that after the conjugate is performed, the phases add)

template<typename MatrixType>
class ContractedWallSinkBilinearSpecMomentum{
private:
  typedef _PropagatorBilinear_generics::prop_info prop_info;
  typedef _PropagatorBilinear_generics::prop_info_pair prop_info_pair;

  typedef std::map<std::pair<std::vector<Float>,std::vector<Float> >, int> mom_pair_idx_map_type;
  mom_pair_idx_map_type mom_pair_idx_map;
  int nmompairs;

  typedef std::map<prop_info_pair, Rcomplex *> map_info_type; //elements of the Rcomplex* indexed by mapping from [mat1][mat2][mom][t]
  
  const int nmat;
  int array_size;
  map_info_type results;

  bool cosine_sink; //at sink use momentum combination (e^{ipx}+e^{-ipx})/2 for the quarks
  
  FourierProp<MatrixType> fprop;

  int idx_map(const int &mat1, const int &mat2, const int & mom_idx, const int &t) const;
  void idx_unmap(const int &idx, int &mat1, int &mat2, int & mom_idx, int &t) const;
  void do_superscript(MatrixType &mat, const PropDFT::Superscript &ss);

  void calcAllContractedBilinears(const prop_info_pair &props, Lattice &lat);

  void calculateBilinears(Lattice &lat,
			  const prop_info_pair &props);

  static bool mom_sort_pred(const std::pair<Float,int> &i, const std::pair<Float,int> &j);

  void find_p2sorted(std::vector< std::pair<Float,int> > &p2list, std::map<int,std::pair<std::vector<Float>,std::vector<Float> > > &p2map);
public:
 ContractedWallSinkBilinearSpecMomentum(): array_size(-1), nmat(_PropagatorBilinear_helper<MatrixType>::nidx), nmompairs(0), cosine_sink(false){  fprop.gaugeFixSink(true);  }

  void enableCosineSink();

  void calculateBilinears(Lattice &lat,
			  char const* tag_A, const PropDFT::Superscript &ss_A,  
			  char const* tag_B, const PropDFT::Superscript &ss_B
			  );

  //Get a Fourier transformed wall sink bilinear correlation function as a function of time
  //Sigma is ignored if MatrixType is WilsonMatrix
  std::vector<Rcomplex> getBilinear(Lattice &lat,
				    const std::pair<std::vector<Float>,std::vector<Float> > &sink_momenta, 
				    char const* tag_A, const PropDFT::Superscript &ss_A,  
				    char const* tag_B, const PropDFT::Superscript &ss_B, 
				    const int &Gamma1, const int &Sigma1,
				    const int &Gamma2, const int &Sigma2
				    );

  //for use with WilsonMatrix where sigma (flavour matrix idx) does not play a role
  std::vector<Rcomplex> getBilinear(Lattice &lat,
				    const std::pair<std::vector<Float>,std::vector<Float> > &sink_momenta,
				    char const* tag_A, const PropDFT::Superscript &ss_A,  
				    char const* tag_B, const PropDFT::Superscript &ss_B, 
				    const int &Gamma1, const int &Gamma2
				    );
    

  void add_momentum(const std::pair<std::vector<Float>,std::vector<Float> > &sink_momenta);

  void clear();
  ~ContractedWallSinkBilinearSpecMomentum(){ clear(); }


private:
  void _writeit(FILE *fp,Rcomplex *con,const int &scf_idx1, const int &scf_idx2, const std::vector< std::pair<Float,int> > &p2list, const std::map<int,std::pair<std::vector<Float>,std::vector<Float> > > &p2map);
public:

  void write(char const* tag_A, const PropDFT::Superscript &ss_A,  
	     char const* tag_B, const PropDFT::Superscript &ss_B, 
	     const int &Gamma1, const int &Sigma1,
	     const int &Gamma2, const int &Sigma2,
	     const char *file, Lattice &lat);

  void write(char const* tag_A, const PropDFT::Superscript &ss_A,  
	     char const* tag_B, const PropDFT::Superscript &ss_B, 
	     const int &Gamma1, const int &Sigma1,
	     const int &Gamma2, const int &Sigma2,
	     FILE *fp, Lattice &lat);
  
  //write all combinations
  void write(char const* tag_A, const PropDFT::Superscript &ss_A,  
	     char const* tag_B, const PropDFT::Superscript &ss_B,
	     const std::string &file, Lattice &lat);

  void write(char const* tag_A, const PropDFT::Superscript &ss_A,  
	     char const* tag_B, const PropDFT::Superscript &ss_B, 
	     FILE *fp, Lattice &lat);
};


#include "prop_dft_impl.hxx"

CPS_END_NAMESPACE

#endif

