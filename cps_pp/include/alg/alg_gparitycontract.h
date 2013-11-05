#ifndef ALG_GPARITY_CONTRACT_H
#define ALG_GPARITY_CONTRACT_H

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
CPS_START_NAMESPACE

//Class for determining allowed sink momenta
class QuarkMomCombination{
  const char* cname;
  std::vector< std::vector<std::vector<int> > > quark_mom;
  std::vector<std::pair<PropagatorContainer const*,bool> > props;

  void mom_comb_recurse(std::vector<std::vector<int> > &into, std::vector<int> &cur_branch, const int &depth) const;

  void calc_allowed_combinations();
  
  std::vector<std::vector<int> > allowed_combinations;
  bool allowed_comb_calculated;
  int chosen_momcomb;
public:
  QuarkMomCombination(): cname("QuarkMomCombination"), chosen_momcomb(-1), allowed_comb_calculated(false){};
  void reset();
  void add_prop(const PropagatorContainer &prop, const bool &conj);

  void set_desired_momentum(const std::vector<int> &what);
  void set_desired_momentum(const int *what);

 //see if momentum combination 'what' is a valid combination for this contraction
  bool contains(const std::vector<int> &what);
  bool contains(const int *what);

  std::vector<Float> get_p() const;

  //Calculate the sink complex phase factor. Must have previously set a desired momentum combination
  Rcomplex phase_factor(const int *pos) const;
  Rcomplex phase_factor(const int &site) const;

  const std::vector<std::vector<int> > &get_allowed_momenta() const;
};

class ContractionQuarkMomCombination{
  std::vector<QuarkMomCombination> momcomb;
  std::map<int,int> contraction_map;
public:
  void add_contraction(const int &contraction, const QuarkMomCombination &cmomcomb);
  void same(const int &contraction, const int &same_as_contraction);
  void set_desired_momentum(const std::vector<int> &what);
  void set_desired_momentum(const int *what);

  std::vector<Float> get_p(const int &contraction) const;

  Complex phase_factor(const int &contraction, const int* pos) const;
  Complex phase_factor(const int &contraction, const int& site) const;
};

//Contraction class
class AlgGparityContract : public Alg{
public:
  //Left/right multiply by a gamma matrix structure in QDP-style conventions:
  //\Gamma(n) = \gamma_1^n1 \gamma_2^n2  \gamma_3^n3 \gamma_4^n4    where ni are bit fields. 
  static void qdp_gl(WilsonMatrix &wmat, const int &gidx);
  static void qdp_gr(WilsonMatrix &wmat, const int &gidx);
  static void qdp_gl(SpinColorFlavorMatrix &wmat, const int &gidx);
  static void qdp_gr(SpinColorFlavorMatrix &wmat, const int &gidx);

  //Coefficient when matrix is transposed or conjugated
  static Float qdp_gcoeff(const int &gidx, const bool &transpose, const bool &conj);
  static Float pauli_coeff(const int &pidx, const bool &transpose, const bool &conj);
private:
  char *cname;
  GparityContractArg *args;

  void global_coord(const int &site, int *into_vec);
  
  void meson_LL_std(PropagatorContainer &prop, const int* sink_mom, const int &gamma_idx_1, const int &gamma_idx_2, FILE *fp);
  void meson_LL_gparity(PropagatorContainer &prop, const int* sink_mom, const int &gamma_idx_1, const int &gamma_idx_2, FILE *fp);

  void meson_HL_gparity(PropagatorContainer &prop_H, PropagatorContainer &prop_L, const int* sink_mom, const int &gamma_idx_1, const int &gamma_idx_2, FILE *fp);
  void meson_HL_std(PropagatorContainer &prop_H, PropagatorContainer &prop_L, const int* sink_mom, const int &gamma_idx_1, const int &gamma_idx_2, FILE *fp);

  void contract_OVVpAA_gparity(const ContractionTypeOVVpAA &args, const int &conf_idx);
  void contract_OVVpAA_std(const ContractionTypeOVVpAA &args, const int &conf_idx);
public:
  AlgGparityContract(Lattice & latt, CommonArg& c_arg, GparityContractArg& arg);
  AlgGparityContract(Lattice & latt, CommonArg& c_arg);

  void set_args(GparityContractArg *to){ args = to; }

  //Run the measurement programme defined by the 'job' input. 
  //By default it uses the args pointer provided when this object is constructed or set using set_args, but user can choose a different run script.
  void run(const int &conf_idx);
  void run(const int &conf_idx, const GparityContractArg& job);

  //Measure the quantity specified by the GparityMeasurement input (essentially a factory)
  void spectrum(const GparityMeasurement &measargs,const int &conf_idx);

  void contract_LL_mesons(const ContractionTypeLLMesons &args, const int &conf_idx);
  void contract_HL_mesons(const ContractionTypeHLMesons &args, const int &conf_idx);
  void contract_OVVpAA(const ContractionTypeOVVpAA &args, const int &conf_idx);
  void contract_all_bilinears(const ContractionTypeAllBilinears &args, const int &conf_idx);
  void contract_all_wallsink_bilinears(const ContractionTypeAllWallSinkBilinears &args, const int &conf_idx);
  void contract_all_wallsink_bilinears_specific_momentum(const ContractionTypeAllWallSinkBilinearsSpecificMomentum &args, const int &conf_idx);
  void contract_fourier_prop(const ContractionTypeFourierProp &args, const int &conf_idx);
  void contract_bilinear_vertex(const ContractionTypeBilinearVertex &args, const int &conf_idx);
  void contract_quadrilinear_vertex(const ContractionTypeQuadrilinearVertex &args, const int &conf_idx);
  void measure_topological_charge(const ContractionTypeTopologicalCharge &args, const int &conf_idx);
  void measure_mres(const ContractionTypeMres &args, const int &conf_idx);
  void measure_mres(const ContractionTypeMres &args, CorrelationFunction &pion, CorrelationFunction &j5_q);
};

CPS_END_NAMESPACE
#endif

