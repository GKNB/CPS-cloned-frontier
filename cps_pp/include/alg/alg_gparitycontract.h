CPS_START_NAMESPACE
#ifndef ALG_GPARITY_CONTRACT_H
#define ALG_GPARITY_CONTRACT_H
CPS_END_NAMESPACE

#include<config.h>
#include <alg/alg_base.h>
#include <util/lattice.h>
#include <alg/correlationfunction.h>
#include <alg/gparity_contract_arg.h>
#include <alg/common_arg.h>
#include <util/spincolorflavormatrix.h>
#include <vector>

CPS_START_NAMESPACE

//Class for determining allowed sink momenta
class QuarkMomCombination{
  const char* cname;
  std::vector< std::vector<std::vector<int> > > quark_mom;

  void mom_comb_recurse(std::vector<std::vector<int> > &into, std::vector<int> &cur_branch, const int &depth) const;
public:
  QuarkMomCombination(): cname("QuarkMomCombination"){};
  void reset();
  void add_prop(const PropagatorContainer &prop, const bool &conj);
  std::vector<std::vector<int> > get_total_momenta() const;

  //is 'what' one of the allowed momenta in the set 'in'
  static bool contains(const std::vector<int> &what, const std::vector<std::vector<int> > &in);
  
  //same as above but generates 'in' using get_total_momenta
  bool contains(const std::vector<int> &what) const;
  bool contains(const int *what) const;
};

//Contraction class
class AlgGparityContract : public Alg{
private:
  char *cname;
  GparityContractArg *args;

  //lattice properties
  int shift_x;
  int shift_y;
  int shift_z;
  int shift_t;

  int size_x;
  int size_y;
  int size_z;
  int size_t;

  int size_xy;
  int spatial_vol;

  void global_coord(const int &site, int *into_vec);
  Rcomplex sink_phasefac(const int *momphase, const int *pos, const bool &is_cconj = false); //pos is a 3-vector
  Rcomplex sink_phasefac(const int *momphase, const int &site, const bool &is_cconj = false); //site is the site index

  void sum_momphase(int *into, PropagatorContainer &prop);

  static const Float Pi_const;

  //Left/right multiply by a gamma matrix structure in QDP-style conventions:
  //\Gamma(n) = \gamma_1^n1 \gamma_2^n2  \gamma_3^n3 \gamma_4^n4    where ni are bit fields. 
  void qdp_gl(WilsonMatrix &wmat, const int &gidx) const;
  void qdp_gr(WilsonMatrix &wmat, const int &gidx) const;
  void qdp_gl(SpinColorFlavorMatrix &wmat, const int &gidx) const;
  void qdp_gr(SpinColorFlavorMatrix &wmat, const int &gidx) const;

  //Coefficient when matrix is transposed or conjugated
  Float qdp_gcoeff(const int &gidx, const bool &transpose, const bool &conj) const;

  void meson_LL_std(PropagatorContainer &prop, const int* sink_mom, const int &gamma_idx_1, const int &gamma_idx_2, FILE *fp);
  void meson_LL_gparity(PropagatorContainer &prop, const int* sink_mom, const int &gamma_idx_1, const int &gamma_idx_2, FILE *fp);

  void meson_HL_gparity(PropagatorContainer &prop_H, PropagatorContainer &prop_L, const int* sink_mom, const int &gamma_idx_1, const int &gamma_idx_2, FILE *fp);
  void meson_HL_std(PropagatorContainer &prop_H, PropagatorContainer &prop_L, const int* sink_mom, const int &gamma_idx_1, const int &gamma_idx_2, FILE *fp);

  void contract_OVVpAA_gparity(const ContractionTypeOVVpAA &args, const int &conf_idx);
  void contract_OVVpAA_std(const ContractionTypeOVVpAA &args, const int &conf_idx);
public:
  AlgGparityContract(Lattice & latt, CommonArg& c_arg, GparityContractArg& arg);
  void run(const int &conf_idx);
  void spectrum(const GparityMeasurement &measargs,const int &conf_idx);

  void contract_LL_mesons(const ContractionTypeLLMesons &args, const int &conf_idx);
  void contract_HL_mesons(const ContractionTypeHLMesons &args, const int &conf_idx);
  void contract_OVVpAA(const ContractionTypeOVVpAA &args, const int &conf_idx);

#if 0
  void gparity_
  //Gparity contractions (return true if contraction was performed)
  void pi_plus(const char *q_f0_tag, const char *q_f1_tag, CorrelationFunction &corr);
  void pi_plus_f0src(const char *q0_f0_tag, const char *q1_f0_tag, CorrelationFunction &corr);
  void pi_minus(const char *q_f0_tag, const char *q_f1_tag, CorrelationFunction &corr);
  void degen_kaon(const char *q_f0_tag, const char *q_f1_tag, CorrelationFunction &corr);
  
  //No Gparity contractions
  void pion_nogparity(const char *q_f0_tag, CorrelationFunction &corr);
#endif
};

#endif
CPS_END_NAMESPACE
