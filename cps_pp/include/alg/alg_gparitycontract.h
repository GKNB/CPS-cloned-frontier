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

CPS_START_NAMESPACE


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
  Rcomplex sink_phasefac(int *momphase, int *pos, const bool &is_cconj = false); //pos is a 3-vector
  Rcomplex sink_phasefac(int *momphase, const int &site, const bool &is_cconj = false); //site is the site index

  void sum_momphase(int *into, PropagatorContainer &prop);

  static const Float Pi_const;
public:
  AlgGparityContract(Lattice & latt, CommonArg& c_arg, GparityContractArg& arg);
  void run(const int &conf_idx);
  void spectrum(const GparityMeasurement &measargs,const int &conf_idx);

  //Gparity contractions (return true if contraction was performed)
  bool pi_plus(const char *q_f0_tag, const char *q_f1_tag, CorrelationFunction &corr);
  bool pi_plus_f0src(const char *q0_f0_tag, const char *q1_f0_tag, CorrelationFunction &corr);
  bool pi_minus(const char *q_f0_tag, const char *q_f1_tag, CorrelationFunction &corr);
  bool degen_kaon(const char *q_f0_tag, const char *q_f1_tag, CorrelationFunction &corr);
  
  //No Gparity contractions
  void pion_nogparity(const char *q_f0_tag, CorrelationFunction &corr);
};

#endif
CPS_END_NAMESPACE
