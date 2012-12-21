/*  Stuff   */

enum ContractionType {
  CONTRACTION_TYPE_LL_MESONS,
  CONTRACTION_TYPE_HL_MESONS,
  CONTRACTION_TYPE_O_VV_P_AA
};
struct ContractionTypeLLMesons{
 string prop_L<>;
 int sink_mom[3];
 string file<>;

 rpccommand GENERATE_PRINT_METHOD;
 rpccommand GENERATE_DEEPCOPY_METHOD;
};
struct ContractionTypeHLMesons{
 string prop_H<>;
 string prop_L<>;
 int sink_mom[3];
 string file<>;

 rpccommand GENERATE_PRINT_METHOD;
 rpccommand GENERATE_DEEPCOPY_METHOD;
};
struct ContractionTypeOVVpAA{
 string prop_H_t0<>;
 string prop_L_t0<>;
 string prop_H_t1<>;
 string prop_L_t1<>;
 string file<>;

 rpccommand GENERATE_PRINT_METHOD;
 rpccommand GENERATE_DEEPCOPY_METHOD;
};

union GparityMeasurement{
switch(ContractionType type){
 case CONTRACTION_TYPE_LL_MESONS:
   ContractionTypeLLMesons contraction_type_ll_mesons;
 case CONTRACTION_TYPE_HL_MESONS:
   ContractionTypeHLMesons contraction_type_hl_mesons;
 case CONTRACTION_TYPE_O_VV_P_AA:
   ContractionTypeOVVpAA contraction_type_o_vv_p_aa;
}
  rpccommand GENERATE_UNION_TYPEMAP;
  rpccommand GENERATE_DEEPCOPY_METHOD;
  rpccommand GENERATE_PRINT_METHOD;
};

class GparityContractArg{
  GparityMeasurement meas<>;
  string config_fmt<>; /* Should contain a %d which is replaced by a config index */
  int conf_start;
  int conf_incr;
  int conf_lessthan;
  FixGaugeArg fix_gauge; /* Gauge fixing - Defaults to FIX_GAUGE_NONE */

  memfun GparityContractArg();

 rpccommand GENERATE_DEEPCOPY_METHOD;
};
