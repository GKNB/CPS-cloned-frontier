/*  Stuff   */

struct GparityMeasurement{
  string prop_1<>;
  string prop_2<>;
  string label_stub<>;
  string file_stub<>;
};
class GparityContractArg{
  GparityMeasurement meas<>;
  string config_fmt<>; /* Should contain a %d which is replaced by a config index */
  int conf_start;
  int conf_incr;
  int conf_lessthan;
  FixGaugeArg fix_gauge; /* Gauge fixing - Defaults to FIX_GAUGE_NONE */

  memfun GparityContractArg();
};
