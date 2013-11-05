/*  Stuff   */

/*include <alg/pooh.h>;
include "alg/test.h";*/

enum AttrType {
  GENERIC_PROP_ATTR,
  POINT_SOURCE_ATTR,
  WALL_SOURCE_ATTR,
  VOLUME_SOURCE_ATTR,
  MOMENTUM_ATTR,
  PROP_IO_ATTR,
  GPARITY_FLAVOR_ATTR,
  CG_ATTR,
  GAUGE_FIX_ATTR,
  MOM_COS_ATTR,
  PROP_COMBINATION_ATTR,
  GPARITY_OTHER_FLAV_PROP_ATTR,
  TWISTED_BC_ATTR,
  STORE_MIDPROP_ATTR
};

struct GenericPropAttrArg{
  string tag<>;
  Float mass;
  BndCndType bc[4];

  memfun static AttrType getType();
  memfun GenericPropAttrArg clone();
  rpccommand GENERATE_DEEPCOPY_METHOD;
  rpccommand GENERATE_PRINT_METHOD;
};
struct PointSourceAttrArg {
  int pos[4];
  memfun static AttrType getType();
  memfun PointSourceAttrArg clone();
  rpccommand GENERATE_DEEPCOPY_METHOD;
  rpccommand GENERATE_PRINT_METHOD;
};
struct WallSourceAttrArg {
  int t;
  memfun static AttrType getType();
  memfun WallSourceAttrArg clone();
  rpccommand GENERATE_PRINT_METHOD;
};
struct VolumeSourceAttrArg {
  memfun static AttrType getType();
  memfun VolumeSourceAttrArg clone();
  rpccommand GENERATE_PRINT_METHOD;
};
struct  MomentumAttrArg {
  int p[3];
  memfun static AttrType getType();
  memfun MomentumAttrArg clone();
  rpccommand GENERATE_DEEPCOPY_METHOD;
  rpccommand GENERATE_PRINT_METHOD;
};
struct PropIOAttrArg {
  string qio_filename<>;
  bool prop_on_disk;
  int save_to_disk;
  memfun static AttrType getType();
  memfun PropIOAttrArg clone();
  rpccommand GENERATE_DEEPCOPY_METHOD;
  rpccommand GENERATE_PRINT_METHOD;
};
struct GparityFlavorAttrArg {
  int flavor;
  memfun static AttrType getType();
  memfun GparityFlavorAttrArg clone();
  rpccommand GENERATE_PRINT_METHOD;
};
struct CGAttrArg {
  int max_num_iter;
  Float stop_rsd;
  Float true_rsd;
  memfun static AttrType getType();
  memfun CGAttrArg clone();
  rpccommand GENERATE_PRINT_METHOD;
};
struct GaugeFixAttrArg{
  int gauge_fix_src;
  int gauge_fix_snk;
  memfun static AttrType getType();
  memfun GaugeFixAttrArg clone();
  rpccommand GENERATE_PRINT_METHOD;
};
struct MomCosAttrArg{
  memfun static AttrType getType();
  memfun MomCosAttrArg clone();
  rpccommand GENERATE_PRINT_METHOD;
};
struct GparityOtherFlavPropAttrArg{ /*Give the tag of a propagator with the same source properties but the other G-parity flavour index*/
  string tag<>;

  memfun static AttrType getType();
  memfun GparityOtherFlavPropAttrArg clone();
  rpccommand GENERATE_DEEPCOPY_METHOD;
  rpccommand GENERATE_PRINT_METHOD;
};
struct  TwistedBcAttrArg {
  int theta[3];
  memfun static AttrType getType();
  memfun TwistedBcAttrArg clone();
  rpccommand GENERATE_DEEPCOPY_METHOD;
  rpccommand GENERATE_PRINT_METHOD;
};
struct StoreMidpropAttrArg{
  memfun static AttrType getType();
  memfun StoreMidpropAttrArg clone();
  rpccommand GENERATE_PRINT_METHOD;
};


enum PropCombination {
  A_PLUS_B,
  A_MINUS_B };

struct PropCombinationAttrArg{	
 string prop_A<>;
 string prop_B<>;
 PropCombination combination;
 memfun static AttrType getType();
 memfun PropCombinationAttrArg clone();
 rpccommand GENERATE_DEEPCOPY_METHOD;
 rpccommand GENERATE_PRINT_METHOD;
};


union AttributeContainer{
switch(AttrType type){
 case GENERIC_PROP_ATTR:
   GenericPropAttrArg generic_prop_attr;
 case POINT_SOURCE_ATTR:
   PointSourceAttrArg point_source_attr;
 case WALL_SOURCE_ATTR:
   WallSourceAttrArg wall_source_attr;
 case VOLUME_SOURCE_ATTR:
   VolumeSourceAttrArg volume_source_attr;
 case MOMENTUM_ATTR:
   MomentumAttrArg momentum_attr;
 case PROP_IO_ATTR:
   PropIOAttrArg prop_io_attr;
 case GPARITY_FLAVOR_ATTR:
   GparityFlavorAttrArg gparity_flavor_attr;
 case CG_ATTR:
   CGAttrArg cg_attr;
 case GAUGE_FIX_ATTR:
   GaugeFixAttrArg gauge_fix_attr;
 case MOM_COS_ATTR:
   MomCosAttrArg mom_cos_attr;
 case PROP_COMBINATION_ATTR:
   PropCombinationAttrArg prop_combination_attr;
 case GPARITY_OTHER_FLAV_PROP_ATTR:
   GparityOtherFlavPropAttrArg gparity_other_flav_prop_attr;
 case TWISTED_BC_ATTR:
   TwistedBcAttrArg twisted_bc_attr;
 case STORE_MIDPROP_ATTR:
   StoreMidpropAttrArg store_midprop_attr;
}
  rpccommand GENERATE_UNION_TYPEMAP;
  rpccommand GENERATE_DEEPCOPY_METHOD;
  rpccommand GENERATE_PRINT_METHOD;
};

class PropagatorArg {
  GenericPropAttrArg generics;
  AttributeContainer attributes<>;

  memfun ~PropagatorArg();
  rpccommand GENERATE_DEEPCOPY_METHOD;
  rpccommand GENERATE_PRINT_METHOD;
};

class JobPropagatorArgs {
  PropagatorArg props<>;

  memfun ~JobPropagatorArgs();
  rpccommand GENERATE_DEEPCOPY_METHOD;
  rpccommand GENERATE_PRINT_METHOD;
};
