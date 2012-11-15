/*  Stuff   */

/*include <alg/pooh.h>;
include "alg/test.h";*/

enum AttrType {
  GENERIC_PROP_ATTR,
  POINT_SOURCE_ATTR,
  WALL_SOURCE_ATTR,
  MOMENTUM_ATTR,
  PROP_IO_ATTR,
  GPARITY_FLAVOR_ATTR,
  CG_ATTR,
  GAUGE_FIX_ATTR,
  MOM_COS_ATTR };

struct GenericPropAttrArg{
  string tag<>;
  Float mass;
  BndCndType bc[4]; /*Currently does nothing, BCs are set globally for the job. Could try creating a new function in GJP to reinitialise the boundary conditions*/

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

union AttributeContainer{
switch(AttrType type){
 case GENERIC_PROP_ATTR:
   GenericPropAttrArg generic_prop_attr;
 case POINT_SOURCE_ATTR:
   PointSourceAttrArg point_source_attr;
 case WALL_SOURCE_ATTR:
   WallSourceAttrArg wall_source_attr;
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
