/*
 * Please do not edit this file.
 * It was generated using PAB's VML system.
 */

#ifndef _PROP_ATTRIBUTE_ARG_H_RPCGEN
#define _PROP_ATTRIBUTE_ARG_H_RPCGEN

#include <config.h>
#include <util/vml/types.h>
#include <util/vml/vml.h>
#include <util/enum.h>
#include <util/defines.h>
CPS_START_NAMESPACE

enum AttrType {
	GENERIC_PROP_ATTR = 0,
	POINT_SOURCE_ATTR = 1,
	WALL_SOURCE_ATTR = 2,
	MOMENTUM_ATTR = 3,
	PROP_IO_ATTR = 4,
	GPARITY_FLAVOR_ATTR = 5,
	CG_ATTR = 6,
	GAUGE_FIX_ATTR = 7,
	MOM_COS_ATTR = 8,
	PROP_COMBINATION_ATTR = 9,
};
typedef enum AttrType AttrType;
extern struct vml_enum_map AttrType_map[];


#include <util/vml/vml_templates.h>
struct GenericPropAttrArg {
	char *tag;
	Float mass;
	BndCndType bc[4];
	   static AttrType getType (  ) ;
	   GenericPropAttrArg clone (  ) ;
	   void deep_copy(const GenericPropAttrArg &rhs);
	   void print(const std::string &prefix ="");
};
typedef struct GenericPropAttrArg GenericPropAttrArg;
template<> struct rpc_deepcopy<GenericPropAttrArg>{
	static void doit(GenericPropAttrArg &into, GenericPropAttrArg const &from);
};

#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<GenericPropAttrArg>{
	static void doit(GenericPropAttrArg const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
struct PointSourceAttrArg {
	int pos[4];
	   static AttrType getType (  ) ;
	   PointSourceAttrArg clone (  ) ;
	   void deep_copy(const PointSourceAttrArg &rhs);
	   void print(const std::string &prefix ="");
};
typedef struct PointSourceAttrArg PointSourceAttrArg;
template<> struct rpc_deepcopy<PointSourceAttrArg>{
	static void doit(PointSourceAttrArg &into, PointSourceAttrArg const &from);
};

#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<PointSourceAttrArg>{
	static void doit(PointSourceAttrArg const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
struct WallSourceAttrArg {
	int t;
	   static AttrType getType (  ) ;
	   WallSourceAttrArg clone (  ) ;
	   void print(const std::string &prefix ="");
};
typedef struct WallSourceAttrArg WallSourceAttrArg;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<WallSourceAttrArg>{
	static void doit(WallSourceAttrArg const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
struct MomentumAttrArg {
	int p[3];
	   static AttrType getType (  ) ;
	   MomentumAttrArg clone (  ) ;
	   void deep_copy(const MomentumAttrArg &rhs);
	   void print(const std::string &prefix ="");
};
typedef struct MomentumAttrArg MomentumAttrArg;
template<> struct rpc_deepcopy<MomentumAttrArg>{
	static void doit(MomentumAttrArg &into, MomentumAttrArg const &from);
};

#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<MomentumAttrArg>{
	static void doit(MomentumAttrArg const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
struct PropIOAttrArg {
	char *qio_filename;
	bool_t prop_on_disk;
	int save_to_disk;
	   static AttrType getType (  ) ;
	   PropIOAttrArg clone (  ) ;
	   void deep_copy(const PropIOAttrArg &rhs);
	   void print(const std::string &prefix ="");
};
typedef struct PropIOAttrArg PropIOAttrArg;
template<> struct rpc_deepcopy<PropIOAttrArg>{
	static void doit(PropIOAttrArg &into, PropIOAttrArg const &from);
};

#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<PropIOAttrArg>{
	static void doit(PropIOAttrArg const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
struct GparityFlavorAttrArg {
	int flavor;
	   static AttrType getType (  ) ;
	   GparityFlavorAttrArg clone (  ) ;
	   void print(const std::string &prefix ="");
};
typedef struct GparityFlavorAttrArg GparityFlavorAttrArg;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<GparityFlavorAttrArg>{
	static void doit(GparityFlavorAttrArg const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
struct CGAttrArg {
	int max_num_iter;
	Float stop_rsd;
	Float true_rsd;
	   static AttrType getType (  ) ;
	   CGAttrArg clone (  ) ;
	   void print(const std::string &prefix ="");
};
typedef struct CGAttrArg CGAttrArg;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<CGAttrArg>{
	static void doit(CGAttrArg const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
struct GaugeFixAttrArg {
	int gauge_fix_src;
	int gauge_fix_snk;
	   static AttrType getType (  ) ;
	   GaugeFixAttrArg clone (  ) ;
	   void print(const std::string &prefix ="");
};
typedef struct GaugeFixAttrArg GaugeFixAttrArg;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<GaugeFixAttrArg>{
	static void doit(GaugeFixAttrArg const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
struct MomCosAttrArg {
	   static AttrType getType (  ) ;
	   MomCosAttrArg clone (  ) ;
	   void print(const std::string &prefix ="");
};
typedef struct MomCosAttrArg MomCosAttrArg;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<MomCosAttrArg>{
	static void doit(MomCosAttrArg const &what, const std::string &prefix="" );
};


enum PropCombination {
	A_PLUS_B = 0,
	A_MINUS_B = 1,
};
typedef enum PropCombination PropCombination;
extern struct vml_enum_map PropCombination_map[];


#include <util/vml/vml_templates.h>
struct PropCombinationAttrArg {
	char *prop_A;
	char *prop_B;
	PropCombination combination;
	   static AttrType getType (  ) ;
	   PropCombinationAttrArg clone (  ) ;
	   void deep_copy(const PropCombinationAttrArg &rhs);
	   void print(const std::string &prefix ="");
};
typedef struct PropCombinationAttrArg PropCombinationAttrArg;
template<> struct rpc_deepcopy<PropCombinationAttrArg>{
	static void doit(PropCombinationAttrArg &into, PropCombinationAttrArg const &from);
};

#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<PropCombinationAttrArg>{
	static void doit(PropCombinationAttrArg const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
struct AttributeContainer {
	AttrType type;
	union {
		GenericPropAttrArg generic_prop_attr;
		PointSourceAttrArg point_source_attr;
		WallSourceAttrArg wall_source_attr;
		MomentumAttrArg momentum_attr;
		PropIOAttrArg prop_io_attr;
		GparityFlavorAttrArg gparity_flavor_attr;
		CGAttrArg cg_attr;
		GaugeFixAttrArg gauge_fix_attr;
		MomCosAttrArg mom_cos_attr;
		PropCombinationAttrArg prop_combination_attr;
	} AttributeContainer_u;
	   template <typename T> static AttrType type_map();
	   void deep_copy(const AttributeContainer &rhs);
	   void print(const std::string &prefix ="");
};
typedef struct AttributeContainer AttributeContainer;
template <typename T> AttrType AttributeContainer::type_map(){
	 return -1000;
}
template <> AttrType AttributeContainer::type_map<GenericPropAttrArg>();
template <> AttrType AttributeContainer::type_map<PointSourceAttrArg>();
template <> AttrType AttributeContainer::type_map<WallSourceAttrArg>();
template <> AttrType AttributeContainer::type_map<MomentumAttrArg>();
template <> AttrType AttributeContainer::type_map<PropIOAttrArg>();
template <> AttrType AttributeContainer::type_map<GparityFlavorAttrArg>();
template <> AttrType AttributeContainer::type_map<CGAttrArg>();
template <> AttrType AttributeContainer::type_map<GaugeFixAttrArg>();
template <> AttrType AttributeContainer::type_map<MomCosAttrArg>();
template <> AttrType AttributeContainer::type_map<PropCombinationAttrArg>();
template<> struct rpc_deepcopy<AttributeContainer>{
	static void doit(AttributeContainer &into, AttributeContainer const &from);
};

#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<AttributeContainer>{
	static void doit(AttributeContainer const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
class VML;
class PropagatorArg {
public:
	 bool Encode(char *filename,char *instance);
	 bool Decode(char *filename,char *instance);
	 bool Vml(VML *vmls,char *instance);
	GenericPropAttrArg generics;
	struct {
		u_int attributes_len;
		AttributeContainer *attributes_val;
	} attributes;
	   ~PropagatorArg (  ) ;
	   void deep_copy(const PropagatorArg &rhs);
	   void print(const std::string &prefix ="");
};
template<> struct rpc_deepcopy<PropagatorArg>{
	static void doit(PropagatorArg &into, PropagatorArg const &from);
};

#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<PropagatorArg>{
	static void doit(PropagatorArg const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
class VML;
class JobPropagatorArgs {
public:
	 bool Encode(char *filename,char *instance);
	 bool Decode(char *filename,char *instance);
	 bool Vml(VML *vmls,char *instance);
	struct {
		u_int props_len;
		PropagatorArg *props_val;
	} props;
	   ~JobPropagatorArgs (  ) ;
	   void deep_copy(const JobPropagatorArgs &rhs);
	   void print(const std::string &prefix ="");
};
template<> struct rpc_deepcopy<JobPropagatorArgs>{
	static void doit(JobPropagatorArgs &into, JobPropagatorArgs const &from);
};

#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<JobPropagatorArgs>{
	static void doit(JobPropagatorArgs const &what, const std::string &prefix="" );
};


/* the xdr functions */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__STDC__) || defined(__cplusplus)
extern  bool_t vml_AttrType (VML *, char *instance, AttrType*);
extern  bool_t vml_GenericPropAttrArg (VML *, char *instance, GenericPropAttrArg*);
extern  bool_t vml_PointSourceAttrArg (VML *, char *instance, PointSourceAttrArg*);
extern  bool_t vml_WallSourceAttrArg (VML *, char *instance, WallSourceAttrArg*);
extern  bool_t vml_MomentumAttrArg (VML *, char *instance, MomentumAttrArg*);
extern  bool_t vml_PropIOAttrArg (VML *, char *instance, PropIOAttrArg*);
extern  bool_t vml_GparityFlavorAttrArg (VML *, char *instance, GparityFlavorAttrArg*);
extern  bool_t vml_CGAttrArg (VML *, char *instance, CGAttrArg*);
extern  bool_t vml_GaugeFixAttrArg (VML *, char *instance, GaugeFixAttrArg*);
extern  bool_t vml_MomCosAttrArg (VML *, char *instance, MomCosAttrArg*);
extern  bool_t vml_PropCombination (VML *, char *instance, PropCombination*);
extern  bool_t vml_PropCombinationAttrArg (VML *, char *instance, PropCombinationAttrArg*);
extern  bool_t vml_AttributeContainer (VML *, char *instance, AttributeContainer*);
extern  bool_t vml_PropagatorArg (VML *, char *instance, PropagatorArg*);
extern  bool_t vml_JobPropagatorArgs (VML *, char *instance, JobPropagatorArgs*);

#else /* K&R C */
extern  bool_t vml_AttrType (VML *, char *instance, AttrType*);
extern  bool_t vml_GenericPropAttrArg (VML *, char *instance, GenericPropAttrArg*);
extern  bool_t vml_PointSourceAttrArg (VML *, char *instance, PointSourceAttrArg*);
extern  bool_t vml_WallSourceAttrArg (VML *, char *instance, WallSourceAttrArg*);
extern  bool_t vml_MomentumAttrArg (VML *, char *instance, MomentumAttrArg*);
extern  bool_t vml_PropIOAttrArg (VML *, char *instance, PropIOAttrArg*);
extern  bool_t vml_GparityFlavorAttrArg (VML *, char *instance, GparityFlavorAttrArg*);
extern  bool_t vml_CGAttrArg (VML *, char *instance, CGAttrArg*);
extern  bool_t vml_GaugeFixAttrArg (VML *, char *instance, GaugeFixAttrArg*);
extern  bool_t vml_MomCosAttrArg (VML *, char *instance, MomCosAttrArg*);
extern  bool_t vml_PropCombination (VML *, char *instance, PropCombination*);
extern  bool_t vml_PropCombinationAttrArg (VML *, char *instance, PropCombinationAttrArg*);
extern  bool_t vml_AttributeContainer (VML *, char *instance, AttributeContainer*);
extern  bool_t vml_PropagatorArg (VML *, char *instance, PropagatorArg*);
extern  bool_t vml_JobPropagatorArgs (VML *, char *instance, JobPropagatorArgs*);

#endif /* K&R C */

#ifdef __cplusplus
}
#endif
CPS_END_NAMESPACE

#endif /* !_PROP_ATTRIBUTE_ARG_H_RPCGEN */
