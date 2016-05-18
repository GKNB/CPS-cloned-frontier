/*
 * Please do not edit this file.
 * It was generated using PAB's VML system.
 */

#ifndef _GPARITY_CONTRACT_ARG_H_RPCGEN
#define _GPARITY_CONTRACT_ARG_H_RPCGEN

#include <config.h>
#include <util/vml/types.h>
#include <util/vml/vml.h>
#include <util/enum.h>
#include <util/defines.h>
CPS_START_NAMESPACE

enum ContractionType {
	CONTRACTION_TYPE_LL_MESONS = 0,
	CONTRACTION_TYPE_HL_MESONS = 1,
	CONTRACTION_TYPE_O_VV_P_AA = 2,
	CONTRACTION_TYPE_ALL_BILINEARS = 3,
	CONTRACTION_TYPE_ALL_WALLSINK_BILINEARS_SPECIFIC_MOMENTUM = 4,
	CONTRACTION_TYPE_FOURIER_PROP = 5,
	CONTRACTION_TYPE_BILINEAR_VERTEX = 6,
	CONTRACTION_TYPE_QUADRILINEAR_VERTEX = 7,
	CONTRACTION_TYPE_TOPOLOGICAL_CHARGE = 8,
	CONTRACTION_TYPE_MRES = 9,
	CONTRACTION_TYPE_A2A_BILINEAR = 10,
	CONTRACTION_TYPE_WILSON_FLOW = 11,
	CONTRACTION_TYPE_K_TO_PIPI = 12,
};
typedef enum ContractionType ContractionType;
extern struct vml_enum_map ContractionType_map[];


#include <util/vml/vml_templates.h>
struct ContractionTypeLLMesons {
	char *prop_L;
	int sink_mom[3];
	char *file;
	   void print(const std::string &prefix ="");
	   void deep_copy(const ContractionTypeLLMesons &rhs);
};
typedef struct ContractionTypeLLMesons ContractionTypeLLMesons;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<ContractionTypeLLMesons>{
	static void doit(ContractionTypeLLMesons const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<ContractionTypeLLMesons>{
	static void doit(ContractionTypeLLMesons &into, ContractionTypeLLMesons const &from);
};



#include <util/vml/vml_templates.h>
struct ContractionTypeHLMesons {
	char *prop_H;
	char *prop_L;
	int sink_mom[3];
	char *file;
	   void print(const std::string &prefix ="");
	   void deep_copy(const ContractionTypeHLMesons &rhs);
};
typedef struct ContractionTypeHLMesons ContractionTypeHLMesons;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<ContractionTypeHLMesons>{
	static void doit(ContractionTypeHLMesons const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<ContractionTypeHLMesons>{
	static void doit(ContractionTypeHLMesons &into, ContractionTypeHLMesons const &from);
};



#include <util/vml/vml_templates.h>
struct ContractionTypeOVVpAA {
	char *prop_H_t0;
	char *prop_L_t0;
	char *prop_H_t1;
	char *prop_L_t1;
	char *file;
	   void print(const std::string &prefix ="");
	   void deep_copy(const ContractionTypeOVVpAA &rhs);
};
typedef struct ContractionTypeOVVpAA ContractionTypeOVVpAA;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<ContractionTypeOVVpAA>{
	static void doit(ContractionTypeOVVpAA const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<ContractionTypeOVVpAA>{
	static void doit(ContractionTypeOVVpAA &into, ContractionTypeOVVpAA const &from);
};



#include <util/vml/vml_templates.h>
struct MomArg {
	Float p[3];
	   void print(const std::string &prefix ="");
	   void deep_copy(const MomArg &rhs);
};
typedef struct MomArg MomArg;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<MomArg>{
	static void doit(MomArg const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<MomArg>{
	static void doit(MomArg &into, MomArg const &from);
};



#include <util/vml/vml_templates.h>
struct MomPairArg {
	Float p1[3];
	Float p2[3];
	   void print(const std::string &prefix ="");
	   void deep_copy(const MomPairArg &rhs);
};
typedef struct MomPairArg MomPairArg;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<MomPairArg>{
	static void doit(MomPairArg const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<MomPairArg>{
	static void doit(MomPairArg &into, MomPairArg const &from);
};


enum PropSuperscript {
	OpNone = 0,
	OpTranspose = 1,
	OpConj = 2,
	OpDagger = 3,
	OpFlipMomentum = 4,
	OpTransposeFlipMomentum = 5,
	OpConjFlipMomentum = 6,
	OpDaggerFlipMomentum = 7,
};
typedef enum PropSuperscript PropSuperscript;
extern struct vml_enum_map PropSuperscript_map[];


#include <util/vml/vml_templates.h>
struct ContractionTypeAllBilinears {
	char *prop_1;
	char *prop_2;
	PropSuperscript op1;
	PropSuperscript op2;
	struct {
		u_int momenta_len;
		MomArg *momenta_val;
	} momenta;
	char *file;
	   void print(const std::string &prefix ="");
	   void deep_copy(const ContractionTypeAllBilinears &rhs);
};
typedef struct ContractionTypeAllBilinears ContractionTypeAllBilinears;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<ContractionTypeAllBilinears>{
	static void doit(ContractionTypeAllBilinears const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<ContractionTypeAllBilinears>{
	static void doit(ContractionTypeAllBilinears &into, ContractionTypeAllBilinears const &from);
};



#include <util/vml/vml_templates.h>
struct ContractionTypeAllWallSinkBilinearsSpecificMomentum {
	char *prop_1;
	char *prop_2;
	PropSuperscript op1;
	PropSuperscript op2;
	struct {
		u_int momenta_len;
		MomPairArg *momenta_val;
	} momenta;
	int cosine_sink;
	char *file;
	   void print(const std::string &prefix ="");
	   void deep_copy(const ContractionTypeAllWallSinkBilinearsSpecificMomentum &rhs);
};
typedef struct ContractionTypeAllWallSinkBilinearsSpecificMomentum ContractionTypeAllWallSinkBilinearsSpecificMomentum;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<ContractionTypeAllWallSinkBilinearsSpecificMomentum>{
	static void doit(ContractionTypeAllWallSinkBilinearsSpecificMomentum const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<ContractionTypeAllWallSinkBilinearsSpecificMomentum>{
	static void doit(ContractionTypeAllWallSinkBilinearsSpecificMomentum &into, ContractionTypeAllWallSinkBilinearsSpecificMomentum const &from);
};



#include <util/vml/vml_templates.h>
struct ContractionTypeFourierProp {
	char *prop;
	int gauge_fix;
	struct {
		u_int momenta_len;
		MomArg *momenta_val;
	} momenta;
	char *file;
	   void print(const std::string &prefix ="");
	   void deep_copy(const ContractionTypeFourierProp &rhs);
};
typedef struct ContractionTypeFourierProp ContractionTypeFourierProp;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<ContractionTypeFourierProp>{
	static void doit(ContractionTypeFourierProp const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<ContractionTypeFourierProp>{
	static void doit(ContractionTypeFourierProp &into, ContractionTypeFourierProp const &from);
};



#include <util/vml/vml_templates.h>
struct ContractionTypeBilinearVertex {
	char *prop_1;
	char *prop_2;
	struct {
		u_int momenta_len;
		MomArg *momenta_val;
	} momenta;
	char *file;
	   void print(const std::string &prefix ="");
	   void deep_copy(const ContractionTypeBilinearVertex &rhs);
};
typedef struct ContractionTypeBilinearVertex ContractionTypeBilinearVertex;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<ContractionTypeBilinearVertex>{
	static void doit(ContractionTypeBilinearVertex const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<ContractionTypeBilinearVertex>{
	static void doit(ContractionTypeBilinearVertex &into, ContractionTypeBilinearVertex const &from);
};



#include <util/vml/vml_templates.h>
struct QuadrilinearSpinStructure {
	struct {
		u_int Gamma1_len;
		int *Gamma1_val;
	} Gamma1;
	struct {
		u_int Gamma2_len;
		int *Gamma2_val;
	} Gamma2;
	struct {
		u_int Sigma1_len;
		int *Sigma1_val;
	} Sigma1;
	struct {
		u_int Sigma2_len;
		int *Sigma2_val;
	} Sigma2;
	   void print(const std::string &prefix ="");
};
typedef struct QuadrilinearSpinStructure QuadrilinearSpinStructure;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<QuadrilinearSpinStructure>{
	static void doit(QuadrilinearSpinStructure const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
struct ContractionTypeQuadrilinearVertex {
	char *prop_1;
	char *prop_2;
	char *prop_3;
	char *prop_4;
	struct {
		u_int momenta_len;
		MomArg *momenta_val;
	} momenta;
	char *file;
	struct {
		u_int spin_structs_len;
		QuadrilinearSpinStructure *spin_structs_val;
	} spin_structs;
	   void print(const std::string &prefix ="");
	   void deep_copy(const ContractionTypeQuadrilinearVertex &rhs);
};
typedef struct ContractionTypeQuadrilinearVertex ContractionTypeQuadrilinearVertex;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<ContractionTypeQuadrilinearVertex>{
	static void doit(ContractionTypeQuadrilinearVertex const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<ContractionTypeQuadrilinearVertex>{
	static void doit(ContractionTypeQuadrilinearVertex &into, ContractionTypeQuadrilinearVertex const &from);
};



#include <util/vml/vml_templates.h>
struct ContractionTypeTopologicalCharge {
	int n_ape_smearing_cycles;
	int ape_smear_su3_project;
	Float ape_su3_proj_tolerance;
	int ape_orthog;
	Float ape_coef;
	char *file;
	   void print(const std::string &prefix ="");
	   void deep_copy(const ContractionTypeTopologicalCharge &rhs);
};
typedef struct ContractionTypeTopologicalCharge ContractionTypeTopologicalCharge;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<ContractionTypeTopologicalCharge>{
	static void doit(ContractionTypeTopologicalCharge const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<ContractionTypeTopologicalCharge>{
	static void doit(ContractionTypeTopologicalCharge &into, ContractionTypeTopologicalCharge const &from);
};



#include <util/vml/vml_templates.h>
struct ContractionTypeMres {
	char *prop;
	char *file;
	   void print(const std::string &prefix ="");
	   void deep_copy(const ContractionTypeMres &rhs);
};
typedef struct ContractionTypeMres ContractionTypeMres;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<ContractionTypeMres>{
	static void doit(ContractionTypeMres const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<ContractionTypeMres>{
	static void doit(ContractionTypeMres &into, ContractionTypeMres const &from);
};



#include <util/vml/vml_templates.h>
struct ContractionTypeWilsonFlow {
	int n_steps;
	Float time_step;
	char *file;
	   void print(const std::string &prefix ="");
	   void deep_copy(const ContractionTypeWilsonFlow &rhs);
};
typedef struct ContractionTypeWilsonFlow ContractionTypeWilsonFlow;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<ContractionTypeWilsonFlow>{
	static void doit(ContractionTypeWilsonFlow const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<ContractionTypeWilsonFlow>{
	static void doit(ContractionTypeWilsonFlow &into, ContractionTypeWilsonFlow const &from);
};


enum A2ASmearingType {
	BOX_3D_SMEARING = 0,
	EXPONENTIAL_3D_SMEARING = 1,
};
typedef enum A2ASmearingType A2ASmearingType;
extern struct vml_enum_map A2ASmearingType_map[];


#include <util/vml/vml_templates.h>
struct Box3dSmearing {
	int side_length;
	   void print(const std::string &prefix ="");
};
typedef struct Box3dSmearing Box3dSmearing;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<Box3dSmearing>{
	static void doit(Box3dSmearing const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
struct Exponential3dSmearing {
	Float radius;
	   void print(const std::string &prefix ="");
};
typedef struct Exponential3dSmearing Exponential3dSmearing;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<Exponential3dSmearing>{
	static void doit(Exponential3dSmearing const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
struct A2ASmearing {
	A2ASmearingType type;
	union {
		Box3dSmearing box_3d_smearing;
		Exponential3dSmearing exponential_3d_smearing;
	} A2ASmearing_u;
	   template <typename T> static A2ASmearingType type_map();
	   void deep_copy(const A2ASmearing &rhs);
	   void print(const std::string &prefix ="");
};
typedef struct A2ASmearing A2ASmearing;
template <typename T> A2ASmearingType A2ASmearing::type_map(){
	 return -1000;
}
template <> A2ASmearingType A2ASmearing::type_map<Box3dSmearing>();
template <> A2ASmearingType A2ASmearing::type_map<Exponential3dSmearing>();
template<> struct rpc_deepcopy<A2ASmearing>{
	static void doit(A2ASmearing &into, A2ASmearing const &from);
};

#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<A2ASmearing>{
	static void doit(A2ASmearing const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
struct MatIdxAndCoeff {
	int idx;
	Float coeff;
	   void print(const std::string &prefix ="");
};
typedef struct MatIdxAndCoeff MatIdxAndCoeff;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<MatIdxAndCoeff>{
	static void doit(MatIdxAndCoeff const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
struct ContractionTypeA2ABilinear {
	char *prop_src_snk;
	char *prop_snk_src;
	A2ASmearing source_smearing;
	A2ASmearing sink_smearing;
	struct {
		u_int source_spin_matrix_len;
		MatIdxAndCoeff *source_spin_matrix_val;
	} source_spin_matrix;
	struct {
		u_int sink_spin_matrix_len;
		MatIdxAndCoeff *sink_spin_matrix_val;
	} sink_spin_matrix;
	struct {
		u_int source_flavor_matrix_len;
		MatIdxAndCoeff *source_flavor_matrix_val;
	} source_flavor_matrix;
	struct {
		u_int sink_flavor_matrix_len;
		MatIdxAndCoeff *sink_flavor_matrix_val;
	} sink_flavor_matrix;
	char *file;
	   void print(const std::string &prefix ="");
	   void deep_copy(const ContractionTypeA2ABilinear &rhs);
};
typedef struct ContractionTypeA2ABilinear ContractionTypeA2ABilinear;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<ContractionTypeA2ABilinear>{
	static void doit(ContractionTypeA2ABilinear const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<ContractionTypeA2ABilinear>{
	static void doit(ContractionTypeA2ABilinear &into, ContractionTypeA2ABilinear const &from);
};



#include <util/vml/vml_templates.h>
struct ContractionTypeKtoPiPi {
	char *prop_L;
	char *prop_H;
	MomPairArg p_qpi1;
	MomPairArg p_qpi2;
	Float p_qK[3];
	int gparity_use_transconv_props;
	A2ASmearing pion_source;
	A2ASmearing kaon_source;
	int t_sep_pi_k;
	int t_sep_pion;
	char *file;
	   void print(const std::string &prefix ="");
	   void deep_copy(const ContractionTypeKtoPiPi &rhs);
};
typedef struct ContractionTypeKtoPiPi ContractionTypeKtoPiPi;
#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<ContractionTypeKtoPiPi>{
	static void doit(ContractionTypeKtoPiPi const &what, const std::string &prefix="" );
};

template<> struct rpc_deepcopy<ContractionTypeKtoPiPi>{
	static void doit(ContractionTypeKtoPiPi &into, ContractionTypeKtoPiPi const &from);
};



#include <util/vml/vml_templates.h>
struct GparityMeasurement {
	ContractionType type;
	union {
		ContractionTypeLLMesons contraction_type_ll_mesons;
		ContractionTypeHLMesons contraction_type_hl_mesons;
		ContractionTypeOVVpAA contraction_type_o_vv_p_aa;
		ContractionTypeAllBilinears contraction_type_all_bilinears;
		ContractionTypeAllWallSinkBilinearsSpecificMomentum contraction_type_all_wallsink_bilinears_specific_momentum;
		ContractionTypeFourierProp contraction_type_fourier_prop;
		ContractionTypeBilinearVertex contraction_type_bilinear_vertex;
		ContractionTypeQuadrilinearVertex contraction_type_quadrilinear_vertex;
		ContractionTypeTopologicalCharge contraction_type_topological_charge;
		ContractionTypeMres contraction_type_mres;
		ContractionTypeA2ABilinear contraction_type_a2a_bilinear;
		ContractionTypeWilsonFlow contraction_type_wilson_flow;
		ContractionTypeKtoPiPi contraction_type_k_to_pipi;
	} GparityMeasurement_u;
	   template <typename T> static ContractionType type_map();
	   void deep_copy(const GparityMeasurement &rhs);
	   void print(const std::string &prefix ="");
};
typedef struct GparityMeasurement GparityMeasurement;
template <typename T> ContractionType GparityMeasurement::type_map(){
	 return -1000;
}
template <> ContractionType GparityMeasurement::type_map<ContractionTypeLLMesons>();
template <> ContractionType GparityMeasurement::type_map<ContractionTypeHLMesons>();
template <> ContractionType GparityMeasurement::type_map<ContractionTypeOVVpAA>();
template <> ContractionType GparityMeasurement::type_map<ContractionTypeAllBilinears>();
template <> ContractionType GparityMeasurement::type_map<ContractionTypeAllWallSinkBilinearsSpecificMomentum>();
template <> ContractionType GparityMeasurement::type_map<ContractionTypeFourierProp>();
template <> ContractionType GparityMeasurement::type_map<ContractionTypeBilinearVertex>();
template <> ContractionType GparityMeasurement::type_map<ContractionTypeQuadrilinearVertex>();
template <> ContractionType GparityMeasurement::type_map<ContractionTypeTopologicalCharge>();
template <> ContractionType GparityMeasurement::type_map<ContractionTypeMres>();
template <> ContractionType GparityMeasurement::type_map<ContractionTypeA2ABilinear>();
template <> ContractionType GparityMeasurement::type_map<ContractionTypeWilsonFlow>();
template <> ContractionType GparityMeasurement::type_map<ContractionTypeKtoPiPi>();
template<> struct rpc_deepcopy<GparityMeasurement>{
	static void doit(GparityMeasurement &into, GparityMeasurement const &from);
};

#ifndef _USE_STDLIB
#error "Cannot generate rpc_print commands without the standard library"
#endif
template<> struct rpc_print<GparityMeasurement>{
	static void doit(GparityMeasurement const &what, const std::string &prefix="" );
};



#include <util/vml/vml_templates.h>
class VML;
class GparityContractArg {
public:
	 bool Encode(char *filename,char *instance);
	 bool Decode(char *filename,char *instance);
	 bool Vml(VML *vmls,char *instance);
	struct {
		u_int meas_len;
		GparityMeasurement *meas_val;
	} meas;
	char *config_fmt;
	int conf_start;
	int conf_incr;
	int conf_lessthan;
	FixGaugeArg fix_gauge;
	   GparityContractArg (  ) ;
	   void deep_copy(const GparityContractArg &rhs);
};
template<> struct rpc_deepcopy<GparityContractArg>{
	static void doit(GparityContractArg &into, GparityContractArg const &from);
};



#include <util/vml/vml_templates.h>
class VML;
class GparityAMAarg {
public:
	 bool Encode(char *filename,char *instance);
	 bool Decode(char *filename,char *instance);
	 bool Vml(VML *vmls,char *instance);
	struct {
		u_int bilinear_args_len;
		ContractionTypeAllBilinears *bilinear_args_val;
	} bilinear_args;
	struct {
		u_int exact_solve_timeslices_len;
		int *exact_solve_timeslices_val;
	} exact_solve_timeslices;
	Float exact_precision;
	Float sloppy_precision;
	char *config_fmt;
	int conf_start;
	int conf_incr;
	int conf_lessthan;
	FixGaugeArg fix_gauge;
	   void deep_copy(const GparityAMAarg &rhs);
};
template<> struct rpc_deepcopy<GparityAMAarg>{
	static void doit(GparityAMAarg &into, GparityAMAarg const &from);
};



#include <util/vml/vml_templates.h>
class VML;
class GparityAMAbilBKarg {
public:
	 bool Encode(char *filename,char *instance);
	 bool Decode(char *filename,char *instance);
	 bool Vml(VML *vmls,char *instance);
	struct {
		u_int exact_solve_timeslices_len;
		int *exact_solve_timeslices_val;
	} exact_solve_timeslices;
	Float exact_precision;
	Float sloppy_precision;
	Float ml;
	Float mh;
	char *config_fmt;
	int conf_start;
	int conf_incr;
	int conf_lessthan;
	FixGaugeArg fix_gauge;
	   void deep_copy(const GparityAMAbilBKarg &rhs);
};
template<> struct rpc_deepcopy<GparityAMAbilBKarg>{
	static void doit(GparityAMAbilBKarg &into, GparityAMAbilBKarg const &from);
};



#include <util/vml/vml_templates.h>
class VML;
class GparityAMAarg2 {
public:
	 bool Encode(char *filename,char *instance);
	 bool Decode(char *filename,char *instance);
	 bool Vml(VML *vmls,char *instance);
	struct {
		u_int exact_solve_timeslices_len;
		int *exact_solve_timeslices_val;
	} exact_solve_timeslices;
	struct {
		u_int sloppy_solve_timeslices_len;
		int *sloppy_solve_timeslices_val;
	} sloppy_solve_timeslices;
	struct {
		u_int bk_tseps_len;
		int *bk_tseps_val;
	} bk_tseps;
	Float exact_precision;
	Float sloppy_precision;
	Float ml;
	Float mh;
	char *results_dir;
	char *config_fmt;
	char *rng_fmt;
	int conf_start;
	int conf_incr;
	int conf_lessthan;
	FixGaugeArg fix_gauge;
	   void deep_copy(const GparityAMAarg2 &rhs);
};
template<> struct rpc_deepcopy<GparityAMAarg2>{
	static void doit(GparityAMAarg2 &into, GparityAMAarg2 const &from);
};


/* the xdr functions */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__STDC__) || defined(__cplusplus)
extern  bool_t vml_ContractionType (VML *, char *instance, ContractionType*);
extern  bool_t vml_ContractionTypeLLMesons (VML *, char *instance, ContractionTypeLLMesons*);
extern  bool_t vml_ContractionTypeHLMesons (VML *, char *instance, ContractionTypeHLMesons*);
extern  bool_t vml_ContractionTypeOVVpAA (VML *, char *instance, ContractionTypeOVVpAA*);
extern  bool_t vml_MomArg (VML *, char *instance, MomArg*);
extern  bool_t vml_MomPairArg (VML *, char *instance, MomPairArg*);
extern  bool_t vml_PropSuperscript (VML *, char *instance, PropSuperscript*);
extern  bool_t vml_ContractionTypeAllBilinears (VML *, char *instance, ContractionTypeAllBilinears*);
extern  bool_t vml_ContractionTypeAllWallSinkBilinearsSpecificMomentum (VML *, char *instance, ContractionTypeAllWallSinkBilinearsSpecificMomentum*);
extern  bool_t vml_ContractionTypeFourierProp (VML *, char *instance, ContractionTypeFourierProp*);
extern  bool_t vml_ContractionTypeBilinearVertex (VML *, char *instance, ContractionTypeBilinearVertex*);
extern  bool_t vml_QuadrilinearSpinStructure (VML *, char *instance, QuadrilinearSpinStructure*);
extern  bool_t vml_ContractionTypeQuadrilinearVertex (VML *, char *instance, ContractionTypeQuadrilinearVertex*);
extern  bool_t vml_ContractionTypeTopologicalCharge (VML *, char *instance, ContractionTypeTopologicalCharge*);
extern  bool_t vml_ContractionTypeMres (VML *, char *instance, ContractionTypeMres*);
extern  bool_t vml_ContractionTypeWilsonFlow (VML *, char *instance, ContractionTypeWilsonFlow*);
extern  bool_t vml_A2ASmearingType (VML *, char *instance, A2ASmearingType*);
extern  bool_t vml_Box3dSmearing (VML *, char *instance, Box3dSmearing*);
extern  bool_t vml_Exponential3dSmearing (VML *, char *instance, Exponential3dSmearing*);
extern  bool_t vml_A2ASmearing (VML *, char *instance, A2ASmearing*);
extern  bool_t vml_MatIdxAndCoeff (VML *, char *instance, MatIdxAndCoeff*);
extern  bool_t vml_ContractionTypeA2ABilinear (VML *, char *instance, ContractionTypeA2ABilinear*);
extern  bool_t vml_ContractionTypeKtoPiPi (VML *, char *instance, ContractionTypeKtoPiPi*);
extern  bool_t vml_GparityMeasurement (VML *, char *instance, GparityMeasurement*);
extern  bool_t vml_GparityContractArg (VML *, char *instance, GparityContractArg*);
extern  bool_t vml_GparityAMAarg (VML *, char *instance, GparityAMAarg*);
extern  bool_t vml_GparityAMAbilBKarg (VML *, char *instance, GparityAMAbilBKarg*);
extern  bool_t vml_GparityAMAarg2 (VML *, char *instance, GparityAMAarg2*);

#else /* K&R C */
extern  bool_t vml_ContractionType (VML *, char *instance, ContractionType*);
extern  bool_t vml_ContractionTypeLLMesons (VML *, char *instance, ContractionTypeLLMesons*);
extern  bool_t vml_ContractionTypeHLMesons (VML *, char *instance, ContractionTypeHLMesons*);
extern  bool_t vml_ContractionTypeOVVpAA (VML *, char *instance, ContractionTypeOVVpAA*);
extern  bool_t vml_MomArg (VML *, char *instance, MomArg*);
extern  bool_t vml_MomPairArg (VML *, char *instance, MomPairArg*);
extern  bool_t vml_PropSuperscript (VML *, char *instance, PropSuperscript*);
extern  bool_t vml_ContractionTypeAllBilinears (VML *, char *instance, ContractionTypeAllBilinears*);
extern  bool_t vml_ContractionTypeAllWallSinkBilinearsSpecificMomentum (VML *, char *instance, ContractionTypeAllWallSinkBilinearsSpecificMomentum*);
extern  bool_t vml_ContractionTypeFourierProp (VML *, char *instance, ContractionTypeFourierProp*);
extern  bool_t vml_ContractionTypeBilinearVertex (VML *, char *instance, ContractionTypeBilinearVertex*);
extern  bool_t vml_QuadrilinearSpinStructure (VML *, char *instance, QuadrilinearSpinStructure*);
extern  bool_t vml_ContractionTypeQuadrilinearVertex (VML *, char *instance, ContractionTypeQuadrilinearVertex*);
extern  bool_t vml_ContractionTypeTopologicalCharge (VML *, char *instance, ContractionTypeTopologicalCharge*);
extern  bool_t vml_ContractionTypeMres (VML *, char *instance, ContractionTypeMres*);
extern  bool_t vml_ContractionTypeWilsonFlow (VML *, char *instance, ContractionTypeWilsonFlow*);
extern  bool_t vml_A2ASmearingType (VML *, char *instance, A2ASmearingType*);
extern  bool_t vml_Box3dSmearing (VML *, char *instance, Box3dSmearing*);
extern  bool_t vml_Exponential3dSmearing (VML *, char *instance, Exponential3dSmearing*);
extern  bool_t vml_A2ASmearing (VML *, char *instance, A2ASmearing*);
extern  bool_t vml_MatIdxAndCoeff (VML *, char *instance, MatIdxAndCoeff*);
extern  bool_t vml_ContractionTypeA2ABilinear (VML *, char *instance, ContractionTypeA2ABilinear*);
extern  bool_t vml_ContractionTypeKtoPiPi (VML *, char *instance, ContractionTypeKtoPiPi*);
extern  bool_t vml_GparityMeasurement (VML *, char *instance, GparityMeasurement*);
extern  bool_t vml_GparityContractArg (VML *, char *instance, GparityContractArg*);
extern  bool_t vml_GparityAMAarg (VML *, char *instance, GparityAMAarg*);
extern  bool_t vml_GparityAMAbilBKarg (VML *, char *instance, GparityAMAbilBKarg*);
extern  bool_t vml_GparityAMAarg2 (VML *, char *instance, GparityAMAarg2*);

#endif /* K&R C */

#ifdef __cplusplus
}
#endif
CPS_END_NAMESPACE

#endif /* !_GPARITY_CONTRACT_ARG_H_RPCGEN */
