/*
 * Please do not edit this file.
 * It was generated using PAB's VML system.
 */

#ifndef _ENUM_H_RPCGEN
#define _ENUM_H_RPCGEN

#include <config.h>
#include <util/vml/types.h>
#include <util/vml/vml.h>
#include <util/enum.h>
#include <util/defines.h>
CPS_START_NAMESPACE

typedef float pooh;

typedef double Float;

typedef double IFloat;

typedef uint64_t Pointer;

enum DirType {
	DIR_X = 0,
	DIR_Y = 1,
	DIR_Z = 2,
	DIR_T = 3,
	DIR_S = 4,
};
typedef enum DirType DirType;
extern struct vml_enum_map DirType_map[];

enum ChiralProj {
	PL = -6,
	PR = -7,
};
typedef enum ChiralProj ChiralProj;
extern struct vml_enum_map ChiralProj_map[];

enum PrecType {
	PREC_SINGLE = 0,
	PREC_DOUBLE = 1,
};
typedef enum PrecType PrecType;
extern struct vml_enum_map PrecType_map[];

enum FclassType {
	F_CLASS_NONE = 0,
	F_CLASS_STAG = 1,
	F_CLASS_WILSON = 2,
	F_CLASS_CLOVER = 3,
	F_CLASS_DWF = 4,
	F_CLASS_ASQTAD = 5,
	F_CLASS_P4 = 6,
	F_CLASS_HISQ = 7,
	F_CLASS_WILSON_TM = 8,
	F_CLASS_MDWF = 9,
	F_CLASS_BFM = 10,
	F_CLASS_MOBIUS = 11,
	F_CLASS_DWF4D = 12,
	F_CLASS_DWF4D_PAIR = 13,
	F_CLASS_ZMOBIUS = 14,
	F_CLASS_NAIVE = 15,
	F_CLASS_GRID = 16,
	F_CLASS_GRID_GPARITY_MOBIUS = 17,
	F_CLASS_GRID_MOBIUS = 18,
	F_CLASS_GRID_ZMOBIUS = 19,
	F_CLASS_GRID_GPARITY_WILSON_TM = 20,
	F_CLASS_GRID_WILSON_TM = 21,
};
typedef enum FclassType FclassType;
extern struct vml_enum_map FclassType_map[];

enum GclassType {
	G_CLASS_NONE = 0,
	G_CLASS_WILSON = 1,
	G_CLASS_POWER_PLAQ = 2,
	G_CLASS_IMPR_RECT = 3,
	G_CLASS_POWER_RECT = 4,
	G_CLASS_IMPR_OLSYM = 5,
	G_CLASS_TADPOLE_RECT = 6,
};
typedef enum GclassType GclassType;
extern struct vml_enum_map GclassType_map[];

enum StrOrdType {
	CANONICAL = 0,
	STAG = 1,
	WILSON = 2,
	G_WILSON_HB = 3,
	STAG_BLOCK = 4,
	DWF_5D_EOPREC = WILSON,
	DWF_4D_EOPREC = 5,
	DWF_4D_EOPREC_EE = 6,
	S_INNER = 7,
};
typedef enum StrOrdType StrOrdType;
extern struct vml_enum_map StrOrdType_map[];

enum CnvFrmType {
	CNV_FRM_NO = 0,
	CNV_FRM_YES = 1,
};
typedef enum CnvFrmType CnvFrmType;
extern struct vml_enum_map CnvFrmType_map[];

enum FermionFieldDimension {
	FOUR_D = 0,
	FIVE_D = 1,
};
typedef enum FermionFieldDimension FermionFieldDimension;
extern struct vml_enum_map FermionFieldDimension_map[];

enum PreserveType {
	PRESERVE_NO = 0,
	PRESERVE_YES = 1,
};
typedef enum PreserveType PreserveType;
extern struct vml_enum_map PreserveType_map[];

enum StartConfType {
	START_CONF_ORD = 0,
	START_CONF_DISORD = 1,
	START_CONF_FILE = 2,
	START_CONF_LOAD = 3,
	START_CONF_MEM = 4,
};
typedef enum StartConfType StartConfType;
extern struct vml_enum_map StartConfType_map[];

enum StartSeedType {
	START_SEED_FIXED = 0,
	START_SEED_FIXED_UNIFORM = 1,
	START_SEED = 2,
	START_SEED_UNIFORM = 3,
	START_SEED_INPUT = 4,
	START_SEED_INPUT_UNIFORM = 5,
	START_SEED_INPUT_NODE = 6,
	START_SEED_FILE = 7,
};
typedef enum StartSeedType StartSeedType;
extern struct vml_enum_map StartSeedType_map[];

enum ChkbType {
	CHKB_EVEN = 0,
	CHKB_ODD = 1,
};
typedef enum ChkbType ChkbType;
extern struct vml_enum_map ChkbType_map[];

enum DagType {
	DAG_NO = 0,
	DAG_YES = 1,
};
typedef enum DagType DagType;
extern struct vml_enum_map DagType_map[];

enum BndCndType {
	BND_CND_PRD = 0,
	BND_CND_APRD = 1,
	BND_CND_TWISTED = 2,
	BND_CND_GPARITY = 3,
	BND_CND_GPARITY_TWISTED = 4,
};
typedef enum BndCndType BndCndType;
extern struct vml_enum_map BndCndType_map[];

enum FixGaugeType {
	FIX_GAUGE_NONE = -2,
	FIX_GAUGE_LANDAU = -1,
	FIX_GAUGE_COULOMB_X = 0,
	FIX_GAUGE_COULOMB_Y = 1,
	FIX_GAUGE_COULOMB_Z = 2,
	FIX_GAUGE_COULOMB_T = 3,
};
typedef enum FixGaugeType FixGaugeType;
extern struct vml_enum_map FixGaugeType_map[];

enum SprojType {
	SPROJ_XM = 0,
	SPROJ_YM = 1,
	SPROJ_ZM = 2,
	SPROJ_TM = 3,
	SPROJ_XP = 4,
	SPROJ_YP = 5,
	SPROJ_ZP = 6,
	SPROJ_TP = 7,
};
typedef enum SprojType SprojType;
extern struct vml_enum_map SprojType_map[];

enum SigmaprojType {
	SIGMAPROJ_XY = 0,
	SIGMAPROJ_XZ = 1,
	SIGMAPROJ_XT = 2,
	SIGMAPROJ_YZ = 3,
	SIGMAPROJ_YT = 4,
	SIGMAPROJ_YX = 5,
	SIGMAPROJ_ZT = 6,
	SIGMAPROJ_ZX = 7,
	SIGMAPROJ_ZY = 8,
	SIGMAPROJ_TX = 9,
	SIGMAPROJ_TY = 10,
	SIGMAPROJ_TZ = 11,
};
typedef enum SigmaprojType SigmaprojType;
extern struct vml_enum_map SigmaprojType_map[];

enum RitzMatType {
	NONE = 0,
	MAT_HERM = 1,
	MATPC_HERM = 2,
	MATPCDAG_MATPC = 3,
	NEG_MATPCDAG_MATPC = 4,
	MATDAG_MAT = 5,
	NEG_MATDAG_MAT = 6,
	MATDAG_MAT_NORM = 7,
	NEG_MATDAG_MAT_NORM = 8,
	MATPCDAG_MATPC_SHIFT = 9,
	RitzMatType_LAST = 10,
};
typedef enum RitzMatType RitzMatType;
extern struct vml_enum_map RitzMatType_map[];

enum RatApproxType {
	CONSTANT = 0,
	DYNAMIC = 1,
};
typedef enum RatApproxType RatApproxType;
extern struct vml_enum_map RatApproxType_map[];

enum MultiShiftSolveType {
	SINGLE = 0,
	MULTI = 1,
	GENERAL = 2,
};
typedef enum MultiShiftSolveType MultiShiftSolveType;
extern struct vml_enum_map MultiShiftSolveType_map[];

enum MassRenormaliseDir {
	RENORM_BACKWARDS = 0,
	RENORM_FORWARDS = 1,
};
typedef enum MassRenormaliseDir MassRenormaliseDir;
extern struct vml_enum_map MassRenormaliseDir_map[];

enum FieldType {
	FERMION = 0,
	BOSON = 1,
};
typedef enum FieldType FieldType;
extern struct vml_enum_map FieldType_map[];

enum RatType {
	RATIONAL_STANDARD = 0,
	RATIONAL_QUOTIENT = 1,
	RATIONAL_SPLIT = 2,
};
typedef enum RatType RatType;
extern struct vml_enum_map RatType_map[];

enum WbaryonFold {
	BARYON_FOLD = 0,
	BARYON_RAW = 1,
	BARYON_PAST = 2,
};
typedef enum WbaryonFold WbaryonFold;
extern struct vml_enum_map WbaryonFold_map[];

enum SourceKind {
	POINT_W = 0,
	WALL_W = 0 + 1,
	BOX_W = 0 + 2,
	JACOBI_W = 0 + 3,
	MAX_NUM_SINK = 0 + 4,
	Z2 = 0 + 5,
	COMPLEX_Z2 = 0 + 6,
	KURAMASHI = 0 + 7,
};
typedef enum SourceKind SourceKind;
extern struct vml_enum_map SourceKind_map[];

enum SinkKind {
	W_POINT = 0,
	W_WALL = 1,
	W_BOX = 2,
};
typedef enum SinkKind SinkKind;
extern struct vml_enum_map SinkKind_map[];

enum MomentumKind {
	MOM_000 = 0,
	MOM_001 = 0 + 1,
	MOM_002 = 0 + 2,
	MOM_011 = 0 + 3,
	MOM_022 = 0 + 4,
	MOM_111 = 0 + 5,
	MOM_222 = 0 + 6,
	MAX_NUM_MOMENTA = 0 + 7,
};
typedef enum MomentumKind MomentumKind;
extern struct vml_enum_map MomentumKind_map[];

enum DEVOperatorKind {
	UNIT = 0,
	DEV1 = 0 + 1,
	DEV2 = 0 + 2,
	DEV3 = 0 + 3,
	DEV1DEV2 = 0 + 4,
	DEV2DEV1 = 0 + 5,
	DEV2DEV3 = 0 + 6,
	DEV3DEV2 = 0 + 7,
	DEV1DEV3 = 0 + 8,
	DEV3DEV1 = 0 + 9,
	DEV1DEV1 = 0 + 10,
	DEV2DEV2 = 0 + 11,
	DEV3DEV3 = 0 + 12,
	DEV_OP_NUM = 0 + 13,
	SUM_F = 0 + 14,
	SUM_S_ANTISYM = 0 + 15,
	SUM_S_SYM = 0 + 16,
	SUM_S_DIAG = 0 + 17,
	SUM_F_S_ANTISYM = 0 + 18,
	SUM_S_SYM_DIAG = 0 + 19,
	SUM_UNIT_F_S_ANTISYM = 0 + 20,
	END_SUM_OP = 0 + 21,
	BEGIN_BE_OP = 0 + 22,
	FB1_OP = 0,
	FB2_OP = 0 + 1,
	FB3_OP = 0 + 2,
	FE1_OP = 0 + 3,
	FE2_OP = 0 + 4,
	FE3_OP = 0 + 5,
	FUNIT_OP = 0 + 6,
	SUM_MAGN_OP = 0 + 7,
	SUM_ELEC_OP = 0 + 8,
	SUM_MAGN_ELEC_OP = 0 + 9,
	END_BE_OP = 0 + 10,
};
typedef enum DEVOperatorKind DEVOperatorKind;
extern struct vml_enum_map DEVOperatorKind_map[];

enum WMesonOpKind {
	MO_a0xP_x = 0,
	MO_a0xP_y = 1,
	MO_a0xP_z = 2,
	MO_pionxP_x = 3,
	MO_pionxP_y = 4,
	MO_pionxP_z = 5,
	MO_a0_primexP_x = 6,
	MO_a0_primexP_y = 7,
	MO_a0_primexP_z = 8,
	MO_rhoxP_A1 = 9,
	MO_rhoxP_T1_x = 10,
	MO_rhoxP_T1_y = 11,
	MO_rhoxP_T1_z = 12,
	MO_rhoxP_T2_x = 13,
	MO_rhoxP_T2_y = 14,
	MO_rhoxP_T2_z = 15,
	MO_a1xP_A1 = 16,
	MO_a1xP_T2_x = 17,
	MO_a1xP_T2_y = 18,
	MO_a1xP_T2_z = 19,
	MO_a1xP_E_1 = 20,
	MO_a1xP_E_2 = 21,
	MO_b1xP_T1_x = 22,
	MO_b1xP_T1_y = 23,
	MO_b1xP_T1_z = 24,
	MO_b1xD_A2 = 25,
	MO_b1xD_T1_x = 26,
	MO_b1xD_T1_y = 27,
	MO_b1xD_T1_z = 28,
	MO_b1xD_T2_x = 29,
	MO_b1xD_T2_y = 30,
	MO_b1xD_T2_z = 31,
	MO_b1xD_E_1 = 32,
	MO_b1xD_E_2 = 33,
	MO_a0_primexD_x = 34,
	MO_a0_primexD_y = 35,
	MO_a0_primexD_z = 36,
	MO_rhoxB_T1_x = 37,
	MO_rhoxB_T1_y = 38,
	MO_rhoxB_T1_z = 39,
	MO_rhoxB_T2_x = 40,
	MO_rhoxB_T2_y = 41,
	MO_rhoxB_T2_z = 42,
	MO_a1xB_A1 = 43,
	MO_a1xB_T1_x = 44,
	MO_a1xB_T1_y = 45,
	MO_a1xB_T1_z = 46,
	MO_a1xB_T2_x = 47,
	MO_a1xB_T2_y = 48,
	MO_a1xB_T2_z = 49,
	MO_a1xD_A2 = 50,
	MO_a1xD_T1_x = 51,
	MO_a1xD_T1_y = 52,
	MO_a1xD_T1_z = 53,
	MO_a1xD_T2_x = 54,
	MO_a1xD_T2_y = 55,
	MO_a1xD_T2_z = 56,
	MO_a1xD_E_1 = 57,
	MO_a1xD_E_2 = 58,
	MO_rhoxD_A2 = 59,
	MO_rhoxD_T1_x = 60,
	MO_rhoxD_T1_y = 61,
	MO_rhoxD_T1_z = 62,
	MO_rhoxD_T2_x = 63,
	MO_rhoxD_T2_y = 64,
	MO_rhoxD_T2_z = 65,
	MO_pionxB_T1_x = 66,
	MO_pionxB_T1_y = 67,
	MO_pionxB_T1_z = 68,
	MO_pionxD_T2_x = 69,
	MO_pionxD_T2_y = 70,
	MO_pionxD_T2_z = 71,
	NUM_WMESON_OP_KIND = 72,
};
typedef enum WMesonOpKind WMesonOpKind;
extern struct vml_enum_map WMesonOpKind_map[];

enum WMesonState {
	MS_a0xP_x = 0,
	MS_a0xP_y = 1,
	MS_a0xP_z = 2,
	MS_pionxP_x = 3,
	MS_pionxP_y = 4,
	MS_pionxP_z = 5,
	MS_a0_primexP_x = 6,
	MS_a0_primexP_y = 7,
	MS_a0_primexP_z = 8,
	MS_rhoxP_A1_1 = 9,
	MS_rhoxP_T1_x = 10,
	MS_rhoxP_T1_y = 11,
	MS_rhoxP_T1_z = 12,
	MS_rhoxP_T2_x = 13,
	MS_rhoxP_T2_y = 14,
	MS_rhoxP_T2_z = 15,
	MS_a1xP_A1_1 = 16,
	MS_a1xP_T2_x = 17,
	MS_a1xP_T2_y = 18,
	MS_a1xP_T2_z = 19,
	MS_a1xP_E_1 = 20,
	MS_a1xP_E_2 = 21,
	MS_b1xP_T1_x = 22,
	MS_b1xP_T1_y = 23,
	MS_b1xP_T1_z = 24,
	MS_b1xD_A2_1 = 25,
	MS_b1xD_T1_x = 26,
	MS_b1xD_T1_y = 27,
	MS_b1xD_T1_z = 28,
	MS_b1xD_T2_x = 29,
	MS_b1xD_T2_y = 30,
	MS_b1xD_T2_z = 31,
	MS_b1xD_E_1 = 32,
	MS_b1xD_E_2 = 33,
	MS_a0_primexD_x = 34,
	MS_a0_primexD_y = 35,
	MS_a0_primexD_z = 36,
	MS_rhoxB_T1_x = 37,
	MS_rhoxB_T1_y = 38,
	MS_rhoxB_T1_z = 39,
	MS_rhoxB_T2_x = 40,
	MS_rhoxB_T2_y = 41,
	MS_rhoxB_T2_z = 42,
	MS_a1xB_A1_1 = 43,
	MS_a1xB_T1_x = 44,
	MS_a1xB_T1_y = 45,
	MS_a1xB_T1_z = 46,
	MS_a1xB_T2_x = 47,
	MS_a1xB_T2_y = 48,
	MS_a1xB_T2_z = 49,
	MS_a1xD_A2_1 = 50,
	MS_a1xD_T1_x = 51,
	MS_a1xD_T1_y = 52,
	MS_a1xD_T1_z = 53,
	MS_a1xD_T2_x = 54,
	MS_a1xD_T2_y = 55,
	MS_a1xD_T2_z = 56,
	MS_a1xD_E_1 = 57,
	MS_a1xD_E_2 = 58,
	MS_rhoxD_A2_1 = 59,
	MS_rhoxD_T1_x = 60,
	MS_rhoxD_T1_y = 61,
	MS_rhoxD_T1_z = 62,
	MS_rhoxD_T2_x = 63,
	MS_rhoxD_T2_y = 64,
	MS_rhoxD_T2_z = 65,
	MS_pionxB_T1_x = 66,
	MS_pionxB_T1_y = 67,
	MS_pionxB_T1_z = 68,
	MS_pionxD_T2_x = 69,
	MS_pionxD_T2_y = 70,
	MS_pionxD_T2_z = 71,
	NUM_WMESON_STATE = 72,
};
typedef enum WMesonState WMesonState;
extern struct vml_enum_map WMesonState_map[];

enum WMesonOutputName {
	a0xP = 0,
	pionxP = 1,
	a0_primexP = 2,
	rhoxP_A1 = 3,
	rhoxP_T1 = 4,
	rhoxP_T2 = 5,
	a1xP_A1 = 6,
	a1xP_T2 = 7,
	a1xP_E = 8,
	b1xP_T1 = 9,
	b1xD_A2 = 10,
	b1xD_T1 = 11,
	b1xD_T2 = 12,
	b1xD_E = 13,
	a0_primexD = 14,
	rhoxB_T1 = 15,
	rhoxB_T2 = 16,
	a1xB_A1 = 17,
	a1xB_T1 = 18,
	a1xB_T2 = 19,
	a1xD_A2 = 20,
	a1xD_T1 = 21,
	a1xD_T2 = 22,
	a1xD_E = 23,
	rhoxD_A2 = 24,
	rhoxD_T1 = 25,
	rhoxD_T2 = 26,
	pionxB_T1 = 27,
	pionxD_T2 = 28,
	NUM_WMESON_OUTPUT = 29,
};
typedef enum WMesonOutputName WMesonOutputName;
extern struct vml_enum_map WMesonOutputName_map[];

enum WMesonCategory {
	NORMALMESON = 0,
	EXT_FIRSTDEV_MESON = 1,
	EXT_SECONDDEV_SYM_MESON = 2,
	EXT_SECONDDEV_ANTISYM_MESON = 3,
	EXT_SECONDDEV_DIAG_MESON = 4,
	MIXING = 5,
};
typedef enum WMesonCategory WMesonCategory;
extern struct vml_enum_map WMesonCategory_map[];

enum WExtMesonBEOutputName {
	BE_pionxB = 0,
	BE_rhoxB_T1 = 0 + 1,
	NUM_WEXTMESON_BE_OUTPUT = 0 + 2,
};
typedef enum WExtMesonBEOutputName WExtMesonBEOutputName;
extern struct vml_enum_map WExtMesonBEOutputName_map[];

enum WExtMesonBEState {
	BE_MS_pionxB_x = 0,
	BE_MS_pionxB_y = 0 + 1,
	BE_MS_pionxB_z = 0 + 2,
	BE_MS_rhoxB_T1_x = 0 + 3,
	BE_MS_rhoxB_T1_y = 0 + 4,
	BE_MS_rhoxB_T1_z = 0 + 5,
	NUM_WEXTMESON_BE_STATES = 0 + 6,
};
typedef enum WExtMesonBEState WExtMesonBEState;
extern struct vml_enum_map WExtMesonBEState_map[];

enum WExtMesonBEOp {
	BE_MO_pionxB_x = 0,
	BE_MO_pionxB_y = 0 + 1,
	BE_MO_pionxB_z = 0 + 2,
	BE_MO_rhoxB_T1_x = 0 + 3,
	BE_MO_rhoxB_T1_y = 0 + 4,
	BE_MO_rhoxB_T1_z = 0 + 5,
	NUM_WEXTMESON_BE_OPS = 0 + 6,
};
typedef enum WExtMesonBEOp WExtMesonBEOp;
extern struct vml_enum_map WExtMesonBEOp_map[];

enum WExtMesonBECategory {
	ELEC_HYBRID_BE = 0,
	MAG_HYBRID_BE = 0 + 1,
	MIXING_BE = 0 + 2,
};
typedef enum WExtMesonBECategory WExtMesonBECategory;
extern struct vml_enum_map WExtMesonBECategory_map[];

enum FieldTensorId {
	FB1 = 0,
	FB2 = 0 + 1,
	FB3 = 0 + 2,
	FE1 = 0 + 3,
	FE2 = 0 + 4,
	FE3 = 0 + 5,
	NUM_FLDS = 0 + 6,
	FUNIT = 0 + 7,
	SUM_MAGN = 0 + 8,
	SUM_ELEC = 0 + 9,
	SUM_MAGN_ELEC = 0 + 10,
	NUM_FLD_OPS = 0 + 11,
};
typedef enum FieldTensorId FieldTensorId;
extern struct vml_enum_map FieldTensorId_map[];

enum PatternType {
	LIN = 0,
	ARRAY = 0 + 1,
	LOG = 0 + 2,
	FLOW = 0 + 3,
};
typedef enum PatternType PatternType;
extern struct vml_enum_map PatternType_map[];

enum IntegratorType {
	INT_LEAP = 0,
	INT_OMELYAN = 0 + 1,
	INT_CAMPOSTRINI = 0 + 2,
	INT_OMELYAN_44 = 0 + 3,
	INT_OMELYAN_45 = 0 + 4,
	INT_FORCE_GRAD_PQPQP = 0 + 5,
	INT_FORCE_GRAD_QPQPQ = 0 + 6,
	INT_FORCE_GRAD_QPQPQPQ = 0 + 7,
	INT_FORCE_GRAD_PQPQPQPQP = 0 + 8,
	INT_SUM = 0 + 9,
	INT_MOM = 0 + 10,
	INT_GAUGE = 0 + 11,
	INT_FERMION = 0 + 12,
	INT_BOSON = 0 + 13,
	INT_QUOTIENT = 0 + 14,
	INT_RATIONAL = 0 + 15,
	INT_RATIONAL_SPLIT = 0 + 16,
	INT_RATIONAL_QUOTIENT = 0 + 17,
};
typedef enum IntegratorType IntegratorType;
extern struct vml_enum_map IntegratorType_map[];

enum IntegratorLevel {
	EMBEDDED_INTEGRATOR = 0,
	TOP_LEVEL_INTEGRATOR = 0 + 1,
};
typedef enum IntegratorLevel IntegratorLevel;
extern struct vml_enum_map IntegratorLevel_map[];

enum ReunitarizeType {
	REUNITARIZE_NO = 0,
	REUNITARIZE_YES = 1,
};
typedef enum ReunitarizeType ReunitarizeType;
extern struct vml_enum_map ReunitarizeType_map[];

enum ReproduceTest {
	REPRODUCE_NO = 0,
	REPRODUCE_YES = 1,
};
typedef enum ReproduceTest ReproduceTest;
extern struct vml_enum_map ReproduceTest_map[];

enum TestReproduceTest {
	TEST_REPRODUCE_NO = 0,
	TEST_REPRODUCE_YES = 1,
};
typedef enum TestReproduceTest TestReproduceTest;
extern struct vml_enum_map TestReproduceTest_map[];

enum ReverseTest {
	REVERSE_NO = 0,
	REVERSE_YES = 1,
};
typedef enum ReverseTest ReverseTest;
extern struct vml_enum_map ReverseTest_map[];

enum MetropolisType {
	METROPOLIS_NO = 0,
	METROPOLIS_YES = 1,
};
typedef enum MetropolisType MetropolisType;
extern struct vml_enum_map MetropolisType_map[];

enum ForceMeasure {
	FORCE_MEASURE_NO = 0,
	FORCE_MEASURE_YES = 1,
};
typedef enum ForceMeasure ForceMeasure;
extern struct vml_enum_map ForceMeasure_map[];

enum EigenMeasure {
	EIGEN_MEASURE_NO = 0,
	EIGEN_MEASURE_YES = 1,
};
typedef enum EigenMeasure EigenMeasure;
extern struct vml_enum_map EigenMeasure_map[];

enum RhmcPolesAction {
	RHMC_POLES_CALC = 0,
	RHMC_POLES_READ = 1,
	RHMC_POLES_CALC_WRITE = 2,
};
typedef enum RhmcPolesAction RhmcPolesAction;
extern struct vml_enum_map RhmcPolesAction_map[];

enum HmdLimits {
	MAX_HMD_MASSES = 200,
	MAX_RAT_DEGREE = 30,
};
typedef enum HmdLimits HmdLimits;
extern struct vml_enum_map HmdLimits_map[];

enum InverterType {
	CG = 0,
	BICGSTAB = 1,
	EIGCG = 2,
	LOWMODEAPPROX = 3,
	CG_LOWMODE_DEFL = 4,
	HDCG = 5,
	FAKE = 6,
};
typedef enum InverterType InverterType;
extern struct vml_enum_map InverterType_map[];

enum RationalApproxType {
	RATIONAL_APPROX_POWER = 0,
	RATIONAL_APPROX_QUOTIENT = 1,
	RATIONAL_APPROX_ZERO_POLE = 2,
};
typedef enum RationalApproxType RationalApproxType;
extern struct vml_enum_map RationalApproxType_map[];

enum RationalBoundsType {
	RATIONAL_BOUNDS_AUTOMATIC = 0,
	RATIONAL_BOUNDS_MANUAL = 1,
};
typedef enum RationalBoundsType RationalBoundsType;
extern struct vml_enum_map RationalBoundsType_map[];

enum StaticBActionLinkSmearType {
	SB_ALS_NONE = 0,
	SB_ALS_APE = 1,
	SB_ALS_APE_NO_PROJ = 2,
	SB_ALS_APE_OLEG = 3,
	SB_ALS_HYP_HK = 4,
	SB_ALS_HYP_L = 5,
	SB_ALS_HYP_2 = 6,
	SB_ALS_STOUT = 7,
};
typedef enum StaticBActionLinkSmearType StaticBActionLinkSmearType;
extern struct vml_enum_map StaticBActionLinkSmearType_map[];

enum GaussianKernelLinkSmearType {
	GKLS_NONE = 0,
	GKLS_APE = 1,
	GKLS_STOUT = 2,
};
typedef enum GaussianKernelLinkSmearType GaussianKernelLinkSmearType;
extern struct vml_enum_map GaussianKernelLinkSmearType_map[];

enum CalcQpropType {
	READ_QPROP = 0,
	NOIO_QPROP = 1,
	WRITE_QPROP = 2,
};
typedef enum CalcQpropType CalcQpropType;
extern struct vml_enum_map CalcQpropType_map[];

enum CalcSeqType {
	READ_SEQ = 0,
	NOIO_SEQ = 1,
	WRITE_SEQ = 2,
	MULT_SEQ = 3,
	READ_MULT_SEQ = 4,
	WRITE_MULT_SEQ = 5,
};
typedef enum CalcSeqType CalcSeqType;
extern struct vml_enum_map CalcSeqType_map[];

enum BfmSolverType {
	BFM_DWF = 0,
	BFM_DWFrb4d = 1,
	BFM_WilsonFermion = 2,
	BFM_WilsonTM = 3,
	BFM_WilsonNN = 4,
	BFM_HwPartFracZolo = 5,
	BFM_HwContFracZolo = 6,
	BFM_HwPartFracTanh = 7,
	BFM_HwContFracTanh = 8,
	BFM_HwCayleyZolo = 9,
	BFM_HtCayleyZolo = 10,
	BFM_HwCayleyTanh = 11,
	BFM_HmCayleyTanh = 12,
	BFM_HtCayleyTanh = 13,
	BFM_DWFTransfer = 14,
	BFM_DWFTransferInv = 15,
	BFM_HtContFracTanh = 16,
	BFM_HtContFracZolo = 17,
};
typedef enum BfmSolverType BfmSolverType;
extern struct vml_enum_map BfmSolverType_map[];

enum A2Apreconditioning {
	SchurOriginal = 0,
	SchurDiagTwo = 1,
};
typedef enum A2Apreconditioning A2Apreconditioning;
extern struct vml_enum_map A2Apreconditioning_map[];

enum A2ACGalgorithm {
	AlgorithmCG = 0,
	AlgorithmMixedPrecisionRestartedCG = 1,
	AlgorithmMixedPrecisionReliableUpdateCG = 2,
	AlgorithmMixedPrecisionReliableUpdateSplitCG = 3,
	AlgorithmMixedPrecisionMADWF = 4,
};
typedef enum A2ACGalgorithm A2ACGalgorithm;
extern struct vml_enum_map A2ACGalgorithm_map[];

/* the xdr functions */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__STDC__) || defined(__cplusplus)
extern  bool_t vml_pooh (VML *, char *instance, pooh*);
extern  bool_t vml_Float (VML *, char *instance, Float*);
extern  bool_t vml_IFloat (VML *, char *instance, IFloat*);
extern  bool_t vml_Pointer (VML *, char *instance, Pointer*);
extern  bool_t vml_DirType (VML *, char *instance, DirType*);
extern  bool_t vml_ChiralProj (VML *, char *instance, ChiralProj*);
extern  bool_t vml_PrecType (VML *, char *instance, PrecType*);
extern  bool_t vml_FclassType (VML *, char *instance, FclassType*);
extern  bool_t vml_GclassType (VML *, char *instance, GclassType*);
extern  bool_t vml_StrOrdType (VML *, char *instance, StrOrdType*);
extern  bool_t vml_CnvFrmType (VML *, char *instance, CnvFrmType*);
extern  bool_t vml_FermionFieldDimension (VML *, char *instance, FermionFieldDimension*);
extern  bool_t vml_PreserveType (VML *, char *instance, PreserveType*);
extern  bool_t vml_StartConfType (VML *, char *instance, StartConfType*);
extern  bool_t vml_StartSeedType (VML *, char *instance, StartSeedType*);
extern  bool_t vml_ChkbType (VML *, char *instance, ChkbType*);
extern  bool_t vml_DagType (VML *, char *instance, DagType*);
extern  bool_t vml_BndCndType (VML *, char *instance, BndCndType*);
extern  bool_t vml_FixGaugeType (VML *, char *instance, FixGaugeType*);
extern  bool_t vml_SprojType (VML *, char *instance, SprojType*);
extern  bool_t vml_SigmaprojType (VML *, char *instance, SigmaprojType*);
extern  bool_t vml_RitzMatType (VML *, char *instance, RitzMatType*);
extern  bool_t vml_RatApproxType (VML *, char *instance, RatApproxType*);
extern  bool_t vml_MultiShiftSolveType (VML *, char *instance, MultiShiftSolveType*);
extern  bool_t vml_MassRenormaliseDir (VML *, char *instance, MassRenormaliseDir*);
extern  bool_t vml_FieldType (VML *, char *instance, FieldType*);
extern  bool_t vml_RatType (VML *, char *instance, RatType*);
extern  bool_t vml_WbaryonFold (VML *, char *instance, WbaryonFold*);
extern  bool_t vml_SourceKind (VML *, char *instance, SourceKind*);
extern  bool_t vml_SinkKind (VML *, char *instance, SinkKind*);
extern  bool_t vml_MomentumKind (VML *, char *instance, MomentumKind*);
extern  bool_t vml_DEVOperatorKind (VML *, char *instance, DEVOperatorKind*);
extern  bool_t vml_WMesonOpKind (VML *, char *instance, WMesonOpKind*);
extern  bool_t vml_WMesonState (VML *, char *instance, WMesonState*);
extern  bool_t vml_WMesonOutputName (VML *, char *instance, WMesonOutputName*);
extern  bool_t vml_WMesonCategory (VML *, char *instance, WMesonCategory*);
extern  bool_t vml_WExtMesonBEOutputName (VML *, char *instance, WExtMesonBEOutputName*);
extern  bool_t vml_WExtMesonBEState (VML *, char *instance, WExtMesonBEState*);
extern  bool_t vml_WExtMesonBEOp (VML *, char *instance, WExtMesonBEOp*);
extern  bool_t vml_WExtMesonBECategory (VML *, char *instance, WExtMesonBECategory*);
extern  bool_t vml_FieldTensorId (VML *, char *instance, FieldTensorId*);
extern  bool_t vml_PatternType (VML *, char *instance, PatternType*);
extern  bool_t vml_IntegratorType (VML *, char *instance, IntegratorType*);
extern  bool_t vml_IntegratorLevel (VML *, char *instance, IntegratorLevel*);
extern  bool_t vml_ReunitarizeType (VML *, char *instance, ReunitarizeType*);
extern  bool_t vml_ReproduceTest (VML *, char *instance, ReproduceTest*);
extern  bool_t vml_TestReproduceTest (VML *, char *instance, TestReproduceTest*);
extern  bool_t vml_ReverseTest (VML *, char *instance, ReverseTest*);
extern  bool_t vml_MetropolisType (VML *, char *instance, MetropolisType*);
extern  bool_t vml_ForceMeasure (VML *, char *instance, ForceMeasure*);
extern  bool_t vml_EigenMeasure (VML *, char *instance, EigenMeasure*);
extern  bool_t vml_RhmcPolesAction (VML *, char *instance, RhmcPolesAction*);
extern  bool_t vml_HmdLimits (VML *, char *instance, HmdLimits*);
extern  bool_t vml_InverterType (VML *, char *instance, InverterType*);
extern  bool_t vml_RationalApproxType (VML *, char *instance, RationalApproxType*);
extern  bool_t vml_RationalBoundsType (VML *, char *instance, RationalBoundsType*);
extern  bool_t vml_StaticBActionLinkSmearType (VML *, char *instance, StaticBActionLinkSmearType*);
extern  bool_t vml_GaussianKernelLinkSmearType (VML *, char *instance, GaussianKernelLinkSmearType*);
extern  bool_t vml_CalcQpropType (VML *, char *instance, CalcQpropType*);
extern  bool_t vml_CalcSeqType (VML *, char *instance, CalcSeqType*);
extern  bool_t vml_BfmSolverType (VML *, char *instance, BfmSolverType*);
extern  bool_t vml_A2Apreconditioning (VML *, char *instance, A2Apreconditioning*);
extern  bool_t vml_A2ACGalgorithm (VML *, char *instance, A2ACGalgorithm*);

#else /* K&R C */
extern  bool_t vml_pooh (VML *, char *instance, pooh*);
extern  bool_t vml_Float (VML *, char *instance, Float*);
extern  bool_t vml_IFloat (VML *, char *instance, IFloat*);
extern  bool_t vml_Pointer (VML *, char *instance, Pointer*);
extern  bool_t vml_DirType (VML *, char *instance, DirType*);
extern  bool_t vml_ChiralProj (VML *, char *instance, ChiralProj*);
extern  bool_t vml_PrecType (VML *, char *instance, PrecType*);
extern  bool_t vml_FclassType (VML *, char *instance, FclassType*);
extern  bool_t vml_GclassType (VML *, char *instance, GclassType*);
extern  bool_t vml_StrOrdType (VML *, char *instance, StrOrdType*);
extern  bool_t vml_CnvFrmType (VML *, char *instance, CnvFrmType*);
extern  bool_t vml_FermionFieldDimension (VML *, char *instance, FermionFieldDimension*);
extern  bool_t vml_PreserveType (VML *, char *instance, PreserveType*);
extern  bool_t vml_StartConfType (VML *, char *instance, StartConfType*);
extern  bool_t vml_StartSeedType (VML *, char *instance, StartSeedType*);
extern  bool_t vml_ChkbType (VML *, char *instance, ChkbType*);
extern  bool_t vml_DagType (VML *, char *instance, DagType*);
extern  bool_t vml_BndCndType (VML *, char *instance, BndCndType*);
extern  bool_t vml_FixGaugeType (VML *, char *instance, FixGaugeType*);
extern  bool_t vml_SprojType (VML *, char *instance, SprojType*);
extern  bool_t vml_SigmaprojType (VML *, char *instance, SigmaprojType*);
extern  bool_t vml_RitzMatType (VML *, char *instance, RitzMatType*);
extern  bool_t vml_RatApproxType (VML *, char *instance, RatApproxType*);
extern  bool_t vml_MultiShiftSolveType (VML *, char *instance, MultiShiftSolveType*);
extern  bool_t vml_MassRenormaliseDir (VML *, char *instance, MassRenormaliseDir*);
extern  bool_t vml_FieldType (VML *, char *instance, FieldType*);
extern  bool_t vml_RatType (VML *, char *instance, RatType*);
extern  bool_t vml_WbaryonFold (VML *, char *instance, WbaryonFold*);
extern  bool_t vml_SourceKind (VML *, char *instance, SourceKind*);
extern  bool_t vml_SinkKind (VML *, char *instance, SinkKind*);
extern  bool_t vml_MomentumKind (VML *, char *instance, MomentumKind*);
extern  bool_t vml_DEVOperatorKind (VML *, char *instance, DEVOperatorKind*);
extern  bool_t vml_WMesonOpKind (VML *, char *instance, WMesonOpKind*);
extern  bool_t vml_WMesonState (VML *, char *instance, WMesonState*);
extern  bool_t vml_WMesonOutputName (VML *, char *instance, WMesonOutputName*);
extern  bool_t vml_WMesonCategory (VML *, char *instance, WMesonCategory*);
extern  bool_t vml_WExtMesonBEOutputName (VML *, char *instance, WExtMesonBEOutputName*);
extern  bool_t vml_WExtMesonBEState (VML *, char *instance, WExtMesonBEState*);
extern  bool_t vml_WExtMesonBEOp (VML *, char *instance, WExtMesonBEOp*);
extern  bool_t vml_WExtMesonBECategory (VML *, char *instance, WExtMesonBECategory*);
extern  bool_t vml_FieldTensorId (VML *, char *instance, FieldTensorId*);
extern  bool_t vml_PatternType (VML *, char *instance, PatternType*);
extern  bool_t vml_IntegratorType (VML *, char *instance, IntegratorType*);
extern  bool_t vml_IntegratorLevel (VML *, char *instance, IntegratorLevel*);
extern  bool_t vml_ReunitarizeType (VML *, char *instance, ReunitarizeType*);
extern  bool_t vml_ReproduceTest (VML *, char *instance, ReproduceTest*);
extern  bool_t vml_TestReproduceTest (VML *, char *instance, TestReproduceTest*);
extern  bool_t vml_ReverseTest (VML *, char *instance, ReverseTest*);
extern  bool_t vml_MetropolisType (VML *, char *instance, MetropolisType*);
extern  bool_t vml_ForceMeasure (VML *, char *instance, ForceMeasure*);
extern  bool_t vml_EigenMeasure (VML *, char *instance, EigenMeasure*);
extern  bool_t vml_RhmcPolesAction (VML *, char *instance, RhmcPolesAction*);
extern  bool_t vml_HmdLimits (VML *, char *instance, HmdLimits*);
extern  bool_t vml_InverterType (VML *, char *instance, InverterType*);
extern  bool_t vml_RationalApproxType (VML *, char *instance, RationalApproxType*);
extern  bool_t vml_RationalBoundsType (VML *, char *instance, RationalBoundsType*);
extern  bool_t vml_StaticBActionLinkSmearType (VML *, char *instance, StaticBActionLinkSmearType*);
extern  bool_t vml_GaussianKernelLinkSmearType (VML *, char *instance, GaussianKernelLinkSmearType*);
extern  bool_t vml_CalcQpropType (VML *, char *instance, CalcQpropType*);
extern  bool_t vml_CalcSeqType (VML *, char *instance, CalcSeqType*);
extern  bool_t vml_BfmSolverType (VML *, char *instance, BfmSolverType*);
extern  bool_t vml_A2Apreconditioning (VML *, char *instance, A2Apreconditioning*);
extern  bool_t vml_A2ACGalgorithm (VML *, char *instance, A2ACGalgorithm*);

#endif /* K&R C */

#ifdef __cplusplus
}
#endif
CPS_END_NAMESPACE

#endif /* !_ENUM_H_RPCGEN */
