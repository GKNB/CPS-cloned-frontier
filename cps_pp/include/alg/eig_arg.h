/*
 * Please do not edit this file.
 * It was generated using PAB's VML system.
 */

#ifndef _EIG_ARG_H_RPCGEN
#define _EIG_ARG_H_RPCGEN

#include <config.h>
#include <util/vml/types.h>
#include <util/vml/vml.h>
#include <util/enum.h>
#include <util/defines.h>
CPS_START_NAMESPACE

enum EIG_LIM {
	MAX_EIG_MASSES = 100,
};
typedef enum EIG_LIM EIG_LIM;
extern struct vml_enum_map EIG_LIM_map[];

class VML;
class EigArg {
public:
	 bool Encode(char *filename,char *instance);
	 bool Decode(char *filename,char *instance);
	 bool Vml(VML *vmls,char *instance);
	PatternType pattern_kind;
	Float Mass_init;
	Float Mass_final;
	Float Mass_step;
	Float Mass[MAX_EIG_MASSES];
	int n_masses;
	int N_eig;
	int Kalk_Sim;
	int MaxCG;
	Float RsdR_a;
	Float RsdR_r;
	Float Rsdlam;
	Float Cv_fact;
	int N_min;
	int N_max;
	int N_KS_max;
	int n_renorm;
	int ProjApsiP;
	enum RitzMatType RitzMatOper;
	int print_hsum;
	int hsum_dir;
	Float mass;
};

/* the xdr functions */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__STDC__) || defined(__cplusplus)
extern  bool_t vml_EIG_LIM (VML *, char *instance, EIG_LIM*);
extern  bool_t vml_EigArg (VML *, char *instance, EigArg*);

#else /* K&R C */
extern  bool_t vml_EIG_LIM (VML *, char *instance, EIG_LIM*);
extern  bool_t vml_EigArg (VML *, char *instance, EigArg*);

#endif /* K&R C */

#ifdef __cplusplus
}
#endif
CPS_END_NAMESPACE

#endif /* !_EIG_ARG_H_RPCGEN */
