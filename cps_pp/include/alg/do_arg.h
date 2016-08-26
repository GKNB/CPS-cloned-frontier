/*
 * Please do not edit this file.
 * It was generated using PAB's VML system.
 */

#ifndef _DO_ARG_H_RPCGEN
#define _DO_ARG_H_RPCGEN

#include <config.h>
#include <util/vml/types.h>
#include <util/vml/vml.h>
#include <util/enum.h>
#include <util/defines.h>
CPS_START_NAMESPACE

class VML;
class BGLAxisMap {
public:
	 bool Encode(char *filename,char *instance);
	 bool Decode(char *filename,char *instance);
	 bool Vml(VML *vmls,char *instance);
	int bgl_machine_dir_x;
	int bgl_machine_dir_y;
	int bgl_machine_dir_z;
	int bgl_machine_dir_t;
};

class VML;
class DoArg {
public:
	 bool Encode(char *filename,char *instance);
	 bool Decode(char *filename,char *instance);
	 bool Vml(VML *vmls,char *instance);
	int x_sites;
	int y_sites;
	int z_sites;
	int t_sites;
	int s_sites;
	int x_node_sites;
	int y_node_sites;
	int z_node_sites;
	int t_node_sites;
	int s_node_sites;
	int x_nodes;
	int y_nodes;
	int z_nodes;
	int t_nodes;
	int s_nodes;
	int updates;
	int measurements;
	int measurefreq;
	int cg_reprod_freq;
	BndCndType x_bc;
	BndCndType y_bc;
	BndCndType z_bc;
	BndCndType t_bc;
	StartConfType start_conf_kind;
	u_long start_conf_load_addr;
	StartSeedType start_seed_kind;
	char *start_seed_filename;
	char *start_conf_filename;
	int start_conf_alloc_flag;
	int wfm_alloc_flag;
	int wfm_send_alloc_flag;
	int start_seed_value;
	Float beta;
	Float c_1;
	Float u0;
	Float dwf_height;
	Float dwf_a5_inv;
	Float power_plaq_cutoff;
	int power_plaq_exponent;
	Float power_rect_cutoff;
	int power_rect_exponent;
	int verbose_level;
	int checksum_level;
	int exec_task_list;
	Float xi_bare;
	int xi_dir;
	Float xi_v;
	Float xi_v_xi;
	Float clover_coeff;
	Float clover_coeff_xi;
	Float xi_gfix;
	int gfix_chkb;
	Float asqtad_KS;
	Float asqtad_naik;
	Float asqtad_3staple;
	Float asqtad_5staple;
	Float asqtad_7staple;
	Float asqtad_lepage;
	Float p4_KS;
	Float p4_knight;
	Float p4_3staple;
	Float p4_5staple;
	Float p4_7staple;
	Float p4_lepage;
	int gparity_1f_X;
	int gparity_1f_Y;
	   DoArg (  ) ;
	   void SetupAsqTadU0 (  double u0 ) ;
};

class VML;
class DoArgExt {
public:
	 bool Encode(char *filename,char *instance);
	 bool Decode(char *filename,char *instance);
	 bool Vml(VML *vmls,char *instance);
	Float twist_bc_x;
	Float twist_bc_y;
	Float twist_bc_z;
	Float twist_bc_t;
	StartConfType start_u1_conf_kind;
	Pointer start_u1_conf_load_addr;
	char *start_u1_conf_filename;
	int start_u1_conf_alloc_flag;
	int mult_u1_conf_flag;
	int save_stride;
	int trajectory;
	Float mobius_b_coeff;
	Float mobius_c_coeff;
	struct {
		u_int zmobius_b_coeff_len;
		Float *zmobius_b_coeff_val;
	} zmobius_b_coeff;
	struct {
		u_int zmobius_c_coeff_len;
		Float *zmobius_c_coeff_val;
	} zmobius_c_coeff;
	   DoArgExt (  ) ;
};

/* the xdr functions */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__STDC__) || defined(__cplusplus)
extern  bool_t vml_BGLAxisMap (VML *, char *instance, BGLAxisMap*);
extern  bool_t vml_DoArg (VML *, char *instance, DoArg*);
extern  bool_t vml_DoArgExt (VML *, char *instance, DoArgExt*);

#else /* K&R C */
extern  bool_t vml_BGLAxisMap (VML *, char *instance, BGLAxisMap*);
extern  bool_t vml_DoArg (VML *, char *instance, DoArg*);
extern  bool_t vml_DoArgExt (VML *, char *instance, DoArgExt*);

#endif /* K&R C */

#ifdef __cplusplus
}
#endif
CPS_END_NAMESPACE

#endif /* !_DO_ARG_H_RPCGEN */
