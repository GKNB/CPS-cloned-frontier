/*
 * Please do not edit this file.
 * It was generated by CPC make system.
 */

#ifndef _CG_ARG_H_RPCGEN
#define _CG_ARG_H_RPCGEN

#include <config.h>
#include <util/vml/types.h>
#include <util/vml/vml.h>
#include <util/enum.h>
#include <util/defines.h>
CPS_START_NAMESPACE

class CgArg {
public:
	 CgArg(char *filename);
	 void Encode(char *filename,char *instance);
	 void Decode(char *filename,char *instance);
	 void Vml(VML *vmls,char *instance);
	Float mass;
	int max_num_iter;
	Float stop_rsd;
	Float true_rsd;
	enum RitzMatType RitzMatOper;
	   CgArg (  ) ;
};

/* the xdr functions */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__STDC__) || defined(__cplusplus)
extern  bool_t vml_CgArg (VML *, char *instance, CgArg*);

#else /* K&R C */
extern  bool_t vml_CgArg (VML *, char *instance, CgArg*);

#endif /* K&R C */

#ifdef __cplusplus
}
#endif
CPS_END_NAMESPACE

#endif /* !_CG_ARG_H_RPCGEN */