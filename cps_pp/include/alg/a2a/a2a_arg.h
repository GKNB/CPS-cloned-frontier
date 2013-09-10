// By Daiqian Sept.27 2012 

#ifndef _A2A_ARG_H_
#define _A2A_ARG_H_

#include<config.h>
#include<util/vml/types.h>
#include<util/vml/vml.h>
#include<util/enum.h>
#include<util/defines.h>
#include<alg/cg_arg.h>
#include<alg/qpropw_arg.h>
CPS_START_NAMESPACE

class VML;
class A2AArg{
	public:
		bool Encode(char *filename, char *instance);
		bool Decode(char *filename, char *instance);
		bool Vml(VML *vmls,char *instance);
		int nl;
		int nhits;
		enum RandomType rand_type;
		int src_width;
};

extern bool_t vml_A2AArg(VML *, char *instance, A2AArg *);

CPS_END_NAMESPACE

#endif
