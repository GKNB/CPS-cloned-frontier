/* Hacked by Peter Boyle for VML 2004 *//*
 * Sun RPC is a product of Sun Microsystems, Inc. and is provided for
 * unrestricted use provided that this legend is included on all tape
 * media and as a part of the software program in whole or part.  Users
 * may copy or modify Sun RPC without charge, but are not authorized
 * to license or distribute it to anyone else except as part of a product or
 * program developed by the user or with the express written consent of
 * Sun Microsystems, Inc.
 *
 * SUN RPC IS PROVIDED AS IS WITH NO WARRANTIES OF ANY KIND INCLUDING THE
 * WARRANTIES OF DESIGN, MERCHANTIBILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE, OR ARISING FROM A COURSE OF DEALING, USAGE OR TRADE PRACTICE.
 *
 * Sun RPC is provided with no support and without any obligation on the
 * part of Sun Microsystems, Inc. to assist in its use, correction,
 * modification or enhancement.
 *
 * SUN MICROSYSTEMS, INC. SHALL HAVE NO LIABILITY WITH RESPECT TO THE
 * INFRINGEMENT OF COPYRIGHTS, TRADE SECRETS OR ANY PATENTS BY SUN RPC
 * OR ANY PART THEREOF.
 *
 * In no event will Sun Microsystems, Inc. be liable for any lost revenue
 * or profits or other special, indirect and consequential damages, even if
 * Sun has been advised of the possibility of such damages.
 *
 * Sun Microsystems, Inc.
 * 2550 Garcia Avenue
 * Mountain View, California  94043
 */

/*
 * From: @(#)rpc_hout.c 1.12 89/02/22 (C) 1987 SMI
 */
char hout_rcsid[] =
  "$Id: rpc_hout.c,v 1.3.358.1 2012-11-15 18:17:08 ckelly Exp $";

/*
 * rpc_hout.c, Header file outputter for the RPC protocol compiler
 */
#include <stdio.h>
#include <ctype.h>
#include "rpc_parse.h"
#include "rpc_util.h"
#include "proto.h"

static void pconstdef (definition * def);
static void pargdef (definition * def);
static void pstructdef (definition * def);
static void pclassdef (definition * def);
static void pincludedef (definition * def);
static void puniondef (definition * def);
static void pdefine (const char *name, const char *num);
static int define_printed (proc_list * stop, version_list * start);
static void pprogramdef (definition * def);
static void parglist (proc_list * proc, const char *addargtype);
static void penumdef (definition * def);
static void ptypedef (definition * def);
static int undefined2 (const char *type, const char *stop);

extern int VMLoutput;

/* store away enough information to allow the XDR functions to be spat
    out at the end of the file */

void
storexdrfuncdecl (const char *name, int pointerp)
{
  xdrfunc * xdrptr;

  xdrptr = (xdrfunc *) malloc(sizeof (struct xdrfunc));

  xdrptr->name = (char *)name;
  xdrptr->pointerp = pointerp;
  xdrptr->next = NULL;

  if (xdrfunc_tail == NULL)
    {
      xdrfunc_head = xdrptr;
      xdrfunc_tail = xdrptr;
    }
  else
    {
      xdrfunc_tail->next = xdrptr;
      xdrfunc_tail = xdrptr;
    }
}

/*
 * Print the C-version of an xdr definition
 */
void
print_datadef (definition *def)
{

  if (def->def_kind == DEF_PROGRAM)	/* handle data only */
    return;

  if (def->def_kind != DEF_CONST)
    {
      f_print (fout, "\n");
    }
  switch (def->def_kind)
    {
    case DEF_STRUCT:
      pstructdef (def);
      break;
    case DEF_CLASS: /*PAB*/
      pclassdef (def);
      break;
    case DEF_INCLUDEPRAGMA: /*CK*/
      pincludedef (def);
      break;
    case DEF_UNION:
      puniondef (def);
      break;
    case DEF_ENUM:
      penumdef (def);
      break;
    case DEF_TYPEDEF:
      ptypedef (def);
      break;
    case DEF_PROGRAM:
      pprogramdef (def);
      break;
    case DEF_CONST:
      pconstdef (def);
      break;
    }
  if (def->def_kind != DEF_PROGRAM && def->def_kind != DEF_CONST)
    {
      storexdrfuncdecl(def->def_name,
 		       def->def_kind != DEF_TYPEDEF ||
		       !isvectordef(def->def.ty.old_type,
				    def->def.ty.rel));
    }
}


void
print_funcdef (definition *def)
{
  switch (def->def_kind)
    {
    case DEF_PROGRAM:
      f_print (fout, "\n");
      pprogramdef (def);
      break;
    default:
      /* ?... shouldn't happen I guess */
      break;
    }
}

void
print_xdr_func_def (char *name, int pointerp, int i)
{
  if ( ! VMLoutput ) {

    if (i == 2)
      {
	f_print (fout, "extern bool_t xdr_%s ();\n", name);
	return;
      }
    else
      f_print(fout, "extern  bool_t xdr_%s (XDR *, %s%s);\n", name,
	      name, pointerp ? "*" : "");
  } else { 

      f_print(fout, "extern  bool_t vml_%s (VML *, char *instance, %s%s);\n", name,
	      name, pointerp ? "*" : "");
  }
}

/*
 *PAB... output prototypes for VML/text serialisation
 */
void
print_vml_func_def (char *name, int pointerp, int i)
{
  if (i == 2)
    {
      f_print (fout, "extern bool_t vml_%s ();\n", name);
      return;
    }
  else
    f_print(fout, "extern  bool_t vml_%s (VML *, char *name,%s%s);\n", name,
	    name, pointerp ? "*" : "");
}

static void
pconstdef (definition *def)
{
  pdefine (def->def_name, def->def.co);
}

/* print out the definitions for the arguments of functions in the
   header file
 */
static void
pargdef (definition * def)
{
  decl_list *l;
  version_list *vers;
  const char *name;
  proc_list *plist;

  for (vers = def->def.pr.versions; vers != NULL; vers = vers->next)
    {
      for (plist = vers->procs; plist != NULL;
	   plist = plist->next)
	{

	  if (!newstyle || plist->arg_num < 2)
	    {
	      continue;		/* old style or single args */
	    }
	  name = plist->args.argname;
	  f_print (fout, "struct %s {\n", name);
	  for (l = plist->args.decls;
	       l != NULL; l = l->next)
	    {
	      pdeclaration (name, &l->decl, 1, ";\n");
	    }
	  f_print (fout, "};\n");
	  f_print (fout, "typedef struct %s %s;\n", name, name);
	  storexdrfuncdecl (name, 0);
	  f_print (fout, "\n");
	}
    }

}

/*CK hard-coded command support*/
static void 
prpc_generate_member_lambda_def(){
  f_print (fout, "\t   template <typename T> void datamem_lambda(T &func);\n");
}
static void 
prpc_generate_member_lambda_code(const char *name, decl_list *dl){
  f_print (fout, "template <typename T> void %s::datamem_lambda(T &func){\n",name);

  decl_list *l;
  for (l = dl; l != NULL; l = l->next){
    declaration *dec = &l->decl;
    if (streq (dec->type, "rpccommand") || streq (dec->type, "") ) continue;

    f_print(fout,"\t");
    
    if (streq (dec->type, "string")){
      f_print (fout, "func.template doit<char *>(&%s::%s,*this);\n",name,dec->name); continue;
    }
    char buf[8];			/* enough to hold "struct ", include NUL */
    const char *prefix = "";
    const char *type;

    if (streq (dec->type, "bool")) type = "bool_t";
    else if (streq (dec->type, "opaque")) type = "char";
    else{
      if (dec->prefix){
	s_print (buf, "%s ", dec->prefix);
	prefix = buf;
      }
      type = dec->type;
    }
    switch (dec->rel){
    case REL_ALIAS:
      f_print (fout, "func.template doit<%s%s>(&%s::%s,*this);\n", prefix, type, name,dec->name);
      break;
    case REL_VECTOR:
      f_print (fout, "func.template doit_vector<%s%s,%s>(&%s::%s,*this);\n", prefix, type, dec->array_max,name,dec->name);
      break;
    case REL_POINTER:
      f_print (fout, "func.template doit<%s%s *>(&%s::%s,*this);\n", prefix, type, name, dec->name);
      break;
    case REL_ARRAY:
      f_print (fout, "func.template doit_arraystruct<%s%s>(&%s::%s,*this);\n",prefix,type,name,dec->name);
      break;
    }
  }

  f_print (fout,"}\n");
}


static void 
prpc_generate_union_typemap_def(const char *enum_type){
  f_print (fout, "\t   template <typename T> static %s type_map();\n",enum_type);
}
static void 
prpc_generate_union_typemap_code(definition *def){
  const char *enum_type = def->def.un.enum_decl.type;
  const char *name = def->def_name;
  
  const declaration* default_decl = def->def.un.default_decl;
  f_print (fout, "template <typename T> %s %s::type_map(){\n",enum_type,name);
  if(default_decl){
    f_print (fout, "\t return %s;\n",default_decl->type);
  }else{
    f_print (fout, "\t return -1000;\n"); //this should cause a compiler error if it is instantiated to catch invalid types
  }
  f_print( fout, "}\n");

  //now forward declare the specializations for implementation in .C file

  case_list* c = def->def.un.cases;
  case_list* l;
  for (l = c; l != NULL; l = l->next){
    f_print (fout, "template <> %s %s::type_map<%s>();\n",enum_type,name,l->case_decl.type);
  }
}

static void 
prpc_generate_deepcopy_method_def(const char *type){
  f_print (fout, "\t   void deep_copy(const %s &rhs);\n",type);
}
static void 
prpc_generate_deepcopy_method_code(definition *def){
  if(def->def_kind != DEF_STRUCT && def->def_kind != DEF_CLASS && def->def_kind != DEF_UNION){ error("deepcopy method only applicable to struct, class and union\n"); }
  //now specialize the deepcopy class template
  const char *name = def->def_name;
  f_print (fout, "template<> struct rpc_deepcopy<%s>{\n",name);
  f_print (fout, "\tstatic void doit(%s &into, %s const &from);\n",name,name);
  f_print (fout, "};\n\n");
}
static void 
prpc_generate_print_method_def(){
  f_print (fout, "\t   void print(const std::string &prefix =\"\");\n");
}
static void 
prpc_generate_print_method_code(definition *def){
  if(def->def_kind != DEF_STRUCT && def->def_kind != DEF_CLASS && def->def_kind != DEF_UNION){ error("print method only applicable to struct, class and union\n"); }
  
  //now specialize the deepcopy class template
  const char *name = def->def_name;
  f_print (fout, "#ifndef _USE_STDLIB\n#error \"Cannot generate rpc_print commands without the standard library\"\n#endif\n");
  f_print (fout, "template<> struct rpc_print<%s>{\n",name);
  f_print (fout, "\tstatic void doit(%s const &what, const std::string &prefix=\"\" );\n",name,name);
  f_print (fout, "};\n\n");
}
static void
prpccommandprior(definition *def){
  //above the struct/class definition
  const char *name = def->def_name;
  decl_list *dl;
  if(def->def_kind == DEF_STRUCT) dl = def->def.st.decls;
  else if(def->def_kind == DEF_CLASS) dl = def->def.ct.decls;
  else if(def->def_kind == DEF_UNION) dl = def->def.un.other_decls;
  else return;

  int include_vml_templates = 0;
  
  decl_list *l;
  for (l = dl; l != NULL; l = l->next)
    {
      if(streq (l->decl.type, "rpccommand")){
	if(streq(l->decl.name, "GENERATE_DEEPCOPY_METHOD")){
	  include_vml_templates = 1;
	}else if(streq(l->decl.name, "GENERATE_PRINT_METHOD")){
	  include_vml_templates = 1;
	}
      }
    }

  if(include_vml_templates){
    f_print (fout, "\n#include <util/vml/vml_templates.h>\n"); //include the template header
  }
}


static void
prpccommandinternal(definition *def){
  const char *name = def->def_name;
  decl_list *dl;
  if(def->def_kind == DEF_STRUCT) dl = def->def.st.decls;
  else if(def->def_kind == DEF_CLASS) dl = def->def.ct.decls;
  else if(def->def_kind == DEF_UNION) dl = def->def.un.other_decls;
  else return;

  decl_list *l;
  for (l = dl; l != NULL; l = l->next)
    {
      if(streq (l->decl.type, "rpccommand")){
	if(streq(l->decl.name, "GENERATE_MEMBER_LAMBDAFUNC")){
	  prpc_generate_member_lambda_def();
	}else if(streq(l->decl.name, "GENERATE_UNION_TYPEMAP")){
	  if(def->def_kind != DEF_UNION){
	    error("rpccommand GENERATE_UNION_TYPEMAP not valid for non-union types\n");
	  }
	  prpc_generate_union_typemap_def(def->def.un.enum_decl.type);
	}else if(streq(l->decl.name, "GENERATE_DEEPCOPY_METHOD")){
	  prpc_generate_deepcopy_method_def(def->def_name);
	}else if(streq(l->decl.name, "GENERATE_PRINT_METHOD")){
	  prpc_generate_print_method_def();
	}
      }
    }
}

static void
prpccommandexternal(definition *def){
  const char *name = def->def_name;
  decl_list *dl;
  if(def->def_kind == DEF_STRUCT) dl = def->def.st.decls;
  else if(def->def_kind == DEF_CLASS) dl = def->def.ct.decls;
  else if(def->def_kind == DEF_UNION) dl = def->def.un.other_decls;
  else return;

  decl_list *l;
  for (l = dl; l != NULL; l = l->next)
    {
      if(streq (l->decl.type, "rpccommand")){
	if(streq(l->decl.name, "test")){
	  f_print (fout, "EXTERNAL TEST COMMAND HERE\n");
	}else if(streq(l->decl.name, "GENERATE_MEMBER_LAMBDAFUNC")){
	  prpc_generate_member_lambda_code(name,dl);
	}else if(streq(l->decl.name, "GENERATE_UNION_TYPEMAP")){
	  prpc_generate_union_typemap_code(def);
	}else if(streq(l->decl.name, "GENERATE_DEEPCOPY_METHOD")){
	  prpc_generate_deepcopy_method_code(def);
	}else if(streq(l->decl.name, "GENERATE_PRINT_METHOD")){
	  prpc_generate_print_method_code(def);
	}
      }
    }
}
/* End CK */
static void
pstructdef (definition *def)
{
  prpccommandprior(def);
  decl_list *l;
  const char *name = def->def_name;

  f_print (fout, "struct %s {\n", name);
  for (l = def->def.st.decls; l != NULL; l = l->next)
    {
      pdeclaration (name, &l->decl, 1, ";\n");
    }
  prpccommandinternal(def);
  f_print (fout, "};\n");
  f_print (fout, "typedef struct %s %s;\n", name, name);
  prpccommandexternal(def);
}

/*PAB class support*/
static void
pclassdef (definition *def)
{
  prpccommandprior(def);
  
  decl_list *l;
  const char *name = def->def_name;
  f_print (fout, "class VML;\n");
  f_print (fout, "class %s {\npublic:\n", name);

    f_print (fout, "\t bool Encode(char *filename,char *instance);\n");
    f_print (fout, "\t bool Decode(char *filename,char *instance);\n");
    f_print (fout, "\t bool Vml(VML *vmls,char *instance);\n");

  for (l = def->def.st.decls; l != NULL; l = l->next)
    {
      pdeclaration (name, &l->decl, 1, ";\n");
    }
  prpccommandinternal(def);
  f_print (fout, "};\n");
  
  prpccommandexternal(def);
}

/*CK include pragma support*/
static void
pincludedef (definition *def)
{
  f_print (fout,"CPS_END_NAMESPACE\n");
  if(def->def.in.is_relative) f_print (fout,"#include %s\n",def->def.in.file);
  else f_print (fout,"#include <%s>\n",def->def.in.file);
  f_print (fout,"CPS_START_NAMESPACE\n");
}

static void
puniondef (definition *def)
{
  prpccommandprior(def);
  case_list *l;
  const char *name = def->def_name;
  declaration *decl;

  f_print (fout, "struct %s {\n", name);
  decl = &def->def.un.enum_decl;
  if (streq (decl->type, "bool"))
    {
      f_print (fout, "\tbool_t %s;\n", decl->name);
    }
  else
    {
      f_print (fout, "\t%s %s;\n", decl->type, decl->name);
    }
  f_print (fout, "\tunion {\n");
  for (l = def->def.un.cases; l != NULL; l = l->next)
    {
      if (l->contflag == 0)
	pdeclaration (name, &l->case_decl, 2, ";\n");
    }
  decl = def->def.un.default_decl;
  if (decl && !streq (decl->type, "void"))
    {
      pdeclaration (name, decl, 2, ";\n");
    }
  f_print (fout, "\t} %s_u;\n", name);
  /*Added by CK*/
  decl_list *ld;
  for (ld = def->def.un.other_decls; ld != NULL; ld = ld->next)
    {
      pdeclaration (name, &ld->decl, 1, ";\n");
    }
  prpccommandinternal(def);  
  /*End CK*/
  f_print (fout, "};\n");
  f_print (fout, "typedef struct %s %s;\n", name, name);
  prpccommandexternal(def); /*Added by CK*/
}

static void
pdefine (const char *name, const char *num)
{
  f_print (fout, "#define %s %s\n", name, num);
}

static int
define_printed (proc_list *stop, version_list *start)
{
  version_list *vers;
  proc_list *proc;

  for (vers = start; vers != NULL; vers = vers->next)
    {
      for (proc = vers->procs; proc != NULL; proc = proc->next)
	{
	  if (proc == stop)
	    {
	      return 0;
	    }
	  else if (streq (proc->proc_name, stop->proc_name))
	    {
	      return 1;
	    }
	}
    }
  abort ();
  /* NOTREACHED */
}

static void
pfreeprocdef (const char *name, const char *vers, int mode)
{
  f_print (fout, "extern int ");
  pvname (name, vers);
  if (mode == 1)
    f_print (fout,"_freeresult (SVCXPRT *, xdrproc_t, caddr_t);\n");
  else
    f_print (fout,"_freeresult ();\n");
}

static void
pprogramdef (definition *def)
{
  version_list *vers;
  proc_list *proc;
  int i;
  const char *ext;

  pargdef (def);

  pdefine (def->def_name, def->def.pr.prog_num);
  for (vers = def->def.pr.versions; vers != NULL; vers = vers->next)
    {
      if (tblflag)
	{
	  f_print (fout, "extern struct rpcgen_table %s_%s_table[];\n",
		   locase (def->def_name), vers->vers_num);
	  f_print (fout, "extern %s_%s_nproc;\n",
		   locase (def->def_name), vers->vers_num);
	}
      pdefine (vers->vers_name, vers->vers_num);

      /*
       * Print out 2 definitions, one for ANSI-C, another for
       * old K & R C
       */

      if(!Cflag)
	{
	  ext = "extern  ";
	  for (proc = vers->procs; proc != NULL;
	       proc = proc->next)
	    {
	      if (!define_printed(proc, def->def.pr.versions))
		{
		  pdefine (proc->proc_name, proc->proc_num);
		}
	      f_print (fout, "%s", ext);
	      pprocdef (proc, vers, NULL, 0, 2);

	      if (mtflag)
		{
		  f_print(fout, "%s", ext);
		  pprocdef (proc, vers, NULL, 1, 2);
		}
	    }
	  pfreeprocdef (def->def_name, vers->vers_num, 2);
	}
      else
	{
	  for (i = 1; i < 3; i++)
	    {
	      if (i == 1)
		{
		  f_print (fout, "\n#if defined(__STDC__) || defined(__cplusplus)\n");
		  ext = "extern  ";
		}
	      else
		{
		  f_print (fout, "\n#else /* K&R C */\n");
		  ext = "extern  ";
		}

	      for (proc = vers->procs; proc != NULL; proc = proc->next)
		{
		  if (!define_printed(proc, def->def.pr.versions))
		    {
		      pdefine(proc->proc_name, proc->proc_num);
		    }
		  f_print (fout, "%s", ext);
		  pprocdef (proc, vers, "CLIENT *", 0, i);
		  f_print (fout, "%s", ext);
		  pprocdef (proc, vers, "struct svc_req *", 1, i);
		}
	      pfreeprocdef (def->def_name, vers->vers_num, i);
	    }
	  f_print (fout, "#endif /* K&R C */\n");
	}
    }
}

void
pprocdef (proc_list * proc, version_list * vp,
	  const char *addargtype, int server_p, int mode)
{
  if (mtflag)
    {/* Print MT style stubs */
      if (server_p)
	f_print (fout, "bool_t ");
      else
	f_print (fout, "enum clnt_stat ");
    }
  else
    {
      ptype (proc->res_prefix, proc->res_type, 1);
      f_print (fout, "* ");
    }
  if (server_p)
    pvname_svc (proc->proc_name, vp->vers_num);
  else
    pvname (proc->proc_name, vp->vers_num);

  /*
   * mode  1 = ANSI-C, mode 2 = K&R C
   */
  if (mode == 1)
    parglist (proc, addargtype);
  else
    f_print (fout, "();\n");
}

/* print out argument list of procedure */
static void
parglist (proc_list *proc, const char *addargtype)
{
  decl_list *dl;

  f_print(fout,"(");
  if (proc->arg_num < 2 && newstyle &&
      streq (proc->args.decls->decl.type, "void"))
    {
      /* 0 argument in new style:  do nothing */
    }
  else
    {
      for (dl = proc->args.decls; dl != NULL; dl = dl->next)
	{
	  ptype (dl->decl.prefix, dl->decl.type, 1);
	  if (!newstyle)
	    f_print (fout, "*");	/* old style passes by reference */

	  f_print (fout, ", ");
	}
    }
  if (mtflag)
    {
      ptype(proc->res_prefix, proc->res_type, 1);
      f_print(fout, "*, ");
    }

  f_print (fout, "%s);\n", addargtype);
}

static void
penumdef (definition *def)
{
  const char *name = def->def_name;
  enumval_list *l;
  const char *last = NULL;
  int count = 0;

  f_print (fout, "enum %s {\n", name);
  for (l = def->def.en.vals; l != NULL; l = l->next)
    {
      f_print (fout, "\t%s", l->name);
      if (l->assignment)
	{
	  f_print (fout, " = %s", l->assignment);
	  last = l->assignment;
	  count = 1;
	}
      else
	{
	  if (last == NULL)
	    {
	      f_print (fout, " = %d", count++);
	    }
	  else
	    {
	      f_print (fout, " = %s + %d", last, count++);
	    }
	}
      f_print (fout, ",\n");
    }
  f_print (fout, "};\n");
  f_print (fout, "typedef enum %s %s;\n", name, name);

  f_print (fout, "extern struct vml_enum_map %s_map[];\n", name);

}

static void
ptypedef (definition *def)
{
  const char *name = def->def_name;
  const char *old = def->def.ty.old_type;
  char prefix[8];	  /* enough to contain "struct ", including NUL */
  relation rel = def->def.ty.rel;

  if (!streq (name, old))
    {
      if (streq (old, "string"))
	{
	  old = "char";
	  rel = REL_POINTER;
	}
      else if (streq (old, "opaque"))
	{
	  old = "char";
	}
      else if (streq (old, "bool"))
	{
	  old = "bool_t";
	}
      if (undefined2 (old, name) && def->def.ty.old_prefix)
	{
	  s_print (prefix, "%s ", def->def.ty.old_prefix);
	}
      else
	{
	  prefix[0] = 0;
	}
      f_print (fout, "typedef ");
      switch (rel)
	{
	case REL_ARRAY:
	  f_print (fout, "struct {\n");
	  f_print (fout, "\tu_int %s_len;\n", name);
	  f_print (fout, "\t%s%s *%s_val;\n", prefix, old, name);
	  f_print (fout, "} %s", name);
	  break;
	case REL_POINTER:
	  f_print (fout, "%s%s *%s", prefix, old, name);
	  break;
	case REL_VECTOR:
	  f_print (fout, "%s%s %s[%s]", prefix, old, name,
		   def->def.ty.array_max);
	  break;
	case REL_ALIAS:
	  f_print (fout, "%s%s %s", prefix, old, name);
	  break;
	}
      f_print (fout, ";\n");
    }
}

void
pdeclaration (const char *name, declaration * dec, int tab,
	      const char *separator)
{
  char buf[8];			/* enough to hold "struct ", include NUL */
  const char *prefix;
  const char *type;

  if (streq (dec->type, "void"))
    {
      return;
    }
  if (streq (dec->type, "rpccommand"))
    {
      /*CK: This is a command for the preprocessor to do something, print nothing out here*/ 
      return;
    }
  tabify (fout, tab);
  if (streq (dec->type, name) && !dec->prefix)
    {
      f_print (fout, "struct ");
    }
  if (streq (dec->type, "string"))
    {
      f_print (fout, "char *%s", dec->name);
    }
  else
    {
      prefix = "";
      if (streq (dec->type, "bool"))
	{
	  type = "bool_t";
	}
      else if (streq (dec->type, "opaque"))
	{
	  type = "char";
	}
      else
	{
	  if (dec->prefix)
	    {
	      s_print (buf, "%s ", dec->prefix);
	      prefix = buf;
	    }
	  type = dec->type;
	}
      switch (dec->rel)
	{
	case REL_ALIAS:
	  f_print (fout, "%s%s %s", prefix, type, dec->name);
	  break;
	case REL_VECTOR:
	  f_print (fout, "%s%s %s[%s]", prefix, type, dec->name,
		   dec->array_max);
	  break;
	case REL_POINTER:
	  f_print (fout, "%s%s *%s", prefix, type, dec->name);
	  break;
	case REL_ARRAY:
	  f_print (fout, "struct {\n");
	  tabify (fout, tab);
	  f_print (fout, "\tu_int %s_len;\n", dec->name);
	  tabify (fout, tab);
	  f_print (fout, "\t%s%s *%s_val;\n", prefix, type, dec->name);
	  tabify (fout, tab);
	  f_print (fout, "} %s", dec->name);
	  break;
	}
    }
  f_print (fout, separator);
}

static int
undefined2 (const char *type, const char *stop)
{
  list *l;
  definition *def;

  for (l = defined; l != NULL; l = l->next)
    {
      def = (definition *) l->val;
      if (def->def_kind != DEF_PROGRAM)
	{
	  if (streq (def->def_name, stop))
	    {
	      return 1;
	    }
	  else if (streq (def->def_name, type))
	    {
	      return 0;
	    }
	}
    }
  return 1;
}
