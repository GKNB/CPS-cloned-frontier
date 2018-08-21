#include<config.h>
CPS_START_NAMESPACE
/*!\file
  \brief  Implementation of Fmobius class.

*/
//--------------------------------------------------------------------
//  CVS keywords
//
//  $Source: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/src/util/lattice/f_mobius/f_mobius.C,v $
//  $State: Exp $
//
//--------------------------------------------------------------------
//------------------------------------------------------------------
//
// f_mobius.C
//
// Fmobius is derived from FwilsonTypes and is relevant to
// domain wall fermions
//
//------------------------------------------------------------------

CPS_END_NAMESPACE
#include <config.h>
#include <util/lattice.h>
#include <util/dirac_op.h>
#include <util/time_cps.h>
#include <util/enum_func.h>
#include <util/time_cps.h>
#include <util/timer.h>
#include <util/lattice/fforce_wilson_type.h>

#include <util/zmobius.h> // for debug remove later 

//USING_NAMESPACE_CPS
CPS_START_NAMESPACE


FclassType Fmobius::Fclass(void) const {
  return F_CLASS_MOBIUS;
}

int Fmobius::FmatInv(Vector *f_out, Vector *f_in, 
		     CgArg *cg_arg, 
		     Float *true_res,
		     CnvFrmType cnv_frm,
		     PreserveType prs_f_in, int dminus)
{
  int iter;
  char *fname = "FmatInv(CgArg*,V*,V*,F*,CnvFrmType)";
  VRB.Func(cname,fname);
  Vector *temp, *dminus_in;

  unsigned long size = GJP.VolNodeSites() * GJP.SnodeSites() * 2 * Colors() * SpinComponents();
  if(prs_f_in==PRESERVE_YES){ 
    temp = (Vector *) smalloc(cname,fname, "temp",size * sizeof(Float));
    moveFloat((IFloat*)temp,(IFloat*)f_in, size);
  }

  DiracOpMobius dop(*this, f_out, f_in, cg_arg, cnv_frm);

  if (dminus){
  VRB.Result(cname,fname,"Dminus applied\n");
    dminus_in = (Vector *) smalloc(cname,fname, "temp",size * sizeof(Float));
  //TIZB: this is bug !  below Dminus multiplication is not in effect.
  // mult by Dminus
    dop.Dminus(dminus_in,f_in);
  // fixed. TB
//    f_in->CopyVec(f_out, size);
    moveFloat((IFloat*)f_in, (IFloat*)dminus_in, size);
    sfree(cname, fname,  "dminus_in",  dminus_in);
  }

#if 0
if (!dminus){
  int local_ls = GJP.SnodeSites();
  const int s_node_coor = GJP.SnodeCoor();
  const int ls_stride = 24 * GJP.VolNodeSites()/2;
  // Multiply 2*kappa
  // do even / odd 
  for(int ieo=0;ieo<2;++ieo){
    for(int s=0; s<local_ls;++s){
      int glb_s = s + local_ls*s_node_coor;
      const Complex kappa_b =
	1.0 / ( 2 * (GJP.Mobius_b()
		     *(4 - GJP.DwfHeight()) + GJP.DwfA5Inv()) );
 	VRB.Flow(cname,fname,"s=%d Mobius_b=%e kappa_b=%e %e\n",
	glb_s,GJP.Mobius_b(),kappa_b.real(),kappa_b.imag());
      int idx = s*ls_stride/2;// "/2" is for complex
      vecTimesEquComplex((Complex*)f_in+idx+ieo*size/4,
			 2.0*kappa_b, ls_stride);
    }
  }
  //moveFloat((IFloat*)f_in,(IFloat*)dminus_in, size);
}
#endif

  Float inv_time =-dclock();
  iter = dop.MatInv(true_res, prs_f_in);
  inv_time +=dclock();
  print_time(fname,"MatInv()",inv_time);
  if(prs_f_in==PRESERVE_YES){
    moveFloat((IFloat*)f_in,(IFloat*)temp, size);


    // TIZB check
if (1){
    Float norm;
    norm = f_out->NormSqGlbSum(size);
    if(!UniqueID()) printf("f_mobius  Norm out %.14e\n",norm);
    norm = f_in->NormSqGlbSum(size);
    if(!UniqueID()) printf("f_mobius Norm in %.14e\n",norm);
    dop.Mat(temp,f_out);  
    norm = temp->NormSqGlbSum(size);
    if(!UniqueID()) printf("f_mobius  Norm Mat*out %.14e\n",norm);
}

    sfree(cname, fname,  "temp",  temp);
  }

  // Return the number of iterations
  return iter;
}


// FmatInvMobius: same as FmatInv, except that we use mobius DWF
// formalism to speed up the CG inversion (via constructing initial guess).
// n_restart: How many restarts we perform
int Fmobius::FmatInv(Vector *f_out,
		     Vector *f_in,
		     MobiusArg *mob_l,
		     MobiusArg *mob_s,
		     Float *true_res,
		     CnvFrmType cnv_frm,
		     PreserveType prs_f_in)

{
  const char *fname = "FmatInv(V*, V*, mdwfArg, mdwfArg, F, Cnv, Preserv, int, F)";
  VRB.Func(cname,fname);
  
  // n_restart and stop_rsd is taken from mob_l
  int n_restart = mob_l->rsd_vec.rsd_vec_len;
  Float *rsd_vec = mob_l->rsd_vec.rsd_vec_val;

  // this implementation doesn't allow splitting in s direction(yet).
  if(GJP.Snodes() != 1){
    ERR.NotImplemented(cname, fname);
  }
  if(cnv_frm != CNV_FRM_YES){
    ERR.NotImplemented(cname, fname);
  }
  if(n_restart < 2){
    ERR.General(cname, fname, "Value %d is invalid for n_restart.\n", n_restart);
  }

  //----------------------------------------------------------------
  // Initialize kappa and ls. This has already been done by the Fmobius
  // call to mobius_init but is done here again in case the user
  // has modified the GJP.DwfA5Inv(), GJP.DwfHeight() or
  // GJP.SnodeSites() while in the scope of the Fmobius object.
  //----------------------------------------------------------------


  
  int ls = GJP.SnodeSites();
  int mob_l_ls = mob_l->ls;
  int mob_s_ls = mob_s->ls;
  Float mobius_b_l = mob_l->mobius_b_coeff;
  Float mobius_c_l = mob_l->mobius_c_coeff;
  Float mobius_b_s = mob_s->mobius_b_coeff;
  Float mobius_c_s = mob_s->mobius_c_coeff;
  int size_4d = GJP.VolNodeSites() * 2 * Colors() * SpinComponents();
  //int size_4d = GJP.VolNodeSites() * 24 ;
  int mob_l_size_5d = size_4d * mob_l_ls;
  int mob_s_size_5d = size_4d * mob_s_ls;

  // input times D_-
  Vector *dminus_in = (Vector *) smalloc(cname, fname, "dminus_in", sizeof(Float) * mob_l_size_5d);
  Vector *tmp_mob_l_5d = (Vector *) smalloc(cname, fname, "tmp_mob_l_5d", sizeof(Float) * mob_l_size_5d);
  Vector *tmp2_mob_l_5d = (Vector *) smalloc(cname, fname, "tmp2_mob_l_5d", sizeof(Float) * mob_l_size_5d);
  Vector *tmp_mob_s_5d = (Vector *) smalloc(cname, fname, "tmp_mob_s_5d", sizeof(Float) * mob_s_size_5d);

  CgArg *cg_arg_l = &mob_l->cg;
  CgArg *cg_arg_s = &mob_s->cg;

  
  // the first time, we solve in Mobius to some degree of accuracy
  {
    Float rsd = cg_arg_l->stop_rsd;
    cg_arg_l->stop_rsd = rsd_vec[0];
    GJP.SnodeSites(mob_l_ls);
    GJP.Mobius_b(mobius_b_l);
    GJP.Mobius_c(mobius_c_l);

    DiracOpMobius dop(*this, f_out, f_in, cg_arg_l, cnv_frm);
    dop.Dminus(dminus_in,f_in);
    int iter = dop.MatInv(f_out, dminus_in, PRESERVE_YES);
#if 1
      for(int i=0;i<GJP.VolNodeSites()*GJP.SnodeSites()*24;i++)
	printf("IN OUT1 %d %e %e\n",i,*((Float*)dminus_in+i),*((Float*)f_out+i) );
#endif
    cg_arg_l->stop_rsd = rsd;
    this->Convert(CANONICAL,dminus_in);
  }

  int restart_cnt;
  for(restart_cnt = 1; restart_cnt < n_restart; ++restart_cnt){
    //constructing the new residue
    tmp2_mob_l_5d->CopyVec(dminus_in, mob_l_size_5d);
    {
      GJP.SnodeSites(mob_l_ls);
      GJP.Mobius_b(mobius_b_l);
      GJP.Mobius_c(mobius_c_l);
      DiracOpMobius dop(*this, tmp_mob_l_5d, f_out, cg_arg_l, cnv_frm);
      dop.Mat(tmp_mob_l_5d, f_out);
    }
    tmp2_mob_l_5d->VecMinusEquVec(tmp_mob_l_5d, mob_l_size_5d);
    //end constructing the new residue(new residue now in tmp2_mob_l_5d)

#if 0
      for(int i=0;i<GJP.VolNodeSites()*GJP.SnodeSites()*24;i++)
	printf("IN OUT2 %d %e\n",i,*((Float*)tmp2_mob_l_5d+i));
#endif

    
    cg_arg_s->stop_rsd = rsd_vec[restart_cnt];
    tmp_mob_l_5d->VecZero(mob_l_size_5d);

    // Large PV
    {
      Float mass = cg_arg_l->mass;
      Float stop_rsd = cg_arg_l->stop_rsd;
      int max_num_iter = cg_arg_l->max_num_iter;

      cg_arg_l->mass = 1.0;
      cg_arg_l->stop_rsd = rsd_vec[restart_cnt];
      cg_arg_l->max_num_iter = cg_arg_s->max_num_iter;

      GJP.SnodeSites(mob_l_ls);
      GJP.Mobius_b(mobius_b_l);
      GJP.Mobius_c(mobius_c_l);
      DiracOpMobius dop(*this, tmp_mob_l_5d, tmp2_mob_l_5d, cg_arg_l, cnv_frm);
      int iter = dop.MatInv(tmp_mob_l_5d, tmp2_mob_l_5d, PRESERVE_YES);

      cg_arg_l->mass = mass;
      cg_arg_l->stop_rsd = stop_rsd;
      cg_arg_l->max_num_iter = max_num_iter;
    }
    SpinProject(tmp2_mob_l_5d, tmp_mob_l_5d, mob_l_ls, 1);
    
#if 1
      for(int i=0;i<GJP.VolNodeSites()*GJP.SnodeSites()*24;i++)
	printf("IN OUT3 %d %e\n",i,*((Float*)tmp2_mob_l_5d+i));
#endif

    // go to the small Mobius lattice
    tmp_mob_s_5d->VecZero(mob_s_size_5d);
    tmp_mob_s_5d->CopyVec(tmp2_mob_l_5d, size_4d);
    
    SpinProject(tmp_mob_l_5d, tmp_mob_s_5d, mob_s_ls, 0);
    

    // SMALL PV
    {
      Float mass= cg_arg_s->mass; 
      cg_arg_s->mass = 1.0;
      
      GJP.SnodeSites(mob_s_ls);
      GJP.Mobius_b(mobius_b_s);
      GJP.Mobius_c(mobius_c_s);
      DiracOpMobius dop(*this, tmp_mob_s_5d, tmp_mob_l_5d, cg_arg_s, cnv_frm);
      dop.Mat(tmp_mob_s_5d, tmp_mob_l_5d);
      
      cg_arg_s->mass = mass;
    }
#if 1
      for(int i=0;i<GJP.VolNodeSites()*GJP.SnodeSites()*24;i++)
	printf("IN OUT4 %d %e\n",i,*((Float*)tmp_mob_s_5d+i));
#endif

    
    // SMALL SLOVE
    tmp_mob_l_5d->VecZero(mob_s_size_5d);
    {
      //DiracOpMobius mdwf(*this, mob_s);
      GJP.SnodeSites(mob_s_ls);
      GJP.Mobius_b(mobius_b_s);
      GJP.Mobius_c(mobius_c_s);

      //if(!UniqueID())printf("TIZB stop_rsd=%e\n", cg_arg_s->stop_rsd);
      
      DiracOpMobius dop(*this, tmp_mob_l_5d, tmp_mob_s_5d, cg_arg_s, cnv_frm);
      dop.MatInv(tmp_mob_l_5d, tmp_mob_s_5d, true_res, PRESERVE_YES);
    }
    SpinProject(tmp_mob_s_5d, tmp_mob_l_5d, mob_s_ls, 1);

#if 1
      for(int i=0;i<GJP.VolNodeSites()*GJP.SnodeSites()*24;i++)
	printf("IN OUT5 %d %e\n",i,*((Float*)tmp_mob_s_5d+i));
#endif

    
    //OV2DWF
    tmp_mob_l_5d->VecNegative(tmp_mob_s_5d, size_4d);
    {
      Vector * tmp_2nd_plane = (Vector *)((Float *)tmp_mob_l_5d + size_4d);
      Vector * src_2nd_plane = (Vector *)((Float *)tmp2_mob_l_5d + size_4d);
      tmp_2nd_plane->CopyVec(src_2nd_plane, mob_l_size_5d - size_4d);
    }
    
    SpinProject(tmp2_mob_l_5d, tmp_mob_l_5d, mob_l_ls, 0);
    
    {
      GJP.SnodeSites(mob_l_ls);
      GJP.Mobius_b(mobius_b_l);
      GJP.Mobius_c(mobius_c_l);
      DiracOpMobius dop(*this, tmp_mob_l_5d, tmp2_mob_l_5d, cg_arg_l, cnv_frm);
      dop.Mat(tmp_mob_l_5d, tmp2_mob_l_5d);
    }
    
    // LARGE PV
    tmp2_mob_l_5d->VecZero(mob_l_size_5d);
    {
      Float mass = cg_arg_l->mass;
      Float stop_rsd = cg_arg_l->stop_rsd;
      int max_num_iter = cg_arg_l->max_num_iter;

      cg_arg_l->mass = 1.0;
      cg_arg_l->stop_rsd = cg_arg_s->stop_rsd;
      cg_arg_l->max_num_iter = cg_arg_s->max_num_iter;

      GJP.SnodeSites(mob_l_ls);
      GJP.Mobius_b(mobius_b_l);
      GJP.Mobius_c(mobius_c_l);
      DiracOpMobius dop(*this, tmp2_mob_l_5d, tmp_mob_l_5d, cg_arg_l, cnv_frm);
      dop.MatInv(tmp2_mob_l_5d, tmp_mob_l_5d, PRESERVE_NO);

      cg_arg_l->mass = mass;
      cg_arg_l->stop_rsd = stop_rsd;
      cg_arg_l->max_num_iter = max_num_iter;
    }
#if 1
      for(int i=0;i<GJP.VolNodeSites()*GJP.SnodeSites()*24;i++)
	printf("IN OUT6 %d %e\n",i,*((Float*)tmp_mob_s_5d+i));
#endif

      SpinProject(tmp_mob_l_5d, tmp2_mob_l_5d, mob_l_ls, 1);
    
    tmp_mob_l_5d->CopyVec(tmp_mob_s_5d, size_4d);
    
    SpinProject(tmp2_mob_l_5d, tmp_mob_l_5d, mob_l_ls, 0);

    f_out->VecAddEquVec(tmp2_mob_l_5d, mob_l_size_5d);

#if 1
      for(int i=0;i<GJP.VolNodeSites()*GJP.SnodeSites()*24;i++)
	printf("IN OUT7 %d %e\n",i,*((Float*)f_out+i));
#endif
  }

  // final solve
  int iter;
  {
    GJP.SnodeSites(mob_l_ls);
    GJP.Mobius_b(mobius_b_l);
    GJP.Mobius_c(mobius_c_l);

    DiracOpMobius dop(*this, f_out, dminus_in, cg_arg_l, cnv_frm);

    for(int i=0;i<GJP.VolNodeSites()*GJP.SnodeSites()*24;i++)
      printf("IN OUT9 %d %e\n",i,*((Float*)f_out+i));

    iter = dop.MatInv(f_out, dminus_in, true_res, PRESERVE_YES);


  }

  sfree(cname, fname,  "tmp_mob_l_5d",  tmp_mob_l_5d);
  sfree(cname, fname, "tmp2_mob_l_5d", tmp2_mob_l_5d);
  sfree(cname, fname,  "tmp_mob_s_5d",  tmp_mob_s_5d);
  sfree(cname, fname,  "dminus_in",  dminus_in);

  GJP.SnodeSites(ls);

  return iter;
}




//Copied from base class since it uses Dwf explicitly. 

//------------------------------------------------------------------
// int FeigSolv(Vector **f_eigenv, Float *lambda, int *valid_eig,
//              EigArg *eig_arg, 
//              CnvFrmType cnv_frm = CNV_FRM_YES):
// It solve  A * f_eigenv = lambda * f_eigenv where
// A is the fermion matrix (Dirac operator). The solution
// is done with the Ritz algorithm. eig_arg is the 
// structure that contains all the control parameters, f_eigenv
// is the fermion field eigenvectors, lambda are the
// returned eigenvalues.
// f_eigenv is defined on the whole lattice.
// The function returns the total number of Ritz iterations.
//------------------------------------------------------------------
int Fmobius::FeigSolv(Vector **f_eigenv, Float *lambda,
		      Float *chirality, int *valid_eig,
		      Float **hsum,
		      EigArg *eig_arg, 
		      CnvFrmType cnv_frm)
{

  int iter;
  char *fname = "FeigSolv(EigArg*,V*,F*,CnvFrmType)";
  VRB.Func(cname,fname);
  VRB.Warn(cname,fname,"Not checked\n");
  CgArg cg_arg;
  cg_arg.mass = eig_arg->mass;
  cg_arg.RitzMatOper = eig_arg->RitzMatOper;
  int N_eig = eig_arg->N_eig;
 
  if(cnv_frm == CNV_FRM_YES)
    for(int i=0; i < N_eig; ++i)
      Fconvert(f_eigenv[i], DWF_4D_EOPREC_EE, StrOrd());
 
  // Call constructor and solve for eigenvectors.
  // Use null pointers to fake out constructor.
  Vector *v1 = (Vector *)0;
  Vector *v2 = (Vector *)0;
 
  DiracOpMobius dop(*this, v1, v2, &cg_arg, CNV_FRM_NO);
  
  iter = dop.RitzEig(f_eigenv, lambda, valid_eig, eig_arg);
 
  if(cnv_frm == CNV_FRM_YES)
    for(int i=0; i < N_eig; ++i)
      Fconvert(f_eigenv[i], CANONICAL, StrOrd());
 
  // rescale eigenvalues to normal convention
  Float factor = 5. - GJP.DwfHeight();
  int i;
  //for(i=0; i<N_eig; ++i)
  //lambda[i] *= factor;

  // calculate chirality
  int Ncb = NumChkb(cg_arg.RitzMatOper);
  size_t f_size = GJP.VolNodeSites()*2*Colors()*SpinComponents()*Ncb/2;
  Vector *four = (Vector *) smalloc (cname,fname, "four", f_size * sizeof(Float));
  Vector *fourg5 = (Vector *) smalloc (cname,fname, "fourg5", f_size * sizeof(Float));
  Float help;

  for (i=0; i<N_eig; i++) {
    Ffive2four (four, f_eigenv[i], 0, GJP.Snodes()*GJP.SnodeSites()-1,Ncb);

    // normalize four
    factor=four->NormSqNode(f_size);
    glb_sum(&factor);
    factor=1./sqrt(factor);
    four->VecTimesEquFloat(factor,f_size);
    Gamma5(fourg5,four,GJP.VolNodeSites()*Ncb/2);
    chirality[i]= four->ReDotProductNode(fourg5, f_size);
    glb_sum(&chirality[i]);
  }

  // calculate hsum
  // slice sum the eigenvector density to make a 1D vector
  if (eig_arg->print_hsum) {

    if(cnv_frm == CNV_FRM_NO)
      for(int i=0; i < N_eig; ++i)
	Fconvert(f_eigenv[i], CANONICAL, StrOrd());

    Float *f_in = (Float *) smalloc (cname, fname, "f_in", 
		GJP.VolNodeSites()*GJP.SnodeSites()*sizeof(Float));
    
    for(i=0; i < N_eig; ++i) {
      IFloat *fp= (IFloat *) f_eigenv[i];
      int j;
      for (j=0; j<GJP.VolNodeSites()*GJP.SnodeSites(); j++, fp+= 24) 
	f_in[j]= Float (dotProduct (fp,fp,24));

      if (i==0) {
	for (j=0; j<GJP.VolNodeSites()*GJP.SnodeSites(); j++)
	  printf ("%f ", f_in[j]);
      }
      printf ("\n");

      f_eigenv[i]->SliceArraySumFive (hsum[i], f_in, eig_arg->hsum_dir);
    }

    sfree(cname, fname, "f_in", f_in);
  }
  sfree(cname,fname, "four",four);
  sfree(cname,fname, "fourg5",fourg5);
  // Return the number of iterations
  return iter;
}

//------------------------------------------------------------------
// Solve  A * f_eigenv = lambda * f_eigenv where
// A is the fermion matrix (Dirac operator). The solution
// is done with the Lanczos algorithm. eig_arg is the 
// structure that contains all the control parameters, f_eigenv
// is the fermion field eigenvectors, lambda are the
// returned eigenvalues.
// f_eigenv is defined on the whole lattice.
//------------------------------------------------------------------
int Fmobius::FeigSolv(Vector **f_eigenv, Float *lambda,
		      LanczosArg *eig_arg, 
		      CnvFrmType cnv_frm)
{
  int iter;
  char *fname = "FeigSolv(V*,F*,LanczosArg*,CnvFrmType)";
  VRB.Func(cname,fname);
  CgArg cg_arg;
  cg_arg.mass = eig_arg->mass;
  cg_arg.RitzMatOper = eig_arg->RitzMat_convcheck;
  cg_arg.eigen_shift = eig_arg->eigen_shift;
  // printf("Shift=%e\n",cg_arg.eigen_shift );exit(2);
  //int N_eig = eig_arg->N_eig;
  int nk = eig_arg->nk_lanczos_vectors;
  int np = eig_arg->np_lanczos_vectors;
  int maxiters = eig_arg->maxiters;
  Float stopres = eig_arg->stop_residual;
  MatrixPolynomialArg* cheby_arg = &(eig_arg->matpoly_arg);
  
  if(cnv_frm == CNV_FRM_YES) // convert only nk, not (nk+np)
    for(int i=0; i < nk; ++i) 
      Fconvert(f_eigenv[i], DWF_4D_EOPREC_EE, StrOrd());
 
  // Call constructor and solve for eigenvectors.
  // Use null pointers to fake out constructor.
  Vector *v1 = (Vector *)0;
  Vector *v2 = (Vector *)0;
  DiracOpMobius dop(*this, v1, v2, &cg_arg, CNV_FRM_NO);
  
  iter = dop.ImpResLanczos(f_eigenv, lambda,  eig_arg); 

  if(cnv_frm == CNV_FRM_YES)
    for(int i=0; i < nk; ++i) // convert only nk, not (nk+np)
      Fconvert(f_eigenv[i], CANONICAL, StrOrd());
  
  return iter;
}

void Fmobius::Fdslash(Vector *f_out, Vector *f_in, CgArg *cg_arg,
                 CnvFrmType cnv_frm, int dir_flag){
  char *fname = "Fdslash(V*,V*,CgArg*,CnvFrmType,i)";
  DiracOpMobius dop(*this, f_out, f_in, cg_arg, cnv_frm);
  dop.Dslash(f_out,f_in,CHKB_EVEN,DAG_NO);
#if 1
{
  unsigned long size = GJP.VolNodeSites() * GJP.SnodeSites() * 2 * Colors() * SpinComponents();
  int local_ls = GJP.SnodeSites();
  const int s_node_coor = GJP.SnodeCoor();
  const int ls_stride = 24 * GJP.VolNodeSites()/2;
  // Multiply 2*kappa
  // do even / odd 
  for(int ieo=0;ieo<2;++ieo){
    for(int s=0; s<local_ls;++s){
      int glb_s = s + local_ls*s_node_coor;
      const Complex kappa_b =
	1.0 / ( 2 * (GJP.Mobius_b()
		     *(4 - GJP.DwfHeight()) + GJP.DwfA5Inv()) );
// 	VRB.Flow(cname,fname,"s=%d Mobius_b=%e kappa_b=%e %e\n",
//	glb_s,GJP.Mobius_b(),kappa_b.real(),kappa_b.imag());
      int idx = s*ls_stride/2;// "/2" is for complex
      vecTimesEquComplex((Complex*)f_out+idx+ieo*size/4,
			 2.0*kappa_b, ls_stride);
    }
  }
  //moveFloat((IFloat*)f_in,(IFloat*)dminus_in, size);
}
#endif
}
int Fmobius::FmatEvlInv(Vector *f_out, Vector *f_in, 
		     CgArg *cg_arg, 
		     Float *true_res,
		     CnvFrmType cnv_frm)
{
  int iter;
  char *fname = "FmatEvlInv(CgArg*,V*,V*,F*,CnvFrmType)";
  VRB.Func(cname,fname);
  Float dtime = -dclock();

  DiracOpMobius dop(*this, f_out, f_in, cg_arg, cnv_frm);
  
#ifdef USE_QUDA
    iter = dop.QudaInvert(f_out, f_in, true_res, 1);
#else
    iter = dop.InvCg(f_out,f_in,&(cg_arg->true_rsd));
//  iter = dop.InvCg(&(cg_arg->true_rsd));
#endif
  if (true_res) *true_res = cg_arg ->true_rsd;

  // Return the number of iterations
  dtime += dclock();
  print_flops(cname,fname,0,dtime);
  VRB.FuncEnd(cname,fname);
  return iter;
}


#if 0
//------------------------------------------------------------------
// Overloaded function is same as original but with true_res=0;
//------------------------------------------------------------------
int Fmobius::FmatEvlInv(Vector *f_out, Vector *f_in, 
		     CgArg *cg_arg, 
		     CnvFrmType cnv_frm)
{ return FmatEvlInv(f_out, f_in, cg_arg, 0, cnv_frm); }
#endif


//------------------------------------------------------------------
// int FmatEvlMInv(Vector *f_out, Vector *f_in, 
//                Float shift[], int Nshift, 
//                CgArg **cg_arg, Float *true_res,
//		  CnvFrmType cnv_frm = CNV_FRM_YES):
// It calculates f_out where (A + shift)* f_out = f_in and
// A is the fermion matrix that appears in the HMC 
// evolution ([Dirac^dag Dirac]) and shift is a real shift of the 
// fermion matrix, with Nshift such shifts. The inversion is done 
// with the multishift conjugate gradient. cg_arg is the structure
// that contains all the control parameters, f_in is the
// fermion field source vector, f_out is the array of solution 
// vectors, f_in and f_out are defined on a checkerboard.
// The function returns the total number of CG iterations.
//------------------------------------------------------------------
int Fmobius::FmatEvlMInv(Vector **f_out, Vector *f_in, Float *shift, 
			  int Nshift, int isz, CgArg **cg_arg, 
			  CnvFrmType cnv_frm, MultiShiftSolveType type, 
			  Float *alpha, Vector **f_out_d)
{
  int iter;
  char *fname = "FmatEvlMInv(V**,V*,.....)";
  VRB.Func(cname,fname);
  Float dtime = -dclock();

  size_t f_size = GJP.VolNodeSites() * FsiteSize() / (FchkbEvl()+1);
  if(GJP.Gparity()) f_size*=2;

  Float dot = f_in -> NormSqGlbSum(f_size);
  VRB.Result(cname,fname,"f_size=%d\n",f_size);

  Float *RsdCG = new Float[Nshift];
  for (int s=0; s<Nshift; s++) RsdCG[s] = cg_arg[s]->stop_rsd;

  Float * out_p = (Float *)f_in;
  VRB.Result(cname,fname,"f_in=%g\n",out_p[0]);
  for(int i =0;i<Nshift;i++){
	Float a_tmp=1.; 
	if (alpha) a_tmp=alpha[i];
  VRB.Result(cname,fname,"%d: shift=%g alpha=%g RsdCG=%g\n",i,shift[i],a_tmp,RsdCG[i]);
  }
  //Fake the constructor
  DiracOpMobius dop(*this, f_out[0], f_in, cg_arg[0], cnv_frm);

  int return_value= dop.MInvCG(f_out,f_in,dot,shift,Nshift,isz,RsdCG,type,alpha);  
  out_p = (Float *)f_out[0];
  VRB.Result(cname,fname,"f_out[0]=%g\n",out_p[0]);
  for (int s=0; s<Nshift; s++) cg_arg[s]->true_rsd = RsdCG[s];  

  delete[] RsdCG;
  dtime += dclock();
  print_flops(cname,fname,0,dtime);
  return return_value;
}

// It evolves the canonical Momemtum mom:
// mom += coef * (phi1^\dag e_i(M) \phi2 + \phi2^\dag e_i(M^\dag) \phi1)
//
ForceArg Fmobius::EvolveMomFforce(Matrix *mom,
                                   Vector *phi1,
                                   Vector *phi2,
                                   Float mass, 
                                   Float coef)
{
    const char *fname = "EvolveMomFforceBase()";
    static Timer time(cname, fname);
    time.start(true);

    VRB.Result(cname,fname,"started\n");

    long f_size = (long)SPINOR_SIZE * GJP.VolNodeSites() * GJP.SnodeSites();
//    long f_size = (long)SPINOR_SIZE * GJP.VolNodeSites() * Fmobius::bfm_args[current_arg_idx].Ls;
    if(GJP.Gparity()) f_size*=2;

    Vector *v1 = (Vector *)smalloc(cname, fname, "v1", sizeof(Float) * f_size);
    Vector *v2 = (Vector *)smalloc(cname, fname, "v2", sizeof(Float) * f_size);

  {
    CgArg cg_arg ;
    cg_arg.mass = mass ;
#ifdef PROFILE
  time = -dclock();
#endif
    DiracOpMobius dwf(*this, v1,v2, &cg_arg, CNV_FRM_NO) ;
// need to be added here
//    dwf.CalcHmdForceVecs(v1,v2, phi1,phi2) ;
//    dwf.CalcHmdForceVecs(chi) ;
#ifdef PROFILE
  time += dclock();
  print_flops(fname,"CalcHmdForceVecs()",0,time);
  time = -dclock();
#endif

    Fconvert(v1,CANONICAL,DWF_4D_EOPREC_EE);
    Fconvert(v2,CANONICAL,DWF_4D_EOPREC_EE);
#ifdef PROFILE
  time += dclock();
  print_flops(fname,"Fconvert()",0,time);
#endif
  }

    FforceWilsonType cal_force(mom, this->GaugeField(), (Float*)v1, (Float*)v2, GJP.SnodeSites(), coef);
    ForceArg ret = cal_force.run();

    sfree(cname, fname, "v1", v1);
    sfree(cname, fname, "v2", v2);

    time.stop(true);
    return ret;
}

ForceArg Fmobius::EvolveMomFforce(Matrix *mom, Vector *frm, 
                               Float mass, Float step_size)
{
    const char *fname = "EvolveMomFforce()";
  

    size_t f_size_4d = SPINOR_SIZE * GJP.VolNodeSites();
    if(GJP.Gparity()) f_size_4d *= 2;
    const size_t f_size_cb = f_size_4d * GJP.SnodeSites() / 2;
    const size_t f_size= f_size_4d * GJP.SnodeSites() ;
  

    CgArg cg_arg;
    cg_arg.mass=mass;
    Vector *tmp = (Vector *)smalloc(cname, fname, "v1", sizeof(Float)*f_size);
{
    DiracOpMobius dwf(*this, tmp, frm, &cg_arg, CNV_FRM_NO) ;
    dwf.MatPc(tmp, frm);
}

    ForceArg f_arg = EvolveMomFforce(mom, tmp, frm, mass, -step_size);
    sfree(cname, fname, "tmp", tmp);

    return f_arg;
}

ForceArg Fmobius::RHMC_EvolveMomFforce(Matrix *mom, Vector **sol, int degree,
                                    int isz, Float *alpha, Float mass, Float dt,
                                    Vector **sol_d, ForceMeasure force_measure)
{
    const char *fname = "RHMC_EvolveMomFforce()";
    static Timer time(cname, fname);
    time.start(true);

    char *force_label=NULL;

    int g_size = GJP.VolNodeSites() * GsiteSize();
    if(GJP.Gparity()) g_size *= 2;

    Matrix *mom_tmp;

    if (force_measure == FORCE_MEASURE_YES) {
        mom_tmp = (Matrix*)smalloc(g_size * sizeof(Float),cname, fname, "mom_tmp");
        ((Vector*)mom_tmp) -> VecZero(g_size);
        force_label = new char[100];
    } else {
        mom_tmp = mom;
    }

    if(!UniqueID()){
	Float pvals[4];
	for(int ii=0;ii<4;ii++){
	    int off = 18 * ii + 2;
	    pvals[ii] = ((Float*)mom)[off];
	}
	if(UniqueID()==0) printf("Fmobius::RHMC_EvolveMomFforce initial mom Px(0) = %e, Py(0) = %e, Pz(0) = %e, Pt(0) = %e\n",pvals[0],pvals[1],pvals[2],pvals[3]);
    }  

    for (int i=0; i<degree; i++) {
        ForceArg Fdt = EvolveMomFforce(mom_tmp, sol[i], mass, alpha[i]*dt);

        if (force_measure == FORCE_MEASURE_YES) {
            sprintf(force_label, "Rational, mass = %e, pole = %d:", mass, i+isz);
            Fdt.print(dt, force_label);
        }

	if(!UniqueID()){
	    Float pvals[4];
	    for(int ii=0;ii<4;ii++){
		int off = 18 * ii + 2;
		pvals[ii] = ((Float*)mom_tmp)[off];
	    }
	    if(UniqueID()==0) printf("Fmobius::RHMC_EvolveMomFforce mom_tmp after pole %d:  Px(0) = %e, Py(0) = %e, Pz(0) = %e, Pt(0) = %e\n",i,pvals[0],pvals[1],pvals[2],pvals[3]);
	}  
    }

    ForceArg ret;

    // If measuring the force, need to measure and then sum to mom
    if (force_measure == FORCE_MEASURE_YES) {
        ret.measure(mom_tmp);
        ret.glb_reduce();

        fTimesV1PlusV2((IFloat*)mom, 1.0, (IFloat*)mom_tmp, (IFloat*)mom, g_size);

        delete[] force_label;
        sfree(mom_tmp, cname, fname, "mom_tmp");
    }

    time.stop(true);
    return ret;
}

Float Fmobius::SetPhi(Vector *phi, Vector *frm1, Vector *frm2,
		       Float mass, DagType dag){
  char *fname = "SetPhi(V*,V*,V*,F)";
  VRB.Func(cname,fname);
  CgArg cg_arg;
  cg_arg.mass = mass;

  if (phi == 0)
    ERR.Pointer(cname,fname,"phi") ;

  if (frm1 == 0)
    ERR.Pointer(cname,fname,"frm1") ;
  
  DiracOpMobius dwf(*this, frm1, 0, &cg_arg, CNV_FRM_NO) ;
  if (dag == DAG_YES) dwf.MatPcDag(phi, frm1) ;
  else dwf.MatPc(phi, frm1) ;

  Float temp= FhamiltonNode(frm1, frm1);
  VRB.FuncEnd(cname,fname);
  return temp;
}

CPS_END_NAMESPACE
