#include<config.h>

CPS_START_NAMESPACE
/*!\file
  \brief  Implementation of FwilsonTm class.

  $Id: f_wilsonTm.C,v 1.3 2012/03/26 13:50:12 chulwoo Exp $
*/
//--------------------------------------------------------------------
//  CVS keywords
//
//  $Source: /space/cvs/cps/cps++/src/util/lattice/f_wilsonTm/f_wilsonTm.C,v $
//  $State: Exp $
//
//--------------------------------------------------------------------
//------------------------------------------------------------------
//
// f_wilsonTm.C
//
// FwilsonTm is derived from Fwilson and is relevant to
// twisted-mass wilson fermions
//
//------------------------------------------------------------------

CPS_END_NAMESPACE
#ifdef USE_BFM_TM
#include <util/lattice/bfm_evo.h>
#define Printf if ( !UniqueID() ) printf
#endif
#include <util/lattice.h>
#include <util/dirac_op.h>
#include <util/wilson.h>
#include <util/verbose.h>
#include <util/gjp.h>
#include <util/error.h>
#include <util/enum_func.h>
#include <comms/scu.h>
#include <comms/glb.h>

#define BENCHMARK
#ifdef BENCHMARK
#include <util/qcdio.h>
#include <sys/time.h>
unsigned long WfmFlopsTm;
#ifndef timersub
#define timersub(a, b, result)                                                \
  do {                                                                        \
    (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;                             \
    (result)->tv_usec = (a)->tv_usec - (b)->tv_usec;                          \
    if ((result)->tv_usec < 0) {                                              \
      --(result)->tv_sec;                                                     \
      (result)->tv_usec += 1000000;                                           \
    }                                                                         \
  } while (0)
#endif
#endif

CPS_START_NAMESPACE

//------------------------------------------------------------------
// Initialize static variables.
//------------------------------------------------------------------

//------------------------------------------------------------------
// This constructor does nothing.
// All initialization done by Fwilson constructor.
//------------------------------------------------------------------
FwilsonTm::FwilsonTm()
: Fwilson()
{
  cname = "FwilsonTm";
  char *fname = "FwilsonTm()";
  VRB.Func(cname,fname);
}

//------------------------------------------------------------------
// This destructor does nothing.
// All termination done by Fwilson destructor.
//------------------------------------------------------------------
FwilsonTm::~FwilsonTm()
{
  char *fname = "~FwilsonTm()";
  VRB.Func(cname,fname);
}

//------------------------------------------------------------------
// FclassType Fclass():
// It returns the type of fermion class.
//------------------------------------------------------------------
FclassType FwilsonTm::Fclass() const{
  return F_CLASS_WILSON_TM;
}

//------------------------------------------------------------------
// int FsiteSize() and int FchkbEvl() not changed for twisted-mass Wilson fermions
//------------------------------------------------------------------

//------------------------------------------------------------------
//~~ modified in f_wilsonTm to create wilsonTm fermions 
//~~ see full notes in f_wilson
//------------------------------------------------------------------
int FwilsonTm::FmatEvlInv(Vector *f_out, Vector *f_in, 
			CgArg *cg_arg, 
			Float *true_res,
			CnvFrmType cnv_frm)
{
  int iter;
  char *fname = "FmatEvlInv(CgArg*,V*,V*,F*,CnvFrmType)";
  VRB.Func(cname,fname);

#if 1
{
  DiracOpWilsonTm wilson(*this, f_out, f_in, cg_arg, cnv_frm);

  WfmFlopsTm = 0;
  struct timeval t_start, t_stop;
  gettimeofday(&t_start,NULL);
  
  iter = wilson.InvCg(&(cg_arg->true_rsd));
  if (true_res) *true_res = cg_arg ->true_rsd;

  gettimeofday(&t_stop,NULL);
  timersub(&t_stop,&t_start,&t_start);
  double flops= (double)WfmFlopsTm;
  double secs = t_start.tv_sec + 1.E-6 *t_start.tv_usec;
//  printf("Wilson solve: %d iteratations %d flops %f Mflops per node\n",
//	 iter,WfmFlopsTm,flops/(secs*1000000) );
 }
#endif

  
  // Return the number of iterations
  return iter;
}

//------------------------------------------------------------------
// int FmatEvlMInv not changed for twisted-mass Wilson fermions
//------------------------------------------------------------------

//------------------------------------------------------------------
// void FminResExt not changed for twisted-mass Wilson fermions
//------------------------------------------------------------------

//------------------------------------------------------------------
// int FmatInv not changed for twisted-mass Wilson fermions
// NOTE: this call MatInv
//------------------------------------------------------------------

//------------------------------------------------------------------
// int FeigSolv not changed for twisted-mass Wilson fermions
//CK: This is incorrect as the RitzEig function calls up to MatPcMatPcDag etc which differ between wilson and wilsonTm. We 
//    must therefore use the correct Dirac operator
//    I have also added a BFM version used when the compile switch USE_BFM_TM is active
//------------------------------------------------------------------

int FwilsonTm::FeigSolv(Vector **f_eigenv, Float *lambda,
			Float *chirality, int *valid_eig,
			Float **hsum,
			EigArg *eig_arg, 
			CnvFrmType cnv_frm)
{
  char *fname = "FeigSolv(EigArg*,V*,F*,CnvFrmType)";
  VRB.Func(cname,fname);

  //Greg: I don't feel like added epsilon to EigArg
  VRB.Result(cname, fname, "Warning!! FwilsonTm::FeigSolv assumes epsilon = 0!!\n");

//#ifdef USE_BFM_TM
#if 0
  //CK: Added a BFM implementation using Hantao's bfm_evo class

  // only 1 eigenvalue can be computed now.
  if(eig_arg->N_eig != 1) {
    ERR.General(cname,fname,"BFM FeigSolv can only calculate a single eigenvalue\n");
  }
  if(eig_arg->RitzMatOper != MATPCDAG_MATPC &&
     eig_arg->RitzMatOper != NEG_MATPCDAG_MATPC) {
    ERR.NotImplemented(cname, fname);
  }
    
  if(cnv_frm == CNV_FRM_YES) ERR.General(cname,fname,"BFM FeigSolv not implemented for non-Wilson ordered fermions");

  bfmarg bfm_arg;

#if TARGET == BGQ
  omp_set_num_threads(64);
#else 
  omp_set_num_threads(1);
#endif

  bfm_arg.solver = WilsonTM;
  for(int i=0;i<4;i++) bfm_arg.node_latt[i] = GJP.NodeSites(i);
  bfm_arg.verbose=1;
  bfm_arg.reproduce=0;

#if TARGET == BGQ
  bfmarg::Threads(64);
#else
  bfmarg::Threads(1);
#endif

  bfmarg::Reproduce(0);
  bfmarg::ReproduceChecksum(0);
  bfmarg::ReproduceMasterCheck(0);
  bfmarg::Verbose(1);
  bfmarg::onepluskappanorm = 0;

  multi1d<int> ncoor = QDP::Layout::nodeCoord();
  multi1d<int> procs = QDP::Layout::logicalSize();


  Printf("%d dim machine\n\t", procs.size());
  for(int mu=0;mu<4;mu++){
    Printf("%d ", procs[mu]);
    if ( procs[mu]>1 ) bfm_arg.local_comm[mu] = 0;
    else bfm_arg.local_comm[mu] = 1;
  }
  Printf("\nLocal comm = ");
  for(int mu=0;mu<4;mu++){
    Printf("%d ", bfm_arg.local_comm[mu]);
  }
  Printf("\n");

  double mq= eig_arg->mass;
  double epsilon = 0; //eig_arg->epsilon;
  Printf("mq=%g epsilon=%g\n",mq,epsilon);

  bfm_arg.precon_5d = 0;
  bfm_arg.Ls   = 1;
  bfm_arg.M5   = 0.0;
  bfm_arg.mass = toDouble(mq);
  bfm_arg.twistedmass = toDouble(epsilon);
  bfm_arg.Csw  = 0.0;
  bfm_arg.max_iter = 10000;
  bfm_arg.residual = eig_arg->Rsdlam;
  bfm_arg.max_iter = eig_arg->MaxCG;
  Printf("Initialising bfm operator\n");

  bfm_evo<double> bd;
  bd.init(bfm_arg);

  VRB.Result(cname, fname, "residual = %17.10e max_iter = %d mass = %17.10e\n",
	     bd.residual, bd.max_iter, bd.mass);
  BondCond();
  bd.cps_importGauge((Float*)GaugeField());

  Fermion_t in = bd.allocFermion();
  bd.cps_impexcbFermion((Float *)f_eigenv[0], in, 1, 1);

#pragma omp parallel
  {
    lambda[0] = bd.ritz(in, eig_arg->RitzMatOper == MATPCDAG_MATPC);
  }

  bd.cps_impexcbFermion((Float *)f_eigenv[0], in, 0, 1);

  // correct the eigenvalue for a dumb convention problem.
  if(eig_arg->RitzMatOper == NEG_MATPCDAG_MATPC) lambda[0] = -lambda[0];

  valid_eig[0] = 1;
  bd.freeFermion(in);

  BondCond();

  //Compute chirality and whatnot
  int f_size = (GJP.VolNodeSites() * FsiteSize())/ 2; //single checkerboard field
  Vector* v1 = (Vector *)pmalloc(f_size*sizeof(Float));

  int nspinvect = GJP.VolNodeSites()/2;

  Gamma5(v1, f_eigenv[0], nspinvect);
  chirality[0] = f_eigenv[0]->ReDotProductGlbSum4D(v1, f_size);
  pfree(v1);

  // Regular CPS code rescales wilson eigenvalues to the convention  m + Dslash(U), i.e. lambda *= (4+m)
  // There is also a normalization difference between CPS and BFM preconditioned matrices:
  //MdagM_BFM = 0.25/kappa^2 * MdagM CPS
  //Hence should multiply BFM eigenvalues by 4*kappa^2 = 1/(4+m)^2  as well as the above factor 
  
  //For twisted mass
  Float kappa = 1.0/2.0/sqrt( (eig_arg->mass + 4.0)*(eig_arg->mass + 4.0) /*+ eig_arg->epsilon * eig_arg->epsilon*/ );
  Float factor = 4*kappa*kappa*(4+eig_arg->mass);

//1.0/(4.0 + eig_arg->mass);
   
  FILE* fp=Fopen(eig_arg->fname,"a");
  lambda[0] *= factor;  //rescale eigenvalue
  lambda[1] =  lambda[0]*lambda[0]; //squared evalue
  
  //print out eigenvalue, eigenvalue^2, chirality 
  Fprintf(fp,"%d %g %g %g %d\n",0,
	  (float)lambda[0],
	  (float)lambda[1],
	  (float)chirality[0],valid_eig[0]);

  Fclose(fp);

  if (eig_arg->print_hsum) ERR.General(cname,fname,"Hsum print not yet implemented (too lazy to copy/paste code)");

  return 0;


#else
  //CK: The regular CPS version

  int iter;

  CgArg cg_arg;
  cg_arg.mass = eig_arg->mass;
  cg_arg.epsilon = 0; //eig_arg->epsilon; //CK: passes down epsilon parameter for twisted mass Wilson fermions. Irrelevant here but this same function is used in FwilsonTm
  cg_arg.RitzMatOper = eig_arg->RitzMatOper;
  int N_eig = eig_arg->N_eig;
  int i;

  //=========================
  // convert fermion field
  //=========================

  if(cnv_frm == CNV_FRM_YES) //Fixed by CK to allow for single checkerboard input vectors as used in AlgActionRational. Previously it would always convert from CANONICAL to WILSON
    for(i=0; i < N_eig; ++i)  Fconvert(f_eigenv[i], WILSON, CANONICAL);

  //------------------------------------------------------------------
  //  we want both the eigenvalues of D_{hermitian} and
  //  D^{+}D.  To not change the arguments passed to RitzEig,
  //  we pass a float pointer which points to 2 * N_eig values
  //  and return both lambda and lambda^2 from RitzEig
  //------------------------------------------------------------------

  Float * lambda2 = (Float * ) smalloc (N_eig*2*sizeof(Float));
  if ( lambda2 == 0 ) ERR.Pointer(cname,fname, "lambda2");
  
  {
    DiracOpWilsonTm wilson(*this, (Vector*) 0 , (Vector*) 0, &cg_arg, CNV_FRM_NO);
    iter = wilson.RitzEig(f_eigenv, lambda2, valid_eig, eig_arg);
  }

  if(cnv_frm == CNV_FRM_YES) 
    for(i=0; i < N_eig; ++i) Fconvert(f_eigenv[i], CANONICAL, WILSON);


  /*
    the call to RitzEig returns a negative number if either the KS or CG maxes
    out, we wish to cope with this in alg_eig, so "pass it up". Clean up the
    storage order first in case we still want to use the eigenvectors as a
    guess.
  */
  if ( iter < 0 ) { return iter ; }


  // Compute chirality
  int Ncb = NumChkb(cg_arg.RitzMatOper);
  int f_size = (GJP.VolNodeSites() * FsiteSize()) * Ncb / 2; //CK: fixed

  Vector* v1 = (Vector *)smalloc(f_size*sizeof(Float));
  if (v1 == 0)
    ERR.Pointer(cname, fname, "v1");
  VRB.Smalloc(cname, fname, "v1", v1, f_size*sizeof(Float));

  int nspinvect = GJP.VolNodeSites() * Ncb/2;

  for(i=0; i < N_eig; ++i)
  {
    Gamma5(v1, f_eigenv[i], nspinvect);
    chirality[i] = f_eigenv[i]->ReDotProductGlbSum4D(v1, f_size);
  }

  VRB.Sfree(cname, fname, "v1", v1);
  sfree(v1);


  // rescale wilson eigenvalues to the convention  m + Dslash(U)
  Float factor = 4.0 + eig_arg->mass;
    
  
  FILE* fp=Fopen(eig_arg->fname,"a");
  for(i=0; i<N_eig; ++i)
    {
      lambda2[i] *= factor;	 		 //rescale eigenvalue
      lambda2[N_eig + i] *= ( factor * factor ); //rescale squared evalue
      lambda[i]=lambda2[i];                      //copy back
      
      //print out eigenvalue, eigenvalue^2, chirality 
      Fprintf(fp,"%d %g %g %g %d\n",i,
              (float)lambda2[i],
              (float)lambda2[N_eig + i],
	      (float)chirality[i],valid_eig[i]);
    }
  Fclose(fp);
  sfree(lambda2); 


  // Slice-sum the eigenvector density to make a 1D vector
  if (eig_arg->print_hsum){
    for(i=0; i < N_eig; ++i){
      //CK: The vector needs to be in canonical ordering. Thus if CNV_FRM_NO we need to convert to CANONICAL. If Ncb==1 we have
      //    only the odd part, so we will need to fill the even part with zeroes prior to conversion
      Vector* tosum = f_eigenv[i];
      if(cnv_frm == CNV_FRM_NO){
	//Create a temp copy of the eigenvector
	int alloc_size = f_size; if(Ncb==1) alloc_size *= 2;
	Float* full = (Float *)smalloc(alloc_size*sizeof(Float));
	for(int j=0;j<f_size;j++) full[j] = ((Float*)f_eigenv[i])[j];

	//Fill in even part with zero for Ncb==1
	if(Ncb==1) for(int j=f_size;j<alloc_size;j++) full[j] = 0; //zero even part

	//Convert
	if(cnv_frm == CNV_FRM_NO) Fconvert((Vector*)full, CANONICAL, WILSON);
	tosum = (Vector*)full;
      }
      tosum->NormSqArraySliceSum(hsum[i], FsiteSize(), eig_arg->hsum_dir);

      if(cnv_frm == CNV_FRM_NO) sfree(tosum);

    }
    
  }

  // The remaining part in QCDSP version are all about "downloading
  // eigenvectors", supposedly not applicable here.

  // Return the number of iterations
  return iter;
#endif
}

//------------------------------------------------------------------
// SetPhi(Vector *phi, Vector *frm1, Vector *frm2, Float mass,
//        Float epsilon, DagType dag):
// It sets the pseudofermion field phi from frm1, frm2.
// Note that frm2 is not used.
// Modified - now returns the (trivial) value of the action
// Now sets epsilon in cg_arg from new input parameter
//------------------------------------------------------------------
Float FwilsonTm::SetPhi(Vector *phi, Vector *frm1, Vector *frm2,
		      Float mass, Float epsilon, DagType dag){
  char *fname = "SetPhi(V*,V*,V*,F,F)";
  VRB.Func(cname,fname);

  CgArg cg_arg;
  cg_arg.mass = mass;
  cg_arg.epsilon = epsilon;

  if (phi == 0)
    ERR.Pointer(cname,fname,"phi") ;

  if (frm1 == 0)
    ERR.Pointer(cname,fname,"frm1") ;

  DiracOpWilsonTm wilson(*this, frm1, frm2, &cg_arg, CNV_FRM_NO) ;
  
  if (dag == DAG_YES) wilson.MatPcDag(phi, frm1) ;
  else wilson.MatPc(phi, frm1) ;

  return FhamiltonNode(frm1, frm1);
}

//------------------------------------------------------------------
// SetPhi(Vector *phi, Vector *frm1, Vector *frm2, Float mass, DagType dag):
// should never be called by wilsonTm fermions
//------------------------------------------------------------------
Float FwilsonTm::SetPhi(Vector *phi, Vector *frm1, Vector *frm2,
		      Float mass, DagType dag){
  char *fname = "SetPhi(V*,V*,V*,F,DagType)";
  VRB.Func(cname,fname);

  ERR.General(cname,fname,"f_wilsonTm: SetPhi(V*,V*,V*,F,DagType) not implemented here\n");

  return Float(0.0);
}

//------------------------------------------------------------------
// ForceArg RHMC_EvolveMomFforce not changed for twisted-mass Wilson fermions
//------------------------------------------------------------------

//------------------------------------------------------------------
// Float BhamiltonNode(Vector *boson, Float mass, Float epsilon):
// The boson Hamiltonian of the node sublattice.
// Now sets epsilon in cg_arg from new input parameter
//------------------------------------------------------------------
Float FwilsonTm::BhamiltonNode(Vector *boson, Float mass, Float epsilon){
  char *fname = "BhamiltonNode(V*,F,F)";
  VRB.Func(cname,fname);
  
  CgArg cg_arg;
  cg_arg.mass = mass;
  cg_arg.epsilon = epsilon;

  if (boson == 0)
    ERR.Pointer(cname,fname,"boson");

  int f_size = (GJP.VolNodeSites() * FsiteSize()) >> 1 ;

  Vector *bsn_tmp = (Vector *)
    smalloc(f_size*sizeof(Float));

  char *str_tmp = "bsn_tmp" ;

  if (bsn_tmp == 0)
    ERR.Pointer(cname,fname,str_tmp) ;

  VRB.Smalloc(cname,fname,str_tmp,bsn_tmp,f_size*sizeof(Float));

  DiracOpWilsonTm wilson(*this, boson, bsn_tmp, &cg_arg, CNV_FRM_NO) ;

  wilson.MatPc(bsn_tmp,boson);

  Float ret_val = bsn_tmp->NormSqNode(f_size) ;

  VRB.Sfree(cname,fname,str_tmp,bsn_tmp);

  sfree(bsn_tmp) ;

  return ret_val;
}

//------------------------------------------------------------------
// Float BhamiltonNode(Vector *boson, Float mass, Float epsilon):
// should never be called by wilsonTm fermions
//------------------------------------------------------------------
Float FwilsonTm::BhamiltonNode(Vector *boson, Float mass){
  char *fname = "BhamiltonNode(V*,F)";
  VRB.Func(cname,fname);
  
  ERR.General(cname,fname,"f_wilsonTm: BhamiltonNode(V*,F) not implemented here\n");

  return Float(0.0);
}

//------------------------------------------------------------------
// EvolveMomFforce(Matrix *mom, Vector *frm, Float mass, 
//                 Float epsilon, Float dt):
// It evolves the canonical momentum mom by dt
// using the fermion force.
// Now sets epsilon in cg_arg from new input parameter
// chi <- (M_f^\dag M_f)^{-1} M_f^\dag (RGV)
//------------------------------------------------------------------
ForceArg FwilsonTm::EvolveMomFforce(Matrix *mom, Vector *chi, 
			      Float mass, Float epsilon, Float dt)
{
  char *fname = "EvolveMomFforce(M*,V*,F,F,F)";
  VRB.Func(cname,fname);

  Matrix *gauge = GaugeField() ;

  if (Colors() != 3) ERR.General(cname,fname,"Wrong nbr of colors.") ;
  if (SpinComponents() != 4) ERR.General(cname,fname,"Wrong nbr of spin comp.") ;
  if (mom == 0) ERR.Pointer(cname,fname,"mom") ;
  if (chi == 0) ERR.Pointer(cname,fname,"chi") ;

//------------------------------------------------------------------
// allocate space for two CANONICAL fermion fields.
//------------------------------------------------------------------
  int f_size = FsiteSize() * GJP.VolNodeSites() ;

  char *str_v1 = "v1" ;
  Vector *v1 = (Vector *)smalloc(cname, fname, str_v1, f_size*sizeof(Float)) ;

  char *str_v2 = "v2" ;
  Vector *v2 = (Vector *)smalloc(cname, fname, str_v2, f_size*sizeof(Float)) ;

//------------------------------------------------------------------
// allocate space for two CANONICAL fermion field on a site.
//------------------------------------------------------------------

  char *str_site_v1 = "site_v1";
  Float *site_v1 = (Float *)smalloc(cname, fname, str_site_v1, FsiteSize()*sizeof(Float));

  char *str_site_v2 = "site_v2";
  Float *site_v2 = (Float *)smalloc(cname, fname, str_site_v2, FsiteSize()*sizeof(Float));

  Float L1 = 0.0;
  Float L2 = 0.0;
  Float Linf = 0.0;


  {
    CgArg cg_arg ;
    cg_arg.mass = mass ;
    cg_arg.epsilon = epsilon;

    DiracOpWilsonTm wilson(*this, v1, v2, &cg_arg, CNV_FRM_YES) ;
    // chi <- (M_f^\dag M_f)^{-1} M_f^\dag (RGV)
    wilson.CalcHmdForceVecs(chi) ;
  }
#if 0
  VRB.Result(cname,fname,"Being skipped for debugging!");
#else

  int x, y, z, t, lx, ly, lz, lt ;

  lx = GJP.XnodeSites() ;
  ly = GJP.YnodeSites() ;
  lz = GJP.ZnodeSites() ;
  lt = GJP.TnodeSites() ;

//------------------------------------------------------------------
// start by summing first over direction (mu) and then over site
// to allow SCU transfers to happen face-by-face in the outermost
// loop.
//------------------------------------------------------------------

  int mu ;

  Matrix tmp, f ;

  for (mu=0; mu<4; mu++) {
    for (t=0; t<lt; t++)
    for (z=0; z<lz; z++)
    for (y=0; y<ly; y++)
    for (x=0; x<lx; x++) {
      int gauge_offset = x+lx*(y+ly*(z+lz*t)) ;
      int vec_offset = FsiteSize()*gauge_offset ;
      gauge_offset = mu+4*gauge_offset ;

      Float *v1_plus_mu ;
      Float *v2_plus_mu ;
      int vec_plus_mu_offset = FsiteSize() ;

      Float coeff = -2.0 * dt ;

      switch (mu) {
        case 0 :
          vec_plus_mu_offset *= (x+1)%lx+lx*(y+ly*(z+lz*t)) ;
          if ((x+1) == lx) {
            getPlusData( (IFloat *)site_v1,
                         (IFloat *)v1+vec_plus_mu_offset, FsiteSize(), mu) ;
            getPlusData( (IFloat *)site_v2,
                         (IFloat *)v2+vec_plus_mu_offset, FsiteSize(), mu) ;
            v1_plus_mu = site_v1 ;                        
            v2_plus_mu = site_v2 ;                        
            if (GJP.XnodeBc()==BND_CND_APRD) coeff = -coeff ;
          } else {
            v1_plus_mu = (Float *)v1+vec_plus_mu_offset ;
            v2_plus_mu = (Float *)v2+vec_plus_mu_offset ;
          }
          break ;
        case 1 :
          vec_plus_mu_offset *= x+lx*((y+1)%ly+ly*(z+lz*t)) ;
          if ((y+1) == ly) {
            getPlusData( (IFloat *)site_v1,
                         (IFloat *)v1+vec_plus_mu_offset, FsiteSize(), mu) ;
            getPlusData( (IFloat *)site_v2,
                         (IFloat *)v2+vec_plus_mu_offset, FsiteSize(), mu) ;
            v1_plus_mu = site_v1 ;                        
            v2_plus_mu = site_v2 ;                        
            if (GJP.YnodeBc()==BND_CND_APRD) coeff = -coeff ;
          } else {
            v1_plus_mu = (Float *)v1+vec_plus_mu_offset ;
            v2_plus_mu = (Float *)v2+vec_plus_mu_offset ;
          }
          break ;
        case 2 :
          vec_plus_mu_offset *= x+lx*(y+ly*((z+1)%lz+lz*t)) ;
          if ((z+1) == lz) {
            getPlusData( (IFloat *)site_v1,
                         (IFloat *)v1+vec_plus_mu_offset, FsiteSize(), mu) ;
            getPlusData( (IFloat *)site_v2,
                         (IFloat *)v2+vec_plus_mu_offset, FsiteSize(), mu) ;
            v1_plus_mu = site_v1 ;
            v2_plus_mu = site_v2 ;
            if (GJP.ZnodeBc()==BND_CND_APRD) coeff = -coeff ;
          } else {
            v1_plus_mu = (Float *)v1+vec_plus_mu_offset ;
            v2_plus_mu = (Float *)v2+vec_plus_mu_offset ;
          }
          break ;
        case 3 :
          vec_plus_mu_offset *= x+lx*(y+ly*(z+lz*((t+1)%lt))) ;
          if ((t+1) == lt) {
            getPlusData( (IFloat *)site_v1,
                         (IFloat *)v1+vec_plus_mu_offset, FsiteSize(), mu) ;
            getPlusData( (IFloat *)site_v2,
                         (IFloat *)v2+vec_plus_mu_offset, FsiteSize(), mu) ;
            v1_plus_mu = site_v1 ;
            v2_plus_mu = site_v2 ;
            if (GJP.TnodeBc()==BND_CND_APRD) coeff = -coeff ;
          } else {
            v1_plus_mu = (Float *)v1+vec_plus_mu_offset ;
            v2_plus_mu = (Float *)v2+vec_plus_mu_offset ;
          }
      } // end switch mu

      sproj_tr[mu](   (IFloat *)&tmp,
                      (IFloat *)v1_plus_mu,
                      (IFloat *)v2+vec_offset, 1, 0, 0);

      sproj_tr[mu+4]( (IFloat *)&f,
                      (IFloat *)v2_plus_mu,
                      (IFloat *)v1+vec_offset, 1, 0, 0);

      tmp += f ;

      f.DotMEqual(*(gauge+gauge_offset), tmp) ;

      tmp.Dagger(f) ;

      f.TrLessAntiHermMatrix(tmp) ;

      f *= coeff ;

      *(mom+gauge_offset) += f ;
      Float norm = f.norm();
      Float tmp = sqrt(norm);
      L1 += tmp;
      L2 += norm;
      Linf = (tmp>Linf ? tmp : Linf);
    }
  }
#endif

//------------------------------------------------------------------
// deallocate space for two CANONICAL fermion fields on a site.
//------------------------------------------------------------------
  VRB.Sfree(cname, fname, str_site_v2, site_v2) ;
  sfree(site_v2) ;

  VRB.Sfree(cname, fname, str_site_v1, site_v1) ;
  sfree(site_v1) ;

//------------------------------------------------------------------
// deallocate space for two CANONICAL fermion fields.
//------------------------------------------------------------------
  VRB.Sfree(cname, fname, str_v2, v2) ;
  sfree(v2) ;

  VRB.Sfree(cname, fname, str_v1, v1) ;
  sfree(v1) ;

  glb_sum(&L1);
  glb_sum(&L2);
  glb_max(&Linf);

  L1 /= 4.0*GJP.VolSites();
  L2 /= 4.0*GJP.VolSites();
  VRB.FuncEnd(cname,fname);

  return ForceArg(L1, sqrt(L2), Linf);
}

//------------------------------------------------------------------
// EvolveMomFforce(Matrix *mom, Vector *frm, Float mass, Float dt):
// should never be called by wilsonTm fermions
//------------------------------------------------------------------
ForceArg FwilsonTm::EvolveMomFforce(Matrix *mom, Vector *chi, 
			      Float mass, Float dt)
{
  char *fname = "EvolveMomFforce(M*,V*,F,F)";
  VRB.Func(cname,fname);

  ERR.General(cname,fname,"f_wilsonTm: EvolveMomFforce(M*,V*,F,F) not implemented here\n");

  return ForceArg(0.0,0.0,0.0);
}


//------------------------------------------------------------------
// EvolveMomFforce(Matrix *mom, Vector *frm, Vector *frm,
//                 Float mass, Float epsilon, Float dt):
// It evolves the canonical momentum mom by dt
// using the boson-component of the alg_quotient force.
// Now sets epsilon in cg_arg from new input parameter
// chi <- (M_f^\dag M_f)^{-1} M_f^\dag (RGV)
// phi <- M_b (M_b^\dag M_b)^{-1} M_f^\dag (RGV)
//------------------------------------------------------------------
ForceArg FwilsonTm::EvolveMomFforce(Matrix *mom, Vector *chi, Vector *eta,
		      Float mass, Float epsilon, Float dt) {
  char *fname = "EvolveMomFforce(M*,V*,V*,F,F,F)";
  VRB.Func(cname,fname);

  Matrix *gauge = GaugeField() ;

  if (Colors() != 3)       { ERR.General(cname,fname,"Wrong nbr of colors.") ; }
  if (SpinComponents()!=4) { ERR.General(cname,fname,"Wrong nbr of spin comp.") ;}
  if (mom == 0)            { ERR.Pointer(cname,fname,"mom") ; }
  if (eta == 0)            { ERR.Pointer(cname,fname,"eta") ; }

//------------------------------------------------------------------
// allocate space for two CANONICAL fermion fields.
// these are all full fermion vector sizes ( i.e. *not* preconditioned )
//------------------------------------------------------------------

  const int f_size        ( FsiteSize() * GJP.VolNodeSites() );

  char *str_v1 = "v1" ;
  Vector *v1 = (Vector *)smalloc(f_size*sizeof(Float)) ;
  if (v1 == 0) ERR.Pointer(cname, fname, str_v1) ;
  VRB.Smalloc(cname, fname, str_v1, v1, f_size*sizeof(Float)) ;

  char *str_v2 = "v2" ;
  Vector *v2 = (Vector *)smalloc(f_size*sizeof(Float)) ;
  if (v2 == 0) ERR.Pointer(cname, fname, str_v2) ;
  VRB.Smalloc(cname, fname, str_v2, v2, f_size*sizeof(Float)) ;

//------------------------------------------------------------------
// two fermion vectors at a single position
//    - these will be used to store off-node
//      field components
//------------------------------------------------------------------

  char *str_site_v1 = "site_v1";
  Float *site_v1 = (Float *)smalloc(FsiteSize()*sizeof(Float));
  if (site_v1 == 0) ERR.Pointer(cname, fname, str_site_v1) ;
  VRB.Smalloc(cname, fname, str_site_v1, site_v1,
    FsiteSize()*sizeof(Float)) ;

  char *str_site_v2 = "site_v2";
  Float *site_v2 = (Float *)smalloc(FsiteSize()*sizeof(Float));
  if (site_v2 == 0) ERR.Pointer(cname, fname, str_site_v2) ;
  VRB.Smalloc(cname, fname, str_site_v2, site_v2,
    FsiteSize()*sizeof(Float)) ;
  
  Float L1 = 0.0;
  Float L2 = 0.0;
  Float Linf = 0.0;


  {
    CgArg cg_arg ;
    cg_arg.mass = mass ;
    cg_arg.epsilon = epsilon;

    DiracOpWilsonTm wilson(*this, v1, v2, &cg_arg, CNV_FRM_YES) ;

//~~
//~~ fermion version:  	wilson.CalcHmdForceVecs(chi)
//~~ boson version:  	wilson.CalcBsnForceVecs(chi, eta)
//~~
    // chi <- (M_f^\dag M_f)^{-1} M_f^\dag (RGV)
    // phi <- M_b (M_b^\dag M_b)^{-1} M_f^\dag (RGV)
    wilson.CalcBsnForceVecs(chi, eta) ;
  }

#if 0
  VRB.Result(cname,fname,"Being skipped for debugging!");
#else

  // evolve the momenta by the fermion force
  int mu, x, y, z, t;
 
  const int lx(GJP.XnodeSites());
  const int ly(GJP.YnodeSites());
  const int lz(GJP.ZnodeSites());
  const int lt(GJP.TnodeSites());

//------------------------------------------------------------------
// start by summing first over direction (mu) and then over site
// to allow SCU transfers to happen face-by-face in the outermost
// loop.
//------------------------------------------------------------------

// (1-gamma_\mu) Tr_s[v1(x+\mu) v2^{\dagger}(x)] +          
// 		(1+gamma_\mu) Tr_s [v2(x+\mu) v1^{\dagger}(x)]
  for (mu=0; mu<4; mu++) {
    for (t=0; t<lt; t++)
      for (z=0; z<lz; z++)
        for (y=0; y<ly; y++)
          for (x=0; x<lx; x++) {
            // position offset
            int gauge_offset = x+lx*(y+ly*(z+lz*t)) ;
            
            // offset for vector field at this point
            int vec_offset = FsiteSize()*gauge_offset ;

            // offset for link in mu direction from this point
            gauge_offset = mu+4*gauge_offset ;

            Float *v1_plus_mu ;
            Float *v2_plus_mu ;
            int vec_plus_mu_offset = FsiteSize() ;

            // sign of coeff (look at momenta update)
            Float coeff = -2.0 * dt ;

            switch (mu) {
              case 0 :
                // next position in mu direction
                vec_plus_mu_offset *= (x+1)%lx+lx*(y+ly*(z+lz*t)) ;
                if ((x+1) == lx) {
                   // off-node
                   // fill site_v1 and site_v2 with v1 and v2 data
                   // from x=0 on next node, need loop because
                   // data is not contiguous in memory 
                   getPlusData( (IFloat *)site_v1,
                              (IFloat *)v1+vec_plus_mu_offset, FsiteSize(), mu) ;
                   getPlusData( (IFloat *)site_v2,
                              (IFloat *)v2+vec_plus_mu_offset, FsiteSize(), mu) ;

                   v1_plus_mu = site_v1 ;                        
                   v2_plus_mu = site_v2 ;                        

                   // GJP.XnodeBc() gives the forward boundary
                   // condition only (so this should work).
                   if (GJP.XnodeBc()==BND_CND_APRD) coeff = -coeff ;
                } else {
                   //
                   //  on - node: just add offset to v1 and v2
                   // (they are now 1 forward in the mu direction )
                   //
                   v1_plus_mu = (Float *)v1+vec_plus_mu_offset ;
                   v2_plus_mu = (Float *)v2+vec_plus_mu_offset ;
                }
                break ;

        // Repeat for the other directions
        case 1 :
          vec_plus_mu_offset *= x+lx*((y+1)%ly+ly*(z+lz*t)) ;
          if ((y+1) == ly) {
            getPlusData( (IFloat *)site_v1,
                         (IFloat *)v1+vec_plus_mu_offset, FsiteSize(), mu) ;
            getPlusData( (IFloat *)site_v2,
                         (IFloat *)v2+vec_plus_mu_offset, FsiteSize(), mu) ;
            v1_plus_mu = site_v1 ;                        
            v2_plus_mu = site_v2 ;                        
            if (GJP.YnodeBc()==BND_CND_APRD) coeff = -coeff ;
          } else {
            v1_plus_mu = (Float *)v1+vec_plus_mu_offset ;
            v2_plus_mu = (Float *)v2+vec_plus_mu_offset ;
          }
          break ;
        case 2 :
          vec_plus_mu_offset *= x+lx*(y+ly*((z+1)%lz+lz*t)) ;
          if ((z+1) == lz) {
            getPlusData( (IFloat *)site_v1,
                         (IFloat *)v1+vec_plus_mu_offset, FsiteSize(), mu) ;
            getPlusData( (IFloat *)site_v2,
                         (IFloat *)v2+vec_plus_mu_offset, FsiteSize(), mu) ;
            v1_plus_mu = site_v1 ;
            v2_plus_mu = site_v2 ;
            if (GJP.ZnodeBc()==BND_CND_APRD) coeff = -coeff ;
          } else {
            v1_plus_mu = (Float *)v1+vec_plus_mu_offset ;
            v2_plus_mu = (Float *)v2+vec_plus_mu_offset ;
          }
          break ;
        case 3 :
          vec_plus_mu_offset *= x+lx*(y+ly*(z+lz*((t+1)%lt))) ;
          if ((t+1) == lt) {
            getPlusData( (IFloat *)site_v1,
                         (IFloat *)v1+vec_plus_mu_offset, FsiteSize(), mu) ;
            getPlusData( (IFloat *)site_v2,
                         (IFloat *)v2+vec_plus_mu_offset, FsiteSize(), mu) ;
            v1_plus_mu = site_v1 ;
            v2_plus_mu = site_v2 ;
            if (GJP.TnodeBc()==BND_CND_APRD) coeff = -coeff ;
          } else {
            v1_plus_mu = (Float *)v1+vec_plus_mu_offset ;
            v2_plus_mu = (Float *)v2+vec_plus_mu_offset ;
          }
      } // end (the evil) mu switch 

      Matrix tmp_mat1, tmp_mat2;  

// ( 1 - gamma_\mu ) Tr_s [ v1(x+\mu) v2^{\dagger}(x) ]           
      sproj_tr[mu](   (IFloat *)&tmp_mat1,   	// output color matrix
                      (IFloat *)v1_plus_mu,		// row vector, NOT conjugated
                      (IFloat *)v2+vec_offset, 	// col vector, IS conjugated
                      1, 0, 0);				// 1 block, 0 strides

// (1 + gamma_\mu)  Tr_s [ v2(x+\mu) v1^{\dagger}(x) ]
      sproj_tr[mu+4]( (IFloat *)&tmp_mat2,		// output color matrix
                      (IFloat *)v2_plus_mu,		// row vector, NOT conjugated
                      (IFloat *)v1+vec_offset, 	// col vector, IS conjugated
                      1, 0, 0);				// 1 block, 0 strides

      // exactly what this sounds like
      tmp_mat1 += tmp_mat2 ;
            
      // multiply sum by the link in the \mu direction
      tmp_mat2.DotMEqual(*(gauge+gauge_offset), tmp_mat1) ;
      
      // take tracless antihermitian piece
      // TrLessAntiHermMatrix need to be passed
      // the dagger of the matrix in question
      tmp_mat1.Dagger(tmp_mat2) ;
      tmp_mat2.TrLessAntiHermMatrix(tmp_mat1) ;

      tmp_mat2 *= coeff ;
            
//~~
//~~ fermion version:  	(mom+gauge_offset) += f
//~~ boson version:  	(mom+gauge_offset) -= f
//~~
      *(mom+gauge_offset) -= tmp_mat2 ;

	 Float norm = tmp_mat2.norm();
	 Float tmp = sqrt(norm);
	 L1 += tmp;
	 L2 += norm;
	 Linf = (tmp>Linf ? tmp : Linf);
	 
    }
  }

//------------------------------------------------------------------
// deallocate space for two CANONICAL fermion fields on a site.
//------------------------------------------------------------------
  VRB.Sfree(cname, fname, str_site_v2, site_v2) ;
  sfree(site_v2) ;

  VRB.Sfree(cname, fname, str_site_v1, site_v1) ;
  sfree(site_v1) ;

//------------------------------------------------------------------
// deallocate space for two CANONICAL fermion fields.
//------------------------------------------------------------------
  VRB.Sfree(cname, fname, str_v2, v2) ;
  sfree(v2) ;

  VRB.Sfree(cname, fname, str_v1, v1) ;
  sfree(v1) ;

  glb_sum(&L1);
  glb_sum(&L2);
  glb_max(&Linf);

  L1 /= 4.0*GJP.VolSites();
  L2 /= 4.0*GJP.VolSites();
#endif

  VRB.FuncEnd(cname,fname);
  return ForceArg(L1, sqrt(L2), Linf);
}


//------------------------------------------------------------------
// ForceArg EvolveMomFforce(Matrix *mom, Vector *phi, Vector *eta,
//		      Float mass, Float epsilon, Float dt)
// should never be called by wilsonTm fermions
//------------------------------------------------------------------
ForceArg FwilsonTm::EvolveMomFforce(Matrix *mom, Vector *phi, Vector *eta,
		      Float mass, Float dt) {
  char *fname = "EvolveMomFforce(M*,V*,V*,F,F)";
  ERR.General(cname,fname,"f_wilsonTm: EvolveMomFForce(M*,V*,V*,F,F) not implemented here\n");

  return ForceArg(0.0,0.0,0.0);
}

CPS_END_NAMESPACE
