#include<omp.h>
#include<config.h>
#include <util/qcdio.h>
#include <math.h>
#include<util/lattice.h>
#include<util/gjp.h>
#include<util/verbose.h>
#include<util/dirac_op.h>
#include<util/wilson.h>
#include<util/error.h>
#include<util/time_cps.h>
#include<util/lattice/fgrid.h>
#include<comms/scu.h>
#include<alg/alg_hmd.h>
#include<alg/do_arg.h>

//#define OMP(A) #pragma omp A


USING_NAMESPACE_CPS

DoArg do_arg;
DoArgExt do_ext;
MobiusArg mobius_arg;


//const char *f_wilson_test_filename = CWDPREFIX("f_wilson_test");
//const char *psi_filename = CWDPREFIX("psi");

static int nx,ny,nz,nt,ns;
static CgArg cg_arg;

#if 0
#include <omp.h>
#else
inline void omp_set_num_threads(int num){}
inline int omp_get_num_threads(){return 1;}
inline int omp_get_thread_num(){return 0;}
#endif

//void run_inv(Lattice &lat, DiracOp &dirac, StrOrdType str_ord, char *out_name, int DO_CHECK);
void run_inv(Lattice &lat, StrOrdType str_ord, char *out_name, int DO_CHECK);

int main(int argc,char *argv[]){

    Start(&argc, &argv);
    char *fname = "main()";
//omp_set_num_threads(16);
#pragma omp parallel default(shared)
{
  int tnum = omp_get_num_threads();


#pragma omp for
for(int  i = 0;i<100;i++){
  if (!UniqueID() && (i==0) ) 
  printf("thread %d of %d i=%d\n",omp_get_thread_num(),tnum,i);
}
}

    //----------------------------------------------------------------
    // Initializes all Global Job Parameters
    //----------------------------------------------------------------
    char *out_file=NULL;

    if(argc>1) out_file=argv[1];
  if ( !do_arg.Decode("do_arg.vml","do_arg") )
    {
      ERR.General("",fname,"Decoding of do_arg failed\n");
    }
  do_arg.Encode("do_arg.dat","do_arg");
  if ( !do_ext.Decode("do_ext.vml","do_ext") )
    {
      ERR.General("",fname,"Decoding of do_ext failed\n");
    }
  do_ext.Encode("do_ext.dat","do_ext");
  if ( !mobius_arg.Decode("mobius_arg.vml","mobius_arg") )
    {
      mobius_arg.Encode("mobius_arg.dat","mobius_arg");
      ERR.General("",fname,"Decoding of mobius_arg failed\n");
    }
  mobius_arg.Encode("mobius_arg.dat","mobius_arg");

  GJP.Initialize(do_arg);
  GJP.InitializeExt(do_ext);
  LRG.Initialize();
#if 0
  GJP.SnodeSites(mobius_arg.ls);
  GJP.ZMobius_b (mobius_arg.zmobius_b_coeff.zmobius_b_coeff_val,
                  mobius_arg.ls);
  GJP.ZMobius_c (mobius_arg.zmobius_c_coeff.zmobius_c_coeff_val,
                   mobius_arg.ls);
#endif

//HACK!! Needs to be fixed, as cg_arg has dynamical allocation now.
    cg_arg = mobius_arg.cg; 
    LRGState rng_state;
{
    GnoneFmobius lat;
    rng_state.GetStates();
	run_inv(lat,DWF_4D_EOPREC_EE,"Fmobius",1);
}

#ifdef USE_GRID
{
//   LRG.Initialize();
   FgridParams params;
   params.mobius_scale=GJP.GetMobius();
   params.epsilon=0.;
   GnoneFgridMobius lat(params);
    rng_state.SetStates();
   run_inv(lat,CANONICAL,"FgridMobius",1);
}
#endif

	
	

    End();
    return 0; 
}

//void run_inv(Lattice &lat, DiracOp &dirac, StrOrdType str_ord, char *out_name, int DO_CHECK){
void run_inv(Lattice &lat,  StrOrdType str_ord, char *out_name, int DO_CHECK){
    FILE *fp;
    double dtime;
	int DO_IO =1;
	if(out_name==NULL) DO_IO=0;
	if(DO_IO) fp = Fopen(ADD_ID,out_name,"w");
    	else fp = stdout;
    VRB.Result("","","DO_CHECK=%d DO_IO=%d\n",DO_CHECK,DO_IO);

    Vector *result = (Vector*)smalloc("","","result",GJP.VolNodeSites()*lat.FsiteSize()*sizeof(IFloat));
    Vector *X_out = (Vector*)smalloc("","","X_out",GJP.VolNodeSites()*lat.FsiteSize()*sizeof(IFloat));
    Vector *X_out2 = (Vector*)smalloc("","","X_out2",GJP.VolNodeSites()*lat.FsiteSize()*sizeof(IFloat));
    Vector *tmp = (Vector*)smalloc("","","tmp",GJP.VolNodeSites()*lat.FsiteSize()*sizeof(IFloat));

	int s_size = 1;
	if (lat.F5D()) s_size = GJP.SnodeSites();

    int s[5];
    Vector *X_in =
	(Vector*)smalloc(GJP.VolNodeSites()*lat.FsiteSize()*sizeof(IFloat));
    if(!X_in) ERR.Pointer("","","X_in");
	memset(X_in,0,GJP.VolNodeSites()*lat.FsiteSize()*sizeof(IFloat));
#if 1
	lat.RandGaussVector(X_in,1.0);
#else

//    lat.RandGaussVector(X_in,1.0);
    Matrix *gf = lat.GaugeField();
    IFloat *gf_p = (IFloat *)lat.GaugeField();
    int fsize = lat.FsiteSize()/s_size;
    VRB.Result("","main()","fsize=%d",fsize);

    for(s[4]=0; s[4]<s_size; s[4]++)
    for(s[3]=0; s[3]<GJP.NodeSites(3); s[3]++)
	for(s[2]=0; s[2]<GJP.NodeSites(2); s[2]++)
	    for(s[1]=0; s[1]<GJP.NodeSites(1); s[1]++)
		for(s[0]=0; s[0]<GJP.NodeSites(0); s[0]++) {

		    int n = lat.FsiteOffset(s)+s[4]*GJP.VolNodeSites();

		int crd=1.;
		  if(CoorX()==0 && CoorY()==0 && CoorZ()==0 && CoorT()==0) crd=1.0; else crd = 0.0;
		  if(s[0]!=0 ) crd = 0.;
		  if(s[1]!=0 ) crd = 0.;
		  if(s[2]!=0 ) crd = 0.;
		  if(s[3]!=0 ) crd = 0.;
			if(s[4]!=0 ) crd = 0.;
					
			IFloat *X_f = (IFloat *)(X_in)+(n*fsize);
		    for(int v=0; v<fsize ; v+=1){ 
			if (v==0)  *(X_f+v) = crd;
			else
			*(X_f+v) = 0;
		    }
		}
#endif

    Vector *out;
    Float true_res;

	for(int k = 0; k< 1; k++){
    	double maxdiff=0.;
//		printf("k=%d ",k);
		if (k ==0)
			out = result;
		else
			out = X_out;
		memset((char *)out, 0,GJP.VolNodeSites()*lat.FsiteSize()*sizeof(IFloat));
		lat.Fconvert(result,str_ord,CANONICAL);
		lat.Fconvert(X_in,str_ord,CANONICAL);
		int offset = GJP.VolNodeSites()*lat.FsiteSize()/ (2*6);
#if 1
		dtime = -dclock();
// With dminus
   		int iter = lat.FmatInv(result,X_in,&cg_arg,&true_res,CNV_FRM_NO,PRESERVE_YES);
// Without dminus
//   		int iter = lat.FmatInvTest(result,X_in,&cg_arg,&true_res,CNV_FRM_NO,PRESERVE_YES);
		dtime +=dclock();
#else
		lat.Fdslash(result,X_in,&cg_arg,CNV_FRM_NO,0);
//		dirac.Dslash(result,X_in+offset,CHKB_EVEN,DAG_NO);
//		dirac.Dslash(result+offset,X_in,CHKB_ODD,DAG_NO);
#endif

//if (DO_CHECK){
if (1){
		if (k == 0){
			memset((char *)X_out2, 0,GJP.VolNodeSites()*lat.FsiteSize()*sizeof(IFloat));
			lat.Fdslash(X_out2,result,&cg_arg,CNV_FRM_NO,0);
//			dirac.Dslash(X_out2,result+offset,CHKB_EVEN,DAG_NO);
//			dirac.Dslash(X_out2+offset,result,CHKB_ODD,DAG_NO);
		}
}
		lat.Fconvert(result,CANONICAL,str_ord);
		lat.Fconvert(X_in,CANONICAL,str_ord);
		lat.Fconvert(X_out2,CANONICAL,str_ord);
//	if (lat.F5D())
	if (lat.Fclass()== F_CLASS_DWF)
		X_out2->FTimesV1PlusV2(-0.5/(5.0-GJP.DwfHeight()),X_out2,out,GJP.VolNodeSites()*lat.FsiteSize());
	else if (lat.Fclass()== F_CLASS_WILSON)
		X_out2->FTimesV1PlusV2(-0.5/(cg_arg.mass+4.0),X_out2,out,GJP.VolNodeSites()*lat.FsiteSize()); 
    
    Float dummy;
    Float dt = 2;
if (DO_CHECK){
    for(s[4]=0; s[4]<s_size; s[4]++) 
    for(s[3]=0; s[3]<GJP.NodeSites(3); s[3]++) 
	for(s[2]=0; s[2]<GJP.NodeSites(2); s[2]++)
	    for(s[1]=0; s[1]<GJP.NodeSites(1); s[1]++)
		for(s[0]=0; s[0]<GJP.NodeSites(0); s[0]++) {

//		    int n = lat.FsiteOffset(s)*lat.SpinComponents()*GJP.SnodeSites();
		    int n = (lat.FsiteOffset(s)+GJP.VolNodeSites()*s[4]) *lat.SpinComponents();
			for(int i=0; i<(3*lat.SpinComponents()); i++){
	double re_re =	*((IFloat*)&result[n]+i*2);
	double in_re =	*((IFloat*)&X_in[n]+i*2);
	double re_im =	*((IFloat*)&result[n]+i*2+1);
	double in_im =	*((IFloat*)&X_in[n]+i*2+1);
if((re_re*re_re+re_im*re_im + in_re*in_re+in_im*in_im)>1e-8)
if(DO_IO){
		    if ( k==0 )
				Fprintf(ADD_ID,fp," %d %d %d %d %d (%d) ", 
			CoorX()*GJP.NodeSites(0)+s[0], 
			CoorY()*GJP.NodeSites(1)+s[1], 
			CoorZ()*GJP.NodeSites(2)+s[2], 
			CoorT()*GJP.NodeSites(3)+s[3], 
			CoorS()*GJP.NodeSites(4)+s[4], 
			i, n);
		    if ( k==0 )
				Fprintf(ADD_ID, fp," ( %0.7e %0.7e ) (%0.7e %0.7e)",
				*((IFloat*)&result[n]+i*2), *((IFloat*)&result[n]+i*2+1),
				*((IFloat*)&X_in[n]+i*2), *((IFloat*)&X_in[n]+i*2+1));
#if 1
				Fprintf(ADD_ID, fp," ( %0.2e %0.2e )\n",
	#if 0
		*((IFloat*)&X_out2[n]+i*2)-*((IFloat*)&X_in[n]+i*2), 
	*((IFloat*)&X_out2[n]+i*2+1)-*((IFloat*)&X_in[n]+i* 2+1));
	#else
		*((IFloat*)&X_out2[n]+i*2),
	*((IFloat*)&X_out2[n]+i*2+1));
	#endif
#else
				Fprintf(ADD_ID, fp,"\n");
#endif
} //DO_IO
	double diff =	*((IFloat*)&X_out2[n]+i*2)-*((IFloat*)&X_in[n]+i*2);
        if (fabs(diff)>maxdiff) maxdiff = fabs(diff);
 	diff = *((IFloat*)&X_out2[n]+i*2+1)-*((IFloat*)&X_in[n]+i* 2+1);
        if (fabs(diff)>maxdiff) maxdiff = fabs(diff);
			}
		}
    VRB.Result("","run_inv()","Max diff between X_in and M*X_out = %0.2e\n", maxdiff);
}
}
	if (DO_IO) Fclose(fp);
    
    sfree(X_in);
    sfree(result);
    sfree(X_out);
    sfree(X_out2);
}
