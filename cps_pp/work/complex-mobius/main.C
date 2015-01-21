#include <conf.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <alg/alg_plaq.h>
#include <alg/alg_lanczos.h>
#include <alg/alg_meas.h>
#include <alg/alg_fix_gauge.h>
#include <util/lattice.h>
#include <util/gjp.h>
#include <util/verbose.h>
#include <util/error.h>
#include <util/command_line.h>
#include <util/ReadLatticePar.h>
#include <util/ReadU1LatticePar.h>
#include <util/qio_readLattice.h>
#include <util/qcdio.h>
#include <util/site.h>
#include <util/eigen_container.h>
#include <util/random.h>
#include <util/zmobius.h>

using namespace std;
USING_NAMESPACE_CPS

void print_vec(char* str, Float* vec)
{

  int node_coor[4]= {
    GJP.NodeCoor(0),
    GJP.NodeCoor(1),
    GJP.NodeCoor(2),
    GJP.NodeCoor(3)} ;
  
  char fname[256];
  sprintf(fname,"%s.%d.%d.%d.%d"
	  ,str
	      ,node_coor[0]
	      ,node_coor[1]
	      ,node_coor[2]
	      ,node_coor[3]);
  FILE* fp2=fopen(fname,"w");
      

  Site site;

  while (site.LoopsOverNode()) {
    // site offset 
    const int s_off(site.Index());
	
    int Ls(GJP.Snodes());
    for(int is=0;is<Ls;++is) {
      fprintf(fp2,"%d %d %d %d  %d ",
	      site.physX(),
	      site.physY(),
	      site.physZ(),
	      site.physT(),
	      is);
      for(int i=0;i<24;++i)
	fprintf(fp2,"%e ",  vec[24*s_off+i]);
      fprintf(fp2, "\n");
    }
  }
  fclose(fp2);
}

DoArg do_arg;
DoArgExt doext_arg;

MobiusArg mobius_arg;
MobiusArg mobius_arg2;


NoArg no_arg;
CommonArg common_arg;
LanczosArg lanczos_arg;


MdwfArg real_mdwf_arg;


void movefloattoFloat(Float* out, float* in, int f_size);

// needed to declare globally
std::vector<EigenCache*> cps::EigenCacheList(0);


//Search contents that match to arguments, return 0 if not found
EigenCache* cps::EigenCacheListSearch( char* fname_root_bc, int neig )
{
  EigenCache* ecache=0;

  for(int i=0; i< EigenCacheList.size(); ++i  )
    if( EigenCacheList[i]-> is_cached( fname_root_bc, neig ) )
      ecache = EigenCacheList[i];

  return ecache;
}


void do_CG(Lattice* lattice, Vector* out, int flag_src_kappa_b, int flag_madwf=0)
{

  Vector* in;
  //Vector* out;
  CgArg cg;
  int ls = GJP.SnodeSites();

  printf("doCG(..):  Ls=%d\n",ls);
  //out = (Vector*)smalloc(GJP.VolNodeSites()*ls*4*sizeof(Vector));
  in  = (Vector*)smalloc(GJP.VolNodeSites()*ls*4*sizeof(Vector));
  Vector* in_4d  = (Vector*)smalloc(GJP.VolNodeSites()*4*sizeof(Vector));

  // zero the source, solution
  Site site;
  while (site.LoopsOverNode()) {
    int coord_idx = site.Index();
    for(int id=0;id<12;++id){
      int idx = id + 12 * coord_idx;
      Float val =
	0.1* site.physX()+
	0.2* site.physY()+
	0.3* site.physZ()+	    
	0.4* site.physT();

	*((Complex*)in_4d  + idx) =Complex(val,0.0);
    }
  }

  
  lattice-> Ffour2five(in, in_4d, 0, ls-1);

  for(int i=0;i<GJP.VolNodeSites()*ls*24;++i)
    *((IFloat*)out+i) = 0.0;


  // for f_mobius, we multiply 2 kappa to make the normalization same
  // with f_zmobius and f_mdwf
  if(flag_src_kappa_b){
    
    int local_ls = GJP.SnodeSites();
    const int s_node_coor = GJP.SnodeCoor();
    const int ls_stride = 24 * GJP.VolNodeSites();

    for(int s=0; s<local_ls;++s){
      int glb_s = s + local_ls*s_node_coor;
      const Complex kappa_b =
	1.0 / ( 2 * (GJP.ZMobius_b()[glb_s]
		     *(4 - GJP.DwfHeight()) + GJP.DwfA5Inv()) );
      int idx = s*ls_stride/2;// "/2" is for complex
      vecTimesEquComplex((Complex*)in+idx, 2.0*kappa_b, ls_stride);
    }

  }

  //point source at origin and spin=color=0
  //if(!UniqueID()) *((Float*)in) = 1.0;

  Float res;
  res = 0.0;
  PreserveType pres =  PRESERVE_YES;
  CnvFrmType cnv =  CNV_FRM_YES;
  int nrestart =  3;
  Float rsd_vec[4];
  rsd_vec[0] = 1e-2;
  rsd_vec[1] = 2e-5;
  rsd_vec[2] = 2e-5;
  rsd_vec[3] = 2e-5;

  cg = mobius_arg.cg;

  //int iter = lattice.FmatInv(out, in, &mdwf_arg, &mdwf_arg2, &res, cnv, pres, nrestart, mdwf_arg.rsd_vec);

  int iter;
  if(flag_madwf)
    iter = lattice->FmatInv(out, in,
					       &mobius_arg,   &mobius_arg2,
			  &res, cnv, pres);
  else 
    iter = lattice->FmatInv(out, in, &cg, &res, cnv, pres);

  print_vec("mob.sol", (Float*)out);

  //  sfree(out);
  sfree(in);

}

int main(int argc,char *argv[])
{

  char *cname = argv[0] ;
  char *fname = "main()" ;
  char* filename;
  filename = (char*) smalloc( 128*sizeof(char) );
  
  Start(&argc, &argv);

  //if ( argc!=3) { 
  //if(!UniqueID())printf("(exe) do_arg doext_arg mdwf_arg mdwf_arg eig_arg work-directory\n");
  //exit(-1);
  //}
  
  //int flag_test = atoi(argv[1]);
  //chdir(argv[1]);

  real_mdwf_arg.Encode("real_mdwf_arg.dat", "real_mdwf_arg");
  doext_arg.Encode("doext_arg.dat","doext_arg");

  
  if ( !do_arg.Decode("do_arg.vml","do_arg") ) 
    { 
      ERR.General(fname,fname,"Decoding of do_arg failed\n");
    }
  if ( !doext_arg.Decode("doext_arg.vml","doext_arg") ) 
    { 
      ERR.General(fname,fname,"Decoding of doext_arg failed\n");
    }
#if 1
  if ( !mobius_arg.Decode("mobius_arg.vml","mobius_arg") ) 
    { 
      mobius_arg.Encode("mobius_arg.dat","mobius_arg");
      ERR.General(fname,fname,"Decoding of mobius_arg failed\n");  
    }
  if ( !mobius_arg2.Decode("mobius_arg2.vml","mobius_arg2") ) 
    { 
      ERR.General(fname,fname,"Decoding of mobius_arg2 failed\n");  
    }
#endif
  if ( !lanczos_arg.Decode("lanczos_arg.vml","lanczos_arg") ) 
    { 
      lanczos_arg.Encode("lanczos_arg.dat","lanczos_arg");
      if(!UniqueID())printf("Decoding of lanczos_arg failed\n"); exit(-1);
    }

  // make a record of what was run
  do_arg.Encode("do_arg.dat","do_arg");
  doext_arg.Encode("doext_arg.dat","doext_arg");
  lanczos_arg.Encode("lanczos_arg.dat","lanczos_arg");
  mobius_arg.Encode("mobius_arg.dat","mobius_arg");
  mobius_arg2.Encode("mobius_arg2.dat","mobius_arg2");
  real_mdwf_arg.Decode("real_mdwf_arg.vml","real_mdwf_arg");


  int long_ls=24;
  int short_ls=mobius_arg2.zmobius_b_coeff.zmobius_b_coeff_len/2;

  

  GJP.Initialize(do_arg);
  GJP.InitializeExt(doext_arg);
  VRB.Level(do_arg.verbose_level);

  GJP.SnodeSites(mobius_arg.ls);
  GJP.ZMobius_b (mobius_arg.zmobius_b_coeff.zmobius_b_coeff_val, mobius_arg.ls);
  GJP.ZMobius_c (mobius_arg.zmobius_c_coeff.zmobius_c_coeff_val, mobius_arg.ls);

  
  //Lattice &lattice = LatticeFactory::Create(F_CLASS_DWF, G_CLASS_WILSON);
  
  //Lattice &lattice = LatticeFactory::Create(F_CLASS_MOBIUS, G_CLASS_NONE);
  //Lattice &lattice = LatticeFactory::Create(F_CLASS_DWF, G_CLASS_NONE);


  // mult su3 links by qed phase
  //lattice.mult_su3_links_by_u1(1.0);  
  //qio_readLattice rl(do_arg.start_conf_filename, lattice, argc, argv);
 

  //int ls = GJP.SnodeSites();
  
  Vector* out_mob = (Vector*)smalloc(GJP.VolNodeSites()*long_ls*4*sizeof(Vector));
  Vector* out_mob_4d = (Vector*)smalloc(GJP.VolNodeSites()*4*sizeof(Vector));

  GJP.SnodeSites( long_ls );
    
  int comp_flag=atoi(argv[1]);
  switch( comp_flag ){
  case 1 :
    {
      GJP.SetMdwfArg( &real_mdwf_arg );
      //Lattice &lattice = LatticeFactory::Create(F_CLASS_MDWF, G_CLASS_WILSON);
      GnoneFmdwf lattice;
      do_CG(&lattice,out_mob,0);
      //LatticeFactory::Destroy();
      lattice.Ffive2four(out_mob_4d, out_mob, GJP.SnodeSites()-1, 0);
    }
    break;
  case 2:
    {
      //Lattice &lattice = LatticeFactory::Create(F_CLASS_MOBIUS, G_CLASS_NONE);
      GnoneFmobius lattice;
      do_CG(&lattice,out_mob,1);
      //LatticeFactory::Destroy();
      lattice.Ffive2four(out_mob_4d, out_mob, GJP.SnodeSites()-1, 0);
    }
    break;
  }

  
  //  LatticeFactory::Destroy();
  //  lattice = LatticeFactory::Create(F_CLASS_MOBIUS, G_CLASS_NONE);



  Vector* out_zmob_4d = (Vector*)smalloc(GJP.VolNodeSites()*4*sizeof(Vector));
  Vector* out_zmob = (Vector*)smalloc(GJP.VolNodeSites()*4*short_ls*sizeof(Vector));
  {
    //Lattice &lattice = LatticeFactory::Create(F_CLASS_ZMOBIUS, G_CLASS_NONE);
    GJP.SnodeSites( short_ls );
    GnoneFzmobius lattice;
    do_CG(&lattice,out_zmob, 0, 0);
    lattice.Ffive2four(out_zmob_4d, out_zmob, GJP.SnodeSites()-1, 0);
    //LatticeFactory::Destroy();
  }


  if(comp_flag){

  Float max_diff = -100;
  Float max_rel_diff = -100;

  //  for(int s=0; s< ls;++s){
  int s=0;{
    Site site;
    int vol=GJP.VolNodeSites();
    while (site.LoopsOverNode()) {
      int i= site.Index()+s*vol;
      for(int ic=0;ic<12;++ic){
	int idx=ic+12*i;
	Complex *z1 = (Complex*)out_mob_4d+idx;
	Complex *z2 = (Complex*)out_zmob_4d+idx;

	
	//printf("%d %e %e %e %e\n", i, z1->real(), z1->imag(), z2->real(), z2->imag());
	Complex zdiff=*z1-*z2;

	if(max_diff < norm(zdiff)){
	  max_diff=norm(zdiff);
	}
	
	if(max_rel_diff < norm(2*zdiff/(*z1+*z2))){
	  max_rel_diff=norm(2*zdiff/(*z1+*z2));
	}
      }
      
    }
  }

  glb_max(&max_diff);
  glb_max(&max_rel_diff);  
  if(!UniqueID()){
    printf("max_difference_sol %e %e\n",max_diff, max_rel_diff);
  }

  {
    int f_size = GJP.VolNodeSites()*24;
    
    Float norm_zmob = out_zmob_4d->NormSqGlbSum(f_size);
    Float norm_mob = out_mob_4d->NormSqGlbSum(f_size);
    
    IFloat* fp = (IFloat*) out_mob_4d;
    IFloat* fpz = (IFloat*) out_zmob_4d;
    for(int i=0;i<f_size;++i) fpz[i]-=fp[i];
    Float norm_diff = out_zmob_4d->NormSqGlbSum(f_size);
    
    if(!UniqueID())
      printf("difference L2 norm %e %e %e rel = %e\n",
	     norm_mob, norm_zmob, norm_diff,
	     norm_diff / norm_mob);
    }
  }


  
  //  LatticeFactory::Destroy();

  sfree(filename);


  //  LatticeFactory::Destroy();
  
  //EigenCacheListCleanup();

  End();
  
  return 0;
} 


// Cleanup list EigenCache, it also destroies contents pointed by the elements.
void cps::EigenCacheListCleanup( )
{
  for(size_t i=0;i< EigenCacheList. size(); ++i){
    EigenCacheList[i]-> dealloc();
  }
  EigenCacheList.clear() ;
}

void movefloattoFloat(Float* out, float* in, int f_size){

  float flt;
  for(int i=0;i<f_size;i++){
    flt = in[i];
    out[i] = (Float)flt;
  }
};

