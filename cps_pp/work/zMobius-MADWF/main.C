#include <conf.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <alg/alg_plaq.h>
#include <alg/alg_lanczos.h>
#include <alg/alg_meas.h>
#include <alg/alg_fix_gauge.h>
#include <alg/qpropw.h>
#include <alg/meson.h>
#include <alg/nuc2pt.h>
#include <util/lattice.h>
#include <util/gjp.h>
#include <util/verbose.h>
#include <util/error.h>
#include <util/command_line.h>
#include <util/ReadLatticePar.h>
#include <util/ReadU1LatticePar.h>
#include <util/qio_readLattice.h>
#include <util/qcdio.h>
#include <util/eigen_container.h>



USING_NAMESPACE_CPS
using namespace std;


DoArg do_arg;
DoArgExt doext_arg;

// It is very imporatnt that mobius_arg is global and exits for ever
// as in GJP, we use zmobius_c_coeff and zmobius_b_coeff is pointing the contetnts of mobius_arg
// We may want to fix this later,  I am not fixing this due to lack of time
MobiusArg mobius_arg;
MobiusArg mobius_arg2;
MdwfArg mdwf_arg;

NoArg no_arg;
CommonArg common_arg;
LanczosArg lanczos_arg;
QPropWArg qp_arg;


void movefloattoFloat(Float* out, float* in, int f_size);


// needed to declare globally
std::vector<EigenCache*> cps::EigenCacheList(0);

//Search contents that match to arguments, return 0 if not found
EigenCache* cps::EigenCacheListSearch( char* fname_root_bc, int neig )
{
  EigenCache* ecache=0;

  for(int i=0; i< EigenCacheList.size(); ++i  ){
    if( EigenCacheList[i]-> is_cached( fname_root_bc, neig ) )
      ecache = EigenCacheList[i];
  }

  return ecache;
}


EigenCache *ecache;
char* cname;

void CalMesons(QPropW &q, char *out_dir, int traj, char *src_str);
void CalNucleons(QPropW &q, char *out_dir, int traj, char *src_str);
 
  char evecname_bc[1024];
void comp_read_eigenvectors(Lattice& lattice)
{
  char* fname= "comp_read_eigenvectors()";
  //GnoneFzmobius lattice;
  /*
   *  Compute eigen vectors and values
   *
   */
  Float etime = time_elapse();
  char cache_name[1024];
  snprintf(cache_name,1024,"cache_0_mass%g", lanczos_arg.mass);
  ecache = new EigenCache(cache_name);
  char evecname[1024];
  snprintf(evecname, 1024,"%s/eig4dee.ls%d.pc%d.mass%g",
	   lanczos_arg.file, mobius_arg2.ls, GJP.ZMobius_PC_Type(),lanczos_arg.mass );
  lanczos_arg.file = evecname;

  
  snprintf(evecname_bc,1024, "%s.bc%d%d%d%d", lanczos_arg.file, GJP.Bc(0),GJP.Bc(1),GJP.Bc(2),GJP.Bc(3));

  
  AlgLanczos  eig(lattice, &common_arg, &lanczos_arg, ecache);
  int Ncb = eig.NumChkb(lanczos_arg.RitzMat_lanczos);
  int ls_save = GJP.SnodeSites();
  GJP.SnodeSites(mobius_arg2.ls);
  GJP.ZMobius_b (mobius_arg2.zmobius_b_coeff.zmobius_b_coeff_val, mobius_arg2.ls);
  GJP.ZMobius_c (mobius_arg2.zmobius_c_coeff.zmobius_c_coeff_val, mobius_arg2.ls  );
  int fsize = GJP.VolNodeSites() * lattice.FsiteSize() * Ncb / 2 / 2; //last 2 for single prec.;
  EigenCacheList. push_back( ecache );
  int neig;
 

#if 1
  if( lanczos_arg.nk_lanczos_vectors > 0) {
            
    int init_flag = 0; // 0 : wall start, 1 :  compression, 2 : compress & decompress, 3: refinement using Ritz
    int ncompress = 10; // number of evecs in the compressed linear combination
      
    char* comp_file;
      
    neig  = lanczos_arg.nk_lanczos_vectors+lanczos_arg.np_lanczos_vectors;
    ecache->alloc( evecname_bc, neig, fsize );
    eig.run( init_flag, ncompress, comp_file );
    neig  = mobius_arg2.cg.neig;
    ecache->set_neig(neig);
  } else {
    neig  = mobius_arg2.cg.neig;
    if(neig>0){
      ecache->alloc( evecname_bc, neig, fsize );
      {//read in only
	const int n_fields =  GJP.SnodeSites();
	const int f_size_per_site = lattice.FsiteSize() / n_fields / 2 ;
	EigenContainer eigcon( lattice, evecname_bc, neig, f_size_per_site/2, n_fields, ecache );
	// factor of 2 for single-prec.
	// have to do this if stride != 1 FIX!
	for(int iev=0; iev < neig ; iev++){
	  Vector* evec= eigcon.nev_load( iev );
	  ecache->set_index(iev);
	}
      }
    }
  }

  etime = time_elapse();
  if(!UniqueID())printf("Time for Lanczos %g\n",etime);
#endif



  // Read eigenvectors for small Ls mobius in mobius_arg2
  //-------------------------------------------------------
#if 0
  if(mobius_arg2.cg.neig>0)
  {
    if( GJP.Snodes() != 1) ERR.NotImplemented(cname,fname,"currently only doing I/O for local Ls\n");

    int neig  = mobius_arg2.cg.neig;

    int Ncb = eig.NumChkb(lanczos_arg.RitzMat_lanczos);
    int ls_save = GJP.SnodeSites();

    int fsize = GJP.VolNodeSites() * lattice.FsiteSize() * Ncb / 2 / 2; //last 2 for single prec.;
    const int n_fields =  GJP.SnodeSites();
    const int f_size_per_site = lattice.FsiteSize() / n_fields / 2 ;
    // the two is for checkerboard?
    EigenContainer eigcon( lattice, evecname_bc, neig, f_size_per_site, n_fields, ecache );
    Float* eval = eigcon.load_eval();
    Float* evecFloat = (Float*)smalloc(2 * fsize * sizeof(Float));

    for(int iev=0; iev< neig; iev++){

      Vector* evec= eigcon.nev_load( iev );
      movefloattoFloat(evecFloat,(float*)evec,fsize*2);

      if(iev % 1 == 0){  // Let's check all

        Float res_chk, eval_chk;
        Float mass = lanczos_arg.mass;

        eigcon. nev_check( (Vector*)evecFloat, mass, &res_chk, &eval_chk );

        Float ev = *(eval+iev);
        if( fabs(res_chk) > 1e-5 )
          VRB.Warn("main","main","nev_check index %d eigval %g mass %g res %e > 1e-5\n",
                           iev, ev, mass, res_chk);

        if( fabs(eval_chk- ev) > 1e-5 )
          VRB.Warn("main","main","nev_check index %d mass %g eval_chk %e eval %e, abs_err %e > 1e-5\n",
                   iev, mass, eval_chk, ev, fabs(eval_chk-ev) );

      }
    }
    sfree(evecFloat);
  }
#endif 

}




int main(int argc,char *argv[])
{

  cname = argv[0] ;
  char *fname = "main()" ;
  
  Start(&argc, &argv);

  //  if ( argc!=9) { 
  //    if(!UniqueID())printf("(exe) do_arg doext_arg mobius_arg-l mobius_arg-s qprop_arg eig_arg mdw_arg do_mdw d_mob do_zmob_lg do_zmob_sl work-directory\n");
  //    exit(-1);
  //  }

  if ( argc!=6) { 
    if(!UniqueID())printf("(exe) working_dir do_mdwf do_mob do_zmob_lg do_zmob_sm , but argc=%d\n", argc);
    exit(-1);
  }

  chdir(argv[1]);
  int do_mdwf=atoi(argv[2]);
  int do_mob=atoi(argv[3]);  
  int do_zmob_lg=atoi(argv[4]);
  int do_zmob_sm=atoi(argv[5]);
  
  //  qp_arg.Encode("qpropw_arg.dat","qpropw_arg");
  
  if ( !do_arg.Decode("do_arg.vml","do_arg") ) 
    { 
      ERR.General(fname,fname,"Decoding of do_arg failed\n");
    }
  if ( !doext_arg.Decode("doext_arg.vml","doext_arg") ) 
    { 
      ERR.General(fname,fname,"Decoding of doext_arg failed\n");
    }
  if ( !mobius_arg.Decode("mobius_arg.vml","mobius_arg") ) 
    { 
      mobius_arg.Encode("mobius_arg.dat","mobius_arg");
      ERR.General(fname,fname,"Decoding of mobius_arg failed\n");  
    }
  if ( !mobius_arg2.Decode("mobius_arg2.vml","mobius_arg2") ) 
    { 
      ERR.General(fname,fname,"Decoding of mobius_arg2 failed\n");  
    }
  if ( !qp_arg.Decode("qpropw_arg.vml","qpropw_arg") ) 
    { 
      ERR.General(fname,fname,"Decoding of qpropw_arg failed\n");  
    }
  if ( !lanczos_arg.Decode("lanczos_arg.vml","lanczos_arg") ) 
    { 
      lanczos_arg.Encode("lanczos_arg.dat","lanczos_arg");
      if(!UniqueID())printf("Decoding of lanczos_arg failed\n"); exit(-1);
    }
  if ( !mdwf_arg.Decode("mdwf_arg.vml","mdwf_arg") ) 
    { 
      lanczos_arg.Encode("mdwf_arg.dat","mdwf_arg");
      if(!UniqueID())printf("Decoding of mdwf_arg failed\n"); exit(-1);
    }

  // make a record of what was run
  do_arg.Encode("do_arg.dat","do_arg");
  doext_arg.Encode("doext_argp.dat","doext_arg");
  qp_arg.Encode("qpropw_arg.dat","qpropw_arg");
  lanczos_arg.Encode("lanczos_arg.dat","lanczos_arg");
  mobius_arg.Encode("mboius_arg.dat","mobius_arg");
  mobius_arg2.Encode("mobius_arg2.dat","mobius_arg2");
  mdwf_arg.Encode("mdwf_arg.dat","mdwf_arg");
  
  GJP.Initialize(do_arg);
  GJP.InitializeExt(doext_arg);
  VRB.Level(do_arg.verbose_level);

  
   GJP.ZMobius_PC_Type(ZMOB_PC_SYM2 ); 
  //int zmob_pc_type = atoi(argv[7]);
   //GJP.ZMobius_PC_Type((ZMobiusPCType)zmob_pc_type );

  


  // Solve  Large Ls with naitive (slow) way
  //--------------------------------------------
  if(do_mob)  {  
    Lattice& lattice=
      LatticeFactory::Create(F_CLASS_MOBIUS, G_CLASS_NONE);

    
    GJP.SnodeSites(mobius_arg.ls);
    GJP.Mobius_b (mobius_arg.mobius_b_coeff);
    GJP.Mobius_c (mobius_arg.mobius_c_coeff);

    CommonArg carg;
    QPropWPointSrc qp(lattice, &qp_arg, &carg);
    //qp.Run();

    system("mkdir -p mob");
    CalMesons(qp, "mob", 0, "point");
    CalNucleons(qp, "mob", 0, "point");
    LatticeFactory::Destroy(); 
  }

  if(do_mdwf){
    GJP.SetMdwfArg( &mdwf_arg );
    GnoneFmdwf lattice;

    CommonArg carg;
    QPropWPointSrc qp(lattice, &qp_arg, &carg);
    //qp.Run();

    system("mkdir -p mdwf");
    CalMesons(qp, "mdwf", 0, "point");
    CalNucleons(qp, "mdwf", 0, "point");
    //LatticeFactory::Destroy(); 
  }

  int flag_comp_read_eigv=0;


  
  // Solve  Large Ls with Zmobius with MADWF
  //--------------------------------------------
if(do_zmob_lg)  {  
    //    Lattice& lattice=
    //      LatticeFactory::Create(F_CLASS_ZMOBIUS, G_CLASS_NONE);
    GnoneFzmobius lattice;
    if(!flag_comp_read_eigv)
      {
	comp_read_eigenvectors(lattice)  ;
	flag_comp_read_eigv=1;
      }
    
    GJP.SnodeSites(mobius_arg.ls);
    GJP.ZMobius_b (mobius_arg.zmobius_b_coeff.zmobius_b_coeff_val,
		   mobius_arg.ls);
    GJP.ZMobius_c (mobius_arg.zmobius_c_coeff.zmobius_c_coeff_val,
		   mobius_arg.ls);
    

       *(MobiusArg**)&(qp_arg.mob_arg_l) = &mobius_arg;
       *(MobiusArg**)&(qp_arg.mob_arg_s) = &mobius_arg2;
    
      
   CommonArg carg;
    QPropWPointSrc qp(lattice, &qp_arg, &carg);
    //qp.Run();

    system("mkdir -p zmob_lg");
    CalMesons(qp, "zmob_lg", 0, "point");
    CalNucleons(qp, "zmob_lg", 0, "point");

    qp_arg.mob_arg_l = 0;
    qp_arg.mob_arg_s = 0;

    //LatticeFactory::Destroy();
  }

   // Solve  Small Ls with Zmobius
  //--------------------------------------------
if(do_zmob_sm)  {  
    //    Lattice& lattice=
    //      LatticeFactory::Create(F_CLASS_ZMOBIUS, G_CLASS_NONE);
    GnoneFzmobius lattice;
    
    if(!flag_comp_read_eigv)
	comp_read_eigenvectors(lattice)  ;

	GJP.SnodeSites(mobius_arg2.ls);
    GJP.ZMobius_b (mobius_arg2.zmobius_b_coeff.zmobius_b_coeff_val,
		   mobius_arg2.ls);
    GJP.ZMobius_c (mobius_arg2.zmobius_c_coeff.zmobius_c_coeff_val,
		   mobius_arg2.ls);
    

    //    *(MobiusArg**)&(qp_arg.mob_arg_l) = &mobius_arg;
    //    *(MobiusArg**)&(qp_arg.mob_arg_s) = &mobius_arg2;
    CgArg cg_save = qp_arg.cg;
    qp_arg.cg = mobius_arg2.cg;
    
    CommonArg carg;

    QPropWPointSrc qp(lattice, &qp_arg, &carg);
    //qp.Run();
    // 
    system("mkdir -p zmob_sm");
    CalMesons(qp, "zmob_sm", 0, "point");
    CalNucleons(qp, "zmob_sm", 0, "point");
    
    //    qp_arg.mob_arg_l = 0;
    ///    qp_arg.mob_arg_s = 0;

    //LatticeFactory::Destroy();
    qp_arg.cg=cg_save;
 }



  
  EigenCacheListCleanup();

  //End();
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

 
 void CalNucleons(QPropW &q, char *out_dir, int traj, char *src_str){
  Nuc2pt nuc(NUC_G5C, POINT);
  nuc.calcNucleon( q );

  char file[256];
  sprintf(file, "%s/nucleon_%s.dat.%d", out_dir,src_str,traj);
  FILE *fp = Fopen(&file[0],"w");

  {
    Nuc2pt nuc(NUC_G5C, POINT);
    nuc.calcNucleon( q );
    nuc.Print(fp);
  }

  Fclose(fp);
  
  
}

void CalMesons(QPropW &q, char *out_dir, int traj, char *src_str){
 
  char file[256];
  sprintf(file, "%s/meson_%s.dat.%d", out_dir,src_str,traj);
  FILE *fp = Fopen(&file[0],"w");
 
  int mu, nu;
  //vector and scalar
  for ( mu = 0; mu < 4; mu++ ){
    char str[256];
    sprintf(str, "GAM_%d", mu+1);
    Meson mes(mu, str);
    mes.Zero();
    mes.setMass(q.Mass(),q.Mass());
    mes.calcMeson(q,q);
    if(!UniqueID())
      mes.Print(fp);
  }

  //pseudoscalar
  {
    mu = -5;
    char str[256];
    sprintf(str, "GAM_5");
    Meson mes(mu, str);
    mes.Zero();
    mes.setMass(q.Mass(),q.Mass());
    mes.calcMeson(q,q);
    if(!UniqueID())
    mes.Print(fp);
  }
  //tensor
  for ( mu = 0; mu < 4; mu++ ){
    for ( nu = mu+1; nu < 4; nu++ ){
      char str[256];
      sprintf(str, "GAM_%d%d",mu+1,nu+1);
      Meson mes1(mu, nu, str);
      mes1.Zero();
      mes1.setMass(q.Mass(),q.Mass());
      mes1.calcMeson(q,q);
      if(!UniqueID())mes1.Print(fp);

      sprintf(str, "GAM_%d%d",nu+1,mu+1);
      Meson mes2(nu, mu, str);
      mes2.Zero();
      mes2.setMass(q.Mass(),q.Mass());
      mes2.calcMeson(q,q);
      if(!UniqueID()) mes2.Print(fp);
    }
  }
  //axialvector
  nu = -5;
  for ( mu = 0; mu < 4; mu++ ){
    char str[256];
    sprintf(str, "GAM_%d5", mu+1);
    Meson mes(mu, nu, str);
    mes.Zero();
    mes.setMass(q.Mass(),q.Mass());
    mes.calcMeson(q,q);
    if(!UniqueID())
    mes.Print(fp);
  }

  // FIXME:  PUT IN THE  A0  SCALAR MESON, or WE WILL MISS A LOT !!
  
  Fclose(fp);

}

