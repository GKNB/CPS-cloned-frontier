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
NoArg no_arg;
CommonArg common_arg;
LanczosArg lanczos_arg;

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





int main(int argc,char *argv[])
{

  char *cname = argv[0] ;
  char *fname = "main()" ;
  char* filename;
  filename = (char*) smalloc( 128*sizeof(char) );
  
  Start(&argc, &argv);

  if ( argc!=8) { 
    if(!UniqueID())printf("(exe) do_arg doext_arg mobius_arg-l mobius_arg-s eig_arg pc_type work-directory\n");
    exit(-1);
  }
  
  chdir(argv[7]);

  if ( !do_arg.Decode(argv[1],"do_arg") ) 
    { 
      ERR.General(fname,fname,"Decoding of do_arg failed\n");
    }
  if ( !doext_arg.Decode(argv[2],"doext_arg") ) 
    { 
      ERR.General(fname,fname,"Decoding of doext_arg failed\n");
    }
#if 1
  if ( !mobius_arg.Decode(argv[3],"mobius_arg") ) 
    { 
      mobius_arg.Encode("mobius_arg.dat","mobius_arg");
      ERR.General(fname,fname,"Decoding of mobius_arg failed\n");  
    }
  if ( !mobius_arg2.Decode(argv[4],"mobius_arg2") ) 
    { 
      ERR.General(fname,fname,"Decoding of mobius_arg2 failed\n");  
    }
#endif
  if ( !lanczos_arg.Decode(argv[5],"lanczos_arg") ) 
    { 
      lanczos_arg.Encode("lanczos_arg.dat","lanczos_arg");
      if(!UniqueID())printf("Decoding of lanczos_arg failed\n"); exit(-1);
    }

  
  // make a record of what was run
  do_arg.Encode("do_arg.dat","do_arg");
  doext_arg.Encode("doext_argp.dat","doext_arg");
  lanczos_arg.Encode("lanczos_arg.dat","lanczos_arg");
  mobius_arg.Encode("mboius_arg.dat","mobius_arg");
  mobius_arg2.Encode("mobius_arg2.dat","mobius_arg2");

  
  GJP.Initialize(do_arg);
  GJP.InitializeExt(doext_arg);
  VRB.Level(do_arg.verbose_level);

  // GJP.ZMobius_PC_Type(ZMOB_PC_SYM2 ); 
  int zmob_pc_type = atoi(argv[6]);
  GJP.ZMobius_PC_Type((ZMobiusPCType)zmob_pc_type );

  

  
  //Lattice &lattice = LatticeFactory::Create(F_CLASS_DWF, G_CLASS_WILSON);
  //Lattice &lattice = LatticeFactory::Create(F_CLASS_ZMOBIUS, G_CLASS_NONE);
  //Lattice &lattice = LatticeFactory::Create(F_CLASS_MOBIUS, G_CLASS_NONE);

  GnoneFzmobius lattice;
  //GnoneFmobius lattice;

  // mult su3 links by qed phase
  //lattice.mult_su3_links_by_u1(1.0);  
  //qio_readLattice rl(do_arg.start_conf_filename, lattice, argc, argv);
 
  
    /*
     *  Compute eigen vectors and values
     *
     */
  Float etime = time_elapse();
  EigenCache *ecache;
  char cache_name[1024];
  snprintf(cache_name,1024,"cache_0_mass%g", lanczos_arg.mass);
  ecache = new EigenCache(cache_name);
  char evecname[1024];
  snprintf(evecname, 1024,"%s/eig4dee.ls%d.pc%d.mass%g",
	   lanczos_arg.file, mobius_arg2.ls, GJP.ZMobius_PC_Type(),lanczos_arg.mass );
  lanczos_arg.file = evecname;
  
  AlgLanczos  eig(lattice, &common_arg, &lanczos_arg, ecache);
  int Ncb = eig.NumChkb(lanczos_arg.RitzMat_lanczos);
  int ls_save = GJP.SnodeSites();
  GJP.SnodeSites(mobius_arg2.ls);
  GJP.ZMobius_b (mobius_arg2.zmobius_b_coeff.zmobius_b_coeff_val, mobius_arg2.ls);
  GJP.ZMobius_c (mobius_arg2.zmobius_c_coeff.zmobius_c_coeff_val, mobius_arg2.ls  );
  int fsize = GJP.VolNodeSites() * lattice.FsiteSize() * Ncb / 2 / 2; //last 2 for single prec.;
  EigenCacheList. push_back( ecache );
  int neig;
 
  char evecname_bc[1024];
  
  snprintf(evecname_bc,1024, "%s.bc%d%d%d%d", lanczos_arg.file, GJP.Bc(0),GJP.Bc(1),GJP.Bc(2),GJP.Bc(3));

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
  
#if 1
  if(mobius_arg2.cg.neig>0)
  {
    if( GJP.Snodes() != 1) ERR.NotImplemented(cname,fname,"currently only doing I/O for local Ls\n");

    int neig  = mobius_arg2.cg.neig;

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


 
  
#if 1
  Vector* in;
  Vector* out;
  CgArg cg;
  int ls = mobius_arg.ls;
  out = (Vector*)smalloc(GJP.VolNodeSites()*ls*4*sizeof(Vector));
  in = (Vector*)smalloc(GJP.VolNodeSites()*ls*4*sizeof(Vector));
  // zero the source, solution
  for(int i=0;i<GJP.VolNodeSites()*ls*24;i++){
    *((Float*)in+i) =0.0;
    *((Float*)out+i) =0.0;
  }
  //point source at origin and spin=color=0
  if(!UniqueID()) *((Float*)in) = 1.0;
  Float res;
  res = 0.0;
  PreserveType pres =  PRESERVE_NO;
  CnvFrmType cnv =  CNV_FRM_YES;



  GJP.SnodeSites(ls_save);
  mobius_arg2.cg.fname_eigen = evecname;
  cg = mobius_arg.cg;


  int iter = lattice.FmatInv(out, in, &mobius_arg,   &mobius_arg2,
			     &res, cnv, pres);
  
  //int iter = lattice.FmatInv(out, in, &cg, &res, cnv, pres);

  sfree(out);
  sfree(in);
  sfree(filename);

#endif

  //LatticeFactory::Destroy();
  
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

