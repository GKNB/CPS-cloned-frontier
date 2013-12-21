#include <conf.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <alg/alg_plaq.h>
#include <alg/qpropw.h>
#include <alg/alg_lanczos.h>
#include <alg/alg_meas.h>
#include <alg/meson.h>
#include <alg/alg_nuc3pt.h>
#include <alg/alg_fix_gauge.h>
#include <util/lattice.h>
#include <util/gjp.h>
#include <util/verbose.h>
#include <util/error.h>
#include <util/command_line.h>
#include <util/ReadLatticePar.h>
#include <util/qcdio.h>
#include <util/site.h>

#include <alg/lanczos_arg.h>

#include <util/eigen_container.h>


USING_NAMESPACE_CPS
using namespace std;

DoArg do_arg;
DoArgExt doext_arg;
NoArg no_arg;
CommonArg common_arg;
LanczosArg eig_arg;
MeasArg meas_arg;
Nuc3ptArg nuc3pt_arg;
QPropWArg qpropw_arg;
QPropWBoxArg qpropw_box_arg;
MatrixPolynomialArg cheby_arg;

void  movefloattoFloat(Float*,float*,int);

//------------------------------------------
// should go to eigen_container.C later, someday
//------------------------------------------
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

// Cleanup list EigenCache, it also destroies contents pointed by the elements.
void cps::EigenCacheListCleanup( )
{
  for(size_t i=0;i< EigenCacheList. size(); ++i){
    EigenCacheList[i]-> dealloc();
  }
  EigenCacheList.clear() ;
}

int main(int argc,char *argv[])
{

  char *cname = argv[0] ;
  char *fname = "main()" ;
  char* filename;
  filename = (char*) smalloc( 128*sizeof(char) );
  
  Start(&argc, &argv);
  
  //------------------------------
  //set log directory
  //------------------------------
  char log_dir[255];
  sprintf(&log_dir[0],"IOLOG");
  
  if ( argc!=10) { 
    if(!UniqueID())printf("(exe) do_arg doext_arg lanczos_arg nuc_arg meas_arg work-directory lmaxshft lmatshft StartShift\n");
    exit(-1);
  }
  
  // load defaults
  
  // move to working sub-directory (first arg on command line after exec.)
  // should be relative to /host.../username
  
  chdir(argv[6]);
  if ( !do_arg.Decode(argv[1],"do_arg") ) 
    { 
      ERR.General(fname,fname,"Decoding of do_arg failed\n");
    }
  if ( !doext_arg.Decode(argv[2],"doext_arg") ) 
    { 
      doext_arg.Encode("doext_arg.dat","doext_arg");
      ERR.General(fname,fname,"Decoding of doext_arg failed\n");
    }
  if ( !eig_arg.Decode(argv[3],"eig_arg") ) 
    { 
      eig_arg.Encode("lanczos_arg.dat","eig_arg");
      ERR.General(fname,fname,"Decoding of eig_arg failed\n");
    }
#if 0
  if ( !cheby_arg.Decode(argv[4],"cheby_arg") ) 
    { 
      ERR.General(fname,fname,"Decoding of cheby_arg failed\n");
    }
  eig_arg.matpoly_arg = (Float*)&cheby_arg; // ugly thing to workaround vml 
#endif
  if ( !nuc3pt_arg.Decode(argv[4],"nuc_arg") ) 
    { 
      nuc3pt_arg.Encode("nuc3pt_arg.dat","nuc3pt_arg");
      ERR.General(fname,fname,"Decoding of nuc_arg failed\n");
    }
  if ( !meas_arg.Decode(argv[5],"meas_arg") ) 
    { 
      ERR.General(fname,fname,"Decoding of meas_arg failed\n");
    }
 
  int lma_xshift = atoi(argv[7]);
  int lma_tshift = atoi(argv[8]);
  int StartShift = atoi(argv[9]);
 
  VRB.Level(do_arg.verbose_level);
  GJP.Initialize(do_arg);
  GJP.InitializeExt(doext_arg);
  
  //Lattice &lattice = LatticeFactory::Create(F_CLASS_DWF, G_CLASS_WILSON);
  Lattice &lattice = LatticeFactory::Create(F_CLASS_MOBIUS, G_CLASS_WILSON);
  
  
  do_arg.Encode("do_arg.dat","do_arg");
  eig_arg.Encode("lanczos_arg.dat","eig_arg");
  meas_arg.Encode("meas_arg.dat","meas_arg");
  nuc3pt_arg.Encode("nuc3pt_arg.dat","nuc3pt_arg");

  // loop over gauge field ensemble  
  for(meas_arg.TrajCur=meas_arg.TrajStart;
      meas_arg.TrajCur<=meas_arg.TrajLessThanLimit;
      meas_arg.TrajCur+=meas_arg.TrajIncrement) {
    
    int traj = meas_arg.TrajCur;
    char fname[1024];
    char eigfname[1024];
    char fname2[1024];
    
    AlgPlaq plaq(lattice,&common_arg,&no_arg);
    plaq.run();
    
    /*
     *  Compute eigen vectors and values
     *
     */
    Float etime = time_elapse();
    EigenCache *ecache;
    char cache_name[1024];
    snprintf(cache_name,1024,"cache_0_mass%g", eig_arg.mass);
    ecache = new EigenCache(cache_name);
    //snprintf(fname, 1024,"%s/eig4dee.mass%g.traj%04d", eig_arg.file, eig_arg.mass, traj );
    snprintf(eigfname, 1024,"%s/eig4dee.mass%g", eig_arg.file, eig_arg.mass);
    eig_arg.file = eigfname;
    
    AlgLanczos  eig(lattice, &common_arg, &eig_arg, ecache);
    int Ncb = eig.NumChkb(eig_arg.RitzMat_lanczos);
    int fsize = GJP.VolNodeSites() * lattice.FsiteSize() * Ncb / 2 / 2; //last 2 for single prec.;
    snprintf(fname2,1024, "%s.bc%d%d%d%d", eig_arg.file, GJP.Bc(0),GJP.Bc(1),GJP.Bc(2),GJP.Bc(3));
    EigenCacheList. push_back( ecache );
    int neig;

    if( eig_arg.nk_lanczos_vectors > 0) {
            
      int init_flag = 0; // 0 : wall start, 1 :  compression, 2 : compress & decompress, 3: refinement using Ritz
      int ncompress = 10; // number of evecs in the compressed linear combination
      
      char* comp_file;
      
      if( init_flag == 1){
	eig_arg. nk_lanczos_vectors = ncompress;
	eig_arg. np_lanczos_vectors = ncompress;
      }
      
      neig  = eig_arg.nk_lanczos_vectors+eig_arg.np_lanczos_vectors;
      ecache->alloc( fname2, neig, fsize );
      eig.run( init_flag, ncompress, comp_file );
      neig = nuc3pt_arg.cg.neig;
      ecache->set_neig(neig);
    } else {
      neig = nuc3pt_arg.cg.neig;
      ecache->alloc( fname2, neig, fsize );
      {//read in only
        const int n_fields =  GJP.SnodeSites();
        const int f_size_per_site = lattice.FsiteSize() / n_fields / 2 ;
        EigenContainer eigcon( lattice, fname2, neig, f_size_per_site/2, n_fields, ecache );
			// factor of 2 for single-prec.
        // have to do this if stride != 1 FIX!
        for(int iev=0; iev < neig ; iev++){
          Vector* evec= eigcon.nev_load( iev );
          ecache->set_index(iev);
        }
      }
    }

    etime = time_elapse();
    if(!UniqueID())printf("Time for Lanczos %g\n",etime);
    
#if 1
  {
    if( GJP.Snodes() != 1) ERR.NotImplemented(cname,fname,"currently only doing I/O for local Ls\n");

    int neig = nuc3pt_arg.cg.neig;

    const int n_fields =  GJP.SnodeSites();
    const int f_size_per_site = lattice.FsiteSize() / n_fields / 2 ;
                                                              // the two is for checkerboard?
    EigenContainer eigcon( lattice, fname2, neig, f_size_per_site, n_fields, ecache );
    Float* eval = eigcon.load_eval();
    Float* evecFloat = (Float*)smalloc(2 * fsize * sizeof(Float));

    for(int iev=0; iev< neig; iev++){

      Vector* evec= eigcon.nev_load( iev );
      movefloattoFloat(evecFloat,(float*)evec,fsize*2);

      if(iev % 1 == 0){  // Let's check all

        Float res_chk, eval_chk;
        Float mass = eig_arg.mass;

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
    if(!UniqueID())printf("main using cache n=%d name=%s\n",ecache->Neig(),ecache->Name());

    // I'm using mt to set the source time slice
    nuc3pt_arg.t_source = nuc3pt_arg. mt[0];

#if 1
  {
    snprintf(fname, 1024,"nuc3pt-exact.dat.%d", traj );
    common_arg.set_filename(fname);
    nuc3pt_arg. cg. fname_eigen = eig_arg.file;
    AlgNuc3pt nuc(lattice,&common_arg,&nuc3pt_arg); 
    nuc.run();
    etime = time_elapse();
    if(!UniqueID())printf("Time for Exact cg+contraction calc %g %d %d %d %d\n",
			   etime, nuc3pt_arg. x[0], nuc3pt_arg. x[1], nuc3pt_arg. x[2], nuc3pt_arg. mt[0]);
  }
#endif
#if 1 
  {
    // save the props this time
    nuc3pt_arg. ensemble_label = "nuc3pt4c"; 
    nuc3pt_arg. ensemble_id = traj; 
    nuc3pt_arg. calc_QProp = WRITE_QPROP;
    nuc3pt_arg. cg. max_num_iter = 100;
    snprintf(fname, 1024,"nuc3pt-rest-approx.dat.%d", traj );
    common_arg.set_filename(fname);
    nuc3pt_arg. cg. fname_eigen = eig_arg.file;
    AlgNuc3pt nuc(lattice,&common_arg,&nuc3pt_arg); 
    nuc.run();
    etime = time_elapse();
    if(!UniqueID())printf("Time for sloppy cg+contraction calc %g %d %d %d %d\n",
			   etime, nuc3pt_arg. x[0], nuc3pt_arg. x[1], nuc3pt_arg. x[2], nuc3pt_arg. mt[0]);
  }
#endif

#if 1
    /*
     *  Now do the LMA(Low Mode Approximation)
     */
    // set inverter to be  LMA
    // set in vml instead
    //nuc3pt_arg. cg. Inverter = LOWMODEAPPROX;
    //nuc3pt_arg. cg. stop_rsd = nuc3pt_arg. cg. ama_stop_rsd;
    snprintf(fname, 1024,"nuc3pt-approx.dat.%d", traj );
    common_arg.set_filename(fname);
    
    // number of shifts in each direction
    //int NUM_SHFT[4] = {2,2,2,4};
    int NUM_SHFT[4];
    NUM_SHFT[0] = lma_xshift;
    NUM_SHFT[1] = lma_xshift;
    NUM_SHFT[2] = lma_xshift;
    NUM_SHFT[3] = lma_tshift;
    int nshft = NUM_SHFT[0]*NUM_SHFT[1]*NUM_SHFT[2]*NUM_SHFT[3];
    
    // original gaussian source spatial location
    int orig_src[3]= { nuc3pt_arg.x[0], nuc3pt_arg.x[1], nuc3pt_arg.x[2] };
    int orig_mt[5]= { nuc3pt_arg.mt[0], nuc3pt_arg.mt[1],nuc3pt_arg.mt[2], nuc3pt_arg.mt[3], nuc3pt_arg.mt[4] };
    
    int sft[4];
    for(int isft_=StartShift; isft_< nshft;isft_++){
      
      // reset the source location
      nuc3pt_arg. x[0] =  orig_src[0];
      nuc3pt_arg. x[1] =  orig_src[1];
      nuc3pt_arg. x[2] =  orig_src[2];
      for(int i=0;i<nuc3pt_arg.num_mult;i++)
    	nuc3pt_arg. mt[i] =  orig_mt[i];
      
      // compute the shift steps
      int isft = isft_;
      sft[0] = isft % NUM_SHFT[0]; isft /= NUM_SHFT[0];
      sft[1] = isft % NUM_SHFT[1]; isft /= NUM_SHFT[1];
      sft[2] = isft % NUM_SHFT[2]; isft /= NUM_SHFT[2];
      sft[3] = isft % NUM_SHFT[3]; isft /= NUM_SHFT[3];
      
      // do shifts
      nuc3pt_arg. x[0] += (GJP.Sites(0)/ NUM_SHFT[0]) *sft[0]; 
      nuc3pt_arg. x[1] += (GJP.Sites(1)/ NUM_SHFT[1]) *sft[1];
      nuc3pt_arg. x[2] += (GJP.Sites(2)/ NUM_SHFT[2]) *sft[2];
      for(int i=0;i<nuc3pt_arg.num_mult;i++)
    	nuc3pt_arg. mt[i] += (GJP.Sites(3)/ NUM_SHFT[3]) *sft[3];
      
      // make sure this is within the nodes
      nuc3pt_arg. x[0] %= GJP.Sites(0);
      nuc3pt_arg. x[1] %= GJP.Sites(1);
      nuc3pt_arg. x[2] %= GJP.Sites(2);
      for(int i=0;i<nuc3pt_arg.num_mult;i++)
    	nuc3pt_arg. mt[i] %= GJP.Sites(3);
      // I'm using mt to set the source time slice
      nuc3pt_arg.t_source = nuc3pt_arg. mt[0];

      // root name of eigen value/vector to be read 
      nuc3pt_arg. cg. fname_eigen = eig_arg.file;
      AlgNuc3pt nuc(lattice,&common_arg,&nuc3pt_arg); 
      nuc.run();
      etime = time_elapse();
      if(!UniqueID())printf("Time for LMA/AMA %g shift %d %d %d %d\n",
			    etime, nuc3pt_arg. x[0], nuc3pt_arg. x[1], nuc3pt_arg. x[2], nuc3pt_arg. mt[0]);
    }

#endif
    
  }
  LatticeFactory::Destroy();
  sfree(filename);
  
  EigenCacheListCleanup();
  
  End();
  
  return 0;
} 

void movefloattoFloat(Float* out, float* in, int f_size){

  float flt;
  for(int i=0;i<f_size;i++){
    flt = in[i];
    out[i] = (Float)flt;
  }
};

