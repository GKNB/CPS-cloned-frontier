//Test G-parity modified version of Daiqian's a2a code. This presupposes that the Lanczos works, so test this first (gparity_lanczos)
//NOTE: You will need to link against libfftw3 and libfftw3_threads

#include <alg/a2a/lanc_arg.h>
#include <alg/eigen/Krylov_5d.h>



#include<chroma.h>
//bfm headers
#include<actions/ferm/invert/syssolver_linop_cg_array.h>
#include<bfm.h>
#include<bfm_qdp.h>
#include<bfm_cg.h>
#include<bfm_mprec.h>


//cps headers
#include<alg/fermion_vector.h>
#include<alg/do_arg.h>
#include<alg/meas_arg.h>
#include<util/qioarg.h>
#include<util/ReadLatticePar.h>

//c++ classes
#include<sys/stat.h>
#include<util/qcdio.h>
//#include<fftw3.h>

#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <util/qcdio.h>
#ifdef PARALLEL
#include <comms/sysfunc_cps.h>
#endif
#include <comms/scu.h>
#include <comms/glb.h>

#include <util/lattice.h>
#include <util/dirac_op.h>
#include <util/time_cps.h>
#include <alg/do_arg.h>
#include <alg/no_arg.h>
#include <alg/common_arg.h>
#include <alg/hmd_arg.h>
#include <alg/alg_plaq.h>
#include <alg/alg_rnd_gauge.h>
#include <alg/threept_arg.h>
#include <alg/threept_prop_arg.h>
#include <alg/alg_threept.h>
#include <util/smalloc.h>

#include <util/ReadLatticePar.h>
#include <util/WriteLatticePar.h>

#include <util/command_line.h>
#include <sstream>

#include<unistd.h>
#include<config.h>

#include <alg/qpropw.h>
#include <alg/qpropw_arg.h>

#include <alg/alg_fix_gauge.h>
#include <alg/fix_gauge_arg.h>

#include <util/data_shift.h>

#include <alg/prop_attribute_arg.h>
#include <alg/gparity_contract_arg.h>
#include <alg/propmanager.h>
#include <alg/alg_gparitycontract.h>

#include <util/gparity_singletodouble.h>

#include <alg/a2a/alg_a2a.h>
#include <alg/a2a/MesonField.h>
//some piece of **** defines these elsewhere, so the bfm header gets screwed up
#undef ND
#undef SPINOR_SIZE
#undef HALF_SPINOR_SIZE
#undef GAUGE_SIZE
#undef Nmu
#undef Ncb
#undef NMinusPlus
#undef Minus
#undef Plus
#undef DaggerYes
#undef DaggerNo
#undef SingleToDouble
#undef DoubleToSingle
#undef Odd
#undef Even


#include <util/lattice/bfm_evo.h>
#include <util/lattice/bfm_eigcg.h>
#include <util/lattice/fbfm.h>
#include <util/wilson.h>
#include <util/verbose.h>
#include <util/gjp.h>
#include <util/error.h>
#include <comms/scu.h>
#include <comms/glb.h>
#include <util/enum_func.h>
#include <util/sproj_tr.h>
#include <util/time_cps.h>
#include <util/lattice/fforce_wilson_type.h>

#include<omp.h>

#ifdef HAVE_BFM
#include <chroma.h>
#endif

using namespace std;
USING_NAMESPACE_CPS

void setup_double_latt(Lattice &double_latt, cps::Matrix* orig_gfield, bool gparity_X, bool gparity_Y){
  //orig latt ( U_0 U_1 ) ( U_2 U_3 ) ( U_4 U_5 ) ( U_6 U_7 )
  //double tatt ( U_0 U_1 U_2 U_3 ) ( U_4 U_5 U_6 U_7 ) ( U_0* U_1* U_2* U_3* ) ( U_4* U_5* U_6* U_7* )

  cps::Matrix *dbl_gfield = double_latt.GaugeField();

  if(!UniqueID()){ printf("Setting up 1f lattice.\n"); fflush(stdout); }
  SingleToDoubleLattice lattdoubler(gparity_X,gparity_Y,orig_gfield,double_latt);
  lattdoubler.Run();
  if(!UniqueID()){ printf("Finished setting up 1f lattice\n"); fflush(stdout); }
}
void setup_double_rng(bool gparity_X, bool gparity_Y){
  //orig 4D rng 2 stacked 4D volumes
  //orig ([R_0 R_1][R'_0 R'_1])([R_2 R_3][R'_2 R'_3])([R_4 R_5][R'_4 R'_5])([R_6 R_7][R'_6 R'_7])
  //double (R_0 R_1 R_2 R_3)(R_4 R_5 R_6 R_7)(R'_0 R'_1 R'_2 R'_3)(R'_4 R'_5 R'_6 R'_7)
  
  //orig 5D rng 2 stacked 4D volumes per ls/2 slice (ls/2 as only one RNG per 2^4 block)

  SingleToDouble4dRNG fourDsetup(gparity_X,gparity_Y);
  SingleToDouble5dRNG fiveDsetup(gparity_X,gparity_Y);
  
  LRG.Reinitialize(); //reset the LRG and prepare for doubled lattice form
  
  if(!UniqueID()){ printf("Setting up 1f 4D RNG\n"); fflush(stdout); }
  fourDsetup.Run();      
  if(!UniqueID()){ printf("Setting up 1f 5D RNG\n"); fflush(stdout); }
  fiveDsetup.Run();    
}
void setup_double_matrixfield(cps::Matrix* double_mat, cps::Matrix* orig_mat, int nmat_per_site, bool gparity_X, bool gparity_Y){
  if(!UniqueID()){ printf("Setting up 1f matrix field.\n"); fflush(stdout); }
  SingleToDoubleMatrixField doubler(gparity_X,gparity_Y,nmat_per_site,orig_mat,double_mat);
  doubler.Run();
  if(!UniqueID()){ printf("Finished setting up 1f matrixfield\n"); fflush(stdout); }
}
void setup_double_5d_vector(Vector *double_vect, Vector* orig_vect, bool gparity_X, bool gparity_Y){
  if(!UniqueID()){ printf("Setting up 1f vector field.\n"); fflush(stdout); }
  SingleToDouble5dVectorField doubler(gparity_X, gparity_Y, orig_vect, double_vect, CANONICAL);
  doubler.Run();
  if(!UniqueID()){ printf("Finished setting up 1f vector field\n"); fflush(stdout); }
}
  
void GaugeTransformU(cps::Matrix *gtrans, Lattice &lat);

void convert_ferm_cpsord_sord(Float *cps, Float* &sord, bfm_evo<Float> &bfm){
  Fermion_t handle[2] = { bfm.allocFermion(), bfm.allocFermion() };
  bfm.cps_impexFermion(cps,handle,1);
  
  long f_size = (long)24 * GJP.VolNodeSites() * GJP.SnodeSites();
  if(GJP.Gparity()) f_size*=2;
  sord = (Float *)pmalloc(sizeof(Float) * f_size);
  bfm.cps_impexFermion_s(sord,handle,0);

  bfm.freeFermion(handle[0]);
  bfm.freeFermion(handle[1]);
}
void convert_ferm_sord_cpsord(Float *sord, Float* &cps, bfm_evo<Float> &bfm){
  Fermion_t handle[2] = { bfm.allocFermion(), bfm.allocFermion() };
  bfm.cps_impexFermion_s(sord,handle,1);
  
  long f_size = (long)24 * GJP.VolNodeSites() * GJP.SnodeSites();
  if(GJP.Gparity()) f_size*=2;
  cps = (Float *)pmalloc(sizeof(Float) * f_size);
  bfm.cps_impexFermion(cps,handle,0);

  bfm.freeFermion(handle[0]);
  bfm.freeFermion(handle[1]);
}


void setup_bfmargs(bfmarg &dwfa, const BfmSolver &solver = DWF){
  printf("Setting up bfmargs\n");

   int nthreads = 1; 
#if TARGET == BGQ
   nthreads = 64;
#endif
   omp_set_num_threads(nthreads);

  dwfa.node_latt[0]  = GJP.XnodeSites();
  dwfa.node_latt[1]  = GJP.YnodeSites();
  dwfa.node_latt[2]  = GJP.ZnodeSites();
  dwfa.node_latt[3]  = GJP.TnodeSites();
  
  multi1d<int> ncoor(4);
  multi1d<int> procs(4);
  for(int i=0;i<4;i++){ ncoor[i] = GJP.NodeCoor(i); procs[i] = GJP.Nodes(i); }

  if(GJP.Gparity()){
    dwfa.gparity = 1;
    printf("G-parity directions: ");
    for(int d=0;d<3;d++)
      if(GJP.Bc(d) == BND_CND_GPARITY){ dwfa.gparity_dir[d] = 1; printf("%d ",d); }
      else dwfa.gparity_dir[d] = 0;
    for(int d=0;d<4;d++){
      dwfa.nodes[d] = procs[d];
      dwfa.ncoor[d] = ncoor[d];
    }
    printf("\n");
  }

  dwfa.verbose=1;
  dwfa.reproduce=0;
  bfmarg::Threads(nthreads);
  bfmarg::Reproduce(0);
  bfmarg::ReproduceChecksum(0);
  bfmarg::ReproduceMasterCheck(0);
  bfmarg::Verbose(1);

  for(int mu=0;mu<4;mu++){
    if ( procs[mu]>1 ) {
      dwfa.local_comm[mu] = 0;
      printf("Non-local comms in direction %d\n",mu);
    } else { 
      dwfa.local_comm[mu] = 1;
      printf("Local comms in direction %d\n",mu);
    }
  }

  dwfa.precon_5d = 1;
  if(solver == HmCayleyTanh){
    dwfa.precon_5d = 0; //mobius uses 4d preconditioning
    dwfa.mobius_scale = 2.0; //b = 0.5(scale+1) c=0.5(scale-1), hence this corresponds to b=1.5 and c=0.5, the params used for the 48^3
  }
  dwfa.Ls   = GJP.SnodeSites();
  dwfa.solver = solver;
  dwfa.M5   = toDouble(GJP.DwfHeight());
  dwfa.mass = toDouble(0.5);
  dwfa.Csw  = 0.0;
  dwfa.max_iter = 5000;
  dwfa.residual = 1e-08;
  printf("Finished setting up bfmargs\n");
}

Float* rand_5d_canonical_fermion(Lattice &lat){
  long f_size = (long)24 * GJP.VolNodeSites() * GJP.SnodeSites();
  if(GJP.Gparity()) f_size*=2;
  Float *v1 = (Float *)pmalloc(sizeof(Float) * f_size);
  printf("Making random gaussian 5d vector\n");
  lat.RandGaussVector((Vector*)v1, 0.5, 2, CANONICAL, FIVE_D);
  printf("Finished making random gaussian vector\n");
  return v1;
}

void lanczos_arg(LancArg &into, const bool &precon){
  into.mass = 0.01;
  into.stop_rsd = 1e-06;
  into.qr_rsd = 1e-14; ///convergence of intermediate QR solves, defaults to 1e-14
  into.EigenOper = DDAGD;
  into.precon = precon; //also try this with true
  into.N_get = 10;///Want K converged vectors
  into.N_use = 11;///Dimension M Krylov space
  into.N_true_get = 10;//Actually number of eigen vectors you will get
  into.ch_ord = 3;///Order of Chebyshev polynomial
  into.ch_alpha = 5.5;///Spectral radius
  into.ch_beta = 0.5;///Spectral offset (ie. find eigenvalues of magnitude less than this)
  into.ch_sh = false;///Shifting or not
  into.ch_mu = 0;///Shift the peak
  into.lock = false;///Use locking transofrmation or not
  into.maxits =10000;///maxiterations
  into.fname = "Lanczos";
}

void a2a_arg(A2AArg &into){
  into.nl = 8;
  into.nhits = 1;
  into.rand_type = UONE;
  into.src_width = 1;
}

void create_eig(GwilsonFdwf* lattice, Lanczos_5d<double>* &eig, const bool &precon){
  //Run in 2f environment. Version for 1f is directly converted from this
  bfm_evo<double>* dwf = new bfm_evo<double>();
  bfmarg dwfa;
  setup_bfmargs(dwfa);

  dwf->init(dwfa);
  
  lattice->BondCond(); //Don't forget to apply the boundary conditions!
  Float* gauge = (Float*) lattice->GaugeField();
  dwf->cps_importGauge(gauge); 

  //Setup and run the lanczos algorithm
  LancArg lanc_arg; lanczos_arg(lanc_arg,precon);
  eig = new Lanczos_5d<double>(*dwf,lanc_arg);
  eig->Run();
  lattice->BondCond(); //Don't forget to un-apply the boundary conditions!
}
  
//Note, only the eigenvectors and eigenvalues of the Lanczos_5d instance are used, hence for a cleaner test I generate these on the 2f lattice
//and simply convert them to the 1f lattice
//The extraction of the eigenvectors to CPS format (needed for conversion) should be done BEFORE moving to the 1f environment
void eig_convert_cps(Lanczos_5d<double> &eig, Float** &eigv_2f_cps, const bool &precon){
  //EXECUTED IN 2F ENVIRONMENT
  long f_size = (long)2*24 * GJP.VolNodeSites() * GJP.SnodeSites();
  multi1d<bfm_fermion> &eigenvecs = eig.bq;
  
  eigv_2f_cps = (Float**)pmalloc(eigenvecs.size() * sizeof(Float*) );
  for(int i=0;i<eigenvecs.size();i++){
    eigv_2f_cps[i] = (Float *)pmalloc(sizeof(Float) * f_size); 
    if(precon){ eigenvecs[i][0] = eig.dop.allocCompactFermion();  eig.dop.set_zero(eigenvecs[i][0]); } //for precon ferms this is not allocated    
    eig.dop.cps_impexFermion(eigv_2f_cps[i],eigenvecs[i],0);
  }
}

void eig_convert_2f_1f(Lanczos_5d<double> &eig, Float** eigv_2f_cps, const std::vector<double> &eval_2f, const bool &precon, const bool &gparity_X, const bool &gparity_Y){
  //THIS SHOULD BE EXECUTED WITHIN THE 1F ENVIRONMENT
  long f_size = (long)24 * GJP.VolNodeSites() * GJP.SnodeSites();
  multi1d<bfm_fermion> &eigenvecs = eig.bq;
  //temp storage
  Float *v_1f_cps = (Float *)pmalloc(sizeof(Float) * f_size); 

  for(int i=0;i<eigenvecs.size();i++){
    setup_double_5d_vector((Vector*)v_1f_cps,(Vector*)eigv_2f_cps[i], gparity_X,gparity_Y);
    if(precon){ eigenvecs[i][0] = eig.dop.allocCompactFermion(); } //for precon ferms this is not allocated 
    eig.dop.cps_impexFermion(v_1f_cps,eigenvecs[i],1); //import
    if(precon){ eig.dop.freeFermion(eigenvecs[i][0]); } //delete the even checkerboard that we allocated above
    pfree(eigv_2f_cps[i]);
  }
  eig.evals = eval_2f;

  pfree(v_1f_cps);
  pfree(eigv_2f_cps);
}

//Create eig but do not run, instead copy over evals and converted evecs from 2f
void create_eig_1f(GwilsonFdwf* lattice, Lanczos_5d<double>* &eig_1f, const Lanczos_5d<double> &eig_2f, Float** eigv_2f_cps, const bool &precon, const bool &gparity_X, const bool &gparity_Y){
  bfm_evo<double>* dwf = new bfm_evo<double>();
  bfmarg dwfa;
  setup_bfmargs(dwfa);

  dwf->init(dwfa);
  
  lattice->BondCond(); //Don't forget to apply the boundary conditions!
  Float* gauge = (Float*) lattice->GaugeField();
  dwf->cps_importGauge(gauge); 
  
  LancArg lanc_arg; lanczos_arg(lanc_arg,precon);
  eig_1f = new Lanczos_5d<double>(*dwf,lanc_arg);  
  eig_1f->do_init(); //normally called in Run
  eig_convert_2f_1f(*eig_1f, eigv_2f_cps, eig_2f.evals, precon, gparity_X, gparity_Y);
  lattice->BondCond(); //Don't forget to un-apply the boundary conditions!
}


//Lanczos should be precomputed. prop pointer will be assigned to an a2a propagator
void a2a_prop_gen(A2APropbfm* &prop, GwilsonFdwf* lattice, Lanczos_5d<double> &eig){
  A2AArg arg;
  a2a_arg(arg);
  
  bfm_evo<double> &dwf = eig.dop;

  CommonArg carg;
  prop = new A2APropbfm(*lattice,arg,carg,&eig);

  prop->allocate_vw();
  QDPIO::cout << "Computing A2a low modes component\n";
  prop->compute_vw_low(dwf);
  QDPIO::cout << "Computing A2a high modes component\n";
  prop->compute_vw_high(dwf);
  QDPIO::cout << "Completed A2A prop\n";
}
void convert_2f_A2A_prop_to_1f(A2APropbfm &prop_2f, bool gparity_X, bool gparity_Y){
  //Called in 1f context!

  int four_vol = GJP.VolNodeSites(); //size includes factor of 2 for second G-parity flavour as we are in the 1f context
  //Vector **v; // v  size   [nvec][24*four_vol *2(gp)]
  for(int vec = 0; vec < prop_2f.get_nvec(); ++vec){
    int vsz = 24*four_vol*sizeof(Float);
    Float* new_v = (Float*)pmalloc(vsz);
    SingleToDoubleField doubler(gparity_X, gparity_Y, 24, (Float*)prop_2f.get_v(vec), new_v);
    doubler.Run();
    memcpy( (void*)prop_2f.get_v(vec), (void*)new_v, vsz);
    pfree(new_v);
  }
  
  //Vector **wl; // low modes w [A2AArg.nl][24*four_vol *2(gp)]

  for(int l = 0; l < prop_2f.get_nl(); ++l){
    int wlsz = 24*four_vol*sizeof(Float);
    Float* new_wl = (Float*)pmalloc(wlsz);
    SingleToDoubleField doubler(gparity_X, gparity_Y, 24, (Float*)prop_2f.get_wl(l), new_wl);
    doubler.Run();
    memcpy( (void*)prop_2f.get_wl(l), (void*)new_wl, wlsz);
    pfree(new_wl);
  }

  //Vector *wh; // high modes w, compressed  [2*four_vol*nh_site*2]   where nh_site = A2AArg.nhits

  //G-parity wh mapping  (re/im=[0,1], site=[0..four_vol-1],flav=[0,1],hit=[0..nh_site]) ->    re/im + 2*( site + flav*four_vol + hit*2*four_vol )
  //Standard  (re/im=[0,1], site=[0..four_vol-1],hit=[0..nh_site]) ->    re/im + 2*( site + hit*four_vol )
  {
    int wh_bytes = 2*four_vol*prop_2f.get_nhits()*sizeof(Float);
    Float* new_wh = (Float*)pmalloc(wh_bytes);
    for(int hit = 0; hit < prop_2f.get_nhits(); hit++){
      int hit_off = 2*four_vol*hit;
      SingleToDoubleField doubler(gparity_X, gparity_Y, 2, (Float*)prop_2f.get_wh() + hit_off, new_wh + hit_off);
      doubler.Run();
    }
    memcpy( (void*)prop_2f.get_wh(), (void*)new_wh, wh_bytes);
    pfree(new_wh);
  }

  //FFTW fields

  if(prop_2f.get_v_fftw(0)!=NULL){

    //Assumes just 1 set of v and w (i.e. no strange quark)
    //v_fftw   size a2a_prop->get_nvec()
    for(int i=0;i< prop_2f.get_nvec(); i++){
      //On each index is a CPS fermion
      int vsz = 24*four_vol*sizeof(Float);
      Float* new_v = (Float*)pmalloc(vsz);
      SingleToDoubleField doubler(gparity_X, gparity_Y, 24, (Float*)prop_2f.get_v_fftw(i), new_v);
      doubler.Run();
      memcpy( (void*)prop_2f.get_v_fftw(i), (void*)new_v, vsz);
      pfree(new_v);
    }
    //wl_fftw[0] (light part)  size a2a_prop->get_nl()
    for(int i=0;i< prop_2f.get_nl(); i++){
      //On each index is a CPS fermion
      int vsz = 24*four_vol*sizeof(Float);
      Float* new_v = (Float*)pmalloc(vsz);
      SingleToDoubleField doubler(gparity_X, gparity_Y, 24, (Float*)prop_2f.get_wl_fftw(i), new_v);
      doubler.Run();
      memcpy( (void*)prop_2f.get_wl_fftw(i), (void*)new_v, vsz);
      pfree(new_v);
    }
    //wh_fftw[0] is 12*a2a_prop->get_nhits() stacked fermion vectors
    for(int i=0;i< 12*prop_2f.get_nhits(); i++){
      int off = i*24*four_vol; //(float units)
      Float* fld = (Float*)prop_2f.get_wh_fftw() + off;

      int vsz = 24*four_vol*sizeof(Float);
      Float* new_v = (Float*)pmalloc(vsz);
      SingleToDoubleField doubler(gparity_X, gparity_Y, 24, fld, new_v);
      doubler.Run();
      memcpy( fld, (void*)new_v, vsz);
      pfree(new_v);
    }     

  }
}


//Compare 1f 2f A2A propagator components (v and w). 2f propagator should have been converted to 1f layout using the above
void compare_1f_2f_A2A_prop(A2APropbfm &prop_2f, A2APropbfm &prop_1f){
  //CALL IN 1F CONTEXT
  int four_vol = GJP.VolNodeSites(); 
  printf("Comparing 1f and 2f A2A propagators\n");
  //Compare wl
  for(int l = 0; l < prop_2f.get_nl(); ++l){
    printf("\n\n1f 2f A2A prop starting wl comparison, mode %d\n",l);
    int wlsz = 24*four_vol;
    Float* wl_2f = (Float*)prop_2f.get_wl(l);
    Float* wl_1f = (Float*)prop_1f.get_wl(l);

    bool fail = false;
    for(int i=0;i<wlsz;++i)
      if( fabs(wl_2f[i]-wl_1f[i]) > 1e-12 ){
	printf("1f 2f A2A prop wl comparison fail mode %d, offset %d.  1f: %.13e     2f: %.13e\n",l,i,wl_1f[i],wl_2f[i]);
	fail = true;
      }
    if(fail){
      printf("1f 2f A2A prop wl comparison failed on mode %d\n",l); exit(-1);
    }
  }
  //Compare wh
  for(int hit = 0; hit < prop_2f.get_nhits(); hit++){
    printf("\n\n1f 2f A2A prop starting wh comparison, hit %d\n",hit);
    int hit_off = 2*four_vol*hit;

    Float* wh_2f = (Float*)prop_2f.get_wh() + hit_off;
    Float* wh_1f = (Float*)prop_1f.get_wh() + hit_off;

    int whsz = 2*four_vol;

    bool fail = false;
    for(int i=0;i<whsz;++i)
      if( fabs(wh_2f[i]-wh_1f[i]) > 1e-12 ){
	printf("1f 2f A2A prop wh comparison fail hit %d, offset %d.  1f: %.13e     2f: %.13e\n",hit,i,wh_1f[i],wh_2f[i]);
	fail = true;
      }
    if(fail){
      printf("1f 2f A2A prop wh comparison failed on hit %d\n",hit); exit(-1);
    }
  }  
  //Compare v
  for(int vec = 0; vec < prop_2f.get_nvec(); ++vec){
    printf("\n\n1f 2f A2A prop starting v comparison, vec %d\n",vec);
    int vsz = 24*four_vol;  
    Float* v_2f = (Float*)prop_2f.get_v(vec);
    Float* v_1f = (Float*)prop_1f.get_v(vec);

    bool fail = false;
    for(int i=0;i<vsz;++i)
      if( fabs(v_2f[i]-v_1f[i]) > 1e-12 ){
	printf("1f 2f A2A prop v comparison fail vec %d, offset %d.  1f: %.13e     2f: %.13e\n",vec,i,v_1f[i],v_2f[i]);
	fail = true;
      }
    if(fail){
      printf("1f 2f A2A prop v comparison failed on vec %d\n",vec); exit(-1);
    }
  }
  printf("Successfully compared 1f and 2f A2A propagators\n");
}




void setup_gfix_args(FixGaugeArg &r){
  r.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
  r.hyperplane_start = 0;
  r.hyperplane_step = 1;
  r.hyperplane_num = GJP.TnodeSites();
  r.stop_cond = 1e-06;
  r.max_iter_num = 6000;
}

CPS_START_NAMESPACE

class MesonFieldTesting{
public:
  static void convert_mesonfield_2f_1f(MesonField &mf, const bool &gparity_X, const bool &gparity_Y){
    //EXECUTE IN 1F ENVIRONMENT

    int four_vol = GJP.VolNodeSites();
    //Assumes just 1 set of v and w (i.e. no strange quark)
    //v_fftw   size a2a_prop->get_nvec()
    for(int i=0;i< mf.a2a_prop->get_nvec(); i++){
      //On each index is a CPS fermion
      int vsz = 24*four_vol*sizeof(Float);
      Float* new_v = (Float*)pmalloc(vsz);
      SingleToDoubleField doubler(gparity_X, gparity_Y, 24, (Float*)mf.v_fftw[i], new_v);
      doubler.Run();
      memcpy( (void*)mf.v_fftw[i], (void*)new_v, vsz);
      pfree(new_v);
    }
    //wl_fftw[0] (light part)  size a2a_prop->get_nl()
    for(int i=0;i< mf.a2a_prop->get_nl(); i++){
      //On each index is a CPS fermion
      int vsz = 24*four_vol*sizeof(Float);
      Float* new_v = (Float*)pmalloc(vsz);
      SingleToDoubleField doubler(gparity_X, gparity_Y, 24, (Float*)mf.wl_fftw[0][i], new_v);
      doubler.Run();
      memcpy( (void*)mf.wl_fftw[0][i], (void*)new_v, vsz);
      pfree(new_v);
    }
    //wh_fftw[0] is 12*a2a_prop->get_nhits() stacked fermion vectors
    for(int i=0;i< 12*mf.a2a_prop->get_nhits(); i++){
      int off = i*24*four_vol; //(float units)
      Float* fld = (Float*)mf.wh_fftw[0] + off;

      int vsz = 24*four_vol*sizeof(Float);
      Float* new_v = (Float*)pmalloc(vsz);
      SingleToDoubleField doubler(gparity_X, gparity_Y, 24, fld, new_v);
      doubler.Run();
      memcpy( fld, (void*)new_v, vsz);
      pfree(new_v);
    }       
    //Note 1f and 2f mesonfields themselves have the same layout
  }
  static void compare_fftw_vecs(MesonField &_1f, MesonField &_2f){
    printf("Comparing meson fields\n");
    int four_vol = GJP.VolNodeSites();

    for(int i=0;i< _1f.a2a_prop->get_nl(); i++){
      printf("Comparing wl_fftw for vec idx %d\n",i);
      bool fail(false);
      for(int j=0;j<24*four_vol;j++){
	Float v1f = ((Float*)_1f.wl_fftw[0][i])[j];
	Float v2f = ((Float*)_2f.wl_fftw[0][i])[j];
	if( fabs( v1f  - v2f ) > 1e-12 ){ printf("Failed on wl_fftw mode idx %d at ferm offset %d: %.14e  %.14e\n",i,j,v1f,v2f); fail = true; }
      }
      if(fail){ printf("Comparison of wl_fftw for vec idx %d failed\n",i); exit(-1); }
    }
    //wh_fftw[0] is 12*a2a_prop->get_nhits() stacked fermion vectors
    for(int i=0;i< 12*_1f.a2a_prop->get_nhits(); i++){
      int off = i*24*four_vol; //(float units)
      Float* fl1f = (Float*)_1f.wh_fftw[0] + off;
      Float* fl2f = (Float*)_2f.wh_fftw[0] + off;

      printf("Comparing wh_fftw for hit/spin-color idx %d\n",i);
      bool fail(false);
      for(int j=0;j<24*four_vol;j++){
	Float v1f = fl1f[j];
	Float v2f = fl2f[j];
	if( fabs( v1f  - v2f ) > 1e-12 ){ printf("Failed on wh_fftw hit/spin-color idx %d at ferm offset %d: %.14e  %.14e\n",i,j,v1f,v2f); fail = true; }
      }
      if(fail){ printf("Comparison of wh_fftw for hit/spin-color idx %d failed\n",i); exit(-1); }
    }
    for(int i=0;i< _1f.a2a_prop->get_nvec(); i++){
      printf("Comparing v_fftw for vec idx %d\n",i);
      bool fail(false);
      for(int j=0;j<24*four_vol;j++){
	Float v1f = ((Float*)_1f.v_fftw[i])[j];
	Float v2f = ((Float*)_2f.v_fftw[i])[j];
	if( fabs( v1f  - v2f ) > 1e-12 ){ printf("Failed on v_fftw vec idx %d at ferm offset %d: %.14e  %.14e\n",i,j,v1f,v2f); fail = true; }
      }
      if(fail){ printf("Comparison of v_fftw for vec idx %d failed\n",i); exit(-1); }
    }


    printf("Passed comparison of meson fields\n");
  }
  //Ensure that the old MesonField FFT results are the same as those obtained from new A2APropBfm
  static void compare_fftw_fields(MesonField &mf, A2APropbfm &a2a){
    printf("Comparing FFTW fields between MesonField and A2APropbfm\n");
    int four_vol = GJP.VolNodeSites();
    int gpfac = (GJP.Gparity()?2:1);

    for(int i=0;i< a2a.get_nl(); i++){
      printf("Comparing wl_fftw for vec idx %d\n",i);
      bool fail(false);
      for(int j=0;j< gpfac*24*four_vol;j++){
	Float v1f = ((Float*)mf.wl_fftw[0][i])[j];
	Float v2f = ((Float*)a2a.get_wl_fftw(i))[j];
	if( fabs( v1f  - v2f ) > 1e-12 ){ printf("Failed on wl_fftw mode idx %d at ferm offset %d: %.14e  %.14e\n",i,j,v1f,v2f); fail = true; }
      }
      if(fail){ printf("Comparison of wl_fftw for vec idx %d failed\n",i); exit(-1); }
    }
    for(int i=0;i< 12* a2a.get_nhits(); i++){
      int off = i*gpfac*24*four_vol; //(float units)
      Float* fl1f = (Float*)mf.wh_fftw[0] + off;
      Float* fl2f = (Float*)a2a.get_wh_fftw() + off;

      printf("Comparing wh_fftw for hit/spin-color idx %d\n",i);
      bool fail(false);
      for(int j=0;j<gpfac*24*four_vol;j++){
	Float v1f = fl1f[j];
	Float v2f = fl2f[j];
	if( fabs( v1f  - v2f ) > 1e-12 ){ printf("Failed on wh_fftw hit/spin-color idx %d at ferm offset %d: %.14e  %.14e\n",i,j,v1f,v2f); fail = true; }
      }
      if(fail){ printf("Comparison of wh_fftw for hit/spin-color idx %d failed\n",i); exit(-1); }
    }
    for(int i=0;i< a2a.get_nvec(); i++){
      printf("Comparing v_fftw for vec idx %d\n",i);
      bool fail(false);
      for(int j=0;j<gpfac*24*four_vol;j++){
	Float v1f = ((Float*)mf.v_fftw[i])[j];
	Float v2f = ((Float*)a2a.get_v_fftw(i))[j];
	if( fabs( v1f  - v2f ) > 1e-12 ){ printf("Failed on v_fftw vec idx %d at ferm offset %d: %.14e  %.14e\n",i,j,v1f,v2f); fail = true; }
      }
      if(fail){ printf("Comparison of v_fftw for vec idx %d failed\n",i); exit(-1); }
    }
    printf("Passed comparison of FFTW fields between MesonField and A2APropbfm\n");
  }
  
  //Compare the 1f and 2f mesonfield object MesonField.mf. Has the same memory layout for the 2 implementations so no conversion is necessary
  static void compare_mf_ll(MesonField &mf_1f, MesonField &mf_2f){
    int t_size = GJP.TnodeSites()*GJP.Tnodes();
    int size = mf_1f.nvec[0]*(mf_1f.nl[0]+12*mf_1f.nhits[0])*t_size*2;
    bool fail(false);
    for(int i=0;i<size;i++){
      int rem = i;
      int reim = rem % 2; rem/=2;
      int glb_t = rem % t_size; rem/=t_size;
      int mat_j = rem % mf_1f.nvec[0]; rem/=mf_1f.nvec[0];
      int mat_i = rem;

      Float* _1f = (Float*)mf_1f.mf + i;
      Float* _2f = (Float*)mf_2f.mf + i;
      if( fabs(*_1f - *_2f) > 1e-12 ){ if(!UniqueID()) printf("Failed on mf_ll offset %d: %.14e   %.14e\n",i,*_1f,*_2f); fail = true; }      
    }
    if(fail){ printf("Failed mf_ll test\n"); exit(-1); }
    else printf("Passed mf_ll test\n");
  }

  static void compare_source_MesonField2_2f(MesonField &mf, MFsource &src){
    bool fail(false);
    for(int i=0;i<2*GJP.VolNodeSites()/GJP.TnodeSites();i++){
      Float* _orig = (Float*)mf.src + i;
      Float* _new = (Float*)src.src + i;
      if( fabs(*_orig - *_new) > 1e-12 ){ if(!UniqueID()) printf("Failed on source comparison offset %d: %.14e   %.14e\n",i,*_orig,*_new); fail = true; }
      else printf("Passed on source comparison offset %d: %.14e   %.14e\n",i,*_orig,*_new);
    }
    if(fail){ printf("Failed source comparison test\n"); exit(-1); }
    else printf("Passed source comparison test\n");
  }

  static void compare_mf_ll_MesonField2_2f(MesonField &mf_2f, MesonField2 &mf2_2f){
    int t_size = GJP.TnodeSites()*GJP.Tnodes();
    int size = mf_2f.nvec[0]*(mf_2f.nl[0]+12*mf_2f.nhits[0])*t_size*2;
    bool fail(false);
    for(int i=0;i<size;i++){
      int rem = i;
      int reim = rem % 2; rem/=2;
      int glb_t = rem % t_size; rem/=t_size;
      int mat_i = rem % mf_2f.nvec[0]; rem/=mf_2f.nvec[0];
      int mat_j = rem;

      Float* _orig = (Float*)mf_2f.mf + i;
      Float* _new = (Float*)mf2_2f.mf + i;
      if( fabs(*_orig - *_new) > 1e-12 ){ if(!UniqueID()) printf("Failed on mf_ll_MesonField2_2f offset %d (i %d j %d t %d reim %d): %.14e   %.14e\n",i,mat_i,mat_j,glb_t,reim,*_orig,*_new); fail = true; }      
      else if(!UniqueID()) printf("Passed on mf_ll_MesonField2_2f offset %d (i %d j %d t %d reim %d): %.14e   %.14e\n",i,mat_i,mat_j,glb_t,reim,*_orig,*_new);
    }
    if(fail){ printf("Failed mf_ll_MesonField2_2f test\n"); exit(-1); }
    else printf("Passed mf_ll_MesonField2_2f test\n");
  }
  static void compare_mf_sl_MesonField2_2f(MesonField &mf_2f, MesonField2 &mf2_2f){
    int t_size = GJP.TnodeSites()*GJP.Tnodes();
    int size = mf_2f.nvec[0]*(mf_2f.nl[1]+12*mf_2f.nhits[1])*t_size*2;
    bool fail(false);
    for(int i=0;i<size;i++){
      int rem = i;
      int reim = rem % 2; rem/=2;
      int glb_t = rem % t_size; rem/=t_size;
      int mat_i = rem % mf_2f.nvec[0]; rem/=mf_2f.nvec[0];
      int mat_j = rem;

      Float* _orig = (Float*)mf_2f.mf_sl + i;
      Float* _new = (Float*)mf2_2f.mf + i;
      if( fabs(*_orig - *_new) > 1e-12 ){ if(!UniqueID()) printf("Failed on mf_sl_MesonField2_2f offset %d (i %d j %d t %d reim %d): %.14e   %.14e\n",i,mat_i,mat_j,glb_t,reim,*_orig,*_new); fail = true; }      
      else if(!UniqueID()) printf("Passed on mf_sl_MesonField2_2f offset %d (i %d j %d t %d reim %d): %.14e   %.14e\n",i,mat_i,mat_j,glb_t,reim,*_orig,*_new);
    }
    if(fail){ printf("Failed mf_sl_MesonField2_2f test\n"); exit(-1); }
    else printf("Passed mf_sl_MesonField2_2f test\n");
  }
  static void compare_mf_ls_MesonField2_2f(MesonField &mf_2f, MesonField2 &mf2_2f){
    int t_size = GJP.TnodeSites()*GJP.Tnodes();
    int size = mf_2f.nvec[1]*(mf_2f.nl[0]+12*mf_2f.nhits[0])*t_size*2;
    bool fail(false);
    for(int i=0;i<size;i++){
      int rem = i;
      int reim = rem % 2; rem/=2;
      int glb_t = rem % t_size; rem/=t_size;
      int mat_i = rem % mf_2f.nvec[1]; rem/=mf_2f.nvec[1];
      int mat_j = rem;

      Float* _orig = (Float*)mf_2f.mf_ls + i;
      Float* _new = (Float*)mf2_2f.mf + i;
      if( fabs(*_orig - *_new) > 1e-12 ){ if(!UniqueID()) printf("Failed on mf_ls_MesonField2_2f offset %d (i %d j %d t %d reim %d): %.14e   %.14e\n",i,mat_i,mat_j,glb_t,reim,*_orig,*_new); fail = true; }      
      else if(!UniqueID()) printf("Passed on mf_ls_MesonField2_2f offset %d (i %d j %d t %d reim %d): %.14e   %.14e\n",i,mat_i,mat_j,glb_t,reim,*_orig,*_new);
    }
    if(fail){ printf("Failed mf_ls_MesonField2_2f test\n"); exit(-1); }
    else printf("Passed mf_ls_MesonField2_2f test\n");
  }


  static void compare_mf_ll_MesonField2_2f_1f(MesonField2 &mf2_2f, MesonField2 &mf2_1f){
    int t_size = GJP.TnodeSites()*GJP.Tnodes();
    int size = mf2_2f.nvec[0]*(mf2_2f.nl[0]+12*mf2_2f.nhits[0])*t_size*2;
    bool fail(false);
    for(int i=0;i<size;i++){
      int rem = i;
      int reim = rem % 2; rem/=2;
      int glb_t = rem % t_size; rem/=t_size;
      int mat_i = rem % mf2_2f.nvec[0]; rem/=mf2_2f.nvec[0];
      int mat_j = rem;

      Float* _orig = (Float*)mf2_2f.mf + i;
      Float* _new = (Float*)mf2_1f.mf + i;
      if( fabs(*_orig - *_new) > 1e-12 ){ if(!UniqueID()) printf("Failed on mf_ll_MesonField2_2f_1f offset %d (i %d j %d t %d reim %d): %.14e   %.14e\n",i,mat_i,mat_j,glb_t,reim,*_orig,*_new); fail = true; }      
      else if(!UniqueID()) printf("Passed on mf_ll_MesonField2_2f_1f offset %d (i %d j %d t %d reim %d): %.14e   %.14e\n",i,mat_i,mat_j,glb_t,reim,*_orig,*_new);
    }
    if(fail){ printf("Failed mf_ll_MesonField2_2f_1f test\n"); exit(-1); }
    else printf("Passed mf_ll_MesonField2_2f_1f test\n");
  }

  static void compare_kaon_corr_MesonField2_2f(std::complex<double> *kaoncorr_orig, CorrelationFunction &kaoncorr_mf2){
    //std::complex<double> *kaoncorr_orig = (std::complex<double> *)pmalloc(sizeof(std::complex<double>)*GJP.Tnodes()*GJP.TnodeSites());
    bool fail(false);

    const char* thr = kaoncorr_mf2.threadType() == CorrelationFunction::THREADED ? "threaded" : "unthreaded";

    for(int t=0;t<GJP.Tnodes()*GJP.TnodeSites();++t){
      double *_orig = (double*) &kaoncorr_orig[t];
      double *_new = (double*) &kaoncorr_mf2(0,t);
      for(int i=0;i<2;i++){	
	if( fabs(_orig[i] - _new[i]) > 1e-12 ){ if(!UniqueID()) printf("Failed on compare_kaon_corr_MesonField2_2f t = %d i = %d: %.14e   %.14e\n",t,i,_orig[i],_new[i]); fail = true; }      
      }
    }
    if(fail){ printf("Failed compare_kaon_corr_MesonField2_2f (%s) test\n",thr); exit(-1); }
    else printf("Passed compare_kaon_corr_MesonField2_2f (%s) test\n",thr);
  }



  static void wallsource_amplitude_2f(MesonField2 &mf2_2f){
    



  }
  

};
CPS_END_NAMESPACE

//(complex<double> *)mf + (j*nvec[0]+i)*t_size*n_flav2 + t_size*(2*g + f) +glb_t;
  // int mf_fac = GJP.Gparity() || GJP.Gparity1fX() ? 4 : 1; //flavour matrix indices also

  // mf = (Vector *)smalloc(cname, fname, "mf", sizeof(Float)*nvec[0]*(nl[0]+sc_size*nhits[0])*t_size*2 * mf_fac);

using namespace Chroma;
using namespace cps;

typedef LatticeFermion T;
typedef multi1d<LatticeFermion> T5;
typedef multi1d<LatticeColorMatrix> U;

int cout_time(char *);
void ReadGaugeField(const MeasArg &meas_arg);
void bfm_init(bfm_evo<double> &dwf,double mq);

int main (int argc,char **argv )
{
  Start(&argc, &argv);

#ifdef HAVE_BFM
  Chroma::initialize(&argc,&argv);
#endif

  CommandLine::is(argc,argv);

  bool gparity_X(false);
  bool gparity_Y(false);

  int arg0 = CommandLine::arg_as_int(0);
  printf("Arg0 is %d\n",arg0);
  if(arg0==0){
    gparity_X=true;
    printf("Doing G-parity HMC test in X direction\n");
  }else{
    printf("Doing G-parity HMC test in X and Y directions\n");
    gparity_X = true;
    gparity_Y = true;
  }

  bool dbl_latt_storemode(false);
  bool save_config(false);
  bool load_config(false);
  bool load_lrg(false);
  bool save_lrg(false);
  char *load_config_file;
  char *save_config_file;
  char *save_lrg_file;
  char *load_lrg_file;
  bool verbose(false);
  bool skip_gparity_inversion(false);
  bool unit_gauge(false);

  int size[] = {2,2,2,2,2};

  int i=2;
  while(i<argc){
    char* cmd = argv[i];  
    if( strncmp(cmd,"-save_config",15) == 0){
      if(i==argc-1){ printf("-save_config requires an argument\n"); exit(-1); }
      save_config=true;
      save_config_file = argv[i+1];
      i+=2;
    }else if( strncmp(cmd,"-load_config",15) == 0){
      if(i==argc-1){ printf("-save_config requires an argument\n"); exit(-1); }
      load_config=true;
      load_config_file = argv[i+1];
      i+=2;
    }else if( strncmp(cmd,"-latt",10) == 0){
      if(i>argc-6){
	printf("Did not specify enough arguments for 'latt' (require 5 dimensions)\n"); exit(-1);
      }
      size[0] = CommandLine::arg_as_int(i); //CommandLine ignores zeroth input arg (i.e. executable name)
      size[1] = CommandLine::arg_as_int(i+1);
      size[2] = CommandLine::arg_as_int(i+2);
      size[3] = CommandLine::arg_as_int(i+3);
      size[4] = CommandLine::arg_as_int(i+4);
      i+=6;
    }else if( strncmp(cmd,"-save_double_latt",20) == 0){
      dbl_latt_storemode = true;
      i++;
    }else if( strncmp(cmd,"-load_lrg",15) == 0){
      if(i==argc-1){ printf("-load_lrg requires an argument\n"); exit(-1); }
      load_lrg=true;
      load_lrg_file = argv[i+1];
      i+=2;
    }else if( strncmp(cmd,"-save_lrg",15) == 0){
      if(i==argc-1){ printf("-save_lrg requires an argument\n"); exit(-1); }
      save_lrg=true;
      save_lrg_file = argv[i+1];
      i+=2;  
    }else if( strncmp(cmd,"-verbose",15) == 0){
      verbose=true;
      i++;
    }else if( strncmp(cmd,"-skip_gparity_inversion",30) == 0){
      skip_gparity_inversion=true;
      i++;
    }else if( strncmp(cmd,"-unit_gauge",15) == 0){
      unit_gauge=true;
      i++;
    }else{
      if(UniqueID()==0) printf("Unrecognised argument: %s\n",cmd);
      exit(-1);
    }
  }
  

  printf("Lattice size is %d %d %d %d\n",size[0],size[1],size[2],size[3],size[4]);

  DoArg do_arg;
  do_arg.x_sites = size[0];
  do_arg.y_sites = size[1];
  do_arg.z_sites = size[2];
  do_arg.t_sites = size[3];
  do_arg.s_sites = size[4];
  do_arg.x_node_sites = 0;
  do_arg.y_node_sites = 0;
  do_arg.z_node_sites = 0;
  do_arg.t_node_sites = 0;
  do_arg.s_node_sites = 0;
  do_arg.x_nodes = 0;
  do_arg.y_nodes = 0;
  do_arg.z_nodes = 0;
  do_arg.t_nodes = 0;
  do_arg.s_nodes = 0;
  do_arg.updates = 0;
  do_arg.measurements = 0;
  do_arg.measurefreq = 0;
  do_arg.cg_reprod_freq = 10;
  do_arg.x_bc = BND_CND_PRD;
  do_arg.y_bc = BND_CND_PRD;
  do_arg.z_bc = BND_CND_PRD;
  do_arg.t_bc = BND_CND_APRD;
  do_arg.start_conf_kind = START_CONF_ORD;
  do_arg.start_conf_load_addr = 0x0;
  do_arg.start_seed_kind = START_SEED_FIXED;
  do_arg.start_seed_filename = "../rngs/ckpoint_rng.0";
  do_arg.start_conf_filename = "../configurations/ckpoint_lat.0";
  do_arg.start_conf_alloc_flag = 6;
  do_arg.wfm_alloc_flag = 2;
  do_arg.wfm_send_alloc_flag = 2;
  do_arg.start_seed_value = 83209;
  do_arg.beta =   2.25;
  do_arg.c_1 =   -3.3100000000000002e-01;
  do_arg.u0 =   1.0000000000000000e+00;
  do_arg.dwf_height =   1.8000000000000000e+00;
  do_arg.dwf_a5_inv =   1.0000000000000000e+00;
  do_arg.power_plaq_cutoff =   0.0000000000000000e+00;
  do_arg.power_plaq_exponent = 0;
  do_arg.power_rect_cutoff =   0.0000000000000000e+00;
  do_arg.power_rect_exponent = 0;
  do_arg.verbose_level = -1202; //VERBOSE_DEBUG_LEVEL; //-1202;
  do_arg.checksum_level = 0;
  do_arg.exec_task_list = 0;
  do_arg.xi_bare =   1.0000000000000000e+00;
  do_arg.xi_dir = 3;
  do_arg.xi_v =   1.0000000000000000e+00;
  do_arg.xi_v_xi =   1.0000000000000000e+00;
  do_arg.clover_coeff =   0.0000000000000000e+00;
  do_arg.clover_coeff_xi =   0.0000000000000000e+00;
  do_arg.xi_gfix =   1.0000000000000000e+00;
  do_arg.gfix_chkb = 1;
  do_arg.asqtad_KS =   0.0000000000000000e+00;
  do_arg.asqtad_naik =   0.0000000000000000e+00;
  do_arg.asqtad_3staple =   0.0000000000000000e+00;
  do_arg.asqtad_5staple =   0.0000000000000000e+00;
  do_arg.asqtad_7staple =   0.0000000000000000e+00;
  do_arg.asqtad_lepage =   0.0000000000000000e+00;
  do_arg.p4_KS =   0.0000000000000000e+00;
  do_arg.p4_knight =   0.0000000000000000e+00;
  do_arg.p4_3staple =   0.0000000000000000e+00;
  do_arg.p4_5staple =   0.0000000000000000e+00;
  do_arg.p4_7staple =   0.0000000000000000e+00;
  do_arg.p4_lepage =   0.0000000000000000e+00;

  if(verbose) do_arg.verbose_level = VERBOSE_DEBUG_LEVEL;

  if(gparity_X) do_arg.x_bc = BND_CND_GPARITY;
  if(gparity_Y) do_arg.y_bc = BND_CND_GPARITY;

  GJP.Initialize(do_arg);

  LRG.Initialize(); //usually initialised when lattice generated, but I pre-init here so I can load the state from file

  if(load_lrg){
    if(UniqueID()==0) printf("Loading RNG state from %s\n",load_lrg_file);
    LRG.Read(load_lrg_file,32);
  }
  if(save_lrg){
    if(UniqueID()==0) printf("Writing RNG state to %s\n",save_lrg_file);
    LRG.Write(save_lrg_file,32);
  }
  
  GwilsonFdwf* lattice = new GwilsonFdwf;
					       
  if(!load_config){
    printf("Creating gauge field\n");
    if(!unit_gauge) lattice->SetGfieldDisOrd();
    else lattice->SetGfieldOrd();
  }else{
    ReadLatticeParallel readLat;
    if(UniqueID()==0) printf("Reading: %s (NERSC-format)\n",load_config_file);
    readLat.read(*lattice,load_config_file);
    if(UniqueID()==0) printf("Config read.\n");
  }

  if(save_config){
    if(UniqueID()==0) printf("Saving config to %s\n",save_config_file);

    QioArg wt_arg(save_config_file,0.001);
    
    wt_arg.ConcurIONumber=32;
    WriteLatticeParallel wl;
    wl.setHeader("disord_id","disord_label",0);
    wl.write(*lattice,wt_arg);
    
    if(!wl.good()) ERR.General("main","()","Failed write lattice %s",save_config_file);

    if(UniqueID()==0) printf("Config written.\n");
  }

  cps_qdp_init(&argc,&argv);
  
  omp_set_num_threads(1);

  bool precon = true;
  //Backup LRG
  LatRanGen LRGbak = LRG;

  //Generate eigenvectors  
  Lanczos_5d<double>* eig_2f;
  create_eig(lattice,eig_2f,precon);

  //Generate A2A prop in 2f environment
  A2APropbfm* prop_2f;
  a2a_prop_gen(prop_2f, lattice, *eig_2f);

  //Pull the eigenvectors out of eig into CPS canonical ordered fields for later 2f->1f conversion
  Float** eigv_2f_cps;
  eig_convert_cps(*eig_2f, eigv_2f_cps,precon);

  //Generate gauge fixing matrices
  CommonArg c_arg;
  FixGaugeArg gfix_arg;
  setup_gfix_args(gfix_arg);
  AlgFixGauge fix_gauge_2f(*lattice,&c_arg,&gfix_arg);
  
  //Generate MesonField object
  MesonField mesonfield_2f(*lattice, prop_2f, &fix_gauge_2f, &c_arg);
  mesonfield_2f.allocate_vw_fftw();

  //Do the FFTW on the A2APropbfm also and test this new code
  fix_gauge_2f.run();
  prop_2f->allocate_vw_fftw();
  prop_2f->fft_vw();
  fix_gauge_2f.free(); //free gauge fixing matrices

  mesonfield_2f.prepare_vw(); //re-performs gauge fixing
  MesonFieldTesting::compare_fftw_fields(mesonfield_2f,*prop_2f);

  //Calculate mesonfield with exponential source with radius 2
  int source_type = 2;   //exp 1 box 2
  double radius = 2;

  MFBasicSource::SourceType source_type2 = MFBasicSource::BoxSource; //MFBasicSource::ExponentialSource;
  mesonfield_2f.cal_mf_ll(radius,source_type);

  //Try to duplicate mf_ll using MesonField2
  MFqdpMatrix structure_2f(MFstructure::W, MFstructure::V, true, false,15,sigma0); //gamma^5 spin, unit mat flavour
  MFBasicSource source_2f(source_type2,radius);
  
  MesonFieldTesting::compare_source_MesonField2_2f(mesonfield_2f,source_2f);  
  MesonField2 mf2_2f(*prop_2f,*prop_2f, structure_2f, source_2f);
  MesonFieldTesting::compare_mf_ll_MesonField2_2f(mesonfield_2f, mf2_2f);

  //Calculate a 'kaon' correlation function in both Daiqian's code and mine (both modified for G-parity)
  {
    AlgFixGauge fix_gauge_2f_ls(*lattice,&c_arg,&gfix_arg);
    MesonField mesonfield_2f_ls(*lattice, prop_2f, prop_2f, &fix_gauge_2f_ls, &c_arg); //pretend the propagator is also a strange quark for testing :)
    mesonfield_2f_ls.allocate_vw_fftw();
    mesonfield_2f_ls.prepare_vw();

    mesonfield_2f_ls.cal_mf_sl(radius,source_type);
    MesonFieldTesting::compare_mf_sl_MesonField2_2f(mesonfield_2f_ls,mf2_2f);

    mesonfield_2f_ls.cal_mf_ls(radius,source_type);
    MesonFieldTesting::compare_mf_ls_MesonField2_2f(mesonfield_2f_ls,mf2_2f);

    std::complex<double> *kaoncorr_orig = (std::complex<double> *)pmalloc(sizeof(std::complex<double>)*GJP.Tnodes()*GJP.TnodeSites());
    mesonfield_2f_ls.run_kaon(kaoncorr_orig);
    
    CorrelationFunction kaoncorr_mf2("",1);
    MesonField2::contract(mf2_2f, mf2_2f, kaoncorr_mf2);
    
    MesonFieldTesting::compare_kaon_corr_MesonField2_2f(kaoncorr_orig, kaoncorr_mf2);

    //Check threaded version
    CorrelationFunction kaoncorr_mf2_thr("",1, CorrelationFunction::THREADED);
    MesonField2::contract(mf2_2f, mf2_2f, kaoncorr_mf2_thr);
    kaoncorr_mf2_thr.sumThreads();

    MesonFieldTesting::compare_kaon_corr_MesonField2_2f(kaoncorr_orig, kaoncorr_mf2_thr);

  }

  //Restore LRG backup to reset RNG for 1f section
  LRG = LRGbak;

  //Move to 1f environment
  if(UniqueID()==0) printf("Starting double lattice section\n");
  
  int array_size = 2*lattice->GsiteSize() * GJP.VolNodeSites() * sizeof(Float);
  cps::Matrix *orig_lattice = (cps::Matrix *) pmalloc(array_size);
  memcpy((void*)orig_lattice, (void*)lattice->GaugeField(), array_size);

  lattice->FreeGauge(); //free memory and reset
  delete lattice; //lattice objects are singleton (scope_lock)
  
  //setup 1f model. Upon calling GJP.Initialize the lattice size will be doubled in the appropriate directions
  //and the boundary condition set to APRD
  if(gparity_X) do_arg.gparity_1f_X = 1;
  if(gparity_Y) do_arg.gparity_1f_Y = 1;

  GJP.Initialize(do_arg);

  if(GJP.Gparity()){ printf("Que?\n"); exit(-1); }
  if(UniqueID()==0) printf("Doubled lattice : %d %d %d %d\n", GJP.XnodeSites()*GJP.Xnodes(),GJP.YnodeSites()*GJP.Ynodes(),
			   GJP.ZnodeSites()*GJP.Znodes(),GJP.TnodeSites()*GJP.Tnodes());
  
#ifdef HAVE_BFM
  {
    QDP::multi1d<int> nrow(Nd);  
    for(int i = 0;i<Nd;i++) nrow[i] = GJP.Sites(i);
    QDP::Layout::setLattSize(nrow);
    QDP::Layout::create();
  }
#endif
  lattice = new GwilsonFdwf;
  setup_double_latt(*lattice,orig_lattice,gparity_X,gparity_Y);
  setup_double_rng(gparity_X,gparity_Y);
   
  //Convert eigenvectors from 2f to 1f to use here (assumes 2f and 1f lanczos agree - this just ensures they agree perfectly, such that any numerical disparities are localised to the A2A code)
  Lanczos_5d<double>* eig_1f;
  create_eig_1f(lattice, eig_1f, *eig_2f, eigv_2f_cps, precon, gparity_X, gparity_Y);

  //Create 1f A2A prop
  A2APropbfm* prop_1f;
  a2a_prop_gen(prop_1f, lattice, *eig_1f);

  //In-place convert 2f A2A prop to 1f format
  convert_2f_A2A_prop_to_1f(*prop_2f, gparity_X, gparity_Y);

  //Compare props
  compare_1f_2f_A2A_prop(*prop_2f, *prop_1f);

  //Fix the gauge
  AlgFixGauge fix_gauge_1f(*lattice,&c_arg,&gfix_arg);
  
  //Generate MesonField object
  MesonField mesonfield_1f(*lattice, prop_1f, &fix_gauge_1f, &c_arg);
  mesonfield_1f.allocate_vw_fftw();
  mesonfield_1f.prepare_vw();

  //Calculate mesonfield with exponential source with radius 2
  mesonfield_1f.cal_mf_ll(radius,source_type);

  MesonFieldTesting::convert_mesonfield_2f_1f(mesonfield_2f,gparity_X,gparity_Y);

  //Compare meson fields
  MesonFieldTesting::compare_fftw_vecs(mesonfield_1f,mesonfield_2f);
  MesonFieldTesting::compare_mf_ll(mesonfield_1f, mesonfield_2f);

  //Test MesonField2 1f code against its 2f version (its 2f version has already been compared to the original MesonField code above)
  fix_gauge_1f.run();
  prop_1f->allocate_vw_fftw();
  prop_1f->fft_vw();
  fix_gauge_1f.free();

  MFqdpMatrix structure_1f(MFstructure::W, MFstructure::V, true, false,15,sigma0); //gamma^5 spin, unit mat flavour
  MFBasicSource source_1f(source_type2,radius);
  
  MesonField2 mf2_1f(*prop_1f,*prop_1f, structure_1f, source_1f);
  MesonFieldTesting::compare_mf_ll_MesonField2_2f_1f(mf2_2f,mf2_1f);



#ifdef HAVE_BFM
  Chroma::finalize();
#endif

  if(UniqueID()==0){
    printf("Main job complete\n"); 
    fflush(stdout);
  }
  
  return 0;
}
