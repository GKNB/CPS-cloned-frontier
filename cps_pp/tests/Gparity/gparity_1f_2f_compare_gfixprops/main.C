//CK: In this test we check that the 1-flavour doubled-lattice approach gives the same propagator as the 2-flavour single-lattice approach

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
#if(0==1)
#include <ReadLattice.h>
#include <WriteLattice.h>
#endif

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
#include <alg/prop_dft.h>
#include <util/gparity_singletodouble.h>

#ifdef HAVE_BFM
#include <chroma.h>
#endif

using namespace std;
USING_NAMESPACE_CPS

#define SETUP_ARRAY(OBJ,ARRAYNAME,TYPE,SIZE)		\
  OBJ . ARRAYNAME . ARRAYNAME##_len = SIZE;		\
  OBJ . ARRAYNAME . ARRAYNAME##_val = new TYPE [SIZE]

#define ELEM(OBJ,ARRAYNAME,IDX) OBJ . ARRAYNAME . ARRAYNAME##_val[IDX]

void setup_double_latt(Lattice &double_latt, Matrix* orig_gfield, bool gparity_X, bool gparity_Y){
  //orig latt ( U_0 U_1 ) ( U_2 U_3 ) ( U_4 U_5 ) ( U_6 U_7 )
  //double tatt ( U_0 U_1 U_2 U_3 ) ( U_4 U_5 U_6 U_7 ) ( U_0* U_1* U_2* U_3* ) ( U_4* U_5* U_6* U_7* )

  Matrix *dbl_gfield = double_latt.GaugeField();

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
void setup_double_gfixmat(Matrix **fix_gauge_to, Matrix**fix_gauge_from, bool gparity_X, bool gparity_Y, FixGaugeType gfix_type){
  if(gfix_type == FIX_GAUGE_LANDAU){
    SingleToDoubleMatrixField dblr(gparity_X,gparity_Y,1,fix_gauge_from[0],fix_gauge_to[0]);
    dblr.Run();
  }else if(gfix_type == FIX_GAUGE_COULOMB_T){
    int orig_3vol = GJP.XnodeSites()*GJP.YnodeSites()*GJP.ZnodeSites();
    if(gparity_X) orig_3vol/=2;
    if(gparity_Y) orig_3vol/=2;
    int orig_4vol = orig_3vol * GJP.TnodeSites();

    int array_size = 2*18 * orig_4vol * sizeof(Float);
    Matrix *tmp = (Matrix *) pmalloc(array_size);

    int dblsize = 18*GJP.VolNodeSites() * sizeof(Float);
    Matrix *tmp2 = (Matrix *) pmalloc(dblsize);

    for(int t=0;t<GJP.TnodeSites();t++){
      for(int site=0;site<orig_3vol;site++){
	for(int f=0;f<2;f++){
	  int pos_into = site + orig_3vol*t + f*orig_4vol;
	  int hplane = t + f*GJP.TnodeSites();
	  tmp[pos_into] = fix_gauge_from[hplane][site];
	}
      }
    }
    SingleToDoubleMatrixField dblr(gparity_X,gparity_Y,1,tmp,tmp2);
    dblr.Run();
    pfree(tmp);

    int now_3vol = GJP.VolNodeSites()/GJP.TnodeSites();
    for(int t=0;t<GJP.TnodeSites();t++){
      for(int site = 0; site < now_3vol;site++){
	fix_gauge_to[t][site] = tmp2[site + now_3vol*t];
      }
    }

    pfree(tmp2);
  }else{
    printf("setup_double_gfixmat only implemented for Coulomb-T and Landau gauges\n"); exit(-1);
  }

}


void GaugeTransformU(Matrix *gtrans, Lattice &lat);

bool test_equals(const Rcomplex &a, const Rcomplex &b, const double &eps){
  if( fabs(a.real()-b.real()) > eps || fabs(a.imag()-b.imag()) > eps ) return false;
  return true;
}
void print(const Rcomplex &w){
  printf("(%.4f %.4f) ",w.real(),w.imag());
}

void print(const WilsonMatrix &w){
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      Rcomplex c = w(i,0,j,0);
      printf("(%.4f %.4f) ",c.real(),c.imag());
    }
    printf("\n");
  }
  printf("\n");
}

bool test_equals(const WilsonMatrix &a, const WilsonMatrix &b, const double &eps){
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int aa=0;aa<3;aa++){
	for(int bb=0;bb<3;bb++){
	  Rcomplex ca = a(i,aa,j,bb);
	  Rcomplex cb = b(i,aa,j,bb);
	  if( fabs(ca.real()-cb.real()) > eps || fabs(ca.imag()-cb.imag()) > eps ) return false;
	}
      }
    }
  }
  return true;
}
bool test_equals(const SpinColorFlavorMatrix &a, const SpinColorFlavorMatrix &b, const double &eps){
  for(int i=0;i<2;i++)
    for(int j=0;j<2;j++)
      if(!test_equals( a(i,j), b(i,j) ,eps) ) return false;

  return true;
}
void print(const SpinColorFlavorMatrix &w){
  for(int i=0;i<2;i++){
    for(int j=0;j<2;j++){
      printf("Flav idx %d %d\n",i,j);
      print(w(i,j));      
    }
  }
}


void print(const Matrix &w){
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
      Rcomplex c = w(i,j);
      printf("(%.4f %.4f) ",c.real(),c.imag());
    }
    printf("\n");
  }
  printf("\n");
}

bool test_equals(const Matrix &a, const Matrix &b, const double &eps){
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
      const Rcomplex &ca = a(i,j);
      const Rcomplex &cb = b(i,j);
      if( fabs(ca.real()-cb.real()) > eps || fabs(ca.imag()-cb.imag()) > eps ) return false;
    }
  }
  return true;
}


int main(int argc,char *argv[])
{
  Start(&argc,&argv); //initialises QMP

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

  int gfix_type = 0; //0 = Coulomb, 1 Landau

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
    }else if( strncmp(cmd,"-gauge_fix_landau",15) == 0){
      gfix_type = 1;
      i++;   
    }else if( strncmp(cmd,"-gauge_fix_coulomb",15) == 0){
      gfix_type = 0;
      i++;   
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

  if(gfix_type == 0) lattice->FixGaugeAllocate(FIX_GAUGE_COULOMB_T);
  else lattice->FixGaugeAllocate(FIX_GAUGE_LANDAU);

  lattice->FixGauge(1e-06,2000);
  if(!UniqueID()){ printf("Gauge fixing finished\n"); fflush(stdout); }

  //Get gfixmat satisfaction
  {
    IFloat* satisfaction = lattice->GaugeFixCondSatisfaction(lattice->FixGaugePtr(), lattice->FixGaugeKind(), 1e-06);
    if(gfix_type == 0) for(int t=0;t<GJP.TnodeSites();t++) printf("t=%d GFix condition satisfaction %f\n",t,satisfaction[t]);
    else printf("GFix condition satisfaction %f\n",satisfaction[0]);
  }

  //Check new source gauge fixing code
  {
    int c(0),s(0),t(0), flav(0);
    FermionVectorTp f1;
    FermionVectorTp f2;
    f1.ZeroSource();    
    f2.ZeroSource();

    if(gfix_type == 0){
      //Coulomb
      f1.SetWallSource(c, s, t, flav);
      f1.GaugeFixVector(*lattice,s);

      f2.SetWallSource(c, s, t, flav);
      f2.GFWallSource(*lattice, s, 3, t,flav);
    }else{
      //Landau
      f1.SetVolSource(c, s, flav); //use a zero momentum volume source
      f1.GaugeFixVector(*lattice,s);
      
      int p[4] = {0,0,0,0};
      f2.SetLandauGaugeMomentaSource(*lattice, c,s, p, flav);
    }
    bool fail(false);
    for(int f=0; f< 2; f++){
      for(int site = 0; site < GJP.VolNodeSites(); site++){
	for(int spn = 0; spn < 4; spn++){
	  for(int clr = 0; clr < 3 ; clr++){
	    int off = 2*(clr + 3*(spn + 4*(site + GJP.VolNodeSites()*f)));
	    Float *f1s = f1.data()+off;
	    Float *f2s = f2.data()+off;
	    if(fabs(f1s[0]-f2s[0])>1e-12 || fabs(f1s[1]-f2s[1])>1e-12){
	      printf("Err %d %d %d %d: (%f,%f) (%f,%f)\n",clr,spn,site,f,f1s[0],f1s[1],f2s[0],f2s[1]);
	      fail = true;
	    }
	  }
	}
      }
    }
    if(fail){
      printf("Source gauge fixing code check failed\n");
      exit(-1);
    }else printf("Source gauge fixing code check passed\n");
  }

  std::vector<SpinColorFlavorMatrix> FTprop;
  std::vector<SpinColorFlavorMatrix> FTprop_nogfix;

  IFloat gparity_prop_f0_norm;
  IFloat gparity_prop_f1_norm;
  {
    PropManager::clear();
    
    JobPropagatorArgs prop_args2;
    SETUP_ARRAY(prop_args2,props,PropagatorArg,2);
  
    char* names[2] = {"prop_f0","prop_f1"};
    BndCndType bndcnd[2] = {BND_CND_APRD,BND_CND_APRD};
    char* otherfprop[2] = {"prop_f1","prop_f0"};
    Float masses[2] = {0.1,0.1};
    int flav[2] = {0,1};

    for(int i=0;i<2;i++){
      PropagatorArg &parg = prop_args2.props.props_val[i];
      
      parg.generics.type = QPROPW_TYPE;
      parg.generics.tag = names[i];
      parg.generics.mass = masses[i];
      parg.generics.bc[0] = GJP.Xbc();
      parg.generics.bc[1] = GJP.Ybc();
      parg.generics.bc[2] = BND_CND_TWISTED; //GJP.Zbc();
      parg.generics.bc[3] = bndcnd[i];

      SETUP_ARRAY(parg,attributes,AttributeContainer,7);
    
      ELEM(parg,attributes,0).type = VOLUME_SOURCE_ATTR;
// WALL_SOURCE_ATTR;
      // WallSourceAttrArg &warg = ELEM(parg,attributes,0).AttributeContainer_u.wall_source_attr;
      // warg.t = 0; //

      ELEM(parg,attributes,1).type = GPARITY_FLAVOR_ATTR;
      GparityFlavorAttrArg &gparg = ELEM(parg,attributes,1).AttributeContainer_u.gparity_flavor_attr;
      gparg.flavor = flav[i];

      ELEM(parg,attributes,2).type = CG_ATTR;
      CGAttrArg &cgattr = ELEM(parg,attributes,2).AttributeContainer_u.cg_attr;
      cgattr.max_num_iter = 5000;
      cgattr.stop_rsd = 1e-08;
      cgattr.true_rsd = 1e-08;

      ELEM(parg,attributes,3).type = GPARITY_OTHER_FLAV_PROP_ATTR;
      GparityOtherFlavPropAttrArg &otherfarg = ELEM(parg,attributes,3).AttributeContainer_u.gparity_other_flav_prop_attr;
      otherfarg.tag = otherfprop[i];

      ELEM(parg,attributes,4).type = MOMENTUM_ATTR;
      MomentumAttrArg &momarg = ELEM(parg,attributes,4).AttributeContainer_u.momentum_attr;
      for(int ii=0;ii<3;ii++)
      	if(GJP.Bc(ii)==BND_CND_GPARITY) momarg.p[ii] = 1;
      	else momarg.p[ii] = 0;

      ELEM(parg,attributes,5).type = GAUGE_FIX_ATTR;
      GaugeFixAttrArg &gfarg = ELEM(parg,attributes,5).AttributeContainer_u.gauge_fix_attr;
      gfarg.gauge_fix_src = 1;
      gfarg.gauge_fix_snk = 0;

      ELEM(parg,attributes,6).type = TWISTED_BC_ATTR;
      TwistedBcAttrArg &tbcarg = ELEM(parg,attributes,6).AttributeContainer_u.twisted_bc_attr;
      tbcarg.theta[0] = 0.0; tbcarg.theta[1] = 0.0; tbcarg.theta[2] = 0.3; //units of pi
    }
    if(UniqueID()==0) printf("prop_args contains %d propagators\n", prop_args2.props.props_len);

    PropManager::setup(prop_args2);   
    PropManager::calcProps(*lattice);
    printf("2f props inverted\n"); fflush(stdout);
    
    gparity_prop_f0_norm = PropManager::getProp(prop_args2.props.props_val[0].generics.tag).convert<QPropWcontainer>().getProp(*lattice).norm();
    gparity_prop_f1_norm = PropManager::getProp(prop_args2.props.props_val[1].generics.tag).convert<QPropWcontainer>().getProp(*lattice).norm();
  }

  //take a copy of the original gauge fixing matrices
  Matrix **gfix_copy;
  if(gfix_type == 0){
    gfix_copy = new Matrix*[GJP.TnodeSites()*2];
    for(int t=0;t<2*GJP.TnodeSites();t++){
      gfix_copy[t] = new Matrix[GJP.XnodeSites()*GJP.YnodeSites()*GJP.ZnodeSites()];
      for(int s=0;s<GJP.XnodeSites()*GJP.YnodeSites()*GJP.ZnodeSites();s++){
	gfix_copy[t][s] = lattice->FixGaugePtr()[t][s];
      }
    }
  }else{
    gfix_copy = new Matrix*[1];
    gfix_copy[0] = new Matrix[2*GJP.VolNodeSites()];
    for(int s=0;s<2*GJP.VolNodeSites();s++){
      gfix_copy[0][s] = lattice->FixGaugePtr()[0][s];
    }
  }

  lattice->FixGaugeFree();
  PropManager::clear();

  if(UniqueID()==0) printf("Starting double lattice inversion\n");

  
  int array_size = 2*lattice->GsiteSize() * GJP.VolNodeSites() * sizeof(Float);
  Matrix *orig_lattice = (Matrix *) pmalloc(array_size);
  memcpy((void*)orig_lattice, (void*)lattice->GaugeField(), array_size);

  lattice->FreeGauge(); //free memory and reset
  delete lattice; //lattice objects are singleton (scope_lock)
  
  //setup 1f model. Upon calling GJP.Initialize the lattice size will be doubled in the appropriate directions
  //and the boundary condition set to APRD
  if(gparity_X){ do_arg.x_sites *= 2; do_arg.x_bc = BND_CND_APRD; }
  if(gparity_Y){ do_arg.y_sites *= 2; do_arg.y_bc = BND_CND_APRD; }

  GJP.Initialize(do_arg);

  if(gparity_X) GJP.setGparity1fX();
  if(gparity_Y) GJP.setGparity1fY();


  if(GJP.Gparity()){ printf("Que?\n"); exit(-1); }
  if(UniqueID()==0) printf("Doubled lattice : %d %d %d %d\n", GJP.XnodeSites()*GJP.Xnodes(),GJP.YnodeSites()*GJP.Ynodes(),
			   GJP.ZnodeSites()*GJP.Znodes(),GJP.TnodeSites()*GJP.Tnodes());
  
#ifdef HAVE_BFM
  {
    QDP::multi1d<int> nrow(Nd);  
    for(int i = 0;i<Nd;i++) nrow[i] = GJP.Sites(i);
    //  multi1d<LatticeFermion> test(Nd);  
    //  nrow=size;
    QDP::Layout::setLattSize(nrow);
    QDP::Layout::create();
  }
#endif

  GwilsonFdwf doubled_lattice;
  setup_double_latt(doubled_lattice,orig_lattice,gparity_X,gparity_Y);
  setup_double_rng(gparity_X,gparity_Y);
 
  GJP.EnableGparity1f2fComparisonCode();

  if(gfix_type == 0) doubled_lattice.FixGaugeAllocate(FIX_GAUGE_COULOMB_T);
  else doubled_lattice.FixGaugeAllocate(FIX_GAUGE_LANDAU);
   
  doubled_lattice.FixGauge(1e-06,2000);
  if(!UniqueID()){ printf("Gauge fixing finished\n"); fflush(stdout); }
   
  //Get gfixmat satisfaction
  {
    IFloat* satisfaction = doubled_lattice.GaugeFixCondSatisfaction(doubled_lattice.FixGaugePtr(), doubled_lattice.FixGaugeKind(), 1e-06);
    if(gfix_type == 0) for(int t=0;t<GJP.TnodeSites();t++) printf("t=%d GFix condition satisfaction %f\n",t,satisfaction[t]);
    else printf("GFix condition satisfaction %f\n",satisfaction[0]);
  }


  //Compare new gauge fixing matrices to old
  {
    bool allsame(true);
    Matrix **gfix_copy_dbl;
    if(gfix_type == 0){
      gfix_copy_dbl = new Matrix*[GJP.TnodeSites()];
      for(int t=0;t<GJP.TnodeSites();t++){
	gfix_copy_dbl[t] = new Matrix[GJP.XnodeSites()*GJP.YnodeSites()*GJP.ZnodeSites()];
      }
    }else{
      gfix_copy_dbl = new Matrix*[1];
      gfix_copy_dbl[0] = new Matrix[GJP.VolNodeSites()];
    }
    printf("Setting up doubled gfix matrices\n"); fflush(stdout);
    setup_double_gfixmat(gfix_copy_dbl, gfix_copy, gparity_X, gparity_Y, doubled_lattice.FixGaugeKind());
    printf("Finished doubling gfix matrices\n"); fflush(stdout);

    if(gfix_type == 1 && gparity_X && !gparity_Y && GJP.Xnodes()==1){ //test first and second half are complex conjugates of each other
      bool fail(false);
      for(int t=0;t<GJP.TnodeSites();t++){
	for(int z=0;z<GJP.ZnodeSites();z++){
	  for(int y=0;y<GJP.YnodeSites();y++){
	    for(int x=GJP.XnodeSites()/2;x<GJP.XnodeSites();x++){
	      int off_hf2 = x+GJP.XnodeSites()*(y+GJP.YnodeSites()*(z+GJP.ZnodeSites()*t));
	      int off_hf1 = off_hf2 - GJP.XnodeSites()/2;
	      Matrix tmp; tmp.Conj(gfix_copy_dbl[0][off_hf1]);
	      Matrix &tmp2 = gfix_copy_dbl[0][off_hf2];
	      if(!test_equals(tmp,tmp2,1e-12)){
		printf("First half second half err on doubled gfix mat at %d %d %d %d\n",x,y,z,t);
		print(tmp);
		print(tmp2);
		fail = true;
	      }
	    }
	  }
	}
      }
      if(fail){ printf("Failed First half second half test on doubled gfix mat\n"); exit(-1); }
      else printf("Passed First half second half test on doubled gfix mat\n"); 
    }


    if(gfix_type == 0){
      for(int t=0;t<GJP.TnodeSites();t++){
	for(int site=0;site<GJP.VolNodeSites()/GJP.TnodeSites();site++){
	  if(!test_equals(gfix_copy_dbl[t][site], doubled_lattice.FixGaugePtr()[t][site],1e-08)){
	    int x[3];
	    int rem = site; 
	    for(int dd=0;dd<3;dd++){ x[dd] = rem % GJP.NodeSites(dd); rem /= GJP.NodeSites(dd); }

	    printf("Gfix mat differ t=%d site=(%d,%d,%d):\n",t,x[0],x[1],x[2]);
	    print(doubled_lattice.FixGaugePtr()[t][site]);
	    print(gfix_copy_dbl[t][site]);
	    allsame=false;
	  }
	}
      }
    }else{ //Landau
      for(int site=0;site<GJP.VolNodeSites();site++){
	if(!test_equals(gfix_copy_dbl[0][site], doubled_lattice.FixGaugePtr()[0][site],1e-08)){
	  int x[4];
	  int rem = site; 
	  for(int dd=0;dd<4;dd++){ x[dd] = rem % GJP.NodeSites(dd); rem /= GJP.NodeSites(dd); }

	  printf("Gfix mat differ site=(%d,%d,%d,%d):\n",x[0],x[1],x[2],x[3]);
	  print(doubled_lattice.FixGaugePtr()[0][site]);
	  print(gfix_copy_dbl[0][site]);
	  allsame=false;
	}
      }
    }
    if(!allsame){ 
      printf("Gauge fixing matrices on 1f lattice differ from those on 2f: different Gribov copies?\n"); 
      printf("Checking gauge condition satisfaction\n");
      bool isgribov(true);
      IFloat* satisfaction = doubled_lattice.GaugeFixCondSatisfaction(gfix_copy_dbl, doubled_lattice.FixGaugeKind(), 1e-06);
      if(gfix_type == 0) for(int t=0;t<GJP.TnodeSites();t++){
	  printf("t=%d GFix condition satisfaction %f\n",t,satisfaction[t]);
	  if(fabs(satisfaction[t])>1) isgribov=false;
	}
      else{
	printf("GFix condition satisfaction %f\n",satisfaction[0]);
	if(fabs(satisfaction[0])>1) isgribov=false;
      }
      if(!isgribov){
	printf("Original gauge fixing matrices do not satisfy gauge fixing condition on doubled lattice: something is wrong\n");
	exit(-1);
      }else{
	printf("Original gauge fixing matrices satisfy gauge fixing condition on doubled lattice\n");
      }

    }
    else printf("Gauge fixing matrices on 1f lattice are equal to those on 2f\n");
  }

  {
    PropManager::clear();
    
    JobPropagatorArgs prop_args2;
    SETUP_ARRAY(prop_args2,props,PropagatorArg,2);
  
    char* names[] = {"prop_f0","prop_f1"};
    BndCndType bndcnd[] = {BND_CND_APRD,BND_CND_APRD};
    int psign[] = {1,1};
    Float masses[] = {0.1,0.1};
    int flav[] = {0,1};

    for(int i=0;i<2;i++){
      PropagatorArg &parg = prop_args2.props.props_val[i];
    
      parg.generics.type = QPROPW_TYPE;
      parg.generics.tag = names[i];
      parg.generics.mass = masses[i];
      parg.generics.bc[0] = GJP.Xbc();
      parg.generics.bc[1] = GJP.Ybc();
      parg.generics.bc[2] = BND_CND_TWISTED; //GJP.Zbc();
      parg.generics.bc[3] = bndcnd[i];

      SETUP_ARRAY(parg,attributes,AttributeContainer,6);
    
      ELEM(parg,attributes,0).type = VOLUME_SOURCE_ATTR;
	// WALL_SOURCE_ATTR;
      // WallSourceAttrArg &warg = ELEM(parg,attributes,0).AttributeContainer_u.wall_source_attr;
      // warg.t = 0;

      ELEM(parg,attributes,1).type = GPARITY_FLAVOR_ATTR;
      GparityFlavorAttrArg &gparg = ELEM(parg,attributes,1).AttributeContainer_u.gparity_flavor_attr;
      gparg.flavor = flav[i];

      ELEM(parg,attributes,2).type = CG_ATTR;
      CGAttrArg &cgattr = ELEM(parg,attributes,2).AttributeContainer_u.cg_attr;
      cgattr.max_num_iter = 5000;
      cgattr.stop_rsd = 1e-08;
      cgattr.true_rsd = 1e-08;

      ELEM(parg,attributes,3).type = MOMENTUM_ATTR;
      MomentumAttrArg &momarg = ELEM(parg,attributes,3).AttributeContainer_u.momentum_attr;
      for(int ii=0;ii<3;ii++)
      	if(GJP.Bc(ii)==BND_CND_APRD) momarg.p[ii] = 1;
      	else momarg.p[ii] = 0;

      ELEM(parg,attributes,4).type = GAUGE_FIX_ATTR;
      GaugeFixAttrArg &gfarg = ELEM(parg,attributes,4).AttributeContainer_u.gauge_fix_attr;
      gfarg.gauge_fix_src = 1;
      gfarg.gauge_fix_snk = 0;

      ELEM(parg,attributes,5).type = TWISTED_BC_ATTR;
      TwistedBcAttrArg &tbcarg = ELEM(parg,attributes,5).AttributeContainer_u.twisted_bc_attr;
      tbcarg.theta[0] = 0.0; tbcarg.theta[1] = 0.0; tbcarg.theta[2] = 0.3; //units of pi
    }
    if(UniqueID()==0) printf("prop_args contains %d propagators\n", prop_args2.props.props_len);

    PropManager::setup(prop_args2);   
    PropManager::calcProps(doubled_lattice);
    printf("1f props inverted\n"); fflush(stdout);

    IFloat dbl_prop_f0_norm = PropManager::getProp(prop_args2.props.props_val[0].generics.tag).convert<QPropWcontainer>().getProp(doubled_lattice).norm();
    IFloat dbl_prop_f1_norm = PropManager::getProp(prop_args2.props.props_val[1].generics.tag).convert<QPropWcontainer>().getProp(doubled_lattice).norm();
    if(gparity_X && gparity_Y){
      dbl_prop_f0_norm/=2; //quad volume but only 2 independent flavors
      dbl_prop_f1_norm/=2;
    }

    printf("Prop norms: G-parity f0 = %f f1 = %f, double latt f0 = %f f1 = %f  Diffs f0 = %f  f1 = %f\n", gparity_prop_f0_norm, gparity_prop_f1_norm,dbl_prop_f0_norm,dbl_prop_f1_norm, gparity_prop_f0_norm-dbl_prop_f0_norm, gparity_prop_f1_norm-dbl_prop_f1_norm );
  }

  doubled_lattice.FixGaugeFree();

#ifdef HAVE_BFM
  Chroma::finalize();
#endif

  if(UniqueID()==0){
    printf("Main job complete\n"); 
    fflush(stdout);
  }
  End();

  return 0;
}




void GaugeTransformU(Matrix *gtrans, Lattice &lat){
  Matrix recv_buf;
  Matrix tmp;
  //apply the gauge transformation to U
  int nflav = 1;
  if(GJP.Gparity()) nflav = 2;

  for(int flav=0;flav<nflav;flav++){
    for(int t=0;t<GJP.TnodeSites();t++){
      for(int z=0;z<GJP.ZnodeSites();z++){
	for(int y=0;y<GJP.YnodeSites();y++){
	  for(int x=0;x<GJP.XnodeSites();x++){
	    int pos[4] = {x,y,z,t};
	    int v_x_off = x + GJP.XnodeSites()*(y+GJP.YnodeSites()*(z+GJP.ZnodeSites()*t)) + flav*GJP.VolNodeSites();
	    Matrix &v_x = *(gtrans + v_x_off);

	    for(int mu=0;mu<4;mu++){
	      int u_x_off = lat.GsiteOffset(pos) + mu + flav*4*GJP.VolNodeSites();
	      Matrix &u_x = *(lat.GaugeField() + u_x_off);

	      //get V_x+mu
	      int posp[4] = {x,y,z,t};
	      posp[mu] = (posp[mu]+1)%GJP.NodeSites(mu);

	      Matrix *v_xpmu_ptr = gtrans + posp[0] + GJP.XnodeSites()*(posp[1]+GJP.YnodeSites()*(posp[2]+GJP.ZnodeSites()*posp[3])) + flav*GJP.VolNodeSites();
	      if(pos[mu] == GJP.NodeSites(mu)-1){
		//if node is on the left wall, send the opposite flavour 
		if(GJP.Bc(mu) == BND_CND_GPARITY && GJP.NodeCoor(mu) == 0){
		  if(flav == 1)
		    v_xpmu_ptr-= GJP.VolNodeSites();
		  else
		    v_xpmu_ptr+= GJP.VolNodeSites();		  
		}

		//doesnt need to be fast!
		getPlusData((double *)&recv_buf, (double *)v_xpmu_ptr, 18, mu);
		v_xpmu_ptr = &recv_buf; 
	      }

	      //dagger/transpose it
	      Matrix vdag_xpmu;
	      vdag_xpmu.Dagger(*v_xpmu_ptr);

	      //gauge transform link
	      tmp.DotMEqual(v_x,u_x);
	      u_x.DotMEqual(tmp,vdag_xpmu);
	    }
	  }
	}
      }
    }

  }

}

