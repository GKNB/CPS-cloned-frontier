#define USE_GRID
#define USE_GRID_A2A
#define USE_GRID_LANCZOS
#include<chroma.h>

//bfm headers
#ifdef USE_BFM
#include<bfm.h>
#include<util/lattice/bfm_eigcg.h> // This is for the Krylov.h function "matrix_dgemm"
#include<util/lattice/bfm_evo.h>
#endif

//cps headers
#include<util/time_cps.h>
#include<alg/common_arg.h>
#include<alg/fix_gauge_arg.h>
#include<alg/do_arg.h>
#include<alg/meas_arg.h>
#include<alg/a2a_arg.h>
#include<alg/lanc_arg.h>
#include<alg/ktopipi_jobparams.h>
#include<util/qioarg.h>
#include<util/ReadLatticePar.h>
#include<alg/alg_fix_gauge.h>
#include<util/flavormatrix.h>
#include<alg/wilson_matrix.h>
#include<util/spincolorflavormatrix.h>

#if defined(USE_GRID) && !defined(DISABLE_GRID_A2A)
#include<util/lattice/fgrid.h>
#endif

#ifdef USE_MPI
//mpi headers
#warning "WARNING : USING MPI"
#include<mpi.h>
#endif

//c++ classes
#include<sys/stat.h>
#include<unistd.h>

//using namespace Chroma;
using namespace cps;

#include <alg/a2a/a2a.h>
#include <alg/a2a/mesonfield.h>

typedef double mf_Float;
typedef std::complex<mf_Float> mf_Complex;
typedef Grid::vComplexD grid_Complex;

inline int toInt(const char* a){
  std::stringstream ss; ss << a; int o; ss >> o;
  return o;
}

void setupDoArg(DoArg &do_arg, int size[5], int ngp, bool verbose = true){
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

  BndCndType* bc[3] = { &do_arg.x_bc, &do_arg.y_bc, &do_arg.z_bc };
  for(int i=0;i<ngp;i++){ 
    *(bc[i]) = BND_CND_GPARITY;
  }
}



int main(int argc,char *argv[])
{
  Start(&argc, &argv);
  int ngp;
  { std::stringstream ss; ss << argv[1]; ss >> ngp; }

  if(!UniqueID()) printf("Doing G-parity in %d directions\n",ngp);

  bool save_config(false);
  bool load_config(false);
  bool load_lrg(false);
  bool save_lrg(false);
  char *load_config_file;
  char *save_config_file;
  char *save_lrg_file;
  char *load_lrg_file;
  bool verbose(false);
  bool unit_gauge(false);

  int size[] = {2,2,2,2,2};
  int nthreads = 1;
  int ntests = 10;
  
  double tol = 1e-8;

  printf("Argc is %d\n",argc);
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
      for(int d=0;d<5;d++)
	size[d] = toInt(argv[i+1+d]);
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
    }else if( strncmp(cmd,"-verbose",15) == 0){
      verbose=true;
      i++;
    }else if( strncmp(cmd,"-nthread",15) == 0){
      nthreads = toInt(argv[i+1]);
      printf("Set nthreads to %d\n", nthreads);
      i+=2;
    }else if( strncmp(cmd,"-ntest",15) == 0){
      ntests = toInt(argv[i+1]);
      printf("Set ntests to %d\n", ntests);
      i+=2;
    }else if( strncmp(cmd,"-unit_gauge",15) == 0){
      unit_gauge=true;
      i++;
    }else if( strncmp(cmd,"-tolerance",15) == 0){
      std::stringstream ss; ss << argv[i+1];
      ss >> tol;
      if(!UniqueID()) printf("Set tolerance to %g\n",tol);
      i+=2;
    }else{
      if(UniqueID()==0) printf("Unrecognised argument: %s\n",cmd);
      exit(-1);
    }
  }

  printf("Lattice size is %d %d %d %d\n",size[0],size[1],size[2],size[3],size[4]);

  CommonArg common_arg;
  DoArg do_arg;  setupDoArg(do_arg,size,ngp,verbose);

  GJP.Initialize(do_arg);
  GJP.SetNthreads(nthreads);
  
#if TARGET == BGQ
  LRG.setSerial();
#endif
  LRG.Initialize(); //usually initialised when lattice generated, but I pre-init here so I can load the state from file

  GnoneFnone lattice;

  if(load_lrg){
    if(UniqueID()==0) printf("Loading RNG state from %s\n",load_lrg_file);
    LRG.Read(load_lrg_file,32);
  }
  if(save_lrg){
    if(UniqueID()==0) printf("Writing RNG state to %s\n",save_lrg_file);
    LRG.Write(save_lrg_file,32);
  }					       
  if(!load_config){
    printf("Creating gauge field\n");
    if(!unit_gauge) lattice.SetGfieldDisOrd();
    else lattice.SetGfieldOrd();
  }else{
    ReadLatticeParallel readLat;
    if(UniqueID()==0) printf("Reading: %s (NERSC-format)\n",load_config_file);
    readLat.read(lattice,load_config_file);
    if(UniqueID()==0) printf("Config read.\n");
  }
  if(save_config){
    if(UniqueID()==0) printf("Saving config to %s\n",save_config_file);

    QioArg wt_arg(save_config_file,0.001);
    
    wt_arg.ConcurIONumber=32;
    WriteLatticeParallel wl;
    wl.setHeader("disord_id","disord_label",0);
    wl.write(lattice,wt_arg);
    
    if(!wl.good()) ERR.General("main","()","Failed write lattice %s",save_config_file);

    if(UniqueID()==0) printf("Config written.\n");
  }

  A2AArg a2a_args;
  a2a_args.nl = 900;
  a2a_args.nhits = 1;
  a2a_args.rand_type = UONE;
  a2a_args.src_width = 1;

  if(0){
    typedef _deduce_a2a_field_policies<mf_Complex> A2Apolicies;

    A2AvectorWfftw<A2Apolicies> W(a2a_args);
    A2AvectorVfftw<A2Apolicies> V(a2a_args);
    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf;
    
    A2AexpSource<> src(2.0);
    SCFspinflavorInnerProduct<typename A2Apolicies::ComplexType,A2AexpSource<> > mf_struct(sigma3,15,src);
    
    Float total_time = 0.;
    for(int iter=0;iter<ntests;iter++){
      W.testRandom();
      V.testRandom();
      
      total_time -= dclock();
      mf.compute(W,mf_struct,V,0);
      total_time += dclock();
    }
    printf("Avg time %d iters: %g secs\n",ntests,total_time/ntests);
  }
    
  int nsimd = Grid::vComplexD::Nsimd();
  int simd_dims[4];
  FourDSIMDPolicy::SIMDdefaultLayout(simd_dims,nsimd,2); //only divide over spatial directions

  printf("Nsimd = %d, SIMD dimensions:\n", nsimd);
  for(int i=0;i<4;i++)
    printf("%d ", simd_dims[i]);
  printf("\n");

  if(0){
    CPSfermion<Grid::vComplexD,FourDSIMDPolicy,DynamicFlavorPolicy,Aligned128AllocPolicy> a(simd_dims);
    CPSfermion4D<cps::ComplexD> b;
    b.testRandom();
    
    a.importField(b);
  }

  NullObject n;
  if(0){
    CPSfield<Grid::vComplexD,1,ThreeDSIMDPolicy,OneFlavorPolicy,Aligned128AllocPolicy> a(simd_dims);
    CPSfield<cps::ComplexD,1,SpatialPolicy,OneFlavorPolicy,StandardAllocPolicy> b(n);
    b.testRandom();
    a.importField(b);

    CPSfield<cps::ComplexD,1,SpatialPolicy,OneFlavorPolicy,StandardAllocPolicy> c(n);
    c.importField(a);

    assert(b.equals(c));
    printf("Test success\n");
  }

  if(0){
    CPSglobalComplexSpatial<cps::ComplexD,OneFlavorPolicy> glb;
    glb.testRandom();

    CPSfield<Grid::vComplexD,1,ThreeDSIMDPolicy,OneFlavorPolicy,Aligned128AllocPolicy> a(simd_dims);
    CPSfield<cps::ComplexD,1,SpatialPolicy,OneFlavorPolicy,StandardAllocPolicy> b(n);

    glb.scatter<Grid::vComplexD,ThreeDSIMDPolicy,Aligned128AllocPolicy>(a);
    glb.scatter<cps::ComplexD,SpatialPolicy,StandardAllocPolicy>(b);

    CPSfield<cps::ComplexD,1,SpatialPolicy,OneFlavorPolicy,StandardAllocPolicy> c(n);
    c.importField(a);

    assert(b.equals(c));
    printf("Test2 success\n");       
  }
  
  if(0){
    A2AexpSource<StandardSourcePolicies> std_exp(2.0);
    A2AexpSource<GridSIMDSourcePolicies> grid_exp(2.0, simd_dims);

    CPSfield<cps::ComplexD,1,SpatialPolicy,OneFlavorPolicy,StandardAllocPolicy> b(n);
    b.importField(grid_exp.getSource());

    assert( b.equals(std_exp.getSource()) );
    printf("Test3 success\n");  
  }

  CPSfermion4D<cps::ComplexD> tmp;
  int ns = tmp.nodeSites(0);

  
  {
    typedef _deduce_a2a_field_policies<mf_Complex> A2Apolicies;
    typedef _deduce_a2a_field_policies<grid_Complex> GridA2Apolicies;

    typename my_enable_if< _equal<typename A2Apolicies::ScalarComplexType, typename GridA2Apolicies::ScalarComplexType>::value, int>::type dummy = 0;

    typedef typename A2Apolicies::ScalarComplexType Ctype;
    typedef typename Ctype::value_type Ftype;
	  
    A2AvectorWfftw<A2Apolicies> W(a2a_args);
    A2AvectorVfftw<A2Apolicies> V(a2a_args);
    
    A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
    A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);
    A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_grid;
    
    A2AexpSource<GridSIMDSourcePolicies> src_grid(2.0, simd_dims);
    SCFspinflavorInnerProduct<typename GridA2Apolicies::ComplexType,A2AexpSource<GridSIMDSourcePolicies> > mf_struct_grid(sigma3,15,src_grid);

    A2AexpSource<> src(2.0);
    SCFspinflavorInnerProduct<typename A2Apolicies::ComplexType,A2AexpSource<> > mf_struct(sigma3,15,src);

    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf;

    Float total_time = 0.;
    for(int iter=0;iter<ntests;iter++){
      W.testRandom();
      V.testRandom();
      Wgrid.importFields(W);
      Vgrid.importFields(V);
      
      total_time -= dclock();
      mf_grid.compute(Wgrid,mf_struct_grid,Vgrid,0);
      total_time += dclock();

      mf.compute(W,mf_struct,V,0);

      bool fail = false;
      for(int i=0;i<mf.size();i++){
	const Ctype& gd = mf_grid.ptr()[i];
	const Ctype& cp = mf.ptr()[i];
	Ftype rdiff = fabs(gd.real()-cp.real());
	Ftype idiff = fabs(gd.imag()-cp.imag());
	if(rdiff > 1e-10 || idiff > 1e-10){
	  printf("Fail: Iter %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",iter, gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
	  fail = true;
	}
      }
      if(fail) ERR.General("","","Standard vs Grid implementation test failed\n");	
    }
    printf("Avg time %d iters: %g secs\n",ntests,total_time/ntests);
  }

  
    
  //A2Asource<Grid::vComplexD,FourDSIMDPolicy,Aligned128AllocPolicy> gsrc;

  

  //OptimizedSpinColorContract<double,true,true> M;
  // {
  //   A2AexpSource<> src(2.0);
  //   A2AexpSource<GridSIMDSourcePolicies> grid_src(2.0, simd_dims);
    
  //   FlavorMatrixGeneral<Grid::vComplexD> vf;
  //   vf.zero();
    
  //   SCFspinflavorInnerProduct<cps::ComplexD,A2AexpSource<> ,true,true> ip(sigma3,0,src);
  //   SCFspinflavorInnerProduct<Grid::vComplexD,A2AexpSource<> ,true,true> ip2(sigma3,0,src);
    
  //   SCFvectorPtr<Grid::vComplexD> aa(NULL,NULL);
  //   ip2(aa,aa,0,0);
  // }
  


  
  return 0;
}
