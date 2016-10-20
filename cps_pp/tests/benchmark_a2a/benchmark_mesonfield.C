#define USE_GRID
#define USE_GRID_A2A
#define USE_GRID_LANCZOS
#include<chroma.h>

#ifdef USE_CALLGRIND
#include<valgrind/callgrind.h>
#else
#define CALLGRIND_START_INSTRUMENTATION ;
#define CALLGRIND_STOP_INSTRUMENTATION ;
#define CALLGRIND_TOGGLE_COLLECT ;
#endif

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

#include <alg/a2a/template_wizardry.h>
#include <alg/a2a/spin_color_matrices.h>
#include <alg/a2a/a2a.h>
#include <alg/a2a/mesonfield.h>
#include <alg/a2a/compute_ktopipi_base.h>

#include "benchmark_mesonfield.h"
 typedef double mf_Float;
 typedef Grid::vComplexD grid_Complex;
 typedef GridSIMDSourcePolicies GridSrcPolicy;

//typedef float mf_Float;
//typedef Grid::vComplexF grid_Complex;
//typedef GridSIMDSourcePoliciesSingle GridSrcPolicy;


typedef std::complex<mf_Float> mf_Complex;


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
  int nlowmodes = 100;
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
    }else if( strncmp(cmd,"-nl",15) == 0){
      std::stringstream ss; ss << argv[i+1];
      ss >> nlowmodes;
      if(!UniqueID()) printf("Set nl to %d\n",nlowmodes);
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

  int nscf = 10;
  int nv = 10;


  A2AArg a2a_args;
  a2a_args.nl = nlowmodes;
  a2a_args.nhits = 1;
  a2a_args.rand_type = UONE;
  a2a_args.src_width = 1;

  if(1) testCyclicPermute();

  if(0) testGenericFFT();
  
  if(0) demonstrateFFTreln<mf_Complex>(a2a_args);


  if(0) testA2AvectorFFTrelnGparity<mf_Complex>(a2a_args,lattice);
  if(1) testA2AvectorFFTrelnGparity<Grid::vComplexD>(a2a_args,lattice);

  
  if(0){
    Grid::vComplexF v = randomvType<Grid::vComplexF>();
    printvType(v);

    
    
    
    Grid::vComplexF::conv_t* conv = reinterpret_cast<Grid::vComplexF::conv_t*>(&v.v);
    conv->s[0] = 0.0; conv->s[1] = 1.0; conv->s[2] = 2.0; conv->s[3] = 3.0;

    printvType(v);

  }






  
  if(0){ //benchmark single timeslice mf contraction
    typedef _deduce_a2a_field_policies<mf_Complex> A2Apolicies;

    A2AvectorWfftw<A2Apolicies> W(a2a_args);
    A2AvectorVfftw<A2Apolicies> V(a2a_args);
    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf;
    
    A2AexpSource<> src(2.0);
    SCFspinflavorInnerProduct<15,typename A2Apolicies::ComplexType,A2AexpSource<> > mf_struct(sigma3,src);
    
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
    
  int nsimd = grid_Complex::Nsimd();
  
  FourDSIMDPolicy::ParamType simd_dims;
  FourDSIMDPolicy::SIMDdefaultLayout(simd_dims,nsimd,2); //only divide over spatial directions

  ThreeDSIMDPolicy::ParamType simd_dims_3d;
  ThreeDSIMDPolicy::SIMDdefaultLayout(simd_dims_3d,nsimd);

  
  printf("Nsimd = %d, SIMD dimensions:\n", nsimd);
  for(int i=0;i<4;i++)
    printf("%d ", simd_dims[i]);
  printf("\n");

  if(0) benchmarkTrace(ntests,tol);
  if(0) benchmarkSpinFlavorTrace(ntests,tol);
  if(0) benchmarkTraceProd(ntests,tol);
  if(0) benchmarkColorTranspose(ntests,tol);
  if(0) benchmarkmultGammaLeft(ntests, tol);
  
  NullObject n;
  if(0){
    CPSfield<grid_Complex,1,ThreeDSIMDPolicy,OneFlavorPolicy,Aligned128AllocPolicy> a(simd_dims_3d);
    CPSfield<mf_Complex,1,SpatialPolicy,OneFlavorPolicy,StandardAllocPolicy> b(n);
    b.testRandom();
    a.importField(b);

    CPSfield<mf_Complex,1,SpatialPolicy,OneFlavorPolicy,StandardAllocPolicy> c(n);
    c.importField(a);

    assert(b.equals(c));
    printf("Test success\n");
  }

  if(0){
    CPSglobalComplexSpatial<mf_Complex,OneFlavorPolicy> glb;
    glb.testRandom();

    CPSfield<grid_Complex,1,ThreeDSIMDPolicy,OneFlavorPolicy,Aligned128AllocPolicy> a(simd_dims_3d);
    CPSfield<mf_Complex,1,SpatialPolicy,OneFlavorPolicy,StandardAllocPolicy> b(n);

    glb.scatter<grid_Complex,ThreeDSIMDPolicy,Aligned128AllocPolicy>(a);
    glb.scatter<mf_Complex,SpatialPolicy,StandardAllocPolicy>(b);

    CPSfield<mf_Complex,1,SpatialPolicy,OneFlavorPolicy,StandardAllocPolicy> c(n);
    c.importField(a);

    assert(b.equals(c));
    printf("Test2 success\n");       
  }
  
  if(0){
    typedef _deduce_a2a_field_policies<mf_Complex> A2Apolicies;
    typedef _deduce_a2a_field_policies<grid_Complex> GridA2Apolicies;
    
    A2AexpSource<typename A2Apolicies::SourcePolicies> std_exp(2.0);
    A2AexpSource<typename GridA2Apolicies::SourcePolicies> grid_exp(2.0, simd_dims_3d);

    CPSfield<typename A2Apolicies::SourcePolicies::ComplexType,1,typename A2Apolicies::SourcePolicies::DimensionPolicy,OneFlavorPolicy,typename A2Apolicies::SourcePolicies::AllocPolicy> b(n);
    b.importField(grid_exp.getSource());

    assert( b.equals(std_exp.getSource(),tol));
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
    
    //A2AexpSource<GridSrcPolicy> src_grid(2.0, simd_dims_3d);
    //SCFspinflavorInnerProduct<typename GridA2Apolicies::ComplexType,A2AexpSource<GridSrcPolicy> > mf_struct_grid(sigma3,15,src_grid);
    int p[3] = {1,1,1};
    A2AflavorProjectedExpSource<GridSrcPolicy> src_grid(2.0,p,simd_dims_3d);
    SCFspinflavorInnerProduct<15,typename GridA2Apolicies::ComplexType,A2AflavorProjectedExpSource<GridSrcPolicy> > mf_struct_grid(sigma3,src_grid);
      

    //A2AexpSource<> src(2.0);
    //SCFspinflavorInnerProduct<typename A2Apolicies::ComplexType,A2AexpSource<> > mf_struct(sigma3,15,src);
    A2AflavorProjectedExpSource<> src(2.0,p);
    SCFspinflavorInnerProduct<15,typename A2Apolicies::ComplexType,A2AflavorProjectedExpSource<> > mf_struct(sigma3,src);
    
    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf;

    if(0){
      // GridVectorizedSpinColorContract benchmark
      typedef typename GridA2Apolicies::ComplexType GVtype;
      typedef typename GridA2Apolicies::ScalarComplexType GCtype;
      const int nsimd = GVtype::Nsimd();      

      NullObject n;
      CPSfield<GCtype,12,FourDpolicy,OneFlavorPolicy> a(n); a.testRandom();
      CPSfield<GCtype,12,FourDpolicy,OneFlavorPolicy> b(n); b.testRandom();
      CPSfield<GVtype,12,FourDSIMDPolicy,OneFlavorPolicy,Aligned128AllocPolicy> aa(simd_dims); aa.importField(a);
      CPSfield<GVtype,12,FourDSIMDPolicy,OneFlavorPolicy,Aligned128AllocPolicy> bb(simd_dims); bb.importField(b);
      CPSfield<GVtype,1,FourDSIMDPolicy,OneFlavorPolicy,Aligned128AllocPolicy> cc(simd_dims);

      int ntests_scaled = 5000;
      printf("Max threads %d\n",omp_get_max_threads());
      double t0 = Grid::usecond();

#pragma omp parallel //avoid thread creation overheads
      {
	int me = omp_get_thread_num();
	int work, off;
	thread_work(work, off, aa.nfsites(), me, omp_get_num_threads());

	for(int test=0;test<ntests_scaled;test++){
	  for(int i=off;i<off+work;i++){
	    GVtype *ai = aa.fsite_ptr(i);
	    GVtype *bi = bb.fsite_ptr(i);
	    GVtype *ci = cc.fsite_ptr(i);
	    *ci = GridVectorizedSpinColorContract<GVtype,true,false>::g5(ai,bi);
	  }
	}
      }
      
      double t1 = Grid::usecond();
      double dt = t1 - t0;
      
      int FLOPs = 12*6*nsimd //12 vectorized conj(a)*b
	+ 12*2*nsimd; //12 vectorized += or -=
      double total_FLOPs = double(FLOPs) * double(aa.nfsites()) * double(ntests_scaled);
      
      double flops = total_FLOPs/dt; //dt in us   dt/(1e-6 s) in Mflops
      std::cout << "GridVectorizedSpinColorContract( conj(a)*b ): New code " << ntests_scaled << " tests over " << nthreads << " threads: Time " << dt << " usecs  flops " << flops/1e3 << " Gflops\n";
      
      //printf("GridVectorizedSpinColorContract( conj(a)*b ): New code %d tests over %d threads: Time %g secs  flops %g\n",ntests,nthreads,dt,flops);
    }

    if(0){ //All-time mesonfield contract
      std::cout << "Starting all-time mesonfield contract benchmark\n";
      Float total_time = 0.;
      std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_grid_t;

      W.testRandom();
      V.testRandom();
      Wgrid.importFields(W);
      Vgrid.importFields(V);
      
      CALLGRIND_START_INSTRUMENTATION ;
      CALLGRIND_TOGGLE_COLLECT ;
      
      for(int iter=0;iter<ntests;iter++){
	total_time -= dclock();
	A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_grid_t,Wgrid,mf_struct_grid,Vgrid);
	total_time += dclock();
      }

      CALLGRIND_TOGGLE_COLLECT ;
      CALLGRIND_STOP_INSTRUMENTATION ;

      int g5_FLOPs = 12*6*nsimd + 12*2*nsimd;//4 flav * 12 vectorized conj(a)*b  + 12 vectorized += or -=         
      int siteFmat_FLOPs = 3*nsimd;  //1 vectorized z.im*-1, 1 vectorized -1*z                                                                                                                             
      int s3_FLOPs = 4*nsimd; //2 vectorized -1*z                                                                                                                                                          
      int TransLeftTrace_FLOPs = nsimd*4*6 + nsimd*3*2; //4 vcmul + 3vcadd                                                                                                                                 
      int reduce_FLOPs = (nsimd - 1)*2; //nsimd-1 cadd                                                                                                                                                    

      double FLOPs_per_site = 0.;
      for(int t=GJP.TnodeCoor()*GJP.TnodeSites(); t<(GJP.TnodeCoor()+1)*GJP.TnodeSites(); t++){
        const int nl_l = mf_grid_t[t].getRowParams().getNl();
        const int nl_r = mf_grid_t[t].getColParams().getNl();

        int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();

        for(int i = 0; i < mf_grid_t[t].getNrows(); i++){
          modeIndexSet i_high_unmapped; if(i>=nl_l) mf_grid_t[t].getRowParams().indexUnmap(i-nl_l,i_high_unmapped);
          SCFvectorPtr<typename GridA2Apolicies::FermionFieldType::FieldSiteType> lscf = Wgrid.getFlavorDilutedVect(i,i_high_unmapped,0,t_lcl); //dilute flavor in-place if it hasn't been already \
                                                                                                                                                                                                           
          for(int j = 0; j < mf_grid_t[t].getNcols(); j++) {
            modeIndexSet j_high_unmapped; if(j>=nl_r) mf_grid_t[t].getColParams().indexUnmap(j-nl_r,j_high_unmapped);
            SCFvectorPtr<typename GridA2Apolicies::FermionFieldType::FieldSiteType> rscf = Vgrid.getFlavorDilutedVect(j,j_high_unmapped,0,t_lcl);

            for(int a=0;a<2;a++)
              for(int b=0;b<2;b++)
                if(!lscf.isZero(a) && !rscf.isZero(b))
                  FLOPs_per_site += g5_FLOPs;
            FLOPs_per_site += siteFmat_FLOPs + s3_FLOPs + TransLeftTrace_FLOPs + reduce_FLOPs;
          }
        }
      }
      const typename GridA2Apolicies::FermionFieldType &mode0 = Wgrid.getMode(0);
      const int size_3d = mode0.nodeSites(0)*mode0.nodeSites(1)*mode0.nodeSites(2);
      double total_FLOPs = double(FLOPs_per_site) * double(size_3d) * double(ntests);

      printf("MF contract all t: Avg time new code %d iters: %g secs. Avg flops %g Gflops\n",ntests,total_time/ntests, total_FLOPs/total_time/1e9);      
    }


    
    
    if(0){ //test mesonfield contract
      //#define MF_CONTR_CPS
#define MF_CONTR_GRID
      std::cout << "Starting mesonfield contract benchmark\n";
      Float total_time = 0.;
      Float total_time_orig = 0.;
      for(int iter=0;iter<ntests;iter++){
	W.testRandom();
	V.testRandom();
	Wgrid.importFields(W);
	Vgrid.importFields(V);
      
#ifdef MF_CONTR_GRID
	total_time -= dclock();
	// CALLGRIND_START_INSTRUMENTATION ;
	// CALLGRIND_TOGGLE_COLLECT ;
	mf_grid.compute(Wgrid,mf_struct_grid,Vgrid,0);
	// CALLGRIND_TOGGLE_COLLECT ;
	// CALLGRIND_STOP_INSTRUMENTATION ;
	total_time += dclock();
#endif
#ifdef MF_CONTR_CPS
	total_time_orig -= dclock();
	mf.compute(W,mf_struct,V,0);
	total_time_orig += dclock();
#endif
#if defined(MF_CONTR_CPS) && defined(MF_CONTR_GRID)
	bool fail = false;
	for(int i=0;i<mf.size();i++){
	  const Ctype& gd = mf_grid.ptr()[i];
	  const Ctype& cp = mf.ptr()[i];
	  Ftype rdiff = fabs(gd.real()-cp.real());
	  Ftype idiff = fabs(gd.imag()-cp.imag());
	  if(rdiff > tol|| idiff > tol){
	    printf("Fail: Iter %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",iter, gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
	    fail = true;
	  }
	}
	if(fail) ERR.General("","","Standard vs Grid implementation test failed\n");	
#endif
      }
#if defined MF_CONTR_GRID
      printf("MF contract: Avg time new code %d iters: %g secs\n",ntests,total_time/ntests);
#endif
#if defined MF_CONTR_CPS
      printf("MF contract: Avg time old code %d iters: %g secs\n",ntests,total_time_orig/ntests);
#endif
    }else{
      W.testRandom();
      V.testRandom();
      Wgrid.importFields(W);
      Vgrid.importFields(V);

      mf.setup(W,V,0,0);
      mf_grid.setup(Wgrid,Vgrid,0,0);     
      mf.testRandom();
      for(int i=0;i<mf.getNrows();i++)
	for(int j=0;j<mf.getNcols();j++)
	  mf_grid(i,j) = mf(i,j); //both are scalar complex
    }

    if(0){ //test mf read/write
      mf.write("mesonfield.dat",FP_IEEE64BIG);
      
      A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfr;
      mfr.read("mesonfield.dat");
    }




    
    if(0){ //test vMv implementation
      //#define CPS_VMV
      //#define GRID_VMV
      //#define CPS_SPLIT_VMV
#define GRID_SPLIT_VMV
      //#define CPS_SPLIT_VMV_XALL
      //#define GRID_SPLIT_VMV_XALL

      std::cout << "Starting vMv benchmark\n";
      Float total_time = 0.;
      Float total_time_orig = 0.;
      Float total_time_split_orig = 0.;
      Float total_time_split_grid = 0.;
      Float total_time_split_orig_xall = 0.;
      Float total_time_split_grid_xall = 0.;
       
      CPSspinColorFlavorMatrix<mf_Complex> orig_sum[nthreads];
      CPSspinColorFlavorMatrix<grid_Complex> grid_sum[nthreads];

      CPSspinColorFlavorMatrix<mf_Complex> orig_tmp[nthreads];
      CPSspinColorFlavorMatrix<grid_Complex> grid_tmp[nthreads];

      CPSspinColorFlavorMatrix<mf_Complex> orig_sum_split[nthreads];
      CPSspinColorFlavorMatrix<grid_Complex> grid_sum_split[nthreads];

      CPSspinColorFlavorMatrix<mf_Complex> orig_sum_split_xall[nthreads];
      CPSspinColorFlavorMatrix<grid_Complex> grid_sum_split_xall[nthreads];

      
      int orig_3vol = GJP.VolNodeSites()/GJP.TnodeSites();
      int grid_3vol = Vgrid.getMode(0).nodeSites(0) * Vgrid.getMode(0).nodeSites(1) *Vgrid.getMode(0).nodeSites(2);

      mult_vMv_split<A2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw> vmv_split_orig;
      mult_vMv_split<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw> vmv_split_grid;

      std::vector<CPSspinColorFlavorMatrix<mf_Complex>> orig_split_xall_tmp(orig_3vol);
      Grid::Vector<CPSspinColorFlavorMatrix<grid_Complex> > grid_split_xall_tmp(grid_3vol);
      
      for(int iter=0;iter<ntests;iter++){
	for(int i=0;i<nthreads;i++){
	  orig_sum[i].zero(); grid_sum[i].zero();
	  orig_sum_split[i].zero(); grid_sum_split[i].zero();
	  orig_sum_split_xall[i].zero(); grid_sum_split_xall[i].zero();
	}
	
	for(int top = 0; top < GJP.TnodeSites(); top++){
#ifdef CPS_VMV
	  //ORIG VMV
	  total_time_orig -= dclock();	  
#pragma omp parallel for
	  for(int xop=0;xop<orig_3vol;xop++){
	    int me = omp_get_thread_num();
	    mult(orig_tmp[me], V, mf, W, xop, top, false, true);
	    orig_sum[me] += orig_tmp[me];
	  }
	  total_time_orig += dclock();
#endif
#ifdef GRID_VMV
	  //GRID VMV
	  total_time -= dclock();
#pragma omp parallel for
	  for(int xop=0;xop<grid_3vol;xop++){
	    int me = omp_get_thread_num();
	    mult(grid_tmp[me], Vgrid, mf_grid, Wgrid, xop, top, false, true);
	    grid_sum[me] += grid_tmp[me];
	  }
	  total_time += dclock();
#endif

#ifdef CPS_SPLIT_VMV
	  //SPLIT VMV
	  total_time_split_orig -= dclock();	  
	  vmv_split_orig.setup(V, mf, W, top);

#pragma omp parallel for
	  for(int xop=0;xop<orig_3vol;xop++){
	    int me = omp_get_thread_num();
	    vmv_split_orig.contract(orig_tmp[me], xop, false, true);
	    orig_sum_split[me] += orig_tmp[me];
	  }
	  total_time_split_orig += dclock();
#endif

#ifdef GRID_SPLIT_VMV
	  //SPLIT VMV GRID
	  total_time_split_grid -= dclock();	  
	  vmv_split_grid.setup(Vgrid, mf_grid, Wgrid, top);

#pragma omp parallel for
	  for(int xop=0;xop<grid_3vol;xop++){
	    int me = omp_get_thread_num();
	    vmv_split_grid.contract(grid_tmp[me], xop, false, true);
	    grid_sum_split[me] += grid_tmp[me];
	  }
	  total_time_split_grid += dclock();
#endif

#ifdef CPS_SPLIT_VMV_XALL	  	 
	  //SPLIT VMV THAT DOES IT FOR ALL SITES
	  total_time_split_orig_xall -= dclock();	  
	  vmv_split_orig.setup(V, mf, W, top);
	  vmv_split_orig.contract(orig_split_xall_tmp, false, true);
#pragma omp parallel for
	  for(int xop=0;xop<orig_3vol;xop++){
	    int me = omp_get_thread_num();
	    orig_sum_split_xall[me] += orig_split_xall_tmp[xop];
	  }
	  total_time_split_orig_xall += dclock();
#endif

#ifdef GRID_SPLIT_VMV_XALL
	  //SPLIT VMV GRID THAT DOES IT FOR ALL SITES
	  total_time_split_grid_xall -= dclock();	  
	  vmv_split_grid.setup(Vgrid, mf_grid, Wgrid, top);
	  vmv_split_grid.contract(grid_split_xall_tmp, false, true);
#pragma omp parallel for
	  for(int xop=0;xop<grid_3vol;xop++){
	    int me = omp_get_thread_num();	    
	    grid_sum_split_xall[me] += grid_split_xall_tmp[xop];
	  }
	  total_time_split_grid_xall += dclock();
#endif	  
	}//end top loop
	for(int i=1;i<nthreads;i++){
	  orig_sum[0] += orig_sum[i];
	  grid_sum[0] += grid_sum[i];
	  orig_sum_split[0] += orig_sum_split[i];
	  grid_sum_split[0] += grid_sum_split[i];
	  orig_sum_split_xall[0] += orig_sum_split_xall[i];
	  grid_sum_split_xall[0] += grid_sum_split_xall[i];  
	}
#ifdef CPS_VMV
	if(iter == 0){
#  ifdef GRID_VMV
            if(!compare(orig_sum[0],grid_sum[0],tol)) ERR.General("","","Standard vs Grid implementation test failed\n");
#  endif
#  ifdef CPS_SPLIT_VMV
	    if(!compare(orig_sum[0],orig_sum_split[0],tol)) ERR.General("","","Standard vs Split implementation test failed\n");
#  endif
#  ifdef GRID_SPLIT_VMV
	    if(!compare(orig_sum[0],grid_sum_split[0],tol)) ERR.General("","","Standard vs Grid Split implementation test failed\n");
#  endif
#  ifdef CPS_SPLIT_VMV_XALL
	    if(!compare(orig_sum[0],orig_sum_split_xall[0],tol)) ERR.General("","","Standard vs Split xall implementation test failed\n");
#  endif
#  ifdef GRID_SPLIT_VMV_XALL
	    if(!compare(orig_sum[0],grid_sum_split_xall[0],tol)) ERR.General("","","Standard vs Grid split xall implementation test failed\n");
#  endif
        }
#endif
      }
#ifdef CPS_VMV
      printf("vMv: Avg time old code %d iters: %g secs\n",ntests,total_time_orig/ntests);
#endif
#ifdef GRID_VMV
      printf("vMv: Avg time new code %d iters: %g secs\n",ntests,total_time/ntests);
#endif
#ifdef CPS_SPLIT_VMV
      printf("vMv: Avg time old code split %d iters: %g secs\n",ntests,total_time_split_orig/ntests);
#endif
#ifdef GRID_SPLIT_VMV
      printf("vMv: Avg time new code split %d iters: %g secs\n",ntests,total_time_split_grid/ntests);
#endif
#ifdef CPS_SPLIT_VMV_XALL
      printf("vMv: Avg time old code split xall %d iters: %g secs\n",ntests,total_time_split_orig_xall/ntests);
#endif
#ifdef CPS_SPLIT_VMV_XALL
      printf("vMv: Avg time new code split xall %d iters: %g secs\n",ntests,total_time_split_grid_xall/ntests);
#endif
    }
    

    if(0){ //test vv implementation
      std::cout << "Starting vv benchmark\n";
      Float total_time = 0.;
      Float total_time_orig = 0.;
      CPSspinColorFlavorMatrix<mf_Complex> orig_sum[nthreads];
      CPSspinColorFlavorMatrix<grid_Complex> grid_sum[nthreads];

      CPSspinColorFlavorMatrix<mf_Complex> orig_tmp[nthreads];
      CPSspinColorFlavorMatrix<grid_Complex> grid_tmp[nthreads];

      int orig_3vol = GJP.VolNodeSites()/GJP.TnodeSites();
      int grid_3vol = Vgrid.getMode(0).nodeSites(0) * Vgrid.getMode(0).nodeSites(1) *Vgrid.getMode(0).nodeSites(2);
      
      for(int iter=0;iter<ntests;iter++){
	for(int i=0;i<nthreads;i++){
	  orig_sum[i].zero(); grid_sum[i].zero();
	}
	
	for(int top = 0; top < GJP.TnodeSites(); top++){
	  //std::cout << "top " << top << std::endl;
	  //std::cout << "Starting orig\n";
	  total_time_orig -= dclock();	  
#pragma omp parallel for
	  for(int xop=0;xop<orig_3vol;xop++){
	    int me = omp_get_thread_num();
	    mult(orig_tmp[me], V, W, xop, top, false, true);
	    orig_sum[me] += orig_tmp[me];
	  }
	  total_time_orig += dclock();
	  //std::cout << "Starting Grid\n";
	  total_time -= dclock();
#pragma omp parallel for
	  for(int xop=0;xop<grid_3vol;xop++){
	    int me = omp_get_thread_num();
	    mult(grid_tmp[me], Vgrid, Wgrid, xop, top, false, true);
	    grid_sum[me] += grid_tmp[me];
	  }
	  total_time += dclock();	  
	}
	for(int i=1;i<nthreads;i++){
	  orig_sum[0] += orig_sum[i];
	  grid_sum[0] += grid_sum[i];
	}

	
	bool fail = false;
	
	Ctype gd;
	for(int sl=0;sl<4;sl++)
	  for(int cl=0;cl<3;cl++)
	    for(int fl=0;fl<2;fl++)
	      for(int sr=0;sr<4;sr++)
		for(int cr=0;cr<3;cr++)
		  for(int fr=0;fr<2;fr++){
		    gd = Reduce( grid_sum[0](sl,sr)(cl,cr)(fl,fr) );
		    const mf_Complex &cp = orig_sum[0](sl,sr)(cl,cr)(fl,fr);

		    double rdiff = fabs(gd.real()-cp.real());
		    double idiff = fabs(gd.imag()-cp.imag());
		    if(rdiff > tol|| idiff > tol){
		      printf("Fail: Iter %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",iter, gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		      fail = true;
		    }
		  }

	if(fail) ERR.General("","","Standard vs Grid implementation test failed\n");
      }

      printf("vv: Avg time new code %d iters: %g secs\n",ntests,total_time/ntests);
      printf("vv: Avg time old code %d iters: %g secs\n",ntests,total_time_orig/ntests);
    }

    
  }

  printf("Finished\n"); fflush(stdout);
  
  return 0;
}
