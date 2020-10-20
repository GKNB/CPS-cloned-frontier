#ifndef _BENCHMARK_MESONFIELD_H
#define _BENCHMARK_MESONFIELD_H

#ifdef USE_CALLGRIND
#include<valgrind/callgrind.h>
#else
#define CALLGRIND_START_INSTRUMENTATION ;
#define CALLGRIND_STOP_INSTRUMENTATION ;
#define CALLGRIND_TOGGLE_COLLECT ;
#endif

#ifdef USE_VTUNE
#include<ittnotify.h>
#else
void __itt_pause(){}
void __itt_resume(){}
void __itt_detach(){}
#endif

#ifdef USE_GPERFTOOLS
#include <gperftools/profiler.h>
#else
int ProfilerStart(const char* fname){}
void ProfilerStop(){}
#endif

#include<alg/a2a/a2a_fields.h>
#include<alg/a2a/mesonfield.h>
#include<alg/a2a/ktopipi_gparity.h>


#ifdef GRID_CUDA
#include <cuda_profiler_api.h>
#else
void cudaProfilerStart(){}
void cudaProfilerStop(){}
#endif

CPS_START_NAMESPACE

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



void randomMatrix(SpinColorFlavorMatrix &A, CPSspinColorFlavorMatrix<cps::ComplexD> &B){
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  for(int f1=0;f1<2;f1++)
	    for(int f2=0;f2<2;f2++){
	      cps::ComplexD tmp;
	      _testRandom<cps::ComplexD>::rand(&tmp,1, 3.0, -3.0);
	      A(s1,c1,f1,s2,c2,f2) = tmp;
	      B(s1,s2)(c1,c2)(f1,f2) = tmp;
	    }
}
void randomMatrix(SpinColorFlavorMatrix &A){
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  for(int f1=0;f1<2;f1++)
	    for(int f2=0;f2<2;f2++){
	      cps::ComplexD tmp;
	      _testRandom<cps::ComplexD>::rand(&tmp,1, 3.0, -3.0);
	      A(s1,c1,f1,s2,c2,f2) = tmp;
	    }
}
void randomMatrix(CPSspinColorFlavorMatrix<cps::ComplexD> &B){
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  for(int f1=0;f1<2;f1++)
	    for(int f2=0;f2<2;f2++){
	      cps::ComplexD tmp;
	      _testRandom<cps::ComplexD>::rand(&tmp,1, 3.0, -3.0);
	      B(s1,s2)(c1,c2)(f1,f2) = tmp;
	    }
}


void benchmarkTrace(const int ntests, const double tol){
  typedef CPSsquareMatrix<CPSsquareMatrix<CPSsquareMatrix<cps::ComplexD,2>,3>,4> SCFmat;
 typedef CPSsquareMatrix<cps::ComplexD,3> Cmat;
    
  //Test they give the same answer
  {
    SpinColorFlavorMatrix old_mat;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;
    randomMatrix(old_mat,new_mat);

    cps::Complex cp =  old_mat.Trace();
    cps::Complex gd = new_mat.Trace();
      
    bool fail = false;
    double rdiff = fabs(gd.real()-cp.real());
    double idiff = fabs(gd.imag()-cp.imag());
    if(rdiff > tol|| idiff > tol){
      printf("Fail: Trace Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
      fail = true;
    }
    if(fail) ERR.General("","","Trace test failed\n");
    else printf("Trace pass\n");
  }
  {
    //Benchmark it    
    SpinColorFlavorMatrix old_mat;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;
    
    //SpinFlavorTrace of SpinColorFlavorMatrix
    Float total_time_old = 0.;
    cps::ComplexD tmp_old;
    for(int iter=0;iter<ntests;iter++){
      randomMatrix(old_mat);      
      total_time_old -= dclock();
      tmp_old = old_mat.Trace();
      total_time_old += dclock();
    }
    Float total_time_new = 0.;
    cps::ComplexD tmp_new;
    for(int iter=0;iter<ntests;iter++){
      randomMatrix(new_mat);
      total_time_new -= dclock();
      tmp_new = new_mat.Trace();
      total_time_new += dclock();
    }
    printf("Trace: Avg time new code %d iters: %g secs\n",ntests,total_time_new/ntests);
    printf("Trace: Avg time old code %d iters: %g secs\n",ntests,total_time_old/ntests);
  }
}



void benchmarkSpinFlavorTrace(const int ntests, const double tol){
  typedef CPSsquareMatrix<CPSsquareMatrix<CPSsquareMatrix<cps::ComplexD,2>,3>,4> SCFmat;
  typedef CPSsquareMatrix<cps::ComplexD,3> Cmat;
    
  //Test they give the same answer
  {
    SpinColorFlavorMatrix old_mat;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;
    randomMatrix(old_mat,new_mat);

    Matrix tmp_mat_old =  old_mat.SpinFlavorTrace();
    Cmat tmp_mat_new = new_mat.SpinFlavorTrace();//TraceTwoIndices<0,2>();
      
    bool fail = false;
    for(int c1=0;c1<3;c1++)
      for(int c2=0;c2<3;c2++){
	cps::ComplexD gd = tmp_mat_old(c1,c2);
	cps::ComplexD cp = tmp_mat_new(c1,c2);
	  
	double rdiff = fabs(gd.real()-cp.real());
	double idiff = fabs(gd.imag()-cp.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: SFtrace Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
	  fail = true;
	}
      }
    if(fail) ERR.General("","","SFtrace test failed\n");
    else printf("SFtrace pass\n");
  }
  {
    //Benchmark it    
    SpinColorFlavorMatrix old_mat;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;
    
    //SpinFlavorTrace of SpinColorFlavorMatrix
    Float total_time_old = 0.;
    Matrix tmp_mat_old;
    for(int iter=0;iter<ntests;iter++){
      randomMatrix(old_mat);      
      total_time_old -= dclock();
      tmp_mat_old = old_mat.SpinFlavorTrace();
      total_time_old += dclock();
    }
    Float total_time_new = 0.;
    Cmat tmp_mat_new;
    for(int iter=0;iter<ntests;iter++){
      randomMatrix(new_mat);
      total_time_new -= dclock();
      //tmp_mat_new.zero();
      //_PartialDoubleTraceImpl<Cmat,CPSspinColorFlavorMatrix<cps::ComplexD>,0,2>::doit(tmp_mat_new,new_mats[iter]);
      tmp_mat_new = new_mat.TraceTwoIndices<0,2>();
      total_time_new += dclock();
    }
    printf("SFtrace: Avg time new code %d iters: %g secs\n",ntests,total_time_new/ntests);
    printf("SFtrace: Avg time old code %d iters: %g secs\n",ntests,total_time_old/ntests);
  }
}


void benchmarkTraceProd(const int ntests, const double tol){
  typedef CPSsquareMatrix<CPSsquareMatrix<CPSsquareMatrix<cps::ComplexD,2>,3>,4> SCFmat;
  typedef CPSsquareMatrix<cps::ComplexD,3> Cmat;
    
  //Test they give the same answer
  {
    SpinColorFlavorMatrix old_mat1, old_mat2;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat1, new_mat2;
    randomMatrix(old_mat1,new_mat1);
    randomMatrix(old_mat2,new_mat2);

    cps::ComplexD tr_old = Trace(old_mat1,old_mat2);
    cps::ComplexD tr_new = Trace(new_mat1,new_mat2);
      
    bool fail = false;
    cps::ComplexD &gd = tr_new;
    cps::ComplexD &cp = tr_old;
	  
    double rdiff = fabs(gd.real()-cp.real());
    double idiff = fabs(gd.imag()-cp.imag());
    if(rdiff > tol|| idiff > tol){
      printf("Fail: Prodtrace Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
      fail = true;
    }
    if(fail) ERR.General("","","Prodtrace test failed\n");
    else printf("Prodtrace pass\n");
  }
  //Benchmark it    
  {
    SpinColorFlavorMatrix old_mat1, old_mat2;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat1, new_mat2;
    
    Float total_time_old = 0.;
    cps::ComplexD tr_old;
    for(int iter=0;iter<ntests;iter++){
      randomMatrix(old_mat1);
      randomMatrix(old_mat2);
      total_time_old -= dclock();
      tr_old = Trace(old_mat1,old_mat2);
      total_time_old += dclock();
    }
    Float total_time_new = 0.;
    cps::ComplexD tr_new;
    for(int iter=0;iter<ntests;iter++){
      randomMatrix(new_mat1);
      randomMatrix(new_mat2);
      total_time_new -= dclock();
      tr_new = Trace(new_mat1,new_mat2);
      total_time_new += dclock();
    }
    printf("Prodtrace: Avg time new code %d iters: %g secs\n",ntests,total_time_new/ntests);
    printf("Prodtrace: Avg time old code %d iters: %g secs\n",ntests,total_time_old/ntests);
  }

}


void benchmarkColorTranspose(const int ntests, const double tol){
  typedef CPSsquareMatrix<CPSsquareMatrix<CPSsquareMatrix<cps::ComplexD,2>,3>,4> SCFmat;
  typedef CPSsquareMatrix<cps::ComplexD,3> Cmat;
    
  //Test they give the same answer
  {
    SpinColorFlavorMatrix old_mat;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;
    randomMatrix(old_mat,new_mat);

    old_mat.transpose_color();
    new_mat = new_mat.TransposeOnIndex<1>();
      
    bool fail = false;
    for(int s1=0;s1<4;s1++)
      for(int s2=0;s2<4;s2++)	
	for(int c1=0;c1<3;c1++)
	  for(int c2=0;c2<3;c2++)
	    for(int f1=0;f1<2;f1++)
	      for(int f2=0;f2<2;f2++){
		cps::ComplexD gd = new_mat(s1,s2)(c1,c2)(f1,f2);
		cps::ComplexD cp = old_mat(s1,c1,f1,s2,c2,f2);
	  
		double rdiff = fabs(gd.real()-cp.real());
		double idiff = fabs(gd.imag()-cp.imag());
		if(rdiff > tol|| idiff > tol){
		  printf("Fail: colortranspose Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		  fail = true;
		}
	      }
    if(fail) ERR.General("","","colortranspose test failed\n");
    else printf("colortranspose pass\n");
  }
  {
    //Benchmark it    
    SpinColorFlavorMatrix old_mat;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;
    
    Float total_time_old = 0.;

    for(int iter=0;iter<ntests;iter++){
      randomMatrix(old_mat);      
      total_time_old -= dclock();
      old_mat.transpose_color();
      total_time_old += dclock();
    }
    Float total_time_new = 0.;

    for(int iter=0;iter<ntests;iter++){
      randomMatrix(new_mat);
      total_time_new -= dclock();
      new_mat = new_mat.TransposeOnIndex<1>();
      total_time_new += dclock();
    }
    printf("colorTranspose: Avg time new code %d iters: %g secs\n",ntests,total_time_new/ntests);
    printf("colorTranspose: Avg time old code %d iters: %g secs\n",ntests,total_time_old/ntests);
  }
}

void multGammaLeftOld(SpinColorFlavorMatrix &M, const int whichGamma, const int i, const int mu){
  assert(whichGamma == 1 || whichGamma==2);
  static int g1[8] = {0,1,0,1,2,3,2,3};
  static int g2[8] = {1,0,3,2,1,0,3,2};

  int gg = whichGamma == 1 ? g1[i] : g2[i];
  switch(gg){
  case 0:
    M.pl(F0).gl(mu);
    break;
  case 1:
    M.pl(F0).glAx(mu);
    break;
  case 2:
    M.pl(F1).gl(mu); M *= -1.0;
    break;
  case 3:
    M.pl(F1).glAx(mu); M *= -1.0;
    break;
  default:
    ERR.General("ComputeKtoPiPiGparityBase","multGammaLeft","Invalid idx\n");
    break;
  }
}

void benchmarkmultGammaLeft(const int ntests, const double tol){
  typedef CPSsquareMatrix<CPSsquareMatrix<CPSsquareMatrix<cps::ComplexD,2>,3>,4> SCFmat;
  
  //Test they give the same answer
  for(int gamma=1;gamma<=2;gamma++)
    for(int i=0;i<8;i++)
      for(int mu=0;mu<4;mu++){
	SpinColorFlavorMatrix old_mat;
	CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;
	randomMatrix(old_mat,new_mat);

	multGammaLeftOld(old_mat,gamma,i,mu);
	ComputeKtoPiPiGparityBase::multGammaLeft(new_mat,gamma,i,mu);
      
	bool fail = false;
	for(int s1=0;s1<4;s1++)
	  for(int s2=0;s2<4;s2++)	
	    for(int c1=0;c1<3;c1++)
	      for(int c2=0;c2<3;c2++)
		for(int f1=0;f1<2;f1++)
		  for(int f2=0;f2<2;f2++){
		    cps::ComplexD gd = new_mat(s1,s2)(c1,c2)(f1,f2);
		    cps::ComplexD cp = old_mat(s1,c1,f1,s2,c2,f2);
		    
		    double rdiff = fabs(gd.real()-cp.real());
		    double idiff = fabs(gd.imag()-cp.imag());
		    if(rdiff > tol|| idiff > tol){
		      printf("Fail: multGammaLeft(%d,%d,%d) Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gamma,i,mu,gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		      fail = true;
		    }
		  }
	if(fail) ERR.General("","","multGammaLeft test %d %d %d failed\n",gamma,i,mu);
	//else printf("multGammaLeft %d %d %d pass\n",gamma,i,mu);
      }

  //Benchmark
  for(int gamma=1;gamma<=2;gamma++)
    for(int i=0;i<8;i++)
      for(int mu=0;mu<4;mu++){
	SpinColorFlavorMatrix old_mat;
	CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;

	Float total_time_old = 0.;
	
	for(int iter=0;iter<ntests;iter++){
	  randomMatrix(old_mat);      
	  total_time_old -= dclock();
	  multGammaLeftOld(old_mat,gamma,i,mu);
	  total_time_old += dclock();
	}
	Float total_time_new = 0.;
	
	for(int iter=0;iter<ntests;iter++){
	  randomMatrix(new_mat);
	  total_time_new -= dclock();
	  ComputeKtoPiPiGparityBase::multGammaLeft(new_mat,gamma,i,mu);
	  total_time_new += dclock();
	}
	printf("multGammaLeft %d %d %d: Avg time new code %d iters: %g secs\n",gamma,i,mu,ntests,total_time_new/ntests);
	printf("multGammaLeft %d %d %d: Avg time old code %d iters: %g secs\n\n",gamma,i,mu,ntests,total_time_old/ntests);
      }  
}

template<typename mf_Complex, typename grid_Complex>
bool compare(const CPSspinColorFlavorMatrix<mf_Complex> &orig, const CPSspinColorFlavorMatrix<grid_Complex> &grid, const double tol){
  bool fail = false;
  
  mf_Complex gd;
  for(int sl=0;sl<4;sl++)
    for(int cl=0;cl<3;cl++)
      for(int fl=0;fl<2;fl++)
	for(int sr=0;sr<4;sr++)
	  for(int cr=0;cr<3;cr++)
	    for(int fr=0;fr<2;fr++){
	      gd = Reduce( grid(sl,sr)(cl,cr)(fl,fr) );
	      const mf_Complex &cp = orig(sl,sr)(cl,cr)(fl,fr);
	      
	      double rdiff = fabs(gd.real()-cp.real());
	      double idiff = fabs(gd.imag()-cp.imag());
	      if(rdiff > tol|| idiff > tol){
		printf("Fail: Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		fail = true;
	      }
	    }
  return !fail;
}

template<typename mf_Complex>
bool compare(const CPSspinColorFlavorMatrix<mf_Complex> &orig, const CPSspinColorFlavorMatrix<mf_Complex> &newimpl, const double tol){
  bool fail = false;
  
  for(int sl=0;sl<4;sl++)
    for(int cl=0;cl<3;cl++)
      for(int fl=0;fl<2;fl++)
	for(int sr=0;sr<4;sr++)
	  for(int cr=0;cr<3;cr++)
	    for(int fr=0;fr<2;fr++){
	      const mf_Complex &gd = newimpl(sl,sr)(cl,cr)(fl,fr);
	      const mf_Complex &cp = orig(sl,sr)(cl,cr)(fl,fr);
	      
	      double rdiff = fabs(gd.real()-cp.real());
	      double idiff = fabs(gd.imag()-cp.imag());
	      if(rdiff > tol|| idiff > tol){
		printf("Fail: Newimpl (%g,%g) Orig (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		fail = true;
	      }
	    }
  return !fail;
}


template< typename mf_Complex, int SiteSize, typename FlavorPolicy, typename AllocPolicy>
void printRow(const CPSfield<mf_Complex,SiteSize,FourDpolicy<FlavorPolicy>,AllocPolicy> &field, const int dir, const std::string &comment,
	       typename my_enable_if< _equal<typename ComplexClassify<mf_Complex>::type, complex_double_or_float_mark>::value, const int>::type = 0
	       ){
  int L = GJP.Nodes(dir)*GJP.NodeSites(dir);
  std::vector<mf_Complex> buf(L,0.);

  int other_dirs[3]; int aa=0;
  for(int i=0;i<4;i++)
    if(i!=dir) other_dirs[aa++] = i;

  
  if(GJP.NodeCoor(other_dirs[0]) == 0 && GJP.NodeCoor(other_dirs[1]) == 0 && GJP.NodeCoor(other_dirs[2]) == 0){
    for(int x=GJP.NodeCoor(dir)*GJP.NodeSites(dir); x < (GJP.NodeCoor(dir)+1)*GJP.NodeSites(dir); x++){
      int lcoor[4] = {0,0,0,0};
      lcoor[dir] = x - GJP.NodeCoor(dir)*GJP.NodeSites(dir);
      
      mf_Complex const* site_ptr = field.site_ptr(lcoor);
      buf[x] = *site_ptr;
    }
  }
  globalSum(buf.data(),L);

  
  if(!UniqueID()){
    printf("%s: (",comment.c_str()); fflush(stdout);
    for(int x=0;x<L;x++){
      if(x % GJP.NodeSites(dir) == 0 && x!=0)
	printf(")(");
      
      printf("[%f,%f] ",buf[x].real(),buf[x].imag());
    }
    printf(")\n"); fflush(stdout);
  }
}

#ifdef USE_GRID
template< typename mf_Complex, int SiteSize, typename FlavorPolicy, typename AllocPolicy>
void printRow(const CPSfield<mf_Complex,SiteSize,FourDSIMDPolicy<FlavorPolicy>,AllocPolicy> &field, const int dir, const std::string &comment,
	       typename my_enable_if< _equal<typename ComplexClassify<mf_Complex>::type, grid_vector_complex_mark>::value, const int>::type = 0
	       ){
  typedef typename mf_Complex::scalar_type ScalarComplex;
  NullObject null_obj;
  CPSfield<ScalarComplex,SiteSize,FourDpolicy<FlavorPolicy>,StandardAllocPolicy> tmp(null_obj);
  tmp.importField(field);
  printRow(tmp,dir,comment);
}
#endif


void testCyclicPermute(){
  NullObject null_obj;
  {//4D
    CPSfield<cps::ComplexD,1,FourDpolicy<FixedFlavorPolicy<1> >,StandardAllocPolicy> from(null_obj);
    CPSfield<cps::ComplexD,1,FourDpolicy<FixedFlavorPolicy<1> >,StandardAllocPolicy> tmp1(null_obj);
    CPSfield<cps::ComplexD,1,FourDpolicy<FixedFlavorPolicy<1> >,StandardAllocPolicy> tmp2(null_obj);

    from.testRandom();

    for(int dir=0;dir<4;dir++){
      int incrs[3] = { GJP.NodeSites(dir)/2, GJP.NodeSites(dir), GJP.NodeSites(dir) + GJP.NodeSites(dir)/2  };
      for(int i=0;i<3;i++){
	int incr = incrs[i];
	for(int pm=-1;pm<=1;pm+=2){
	  if(!UniqueID()) printf("Testing 4D permute in direction %c%d with increment %d\n",pm == 1 ? '+' : '-',dir,incr);
	  //permute in incr until we cycle all the way around
	  tmp1 = from;
	  CPSfield<cps::ComplexD,1,FourDpolicy<FixedFlavorPolicy<1> >,StandardAllocPolicy> *send = &tmp1;
	  CPSfield<cps::ComplexD,1,FourDpolicy<FixedFlavorPolicy<1> >,StandardAllocPolicy> *recv = &tmp2;

	  int shifted = 0;
	  printRow(from,dir,"Initial line      ");

	  int total = GJP.Nodes(dir)*GJP.NodeSites(dir);
	  int perm = 0;
	  while(shifted < total){
	    int amt = std::min(incr, total-shifted);

	    cyclicPermute(*recv,*send,dir,pm,amt);
	    shifted += amt;
	    std::ostringstream comment; comment << "After perm " << perm++ << " by incr " << amt;
	    printRow(*recv,dir,comment.str());
	    
	    if(shifted < total)
	      std::swap(send,recv);
	  }
	  printRow(*recv,dir,"Final line      ");
	  
	  int coor[4];
	  for(coor[0]=0;coor[0]<GJP.XnodeSites();coor[0]++){
	    for(coor[1]=0;coor[1]<GJP.YnodeSites();coor[1]++){
	      for(coor[2]=0;coor[2]<GJP.ZnodeSites();coor[2]++){
		for(coor[3]=0;coor[3]<GJP.TnodeSites();coor[3]++){
		  cps::ComplexD const* orig = from.site_ptr(coor);
		  cps::ComplexD const* permd = recv->site_ptr(coor);
		  if(orig->real() != permd->real() || orig->imag() != permd->imag()){
		    printf("Error node coor (%d,%d,%d,%d) (%d,%d,%d,%d) : (%g,%g) vs (%g,%g) diff (%g,%g)\n",GJP.XnodeCoor(),GJP.YnodeCoor(),GJP.ZnodeCoor(),GJP.TnodeCoor(),coor[0],coor[1],coor[2],coor[3],orig->real(),orig->imag(),permd->real(),permd->imag(), orig->real()-permd->real(),orig->imag()-permd->imag());
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }//End 4D

  {//3D
    CPSfield<cps::ComplexD,1,SpatialPolicy<FixedFlavorPolicy<1> >,StandardAllocPolicy> from(null_obj);
    CPSfield<cps::ComplexD,1,SpatialPolicy<FixedFlavorPolicy<1> >,StandardAllocPolicy> tmp1(null_obj);
    CPSfield<cps::ComplexD,1,SpatialPolicy<FixedFlavorPolicy<1> >,StandardAllocPolicy> tmp2(null_obj);

    from.testRandom();

    for(int dir=0;dir<3;dir++){
      for(int pm=-1;pm<=1;pm+=2){
	if(!UniqueID()) printf("Testing 3D permute in direction %c%d\n",pm == 1 ? '+' : '-',dir);
	//permute in incr until we cycle all the way around
	tmp1 = from;
	CPSfield<cps::ComplexD,1,SpatialPolicy<FixedFlavorPolicy<1> >,StandardAllocPolicy> *send = &tmp1;
	CPSfield<cps::ComplexD,1,SpatialPolicy<FixedFlavorPolicy<1> >,StandardAllocPolicy> *recv = &tmp2;

	int shifted = 0;
	int total = GJP.Nodes(dir)*GJP.NodeSites(dir);
	int incr = GJP.NodeSites(dir)/2;
	int perm = 0;
	while(shifted < total){
	  cyclicPermute(*recv,*send,dir,pm,incr);
	  shifted += incr;
	  if(shifted < total)
	    std::swap(send,recv);
	}
      
	int coor[3];
	for(coor[0]=0;coor[0]<GJP.XnodeSites();coor[0]++){
	  for(coor[1]=0;coor[1]<GJP.YnodeSites();coor[1]++){
	    for(coor[2]=0;coor[2]<GJP.ZnodeSites();coor[2]++){
	      cps::ComplexD const* orig = from.site_ptr(coor);
	      cps::ComplexD const* permd = recv->site_ptr(coor);
	      if(orig->real() != permd->real() || orig->imag() != permd->imag()){
		printf("Error node coor (%d,%d,%d,%d) (%d,%d,%d) : (%g,%g) vs (%g,%g) diff (%g,%g)\n",GJP.XnodeCoor(),GJP.YnodeCoor(),GJP.ZnodeCoor(),GJP.TnodeCoor(),coor[0],coor[1],coor[2],orig->real(),orig->imag(),permd->real(),permd->imag(), orig->real()-permd->real(),orig->imag()-permd->imag());
	      }
	    }
	  }
	}	
      }
    }
  }//End 3D

#ifdef USE_GRID

  {//4D
    typedef FourDSIMDPolicy<FixedFlavorPolicy<1> >::ParamType simd_params;
    simd_params sp;
    FourDSIMDPolicy<FixedFlavorPolicy<1> >::SIMDdefaultLayout(sp, Grid::vComplexD::Nsimd() );
  
    CPSfield<Grid::vComplexD,1,FourDSIMDPolicy<FixedFlavorPolicy<1> >,Aligned128AllocPolicy> from_grid(sp);
    CPSfield<Grid::vComplexD,1,FourDSIMDPolicy<FixedFlavorPolicy<1> >,Aligned128AllocPolicy> tmp1_grid(sp);
    CPSfield<Grid::vComplexD,1,FourDSIMDPolicy<FixedFlavorPolicy<1> >,Aligned128AllocPolicy> tmp2_grid(sp);

    CPSfield<cps::ComplexD,1,FourDpolicy<FixedFlavorPolicy<1> >,StandardAllocPolicy> from(null_obj);
    CPSfield<cps::ComplexD,1,FourDpolicy<FixedFlavorPolicy<1> >,StandardAllocPolicy> tmp1(null_obj);
    from.testRandom();
    from_grid.importField(from);

    for(int dir=0;dir<4;dir++){
      for(int pm=-1;pm<=1;pm+=2){
	if(!UniqueID()) printf("Testing 4D permute in direction %c%d with SIMD layout\n",pm == 1 ? '+' : '-',dir);
	//permute in incr until we cycle all the way around
	tmp1_grid = from_grid;
	CPSfield<Grid::vComplexD,1,FourDSIMDPolicy<FixedFlavorPolicy<1> >,Aligned128AllocPolicy> *send = &tmp1_grid;
	CPSfield<Grid::vComplexD,1,FourDSIMDPolicy<FixedFlavorPolicy<1> >,Aligned128AllocPolicy> *recv = &tmp2_grid;

	int shifted = 0;
	printRow(from_grid,dir,"Initial line      ");

	int total = GJP.Nodes(dir)*GJP.NodeSites(dir);
	int incr = GJP.NodeSites(dir)/2;
	int perm = 0;
	while(shifted < total){
	  cyclicPermute(*recv,*send,dir,pm,incr);
	  shifted += incr;
	  std::ostringstream comment; comment << "After perm " << perm++ << " by incr " << incr;
	  printRow(*recv,dir,comment.str());
	        
	  if(shifted < total)
	    std::swap(send,recv);
	}
	printRow(*recv,dir,"Final line      ");

	tmp1.importField(*recv);
      
	int coor[4];
	for(coor[0]=0;coor[0]<GJP.XnodeSites();coor[0]++){
	  for(coor[1]=0;coor[1]<GJP.YnodeSites();coor[1]++){
	    for(coor[2]=0;coor[2]<GJP.ZnodeSites();coor[2]++){
	      for(coor[3]=0;coor[3]<GJP.TnodeSites();coor[3]++){
		cps::ComplexD const* orig = from.site_ptr(coor);
		cps::ComplexD const* permd = tmp1.site_ptr(coor);
		if(orig->real() != permd->real() || orig->imag() != permd->imag()){
		  printf("Error node coor (%d,%d,%d,%d) (%d,%d,%d,%d) : (%g,%g) vs (%g,%g) diff (%g,%g)\n",GJP.XnodeCoor(),GJP.YnodeCoor(),GJP.ZnodeCoor(),GJP.TnodeCoor(),coor[0],coor[1],coor[2],coor[3],orig->real(),orig->imag(),permd->real(),permd->imag(), orig->real()-permd->real(),orig->imag()-permd->imag());
		}
	      }
	    }
	  }
	}
      }
    }
  }

  {//3D
    typedef ThreeDSIMDPolicy<FixedFlavorPolicy<1> >::ParamType simd_params;
    simd_params sp;
    ThreeDSIMDPolicy<FixedFlavorPolicy<1> >::SIMDdefaultLayout(sp, Grid::vComplexD::Nsimd() );
  
    CPSfield<Grid::vComplexD,1,ThreeDSIMDPolicy<FixedFlavorPolicy<1> >,Aligned128AllocPolicy> from_grid(sp);
    CPSfield<Grid::vComplexD,1,ThreeDSIMDPolicy<FixedFlavorPolicy<1> >,Aligned128AllocPolicy> tmp1_grid(sp);
    CPSfield<Grid::vComplexD,1,ThreeDSIMDPolicy<FixedFlavorPolicy<1> >,Aligned128AllocPolicy> tmp2_grid(sp);

    CPSfield<cps::ComplexD,1,SpatialPolicy<FixedFlavorPolicy<1> >,StandardAllocPolicy> from(null_obj);
    CPSfield<cps::ComplexD,1,SpatialPolicy<FixedFlavorPolicy<1> >,StandardAllocPolicy> tmp1(null_obj);
    from.testRandom();
    from_grid.importField(from);

    for(int dir=0;dir<3;dir++){
      for(int pm=-1;pm<=1;pm+=2){
	if(!UniqueID()) printf("Testing 3D permute in direction %c%d with SIMD layout\n",pm == 1 ? '+' : '-',dir);
	//permute in incr until we cycle all the way around
	tmp1_grid = from_grid;
	CPSfield<Grid::vComplexD,1,ThreeDSIMDPolicy<FixedFlavorPolicy<1> >,Aligned128AllocPolicy> *send = &tmp1_grid;
	CPSfield<Grid::vComplexD,1,ThreeDSIMDPolicy<FixedFlavorPolicy<1> >,Aligned128AllocPolicy> *recv = &tmp2_grid;

	int shifted = 0;
	int total = GJP.Nodes(dir)*GJP.NodeSites(dir);
	int incr = GJP.NodeSites(dir)/2;
	int perm = 0;
	while(shifted < total){
	  cyclicPermute(*recv,*send,dir,pm,incr);
	  shifted += incr;
	  if(shifted < total)
	    std::swap(send,recv);
	}
	tmp1.importField(*recv);
      
	int coor[3];
	for(coor[0]=0;coor[0]<GJP.XnodeSites();coor[0]++){
	  for(coor[1]=0;coor[1]<GJP.YnodeSites();coor[1]++){
	    for(coor[2]=0;coor[2]<GJP.ZnodeSites();coor[2]++){
	      cps::ComplexD const* orig = from.site_ptr(coor);
	      cps::ComplexD const* permd = tmp1.site_ptr(coor);
	      if(orig->real() != permd->real() || orig->imag() != permd->imag()){
		printf("Error node coor (%d,%d,%d,%d) (%d,%d,%d) : (%g,%g) vs (%g,%g) diff (%g,%g)\n",GJP.XnodeCoor(),GJP.YnodeCoor(),GJP.ZnodeCoor(),GJP.TnodeCoor(),coor[0],coor[1],coor[2],orig->real(),orig->imag(),permd->real(),permd->imag(), orig->real()-permd->real(),orig->imag()-permd->imag());
	      }
	    }
	  }
	}
      }

    }
  }
#endif

  if(!UniqueID()){ printf("Passed permute test\n"); fflush(stdout); }
} 




template<typename A2Apolicies>
void demonstrateFFTreln(const A2AArg &a2a_args){
  //Demonstrate relation between FFTW fields
  A2AvectorW<A2Apolicies> W(a2a_args);
  A2AvectorV<A2Apolicies> V(a2a_args);
  W.testRandom();
  V.testRandom();

  int p1[3] = {1,1,1};
  int p5[3] = {5,1,1};

  twist<typename A2Apolicies::FermionFieldType> twist_p1(p1);
  twist<typename A2Apolicies::FermionFieldType> twist_p5(p5);
    
  A2AvectorVfftw<A2Apolicies> Vfftw_p1(a2a_args);
  Vfftw_p1.fft(V,&twist_p1);

  A2AvectorVfftw<A2Apolicies> Vfftw_p5(a2a_args);
  Vfftw_p5.fft(V,&twist_p5);

  //f5(n) = f1(n+1)
  for(int i=0;i<Vfftw_p1.getNmodes();i++)
    cyclicPermute(Vfftw_p1.getMode(i), Vfftw_p1.getMode(i), 0, -1, 1);
    
  printRow(Vfftw_p1.getMode(0),0, "T_-1 V(p1) T_-1");
  printRow(Vfftw_p5.getMode(0),0, "V(p5)          ");

  for(int i=0;i<Vfftw_p1.getNmodes();i++)
    assert( Vfftw_p1.getMode(i).equals( Vfftw_p5.getMode(i), 1e-7, true ) );

  A2AvectorWfftw<A2Apolicies> Wfftw_p1(a2a_args);
  Wfftw_p1.fft(W,&twist_p1);

  A2AvectorWfftw<A2Apolicies> Wfftw_p5(a2a_args);
  Wfftw_p5.fft(W,&twist_p5);

  for(int i=0;i<Wfftw_p1.getNmodes();i++)
    cyclicPermute(Wfftw_p1.getMode(i), Wfftw_p1.getMode(i), 0, -1, 1);

  printRow(Wfftw_p1.getMode(0),0, "T_-1 W(p1) T_-1");
  printRow(Wfftw_p5.getMode(0),0, "W(p5)          ");

  for(int i=0;i<Wfftw_p1.getNmodes();i++)
    assert( Wfftw_p1.getMode(i).equals( Wfftw_p5.getMode(i), 1e-7, true ) );

  if(!UniqueID()) printf("Passed FFT relation test\n");
}

template<typename ParamType, typename mf_Complex>
struct defaultFieldParams{
  static void get(ParamType &into){}
};

template<int N, typename mf_Complex>
struct defaultFieldParams< SIMDdims<N>, mf_Complex >{
  static void get(SIMDdims<N> &into){
    SIMDpolicyBase<N>::SIMDdefaultLayout(into, mf_Complex::Nsimd(), 2);
  }
};

template<typename A2Apolicies>
void testA2AvectorFFTrelnGparity(const A2AArg &a2a_args,Lattice &lat){
  assert(GJP.Gparity());

  if(lat.FixGaugeKind() == FIX_GAUGE_NONE){
    FixGaugeArg fix_gauge_arg;
    fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
    fix_gauge_arg.hyperplane_start = 0;
    fix_gauge_arg.hyperplane_step = 1;
    fix_gauge_arg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites();
    fix_gauge_arg.stop_cond = 1e-08;
    fix_gauge_arg.max_iter_num = 10000;

    CommonArg common_arg;
  
    AlgFixGauge fix_gauge(lat,&common_arg,&fix_gauge_arg);
    fix_gauge.run();
  }
  
  //Demonstrate relation between FFTW fields
  typedef typename A2Apolicies::ComplexType mf_Complex;
  typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);
  
  A2AvectorW<A2Apolicies> W(a2a_args,fp);
  W.testRandom();

  int p_p1[3];
  GparityBaseMomentum(p_p1,+1);

  int p_m1[3];
  GparityBaseMomentum(p_m1,-1);

  //Perform base FFTs
  //twist<typename A2Apolicies::FermionFieldType> twist_p1(p_p1);
  //twist<typename A2Apolicies::FermionFieldType> twist_m1(p_m1);

  gaugeFixAndTwist<typename A2Apolicies::FermionFieldType> twist_p1(p_p1,lat);
  gaugeFixAndTwist<typename A2Apolicies::FermionFieldType> twist_m1(p_m1,lat);
  
  A2AvectorWfftw<A2Apolicies> Wfftw_p1(a2a_args,fp);
  Wfftw_p1.fft(W,&twist_p1);

  A2AvectorWfftw<A2Apolicies> Wfftw_m1(a2a_args,fp);
  Wfftw_m1.fft(W,&twist_m1);


  int p[3];  
  A2AvectorWfftw<A2Apolicies> result(a2a_args,fp);
  A2AvectorWfftw<A2Apolicies> compare(a2a_args,fp);
  
  //Get twist for first excited momentum in p1 set
  {
    memcpy(p,p_p1,3*sizeof(int));
    p[0] = 5;
    //twist<typename A2Apolicies::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies::FermionFieldType> twist_p(p,lat);    
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in x direction\n",p[0],p[1],p[2]);

    printRow(result.getMode(0),0,  "Result ");
    printRow(compare.getMode(0),0, "Compare");
    
    for(int i=0;i<compare.getNmodes();i++)
      assert( compare.getMode(i).equals( result.getMode(i), 1e-8, true ) );
  }
  //Get twist for first negative excited momentum in p1 set
  if(GJP.Bc(1) == BND_CND_GPARITY){
    memcpy(p,p_p1,3*sizeof(int));
    p[1] = -3;
    //twist<typename A2Apolicies::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies::FermionFieldType> twist_p(p,lat);    
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in y direction\n",p[0],p[1],p[2]);

    printRow(result.getMode(0),0,  "Result ");
    printRow(compare.getMode(0),0, "Compare");
    
    for(int i=0;i<compare.getNmodes();i++)
      assert( compare.getMode(i).equals( result.getMode(i), 1e-8, true ) );
  }
  //Try two directions
  if(GJP.Bc(1) == BND_CND_GPARITY){
    memcpy(p,p_p1,3*sizeof(int));
    p[0] = -3;
    p[1] = -3;
    //twist<typename A2Apolicies::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies::FermionFieldType> twist_p(p,lat);    
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in y direction\n",p[0],p[1],p[2]);

    printRow(result.getMode(0),0,  "Result ");
    printRow(compare.getMode(0),0, "Compare");
    
    for(int i=0;i<compare.getNmodes();i++)
      assert( compare.getMode(i).equals( result.getMode(i), 1e-7, true ) );
  }
  //Try 3 directions
  if(GJP.Bc(1) == BND_CND_GPARITY && GJP.Bc(2) == BND_CND_GPARITY){
    memcpy(p,p_p1,3*sizeof(int));
    p[0] = -3;
    p[1] = -3;
    p[2] = -3;
    //twist<typename A2Apolicies::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies::FermionFieldType> twist_p(p,lat);    
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in y direction\n",p[0],p[1],p[2]);

    printRow(result.getMode(0),0,  "Result ");
    printRow(compare.getMode(0),0, "Compare");
    
    for(int i=0;i<compare.getNmodes();i++)
      assert( compare.getMode(i).equals( result.getMode(i), 1e-7, true ) );
  }
  //Get twist for first excited momentum in m1 set
  {
    memcpy(p,p_m1,3*sizeof(int));
    p[0] = 3;
    //twist<typename A2Apolicies::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies::FermionFieldType> twist_p(p,lat);   
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in x direction\n",p[0],p[1],p[2]);

    printRow(result.getMode(0),0,  "Result ");
    printRow(compare.getMode(0),0, "Compare");
    
    for(int i=0;i<compare.getNmodes();i++)
      assert( compare.getMode(i).equals( result.getMode(i), 1e-8, true ) );
  }
  
}



template<typename T>
struct _printit{
  static void printit(const T d[], const int n){
    for(int i=0;i<n;i++){
      std::cout << d[i] << " ";
    }
    std::cout << std::endl;
  }
};

template<typename T>
struct _printit<std::complex<T> >{
  static void printit(const std::complex<T> d[], const int n){
    for(int i=0;i<n;i++){
      std::cout << '[' << d[i].real() << ',' << d[i].imag() << "] ";
    }
    std::cout << std::endl;
  }
};

  
template<typename T>
void printit(const T d[], const int n){
  _printit<T>::printit(d,n);
}



template<typename T>
void printvType(const T& v){
  typedef typename T::scalar_type S;
  int Nsimd = T::Nsimd();
  S to[Nsimd];
  vstore(v,to);
  printit(to,Nsimd);
}


template<typename T>
struct _rand{
  inline static T rand(){
    return LRG.Urand();
  }
};



template<typename T>
struct _rand<std::complex<T> >{
  inline static std::complex<T> rand(){
    return std::complex<T>(LRG.Urand(),LRG.Urand());
  }
};

template<typename T>
T randomvType(){
  T out;
  typedef typename T::scalar_type S;
  int Nsimd = T::Nsimd();
  S r[Nsimd];
  for(int i=0;i<Nsimd;i++) r[i] = _rand<S>::rand();
  vset(out,r);
  return out;
}

template<typename A2Apolicies>
void testMultiSource(const A2AArg &a2a_args,Lattice &lat){
  assert(GJP.Gparity());
  
  if(lat.FixGaugeKind() == FIX_GAUGE_NONE){
    FixGaugeArg fix_gauge_arg;
    fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
    fix_gauge_arg.hyperplane_start = 0;
    fix_gauge_arg.hyperplane_step = 1;
    fix_gauge_arg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites();
    fix_gauge_arg.stop_cond = 1e-08;
    fix_gauge_arg.max_iter_num = 10000;

    CommonArg common_arg;
  
    AlgFixGauge fix_gauge(lat,&common_arg,&fix_gauge_arg);
    fix_gauge.run();
  }
  
  //Demonstrate relation between FFTW fields
  typedef typename A2Apolicies::ComplexType mf_Complex;
  typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);
  
  A2AvectorW<A2Apolicies> W(a2a_args,fp);
  A2AvectorV<A2Apolicies> V(a2a_args,fp);
  W.testRandom();
  V.testRandom();

  int p[3];
  GparityBaseMomentum(p,+1);
  ThreeMomentum pp(p);

  GparityBaseMomentum(p,-1);
  ThreeMomentum pm(p);

  ThreeMomentum pp3 = pp * 3;
  ThreeMomentum pm3 = pm * 3;

  if(1){ //1s + 2s source
    typedef typename A2AflavorProjectedExpSource<typename A2Apolicies::SourcePolicies>::FieldParamType SrcFieldParamType;
    typedef typename A2AflavorProjectedExpSource<typename A2Apolicies::SourcePolicies>::ComplexType SrcComplexType;
    SrcFieldParamType sfp; defaultFieldParams<SrcFieldParamType, SrcComplexType>::get(sfp);

    typedef A2AflavorProjectedExpSource<typename A2Apolicies::SourcePolicies> ExpSrcType;
    typedef A2AflavorProjectedHydrogenSource<typename A2Apolicies::SourcePolicies> HydSrcType;
  
    ExpSrcType _1s_src(2.0, pp.ptr(), sfp);
    HydSrcType _2s_src(2,0,0, 2.0, pp.ptr(), sfp);

    typedef SCFspinflavorInnerProduct<15,mf_Complex,ExpSrcType,true,false> ExpInnerType;
    typedef SCFspinflavorInnerProduct<15,mf_Complex,HydSrcType,true,false> HydInnerType;
  
    ExpInnerType _1s_inner(sigma3, _1s_src);
    HydInnerType _2s_inner(sigma3, _2s_src);

    A2AvectorWfftw<A2Apolicies> Wfftw_pp(a2a_args,fp);
    Wfftw_pp.gaugeFixTwistFFT(W,pp.ptr(),lat);

    A2AvectorVfftw<A2Apolicies> Vfftw_pp(a2a_args,fp);
    Vfftw_pp.gaugeFixTwistFFT(V,pp.ptr(),lat);
  
    std::vector< A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_std_1s_pp_pp;
    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_std_1s_pp_pp, Wfftw_pp, _1s_inner, Vfftw_pp);

    typedef GparityFlavorProjectedBasicSourceStorage<A2Apolicies, ExpInnerType> ExpStorageType;
  
    ExpStorageType exp_store_1s_pp_pp(_1s_inner);
    exp_store_1s_pp_pp.addCompute(0,0,pp,pp);

    typename ComputeMesonFields<A2Apolicies,ExpStorageType>::WspeciesVector Wspecies(1, &W);
    typename ComputeMesonFields<A2Apolicies,ExpStorageType>::VspeciesVector Vspecies(1, &V);

    std::cout << "Start 1s ExpStorage compute\n";
    ComputeMesonFields<A2Apolicies,ExpStorageType>::compute(exp_store_1s_pp_pp,Wspecies,Vspecies,lat);

    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    for(int t=0;t<Lt;t++){
      if(!UniqueID()) printf("Comparing test 1 t=%d\n",t);
      assert( exp_store_1s_pp_pp[0][t].equals(mf_std_1s_pp_pp[t],1e-10,true) );
    }
    if(!UniqueID()) printf("Passed equivalence test 1\n");

    typedef Elem<ExpSrcType,Elem<HydSrcType,ListEnd> > SrcList;
    typedef A2AmultiSource<SrcList> MultiSrcType;
    typedef SCFspinflavorInnerProduct<15,mf_Complex,MultiSrcType,true,false> ExpHydMultiInnerType;

    MultiSrcType exp_hyd_multi_src;
    exp_hyd_multi_src.template getSource<0>().setup(2.0,pp.ptr(),sfp);
    exp_hyd_multi_src.template getSource<1>().setup(2,0,0, 2.0, pp.ptr(), sfp);
  
    ExpHydMultiInnerType exp_hyd_multi_inner(sigma3,exp_hyd_multi_src);

    typedef GparityFlavorProjectedBasicSourceStorage<A2Apolicies, HydInnerType> HydStorageType;
    HydStorageType exp_store_2s_pp_pp(_2s_inner);
    exp_store_2s_pp_pp.addCompute(0,0,pp,pp);
    exp_store_2s_pp_pp.addCompute(0,0,pm,pp);
    exp_store_2s_pp_pp.addCompute(0,0,pp3,pp);

  
    ComputeMesonFields<A2Apolicies,HydStorageType>::compute(exp_store_2s_pp_pp,Wspecies,Vspecies,lat);

  
    typedef GparityFlavorProjectedMultiSourceStorage<A2Apolicies, ExpHydMultiInnerType> ExpHydMultiStorageType;
    ExpHydMultiStorageType exp_store_1s_2s_pp_pp(exp_hyd_multi_inner, exp_hyd_multi_src);
    exp_store_1s_2s_pp_pp.addCompute(0,0,pp,pp);

    std::cout << "Start 1s/2s ExpHydMultiStorage compute\n";
    ComputeMesonFields<A2Apolicies,ExpHydMultiStorageType>::compute(exp_store_1s_2s_pp_pp,Wspecies,Vspecies,lat);
  
    for(int t=0;t<Lt;t++){
      if(!UniqueID()) printf("Comparing test 2 t=%d\n",t);
      assert( exp_store_1s_2s_pp_pp(0,0)[t].equals(mf_std_1s_pp_pp[t],1e-10,true) );
    }
    if(!UniqueID()) printf("Passed equivalence test 2\n");
    for(int t=0;t<Lt;t++){
      if(!UniqueID()) printf("Comparing test 3 t=%d\n",t);
      assert( exp_store_1s_2s_pp_pp(1,0)[t].equals(exp_store_2s_pp_pp[0][t],1e-10,true) );
    }
    if(!UniqueID()) printf("Passed equivalence test 3\n");
  }

  if(1){ //1s + point source
    if(!UniqueID()) printf("Doing 1s+point source\n");
    typedef typename A2AflavorProjectedExpSource<typename A2Apolicies::SourcePolicies>::FieldParamType SrcFieldParamType;
    typedef typename A2AflavorProjectedExpSource<typename A2Apolicies::SourcePolicies>::ComplexType SrcComplexType;
    SrcFieldParamType sfp; defaultFieldParams<SrcFieldParamType, SrcComplexType>::get(sfp);

    typedef A2AflavorProjectedExpSource<typename A2Apolicies::SourcePolicies> ExpSrcType;
    typedef A2AflavorProjectedPointSource<typename A2Apolicies::SourcePolicies> PointSrcType;
    typedef A2ApointSource<typename A2Apolicies::SourcePolicies> PointSrcBasicType;

    ExpSrcType _1s_src(2.0, pp3.ptr(), sfp);
    PointSrcType _pt_src(sfp);
    PointSrcBasicType _pt_basic_src(sfp);
    
    typedef SCFspinflavorInnerProduct<15,mf_Complex,ExpSrcType,true,false> ExpInnerType;
    typedef SCFspinflavorInnerProduct<15,mf_Complex,PointSrcType,true,false> PointInnerType;
    typedef SCFspinflavorInnerProduct<15,mf_Complex,PointSrcBasicType,true,false> PointBasicInnerType;
  
    ExpInnerType _1s_inner(sigma3, _1s_src);
    PointInnerType _pt_inner(sigma3, _pt_src);
    PointBasicInnerType _pt_basic_inner(sigma3, _pt_basic_src);

    A2AvectorWfftw<A2Apolicies> Wfftw_pp(a2a_args,fp);
    Wfftw_pp.gaugeFixTwistFFT(W,pp.ptr(),lat);

    A2AvectorVfftw<A2Apolicies> Vfftw_pp(a2a_args,fp);
    Vfftw_pp.gaugeFixTwistFFT(V,pp3.ptr(),lat);
  
    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    //Do the point and 1s by regular means
    if(!UniqueID()){ printf("Computing with point source\n"); fflush(stdout); }       
    std::vector< A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_pt_std;
    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_pt_std, Wfftw_pp, _pt_inner, Vfftw_pp);

    if(!UniqueID()){ printf("Computing with 1s source\n"); fflush(stdout); }
    std::vector< A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_1s_std;
    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_1s_std, Wfftw_pp, _1s_inner, Vfftw_pp);

    //1) Check flavor projected point and basic point give the same result (no projector for point)
    {
      if(!UniqueID()){ printf("Computing with non-flavor projected point source\n"); fflush(stdout); }
      std::vector< A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_basic;
      A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_basic, Wfftw_pp, _pt_basic_inner, Vfftw_pp);
      
      for(int t=0;t<Lt;t++){
	if(!UniqueID()) printf("1) Comparing flavor projected point src to basic point src t=%d\n",t);
	assert( mf_pt_std[t].equals(mf_basic[t],1e-10,true) );
      }
      if(!UniqueID()) printf("Passed point check test\n");
    }
    
    //Prepare the compound src
    typedef Elem<ExpSrcType,Elem<PointSrcType,ListEnd> > SrcList;
    typedef A2AmultiSource<SrcList> MultiSrcType;

    if(1){
      typedef SCFspinflavorInnerProduct<15,mf_Complex,MultiSrcType,true,false> MultiInnerType;
      
      MultiSrcType multi_src;    
      multi_src.template getSource<0>().setup(2.0,pp3.ptr(),sfp);
      multi_src.template getSource<1>().setup(sfp);      
      
      MultiInnerType multi_inner(sigma3,multi_src);
      
      typedef GparityFlavorProjectedMultiSourceStorage<A2Apolicies, MultiInnerType> MultiStorageType;
      MultiStorageType store(multi_inner, multi_src);
      store.addCompute(0,0,pp,pp3);

      std::cout << "Start 1s/point MultiStorage compute\n";
      typename ComputeMesonFields<A2Apolicies,MultiStorageType>::WspeciesVector Wspecies(1, &W);
      typename ComputeMesonFields<A2Apolicies,MultiStorageType>::VspeciesVector Vspecies(1, &V);

      ComputeMesonFields<A2Apolicies,MultiStorageType>::compute(store,Wspecies,Vspecies,lat);
  
      //Test 1s
      for(int t=0;t<Lt;t++){
	if(!UniqueID()) printf("Comparing 1s t=%d\n",t);
	assert( store(0,0)[t].equals(mf_1s_std[t],1e-6,true) );
      }
      if(!UniqueID()) printf("Passed 1s multisrc equivalence test\n");
      
      //Test point
      for(int t=0;t<Lt;t++){
	if(!UniqueID()) printf("Comparing point t=%d\n",t);
	assert( store(1,0)[t].equals(mf_pt_std[t],1e-6,true) );
      }
      if(!UniqueID()) printf("Passed point multisrc equivalence test\n");
    }

    //Test the compound shift source also
    {
      typedef GparitySourceShiftInnerProduct<mf_Complex,MultiSrcType, flavorMatrixSpinColorContract<15,true,false> > MultiInnerType;
      
      MultiSrcType multi_src;    
      multi_src.template getSource<0>().setup(2.0,pp3.ptr(),sfp);
      multi_src.template getSource<1>().setup(sfp);      
      
      MultiInnerType multi_inner(sigma3,multi_src);
      
      typedef GparityFlavorProjectedShiftSourceStorage<A2Apolicies, MultiInnerType> MultiStorageType;
      MultiStorageType store(multi_inner, multi_src);
      store.addCompute(0,0,pp,pp3);

      if(!UniqueID()){ printf("Start 1s/point shift multiStorage compute\n"); fflush(stdout); }
      typename ComputeMesonFields<A2Apolicies,MultiStorageType>::WspeciesVector Wspecies(1, &W);
      typename ComputeMesonFields<A2Apolicies,MultiStorageType>::VspeciesVector Vspecies(1, &V);

      ComputeMesonFields<A2Apolicies,MultiStorageType>::compute(store,Wspecies,Vspecies,lat);
  
      //Test 1s
      for(int t=0;t<Lt;t++){
	if(!UniqueID()) printf("Comparing shift 1s t=%d\n",t);
	assert( store(0,0)[t].equals(mf_1s_std[t],1e-6,true) );
      }
      if(!UniqueID()) printf("Passed 1s shift multisrc equivalence test\n");
      
      //Test point
      for(int t=0;t<Lt;t++){
	if(!UniqueID()) printf("Comparing shift point t=%d\n",t);
	assert( store(1,0)[t].equals(mf_pt_std[t],1e-6,true) );
      }
      if(!UniqueID()) printf("Passed point shift multisrc equivalence test\n");
    }
  }
  
}



//Added to attempt to isolate a SEGV on BG/Q!
template<typename A2Apolicies, int isGparity>
struct _testKtoPiPiType3{};

template<typename A2Apolicies>
struct _testKtoPiPiType3<A2Apolicies, 0>{
  static void run(const A2AArg &a2a_args,Lattice &lat){}
};
template<typename A2Apolicies>
struct _testKtoPiPiType3<A2Apolicies, 1>{
  static void run(const A2AArg &a2a_args,Lattice &lat){
    if(!UniqueID()){ printf("Test run of K->pipi type 3\n"); fflush(stdout); }
    assert(GJP.Gparity());

    typedef typename A2Apolicies::ComplexType mf_Complex;
    typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;
    FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);
  
    A2AvectorW<A2Apolicies> W(a2a_args,fp);
    A2AvectorV<A2Apolicies> V(a2a_args,fp);
    W.testRandom();
    V.testRandom();

    int p[3];
    GparityBaseMomentum(p,+1);
    ThreeMomentum pp(p);

    GparityBaseMomentum(p,-1);
    ThreeMomentum pm(p);

    MesonFieldMomentumContainer<A2Apolicies> mf_pions;

    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    A2Aparams params(a2a_args);

    typedef std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_WV;
    typedef std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw> > mf_WW;
  
    mf_WV tmp_WV(Lt);
    mf_WW tmp_WW(Lt);
    for(int t=0;t<Lt;t++){
      tmp_WV[t].setup(params,params,t,t);
      tmp_WV[t].testRandom();

      tmp_WW[t].setup(params,params,t,t);
      tmp_WW[t].testRandom();
    }
    mf_pions.copyAdd(pp,tmp_WV);
    mf_pions.copyAdd(pm,tmp_WV);

    int pipi_sep = 2;
    int tsep_k_pi = 6;
    int tstep = 1;
  
    typedef typename ComputeKtoPiPiGparity<A2Apolicies>::ResultsContainerType ResultsContainerType;
    typedef typename ComputeKtoPiPiGparity<A2Apolicies>::MixDiagResultsContainerType MixDiagResultsContainerType;
  
    ResultsContainerType type3;
    MixDiagResultsContainerType mix3;
    ComputeKtoPiPiGparity<A2Apolicies>::type3(type3,mix3,
					      tsep_k_pi, pipi_sep, tstep, pp,
					      tmp_WW, mf_pions,
					      V, V,
					      W, W);
    if(!UniqueID()){ printf("End of test run of K->pipi type 3\n"); fflush(stdout); }
  }
};


template<typename A2Apolicies>
void testKtoPiPiType3(const A2AArg &a2a_args,Lattice &lat){
  _testKtoPiPiType3<A2Apolicies, A2Apolicies::GPARITY>::run(a2a_args,lat);
}




template<typename A2Apolicies, int isGparity>
struct _benchmarkKtoPiPiOffload{};

template<typename A2Apolicies>
struct _benchmarkKtoPiPiOffload<A2Apolicies, 0>{
  static void run(const A2AArg &a2a_args,Lattice &lat){}
};
template<typename A2Apolicies>
struct _benchmarkKtoPiPiOffload<A2Apolicies, 1>{
  typedef typename A2Apolicies::ComplexType mf_Complex;
  typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;    
  typedef std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_WV;
  typedef std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorWfftw> > mf_WW;
  typedef typename ComputeKtoPiPiGparity<A2Apolicies>::ResultsContainerType ResultsContainerType;
  typedef typename ComputeKtoPiPiGparity<A2Apolicies>::MixDiagResultsContainerType MixDiagResultsContainerType;

  Lattice &lat;
  const A2AArg &a2a_args;
  A2Aparams params;
  int Lt;

  FieldInputParamType fp;
  A2AvectorW<A2Apolicies> *W;
  A2AvectorV<A2Apolicies> *V;
  A2AvectorW<A2Apolicies> *Wh;
  A2AvectorV<A2Apolicies> *Vh;

  MesonFieldMomentumContainer<A2Apolicies> mf_pions;

  mf_WV tmp_WV;
  mf_WW tmp_WW;
    
  ThreeMomentum pp;
  ThreeMomentum pm;

  int pipi_sep;
  int tstep;
  std::vector<int> tsep_k_pi;

  ~_benchmarkKtoPiPiOffload(){
    delete W;
    delete V;
    delete Wh;
    delete Vh;
  }

  _benchmarkKtoPiPiOffload(const A2AArg &a2a_args,Lattice &lat): a2a_args(a2a_args), lat(lat), params(a2a_args), Lt(GJP.Tnodes()*GJP.TnodeSites()), 
								 tmp_WV(Lt), tmp_WW(Lt), pipi_sep(2), tstep(1), tsep_k_pi({6}){
    assert(GJP.Gparity());

    defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);
  
    W = new A2AvectorW<A2Apolicies>(a2a_args,fp);
    V = new A2AvectorV<A2Apolicies>(a2a_args,fp);
    W->testRandom();
    V->testRandom();

    Wh = new A2AvectorW<A2Apolicies>(a2a_args,fp);
    Vh = new A2AvectorV<A2Apolicies>(a2a_args,fp);
    Wh->testRandom();
    Vh->testRandom();

    int p[3];
    GparityBaseMomentum(p,+1);
    pp = ThreeMomentum(p);

    GparityBaseMomentum(p,-1);
    pm = ThreeMomentum(p);

    for(int t=0;t<Lt;t++){
      tmp_WV[t].setup(params,params,t,t);
      tmp_WV[t].testRandom();

      tmp_WW[t].setup(params,params,t,t);
      tmp_WW[t].testRandom();
    }
    mf_pions.copyAdd(pp,tmp_WV);
    mf_pions.copyAdd(pm,tmp_WV);
  }

  void type1(){
    if(!UniqueID()){ printf("Timing K->pipi type 1 field version\n"); fflush(stdout); }
  
    ResultsContainerType result;
    ComputeKtoPiPiGparity<A2Apolicies>::type1_field(&result, tsep_k_pi, pipi_sep, tstep, pp, tmp_WW, mf_pions, *V, *Vh, *W, *Wh);
    if(!UniqueID()){ printf("End of timing of K->pipi type 1 field version\n"); fflush(stdout); }
  }

  void type4(){
    if(!UniqueID()){ printf("Timing K->pipi type 4 field version\n"); fflush(stdout); }
  
    ResultsContainerType result;
    MixDiagResultsContainerType mix;

    ComputeKtoPiPiGparity<A2Apolicies>::type4_field(result, mix, 1, tmp_WW, *V, *Vh, *W, *Wh);
    if(!UniqueID()){ printf("End of timing of K->pipi type 4 field version\n"); fflush(stdout); }
  }
};


template<typename A2Apolicies>
void benchmarkKtoPiPiType1offload(const A2AArg &a2a_args,Lattice &lat){
  _benchmarkKtoPiPiOffload<A2Apolicies, A2Apolicies::GPARITY> calc(a2a_args,lat);
  calc.type1();
}

template<typename A2Apolicies>
void benchmarkKtoPiPiType4offload(const A2AArg &a2a_args,Lattice &lat){
  _benchmarkKtoPiPiOffload<A2Apolicies, A2Apolicies::GPARITY> calc(a2a_args,lat);
  calc.type4();
}







//Test the compute-many storage that sums meson fields on the fly
template<typename A2Apolicies>
void testSumSource(const A2AArg &a2a_args,Lattice &lat){
  assert(GJP.Gparity());
  
  if(lat.FixGaugeKind() == FIX_GAUGE_NONE){
    FixGaugeArg fix_gauge_arg;
    fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
    fix_gauge_arg.hyperplane_start = 0;
    fix_gauge_arg.hyperplane_step = 1;
    fix_gauge_arg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites();
    fix_gauge_arg.stop_cond = 1e-08;
    fix_gauge_arg.max_iter_num = 10000;

    CommonArg common_arg;
  
    AlgFixGauge fix_gauge(lat,&common_arg,&fix_gauge_arg);
    fix_gauge.run();
  }
  
  int Lt = GJP.TnodeSites() * GJP.Tnodes();

  typedef typename A2Apolicies::ComplexType mf_Complex;
  typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);
  
  A2AvectorW<A2Apolicies> W(a2a_args,fp);
  A2AvectorV<A2Apolicies> V(a2a_args,fp);
  W.testRandom();
  V.testRandom();


  std::vector<ThreeMomentum> p_wdag;
  std::vector<ThreeMomentum> p_v;



  //Total mom (-2,*,*)

  //Base
  p_wdag.push_back(ThreeMomentum(-3,0,0));
  p_v.push_back(ThreeMomentum(1,0,0));
  
  //Symmetrized (lives in same momentum set as base)
  p_wdag.push_back(ThreeMomentum(1,0,0));
  p_v.push_back(ThreeMomentum(-3,0,0));
  
  //Alt (lives in other momentum set)
  p_wdag.push_back(ThreeMomentum(3,0,0));
  p_v.push_back(ThreeMomentum(-5,0,0));
  
  //Alt symmetrized
  p_wdag.push_back(ThreeMomentum(-5,0,0));
  p_v.push_back(ThreeMomentum(3,0,0));

  int nmom = p_v.size();

  for(int i=1;i<3;i++){
    if(GJP.Bc(i) == BND_CND_GPARITY){
      for(int p=0;p<nmom;p++){
	p_wdag[p](i) = p_wdag[p](0);
	p_v[p](i) = p_v[p](0);
      }
    }
  }
  
  typedef A2AflavorProjectedExpSource<typename A2Apolicies::SourcePolicies> ExpSrcType;
  typedef typename ExpSrcType::FieldParamType SrcFieldParamType;
  typedef typename ExpSrcType::ComplexType SrcComplexType;
  SrcFieldParamType sfp; defaultFieldParams<SrcFieldParamType, SrcComplexType>::get(sfp);

  typedef SCFspinflavorInnerProduct<15,mf_Complex,ExpSrcType,true,false> ExpInnerType;
  
  ExpSrcType src(2.0, p_v[0].ptr(), sfp); //momentum is not relevant as it is shifted internally
  ExpInnerType inner(sigma3, src);

  typedef GparityFlavorProjectedBasicSourceStorage<A2Apolicies, ExpInnerType> BasicStorageType;

  typename ComputeMesonFields<A2Apolicies,BasicStorageType>::WspeciesVector Wspecies(1, &W);
  typename ComputeMesonFields<A2Apolicies,BasicStorageType>::VspeciesVector Vspecies(1, &V);

  BasicStorageType store_basic(inner);
  for(int p=0;p<nmom;p++){
    store_basic.addCompute(0,0,-p_wdag[p],p_v[p]);
  }

  ComputeMesonFields<A2Apolicies,BasicStorageType>::compute(store_basic,Wspecies,Vspecies,lat);

  typedef std::vector< A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > MFvectorType;
  
  for(int p=0;p<nmom;p++)
    nodeGetMany(1, &store_basic(p));

  MFvectorType mf_basic = store_basic(0);
  for(int t=0;t<Lt;t++){
    for(int p=1;p<nmom;p++){
      mf_basic[t].plus_equals(store_basic(p)[t]);
    }
    mf_basic[t].times_equals(1./nmom);
  }
  
  std::vector< std::pair<ThreeMomentum,ThreeMomentum> > set_mom;
  for(int p=0;p<nmom;p++)
    set_mom.push_back( std::pair<ThreeMomentum,ThreeMomentum>(-p_wdag[p],p_v[p]) );
  
  typedef GparityFlavorProjectedSumSourceStorage<A2Apolicies, ExpInnerType> SumStorageType;

  SumStorageType store_sum(inner);
  store_sum.addComputeSet(0,0, set_mom);
  
  ComputeMesonFields<A2Apolicies,SumStorageType>::compute(store_sum,Wspecies,Vspecies,lat);

  store_sum.sumToAverage();

  nodeGetMany(1, &store_sum(0) );

  for(int t=0;t<Lt;t++){
    if(!UniqueID()) printf("Comparing mf avg t=%d\n",t);
    assert( mf_basic[t].equals( store_sum(0)[t],1e-6,true) );
  }
  if(!UniqueID()) printf("Passed mf avg sum source equivalence test\n");

  typedef typename A2Apolicies::ComplexType VectorComplexType;
  
  typedef GparitySourceShiftInnerProduct<VectorComplexType,ExpSrcType, flavorMatrixSpinColorContract<15,true,false> > ShiftInnerType;
  typedef GparityFlavorProjectedShiftSourceSumStorage<A2Apolicies, ShiftInnerType> ShiftSumStorageType;
  
  ShiftInnerType shift_inner(sigma3, src);

  ShiftSumStorageType shift_store_sum(shift_inner,src);
  shift_store_sum.addComputeSet(0,0, set_mom, true);
  
  ComputeMesonFields<A2Apolicies,ShiftSumStorageType>::compute(shift_store_sum,Wspecies,Vspecies,lat);

  typedef std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mfVector;
  const mfVector &avgd = shift_store_sum(0);
  printf("Index 0 points to %p\n", &avgd); fflush(stdout);
  
  nodeGetMany(1, &shift_store_sum(0) );

  for(int t=0;t<Lt;t++){
    if(!UniqueID()) printf("Comparing mf avg t=%d\n",t);
    assert( mf_basic[t].equals( shift_store_sum(0)[t],1e-6,true) );
  }
  if(!UniqueID()) printf("Passed mf avg sum source equivalence test\n");
}



template<typename A2Apolicies>
void testMfFFTreln(const A2AArg &a2a_args,Lattice &lat){
  assert(GJP.Gparity());

  if(lat.FixGaugeKind() == FIX_GAUGE_NONE){
    FixGaugeArg fix_gauge_arg;
    fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
    fix_gauge_arg.hyperplane_start = 0;
    fix_gauge_arg.hyperplane_step = 1;
    fix_gauge_arg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites();
    fix_gauge_arg.stop_cond = 1e-08;
    fix_gauge_arg.max_iter_num = 10000;

    CommonArg common_arg;
  
    AlgFixGauge fix_gauge(lat,&common_arg,&fix_gauge_arg);
    fix_gauge.run();
  }
  typedef typename A2Apolicies::ComplexType mf_Complex;
  typedef typename A2Apolicies::SourcePolicies SourcePolicies;
  
  typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);

  typedef typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType SrcInputParamType;
  SrcInputParamType sp; defaultFieldParams<SrcInputParamType, mf_Complex>::get(sp);

  A2AvectorW<A2Apolicies> W(a2a_args,fp);
  A2AvectorV<A2Apolicies> V(a2a_args,fp);
  W.testRandom();
  V.testRandom();

  int pp[3]; GparityBaseMomentum(pp,+1); //(1,1,1)
  int pm[3]; GparityBaseMomentum(pm,-1); //(-1,-1,-1)

  //M_ij^{4a+k,4b+l} =  \sum_{n=0}^{L-1} \Omega^{\dagger,4a+k}_i(n) \Gamma \gamma(n) N^{4b+l}_j(n)     (1)
  //                    \sum_{n=0}^{L-1} \Omega^{\dagger,k}_i(n-a-b) \Gamma \gamma(n-b) N^l_j(n)         (2)
  
  //\Omega^{\dagger,k}_i(n) = [ \sum_{x=0}^{L-1} e^{-2\pi i nx/L} e^{- (-k) \pi ix/2L} W_i(x) ]^\dagger
  //N^l_j(n) = \sum_{x=0}^{L-1} e^{-2\pi ix/L} e^{-l \pi ix/2L} V_i(x)

  //Use a state with total momentum 0; k=1 l=-1 a=-1 b=1  so total momentum  -3 + 3  = 0

  int a = -1;
  int b = 1;
  int k = 1;
  int l = -1;

  assert(a+b == 0); //don't want to permute W V right now
  
  //For (1) 
  int p1w[3] = { -(4*a+k), pm[1],pm[1] };  //fix other momenta to first allowed
  int p1v[3] = { 4*b+l, pm[1],pm[1] };
  
  //For (2)
  int p2w[3] = {-k, pm[1],pm[1]};
  int p2v[3] = {l, pm[1],pm[1]};
  
  typedef A2AflavorProjectedExpSource<SourcePolicies> SrcType;
  typedef SCFspinflavorInnerProduct<0,mf_Complex,SrcType,true,false> InnerType; //unit matrix spin structure
  typedef GparityFlavorProjectedBasicSourceStorage<A2Apolicies, InnerType> StorageType;

  SrcType src1(2., pp, sp);
  SrcType src2(2., pp, sp);
  cyclicPermute( src2.getSource(), src2.getSource(), 0, 1, b);

  InnerType inner1(sigma0,src1);
  InnerType inner2(sigma0,src2);
  StorageType mf_store1(inner1);
  StorageType mf_store2(inner2);

  mf_store1.addCompute(0,0, ThreeMomentum(p1w), ThreeMomentum(p1v) );
  mf_store1.addCompute(0,0, ThreeMomentum(p2w), ThreeMomentum(p2v) );
  
  mf_store2.addCompute(0,0, ThreeMomentum(p2w), ThreeMomentum(p2v) );

  typename ComputeMesonFields<A2Apolicies,StorageType>::WspeciesVector Wspecies(1, &W);
  typename ComputeMesonFields<A2Apolicies,StorageType>::VspeciesVector Vspecies(1, &V);

  ComputeMesonFields<A2Apolicies,StorageType>::compute(mf_store1,Wspecies,Vspecies,lat);
  ComputeMesonFields<A2Apolicies,StorageType>::compute(mf_store2,Wspecies,Vspecies,lat);

  printf("Testing mf relation\n"); fflush(stdout);
  assert( mf_store1[0][0].equals( mf_store2[0][0], 1e-6, true) );
  printf("MF Relation proven\n");

  // StorageType mf_store3(inner1);
  // mf_store3.addCompute(0,0, ThreeMomentum(p1w), ThreeMomentum(p1v), true );
#if 1
  
  typedef GparitySourceShiftInnerProduct<mf_Complex,SrcType,flavorMatrixSpinColorContract<0,true,false> > ShiftInnerType;
  typedef GparityFlavorProjectedShiftSourceStorage<A2Apolicies, ShiftInnerType> ShiftStorageType;
  
  SrcType src3(2., pp, sp);
  ShiftInnerType shift_inner(sigma0,src3);
  ShiftStorageType mf_shift_store(shift_inner,src3);
  mf_shift_store.addCompute(0,0, ThreeMomentum(p1w), ThreeMomentum(p1v) );
  mf_shift_store.addCompute(0,0, ThreeMomentum(p2w), ThreeMomentum(p2v) );
  int nc = mf_shift_store.nCompute();
  printf("Number of optimized computations: %d\n",nc);

  ComputeMesonFields<A2Apolicies,ShiftStorageType>::compute(mf_shift_store,Wspecies,Vspecies,lat);

  assert( mf_shift_store[0][0].equals( mf_store1[0][0], 1e-6, true) );
  assert( mf_shift_store[1][0].equals( mf_store1[1][0], 1e-6, true) );
  printf("Passed test of shift storage for single source type\n");

  typedef Elem<SrcType, Elem<SrcType,ListEnd > > SrcList;
  typedef A2AmultiSource<SrcList> MultiSrcType;
  typedef GparitySourceShiftInnerProduct<mf_Complex,MultiSrcType,flavorMatrixSpinColorContract<0,true,false> > ShiftMultiSrcInnerType;
  typedef GparityFlavorProjectedShiftSourceStorage<A2Apolicies, ShiftMultiSrcInnerType> ShiftMultiSrcStorageType;

  MultiSrcType multisrc;
  multisrc.template getSource<0>().setup(3.,pp, sp);
  multisrc.template getSource<1>().setup(2.,pp, sp);
  ShiftMultiSrcInnerType shift_inner_multisrc(sigma0,multisrc);
  ShiftMultiSrcStorageType mf_shift_multisrc_store(shift_inner_multisrc, multisrc);
  mf_shift_multisrc_store.addCompute(0,0, ThreeMomentum(p1w), ThreeMomentum(p1v) );
  mf_shift_multisrc_store.addCompute(0,0, ThreeMomentum(p2w), ThreeMomentum(p2v) );
  
  ComputeMesonFields<A2Apolicies,ShiftMultiSrcStorageType>::compute(mf_shift_multisrc_store,Wspecies,Vspecies,lat);

  assert( mf_shift_multisrc_store(1,0)[0].equals( mf_store1[0][0], 1e-6, true) );
  assert( mf_shift_multisrc_store(1,1)[0].equals( mf_store1[1][0], 1e-6, true) );
  
  
#endif
}
  
//  static void ComputeKtoPiPiGparityBase::multGammaLeft(CPSspinColorFlavorMatrix<ComplexType> &M, const int whichGamma, const int i, const int mu){



//non-SIMD data
template<typename A2Apolicies>
void benchmarkFFT(const int ntest){
  typedef typename A2Apolicies::FermionFieldType::FieldSiteType mf_Complex;
  typedef typename A2Apolicies::FermionFieldType::FieldMappingPolicy MappingPolicy;
  typedef typename A2Apolicies::FermionFieldType::FieldAllocPolicy AllocPolicy;

  typedef typename MappingPolicy::template Rebase<OneFlavorPolicy>::type OneFlavorMap;
  
  typedef CPSfield<mf_Complex,12,OneFlavorMap, AllocPolicy> FieldType;
  typedef typename FieldType::InputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);

  bool do_dirs[4] = {1,1,1,0}; //3D fft
  
  FieldType in(fp);
  in.testRandom();

  FieldType out1(fp);
  double t_orig = -dclock();
  for(int i=0;i<ntest;i++){
    if(!UniqueID()) printf("FFT orig %d\n",i);
    fft(out1,in,do_dirs);
  }
  t_orig += dclock();
  t_orig /= ntest;

  FieldType out2(fp);
  double t_opt = -dclock();
  for(int i=0;i<ntest;i++){  
    if(!UniqueID()) printf("FFT opt %d\n",i);
    fft_opt(out2,in,do_dirs);
  }
  t_opt += dclock();
  t_opt /= ntest;

  if(!UniqueID()){
    printf("3D FFT timings: orig %f s   opt %f s\n", t_orig, t_opt);    
    fft_opt_mu_timings::get().print();
  }
  
}







template<typename A2Apolicies>
void testA2AFFTinv(const A2AArg &a2a_args,Lattice &lat){
  assert(GJP.Gparity());

  if(lat.FixGaugeKind() == FIX_GAUGE_NONE){
    FixGaugeArg fix_gauge_arg;
    fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
    fix_gauge_arg.hyperplane_start = 0;
    fix_gauge_arg.hyperplane_step = 1;
    fix_gauge_arg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites();
    fix_gauge_arg.stop_cond = 1e-08;
    fix_gauge_arg.max_iter_num = 10000;

    CommonArg common_arg;
  
    AlgFixGauge fix_gauge(lat,&common_arg,&fix_gauge_arg);
    fix_gauge.run();
  }
  typedef typename A2Apolicies::ComplexType mf_Complex;
  typedef typename A2Apolicies::SourcePolicies SourcePolicies;
  
  typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);

  typedef typename A2Apolicies::SourcePolicies::MappingPolicy::ParamType SrcInputParamType;
  SrcInputParamType sp; defaultFieldParams<SrcInputParamType, mf_Complex>::get(sp);

  A2AvectorW<A2Apolicies> W(a2a_args,fp);
  A2AvectorV<A2Apolicies> V(a2a_args,fp);
  W.testRandom();
  V.testRandom();

  int pp[3]; GparityBaseMomentum(pp,+1); //(1,1,1)
  int pm[3]; GparityBaseMomentum(pm,-1); //(-1,-1,-1)
  
  A2AvectorVfftw<A2Apolicies> Vfft(a2a_args,fp);
  Vfft.fft(V);

  A2AvectorV<A2Apolicies> Vrec(a2a_args,fp);
  Vfft.inversefft(Vrec);

  for(int i=0;i<V.getNmodes();i++){
    assert( Vrec.getMode(i).equals( V.getMode(i), 1e-08, true) ); 
  }
  if(!UniqueID()) printf("Passed V fft/inverse test\n");

  A2AvectorWfftw<A2Apolicies> Wfft(a2a_args,fp);
  Wfft.fft(W);

  A2AvectorW<A2Apolicies> Wrec(a2a_args,fp);
  Wfft.inversefft(Wrec);

  for(int i=0;i<W.getNl();i++){
    assert( Wrec.getWl(i).equals( W.getWl(i), 1e-08, true) ); 
  }
  if(!UniqueID()) printf("Passed Wl fft/inverse test\n"); 

  for(int i=0;i<W.getNhits();i++){
    assert( Wrec.getWh(i).equals( W.getWh(i), 1e-08, true) ); 
  }
  if(!UniqueID()) printf("Passed Wh fft/inverse test\n"); 
}

#ifdef USE_GRID
template<typename vComplexType, bool conj_left, bool conj_right>
class GridVectorizedSpinColorContractBasic{
public:
  inline static vComplexType g5(const vComplexType *const l, const vComplexType *const r){
    const static int sc_size =12;
    const static int half_sc = 6;

    vComplexType v3; zeroit(v3);

    for(int i = half_sc; i < sc_size; i++){ 
      v3 -= MconjGrid<vComplexType,conj_left,conj_right>::doit(l+i,r+i);
    }
    for(int i = 0; i < half_sc; i ++){ 
      v3 += MconjGrid<vComplexType,conj_left,conj_right>::doit(l+i,r+i);
    }
    return v3;
  }
};

template<typename T>
typename my_enable_if< is_complex_double_or_float<typename T::scalar_type>::value, bool>::type 
vTypeEquals(const T& a, const T &b, const double tolerance = 1e-12, bool verbose = false){
  typedef typename T::scalar_type S;
  int Nsimd = T::Nsimd();
  S ato[Nsimd];
  vstore(a,ato);
  S bto[Nsimd];
  vstore(b,bto);
  
  bool eq = true;
  for(int i=0;i<Nsimd;i++)
    if( fabs(ato[i].real() - bto[i].real()) > tolerance || fabs(ato[i].imag() - bto[i].imag()) > tolerance ){
      if(verbose && !UniqueID()){	
	double rdiff = fabs(ato[i].real() - bto[i].real());
	double idiff = fabs(ato[i].imag() - bto[i].imag());
	printf("Mismatch index %d: (%g,%g) vs (%g,%g) with diffs (%g,%g)\n",i,ato[i].real(),bto[i].real(),ato[i].imag(),bto[i].imag(),rdiff,idiff);
      }
      eq = false; break;
    }

  if(!eq && verbose && !UniqueID()){
    printf("NOT EQUAL:\n");
    printit(ato,Nsimd);
    printit(bto,Nsimd);
  }    
  return eq;
}

#endif

template<typename mf_Complex>
void testGridg5Contract(){
#ifdef USE_GRID
  Grid::Vector<mf_Complex> vec1(12);
  Grid::Vector<mf_Complex> vec2(12);
  for(int i=0;i<12;i++){
    vec1[i] = randomvType<mf_Complex>();
    vec2[i] = randomvType<mf_Complex>();
  }

  mf_Complex a = GridVectorizedSpinColorContractBasic<mf_Complex,true,false>::g5(vec1.data(),vec2.data());
  mf_Complex b = GridVectorizedSpinColorContract<mf_Complex,true,false>::g5(vec1.data(),vec2.data());
  assert(vTypeEquals(a,b,1e-6,true) == true);
  if(!UniqueID()){ printf("Passed g5 contract repro\n"); fflush(stdout); }
#endif
}

template<typename A2Apolicies>
void testVVdag(Lattice &lat){
  typedef typename A2Apolicies::FermionFieldType::FieldSiteType mf_Complex;
  typedef typename A2Apolicies::FermionFieldType::FieldMappingPolicy MappingPolicy;
  typedef typename A2Apolicies::FermionFieldType::FieldAllocPolicy AllocPolicy;

  typedef typename MappingPolicy::template Rebase<OneFlavorPolicy>::type OneFlavorMap;
  
  typedef CPSfermion4D<mf_Complex,OneFlavorMap, AllocPolicy> FieldType;
  
  typedef typename FieldType::InputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);
  
  if(lat.FixGaugeKind() == FIX_GAUGE_NONE){
    FixGaugeArg fix_gauge_arg;
    fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
    fix_gauge_arg.hyperplane_start = 0;
    fix_gauge_arg.hyperplane_step = 1;
    fix_gauge_arg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites();
    fix_gauge_arg.stop_cond = 1e-08;
    fix_gauge_arg.max_iter_num = 10000;

    CommonArg common_arg;
  
    AlgFixGauge fix_gauge(lat,&common_arg,&fix_gauge_arg);
    fix_gauge.run();
  }

  FieldType a(fp);
  a.testRandom();
  
  FieldType Va(a);
  Va.gaugeFix(lat,true,false); //parallel, no dagger

  printRow(a,0,"a");
  printRow(Va,0,"Va");
  
  FieldType VdagVa(Va);
  VdagVa.gaugeFix(lat,true,true);

  printRow(VdagVa,0,"VdagVa");

  assert( VdagVa.equals(a, 1e-8, true) );

  FieldType diff = VdagVa - a;
  printRow(diff,0,"diff");

  double n2 = diff.norm2();
  printf("Norm diff = %g\n",n2);

  FieldType zro(fp); zro.zero();

  assert( diff.equals(zro,1e-12,true));
}


template<typename ManualAllocA2Apolicies>
void testDestructiveFFT(const A2AArg &a2a_args,Lattice &lat){
  assert(GJP.Gparity());

  if(lat.FixGaugeKind() == FIX_GAUGE_NONE){
    FixGaugeArg fix_gauge_arg;
    fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
    fix_gauge_arg.hyperplane_start = 0;
    fix_gauge_arg.hyperplane_step = 1;
    fix_gauge_arg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites();
    fix_gauge_arg.stop_cond = 1e-08;
    fix_gauge_arg.max_iter_num = 10000;

    CommonArg common_arg;
  
    AlgFixGauge fix_gauge(lat,&common_arg,&fix_gauge_arg);
    fix_gauge.run();
  }
  typedef typename ManualAllocA2Apolicies::FermionFieldType FermionFieldType;
  typedef typename ManualAllocA2Apolicies::SourcePolicies SourcePolicies;
  typedef typename ManualAllocA2Apolicies::ComplexType mf_Complex;
  
  typedef typename A2AvectorWfftw<ManualAllocA2Apolicies>::FieldInputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);

  typedef typename ManualAllocA2Apolicies::SourcePolicies::MappingPolicy::ParamType SrcInputParamType;
  SrcInputParamType sp; defaultFieldParams<SrcInputParamType, mf_Complex>::get(sp);

  A2AvectorW<ManualAllocA2Apolicies> W(a2a_args,fp);
  A2AvectorV<ManualAllocA2Apolicies> V(a2a_args,fp);
  
  for(int i=0;i<V.getNmodes();i++) assert( &V.getMode(i) == NULL);
  V.allocModes();
  for(int i=0;i<V.getNmodes();i++) assert( &V.getMode(i) != NULL);
  
  V.testRandom();

  W.allocModes();
  for(int i=0;i<W.getNl();i++) assert( &W.getWl(i) != NULL);
  for(int i=0;i<W.getNhits();i++) assert( &W.getWh(i) != NULL);
  W.testRandom();

  
  A2AvectorV<ManualAllocA2Apolicies> Vcopy = V;
  A2AvectorW<ManualAllocA2Apolicies> Wcopy = W;
  
  int pp[3]; GparityBaseMomentum(pp,+1); //(1,1,1)
  int pm[3]; GparityBaseMomentum(pm,-1); //(-1,-1,-1)

  gaugeFixAndTwist<FermionFieldType> fft_op(pp,lat);  
  reverseGaugeFixAndTwist<FermionFieldType> invfft_op(pp,lat);
  
  A2AvectorVfftw<ManualAllocA2Apolicies> Vfft(a2a_args,fp); //no allocation yet performed
  Vfft.destructivefft(V, &fft_op);

  for(int i=0;i<V.getNmodes();i++) assert( &V.getMode(i) == NULL);
  for(int i=0;i<Vfft.getNmodes();i++) assert( &Vfft.getMode(i) != NULL);

  
  A2AvectorV<ManualAllocA2Apolicies> Vrec(a2a_args,fp);
  Vfft.destructiveInversefft(Vrec, &invfft_op);

  for(int i=0;i<Vrec.getNmodes();i++) assert( &Vrec.getMode(i) != NULL);
  for(int i=0;i<Vfft.getNmodes();i++) assert( &Vfft.getMode(i) == NULL); 

  for(int i=0;i<Vrec.getNmodes();i++) assert( Vrec.getMode(i).equals( Vcopy.getMode(i), 1e-08, true) );

  
  printf("Passed V destructive fft/inverse test\n");
   
  A2AvectorWfftw<ManualAllocA2Apolicies> Wfft(a2a_args,fp);
  Wfft.destructiveGaugeFixTwistFFT(W,pp,lat);

  for(int i=0;i<W.getNl();i++) assert( &W.getWl(i) == NULL);
  for(int i=0;i<W.getNhits();i++) assert( &W.getWh(i) == NULL);
  
  for(int i=0;i<Wfft.getNmodes();i++) assert( &Wfft.getMode(i) != NULL);
  
  A2AvectorW<ManualAllocA2Apolicies> Wrec(a2a_args,fp);
  Wfft.destructiveUnapplyGaugeFixTwistFFT(Wrec, pp,lat);
  
  for(int i=0;i<Wfft.getNmodes();i++) assert( &Wfft.getMode(i) == NULL);

  for(int i=0;i<Wrec.getNl();i++) assert( &Wrec.getWl(i) != NULL);
  for(int i=0;i<Wrec.getNhits();i++) assert( &Wrec.getWh(i) != NULL);
  
  for(int i=0;i<Wrec.getNl();i++){
    assert( Wrec.getWl(i).equals( Wcopy.getWl(i), 1e-08, true) ); 
  }
  if(!UniqueID()) printf("Passed Wl destructive fft/inverse test\n"); 

  for(int i=0;i<Wrec.getNhits();i++){
    assert( Wrec.getWh(i).equals( Wcopy.getWh(i), 1e-08, true) ); 
  }
  if(!UniqueID()) printf("Passed Wh destructive fft/inverse test\n"); 
  
  
}


void testA2AallocFree(const A2AArg &a2a_args,Lattice &lat){
#ifdef USE_GRID
  typedef A2ApoliciesSIMDdoubleManualAlloc A2Apolicies;
#else  
  typedef A2ApoliciesDoubleManualAlloc A2Apolicies;
#endif
  
  typedef typename A2Apolicies::FermionFieldType FermionFieldType;
  typedef typename A2Apolicies::SourcePolicies SourcePolicies;
  typedef typename A2Apolicies::ComplexType mf_Complex;
  
  typedef typename A2AvectorWfftw<A2Apolicies>::FieldInputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);

  A2AvectorVfftw<A2Apolicies> Vfft(a2a_args,fp);
  double size =  A2AvectorVfftw<A2Apolicies>::Mbyte_size(a2a_args,fp);
  
  for(int i=0;i<100;i++){
    if(!UniqueID()) printf("Pre-init\n");
    printMem(); fflush(stdout);

    if(!UniqueID()) printf("Expected size %f MB\n",size);
    
    if(!UniqueID()) printf("Post-init\n");
    printMem(); fflush(stdout);

    Vfft.allocModes();

    for(int i=0;i<Vfft.getNmodes();i++){
      assert(&Vfft.getMode(i) != NULL);
      Vfft.getMode(i).zero();
    }
    if(!UniqueID()) printf("Post-alloc\n");
    printMem(); fflush(stdout);

    Vfft.freeModes();

    for(int i=0;i<Vfft.getNmodes();i++)
      assert(&Vfft.getMode(i) == NULL);
    
    if(!UniqueID()) printf("Post-free\n");
    printMem(); fflush(stdout);
  }
}


template<typename GridA2Apolicies>
void benchmarkMFcontractKernel(const int ntests, const int nthreads){
#ifdef USE_GRID
  // GridVectorizedSpinColorContract benchmark
  typedef typename GridA2Apolicies::ComplexType GVtype;
  typedef typename GridA2Apolicies::ScalarComplexType GCtype;
  const int nsimd = GVtype::Nsimd();      

  FourDSIMDPolicy<OneFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<OneFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
  
  NullObject n;
  CPSfield<GCtype,12,FourDpolicy<OneFlavorPolicy> > a(n); a.testRandom();
  CPSfield<GCtype,12,FourDpolicy<OneFlavorPolicy> > b(n); b.testRandom();
  CPSfield<GVtype,12,FourDSIMDPolicy<OneFlavorPolicy>,Aligned128AllocPolicy> aa(simd_dims); aa.importField(a);
  CPSfield<GVtype,12,FourDSIMDPolicy<OneFlavorPolicy>,Aligned128AllocPolicy> bb(simd_dims); bb.importField(b);
  CPSfield<GVtype,1,FourDSIMDPolicy<OneFlavorPolicy>,Aligned128AllocPolicy> cc(simd_dims);

  int ntests_scaled = ntests;// * 1000;
  printf("Max threads %d\n",omp_get_max_threads());
#ifdef TIMERS_OFF
  printf("Timers are OFF\n"); fflush(stdout);
#else
  printf("Timers are ON\n"); fflush(stdout);
#endif
  __itt_resume();

  for(int oloop=0; oloop < 1; oloop++){
    double t0 = Grid::usecond();

#if 0
    
#pragma omp parallel //avoid thread creation overheads
    {
      int me = omp_get_thread_num();
      int work, off;
      thread_work(work, off, aa.nfsites(), me, omp_get_num_threads());
	
      GVtype *abase = aa.fsite_ptr(off);
      GVtype *bbase = bb.fsite_ptr(off);
      GVtype *cbase = cc.fsite_ptr(off);

      for(int test=0;test<ntests_scaled;test++){
	GVtype *ai = abase;
	GVtype *bi = bbase;
	GVtype *ci = cbase;
	__SSC_MARK(0x1);
	for(int i=0;i<work;i++){
	  *ci = GridVectorizedSpinColorContract<GVtype,true,false>::g5(ai,bi);
	  ai += 12;
	  bi += 12;
	  ci += 1;
	}
	__SSC_MARK(0x2);
      }
    }


#else
    //Should operate entirely out of GPU memoru
    size_t work = aa.nfsites();
    static const int Nsimd = GVtype::Nsimd();

    size_t site_size_ab = aa.siteSize();
    size_t site_size_c = cc.siteSize();
    
    GVtype const* adata = aa.ptr();
    GVtype const* bdata = bb.ptr();
    GVtype * cdata = cc.ptr();

    for(int test=0;test<ntests_scaled;test++){   
      {
	using namespace Grid;
	if(test == ntests_scaled -1) cudaProfilerStart();

	accelerator_for(item, work, Nsimd, 
			{
			  size_t x = item;
			  GVtype const* ax = adata + site_size_ab*x;
			  GVtype const* bx = bdata + site_size_ab*x;
			  GVtype *cx = cdata + site_size_c*x;
			
			  typename SIMT<GVtype>::value_type v = GridVectorizedSpinColorContract<GVtype,true,false>::g5(ax,bx);

			  SIMT<GVtype>::write(*cx, v);			  
			});
	if(test == ntests_scaled -1) cudaProfilerStop();
      }   
    }    

#endif

    

    double t1 = Grid::usecond();
    double dt = t1 - t0;
      
    int FLOPs = 12*6*nsimd //12 vectorized conj(a)*b
      + 12*2*nsimd; //12 vectorized += or -=
    double call_FLOPs = double(FLOPs) * double(aa.nfsites());

    double call_time_us = dt/ntests_scaled;
    double call_time_s = dt/ntests_scaled /1e6;
    
    double flops = call_FLOPs/call_time_us; //dt in us   dt/(1e-6 s) in Mflops

    double bytes_read = 2* 12 * 2*8 * nsimd; 
    double bytes_store = 2 * nsimd;

    double call_bytes = (bytes_read + bytes_store) * double(aa.nfsites());
        
    double bandwidth = call_bytes/call_time_s / 1024./1024.; // in MB/s

    double FLOPS_per_byte = FLOPs/(bytes_read + bytes_store);
    double theor_perf = FLOPS_per_byte * bandwidth; //in Mflops (assuming bandwidth bound)
    
    std::cout << "GridVectorizedSpinColorContract( conj(a)*b ): New code " << ntests_scaled << " tests over " << nthreads << " threads: Time " << dt << " usecs  flops " << flops/1e3 << " Gflops\n";
    std::cout << "Time per call " << call_time_us << " usecs" << std::endl;
    std::cout << "Memory bandwidth " << bandwidth << " MB/s" << std::endl;
    std::cout << "FLOPS/byte " << FLOPS_per_byte << std::endl;
    std::cout << "Theoretical performance " << theor_perf/1e3 << " Gflops\n";    
    std::cout << "Total work is " << work << " and Nsimd = " << Nsimd << std::endl;
  }
  __itt_detach();
#endif
}


template<typename ScalarA2Apolicies>
void testMesonFieldReadWrite(const A2AArg &a2a_args){
  A2AvectorWfftw<ScalarA2Apolicies> W(a2a_args);
  A2AvectorVfftw<ScalarA2Apolicies> V(a2a_args);
  
  A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf;
  mf.testRandom();
  
  {
    mf.write("mesonfield.dat",FP_IEEE64BIG);
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfr;
    mfr.read("mesonfield.dat");
    assert( mfr.equals(mf,1e-18,true));
    if(!UniqueID()) printf("Passed mf single IO test\n");
  }
  {
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfa;
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfb;
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfc;
    mfa.setup(W,V,0,0);
    mfb.setup(W,V,1,1);
    mfc.setup(W,V,2,2);		
      
    mfa.testRandom();
    mfb.testRandom();
    mfc.testRandom();

    std::ofstream *fp = !UniqueID() ? new std::ofstream("mesonfield_many.dat") : NULL;

    mfa.write(fp,FP_IEEE64BIG);
    mfb.write(fp,FP_IEEE64LITTLE);
    mfc.write(fp,FP_IEEE64BIG);

    if(!UniqueID()) fp->close();

    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfra;
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfrb;
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mfrc;

    std::ifstream *ifp = !UniqueID() ? new std::ifstream("mesonfield_many.dat") : NULL;

    mfra.read(ifp);
    mfrb.read(ifp);
    mfrc.read(ifp);

    if(!UniqueID()) ifp->close();

    assert( mfra.equals(mfa,1e-18,true) );
    assert( mfrb.equals(mfb,1e-18,true) );
    assert( mfrc.equals(mfc,1e-18,true) );
    if(!UniqueID()) printf("Passed mf multi IO test\n");
  }
  {
    std::vector< A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mfv(3);
    for(int i=0;i<3;i++){
      mfv[i].setup(W,V,i,i);
      mfv[i].testRandom();
    }
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::write("mesonfield_vec.dat", mfv, FP_IEEE64LITTLE);
	
    std::vector< A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mfrv;
    A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::read("mesonfield_vec.dat", mfrv);

    for(int i=0;i<3;i++)
      assert( mfrv[i].equals(mfv[i], 1e-18, true) );
    if(!UniqueID()) printf("Passed mf vector IO test\n");
  }
}	


template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void testMFcontract(const A2AArg &a2a_args, const int nthreads, const double tol){
 
#ifdef USE_GRID
  std::cout << "Starting MF contraction test\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorWfftw<ScalarA2Apolicies> W(a2a_args);
  A2AvectorVfftw<ScalarA2Apolicies> V(a2a_args);
    
  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);
  
  std::vector<A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_grid;
  
  typedef typename GridA2Apolicies::ComplexType grid_Complex;
  typedef typename ScalarA2Apolicies::ComplexType mf_Complex;

  typedef typename GridA2Apolicies::ScalarComplexType Ctype;
  typedef typename Ctype::value_type Ftype;
  
  ThreeDSIMDPolicy<OneFlavorPolicy>::ParamType simd_dims_3d;
  ThreeDSIMDPolicy<OneFlavorPolicy>::SIMDdefaultLayout(simd_dims_3d,nsimd);

  typedef typename GridA2Apolicies::SourcePolicies GridSrcPolicy;    
  int p[3] = {1,1,1};
  A2AflavorProjectedExpSource<GridSrcPolicy> src_grid(2.0,p,simd_dims_3d);
  typedef SCFspinflavorInnerProduct<15,typename GridA2Apolicies::ComplexType,A2AflavorProjectedExpSource<GridSrcPolicy> > GridInnerProduct;
  GridInnerProduct mf_struct_grid(sigma3,src_grid);


  //typedef GparityNoSourceInnerProduct<typename GridA2Apolicies::ComplexType, flavorMatrixSpinColorContract<15,true,false> > GridInnerProduct;
  //GridInnerProduct mf_struct_grid(sigma3);
  
  
  A2AflavorProjectedExpSource<> src(2.0,p);
  typedef SCFspinflavorInnerProduct<15,typename ScalarA2Apolicies::ComplexType,A2AflavorProjectedExpSource<> > StdInnerProduct;
  StdInnerProduct mf_struct(sigma3,src);

  W.testRandom();
  V.testRandom();
  Wgrid.importFields(W);
  Vgrid.importFields(V);
  
#ifndef GRID_NVCC
  //Original Grid implementation
  {
    typedef typename std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> >::allocator_type Allocator;
    typedef SingleSrcVectorPoliciesSIMD<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,GridInnerProduct> VectorPolicies;
    mfComputeGeneral<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw, GridInnerProduct, VectorPolicies> cg;
    cg.compute(mf_grid,Wgrid,mf_struct_grid,Vgrid, true);
  }
#else
  {
    typedef typename std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> >::allocator_type Allocator;
    typedef SingleSrcVectorPoliciesSIMDoffload<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,GridInnerProduct> VectorPolicies;
    mfComputeGeneralOffload<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw, GridInnerProduct, VectorPolicies> cg;
    cg.compute(mf_grid,Wgrid,mf_struct_grid,Vgrid, true);
  }
#endif
  
  A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf,W,mf_struct,V);

  bool fail = false;
  for(int t=0;t<mf.size();t++){
    for(int i=0;i<mf[t].size();i++){
      const Ctype& gd = mf_grid[t].ptr()[i];
      const Ctype& cp = mf[t].ptr()[i];
      Ftype rdiff = fabs(gd.real()-cp.real());
      Ftype idiff = fabs(gd.imag()-cp.imag());
      if(rdiff > tol|| idiff > tol){
	printf("Fail: t %d idx %d Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",t, i,gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
	fail = true;
      }
    }
  }
  if(fail) ERR.General("","","Standard vs Grid implementation test failed\n");
  else if(!UniqueID()){ printf("Passed MF contraction test\n"); fflush(stdout); }
#endif
}



template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void benchmarkMFcontract(const A2AArg &a2a_args, const int ntests, const int nthreads){
#ifdef USE_GRID
  typedef typename GridA2Apolicies::SourcePolicies GridSrcPolicy;
  typedef typename ScalarA2Apolicies::ScalarComplexType Ctype;
  typedef typename Ctype::value_type Ftype;

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);

  A2AvectorWfftw<ScalarA2Apolicies> W(a2a_args);
  A2AvectorVfftw<ScalarA2Apolicies> V(a2a_args);
  
  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);
  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_grid;

  ThreeDSIMDPolicy<OneFlavorPolicy>::ParamType simd_dims_3d;
  ThreeDSIMDPolicy<OneFlavorPolicy>::SIMDdefaultLayout(simd_dims_3d,nsimd);

  printf("Nsimd = %d, SIMD dimensions:\n", nsimd);
  for(int i=0;i<4;i++)
    printf("%d ", simd_dims[i]);
  printf("\n");
  
  int p[3] = {1,1,1};
  A2AflavorProjectedExpSource<GridSrcPolicy> src_grid(2.0,p,simd_dims_3d);
  //typedef SCFspinflavorInnerProductCT<15,sigma3,typename GridA2Apolicies::ComplexType,A2AflavorProjectedExpSource<GridSrcPolicy> > GridInnerProduct;
  //GridInnerProduct mf_struct_grid(src_grid);

  typedef SCFspinflavorInnerProduct<15,typename GridA2Apolicies::ComplexType,A2AflavorProjectedExpSource<GridSrcPolicy> > GridInnerProduct;
  GridInnerProduct mf_struct_grid(sigma3,src_grid);
  
  std::cout << "Starting all-time mesonfield contract benchmark\n";
  if(!UniqueID()) printf("Using outer blocking bi %d bj %d bp %d\n",BlockedMesonFieldArgs::bi,BlockedMesonFieldArgs::bj,BlockedMesonFieldArgs::bp);
  if(!UniqueID()) printf("Using inner blocking bi %d bj %d bp %d\n",BlockedMesonFieldArgs::bii,BlockedMesonFieldArgs::bjj,BlockedMesonFieldArgs::bpp);

  Float total_time = 0.;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_grid_t;

  W.testRandom();
  V.testRandom();
  Wgrid.importFields(W);
  Vgrid.importFields(V);
      
  CALLGRIND_START_INSTRUMENTATION ;
  CALLGRIND_TOGGLE_COLLECT ;
  
  typedef typename std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> >::allocator_type Allocator;

#ifndef GRID_NVCC
    typedef SingleSrcVectorPoliciesSIMD<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,GridInnerProduct> VectorPolicies;
  //typedef SingleSrcVectorPolicies<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,GridInnerProduct> VectorPolicies;
    mfComputeGeneral<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw, GridInnerProduct, VectorPolicies> cg;
#else
  typedef SingleSrcVectorPoliciesSIMDoffload<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,GridInnerProduct> VectorPolicies;
  mfComputeGeneralOffload<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw, GridInnerProduct, VectorPolicies> cg;
#endif

  BlockedMesonFieldArgs::enable_profiling = false; 
  
  ProfilerStart("SingleSrcProfile.prof");
  for(int iter=0;iter<ntests;iter++){
    total_time -= dclock();

    //__itt_resume();
    if(iter == ntests-1) BlockedMesonFieldArgs::enable_profiling = true;
    cg.compute(mf_grid_t,Wgrid,mf_struct_grid,Vgrid, true);
    if(iter == ntests-1) BlockedMesonFieldArgs::enable_profiling = false;
    //__itt_pause();

    //A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_grid_t,Wgrid,mf_struct_grid,Vgrid);
    total_time += dclock();
  }
  __itt_detach();
  ProfilerStop();  

  CALLGRIND_TOGGLE_COLLECT ;
  CALLGRIND_STOP_INSTRUMENTATION ;

  int g5_FLOPs = 12*6*nsimd + 12*2*nsimd;//4 flav * 12 vectorized conj(a)*b  + 12 vectorized += or -=         
  int siteFmat_FLOPs = 3*nsimd;  //1 vectorized z.im*-1, 1 vectorized -1*z
  int s3_FLOPs = 4*nsimd; //2 vectorized -1*z
  int TransLeftTrace_FLOPs = nsimd*4*6 + nsimd*3*2; //4 vcmul + 3vcadd
  int reduce_FLOPs = nsimd*2; //nsimd cadd  (reduce over lanes and sites)
  
  double FLOPs_per_site = 0.;
  for(int t=GJP.TnodeCoor()*GJP.TnodeSites(); t<(GJP.TnodeCoor()+1)*GJP.TnodeSites(); t++){
    const int nl_l = mf_grid_t[t].getRowParams().getNl();
    const int nl_r = mf_grid_t[t].getColParams().getNl();

    int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();

    for(int i = 0; i < mf_grid_t[t].getNrows(); i++){
      modeIndexSet i_high_unmapped; if(i>=nl_l) mf_grid_t[t].getRowParams().indexUnmap(i-nl_l,i_high_unmapped);
      SCFvectorPtr<typename GridA2Apolicies::FermionFieldType::FieldSiteType> lscf = Wgrid.getFlavorDilutedVect(i,i_high_unmapped,0,t_lcl); //dilute flavor in-place if it hasn't been already
                                                                                                                                                                       
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
#endif
}









template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void benchmarkMultiSrcMFcontract(const A2AArg &a2a_args, const int ntests, const int nthreads){
#ifdef USE_GRID
  typedef typename GridA2Apolicies::SourcePolicies GridSrcPolicy;
  typedef typename ScalarA2Apolicies::ScalarComplexType Ctype;
  typedef typename Ctype::value_type Ftype;

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);

  A2AvectorWfftw<ScalarA2Apolicies> W(a2a_args);
  A2AvectorVfftw<ScalarA2Apolicies> V(a2a_args);
  
  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);
  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_grid;

  ThreeDSIMDPolicy<OneFlavorPolicy>::ParamType simd_dims_3d;
  ThreeDSIMDPolicy<OneFlavorPolicy>::SIMDdefaultLayout(simd_dims_3d,nsimd);

  printf("Nsimd = %d, SIMD dimensions:\n", nsimd);
  for(int i=0;i<4;i++)
    printf("%d ", simd_dims[i]);
  printf("\n");
  
  int p[3] = {1,1,1};

  typedef typename GridA2Apolicies::ComplexType ComplexType;  
  typedef A2AflavorProjectedExpSource<GridSrcPolicy> ExpSrcType;
  typedef A2AflavorProjectedHydrogenSource<GridSrcPolicy> HydSrcType;
  typedef Elem<ExpSrcType, Elem<HydSrcType,ListEnd > > SrcList;
  typedef A2AmultiSource<SrcList> MultiSrcType;
  typedef GparitySourceShiftInnerProduct<ComplexType,MultiSrcType, flavorMatrixSpinColorContract<15,true,false> > MultiInnerType;

  const double rad = 2.0;
  MultiSrcType src;
  src.template getSource<0>().setup(rad,p,simd_dims_3d); //1s
  src.template getSource<1>().setup(2,0,0,rad,p,simd_dims_3d); //2s

  MultiInnerType g5_s3_inner(sigma3, src);
  std::vector<std::vector<int> > shifts(1, std::vector<int>(3,0));
  g5_s3_inner.setShifts(shifts);
  
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();
  
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_exp;
  std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_hyd;

  std::vector< std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> >* > mf_st(2);
  mf_st[0] = &mf_exp;
  mf_st[1] = &mf_hyd;
  
  std::cout << "Starting all-time mesonfield contract benchmark with multi-src (1s, 2s hyd)\n";
  if(!UniqueID()) printf("Using outer blocking bi %d bj %d bp %d\n",BlockedMesonFieldArgs::bi,BlockedMesonFieldArgs::bj,BlockedMesonFieldArgs::bp);
  if(!UniqueID()) printf("Using inner blocking bi %d bj %d bp %d\n",BlockedMesonFieldArgs::bii,BlockedMesonFieldArgs::bjj,BlockedMesonFieldArgs::bpp);

  Float total_time = 0.;

  W.testRandom();
  V.testRandom();
  Wgrid.importFields(W);
  Vgrid.importFields(V);
      
  CALLGRIND_START_INSTRUMENTATION ;
  CALLGRIND_TOGGLE_COLLECT ;
  
  typedef typename std::vector<A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> >::allocator_type Allocator;

#ifndef GRID_NVCC
  typedef MultiSrcVectorPoliciesSIMD<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,MultiInnerType> VectorPolicies;
  //typedef MultiSrcVectorPolicies<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,MultiInnerType> VectorPolicies;
  mfComputeGeneral<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw, MultiInnerType, VectorPolicies> cg;
#else
  typedef MultiSrcVectorPoliciesSIMDoffload<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw,Allocator,MultiInnerType> VectorPolicies;
  mfComputeGeneralOffload<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw, MultiInnerType, VectorPolicies> cg;
#endif  

  ProfilerStart("MultiSrcProfile.prof");
  for(int iter=0;iter<ntests;iter++){
    total_time -= dclock();

    __itt_resume();
    cg.compute(mf_st,Wgrid,g5_s3_inner,Vgrid, true);
    __itt_pause();
    
    //A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_grid_t,Wgrid,mf_struct_grid,Vgrid);
    total_time += dclock();
  }
  ProfilerStop();
  __itt_detach();


  CALLGRIND_TOGGLE_COLLECT ;
  CALLGRIND_STOP_INSTRUMENTATION ;

  int nsrc = 2;
  
  int g5_FLOPs = 12*6*nsimd + 12*2*nsimd;//4 flav * 12 vectorized conj(a)*b  + 12 vectorized += or -=
  int siteFmat_FLOPs = nsrc*3*nsimd;  //1 vectorized z.im*-1, 1 vectorized -1*z
  int s3_FLOPs = nsrc*4*nsimd; //2 vectorized -1*z
  int TransLeftTrace_FLOPs = nsrc*nsimd*4*6 + nsrc*nsimd*3*2; //4 vcmul + 3vcadd
  int reduce_FLOPs = 0; // (nsimd - 1)*2; //nsimd-1 cadd

  double FLOPs_per_site = 0.;
  for(int t=GJP.TnodeCoor()*GJP.TnodeSites(); t<(GJP.TnodeCoor()+1)*GJP.TnodeSites(); t++){
    const int nl_l = mf_exp[t].getRowParams().getNl();
    const int nl_r = mf_exp[t].getColParams().getNl();

    int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();

    for(int i = 0; i < mf_exp[t].getNrows(); i++){
      modeIndexSet i_high_unmapped; if(i>=nl_l) mf_exp[t].getRowParams().indexUnmap(i-nl_l,i_high_unmapped);
      SCFvectorPtr<typename GridA2Apolicies::FermionFieldType::FieldSiteType> lscf = Wgrid.getFlavorDilutedVect(i,i_high_unmapped,0,t_lcl); //dilute flavor in-place if it hasn't been already \
                                                                                                                                                                                                           
      for(int j = 0; j < mf_exp[t].getNcols(); j++) {
	modeIndexSet j_high_unmapped; if(j>=nl_r) mf_exp[t].getColParams().indexUnmap(j-nl_r,j_high_unmapped);
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

  printf("MF contract all t multi-src: Avg time new code %d iters: %g secs. Avg flops %g Gflops\n",ntests,total_time/ntests, total_FLOPs/total_time/1e9);
#endif
}




template<typename A2Apolicies>
void testTraceSingle(const A2AArg &a2a_args, const double tol){
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_grid;
  mf_grid.setup(a2a_args,a2a_args,0,0);

  LRG.AssignGenerator(0); //always uses the RNG at coord 0 on node 0 - should always be the same one!
  mf_grid.testRandom();

  typedef typename A2Apolicies::ScalarComplexType mf_Complex;  
  mf_Complex fast = trace(mf_grid);
  mf_Complex slow = trace_slow(mf_grid);

  bool fail = false;
  if(!UniqueID()) printf("Trace Fast (%g,%g) Slow (%g,%g) Diff (%g,%g)\n",fast.real(),fast.imag(), slow.real(),slow.imag(), fast.real()-slow.real(), fast.imag()-slow.imag());
  double rdiff = fabs(fast.real()-slow.real());
  double idiff = fabs(fast.imag()-slow.imag());
  if(rdiff > tol|| idiff > tol){
    fail = true;
  }
  if(fail) ERR.General("","","MF single trace test failed\n");
  else if(!UniqueID()) printf("MF single trace pass\n");

  //Manually test node number independence of the node distributed trace
  std::vector<typename A2Apolicies::ScalarComplexType> into;
  std::vector<A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > m(3);
  for(int i=0;i<m.size();i++){
    m[i].setup(a2a_args,a2a_args,0,0);
    LRG.AssignGenerator(0);
    m[i].testRandom();
  }
  trace(into,m);

  if(!UniqueID()){
    printf("Distributed traces:");
    for(int i=0;i<into.size();i++){
      printf(" (%g,%g)",into[i].real(),into[i].imag());
    }
    printf("\n");
  } 
}


template<typename A2Apolicies>
void benchmarkMFmult(const A2AArg &a2a_args, const int ntests){
  typedef A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_WV; 
  mf_WV l;
  l.setup(a2a_args,a2a_args,0,0);
  l.testRandom();  

  int nodes = 1; for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);

  if(!UniqueID()) printf("mf_WV sizes %d %d. Using %d threads\n",l.getNrows(),l.getNcols(), omp_get_max_threads());

  mf_WV r;
  r.setup(a2a_args,a2a_args,1,1);
  r.testRandom();  

  const size_t ni = l.getNrows();
  const size_t nk = r.getNcols();

  typedef typename mf_WV::RightDilutionType ConLeftDilutionType;
  typedef typename mf_WV::LeftDilutionType ConRightDilutionType;

  ModeContractionIndices<ConLeftDilutionType,ConRightDilutionType> ind(l.getColParams());
    
  modeIndexSet lmodeparams; lmodeparams.time = l.getColTimeslice();
  modeIndexSet rmodeparams; rmodeparams.time = r.getRowTimeslice();
    
  const size_t nj = ind.getNindices(lmodeparams,rmodeparams);

  //zmul 6 Flops
  //zmadd 8 Flops
  //zvecdot (N) = 6 + (N-1)*8 Flops

  size_t Flops = ni * nk * ( 6 + (nj-1)*8 );
  double time, Mflops, Mflops_per_node;

  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> c;

  time = -dclock();
  for(int i=0;i<ntests;i++){
    mult(c, l, r, true); //NODE LOCAL, used in pipi
  }
  time += dclock();

  Mflops = double(Flops)/time*double(ntests)/double(1.e6);

  if(!UniqueID()) printf("MF mult node local (ni=%d nj=%d nk=%d) %f Mflops\n",ni,nj,nk,Mflops);

#ifdef MULT_IMPL_CUBLASXT
  if(!UniqueID()) _mult_impl_base::getTimers().print();
  _mult_impl_base::getTimers().reset();
#endif

  time = -dclock();
  for(int i=0;i<ntests;i++){
    mult(c, l, r, false); //NODE DISTRIBUTED, used in K->pipi
  }
  time += dclock();

  Mflops = double(Flops)/time*double(ntests)/double(1.e6);
  Mflops_per_node = Mflops/nodes;
  
  if(!UniqueID()) printf("MF mult node distributed (ni=%d nj=%d nk=%d) %f Mflops,  %f Mflops/node\n",ni,nj,nk,Mflops, Mflops_per_node);

#ifdef MULT_IMPL_CUBLASXT
  if(!UniqueID()) _mult_impl_base::getTimers().print();
#endif


  //////////////////////////////////////////
#ifdef MULT_IMPL_GSL

  time = -dclock();
  for(int i=0;i<ntests;i++){
    _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_orig(c, l, r, true); //NODE LOCAL, used in pipi
  }
  time += dclock();

  Mflops = double(Flops)/time*double(ntests)/double(1.e6);

  if(!UniqueID()) printf("MF mult_orig node local (ni=%d nj=%d nk=%d) %f Mflops\n",ni,nj,nk,Mflops);

  time = -dclock();
  for(int i=0;i<ntests;i++){
    _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_orig(c, l, r, false); //NODE DISTRIBUTED, used in K->pipi
  }
  time += dclock();

  Mflops = double(Flops)/time*double(ntests)/double(1.e6);
  Mflops_per_node = Mflops/nodes;
  
  if(!UniqueID()) printf("MF mult_orig node distributed (ni=%d nj=%d nk=%d) %f Mflops,  %f Mflops/node\n",ni,nj,nk,Mflops, Mflops_per_node);


  ////////////////////////////////////////////
  time = -dclock();
  for(int i=0;i<ntests;i++){
    _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt1(c, l, r, true); //NODE LOCAL, used in pipi
  }
  time += dclock();

  Mflops = double(Flops)/time*double(ntests)/double(1.e6);

  if(!UniqueID()) printf("MF mult_opt1 node local (ni=%d nj=%d nk=%d) %f Mflops\n",ni,nj,nk,Mflops);

  time = -dclock();
  for(int i=0;i<ntests;i++){
    _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt1(c, l, r, false); //NODE DISTRIBUTED, used in K->pipi
  }
  time += dclock();

  Mflops = double(Flops)/time*double(ntests)/double(1.e6);
  Mflops_per_node = Mflops/nodes;
  
  if(!UniqueID()) printf("MF mult_opt1 node distributed (ni=%d nj=%d nk=%d) %f Mflops,  %f Mflops/node\n",ni,nj,nk,Mflops, Mflops_per_node);


  ////////////////////////////////////////////
  time = -dclock();
  for(int i=0;i<ntests;i++){
    _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt2(c, l, r, true); //NODE LOCAL, used in pipi
  }
  time += dclock();

  Mflops = double(Flops)/time*double(ntests)/double(1.e6);

  if(!UniqueID()) printf("MF mult_opt2 node local (ni=%d nj=%d nk=%d) %f Mflops\n",ni,nj,nk,Mflops);

  time = -dclock();
  for(int i=0;i<ntests;i++){
    _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt2(c, l, r, false); //NODE DISTRIBUTED, used in K->pipi
  }
  time += dclock();

  Mflops = double(Flops)/time*double(ntests)/double(1.e6);
  Mflops_per_node = Mflops/nodes;
  
  if(!UniqueID()) printf("MF mult_opt2 node distributed (ni=%d nj=%d nk=%d) %f Mflops,  %f Mflops/node\n",ni,nj,nk,Mflops, Mflops_per_node);
#endif //MULT_IMPL_GSL
}




void testCPSfieldImpex(){
  { //4D fields
    typedef CPSfermion4D<cps::ComplexD> CPSfermion4DBasic;
    CPSfermion4DBasic a;
    a.testRandom();
    
    {
      CPSfermion4DBasic b;
      a.exportField(b);
      CPSfermion4DBasic c;
      c.importField(b);
      assert( a.equals(c) );
    }

    {
      CPSfermion4DBasic b_odd, b_even;
      IncludeCBsite<4> odd_mask(1);
      IncludeCBsite<4> even_mask(0);
      a.exportField(b_odd, &odd_mask);
      a.exportField(b_even, &even_mask);
          
      CPSfermion4DBasic c;
      c.importField(b_odd, &odd_mask);
      c.importField(b_even, &even_mask);

      assert( a.equals(c) );
    }

#ifdef USE_GRID
    typedef CPSfermion4D<Grid::vComplexD, FourDSIMDPolicy<DynamicFlavorPolicy>,Aligned128AllocPolicy> CPSfermion4DGrid;
    typedef typename CPSfermion4DGrid::InputParamType CPSfermion4DGridParams;
    CPSfermion4DGridParams gp;
    setupFieldParams<CPSfermion4DGrid>(gp);
    
    {
      CPSfermion4DGrid b(gp);
      a.exportField(b);
      
      CPSfermion4DBasic c;
      c.importField(b);
      
      assert( a.equals(c) );
    }
    {
      CPSfermion4DGrid b(gp);
      b.importField(a);
      
      CPSfermion4DBasic c;
      b.exportField(c);
      
      assert( a.equals(c) );
    }

    {
      CPSfermion4DGrid b_odd(gp), b_even(gp);
      IncludeCBsite<4> odd_mask(1);
      IncludeCBsite<4> even_mask(0);
      a.exportField(b_odd, &odd_mask);
      a.exportField(b_even, &even_mask);
          
      CPSfermion4DBasic c;
      c.importField(b_odd, &odd_mask);
      c.importField(b_even, &even_mask);

      assert( a.equals(c) );
    }
#endif //USE_GRID
    
  }

  

  { //5D fields
    typedef CPSfermion5D<cps::ComplexD> CPSfermion5DBasic;
    CPSfermion5DBasic a;
    a.testRandom();

    {
      CPSfermion5DBasic b;
      a.exportField(b);
      CPSfermion5DBasic c;
      c.importField(b);
      assert( a.equals(c) );
    }

    {
      CPSfermion5DBasic b_odd, b_even;
      IncludeCBsite<5> odd_mask(1); //4d prec
      IncludeCBsite<5> even_mask(0);
      a.exportField(b_odd, &odd_mask);
      a.exportField(b_even, &even_mask);
          
      CPSfermion5DBasic c;
      c.importField(b_odd, &odd_mask);
      c.importField(b_even, &even_mask);

      assert( a.equals(c) );
    }

    {//The reduced size checkerboarded fields
      CPSfermion5Dcb4Dodd<cps::ComplexD> b_odd;
      CPSfermion5Dcb4Deven<cps::ComplexD> b_even;

      IncludeCBsite<5> odd_mask(1); //4d prec
      IncludeCBsite<5> even_mask(0);
      a.exportField(b_odd, &odd_mask);
      a.exportField(b_even, &even_mask);
          
      CPSfermion5DBasic c;
      c.importField(b_odd, &odd_mask);
      c.importField(b_even, &even_mask); //shouldn't need mask because only the cb sites are contained in the imported field but it disables the site number check

      assert( a.equals(c) );
    }    
  }//end of 5d field testing
}

#ifdef USE_GRID







template<typename GridA2Apolicies>
void testGridFieldImpex(typename GridA2Apolicies::FgridGFclass &lattice){

  { //test my peek poke
    typedef Grid::iVector<Grid::iScalar<Grid::vRealD>, 3> vtype;
    typedef typename Grid::GridTypeMapper<vtype>::scalar_object stype;
    typedef typename Grid::GridTypeMapper<vtype>::scalar_type rtype;

    const int Nsimd = vtype::vector_type::Nsimd();
    
    vtype* vp = (vtype*)memalign(128,sizeof(vtype));
    stype* sp = (stype*)memalign(128,Nsimd*sizeof(stype));
    stype* sp2 = (stype*)memalign(128,Nsimd*sizeof(stype));

    for(int i=0;i<Nsimd;i++)
      for(int j=0;j<sizeof(stype)/sizeof(rtype);j++)
	(  (rtype*)(sp+i) )[j] = rtype(j+Nsimd*i);

    std::cout << "Poking:\n";
    for(int i=0;i<Nsimd;i++) std::cout << sp[i] << std::endl;

    
    for(int lane=0;lane<Nsimd;lane++)
      pokeLane(*vp, sp[lane], lane);

    
    std::cout << "\nAfter poke: " << *vp << std::endl;


    std::cout << "Peeked:\n";    
    for(int lane=0;lane<Nsimd;lane++){
      peekLane(sp[lane], *vp, lane);
      std::cout << sp[lane] << std::endl;
    }
  }


  
  Grid::GridCartesian *FGrid = lattice.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();
  Grid::GridCartesian *UGrid = lattice.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();

  typedef typename GridA2Apolicies::GridFermionField GridFermionField;

  typedef CPSfermion5D<cps::ComplexD> CPSfermion5DBasic;
  CPSfermion5DBasic a;
  a.testRandom();

  GridFermionField a_grid(FGrid);
  a.exportGridField(a_grid);

  {
    CPSfermion5DBasic b;
    b.importGridField(a_grid);
    assert(b.equals(a));    
  }

  {
    CPSfermion5DBasic b_odd, b_even;
    IncludeCBsite<5> odd_mask(1); //4d prec
    IncludeCBsite<5> even_mask(0);
    b_odd.importGridField(a_grid, &odd_mask);
    b_even.importGridField(a_grid, &even_mask);
          
    CPSfermion5DBasic c;
    c.importField(b_odd, &odd_mask);
    c.importField(b_even, &even_mask);

    assert( a.equals(c) );
  }

  
  {//The reduced size checkerboarded fields
    CPSfermion5Dcb4Dodd<cps::ComplexD> b_odd;
    CPSfermion5Dcb4Deven<cps::ComplexD> b_even;

    IncludeCBsite<5> odd_mask(1); //4d prec
    IncludeCBsite<5> even_mask(0);
    b_odd.importGridField(a_grid, &odd_mask);
    b_even.importGridField(a_grid, &even_mask);
          
    CPSfermion5DBasic c;
    c.importField(b_odd, &odd_mask);
    c.importField(b_even, &even_mask); //shouldn't need mask because only the cb sites are contained in the imported field but it disables the site number check

    assert( a.equals(c) );
  }    
    
  
  

}



#endif //USE_GRID




  
void testCPSfieldIO(){
  if(!UniqueID()) printf("testCPSfieldIO called\n");

  CPSfield_checksumType cksumtype[2] = { checksumBasic, checksumCRC32 };
  FP_FORMAT fileformat[2] = { FP_IEEE64BIG, FP_IEEE64LITTLE };
  
  for(int i=0;i<2;i++){
    for(int j=0;j<2;j++){
      {
	CPSfermion4D<cps::ComplexD> a;
	a.testRandom();
    
	a.writeParallel("field", fileformat[j], cksumtype[i]);
    
	CPSfermion4D<cps::ComplexD> b;
	b.readParallel("field");
    
	assert( a.equals(b) );
      }
#ifdef USE_GRID
      {
	//Native write with SIMD intact
	typedef CPSfield<Grid::vComplexD,12,FourDSIMDPolicy<DynamicFlavorPolicy>,Aligned128AllocPolicy> GridFieldType;
	typedef CPSfield<cps::ComplexD,12,FourDpolicy<DynamicFlavorPolicy> > ScalarFieldType;
	typedef GridFieldType::InputParamType ParamType;

	ParamType params;
	GridFieldType::SIMDdefaultLayout(params, Grid::vComplexD::Nsimd());
	
	GridFieldType a(params);
	a.testRandom();

	a.writeParallel("field_simd", fileformat[j], cksumtype[i]);
    
	GridFieldType b(params);
	b.readParallel("field_simd");
    
	assert( a.equals(b) );

	//Impex to non-SIMD
	NullObject null;
	ScalarFieldType c(null);
	c.importField(a);

	c.writeParallel("field_scalar", fileformat[j], cksumtype[i]);

	ScalarFieldType d(null);
	d.readParallel("field_scalar");
	b.importField(d);
	
	assert( a.equals(b) );
      }
#endif
      
      {
	CPScomplex4D<cps::ComplexD> a;
	a.testRandom();
    
	a.writeParallel("field", fileformat[j], cksumtype[i]);
    
	CPScomplex4D<cps::ComplexD> b;
	b.readParallel("field");
    
	assert( a.equals(b) );
      }

      {
	typedef CPSfermion5D<cps::ComplexD> CPSfermion5DBasic;
	CPSfermion5DBasic a;
	a.testRandom();

	CPSfermion5Dcb4Dodd<cps::ComplexD> b_odd;
	CPSfermion5Dcb4Deven<cps::ComplexD> b_even;
    
	IncludeCBsite<5> odd_mask(1); //4d prec
	IncludeCBsite<5> even_mask(0);
	a.exportField(b_odd, &odd_mask);
	a.exportField(b_even, &even_mask);

	b_odd.writeParallel("field_odd", fileformat[j], cksumtype[i]);
	b_even.writeParallel("field_even", fileformat[j], cksumtype[i]);
    
	CPSfermion5Dcb4Dodd<cps::ComplexD> c_odd;
	CPSfermion5Dcb4Deven<cps::ComplexD> c_even;
	c_odd.readParallel("field_odd");
	c_even.readParallel("field_even");

	CPSfermion5DBasic d;
	d.importField(c_odd, &odd_mask);
	d.importField(c_even, &even_mask); 
    
	assert( a.equals(d) );
      }
    }
  }

  //Test parallel write with separate metadata
  
  {
    FP_FORMAT fileformat[2] = { FP_IEEE64BIG, FP_IEEE64LITTLE };
    for(int i=0;i<2;i++){
      {
	CPSfermion4D<cps::ComplexD> a;
	a.testRandom();
	
	a.writeParallelSeparateMetadata("field_split", fileformat[i]);
	
	CPSfermion4D<cps::ComplexD> b;
	b.readParallelSeparateMetadata("field_split");
	assert(a.equals(b));
      }
      {
    	CPSfermion4D<cps::ComplexD> a;
    	CPSfermion4D<cps::ComplexD> b;
    	a.testRandom();
    	b.testRandom();

    	typedef typename baseCPSfieldType<CPSfermion4D<cps::ComplexD> >::type baseField;
	
    	std::vector<baseField const*> ptrs_wr(2);
    	ptrs_wr[0] = &a;
    	ptrs_wr[1] = &b;

    	writeParallelSeparateMetadata<typename baseField::FieldSiteType,baseField::FieldSiteSize,
				      typename baseField::FieldMappingPolicy> wr(fileformat[i]);

	wr.writeManyFields("field_split_multi", ptrs_wr);
	
	CPSfermion4D<cps::ComplexD> c;
    	CPSfermion4D<cps::ComplexD> d;

	std::vector<baseField*> ptrs_rd(2);
    	ptrs_rd[0] = &c;
    	ptrs_rd[1] = &d;

	readParallelSeparateMetadata<typename baseField::FieldSiteType,baseField::FieldSiteSize,
				     typename baseField::FieldMappingPolicy> rd;
	
	rd.readManyFields(ptrs_rd, "field_split_multi");

	assert(a.equals(c));
	assert(b.equals(d));

#ifdef USE_GRID
	//Test for SIMD types too
	typedef CPSfield<Grid::vComplexD,12,FourDSIMDPolicy<DynamicFlavorPolicy>,Aligned128AllocPolicy> GridFieldType;
	typedef GridFieldType::InputParamType ParamType;

	ParamType params;
	GridFieldType::SIMDdefaultLayout(params, Grid::vComplexD::Nsimd());
	
	GridFieldType asimd(params);
	asimd.importField(a);
	
	GridFieldType bsimd(params);
	bsimd.importField(b);
	
	//First save in SIMD format and re-read in SIMD format
	std::vector<GridFieldType const*> ptrs_wrsimd(2);
	ptrs_wrsimd[0] = &asimd;
	ptrs_wrsimd[1] = &bsimd;
	
	wr.writeManyFields("field_split_multi_simd", ptrs_wrsimd);

	GridFieldType csimd(params);
	GridFieldType dsimd(params);

	std::vector<GridFieldType*> ptrs_rdsimd(2);
	ptrs_rdsimd[0] = &csimd;
	ptrs_rdsimd[1] = &dsimd;

	rd.readManyFields(ptrs_rdsimd, "field_split_multi_simd");
	
	assert(asimd.equals(csimd));
	assert(bsimd.equals(dsimd));

	//Also try loading SIMD field as non-SIMD
	rd.readManyFields(ptrs_rd, "field_split_multi_simd");
	assert(a.equals(c));
	assert(b.equals(d));

	//Finally try loading non-SIMD field as SIMD
	rd.readManyFields(ptrs_rdsimd, "field_split_multi");

	assert(asimd.equals(csimd));
	assert(bsimd.equals(dsimd));	
#endif
      }
      
    }
  }
    
  
}

template<typename A2Apolicies, typename ComplexClass>
struct setupFieldParams2{};

template<typename A2Apolicies>
struct setupFieldParams2<A2Apolicies,complex_double_or_float_mark>{
  NullObject params;
};

template<typename A2Apolicies>
struct setupFieldParams2<A2Apolicies, grid_vector_complex_mark>{
  typename FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType params;
  setupFieldParams2(){
    const int nsimd = A2Apolicies::ComplexType::Nsimd();
    FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(params,nsimd,2);
  }
};



template<typename A2Apolicies>
void testA2AvectorIO(const A2AArg &a2a_args){
  if(!UniqueID()) printf("testA2AvectorIO called\n");
  typedef typename A2AvectorV<A2Apolicies>::FieldInputParamType FieldParams;
  
  setupFieldParams2<A2Apolicies, typename ComplexClassify<typename A2Apolicies::ComplexType>::type> p;

  CPSfield_checksumType cksumtype[2] = { checksumBasic, checksumCRC32 };
  FP_FORMAT fileformat[2] = { FP_IEEE64BIG, FP_IEEE64LITTLE };
  
  for(int i=0;i<2;i++){
    for(int j=0;j<2;j++){
  
      {
  	A2AvectorV<A2Apolicies> Va(a2a_args, p.params);
  	Va.testRandom();

  	Va.writeParallel("Vvector", fileformat[j], cksumtype[i]);

  	A2AArg def;
  	def.nl = 1; def.nhits = 1; def.rand_type = UONE; def.src_width = 1;

  	A2AvectorV<A2Apolicies> Vb(def, p.params);
  	Vb.readParallel("Vvector");

  	assert( Va.paramsEqual(Vb) );
  	assert( Va.getNmodes() == Vb.getNmodes() );
  
  	for(int i=0;i<Va.getNmodes();i++){
  	  assert( Va.getMode(i).equals(Vb.getMode(i)) );
  	}
      }

  
      {
  	A2AvectorW<A2Apolicies> Wa(a2a_args, p.params);
  	Wa.testRandom();

  	Wa.writeParallel("Wvector", fileformat[j], cksumtype[i]);

  	A2AArg def;
  	def.nl = 1; def.nhits = 1; def.rand_type = UONE; def.src_width = 1;

  	A2AvectorW<A2Apolicies> Wb(def, p.params);
  	Wb.readParallel("Wvector");

  	assert( Wa.paramsEqual(Wb) );
  	assert( Wa.getNmodes() == Wb.getNmodes() );
  
  	for(int i=0;i<Wa.getNl();i++){
  	  assert( Wa.getWl(i).equals(Wb.getWl(i)) );
  	}
  	for(int i=0;i<Wa.getNhits();i++){
  	  assert( Wa.getWh(i).equals(Wb.getWh(i)) );
  	}    
      }      
    }
  }




  //Test parallel read/write with separate metadata
  for(int i=0;i<2;i++){
    {//V  
      A2AvectorV<A2Apolicies> Va(a2a_args, p.params);
      Va.testRandom();
    
      Va.writeParallelSeparateMetadata("Vvector_split", fileformat[i]);
    
      A2AvectorV<A2Apolicies> Vb(a2a_args, p.params);

      Vb.readParallelSeparateMetadata("Vvector_split");
    
      assert( Va.paramsEqual(Vb) );
      assert( Va.getNmodes() == Vb.getNmodes() );
    
      for(int i=0;i<Va.getNmodes();i++){
	assert( Va.getMode(i).equals(Vb.getMode(i)) );
      }
    }//V
    {//W
      A2AvectorW<A2Apolicies> Wa(a2a_args, p.params);
      Wa.testRandom();
      
      Wa.writeParallelSeparateMetadata("Wvector_split", fileformat[i]);

      A2AvectorW<A2Apolicies> Wb(a2a_args, p.params);
      Wb.readParallelSeparateMetadata("Wvector_split");

      assert( Wa.paramsEqual(Wb) );
      assert( Wa.getNmodes() == Wb.getNmodes() );
      
      for(int i=0;i<Wa.getNl();i++){
	assert( Wa.getWl(i).equals(Wb.getWl(i)) );
      }
      for(int i=0;i<Wa.getNhits();i++){
	assert( Wa.getWh(i).equals(Wb.getWh(i)) );
      }    
    }//W
  }



  
}


void benchmarkCPSfieldIO(){
  const int nfield_tests[7] = {1,10,50,100,250,500,1000};

  for(int n=0;n<7;n++){
    const int nfield = nfield_tests[n];
    
    std::vector<CPSfermion4D<cps::ComplexD> > a(nfield);
    for(int i=0;i<nfield;i++) a[i].testRandom();
    const double mb_written = double(a[0].byte_size())/1024/1024*nfield;
    
    const int ntest = 10;

    double avg_rate = 0;
    
    for(int i=0;i<ntest;i++){
      std::ostringstream fname; fname << "field.test" << i << ".node" << UniqueID();
      std::ofstream f(fname.str().c_str());
      double time = -dclock();
      for(int j=0;j<nfield;j++) a[j].writeParallel(f);
      f.close();
      time += dclock();
      
      const double rate = mb_written/time;
      avg_rate += rate;
      if(!UniqueID()) printf("Test %d, wrote %f MB in %f s: rate %f MB/s\n",i,mb_written,time,rate);
    }
    avg_rate /= ntest;
    
    if(!UniqueID()) printf("Data size %f MB, avg rate %f MB/s\n",mb_written,avg_rate);
    
  }
}


void testPointSource(){
  typedef A2ApointSource<StandardSourcePolicies> SrcType;
  
  const int glb_size[3] = {GJP.XnodeSites()*GJP.Xnodes(), GJP.YnodeSites()*GJP.Ynodes(), GJP.ZnodeSites()*GJP.Znodes() };
  int V = glb_size[0] * glb_size[1] * glb_size[2];

  NullObject n;
  SrcType src(n);
  for(int i=0;i<V;i++){
    int rem = i; 
    int pos[3];
    for(int d=0;d<3;d++){ pos[d] = rem % glb_size[d];  rem /= glb_size[d]; }
    
    double v = src.value(pos, glb_size).real();
    ComplexD f = src.siteComplex(i);

    if(!UniqueID()){
      printf("%d %d %d  %f  (%f,%f)\n", pos[0],pos[1],pos[2], v, f.real(),f.imag());
    }
  }
}

#ifdef USE_GRID

template<typename T>
bool GridTensorEquals(const T &a, const T &b){
  typedef typename T::vector_type vtype;
  const int sz = sizeof(T)/sizeof(vtype);

  vtype const* va = (vtype const*)&a;
  vtype const* vb = (vtype const*)&b;
  
  for(int i=0;i<sz;i++){
    if( ! equals(va[i], vb[i])) return false;  
  }
  return true;
}



template<typename GridA2Apolicies>
void testLanczosIO(typename GridA2Apolicies::FgridGFclass &lattice){
  LancArg lanc_arg;
  lanc_arg.mass = 0.01;
  lanc_arg.stop_rsd = 1e-08;
  lanc_arg.N_true_get = 50;
  GridLanczosWrapper<GridA2Apolicies> lanc;
  lanc.randomizeEvecs(lanc_arg,lattice);

  lanc.writeParallel("lanc");

  {
    GridLanczosWrapper<GridA2Apolicies> lanc2;
    lanc2.readParallel("lanc");

    assert(lanc2.evec_f.size() == 0);
    assert(lanc.evec.size() == lanc2.evec.size());
    assert(lanc.eval.size() == lanc2.eval.size());

    CPSfermion5Dcb4Dodd<cps::ComplexD> c_odd_d_1;
    CPSfermion5Dcb4Dodd<cps::ComplexD> c_odd_d_2;
  
    for(int i=0;i<lanc.eval.size();i++){
      assert(lanc.eval[i] == lanc2.eval[i]);
      c_odd_d_1.importGridField(lanc.evec[i]);
      c_odd_d_2.importGridField(lanc2.evec[i]);
      
      assert( c_odd_d_1.equals( c_odd_d_2 ) );

      auto view = lanc.evec[i].View(Grid::CpuRead);
      auto view2 = lanc2.evec[i].View(Grid::CpuRead);

      for(int s=0;s<lanc.evec[i].Grid()->oSites();s++)
	assert( GridTensorEquals(view[s] , view2[s]) );
      
      
    }
  }

  lanc.toSingle();
  lanc.writeParallel("lanc");
  
  {
    GridLanczosWrapper<GridA2Apolicies> lanc2;
    lanc2.readParallel("lanc");

    assert(lanc2.evec.size() == 0);
    assert(lanc.evec_f.size() == lanc2.evec_f.size());
    assert(lanc.eval.size() == lanc2.eval.size());

    CPSfermion5Dcb4Dodd<cps::ComplexF> c_odd_f_1;
    CPSfermion5Dcb4Dodd<cps::ComplexF> c_odd_f_2;
  
    for(int i=0;i<lanc.eval.size();i++){
      assert(lanc.eval[i] == lanc2.eval[i]);
      c_odd_f_1.importGridField(lanc.evec_f[i]);
      c_odd_f_2.importGridField(lanc2.evec_f[i]);
      
      assert( c_odd_f_1.equals( c_odd_f_2 ) );

      auto view = lanc.evec_f[i].View(Grid::CpuRead);
      auto view2 = lanc2.evec_f[i].View(Grid::CpuRead);

      for(int s=0;s<lanc.evec_f[i].Grid()->oSites();s++)
	assert( GridTensorEquals(view[s] , view2[s]) );
    }
  }
}



void benchmarkTest(){
  CPSfermion4D<cps::ComplexD> rnd4d;
  rnd4d.testRandom();

  const int nsimd = Grid::vComplexD::Nsimd();      
  
  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd);
  
  CPSfermion4D<Grid::vComplexD,FourDSIMDPolicy<DynamicFlavorPolicy> > in(simd_dims);
  in.importField(rnd4d);
}


template<typename GridA2Apolicies>
void testLMAprop(typename GridA2Apolicies::FgridGFclass &lattice, int argc, char* argv[]){
#if defined(USE_GRID_LANCZOS) && defined(USE_GRID_A2A)

  NullObject null_obj;
  lattice.BondCond();
  CPSfield<cps::ComplexD,4*9,FourDpolicy<OneFlavorPolicy> > cps_gauge((cps::ComplexD*)lattice.GaugeField(),null_obj);
  cps_gauge.exportGridField(*lattice.getUmu());
  lattice.BondCond();

  if(lattice.FixGaugeKind() == FIX_GAUGE_NONE){
    FixGaugeArg fix_gauge_arg;
    fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
    fix_gauge_arg.hyperplane_start = 0;
    fix_gauge_arg.hyperplane_step = 1;
    fix_gauge_arg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites();
    fix_gauge_arg.stop_cond = 1e-08;
    fix_gauge_arg.max_iter_num = 10000;

    CommonArg common_arg;
  
    AlgFixGauge fix_gauge(lattice,&common_arg,&fix_gauge_arg);
    fix_gauge.run();
  }

  LancArg lanc_arg;
  bool read_larg = false;
  for(int i=1;i<argc;i++){
    if(std::string(argv[i]) == "-lanc_arg"){
      if(!lanc_arg.Decode(argv[i+1],"lanc_arg")){
	ERR.General("Parameters","Parameters","Can't open %s!\n",argv[i+1]);
      }
      read_larg = true;
      break;
    }
  }
  if(!read_larg){
    lanc_arg.mass = 0.01;
    lanc_arg.stop_rsd = 1e-08;
    lanc_arg.qr_rsd = 1e-13;
    lanc_arg.N_true_get = 100;
    lanc_arg.N_get = 100;
    lanc_arg.N_use = 120;
    lanc_arg.EigenOper = DDAGD;
    lanc_arg.precon = 1;
    lanc_arg.ch_ord = 80;
    lanc_arg.ch_alpha = 6.6;
    lanc_arg.ch_beta = 1;
    lanc_arg.lock = 0;
    lanc_arg.maxits = 10000;
  }    

  GridLanczosWrapper<GridA2Apolicies> lanc;

  lanc.compute(lanc_arg, lattice);

  A2AArg a2a_args;
  a2a_args.nl = 100;
  a2a_args.nhits = 0;
  a2a_args.rand_type = UONE;
  a2a_args.src_width = 1;
  
  const int nsimd = Grid::vComplexD::Nsimd();      
  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd);

  A2AvectorV<GridA2Apolicies> V(a2a_args, simd_dims);
  A2AvectorW<GridA2Apolicies> W(a2a_args, simd_dims);

  CGcontrols cg_controls;
  cg_controls.CGalgorithm = AlgorithmCG;
  computeVWlow(V,W, lattice, lanc.evec, lanc.eval, lanc.mass, cg_controls);
  
  //v_i^dag G v_i = (1/L_i)
  //v_i^dag G^dag v_i = (1/L_i*)

  //Let M be the number of eigenvectors of the matrix, and N < M

  //LMA = \sum_{i=0}^N v_i (1/L_i) v_i^dag
  //LMA^dag = \sum_{i=0}^N v_i (1/L_i*) v^dag_i
  //        = \sum_{i=0}^N v_i v_i^dag G^dag v_i v^dag_i
  //        = \sum_{i=0}^N v_i v_i^dag g5 G g5 v_i v^dag_i

  //G = \sum_{i=0}^M v_i (1/L_i) v_i^dag   because  G v_j = (1/L_j) v_j    v_i^dag v_j = delta_ij   

  //D G = 1 = \sum_{i=0}^M v_i v_i^dag

  //thus for N=M
  //LMA^dag
  //        = g5 G g5 = g5 LMA g5
  




  //LMA - g5 LMA^dag g5 
  //= \sum_{i=0}^N   [  v_i (1/L_i) v_i^dag  -  g5 v_i (1/L^*_i) v^dag_i g5 ]
  //= 


  //\sum_y D(x,y) v_i(y) = L_i v_i(x)
  //\sum_xy G(z,x) D(x,y) v_i(y) = \sum_x G(z,x) L_i v_i(x)
  //v_i(z) = \sum_x G(z,x) L_i v_i(x)

  //v_i(z) = \sum_x g5 G^dag(x,z) g5 L_i v_i(x)
  //       = [\sum_x v_i^dag(x) L_i g5 G(x,z) g5]^dag
  
  //v_i^dag(z) = \sum_x  v_i^dag(x) L_i g5 G(x,z) g5

  //\sum_y G(x,y) v_i(y) = (1/L_i) v_i(x)
  //\sum_y v_i^dag(y) G^dag(y,x) = (1/L_i) v_i^dag(x)
  



  //LMA(x) =  \sum_{i=0}^{N} v_i(x) (1/L_i) v_i^dag(x)
  //       =  \sum_{i=0}^{N} v_i(x) (1/L_i) \sum_z v_i^dag(z) L_i g5 G(z,x) g5
  //       =  \sum_z \sum_{i=0}^{N} v_i(x) v_i^dag(z) g5 G(z,x) g5
  //       == \sum_z \sum_{i=0}^{N} 


  //Make meson fields
  //Test g5 hermiticity and cc reln exactness 

  //g5-herm
  //sum_{x,y}e^{-ip1x} e^{-ip2y}  tr( G^dag(x,y) ) == sum_{x,y}e^{-ip1x} e^{-ip2y} tr( G(y,x) )
  //sum_{x,y}e^{-ip1x} e^{-ip2y} tr( [V_i(x)W_i^dag(y)]^dag )  = tr( [sum_{x,y}e^{+ip1x} e^{+ip2y}V_i(x)W_i^dag(y)] )^* = M_ii(-p2,-p1)^* 
  //sum_{x,y}e^{-ip1x} e^{-ip2y} tr( G(y,x) ) = tr( [sum_{x,y}e^{-ip1x} e^{-ip2y} V(y)W^dag(x)] ) = M(p1,p2)
  

  //sum_{x,y}e^{-ip1x} e^{-ip2y}  tr( G^dag(x,y) O(x-y) A^dag(x) A(y) s3(1 + q(p1)s2) ) == sum_{x,y}e^{-ip1x} e^{-ip2y} tr( G(y,x) O(x-y) A^dag(x) A(y) s3(1 + q(p1)s2) )

  //q(p) = exp(i n(p) \pi)   = +/- 1
  //q^dag(p) = exp(-i n(p)\pi ) = q(p)

  //sum_{x,y}e^{-ip1x} e^{-ip2y} tr(  G^dag(x,y) O(x-y) A^dag(x)  A(y) s3 (1 + q(p1)s2) )  
  //[ sum_{x,y}e^{+ip1x} e^{+ip2y} tr(  (1 + q(p1)s2) )s3 A^dag(y) A(x) O(x-y) G(x,y)    ]^*
  //[ sum_{x,y}e^{+ip1x} e^{+ip2y} tr( G(x,y) O(x-y) A^dag(y) A(x) s3(1 + q(-p1)s2) ) ]^*
  //[ M_ii(-p2,-p1) ]^*

  //sum_{x,y}e^{-ip1x} e^{-ip2y} tr( G(y,x) O(x-y) A^dag(x) A(y) s3(1 + q(p1)s2) )
  //M_ii(p1,p2)

  typedef typename A2AflavorProjectedExpSource<typename GridA2Apolicies::SourcePolicies>::FieldParamType SrcFieldParamType;
  typedef typename A2AflavorProjectedExpSource<typename GridA2Apolicies::SourcePolicies>::ComplexType SrcComplexType;
  SrcFieldParamType sfp; defaultFieldParams<SrcFieldParamType, SrcComplexType>::get(sfp);

  typedef A2AflavorProjectedExpSource<typename GridA2Apolicies::SourcePolicies> ExpSrcType;
  
  int p1[3] = {1,1,1}; //n1 = (0,0,0)    exp(in1) = 1
  int p2[3] = {-3,1,-3}; //n2 = (-2,0,-2)    exp(in2) = 1
  int mp1[3] = {-1,-1,-1}; //nmp1 = (-1,-1,-1)   exp(imp1) = -1
  int mp2[3] = {3,-1,3}; //nmp2 = (1,-1,1)  exp(inmp2) = -1

  typedef typename GridA2Apolicies::ComplexType ComplexType;
  typedef typename GridA2Apolicies::ScalarComplexType ScalarComplexType;

  typedef SCFspinflavorInnerProduct<0,ComplexType,ExpSrcType,true,false> ExpInnerType;

  int Lt = GJP.Tnodes()*GJP.TnodeSites();

  std::vector< A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_mp2_mp1;
  {
    ExpSrcType src_mp2_mp1(2.0, mp1, sfp); //momentum associated with the V
    ExpInnerType inner_mp2_mp1(sigma3, src_mp2_mp1);

    A2AvectorVfftw<GridA2Apolicies> Vfftw(a2a_args,simd_dims);
    Vfftw.gaugeFixTwistFFT(V,mp1,lattice);
    
    A2AvectorWfftw<GridA2Apolicies> Wfftw(a2a_args,simd_dims);
    Wfftw.gaugeFixTwistFFT(W,p2,lattice);
  
    A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_mp2_mp1, Wfftw, inner_mp2_mp1, Vfftw);
  }

  std::vector< A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> > mf_p1_p2;
  {
    ExpSrcType src_p1_p2(2.0, p2, sfp);  
    ExpInnerType inner_p1_p2(sigma3, src_p1_p2);

    A2AvectorVfftw<GridA2Apolicies> Vfftw(a2a_args,simd_dims);
    Vfftw.gaugeFixTwistFFT(V,p2,lattice);
    
    A2AvectorWfftw<GridA2Apolicies> Wfftw(a2a_args,simd_dims);
    Wfftw.gaugeFixTwistFFT(W,mp1,lattice);
  
    A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_p1_p2, Wfftw, inner_p1_p2, Vfftw);
  }

  for(int t=0;t<Lt;t++){
    ScalarComplexType a = cconj(trace(mf_mp2_mp1[t]));
    ScalarComplexType b = trace(mf_p1_p2[t]);

    std::cout << t << " " << a.real() << " " << a.imag() << " " << b.real() << " " << b.imag() << std::endl;
  }



  //g5-herm:
  //Tr( G^dag(x,y) G^dag(y,x) ) = Tr ( G(y,x) G(x,y) )
#endif
}


void testSCFmat(){
  typedef std::complex<double> ComplexType;
  typedef CPSspinMatrix<ComplexType> SpinMat;

  SpinMat one; one.unit();
  SpinMat minusone(one); minusone *= -1.;
  SpinMat zero; zero.zero();

  //Test 1.gr(i) == 1.gl(i)
  {
    std::cout << "Test 1.gr(i) == 1.gl(i)\n";
    for(int i=0;i<5;i++){
      int mu = i<4 ? i : -5;
      SpinMat a(one); a.gr(mu);
      SpinMat b(one); b.gl(mu);
      std::cout << mu << " " << a << "\n" << b << std::endl;
      assert(a==b);
    }
  }

  SpinMat gamma[6] = {one,one,one,one,one,one};
  for(int i=0;i<4;i++) gamma[i].gr(i);
  gamma[5].gr(-5);

  //Test anticommutation reln
  {
    SpinMat two(one); two *= 2.; 
    std::cout << "Test anticommutation reln\n";

    for(int mu=0;mu<4;mu++){
      for(int nu=0; nu<4; nu++){
	SpinMat c = gamma[mu]*gamma[nu] + gamma[nu]*gamma[mu];
	std::cout << mu << " " << nu << " " << c << std::endl;
	if(mu == nu) assert(c == two);
	else assert(c == zero);
      }
    }
  }

  //Test glAx
  {
    std::cout << "Testing glAx\n";
    for(int mu=0;mu<4;mu++){
      SpinMat a(one); a.glAx(mu);
      SpinMat b(one); b.gl(-5).gl(mu);
      std::cout << mu << " " << a << "\n" << b << std::endl;
      assert(a==b);
    }
  }

  //Test grAx
  {
    std::cout << "Testing grAx\n";
    for(int mu=0;mu<4;mu++){
      SpinMat a(one); a.grAx(mu);
      SpinMat b(one); b.gr(mu).gr(-5);
      std::cout << mu << " " << a << "\n" << b << std::endl;
      assert(a==b);

      SpinMat c(one); c.glAx(mu); c.grAx(mu); 
      
      std::cout << mu << " pow2 " << c << std::endl;
      assert(c == minusone);
    }
  }



}

#endif //USE_GRID



void timeAllReduce(bool huge_pages){
#ifdef USE_MPI

  size_t bytes = 1024;
  size_t scale = 2;
  size_t max_bytes = 1024*1024*1024 + 1; //512MB max comm
  
  while(bytes < max_bytes){
    assert(bytes % sizeof(double) == 0);
    size_t ndouble = bytes / sizeof(double);

    assert( (size_t) ((int)ndouble) == ndouble ); //make sure it can be downcast to int

    double* buf;
    
    if(huge_pages){
      char shm_name [1024];
      struct passwd *pw = getpwuid (getuid());
      sprintf(shm_name,"/shm_%s",pw->pw_name);

      shm_unlink(shm_name);
      int fd=shm_open(shm_name,O_RDWR|O_CREAT,0666);
      if ( fd < 0 ) {   perror("failed shm_open");      assert(0);      }
      ftruncate(fd, bytes);
      
      int mmap_flag = MAP_SHARED;
#ifdef MAP_POPULATE
      mmap_flag |= MAP_POPULATE;
#endif
      //if (huge) mmap_flag |= MAP_HUGETLB;
      mmap_flag |= MAP_HUGETLB;
      buf =  (double*)mmap(NULL,bytes, PROT_READ | PROT_WRITE, mmap_flag, fd, 0);

      //std::cout << "SHM "<<ptr<< "("<< size<< "bytes)"<<std::endl;
      if ( (void*)buf == (void*)MAP_FAILED ) {
        perror("failed mmap");
        assert(0);
      }

    }else{
      buf = (double*)malloc(bytes);
    }

    int nrpt = 10;

    Float time = -dclock();
    {
      for(int n=0;n<nrpt;n++)
	MPI_Allreduce(MPI_IN_PLACE, buf, ndouble, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    time += dclock();

    time /= nrpt;

    Float MB =  ( (double)bytes )/1024./1024.;

    Float MB_per_s = MB / time;
    std::cout << "Size " << MB << " MB  time " << time << " secs, " << MB_per_s << " MB/s" << std::endl;

    if(huge_pages) munmap(buf,bytes);
    else free(buf);

    bytes *= scale;
  }
#endif
}




#ifdef USE_GRID
template<typename GridA2Apolicies>
void test4DlowmodeSubtraction(A2AArg a2a_args, const int ntests, const int nthreads, typename GridA2Apolicies::FgridGFclass &lattice){
  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);

  a2a_args.nl = 10;

  assert(GJP.Snodes() == 1);
  int Ls = GJP.SnodeSites();
  double b_minus_c_outer = lattice.get_mob_b() - lattice.get_mob_c();
  assert(b_minus_c_outer == 1.0);
  double b_plus_c_outer = lattice.get_mob_b() + lattice.get_mob_c();
  double mass = 0.01;

  if(!UniqueID()) printf("test4DlowmodeSubtraction outer b+c=%g\n", b_plus_c_outer);
  
  CGcontrols cg_controls_4dsub;
  cg_controls_4dsub.CGalgorithm = AlgorithmMixedPrecisionMADWF; //currently only MADWF version supports 4d subtraction, but it will use regular CG internally if the inner and outer Dirac ops match
  cg_controls_4dsub.CG_tolerance = 1e-8;
  cg_controls_4dsub.CG_max_iters = 10000;
  cg_controls_4dsub.mixedCG_init_inner_tolerance = 1e-4;
  cg_controls_4dsub.madwf_params.Ls_inner = Ls;
  cg_controls_4dsub.madwf_params.b_plus_c_inner = b_plus_c_outer;
  cg_controls_4dsub.madwf_params.precond = SchurOriginal;
  cg_controls_4dsub.madwf_params.use_ZMobius = false;
  cg_controls_4dsub.madwf_params.ZMobius_params.compute_lambda_max = 1.42;
  cg_controls_4dsub.madwf_params.ZMobius_params.gamma_src = A2A_ZMobiusGammaSourceCompute;

  CGcontrols cg_controls_5dsub(cg_controls_4dsub);
  cg_controls_5dsub.CGalgorithm = AlgorithmMixedPrecisionRestartedCG;

  //Random evecs or evals
  std::vector<typename GridA2Apolicies::GridFermionFieldF> evec(a2a_args.nl, typename GridA2Apolicies::GridFermionFieldF(lattice.getFrbGridF()) );
  std::vector<Grid::RealD> eval(a2a_args.nl);
  CPSfermion5D<typename GridA2Apolicies::ScalarComplexType> tmp;
  for(int i=0;i<a2a_args.nl;i++){
    eval[i] = fabs(LRG.Urand(FOUR_D));
    tmp.testRandom();
    tmp.exportGridField(evec[i]);
    evec[i].Checkerboard() = Grid::Odd;
  }

  EvecInterfaceGridSinglePrec<GridA2Apolicies> eve_4dsub(evec, eval, lattice, lattice.get_mass(), cg_controls_4dsub);
  EvecInterfaceGridSinglePrec<GridA2Apolicies> eve_5dsub(evec, eval, lattice, lattice.get_mass(), cg_controls_5dsub);

  A2AvectorW<GridA2Apolicies> W_5dsub(a2a_args, simd_dims);
  A2AvectorW<GridA2Apolicies> W_4dsub(a2a_args, simd_dims);
  
  LatRanGen LRGbak(LRG);
  W_5dsub.setWhRandom();
  LRG = LRGbak;
  W_4dsub.setWhRandom();

  assert( W_5dsub.getWh(0).equals( W_4dsub.getWh(0) ));


  A2AvectorV<GridA2Apolicies> V_5dsub(a2a_args, simd_dims);
  A2AvectorV<GridA2Apolicies> V_4dsub(a2a_args, simd_dims);

  computeVWhigh(V_4dsub, W_4dsub, lattice, eve_4dsub, mass, cg_controls_4dsub);
  computeVWhigh(V_5dsub, W_5dsub, lattice, eve_5dsub, mass, cg_controls_5dsub);
  
  std::cout << "V " << std::endl;
  typename GridA2Apolicies::ScalarFermionFieldType v_scal1, v_scal2;

  for(int i=0;i<V_5dsub.getNv();i++){
    printf("Testing %d\n",i);
    v_scal1.importField(V_5dsub.getMode(i));
    v_scal2.importField(V_4dsub.getMode(i));
    compareField(v_scal1, v_scal2, "Field", 1e-4, true);
  }
}

template<typename ScalarA2Apolicies, typename GridA2Apolicies>
void benchmarkvMvGridOrig(const A2AArg &a2a_args, const int ntests, const int nthreads){
#ifdef USE_GRID
#define CPS_VMV
  //#define GRID_VMV
#define GRID_SPLIT_LITE_VMV;

  std::cout << "Starting vMv benchmark\n";
  std::cout << "nl=" << a2a_args.nl << "\n";
  

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  typename FourDSIMDPolicy<typename ScalarA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<typename ScalarA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorWfftw<ScalarA2Apolicies> W(a2a_args);
  A2AvectorVfftw<ScalarA2Apolicies> V(a2a_args);
    
  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);

  W.testRandom();
  V.testRandom();
  Wgrid.importFields(W);
  Vgrid.importFields(V);
  
  A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf;
  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_grid;
  mf.setup(W,V,0,0);
  mf_grid.setup(Wgrid,Vgrid,0,0);     
  mf.testRandom();
  for(int i=0;i<mf.getNrows();i++)
    for(int j=0;j<mf.getNcols();j++)
      mf_grid(i,j) = mf(i,j); //both are scalar complex
  
  typedef typename GridA2Apolicies::ComplexType grid_Complex;
  typedef typename ScalarA2Apolicies::ComplexType mf_Complex;

  int nf = GJP.Gparity()+1;

  //Compute Flops
  size_t Flops = 0;

  {
    typedef typename A2AvectorVfftw<ScalarA2Apolicies>::DilutionType iLeftDilutionType;
    typedef typename A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::LeftDilutionType iRightDilutionType;

    typedef typename A2AmesonField<ScalarA2Apolicies,A2AvectorWfftw,A2AvectorVfftw>::RightDilutionType jLeftDilutionType;    
    typedef typename A2AvectorWfftw<ScalarA2Apolicies>::DilutionType jRightDilutionType;

    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(V);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(W);
    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    size_t vol3d = GJP.XnodeSites()*GJP.YnodeSites()*GJP.ZnodeSites();

    //Count Flops on node
    for(int t=GJP.TnodeCoor()*GJP.TnodeSites();t<(GJP.TnodeCoor()+1)*GJP.TnodeSites();t++){
      //Count elements of Mr actually used
      std::vector<bool> ir_used(mf.getNrows(),false);
      for(int scl=0;scl<12;scl++){
	for(int fl=0;fl<nf;fl++){
	  modeIndexSet ilp, irp;
	  ilp.time = t;
	  irp.time = mf.getRowTimeslice();
	      	      
	  ilp.spin_color = scl;
	  ilp.flavor = fl;

	  int ni = i_ind.getNindices(ilp,irp);
	  for(int i=0;i<ni;i++){
	    int il = i_ind.getLeftIndex(i, ilp,irp);
	    ir_used[il] = true;
	  }
	}
      }
      int nir_used = 0;
      for(int i=0;i<ir_used.size();i++)
	if(ir_used[i]) nir_used++;
      
      //Mr[scr][fr]
      for(int scr=0;scr<12;scr++){
	for(int fr=0;fr<nf;fr++){

	  modeIndexSet jlp, jrp;
	      
	  jlp.time = mf.getColTimeslice();
	  jrp.time = t;
	  
	  jrp.spin_color = scr;
	  jrp.flavor = fr;
	  
	  int nj = j_ind.getNindices(jlp,jrp);
	      	      
	  //Mr =  nir_used * nj * (cmul + cadd)  per site
	  Flops += nir_used * nj * 8 * vol3d;
	}
      }

      //l[scl][fl](Mr[scr][fr])
      for(int scl=0;scl<12;scl++){
	for(int fl=0;fl<nf;fl++){
	  for(int scr=0;scr<12;scr++){
	    for(int fr=0;fr<nf;fr++){

	      modeIndexSet ilp, irp, jlp, jrp;
	      ilp.time = t;
	      irp.time = mf.getRowTimeslice();
	            
	      ilp.spin_color = scl;
	      ilp.flavor = fl;

	      int ni = i_ind.getNindices(ilp,irp);
	      	      
	      //l( Mr) = ni  * (cmul + cadd)  per site
	      Flops += ni * 8 * vol3d;
	    }
	  }
	}
      } 
    }//t

  }//Flops count
  size_t MFlops = Flops/1024/1024;
  
      
  Float total_time = 0.;
  Float total_time_orig = 0.;
  Float total_time_split_lite_grid = 0.;
  Float total_time_field_offload = 0.;
  mult_vMv_field_offload_timers::get().reset();

  typedef typename AlignedVector<CPSspinColorFlavorMatrix<mf_Complex> >::type BasicVector;
  typedef typename AlignedVector<CPSspinColorFlavorMatrix<grid_Complex> >::type SIMDvector;

  BasicVector orig_sum(nthreads);
  SIMDvector grid_sum(nthreads);

  BasicVector orig_tmp(nthreads);
  SIMDvector grid_tmp(nthreads);

  SIMDvector grid_sum_split_lite(nthreads);      

  int orig_3vol = GJP.VolNodeSites()/GJP.TnodeSites();
  int grid_3vol = Vgrid.getMode(0).nodeSites(0) * Vgrid.getMode(0).nodeSites(1) *Vgrid.getMode(0).nodeSites(2);

  mult_vMv_split_lite<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw> vmv_split_lite_grid;

  for(int iter=0;iter<ntests;iter++){
    for(int i=0;i<nthreads;i++){
      orig_sum[i].zero(); grid_sum[i].zero();
      grid_sum_split_lite[i].zero();
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

#ifdef GRID_SPLIT_LITE_VMV
      //SPLIT LITE VMV GRID
      total_time_split_lite_grid -= dclock();	  
      vmv_split_lite_grid.setup(Vgrid, mf_grid, Wgrid, top);

#pragma omp parallel for
      for(int xop=0;xop<grid_3vol;xop++){
	int me = omp_get_thread_num();
	vmv_split_lite_grid.contract(grid_tmp[me], xop, false, true);
	grid_sum_split_lite[me] += grid_tmp[me];
      }
      total_time_split_lite_grid += dclock();
#endif

    }//end top loop
    for(int i=1;i<nthreads;i++){
      orig_sum[0] += orig_sum[i];
      grid_sum[0] += grid_sum[i];
      grid_sum_split_lite[0] += grid_sum_split_lite[i];  
    }

    //Offload version computes all x,t, so we just have to sum over 4 volume afterwards
    total_time_field_offload -= dclock();
    typedef typename mult_vMv_field<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw>::PropagatorField PropagatorField;
    PropagatorField pfield(simd_dims);
    
    mult(pfield, Vgrid, mf_grid, Wgrid, false, true);
    total_time_field_offload += dclock();

    CPSspinColorFlavorMatrix<grid_Complex> vmv_offload_sum4;
    vmv_offload_sum4.zero();
    for(size_t i=0;i<pfield.size();i++){
      vmv_offload_sum4 += *pfield.fsite_ptr(i);
    }
  } //tests loop
#ifdef CPS_VMV
  printf("vMv: Avg time vMv (non-SIMD) %d iters: %g secs/iter  %g Mflops\n",ntests,total_time_orig/ntests,  MFlops/(total_time_orig/ntests) );
#endif
#ifdef GRID_VMV
  printf("vMv: Avg time vMv (SIMD) code %d iters: %g secs/iter  %g Mflops\n",ntests,total_time/ntests, MFlops/(total_time/ntests) );
#endif
#ifdef GRID_SPLIT_LITE_VMV
  printf("vMv: Avg time split vMv lite (SIMD) %d iters: %g secs/iter  %g Mflops\n",ntests,total_time_split_lite_grid/ntests, MFlops/(total_time_split_lite_grid/ntests) );
#endif
  printf("vMv: Avg time vMv field offload %d iters: %g secs/iter  %g Mflops\n",ntests,total_time_field_offload/ntests, MFlops/(total_time_field_offload/ntests) );

  if(!UniqueID()){
    printf("vMv offload timings:\n");
    mult_vMv_field_offload_timers::get().print();
  }

#endif
}




template<typename GridA2Apolicies>
void benchmarkvMvGridOffload(const A2AArg &a2a_args, const int ntests, const int nthreads){
  std::cout << "Starting vMv offload benchmark\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  typename FourDSIMDPolicy<typename GridA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<typename GridA2Apolicies::FermionFieldType::FieldMappingPolicy::FieldFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);
      
  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);

  Wgrid.testRandom();
  Vgrid.testRandom();
  
  A2AmesonField<GridA2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_grid;
  mf_grid.setup(Wgrid,Vgrid,0,0);     
  mf_grid.testRandom();
  
  typedef mult_vMv_field<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, A2AvectorVfftw, A2AvectorWfftw> offload;
  typedef typename offload::PropagatorField PropagatorField;
  PropagatorField pfield(simd_dims);
  
  Float total_time_field_offload = 0.;
  mult_vMv_field_offload_timers::get().reset();
  
  for(int i=0;i<ntests;i++){
    if(!UniqueID()){ printf("."); fflush(stdout); }
    total_time_field_offload -= dclock();    
    mult(pfield, Vgrid, mf_grid, Wgrid, false, true);
    total_time_field_offload += dclock();
  }
  if(!UniqueID()){ printf("\n"); fflush(stdout); }

  int nf = GJP.Gparity() + 1;

  //Count flops (over all nodes)
  //\sum_i\sum_j v(il)_{scl,fl}(x)  M(ir, jl) * v(jr)_{scr,fr}(x)  for all t, x3d

  //Simple method
  //for(int x4d=0;x4d<vol4d;x4d++)
  // for(int scl=0;scl<12;scl++)
  //  for(int fl=0;fl<nf;fl++)
  //   for(int scr=0;scl<12;scl++)
  //    for(int fr=0;fr<nf;fr++)
  //     for(int i=0;i<ni;i++)
  //      for(int j=0;j<nj;j++)
  //        out(fl,scl; fr, scr)(x) += v(il[i])_{scl,fl}(x) * M(ir[i], jl[j]) * v(jr[j])_{scr,fr}(x)     
  //
  //vol4d * 12*nf * 12*nf * ni * nj * 14 flops
  


  //Split method
  //for(int x4d=0;x4d<vol4d;x4d++)
  // for(int scr=0;scr<12;scr++)
  //  for(int fr=0;fr<nf;fr++)
  //    for(int i=0;i<ni;i++)
  //     for(int j=0;j<nj;j++)
  //        Mr(ir[i])_{scr,fr}(x)   +=   M(ir[i], jl[j]) * v(jr[j])_{scr,fr}(x) 
  //
  //vol4d * 12 * nf *  ni * nj * 8 flops

  //+

  //for(int x4d=0;x4d<vol4d;x4d++)
  // for(int scl=0;scl<12;scl++)
  //  for(int fl=0;fl<nf;fl++)
  //   for(int scr=0;scl<12;scl++)
  //    for(int fr=0;fr<nf;fr++)
  //     for(int i=0;i<ni;i++)
  //        out(fl,scl; fr, scr)(x) += v(il[i])_{scl,fl}(x) * Mr(ir[i])_{scr,fr}(x)    
  //vol4d * 12 * nf * 12 * nf *  ni * 8 flops   


  //vol4d * 12 * nf * ni * ( nj * 8 + 12*nf*ni *8)


  ModeContractionIndices<typename offload::iLeftDilutionType, typename offload::iRightDilutionType> i_ind(Vgrid);
  ModeContractionIndices<typename offload::jLeftDilutionType, typename offload::jRightDilutionType> j_ind(Vgrid);
  size_t Flops = 0;
  for(int t_glob=0;t_glob<GJP.TnodeSites()*GJP.Tnodes();t_glob++){
    modeIndexSet ilp, irp, jlp, jrp;
    ilp.time = jrp.time = t_glob;
    irp.time = mf_grid.getRowTimeslice();
    jlp.time = mf_grid.getColTimeslice();
    
    //ni is actually a function of scl, fl, but we can work out exactly which ir are used for any of the scl,fl
    std::set<int> ir_used;
    for(int fl=0;fl<nf;fl++){
      ilp.flavor = irp.flavor = fl;
      for(int scl=0;scl<12;scl++){
	ilp.spin_color = irp.spin_color = scl;
	auto const &ivec = i_ind.getIndexVector(ilp,irp);
	for(int i=0;i<ivec.size();i++)
	  ir_used.insert(ivec[i].second);
      }
    }
    for(int fr=0;fr<nf;fr++){
      jlp.flavor = jrp.flavor = fr;
      for(int scr=0;scr<12;scr++){
	jlp.spin_color = jrp.spin_color = scr;
	size_t nj = j_ind.getIndexVector(jlp,jrp).size();
	
	Flops += ir_used.size() * nj * 8;
      }
    }

    for(int fr=0;fr<nf;fr++){
      for(int scr=0;scr<12;scr++){
	
	for(int fl=0;fl<nf;fl++){
	  ilp.flavor = irp.flavor = fl;
	  for(int scl=0;scl<12;scl++){
	    ilp.spin_color = irp.spin_color = scl;
	    size_t ni = i_ind.getIndexVector(ilp,irp).size();
	    
	    Flops += ni * 8;
	  }
	}
      }
    }
  }
  Flops *= GJP.TotalNodes()*GJP.VolNodeSites()/GJP.TnodeSites(); //the above is done for every 3d site

  double tavg = total_time_field_offload/ntests;
  double Mflops = double(Flops)/tavg/1024./1024.;

  if(!UniqueID()){
    printf("vMv: Avg time offload %d iters: %g secs  perf %f Mflops\n",ntests,tavg,Mflops);
    printf("vMv offload timings:\n");
    mult_vMv_field_offload_timers::get().print();
  }
}





template<typename GridA2Apolicies>
void benchmarkVVgridOffload(const A2AArg &a2a_args, const int ntests, const int nthreads){
  std::cout << "Starting vv benchmark\n";

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);

  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);

  Wgrid.testRandom();
  Vgrid.testRandom();
  
  typedef mult_vv_field<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw> offload;
  typedef typename offload::PropagatorField PropagatorField;
  PropagatorField pfield(simd_dims);
      
  Float total_time_field_offload = 0;

  for(int iter=0;iter<ntests;iter++){
    total_time_field_offload -= dclock();    
    mult(pfield, Vgrid, Wgrid, false, true);
    total_time_field_offload += dclock();
  }


  int nf = GJP.Gparity() + 1;

  //Count flops (over all nodes)
  //\sum_i v(il)_{scl,fl}(x) * v(ir)_{scr,fr}(x) for all t, x3d
  ModeContractionIndices<typename offload::leftDilutionType, typename offload::rightDilutionType> i_ind(Vgrid);
  size_t Flops = 0;
  for(int t_glob=0;t_glob<GJP.TnodeSites()*GJP.Tnodes();t_glob++){
    modeIndexSet ilp, irp, jlp, jrp;
    ilp.time = irp.time = t_glob;
    
    for(ilp.flavor=0;ilp.flavor<nf;ilp.flavor++){
      for(ilp.spin_color=0;ilp.spin_color<12;ilp.spin_color++){
	for(irp.flavor=0;irp.flavor<nf;irp.flavor++){
	  for(irp.spin_color=0;irp.spin_color<12;irp.spin_color++){
	    size_t ni = i_ind.getIndexVector(ilp,irp).size();
	    Flops +=  ni * 8; //z = z + (z*z)   z*z=6flops
	  }
	}
      }
    }
  }
  Flops *= GJP.TotalNodes()*GJP.VolNodeSites()/GJP.TnodeSites(); //the above is done for every global 3d site

  double tavg = total_time_field_offload/ntests;
  double Mflops = double(Flops)/tavg/1024./1024.;

  printf("vv: Avg time field offload code %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);

  if(!UniqueID()){
    printf("vv offload timings:\n");
    mult_vv_field_offload_timers::get().print();
  }

}





struct _tr{
  template<typename MatrixType>
  accelerator_inline auto operator()(const MatrixType &matrix) const ->decltype(matrix.Trace()){ return matrix.Trace(); }  
};
struct _trtr{
  template<typename MatrixType>
  accelerator_inline auto operator()(const MatrixType &a, const MatrixType &b) const ->decltype(a.Trace()*b.Trace()){ return a.Trace()*b.Trace(); }  
};

template<typename VectorMatrixType>
struct _trtrV{
  typedef typename VectorMatrixType::scalar_type OutputType;
  accelerator_inline void operator()(OutputType &out, const VectorMatrixType &a, const VectorMatrixType &b, const int lane) const{ 
    typename VectorMatrixType::scalar_type tmp, tmp2; //each thread will have one of these but will only write to a single thread    
    Trace(tmp, a, lane);
    Trace(tmp2, b, lane);
    mult(out, tmp, tmp2, lane);
  }
};



template<typename GridA2Apolicies>
void benchmarkCPSmatrixField(const int ntests){
  std::cout << "Starting CPSmatrixField benchmark\n";

  typedef typename GridA2Apolicies::ComplexType ComplexType;
  const int nsimd = ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);

  typedef CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > SCFmatrixField;

  static_assert(isCPSsquareMatrix<typename SCFmatrixField::FieldSiteType>::value == 1);


  SCFmatrixField m1(simd_dims);
  m1.testRandom();
  SCFmatrixField m2(simd_dims);
  m2.testRandom();



  typename std::decay<decltype(Trace(m1))>::type tr_m1(simd_dims);
  
  if(0){
    //Trace
    Float total_time_trace = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = Trace(m1);
    }
    total_time_trace += dclock();
    
    double Flops = m1.size() * 24 * 2 * nsimd; //sum of diagonal elements per site, each sum re/im with SIMD vectorization
    double tavg = total_time_trace/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Trace(SCFmatrixField) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }

  if(0){
    //unop Trace
    Float total_time_trace = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = unop(m1, _tr());
    }
    total_time_trace += dclock();
    
    double Flops = m1.size() * 24 * 2 * nsimd; //sum of diagonal elements per site, each sum re/im with SIMD vectorization
    double tavg = total_time_trace/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Unop trace(SCFmatrixField) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }


  if(0){
    //unopV Trace
    Float total_time_trace = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = unop_v(m1, _trV<typename SCFmatrixField::FieldSiteType>());
    }
    total_time_trace += dclock();
    
    double Flops = m1.size() * 24 * 2 * nsimd; //sum of diagonal elements per site, each sum re/im with SIMD vectorization
    double tavg = total_time_trace/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Unop_v trace(SCFmatrixField) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }


  if(0){
    //Trace * trace
    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = Trace(m1) * Trace(m2);
    }
    total_time += dclock();
    
    double tr_Flops = m1.size() * 24 * 2 * nsimd; //sum of diagonal elements per site, each sum re/im with SIMD vectorization
    double mul_flops = m1.size() * nsimd * 6; //complex mult

    double Flops = 2*tr_Flops + mul_flops;
    double tavg = total_time/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Trace(SCFmatrixField)*Trace(SCFmatrixField) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }

  if(0){
    //binop(Trace * trace)
    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = binop(m1, m2, _trtr());
    }
    total_time += dclock();
    
    double tr_Flops = m1.size() * 24 * 2 * nsimd; //sum of diagonal elements per site, each sum re/im with SIMD vectorization
    double mul_flops = m1.size() * nsimd * 6; //complex mult

    double Flops = 2*tr_Flops + mul_flops;
    double tavg = total_time/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Binop Trace(SCFmatrixField)*Trace(SCFmatrixField) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }

  if(0){
    //binop_v(Trace * trace)
    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = binop_v(m1,m2, _trtrV<typename SCFmatrixField::FieldSiteType>());
    }
    total_time += dclock();
    
    double tr_Flops = m1.size() * 24 * 2 * nsimd; //sum of diagonal elements per site, each sum re/im with SIMD vectorization
    double mul_flops = m1.size() * nsimd * 6; //complex mult

    double Flops = 2*tr_Flops + mul_flops;
    double tavg = total_time/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Binop_v Trace(SCFmatrixField)*Trace(SCFmatrixField) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }

  if(0){
    //operator* 
    SCFmatrixField m3(simd_dims);

    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      m3 = m1 * m2;
    }
    total_time += dclock();
    
    //24*24 matrix multiply
    //Flops = 24* 24 * ( 24 madds )     madd = 8 Flops * nsimd
    double Flops = m1.size() * 24 * 24 * ( 24 * 8 * nsimd );
    double tavg = total_time/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("SCFmatrixField*SCFmatrixField %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }

  if(0){
    //operator* binop_v
    SCFmatrixField m3(simd_dims);

    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      m3 = binop_v(m1,m2, _timesV<typename SCFmatrixField::FieldSiteType>());
    }
    total_time += dclock();
    
    //24*24 matrix multiply
    //Flops = 24* 24 * ( 24 madds )     madd = 8 Flops * nsimd
    double Flops = m1.size() * 24 * 24 * ( 24 * 8 * nsimd );
    double tavg = total_time/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Binop_v SCFmatrixField*SCFmatrixField %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }

  if(0){
    //gl
    SCFmatrixField m3 = m1;

    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      gl(m3, 0);
    }
    total_time += dclock();
    double tavg = total_time/ntests;
    
    printf("gl(SCFmatrixField, 0) %d iters: %g secs\n",ntests,tavg);
  }


  if(0){
    //gl_r
    SCFmatrixField m3 = gl_r(m1,0);

    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      m3 = gl_r(m1, 0);
    }
    total_time += dclock();
    double tavg = total_time/ntests;
    
    printf("gl_r(SCFmatrixField, 0) %d iters: %g secs\n",ntests,tavg);
  }


  if(0){
    //Trace(M1*M2)
    tr_m1 = Trace(m1,m2);

    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = Trace(m1,m2);
    }
    total_time += dclock();
    
    //\sum_{ij} a_{ij}b_{ji}
    double Flops = m1.size() * 24 * 24 * 8 * nsimd;
    double tavg = total_time/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Trace(SCFmatrixField*SCFmatrixField) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }


  if(1){    
    //Trace(M1*Ms)  where Ms is not a field
    CPSspinColorFlavorMatrix<ComplexType> Ms = *m2.site_ptr(size_t(0));
    tr_m1 = Trace(m1,Ms);

    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = Trace(m1,Ms);
    }
    total_time += dclock();
    
    //\sum_{ij} a_{ij}b_{ji}
    double Flops = m1.size() * 24 * 24 * 8 * nsimd;
    double tavg = total_time/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Trace(SCFmatrixField*SCFmatrix) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }





}



#endif //USE_GRID





CPS_END_NAMESPACE

#endif
