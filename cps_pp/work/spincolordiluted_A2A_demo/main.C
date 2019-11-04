#include <alg/a2a/a2a.h>
#include <alg/a2a/utils_main.h>
#include <alg/a2a/grid_wrappers.h>


USING_NAMESPACE_CPS

//A propagator G(x,x) = \sum_i V_i(x) W^\dag_i(x)
//Fortunately this is node local

static   QPropWArg qpropw_arg;

//x is a *node local* lattice size 
template<typename A2Apolicies>
WilsonMatrix selfContractionBasic(int x[4], const A2AvectorV<A2Apolicies> &V, const A2AvectorW<A2Apolicies> &W){
  WilsonMatrix out;
  memset( (void*)&out, 0, 12*12*2*sizeof(double));

  int x3d = x[0] + GJP.XnodeSites()*( x[1] + GJP.YnodeSites()*x[2] );

  const int nmode = V.getNmodes();

  Rcomplex Vx[12], Wdagx[12];
  for(int mode=0;mode<nmode;mode++){
    for(int sc=0;sc<12;sc++){
      Vx[sc] = V.elem(mode, x3d, x[3], sc, 0);
      Wdagx[sc] = std::conj(W.elem(mode, x3d, x[3], sc, 0));
    }
    for(int s1=0;s1<4;s1++){
      for(int c1=0;c1<3;c1++){
	int sc1 = c1 + 3*s1;

	for(int s2=0;s2<4;s2++){
	  for(int c2=0;c2<3;c2++){
	    int sc2 = c2 + 3*s2;
	    
	    out(s1,c1,s2,c2) += Vx[sc1] * Wdagx[sc2];
	  }
	}
      }
    }
  }    
  return out;
}

//This should be tested by running with 0 low modes and using the high mode sources contained in the W field to generate normal volume stochastic propagators
//QPropWRandVolSrc uses the same 4D complex field for all spin and color
template<typename A2Apolicies>
void testRandVol(const A2AvectorW<A2Apolicies> &W, Lattice &lat, double mass){
//  assert(W.getNl() == 0); //makes sure 0 low modes
  assert(W.getNhits() == 1);
  typedef CPScomplex4D<cps::Rcomplex, FourDpolicy<DynamicFlavorPolicy>, StandardAllocPolicy> ComplexFieldType;
  const ComplexFieldType &rand_field = W.getWh(0);

  CgArg &cg = qpropw_arg.cg;
  cg.mass = mass;
  cg.epsilon = 0;
  cg.max_num_iter = 10000;
  cg.stop_rsd = 1e-08;
  cg.true_rsd = 1e-08;
  cg.RitzMatOper = MATPCDAG_MATPC;
  cg.Inverter = CG_LOWMODE_DEFL;

  qpropw_arg.file = "";
  qpropw_arg.flavor = 0;
  qpropw_arg.gauge_fix_src = 0;
  qpropw_arg.gauge_fix_snk = 0;
  qpropw_arg.store_midprop = 0;
  qpropw_arg.save_ls_prop = 0;
  qpropw_arg.do_half_fermion = 0;
  qpropw_arg.ensemble_label = "";
  qpropw_arg.ensemble_id = "";
  qpropw_arg.StartSrcSpin = 0;
  qpropw_arg.EndSrcSpin = 3;
  qpropw_arg.StartSrcColor = 0;
  qpropw_arg.EndSrcColor = 2;

  CommonArg carg;
  QPropWRandArg rand_arg;
  
  QPropWRandVolSrc prop(lat, &qpropw_arg, &rand_arg, &carg, (Complex const*)rand_field.ptr());
  
  if(!UniqueID()){
    int x[4] = {0,0,0,0};
    int site = x[0] + GJP.XnodeSites()*(x[1] + GJP.YnodeSites()*(x[2] + GJP.ZnodeSites()*x[3]));
    WilsonMatrix loop = prop[site];
    Complex wdag = std::conj(*rand_field.site_ptr(x));
    loop *= wdag;
    
    Rcomplex tr = loop.Trace();

    std::cout << "TEST: " << std::real(tr) << " " << std::imag(tr) << std::endl;
  }

}




//Defines a number of types used internally to the A2A library including the Grid fields and Dirac operator type
typedef A2ApoliciesDoubleAutoAlloc A2Apolicies;

int main (int argc,char **argv )
{
  const char *fname="main(int,char**)";
  Start(&argc, &argv);

  CommonArg carg;
  DoArg do_arg;
  LancArg lanc_arg;
  A2AArg a2a_arg; //note: src_width is the number of timeslices upon which the random source lives. To remove time dilution set src_width = Lt
  CGcontrols cg_arg; //controls for the CG used to form the V vectors

  //---This stuff is needed so that the template can be produced without crashing!
#define I(A,B) do_arg.A = B
  I(x_bc,BND_CND_PRD);
  I(y_bc,BND_CND_PRD);
  I(z_bc,BND_CND_PRD);
  I(t_bc,BND_CND_PRD);
#undef I
  //---

  if(!do_arg.Decode("do_arg.vml","do_arg")){
    do_arg.Encode("do_arg.templ","do_arg");
    VRB.Result("","main","Can't open do_arg.vml!\n");exit(1);
  }
  if(!lanc_arg.Decode("lanc_arg.vml","lanc_arg")){
    lanc_arg.Encode("lanc_arg.templ","lanc_arg");
    VRB.Result("","main","Can't open lanc_arg.vml!\n");exit(1);
  }
  if(!a2a_arg.Decode("a2a_arg.vml","a2a_arg")){
    a2a_arg.Encode("a2a_arg.templ","a2a_arg");
    VRB.Result("","main","Can't open a2a_arg.vml!\n");exit(1);
  }
  if(!cg_arg.Decode("cg_arg.vml","cg_arg")){
    cg_arg.Encode("cg_arg.templ","cg_arg");
    VRB.Result("","main","Can't open cg_arg.vml!\n");exit(1);
  }


  int nthreads = 1;
  //Read command line args                                                                                                                                                                                 
//  nthreads = 4;
  std::cout << "CC " << nthreads << std::endl;
  //Setup CPS
  GJP.Initialize(do_arg);
  GJP.SetNthreads(nthreads);
  //omp_set_num_threads(64);
  //std::cout << "Number of threads " << omp_get_max_threads() << std::endl;
#ifdef USE_GRID
  Grid::GridThread::SetThreads(nthreads);
#endif
  
  //Initialize FGrid
  FgridParams fgp;

  typedef A2Apolicies::FgridGFclass LatticeType;
  typedef typename A2Apolicies::GridFermionField GridFermionField;
  typedef typename A2Apolicies::GridFermionFieldF GridFermionFieldF;
  typedef typename A2Apolicies::FgridFclass FgridFclass;
  typedef typename A2Apolicies::GridDirac GridDirac;
  LatticeType lattice(fgp);
  lattice.ImportGauge();
  //Mobius parameters
  const double mob_b = 1.0;
  const double mob_c = 0.0;   //b-c = 1
  fgp.mobius_scale = mob_b + mob_c; //b+c
  const double M5 = do_arg.dwf_height;
  printf("Grid b=%g c=%g b+c=%g\n",mob_b,mob_c,mob_b+mob_c);

  //Do the Lanczos
  std::cout << "GridLanczosWrapper<A2Apolicies> lanczos" << endl;
  GridLanczosWrapper<A2Apolicies> lanczos;
  std::cout << "lanczos.compute(lanc_arg, lattice)" << endl;
  lanczos.compute(lanc_arg, lattice);

  //Typically we convert the evecs to single precision to save memory
  //(Note split Grid not yet supported if we don't have single-prec evecs)
  lanczos.toSingle(lattice);
  
  A2AvectorW<A2Apolicies> W(a2a_arg);
  A2AvectorV<A2Apolicies> V(a2a_arg);

  W.computeVW(V, lattice, lanczos.evec_f, lanczos.eval, lanc_arg.mass, cg_arg);

  EvecInterfaceGridSinglePrec<A2Apolicies> ev(lanczos.evec_f,lanczos.eval,lattice,lanc_arg.mass);

  FgridFclass &latg = dynamic_cast<FgridFclass&>(lattice);
  //Grids and gauge field
  Grid::GridCartesian *UGrid = latg.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = latg.getUrbGrid();
  Grid::GridCartesian *FGrid = latg.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = latg.getFrbGrid();
  Grid::LatticeGaugeFieldD *Umu = latg.getUmu();
  std::cout<<"UGrid= "<<UGrid<<" UrbGrid= "<<UrbGrid<<" FGrid= "<<FGrid<<" FrbGrid= "<<FrbGrid<<std::endl;

  //const int gparity = GJP.Gparity();

  //Setup Grid Dirac operator
  typename GridDirac::ImplParams params;
  latg.SetParams(params);

  if(0){
  GridDirac Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,lanc_arg.mass,M5,mob_b,mob_c, params);
  Grid::SchurDiagMooeeOperator<GridDirac, GridFermionField> linop(Ddwf);

  GridFermionField sol(FGrid);
  GridFermionField src(FGrid);
  std::cout <<  "Before CGNE_MdagM"<< std::endl;
  ev.CGNE_MdagM(linop,sol,src,cg_arg);
  std::cout <<  "After CGNE_MdagM"<< std::endl;
  //Grid_CGNE_M_high<A2Apolicies>(sol);
  }

  if(!UniqueID()){
    int x[4] = {0,0,0,0};
    WilsonMatrix loop = selfContractionBasic(x,V,W);
    
    Rcomplex tr = loop.Trace();

    std::cout << "Loop trace, site 0: " << std::real(tr) << " " << std::imag(tr) << std::endl;
  }


  const char *evec_name = "light_evec";

  qpropw_arg.cg.fname_eigen = (char *) evec_name;

  int N_evec= lanc_arg.N_true_get;
  qpropw_arg.cg.neig=N_evec;

  if(N_evec>0)
  {

    EigenCacheGrid<GridFermionFieldF>  *ecache = new EigenCacheGrid <GridFermionFieldF> (evec_name);
//    const int n_fields = GJP.SnodeSites ();
//    const size_t f_size_per_site = lattice.FsiteSize () / n_fields / 2;     // checkerboarding
//    size_t evec_size = (size_t) (GJP.VolNodeSites () / 2) * lattice.FsiteSize ();
//    assert(evec_size != lattice.half_size)
//    size_t fsize = evec_size;
//    int data_size = sizeof (Float);
//    if (lanczos_arg.precision == PREC_SINGLE)
//      data_size = sizeof (float);
//    ecache->alloc (N_evec,evec_size, data_size);
    ecache->load(lanczos.eval, lanczos.evec_f);
//    ecache->read_compressed ((char*)evec_dir);
    EigenCacheList.push_back (ecache);
  }



  VRB.Result("",fname,"W.getNl()=%d\n",W.getNl());
//  if(W.getNl() == 0) 
  testRandVol(W, lattice, lanc_arg.mass);
  /*
  */
  if(!UniqueID()) printf("Done\n");
  End();
}

