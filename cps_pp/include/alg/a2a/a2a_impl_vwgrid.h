CPS_END_NAMESPACE
#include<util/lattice/fgrid.h>
CPS_START_NAMESPACE


inline void compareFermion(const CPSfermion5D<double> &A, const CPSfermion5D<double> &B, const std::string &descr = "Ferms", const double tol = 1e-9){
  assert(sizeof(std::complex<double>) == 2*sizeof(double));
  double fail = 0.;
  for(int i=0;i<GJP.VolNodeSites()*GJP.SnodeSites();i++){
    int x[5]; int rem = i;
    for(int ii=0;ii<5;ii++){ x[ii] = rem % GJP.NodeSites(ii); rem /= GJP.NodeSites(ii); }
    
    for(int f=0;f<GJP.Gparity()+1;f++){
      for(int sc=0;sc<24;sc++){
	double vbfm = *((double*)A.site_ptr(i,f) + sc);
	double vgrid = *((double*)B.site_ptr(i,f) + sc);
	    
	double diff_rat = fabs( 2.0 * ( vbfm - vgrid )/( vbfm + vgrid ) );
	double rat_grid_bfm = vbfm/vgrid;
	if(vbfm == 0.0 && vgrid == 0.0){ diff_rat = 0.;	 rat_grid_bfm = 1.; }
	if( (vbfm == 0.0 && fabs(vgrid) < 1e-50) || (vgrid == 0.0 && fabs(vbfm) < 1e-50) ){ diff_rat = 0.;	 rat_grid_bfm = 1.; }

	if(diff_rat > tol){
	  printf("Fail: (%d,%d,%d,%d,%d; %d; %d) A %g B %g rat_A_B %g fracdiff %g\n",x[0],x[1],x[2],x[3],x[4],f,sc,vbfm,vgrid,rat_grid_bfm,diff_rat);
	  fail = 1.0;
	}//else printf("Pass: (%d,%d,%d,%d,%d; %d; %d) A %g B %g rat_A_B %g fracdiff %g\n",x[0],x[1],x[2],x[3],x[4],f,sc,vbfm,vgrid,rat_grid_bfm,diff_rat);
      }
    }
  }
  glb_max(&fail);
  
  if(fail!=0.0){
    if(!UniqueID()){ printf("Failed %s check\n", descr.c_str()); fflush(stdout); } 
    exit(-1);
  }else{
    if(!UniqueID()){ printf("Passed %s check\n", descr.c_str()); fflush(stdout); }
  }
}

#ifdef USE_BFM
inline void exportBFMcb(CPSfermion5D<double> &into, Fermion_t from, bfm_evo<double> &dwf, int cb, bool singleprec_evec = false){
  Fermion_t zero_a = dwf.allocFermion();
#pragma omp parallel
  {   
    dwf.set_zero(zero_a); 
  }
  Fermion_t etmp = dwf.allocFermion(); 
  Fermion_t tmp[2];
  tmp[!cb] = zero_a;
  if(singleprec_evec){
    const int len = 24 * dwf.node_cbvol * (1 + dwf.gparity) * dwf.cbLs;
#pragma omp parallel for
    for(int j = 0; j < len; j++) {
      ((double*)etmp)[j] = ((float*)(from))[j];
    }
    tmp[cb] = etmp;
  }else tmp[cb] = from;

  dwf.cps_impexFermion(into.ptr(),tmp,0);
  dwf.freeFermion(zero_a);
  dwf.freeFermion(etmp);
}
#endif

#ifdef USE_GRID
inline void exportGridcb(CPSfermion5D<double> &into, LATTICE_FERMION &from, FGRID &latg){
  Grid::GridCartesian *FGrid = latg.getFGrid();
  LATTICE_FERMION tmp_g(FGrid);
  tmp_g = Grid::zero;

  setCheckerboard(tmp_g, from);
  latg.ImportFermion((Vector*)into.ptr(), tmp_g);
}
#endif

//BFM evecs
#ifdef USE_BFM_LANCZOS

class EvecInterfaceBFM: public EvecInterface{
  BFM_Krylov::Lanczos_5d<double> &eig;
  bfm_evo<double> &dwf;
  FGRID *latg;
  double *cps_tmp_d;
  Fermion_t bq_tmp_bfm;
  bool singleprec_evecs;
  int len;
  LATTICE_FERMION *tmp_full;
public:
  EvecInterfaceBFM(BFM_Krylov::Lanczos_5d<double> &_eig, bfm_evo<double> &_dwf, Lattice &lat, const bool _singleprec_evecs): eig(_eig), dwf(_dwf), singleprec_evecs(_singleprec_evecs){
    len = 24 * eig.dop.node_cbvol * (1 + dwf.gparity) * eig.dop.cbLs;
    cps_tmp_d = (double*)malloc(len * sizeof(double));
    bq_tmp_bfm = dwf.allocCompactFermion(); 

    assert(lat.Fclass() == F_CLASS_GRID);
    assert(dwf.precon_5d == 0);
    latg = dynamic_cast<FGRID*>(&lat);

    Grid::GridCartesian *FGrid = latg->getFGrid();
    tmp_full = new LATTICE_FERMION(FGrid);

    const int gparity = GJP.Gparity();
    if(eig.dop.gparity != gparity){ ERR.General("EvecInterfaceBFM","EvecInterfaceBFM","Gparity must be disabled/enabled for *both* CPS and the eigenvectors"); }
  }
  Float getEvec(LATTICE_FERMION &into, const int idx){
    omp_set_num_threads(bfmarg::threads);
    
    //Copy bq[i][1] into bq_tmp
    if(singleprec_evecs){ // eig->bq is in single precision
      //Upcast the float type to double
#pragma omp parallel for 
      for(int j = 0; j < len; j++) {
	((double*)bq_tmp_bfm)[j] = ((float*)(eig.bq[idx][1]))[j];
      }
      //Use bfm_evo to convert to a CPS field
      dwf.cps_impexcbFermion<double>(cps_tmp_d, bq_tmp_bfm, 0, Odd);

    }else{ // eig.bq is in double precision
      //Use bfm_evo to convert to a CPS field
      dwf.cps_impexcbFermion<double>(cps_tmp_d, eig.bq[idx][1], 0, Odd);     
    }
    //Use Fgrid to convert to a Grid field
    *tmp_full = Grid::zero;
    latg->ImportFermion(*tmp_full, (Vector*)cps_tmp_d, FgridBase::Odd);
    pickCheckerboard(Odd,into,*tmp_full);

    return eig.evals[idx];
  }
  int nEvecs() const{
    return eig.get;
  }

  ~EvecInterfaceBFM(){
    free(cps_tmp_d);
    dwf.freeFermion(bq_tmp_bfm);
    delete tmp_full;
  }

};


//Compute the low mode part of the W and V vectors. In the Lanczos class you can choose to store the vectors in single precision (despite the overall precision, which is fixed to double here)
//Set 'singleprec_evecs' if this has been done
template< typename mf_Float>
void A2AvectorW<mf_Float>::computeVWlow(A2AvectorV<mf_Float> &V, Lattice &lat, BFM_Krylov::Lanczos_5d<double> &eig, bfm_evo<double> &dwf, bool singleprec_evecs){
  EvecInterfaceBFM ev(eig,dwf,lat,singleprec_evecs);
  return computeVWlow(V,lat,ev,dwf.mass);
}

template< typename mf_Float>
void A2AvectorW<mf_Float>::computeVWhigh(A2AvectorV<mf_Float> &V, BFM_Krylov::Lanczos_5d<double> &eig, bool singleprec_evecs, Lattice &lat, bfm_evo<double> &dwf_d, bfm_evo<float> *dwf_fp){
  bool mixed_prec_cg = dwf_fp != NULL; 
  if(mixed_prec_cg){
    //NOT IMPLEMENTED YET
    ERR.General(cname.c_str(),"computeVWhigh","No grid implementation of mixed precision CG\n");
  }

  if(mixed_prec_cg && !singleprec_evecs){ ERR.General(cname.c_str(),"computeVWhigh","If using mixed precision CG, input eigenvectors must be stored in single precision"); }

  EvecInterfaceBFM ev(eig,dwf_d,lat,singleprec_evecs);
  return computeVWhigh(V,lat,ev,dwf_d.mass,dwf_d.residual,dwf_d.max_iter);
}

#endif



//Grid evecs
#ifdef USE_GRID_LANCZOS

class EvecInterfaceGrid: public EvecInterface{
  const std::vector<Grid::RealD> &eval; 
  const std::vector<LATTICE_FERMION> &evec;

public:
  EvecInterfaceGrid(const std::vector<LATTICE_FERMION> &_evec, const std::vector<Grid::RealD> &_eval): evec(_evec), eval(_eval){}

  Float getEvec(LATTICE_FERMION &into, const int idx){
    into = evec[idx];
    return eval[idx];
  }
  int nEvecs() const{
    return eval.size();
  }
};

template< typename mf_Float>
void A2AvectorW<mf_Float>::computeVWlow(A2AvectorV<mf_Float> &V, Lattice &lat, const std::vector<LATTICE_FERMION> &evec, const std::vector<Grid::RealD> &eval, const double mass){
  EvecInterfaceGrid ev(evec,eval);
  return computeVWlow(V,lat,ev,mass);
}

template< typename mf_Float>
void A2AvectorW<mf_Float>::computeVWhigh(A2AvectorV<mf_Float> &V, Lattice &lat, const std::vector<LATTICE_FERMION> &evec, const std::vector<Grid::RealD> &eval, const double mass, const Float residual, const int max_iter){
  EvecInterfaceGrid ev(evec,eval);
  return computeVWhigh(V,lat,ev,mass,residual,max_iter);
}
#endif




template< typename mf_Float>
void A2AvectorW<mf_Float>::computeVWlow(A2AvectorV<mf_Float> &V, Lattice &lat, EvecInterface &evecs, const Float mass){
  if(!UniqueID()) printf("Computing VWlow using Grid\n");
  
  const char *fname = "computeVQlow(....)";

  int ngp = 0;
  for(int i=0;i<3;i++) if(GJP.Bc(i) == BND_CND_GPARITY) ++ngp;

#ifdef GRID_GPARITY
  if(ngp == 0) ERR.General("A2AvectorW","computeVWlow","Fgrid is currently compiled for G-parity\n");
#else
  if(ngp != 0) ERR.General("A2AvectorW","computeVWlow","Fgrid is not currently compiled for G-parity\n");
#endif

  assert(lat.Fclass() == FGRID_CLASS_NAME);
  assert(sizeof(std::complex<mf_Float>) == 2*sizeof(mf_Float));
  FGRID &latg = dynamic_cast<FGRID&>(lat);

  //Grids and gauge field
  Grid::GridCartesian *UGrid = latg.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = latg.getUrbGrid();
  Grid::GridCartesian *FGrid = latg.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = latg.getFrbGrid();
  Grid::QCD::LatticeGaugeFieldD *Umu = latg.getUmu();
  
  //Mobius parameters
  const double mob_b = latg.get_mob_b();
  const double mob_c = mob_b - 1.;   //b-c = 1
  const double M5 = GJP.DwfHeight();
  printf("Grid b=%g c=%g b+c=%g\n",mob_b,mob_c,mob_b+mob_c);

  const int gparity = GJP.Gparity();

  //Double precision temp fields
  CPSfermion4D<Float> afield;  Vector* a = (Vector*)afield.ptr(); //breaks encapsulation, but I can sort this out later.
  CPSfermion5D<Float> bfield;  Vector* b = (Vector*)bfield.ptr();

  int afield_fsize = afield.size()*sizeof(CPSfermion4D<Float>::FieldSiteType)/sizeof(Float); //number of floats in field
  
  const int glb_ls = GJP.SnodeSites() * GJP.Snodes();

  //Setup Grid Dirac operator
  DIRAC ::ImplParams params;
  latg.SetParams(params);

  DIRAC Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c, params);
  Grid::SchurDiagMooeeOperator<DIRAC, LATTICE_FERMION> linop(Ddwf);

  //Eigenvectors exist on odd checkerboard
  LATTICE_FERMION bq_tmp(FrbGrid);
  LATTICE_FERMION tmp(FrbGrid);
  LATTICE_FERMION tmp2(FrbGrid);
  LATTICE_FERMION tmp3(FrbGrid);

  LATTICE_FERMION tmp_full(FGrid);
  LATTICE_FERMION tmp_full2(FGrid);

  //The general method is described by page 60 of Daiqian's thesis
  for(int i = 0; i < nl; i++) {
    //Step 1) Compute V
    mf_Float* vi = (mf_Float*)V.getVl(i).ptr();
    
    Float eval = evecs.getEvec(bq_tmp,i);
    assert(bq_tmp.checkerboard == Grid::Odd);

    //Compute  [ -(Mee)^-1 Meo bq_tmp, bg_tmp ]
    Ddwf.Meooe(bq_tmp,tmp2);	//tmp2 = Meo bq_tmp 
    Ddwf.MooeeInv(tmp2,tmp);   //tmp = (Mee)^-1 Meo bq_tmp
    tmp = -tmp; //even checkerboard
    
    assert(tmp.checkerboard == Grid::Even);
    
    setCheckerboard(tmp_full, tmp); //even checkerboard
    setCheckerboard(tmp_full, bq_tmp); //odd checkerboard

    //Get 4D part and poke into a
    latg.ImportFermion(b,tmp_full,FgridBase::All);
    lat.Ffive2four(a,b,glb_ls-1,0,2); // a[4d] = b[5d walls]
    //Multiply by 1/lambda[i] and copy into v (with precision change if necessary)
    VecTimesEquFloat<mf_Float,Float>(vi, (Float*)a, 1.0 / eval, afield_fsize);

    //Step 2) Compute Wl

    //Do tmp = [ -[Mee^-1]^dag [Meo]^dag Doo bq_tmp,  Doo bq_tmp ]    (Note that for the Moe^dag in Daiqian's thesis, the dagger also implies a transpose of the spatial indices, hence the Meo^dag in the code)
    linop.Mpc(bq_tmp,tmp2);  //tmp2 = Doo bq_tmp
    
    Ddwf.MeooeDag(tmp2,tmp3); //tmp3 = Meo^dag Doo bq_tmp
    Ddwf.MooeeInvDag(tmp3,tmp); //tmp = [Mee^-1]^dag Meo^dag Doo bq_tmp
    tmp = -tmp;
    
    assert(tmp.checkerboard == Grid::Even);
    assert(tmp2.checkerboard == Grid::Odd);

    setCheckerboard(tmp_full, tmp);
    setCheckerboard(tmp_full, tmp2);

    //Left-multiply by D-^dag.  D- = (1-c*DW)
    Ddwf.DW(tmp_full, tmp_full2, 1);
    axpy(tmp_full, -mob_c, tmp_full2, tmp_full); 

    //Get 4D part, poke onto a then copy into wl
    latg.ImportFermion(b,tmp_full,FgridBase::All);
    lat.Ffive2four(a,b,0,glb_ls-1, 2);
    VecTimesEquFloat<mf_Float,Float>((mf_Float*)wl[i].ptr(), (Float*)a, 1.0, afield_fsize);
  }
}








//nLowMode is the number of modes we actually use to deflate. This must be <= evals.size(). The full set of computed eigenvectors is used to improve the guess.
inline void Grid_CGNE_M_high(LATTICE_FERMION &solution, const LATTICE_FERMION &source, double resid, int max_iters, EvecInterface &evecs, int nLowMode, 
			     FGRID &latg, DIRAC &Ddwf, Grid::GridCartesian *FGrid, Grid::GridRedBlackCartesian *FrbGrid){
  double f = norm2(source);
  if (!UniqueID()) printf("Grid_CGNE_M_high: Source norm is %le\n",f);
  f = norm2(solution);
  if (!UniqueID()) printf("Grid_CGNE_M_high: Guess norm is %le\n",f);

  Grid::SchurDiagMooeeOperator<DIRAC, LATTICE_FERMION> linop(Ddwf);

  LATTICE_FERMION tmp_cb1(FrbGrid);
  LATTICE_FERMION tmp_cb2(FrbGrid);
  LATTICE_FERMION tmp_cb3(FrbGrid);
  LATTICE_FERMION tmp_cb4(FrbGrid);

  LATTICE_FERMION tmp_full(FGrid);

  // src_o = Mprecdag * (source_o - Moe MeeInv source_e)  , cf Daiqian's thesis page 60
  LATTICE_FERMION src_o(FrbGrid);

  pickCheckerboard(Grid::Even,tmp_cb1,source);  //tmp_cb1 = source_e
  pickCheckerboard(Grid::Odd,tmp_cb2,source);   //tmp_cb2 = source_o

  Ddwf.MooeeInv(tmp_cb1,tmp_cb3);
  Ddwf.Meooe     (tmp_cb3,tmp_cb4); //tmp_cb4 = Moe MeeInv source_e       (tmp_cb3 free)
  axpy    (tmp_cb3,-1.0,tmp_cb4, tmp_cb2); //tmp_cb3 = (source_o - Moe MeeInv source_e)    (tmp_cb4 free)
  linop.MpcDag(tmp_cb3, src_o); //src_o = Mprecdag * (source_o - Moe MeeInv source_e)    (tmp_cb3, tmp_cb4 free)

  //Compute low-mode projection and CG guess
  int Nev = evecs.nEvecs();

  LATTICE_FERMION lsol_full(FrbGrid); //full low-mode part (all evecs)
  lsol_full = Grid::zero;

  LATTICE_FERMION lsol_defl(FrbGrid); //low-mode part for subset of evecs with index < nLowMode

  LATTICE_FERMION sol_o(FrbGrid); //CG solution
  sol_o = Grid::zero;

  if(Nev < nLowMode)
    ERR.General("","Grid_CGNE_M_High","Number of low eigen modes to do deflation is smaller than number of low modes to be substracted!\n");

  if(Nev > 0){
    if (!UniqueID()) printf("Grid_CGNE_M_High: deflating with %d evecs\n",Nev);

    for(int n = 0; n < Nev; n++){
      double eval = evecs.getEvec(tmp_cb1,n);
      Grid::ComplexD cn = innerProduct(tmp_cb1, src_o);	
      axpy(lsol_full, cn / eval, tmp_cb1, lsol_full);

      if(n == nLowMode - 1) lsol_defl = lsol_full;
    }
    sol_o = lsol_full; //sol_o = lsol   Set guess equal to low mode projection 
  }
  
  //Do CG
  Grid::ConjugateGradient<LATTICE_FERMION> CG(resid, max_iters);


  f = norm2(src_o);
  if (!UniqueID()) printf("Grid_CGNE_M_high: CGNE_prec_MdagM src norm %le\n",f);
  f = norm2(sol_o);
  if (!UniqueID()) printf("Grid_CGNE_M_high: CGNE_prec_MdagM guess norm %le\n",f);

  CG(linop, src_o, sol_o);

  f = norm2(sol_o);
  if (!UniqueID()) printf("Grid_CGNE_M_high: CGNE_prec_MdagM sol norm %le\n",f);


  //Pull low-mode part out of solution
  axpy(sol_o, -1.0, lsol_defl, sol_o);
  setCheckerboard(solution, sol_o);

  // sol_e = M_ee^-1 * ( src_e - Meo sol_o )...
  pickCheckerboard(Grid::Even,tmp_cb1,source);  //tmp_cb1 = src_e

  Ddwf.Meooe(sol_o,tmp_cb2); //tmp_cb2 = Meo sol_o
  axpy(tmp_cb1, -1.0, tmp_cb2, tmp_cb1); //tmp_cb1 = (-Meo sol_o + src_e)   (tmp_cb2 free)
  Ddwf.MooeeInv(tmp_cb1,tmp_cb2);  //tmp_cb2 = Mee^-1(-Meo sol_o + src_e)   (tmp_cb1 free)
  
  setCheckerboard(solution, tmp_cb2);

  f = norm2(solution);
  if (!UniqueID()) printf("Grid_CGNE_M_high: unprec sol norm is %le\n",f);
}


//Compute the high mode parts of V and W. 
//singleprec_evecs specifies whether the input eigenvectors are stored in single preciison
//You can optionally pass a single precision bfm instance, which if given will cause the underlying CG to be performed in mixed precision.
//WARNING: if using the mixed precision solve, the eigenvectors *MUST* be in single precision (there is a runtime check)
template< typename mf_Float>
void A2AvectorW<mf_Float>::computeVWhigh(A2AvectorV<mf_Float> &V, Lattice &lat, EvecInterface &evecs, const Float mass, const Float residual, const int max_iter){
  const char *fname = "computeVWhigh(....)";

  int ngp = 0;
  for(int i=0;i<3;i++) if(GJP.Bc(i) == BND_CND_GPARITY) ++ngp;

#ifdef GRID_GPARITY
  if(ngp == 0) ERR.General("A2AvectorW","computeVWlow","Fgrid is currently compiled for G-parity\n");
#else
  if(ngp != 0) ERR.General("A2AvectorW","computeVWlow","Fgrid is not currently compiled for G-parity\n");
#endif

  assert(lat.Fclass() == FGRID_CLASS_NAME);
  FGRID &latg = dynamic_cast<FGRID&>(lat);

  //Grids and gauge field
  Grid::GridCartesian *UGrid = latg.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = latg.getUrbGrid();
  Grid::GridCartesian *FGrid = latg.getFGrid();
  Grid::GridRedBlackCartesian *FrbGrid = latg.getFrbGrid();
  Grid::QCD::LatticeGaugeFieldD *Umu = latg.getUmu();
  
  //Mobius parameters
  const double mob_b = latg.get_mob_b();
  const double mob_c = mob_b - 1.;   //b-c = 1
  const double M5 = GJP.DwfHeight();
  printf("Grid b=%g c=%g b+c=%g\n",mob_b,mob_c,mob_b+mob_c);

  const int gparity = GJP.Gparity();

  //Setup Grid Dirac operator
  DIRAC ::ImplParams params;
  latg.SetParams(params);

  DIRAC Ddwf(*Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c, params);
  Grid::SchurDiagMooeeOperator<DIRAC, LATTICE_FERMION> linop(Ddwf);

  VRB.Result(cname.c_str(), fname, "Start computing high modes using Grid.\n");
    
  //Generate the compact random sources for the high modes
  setWhRandom(args.rand_type);

  //Allocate temp *double precision* storage for fermions
  CPSfermion5D<Float> afield,bfield;
  CPSfermion4D<Float> v4dfield;
  
  int v4dfield_fsize = v4dfield.size()*sizeof(CPSfermion4D<Float>::FieldSiteType)/sizeof(Float); //number of floats in field
  
  Vector *a = (Vector*)afield.ptr(), *b = (Vector*)bfield.ptr(), *v4d = (Vector*)v4dfield.ptr();

  const int glb_ls = GJP.SnodeSites() * GJP.Snodes();

  LATTICE_FERMION gtmp(FrbGrid);
  LATTICE_FERMION gtmp2(FrbGrid);
  LATTICE_FERMION gtmp3(FrbGrid);

  LATTICE_FERMION gsrc(FGrid);
  LATTICE_FERMION gtmp_full(FGrid);
  LATTICE_FERMION gtmp_full2(FGrid);

  //Details of this process can be found in Daiqian's thesis, page 60
  for(int i=0; i<nh; i++){
    //Step 1) Get the diluted W vector to invert upon
    getDilutedSource(v4dfield, i);

    //Step 2) Solve V
    lat.Ffour2five(a, v4d, 0, glb_ls-1, 2); // poke the diluted 4D source onto a 5D source    
    latg.ImportFermion(gsrc, (Vector*)a);

    //Left-multiply by D-.  D- = (1-c*DW)
    Ddwf.DW(gsrc, gtmp_full, Grid::DaggerNo);
    axpy(gsrc, -mob_c, gtmp_full, gsrc); 

    //We can re-use previously computed solutions to speed up the calculation if rerunning for a second mass by using them as a guess
    //If no previously computed solutions this wastes a few flops, but not enough to care about
    //V vectors default to zero, so this is a zero guess if not reusing existing solutions
    VecTimesEquFloat<Float,mf_Float>((Float*)v4d, (mf_Float*)V.getVh(i).ptr(), 1.0, v4dfield_fsize); // v[i]->v4d to double precision
    lat.Ffour2five(a, v4d, 0, glb_ls-1, 2); // to 5d

    latg.ImportFermion(gtmp_full, (Vector*)a);
    Ddwf.DW(gtmp_full, gtmp_full2, Grid::DaggerNo);
    axpy(gtmp_full, -mob_c, gtmp_full2, gtmp_full); 

    //Do the CG
    Grid_CGNE_M_high(gtmp_full, gsrc, residual, max_iter, evecs, nl, latg, Ddwf, FGrid, FrbGrid);
 
    //CPSify the solution, including 1/nhit for the hit average
    latg.ImportFermion((Vector*)b, gtmp_full);
    lat.Ffive2four(v4d, b, glb_ls-1, 0, 2);
    VecTimesEquFloat<mf_Float,Float>((mf_Float*)V.getVh(i).ptr(), (Float*)v4d, 1.0 / nhits, v4dfield_fsize);
  }
}
