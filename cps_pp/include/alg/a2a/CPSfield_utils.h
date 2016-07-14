#ifndef CPS_FIELD_UTILS_H
#define CPS_FIELD_UTILS_H

CPS_START_NAMESPACE

inline void compareFermion(const CPSfermion5D<ComplexD> &A, const CPSfermion5D<ComplexD> &B, const std::string &descr = "Ferms", const double tol = 1e-9){
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

template<typename FieldType, typename my_enable_if<_equal<typename ComplexClassify<typename FieldType::FieldSiteType>::type, complex_double_or_float_mark>::value,int>::type = 0>
inline void compareField(const FieldType &A, const FieldType &B, const std::string &descr = "Field", const double tol = 1e-9, bool print_all = false){
  typedef typename FieldType::FieldSiteType::value_type value_type;
  
  double fail = 0.;
  for(int xf=0;xf<A.nfsites();xf++){
    int f; int x[FieldType::FieldDimensionPolicy::EuclideanDimension];
    A.fsiteUnmap(xf, x,f);

    for(int i=0;i<FieldType::FieldSiteSize;i++){
      value_type const* av = (value_type const*)(A.fsite_ptr(xf)+i);
      value_type const* bv = (value_type const*)(B.fsite_ptr(xf)+i);
      for(int reim=0;reim<2;reim++){
	value_type diff_rat = (av[reim] == 0.0 && bv[reim] == 0.0) ? 0.0 : fabs( 2.*(av[reim]-bv[reim])/(av[reim]+bv[reim]) );
	if(diff_rat > tol || print_all){
	  if(!print_all) std::cout << "Fail: (";
	  else std::cout << "Pass: (";
	  
	  for(int xx=0;xx<FieldType::FieldDimensionPolicy::EuclideanDimension-1;xx++)
	    std::cout << x[xx] << ", ";
	  std::cout << x[FieldType::FieldDimensionPolicy::EuclideanDimension-1];

	  std::cout << ") f=" << f << " reim " << reim << " A " << av[reim] << " B " << bv[reim] << " fracdiff " << diff_rat << std::endl;
	  if(!print_all) fail = 1.;
	}
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
inline void exportBFMcb(CPSfermion5D<ComplexD> &into, Fermion_t from, bfm_evo<double> &dwf, int cb, bool singleprec_evec = false){
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
template<typename GridPolicies>
inline void exportGridcb(CPSfermion5D<ComplexD> &into, typename GridPolicies::GridFermionField &from, typename GridPolicies::FgridFclass &latg){
  Grid::GridCartesian *FGrid = latg.getFGrid();
  typename GridPolicies::GridFermionField tmp_g(FGrid);
  tmp_g = Grid::zero;

  setCheckerboard(tmp_g, from);
  latg.ImportFermion((Vector*)into.ptr(), tmp_g);
}
#endif

CPS_END_NAMESPACE
#endif
