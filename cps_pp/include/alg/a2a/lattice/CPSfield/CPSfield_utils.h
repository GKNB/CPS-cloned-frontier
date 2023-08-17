#ifndef CPS_FIELD_UTILS_H
#define CPS_FIELD_UTILS_H

#include <map>
#include "CPSfield.h"

CPS_START_NAMESPACE
#include "implementation/CPSfield_utils_impl.tcc"
#include "implementation/CPSfield_utils_cyclicpermute.tcc"


//Compare 5d fermion fields
inline void compareFermion(const CPSfermion5D<ComplexD> &A, const CPSfermion5D<ComplexD> &B, const std::string &descr = "Ferms", const double tol = 1e-9){
  double fail = 0.;
  for(int i=0;i<GJP.VolNodeSites()*GJP.SnodeSites();i++){
    int x[5]; int rem = i;
    for(int ii=0;ii<5;ii++){ x[ii] = rem % GJP.NodeSites(ii); rem /= GJP.NodeSites(ii); }
    
    CPSautoView(A_v,A,HostRead);
    CPSautoView(B_v,B,HostRead);
    
    for(int f=0;f<GJP.Gparity()+1;f++){
      for(int sc=0;sc<24;sc++){
	double vbfm = *((double*)A_v.site_ptr(i,f) + sc);
	double vgrid = *((double*)B_v.site_ptr(i,f) + sc);
	    
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
    a2a_printf("Failed %s check\n", descr.c_str());
    exit(-1);
  }else{
    a2a_printf("Passed %s check\n", descr.c_str());
  }
}

template<typename FieldTypeA,typename FieldTypeB>
struct CPSfieldTypesSameUpToAllocPolicy{
  enum { value =
    std::is_same<typename FieldTypeA::FieldSiteType,typename FieldTypeB::FieldSiteType>::value
    &&
    FieldTypeA::FieldSiteSize == FieldTypeA::FieldSiteSize
    &&
    std::is_same<typename FieldTypeA::FieldMappingPolicy,typename FieldTypeB::FieldMappingPolicy>::value
  };
};


//Compare general CPSfield
template<typename FieldTypeA, typename FieldTypeB, typename std::enable_if<
						     std::is_same<typename ComplexClassify<typename FieldTypeA::FieldSiteType>::type, complex_double_or_float_mark>::value
						     &&
						     CPSfieldTypesSameUpToAllocPolicy<FieldTypeA,FieldTypeB>::value
						     ,int>::type = 0>
inline void compareField(const FieldTypeA &A, const FieldTypeB &B, const std::string &descr = "Field", const double tol = 1e-9, bool print_all = false){
  typedef typename FieldTypeA::FieldSiteType::value_type value_type;

  CPSautoView(A_v,A,HostRead);
  CPSautoView(B_v,B,HostRead);
  double fail = 0.;
  for(int xf=0;xf<A.nfsites();xf++){
    int f; int x[FieldTypeA::FieldMappingPolicy::EuclideanDimension];
    A.fsiteUnmap(xf, x,f);

    for(int i=0;i<FieldTypeA::FieldSiteSize;i++){
      value_type const* av = (value_type const*)(A_v.fsite_ptr(xf)+i);
      value_type const* bv = (value_type const*)(B_v.fsite_ptr(xf)+i);
      for(int reim=0;reim<2;reim++){
	value_type diff_rat = (av[reim] == 0.0 && bv[reim] == 0.0) ? 0.0 : fabs( 2.*(av[reim]-bv[reim])/(av[reim]+bv[reim]) );
	if(diff_rat > tol || print_all){
	  if(diff_rat > tol) std::cout << "!FAIL: ";
	  else std::cout << "Pass: ";
	  
	  std::cout << "coord=(";
	  for(int xx=0;xx<FieldTypeA::FieldMappingPolicy::EuclideanDimension-1;xx++)
	    std::cout << x[xx] << ", ";
	  std::cout << x[FieldTypeA::FieldMappingPolicy::EuclideanDimension-1];

	  std::cout << ") f=" << f << " i=" << i << " reim=" << reim << " A " << av[reim] << " B " << bv[reim] << " fracdiff " << diff_rat << std::endl;
	  if(diff_rat > tol) fail = 1.;
	}
      }
    }
  }
  glb_max(&fail);
  
  if(fail!=0.0){
    a2a_printf("Failed %s check\n", descr.c_str());
    exit(-1);
  }else{
    a2a_printf("Passed %s check\n", descr.c_str());
  }
}


#ifdef USE_GRID
//Export a checkerboarded Grid field to an un-checkerboarded CPS field
template<typename GridPolicies>
inline void exportGridcb(CPSfermion5D<cps::ComplexD> &into, typename GridPolicies::GridFermionField &from, typename GridPolicies::FgridFclass &latg){
  Grid::GridCartesian *FGrid = latg.getFGrid();
  typename GridPolicies::GridFermionField tmp_g(FGrid);
  tmp_g = Grid::Zero();

  setCheckerboard(tmp_g, from);
  CPSautoView(into_v,into,HostWrite);
  latg.ImportFermion((Vector*)into_v.ptr(), tmp_g);
}
#endif



//Cyclic permutation of *4D* and *3D* CPSfield with std::complex type and FourDpolicy dimension policy
//Conventions are direction of *data flow*: For shift n in direction +1   f'(x) = f(x-\hat i)  so data is sent in the +x direction.
template< typename mf_Complex, int SiteSize, typename MappingPolicy, typename AllocPolicy>
void cyclicPermute(CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> &to, const CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> &from,
		   const int dir, const int pm, const int n){
  if(n >= GJP.NodeSites(dir)){ //deal with n > node size
    CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> tmp1(from);
    CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy> tmp2(from);

    CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy>* from_i = &tmp1;
    CPSfield<mf_Complex,SiteSize,MappingPolicy,AllocPolicy>* to_i = &tmp2;
    int nn = n;
    while(nn >= GJP.NodeSites(dir)){
      cyclicPermuteImpl(*to_i, *from_i, dir, pm, GJP.NodeSites(dir)-1);
      nn -= (GJP.NodeSites(dir)-1);
      std::swap(from_i, to_i); //on last iteration the data will be in from_i after leaving the loop
    }
    cyclicPermuteImpl(to, *from_i, dir, pm, nn);
  }else{
    cyclicPermuteImpl(to, from, dir, pm, n);
  }
}


//+1 if of>0,  -1 otherwise
inline int getShiftSign(const int of){ return of > 0 ? +1 : -1; }


//Invoke multiple independent permutes to offset field by vector 'shift' assuming field is periodic
template<typename FieldType>
void shiftPeriodicField(FieldType &to, const FieldType &from, const std::vector<int> &shift){
  int nd = shift.size(); //assume ascending: x,y,z,t
  int nshift_dirs = 0;
  for(int i=0;i<nd;i++) if(shift[i]!=0) ++nshift_dirs;

  if(nshift_dirs == 0){
    if(&to != &from) to = from;
    return;
  }else if(nshift_dirs == 1){
    for(int d=0;d<nd;d++){
      if(shift[d] != 0){
	cyclicPermute(to,from,d,getShiftSign(shift[d]),abs(shift[d]) );
	return;
      }
    }    
  }else{
    FieldType tmp1 = from;
    FieldType tmp2 = from;
    FieldType * send = &tmp1;
    FieldType * recv = &tmp2;

    int shifts_done = 0;
    for(int d=0;d<nd;d++){
      if(shift[d] != 0){
	cyclicPermute(shifts_done < nshift_dirs-1 ? *recv : to,*send,d,getShiftSign(shift[d]),abs(shift[d]) );
	++shifts_done;
	if(shifts_done < nshift_dirs) std::swap(send,recv);
	else return;
      }
    }   
  }
}

//Setup the input parameters for a CPSfield to default settings. This is specifically relevant to SIMD data
//NOTE: If you choose to choose a different SIMD layout, *ensure that the time direction is not SIMD-folded otherwise many A2A algorithms will not work*
template<typename FieldType>
void setupFieldParams(typename FieldType::InputParamType &p){
  _setupFieldParams<typename FieldType::FieldSiteType, typename FieldType::FieldMappingPolicy, typename FieldType::FieldMappingPolicy::ParamType>::doit(p);
}

//Get the underlying CPSfield type associated with a class derived from CPSfield
template<typename DerivedType>
struct baseCPSfieldType{
  typedef CPSfield<typename DerivedType::FieldSiteType, DerivedType::FieldSiteSize, typename DerivedType::FieldMappingPolicy, typename DerivedType::FieldAllocPolicy> type;
};

inline std::vector<int> defaultConjDirs(){
  std::vector<int> c(4,0);
  for(int i=0;i<4;i++) c[i] = (GJP.Bc(i)==BND_CND_GPARITY ? 1 : 0);
  return c;
}

//Cshift a field in the direction mu, for a field with complex-conjugate BCs in chosen directions
//Cshift(+mu) : f'(x) = f(x-\hat mu)
//Cshift(-mu) : f'(x) = f(x+\hat mu)
template<typename T, int SiteSize, typename FlavorPol, typename Alloc>
CPSfield<T,SiteSize,FourDpolicy<FlavorPol>,Alloc> CshiftCconjBc(const CPSfield<T,SiteSize,FourDpolicy<FlavorPol>,Alloc> &field, int mu, int pm, const std::vector<int> &conj_dirs= defaultConjDirs()){
  assert(conj_dirs.size() == 4);
  assert(abs(pm) == 1);
  NullObject null_obj;
  CPSfield<T,SiteSize,FourDpolicy<FlavorPol>,Alloc> out(null_obj);

  cyclicPermute(out, field, mu, pm, 1);

  int orthdirs[3];
  size_t orthlen[3];
  size_t orthsize = 1;
  int jj=0;
  for(int i=0;i<4;i++){
    if(i!=mu){
      orthdirs[jj] = i;      
      orthlen[jj] = GJP.NodeSites(i);
      orthsize *= GJP.NodeSites(i);
      ++jj;
    }
  }
  
  if(pm == 1 && conj_dirs[mu] && GJP.NodeCoor(mu) == 0){ //pulled across lower boundary
    CPSautoView(out_v,out,HostReadWrite);
#pragma omp parallel for
    for(size_t o=0;o<orthsize;o++){
      int coord[4]; 
      coord[mu] = 0;
      size_t rem = o;
      for(int i=0;i<3;i++){
	coord[orthdirs[i]] = rem % orthlen[i]; rem /= orthlen[i];
      }
      for(int f=0;f<out_v.nflavors();f++){
	T *p = out_v.site_ptr(coord,f);
	for(int s=0;s<SiteSize;s++)
	  p[s] = cps::cconj(p[s]);
      }
    }
  }else if(pm == -1 && conj_dirs[mu] && GJP.NodeCoor(mu) == GJP.Nodes(mu)-1){ //pulled across upper boundary
    CPSautoView(out_v,out,HostReadWrite);
#pragma omp parallel for
    for(size_t o=0;o<orthsize;o++){
      int coord[4]; 
      coord[mu] = GJP.NodeSites(mu)-1;
      size_t rem = o;
      for(int i=0;i<3;i++){
	coord[orthdirs[i]] = rem % orthlen[i]; rem /= orthlen[i];
      }
      for(int f=0;f<out_v.nflavors();f++){
	T *p = out_v.site_ptr(coord,f);
	for(int s=0;s<SiteSize;s++)
	  p[s] = cps::cconj(p[s]);
      }
    }
  }
  return out;
}

//Call the above but with intermediate conversion for non-basic layout (currently cannot deal directly with SIMD)
template<typename T, int SiteSize, typename DimPol, typename Alloc, 
	 typename std::enable_if<DimPol::EuclideanDimension==4 && 
				 !std::is_same<FourDpolicy<typename DimPol::FieldFlavorPolicy>, DimPol>::value
					       ,int>::type = 0 >
CPSfield<T,SiteSize,DimPol,Alloc> CshiftCconjBc(const CPSfield<T,SiteSize,DimPol,Alloc> &field, int mu, int pm, const std::vector<int> &conj_dirs = defaultConjDirs()){
  NullObject null_obj;
  CPSfield<typename T::scalar_type,SiteSize,FourDpolicy<typename DimPol::FieldFlavorPolicy>,Alloc> tmp(null_obj);
  tmp.importField(field);
  tmp = CshiftCconjBc(tmp,mu,pm,conj_dirs);
  CPSfield<T,SiteSize,DimPol,Alloc> out(field.getDimPolParams());
  out.importField(tmp);
  return out;
}

//Obtain the gauge fixing matrix from the CPS lattice class
inline CPSfield<cps::ComplexD,9,FourDpolicy<DynamicFlavorPolicy> > getGaugeFixingMatrix(Lattice &lat){
  NullObject null_obj;
  CPSfield<cps::ComplexD,9,FourDpolicy<DynamicFlavorPolicy> > gfmat(null_obj);
  {
    CPSautoView(gfmat_v,gfmat,HostWrite);
      
#pragma omp parallel for
    for(int s=0;s<GJP.VolNodeSites();s++){
      for(int f=0;f<GJP.Gparity()+1;f++){
	cps::ComplexD* to = gfmat_v.site_ptr(s,f);
	cps::ComplexD const* from = (cps::ComplexD const*)lat.FixGaugeMatrix(s,f);
	memcpy(to, from, 9*sizeof(cps::ComplexD));
      }
    }
  }
  return gfmat;
}
//Obtain the flavor-0 component of the gauge fixing matrix from the CPS lattice class
inline CPSfield<cps::ComplexD,9,FourDpolicy<OneFlavorPolicy> > getGaugeFixingMatrixFlavor0(Lattice &lat){
  NullObject null_obj;
  CPSfield<cps::ComplexD,9,FourDpolicy<OneFlavorPolicy> > gfmat(null_obj);
  {
    CPSautoView(gfmat_v,gfmat,HostWrite);
      
#pragma omp parallel for
    for(int s=0;s<GJP.VolNodeSites();s++){
      cps::ComplexD* to = gfmat_v.site_ptr(s,0);
      cps::ComplexD const* from = (cps::ComplexD const*)lat.FixGaugeMatrix(s,0);
      memcpy(to, from, 9*sizeof(cps::ComplexD));
    }
  }
  return gfmat;
}
//Apply the gauge fixing matrices to the lattice gauge links, obtaining a gauge fixed configuration
inline void gaugeFixCPSlattice(Lattice &lat){
  typedef CPSfield<cps::ComplexD,9,FourDpolicy<DynamicFlavorPolicy> > GaugeRotLinField;
  GaugeRotLinField gfmat = getGaugeFixingMatrix(lat);

  for(int mu=0;mu<4;mu++){
    GaugeRotLinField gfmat_plus = CshiftCconjBc(gfmat, mu, -1); 

    CPSautoView(gfmat_v, gfmat, HostRead);
    CPSautoView(gfmat_plus_v, gfmat_plus, HostRead);
 #pragma omp parallel for
    for(size_t i=0;i<GJP.VolNodeSites();i++){
      for(int f=0;f<GJP.Gparity()+1;f++){
	Matrix g_plus_dag;  g_plus_dag.Dagger( *((Matrix*)gfmat_plus_v.site_ptr(i,f)) );
	Matrix g = *((Matrix*)gfmat_v.site_ptr(i,f) );
	Matrix *U = lat.GaugeField()+ mu + 4*(i + GJP.VolNodeSites()*f);
	*U = g*(*U)*g_plus_dag;
      }
    }
  }
}

CPS_END_NAMESPACE
#endif
