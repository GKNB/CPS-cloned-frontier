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
    if(!UniqueID()){ printf("Failed %s check\n", descr.c_str()); fflush(stdout); } 
    exit(-1);
  }else{
    if(!UniqueID()){ printf("Passed %s check\n", descr.c_str()); fflush(stdout); }
  }
}

//Compare general CPSfield
template<typename FieldType, typename my_enable_if<_equal<typename ComplexClassify<typename FieldType::FieldSiteType>::type, complex_double_or_float_mark>::value,int>::type = 0>
inline void compareField(const FieldType &A, const FieldType &B, const std::string &descr = "Field", const double tol = 1e-9, bool print_all = false){
  typedef typename FieldType::FieldSiteType::value_type value_type;
  
  double fail = 0.;
  for(int xf=0;xf<A.nfsites();xf++){
    int f; int x[FieldType::FieldMappingPolicy::EuclideanDimension];
    A.fsiteUnmap(xf, x,f);

    for(int i=0;i<FieldType::FieldSiteSize;i++){
      value_type const* av = (value_type const*)(A.fsite_ptr(xf)+i);
      value_type const* bv = (value_type const*)(B.fsite_ptr(xf)+i);
      for(int reim=0;reim<2;reim++){
	value_type diff_rat = (av[reim] == 0.0 && bv[reim] == 0.0) ? 0.0 : fabs( 2.*(av[reim]-bv[reim])/(av[reim]+bv[reim]) );
	if(diff_rat > tol || print_all){
	  if(diff_rat > tol) std::cout << "!FAIL: ";
	  else std::cout << "Pass: ";
	  
	  std::cout << "coord=(";
	  for(int xx=0;xx<FieldType::FieldMappingPolicy::EuclideanDimension-1;xx++)
	    std::cout << x[xx] << ", ";
	  std::cout << x[FieldType::FieldMappingPolicy::EuclideanDimension-1];

	  std::cout << ") f=" << f << " i=" << i << " reim=" << reim << " A " << av[reim] << " B " << bv[reim] << " fracdiff " << diff_rat << std::endl;
	  if(diff_rat > tol) fail = 1.;
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

CPS_END_NAMESPACE
#endif
