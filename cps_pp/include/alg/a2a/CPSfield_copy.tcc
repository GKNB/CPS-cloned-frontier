#ifndef _CPSFIELD_COPY_H
#define _CPSFIELD_COPY_H

template<int SiteSize,
	 typename TypeA, typename MapPolA, typename AllocPolA,
	 typename TypeB, typename MapPolB, typename AllocPolB,
	 typename Enable = void>
class CPSfieldCopy;

//Generic copy. SiteSize and number of Euclidean dimensions must be the same
#ifdef USE_GRID
#define CONDITION sameDim<MapPolA,MapPolB>::val && !Grid::is_simd<TypeA>::value && !Grid::is_simd<TypeB>::value
#else
#define CONDITION sameDim<MapPolA,MapPolB>::val
#endif

template<int SiteSize,
	 typename TypeA, typename MapPolA, typename AllocPolA,
	 typename TypeB, typename MapPolB, typename AllocPolB>
class CPSfieldCopy<SiteSize,TypeA,MapPolA,AllocPolA, TypeB,MapPolB,AllocPolB, typename my_enable_if<CONDITION,void>::type>{
public: 
  static void copy(CPSfield<TypeA,SiteSize,MapPolA,AllocPolA> &into,
		   const CPSfield<TypeB,SiteSize,MapPolB,AllocPolB> &from, IncludeSite<MapPolB::EuclideanDimension> const* fromsitemask){
    if(fromsitemask == NULL) assert(into.nfsites() == from.nfsites()); //should be true in # Euclidean dimensions the same, but not guaranteed.
    //If a mask is provided its up to the user to ensure all masked sites are valid for the destination field
    
#pragma omp parallel for
    for(int fs=0;fs<from.nfsites();fs++){
      int x[5], f; from.fsiteUnmap(fs,x,f); //doesn't matter if the linearization differs between the two
      if(fromsitemask == NULL || fromsitemask->query(x,f)){
	TypeA* toptr = into.site_ptr(x,f);
	TypeB const* fromptr = from.fsite_ptr(fs);
	for(int i=0;i<SiteSize;i++) toptr[i] = fromptr[i];
      }
    }
  }
};
#undef CONDITION

#ifdef USE_GRID

std::string vtostring(const int* v, const int ndim){
  std::ostringstream os;
  os << '(';
  for(int i=0;i<ndim-1;i++) os << v[i] << ", ";
  os << v[ndim-1] << ')';
  return os.str();
}

//TypeA is Grid_simd type
#define CONDITION sameDim<MapPolA,MapPolB>::val && Grid::is_simd<GridSIMDTypeA>::value && !Grid::is_simd<TypeB>::value

template<int SiteSize,
	 typename GridSIMDTypeA, typename MapPolA, typename AllocPolA,
	 typename TypeB, typename MapPolB, typename AllocPolB>
class CPSfieldCopy<SiteSize,
		   GridSIMDTypeA, MapPolA, AllocPolA,
		   TypeB, MapPolB, AllocPolB, typename my_enable_if<CONDITION,void>::type>
{
public:
  static void copy(CPSfield<GridSIMDTypeA,SiteSize,MapPolA,AllocPolA> &into,
		   const CPSfield<TypeB,SiteSize,MapPolB,AllocPolB> &from, IncludeSite<MapPolB::EuclideanDimension> const* fromsitemask){
    const int nsimd = GridSIMDTypeA::Nsimd();
    const int ndim = MapPolA::EuclideanDimension;
    if(fromsitemask == NULL) if(from.nfsites()/nsimd != into.nfsites()) ERR.General("CPSfieldCopy","copy(<SIMD field> &into, const <non-SIMD field> &from)","Expected from.nfsites/nsimd = into.nfsites, got %d/%d (=%d) != %d\n",from.nfsites(),nsimd, from.nfsites()/nsimd, into.nfsites());
    //If a mask is provided its up to the user to ensure all masked sites are valid for the destination fiedl
    
    typedef typename GridSIMDTypeA::scalar_type GridTypeScalar;
    
#pragma omp parallel for
    for(int fs=0;fs<from.nfsites();fs++){
      int x[ndim], f; from.fsiteUnmap(fs,x,f);
      if(fromsitemask == NULL || fromsitemask->query(x,f)){
	int lane = into.SIMDmap(x);
	GridSIMDTypeA * toptr = into.site_ptr(x,f);
	TypeB const* fromptr = from.fsite_ptr(fs);
	
	for(int s=0;s<SiteSize;s++)
	  *(  (GridTypeScalar*)(toptr+s) + lane ) = fromptr[s];
      }
    }
  }
};
#undef CONDITION

//TypeB is Grid_simd type
#define CONDITION sameDim<MapPolA,MapPolB>::val && !Grid::is_simd<TypeA>::value && Grid::is_simd<GridSIMDTypeB>::value

template<int SiteSize,
	 typename TypeA, typename MapPolA, typename AllocPolA,
	 typename GridSIMDTypeB, typename MapPolB, typename AllocPolB>
class CPSfieldCopy<SiteSize,
		   TypeA, MapPolA, AllocPolA,
		   GridSIMDTypeB, MapPolB, AllocPolB, typename my_enable_if<CONDITION,void>::type>
{
public:
  static void copy(CPSfield<TypeA,SiteSize,MapPolA,AllocPolA> &into,
		   const CPSfield<GridSIMDTypeB,SiteSize,MapPolB,AllocPolB> &from, IncludeSite<MapPolB::EuclideanDimension> const* fromsitemask){
    const int nsimd = GridSIMDTypeB::Nsimd();
    const int ndim = MapPolA::EuclideanDimension;
    if(fromsitemask == NULL) if(into.nfsites()/nsimd != from.nfsites()) ERR.General("CPSfieldCopy","copy(<non-SIMD field> &into, const <SIMD-field> &from)","Expected into.nfsites/nsimd = from.nfsites, got %d/%d (=%d) != %d\n",into.nfsites(),nsimd, into.nfsites()/nsimd, from.nfsites());
    //If a mask is provided its up to the user to ensure all masked sites are valid for the destination field
    
    std::vector<std::vector<int> > packed_offsets(nsimd,std::vector<int>(ndim));
    for(int i=0;i<nsimd;i++) from.SIMDunmap(i,&packed_offsets[i][0]);

    typedef typename GridSIMDTypeB::scalar_type GridTypeScalar;
    
#pragma omp parallel for
    for(int fs=0;fs<from.nfsites();fs++){
      int x[ndim], f; from.fsiteUnmap(fs,x,f);
      GridSIMDTypeB const* fromptr = from.fsite_ptr(fs);
            
      //x is the root coordinate corresponding to SIMD packed index 0      
      int xx[ndim];
      for(int lane=0;lane<nsimd;lane++){
	for(int d=0;d<ndim;d++) xx[d] = x[d] + packed_offsets[lane][d];  //xx = x + offset
	
	if(fromsitemask == NULL || fromsitemask->query(xx,f)){
	  TypeA* toptr = into.site_ptr(xx,f);
	  for(int s=0;s<SiteSize;s++)
	    toptr[s] = *( (GridTypeScalar const*)(fromptr + s) + lane );
	} 
      }
    }
    
  }
};
#undef CONDITION

#endif

template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
template< typename extSiteType, typename extMapPol, typename extAllocPol>
void CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::importField(const CPSfield<extSiteType,SiteSize,extMapPol,extAllocPol> &r, IncludeSite<extMapPol::EuclideanDimension> const* fromsitemask){
  CPSfieldCopy<SiteSize,
	       SiteType,MappingPolicy,AllocPolicy,
	       extSiteType, extMapPol, extAllocPol>::copy(*this,r,fromsitemask);
}
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
template< typename extSiteType, typename extMapPol, typename extAllocPol>
void CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::exportField(CPSfield<extSiteType,SiteSize,extMapPol,extAllocPol> &r, IncludeSite<MappingPolicy::EuclideanDimension> const* fromsitemask) const{
  CPSfieldCopy<SiteSize,
	       extSiteType, extMapPol,  extAllocPol,
	       SiteType,MappingPolicy,AllocPolicy>::copy(r,*this,fromsitemask);
}

#endif
