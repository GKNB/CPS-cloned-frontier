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
		   const CPSfield<TypeB,SiteSize,MapPolB,AllocPolB> &from){
    assert(into.nfsites() == from.nfsites()); //should be true in # Euclidean dimensions the same, but not guaranteed
    
    #pragma omp parallel for
    for(int fs=0;fs<into.nfsites();fs++){
      int x[5], f; into.fsiteUnmap(fs,x,f); //doesn't matter if the linearization differs between the two
      TypeA* toptr = into.fsite_ptr(fs);
      TypeB const* fromptr = from.site_ptr(x,f);
      for(int i=0;i<SiteSize;i++) toptr[i] = fromptr[i];
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
		   const CPSfield<TypeB,SiteSize,MapPolB,AllocPolB> &from){
    const int nsimd = GridSIMDTypeA::Nsimd();
    const int ndim = MapPolA::EuclideanDimension;
    if(from.nfsites()/nsimd != into.nfsites()) ERR.General("CPSfieldCopy","copy(<SIMD field> &into, const <non-SIMD field> &from)","Expected from.nfsites/nsimd = into.nfsites, got %d/%d (=%d) != %d\n",from.nfsites(),nsimd, from.nfsites()/nsimd, into.nfsites());
    
    std::vector<std::vector<int> > packed_offsets(nsimd,std::vector<int>(ndim));
    for(int i=0;i<nsimd;i++){
      into.SIMDunmap(i,&packed_offsets[i][0]);
    }
    
#pragma omp parallel for
    for(int fs=0;fs<into.nfsites();fs++){
      int x[ndim], f; into.fsiteUnmap(fs,x,f);
      GridSIMDTypeA* toptr = into.fsite_ptr(fs);

      //x is the root coordinate corresponding to SIMD packed index 0      
      std::vector<TypeB const*> ptrs(nsimd);
      ptrs[0] = from.site_ptr(x,f);
      
      int xx[ndim];
      for(int i=1;i<nsimd;i++){
	for(int d=0;d<ndim;d++)
	  xx[d] = x[d] + packed_offsets[i][d];  //xx = x + offset
	ptrs[i] = from.site_ptr(xx,f);
      }
      into.SIMDpack(toptr, ptrs, SiteSize);
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
		   const CPSfield<GridSIMDTypeB,SiteSize,MapPolB,AllocPolB> &from){
    const int nsimd = GridSIMDTypeB::Nsimd();
    const int ndim = MapPolA::EuclideanDimension;
    if(into.nfsites()/nsimd != from.nfsites()) ERR.General("CPSfieldCopy","copy(<non-SIMD field> &into, const <SIMD-field> &from)","Expected into.nfsites/nsimd = from.nfsites, got %d/%d (=%d) != %d\n",into.nfsites(),nsimd, into.nfsites()/nsimd, from.nfsites());

    std::vector<std::vector<int> > packed_offsets(nsimd,std::vector<int>(ndim));
    for(int i=0;i<nsimd;i++) from.SIMDunmap(i,&packed_offsets[i][0]);

#pragma omp parallel for
    for(int fs=0;fs<from.nfsites();fs++){
      int x[ndim], f; from.fsiteUnmap(fs,x,f);
      GridSIMDTypeB const* fromptr = from.fsite_ptr(fs);

      //x is the root coordinate corresponding to SIMD packed index 0
      std::vector<TypeA*> ptrs(nsimd);
      ptrs[0] = into.site_ptr(x,f);
      
      int xx[ndim];
      for(int i=1;i<nsimd;i++){
	for(int d=0;d<ndim;d++)
	  xx[d] = x[d] + packed_offsets[i][d];  //xx = x + offset

	ptrs[i] = into.site_ptr(xx,f);
      }
      from.SIMDunpack(ptrs, fromptr, SiteSize);
    }
  }
};
#undef CONDITION

#endif

template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
template< typename extSiteType, typename extMapPol, typename extAllocPol>
void CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::importField(const CPSfield<extSiteType,SiteSize,extMapPol,extAllocPol> &r){
  CPSfieldCopy<SiteSize,
	       SiteType,MappingPolicy,AllocPolicy,
	       extSiteType, extMapPol, extAllocPol>::copy(*this,r);
}
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
template< typename extSiteType, typename extMapPol, typename extAllocPol>
void CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::exportField(CPSfield<extSiteType,SiteSize,extMapPol,extAllocPol> &r) const{
  CPSfieldCopy<SiteSize,
	       extSiteType, extMapPol,  extAllocPol,
	       SiteType,MappingPolicy,AllocPolicy>::copy(r,*this);
}


#endif
