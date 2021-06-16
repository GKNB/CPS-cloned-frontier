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
#pragma omp parallel for
    for(size_t fs=0;fs<into.nfsites();fs++){
      int x[5], f; into.fsiteUnmap(fs,x,f); //doesn't matter if the linearization differs between the two
      if(from.contains(x,f) && (fromsitemask == NULL || fromsitemask->query(x,f)) ){
	TypeA* toptr = into.fsite_ptr(fs);
	TypeB const* fromptr = from.site_ptr(x,f);
	for(int i=0;i<SiteSize;i++) toptr[i] = fromptr[i];
      }
    }
  }
};
#undef CONDITION

#ifdef USE_GRID

inline std::string vtostring(const int* v, const int ndim){
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
    typedef typename GridSIMDTypeA::scalar_type GridTypeScalar;

    std::vector<std::vector<int> > packed_offsets(nsimd,std::vector<int>(ndim));
    for(int i=0;i<nsimd;i++) into.SIMDunmap(i,packed_offsets[i].data());
    
#pragma omp parallel for
    for(size_t fs=0;fs<into.nfsites();fs++){
      int x[ndim], f; into.fsiteUnmap(fs,x,f); //this is the root coordinate for lane 0
      GridSIMDTypeA * toptr = into.fsite_ptr(fs);

      int xx[ndim]; //full coordinate
      for(int lane=0;lane<nsimd;lane++){
	for(int d=0;d<ndim;d++) xx[d] = x[d] + packed_offsets[lane][d];  //xx = x + offset

	if(from.contains(xx,f) && (fromsitemask == NULL || fromsitemask->query(xx,f)) ){
	  TypeB const* fromptr = from.site_ptr(xx,f);
	
	  for(int s=0;s<SiteSize;s++)
	    *(  (GridTypeScalar*)(toptr+s) + lane ) = fromptr[s];
	}
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
    const int ndim = MapPolA::EuclideanDimension;
    typedef typename GridSIMDTypeB::scalar_type GridTypeScalar;
    
#pragma omp parallel for
    for(size_t fs=0;fs<into.nfsites();fs++){
      int x[ndim], f; into.fsiteUnmap(fs,x,f);
      TypeA* toptr = into.fsite_ptr(fs);
      
      if(from.contains(x,f) && (fromsitemask == NULL || fromsitemask->query(x,f)) ){
	int lane = from.SIMDmap(x);
	GridSIMDTypeB const* fromptr = from.site_ptr(x,f);
	for(int s=0;s<SiteSize;s++)
	  toptr[s] = *( (GridTypeScalar const*)(fromptr + s) + lane );      
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


//CPSfield <-> Grid field
#ifdef USE_GRID

template<typename T,typename CPScomplex>
struct GridTensorConvert{};

template<typename complex_scalar, typename CPScomplex>
struct GridTensorConvert<Grid::iSpinColourVector<complex_scalar>, CPScomplex>{
  static_assert(!Grid::isSIMDvectorized<complex_scalar>::value && Grid::isComplex<complex_scalar>::value, "Only applies to scalar complex types");

  //12-component complex spin-color vector
  //We have assured the input is not SIMD vectorized so the output type is the same
  inline static void doit(CPScomplex* cps, const Grid::iSpinColourVector<complex_scalar> &grid, const int f){
    for(int s=0;s<Grid::Ns;s++)
      for(int c=0;c<Grid::Nc;c++)
	*cps++ = grid()(s)(c);
  }
  inline static void doit(Grid::iSpinColourVector<complex_scalar> &grid, CPScomplex const* cps, const int f){
    for(int s=0;s<Grid::Ns;s++)
      for(int c=0;c<Grid::Nc;c++)
	grid()(s)(c) = *cps++;
  }
};
template<typename complex_scalar, typename CPScomplex>
struct GridTensorConvert<Grid::iGparitySpinColourVector<complex_scalar>, CPScomplex>{
  static_assert(!Grid::isSIMDvectorized<complex_scalar>::value && Grid::isComplex<complex_scalar>::value, "Only applies to scalar complex types");

  //12-component complex spin-color vector
  //We have assured the input is not SIMD vectorized so the output type is the same
  inline static void doit(CPScomplex* cps, const Grid::iGparitySpinColourVector<complex_scalar> &grid, const int f){
    for(int s=0;s<Grid::Ns;s++)
      for(int c=0;c<Grid::Nc;c++)
	*cps++ = grid(f)(s)(c);
  }
  inline static void doit(Grid::iGparitySpinColourVector<complex_scalar> &grid, CPScomplex const* cps, const int f){
    for(int s=0;s<Grid::Ns;s++)
      for(int c=0;c<Grid::Nc;c++)
  	grid(f)(s)(c) = *cps++;
  }
};
template<typename complex_scalar, typename CPScomplex>
struct GridTensorConvert<Grid::iLorentzColourMatrix<complex_scalar>, CPScomplex>{
  static_assert(!Grid::isSIMDvectorized<complex_scalar>::value && Grid::isComplex<complex_scalar>::value, "Only applies to scalar complex types");

  //Gauge field  mu=0..3  3*3 complex
  //We have assured the input is not SIMD vectorized so the output type is the same
  inline static void doit(CPScomplex* cps, const Grid::iLorentzColourMatrix<complex_scalar> &grid, const int f){
    for(int mu=0;mu<4;mu++)
      for(int i=0;i<3;i++)
	for(int j=0;j<3;j++)
	  *cps++ = grid(mu)()(i,j);
  }
  inline static void doit(Grid::iLorentzColourMatrix<complex_scalar> &grid, CPScomplex const* cps, const int f){
    for(int mu=0;mu<4;mu++)
      for(int i=0;i<3;i++)
	for(int j=0;j<3;j++)
	  grid(mu)()(i,j) = *cps++;
  }
};


template<int Ndim>
struct dimensionMap{};

template<>
struct dimensionMap<5>{
  const int cps_to_grid[5] = {1,2,3,4,0};
  const int grid_to_cps[5] = {4,0,1,2,3};
};
template<>
struct dimensionMap<4>{
  const int cps_to_grid[4] = {0,1,2,3};
  const int grid_to_cps[4] = {0,1,2,3};
};

//Local coordinate *in Grid's dimension ordering* (cf above)
inline void getLocalLatticeCoord(std::vector<int> &lcoor, const size_t oidx, const size_t iidx, Grid::GridBase const* grid, const int checkerboard){
  Grid::GridBase* gridc = const_cast<Grid::GridBase*>(grid); //the lookup functions are not const for some reason
  const int Nd = grid->Nd();
  Grid::Coordinate ocoor(Nd);
  gridc->oCoorFromOindex(ocoor, oidx);

  Grid::Coordinate icoor(Nd);
  gridc->iCoorFromIindex(icoor, iidx);

  int checker_dim = -1;
  //Get the local coordinate on Grid's local lattice 
  for(int mu=0;mu<Nd;mu++){
    lcoor[mu] = ocoor[mu] + gridc->_rdimensions[mu]*icoor[mu];

    //For checkerboarded Grid fields the above is defined on a reduced lattice of half the size
    if(gridc->CheckerBoarded(mu)){
      lcoor[mu] = lcoor[mu]*2;
      checker_dim = mu;
    }
  }

  if(checker_dim != -1 && gridc->CheckerBoard(lcoor) != checkerboard) lcoor[checker_dim] += 1;
}
//Local coordinate in canonical x,y,z,t,s ordering
template<int Nd>
inline void getLocalCanonicalLatticeCoord(std::vector<int> &lcoor, const size_t oidx, const size_t iidx, Grid::GridBase const* grid, const int checkerboard){
  assert(grid->Nd() == Nd);
  static dimensionMap<Nd> dim_map;
  std::vector<int> lcoor_grid(Nd);
  getLocalLatticeCoord(lcoor_grid, oidx,iidx, grid, checkerboard);
  for(int mu=0;mu<Nd;mu++) lcoor[ dim_map.grid_to_cps[mu] ] = lcoor_grid[mu];
}



template<typename Type, int SiteSize, typename MapPol, typename AllocPol,
	 typename GridField, typename ComplexClass>
class CPSfieldGridImpex{};

template<typename Type, int SiteSize, typename MapPol, typename AllocPol,
	 typename GridField>
class CPSfieldGridImpex<Type,SiteSize,MapPol,AllocPol,GridField,complex_double_or_float_mark>{
  typedef CPSfield<Type,SiteSize,MapPol,AllocPol> CPSfieldType;

public:
  typedef typename Grid::GridTypeMapper<typename GridField::vector_object>::scalar_object sobj;
  
  static void import(CPSfieldType &into, const GridField &from, IncludeSite<MapPol::EuclideanDimension> const* fromsitemask){
    const int Nd = MapPol::EuclideanDimension;
    assert(Nd == from.Grid()->Nd());
    dimensionMap<CPSfieldType::EuclideanDimension> dim_map;
    auto from_view = from.View(Grid::CpuRead);

#pragma omp parallel for
    for(size_t site=0;site<into.nsites();site++){
      std::vector<int> x(Nd);
      into.siteUnmap(site, &x[0]);

      Grid::Coordinate grid_x(Nd);
      for(int i=0;i<Nd;i++)
	grid_x[ dim_map.cps_to_grid[i] ] = x[i];

      if(from.Grid()->CheckerBoard(grid_x) != from.Checkerboard()){ //skip sites not on Grid checkerboard
	continue;
      }	
      
      sobj siteGrid; //contains both flavors if Gparity
      peekLocalSite(siteGrid,from_view,grid_x);

      for(int f=0;f<into.nflavors();f++){
	if(fromsitemask == NULL || fromsitemask->query(&x[0],f)){
	  typename CPSfieldType::FieldSiteType *cps = into.site_ptr(site,f);
	  GridTensorConvert<sobj, typename CPSfieldType::FieldSiteType>::doit(cps, siteGrid, f);
	}
      }      
    }

    from_view.ViewClose();
  }

  
  
  static void exportit(GridField &into, const CPSfieldType &from, IncludeSite<MapPol::EuclideanDimension> const* fromsitemask){
    const int Nd = MapPol::EuclideanDimension;
    assert(Nd == into.Grid()->Nd());
    dimensionMap<CPSfieldType::EuclideanDimension> dim_map;
    const int Nsimd = GridField::vector_type::Nsimd();
    
    auto oview = into.View(Grid::CpuWrite);

#pragma omp parallel for
    for(size_t out_oidx=0;out_oidx<into.Grid()->oSites();out_oidx++){
      std::vector<int> lcoor_cps(Nd);

      for(int lane=0;lane<Nsimd;lane++){
	getLocalCanonicalLatticeCoord<Nd>(lcoor_cps, out_oidx, lane, into.Grid(), into.Checkerboard());

	sobj tmp; peekLane(tmp,oview[out_oidx],lane);
	
	for(int f=0;f<from.nflavors();f++){
	  if(fromsitemask == NULL || fromsitemask->query(lcoor_cps.data(),f)){	    	   
	    typename CPSfieldType::FieldSiteType const* cps = from.site_ptr(lcoor_cps.data(),f);
	    GridTensorConvert<sobj, typename CPSfieldType::FieldSiteType>::doit(tmp, cps, f);
	  }
	}

	pokeLane(oview[out_oidx], tmp, lane);
      }
    }

    oview.ViewClose();
  }
  
};

template<typename Type, int SiteSize, typename MapPol, typename AllocPol,
	 typename GridField>
class CPSfieldGridImpex<Type,SiteSize,MapPol,AllocPol,GridField,grid_vector_complex_mark>{
  typedef CPSfield<Type,SiteSize,MapPol,AllocPol> CPSfieldType;

public:

  static void import(CPSfieldType &into, const GridField &from, IncludeSite<MapPol::EuclideanDimension> const* fromsitemask){
    const int Nd = MapPol::EuclideanDimension;
    assert(Nd == from.Grid()->Nd());
    typedef typename Grid::GridTypeMapper<Type>::scalar_type CPSscalarType;
    typedef typename ComplexClassify<CPSscalarType>::type CPSscalarTypeClass;
    
    //Create temp CPS unvectorized field
    typedef typename StandardDimensionPolicy<MapPol::EuclideanDimension, typename MapPol::FieldFlavorPolicy>::type CPSscalarMapPol;
    NullObject n;
    CPSfield<CPSscalarType,SiteSize,CPSscalarMapPol,StandardAllocPolicy> cps_unpacked(n);

    CPSfieldGridImpex<CPSscalarType,SiteSize,CPSscalarMapPol,StandardAllocPolicy,GridField, CPSscalarTypeClass>::import(cps_unpacked,from,fromsitemask);
    into.importField(cps_unpacked);
  }
  
  static void exportit(GridField &into, const CPSfieldType &from, IncludeSite<MapPol::EuclideanDimension> const* fromsitemask){
    const int Nd = MapPol::EuclideanDimension;
    assert(Nd == into.Grid()->Nd());
    typedef typename Grid::GridTypeMapper<Type>::scalar_type CPSscalarType;
    typedef typename ComplexClassify<CPSscalarType>::type CPSscalarTypeClass;

    //Create temp CPS unvectorized field
    typedef typename StandardDimensionPolicy<MapPol::EuclideanDimension, typename MapPol::FieldFlavorPolicy>::type CPSscalarMapPol;
    NullObject n;
    CPSfield<CPSscalarType,SiteSize,CPSscalarMapPol,StandardAllocPolicy> cps_unpacked(n);
    cps_unpacked.importField(from);
    CPSfieldGridImpex<CPSscalarType,SiteSize,CPSscalarMapPol,StandardAllocPolicy,GridField, CPSscalarTypeClass>::exportit(into, cps_unpacked,fromsitemask);
  }
};

template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
template<typename GridField>
void  CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::importGridField(const GridField &grid, IncludeSite<MappingPolicy::EuclideanDimension> const* fromsitemask){
  typedef typename ComplexClassify<SiteType>::type ComplexClass;
  CPSfieldGridImpex<SiteType,SiteSize,MappingPolicy,AllocPolicy,GridField,ComplexClass>::import(*this, grid,fromsitemask);
}
template< typename SiteType, int SiteSize, typename MappingPolicy, typename AllocPolicy>
template<typename GridField>
void  CPSfield<SiteType,SiteSize,MappingPolicy,AllocPolicy>::exportGridField(GridField &grid, IncludeSite<MappingPolicy::EuclideanDimension> const* fromsitemask) const{
  typedef typename ComplexClassify<SiteType>::type ComplexClass;
  CPSfieldGridImpex<SiteType,SiteSize,MappingPolicy,AllocPolicy,GridField,ComplexClass>::exportit(grid,*this,fromsitemask);
}
#endif


#endif
