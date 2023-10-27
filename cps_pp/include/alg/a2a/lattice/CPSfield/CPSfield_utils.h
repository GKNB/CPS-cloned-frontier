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

inline std::ostream & operator<<(std::ostream &os , const std::vector<int> &v){
  os << "("; 
  for(int const e : v) os << e << " ";
  os << ")";
  return os;
}

class cpsFieldPartIObase{
protected:
  static inline void lexRankToNodeCoor(int rcoor[4], int rank, const std::vector<int> &mpi){
    for(int i=0;i<4;i++){
      rcoor[i] = rank % mpi[i]; rank /= mpi[i];   //r = rx + Nx*( ry + Ny * (rz + Nz * rt))
    }
  }
  static inline int nodeCoorToLexRank(const int rcoor[4], const std::vector<int> &mpi){
    return rcoor[0] + mpi[0]*( rcoor[1] + mpi[1] * ( rcoor[2] + mpi[2] * rcoor[3] ) );
  }

  static inline void offsetToLocalSiteCoor(int lcoor[4], int off, const std::vector<int> &nodesites){
    for(int i=0;i<4;i++){
      lcoor[i] = off % nodesites[i]; off /= nodesites[i];
    }
  }
  static inline int localSiteCoorToOffset(const int lcoor[4], const std::vector<int> &nodesites){
    return lcoor[0] + nodesites[0]*( lcoor[1] + nodesites[1] * ( lcoor[2] + nodesites[2] * lcoor[3] ) );
  }

  //Rank mappings,  4 integers offset by 4*rank
  static std::vector<int> determineRankMapping(int rank, int nrank){
    std::vector<int> nodecoors(4*nrank, 0);
    int* base = nodecoors.data() + 4*rank;
    for(int i=0;i<4;i++) base[i] = GJP.NodeCoor(i);

    // printf("Rank %d node coord %d %d %d %d\n", rank, base[0],base[1],base[2],base[3]);
    assert( MPI_Allreduce(MPI_IN_PLACE, nodecoors.data(), 4*nrank, MPI_INT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS );
    return nodecoors;
  }
};

template<typename CPSfieldType, 
	 typename std::enable_if<std::is_same<typename CPSfieldType::FieldMappingPolicy,  FourDpolicy<typename CPSfieldType::FieldMappingPolicy::FieldFlavorPolicy> >::value, int>::type = 0>
class cpsFieldPartIOwriter: public cpsFieldPartIObase{
  cpsBinaryWriter wr_data;
  size_t bytes_written;

  double t_total;
  double t_write;

  enum { SiteSize = CPSfieldType::FieldSiteSize };
  typedef typename CPSfieldType::FieldSiteType SiteType;

public:
  void open(const std::string &file_stub){
    t_total = -dclock();
    t_write = 0;

    //Write metadata
    int rank;
    assert( MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS );
    int nrank = 1;
    for(int i=0;i<4;i++)
      nrank *= GJP.Nodes(i);
    
    std::vector<int> nodecoors = determineRankMapping(rank, nrank);
    if(!UniqueID()){
      std::vector<int> mpi_orig(4);
      for(int i=0;i<4;i++) mpi_orig[i] = GJP.Nodes(i);
      
      cpsBinaryWriter wr(file_stub+".meta");
      wr.write(mpi_orig.data(), 4*sizeof(int), true);
      wr.write(nodecoors.data(), nodecoors.size()*sizeof(int), true);
    }

    bytes_written = 0;

    //Open data filed
    std::stringstream fn; fn << file_stub << '.' << GJP.NodeCoor(0) << '.' << GJP.NodeCoor(1) << '.' << GJP.NodeCoor(2) << '.' << GJP.NodeCoor(3) << ".dat"; 
    wr_data.open(fn.str());
    t_total += dclock();
  }
  cpsFieldPartIOwriter(): bytes_written(0){}
  cpsFieldPartIOwriter(const std::string &file_stub){ open(file_stub); }
  
  void write(const CPSfieldType &field){
    t_write -= dclock(); t_total -= dclock();

    size_t field_bytes = field.nfsites() * SiteSize * sizeof(SiteType);
    CPSautoView(from_v,field,HostRead);
    wr_data.write(from_v.ptr(),field_bytes);
   
    bytes_written += field_bytes;
    
    t_write += dclock(); t_total += dclock();
  }
  
  void close(){
    if(wr_data.isOpen()){
      t_total -= dclock();;
      wr_data.close();
      LOGA2A << "cpsFieldPartIOwriter: Write bandwidth " << byte_to_MB(bytes_written) / t_write << " MB/s" << std::endl;
      
      t_total += dclock();
      LOGA2A << "cpsFieldPartIOwriter timings - write:" << t_write << "s  total: " << t_total << "s" << std::endl;
    }
  }
  ~cpsFieldPartIOwriter(){ close(); }
};

template<typename CPSfieldType, 
	 typename std::enable_if<std::is_same<typename CPSfieldType::FieldMappingPolicy,  FourDpolicy<typename CPSfieldType::FieldMappingPolicy::FieldFlavorPolicy> >::value, int>::type = 0>
class cpsFieldPartIOreader: public cpsFieldPartIObase{
  std::vector<cpsBinaryReader> rd_data;
  int rank;
  int nrank_new;
  int nf;
  size_t new_foff;
  size_t orig_foff;
  
  size_t bytes_read;

  struct CommInfo{
    int rank_from;
    int rank_to;
    size_t src_off;
    size_t dest_off;
    size_t size;
    int src_blockidx; //which of the blocks of data on the source rank should it send? (potentially > 1)
    int tag; //unique tag for each communication event
  };

  std::vector<CommInfo> sends; //MPI sends from this rank
  std::vector<CommInfo> recvs; //MPI recvs to this rank
  std::vector<CommInfo> copies; //internal copies
  
  size_t orig_nodebytes;
  size_t site_bytes;

  double t_total;
  double t_read;
  double t_comms;
  double t_setup;

  bool verbose;

  enum { SiteSize = CPSfieldType::FieldSiteSize };
  typedef typename CPSfieldType::FieldSiteType SiteType;
  typedef typename CPSfieldType::FieldMappingPolicy::FieldFlavorPolicy FlavorPolicy;

public:
  void open(const std::string &file_stub){
    t_setup -= dclock(); t_total -= dclock(); t_read = t_comms = 0;

    assert( MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS );

    std::vector<int> mpi_orig(4);
    int nrank_orig = 1;
    std::vector<int> orig_rank_nodecoors;

    {
      cpsBinaryReader rd(file_stub+".meta");
      rd.read(mpi_orig.data(),4*sizeof(int), true);
    
      for(int i=0;i<4;i++) nrank_orig *= mpi_orig[i];

      orig_rank_nodecoors.resize(4*nrank_orig);
      rd.read(orig_rank_nodecoors.data(),4*nrank_orig*sizeof(int), true);
    }

    //lexrank is the lexicographic rank built from the node coordinate in the usual fashion
    std::vector<int> lexrank_to_origrank_map(nrank_orig);
    for(int r=0;r<nrank_orig;r++){
      int const* n = orig_rank_nodecoors.data() + 4*r;
      int lexrank = nodeCoorToLexRank(n, mpi_orig);
      lexrank_to_origrank_map[lexrank] = r;
    }

    nrank_new = 1;
    std::vector<int> mpi_new(4);
    size_t orig_nodevol = 1, new_nodevol = 1;

    std::vector<int> nodesites_orig(4), nodesites_new(4);
    std::vector<int> sites(4);

    for(int i=0;i<4;i++){
      nrank_new *= GJP.Nodes(i);

      mpi_new[i] = GJP.Nodes(i);

      int isites = GJP.NodeSites(i) * GJP.Nodes(i);
      nodesites_new[i] = GJP.NodeSites(i);
      nodesites_orig[i] = isites/mpi_orig[i];
    
      orig_nodevol *= nodesites_orig[i];
      new_nodevol *= nodesites_new[i];
    }

    FlavorPolicy flavpol;
    nf = flavpol.nflavors();
    new_foff = new_nodevol;
    orig_foff  = orig_nodevol;

    if(verbose){
      std::cout << "Ranks orig: " << nrank_orig << ", new " << nrank_new << std::endl;
      std::cout << "MPI geometry orig: " << mpi_orig << ",  new: " << mpi_new << std::endl;
    }    

    //Get the mapping of MPI rank to node coordinate in this job
    std::vector<int> new_rank_nodecoors = determineRankMapping(rank, nrank_new);
    
    if(verbose){
      std::cout << "New rank mapping:" << std::endl;
      for(int r=0;r<nrank_new;r++){
	int* base = new_rank_nodecoors.data() + 4*r;
	std::cout << r << ":" << base[0] << " " << base[1] << " " << base[2] << " " << base[3] << std::endl;
      }
      std::cout << "Orig rank mapping:" << std::endl;
      for(int r=0;r<nrank_orig;r++){
	int* base = orig_rank_nodecoors.data() + 4*r;
	std::cout << r << ":" << base[0] << " " << base[1] << " " << base[2] << " " << base[3] << std::endl;
      }
    }
  
    //Generate the mapping information for the MPI IO cross-communication
    std::vector<std::vector<CommInfo> > all_comms(nrank_new);

#pragma omp parallel for
    for(int rank_new = 0 ; rank_new < nrank_new; rank_new++){ //dest rank
      for(size_t s = 0 ; s< new_nodevol; s++){ //dest site
	int xloc[4];
	offsetToLocalSiteCoor(xloc, s, nodesites_new);

	int const* nodecoor_new = new_rank_nodecoors.data() + 4*rank_new;

	int nodecoor_orig[4];     
	int xloc_orig[4];
	for(int i=0;i<4;i++){
	  int xi_full = xloc[i] + nodesites_new[i] * nodecoor_new[i];
	  nodecoor_orig[i] = xi_full / nodesites_orig[i];
	  xloc_orig[i] = xi_full - nodecoor_orig[i] * nodesites_orig[i];
	}
   
	int orig_lexrank = nodeCoorToLexRank(nodecoor_orig, mpi_orig);
	int orig_rank = lexrank_to_origrank_map[orig_lexrank];

	int orig_off = localSiteCoorToOffset(xloc_orig, nodesites_orig);

	int orig_rank_data_owner = orig_rank % nrank_new; //which MPI rank in this job currently owns this data block
	int orig_rank_data_block = orig_rank / nrank_new;
      
	//Merge consecutive sends
	CommInfo *bck = s > 0 ? &all_comms[rank_new].back() : nullptr;

	if(bck && 
	   bck->rank_from == orig_rank_data_owner && bck->src_blockidx == orig_rank_data_block &&
	   bck->src_off + bck->size == orig_off && bck->dest_off + bck->size == s){
	  bck->size += 1;
	}else{
	  CommInfo snd;
	  snd.rank_from = orig_rank_data_owner;
	  snd.rank_to = rank_new;
	  snd.src_off = orig_off;
	  snd.dest_off = s;
	  snd.size = 1;
	  snd.src_blockidx = orig_rank_data_block;
	  snd.tag = rank_new + nrank_new * all_comms[rank_new].size(); //every node will agree with this tag
	  all_comms[rank_new].push_back(snd);
	}
      }//s
    }//rank_new

    //Break out those comms that this node takes part in
    for(int r=0;r< nrank_new; r++){
      for(int s=0;s<all_comms[r].size();s++){
    	auto const &ss = all_comms[r][s];
	if(ss.rank_from == rank && ss.rank_to == rank) copies.push_back(ss);
	else if(ss.rank_from == rank) sends.push_back(ss);
	else if(ss.rank_to == rank) recvs.push_back(ss);
      }
    }

    rd_data.clear();
    //Load the original data blocks with a particular order defined as follows:
    //We choose a lexicographic mapping of original node coordinates to original "ranks" (doesn't matter if this is the same as the original, actual MPI rank mapping)
    //The MPI ranks of this job load the data from the original ranks round-robin (so some ranks may have more data than others)
    //The data blocks are stored originally with a filename corresponding to their original node offset, so the original mapping of this to MPI rank is unimportant
    for(int rank_orig=0;rank_orig<nrank_orig;rank_orig++){ 
      if(rank_orig % nrank_new == rank){ //this rank loads data from original "ranks" according to a known mapping
	int const* node_coor_orig = orig_rank_nodecoors.data() + 4*rank_orig;
	
	//Open file  file_stub + .nx.ny.nz.nt -> d
	std::stringstream fn; fn << file_stub << '.' << node_coor_orig[0] << '.' << node_coor_orig[1] << '.' << node_coor_orig[2] << '.' << node_coor_orig[3] << ".dat"; 
	if(verbose) printf("Rank %d reading file %s\n", rank, fn.str().c_str());
	rd_data.push_back(cpsBinaryReader(fn.str()));
      }
    }
    site_bytes = SiteSize*sizeof(SiteType);
    orig_nodebytes = nf * orig_nodevol * site_bytes;
    bytes_read=0;
    
    t_setup += dclock(); t_total += dclock();
  }

  cpsFieldPartIOreader(): bytes_read(0), verbose(false){}
  cpsFieldPartIOreader(const std::string &file_stub): verbose(false){ open(file_stub); }

  inline void setVerbose(bool to){ verbose = to; }

  void read(CPSfieldType &into){
    t_read -= dclock(); t_total -= dclock();
    double t_rd_data = -dclock();

    std::vector<char*> node_data(rd_data.size());
    for(int d=0;d<rd_data.size();d++){
      node_data[d]=(char*)memalign_check(32,orig_nodebytes);
      rd_data[d].read(node_data[d],orig_nodebytes,true);
    }

    t_rd_data += dclock();
    t_read += dclock();
    bytes_read += node_data.size()*orig_nodebytes;
    double rate = byte_to_MB(node_data.size()*orig_nodebytes) / t_rd_data;
    LOGA2A << "cpsFieldPartIOreader: Field part read bandwidth " << rate << " MB/s" << std::endl;

    t_comms -= dclock();
    CPSautoView(into_v, into, HostWrite);
    char* into_p = (char*)into_v.ptr();

    int nsend=nf*sends.size(), nrecv=nf*recvs.size(), ncp=nf*copies.size();    

    std::vector<MPI_Request> comms(nsend+nrecv);

    //Start sends
    MPI_Request *sbase = comms.data();
#pragma omp parallel for
    for(int s=0; s < sends.size(); s++){
      auto const &ss = sends[s];
      for(int f=0;f<nf;f++)
	assert( MPI_Isend(node_data[ss.src_blockidx] + (ss.src_off + orig_foff*f)*site_bytes , ss.size*site_bytes, MPI_CHAR, ss.rank_to, f+nf*ss.tag, MPI_COMM_WORLD, sbase + f + nf*s) == MPI_SUCCESS );
    }
    //Start recvs
    MPI_Request *rbase = comms.data() + nsend;
#pragma omp parallel for
    for(int s=0; s < recvs.size(); s++){
      auto const &ss = recvs[s];
      for(int f=0;f<nf;f++)
	assert( MPI_Irecv(into_p + (ss.dest_off + new_foff*f)*site_bytes, ss.size*site_bytes, MPI_CHAR, ss.rank_from, f+nf*ss.tag, MPI_COMM_WORLD, rbase + f + nf*s) == MPI_SUCCESS );
    }
    //Do copies while waiting for MPI
#pragma omp parallel for
    for(int s=0; s < copies.size(); s++){
      auto const &ss = copies[s];
      for(int f=0;f<nf;f++)
	memcpy(into_p + (ss.dest_off + new_foff*f)*site_bytes, node_data[ss.src_blockidx] + (ss.src_off + orig_foff*f)*site_bytes, ss.size*site_bytes);
    }
    if(verbose) printf("Rank %d comm events %d : sends %d recvs %d copies %d\n", rank, comms.size(), nsend, nrecv, ncp);

    assert(MPI_Waitall(comms.size(), comms.data(), MPI_STATUSES_IGNORE) == MPI_SUCCESS );
    t_comms += dclock();
  
    for(int b=0;b<node_data.size();b++) free(node_data[b]);
    t_total += dclock();
  }

  void close(){
    if(rd_data.size()){
      t_total -= dclock();;
      for(int d=0;d<rd_data.size();d++) rd_data[d].close();
      rd_data.clear();
      sends.clear();

      LOGA2A << "cpsFieldPartIOreader: Read bandwidth " << byte_to_MB(bytes_read) / t_read << " MB/s" << std::endl;
      
      t_total += dclock();
      LOGA2A << "cpsFieldPartIOreader timings - setup:" << t_setup << "s  read:" << t_read << "s  comms:" << t_comms << "s  total: " << t_total << "s" << std::endl;
    }
  }
  ~cpsFieldPartIOreader(){ close(); }
};

CPS_END_NAMESPACE
#endif
