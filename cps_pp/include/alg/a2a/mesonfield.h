#ifndef _CK_MESON_FIELD
#define _CK_MESON_FIELD

#include<set>
#include<util/time_cps.h>
#include<alg/a2a/scfvectorptr.h>
#include<alg/a2a/mode_contraction_indices.h>
#include<alg/a2a/spin_color_matrices.h>
#include<alg/a2a/a2a_dilutions.h>
#include<alg/a2a/CPSfield.h>
#include<alg/a2a/mesonfield_controls.h>

CPS_START_NAMESPACE

template<typename Dilution>
struct FlavorUnpacked{};

template<>
struct FlavorUnpacked<TimeFlavorPackedIndexDilution>{
  typedef TimePackedIndexDilution UnpackedType;
};
template<>
struct FlavorUnpacked<StandardIndexDilution>{
  typedef StandardIndexDilution UnpackedType;
};

//We have 2 different mode mappings in play in the high-mode part of the inputs: the fully unpacked dilution mapping used by Vfftw, and the time/flavor packed index  spin_color + 12*hit  used by Wfftw
//For the meson field we want to unpack W in flavor also, introducing a new mapping    sc + 12*( flav + nflav*hit)

//We allow for the timeslices associated with the left and right vectors of the outer product to differ
//In the majority of cases the timeslice of the right-hand vector is equal to that of the left-hand, but in some cases we might desire it to be different,
//for example when taking the product of [[W(t1)*V(t1)]] [[W(t2)*W(t2)]] -> [[W(t1)*W(t2)]]

//If using the NODE_DISTRIBUTE_MESONFIELDS option, we can choose whether to distribute over the memory of the nodes or to read/write from disk
#define MESONFIELD_USE_DISTRIBUTED_STORAGE

#ifdef MESONFIELD_USE_BURSTBUFFER
typedef BurstBufferMemoryStorage MesonFieldDistributedStorageType;
#elif defined(MESONFIELD_USE_NODE_SCRATCH)
typedef IndependentDiskWriteStorage MesonFieldDistributedStorageType;
#elif defined(MESONFIELD_USE_DISTRIBUTED_STORAGE)
typedef DistributedMemoryStorage MesonFieldDistributedStorageType;
#else
#error "Meson field invalid storage type"
#endif


template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
class A2AmesonField: public MesonFieldDistributedStorageType{
public:
  //Deduce the dilution types for the meson field. We unpack the flavor index in W fields
  typedef typename A2AfieldL<mf_Policies>::DilutionType LeftInputDilutionType;
  typedef typename A2AfieldR<mf_Policies>::DilutionType RightInputDilutionType;

  typedef typename FlavorUnpacked<LeftInputDilutionType>::UnpackedType LeftDilutionType;
  typedef typename FlavorUnpacked<RightInputDilutionType>::UnpackedType RightDilutionType;
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
 private:
  int nmodes_l, nmodes_r;
  int fsize; //in units of ScalarComplexType

  LeftDilutionType lindexdilution;
  RightDilutionType rindexdilution;

  int tl, tr; //time coordinates associated with left and right fields of the outer-product

  template<typename, template <typename> class ,  template <typename> class >
  friend class A2AmesonField; //friend this class but with other field types

public:
  A2AmesonField(): fsize(0), nmodes_l(0), nmodes_r(0), MesonFieldDistributedStorageType(){ }

  //Just setup memory (setup is automatically called when 'compute' is called, so this is not necessary. However if you disable the setup at compute time you should setup the memory beforehand)
  A2AmesonField(const A2AfieldL<mf_Policies> &l, const A2AfieldR<mf_Policies> &r): fsize(0), nmodes_l(0), nmodes_r(0), MesonFieldDistributedStorageType(){
    setup(l,r,-1,-1);
  }

 A2AmesonField(const A2AmesonField &r): nmodes_l(r.nmodes_l), nmodes_r(r.nmodes_r),
					 fsize(r.fsize), lindexdilution(r.lindexdilution), rindexdilution(r.rindexdilution),
					 tl(r.tl), tr(r.tr), MesonFieldDistributedStorageType(r){ }

  //Call this when you use the default constructor if not automatically called (it is called automatically in ::compute)
  void setup(const A2Aparams &lp, const A2Aparams &rp, const int _tl, const int _tr){
    tl = _tl; tr = _tr;
    lindexdilution = lp; rindexdilution = rp;

    nmodes_l = lindexdilution.getNmodes();
    nmodes_r = rindexdilution.getNmodes();

    int old_fsize = fsize;
    fsize = nmodes_l*nmodes_r;

    if(this->data() != NULL && old_fsize != fsize ){
      this->freeMem();
      this->alloc(128, fsize * sizeof(ScalarComplexType)); 
    }else if(this->data() == NULL){
      this->alloc(128, fsize * sizeof(ScalarComplexType)); 
    }
    zero();
  }

  //Return size in bytes
  size_t byte_size() const{ return fsize * sizeof(ScalarComplexType); }

  //Can compute the byte size without an instance as long as we know the params
  static size_t byte_size(const A2Aparams &lp, const A2Aparams &rp){
    LeftDilutionType lindexdilution = lp;
    RightDilutionType rindexdilution = rp;
    int nmodes_l = lindexdilution.getNmodes();
    int nmodes_r = rindexdilution.getNmodes();
    size_t fsize = nmodes_l*nmodes_r;
    return fsize * sizeof(ScalarComplexType);
  }

  void free_mem(){
    this->freeMem();
  }

  bool equals(const A2AmesonField &r, const double tolerance = 1e-10, bool verbose = false) const{
    for(int i=0;i<nmodes_l;i++){
      for(int j=0;j<nmodes_r;j++){
	const ScalarComplexType &lval = (*this)(i,j);
	const ScalarComplexType &rval = r(i,j);
	
	if( fabs(lval.real() - rval.real()) > tolerance || fabs(lval.imag() - rval.imag()) > tolerance ){
	  if(verbose && !UniqueID()){
	    printf("Err: (%d,%d) : this[%g,%g] vs that[%g,%g] : diff [%g,%g]\n",i,j,
		   lval.real(),lval.imag(),rval.real(),rval.imag(),fabs(lval.real()-rval.real()), fabs(lval.imag()-rval.imag()) ); fflush(stdout);
	  }
	  return false;
	}
      }
    }
    return true;
  }    
  
  A2AmesonField &operator=(const A2AmesonField &r){    
    nmodes_l = r.nmodes_l; nmodes_r = r.nmodes_r; fsize = r.fsize;
    lindexdilution = r.lindexdilution;  rindexdilution = r.rindexdilution;
    tl = r.tl; tr = r.tr;
    ((MesonFieldDistributedStorageType*)this)->operator=(r);
    return *this;
  }


  inline ScalarComplexType* ptr(){ return (ScalarComplexType*)this->data(); } //Use at your own risk
  inline ScalarComplexType const* ptr() const{ return (ScalarComplexType const*)this->data(); }

  void move(A2AmesonField &from){
    nmodes_l = from.nmodes_l; nmodes_r = from.nmodes_r; fsize = from.fsize;
    lindexdilution = from.lindexdilution; rindexdilution = from.rindexdilution; 
    tl = from.tl; tr = from.tr;
    ((MesonFieldDistributedStorageType*)this)->move(from);
  }
  
  //Size in complex
  inline const int size() const{ return fsize; }

  //Access elements with compressed mode index
  inline ScalarComplexType & operator()(const int i, const int j){ //Use at your own risk
    return this->ptr()[j + nmodes_r*i]; //right mode index changes most quickly
  }
  
  inline const ScalarComplexType & operator()(const int i, const int j) const{
    return this->ptr()[j + nmodes_r*i];
  }
  
  inline const int getRowTimeslice() const{ return tl; }
  inline const int getColTimeslice() const{ return tr; }
  
  //A slow implementation to access elements from full unpacked indices
  inline const ScalarComplexType & elem(const int full_i, const int full_j) const{
    StaticAssert< _equal<LeftDilutionType,StandardIndexDilution>::value || _equal<LeftDilutionType,TimePackedIndexDilution>::value >();
    StaticAssert< _equal<RightDilutionType,StandardIndexDilution>::value || _equal<RightDilutionType,TimePackedIndexDilution>::value >();
    
    static ScalarComplexType zero(0.0,0.0);

    int nll = lindexdilution.getNl();
    int nlr = rindexdilution.getNl();

    int packed_i;
    if(_equal<LeftDilutionType,StandardIndexDilution>::value || full_i < nll) packed_i = full_i; //  lindexdilution.getModeType() == StandardIndex
    else{ // W *
      StandardIndexDilution lfulldil(lindexdilution);
      modeIndexSet i_idx; lfulldil.indexUnmap(full_i-nll, i_idx);
      if(i_idx.time != tl) return zero; //delta function in time
      else packed_i = nll + lindexdilution.indexMap(i_idx);
    }
    int packed_j;
    if(_equal<RightDilutionType,StandardIndexDilution>::value || full_j < nlr) packed_j = full_j; //rindexdilution.getModeType() == StandardIndex
    else{ //* W
      StandardIndexDilution rfulldil(rindexdilution);
      modeIndexSet j_idx; rfulldil.indexUnmap(full_j-nlr, j_idx);
      if(j_idx.time != tr) return zero;
      else packed_j = nlr + rindexdilution.indexMap(j_idx);
    }
    return this->operator()(packed_i,packed_j);
  }

  //Version of the above that returns a pointer so that the element can be modified. A NULL pointer will be returned for elements that are enforced to be zero by the index packing
  inline ScalarComplexType* elem_ptr(const int full_i, const int full_j){
    StaticAssert< _equal<LeftDilutionType,StandardIndexDilution>::value || _equal<LeftDilutionType,TimePackedIndexDilution>::value >();
    StaticAssert< _equal<RightDilutionType,StandardIndexDilution>::value || _equal<RightDilutionType,TimePackedIndexDilution>::value >();
    
    int nll = lindexdilution.getNl();
    int nlr = rindexdilution.getNl();

    int packed_i;
    if(_equal<LeftDilutionType,StandardIndexDilution>::value || full_i < nll) packed_i = full_i; //  lindexdilution.getModeType() == StandardIndex
    else{ // W *
      StandardIndexDilution lfulldil(lindexdilution);
      modeIndexSet i_idx; lfulldil.indexUnmap(full_i-nll, i_idx);
      if(i_idx.time != tl) return NULL; //delta function in time
      else packed_i = nll + lindexdilution.indexMap(i_idx);
    }
    int packed_j;
    if(_equal<RightDilutionType,StandardIndexDilution>::value || full_j < nlr) packed_j = full_j; //rindexdilution.getModeType() == StandardIndex
    else{ //* W
      StandardIndexDilution rfulldil(rindexdilution);
      modeIndexSet j_idx; rfulldil.indexUnmap(full_j-nlr, j_idx);
      if(j_idx.time != tr) return NULL;
      else packed_j = nlr + rindexdilution.indexMap(j_idx);
    }
    return &this->operator()(packed_i,packed_j);
  }

  inline void zero(const bool parallel = true){
    memset(this->data(), 0, sizeof(ScalarComplexType) * fsize);      
  }
  //For all mode indices l_i and r_j, compute the meson field  V^-1 \sum_p l_i^\dagger(p,t) M(p,t) r_j(p,t)
  //It is assumed that A2AfieldL and A2AfieldR are Fourier transformed field containers
  //M(p,t) is a completely general momentum-space spin/color/flavor matrix per temporal slice
  //do_setup allows you to disable the reassignment of the memory (it will still reset to zero). Use wisely!
  template<typename InnerProduct>
  void compute(const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r,const int &t, bool do_setup = true);

  //This version is more efficient on multi-nodes
  template<typename InnerProduct, typename Allocator>
  static void compute(std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator > &mf_t, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup = true);

  //Version of the above for multi-src inner products (output vector indexed by [src idx][t]
  template<typename InnerProduct, typename Allocator>
  static void compute(std::vector< std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator >* > &mf_st, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup = true);

  inline const int getNrows() const{ return nmodes_l; }
  inline const int getNcols() const{ return nmodes_r; }

  inline const LeftDilutionType & getRowParams() const{ return lindexdilution; }
  inline const RightDilutionType & getColParams() const{ return rindexdilution; }
  
  void plus_equals(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &with, const bool parallel = true);
  void times_equals(const ScalarComplexType f, const bool parallel = true);

  //Replace this meson field with the average of this and a second field, 'with'
  void average(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &with, const bool parallel = true);

  //Set each float to a uniform random number in the specified range
  //Uses a fixed-seed uniform RNG that every node has an identical copy of
  void testRandom(const Float hi=0.5, const Float lo=-0.5){    
    static UniformRandomGenerator urng(hi,lo);
    static bool init = false;
    if(!init){ urng.Reset(1234); init = true; }

    for(int i=0;i<this->fsize;i++) this->ptr()[i] = ScalarComplexType(urng.Rand(hi,lo), urng.Rand(hi,lo) );
  }

  //Reorder the rows so that all the elements in idx_map are sequential. Indices not in map may be written over. Use at your own risk
  void rowReorder(A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &into, const int idx_map[], int map_size, bool parallel = true) const;
  void colReorder(A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &into, const int idx_map[], int map_size, bool parallel = true) const;

  //Do a column reorder but where we pack the row indices to exclude those not used (as indicated by input bool array)
  //Output as a GSL matrix. Can reuse previously allocated matrix providing its big enough
  typename gsl_wrapper<typename ScalarComplexType::value_type>::matrix_complex * GSLpackedColReorder(const int idx_map[], int map_size, bool rowidx_used[], typename gsl_wrapper<typename ScalarComplexType::value_type>::matrix_complex *reuse = NULL) const;

#ifdef USE_GRID
  //Do a column reorder but where we pack the row indices to exclude those not used (as indicated by input bool array)
  //Output to a linearized matrix of Grid SIMD vectors where we have splatted the scalar onto all SIMD lanes
  //Option not to resize the output vector, allowing reuse of a previously allocated vector providing it's large enough
  void splatPackedColReorder(typename AlignedVector<typename mf_Policies::ComplexType>::type &into, const int idx_map[], int map_size, bool rowidx_used[], bool do_resize = true) const;
  void scalarPackedColReorder(typename AlignedVector<typename mf_Policies::ComplexType>::type &into, const int idx_map[], int map_size, bool rowidx_used[], bool do_resize = true) const;
#endif
  
  //Transpose the meson field! (parallel)
  void transpose(A2AmesonField<mf_Policies,A2AfieldR,A2AfieldL> &into) const;

  //Delete all the data associated with this meson field apart from on node with UniqueID 'node'. The node index is saved so that the data can be later retrieved.
  //The memory will be distributed according to a global index that cycles between 0... nodes-1 (with looping) to ensure even distribution
  void nodeDistribute();
  //Get back the data. After the call, all nodes will have a complete copy
  //All nodes must call nodeGet simultaneously. If 'require' is false the comms will be performed but the data will not be kept on this node
  void nodeGet(bool require = true);

  void write(std::ostream *file_ptr, FP_FORMAT fileformat = FP_AUTOMATIC) const;
  void write(const std::string &filename, FP_FORMAT fileformat = FP_AUTOMATIC) const;
  void read(std::istream *file_ptr); //istream pointer should only be open on node 0 - should be NULL otherwise
  void read(const std::string &filename);

  static void write(const std::string &filename, const std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> > &mfs, FP_FORMAT fileformat = FP_AUTOMATIC);
  static void write(std::ostream *file_ptr, const std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> > &mfs, FP_FORMAT fileformat = FP_AUTOMATIC);
  static void read(const std::string &filename, std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> > &mfs);
  static void read(std::istream *file_ptr, std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> > &mfs);

  void nodeSum(){ //don't call unless you know what you're doing
    QMP_sum_array( (typename ScalarComplexType::value_type*)this->data(),2*fsize);
  }
};

//Matrix product of meson field pairs
//out(t1,t4) = l(t1,t2) * r(t3,t4)     (The stored timeslices are only used to unpack TimePackedIndex so it doesn't matter if t2 and t3 are thrown away; their indices are contracted over hence the times are not needed)
//Threaded and node distributed. 
//Node-locality can be enabled with 'node_local = true'
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
void mult(A2AmesonField<mf_Policies,lA2AfieldL,rA2AfieldR> &out, const A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> &l, const A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> &r, const bool node_local = false);


// l^i(xop,top) M^ij r^j(xop,top)
//argument xop is the *local* 3d site index in canonical ordering, top is the *local* time coordinate
// Node local and unthreaded
template<typename mf_Policies, 
	 template <typename> class lA2Afield,  
	 template <typename> class MA2AfieldL,  template <typename> class MA2AfieldR,
	 template <typename> class rA2Afield  
	 >
void mult(CPSspinColorFlavorMatrix<typename mf_Policies::ComplexType> &out, const lA2Afield<mf_Policies> &l,  const A2AmesonField<mf_Policies,MA2AfieldL,MA2AfieldR> &M, const rA2Afield<mf_Policies> &r, const int &xop, const int &top, const bool &conj_l, const bool &conj_r);

// l^i(xop,top) r^i(xop,top)
//argument xop is the *local* 3d site index in canonical ordering, top is the *local* time coordinate
// Node local and unthreaded
template<typename mf_Policies, 
	 template <typename> class lA2Afield,  
	 template <typename> class rA2Afield  
	 >
void mult(CPSspinColorFlavorMatrix<typename mf_Policies::ComplexType> &out, const lA2Afield<mf_Policies> &l, const rA2Afield<mf_Policies> &r, const int &xop, const int &top, const bool &conj_l, const bool &conj_r);



//Compute   l^ij(t1,t2) r^ji(t3,t4)
//Threaded but *node local*
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
typename mf_Policies::ScalarComplexType trace(const A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> &l, const A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> &r);

//Compute   l^ij(t1,t2) r^ji(t3,t4) for all t1, t4  and place into matrix element t1,t4
//This is both threaded and distributed over nodes
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
void trace(fMatrix<typename mf_Policies::ScalarComplexType> &into, const std::vector<A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> > &l, const std::vector<A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> > &r);


//Compute   m^ii(t1,t2)
//Threaded but *node local*
template<typename mf_Policies, 
	 template <typename> class A2AfieldL,  template <typename> class A2AfieldR
	 >
typename mf_Policies::ScalarComplexType trace(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &m);

//Compute   m^ii(t1,t2)  for an arbitrary vector of meson fields
//This is both threaded and distributed over nodes
template<typename mf_Policies, 
	 template <typename> class A2AfieldL,  template <typename> class A2AfieldR
	 >
void trace(std::vector<typename mf_Policies::ScalarComplexType> &into, const std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> > &m);


template<typename T>
void nodeGetMany(const int n, std::vector<T> *a, ...);

//Same as above but the user can pass in a set of bools that tell the gather whether the MF on that timeslice is required. If not it is internally deleted, freeing memory
template<typename T>
void nodeGetMany(const int n, std::vector<T> *a, std::vector<bool> const* a_timeslice_mask,  ...);

template<typename T>
void nodeDistributeMany(const int n, std::vector<T> *a, ...);

//Distribute only meson fields in 'from' that are *not* present in any of the sets 'notina' and following
template<typename T>
void nodeDistributeUnique(std::vector<T> &from, const int n, std::vector<T> const* notina, ...);


template<typename T>
bool mesonFieldsOnNode(const std::vector<T> &mf){
  for(int i=0;i<mf.size();i++) if(!mf[i].isOnNode()) return false;
  return true;
}

#include<alg/a2a/mesonfield_mult_impl.tcc>
#include<alg/a2a/mesonfield_mult_vMv_impl.tcc>
#include<alg/a2a/mesonfield_mult_vv_impl.tcc>
#include<alg/a2a/mesonfield_io.tcc>
#include<alg/a2a/mesonfield_compute_impl.tcc>
#include<alg/a2a/mesonfield_impl.tcc>
CPS_END_NAMESPACE





#endif
