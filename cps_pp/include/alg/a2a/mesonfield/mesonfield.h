#ifndef _CK_MESON_FIELD
#define _CK_MESON_FIELD

#include<set>
#include<util/time_cps.h>
#include<alg/a2a/utils.h>
#include<alg/a2a/base.h>
#include<alg/a2a/lattice.h>

#include "mesonfield_controls.h"
#include "mesonfield_distributed_storage.h"

CPS_START_NAMESPACE


//We have 2 different mode mappings in play in the high-mode part of the inputs: the fully unpacked dilution mapping used by Vfftw, and the time/flavor packed index  spin_color + 12*hit  used by Wfftw
//For the meson field we want to unpack W in flavor also, introducing a new mapping    sc + 12*( flav + nflav*hit)

//We allow for the timeslices associated with the left and right vectors of the outer product to differ
//In the majority of cases the timeslice of the right-hand vector is equal to that of the left-hand, but in some cases we might desire it to be different,
//for example when taking the product of [[W(t1)*V(t1)]] [[W(t2)*W(t2)]] -> [[W(t1)*W(t2)]]


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

  inline double norm2() const{
    double out = 0.;
    for(int i=0;i<size();i++) out += norm(ptr()[i]);
    QMP_sum_array(&out, 1);
    return out;
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

  //These functions return the number of *packed* modes not the full number of modes
  inline const int getNrows() const{ return nmodes_l; }
  inline const int getNcols() const{ return nmodes_r; }

  //These functions return the number of *full* modes
  inline const int getNrowsFull() const{ return lindexdilution.getNv(); }
  inline const int getNcolsFull() const{ return rindexdilution.getNv(); }

  //Return the full set of dilution parameters
  inline const LeftDilutionType & getRowParams() const{ return lindexdilution; }
  inline const RightDilutionType & getColParams() const{ return rindexdilution; }
  
  void plus_equals(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &with, const bool parallel = true);
  void times_equals(const ScalarComplexType f, const bool parallel = true);

  //Replace this meson field with the average of this and a second field, 'with'
  void average(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &with, const bool parallel = true);

  //Set each float to a uniform random number in the specified range
  //Uses a fixed-seed uniform RNG that every node has an identical copy of
  void testRandom(const Float hi=0.5, const Float lo=-0.5);

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

  //Take the complex conjugate of the meson field (parallel)
  void conj(A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &into) const;

  //Take the hermitian conjugate of the meson field (parallel)
  void hconj(A2AmesonField<mf_Policies,A2AfieldR,A2AfieldL> &into) const;

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



#include "implementation/mesonfield_io.tcc"
#include "implementation/mesonfield_compute_impl.tcc"
#include "implementation/mesonfield_impl.tcc"

CPS_END_NAMESPACE





#endif