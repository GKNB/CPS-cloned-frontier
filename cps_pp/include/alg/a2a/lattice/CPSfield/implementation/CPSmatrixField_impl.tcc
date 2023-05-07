template<typename VectorMatrixType>
double CPSmatrixFieldNorm2(const CPSmatrixField<VectorMatrixType> &f){
  typedef typename VectorMatrixType::scalar_type scalar_type;
  constexpr int nscalar= VectorMatrixType::nScalarType();
  CPSfield<scalar_type,nscalar, FourDSIMDPolicy<OneFlavorPolicy>, HostAllocPolicy> tmp(f.getDimPolParams());
  {
    CPSautoView(tmp_v,tmp,HostWrite);
    CPSautoView(f_v,f,HostRead);
    memcpy(tmp_v.ptr(), f_v.ptr(), f.byte_size());
  }
  return tmp.norm2();
}

template<typename T>
void _testRandom<T, typename std::enable_if<isCPSsquareMatrix<T>::value, void>::type>::rand(T* f, size_t fsize, const Float hi, const Float lo){
  static_assert(sizeof(T) == T::nScalarType()*sizeof(typename T::scalar_type));
  _testRandom<typename T::scalar_type>::rand( (typename T::scalar_type*)f, T::nScalarType()*fsize, hi, lo);
}

template<typename VectorMatrixType>
inline auto Trace(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trV<VectorMatrixType>()) ){
  return unop_v(a, _trV<VectorMatrixType>());
}

template<int Index, typename VectorMatrixType>
inline auto TraceIndex(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trIndexV<Index,VectorMatrixType>()) ){
  return unop_v(a, _trIndexV<Index,VectorMatrixType>());
}
template<int Index, typename VectorMatrixType>
inline void TraceIndex(CPSmatrixField<typename _trIndexV<Index,VectorMatrixType>::OutputType > &out,  const CPSmatrixField<VectorMatrixType> &in){
  unop_v(out, in, _trIndexV<Index,VectorMatrixType>());
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > ColorTrace(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in){
  return TraceIndex<1>(in);
}
template<typename ComplexType>
inline void ColorTrace(CPSmatrixField<CPSspinMatrix<CPSflavorMatrix<ComplexType> > > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in){
  return TraceIndex<1>(out, in);
}

//Trace over two indices of a nested matrix. Requires  Index1 < Index2
template<int Index1, int Index2, typename VectorMatrixType>
inline auto TraceTwoIndices(const CPSmatrixField<VectorMatrixType> &a)->decltype( unop_v(a, _trTwoIndicesV<Index1,Index2,VectorMatrixType>()) ){
  return unop_v(a, _trTwoIndicesV<Index1,Index2,VectorMatrixType>());
}

template<int Index1, int Index2, typename VectorMatrixType>
inline void TraceTwoIndices(CPSmatrixField<typename _trTwoIndicesV<Index1,Index2,VectorMatrixType>::OutputType > &out, const CPSmatrixField<VectorMatrixType> &in){
  unop_v(out, in, _trTwoIndicesV<Index1,Index2,VectorMatrixType>());
}

template<typename ComplexType>
inline CPSmatrixField<CPScolorMatrix<ComplexType> > SpinFlavorTrace(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in){
  return TraceTwoIndices<0,2>(in);
}

template<typename ComplexType>
inline void SpinFlavorTrace(CPSmatrixField<CPScolorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in){
  TraceTwoIndices<0,2>(out,in);
}

template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> Transpose(const CPSmatrixField<VectorMatrixType> &a){
  return unop_v(a, _transposeV<VectorMatrixType>());
}

template<int Index, typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> TransposeOnIndex(const CPSmatrixField<VectorMatrixType> &in){
  return unop_v(in, _transIdx<Index,VectorMatrixType>());
}

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > TransposeColor(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in){
  return TransposeOnIndex<1>(in);
}

template<int Index, typename VectorMatrixType>
inline void TransposeOnIndex(CPSmatrixField<VectorMatrixType> &out, const CPSmatrixField<VectorMatrixType> &in){
  unop_v(out, in, _transIdx<Index,VectorMatrixType>());
}

template<typename ComplexType>
inline void TransposeColor(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in){
  return TransposeOnIndex<1>(out, in);
}

//Complex conjugate
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> cconj(const CPSmatrixField<VectorMatrixType> &a){
  return unop_v(a, _cconjV<VectorMatrixType>());
}

//Left multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > gl_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_v(in,_gl_r_V<ComplexType>(dir));
}
//Right multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > gr_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_v(in,_gr_r_V<ComplexType>(dir));
}
//Left multiplication by gamma(dir)gamma(5)
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > glAx_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_v(in,_glAx_r_V<ComplexType>(dir));
}
//Right multiplication by gamma(dir)gamma(5)
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > grAx_r(const CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_v(in,_grAx_r_V<ComplexType>(dir));
}

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > gl_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_v_2d(in,_gl_r_scf_V_2d<ComplexType>(dir));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > glAx_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_v_2d(in,_glAx_r_scf_V_2d<ComplexType>(dir));
}

template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > gr_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_v_2d(in,_gr_r_scf_V_2d<ComplexType>(dir));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > grAx_r(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_v_2d(in,_grAx_r_scf_V_2d<ComplexType>(dir));
}

template<typename ComplexType>
inline void gl_r(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  unop_v_2d(out,in,_gl_r_scf_V_2d<ComplexType>(dir));
}
template<typename ComplexType>
inline void glAx_r(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  unop_v_2d(out,in,_glAx_r_scf_V_2d<ComplexType>(dir));
}

template<typename ComplexType>
inline void gr_r(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  unop_v_2d(out,in,_gr_r_scf_V_2d<ComplexType>(dir));
}
template<typename ComplexType>
inline void grAx_r(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out, const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  unop_v_2d(out,in,_grAx_r_scf_V_2d<ComplexType>(dir));
}

template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> operator*(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b){
  return binop_v(a,b,_timesV<VectorMatrixType>());
}
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> operator+(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b){
  return binop_v(a,b,_addV<VectorMatrixType>());
}
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> operator-(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b){
  return binop_v(a,b,_subV<VectorMatrixType>());
}

//Trace(a*b) = \sum_{ij} a_{ij}b_{ji}
template<typename VectorMatrixType>
inline CPSmatrixField<typename _traceProdV<VectorMatrixType>::OutputType> Trace(const CPSmatrixField<VectorMatrixType> &a, const CPSmatrixField<VectorMatrixType> &b){
  return binop_v(a,b,_traceProdV<VectorMatrixType>());
}

template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & unit(CPSmatrixField<VectorMatrixType> &in){
  return unop_self_v(in,_unitV<VectorMatrixType>());
}

//in -> i * in
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & timesI(CPSmatrixField<VectorMatrixType> &in){
  return unop_self_v(in,_timesIV<VectorMatrixType>());
}

//in -> -i * in
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & timesMinusI(CPSmatrixField<VectorMatrixType> &in){
  return unop_self_v(in,_timesMinusIV<VectorMatrixType>());
}

//in -> -in
template<typename VectorMatrixType>
inline CPSmatrixField<VectorMatrixType> & timesMinusOne(CPSmatrixField<VectorMatrixType> &in){
  return unop_self_v(in,_timesMinusOneV<VectorMatrixType>());
}

//Left multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & gl(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_gl_V<ComplexType>(dir));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & gl(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_gl_scf_V<ComplexType>(dir));
}

//Right multiplication by gamma matrix
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & gr(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_gr_V<ComplexType>(dir));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & gr(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_gr_scf_V<ComplexType>(dir));
}

//Left multiplication by gamma^dir gamma^5
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & glAx(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_glAx_V<ComplexType>(dir));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & glAx(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_glAx_scf_V<ComplexType>(dir));
}

//Right multiplication by gamma^dir gamma^5
template<typename ComplexType>
inline CPSmatrixField<CPSspinMatrix<ComplexType> > & grAx(CPSmatrixField<CPSspinMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_grAx_V<ComplexType>(dir));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & grAx(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const int dir){
  return unop_self_v(in,_grAx_scf_V<ComplexType>(dir));
}

//Left multiplication by flavor matrix
template<typename ComplexType>
inline CPSmatrixField<CPSflavorMatrix<ComplexType> > & pl(CPSmatrixField<CPSflavorMatrix<ComplexType> > &in, const FlavorMatrixType type){
  return unop_self_v(in,_pl_V<ComplexType>(type));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & pl(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const FlavorMatrixType type){
  return unop_self_v(in,_pl_scf_V<ComplexType>(type));
}

//Right multiplication by flavor matrix
template<typename ComplexType>
inline CPSmatrixField<CPSflavorMatrix<ComplexType> > & pr(CPSmatrixField<CPSflavorMatrix<ComplexType> > &in, const FlavorMatrixType type){
  return unop_self_v(in,_pr_V<ComplexType>(type));
}
template<typename ComplexType>
inline CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > & pr(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &in, const FlavorMatrixType type){
  return unop_self_v(in,_pr_scf_V<ComplexType>(type));
}

template<typename VectorMatrixType>			
CPSmatrixField<typename VectorMatrixType::scalar_type> Trace(const CPSmatrixField<VectorMatrixType> &a, const VectorMatrixType &b){
  using namespace Grid;
  CPSmatrixField<typename VectorMatrixType::scalar_type> out(a.getDimPolParams());
  
  //On NVidia GPUs the compiler fails if b is passed by value into the kernel due to "formal parameter space overflow"
  //As we can't assume b is allocated in UVM memory we need to either create a UVM copy or explicitly copy b to the GPU
#ifdef GPU_VEC
  VectorMatrixType* bptr = (VectorMatrixType*)device_alloc_check(sizeof(VectorMatrixType));
  copy_host_to_device(bptr, &b, sizeof(VectorMatrixType));
#else
  VectorMatrixType const* bptr = &b;
#endif  

  static const int nsimd = VectorMatrixType::scalar_type::Nsimd();
  CPSautoView(av,a,DeviceRead);
  CPSautoView(ov,out,DeviceWrite);
  accelerator_for(x4d, a.size(), nsimd,
  		  {
		    int lane = Grid::acceleratorSIMTlane(nsimd);
		    Trace(*ov.site_ptr(x4d), *av.site_ptr(x4d), *bptr, lane);
  		  }
  		  );

#ifdef GPU_VEC
  device_free(bptr);
#endif

  return out;
}

//Sum the matrix field over sides on this node
template<typename VectorMatrixType>			
VectorMatrixType localNodeSumSimple(const CPSmatrixField<VectorMatrixType> &a){
  CPSautoView(a_v,a,HostRead);
  VectorMatrixType out = *a_v.site_ptr(size_t(0));
  for(size_t i=1;i<a.size();i++) out = out + *a_v.site_ptr(i);
  return out;
}


template<typename VectorMatrixType>			
VectorMatrixType localNodeSum(const CPSmatrixField<VectorMatrixType> &a){
  using namespace Grid;
  VectorMatrixType out; memset(&out, 0, sizeof(VectorMatrixType));

  //Want to introduce some paralellism into this thing
  //Parallelize over fundamental complex type
  typedef typename getScalarType<VectorMatrixType, typename MatrixTypeClassify<VectorMatrixType>::type>::type ScalarType; 

  static const size_t nscalar = sizeof(VectorMatrixType)/sizeof(ScalarType);
  size_t field_size = a.size();
  //Have vol/2 threads sum   x[i] + x[i + vol/2]
  //Then vol/4 for the next round, and so on
  static const int nsimd = ScalarType::Nsimd();
  
  assert(field_size % 2 == 0);

  ScalarType *tmp = (ScalarType*)managed_alloc_check(nscalar*field_size/2*sizeof(ScalarType));
  ScalarType *tmp2 = (ScalarType*)managed_alloc_check(nscalar*field_size/2*sizeof(ScalarType));

  ScalarType *into = tmp;

  CPSautoView(av,a,DeviceRead);
  accelerator_for(offset, nscalar * field_size/2, nsimd,
		  {
		    typedef SIMT<ScalarType> ACC;
		    //Map offset as  s + nscalar*x
		    size_t s = offset % nscalar;
		    size_t x = offset / nscalar;

		    ScalarType const* aa1_ptr = (ScalarType const *)av.site_ptr(x); //pointer to complex type
		    auto aa1 = ACC::read(aa1_ptr[s]);

		    ScalarType const* aa2_ptr = (ScalarType const *)av.site_ptr(x + field_size/2); //pointer to complex type
		    auto aa2 = ACC::read(aa2_ptr[s]);
		    
		    ACC::write(into[offset], aa1+aa2);
		  }
		  );
  
  field_size/=2;

  size_t iter = 0;
  while(field_size % 2 == 0){
    size_t work = nscalar * field_size/2;
    size_t threads = Grid::acceleratorThreads();
    if(work % threads != 0){ //not enough work left even for one workgroup, little point in accelerating (it also causes errors in oneAPI, at least for iris GPus)
      break;
    }
      
    //swap back and forth between the two temp buffers,
    ScalarType const* from = (iter % 2 == 0 ? tmp : tmp2);
    into = (iter % 2 == 0 ? tmp2 : tmp);
    
    //std::cout << "Iteration " << iter << " work=" << work << " nthread=" << Grid::acceleratorThreads() << std::endl;
    
    accelerator_for(offset, work, nsimd,
    		    {
    		      typedef SIMT<ScalarType> ACC;
		      
    		      ScalarType const* aa1_ptr = from + offset;
    		      auto aa1 = ACC::read(*aa1_ptr);
		      
    		      ScalarType const* aa2_ptr = from + offset + nscalar*field_size/2;
    		      auto aa2 = ACC::read(*aa2_ptr);
		      
    		      ACC::write(into[offset], aa1+aa2);
    		    }
    		    );
    field_size/=2;
    ++iter;
  }

  ScalarType *out_s = (ScalarType*)&out;
  for(size_t s=0;s<nscalar;s++)
    for(size_t i=0; i<field_size; i++)
      out_s[s] = out_s[s] + into[s+nscalar*i];
  
  managed_free(tmp);
  managed_free(tmp2);
  return out;
}

template<typename VectorMatrixType>			
inline auto globalSumReduce(const CPSmatrixField<VectorMatrixType> &a)
  ->decltype(Reduce(localNodeSum(a)))
{
  VectorMatrixType lsum = localNodeSum(a);
  auto slsum = Reduce(lsum);
  globalSum(&slsum);
  return slsum;
}

template<typename VectorMatrixType>			
ManagedVector<VectorMatrixType>  localNodeSpatialSumSimple(const CPSmatrixField<VectorMatrixType> &a){
  int Lt_loc = GJP.TnodeSites();

  size_t field_size = a.size();
  assert(a.nodeSites(3) == Lt_loc);
  size_t field_size_3d = field_size/Lt_loc; 

  ManagedVector<VectorMatrixType> out(Lt_loc); 
  CPSautoView(a_v,a,HostRead);
  for(int t=0;t<Lt_loc;t++){
    size_t x4d = a.threeToFour(0,t);
    out[t] = *a_v.site_ptr(x4d);
    for(size_t i=1;i<field_size_3d;i++){
      x4d = a.threeToFour(i,t);
      out[t] = out[t] + *a_v.site_ptr(x4d);
    }
  }
  return out;
}

template<typename VectorMatrixType>			
ManagedVector<VectorMatrixType> localNodeSpatialSum(const CPSmatrixField<VectorMatrixType> &a){
  using namespace Grid;
  int Lt_loc = GJP.TnodeSites();
  ManagedVector<VectorMatrixType> out(Lt_loc); 
  memset(out.data(), 0, Lt_loc*sizeof(VectorMatrixType));
  
  typedef typename getScalarType<VectorMatrixType, typename MatrixTypeClassify<VectorMatrixType>::type  >::type ScalarType; 
  static const size_t nscalar = sizeof(VectorMatrixType)/sizeof(ScalarType);
  size_t field_size = a.size();
  assert(a.nodeSites(3) == Lt_loc);
  size_t field_size_3d = field_size/Lt_loc; 
  //Have vol/2 threads sum   x[i] + x[i + vol/2]
  //Then vol/4 for the next round, and so on
  static const int nsimd = ScalarType::Nsimd();
  
  assert(field_size_3d % 2 == 0);

  ScalarType *tmp = (ScalarType*)managed_alloc_check(nscalar*field_size/2*sizeof(ScalarType));
  ScalarType *tmp2 = (ScalarType*)managed_alloc_check(nscalar*field_size/2*sizeof(ScalarType));

  ScalarType *into = tmp;

  CPSautoView(av,a,DeviceRead);
  accelerator_for(offset, nscalar * Lt_loc * field_size_3d/2, nsimd,
		  {
		    typedef SIMT<ScalarType> ACC;
		    //Map offset as  s + nscalar*(t + Lt_loc*x3d)
		    size_t rem = offset;
		    size_t s = rem % nscalar; rem /= nscalar;
		    size_t t = rem % Lt_loc; rem /= Lt_loc;
		    size_t x3d = rem;

		    size_t x4d = av.threeToFour(x3d,t);
		    size_t x4d_shift = av.threeToFour(x3d + field_size_3d/2, t);

		    ScalarType const* aa1_ptr = (ScalarType const *)av.site_ptr(x4d); //pointer to complex type
		    auto aa1 = ACC::read(aa1_ptr[s]);

		    ScalarType const* aa2_ptr = (ScalarType const *)av.site_ptr(x4d_shift); //pointer to complex type
		    auto aa2 = ACC::read(aa2_ptr[s]);
		    
		    ACC::write(into[offset], aa1+aa2);
		  }
		  );
  
  field_size_3d/=2;

  size_t iter = 0;
  while(field_size_3d % 2 == 0){
    //swap back and forth between the two temp buffers,
    ScalarType const* from = (iter % 2 == 0 ? tmp : tmp2);
    into = (iter % 2 == 0 ? tmp2 : tmp);
    
    accelerator_for(offset, nscalar * Lt_loc * field_size_3d/2, nsimd,
    		    {
    		      typedef SIMT<ScalarType> ACC;
		      
    		      ScalarType const* aa1_ptr = from + offset;
    		      auto aa1 = ACC::read(*aa1_ptr);
		      
    		      ScalarType const* aa2_ptr = from + offset + nscalar*Lt_loc*field_size_3d/2;
    		      auto aa2 = ACC::read(*aa2_ptr);
		      
    		      ACC::write(into[offset], aa1+aa2);
    		    }
    		    );
    field_size_3d/=2;
    ++iter;
  }

  for(int t=0;t<Lt_loc;t++){
    ScalarType *out_s = (ScalarType*)&out[t];
    for(size_t s=0;s<nscalar;s++)    
      for(size_t x3d=0; x3d<field_size_3d; x3d++)
	out_s[s] = out_s[s] + into[s+nscalar*(t+Lt_loc*x3d)];
  }

  managed_free(tmp);
  managed_free(tmp2);
  return out;
}

