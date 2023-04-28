/*
  Expect a functor that acts on SIMD data, pulling out the SIMD lane as appropriate
  Example:

  template<typename VectorMatrixType>
  struct _trV{
     typedef typename VectorMatrixType::scalar_type OutputType;  //must contain OutputType typedef
     accelerator_inline void operator()(OutputType &out, const VectorMatrixType &in, const int lane) const{     //must take SIMD lane as parameter
       SIMT<OutputType>::write(out,0,lane); //Should use SIMT accessors
       _LaneRecursiveTraceImpl<OutputType, VectorMatrixType, cps_square_matrix_mark>::doit(out, in, lane);
     }
  };
*/  

//Unary operation returning result
template<typename T, typename Functor>
auto unop_v(const CPSmatrixField<T> &in, const Functor &l)-> CPSmatrixField<typename Functor::OutputType>{
  using namespace Grid;
  constexpr int nsimd = T::scalar_type::Nsimd();
  CPSmatrixField<typename Functor::OutputType> out(in.getDimPolParams());
  CPSautoView(ov,out,DeviceWrite);
  CPSautoView(iv,in,DeviceRead);
  accelerator_for(x4d, iv.size(), nsimd,
		    {
		      int lane = Grid::acceleratorSIMTlane(nsimd);
		      l(*ov.site_ptr(x4d), *iv.site_ptr(x4d), lane);
		    }
		    );
  return out;
}
//Unary operation on output argument
template<typename T, typename Functor>
void unop_v(CPSmatrixField<typename Functor::OutputType> &out, const CPSmatrixField<T> &in, const Functor &l){
  using namespace Grid;
  constexpr int nsimd = T::scalar_type::Nsimd();
  CPSautoView(ov,out,DeviceWrite);
  CPSautoView(iv,in,DeviceRead);
  accelerator_for(x4d, iv.size(), nsimd,
		    {
		      int lane = Grid::acceleratorSIMTlane(nsimd);
		      l(*ov.site_ptr(x4d), *iv.site_ptr(x4d), lane);
		    }
		    );
}

/*
  Expect a functor that acts on SIMD data, pulling out the SIMD lane as appropriate. The per-site work should be divided over some number of parallel operations
  Example:

  template<typename VectorMatrixType>
  struct mystruct{
     typedef typename VectorMatrixType::scalar_type OutputType;  //must contain OutputType typedef
     accelerator_inline int nParallel() const; //amount of parallel work per site

     accelerator_inline void operator()(OutputType &out, const VectorMatrixType &in, const int widx, const int lane) const;  //widx=work idx inside site, lane = simd lane
  };
*/  

//Unary operation returning result. This version allows for an extra level of parallelism
template<typename T, typename Functor>
auto unop_v_2d(const CPSmatrixField<T> &in, const Functor &l)-> CPSmatrixField<typename Functor::OutputType>{
  using namespace Grid;
  constexpr int nsimd = T::scalar_type::Nsimd();
  CPSmatrixField<typename Functor::OutputType> out(in.getDimPolParams());
  CPSautoView(ov,out,DeviceWrite);
  CPSautoView(iv,in,DeviceRead);
  accelerator_for2d(x4d, iv.size(), widx, l.nParallel(), nsimd,
		    {
		      int lane = Grid::acceleratorSIMTlane(nsimd);
		      l(*ov.site_ptr(x4d), *iv.site_ptr(x4d), widx, lane);
		    }
		    );
  return out;
}
//Unary operation on output argument. This version allows for an extra level of parallelism
template<typename T, typename Functor>
void unop_v_2d(CPSmatrixField<typename Functor::OutputType> &out, const CPSmatrixField<T> &in, const Functor &l){
  using namespace Grid;
  constexpr int nsimd = T::scalar_type::Nsimd();
  CPSautoView(iv,in,DeviceRead);
  CPSautoView(ov,out,DeviceWrite);
  accelerator_for2d(x4d, iv.size(), widx, l.nParallel(), nsimd,
		    {
		      int lane = Grid::acceleratorSIMTlane(nsimd);
		      l(*ov.site_ptr(x4d), *iv.site_ptr(x4d), widx, lane);
		    }
		    );
}

/*
  Expect Functor of the form, e.g.

  template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
  struct _unitV{
    accelerator_inline void operator()(VectorMatrixType &out, const int lane) const{ 
      unit(out, lane);
    }
  };
*/
//Unary operation on self
//For convenience this function returns a reference to the modified input field
template<typename T, typename Functor>
CPSmatrixField<T> & unop_self_v(CPSmatrixField<T> &m, const Functor &l){
  using namespace Grid;
  constexpr int nsimd = T::scalar_type::Nsimd();
  CPSautoView(mvr,m,DeviceRead);
  CPSautoView(mv,m,DeviceWrite);
  accelerator_for(x4d, m.size(), nsimd,
		    {
		      int lane = Grid::acceleratorSIMTlane(nsimd);
		      l(*mv.site_ptr(x4d), lane);
		    }
		    );
  return m;
}

/*Generic unop using functor
  Sadly it does not work for lambdas as the matrix type will be evaluated with both the vector and scalar matrix arguments
  Instead use a templated functor e.g.

  struct _tr{
    template<typename MatrixType>
    accelerator_inline auto operator()(const MatrixType &matrix) const ->decltype(matrix.Trace()){ return matrix.Trace(); }  
  };

  then call
  
  unop(myfield, tr_());
*/
//Unary operation for functionality that acts on scalar data
template<typename T, typename Lambda>
auto unop(const CPSmatrixField<T> &in, const Lambda &l)-> CPSmatrixField<typename std::decay<decltype( l( *((T*)nullptr) ) )>::type>{
  typedef typename std::decay<decltype( l( *((T*)nullptr) ) )>::type outMatrixType;
  using namespace Grid;
  constexpr int nsimd = T::scalar_type::Nsimd();
  CPSmatrixField<outMatrixType> out(in.getDimPolParams());
  CPSautoView(ov,out,DeviceWrite);
  CPSautoView(iv,in,DeviceRead);
  accelerator_for(x4d, in.size(), nsimd,
		    {
		      typedef SIMT<T> ACCr;
		      typedef SIMT<outMatrixType> ACCo;
		      auto aa = ACCr::read(*iv.site_ptr(x4d));
		      ACCo::write(*ov.site_ptr(x4d), l(aa) );
		    }
		    );
  return out;
}


/*
  Expect functor of the form, e.g.

  template<typename VectorMatrixType, typename std::enable_if<isCPSsquareMatrix<VectorMatrixType>::value, int>::type = 0>
  struct _subV{
    typedef VectorMatrixType OutputType;
    accelerator_inline void operator()(VectorMatrixType &out, const VectorMatrixType &a, const VectorMatrixType &b, const int lane) const{ 
      sub(out, a, b, lane);
    }
  };
*/
//Binary operation returning result
template<typename T, typename Functor>
auto binop_v(const CPSmatrixField<T> &a, const CPSmatrixField<T> &b, const Functor &l)-> CPSmatrixField<typename Functor::OutputType>{
  using namespace Grid;
  assert(a.size() == b.size());
  constexpr int nsimd = getScalarType<T, typename MatrixTypeClassify<T>::type>::type::Nsimd();
  CPSmatrixField<typename Functor::OutputType> out(a.getDimPolParams());
  CPSautoView(ov,out,DeviceWrite);
  CPSautoView(av,a,DeviceRead); CPSautoView(bv,b,DeviceRead);
  
  accelerator_for(x4d, av.size(), nsimd,
		    {
		      int lane = Grid::acceleratorSIMTlane(nsimd);
		      l(*ov.site_ptr(x4d), *av.site_ptr(x4d), *bv.site_ptr(x4d), lane);
		    }
		    );
  return out;
}

//Binary operation for functionality that acts only on scalar data
template<typename T, typename U, typename Lambda>
auto binop(const CPSmatrixField<T> &a, const CPSmatrixField<U> &b, const Lambda &l)-> 
  CPSmatrixField<typename std::decay<decltype( l( *((T*)nullptr), *((T*)nullptr) ) )>::type>{
  typedef typename std::decay<decltype( l( *((T*)nullptr), *((T*)nullptr) ) )>::type outMatrixType;
  using namespace Grid;
  constexpr int nsimd = T::scalar_type::Nsimd();
  CPSmatrixField<outMatrixType> out(a.getDimPolParams());
  CPSautoView(ov,out,DeviceWrite);
  CPSautoView(av,a,DeviceRead);
  CPSautoView(bv,b,DeviceRead);
  
  accelerator_for(x4d, av.size(), nsimd,
		    {
		      typedef SIMT<T> ACCra;
		      typedef SIMT<U> ACCrb;			
		      typedef SIMT<outMatrixType> ACCo;
		      auto aa = ACCra::read(*av.site_ptr(x4d));
		      auto bb = ACCrb::read(*bv.site_ptr(x4d));
		      ACCo::write(*ov.site_ptr(x4d), l(aa,bb) );
		    }
		    );
  return out;
}
