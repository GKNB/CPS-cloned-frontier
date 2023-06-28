//For derived classes we want methods to return references or instances of the derived type for inherited functions
//Must have "base_type" defined
//DERIVED = the full derived class name, eg CPSflavorMatrix<T>
//DERIVED_CON = the derived class constructor name, eg CPSflavorMatrix
#define INHERIT_METHODS_AND_TYPES(DERIVED, DERIVED_CON)		\
  typedef typename base_type::value_type value_type;	\
  typedef typename base_type::scalar_type scalar_type;			\
									\
  accelerator DERIVED_CON(const base_type &r): base_type(r){} \
  accelerator DERIVED_CON(): base_type(){}			\
  accelerator DERIVED_CON(base_type &&r): base_type(std::move(r)){}	\
									\
  accelerator_inline DERIVED Transpose() const{ return this->base_type::Transpose(); } \
  template<int TransposeDepth>						\
  accelerator_inline DERIVED TransposeOnIndex() const{ return this->base_type::template TransposeOnIndex<TransposeDepth>(); } \
									\
  accelerator_inline DERIVED & zero(){ return static_cast<DERIVED &>(this->base_type::zero()); } \
  accelerator_inline DERIVED & unit(){ return static_cast<DERIVED &>(this->base_type::unit()); } \
  accelerator_inline DERIVED & timesMinusOne(){ return static_cast<DERIVED &>(this->base_type::timesMinusOne()); } \
  accelerator_inline DERIVED & timesI(){ return static_cast<DERIVED &>(this->base_type::timesI()); } \
  accelerator_inline DERIVED & timesMinusI(){ return static_cast<DERIVED &>(this->base_type::timesMinusI()); } \
									\
  accelerator_inline DERIVED & operator=(const base_type &r){ return static_cast<DERIVED &>(this->base_type::operator=(r)); } \
  accelerator_inline DERIVED & operator=(base_type &&r){ return static_cast<DERIVED &>(this->base_type::operator=(std::move(r))); } \
  accelerator_inline DERIVED & operator+=(const DERIVED &r){ return static_cast<DERIVED &>(this->base_type::operator+=(r)); } \
  accelerator_inline DERIVED & operator*=(const scalar_type &r){ return static_cast<DERIVED &>(this->base_type::operator*=(r)); } \
  accelerator_inline DERIVED & operator-=(const DERIVED &r){ return static_cast<DERIVED &>(this->base_type::operator-=(r)); } 
