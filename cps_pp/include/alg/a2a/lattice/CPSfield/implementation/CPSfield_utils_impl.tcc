template<typename ComplexType, typename MappingType, typename ParamType>
struct _setupFieldParams{};

template<typename ComplexType, typename MappingType>
struct _setupFieldParams<ComplexType,MappingType,cps::NullObject>{ //any field with NullObject params
  static inline void doit(cps::NullObject &n){}
}; 

#ifdef USE_GRID
template<typename ComplexType, typename FlavorPolicy>
struct _setupFieldParams<ComplexType, FourDSIMDPolicy<FlavorPolicy>, typename FourDSIMDPolicy<FlavorPolicy>::ParamType>{
  static inline void doit(typename FourDSIMDPolicy<FlavorPolicy>::ParamType &p){
    int nsimd = ComplexType::Nsimd();
    FourDSIMDPolicy<FlavorPolicy>::SIMDdefaultLayout(p,nsimd,2); //only divide over spatial directions

    LOGA2A << "4D field params: Nsimd = " << nsimd << ", SIMD dimensions:";
    for(int i=0;i<4;i++)
      LOGA2ANT << p[i] << " ";
    LOGA2ANT << std::endl;
  }
};
template<typename ComplexType, typename FlavorPolicy>
struct _setupFieldParams<ComplexType, ThreeDSIMDPolicy<FlavorPolicy>, typename ThreeDSIMDPolicy<FlavorPolicy>::ParamType>{
  static inline void doit(typename ThreeDSIMDPolicy<FlavorPolicy>::ParamType &p){
    int nsimd = ComplexType::Nsimd();
    ThreeDSIMDPolicy<FlavorPolicy>::SIMDdefaultLayout(p,nsimd);

    LOGA2A << "3D field params: Nsimd = " << nsimd << ", SIMD dimensions:";
    for(int i=0;i<3;i++)
      LOGA2ANT << p[i] << " ";
    LOGA2ANT << std::endl;
  }
};
#endif
