template<typename mf_Policies>
struct _randomizeVWimpl<mf_Policies,complex_double_or_float_mark>{
  static inline void randomizeVW(A2AvectorV<mf_Policies> &V, A2AvectorW<mf_Policies> &W){
    typedef typename mf_Policies::FermionFieldType FermionFieldType;
    typedef typename mf_Policies::ComplexFieldType ComplexFieldType;
  
    int nl = V.getNl();
    int nh = V.getNh(); //number of fully diluted high-mode indices
    int nhit = V.getNhits();
    assert(nl == W.getNl());
    assert(nh == W.getNh());
    assert(nhit == W.getNhits());
  

    std::vector<FermionFieldType> wl(nl);
    for(int i=0;i<nl;i++) wl[i].setUniformRandom();
  
    std::vector<FermionFieldType> vl(nl);
    for(int i=0;i<nl;i++) vl[i].setUniformRandom();
  
    std::vector<ComplexFieldType> wh(nhit);
    for(int i=0;i<nhit;i++) wh[i].setUniformRandom();
  
    std::vector<FermionFieldType> vh(nh);
    for(int i=0;i<nh;i++) vh[i].setUniformRandom();
    
    for(int i=0;i<nl;i++){
      V.importVl(vl[i],i);
      W.importWl(wl[i],i);
    }

    for(int i=0;i<nh;i++)
      V.importVh(vh[i],i);
  
    for(int i=0;i<nhit;i++)
      W.importWh(wh[i],i);
  }
};

//Ensure this generates randoms in the same order as the scalar version
template<typename mf_Policies>
struct _randomizeVWimpl<mf_Policies,grid_vector_complex_mark>{
  static inline void randomizeVW(A2AvectorV<mf_Policies> &V, A2AvectorW<mf_Policies> &W){
    typedef typename mf_Policies::FermionFieldType::FieldMappingPolicy::EquivalentScalarPolicy ScalarMappingPolicy;
  
    typedef CPSfermion4D<typename mf_Policies::ScalarComplexType, ScalarMappingPolicy, typename mf_Policies::AllocPolicy> ScalarFermionFieldType;
    typedef CPScomplex4D<typename mf_Policies::ScalarComplexType, ScalarMappingPolicy, typename mf_Policies::AllocPolicy> ScalarComplexFieldType;
  
    int nl = V.getNl();
    int nh = V.getNh(); //number of fully diluted high-mode indices
    int nhit = V.getNhits();
    assert(nl == W.getNl());
    assert(nh == W.getNh());
    assert(nhit == W.getNhits());

    ScalarFermionFieldType tmp;
    ScalarComplexFieldType tmp_cmplx;
  
    for(int i=0;i<nl;i++){
      tmp.setUniformRandom();
      W.getWl(i).importField(tmp);
    }
    for(int i=0;i<nl;i++){
      tmp.setUniformRandom();
      V.getVl(i).importField(tmp);
    }
    for(int i=0;i<nhit;i++){
      tmp_cmplx.setUniformRandom();
      W.getWh(i).importField(tmp_cmplx);
    }
    for(int i=0;i<nh;i++){
      tmp.setUniformRandom();
      V.getVh(i).importField(tmp);
    }
  }
};
