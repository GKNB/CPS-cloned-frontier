#ifndef _MODE_MAPPING_IMPL_H
#define _MODE_MAPPING_IMPL_H

/////////Work out the unmapping for a single type
template<int Depth, typename DilutionType>
struct computeModeUnmapping{
  typedef typename IndexVector<Depth>::Type VectorType;
  static void doit(VectorType &v,modeIndexSet &coord, const DilutionType &dil){
    int nidx = IndexConvention<Depth>::getNidx(dil);
    v.resize(nidx);
    for(int i=0;i<nidx;i++){
      IndexConvention<Depth>::set(coord,i);
      computeModeUnmapping<Depth-1,DilutionType>::doit(v[i],coord, dil);
    }
  }
};

template<typename DilutionType>
struct computeModeUnmapping<0,DilutionType>{
  typedef typename IndexVector<0>::Type VectorType;

  static void doit(VectorType &v,modeIndexSet &coord, const DilutionType &dil){
    std::vector<int> &std_to_dil = v.first;
    std::vector<bool> &non_zeroes = v.second;
    dil.getIndexMapping(std_to_dil, non_zeroes, coord);
  }
};

//Compute the mode map between types recursively

//First move through left nested coords   //spincolor=3, flavor=2, time=1
template<int DepthLeftDilution, int DepthRightDilution, typename LeftDilutionType, typename RightDilutionType>
struct computeModeMap{
  typedef typename IndexVector<DepthLeftDilution>::Type VectorTypeLeftDilution;
  typedef typename IndexVector<DepthRightDilution>::Type VectorTypeRightDilution;

  typedef typename IndexTensor<DepthLeftDilution, DepthRightDilution>::Type TensorType;
  
  static void doit(TensorType &v,const VectorTypeLeftDilution &left_unmap, const VectorTypeRightDilution &right_unmap, const A2Aparams &p){
    int nidx = IndexConvention<DepthLeftDilution>::getNidx(p);
    v.resize(nidx);
    for(int i=0;i<nidx;i++)
      computeModeMap<DepthLeftDilution-1,DepthRightDilution, LeftDilutionType,RightDilutionType>::doit(v[i],left_unmap[i],right_unmap, p);
  }
};
//Gotten down to the base modes for left, start on right
template<int DepthRightDilution, typename LeftDilutionType, typename RightDilutionType>
struct computeModeMap<0,DepthRightDilution,LeftDilutionType,RightDilutionType>{ 
  typedef typename IndexVector<0>::Type VectorTypeLeftDilution;
  typedef typename IndexVector<DepthRightDilution>::Type VectorTypeRightDilution;

  typedef typename IndexTensor<0, DepthRightDilution>::Type TensorType;
  
  static void doit(TensorType &v,const VectorTypeLeftDilution &left_unmap, const VectorTypeRightDilution &right_unmap, const A2Aparams &p){
    int nidx = IndexConvention<DepthRightDilution>::getNidx(p);
    v.resize(nidx);
    for(int i=0;i<nidx;i++)
      computeModeMap<0,DepthRightDilution-1, LeftDilutionType,RightDilutionType>::doit(v[i],left_unmap,right_unmap[i],p);
  }
};
template<typename LeftDilutionType, typename RightDilutionType>
struct computeModeMap<0,0,LeftDilutionType,RightDilutionType>{ //gotten down to the base modes for left, start on right
  typedef typename IndexVector<0>::Type VectorTypeLeftDilution;
  typedef typename IndexVector<0>::Type VectorTypeRightDilution;

  typedef typename IndexTensor<0, 0>::Type TensorType; //mode * mode
  
  static void doit(TensorType &v,const VectorTypeLeftDilution &left_unmap, const VectorTypeRightDilution &right_unmap, const A2Aparams &p){
    //Fully unpack both and find the overlap between the sets of non-zero indices
    const std::vector<bool> &non_zeroes_left = left_unmap.second;
    const std::vector<bool> &non_zeroes_right = right_unmap.second;

    const std::vector<int> &std_to_left = left_unmap.first;
    const std::vector<int> &std_to_right = right_unmap.first;

    std::vector<bool> overlap;
    compute_overlap(overlap, non_zeroes_left, non_zeroes_right);

    int n_std = overlap.size();
    assert(std_to_left.size() == n_std);
    assert(std_to_right.size() == n_std);

    v.resize(0); v.reserve(n_std);

    for(int i=0;i<n_std;i++)
      if(overlap[i]){
	std::pair<int,int> idx_pair(std_to_left[i], std_to_right[i]);
	v.push_back(idx_pair);
      }
  }
};



template<typename LeftDilutionType, typename RightDilutionType>
void ModeMapping<LeftDilutionType,RightDilutionType>::compute(TensorType &idx_map, const A2Aparams &p){
  modeIndexSet tmp;
  const LeftDilutionType &left = static_cast<const LeftDilutionType &>(p);
  const RightDilutionType &right = static_cast<const RightDilutionType &>(p);
    
  VectorTypeLeftDilution left_unmap;
  computeModeUnmapping<DepthLeftDilution,LeftDilutionType>::doit(left_unmap,tmp,left);

  VectorTypeRightDilution right_unmap;
  computeModeUnmapping<DepthRightDilution,RightDilutionType>::doit(right_unmap,tmp,right);
  
  computeModeMap<DepthLeftDilution,DepthRightDilution, LeftDilutionType,RightDilutionType>::doit(idx_map,left_unmap,right_unmap,p);
}




#endif
