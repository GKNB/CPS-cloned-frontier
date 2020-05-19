#ifndef _MODE_CONTRACTION_INDICES_H
#define _MODE_CONTRACTION_INDICES_H

#include "mode_mapping.h"

CPS_START_NAMESPACE

//When we contract over modes between two meson fields with different numbers of row/column modes (i.e. because one is packed in time) we need to figure out the mapping between the two
//This class computes and stores the unpacked indices making the contraction transparent. It is a simple wrapper around ModeMapping

//Predefine meson field
template<typename mf_Float,template <typename> class A2AfieldL,template <typename> class A2AfieldR>
class A2AmesonField;


//Get an index
template<int LeftDilutionDepth, int RightDilutionDepth>
struct getIndex{
  typedef typename IndexTensor<LeftDilutionDepth,RightDilutionDepth>::Type TensorType;
  inline static const std::pair<int,int> & doit(const int i, const modeIndexSet &left_coord, const modeIndexSet &right_coord, const TensorType &mode_map){
    int val_left = IndexConvention<LeftDilutionDepth>::get(left_coord); 
    return getIndex<LeftDilutionDepth-1,RightDilutionDepth>::doit(i, left_coord, right_coord,mode_map[val_left] );
  }
};
template<int RightDilutionDepth>
struct getIndex<0,RightDilutionDepth>{
  typedef typename IndexTensor<0,RightDilutionDepth>::Type TensorType;
  inline static const std::pair<int,int> & doit(const int i, const modeIndexSet &left_coord, const modeIndexSet &right_coord, const TensorType &mode_map){
    int val_right = IndexConvention<RightDilutionDepth>::get(right_coord); 
    return getIndex<0,RightDilutionDepth-1>::doit(i, left_coord, right_coord,mode_map[val_right] );
  }
};
template<>
struct getIndex<0,0>{
  typedef typename IndexTensor<0,0>::Type TensorType;
  inline static const std::pair<int,int> & doit(const int i, const modeIndexSet &left_coord, const modeIndexSet &right_coord, const TensorType &mode_map){
    return mode_map[i];
  }
};

//Get number of overlapping indices
template<int LeftDilutionDepth, int RightDilutionDepth>
struct _getNindices{
  typedef typename IndexTensor<LeftDilutionDepth,RightDilutionDepth>::Type TensorType;
  inline static int doit(const modeIndexSet &left_coord, const modeIndexSet &right_coord, const TensorType &mode_map){
    int val_left = IndexConvention<LeftDilutionDepth>::get(left_coord); 
    return _getNindices<LeftDilutionDepth-1,RightDilutionDepth>::doit(left_coord, right_coord,mode_map[val_left] );
  }
};
template<int RightDilutionDepth>
struct _getNindices<0,RightDilutionDepth>{
  typedef typename IndexTensor<0,RightDilutionDepth>::Type TensorType;
  inline static int doit(const modeIndexSet &left_coord, const modeIndexSet &right_coord, const TensorType &mode_map){
    int val_right = IndexConvention<RightDilutionDepth>::get(right_coord); 
    return _getNindices<0,RightDilutionDepth-1>::doit(left_coord, right_coord,mode_map[val_right] );
  }
};
template<>
struct _getNindices<0,0>{
  typedef typename IndexTensor<0,0>::Type TensorType;
  inline static int doit(const modeIndexSet &left_coord, const modeIndexSet &right_coord, const TensorType &mode_map){
    return mode_map.size();
  }
};


template<typename LeftDilutionType, typename RightDilutionType>
class ModeContractionIndices{
  typedef typename ModeMapping<LeftDilutionType,RightDilutionType>::TensorType TensorType;

  enum { DepthLeftDilution = LeftDilutionType::UndilutedIndices };
  enum { DepthRightDilution = RightDilutionType::UndilutedIndices };

  TensorType mode_map;
 public:
  ModeContractionIndices() = default;

  ModeContractionIndices(const A2Aparams &a2a_params){
    compute(a2a_params);
  }

  void compute(const A2Aparams &a2a_params){
    ModeMapping<LeftDilutionType,RightDilutionType>::compute(mode_map,a2a_params);
  }
  
  int getLeftIndex(const int i, const modeIndexSet &left_coord, const modeIndexSet &right_coord) const{
    const std::pair<int,int> &idx_pair = getIndex<DepthLeftDilution,DepthRightDilution>::doit(i, left_coord, right_coord, mode_map);
    return idx_pair.first;
  }
  int getRightIndex(const int i, const modeIndexSet &left_coord, const modeIndexSet &right_coord) const{
    const std::pair<int,int> &idx_pair = getIndex<DepthLeftDilution,DepthRightDilution>::doit(i, left_coord, right_coord, mode_map);
    return idx_pair.second;
  }

  void getBothIndices(int &il, int &ir, const int &i, const modeIndexSet &left_coord, const modeIndexSet &right_coord) const{
    const std::pair<int,int> &idx_pair = getIndex<DepthLeftDilution,DepthRightDilution>::doit(i, left_coord, right_coord, mode_map);
    il = idx_pair.first; 
    ir = idx_pair.second; 
  }

  int getNindices(const modeIndexSet &left_coord, const modeIndexSet &right_coord) const{
    return _getNindices<DepthLeftDilution,DepthRightDilution>::doit(left_coord, right_coord, mode_map);
  }

};

CPS_END_NAMESPACE

#endif
