#ifndef _MODE_CONTRACTION_INDICES_H
#define _MODE_CONTRACTION_INDICES_H

#include "mode_mapping.h"

CPS_START_NAMESPACE

//When we contract over modes between two meson fields with different numbers of row/column modes (i.e. because one is packed in time) we need to figure out the mapping between the two
//This class computes and stores the unpacked indices making the contraction transparent. It is a simple wrapper around ModeMapping

//Predefine meson field
template<typename mf_Float,template <typename> class A2AfieldL,template <typename> class A2AfieldR>
class A2AmesonField;


//Get an index vector
template<int LeftDilutionDepth, int RightDilutionDepth>
struct _getIndexVector{
  typedef typename IndexTensor<LeftDilutionDepth,RightDilutionDepth>::Type TensorType;
  inline static const ModeMapType & doit(const modeIndexSet &left_coord, const modeIndexSet &right_coord, const TensorType &mode_map){
    int val_left = IndexConvention<LeftDilutionDepth>::get(left_coord); 
    return _getIndexVector<LeftDilutionDepth-1,RightDilutionDepth>::doit(left_coord, right_coord,mode_map[val_left] );
  }
};
template<int RightDilutionDepth>
struct _getIndexVector<0,RightDilutionDepth>{
  typedef typename IndexTensor<0,RightDilutionDepth>::Type TensorType;
  inline static const ModeMapType & doit(const modeIndexSet &left_coord, const modeIndexSet &right_coord, const TensorType &mode_map){
    int val_right = IndexConvention<RightDilutionDepth>::get(right_coord); 
    return _getIndexVector<0,RightDilutionDepth-1>::doit(left_coord, right_coord,mode_map[val_right] );
  }
};
template<>
struct _getIndexVector<0,0>{
  typedef typename IndexTensor<0,0>::Type TensorType;
  inline static const ModeMapType & doit(const modeIndexSet &left_coord, const modeIndexSet &right_coord, const TensorType &mode_map){
    return mode_map;
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

  //Get the tensor that contains the set of matching mode indices (i,j) for each choice of field index (sc, f, t) as appropriate (cf ModeMapping)
  inline const TensorType & getIndexTensor() const{ return mode_map; }

  //Get the vector of matching indices (i,j)
  inline const ModeMapType & getIndexVector(const modeIndexSet &left_coord, const modeIndexSet &right_coord) const{
    return _getIndexVector<DepthLeftDilution,DepthRightDilution>::doit(left_coord, right_coord, mode_map);
  }
  
  inline int getLeftIndex(const int i, const modeIndexSet &left_coord, const modeIndexSet &right_coord) const{
    return getIndexVector(left_coord, right_coord)[i].first;
  }
  inline int getRightIndex(const int i, const modeIndexSet &left_coord, const modeIndexSet &right_coord) const{
    return getIndexVector(left_coord, right_coord)[i].second;
  }

  inline void getBothIndices(int &il, int &ir, const int i, const modeIndexSet &left_coord, const modeIndexSet &right_coord) const{
    const auto &p = getIndexVector(left_coord, right_coord)[i];
    il = p.first; 
    ir = p.second; 
  }

  inline int getNindices(const modeIndexSet &left_coord, const modeIndexSet &right_coord) const{
    return _getNindices<DepthLeftDilution,DepthRightDilution>::doit(left_coord, right_coord, mode_map);
  }

};

CPS_END_NAMESPACE

#endif
