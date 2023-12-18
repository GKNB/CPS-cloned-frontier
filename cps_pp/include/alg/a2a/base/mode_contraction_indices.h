#ifndef _MODE_CONTRACTION_INDICES_H
#define _MODE_CONTRACTION_INDICES_H

//#include "mode_mapping.h"

CPS_START_NAMESPACE

//When we contract over modes between two meson fields with different numbers of row/column modes (i.e. because one is packed in time) we need to figure out the mapping between the two
//This class computes and stores the unpacked indices making the contraction transparent. It is a simple wrapper around ModeMapping

//Predefine meson field
template<typename mf_Policies,template <typename> class A2AfieldL,template <typename> class A2AfieldR>
class A2AmesonField;

typedef std::vector<std::pair<int,int>> ModeMapType; 

template<typename LeftDilutionType, typename RightDilutionType>
class ModeContractionIndices{
  size_t lidx_sizes[3];
  size_t ridx_sizes[3];
  std::vector<ModeMapType> mode_map_tensor;
  
 public:
  ModeContractionIndices() = default;

  ModeContractionIndices(const A2Aparams &a2a_params){
    compute(a2a_params);
  }
  
  inline size_t lmap(const size_t t, const size_t f, const size_t sc) const{
    return t + lidx_sizes[0]*( f + lidx_sizes[1]*sc );
  }
  inline size_t rmap(const size_t t, const size_t f, const size_t sc) const{
    return t + ridx_sizes[0]*( f + ridx_sizes[1]*sc );
  }
  inline size_t lrmap(const size_t tl, const size_t fl, const size_t scl, const size_t tr, const size_t fr, const size_t scr) const{
    return tl + lidx_sizes[0]*( fl + lidx_sizes[1]*(scl + lidx_sizes[2]*( tr + ridx_sizes[0]*( fr + ridx_sizes[1]*scr) ) ) );
  }

  void compute(const A2Aparams &a2a_params){
    lidx_sizes[0] = LeftDilutionType::isPacked(0) ? a2a_params.getNtBlocks() : 1;
    lidx_sizes[1] = LeftDilutionType::isPacked(1) ? a2a_params.getNflavors() : 1;
    lidx_sizes[2] = LeftDilutionType::isPacked(2) ? 12 : 1;

    ridx_sizes[0] = RightDilutionType::isPacked(0) ? a2a_params.getNtBlocks() : 1;
    ridx_sizes[1] = RightDilutionType::isPacked(1) ? a2a_params.getNflavors() : 1;
    ridx_sizes[2] = RightDilutionType::isPacked(2) ? 12 : 1;
    
    LeftDilutionType ldil(a2a_params);
    RightDilutionType rdil(a2a_params);

    size_t lvsize = lidx_sizes[0]*lidx_sizes[1]*lidx_sizes[2];
    size_t rvsize = ridx_sizes[0]*ridx_sizes[1]*ridx_sizes[2];
    size_t tsize = lvsize*rvsize;
    
    size_t nv = a2a_params.getNv();

    typedef std::pair< std::vector<int>, std::vector<bool> > Velem;
    std::vector<Velem> lv(lvsize), rv(rvsize);
    modeIndexSet ml, mr;
    for(ml.time =0; ml.time < lidx_sizes[0]; ml.time++){
    for(ml.flavor =0; ml.flavor < lidx_sizes[1]; ml.flavor++){
    for(ml.spin_color =0; ml.spin_color < lidx_sizes[2]; ml.spin_color++){
      Velem &lelem = lv[lmap(ml.time,ml.flavor,ml.spin_color)];
      ldil.getIndexMapping(lelem.first,lelem.second,ml);
    }}}
    for(mr.time =0; mr.time < ridx_sizes[0]; mr.time++){
    for(mr.flavor =0; mr.flavor < ridx_sizes[1]; mr.flavor++){
    for(mr.spin_color =0; mr.spin_color < ridx_sizes[2]; mr.spin_color++){
      Velem &relem = rv[rmap(mr.time,mr.flavor,mr.spin_color)];
      rdil.getIndexMapping(relem.first,relem.second,mr);
    }}}

    mode_map_tensor.resize(tsize);
    for(ml.time =0; ml.time < lidx_sizes[0]; ml.time++){
    for(ml.flavor =0; ml.flavor < lidx_sizes[1]; ml.flavor++){
    for(ml.spin_color =0; ml.spin_color < lidx_sizes[2]; ml.spin_color++){
      const Velem &lelem = lv[lmap(ml.time,ml.flavor,ml.spin_color)];

      for(mr.time =0; mr.time < ridx_sizes[0]; mr.time++){
      for(mr.flavor =0; mr.flavor < ridx_sizes[1]; mr.flavor++){
      for(mr.spin_color =0; mr.spin_color < ridx_sizes[2]; mr.spin_color++){
        const Velem &relem = rv[rmap(mr.time,mr.flavor,mr.spin_color)];
  
	ModeMapType &lrelem = mode_map_tensor[lrmap(ml.time,ml.flavor,ml.spin_color,mr.time,mr.flavor,mr.spin_color)];
	for(int i=0;i<nv;i++)
	  if(lelem.second[i] && relem.second[i]) lrelem.push_back(std::pair<int,int>(lelem.first[i],relem.first[i])); //record l,r indices of elements that match up
    
    }}}}}}
  };

  //Return the tensor. Use lrmap to get the mapping for particular elements
  inline const std::vector<ModeMapType> & getModeMapTensor() const{ return mode_map_tensor; }

  inline size_t tensorSize() const{ return mode_map_tensor.size(); }

  //Get the vector of matching indices (i,j)
  inline const ModeMapType & getIndexVector(const modeIndexSet &left_coord, const modeIndexSet &right_coord) const{
    size_t tl = left_coord.time % lidx_sizes[0]; //gives 0 if not packed
    size_t fl = left_coord.flavor % lidx_sizes[1];
    size_t scl = left_coord.spin_color % lidx_sizes[2];
    size_t tr = right_coord.time % ridx_sizes[0];
    size_t fr = right_coord.flavor % ridx_sizes[1];
    size_t scr = right_coord.spin_color % ridx_sizes[2];

    return mode_map_tensor[ lrmap(tl,fl,scl,tr,fr,scr) ];
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
    return getIndexVector(left_coord,right_coord).size();
  }

};

CPS_END_NAMESPACE

#endif
