#ifndef _UTILS_ARRAY_H_
#define _UTILS_ARRAY_H_

#include <vector>
#include <cassert>
#include <util/gjp.h>

//Utilities for arrays

CPS_START_NAMESPACE

//out[i] = a[i] && b[i]
inline void compute_overlap(std::vector<bool> &out, const std::vector<bool> &a, const std::vector<bool> &b){
  assert(a.size()==b.size());
  out.resize(a.size());
  for(int i=0;i<a.size();i++) out[i] = a[i] && b[i];
}


//Look for contiguous blocks of indices in the idx_map, output a list of start,size pairs
inline void find_contiguous_blocks(std::vector<std::pair<int,int> > &blocks, const int idx_map[], int map_size){
  blocks.resize(0);
  std::pair<int,int> block(0,1); //start, size
  int prev = idx_map[0];
  for(int j_packed=1;j_packed<map_size;j_packed++){
    int j_unpacked = idx_map[j_packed];
    if(j_unpacked == prev+1){
      ++block.second;
    }else{
      blocks.push_back(block);
      block.first = j_packed;
      block.second = 1;      
    }
    prev = j_unpacked;
  }
  blocks.push_back(block);

  int sum = 0;
  for(int b=0;b<blocks.size();b++){
    //printf("Block %d, start %d, size %d\n",b,blocks[b].first,blocks[b].second);
    sum += blocks[b].second;
  }
  if(sum != map_size)
    ERR.General("find_contiguous_blocks","","Sum of block sizes %d, expect %d\n",sum,map_size);
}

//Vector resize
template<typename T>
inline void resize_2d(std::vector<std::vector<T> > &v, const size_t i, const size_t j){
  v.resize(i);
  for(int a=0;a<i;a++) v[a].resize(j);
}
template<typename T>
inline void resize_3d(std::vector<std::vector<std::vector<T> > > &v, const size_t i, const size_t j, const size_t k){
  v.resize(i);
  for(int a=0;a<i;a++){
    v[a].resize(j);
    for(int b=0;b<j;b++)
      v[a][b].resize(k);
  }
}


CPS_END_NAMESPACE

#endif
