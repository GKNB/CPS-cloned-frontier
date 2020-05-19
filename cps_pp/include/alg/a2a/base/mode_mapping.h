#ifndef _MODE_MAPPING_H
#define _MODE_MAPPING_H

#include "a2a_dilutions.h"
#include<alg/a2a/utils.h>

CPS_START_NAMESPACE

//As we dilute we unpack in the following order: spincolor , flavor, time. We assign these indices  spincolor=3, flavor=2, time=1
template<int TypeIndex>
struct IndexConvention{};

template<>
struct IndexConvention<3>{
  static int getNidx(){ return 12; }
  static std::string getStr(){ return "spin-color"; }
  static void set(modeIndexSet &into, const int &val){ into.spin_color = val; } 
  static const int & get(const modeIndexSet &from){ return from.spin_color; } 
};
template<>
struct IndexConvention<2>{
  static int getNidx(){ return GJP.Gparity() ? 2:1; }
  static std::string getStr(){ return "flavor"; }  
  static void set(modeIndexSet &into, const int &val){ into.flavor = val; } 
  static const int & get(const modeIndexSet &from){ return from.flavor; } 
};
template<>
struct IndexConvention<1>{
  static int getNidx(){ return GJP.Tnodes()*GJP.TnodeSites(); }
  static std::string getStr(){ return "time"; }
  static void set(modeIndexSet &into, const int &val){ into.time = val; } 
  static const int & get(const modeIndexSet &from){ return from.time; } 
};


//To store an index mapping we need a large number of matrices
typedef std::vector<std::pair<int,int> > ModeMapType;

//Unmapping a single index
template<int Depth>
struct IndexVector{
  typedef typename IndexVector<Depth-1>::Type SubType;
  typedef std::vector<SubType> Type;
};
template<>
struct IndexVector<0>{
  typedef std::pair< std::vector<int>, std::vector<bool> > Type;
};

//We want a big set of nested vectors where the first Ldepth vectors are associated with the unmappings for the left index, and the remaining Rdepth with the right index
template<int Ldepth, int Rdepth>
struct IndexTensor{
  typedef typename IndexTensor<Ldepth-1,Rdepth>::Type SubType;
  typedef std::vector<SubType> Type;
};
template<int Rdepth>
struct IndexTensor<0,Rdepth>{
  typedef typename IndexTensor<0,Rdepth-1>::Type SubType;
  typedef std::vector<SubType> Type;
};
template<>
struct IndexTensor<0,0>{
  typedef ModeMapType Type; //mode * mode
};





/*
When performing products of two vectors A and B with different packed indices, we need to work out which index of A corresponds to which index of B
For clarity let us work with specific examples, say A=FullyPackedIndexDilution and B=TimeFlavorPackedIndexDilution
Given these classifications we know
A_{sc, f, t, h} = a_h \delta_{sc, sc'_A}\delta_{f,f'_A}\delta_{t, t'_A}
B_{sc, f, t, h} = b_{sc,h} \delta_{f,f'_B}\delta_{t, t'_B}

In memory A and B are indexed by packed indices  i_A \in {0..nh}  ,  i_B \in {0..12*nh}

Let's say we aim to compute  A_{sc, f, t, h} B_{sc, f, t, h}  for all sc,f,t,h
We know that the delta-function structure highly-constrains the number of non-zero values that need to be computed

A_{sc, f, t, h} B_{sc, f, t, h} = a_h b_{sc,h} \delta_{sc, sc'_A} [\delta_{f,f'_A}\delta_{f,f'_B}] [\delta_{t, t'_B}\delta_{t, t'_A}]

Thus non-zero values only occur if      sc == sc'_A  && (f == f'_A && f == f'_B) && (t == t'_A && t == t'_B )

The ModeMapping class determines the set of pairs  (i_A, i_B) that must be computed given values of sc'_A, f'_A, t'_A  and  f'_B, t'_B

It is assumed that the dilution type specified by the left "PackedType" is more packed than the right "UnpackedType"
(Note that UnpackedType doesn't have to be fully diluted, it just has to be more or equally diluted than "PackedType"  (IS THIS ACTUALLY NECESSARY?) )

As such for the example above we will use PackedType=FullyPackedIndexDilution   UnpackedType=TimeFlavorPackedIndexDilution
The result will be an object 'idx_map' for which a vector of pairs (i_A, i_B) can be obtained as

vector<pair(i_A,i_B)> = idx_map[sc'_A][f'_A][t'_A] [f'_B][t'_B]

*/

template<typename LeftDilutionType, typename RightDilutionType>
class ModeMapping{
public:
  enum { DepthLeftDilution = LeftDilutionType::UndilutedIndices };
  enum { DepthRightDilution = RightDilutionType::UndilutedIndices };

  typedef typename IndexVector<DepthLeftDilution>::Type VectorTypeLeftDilution;
  typedef typename IndexVector<DepthRightDilution>::Type VectorTypeRightDilution;

  typedef typename IndexTensor<DepthLeftDilution, DepthRightDilution>::Type TensorType;

  static void compute(TensorType &idx_map, const A2Aparams &p);
};



#include "implementation/mode_mapping_impl.tcc"

CPS_END_NAMESPACE


#endif
