#ifndef _A2A_DILUTIONS_H
#define _A2A_DILUTIONS_H

#include<cassert>
#include<vector>

#include<alg/a2a/lattice/fmatrix.h>
#include "a2a_params.h"

CPS_START_NAMESPACE

/*
----------Dilution and packing--------------
In the A2A world we have a lot of vector like objects that are delta functions in one or more indices.
We have 3 types of coordinate that a vector might be a delta function in:  spin_color (these are treated always together),  flavor and time
In order to save memory, rather than storing a lot of zeroes we need only keep the non-zero entries.

***************
Simple vectors:
***************
Let us consider a flavor-time vector v_{f,t} of size nf*Lt, that is actually a delta function in time,   v_{f,t} = x_f\delta_{t,t'}

In order to save memory we need only store the nf non-zero elements x_f (and the value of t'), thus reducing the memory requirement by a factor of Lt

The resulting "packed" vector has size nf. Given that the time index was removed, we refer to this as "time packed". 
* The remaining index, flavor here, is referred to as a "diluted" index.
* A packed index may also be referred to as an "undiluted" index.

For our purposes we need conside the following 4 classes of "dilution" or "packing":
StandardIndexDilution -           packed: none                        diluted:  spin_color, flavor, time  (hit)
TimePackedIndexDilution -         packed: time                        diluted:  spin_color, flavor  (hit)
TimeFlavorPackedIndexDilution -   packed: flavor, time                diluted:  spin_color  (hit)
FullyPackedIndexDilution -        packed: spin_color, flavor, time    diluted: (hit)


*************
Vectors of fields
*************
Let us now consider a more complex object, a vector of nf*Lt (flavor) fields v^(f',t')_{f}(x,t)
where the superscript in parentheses indexes the fields

If v satisfies the following form:

v^(f',t')_{f}(x,t) = c^(f',t')_f(x) \delta{t', t}

the only information we are required to store are  c^(f',t')_f(x), again reducing the memory requirement by  a factor of Lt

Note that rather than reducing the dimension of the fields, for convenience we will instead reduce the *number of fields*
as follows:

w^(f')_{f}(x,t') = c^(f',t')_f(x)

where t' is now the time coordinate of the fields. Thus for vectors of fields, the superscript becomes a packed index.

In this library we will typically combine the multi-dimensional superscript indices lexicographically 
A compound index for a fully unpacked vector coordinate is referred to as a "full" or perhaps "unpacked" index
A compound index corresponding to a coordinate of a packed vector is referred to as a "packed" index

**************
V and W vectors
**************
We use these packing strategies to reduce the number of "high mode" fields that need to be stored for the A2A vectors.

Start with the W-vector high mode fields,   w^(i)_j(x)
Here the superscript (i) is the high mode index,  j is the spin/color/flavor index and x the space/time index of the field element

The superscript high-mode index implicitly contains the index of the non-zero row for each dimension, the equivalent of t' in our flavor-time example above

Let the indices 'unmap' as follows (i) -> (i_sc, i_f, i_t, i_h)   and   j -> {j_sc, j_f}   x->{x_3d, x_t}     where 'sc'=spin_color,  'f'=flavor,  't'=time   'h'=hit

This library assumes full dilution, i.e. that the high modes are delta functions in all indices (although the "time block" size can be changed to Lt to dynamically remove the time dilution).

Thus:
w^(i)_j(x) = \eta^{i_h}(x_3d) \delta_{i_f,j_f}\delta_{i_sc, j_sc}\delta_{i_t, x_t}

where \eta^{i_h}(x_3d) is a 3d complex field

Thus we need only store n_hit 3d complex fields \eta^{i_h}(x_3d)  to reconstruct this entire massive object, which we store as above as 4d fields

w'^(i_h)(x_3d, i_t) =  \eta^{i_h}(x_3d)

The packed index is just the hit index, therefore the "dilution" strategy is FullyPackedIndexDilution

Note that this object has no spin,color or flavor indices; it is just a complex field. 

Note also that in principle this results in a copy of the same complex field on each timeslice. However for convenience in implementation 
we actually allow \eta to vary between timeslices (and also between flavors)



The V-vector high mode fields are solution vectors and therefore do not have a delta-function structure. Therefore both V and Vfftw are indexed as StandardIndexDilution 

Finally when constructing the Wfftw fields we must first unpack the spin_color index and multiply by the gauge-fixing matrices prior to performing the FFT. Thus 

w_FFT^(i)_{sc,f}(p,t) = \rho^(i_sc,i_h)_{sc}(p) \delta_{t,i_t} \delta_{f,i_f}

which can be stored as

w_FFT'^(i_sc,i_h)_{sc}(p,i_t) = \rho^(i_sc,i_h)_{sc}(p)

where the superscript has TimeFlavorPackedIndexDilution.  Again in implementation we make a rho also a function of i_t (and i_f)


**************
Meson fields
**************

Meson fields are matrices formed (in the typical case) as

M^(i,j)(t) = \sum_{p,sc,f,sc',f'} v_FFT^(i)_{sc,f}(p,t) F_{sc,f; sc', f'}(p)  w_FFT^\dagger (j)_{sc',f'}(p,t)   for some 3d spin-color-flavor matrix field  F_{sc,f; sc', f'}(p)
           = \sum_{p,sc,f,sc',f'} v_FFT^(i)_{sc,f}(p,t) F_{sc,f; sc', f'}(p)  \rho^(j_sc,j_h)_{sc'}(p) \delta_{t,j_t} \delta_{f',j_f}
           = \sum_{p,sc,f,sc'} v_FFT^(i)_{sc,f}(p,t) F_{sc,f; sc', j_f}(p) \rho^(j_sc,j_h)_{sc'}(p)  \delta_{t,j_t}

The non-zero terms have t==j_t hence we can store all the non-zero matrix elements in a compact form

M'^(i, j_sc,j_h)(j_t) = \sum_{p,sc,f,sc'} v_FFT^(i)_{sc,f}(p,t) F_{sc,f; sc', j_f}(p) \rho^(j_sc,j_h)_{sc'}(p)
                      = \sum_{p,sc,f,sc'} v_FFT^(i)_{sc,f}(p,t) F_{sc,f; sc', j_f}(p) w_FFT'^(j_sc,j_h)_{sc}(p,j_t)

we see that by storing the meson fields and w_FFT as specified above, we can in practise largely forget about the fact that the indices are compressed


----------------------------------------------------------------------------------------------------------------------------------

NOTE: As it stands all dilutions do not contain any information other than the A2Aparams which is the same for all dilutions. This allows us to create instances of one dilution from another or from an A2Aparams alone without propagating extra information.

NOTE: In this file the index dilutions will be treated as those associated with 'simple vectors' above rather than adding in all the complications associated with the a2a field indices
*/


struct modeIndexSet{
  int hit;
  int spin_color;
  int flavor;
  int time;
  modeIndexSet(): hit(-1),spin_color(-1),flavor(-1),time(-1){}
};

//Various stages of high-mode dilution. Undiluted indices are referred to as 'packed'
//For vectors V_{sc, f, t, h}  = v_{sc, f, t, h}
class StandardIndexDilution: public A2Aparams{
public:
  enum { UndilutedIndices = 0 };

  StandardIndexDilution(): A2Aparams(){}
  StandardIndexDilution(const A2AArg &_args): A2Aparams(_args){}
  StandardIndexDilution(const A2Aparams &_p): A2Aparams(_p){}

  //Map dilution indices to a 'mode' index for the high-mode part of the propagator
  //This is a combination of the hit index, timeslice-block index and spin,color (and flavor) indices
  inline int indexMap(const int hit, const int tblock, const int spin_color, const int flavor = 0) const{
    return spin_color + nspincolor*( flavor + nflavors*( tblock + ntblocks * hit) );
  }
  inline void indexUnmap(int idx, int &hit, int &tblock, int &spin_color, int &flavor) const{
    spin_color = idx % nspincolor; idx/=nspincolor;
    flavor = idx % nflavors; idx/=nflavors;
    tblock = idx % ntblocks; idx/=ntblocks;
    hit = idx;
  }
  inline int indexMap(const modeIndexSet &mset) const{
    return indexMap(mset.hit, mset.time, mset.spin_color, mset.flavor);
  }
  inline void indexUnmap(int idx,modeIndexSet &mset) const{
    return indexUnmap(idx,mset.hit, mset.time, mset.spin_color, mset.flavor);
  }
  inline int getNmodes() const{ return nv; }

  //Compute the mapping between a full (unpacked) index and the packed index. Only the elements map[i] for which non_zeroes[i]==true are meaningful
  //coord_delta is ignored here
  void getIndexMapping(std::vector<int> &map, std::vector<bool> &non_zeroes, const modeIndexSet &coord_delta) const{
    non_zeroes.resize(nv,true);
    map.resize(nv); for(int i=0;i<nv;i++) map[i] = i; //trivial mapping
  }

  static std::string name(){ return "StandardIndexDilution"; }
};

//An index containing only the hit, flavor and spin-color indices
//For vectors   V^(t_delta)_{sc, f, t, h} = v_{sc, f, h} \delta_{t, t_delta}    [t_delta stored separately]
class TimePackedIndexDilution: public A2Aparams{

public:
  enum { UndilutedIndices = 1 };

  TimePackedIndexDilution(): A2Aparams(){}
  TimePackedIndexDilution(const A2AArg &_args): A2Aparams(_args){}
  TimePackedIndexDilution(const A2Aparams &_p): A2Aparams(_p){}

  inline int indexMap(const int hit, const int spin_color, const int flavor = 0) const{
    return spin_color + nspincolor * (flavor + nflavors*hit);
  }
  inline void indexUnmap(int idx, int &hit, int &spin_color, int &flavor) const{
    spin_color = idx % nspincolor; idx/=nspincolor;
    flavor = idx % nflavors; idx/=nflavors;
    hit = idx;
  }
  inline int indexMap(const modeIndexSet &mset) const{
    return indexMap(mset.hit, mset.spin_color, mset.flavor);
  }
  inline void indexUnmap(int idx,modeIndexSet &mset) const{
    return indexUnmap(idx,mset.hit, mset.spin_color, mset.flavor);
  }
  inline int getNmodes() const{ return nl + nhits*nspincolor*nflavors; }

  //Compute the mapping between a full (unpacked) index and the packed index. Only the elements map[i] for which non_zeroes[i]==true are meaningful
  //t_delta is defined above
  void getIndexMapping(std::vector<int> &map, std::vector<bool> &non_zeroes, int t_delta) const{
    non_zeroes.resize(nv,false); map.resize(nv);
    for(int i=0;i<nl;i++){ non_zeroes[i] = true; map[i] = i; } //low modes mapping trivial
    
    const StandardIndexDilution &unpacked = static_cast< const StandardIndexDilution &>(*this);

    //Loop over packed modes and fill gaps
    for(int p = 0; p < nhits*nspincolor*nflavors; p++){
      modeIndexSet pparams;  indexUnmap(p, pparams); pparams.time = t_delta;
      int idx_packed = nl + p;
      int idx_unpacked = nl + unpacked.indexMap(pparams);
      non_zeroes[idx_unpacked] = true;
      map[idx_unpacked] = idx_packed;
    }
  }

  //Compute the mapping between a full (unpacked) index and the packed index. Only the elements map[i] for which non_zeroes[i]==true are meaningful
  //t_delta is passed in as the .time element of the modeIndexSet coord_delta  (other indices ignored)
  inline void getIndexMapping(std::vector<int> &map, std::vector<bool> &non_zeroes, const modeIndexSet &coord_delta) const{
    assert(coord_delta.time != -1);
    getIndexMapping(map, non_zeroes, coord_delta.time);
  }
  static std::string name(){ return "TimePackedIndexDilution"; }
};

//An index containing only the hit and spin-color indices
//For vectors   V_{sc, f, t, h} = v_{sc, h} \delta_{f,f_delta}\delta_{t, t_delta}    [f_delta,t_delta stored separately]
class TimeFlavorPackedIndexDilution: public A2Aparams{

public:
  enum { UndilutedIndices = 2 };

  TimeFlavorPackedIndexDilution(): A2Aparams(){}
  TimeFlavorPackedIndexDilution(const A2AArg &_args): A2Aparams(_args){}
  TimeFlavorPackedIndexDilution(const A2Aparams &_p): A2Aparams(_p){}

  //Mapping used by W_fftw for high 'modes', where only the spin/color index has been diluted out
  inline int indexMap(const int hit, const int spin_color) const{
    return spin_color + nspincolor * hit;
  }
  inline void indexUnmap(int idx, int &hit, int &spin_color) const{
    spin_color = idx % nspincolor; idx/=nspincolor;
    hit = idx;
  }
  inline int indexMap(const modeIndexSet &mset) const{
    return indexMap(mset.hit, mset.spin_color);
  }
  inline void indexUnmap(int idx,modeIndexSet &mset) const{
    return indexUnmap(idx,mset.hit, mset.spin_color);
  }
  inline int getNmodes() const{ return nl + nhits*nspincolor; }

  //Compute the mapping between a full (unpacked) index and the packed index. Only the elements map[i] for which non_zeroes[i]==true are meaningful
  //f_delta, t_delta are defined above
  void getIndexMapping(std::vector<int> &map, std::vector<bool> &non_zeroes, const int f_delta, const int t_delta) const{
    non_zeroes.resize(nv,false); map.resize(nv);
    for(int i=0;i<nl;i++){ non_zeroes[i] = true; map[i] = i; } //low modes mapping trivial
    
    const StandardIndexDilution &unpacked = static_cast< const StandardIndexDilution &>(*this);

    //Loop over packed modes and fill gaps
    for(int p = 0; p < nhits*nspincolor; p++){
      modeIndexSet pparams;  indexUnmap(p, pparams); pparams.time = t_delta; pparams.flavor = f_delta; 
      int idx_packed = nl + p;
      int idx_unpacked = nl + unpacked.indexMap(pparams);
      non_zeroes[idx_unpacked] = true;
      map[idx_unpacked] = idx_packed;
    }
  }

  //Compute the mapping between a full (unpacked) index and the packed index. Only the elements map[i] for which non_zeroes[i]==true are meaningful
  //f_delta, t_delta is passed in as the .flavor and .time elements of the modeIndexSet coord_delta  (other indices ignored)
  inline void getIndexMapping(std::vector<int> &map, std::vector<bool> &non_zeroes, const modeIndexSet &coord_delta) const{
    assert(coord_delta.time != -1 && coord_delta.flavor != -1);
    getIndexMapping(map, non_zeroes, coord_delta.flavor, coord_delta.time);
  }
  static std::string name(){ return "TimeFlavorPackedIndexDilution"; }
};


//An index containing only the hit index
//For vectors   V_{sc, f, t, h} = v_{h} \delta_{sc,sc_delta}\delta_{f,f_delta}\delta_{t, t_delta}    [sc_delta,f_delta,t_delta stored separately]
class FullyPackedIndexDilution: public A2Aparams{
public:
  enum { UndilutedIndices = 3 };

  FullyPackedIndexDilution(): A2Aparams(){}
  FullyPackedIndexDilution(const A2AArg &_args): A2Aparams(_args){}
  FullyPackedIndexDilution(const A2Aparams &_p): A2Aparams(_p){}

  //Mapping used by W_fftw for high 'modes', where only the spin/color index has been diluted out
  inline int indexMap(const int hit) const{
    return hit;
  }
  inline void indexUnmap(int idx, int &hit) const{
    hit = idx;
  }
  inline int indexMap(const modeIndexSet &mset) const{
    return indexMap(mset.hit);
  }
  inline void indexUnmap(int idx,modeIndexSet &mset) const{
    return indexUnmap(idx,mset.hit);
  }

  inline int getNmodes() const{ return nl + nhits*nspincolor; }

  //Compute the mapping between a full (unpacked) index and the packed index. Only the elements map[i] for which non_zeroes[i]==true are meaningful
  //sc_delta, f_delta, t_delta are defined above
  void getIndexMapping(std::vector<int> &map, std::vector<bool> &non_zeroes, const int sc_delta, const int f_delta, const int t_delta) const{
    non_zeroes.resize(nv,false); map.resize(nv);
    for(int i=0;i<nl;i++){ non_zeroes[i] = true; map[i] = i; } //low modes mapping trivial
    
    const StandardIndexDilution &unpacked = static_cast< const StandardIndexDilution &>(*this);

    //Loop over packed modes and fill gaps
    for(int p = 0; p < nhits; p++){
      modeIndexSet pparams;  indexUnmap(p, pparams); pparams.time = t_delta; pparams.flavor = f_delta; pparams.spin_color = sc_delta;
      int idx_packed = nl + p;
      int idx_unpacked = nl + unpacked.indexMap(pparams);
      non_zeroes[idx_unpacked] = true;
      map[idx_unpacked] = idx_packed;
    }
  }

  //Compute the mapping between a full (unpacked) index and the packed index. Only the elements map[i] for which non_zeroes[i]==true are meaningful
  //sc_delta, f_delta, t_delta is passed in as the .spin_color, .flavor and .time elements of the modeIndexSet coord_delta  (other indices ignored)
  inline void getIndexMapping(std::vector<int> &map, std::vector<bool> &non_zeroes, const modeIndexSet &coord_delta) const{
    assert(coord_delta.time != -1 && coord_delta.flavor != -1 && coord_delta.spin_color != -1);
    getIndexMapping(map, non_zeroes, coord_delta.spin_color, coord_delta.flavor, coord_delta.time);
  }

  static std::string name(){ return "FullyPackedIndexDilution"; }
};






template<typename Dilution>
struct FlavorUnpacked{};

template<>
struct FlavorUnpacked<TimeFlavorPackedIndexDilution>{
  typedef TimePackedIndexDilution UnpackedType;
};
template<>
struct FlavorUnpacked<StandardIndexDilution>{
  typedef StandardIndexDilution UnpackedType;
};




CPS_END_NAMESPACE
#endif
