#ifndef _PIPI_MFCONTAINER_H
#define _PIPI_MFCONTAINER_H

#include<alg/a2a/threemomentum.h>
CPS_START_NAMESPACE

//We must construct meson fields with a number of different pion momenta. This class holds the fields and allows access in a flexible and transparent manner
//The ThreeMomentum is the pion momentum
template<typename mf_Policies>
class MesonFieldMomentumContainer{
private:
  typedef std::map<ThreeMomentum, std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> >* > MapType; //vector is the time index of the meson field
  MapType mf; //store pointers so we don't have to copy
  
public:
  std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > const* getPtr(const ThreeMomentum &p) const{
    typename MapType::const_iterator it = mf.find(p);
    if(it == mf.end()) return NULL;
    else return it->second;
  }
  const std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> >& get(const ThreeMomentum &p) const{
    typename MapType::const_iterator it = mf.find(p);
    if(it == mf.end()){
      std::cout << "MesonFieldMomentumContainer::get Cannot find meson field with ThreeMomentum " << p.str() << std::endl; std::cout.flush();
      exit(-1);
    }      
    else return *it->second;
  }
  std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> >& get(const ThreeMomentum &p){
    typename MapType::iterator it = mf.find(p);
    if(it == mf.end()){
      std::cout << "MesonFieldMomentumContainer::get Cannot find meson field with ThreeMomentum " << p.str() << std::endl; std::cout.flush();
      exit(-1);
    }
    else return *it->second;
  }

  void printMomenta(std::ostream &os) const{
    for(typename MapType::const_iterator it = mf.begin(); it != mf.end(); it++)
      os << it->first.str() << "\n";
  }

  void add(const ThreeMomentum &p, std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mfield){
    mf[p] = &mfield;
  }
  bool contains(const ThreeMomentum &p) const{ return mf.count(p) != 0; }
};


CPS_END_NAMESPACE
#endif

