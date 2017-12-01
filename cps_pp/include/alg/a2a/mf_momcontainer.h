#ifndef _PIPI_MFCONTAINER_H
#define _PIPI_MFCONTAINER_H

#include<alg/a2a/threemomentum.h>
#include<alg/a2a/a2a.h>
#include<alg/a2a/mesonfield.h>

CPS_START_NAMESPACE

//We must construct meson fields with a number of different total momenta. This class holds the fields and allows access in a flexible and transparent manner
//The ThreeMomentum is the total meson momentum
//The class owns the meson fields it stores, and they are deleted when it is destroyed
template<typename mf_Policies>
class MesonFieldMomentumContainer{
private:
  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> MfType;
  typedef std::map<ThreeMomentum, std::vector<MfType >* > MapType; //vector is the time index of the meson field
  MapType mf; //store pointers so we don't have to copy
  
public:
  std::vector<MfType > const* getPtr(const ThreeMomentum &p) const{
    typename MapType::const_iterator it = mf.find(p);
    if(it == mf.end()) return NULL;
    else return it->second;
  }
  const std::vector<MfType >& get(const ThreeMomentum &p) const{
    typename MapType::const_iterator it = mf.find(p);
    if(it == mf.end()){
      std::cout << "MesonFieldMomentumContainer::get Cannot find meson field with ThreeMomentum " << p.str() << std::endl; std::cout.flush();
      exit(-1);
    }      
    else return *it->second;
  }
  std::vector<MfType >& get(const ThreeMomentum &p){
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

  std::vector<MfType >& copyAdd(const ThreeMomentum &p, const std::vector<MfType > &mfield){   
    mf[p] = new std::vector<MfType >(mfield);
    return *mf[p];
  }
  std::vector<MfType >& moveAdd(const ThreeMomentum &p, std::vector<MfType > &mfield){
    mf[p] = new std::vector<MfType >(mfield.size());
    for(int i=0;i<mfield.size();i++) mf[p]->operator[](i).move(mfield[i]);
    std::vector<MfType >().swap(mfield);
    return *mf[p];
  }
  
  bool contains(const ThreeMomentum &p) const{ return mf.count(p) != 0; }

  void average(MesonFieldMomentumContainer<mf_Policies> &r){
    if(!UniqueID()) printf("MesonFieldMomentumContainer::average called\n");
    double time = -dclock();
    for(typename MapType::iterator it = mf.begin(); it != mf.end(); it++){
      assert(r.contains(it->first));
      std::vector<MfType >&rmf = r.get(it->first);
      std::vector<MfType >&lmf = *it->second;

      bool redist_l = false;
      bool redist_r = false;
      if(!mesonFieldsOnNode(lmf)){
	nodeGetMany(1,&lmf);
	redist_l = true;
      }
      if(!mesonFieldsOnNode(rmf)){
	nodeGetMany(1,&rmf);
	redist_r = true;
      }
      
      for(int t=0;t<lmf.size();t++) lmf[t].average(rmf[t]);

      if(redist_l) nodeDistributeMany(1,&lmf);
      if(redist_r) nodeDistributeMany(1,&rmf);
    }
    print_time("MesonFieldMomentumContainer","average",time + dclock());
  }
  
  inline void free_mem(){
    for(typename MapType::iterator it = mf.begin(); it != mf.end(); it++){
      if(it->second != NULL){
	delete it->second;
	it->second = NULL;
      }
    }
  }
  
  ~MesonFieldMomentumContainer(){
    free_mem();
  }
};




//In some cases we are interested in storing multiple momentum combinations with the same total momentum; use this container for easy storage of such
template<typename mf_Policies>
class MesonFieldMomentumPairContainer{
  typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> MfType;
  typedef std::pair<ThreeMomentum,ThreeMomentum> MomentumPair;
  typedef std::map<MomentumPair, std::vector<MfType >* > MapType; //vector is the time index of the meson field
  MapType mf; 
  
public:
  std::vector<MfType > const* getPtr(const ThreeMomentum &p1, const ThreeMomentum &p2) const{
    typename MapType::const_iterator it = mf.find(MomentumPair(p1,p2));
    if(it == mf.end()) return NULL;
    else return it->second;
  }
  const std::vector<MfType >& get(const ThreeMomentum &p1, const ThreeMomentum &p2) const{
    typename MapType::const_iterator it = mf.find(MomentumPair(p1,p2));
    if(it == mf.end()){
      std::cout << "MesonFieldMomentumContainer::get Cannot find meson field with ThreeMomenta " << p1.str() << " + " << p2.str() << std::endl; std::cout.flush();
      exit(-1);
    }      
    else return *it->second;
  }
  std::vector<MfType >& get(const ThreeMomentum &p1, const ThreeMomentum &p2){
    typename MapType::iterator it = mf.find(MomentumPair(p1,p2));
    if(it == mf.end()){
      std::cout << "MesonFieldMomentumContainer::get Cannot find meson field with ThreeMomenta " << p1.str() << " + " << p2.str() << std::endl; std::cout.flush();
      exit(-1);
    }
    else return *it->second;
  }

  void printMomenta(std::ostream &os) const{
    for(typename MapType::const_iterator it = mf.begin(); it != mf.end(); it++)
      os << it->first.first.str() << " + " << it->first.second.str() << "\n";
  }

  std::vector<MfType >& copyAdd(const ThreeMomentum &p1, const ThreeMomentum &p2, const std::vector<MfType > &mfield){
    MomentumPair p(p1,p2);
    mf[p] = new std::vector<MfType >(mfield);
    return *mf[p];
  }
  std::vector<MfType >& moveAdd(const ThreeMomentum &p1, const ThreeMomentum &p2, std::vector<MfType > &mfield){
    MomentumPair p(p1,p2);
    mf[p] = new std::vector<MfType >(mfield.size());
    for(int i=0;i<mfield.size();i++) mf[p]->operator[](i).move(mfield[i]);
    std::vector<MfType >().swap(mfield);
    return *mf[p];
  }
  
  bool contains(const ThreeMomentum &p1, const ThreeMomentum &p2) const{ return mf.count(MomentumPair(p1,p2)) != 0; }

  ~MesonFieldMomentumPairContainer(){
    for(typename MapType::iterator it = mf.begin(); it != mf.end(); it++) delete it->second;
  }
};



CPS_END_NAMESPACE
#endif

