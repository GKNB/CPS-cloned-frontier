#ifndef _PIPI_MFCONTAINER_H
#define _PIPI_MFCONTAINER_H

#include "mesonfield.h"

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
  typedef typename MapType::iterator iterator;
  typedef typename MapType::const_iterator const_iterator;

  //Use these iterators at your own risk!
  iterator begin(){ return mf.begin(); }  
  iterator end(){ return mf.end(); }

  const_iterator begin() const{ return mf.begin(); }  
  const_iterator end() const{ return mf.end(); }
  

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

  void getMomenta(std::vector<ThreeMomentum> &mom){
    mom.resize(0);
    for(typename MapType::const_iterator it = mf.begin(); it != mf.end(); it++)
      mom.push_back(it->first);
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
  
  void freeMesonFieldMem(){
    for(typename MapType::iterator it = mf.begin(); it != mf.end(); it++){
      if(it->second != NULL){
	int n = it->second->size();
	for(int i=0;i<n;i++) it->second->at(i).free_mem();
      }
    }
  }

  void write(const std::string &file_stub, const bool redistribute){
    for(typename MapType::iterator it = mf.begin(); it != mf.end(); it++){
      if(it->second != NULL){
	std::ostringstream f;  f<<file_stub << "_mom" << it->first.file_str() << ".dat";
#ifdef NODE_DISTRIBUTE_MESONFIELDS
	nodeGetMany(1,it->second);
#endif
	MfType::write(f.str(), *it->second);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
	if(redistribute) nodeDistributeMany(1,it->second);
#endif
      }
    }
  }

  //Store the meson fields to disk and free their memory. Intended for temporary storage
  void dumpToDiskAndFree(const std::string &file_stub){
    double time = -dclock();
    for(typename MapType::iterator it = mf.begin(); it != mf.end(); it++){
      if(it->second != NULL){
	std::ostringstream f;  f<<file_stub << "_mom" << it->first.file_str() << ".dat";
#ifdef NODE_DISTRIBUTE_MESONFIELDS
	nodeGetMany(1,it->second);
#endif
	MfType::write(f.str(), *it->second);
	int n = it->second->size();
	for(int i=0;i<n;i++) it->second->at(i).free_mem();
      }
    }
    cps::sync();
    print_time("MesonFieldMomentumContainer","dumpToDiskAndFree",time+dclock());
  }
  void restoreFromDisk(const std::string &file_stub, bool distribute, bool do_delete = true){
    double time = -dclock();
    for(typename MapType::iterator it = mf.begin(); it != mf.end(); it++){
      if(it->second != NULL){
	std::ostringstream f;  f<<file_stub << "_mom" << it->first.file_str() << ".dat";
	MfType::read(f.str(), *it->second);

	if(distribute) nodeDistributeMany(1,it->second);

	if(do_delete && !UniqueID())
	  if(remove(f.str().c_str())){
	    std::perror("Error deleting file");
	    ERR.General("MesonFieldMomentumContainer","Restore from disk", "Could not delete file %s",f.str().c_str());
	  }
      }
    }
    cps::sync();
    print_time("MesonFieldMomentumContainer","restoreFromDisk",time+dclock());
  }


  ~MesonFieldMomentumContainer(){
    free_mem();
  }

#ifdef MESONFIELD_USE_DISTRIBUTED_STORAGE
  
  void rebalance(){
    std::vector<DistributedMemoryStorage*> ptrs;
    for(typename MapType::iterator it = mf.begin(); it != mf.end(); it++){
      if(it->second == NULL) continue; //fixes segfault ~dsh
      for(int t=0;t<it->second->size();t++){
	ptrs.push_back(dynamic_cast<DistributedMemoryStorage*>(&it->second->operator[](t)));	
      }
    }
    DistributedMemoryStorage::rebalance(ptrs);
  }


#endif
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

  inline void free_mem(){
    for(typename MapType::iterator it = mf.begin(); it != mf.end(); it++){
      if(it->second != NULL){
	delete it->second;
	it->second = NULL;
      }
    }
  }

  ~MesonFieldMomentumPairContainer(){
    for(typename MapType::iterator it = mf.begin(); it != mf.end(); it++) delete it->second;
  }
};



CPS_END_NAMESPACE
#endif

