#ifndef _PROP_MOM_CONTAINER_H
#define _PROP_MOM_CONTAINER_H

#include <set>
#include "propwrapper.h"

CPS_START_NAMESPACE

class PropMomContainer{
  std::map<std::string, PropWrapper> props;
public:
  void insert(const PropWrapper &prop, const std::string tag){
    if(props.count(tag) != 0) ERR.General("PropMomContainer","insert","Attempting to insert duplicate of prop with tag '%s'\n", tag.c_str());
    props[tag] = prop;
  }
  PropWrapper & get(const std::string &tag){
    if(props.count(tag) == 0) ERR.General("PropMomContainer","get","Could not find prop with tag '%s'\n", tag.c_str());
    return props[tag];
  } 
  void clear(){
    //This container takes ownership of the memory associated with the propagators.
    //As a propagator may appear in multiple entries we must avoid double deletion
    std::set<QPropW*> deleted;
    for(std::map<std::string, PropWrapper>::iterator it = props.begin(); it != props.end(); it++){
      for(int f=0;f<1+GJP.Gparity();f++){
	QPropW* p = it->second.getPtr(f);
	if(!deleted.count(p)){ 
	  deleted.insert(p);
	  delete p;
	}
      }
    }
    props.clear();
  }
  void printAllTags() const{
    if(!UniqueID()){
      printf("Propagators stored:\n");
      for(std::map<std::string, PropWrapper>::const_iterator it = props.begin(); it != props.end(); it++)
	std::cout << it->first << '\n';
    }
  }


  //Takes ownership and deletes props when destroyed
  ~PropMomContainer(){
    clear();
  }
};



CPS_END_NAMESPACE

#endif
