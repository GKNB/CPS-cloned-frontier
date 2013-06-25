CPS_START_NAMESPACE
#ifndef PROPAGATOR_CONTAINER_H
#define PROPAGATOR_CONTAINER_H
CPS_END_NAMESPACE

#include<vector>
#include<config.h>
#include <alg/qpropw.h>
#include <alg/prop_attribute_arg.h>

CPS_START_NAMESPACE

class PropagatorContainer{
protected:
  QPropW *prop;
  AttributeContainer* attributes[50]; //positions are mapped by the integer values of the enum  AttrType with no duplicate entries
  AttributeContainer* findAttr(const AttrType &type) const; //returns NULL if not present
public:
  void add(const AttributeContainer &p);
  void setup(PropagatorArg &arg);
  template<typename T>
  bool getAttr(T* &to) const;
  
  void readProp(Lattice &latt); //loads prop if on disk, does nothing otherwise
  void calcProp(Lattice &latt); //perform prop inversion
  void deleteProp(); //deletes the prop from memory, used to save space. Will be automatically recalculated if getProp is called again

  QPropW & getProp(Lattice &latt); //get the prop, calculate or load if necessary

  bool tagEquals(const char* what); //check if tag is equal to input

  void momentum(int *into) const; //get the quark 3-momentum
  int flavor() const; //get the quark flavor
  char const* tag() const;

  void printAttribs() const;

  //for props that are formed by combining other props, copy over the attributes
  void propCombSetupAttrib();
  std::vector<std::vector<int> > get_allowed_momenta() const;

  PropagatorContainer();
  ~PropagatorContainer();
};
template<typename T>
bool PropagatorContainer::getAttr(T* &to) const{
  AttributeContainer *c = findAttr(T::getType());
  if(c==NULL) return false;
  to = reinterpret_cast<T*>(&(c->AttributeContainer_u)); //C99 standard: A pointer to a union object, suitably converted, points to each of its members
  return true;
}

#endif
CPS_END_NAMESPACE
