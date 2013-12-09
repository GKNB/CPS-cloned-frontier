CPS_START_NAMESPACE
#ifndef PROPAGATOR_CONTAINER_H
#define PROPAGATOR_CONTAINER_H
CPS_END_NAMESPACE

#include<vector>
#include<config.h>
#include <alg/qpropw.h>
#include <alg/lanc_arg.h>
#include <alg/prop_attribute_arg.h>
//#include <alg/eigen/Krylov_5d.h>

template<class T> class Lanczos_5d;
template<class T> class bfm_evo;

CPS_START_NAMESPACE

class PropagatorContainer{
protected:
  AttributeContainer* attributes[50]; //positions are mapped by the integer values of the enum  AttrType with no duplicate entries
public:
  void add(const AttributeContainer &p);
  void setup(PropagatorArg &arg);

  AttributeContainer* findAttr(const AttrType &type) const; //returns NULL if not present

  template<typename T>
  bool getAttr(T* &to) const;

  template<typename T>
  T* getAttr() const;

  template<typename T>
  bool hasAttr() const{ return findAttr(T::getType())!=NULL; }
  
  virtual void readProp(Lattice &latt) = 0; //loads prop if on disk (or if possible), does nothing otherwise
  virtual void calcProp(Lattice &latt) = 0; //perform prop inversion
  virtual void deleteProp() = 0; //deletes the prop from memory, used to save space. Will be automatically recalculated if getProp is called again

  PropagatorType type() const; //return the type (stored in the generic parameters)

  static PropagatorContainer* create(const PropagatorType &ctype); //Create a new container of the specified type on the stack, return pointer

  //Return a reference to this container cast to the derived class
  template<typename T>
  T & convert(){ return dynamic_cast<T&>(*this); }

  template<typename T>
  const T & convert() const{ return dynamic_cast<const T&>(*this); }

  bool tagEquals(const char* what); //check if tag is equal to input

  char const* tag() const;

  void printAttribs() const;

  PropagatorContainer();
  virtual ~PropagatorContainer();
};
template<typename T>
bool PropagatorContainer::getAttr(T* &to) const{
  AttributeContainer *c = findAttr(T::getType());
  if(c==NULL) return false;
  to = reinterpret_cast<T*>(&(c->AttributeContainer_u)); //C99 standard: A pointer to a union object, suitably converted, points to each of its members
  return true;
}
template<typename T>
T* PropagatorContainer::getAttr() const{
  AttributeContainer *c = findAttr(T::getType());
  if(c==NULL) return NULL;
  return reinterpret_cast<T*>(&(c->AttributeContainer_u));
}


class QPropWcontainer: public PropagatorContainer{
 protected:
  QPropW *prop;

 public:
  QPropWcontainer(): PropagatorContainer(), prop(NULL){}

  void readProp(Lattice &latt); //loads prop if on disk (or if possible), does nothing otherwise
  void calcProp(Lattice &latt); //perform prop inversion
  void deleteProp(); //deletes the prop from memory, used to save space. Will be automatically recalculated if getProp is called again

  //for props that are formed by combining other props, copy over the attributes
  void propCombSetupAttrib();

  //Convenience function to check and then convert a base class reference. Pass in cname and fname
  static QPropWcontainer & verify_convert(PropagatorContainer &pc, const char* cname, const char* fname);
  static const QPropWcontainer & verify_convert(const PropagatorContainer &pc, const char* cname, const char* fname);

  QPropW & getProp(Lattice &latt); //get the prop, calculate or load if necessary

  void momentum(int *into) const; //get the quark 3-momentum
  int flavor() const; //get the quark flavor
  
  std::vector<std::vector<int> > get_allowed_momenta() const;

  ~QPropWcontainer(){ if(prop!=NULL) delete prop; }
};

class A2APropbfm;

class A2ApropContainer: public PropagatorContainer{
 protected:
  A2APropbfm *prop;

 public:
  A2ApropContainer(): PropagatorContainer(), prop(NULL){}

  void readProp(Lattice &latt); //loads prop if on disk (or if possible), does nothing otherwise
  void calcProp(Lattice &latt); //perform prop inversion
  void deleteProp(); //deletes the prop from memory, used to save space. Will be automatically recalculated if getProp is called again

  //Convenience function to check and then convert a base class reference. Pass in cname and fname
  static A2ApropContainer & verify_convert(PropagatorContainer &pc, const char* cname, const char* fname);
  static const A2ApropContainer & verify_convert(const PropagatorContainer &pc, const char* cname, const char* fname);

  A2APropbfm & getProp(Lattice &latt); //get the prop, calculate or load if necessary

  ~A2ApropContainer(){ if(prop!=NULL) delete prop; }
};




class LanczosContainer{
  Lanczos_5d<double> *lanczos;
  bfm_evo<double> *dwf;
  
  bool reload_gauge;

  LanczosContainerArg args;
 public:
  LanczosContainer(LanczosContainerArg &_args): reload_gauge(false), dwf(NULL), lanczos(NULL){ args.deep_copy(_args); } 

  void calcEig(Lattice &latt);
  void deleteEig();

  Lanczos_5d<double> & getEig(Lattice &latt);

  inline bool tagEquals(const char* what){ //check if tag is equal to input
    return (strcmp(args.tag,what)==0);
  }

  inline char const* tag() const{ return args.tag; }

  void reloadGauge(){ reload_gauge = true; }

  void set_lanczos(Lanczos_5d<double> *to); //Does not check arguments match those in args, so use only if you know what you are doing!
};




#endif
CPS_END_NAMESPACE
