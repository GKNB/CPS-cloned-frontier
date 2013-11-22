#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <util/qcdio.h>
#ifdef PARALLEL
#include <comms/sysfunc_cps.h>
#endif
#include <comms/scu.h>
#include <comms/glb.h>

#include <util/lattice.h>
#include <util/time_cps.h>
#include <util/smalloc.h>

#include <util/command_line.h>

#include<unistd.h>
#include<config.h>

#include <alg/qpropw.h>
#include <alg/propvector.h>

CPS_START_NAMESPACE

PropagatorContainer & PropVector::operator[](const int &idx){ return *props[idx]; }
PropVector::PropVector(): sz(0){ for(int i=0;i<MAX_SIZE;i++) props[i] = NULL; }
PropVector::~PropVector(){ for(int i=0;i<MAX_SIZE;i++) if(props[i]!=NULL) delete props[i]; }
  
const int &PropVector::size() const{ return sz; }

PropagatorContainer & PropVector::addProp(PropagatorArg &arg){
  if(sz==MAX_SIZE){ ERR.General("PropVector","addProp(PropagatorArg &arg)","Reached maximum number of allowed propagators: %d\n",MAX_SIZE); }
  PropagatorContainer* p = PropagatorContainer::create(arg.generics.type);
  p->setup(arg);
  props[sz++] = p;
  return *p;
}
void PropVector::clear(){
  for(int i=0;i<MAX_SIZE;i++)
    if(props[i]!=NULL){
      delete props[i];
      props[i]=NULL;
    }
  sz = 0;
}

CPS_END_NAMESPACE
