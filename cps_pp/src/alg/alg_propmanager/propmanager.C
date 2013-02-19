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

#include <alg/propmanager.h>

CPS_START_NAMESPACE


PropVector PropManager::props;

PropagatorContainer & PropManager::getProp(const char *tag){
  for(int i=0;i<props.size();i++) if(props[i].tagEquals(tag)) return props[i];
  ERR.General("PropManager","getProp(const char *tag, Lattice &latt)","Prop '%s' does not exist!\n",tag);
};

PropagatorContainer & PropManager::addProp(PropagatorArg &arg){
  return props.addProp(arg);
}

void PropManager::setup(JobPropagatorArgs &prop_args){
  for(int i=0;i< prop_args.props.props_len; i++){
    addProp(prop_args.props.props_val[i]);
  }
  //after all props are loaded in, the attributes of props that combine other props are copied over
  for(int i=0;i<props.size();i++) props[i].propCombSetupAttrib();
}

void PropManager::startNewTraj(){
  for(int i=0;i<props.size();i++){
    props[i].deleteProp();
  }
}

void PropManager::clear(){
  props.clear();
}

void PropManager::calcProps(Lattice &latt){
  for(int i=0;i<props.size();i++){
    props[i].readProp(latt);
    props[i].calcProp(latt); //will calculate if not read
  }
}

CPS_END_NAMESPACE
