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
#include <alg/do_arg.h>
#include <alg/no_arg.h>
#include <alg/common_arg.h>
#include <util/smalloc.h>

#include <util/ReadLatticePar.h>
#include <util/WriteLatticePar.h>

#include<unistd.h>
#include<config.h>

#include <alg/qpropw.h>
#include <alg/qpropw_arg.h>

#include<alg/propagatorcontainer.h>
#include<alg/propmanager.h>
CPS_START_NAMESPACE


PropagatorContainer::PropagatorContainer(): prop(NULL){ for(int i=0;i<50;i++) attributes[i]=NULL;}


void PropagatorContainer::setup(PropagatorArg &arg){
  for(int i=0;i<50;i++) if(attributes[i]) delete attributes[i];
  
  //special case, generic attributes are always present and maintained outside of the array
  AttributeContainer* p = new AttributeContainer;
  p->type = GENERIC_PROP_ATTR;
  p->AttributeContainer_u.generic_prop_attr.deep_copy(arg.generics);
  attributes[ (int)p->type ] = p;

  if(UniqueID()==0) printf("Got %d additional attributes\n",arg.attributes.attributes_len);

  for(int i=0;i<arg.attributes.attributes_len;i++){
    //no duplicates, later entries of same type overwrite earlier
    int idx = (int)arg.attributes.attributes_val[i].type;
    if(attributes[idx]!=NULL) delete attributes[idx];
    attributes[idx] = new AttributeContainer;
    attributes[idx]->deep_copy(arg.attributes.attributes_val[i]); //make a copy
  }
}

void PropagatorContainer::add(const AttributeContainer &p){
  int idx = (int)p.type;
  if(attributes[idx]!=NULL) delete attributes[idx]; //no duplicates
  attributes[idx] = new AttributeContainer;
  attributes[idx]->deep_copy(p);
}

AttributeContainer*  PropagatorContainer::findAttr(const AttrType &type) const{
  return attributes[(int)type];
}
PropagatorContainer::~PropagatorContainer(){
  for(int i=0;i<50;i++) if(attributes[i]) delete attributes[i];
  if(prop!=NULL) delete prop;
}


void PropagatorContainer::readProp(Lattice &latt){
  if(prop!=NULL) return; //don't load if already inverted
  PropIOAttrArg *io;
  if(!getAttr(io)) return;
  if(!io->prop_on_disk) return;

  if(UniqueID()==0) printf("Prop is on disk, loading from file %s\n",io->qio_filename);
  
  //load prop with info from io
  CommonArg c_arg("label","filename");//find out what this does!
  
  prop = new QPropW(latt,&c_arg);
  prop->ReLoad(io->qio_filename);
}

void PropagatorContainer::calcProp(Lattice &latt){
  //function acts as factory for QPropW objects depending on the attribute objects
  if(prop!=NULL) return; //don't calculate twice

  char *cname = "PropagatorContainer";
  char *fname = "calcProp()";

  CommonArg c_arg("label","filename");//find out what this does!

  GenericPropAttrArg *generics;
  if(!getAttr(generics)) ERR.General(cname,fname,"Propagator attribute list does not contain a GenericPropAttr\n");

  //if propagator is a combination of other propagators, do this separately
  PropCombinationAttrArg *propcomb;
  if(getAttr(propcomb)){
    if(UniqueID()==0) printf("Propagator %s combining props %s and %s\n",generics->tag,propcomb->prop_A,propcomb->prop_B);

    PropagatorContainer &A = PropManager::getProp(propcomb->prop_A);
    PropagatorContainer &B = PropManager::getProp(propcomb->prop_B);

    //copy attributes from A. Does not check that attributes of A and B match
    propCombSetupAttrib();

    //calculate A and B if they have not yet been calculated
    QPropW &A_qpw = A.getProp(latt);
    QPropW &B_qpw = B.getProp(latt);

    prop = new QPropW(A_qpw);
    //perform the combination
    if(UniqueID()==0) printf("Propagator %s starting combination: prop %p, A_qpw %p B_qpw %p\n",generics->tag,prop,&A_qpw,&B_qpw);
    
    if(propcomb->combination == A_PLUS_B){
      prop->Average(B_qpw);
    }else if(propcomb->combination == A_MINUS_B){
      prop->LinComb(B_qpw,0.5,-0.5);
    }else{
      ERR.General(cname,fname,"Unknown PropagatorCombination");
    }
    if(UniqueID()==0) printf("Propagator %s finished combination\n",generics->tag);
    return;
  }

  //calculate the propagator
  if(UniqueID()==0) printf("Calculating propagator %s\n",generics->tag);

  CgArg cg;
  cg.mass =  generics->mass;
  cg.max_num_iter = 5000;
  cg.stop_rsd =   1.0000000000000000e-08;
  cg.true_rsd =   1.0000000000000000e-08;
  cg.RitzMatOper = NONE;
  cg.Inverter = CG;
  cg.bicgstab_n = 0;

  CGAttrArg *cgattr;
  if(getAttr(cgattr)){
    cg.max_num_iter = cgattr->max_num_iter;
    cg.stop_rsd = cgattr->stop_rsd;
    cg.true_rsd = cgattr->true_rsd; 
  }

  QPropWArg qpropw_arg;
  qpropw_arg.cg = cg;
  qpropw_arg.x = 0;
  qpropw_arg.y = 0;
  qpropw_arg.z = 0;
  qpropw_arg.t = 0;
  qpropw_arg.flavor = 0; //default on d field  
  qpropw_arg.ensemble_label = "ens";
  qpropw_arg.ensemble_id = "ens_id";
  qpropw_arg.StartSrcSpin = 0;
  qpropw_arg.EndSrcSpin = 4;
  qpropw_arg.StartSrcColor = 0;
  qpropw_arg.EndSrcColor = 3;

  //boundary conditions
  BndCndType init_bc[4];
  TwistedBcAttrArg *tbcarg;
  for(int i=0;i<4;i++){
    if(i<3 && generics->bc[i] != GJP.Bc(i) && !(generics->bc[i] == BND_CND_TWISTED || generics->bc[i] == BND_CND_GPARITY_TWISTED) )
      ERR.General(cname,fname,"Propagator %s: valence and sea spatial boundary conditions do not match (partially-twisted BCs are allowed)\n",generics->tag);
    if( GJP.Bc(i) != BND_CND_GPARITY && generics->bc[i] == BND_CND_GPARITY_TWISTED ) ERR.General(cname,fname,"Propagator %s: Cannot use twisted G-parity valence BCs in a non-Gparity direction");

    if(generics->bc[i] == BND_CND_TWISTED || generics->bc[i] == BND_CND_GPARITY_TWISTED){
      if(getAttr(tbcarg)) for(int j=0;j<3;j++) GJP.TwistAngle(j, tbcarg->theta[j]);
      else for(int j=0;j<3;j++) GJP.TwistAngle(j,0.0); //default twist angle is zero
    }
    init_bc[i] = GJP.Bc(i);
    GJP.Bc(i,generics->bc[i]);
  }

  //fill out qpropw_arg arguments
  GparityFlavorAttrArg *flav;
  if(getAttr(flav)) qpropw_arg.flavor = flav->flavor;

  //mid-point correlator?
  if(hasAttr<StoreMidpropAttrArg>()) qpropw_arg.store_midprop = 1;

  PropIOAttrArg *io;
  if(getAttr(io)){ 
    qpropw_arg.save_prop = io->save_to_disk;
    #ifndef USE_QIO
    if(io->save_to_disk) ERR.General(cname,fname,"Cannot save propagators without QIO\n");
    #endif

    qpropw_arg.file = io->qio_filename;
  }

  GaugeFixAttrArg *gfix;
  if(getAttr(gfix)){
    if(latt.FixGaugeKind() == FIX_GAUGE_NONE && (gfix->gauge_fix_src==1 || gfix->gauge_fix_snk==1))
      ERR.General(cname,fname,"Gauge fixed source or sink requested but gauge fixing matrices have not been calculated\n");
    
    qpropw_arg.gauge_fix_src = gfix->gauge_fix_src; 
    qpropw_arg.gauge_fix_snk = gfix->gauge_fix_snk;
  }

  PointSourceAttrArg *pt;
  WallSourceAttrArg *wl;
  VolumeSourceAttrArg *vl;
  if(getAttr(pt) && getAttr(wl) || getAttr(pt) && getAttr(vl) || getAttr(vl) && getAttr(wl) )
    ERR.General(cname,fname,"Propagator %s: Must specify only one source type attribute\n",generics->tag);

  //NOTE: Momentum units are:
  //                         Periodic  2\pi/L
  //                     Antiperiodic   \pi/L
  //                         G-parity   \pi/2L
  //In G-parity directions, the propagators are antiperiodic in 2L

  if(getAttr(pt)){
    qpropw_arg.x = pt->pos[0];
    qpropw_arg.y = pt->pos[1];
    qpropw_arg.z = pt->pos[2];
    qpropw_arg.t = pt->pos[3];
    
    //assemble the arg objects
    prop = new QPropWPointSrc(latt,&qpropw_arg,&c_arg);

    MomentumAttrArg *mom;
    if(getAttr(mom)){
      //for a point source with momentum, apply the e^-ipx factor at the source location 
      //such that it can be treated in the same way as a momentum source

      ThreeMom tmom(mom->p);
      Site s(pt->pos[0],pt->pos[1],pt->pos[2],pt->pos[3]);

      Complex phase;

      MomCosAttrArg *cos;
      if(getAttr(cos)){ //a cosine point source
	phase = tmom.FactCos(s);
      }else{
	phase = tmom.Fact(s);
      }

      (*prop)[s.Index()] *= phase; 
      if(GJP.Gparity()){
	int wmat_shift = GJP.VolNodeSites();
	(*prop)[s.Index()+wmat_shift] *= phase; 
      }

      if(!UniqueID()) printf("Propagator %s has 3-momentum (%d,%d,%d) and position (%d,%d,%d,%d), source phase factor is (%e,%e)\n",
			     generics->tag,mom->p[0],mom->p[1],mom->p[2],pt->pos[0],pt->pos[1],pt->pos[2],pt->pos[3],phase.real(),phase.imag());

    }
  }else if(getAttr(wl)){
    qpropw_arg.t = wl->t;
    
    if(qpropw_arg.gauge_fix_src == 1 && latt.FixGaugeKind() != FIX_GAUGE_COULOMB_T)
      ERR.General(cname,fname,"Gauge fixed wall/mom source requested, but gauge is not Coulomb-T\n");

    MomentumAttrArg *mom;
    if(getAttr(mom)){
      MomCosAttrArg *cos;
      if(getAttr(cos)){ //a cosine source
	prop = new QPropWMomCosSrc(latt,&qpropw_arg,mom->p,&c_arg); //knows about correct G-parity units of momentum (cf. ThreeMom::CalcLatMom)
      }else{ //a regular momentum source
	prop = new QPropWMomSrc(latt,&qpropw_arg,mom->p,&c_arg); //knows about correct G-parity units of momentum
      }
    }else{
      prop = new QPropWWallSrc(latt,&qpropw_arg,&c_arg);
    }
  }else if(getAttr(vl)){
    if(qpropw_arg.gauge_fix_src == 1 && latt.FixGaugeKind() == FIX_GAUGE_NONE)
      ERR.General(cname,fname,"Gauge fixed volume source requested, but no gauge fixing has been performed\n");
    
    MomentumAttrArg *mom;
    if(getAttr(mom)){
      MomCosAttrArg *cos;
      if(getAttr(cos)){
	ERR.General(cname,fname,"No volume cosine source implemented");
      }else{
	prop = new QPropWVolMomSrc(latt,&qpropw_arg,mom->p,&c_arg);
      }
    }else{
      prop = new QPropWVolSrc(latt,&qpropw_arg,&c_arg);
    }
  }else{
    ERR.General(cname,fname,"Propagator %s has no source type AttrArg\n",generics->tag);
  }
  
  if(getAttr(io) && io->save_to_disk){
    io->prop_on_disk = true; //QPropW saves the prop
  }
  
  //restore boundary conditions to original
  for(int i=0;i<4;i++) GJP.Bc(i,init_bc[i]);
  for(int j=0;j<3;j++) GJP.TwistAngle(j,0);
}


int PropagatorContainer::flavor() const{
  GparityFlavorAttrArg *flav;
  if(getAttr(flav)) return flav->flavor;
  return 0;
}
void PropagatorContainer::momentum(int *into) const{
    MomentumAttrArg *mom;
    if(getAttr(mom)){
      for(int i=0;i<3;i++) into[i] = mom->p[i];
    }else{
      for(int i=0;i<3;i++) into[i] = 0;
    }
}

QPropW & PropagatorContainer::getProp(Lattice &latt){
  if(prop==NULL){
    readProp(latt);
    calcProp(latt); //will calculate if prop was not read
  }
  return *prop;
}

bool PropagatorContainer::tagEquals(const char* what){
  GenericPropAttrArg *generics;
  if(!getAttr(generics)) ERR.General("PropagatorContainer","tagEquals(const char* what)","Propagator attribute list does not contain a GenericPropAttr\n");
  if(strcmp(generics->tag,what)==0) return true;
  return false;
}
char const* PropagatorContainer::tag() const{
  GenericPropAttrArg *generics;
  if(!getAttr(generics)) ERR.General("PropagatorContainer","tag()","Propagator attribute list does not contain a GenericPropAttr\n");
  return generics->tag;
}


void PropagatorContainer::deleteProp(){
  if(prop!=NULL){ delete prop; prop=NULL; }
}

void PropagatorContainer::printAttribs() const{
  printf("Propagator Attributes:\n");
  for(int i=0;i<50;i++){
    if(attributes[i]!=NULL){
      printf("%s:",AttrType_map[i].name);
      attributes[i]->print();
    }
  }
}

void PropagatorContainer::propCombSetupAttrib(){
  PropCombinationAttrArg *propcomb;
  if(getAttr(propcomb)){
    PropagatorContainer &A = PropManager::getProp(propcomb->prop_A);

    //copy attributes from A. Does not check that attributes of A and B match
    for(int i=0;i<50;i++) 
      if((AttrType)i != GENERIC_PROP_ATTR && attributes[i]==NULL && A.attributes[i]!=NULL) 
  	add(*A.attributes[i]);
  }
}


std::vector<std::vector<int> > PropagatorContainer::get_allowed_momenta() const{
  std::vector<std::vector<int> > out;

  MomCosAttrArg *cos;
  MomentumAttrArg *mom;

  if(getAttr(mom)){
    std::vector<int> p(3); p[0] = mom->p[0]; p[1] = mom->p[1]; p[2] = mom->p[2];
    out.push_back(p);

    if(getAttr(cos)){
      std::vector<int> mp(p); for(int i=0;i<3;i++) mp[i]*=-1;
      out.push_back(mp);
    }
  }else{
    out.push_back(std::vector<int>(3,0));
  }
  return out;
}


CPS_END_NAMESPACE


