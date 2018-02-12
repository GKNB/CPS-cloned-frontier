#ifndef _PROPAGATORS_GPAMA_H  
#define _PROPAGATORS_GPAMA_H

CPS_START_NAMESPACE

template<typename Type>
Type* randomSolutionPropagator(const bool store_midprop, Lattice &latt){
  if(!UniqueID()) printf("Generating random propagator solution vector\n");
  CommonArg c_arg;
  Type* ret = new Type(latt,&c_arg); //this constructor does nothing
  const int msize = 12*12*2;
  ret->Allocate(PROP);
  for(int f=0;f<GJP.Gparity()+1;f++){
    for(int x=0;x<GJP.VolNodeSites();x++){
      LRG.AssignGenerator(x,f);
      Float* off = (Float*)&(ret->SiteMatrix(x,f));
      for(int i=0;i<msize;i++) off[i] = LRG.Urand(0.5,-0.5,FOUR_D);
    }
  }
  if(store_midprop){
    ret->Allocate(MIDPROP);
    for(int f=0;f<GJP.Gparity()+1;f++){
      for(int x=0;x<GJP.VolNodeSites();x++){
	LRG.AssignGenerator(x,f);
	Float* off = (Float*)&(ret->MidPlaneSiteMatrix(x,f));
	for(int i=0;i<msize;i++) off[i] = LRG.Urand(0.5,-0.5,FOUR_D);
      }
    }
  }
  return ret;
}

void setupBasicArgs(CgArg &cg, QPropWArg &qpropw_arg, const double mass, const double stop_prec, const int t, const int flav, bool store_midprop){
  cg.mass = mass;
  cg.max_num_iter = 10000;
  cg.stop_rsd = stop_prec;
  cg.true_rsd = stop_prec;
  cg.RitzMatOper = NONE;
  cg.Inverter = CG;
  cg.bicgstab_n = 0;

  qpropw_arg.cg = cg;
  qpropw_arg.x = 0;
  qpropw_arg.y = 0;
  qpropw_arg.z = 0;
  qpropw_arg.t = t;
  qpropw_arg.flavor = flav; 
  qpropw_arg.ensemble_label = "ens";
  qpropw_arg.ensemble_id = "ens_id";
  qpropw_arg.StartSrcSpin = 0;
  qpropw_arg.EndSrcSpin = 4;
  qpropw_arg.StartSrcColor = 0;
  qpropw_arg.EndSrcColor = 3;
  qpropw_arg.gauge_fix_src = 1;
  qpropw_arg.gauge_fix_snk = 0;
  qpropw_arg.store_midprop = store_midprop ? 1 : 0; //for mres
}

multi1d<float>* latticeSetDeflation(Lattice &latt, BFM_Krylov::Lanczos_5d<double> *deflate){
  multi1d<float> *eval_conv = NULL;

  if(deflate != NULL){
    if(latt.Fclass() != F_CLASS_BFM && latt.Fclass() != F_CLASS_BFM_TYPE2)
      ERR.General("","computePropagator","Deflation only implemented for Fbfm\n");
    if(Fbfm::use_mixed_solver){
      //Have to convert evals to single prec
      eval_conv = new multi1d<float>(deflate->bl.size());
      for(int i=0;i<eval_conv->size();i++) eval_conv->operator[](i) = deflate->bl[i];
      dynamic_cast<Fbfm&>(latt).set_deflation<float>(&deflate->bq,eval_conv,0); //last argument is really obscure - it's the number of eigenvectors subtracted from the solution to produce a high-mode inverse - we want zero here
    }else dynamic_cast<Fbfm&>(latt).set_deflation(&deflate->bq,&deflate->bl,0);
  }
  return eval_conv;
}
void latticeUnsetDeflation(Lattice &latt, BFM_Krylov::Lanczos_5d<double> *deflate, multi1d<float> *eval_conv){
  if(deflate != NULL) dynamic_cast<Fbfm&>(latt).unset_deflation();
  if(eval_conv !=NULL) delete eval_conv;
}

  


//Generate a flavor 'f' gauge fixed wall momentum propagator from given timeslice. Momenta are in units of pi/2L
//Eigenvectors must be those appropriate to the choice of temporal BC, 'time_bc'

//Note: If using Fbfm or FGrid, the current temporal BC listed in GJP.Tbc() must be applied to the bfm/Grid internal gauge field (i.e. minuses on t-links at boundard for APRD) prior to using this method. Internally
//it changes the bc to 'time_bc' but it changes it back at the end.
QPropWMomSrc* computeMomSourcePropagator(const double mass, const double stop_prec, const int t, const int flav, const ThreeMomentum &mom, const BndCndType time_bc, const bool store_midprop, 
					  Lattice &latt,  BFM_Krylov::Lanczos_5d<double> *deflate = NULL, const bool random_solution = false){ 
  if(random_solution) return randomSolutionPropagator<QPropWMomSrc>(store_midprop,latt);
  int const* p = mom.ptr();

  multi1d<float> *eval_conv = latticeSetDeflation(latt, deflate);

  CommonArg c_arg;
  CgArg cg;
  QPropWArg qpropw_arg;
  setupBasicArgs(cg, qpropw_arg,mass,stop_prec,t,flav,store_midprop);

  //Switching boundary conditions is poorly implemented in Fbfm (and probably FGrid)
  //For traditional lattice types the BC is applied by modififying the gauge field when the Dirac operator is created and reverting when destroyed. This only ever happens internally - no global instance of the Dirac operator exists
  //On the other hand, Fbfm does all its inversion internally and doesn't instantiate a CPS Dirac operator. We therefore have to manually force Fbfm to change its internal gauge field by applying BondCond
  bool is_wrapper_type = ( latt.Fclass() == F_CLASS_BFM || latt.Fclass() == F_CLASS_BFM_TYPE2 ); //I hate this!

  BndCndType init_tbc = GJP.Tbc();
  BndCndType target_tbc = time_bc;

  GJP.Bc(3,target_tbc);
  if(is_wrapper_type) latt.BondCond();  //Apply new BC to internal gauge fields

  QPropWMomSrc* ret = new QPropWMomSrc(latt,&qpropw_arg,const_cast<int*>(p),&c_arg);

  //Restore the BCs
  if(is_wrapper_type) latt.BondCond();  //unapply existing BC
  GJP.Bc(3,init_tbc);

  latticeUnsetDeflation(latt, deflate, eval_conv);
  return ret;
}


PropWrapper computeMomSourcePropagator(const double mass, const double stop_prec, const int t, const ThreeMomentum &mom, const BndCndType time_bc, const bool store_midprop, 
					Lattice &latt,  BFM_Krylov::Lanczos_5d<double> *deflate = NULL, const bool random_solution = false){ 
  QPropWMomSrc* prop_f0 = computeMomSourcePropagator(mass,stop_prec,t,0,mom,time_bc,store_midprop,latt,deflate,random_solution);
  QPropWMomSrc* prop_f1 = GJP.Gparity() ? computeMomSourcePropagator(mass,stop_prec,t,1,mom,time_bc,store_midprop,latt,deflate,random_solution) : NULL;
  return PropWrapper(prop_f0, prop_f1);
}



void computeMomSourcePropagators(Props &props, const double mass, const double stop_prec, const std::vector<int> &tslices, const QuarkMomenta &quark_momenta, const BndCndType time_bc, const bool store_midprop, 
				  Lattice &latt,  BFM_Krylov::Lanczos_5d<double> *deflate = NULL, const bool random_solution = false){
  for(int tt=0;tt<tslices.size();tt++){
    const int t = tslices[tt];
    for(int pp=0;pp<quark_momenta.nMom();pp++){
      const ThreeMomentum &p = quark_momenta.getMom(pp);

      props(t,p) = computeMomSourcePropagator(mass,stop_prec,t,p,time_bc,store_midprop,latt,deflate,random_solution);

      if(GJP.Gparity()){
	//Free to add - momentum
	(props(t,-p) = props(t,p)).setFlip(true);
      }	      
    }
  }
}

Complex* getRandomSource(const RandomType rand_type, Lattice &latt){
  CommonArg c_arg;
  CgArg cg;
  QPropWArg qpropw_arg;
  setupBasicArgs(cg, qpropw_arg,0.01,1e-8,0,0,false); //these can be anything

  QPropWRandArg rand_arg;
  rand_arg.rng = rand_type;

  QPropWRand r(latt,&qpropw_arg,&rand_arg, &c_arg);
  void *base = (void*)&r.rand_src(0);
  
  size_t sz = (GJP.Gparity() + 1) * GJP.VolNodeSites() * 2 * sizeof(Float);
  void* out = malloc(sz);
  memcpy(out,base,sz);
  return (Complex*)out;
}
 

QPropWRandMomSrc* computeRandMomSourcePropagator(const RandomType rand_type, const double mass, const double stop_prec, const int t, const int flav, 
						 const ThreeMomentum &mom, const BndCndType time_bc, const bool store_midprop, 
						 Lattice &latt, Complex const* random_src, BFM_Krylov::Lanczos_5d<double> *deflate = NULL, const bool random_solution = false){ 
  if(random_solution) return randomSolutionPropagator<QPropWRandMomSrc>(store_midprop,latt);
  int const* p = mom.ptr();
  multi1d<float> *eval_conv = latticeSetDeflation(latt, deflate);
  
  CommonArg c_arg;
  CgArg cg;
  QPropWArg qpropw_arg;
  setupBasicArgs(cg, qpropw_arg,mass,stop_prec,t,flav,store_midprop);

  QPropWRandArg rand_arg;
  rand_arg.rng = rand_type;
  
  //Switching boundary conditions is poorly implemented in Fbfm (and probably FGrid)
  //For traditional lattice types the BC is applied by modififying the gauge field when the Dirac operator is created and reverting when destroyed. This only ever happens internally - no global instance of the Dirac operator exists
  //On the other hand, Fbfm does all its inversion internally and doesn't instantiate a CPS Dirac operator. We therefore have to manually force Fbfm to change its internal gauge field by applying BondCond
  bool is_wrapper_type = ( latt.Fclass() == F_CLASS_BFM || latt.Fclass() == F_CLASS_BFM_TYPE2 ); //I hate this!

  BndCndType init_tbc = GJP.Tbc();
  BndCndType target_tbc = time_bc;

  GJP.Bc(3,target_tbc);
  if(is_wrapper_type) latt.BondCond();  //Apply new BC to internal gauge fields

  QPropWRandMomSrc* ret = new QPropWRandMomSrc(latt,&qpropw_arg,&rand_arg,const_cast<int*>(p),&c_arg, random_src);

  //Restore the BCs
  if(is_wrapper_type) latt.BondCond();  //unapply existing BC
  GJP.Bc(3,init_tbc);

  latticeUnsetDeflation(latt, deflate, eval_conv);
  return ret;
}

PropWrapper computeRandMomSourcePropagator(const RandomType rand_type,const double mass, const double stop_prec, const int t, const ThreeMomentum &mom, const BndCndType time_bc, const bool store_midprop, 
					   Lattice &latt, Complex const* random_src,  BFM_Krylov::Lanczos_5d<double> *deflate = NULL, const bool random_solution = false){ 
  QPropWRandMomSrc* prop_f0 = computeRandMomSourcePropagator(rand_type,mass,stop_prec,t,0,mom,time_bc,store_midprop,latt,random_src,deflate,random_solution);
  QPropWRandMomSrc* prop_f1 = GJP.Gparity() ? computeRandMomSourcePropagator(rand_type,mass,stop_prec,t,1,mom,time_bc,store_midprop,latt,random_src,deflate,random_solution) : NULL;
  return PropWrapper(prop_f0, prop_f1);
}


//Assume common 4d complex random source
void computeRandMomSourcePropagators(Props &props, const RandomType rand_type,const double mass, const double stop_prec, const std::vector<int> &tslices, const QuarkMomenta &quark_momenta, const BndCndType time_bc, const bool store_midprop, 
				     Lattice &latt, Complex const* random_src, BFM_Krylov::Lanczos_5d<double> *deflate = NULL, const bool random_solution = false){
  for(int tt=0;tt<tslices.size();tt++){
    const int t = tslices[tt];
    for(int pp=0;pp<quark_momenta.nMom();pp++){
      const ThreeMomentum &p = quark_momenta.getMom(pp);

      props(t,p) = computeRandMomSourcePropagator(rand_type,mass,stop_prec,t,p,time_bc,store_midprop,latt,random_src,deflate,random_solution);

      if(GJP.Gparity()){
	//Free to add - momentum
	(props(t,-p) = props(t,p)).setFlip(true);
      }	      
    }
  }
}



//Combine quarks with P and A Tbcs into F=P+A and B=P-A types which are added to the PropMomContainer with appropriate tags
void combinePA(Props &props_F, Props &props_B, const Props &props_P, const Props &props_A){
  Props::const_iterator itP = props_P.begin();
  Props::const_iterator itA = props_A.begin();

  while(itP != props_P.end()){
    const int t = itP->first.first;
    const ThreeMomentum &p = itP->first.second;

    assert(itA->first.first == t);
    assert(itA->first.second == p);

    PropWrapper combF = PropWrapper::combinePA(itP->second,itA->second,CombinationF);
    PropWrapper combB = PropWrapper::combinePA(itP->second,itA->second,CombinationB);

    props_F(t,p) = combF;
    props_B(t,p) = combB;

    itP++; itA++;
  }
}


CPS_END_NAMESPACE

#endif
