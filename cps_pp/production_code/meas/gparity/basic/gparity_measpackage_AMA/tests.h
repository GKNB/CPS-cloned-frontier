#ifndef _TESTS_GPARITY_AMA
#define _TESTS_GPARITY_AMA

CPS_START_NAMESPACE

void setupDoArg(DoArg &do_arg, int size[5], int ngp, bool verbose = true){
  do_arg.x_sites = size[0];
  do_arg.y_sites = size[1];
  do_arg.z_sites = size[2];
  do_arg.t_sites = size[3];
  do_arg.s_sites = size[4];
  do_arg.x_node_sites = 0;
  do_arg.y_node_sites = 0;
  do_arg.z_node_sites = 0;
  do_arg.t_node_sites = 0;
  do_arg.s_node_sites = 0;
  do_arg.x_nodes = 0;
  do_arg.y_nodes = 0;
  do_arg.z_nodes = 0;
  do_arg.t_nodes = 0;
  do_arg.s_nodes = 0;
  do_arg.updates = 0;
  do_arg.measurements = 0;
  do_arg.measurefreq = 0;
  do_arg.cg_reprod_freq = 10;
  do_arg.x_bc = BND_CND_PRD;
  do_arg.y_bc = BND_CND_PRD;
  do_arg.z_bc = BND_CND_PRD;
  do_arg.t_bc = BND_CND_APRD;
  do_arg.start_conf_kind = START_CONF_ORD;
  do_arg.start_conf_load_addr = 0x0;
  do_arg.start_seed_kind = START_SEED_FIXED;
  do_arg.start_seed_filename = "../rngs/ckpoint_rng.0";
  do_arg.start_conf_filename = "../configurations/ckpoint_lat.0";
  do_arg.start_conf_alloc_flag = 6;
  do_arg.wfm_alloc_flag = 2;
  do_arg.wfm_send_alloc_flag = 2;
  do_arg.start_seed_value = 83209;
  do_arg.beta =   2.25;
  do_arg.c_1 =   -3.3100000000000002e-01;
  do_arg.u0 =   1.0000000000000000e+00;
  do_arg.dwf_height =   1.8000000000000000e+00;
  do_arg.dwf_a5_inv =   1.0000000000000000e+00;
  do_arg.power_plaq_cutoff =   0.0000000000000000e+00;
  do_arg.power_plaq_exponent = 0;
  do_arg.power_rect_cutoff =   0.0000000000000000e+00;
  do_arg.power_rect_exponent = 0;
  do_arg.verbose_level = -1202; //VERBOSE_DEBUG_LEVEL; //-1202;
  do_arg.checksum_level = 0;
  do_arg.exec_task_list = 0;
  do_arg.xi_bare =   1.0000000000000000e+00;
  do_arg.xi_dir = 3;
  do_arg.xi_v =   1.0000000000000000e+00;
  do_arg.xi_v_xi =   1.0000000000000000e+00;
  do_arg.clover_coeff =   0.0000000000000000e+00;
  do_arg.clover_coeff_xi =   0.0000000000000000e+00;
  do_arg.xi_gfix =   1.0000000000000000e+00;
  do_arg.gfix_chkb = 1;
  do_arg.asqtad_KS =   0.0000000000000000e+00;
  do_arg.asqtad_naik =   0.0000000000000000e+00;
  do_arg.asqtad_3staple =   0.0000000000000000e+00;
  do_arg.asqtad_5staple =   0.0000000000000000e+00;
  do_arg.asqtad_7staple =   0.0000000000000000e+00;
  do_arg.asqtad_lepage =   0.0000000000000000e+00;
  do_arg.p4_KS =   0.0000000000000000e+00;
  do_arg.p4_knight =   0.0000000000000000e+00;
  do_arg.p4_3staple =   0.0000000000000000e+00;
  do_arg.p4_5staple =   0.0000000000000000e+00;
  do_arg.p4_7staple =   0.0000000000000000e+00;
  do_arg.p4_lepage =   0.0000000000000000e+00;

  if(verbose) do_arg.verbose_level = VERBOSE_DEBUG_LEVEL;

  BndCndType* bc[3] = { &do_arg.x_bc, &do_arg.y_bc, &do_arg.z_bc };
  for(int i=0;i<ngp;i++){ 
    *(bc[i]) = BND_CND_GPARITY;
  }
}


void setup_bfmargs(BfmArg &dwfa, const BfmSolverType solver, const int nthreads, const double mobius_scale = 2.){
  dwfa.verbose=1;
  dwfa.reproduce=0;
  dwfa.precon_5d = 1;
  if(solver == BFM_HmCayleyTanh){
    dwfa.precon_5d = 0; //mobius uses 4d preconditioning
    dwfa.mobius_scale = mobius_scale;
  }else if(solver != BFM_DWF){
    ERR.General("","setup_bfmargs","Unknown solver\n");
  }

  dwfa.threads = nthreads;
  dwfa.Ls = GJP.SnodeSites();
  dwfa.solver = solver;
  dwfa.M5 = toDouble(GJP.DwfHeight());
  dwfa.mass = toDouble(0.04);
  dwfa.Csw = 0.0;
  dwfa.max_iter = 200000;
  dwfa.residual = 1e-08;
}

void setupFixGaugeArg(FixGaugeArg &r){
  r.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
  r.hyperplane_start = 0;
  r.hyperplane_step = 1;
  r.hyperplane_num = GJP.TnodeSites()*GJP.Tnodes();
  r.stop_cond = 1e-06;
  r.max_iter_num = 60000;
}

void compareProps(QPropW &A, QPropW &B, const std::string label, Float tol = 1e-12){
  Float fail = 0;
  for(int f=0;f<GJP.Gparity()+1;f++){
    for(int x=0;x<GJP.VolNodeSites();x++){
      WilsonMatrix & Am = A.SiteMatrix(x,f);
      WilsonMatrix & Bm = B.SiteMatrix(x,f);

      Float* Amp = (Float*)&Am;
      Float* Bmp = (Float*)&Bm;

      for(int i=0;i<24;i++){
	if(fabs(Amp[i]- Bmp[i]) > tol){
	  printf("Node %d fail f=%d x=%d i=%d : %g vs %g\n",UniqueID(),f,x,i,Amp[i],Bmp[i]);
	  fail = 1.;
	}
      }
    }
  }
  glb_sum(&fail);
  if(fail!=0.0){
    if(!UniqueID()) printf("Prop comparison '%s' failed\n",label.c_str());
    exit(-1);
  }else{
    if(!UniqueID()) printf("Prop comparison '%s' passed\n",label.c_str());
  }
}



PropagatorContainer & computePropagatorOld(const std::string &tag, const double mass, const double stop_prec, const int t, const int flav, const int p[3], Lattice &latt, const std::string tag_otherflav = "", BFM_Krylov::Lanczos_5d<double> *deflate = NULL){ 
  if(deflate != NULL) ERR.General("","computePropagatorOld","Deflation not yet implemented\n");

  PropagatorArg parg;
  parg.generics.type = QPROPW_TYPE;
  parg.generics.tag = strdup(tag.c_str());
  parg.generics.mass = mass;
  for(int i=0;i<4;i++)
    parg.generics.bc[i] = GJP.Bc(i);
  
  int len = 5 + (tag_otherflav != "" ? 1 : 0);
  parg.attributes.attributes_len = len;
  parg.attributes.attributes_val = new AttributeContainer[len];

  parg.attributes.attributes_val[0].type = WALL_SOURCE_ATTR;
  WallSourceAttrArg &wa = parg.attributes.attributes_val[0].AttributeContainer_u.wall_source_attr;
  wa.t = t;

  parg.attributes.attributes_val[1].type = GPARITY_FLAVOR_ATTR;
  GparityFlavorAttrArg &fa = parg.attributes.attributes_val[1].AttributeContainer_u.gparity_flavor_attr;
  fa.flavor = flav;

  parg.attributes.attributes_val[2].type = CG_ATTR;
  CGAttrArg &cga = parg.attributes.attributes_val[2].AttributeContainer_u.cg_attr;
  cga.max_num_iter = 10000;
  cga.stop_rsd = stop_prec;
  cga.true_rsd = stop_prec;

  parg.attributes.attributes_val[3].type = GAUGE_FIX_ATTR;
  GaugeFixAttrArg &gfa = parg.attributes.attributes_val[3].AttributeContainer_u.gauge_fix_attr;
  gfa.gauge_fix_src = 1;
  gfa.gauge_fix_snk = 0;

  parg.attributes.attributes_val[4].type = MOMENTUM_ATTR;
  MomentumAttrArg &ma = parg.attributes.attributes_val[4].AttributeContainer_u.momentum_attr;
  memcpy(ma.p,p,3*sizeof(int));

  if(tag_otherflav != ""){
    parg.attributes.attributes_val[5].type = GPARITY_OTHER_FLAV_PROP_ATTR;
    GparityOtherFlavPropAttrArg &gofa = parg.attributes.attributes_val[5].AttributeContainer_u.gparity_other_flav_prop_attr;
    gofa.tag = strdup(tag_otherflav.c_str());
  }
  return PropManager::addProp(parg);
}


bool test_equals(const SpinColorFlavorMatrix &a, const SpinColorFlavorMatrix &b, const double &eps){
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int aa=0;aa<3;aa++){
	for(int bb=0;bb<3;bb++){
	  for(int f0=0;f0<2;f0++){
	    for(int f1=0;f1<2;f1++){
	    
	      Complex ca = a(i,aa,f0,j,bb,f1);
	      Complex cb = b(i,aa,f0,j,bb,f1);
	      if( fabs(ca.real()-cb.real()) > eps || fabs(ca.imag()-cb.imag()) > eps ){		
		printf("FAIL %d %d %d %d %d %d (%.4f %.4f) (%.4f %.4f), diff (%g, %g)\n",i,aa,f0,j,bb,f1,ca.real(),ca.imag(),cb.real(),cb.imag(), std::real(ca-cb), std::imag(ca-cb));
		return false;
	      }
	    }
	  }
	}
      }
    }
  }
  return true;
}

void print2(const SpinColorFlavorMatrix &a,const SpinColorFlavorMatrix &b){
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int aa=0;aa<3;aa++){
	for(int bb=0;bb<3;bb++){
	  for(int f0=0;f0<2;f0++){
	    for(int f1=0;f1<2;f1++){
	      Complex ca = a(i,aa,f0,j,bb,f1);
	      Complex cb = b(i,aa,f0,j,bb,f1);
	      printf("%d %d %d %d %d %d (%.4f %.4f) (%.4f %.4f)\n",i,aa,f0,j,bb,f1,ca.real(),ca.imag(),cb.real(),cb.imag());
	    }
	  }
	}
      }
    }
  }
}


int run_tests(int argc,char *argv[])
{
  Start(&argc, &argv);
  int ngp;
  { std::stringstream ss; ss << argv[1]; ss >> ngp; }

  if(!UniqueID()) printf("Doing G-parity in %d directions\n",ngp);
  assert(ngp > 0);


  bool save_config(false);
  bool load_config(false);
  bool load_lrg(false);
  bool save_lrg(false);
  char *load_config_file;
  char *save_config_file;
  char *save_lrg_file;
  char *load_lrg_file;
  bool verbose(false);
  bool unit_gauge(false);

  int size[] = {2,2,2,2,2};
  int nthreads = 1;

  BfmSolverType solver = BFM_DWF;//DWF; HmCayleyTanh
  double mobius_scale = 2.;

  bool lanc_args_setup = false;
  Fbfm::use_mixed_solver = false;

  int i=2;
  while(i<argc){
    char* cmd = argv[i];  
    if( strncmp(cmd,"-save_config",15) == 0){
      if(i==argc-1){ printf("-save_config requires an argument\n"); exit(-1); }
      save_config=true;
      save_config_file = argv[i+1];
      i+=2;
    }else if( strncmp(cmd,"-load_config",15) == 0){
      if(i==argc-1){ printf("-save_config requires an argument\n"); exit(-1); }
      load_config=true;
      load_config_file = argv[i+1];
      i+=2;
    }else if( strncmp(cmd,"-latt",10) == 0){
      if(i>argc-6){
	printf("Did not specify enough arguments for 'latt' (require 5 dimensions)\n"); exit(-1);
      }
      for(int d=0;d<5;d++)
	size[d] = toInt(argv[i+1+d]);
      i+=6;
    }else if( strncmp(cmd,"-load_lrg",15) == 0){
      if(i==argc-1){ printf("-load_lrg requires an argument\n"); exit(-1); }
      load_lrg=true;
      load_lrg_file = argv[i+1];
      i+=2;
    }else if( strncmp(cmd,"-save_lrg",15) == 0){
      if(i==argc-1){ printf("-save_lrg requires an argument\n"); exit(-1); }
      save_lrg=true;
      save_lrg_file = argv[i+1];
      i+=2;  
    }else if( strncmp(cmd,"-verbose",15) == 0){
      verbose=true;
      i++;
    }else if( strncmp(cmd,"-nthread",15) == 0){
      nthreads = toInt(argv[i+1]);
      i+=2;
    }else if( strncmp(cmd,"-unit_gauge",15) == 0){
      unit_gauge=true;
      i++;
    }else if( strncmp(cmd,"-mobius",15) == 0){
      solver = BFM_HmCayleyTanh;
      i++;
    }else if( strncmp(cmd,"-mobius_scale",15) == 0){
      std::stringstream ss; ss >> argv[i+1];
      ss << mobius_scale;
      if(!UniqueID()) printf("Set Mobius scale to %g\n",mobius_scale);
      i+=2;
    /* }else if( strncmp(cmd,"-load_lanc_arg",20) == 0){ */
    /*   if(!lanc_arg.Decode("lanc_arg.vml","lanc_arg")){VRB.Result("","","Can't open lanc_arg.vml!\n");exit(1);} */
    /*   if(!lanc_arg_s.Decode("lanc_arg_s.vml","lanc_arg")){VRB.Result("","","Can't open lanc_arg_s.vml!\n");exit(1);} */
    /*   lanc_args_setup = true; */
    /*   i++; */
    }else if( std::string(argv[i]) == "-use_mixed_solver" ){
      Fbfm::use_mixed_solver = true;
      i++;
    }else{
      if(UniqueID()==0) printf("Unrecognised argument: %s\n",cmd);
      exit(-1);
    }
  }

  printf("Lattice size is %d %d %d %d\n",size[0],size[1],size[2],size[3],size[4]);

  CommonArg common_arg;
  DoArg do_arg;  setupDoArg(do_arg,size,ngp,verbose);

  GJP.Initialize(do_arg);

#if TARGET == BGQ
  LRG.setSerial();
#endif
  LRG.Initialize(); //usually initialised when lattice generated, but I pre-init here so I can load the state from file

  //cps_qdp_init(&argc,&argv);
  //Chroma::initialize(&argc,&argv);

  BfmArg bfm_arg;
  setup_bfmargs(bfm_arg,solver,nthreads,mobius_scale);
  init_fbfm(&argc,&argv,bfm_arg);

  if(UniqueID()==0)
    if(Fbfm::use_mixed_solver) printf("Using Fbfm mixed precision solver\n");
    else printf("Using Fbfm double precision solver\n");

  GnoneFbfm lattice;
  //GwilsonFdwf lattice;

  if(load_lrg){
    if(UniqueID()==0) printf("Loading RNG state from %s\n",load_lrg_file);
    LRG.Read(load_lrg_file,32);
  }
  if(save_lrg){
    if(UniqueID()==0) printf("Writing RNG state to %s\n",save_lrg_file);
    LRG.Write(save_lrg_file,32);
  }					       
  if(!load_config){
    printf("Creating gauge field\n");
    if(!unit_gauge) lattice.SetGfieldDisOrd();
    else lattice.SetGfieldOrd();
  }else{
    ReadLatticeParallel readLat;
    if(UniqueID()==0) printf("Reading: %s (NERSC-format)\n",load_config_file);
    readLat.read(lattice,load_config_file);
    if(UniqueID()==0) printf("Config read.\n");
  }
  if(save_config){
    if(UniqueID()==0) printf("Saving config to %s\n",save_config_file);

    QioArg wt_arg(save_config_file,0.001);
    
    wt_arg.ConcurIONumber=32;
    WriteLatticeParallel wl;
    wl.setHeader("disord_id","disord_label",0);
    wl.write(lattice,wt_arg);
    
    if(!wl.good()) ERR.General("main","()","Failed write lattice %s",save_config_file);

    if(UniqueID()==0) printf("Config written.\n");
  }

  bfm_evo<double> &dwf_d = static_cast<Fbfm&>(lattice).bd;
  bfm_evo<float> &dwf_f = static_cast<Fbfm&>(lattice).bf;
  lattice.BondCond(); //this applies the BC to the loaded field then imports it to the internal bfm instances. If you don't have the BC right the cconj relation actually fails - cute

  FixGaugeArg fix_gauge_arg; setupFixGaugeArg(fix_gauge_arg);

  AlgFixGauge fix_gauge(lattice,&common_arg,&fix_gauge_arg);
  fix_gauge.run();

  //Generate props
  double prec = 1e-6;
  ThreeMomentum p;
  for(int i=0;i<3;i++) if(GJP.Bc(i) == BND_CND_GPARITY) p(i) = 1;
  ThreeMomentum mp(-p);

  int tsrc = 0;
  PropagatorContainer &prop_f0_pplus = computePropagatorOld("prop_f0_pplus",0.01,prec,tsrc,0,p.ptr(),lattice, "prop_f1_pplus");
  PropagatorContainer &prop_f1_pplus = computePropagatorOld("prop_f1_pplus",0.01,prec,tsrc,1,p.ptr(),lattice, "prop_f0_pplus");

  PropagatorContainer &prop_f0_pminus = computePropagatorOld("prop_f0_pminus",0.01,prec,tsrc,0,mp.ptr(),lattice, "prop_f1_pminus");
  PropagatorContainer &prop_f1_pminus = computePropagatorOld("prop_f1_pminus",0.01,prec,tsrc,1,mp.ptr(),lattice, "prop_f0_pminus");

  PropManager::calcProps(lattice);

  //Check complex conj relation basic
  for(int i=0;i<GJP.VolNodeSites();i++){
    SpinColorFlavorMatrix pm_calc, pm_flip;
    //pw_prop_pplus.siteMatrix(pm_calc,i); //p- formed from p- f0 and p- f1

    pm_calc.generate(prop_f0_pplus.convert<QPropWcontainer>(), prop_f1_pplus.convert<QPropWcontainer>(), lattice,i);
    pm_flip.generate_from_cconj_pair(prop_f0_pplus.convert<QPropWcontainer>(), prop_f0_pminus.convert<QPropWcontainer>(), lattice,i);
    
    bool eq = test_equals(pm_calc,pm_flip,1e-6);
    double fail = eq ? 0. : 1.;
    glb_sum(&fail);
    if(fail != 0.0){
      if(!UniqueID()){
	print2(pm_calc,pm_flip);	
      }
      if(!UniqueID()) printf("cconj reln test plus failed\n");
      exit(-1);
    }
  }
  if(!UniqueID()) printf("cconj reln test plus passed\n");


  QPropWMomSrc* prop_f0_pplus_test = computePropagator(0.01, prec, tsrc, 0, p, lattice);
  
  compareProps(QPropWcontainer::verify_convert(prop_f0_pplus,"","").getProp(lattice), *prop_f0_pplus_test, "f0 p+ test", 1e-12);

  //Test pion 2pt against old code
  int L[4] = {GJP.XnodeSites()*GJP.Xnodes(), GJP.YnodeSites()*GJP.Ynodes(), GJP.ZnodeSites()*GJP.Znodes(), GJP.TnodeSites()*GJP.Tnodes()};


  ContractedBilinear<SpinColorFlavorMatrix> conbil;
  std::vector<Float> p_pi_plus(3,0); //sink phase exp(i p.x)
  for(int i=0;i<ngp;i++) p_pi_plus[i] = M_PI/double(L[i]);

  conbil.add_momentum(p_pi_plus);
  conbil.calculateBilinears(lattice, "prop_f0_pminus", PropDFT::Dagger, "prop_f0_pplus", PropDFT::None);

  Pion2PtSinkOp snk_op_new[] = { AX, AY, AZ, AT, P };
  int snk_spn_old[] = {1,2,4,8,0};

  PropWrapper pw_prop_pplus(&QPropWcontainer::verify_convert(prop_f0_pplus,"","").getProp(lattice),
			    &QPropWcontainer::verify_convert(prop_f1_pplus,"","").getProp(lattice));
  
  PropWrapper pw_prop_pminus(&QPropWcontainer::verify_convert(prop_f0_pminus,"","").getProp(lattice),
			     &QPropWcontainer::verify_convert(prop_f1_pminus,"","").getProp(lattice));


  for(int oo=0;oo<5;oo++){
    //sigma3(1+sigma2) = sigma3 -i sigma1
    //Note, ordering of operators in ContractedBilinear is source, sink
    const std::vector<Rcomplex> &pps3 =  conbil.getBilinear(lattice,p_pi_plus,"prop_f0_pminus", PropDFT::Dagger, "prop_f0_pplus", PropDFT::None,
							    0, 3, snk_spn_old[oo], 3);
    const std::vector<Rcomplex> &pps1 =  conbil.getBilinear(lattice,p_pi_plus,"prop_f0_pminus", PropDFT::Dagger, "prop_f0_pplus", PropDFT::None,
							    0, 1, snk_spn_old[oo], 3);
    std::vector<Rcomplex> ppconbil(pps3);
    for(int i=0;i<pps3.size();i++) ppconbil[i] = (snk_op_new[oo] == P ? -1. : 1.)*0.25*(pps3[i] + Complex(0,-1)*pps1[i]);
    
    fMatrix<double> ppnew(1,L[3]);
    pionTwoPointLWGparity(ppnew,0,snk_op_new[oo],mp,p,pw_prop_pminus,pw_prop_pplus);
    
    if(!UniqueID()) printf("Operator %d\n",oo);
    for(int t=0;t<L[3];t++){
      if(!UniqueID()) printf("t=%d, old=(%g,%g) new=(%g,%g)\n",t,std::real(ppconbil[t]),std::imag(ppconbil[t]),std::real(ppnew(0,t)),std::imag(ppnew(0,t)) );
    }
  }



  //Test 
  for(int i=0;i<GJP.VolNodeSites();i++){
    SpinColorFlavorMatrix pm_calc, pm_flip;
    pw_prop_pminus.siteMatrix(pm_calc,i); //p- formed from p- f0 and p- f1
    pm_flip.generate_from_cconj_pair(QPropWcontainer::verify_convert(prop_f0_pminus,"",""), QPropWcontainer::verify_convert(prop_f0_pplus,"",""), lattice,i);

    double fail = 0.0;

    for(int scf=0;scf<24*24;scf++){
      int rem = scf;
      int s1 = rem % 4; rem /= 4;
      int c1 = rem % 3; rem /= 3;
      int f1 = rem % 2; rem /= 2;
      int s2 = rem % 4; rem /= 4;
      int c2 = rem % 3; rem /= 3;
      int f2 = rem % 2; rem /= 2;
      
      const Complex &v_calc = pm_calc(s1,c1,f1,s2,c2,f2);
      const Complex &v_flip = pm_flip(s1,c1,f1,s2,c2,f2);
      if(fabs(std::real(v_calc - v_flip))>1e-12 ||
	 fabs(std::imag(v_calc - v_flip))>1e-12){
	if(!UniqueID()) printf("cconj reln test fail %d (%d %d %d %d %d %d): (%g, %g) vs (%g, %g)\n",i,s1,c1,f2,s2,c2,f2,std::real(v_calc),std::imag(v_calc),std::real(v_flip),std::imag(v_flip));
	fail = 1.;
      }
    }
    glb_sum(&fail);
    if(fail != 0.0){
      if(!UniqueID()) printf("cconj reln test failed\n");
      exit(-1);
    }
  }
  if(!UniqueID()) printf("Passed cconj reln test 2 (-p)\n");

  PropWrapper pw_prop_pminus_fromflip(&QPropWcontainer::verify_convert(prop_f0_pplus,"","").getProp(lattice),
				      &QPropWcontainer::verify_convert(prop_f1_pplus,"","").getProp(lattice), true);

  //Test the mom flip
  for(int i=0;i<GJP.VolNodeSites();i++){
    SpinColorFlavorMatrix pm_calc, pm_flip;
    pw_prop_pminus.siteMatrix(pm_calc,i);
    pw_prop_pminus_fromflip.siteMatrix(pm_flip,i);

    double fail = 0.0;

    for(int scf=0;scf<24*24;scf++){
      int rem = scf;
      int s1 = rem % 4; rem /= 4;
      int c1 = rem % 3; rem /= 3;
      int f1 = rem % 2; rem /= 2;
      int s2 = rem % 4; rem /= 4;
      int c2 = rem % 3; rem /= 3;
      int f2 = rem % 2; rem /= 2;
      
      const Complex &v_calc = pm_calc(s1,c1,f1,s2,c2,f2);
      const Complex &v_flip = pm_flip(s1,c1,f1,s2,c2,f2);
      if(fabs(std::real(v_calc - v_flip))>1e-12 ||
	 fabs(std::imag(v_calc - v_flip))>1e-12){
	if(!UniqueID()) printf("Flip test fail %d (%d %d %d %d %d %d): (%g, %g) vs (%g, %g)\n",i,s1,c1,f2,s2,c2,f2,std::real(v_calc),std::imag(v_calc),std::real(v_flip),std::imag(v_flip));
	fail = 1.;
      }
    }
    glb_sum(&fail);
    if(fail != 0.0){
      if(!UniqueID()) printf("Flip test failed\n");
      exit(-1);
    }
  }
  if(!UniqueID()) printf("Passed mom flip test\n");


// void pionTwoPointLWGparity(fMatrix<double> &into, const int tsrc, const Pion2PtSinkOp sink_op, const ThreeMomentum &p1, const ThreeMomentum &p2,
// 			   const PropWrapper &prop1, const PropWrapper &prop2,
// 			   const bool use_wrong_proj_sign = false){

  










  //   //Gauge fix lattice if required
  //   if(ama_arg.fix_gauge.fix_gauge_kind != FIX_GAUGE_NONE){
  //     AlgFixGauge fix_gauge(lattice,&carg,&ama_arg.fix_gauge);
  //     fix_gauge.run();
  //   }

  //   //Generate eigenvectors
  //   Float time = -dclock();
  //   BFM_Krylov::Lanczos_5d lanc_l(dwf_d, lanc_arg_l);
  //   lanc_l.Run();
  //   if(Fbfm::use_mixed_solver){
  //     //Convert eigenvectors to single precision
  //     lanc_l.toSingle();
  //   }
  //   time += dclock();    
  //   print_time("main","Light quark Lanczos",time);

  //   time = -dclock();
  //   BFM_Krylov::Lanczos_5d lanc_h(dwf_d, lanc_arg_h);
  //   lanc_h.Run();
  //   if(Fbfm::use_mixed_solver){
  //     //Convert eigenvectors to single precision
  //     lanc_h.toSingle();
  //   }
  //   time += dclock();    
  //   print_time("main","Heavy quark Lanczos",time);
 
  //   //We want stationary mesons and moving mesons. For GPBC there are two inequivalent directions: along the G-parity axis and perpendicular to it. 
  //   PropMomContainer props; //stores generated propagators by tag

  //   bool do_alternative_mom = true;

  //   //Decide on the meson momenta we wish to compute
  //   MesonMomenta pion_momenta;
  //   PionMomenta::setup(pion_momenta,do_alternative_mom);
    
  //   MesonMomenta su2_singlet_momenta;
  //   LightFlavorSingletMomenta::setup(su2_singlet_momenta);

  //   MesonMomenta kaon_momenta;
  //   KaonMomenta::setup(kaon_momenta,do_alternative_mom);

  //   //Determine the quark momenta we will need
  //   QuarkMomenta light_quark_momenta;
  //   QuarkMomenta heavy_quark_momenta;
    
  //   pion_momenta.appendQuarkMomenta(Light, light_quark_momenta); //adds the quark momenta it needs
  //   su2_singlet_momenta.appendQuarkMomenta(Light, light_quark_momenta);
  //   kaon_momenta.appendQuarkMomenta(Light, light_quark_momenta); //each momentum is unique
  //   kaon_momenta.appendQuarkMomenta(Heavy, heavy_quark_momenta);

  //   const int Lt = GJP.Tnodes()*GJP.TnodeSites();

  //   double sloppy_prec, exact_prec;
  //   double ml, mh;
  //   std::string results_dir;

  //   std::vector<int> tslice_sloppy;
  //   std::vector<int> tslice_exact;

  //   for(int status = 0; status < 2; status++){ //sloppy, exact
  //     PropPrecision pp = status == 0 ? Sloppy : Exact;
  //     const std::vector<int> &tslices = status == 0 ? tslice_sloppy : tslice_exact;
  //     double prec = status == 0 ? sloppy_prec : exact_prec;
      
  //     //Light-quark inversions
  //     lightQuarkInvert(props, pp, prec,ml,tslices,light_quark_momenta,lattice,lanc_l);

  //     //Pion 2pt LW functions pseudoscalar and axial sinks	     
  //     measurePion2ptLW(props,pp,tslices,ll_meson_momenta);

  //     //Pion 2pt WW function pseudoscalar sink
  //     measurePion2ptPPWW(props,pp,tslices,ll_meson_momenta,lattice);

  //     props.clear(); //delete all propagators thus far computed
  //   }
  // }//end of conf loop

  if(UniqueID()==0){
    printf("Main job complete\n"); 
    fflush(stdout);
  }
  return 0;
}





CPS_END_NAMESPACE

#endif
