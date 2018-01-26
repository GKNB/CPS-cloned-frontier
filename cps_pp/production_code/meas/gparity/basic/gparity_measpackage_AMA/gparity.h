#ifndef _GPARITY_H
#define _GPARITY_H

#include <string>
#include <vector>
#include <memory>
#include <util/gjp.h>
#include <alg/do_arg.h>
#include <alg/bfm_arg.h>
#include <alg/lanc_arg.h>
#include <alg/fix_gauge_arg.h>
#include <alg/gparity_contract_arg.h>
#include <alg/eigen/Krylov_5d.h>
#include <util/lattice/fbfm.h>
#include <util/ReadLatticePar.h>
CPS_START_NAMESPACE

template<class T>
void decode_vml(const std::string &directory, const std::string &arg_name, T&into){
  std::string file = directory + std::string("/") + arg_name + std::string(".vml");
  if(!UniqueID()) printf("Decoding %s.vml from directory %s\n",arg_name.c_str(),directory.c_str() ); fflush(stdout); 
  if ( ! into.Decode(const_cast<char*>(file.c_str()), const_cast<char*>(arg_name.c_str()) ) ){
    std::string templ = directory + std::string("/") + arg_name + std::string(".templ");
    into.Encode(const_cast<char*>(templ.c_str()),const_cast<char*>(arg_name.c_str()));
    ERR.General("", "decode_vml", "Could not read %s\n",file.c_str());
  }
}


void decode_vml_all(DoArg &do_arg, BfmArg &bfm_arg, LancArg &lanc_arg_l, LancArg &lanc_arg_h, GparityAMAarg2 &ama_arg, const std::string &script_dir){
  const char* cname = "";
  const char *fname = "decode_vml_all()";
  decode_vml(script_dir,"do_arg",do_arg);
  decode_vml(script_dir,"bfm_arg",bfm_arg);
  decode_vml(script_dir,"lanc_arg_l",lanc_arg_l);
  decode_vml(script_dir,"lanc_arg_h",lanc_arg_h);
  decode_vml(script_dir,"ama_arg",ama_arg);
}

void check_bk_tsources(GparityAMAarg2 &ama_arg){
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();
  const int lens[2] = { ama_arg.exact_solve_timeslices.exact_solve_timeslices_len, ama_arg.sloppy_solve_timeslices.sloppy_solve_timeslices_len };
  const int* vals[2] = { ama_arg.exact_solve_timeslices.exact_solve_timeslices_val, ama_arg.sloppy_solve_timeslices.sloppy_solve_timeslices_val };

  for(int p=0;p<2;p++){
    std::string status = p==0 ? "exact" : "sloppy";
    if(!UniqueID()) printf("Checking %s BK sources\n",status.c_str());

    if(lens[p] == 0){
      if(!UniqueID()) printf("Skipping %s BK as no sources\n",status.c_str());
      continue;
    }

    std::vector<int> tsep_meas_count(ama_arg.bk_tseps.bk_tseps_len, 0);

    for(int tt=0;tt<lens[p];tt++){
      int t0 = vals[p][tt];
      for(int tsi=0;tsi<ama_arg.bk_tseps.bk_tseps_len;tsi++){
	int tsep = ama_arg.bk_tseps.bk_tseps_val[tsi];
	int t1 = (t0 + tsep) % Lt;
	bool found = false;
	for(int tt2=0;tt2<lens[p];tt2++) if(vals[p][tt2]==t1){ found = true; break; }
	if(found) tsep_meas_count[tsi]++;

	if(!UniqueID()) printf("BK t0=%d tsep=%d, want t1=%d, status %d, count now %d\n",t0,tsep,t1,found,tsep_meas_count[tsi]);
      }
    }

    for(int tsi=0;tsi<ama_arg.bk_tseps.bk_tseps_len;tsi++){
      int tsep = ama_arg.bk_tseps.bk_tseps_val[tsi];
      if(!UniqueID()) printf("BK with tsep %d will have %d translations\n",tsep,tsep_meas_count[tsi]);
      if(tsep_meas_count[tsi] == 0){
	if(!UniqueID()){
	  printf("Error: BK time sep %d and %s props, no measurements will be performed due to lack of source timeslices\n",tsep,status.c_str());
	  std::cout.flush();
	}
	exit(-1);
      }
    }
  }
}

void init_fbfm(int *argc, char **argv[], const BfmArg &args)
{
  if(!UniqueID()) printf("Initializing Fbfm\n");
  // /*! IMPORTANT: BfmSolverType is not the same as the BfmSolver in the bfm package. BfmSolverType is defined in enum.x. Basically it adds a BFM_ prefix to the corresponding names in the BfmSolver enum. */
  cps_qdp_init(argc,argv);
//Chroma::initialize(argc, argv);
  multi1d<int> nrow(Nd);
  
  for(int i = 0; i< Nd; ++i)
    nrow[i] = GJP.Sites(i);
  
  Layout::setLattSize(nrow);
  Layout::create();

  if(args.solver == BFM_HmCayleyTanh){
    Fbfm::bfm_args[0].solver = HmCayleyTanh;
  }else if(args.solver == BFM_DWF){
    Fbfm::bfm_args[0].solver = DWF;
  }else ERR.General("","init_bfm","CPS solver enum correspondance to bfm enum has not been added for input solver type\n");

  Fbfm::bfm_args[0].precon_5d = args.precon_5d;
  
  Fbfm::bfm_args[0].Ls = GJP.SnodeSites();
  Fbfm::bfm_args[0].M5 = GJP.DwfHeight();
  Fbfm::bfm_args[0].mass = args.mass;
  Fbfm::bfm_args[0].residual = args.residual;
  Fbfm::bfm_args[0].max_iter = args.max_iter;
  Fbfm::bfm_args[0].Csw = args.Csw;
  
  Fbfm::bfm_args[0].node_latt[0] = QDP::Layout::subgridLattSize()[0];
  Fbfm::bfm_args[0].node_latt[1] = QDP::Layout::subgridLattSize()[1];
  Fbfm::bfm_args[0].node_latt[2] = QDP::Layout::subgridLattSize()[2];
  Fbfm::bfm_args[0].node_latt[3] = QDP::Layout::subgridLattSize()[3];

  multi1d<int> procs = QDP::Layout::logicalSize();
  multi1d<int> ncoor = QDP::Layout::nodeCoord();

  Fbfm::bfm_args[0].local_comm[0] = procs[0] > 1 ? 0 : 1;
  Fbfm::bfm_args[0].local_comm[1] = procs[1] > 1 ? 0 : 1;
  Fbfm::bfm_args[0].local_comm[2] = procs[2] > 1 ? 0 : 1;
  Fbfm::bfm_args[0].local_comm[3] = procs[3] > 1 ? 0 : 1;

  for(int i=0;i<4;i++) Fbfm::bfm_args[0].ncoor[i] = ncoor[i];

  if(GJP.Gparity()){
    Fbfm::bfm_args[0].gparity = 1;
    if(!UniqueID()) printf("G-parity directions: ");
    for(int d=0;d<3;d++)
      if(GJP.Bc(d) == BND_CND_GPARITY){ Fbfm::bfm_args[0].gparity_dir[d] = 1; printf("%d ",d); }
      else Fbfm::bfm_args[0].gparity_dir[d] = 0;
    for(int d=0;d<4;d++){
      Fbfm::bfm_args[0].nodes[d] = procs[d];
    }
    printf("\n");
  }else if(!UniqueID()) printf("Standard boundary conditions\n");

  //Fbfm::current_arg_idx = 0;
#ifdef USE_NEW_BFM_GPARITY
  Fbfm::bfm_args[0].threads = args.threads;
  Fbfm::bfm_args[0].mobius_scale = args.mobius_scale;  // mobius_scale = b + c in Andrew's notation
#else
  bfmarg::Threads(args.threads);
  bfmarg::mobius_scale = args.mobius_scale;
#endif
  omp_set_num_threads(args.threads);

  bfmarg::Reproduce(0);
  bfmarg::ReproduceChecksum(0);
  bfmarg::ReproduceMasterCheck(0);
  bfmarg::Verbose(args.verbose);
}


inline int toInt(const char* a){
  std::stringstream ss; ss << a; int o; ss >> o;
  return o;
}

inline bool contains_pctd(const std::string &c){
  return c.find("%d") != std::string::npos;
}

void getTimeslices(std::vector<int> &tslice_sloppy, std::vector<int> &tslice_exact, std::vector<int> &bk_tseps, const GparityAMAarg2 &ama_arg, bool random_exact_tsrc_offset){
  tslice_sloppy = std::vector<int>(ama_arg.sloppy_solve_timeslices.sloppy_solve_timeslices_val, 
				   ama_arg.sloppy_solve_timeslices.sloppy_solve_timeslices_val + ama_arg.sloppy_solve_timeslices.sloppy_solve_timeslices_len);
  tslice_exact = std::vector<int>(ama_arg.exact_solve_timeslices.exact_solve_timeslices_val, 
				  ama_arg.exact_solve_timeslices.exact_solve_timeslices_val + ama_arg.exact_solve_timeslices.exact_solve_timeslices_len);
    
  bk_tseps = std::vector<int> (ama_arg.bk_tseps.bk_tseps_val, ama_arg.bk_tseps.bk_tseps_val + ama_arg.bk_tseps.bk_tseps_len);

  const int Lt = GJP.Tnodes()*GJP.TnodeSites();
  if(random_exact_tsrc_offset){
    int offset = int(floor( LRG.Lrand(Lt,0) )) % Lt;
    if(!UniqueID()) printf("Shifting exact src timeslices by offset %d\n",offset);
    for(int i=0;i<tslice_exact.size();i++){
      int nval = (tslice_exact[i]+offset) % Lt;
      if(!UniqueID()) printf("Exact src timeslice %d -> %d\n",tslice_exact[i], nval);
      tslice_exact[i] = nval;
    }
  }
}
    
typedef std::auto_ptr<BFM_Krylov::Lanczos_5d<double> > LanczosPtrType;

inline std::auto_ptr<BFM_Krylov::Lanczos_5d<double> > doLanczos(GnoneFbfm &lattice, const LancArg lanc_arg, const BndCndType time_bc){
  if(lanc_arg.N_get == 0) return std::auto_ptr<BFM_Krylov::Lanczos_5d<double> >(NULL);

  BndCndType init_tbc = GJP.Tbc();
  BndCndType target_tbc = time_bc;

  GJP.Bc(3,target_tbc);
  lattice.BondCond();  //Apply BC to internal gauge fields

  bfm_evo<double> &dwf_d = static_cast<Fbfm&>(lattice).bd;
  std::auto_ptr<BFM_Krylov::Lanczos_5d<double> > ret(new BFM_Krylov::Lanczos_5d<double>(dwf_d, const_cast<LancArg&>(lanc_arg)));
  ret->Run();
  if(Fbfm::use_mixed_solver){
    //Convert eigenvectors to single precision
    ret->toSingle();
  }

  //Restore the BCs
  lattice.BondCond();  //unapply BC
  GJP.Bc(3,init_tbc);

  return ret;
}

inline void freeLanczos(std::auto_ptr<BFM_Krylov::Lanczos_5d<double> > &lanc){
  if(lanc.get() == NULL) return;

  if(Fbfm::use_mixed_solver){
    BFM_Krylov::Lanczos_5d<double>* eig = lanc.get();
    //Evecs were converted to single in place which changes their size. Thus we need to manually dealloc in order to prevent a crash
    
    int words = 24 * eig->dop.node_cbvol * eig->dop.cbLs * (eig->dop.gparity ? 2:1);
    int bytes = words*sizeof(float);
	
    for(int i = 0; i < eig->bq.size(); i++){
      for(int cb=eig->prec;cb<2;cb++)
	if(eig->bq[i][cb] != NULL){
	  bfm_free(eig->bq[i][cb],bytes);
	  eig->bq[i][cb] = NULL;
	}
    }
    eig->bq.resize(0);
  }
  lanc.reset();
}


//Read/generate the gauge configuration and RNG
void readLatticeAndRNG(Lattice &lattice, const CmdLine &cmdline, const DoArg &do_arg, const GparityAMAarg2 &ama_arg, const int conf){
  char load_config_file[1000];
  char load_rng_file[1000];
  
  if(do_arg.start_conf_kind == START_CONF_ORD || cmdline.rng_test){
    if(!UniqueID()) printf("Using unit gauge links\n");
    lattice.SetGfieldOrd();
  }else if(do_arg.start_conf_kind == START_CONF_DISORD){
    if(!UniqueID()) printf("Using random gauge links\n");
    lattice.SetGfieldDisOrd();
    printf("Gauge checksum = %d\n", lattice.CheckSum());
  }else if(do_arg.start_conf_kind == START_CONF_FILE){    
    if(sprintf(load_config_file,ama_arg.config_fmt,conf) < 0){
      ERR.General("","main()","Configuration filename creation problem : %s | %s",load_config_file,ama_arg.config_fmt);
    }
    //load the configuration
    ReadLatticeParallel readLat;
    if(UniqueID()==0) printf("Reading: %s (NERSC-format)\n",load_config_file);
    if(cmdline.dbl_latt_storemode){
      if(!UniqueID()) printf("Disabling U* field reconstruction\n");
      readLat.disableGparityReconstructUstarField();
    }
    readLat.read(lattice,load_config_file);
  }else{
    ERR.General("","main()","Invalid do_arg.start_conf_kind\n");
  }

  if(do_arg.start_seed_kind == START_SEED_FILE){   
    if(sprintf(load_rng_file,ama_arg.rng_fmt,conf) < 0){
      ERR.General("","main()","RNG filename creation problem : %s | %s",load_rng_file,ama_arg.rng_fmt);
    }
    if(UniqueID()==0) printf("Loading RNG state from %s\n",load_rng_file);
    int default_concur=0;
#if TARGET==BGQ
    default_concur=1;
#endif
    LRG.Read(load_rng_file,default_concur);
    if(UniqueID()==0) printf("RNG read.\n");
  }

#ifdef DOUBLE_TLATT
  if(!UniqueID()) printf("Doubling lattice temporal size\n");
  LatticeTimeDoubler doubler;
  doubler.doubleLattice(lattice,do_arg);
#endif
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();

  if(cmdline.tshift != 0)
    Tshift4D( (Float*)lattice.GaugeField(), 4*3*3*2, cmdline.tshift); //do optional temporal shift
}

		       



CPS_END_NAMESPACE


#endif
