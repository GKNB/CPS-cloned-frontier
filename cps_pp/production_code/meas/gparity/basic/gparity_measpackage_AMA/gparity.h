#ifndef _GPARITY_H
#define _GPARITY_H

CPS_START_NAMESPACE

template<class T>
void decode_vml(const std::string &directory, const std::string &arg_name, T&into){
  std::string file = directory + std::string("/") + arg_name + std::string(".vml");
  if(!UniqueID()) printf("Decoding %s.vml from directory %s\n",arg_name.c_str(),directory.c_str() ); fflush(stdout); 
  if ( ! into.Decode(const_cast<char*>(file.c_str()), const_cast<char*>(arg_name.c_str()) ) )
    ERR.General("", "decode_vml", "Could not read %s\n",file.c_str());
}


void decode_vml_all(DoArg &do_arg, BfmArg &bfm_arg, LancArg &lanc_arg_l, LancArg &lanc_arg_h, GparityAMAarg &ama_arg, const std::string &script_dir){
  const char* cname = "";
  const char *fname = "decode_vml_all()";
  decode_vml(script_dir,"do_arg",do_arg);
  decode_vml(script_dir,"bfm_arg",bfm_arg);
  decode_vml(script_dir,"lanc_arg_l",lanc_arg_l);
  decode_vml(script_dir,"lanc_arg_h",lanc_arg_h);
  decode_vml(script_dir,"ama_arg",ama_arg);
}

void init_fbfm(int *argc, char **argv[], const BfmArg &args)
{
  if(!UniqueID()) printf("Initializing Fbfm\n");
  // /*! IMPORTANT: BfmSolverType is not the same as the BfmSolver in the bfm package. BfmSolverType is defined in enum.x. Basically it adds a BFM_ prefix to the corresponding names in the BfmSolver enum. */
  cps_qdp_init(argc,argv);
  Chroma::initialize(argc, argv);
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

  // mobius_scale = b + c in Andrew's notation
  bfmarg::mobius_scale = args.mobius_scale;

  //Fbfm::current_arg_idx = 0;

  bfmarg::Threads(args.threads); 
  omp_set_num_threads(args.threads);

  bfmarg::Reproduce(0);
  bfmarg::ReproduceChecksum(0);
  bfmarg::ReproduceMasterCheck(0);
  bfmarg::Verbose(args.verbose);
}









// static std::string momTag(const int p[3]){
//   std::ostringstream os;
//   os << 'p';
//   for(int i=0;i<3;i++){
//     if(p[i] < 0) os << '_';
//     os << p[i];
//   }
//   return os.str();
// }







CPS_END_NAMESPACE


#endif
