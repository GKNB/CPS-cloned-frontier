#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sstream>
#include <vector>

#include <alg/enum.h>
#include <util/lattice.h>
#include <util/lattice/lattice_types.h>
#include <util/ReadLatticePar.h>
#include <util/WriteLatticePar.h>

USING_NAMESPACE_CPS

//String copy to *new* buffer
void stringCopy(char* &into, const std::string &from){
  into = (char*)malloc( (from.size()+1)*sizeof(char) );
  strcpy(into, from.c_str());
}

void initializeDoArg(DoArg &do_arg, const std::vector<int> &latt, const std::vector<cps::BndCndType> &bcs, const std::string &gauge_file, const std::string &rng_file){
  do_arg.x_sites = latt[0];
  do_arg.y_sites = latt[1];
  do_arg.z_sites = latt[2];
  do_arg.t_sites = latt[3];
  do_arg.s_sites = latt[4];
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
  do_arg.x_bc = bcs[0];
  do_arg.y_bc = bcs[1];
  do_arg.z_bc = bcs[2];
  do_arg.t_bc = bcs[3];
  do_arg.start_conf_kind = START_CONF_FILE;
  do_arg.start_conf_load_addr = 0x0;
  do_arg.start_seed_kind = START_SEED_FILE;
  
  stringCopy(do_arg.start_seed_filename, rng_file);
  stringCopy(do_arg.start_conf_filename, gauge_file);

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
}

BndCndType parseBC(const std::string &bc){
  if(bc == "PRD") return BND_CND_PRD;
  else if(bc == "APRD") return BND_CND_APRD;
  else if(bc == "GPARITY") return BND_CND_GPARITY;
  else{
    ERR.General("","parseBC","Unsupported boundary condition %s\n",bc.c_str());
  }
}


enum BndCndType {
	BND_CND_PRD = 0,
	BND_CND_APRD = 1,
	BND_CND_TWISTED = 2,
	BND_CND_GPARITY = 3,
	BND_CND_GPARITY_TWISTED = 4,
};

int main(int argc,char *argv[])
{
  Start(&argc,&argv); //initialises QMP
  const int narg_req = 1 + 5 + 4 + 1 + 1; //exe, latt5, bc, gauge file, rng file
  
  if(argc != narg_req)
    ERR.General("","main","Require lattice (5 ints), boundary conditions (4 strings {PRD, APRD, GPARITY}), gauge file, rng file\n");

  int aidx = 1;
  
  std::vector<int> latt5(5);
  std::cout << "Lattice:";
  for(int i=0;i<5;i++){
    std::stringstream ss; ss << argv[aidx++]; ss >> latt5[i];
    std::cout << " " << latt5[i];
  }
  std::cout << std::endl;

  cps::BndCndType test;
  std::vector<cps::BndCndType> bcs(4);
  std::cout << "BCs:";
  for(int i=0;i<4;i++){
    bcs[i] = parseBC(argv[aidx++]);
    std::cout << " " << BndCndType_map[(int)bcs[i]].name;
  }
  std::cout << std::endl;

  std::string gauge_file = argv[aidx++];
  std::string rng_file = argv[aidx++];

  DoArg do_arg;
  initializeDoArg(do_arg,latt5,bcs,gauge_file,rng_file);

  GJP.Initialize(do_arg);

  GnoneFnone latt; //will load the gauge file and check cksum
  LRG.Initialize();
  LRG.Read(rng_file.c_str(),32);
  
  End();
  
  if(UniqueID()==0){
    printf("Check passed\n"); 
    fflush(stdout);
  }
  return 0;
}


