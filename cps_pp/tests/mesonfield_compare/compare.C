#include<alg/a2a/a2a.h>
#include<alg/a2a/mesonfield.h>

using namespace cps;

template<typename T>
inline T toType(const char* a){
  std::stringstream ss; ss << a; T o; ss >> o;
  return o;
}

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



int main(int argc,char *argv[])
{
  Start(&argc, &argv);
  assert(argc == 6);

  const int Lt = toType<int>(argv[1]);
  const int ngp = toType<int>(argv[2]);
  const double tolerance = toType<double>(argv[3]);
  
  std::string file1 = argv[4];
  std::string file2 = argv[5];


  int size[] = {2,2,2,Lt,2};
  DoArg do_arg;  setupDoArg(do_arg,size,ngp,false);
  GJP.Initialize(do_arg);
  
  typedef A2ApoliciesDoubleAutoAlloc A2Apolicies;
  typedef A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> MfType;
  typedef std::vector<MfType> MfVector;
  
  MfVector mf1, mf2;

  MfType::read(file1, mf1);
  MfType::read(file2, mf2);

  assert(mf1.size() == mf2.size());
  
  typedef typename MfType::ScalarComplexType ComplexType;
  typedef typename ComplexType::value_type RealType;

  for(int t=0;t<mf1.size();t++){
    RealType const* mf1_ptr = (RealType const*)mf1[t].ptr();
    RealType const* mf2_ptr = (RealType const*)mf2[t].ptr();

    for(int i=0;i<2*mf1[t].size();i++){
      RealType reldiff = fabs(2.*(mf1_ptr[i] - mf2_ptr[i])/(mf1_ptr[i] + mf2_ptr[i]));
      if(tolerance == 0 || reldiff > tolerance) printf("t %d i %d : %g %g <-> %g\n",t,i,mf1_ptr[i],mf2_ptr[i],reldiff);
    }
  }
  
  printf("Finished\n"); fflush(stdout);
  
  return 0;
}




