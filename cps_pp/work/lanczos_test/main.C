#include<cps.h>
#include "twisted_bc.h"


#undef encode_vml
#define encode_vml(arg_name) do{                                  \
        if( !arg_name.Encode(#arg_name".dat", #arg_name) ){                    \
            ERR.General(cname, fname, #arg_name " encoding failed.\n"); \
        }                                                               \
    }while(0)

#undef decode_vml
#define decode_vml(arg_name)  do{                                       \
        if ( ! arg_name.Decode(CommandLine::arg(), #arg_name) )            \
            ERR.General(cname, fname, "Bad " #arg_name ".vml.\n");      \
    } while(0)



USING_NAMESPACE_CPS DoArg do_arg;
DoArgExt doext_arg;
LanczosArg lanczos_arg;
QPropWArg qpropw_arg;
FixGaugeArg fix_gauge_arg;
FloatArray l_twist_arg;
//const char *evec_dir;

void CalMesons (QPropW & q, const char *out_dir, int traj, char *src_str)
{

  char file[256];
  int status;
  if ((status = mkdir (out_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)) != 0)
    VRB.Result ("", "CalMeons()", "mkdir %s returned %d\n", out_dir, status);
  sprintf (file, "%s/meson_%s.dat.%d", out_dir, src_str, traj);
  FILE *fp = Fopen (&file[0], "w");

  int mu, nu;
  //vector and scalar
  for (mu = 0; mu < 4; mu++) {
    char str[256];
    sprintf (str, "GAM_%d", mu + 1);
    Meson mes (mu, str);
    mes.Zero ();
    mes.setMass (q.Mass (), q.Mass ());
    mes.calcMeson (q, q);
    if (!UniqueID ())
      mes.Print (fp);
  }

  //pseudoscalar
  {
    mu = -5;
    char str[256];
    sprintf (str, "GAM_5");
    Meson mes (mu, str);
    mes.Zero ();
    mes.setMass (q.Mass (), q.Mass ());
    mes.calcMeson (q, q);
    if (!UniqueID ())
      mes.Print (fp);
  }
  //tensor
  for (mu = 0; mu < 4; mu++) {
    for (nu = mu + 1; nu < 4; nu++) {
      char str[256];
      sprintf (str, "GAM_%d%d", mu + 1, nu + 1);
      Meson mes1 (mu, nu, str);
      mes1.Zero ();
      mes1.setMass (q.Mass (), q.Mass ());
      mes1.calcMeson (q, q);
      if (!UniqueID ())
        mes1.Print (fp);

      sprintf (str, "GAM_%d%d", nu + 1, mu + 1);
      Meson mes2 (nu, mu, str);
      mes2.Zero ();
      mes2.setMass (q.Mass (), q.Mass ());
      mes2.calcMeson (q, q);
      if (!UniqueID ())
        mes2.Print (fp);
    }
  }
  //axialvector
  nu = -5;
  for (mu = 0; mu < 4; mu++) {
    char str[256];
    sprintf (str, "GAM_%d5", mu + 1);
    Meson mes (mu, nu, str);
    mes.Zero ();
    mes.setMass (q.Mass (), q.Mass ());
    mes.calcMeson (q, q);
    if (!UniqueID ())
      mes.Print (fp);
  }

  // FIXME:  PUT IN THE  A0  SCALAR MESON, or WE WILL MISS A LOT !!

  Fclose (fp);

}

void CalNucleons (QPropW & q, const char *out_dir, int traj, char *src_str)
{
  Nuc2pt nuc (NUC_G5C, POINT);
  nuc.calcNucleon (q);

  char file[256];
  sprintf (file, "%s/nucleon_%s.dat.%d", out_dir, src_str, traj);
  FILE *fp = Fopen (&file[0], "w");

  {
    Nuc2pt nuc (NUC_G5C, POINT);
    nuc.calcNucleon (q);
    nuc.Print (fp);
  }

  Fclose (fp);
}





static int nx, ny, nz, nt, ns;
//static CgArg cg_arg;


static void SetZmobiusPC (int flag)
{
  switch (flag) {
  case 0:
    GJP.ZMobius_PC_Type (ZMOB_PC_ORIG);
    break;
  case 1:
    GJP.ZMobius_PC_Type (ZMOB_PC_SYM1);
    break;
  case 2:
    GJP.ZMobius_PC_Type (ZMOB_PC_SYM2);
    break;
  default:
    ERR.General ("", "SetZmobiusPC()", "%d Not implemented\n", flag);
  }
}

void run_lanczos (Lattice & lat, const char *evec_dir, const char * result_dir,
                  const Float * twist = NULL);

int main (int argc, char *argv[])
{

  const char *cname = "";
  const char *fname = "main()";

  Start (&argc, &argv);
#pragma omp parallel default(shared)
  {
    int tnum = omp_get_num_threads ();


#pragma omp for
    for (int i = 0; i < 10; i++) {
      if (!UniqueID ())
        printf ("thread %d of %d i=%d\n", omp_get_thread_num (), tnum, i);
    }
  }


  //----------------------------------------------------------------
  // Initializes all Global Job Parameters
  //----------------------------------------------------------------
  char *out_file = NULL;
  CommandLine::is (argc, argv);
  chdir (CommandLine::arg ());
//  if (argc > 1)
  out_file = CommandLine::arg ();

  decode_vml (do_arg);
  encode_vml (do_arg);

  decode_vml (doext_arg);
  encode_vml (doext_arg);

  decode_vml (lanczos_arg);
  encode_vml (lanczos_arg);

  decode_vml (qpropw_arg);
  encode_vml (qpropw_arg);

  decode_vml (fix_gauge_arg);
  encode_vml (fix_gauge_arg);

#ifdef USE_QUDA
  if (!QudaParam.Decode (CommandLine::arg (), "QudaParam")) {
    printf ("Bum quda_arg\n");
    exit (-1);
  }
#endif

  SetZmobiusPC (CommandLine::arg_as_int ());

//  evec_dir = CommandLine::arg ();

  const double *ltw = NULL;
//  if (!CommandLine::end ()) {
    decode_vml (l_twist_arg);
    encode_vml (l_twist_arg);
    assert (l_twist_arg.Floats.Floats_len == 4);
    ltw = l_twist_arg.Floats.Floats_val;
//  }


  GJP.Initialize (do_arg);
  GJP.InitializeExt (doext_arg);
  LRG.Initialize ();

if(0)
{
   GnoneFnone lat;
   WriteLatticeParallel(lat,"lat.out");
}


  {
//    GwilsonFmobius lat;
    FgridParams fgp;
    fgp.mobius_scale = GJP.GetMobius();
    VRB.Result("","main()","Grid b+c=%g\n",fgp.mobius_scale);
    GnoneFgridMobius lat(fgp);
{
  const char *gfix_in = "gfix_lat.in";
  const char *gfix_out = "gfix_lat.out";
  CommonArg com_fg ("fg.out");
  AlgFixGauge fg (lat, &com_fg, &fix_gauge_arg);
  FILE * fp = fopen (gfix_in, "r");
  if (!fp) {
    fg.run ();
  } else {
    QioArg rd_arg (gfix_in);
    fg.run (&rd_arg);
  }
  fg.Save (gfix_out, 0);
}
    run_lanczos (lat,  "evec", "light");
    run_lanczos (lat,  "evec_tw", "light_tw", ltw);
  }

  End ();
}

void run_lanczos (Lattice & lat, const char *evec_dir, const char * result_dir,
                  const Float * twist  )
{
  const char *fname = "run_lanczos()";
  double dtime;

  int s_size = 1;
  if (lat.F5D ())
    s_size = GJP.SnodeSites ();

  int s[5];

  if (twist)
    twistBc (lat, twist);
  int N_evec = lanczos_arg.nt_lanczos_vectors;
  VRB.Result("",fname,"N_evec=%d\n",N_evec);
//  exit(-2);
  Vector **out=NULL;
  Float *lambda=NULL;
  std::vector < Vector * >v_out;
  std::vector < Float > v_eval;
  float *out_f;
  size_t total_size = (size_t) (GJP.VolNodeSites () / 2) * lat.FsiteSize ();
  size_t evec_size = total_size;
  Float *out_d;
if (N_evec>0){
  v_out.resize(N_evec,NULL);
  v_eval.resize(N_evec,NULL);
  lambda = v_eval.data ();
  out = v_out.data ();

  if (evec_size != lat.half_size)
    ERR.General ("", fname, "evec_size(%d)!=half_size(%d)\n");
  out_d = (Float *) smalloc (evec_size * sizeof (Float) * N_evec);
  out_f = (float *) out_d;
  for (int i = 0; i < N_evec; i++) {
    if (lanczos_arg.precision == PREC_SINGLE)
      out[i] = (Vector *) (out_f + i * evec_size);
    else
      out[i] = (Vector *) (out_d + i * evec_size);
  }
}

  Float true_res;

//   const char *evec_meta ="evec/metadata.txt";
  const int MAX_LEN = 256;
  char evec_meta[MAX_LEN];
  snprintf (evec_meta, MAX_LEN, "%s/metadata.txt", evec_dir);
  VRB.Result("",fname,"evec_meta=%s\n",evec_meta);
  char res_e[MAX_LEN];
  snprintf (res_e, MAX_LEN, "%s_exact", result_dir);
  VRB.Result("",fname,"res_e=%s\n",res_e);
  char res_sloppy[MAX_LEN];
  snprintf (res_sloppy, MAX_LEN, "%s_sloppy", result_dir);
  VRB.Result("",fname,"res_sloppy=%s\n",res_sloppy);
  char res_s[MAX_LEN];
  snprintf (res_s, MAX_LEN, "%s_s", result_dir);
  VRB.Result("",fname,"res_s=%s\n",res_s);


  int offset = GJP.VolNodeSites () * lat.FsiteSize () / (2 * 6);

  FILE *fp = fopen (evec_meta, "r");
  if (!fp && N_evec>0) {

    dtime = -dclock ();
    int iter = lat.FeigSolv (out, lambda, &lanczos_arg, CNV_FRM_NO);
    dtime += dclock ();
    print_time ("", "FeigSolv()", dtime);


    EvecWriter writer;
    evec_write args;
    for (int i = 0; i < 4; i++)
      args.b[i] = 2;
    args.b[4] = GJP.NodeSites (4);
#if 0
    args.nkeep = N_evec / 2;
    args.nkeep_single = N_evec / 4;
#else
      args.nkeep_single = 100;
    if (args.nkeep_single > N_evec/2) args.nkeep_single = N_evec/2;
      args.nkeep = 250;
    if (args.nkeep > N_evec) args.nkeep = N_evec;
#endif
    args.n_dir = 32;
    args.bigendian = 0;
    args.vrb_nkeep_res = 0;
    args.vrb_evec_res = 0;
    args.concur = 32;
    dtime = -dclock ();
    if (lanczos_arg.precision == PREC_SINGLE) {
      writer.writeCompressedVector (evec_dir, out_f, args, v_eval);
    } else {
      for (size_t i = 0; i < total_size * N_evec; i++)
        out_f[i] = out_d[i];
      writer.writeCompressedVector (evec_dir, out_f, args, v_eval);
    }
    dtime += dclock ();
    print_time ("main()", "writeCompressedVector()", dtime);
//    VRB.Result ("", "main()", "evec written\n");
  }
  cps::sync ();
  EigenCacheGrid<FgridMobius::GridFermionFieldF> *ecache = NULL;

  const char *evec_name = "light_evec";
#if 1
  Float mass = lanczos_arg.mass;
  qpropw_arg.cg.mass=mass;
#else
  if (fabs (lanczos_arg.mass - qpropw_arg.cg.mass) > 1e-8)
    ERR.General ("", fname,
                 "lanczos_arg.mass(%0.4e)!=qpropw_arg.cg.mass(%0.4e)\n",
                 lanczos_arg.mass, qpropw_arg.cg.mass);
#endif
  qpropw_arg.cg.fname_eigen = (char *) evec_name;
  if(N_evec>0)
  {

    const int n_fields = GJP.SnodeSites ();
    const size_t f_size_per_site = lat.FsiteSize () / n_fields / 2;     // checkerboarding

//    ecache = new EigenCache (evec_name);
    ecache = new EigenCacheGrid<FgridMobius::GridFermionFieldF> (evec_name);
    size_t fsize = evec_size;
    int data_size = sizeof (Float);
    if (lanczos_arg.precision == PREC_SINGLE)
      data_size = sizeof (float);
    ecache->alloc (N_evec, fsize, data_size);
    ecache->read_compressed ((char*)evec_dir);
    EigenCacheList.push_back (ecache);
//  EigenContainer eigcon(lat, evec_dir.c_str(), N_evec,f_size_per_site/2, n_fields, ecache);
  }

  CommonArg carg;
if(N_evec>0)
  {
    qpropw_arg.cg.Inverter = CG_LOWMODE_DEFL;
//    QPropWPointSrc light_point_defl (lat, &qpropw_arg, &carg);
{
    QPropWWallSrc light_point_e (lat, &qpropw_arg, &carg);
    CalMesons (light_point_e, result_dir, 0, res_e);
    CalNucleons (light_point_e, result_dir, 0, res_e);
}
    qpropw_arg.cg.stop_rsd = 1e-4;
    qpropw_arg.cg.true_rsd = 1e-4;
{
    QPropWWallSrc light_point_s (lat, &qpropw_arg, &carg);
    CalMesons (light_point_s, result_dir, 0, res_sloppy);
    CalNucleons (light_point_s, result_dir, 0, res_sloppy);
}
   ecache->dealloc ();
  }

if (1) {
    qpropw_arg.cg.Inverter = CG;
//    qpropw_arg.cg.mass = 0.5;
    qpropw_arg.cg.mass = 0.02661;
    qpropw_arg.cg.stop_rsd = 1e-8;
    qpropw_arg.cg.true_rsd = 1e-8;
//    QPropWPointSrc light_point (lat, &qpropw_arg, &carg);
    QPropWWallSrc light_point (lat, &qpropw_arg, &carg);
    CalMesons (light_point, result_dir, 0, (char*)res_s);
    CalNucleons (light_point, result_dir , 0, (char*)res_s);
}
  if (twist)
    untwistBc (lat, twist);

  sfree (out_d);
}
