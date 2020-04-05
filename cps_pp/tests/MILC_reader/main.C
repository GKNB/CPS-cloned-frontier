#include <cps.h>

#undef encode_vml
#define encode_vml(arg_name) do{                                  \
        if( !arg_name.Encode(#arg_name".dat", #arg_name) ){                    \
            ERR.General(cname, fname, #arg_name " encoding failed.\n"); \
        }                                                               \
    }while(0)

#undef decode_vml
#define decode_vml(arg_name)  do{                                       \
        if ( ! arg_name.Decode(#arg_name".vml", #arg_name) )            \
            ERR.General(cname, fname, "Bad " #arg_name ".vml.\n");      \
    } while(0)



USING_NAMESPACE_CPS 

int read_lattice (int argc, char **argv)
{
  cout << "ReadLatticeParallel class test!" << endl;

  GwilsonFnone lat;             // create a lattice at the GJP-specified address


  // load lattice, and use the above created lattice to check plaq and link trace
  ReadMILCParallel rd;
  rd.setLogDir ("logs");
  //  rd.SimQCDSP(1);
  rd.read (lat, argv[2]);

  //// to have more control, use
  //
  // QioArg  rd_arg(char * filename, Float check_precision /* optional */ );
  // 
  //   (modify rd_arg members....)   // including setting ConcurIONumber // see  qioarg.h
  //
  // ReadLatticeParallel rd(lat, rd_arg);
  //


  if (!rd.good ()) {
    cout << "Loading failed" << endl;
    exit (-13);
  }


  cout << "Load complete" << endl << endl;


  return 0;
}



int main (int argc, char **argv)
{
  const char *cname = "";
  const char *fname = "main(argc,argv)";

  Start (&argc, &argv);
  // init  GJP
  DoArg do_arg;

  decode_vml (do_arg);
  encode_vml (do_arg);

  GJP.Initialize (do_arg);

  cout << "Initialized ok" << endl;

  int if_read = 0;
  if (argc > 2)
    if_read = 1;
  if (if_read)
    read_lattice (argc, argv);

  std::string latname;
  LatticeContainer lat_cont;
  {
    GnoneFnone lattice;
  }
  {
    GnoneFhisq lattice;
    if (if_read)
      latname = std::string (argv[2]) + ".NERSC";
    else
      latname = std::string (do_arg.start_conf_filename) + ".NERSC";
    WriteLatticeParallel (lattice, latname.c_str ());
    lat_cont.Get (lattice);
    lattice.Smear ();
    Matrix *field = lattice.Fields (0);
    Matrix *gauge = lattice.GaugeField ();
    size_t vol = GJP.VolNodeSites ();
    VRB.Result (cname, fname, "vol=%u \n", vol);
    size_t interval = vol / 100;
    time_elapse ();
    if(0)
    for (size_t i = 0; i < vol; i++) {
      Site s (i);
      if (i % interval == 0)
        VRB.Result (cname, fname, "i=%u %d %%\n", i, i / interval);
      for (int mu = 0; mu < 4; mu++) {
        (gauge + i * 4 + mu)->Dagger (*(field + mu * vol + i));
//        *(gauge + i * 4 + mu) = (*(field + mu * vol + i));
        Float *tmp_p = (Float *) (gauge + i * 4 + mu);
	Float det[2];
	(gauge + i * 4 + mu)->Det(det);
	Complex  trace = (gauge + i * 4 + mu)->Trace();

        VRB.Result (cname, fname,
                    "%d: %d %d %d %d %d = %0.4e %0.4e %0.4e %0.4e %0.4e %0.4e : %0.6e %0.6e %0.6e %0.6e\n",
                    i, s.physX (), s.physY (), s.physZ (), s.physT (), mu,
                    tmp_p[0], tmp_p[1], tmp_p[2], tmp_p[3], tmp_p[4], tmp_p[5],
		    trace.real(),trace.imag(),det[0],det[1]);
      }
    }
    if (if_read)
      latname = std::string (argv[2]) + ".hisq";
    else
      latname = std::string (do_arg.start_conf_filename) + ".hisq";
    WriteLatticeParallel (lattice, latname.c_str ());
    lat_cont.Set (lattice);
    print_flops ("", "convert", 0, time_elapse ());
  }
  {
    GnoneFnone lattice;
  }

}
