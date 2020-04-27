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



int read_lattice(int argc, char ** argv) {
  cout << "ReadLatticeParallel class test!" << endl;

  GwilsonFnone lat;  // create a lattice at the GJP-specified address

  
  // load lattice, and use the above created lattice to check plaq and link trace
  ReadMILCParallel  rd;
  rd.setLogDir("logs");
  //  rd.SimQCDSP(1);
  rd.read(lat,argv[2]);

  //// to have more control, use
  //
  // QioArg  rd_arg(char * filename, Float check_precision /* optional */ );
  // 
  //   (modify rd_arg members....)   // including setting ConcurIONumber // see  qioarg.h
  //
  // ReadLatticeParallel rd(lat, rd_arg);
  //

  
  if(!rd.good()) {
    cout << "Loading failed" << endl;
    exit(-13);
  }
  

  cout << "Load complete" << endl << endl;

#if 0
  cout << "================== PARALLEL IO ======================" << endl;
  cout << "-----------------------------------------------------" << endl;
  cout << "================== SERIAL   IO ======================" << endl << endl;

  ReadLatticeSerial rd2;
  rd2.setLogDir("logs");
  //  rd2.SimQCDSP(1);
  rd2.read(lat,argv[2]);

  if(!rd2.good()) {
    cout << "Serial Loading failed" << endl;
    exit(-13);
  }

  cout << "Load Complete" << endl;
#endif

  return 0;
}


int write_lattice(int argc, char ** argv) {
  cout << "WriteLatticeParallel class test!" << endl;

  GwilsonFnone lat;

  // TWO-STEP writing, to set some labels in header
  WriteLatticeParallel wt;
  wt.setHeader("EnsTest","Testing Ensemble by Sam Oct,2004",1001);
  wt.setLogDir("logs");
  wt.write(lat,argv[2]);

  if(!wt.good()) {
    cout << "Unloading failed" << endl;
    exit(-13);
  }

  cout << "===================  PARALLEL UNLOADING ["<<argv[2]<<"] =========================" << endl;
  cout << "--------------------------------------------------------------------------------------" << endl;
  cout << "===================  SERIAL   UNLOADING ["<<argv[2]<<".serial] ==================" << endl << endl;

  WriteLatticeSerial wt2;
  wt2.setHeader("EnsSerial", "Serial unloading test Nov, 2004", 1002);
  wt2.setLogDir("logs");

  char filename2[256];
  strcpy(filename2,argv[2]);
  strcat(filename2,".serial");

  wt2.write(lat,filename2);
  if(!wt2.good()) {
    cout << "Unloading failed" << endl;
    exit(-13);
  }

  cout << "Unload complete" <<endl << endl;

  return 0;
}


int main(int argc, char ** argv) {
  const char *cname="";
  const char *fname="main(argc,argv)";
  if(0)
  if(argc<11) {
    cout << "Usage:" << endl<<"      qrun QCDOC.x  -[r|w]  <conf.dat>  <x sites> <y sites> <z sites> <t sites>  <Xbc> <Ybc> <Zbc> <Tbc>"<< endl;
    cout << "(use letter \'P\' or \'A\' for arguments of gauge BC's)" << endl;
    cout << "Eg,   qrun QCDOC.x -r  conf8x8x8x16.file   8 8 8 16  P P P P"<< endl;
    cout << "      qrun QCDOC.x -w  conf4x4x4x32.file   4 4 4 32  P P A A"<< endl;
    exit(1);
  }


  Start(&argc,&argv);
  // init  GJP
  DoArg do_arg;

    decode_vml (do_arg);
    encode_vml (do_arg);

    GJP.Initialize(do_arg);

    cout << "Initialized ok" << endl;

    int if_read=0;
    if (argc > 2) if_read=1;
    if(if_read) read_lattice(argc,argv);

    std::string latname;
  LatticeContainer lat_cont;
{
    GnoneFnone lattice;
}
  {
    GnoneFhisq lattice;
    if(if_read) latname = std::string(argv[2])+".NERSC";
     else latname = std::string(do_arg.start_conf_filename)+".NERSC";
    WriteLatticeParallel(lattice,latname.c_str());
    lat_cont.Get(lattice);
    lattice.Smear();
	Matrix *field = lattice.Fields(0);
	Matrix *gauge = lattice.GaugeField();
	size_t vol = GJP.VolNodeSites();
	    VRB.Result(cname,fname,"vol=%u \n",vol);
	size_t interval=vol/100;
    time_elapse();
    for(size_t i=0;i<vol;i++){
//	  Site s (i);
         if(i%interval==0)
	 VRB.Result(cname,fname,"i=%u %d %%\n",i,i/interval);
	  for(int mu=0;mu<4;mu++){
	    (gauge+i*4+mu)->Dagger(  *(field+mu*vol+i));
	  }
    }
    if(if_read) latname = std::string(argv[2])+".hisq";
     else latname = std::string(do_arg.start_conf_filename)+".hisq";
    WriteLatticeParallel(lattice,latname.c_str());
    lat_cont.Set(lattice);
    print_flops("","convert",0,time_elapse());
  }
{
    GnoneFnone lattice;
}

}

  
