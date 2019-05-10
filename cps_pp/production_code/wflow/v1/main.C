
#include <config.h>

#include <util/lattice.h>
#include <util/lat_cont.h>
#include <util/time_cps.h>

#include <util/verbose.h>
#include <util/error.h>

#include <util/qcdio.h>
#include <util/ReadLatticePar.h>
#include <util/command_line.h>
#include <util/qioarg.h>

#include <alg/common_arg.h>
#include <alg/meas_arg.h>
#include <alg/no_arg.h>
#include <alg/do_arg.h>

#include <alg/alg_plaq.h>
#include <alg/alg_meas.h>
#include <alg/alg_actiondensity.h>
#include <alg/alg_wilsonflow.h>
#include <alg/alg_tcharge.h>

#include <fenv.h>


USING_NAMESPACE_CPS

static const char* cname = "main";

DoArg do_arg;
MeasArg meas_arg;
  
#define decode_vml(arg_name)  do{                                      \
    if ( ! arg_name.Decode("./" #arg_name".vml", #arg_name) )               \
      ERR.General(cname, fname, "Bad ./" #arg_name ".vml.\n");           \
  } while(0)  

void erase_file(const char* filename)
{
  FILE* f = Fopen(filename, "w");
  Fclose(f);
}

void load_checkpoint(int traj)
{
  const char *cname = "cps";
  const char *fname = "load_checkpoint()";

  Float timer = dclock();

  char lat_file[256];
  Lattice &lat = LatticeFactory::Create(F_CLASS_NONE, G_CLASS_NONE);
  sprintf(lat_file, "%s.%d", meas_arg.GaugeStem, traj);
  QioArg rd_arg(lat_file, 0.001);
  rd_arg.ConcurIONumber = meas_arg.IOconcurrency;
  ReadLatticeParallel rl;
  rl.read(lat,rd_arg);
  if(!rl.good()) ERR.General(cname,fname,"Failed read lattice %s\n",lat_file);
  LatticeFactory::Destroy();
  
  VRB.Result(cname, fname, "Loaded lattice in %e seconds\n", dclock() - timer);
}

void measure_wflow(int traj,
                   Float dt,
                   int num_time_steps,
                   int measure_start_step,
                   int measure_interval)
{
  const char* fname = "measure_wflow_scale()";
  VRB.Result(cname, fname, "\n\n\n ************ Starting Wilson flow scale measurement *************\n\n\n");
  Float timer = dclock();

  GwilsonFnone lat;

  // save the unsmeared lattice
  LatticeContainer lat_cont;
  lat_cont.Get(lat);

  CommonArg common_arg,tcharge_arg;
  char filename[512];
  sprintf(filename, "results-dt%0.3f/wflow.%d", dt, traj);
  erase_file(filename);
  common_arg.set_filename(filename);

//  char filename2[512];
//  sprintf(filename2, "results-dt%0.3f/wflow.%d", dt, traj);
//  erase_file(filename2);
//  tcharge_arg.set_filename(filename);

  NoArg no_arg;

  AlgWilsonFlow wilson_flow(lat, &common_arg, dt);
  AlgActionDensity action_density(lat, &common_arg);
  AlgPlaq plaq(lat, &common_arg, &no_arg);
  AlgTcharge tcharge(lat, &common_arg);

  int wflow_count = 0, action_count = 0, plaq_count = 0, tcharge_count=0;
  Float wflow_time = 0, action_time = 0, plaq_time = 0, tcharge_time=0;
  Float wflow_target_t = dt*(Float)num_time_steps;

  int tc_int = (int) 1.0/dt;
  int step;
   double input_dt = dt;
    double actual_dt; // the dt that's actually been run
    double accumulate_t = 0.;
    while(accumulate_t < wflow_target_t){
      action_density.run();
      tcharge.run();
      actual_dt = wilson_flow.run_adaptive(input_dt); // This function returns the actual dt and modify input_dt to the next target value
      accumulate_t += actual_dt;
      if(UniqueID() == 0) printf("This dt = %16.12e, next dt = %16.12e, accumulate t = %16.12e\n", actual_dt, input_dt, accumulate_t);
    }

    action_density.run();
    tcharge.run();



  // restore the unsmeared lattice
  lat_cont.Set(lat);

  VRB.Result(cname, fname, "************ Finished Wilson flow scale measurement *************");
  VRB.Result(cname, fname, "Total time                              = %e seconds\n", dclock() - timer);
  VRB.Result(cname, fname, "Time for %d wilson flow steps           = %e seconds (%e seconds each)\n", wflow_count, wflow_time, wflow_time / wflow_count);
  VRB.Result(cname, fname, "Time for %d action density measurements = %e seconds (%e seconds each)\n", action_count, action_time, action_time / action_count);
  VRB.Result(cname, fname, "Time for %d plaquette measurements      = %e seconds (%e seconds each)\n", plaq_count, plaq_time, plaq_time / plaq_count);
}

void setup(int argc, char *argv[]) 
{
  const char* fname = "setup()";
  
  Start(&argc, &argv);
  CommandLine::is(argc,argv);

  decode_vml(do_arg);
  decode_vml(meas_arg);
  meas_arg.TrajStart = CommandLine::arg_as_int();
  meas_arg.TrajLessThanLimit = CommandLine::arg_as_int();
 
  GJP.Initialize(do_arg);

  //Throw floating point exceptions to crash the program if bad stuff happens:
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
}
 

int main(int argc, char *argv[])
{
  const char* fname = "main()";

  setup(argc, argv);
    Float dt, num_time_steps;

    dt = CommandLine::arg_as_Float();
    num_time_steps = CommandLine::arg_as_int();

  for(int traj = meas_arg.TrajStart; traj < meas_arg.TrajLessThanLimit; traj += meas_arg.TrajIncrement) {
    VRB.Result(cname, fname, "\n\n/////////// Measuring configuration %d /////////////\n\n", traj);

    load_checkpoint(traj);

    int measure_start = 0;
    int measure_interval = 1;


    measure_wflow(traj, dt, num_time_steps, measure_start, measure_interval);
  }

  VRB.Result(cname, fname, "Program exiting normally\n");

  End();

  return 0;
}

