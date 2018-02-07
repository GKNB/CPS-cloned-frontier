#include <iostream>
#include <complex>
#include <stdio.h>
#include <stdlib.h>
#include <util/random.h>
#include <util/gjp.h>
#include <util/time_cps.h>
#include <util/error.h>
#include <util/lat_data.h>
#include <alg/alg_fix_gauge.h>
#include <alg/do_arg.h>

USING_NAMESPACE_CPS

FixGaugeArg fix_gauge_arg;
DoArg do_arg;


int main (int argc, char **argv)
{

  const char *cname = "";
  const char *fname = "main()";
  Start (&argc, &argv);

  do_arg.Decode("do_arg.vml","do_arg");
  fix_gauge_arg.Decode("fix_gauge_arg.vml","fix_gauge_arg");
  GJP.Initialize(do_arg);
  int traj=0;

  GnoneFwilson lat;
  std::stringstream buf;
  buf << "fg-bc." << traj <<std::endl;
  CommonArg com_fg;
  com_fg.set_filename(buf.str().c_str());
  AlgFixGauge gf(lat,&com_fg,&fix_gauge_arg);
  std::string gfname("gfix.in");
  FILE *fp = fopen(gfname.c_str(),"r");
  if(fp) {
    QioArg rd_arg(gfname.c_str());
    gf.run(&rd_arg);
  } else 
    gf.run();
    QioArg wt_arg("gfix.out");
    gf.Save(wt_arg);

}
