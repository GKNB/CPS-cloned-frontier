#include <config.h>
#ifdef USE_QMP
#include <util/gjp.h>
#include <util/error.h>
#include <qmp.h>
#ifdef USE_GRID
#include <util/lattice/fgrid.h>
#endif

//QMP definitions for CPS::Start() and CPS::End()

//extern "C" void _mcleanup();
CPS_START_NAMESPACE
void Start(){
  ERR.General("cps","Start()","Start(&argc,&argv should be used with QMP");
//  QMPSCU::init_qmp();
}

void Start(int * argc, char *** argv) {
//  printf("Start(%d %p)\n",*argc,*argv);
  //Initialize QMP
#ifdef USE_GRID
//  Grid::Grid_init(argc,argv);
//  Fgrid::grid_initted=true;
#endif
  QMPSCU::init_qmp(argc, argv);
  GJP.setArg(argc,argv);
}
void End(){
  QMPSCU::destroy_qmp();
//  _mcleanup();
}
CPS_END_NAMESPACE
#endif
