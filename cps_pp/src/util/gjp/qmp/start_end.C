#include <iostream>
#include <config.h>
#include <version.h>
#include <util/gjp.h>
#include <util/error.h>
#ifdef USE_QMP
#include <qmp.h>
#endif

#ifdef USE_GRID
#include <util/lattice/fgrid.h>
#endif

#ifdef USE_QUDA
#include <invert_quda.h>
#endif

CPS_START_NAMESPACE
#ifdef USE_QMP
void Start(){
  ERR.General("cps","Start()","Start(&argc,&argv should be used with QMP");
//  QMPSCU::init_qmp();
}

void printHash(){
#ifdef GITHASH
    std::cout << "CPS git commit hash=" << GITHASH << std::endl;
#else
    std::cout << "CPS git commit hash is undefined. Check version.h ." << std::endl;
#endif
#undef GITHASH
}


void Start(int * argc, char *** argv) {
//  printf("Start(%d %p)\n",*argc,*argv);
  //Initialize QMP
  QMPSCU::init_qmp(argc, argv);
  GJP.setArg(argc,argv);
  printHash();
#ifdef USE_QUDA
  initQuda(-1000);
#endif
}
void End(){
  QMPSCU::destroy_qmp();
  // printf("destroy_qmp()\n");
#ifdef USE_QUDA
  endQuda();
#endif
}
#else
void Start(int * argc, char *** argv) {
  GJP.setArg(argc,argv);
#ifdef USE_QUDA
  initQuda(QudaParam.device);
#endif
}
void End(){
#ifdef USE_QUDA
  endQuda();
#endif
}
#endif
CPS_END_NAMESPACE
