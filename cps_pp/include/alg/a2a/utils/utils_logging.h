#pragma once

#include <util/gjp.h>
#ifdef USE_GRID
#include <Grid/Grid.h>
#endif

CPS_START_NAMESPACE

#ifdef USE_GRID
#define LOGA2A std::cout << Grid::GridLogMessage
#define LOGA2ANT std::cout
#else
#define LOGA2A if(!UniqueID()) std::cout
#define LOGA2ANT std::cout
#endif

//printf only to head node through LOGA2A stream
void a2a_printf(const char* format, ...);
//printf only to head node through LOGA2ANT stream (no timing)
void a2a_printfnt(const char* format, ...);


//print_time only to head node through LOGA2A stream
inline void a2a_print_time(const char *cname, const char *fname, Float time){
  a2a_printf("%s::%s: %e seconds\n",cname,fname,time);
}

CPS_END_NAMESPACE
