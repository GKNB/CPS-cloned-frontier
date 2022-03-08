#pragma once

#ifdef USE_CALLGRIND
#include<valgrind/callgrind.h>
#else
#define CALLGRIND_START_INSTRUMENTATION ;
#define CALLGRIND_STOP_INSTRUMENTATION ;
#define CALLGRIND_TOGGLE_COLLECT ;
#endif

#ifdef USE_VTUNE
#include<ittnotify.h>
#else
void __itt_pause(){}
void __itt_resume(){}
void __itt_detach(){}
#endif

#ifdef USE_GPERFTOOLS
#include <gperftools/profiler.h>
#else
int ProfilerStart(const char* fname){}
void ProfilerStop(){}
#endif

#ifdef GRID_CUDA
#include <cuda_profiler_api.h>
#else
void cudaProfilerStart(){}
void cudaProfilerStop(){}
#endif
