#include<config.h>
#include <alg/a2a/mesonfield/mesonfield_controls.h>

CPS_START_NAMESPACE

int BlockedMesonFieldArgs::bi(MF_COMPUTE_BI);
int BlockedMesonFieldArgs::bj(MF_COMPUTE_BJ);
int BlockedMesonFieldArgs::bp(MF_COMPUTE_BP);  

int BlockedMesonFieldArgs::bii(MF_COMPUTE_BII);
int BlockedMesonFieldArgs::bjj(MF_COMPUTE_BJJ);
int BlockedMesonFieldArgs::bpp(MF_COMPUTE_BPP);  

bool BlockedMesonFieldArgs::enable_profiling(false);  

int BlockedvMvOffloadArgs::b(100);


CPS_END_NAMESPACE
