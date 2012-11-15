/*!\file
  \brief Declaration of functions for timing and performance measurement.

  $Id: time_cps.h,v 1.6.28.1 2012-11-15 18:17:08 ckelly Exp $
*/

#ifndef UTIL_TIME_H
#define UTIL_TIME_H

#include <config.h>
#include <util/data_types.h>
#include <sys/time.h>

CPS_START_NAMESPACE
/*! \defgroup profiling Timing and performance functions
  @{
*/

//! Gets the wall-clock time.

Float dclock(void);
Float print_time(const char *cname, const char *fname, Float time);

//! Prints the FLOPS rate to stdout
Float print_flops(double nflops, Float time);
//! Prints the FLOPS rate to stdout
//Float print_flops(const char cname[], const char fname[],unsigned long long nflops, Float time);
Float print_flops(const char cname[], const char fname[],double nflops, Float time);
//! Prints the FLOPS rate to stdout
Float print_flops(double nflops, struct timeval *start, struct timeval *end);
//! Prints the FLOPS rate to stdout
//Float print_flops(const char cname[], const char fname[], unsigned long long nflops, struct timeval *start, struct timeval *end);
Float print_flops(const char cname[], const char fname[], double nflops, struct timeval *start, struct timeval *end);

/*! @} */

//CK: static timestamp class with optional output stream and 'depth' tagging of stamps enabling easy parsing of output
//    writes output to a file

class TimeStamp{
 protected:
  static int cur_depth;
  static FILE *stream;
  static bool enabled;
  static Float start;
  
  static void vstamp(const char *format, va_list args);
 public:
  static void set_file(const char *filename);
  static void reset();
  static void close_file();
  static void stamp(const char *format,...);
  static void incr_depth();
  static void decr_depth();
  
  static void stamp_incr(const char *format,...);
  static void stamp_decr(const char *format,...);

  static void incr_stamp(const char *format,...);
  static void decr_stamp(const char *format,...);

  static void start_func(const char* cls, const char *fnc);
  static void end_func(const char* cls, const char *fnc);
};


CPS_END_NAMESPACE
#endif
