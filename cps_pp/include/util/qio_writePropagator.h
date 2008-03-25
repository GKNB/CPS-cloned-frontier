#ifndef __QIOWRITEPROP__
#define __QIOWRITEPROP__

#include <util/qio_general.h>
#include <util/fpconv.h>


CPS_START_NAMESPACE
using namespace std;


class qio_writePropagator: private qio_init {

 private:

  char *cname;

 public:

  // version with argc/argv

  qio_writePropagator( int argc, char *argv[]): qio_init(argc, argv), cname("qio_writePropagator"), source_hypercube(0)
    {initHeader();}


  // writing Scalar Source + 12 Sinks
  qio_writePropagator(char *outfile, const void *prop, const void *scSource, 
		      int argc, char *argv[], int volFormat=QIO_VOLFMT, FP_FORMAT floatFormat=FP_AUTOMATIC):
    qio_init(argc, argv), cname("qio_writePropagator"), source_hypercube(0)
    { initHeader(); write_ScS_12sink(outfile, prop, scSource, volFormat, floatFormat);}

   
  // writing Scalar Source + 12 Sinks, setting header id, label, traj
  qio_writePropagator(char *outfile, const void *prop, const void *scSource, 
		      const char * ensemble_id, const char * ensemble_label, const int traj,
		      int argc, char *argv[], int volFormat=QIO_VOLFMT, FP_FORMAT floatFormat=FP_AUTOMATIC):
    qio_init(argc, argv), cname("qio_writePropagator"), source_hypercube(0)
    { setHeader(ensemble_id, ensemble_label, traj); write_ScS_12sink(outfile, prop, scSource, volFormat, floatFormat);}

  // writing Scalar Source + 12 Sinks, setting header id, label, traj, prop_type, source_type
  qio_writePropagator(char *outfile, const void *prop, const void *scSource, 
		      const char * ensemble_id, const char * ensemble_label, const int traj,
		      const char * propagator_type, const char * source_type,
		      int argc, char *argv[], int volFormat=QIO_VOLFMT, FP_FORMAT floatFormat=FP_AUTOMATIC):
    qio_init(argc, argv), cname("qio_writePropagator"), source_hypercube(0)
    { setHeader(ensemble_id, ensemble_label, traj, propagator_type, source_type); 
    write_ScS_12sink(outfile, prop, scSource, volFormat, floatFormat);}

  
  
  // writing 12 pairs of Scalar/Full Source and full Sink 
  qio_writePropagator(char *outfile, const QIO_PROP_SOURCE_TYPES sType, const void *prop, const void *source, 
		      int argc, char *argv[], int volFormat=QIO_VOLFMT, FP_FORMAT floatFormat=FP_AUTOMATIC):
    qio_init(argc, argv), cname("qio_writePropagator"), source_hypercube(0)
    { initHeader(); write_12pairs(outfile, sType, prop, source, volFormat, floatFormat);}

  // writing 12 pairs of Scalar/Full Source and full Sink, setting header id, label, traj
  qio_writePropagator(char *outfile, const QIO_PROP_SOURCE_TYPES sType, const void *prop, const void *source, 
		      const char * ensemble_id, const char * ensemble_label, const int traj, 
		      int argc, char *argv[], int volFormat=QIO_VOLFMT, FP_FORMAT floatFormat=FP_AUTOMATIC):
    qio_init(argc, argv), cname("qio_writePropagator"), source_hypercube(0)
    { setHeader(ensemble_id, ensemble_label, traj); write_12pairs(outfile, sType, prop, source, volFormat, floatFormat);}
  
  // writing 12 pairs of Scalar/Full Source and full Sink, setting header id, label, traj
  qio_writePropagator(char *outfile, const QIO_PROP_SOURCE_TYPES sType, const void *prop, const void *source, 
		      const char * ensemble_id, const char * ensemble_label, const int traj,
		      const char * propagator_type, const char * source_type,
		      int argc, char *argv[], int volFormat=QIO_VOLFMT, FP_FORMAT floatFormat=FP_AUTOMATIC):
    qio_init(argc, argv), cname("qio_writePropagator"), source_hypercube(0)
    { setHeader(ensemble_id, ensemble_label, traj, propagator_type, source_type); 
    write_12pairs(outfile, sType, prop, source, volFormat, floatFormat);}
  
  // version w/o argc/argv, using GJP

  qio_writePropagator(): qio_init(GJP.argc(), GJP.argv()), cname("qio_writePropagator"), source_hypercube(0)
    {initHeader();}
  
  
  // writing Scalar Source + 12 Sinks
  qio_writePropagator(char *outfile, const void *prop, const void *scSource, 
		      int volFormat=QIO_VOLFMT, FP_FORMAT floatFormat=FP_AUTOMATIC):
    qio_init(GJP.argc(), GJP.argv()), cname("qio_writePropagator"), source_hypercube(0)
    { initHeader(); write_ScS_12sink(outfile, prop, scSource, volFormat, floatFormat);}

   
  // writing Scalar Source + 12 Sinks, setting header id, label, traj
  qio_writePropagator(char *outfile, const void *prop, const void *scSource, 
		      const char * ensemble_id, const char * ensemble_label, const int traj,
		      int volFormat=QIO_VOLFMT, FP_FORMAT floatFormat=FP_AUTOMATIC):
    qio_init(GJP.argc(), GJP.argv()), cname("qio_writePropagator"), source_hypercube(0)
    { setHeader(ensemble_id, ensemble_label, traj); write_ScS_12sink(outfile, prop, scSource, volFormat, floatFormat);}

  // writing Scalar Source + 12 Sinks, setting header id, label, traj, prop_type, source_type
  qio_writePropagator(char *outfile, const void *prop, const void *scSource, 
		      const char * ensemble_id, const char * ensemble_label, const int traj,
		      const char * propagator_type, const char * source_type,
		      int volFormat=QIO_VOLFMT, FP_FORMAT floatFormat=FP_AUTOMATIC):
    qio_init(GJP.argc(), GJP.argv()), cname("qio_writePropagator"), source_hypercube(0)
    { setHeader(ensemble_id, ensemble_label, traj, propagator_type, source_type); 
    write_ScS_12sink(outfile, prop, scSource, volFormat, floatFormat);}

  
  
  // writing 12 pairs of Scalar/Full Source and full Sink 
  qio_writePropagator(char *outfile, const QIO_PROP_SOURCE_TYPES sType, const void *prop, const void *source, 
		      int volFormat=QIO_VOLFMT, FP_FORMAT floatFormat=FP_AUTOMATIC):
    qio_init(GJP.argc(), GJP.argv()), cname("qio_writePropagator"), source_hypercube(0)
    { initHeader(); write_12pairs(outfile, sType, prop, source, volFormat, floatFormat);}

  // writing 12 pairs of Scalar/Full Source and full Sink, setting header id, label, traj
  qio_writePropagator(char *outfile, const QIO_PROP_SOURCE_TYPES sType, const void *prop, const void *source, 
		      const char * ensemble_id, const char * ensemble_label, const int traj, 
		      int volFormat=QIO_VOLFMT, FP_FORMAT floatFormat=FP_AUTOMATIC):
    qio_init(GJP.argc(), GJP.argv()), cname("qio_writePropagator"), source_hypercube(0)
    { setHeader(ensemble_id, ensemble_label, traj); write_12pairs(outfile, sType, prop, source, volFormat, floatFormat);}
  
  // writing 12 pairs of Scalar/Full Source and full Sink, setting header id, label, traj
  qio_writePropagator(char *outfile, const QIO_PROP_SOURCE_TYPES sType, const void *prop, const void *source, 
		      const char * ensemble_id, const char * ensemble_label, const int traj,
		      const char * propagator_type, const char * source_type,
		      int volFormat=QIO_VOLFMT, FP_FORMAT floatFormat=FP_AUTOMATIC):
    qio_init(GJP.argc(), GJP.argv()), cname("qio_writePropagator"), source_hypercube(0)
    { setHeader(ensemble_id, ensemble_label, traj, propagator_type, source_type); 
    write_12pairs(outfile, sType, prop, source, volFormat, floatFormat);}
  

  

  virtual ~qio_writePropagator(){ 
    #ifdef DEBUG_Init
    printf("finished: qio_writePropagator\n");
    #endif // DEBUG_Init
  }

  void write_ScS_12sink(char *outfile, const void *prop, const void *scSource, 
			int volFormat=QIO_VOLFMT, FP_FORMAT floatFormat=FP_AUTOMATIC);

  void write_12pairs(char *outfile, const QIO_PROP_SOURCE_TYPES sType, const void *prop, const void *source, 
		     int volFormat=QIO_VOLFMT, FP_FORMAT floatFormat=FP_AUTOMATIC);

  void setHeader(const char * ensemble_id, const char * ensemble_label, const int traj);

  void setHeader(const char * propagator_type, const char * source_type);

  void setHeader(const char * ensemble_id, const char * ensemble_label, const int traj, 
		 const char * propagator_type, const char * source_type)
    {
      setHeader(ensemble_id, ensemble_label, traj);
      setHeader(propagator_type, source_type);
    }

  void setSourceTslice( const int tslice);

  void setSourceTslices( const int t_start, const int t_end);

  void setSourceHypercube( const int start[4], const int end[4]);

  void setSourceComplete(void)
    { source_hypercube=0;}

  int SourceHypercube(void)
    { return source_hypercube;}

 
 private:

  QIO_Writer *qio_Output;

  void qio_openOutput(char *filename, QIO_String *record_file, int volFormat);

  void qio_closeOutput()
    { QIO_close_write(qio_Output);}


  void initHeader()
    { 
      header_traj = -1; 
      strcpy(header_ensemble_id, "not specified" );
      strcpy(header_ensemble_label, "not specified");
      strcpy(header_propagator_type, "not specified");
      strcpy(header_source_type, "not specified");
    }

  int header_traj;
  char header_ensemble_id[MAX_HEADER_LINE];
  char header_ensemble_label[MAX_HEADER_LINE];
  char header_propagator_type[MAX_HEADER_LINE];
  char header_source_type[MAX_HEADER_LINE];

  int source_hypercube; 
  int source_start[QIO_RW_DIMENSION];
  int source_end[QIO_RW_DIMENSION];
  


};



CPS_END_NAMESPACE
#endif // __QIOWRITEPROP__
