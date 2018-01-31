#ifndef _WRITENERSC_H_
#define _WRITENERSC_H_
// Write the format {load,unload}_lattice 

#include <stdlib.h>	// exit()
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <util/gjp.h>
#include <util/vector.h>
#include <util/lattice.h>
#include <util/verbose.h>
#include <util/error.h>
#include <util/qcdio.h>

#include <util/qioarg.h>
//#include <util/fpconv.h>
#include <util/iostyle.h>
#include <util/latheader.h>


CPS_START_NAMESPACE

//----------------------------------------
// WriteNERSC class
template <class headertype, int _Ndim, typename dtype>
class WriteNERSC : public QioControl
{
  // IoStyle provides a function  IoStyle::store() 
  // which determines Parallel or Serial storing

 private:
  //    FPConv fpconv;
    int data_per_site;
    int csum_pos;
//    bool recon_row_3;
    const char *cname;
    bool UseParIO;
    int Ndim;

 public:
    headertype hd;

 public:
    // ctor for 2-step unloading
    WriteNERSC(int ndata)  
      : QioControl(), cname("WriteNERSC"),Ndim(_Ndim),data_per_site(ndata){
    }

    // ctor containing unloading behavior
    WriteNERSC(int ndata,dtype * data, const char * filename,
			 const FP_FORMAT dataFormat = FP_AUTOMATIC)
      : QioControl(), cname("WriteNERSC"),Ndim(_Ndim),data_per_site(ndata){
      QioArg  wt_arg(filename, dataFormat);
      write(data, wt_arg);
    }

    // ctor containing unloading behavior
    WriteNERSC(int ndata,dtype *data, const QioArg & wt_arg)
      : QioControl(), cname("WriteNERSC"),Ndim(_Ndim),data_per_site(ndata){
      write(data, wt_arg);
    }

    ~WriteNERSC() {}

#if 0

    void setHeader(const char * EnsembleId, const char * EnsembleLabel,
		   const int SequenceNumber) {
      hd.setHeader(EnsembleId, EnsembleLabel, SequenceNumber);
    }
#endif

    void write(void * data, const char * filename,
	       const FP_FORMAT dataFormat = FP_AUTOMATIC) {
      QioArg  wt_arg(filename, dataFormat);
      write(data, wt_arg);
    }

 public:
    inline void setParallel() { UseParIO = 1; }
    inline void setSerial() { UseParIO = 0; }
    inline int parIO() const { return UseParIO; }
//    void write(void *, const QioArg & wt_arg);
#define PROFILE
//template <class headertype, int _Ndim, typename dtype>
void write(void *data, const QioArg & wt_arg)
{
  const char * fname = "write()";
  VRB.Func(cname,fname);

  char loginfo[100];
  sprintf(loginfo,"Unload %s",wt_arg.FileName);
  startLogging(loginfo);

#ifdef PROFILE
  struct timeval start,end;
  gettimeofday(&start,NULL);
#endif
  sync();

  // init
  int error = 0;
  io_good = false;

  //  const char * filename = wt_arg.FileName;
  FP_FORMAT dataFormat = wt_arg.FileFpFormat;
  fpconv.setFileFormat(dataFormat);
  if(fpconv.fileFormat == FP_UNKNOWN) {
    ERR.General(cname,fname,"Output Floating Point format UNKNOWN\n");
  }

  sync();

  log();
  
  // write lattice data, in Parallel or Serial manner
  // determined by the template parameter "IoStyle" of this class
//  int data_per_site = recon_row_3 ? 4*12 : 4*18;
//  const size_t chars_per_site = data_per_site * fpconv.fileFpSize();
  const size_t chars_per_site = data_per_site * sizeof(dtype);

  unsigned int csum = 0;

// #endif
  
  std::fstream output;

  if (isRoot()){
    output.open(wt_arg.FileName,std::ios::out);
	output.close();
  }
//  Float temp=0.;
//  glb_sum(&temp);
  sync();
  if(parIO()) {
    // all open file, start writing
    output.open(wt_arg.FileName);
    if(!output.good())    {
            ERR.General(cname,fname, "Could not open file: [%s] for output.\n",wt_arg.FileName);
      //      VRB.Flow(cname,fname,"USER: maybe you should kill the process\n");
      
      printf("Node %d:Could not open file: [%s] for output.\n",UniqueID(),wt_arg.FileName);
      error = 1;
    }
  }
  else {
    // only node 0 open file, start writing
    if(isRoot()) {
      FILE *fp = Fopen(wt_arg.FileName,"w");
      Fclose(fp);
      output.open(wt_arg.FileName);
      if(!output.good())    {
	//	VRB.Flow(cname,fname, "Could not open file: [%s] for output.\n",wt_arg.FileName);
	//	VRB.Flow(cname,fname,"USER: maybe you should kill the process\n");
	error = 1;
      }
    }
  }
  if (error)
    printf("Node %d: says opening %s failed\n",UniqueID(),wt_arg.FileName);
//  if(synchronize(error) > 0)  
//    ERR.FileW(cname,fname,wt_arg.FileName);
   error=0;

  // write header
  if(isRoot()){
    hd.init(wt_arg, fpconv.fileFormat);
    hd.setHeader(data_per_site);
    hd.write(output);
  }
  if (error)
    printf("Node %d: says Writing header failed  %s failed\n",UniqueID(),wt_arg.FileName);
//  if(synchronize(error) > 0) 
//    ERR.General(cname,fname,"Writing header failed\n");
  log();
  int temp_start = hd.data_start;
  broadcastInt(&temp_start);
  hd.data_start = temp_start;

  if(parIO()) {
    ParallelIO pario(wt_arg);
    if(!pario.store(output, (char*)data, data_per_site, chars_per_site,
		    hd, fpconv, Ndim, &csum)) 
      ERR.General(cname, fname, "Unload failed\n");

  }
  else {
    SerialIO serio(wt_arg);
    if(!serio.store(output, (char*)data, data_per_site, chars_per_site,
		    hd, fpconv, Ndim, &csum)) 
      ERR.General(cname, fname, "Unload failed\n");
  }
  //    printf("Node %d: lattice write csum=%x\n",UniqueID(),csum);
  if(wt_arg.Scoor() == 0)
      csum = globalSumUint(csum);
  else
      globalSumUint(0);
//  output.flush();

  log();

  // after unloading, fill in checksum
  if(isRoot()) 
    hd.fillInChecksum(output,csum);

  if(parIO()) {
    output.close();
  }
  else {
    if(isRoot())
      output.close();
  }
  if(synchronize(error) != 0)  
    ERR.General(cname, fname, "Writing checksum failed\n");
  
#ifdef PROFILE
  gettimeofday(&end,NULL);
  print_flops(cname,"write",0,&start,&end);
#endif

  io_good = true;

  log();
  finishLogging();
  sync();
  VRB.FuncEnd(cname,fname);
}

};


CPS_END_NAMESPACE
#endif
