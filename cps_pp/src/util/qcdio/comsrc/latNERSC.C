#include <config.h>
#include <util/latheader.h>
#include <util/qioarg.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <unistd.h>
using namespace std;

CPS_START_NAMESPACE


/////////////////////////////////////////////////////////////////////
// LatNERSCHeader members 
/////////////////////////////////////////////////////////////////////
void LatNERSCHeader::init(const QioArg & qio_arg, FP_FORMAT FileFormat) {
  hdr_version = "1.0";
  storage_format = "1.0";

  for(int i=0;i<4;i++)
    dimension[i] = qio_arg.Nodes(i)*qio_arg.NodeSites(i);


  for(int i=0;i<4;i++) 
    boundary[i] = qio_arg.Bc(i);

  checksum = 0;

  //  ensemble_id = "unspecified";
  //  ensemble_label = "unspecified";
  //  sequence_number = 0;
  

#if TARGET == BGL 
    creator = "RBC"; // getlogin() not supported on QCDOC yet
    creator_hardware = "BGL";
#elif TARGET == BGQ 
    creator = "RBC";
    creator_hardware = "BGQ";
#else
    creator = "RBC";
    creator_hardware = "NOARCH";
#endif

    char buf[256];
    gethostname(buf,256);
    creator_hardware += buf;

  // default archive time is current time
    struct timeval tp;
    gettimeofday(&tp,NULL);
    time_t ptm = tp.tv_sec;

    cout << "time = " << ptm << endl;
    archive_date = asctime(localtime(&ptm));
    int i1( archive_date.find_last_not_of("\n"));
    archive_date = archive_date.substr(0,i1+1);

    creation_date = archive_date;

    floating_point = FileFormat;

}


void LatNERSCHeader::setHeader(int ndata, const char *CreatorName , const char *CreatorHardware ){
//  ensemble_id = EnsembleId;
//  ensemble_label= EnsembleLabel;
//  sequence_number = SequenceNumber;
  data_per_site = ndata;
  if (CreatorName) creator = CreatorName;
  if (CreatorHardware) creator_hardware = CreatorHardware;
}

void LatNERSCHeader::writeContent(ostream & fout) {
  const char *fname="writeContent(ostream&)";
//  fout.seekp(0,ios::beg);
//  fout << "BEGIN_HEADER" << endl;
  fout << "HDR_VERSION = " << hdr_version << endl;
  fout << "STORAGE_FORMAT = " << storage_format << endl;

  for(int i=0;i<4;i++){
    fout << "DIMENSION_" << i+1 << " = " << dimension[i] << endl ;
  }
  // just to keep the space and write it later

  fout << "CHECKSUM = ";
  // store checksum position
  csum_pos = fout.tellp();
  fout << hex << setw(8) << 0 << dec << endl;
  
  fout << "FLOATING_POINT = " << FPConv::name(floating_point) <<endl;
  fout << "DATA_PER_SITE = " <<data_per_site << endl;

//  fout << "END_HEADER" << endl;
//  data_start = fout.tellp();
//  VRB.Result(cname,fname,"data_start=%d csum_pos=%d\n",data_start,csum_pos);
}


void LatNERSCHeader::fillInChecksum(ostream & fout, unsigned int checksum) const{
  fout.seekp(csum_pos);
  fout << hex << setw(8) << checksum << dec;
}


void LatNERSCHeader::read(istream & fin) {
  
  const char *fname="read(istream&)";
  string line;
  do {
    getline(fin,line); // read one line
    hd.add(line);
  } while( line.find("END_HEADER") == string::npos);
  
  data_start = fin.tellg();
  
  // interpret header
  hdr_version = hd.asString("HDR_VERSION");
  storage_format = hd.asString("STORAGE_FORMAT");

  dimension[0] = hd.asInt("DIMENSION_1");
  dimension[1] = hd.asInt("DIMENSION_2");
  dimension[2] = hd.asInt("DIMENSION_3");
  dimension[3] = hd.asInt("DIMENSION_4");


  string bcs[] = {hd.asString("BOUNDARY_1"), hd.asString("BOUNDARY_2"), hd.asString("BOUNDARY_3"), hd.asString("BOUNDARY_4") };
  for(int d=0;d<4;d++){
    if(bcs[d] == "PERIODIC") boundary[d] == BND_CND_PRD;
    else if(bcs[d] == "ANTIPERIODIC") boundary[d] == BND_CND_APRD;
    else if(bcs[d] == "GPARITY") boundary[d] == BND_CND_GPARITY;
  }
  checksum = hd.asHex("CHECKSUM");
  //  sscanf(hd.asString("CHECKSUM").c_str(), "%x ", &checksum);

//  ensemble_id = hd.asString("ENSEMBLE_ID");
//  ensemble_label = hd.asString("ENSEMBLE_LABEL");
//  sequence_number = hd.asInt("SEQUENCE_NUMBER");
  creator = hd.asString("CREATOR");
  creator_hardware = hd.asString("CREATOR_HARDWARE");
  creation_date = hd.asString("CREATION_DATE");
  archive_date = hd.asString("ARCHIVE_DATE");

  FPConv fp;
  floating_point = fp.setFileFormat(hd.asString("FLOATING_POINT").c_str());
  std::stringstream dump;
  dump <<  hd.asString("DATA_PER_SITE");
  dump >> data_per_site;
  VRB.Result(cname,fname,"data_per_site=%d\n",data_per_site);
//  exit(-44);
  
}


template<FixGaugeType gfix_type>
void LatGfixHeader<gfix_type>::writeContent(ostream & fout) {
  const char *fname="writeContent(ostream&)";
//  fout.seekp(0,ios::beg);
//  fout << "BEGIN_HEADER" << endl;
  LatNERSCHeader::writeContent(fout);
  switch (gfix_type){
	case FIX_GAUGE_LANDAU:
  	fout << "GF_TYPE = "<< "LANDAU"<< endl; break;
	case FIX_GAUGE_COULOMB_X:
  	fout << "GF_TYPE = "<< "COULOMB_X" << endl; break;
	case FIX_GAUGE_COULOMB_Y:
  	fout << "GF_TYPE = "<< "COULOMB_Y" << endl; break;
	case FIX_GAUGE_COULOMB_Z:
  	fout << "GF_TYPE = "<< "COULOMB_Z" << endl; break;
	case FIX_GAUGE_COULOMB_T:
  	fout << "GF_TYPE = "<< "COULOMB_T" << stp_cnd << endl; break;
	default:
	ERR.General(cname,fname,"Gauge fixing type not defined\n");
  }
  fout << "GF_ACCURACY"<< setprecision(8)  << stp_cnd << endl;
}

template<FixGaugeType gfix_type>
void LatGfixHeader<gfix_type>::read(istream & fin) {
  
  const char *fname="read(istream&)";

  LatNERSCHeader::read(fin);
//  std::sting gf_type = hd.asString("GF_TYPE");
//  Checks should be added   
  
}

CPS_END_NAMESPACE
