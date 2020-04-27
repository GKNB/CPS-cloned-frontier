#include<config.h>
CPS_START_NAMESPACE
//------------------------------------------------------------------
/*!\file
  \brief Implementation of AlgFixGauge class methods.

  $Id: alg_fix_gauge.C,v 1.11 2007/06/25 15:49:20 chulwoo Exp $
*/
//------------------------------------------------------------------
//  CVS keywords
//
//  $Author: chulwoo $
//  $Date: 2007/06/25 15:49:20 $
//  $Header: /space/cvs/cps/cps++/src/alg/alg_fix_gauge/alg_fix_gauge.C,v 1.11 2007/06/25 15:49:20 chulwoo Exp $
//  $Id: alg_fix_gauge.C,v 1.11 2007/06/25 15:49:20 chulwoo Exp $
//  $Name: v5_0_16_hantao_io_test_v7 $
//  $Locker:  $
//  $RCSfile: alg_fix_gauge.C,v $
//  $Revision: 1.11 $
//  $Source: /space/cvs/cps/cps++/src/alg/alg_fix_gauge/alg_fix_gauge.C,v $
//  $State: Exp $
//
//--------------------------------------------------------------------


CPS_END_NAMESPACE
#include <util/lattice.h>
#include <util/gjp.h>
#include <util/site.h>
#include <alg/alg_fix_gauge.h>
#include <util/smalloc.h>
#include <util/vector.h>
#include <util/verbose.h>
#include <util/error.h>
#include <util/ReadNERSC.h>
#include <util/WriteNERSC.h>
CPS_START_NAMESPACE



//------------------------------------------------------------------
/*!
  \param latt The lattice object containing the gauge field to be fixed.
  \param c_arg Generic algorithm parameters
  \param arg Gauge fixing parameters.
*/
//------------------------------------------------------------------
AlgFixGauge::AlgFixGauge(Lattice& latt, 
			 CommonArg *c_arg,
			 FixGaugeArg *arg) :
			 Alg(latt, c_arg) 
{
  cname = "AlgFixGauge";
  const char *fname = "AlgFixGauge";
  VRB.Func(cname,fname);

  // Initialize the argument pointer
  //----------------------------------------------------------------
  if(arg == 0)
    ERR.Pointer(cname,fname, "arg");
  alg_fix_gauge_arg = arg;

}


//------------------------------------------------------------------
/*!
  \note The destructor does not free the memory allocated.
  Use AlgFixGauge::free for this or Lattice::FixGaugeFree
*/
//------------------------------------------------------------------
AlgFixGauge::~AlgFixGauge() {
  const char *fname = "~AlgFixGauge";
  VRB.Func(cname,fname);
}


//------------------------------------------------------------------
// Allocates memory and constructs the gauge fixing matrices
//------------------------------------------------------------------
void AlgFixGauge::run(const QioArg *rd_arg)
{
  const char *fname = "run";
  VRB.Func(cname,fname);

  // Set the Lattice pointer
  //----------------------------------------------------------------
  Lattice& lat = AlgLattice();

  // Set up arguments
  //----------------------------------------------------------------
  FixGaugeType fix = alg_fix_gauge_arg->fix_gauge_kind;
  int start = alg_fix_gauge_arg->hyperplane_start;
  int step = alg_fix_gauge_arg->hyperplane_step;
  num = alg_fix_gauge_arg->hyperplane_num;
  int *h_planes = NULL;

  // If coulomb gauge
  //----------------------------------------------------------------
  if( (fix == FIX_GAUGE_COULOMB_X) ||
      (fix == FIX_GAUGE_COULOMB_Y) ||
      (fix == FIX_GAUGE_COULOMB_Z) ||
      (fix == FIX_GAUGE_COULOMB_T)  ){ 

      switch(fix){
      case FIX_GAUGE_COULOMB_X:
	  lattice_dir_size = GJP.XnodeSites() * GJP.Xnodes();
	  break;
      case FIX_GAUGE_COULOMB_Y:
	  lattice_dir_size = GJP.YnodeSites() * GJP.Ynodes();
	  break;
      case FIX_GAUGE_COULOMB_Z:
	  lattice_dir_size = GJP.ZnodeSites() * GJP.Znodes();
	  break;
      case FIX_GAUGE_COULOMB_T:
	  lattice_dir_size = GJP.TnodeSites() * GJP.Tnodes();
	  break;
      case FIX_GAUGE_LANDAU:
      case FIX_GAUGE_NONE:
	  break;
      }

    if(start+step*(num-1) >= lattice_dir_size)
	ERR.General(cname, fname,
		    "The coordinate of the last hyperplane (%d+%d*%d) is greater than the global lattice size.", 
		    start, step, num-1, lattice_dir_size);

    h_planes = (int *) smalloc(num * sizeof(int));
    if(h_planes == 0) ERR.Pointer(cname,fname, "h_planes");
    VRB.Smalloc(cname,fname, "h_planes", h_planes, num * sizeof(int));
    
    for(int i=0; i<num; i++) h_planes[i] = start + step * i;

  } 

  // Allocate gauge fixing matrices and set them to 1
  //----------------------------------------------------------------
  lat.FixGaugeAllocate(fix, num, h_planes);
  if(rd_arg)
  this->Load(*rd_arg);


  // Calculate the gauge fixing matrices
  //----------------------------------------------------------------
// added to make it possible to explicitly bypass gauge fixin:w
  if ( alg_fix_gauge_arg->max_iter_num > 0 ) 
  lat.FixGauge(alg_fix_gauge_arg->stop_cond, 
	       alg_fix_gauge_arg->max_iter_num);

  VRB.Sfree(cname,fname, "h_planes",h_planes);
  sfree(h_planes);

}

//------------------------------------------------------------------
// Free the memory of the gauge fixing matrices.
//------------------------------------------------------------------
void AlgFixGauge::free()
{
  char *fname = "free()";
  VRB.Func(cname,fname);

  // Set the Lattice pointer
  //----------------------------------------------------------------
  Lattice& lat = AlgLattice();

  // Free gauge fixing matrices
  //----------------------------------------------------------------
  lat.FixGaugeFree();
}

void AlgFixGauge::Save(const QioArg &wt_arg, int pario){

  const char *fname("Save()");
  Lattice& lat = AlgLattice();
  LatMatrix gfix_mat(1);

  Site s;
  while(s.LoopsOverNode()){
//	int *pos = s.pos();
	const Matrix *mat1 = lat.FixGaugeMatrix(s.pos());
	if(mat1 !=NULL ){
		Matrix *mat2 = gfix_mat.Mat(s.Index());
		*mat2 = *mat1;
	Float *tmp_p = (Float*)mat2;
//	printf("LoopsOverNode: %d %d %d %d index %d %0.6e %0.6e %0.6e %0.6e %0.6e %0.6e\n", 
	VRB.Debug(cname,fname,"LoopsOverNode: %d %d %d %d index %d %0.6e %0.6e %0.6e %0.6e %0.6e %0.6e\n", 
	s.physX(),s.physY(),s.physZ(),s.physT(), s.Index(),
	tmp_p[0],tmp_p[1],tmp_p[2],tmp_p[3],tmp_p[4],tmp_p[5]);
        } else {
	ERR.General(cname,fname,"Gauge fix matrix should be available on every sites!\n");
        }
  }
//  WriteNERSC<LatGfixHeader < alg_fix_gauge_arg->fix_gauge_kind >,4,Float> nersc_write(gfix_mat.Nelem());
  std::vector <std::string > key;
  key.push_back("GF_TYPE");
  key.push_back("GF_ACCRUACY");

  std::vector <std::string > value;
  switch ( alg_fix_gauge_arg->fix_gauge_kind){
        case FIX_GAUGE_LANDAU:
        value.push_back( "LANDAU"); break;
        case FIX_GAUGE_COULOMB_X:
        value.push_back( "COULOMB_X"); break;
        case FIX_GAUGE_COULOMB_Y:
        value.push_back( "COULOMB_Y"); break;
        case FIX_GAUGE_COULOMB_Z:
        value.push_back( "COULOMB_Z"); break;
        case FIX_GAUGE_COULOMB_T:
        value.push_back( "COULOMB_T"); break;
        default:
        ERR.General(cname,fname,"Gauge fixing type not defined\n");
  }
  std::stringstream accuracy;
  accuracy.precision(8);  
  accuracy << alg_fix_gauge_arg->stop_cond;
  value.push_back(accuracy.str());


  WriteNERSC<LatNERSCHeader,4,Float> nersc_write(gfix_mat.Nelem(),key,value);
  VRB.Result(cname,fname,"pario=%d resetting to serial\n",pario);
//  if (!pario ) 
  nersc_write.setSerial();
  nersc_write.write(gfix_mat.Field(),wt_arg);
}

void AlgFixGauge::Load(const QioArg &rd_arg){

  const char *fname("Load()");
  Lattice& lat = AlgLattice();
  LatMatrix gfix_mat(1);

  ReadNERSC<LatNERSCHeader,4,Float> nersc_read(gfix_mat.Nelem());
//  Being lazy. Should be fixed eventually!
//  ReadNERSC<LatNERSCHeader,4,Float> nersc_read(18);
  nersc_read.read(gfix_mat.Field(),rd_arg);

  Site s;
  while(s.LoopsOverNode()){
//	int *pos = s.pos();
//	const Matrix *mat1 = lat.FixGaugeMatrix(s.pos());
	Matrix *mat2 = gfix_mat.Mat(s.Index());
	Float *tmp_p = (Float*)mat2;
	VRB.Debug(cname,fname,"LoopsOverNode: %d %d %d %d index %d %0.6e %0.6e %0.6e %0.6e %0.6e %0.6e\n", 
	s.physX(),s.physY(),s.physZ(),s.physT(), s.Index(),
	tmp_p[0],tmp_p[1],tmp_p[2],tmp_p[3],tmp_p[4],tmp_p[5]);
	lat.SetFixGaugeMatrix(*mat2,s.pos());
  }
}


CPS_END_NAMESPACE
