#include<config.h>
CPS_START_NAMESPACE
//------------------------------------------------------------------
/*!\file
  \brief  Definitions of the AlgFixGauge class.

  $Id: alg_fix_gauge.h,v 1.3 2004/09/02 16:56:28 zs Exp $
*/
//------------------------------------------------------------------


#ifndef INCLUDED_ALG_FIX_GAUGE_H
#define INCLUDED_ALG_FIX_GAUGE_H

CPS_END_NAMESPACE
#include <util/lattice.h>
#include <util/qioarg.h>
#include <alg/alg_base.h>
#include <alg/common_arg.h>
#include <alg/fix_gauge_arg.h>
CPS_START_NAMESPACE

//! Performs gauge fixing.
/*!
  The gauge can be fixed to Landau gauge, in which case this is done
  at every lattice site, or to Coulomb gauge, in which case the gauge is
  fixed on hyperplanes normal to a specified direction.
  You can specify the first hyperplane, the number of hyperplanes and the
  distance between hyperplanes.

    \ingroup alg
 */
class AlgFixGauge : public Alg
{
 private:
    char *cname;
  int num, lattice_dir_size;


    FixGaugeArg *alg_fix_gauge_arg;
        // The argument structure for the AlgFixGauge algorithm
 

 public:
    AlgFixGauge(Lattice & latt, CommonArg *c_arg, FixGaugeArg *arg);
//:Alg(),num(0),lattice_dir_size(0);

    virtual ~AlgFixGauge();

    //  Constructs the gauge fixing matrices.
    /*!\post The Lattice class allocates memory for the gauge fixing matrices
      which can be accessed using Lattice::FixGaugePtr.
     */
    void run(const QioArg *rd_arg=NULL);  

    void Save(const QioArg &wt_arg, int pario = 1);
    void Save(const char *filename, int pario = 1){
        QioArg wt_arg(filename, pario);
        this->Save(wt_arg);
    }
    void Load(const QioArg &rd_arg);
    void Load(const char *filename){
        QioArg rd_arg(filename);
        this->Load(rd_arg);
    }

    // Frees the memory allocated for the gauge fixing matrices.
    void free(void);

};




#endif





CPS_END_NAMESPACE
