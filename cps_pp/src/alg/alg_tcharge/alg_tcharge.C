#include <config.h>
#include <stdio.h>
#include <util/gjp.h>
#include <util/site.h>
#include <util/qcdio.h>
#include <util/gauge_field.h>
#include <alg/alg_tcharge.h>
#include <comms/glb.h>
#include <comms/scu.h>
#include <omp.h>
#include <cassert>
CPS_START_NAMESPACE

//----------------------------------------------------------
//
// alg_tcharge.C
// 
// measures a simple defintion of the
// topological charge
//  
//  Using Clover Leaf   c.f   hep-lat/010610  Eq (6) -- (10)
//-----------------------------------------------------------


/*!
  takes the imaginary part of a matrix
*/
void ZeroReal(Matrix& m)
{
/*  Matrix temp;
  temp.Dagger( m );
  m-=temp;
  m*=0.5;*/

  for(int i = 0; i < 3; i++) {
    m(i, i).real(0);
  }

  for(int x = 0; x < 2; x++) {
    for(int y = x+1; y < 3; y++) {
      Float a = m(x, y).real();
      Float b = m(x, y).imag();
      Float c = m(y, x).real();
      Float d = m(y, x).imag();
      m(x, y).real(0.5 * (a - c));
      m(x, y).imag(0.5 * (b + d));
      m(y, x).real(0.5 * (c - a));
      m(y, x).imag(0.5 * (b + d));
    }
  }
}

/*!
  constructs 

  \f[
  \frac{1}{32 \pi^2} \epsilon_{\mu \nu \eta \nu } {\rm Tr} \left\{ F_{\mu \nu} F_{\eta \nu} \right\}
  \f]

  passed two arrays of matrices holding the approximations for F:

  plaqs[0]  = F_01
  plaqs[1]  = F_02
  plaqs[2]  = F_03
  plaqs[3]  = F_12
  plaqs[4]  = F_13
  plaqs[5]  = F_23
  
*/

Complex MkTop(Matrix plaqs[])
{
  const Float nfactor(-1.0/( 4 * 3.141592654 * 3.141592654 ));

  Matrix Top;

  // negative weight

  Top.DotMEqual( plaqs[1] , plaqs[4] );
  
  Top *= -1.0 ;
  // positive weight 
  
  Top.DotMPlus ( plaqs[2] , plaqs[3] );
  Top.DotMPlus ( plaqs[5] , plaqs[0] );
  
  return nfactor*Top.Tr();
}



/*!
  Calculate Clover leaf (1x1 size) SU(3) Matrix 
  Sheikholeslami, Wohlert, NPB259, 572 (1985)  Eq. (2.9)
*/
void CloverLeaf(GaugeField& gf, Matrix& pl,  int* pos, int mu, int nu)
{
   Matrix P0,P1,P2,P3;

   P0.ZeroMatrix();
   P1.ZeroMatrix();
   P2.ZeroMatrix();
   P3.ZeroMatrix();

   // each direction could be {0,1,2,3,4,5,6,7} coresponding to
   // the directions {n_x, n_y, n_z, n_t, -n_x, -n_y, -n_z, -n_t}


   int dirs0[4]={mu,nu, mu+4, nu+4};
   gfPathOrdProdPlus(P0, gf, pos, dirs0, 4);

   int dirs1[4]={nu+4,mu+4, nu, mu};
   gfPathOrdProdPlus(P1, gf, pos, dirs1, 4);

   int dirs2[4]={nu,mu+4, nu+4, mu};
   gfPathOrdProdPlus(P2, gf, pos, dirs2, 4);

   int dirs3[4]={mu,nu+4, mu+4, nu};
   gfPathOrdProdPlus(P3, gf, pos, dirs3, 4);


   
   P0 -=  P1;
   P0 +=  P2;
   P0 -=  P3;
   P0 *= 0.25;
   
   moveMem((Float*) &pl,(Float*) &P0, 18 * sizeof(Float) );

}


// Calculate Clover leaf (2x1, 1x2 size) SU(3) Matrix 
// Sheikholeslami, Wohlert, NPB259, 572 (1985)  Eq. (2.9)
// hep-lat/010610  Eq (8)
void CloverLeafRect(GaugeField& gf, Matrix& pl,  int* pos, int mu, int nu)
{
   Matrix P0,P1,P2,P3;


   // 1x2 size
   P0.ZeroMatrix();
   P1.ZeroMatrix();
   P2.ZeroMatrix();
   P3.ZeroMatrix();

   // each direction could be {0,1,2,3,4,5,6,7} coresponding to
   // the directions {n_x, n_y, n_z, n_t, -n_x, -n_y, -n_z, -n_t}


   {
     int dirs0[6]={mu,mu, nu, mu+4,mu+4, nu+4};
     gfPathOrdProdPlus(P0, gf, pos, dirs0, 6);
     

     int dirs1[6]={nu+4,mu+4,mu+4, nu, mu,mu};
     gfPathOrdProdPlus(P1, gf, pos, dirs1, 6);

     int dirs2[6]={nu,mu+4,mu+4, nu+4, mu,mu};
     gfPathOrdProdPlus(P2, gf, pos, dirs2, 6);

     int dirs3[6]={mu,mu, nu+4, mu+4,mu+4, nu};
     gfPathOrdProdPlus(P3, gf, pos, dirs3, 6);
   }

   P0 -=  P1;
   P0 +=  P2;
   P0 -=  P3;
   

   moveMem((Float*) &pl,(Float*) &P0, 18 * sizeof(Float) );
   
   // 2x1  size
   P0.ZeroMatrix();
   P1.ZeroMatrix();
   P2.ZeroMatrix();
   P3.ZeroMatrix();

   // each direction could be {0,1,2,3,4,5,6,7} coresponding to
   // the directions {n_x, n_y, n_z, n_t, -n_x, -n_y, -n_z, -n_t}


   {
     int dirs0[6]={mu,nu,nu, mu+4, nu+4,nu+4};
     gfPathOrdProdPlus(P0, gf, pos, dirs0, 6);
     

     int dirs1[6]={nu+4,nu+4,mu+4, nu,nu, mu};
     gfPathOrdProdPlus(P1, gf, pos, dirs1, 6);

     int dirs2[6]={nu,nu,mu+4, nu+4,nu+4, mu};
     gfPathOrdProdPlus(P2, gf, pos, dirs2, 6);
     
     int dirs3[6]={mu,nu+4,nu+4, mu+4, nu, nu};
     gfPathOrdProdPlus(P3, gf, pos, dirs3, 6);
   }

   P0 -=  P1;
   P0 +=  P2;
   P0 -=  P3;

   pl += P0;
   pl *= 1.0/16.0;
}

void CloverLeaf1x3(GaugeField& gf, Matrix& pl,  int* pos, int mu, int nu)
{
   Matrix P0,P1,P2,P3;


   // 1x3 size
   P0.ZeroMatrix();
   P1.ZeroMatrix();
   P2.ZeroMatrix();
   P3.ZeroMatrix();

   // each direction could be {0,1,2,3,4,5,6,7} coresponding to
   // the directions {n_x, n_y, n_z, n_t, -n_x, -n_y, -n_z, -n_t}


   {
     int dirs0[8]={mu,mu, mu,
		   nu, 
		   mu+4,mu+4,mu+4,
		   nu+4};
     gfPathOrdProdPlus(P0, gf, pos, dirs0, 8);
     

     int dirs1[8]={nu+4,
		   mu+4,mu+4, mu+4,
		   nu, 
		   mu,mu,mu };
     gfPathOrdProdPlus(P1, gf, pos, dirs1, 8);

     int dirs2[8]={nu,
		   mu+4,mu+4,mu+4,
		   nu+4, 
		   mu,mu,mu};
     gfPathOrdProdPlus(P2, gf, pos, dirs2, 8);

     int dirs3[8]={mu,mu,mu,
		   nu+4, 
		   mu+4,mu+4, mu+4,
		   nu};
     gfPathOrdProdPlus(P3, gf, pos, dirs3, 8);
   }

   P0 -=  P1;
   P0 +=  P2;
   P0 -=  P3;
   

   moveMem((Float*) &pl,(Float*) &P0, 18 * sizeof(Float) );
   
   // 3x1  size
   P0.ZeroMatrix();
   P1.ZeroMatrix();
   P2.ZeroMatrix();
   P3.ZeroMatrix();

   // each direction could be {0,1,2,3,4,5,6,7} coresponding to
   // the directions {n_x, n_y, n_z, n_t, -n_x, -n_y, -n_z, -n_t}
   {
     int dirs0[8]={mu,
		   nu,nu,nu,
		   mu+4, 
		   nu+4,nu+4,nu+4};
     gfPathOrdProdPlus(P0, gf, pos, dirs0, 8);
     
     int dirs1[8]={nu+4,nu+4,nu+4,
		   mu+4, 
		   nu,nu,nu,
		   mu};
     gfPathOrdProdPlus(P1, gf, pos, dirs1, 8);

     int dirs2[8]={nu,nu,nu,
		   mu+4, 
		   nu+4,nu+4,nu+4,
		   mu};
     gfPathOrdProdPlus(P2, gf, pos, dirs2, 8);
     
     int dirs3[8]={mu,
		   nu+4,nu+4,nu+4,
		   mu+4, 
		   nu, nu, nu};
     gfPathOrdProdPlus(P3, gf, pos, dirs3, 8);
   }

   P0 -=  P1;
   P0 +=  P2;
   P0 -=  P3;

   pl += P0;
   pl *= 1.0/24.0;
}

void CloverLeaf2x2(GaugeField& gf, Matrix& pl,  int* pos, int mu, int nu)
{
  Matrix P0,P1,P2,P3;
  // 1x2 size
  P0.ZeroMatrix();
  P1.ZeroMatrix();
  P2.ZeroMatrix();
  P3.ZeroMatrix();
  
  // each direction could be {0,1,2,3,4,5,6,7} coresponding to
  // the directions {n_x, n_y, n_z, n_t, -n_x, -n_y, -n_z, -n_t}
  int dirs0[8]={mu,mu, nu, nu, mu+4,mu+4, nu+4, nu+4};
  gfPathOrdProdPlus(P0, gf, pos, dirs0, 8);
    
  int dirs1[8]={nu+4, nu+4, mu+4, mu+4, nu, nu, mu,mu };
  gfPathOrdProdPlus(P1, gf, pos, dirs1, 8);
  
  int dirs2[8]={nu, nu, mu+4, mu+4, nu+4, nu+4, mu, mu };
  gfPathOrdProdPlus(P2, gf, pos, dirs2, 8);
  
  int dirs3[8]={mu,mu, nu+4, nu+4, mu+4, mu+4, nu, nu };
  gfPathOrdProdPlus(P3, gf, pos, dirs3, 8);
  
  P0 -=  P1;
  P0 +=  P2;
  P0 -=  P3;
  P0 *= 1.0/16;
   
  moveMem((Float*) &pl,(Float*) &P0, 18 * sizeof(Float) );
  
}

void CloverLeaf3x3(GaugeField& gf, Matrix& pl,  int* pos, int mu, int nu)
{
  Matrix P0,P1,P2,P3;
  // 1x2 size
  P0.ZeroMatrix();
  P1.ZeroMatrix();
  P2.ZeroMatrix();
  P3.ZeroMatrix();
  
  // each direction could be {0,1,2,3,4,5,6,7} coresponding to
  // the directions {n_x, n_y, n_z, n_t, -n_x, -n_y, -n_z, -n_t}
  int dirs0[12]={ mu, mu, mu,
		  nu, nu, nu,
		  mu+4, mu+4, mu+4,
		  nu+4, nu+4, nu+4 };
  gfPathOrdProdPlus(P0, gf, pos, dirs0, 12);
    
  int dirs1[12]={nu+4, nu+4, nu+4,
		 mu+4, mu+4, mu+4,
		 nu,   nu,   nu, 
		 mu,   mu,   mu   };
  gfPathOrdProdPlus(P1, gf, pos, dirs1, 12);
  
  int dirs2[12]={nu, nu, nu,
		 mu+4, mu+4, mu+4,
		 nu+4, nu+4, nu+4,
		 mu,   mu  , mu   };
  gfPathOrdProdPlus(P2, gf, pos, dirs2, 12);
  
  int dirs3[12]={mu  , mu,   mu,
		 nu+4, nu+4, nu+4, 
		 mu+4, mu+4, mu+4, 
		 nu  , nu  , nu };
  gfPathOrdProdPlus(P3, gf, pos, dirs3, 12);
  
  P0 -=  P1;
  P0 +=  P2;
  P0 -=  P3;
  P0 *= 1./(9*4);
   
  moveMem((Float*) &pl,(Float*) &P0, 18 * sizeof(Float) );
}


// number of clover-leaf functions
const int nfunc(5);

typedef void (*leaf_function)(GaugeField&, Matrix&,  int*, int, int);

// map to the functions used
leaf_function leaf_map[5] = { &CloverLeaf,
			      &CloverLeafRect,
			      &CloverLeaf2x2,
			      &CloverLeaf3x3,
			      &CloverLeaf1x3 };
// map to the names
const char* names[5] = { "1x1",
			 "1x2",
			 "2x2",
			 "3x3",
			 "1x3" };


void AlgTcharge::run()
{
  const char fname[] = "run()";
  Lattice& lat( AlgLattice() );  

  static bool initialized = false;
  static Offsets offsets;

  cps::GaugeField gf(lat);
  const int buffer_thickness = 3; //Required expansion in each direction
  gf.resize(buffer_thickness);
  if (!initialized) {
    gf.refresh();
    gf.recordOffsetStart(false, true);
  } else {
    gf.refresh(offsets);
  }

  Float tmat[nfunc] = {0};

  const int t_extent = GJP.Sites(3); //Global number of time slices.
  Float slice_sums[nfunc][t_extent];
  for(int f = 0; f < nfunc; f++) {
    for(int t = 0; t < t_extent; t++) {
      slice_sums[f][t] = 0.0;
    }
  }

#pragma omp parallel
  {
    Float thread_tmat[nfunc] = {0};
    Float thread_slice_sums[nfunc][t_extent];
    for(int f = 0; f < nfunc; f++) {
      for(int t = 0; t < t_extent; t++) {
        thread_slice_sums[f][t] = 0.0;
      }
    }

#pragma omp for
    for(int i = 0; i < GJP.VolNodeSites(); ++i)
    {
      int y[4];
      gf.coordinatesFromIndex(y, i);

      int index = 0;
      Matrix plaqs[nfunc][6];
      for(int mu = 0; mu < 3; ++mu)
        for(int nu = mu + 1; nu < 4; ++nu)
        {
          for(int f(0); f < nfunc; ++f)
          {
            (*(leaf_map[f]))(gf, plaqs[f][index], y, mu, nu);
            ZeroReal(plaqs[f][index]);
          }
          index++;
        }

      int t = y[3] + GJP.TnodeSites() * GJP.TnodeCoor(); //global time slice index

      for(int f = 0; f < nfunc; ++f) {
        Float top = MkTop(plaqs[f]).real();

        thread_tmat[f] += top;
        thread_slice_sums[f][t] += top;
      }
    }

    //reduce results
#pragma omp critical
    {
      for(int f = 0; f < nfunc; f++) {
        tmat[f] += thread_tmat[f];
        for(int t = 0; t < t_extent; t++) {
          slice_sums[f][t] += thread_slice_sums[f][t];
        }
      }
    }
  }

  //Sum results over nodes
#ifdef USE_QMP
  QMP_sum_double_array(tmat, nfunc);
  QMP_sum_double_array(&slice_sums[0][0], nfunc*t_extent);
#endif

  if (!initialized) {
    offsets = gf.getRecordedOffsets();
    gf.recordOffsetEnd();
    initialized = true;
  }


  if(common_arg->filename != 0)
  {
    FILE *fp;
    if((fp = Fopen(common_arg->filename, "a")) == NULL)
      ERR.FileA(cname, fname, common_arg->filename);
    Fprintf(fp, "AlgTcharge:\n");
    Fprintf(fp, "nleaf : %i\n", nfunc);
    for(int f(0); f < nfunc; ++f)
      Fprintf(fp,"   %i : %s\n", f, names[f]);
    for (int f(0); f < nfunc; ++f)
      Fprintf(fp, "%i %i : %15e\n", f, f, tmat[f]);

    for(int f = 0; f < nfunc; f++) {
      Fprintf(fp, "%i : ", f);
      for(int t = 0; t < GJP.Sites(3); t++) {
        Fprintf(fp, "%15e ", slice_sums[f][t]);
      }
      Fprintf(fp, "\n");
    }

    Fclose(fp);
  }
}


CPS_END_NAMESPACE

