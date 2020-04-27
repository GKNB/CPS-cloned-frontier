#include <config.h>
#include <alg/alg_actiondensity.h>
#include <stdio.h>
#include <util/gjp.h>
#include <util/site.h>
#include <util/qcdio.h>
#include <util/gauge_field.h>
#include <comms/glb.h>
#include <comms/scu.h>
#include <omp.h>
#include <util/time_cps.h>
#include <cassert>
CPS_START_NAMESPACE

/*!
  takes the imaginary part of a matrix
*/
void AlgActionDensity::ZeroReal(Matrix& m)
{
/*  Matrix temp;
  temp.Dagger( m );
  m-=temp;
  m*=0.5;*/

  //Replacing above with below gives a factor of 2 speedup for the entire smartrun:
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
  Computes (1/2) tr(F_mu_nu F_mu_nu)

  Input should be a set of six matrices giving the field strength
  tensor F_mu_nu in each direction:

  clovers[0]  = F_01
  clovers[1]  = F_02
  clovers[2]  = F_03
  clovers[3]  = F_12
  clovers[4]  = F_13
  clovers[5]  = F_23
  
*/

Complex AlgActionDensity::ActionDensity(Matrix clovers[])
{
  Matrix action;

  action.ZeroMatrix();

  for(int i = 0; i < 6; i++) {
    action.DotMPlus(clovers[i], clovers[i]);
  }

  //Minus sign accounts for the fact that that the input is actual i*F_mu_nu
  return -action.Tr();
}


/*!
  Calculate Clover leaf (1x1 size) SU(3) Matrix 
  Sheikholeslami, Wohlert, NPB259, 572 (1985)  Eq. (2.9)
*/
void AlgActionDensity::CloverLeaf(GaugeField& gf, Matrix& pl,  int* pos, int mu, int nu)
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

void AlgActionDensity::run(Float *result)
{
  const char fname[] = "run()";
  Lattice& lat( AlgLattice() );  

  static bool initialized = false;
  static Offsets offsets;

  cps::GaugeField gf(lat);
  const int buffer_thickness = 1; //Required expansion in each direction
  gf.resize(buffer_thickness);
  if (!initialized) {
    gf.refresh();
    gf.recordOffsetStart(false, true);
  } else {
    gf.refresh(offsets);
  }

  Float action = 0;  

  const int t_extent = GJP.Sites(3); //Global number of time slices.

  //Action summed over each time slice:
  Float slice_actions[t_extent];
  for(int t = 0; t < t_extent; t++) slice_actions[t] = 0.0;

#pragma omp parallel reduction(+:action)
  {
    //Thread-local sums over time slices, since OpenMP can't automatically
    //reduce arrays.
    Float thread_slice_actions[t_extent];
    for(int t = 0; t < t_extent; t++) thread_slice_actions[t] = 0.0;

#pragma omp for 
    for(int i = 0; i < GJP.VolNodeSites(); ++i)
    {
      int y[4];
      gf.coordinatesFromIndex(y, i);

      int index = 0;
      Matrix clovers[6];
      for(int mu = 0; mu < 3; ++mu)
        for(int nu = mu + 1; nu < 4; ++nu)
        {
          CloverLeaf(gf, clovers[index], y, mu, nu);
          ZeroReal(clovers[index]);
          index++;
        }

      Float a = ActionDensity(clovers).real();
      action += a;

      int t = y[3] + GJP.TnodeSites() * GJP.TnodeCoor(); //global time slice index
      thread_slice_actions[t] += a;
    }

    //Manually reduce the slice actions
#pragma omp critical
    {
      for(int t = 0; t < t_extent; t++) {
        slice_actions[t] += thread_slice_actions[t];
      }
    }
  } //action is automatically reduced

  if (!initialized) {
    offsets = gf.getRecordedOffsets();
    gf.recordOffsetEnd();
    initialized = true;
  }

  //Sum results over all nodes.
#ifdef USE_QMP
  glb_sum(&action);
  QMP_sum_double_array(slice_actions, t_extent);
#endif

  Float action_density = action / GJP.VolSites();
  if(result) *result = action_density;

  const int slice_sites = GJP.Sites(0) * GJP.Sites(1) * GJP.Sites(2);
  Float slice_densities[t_extent];
  for(int t = 0; t < t_extent; t++) {
    slice_densities[t] = slice_actions[t] / slice_sites;
  }

  //Print out results
  if(common_arg->filename != 0) {
    char *fname = "run()";
    FILE *fp;
    if( (fp = Fopen(common_arg->filename, "a")) == NULL ) {
      ERR.FileA(cname,fname,common_arg->filename);
    }

    //Prints mean value of (1/2)tr(F_mu_nu F_mu_nu), in units of a^4
    Fprintf(fp, "%15e\n", action_density);

    for(int t = 0; t < t_extent; t++) {
      Fprintf(fp, "%15e ", slice_densities[t]);
    }
    Fprintf(fp, "\n");
      
    Fclose(fp);
  }
}

#if 0
// A communication efficient way of calculating the t-charge
// Pass the surface slab to adjacent nodes once
// Do all the calculation locally.
// Issues:
// Put the orignal lattice in the center of the local fields
// Assemble/disemble a continuous memory containning all the
// data before/after the communication
// If local size is too small, we construct a larger local 
// volume and pass on the data to the next neighbor
// Local size  >= 2 & Slab = 3
// Twice as large on each dimension will suffice.

void AlgActionDensity::smartrun(Float* result)
{
  const char fname[] = "smartrun()";
  Lattice& lat( AlgLattice() );  
#if 1
  ERR.General(cname,fname,"Not ported to GaugeField yet");
#else

  const int Slab = 1; //Expansion in each direction

  if(GJP.Gparity() && Slab!=1){ //In this case the local field is doubled up and I'm not sure my G-parity changes will work
    if(!UniqueID()) printf("AlgActionDensity::smartrun(Float* result) : Code not implemented to deal with surface slab sizes > 1\n");
    exit(-1);
  }

  const int MatrixSize = 2 * lat.Colors() * lat.Colors();
  const int GsiteSize = 4 * MatrixSize;
  int l_node_sites[4] = {
    GJP.XnodeSites(),
    GJP.YnodeSites(),
    GJP.ZnodeSites(),
    GJP.TnodeSites()};
  int l_dir_offset[4];
  l_dir_offset[0] = GsiteSize;
  l_dir_offset[1] = l_dir_offset[0] * l_node_sites[0];
  l_dir_offset[2] = l_dir_offset[1] * l_node_sites[1];
  l_dir_offset[3] = l_dir_offset[2] * l_node_sites[2];

  int vol_node_sites = GJP.VolNodeSites();
  int flag[4] = {0, 0, 0, 0};  //Flags for too small a local dimension

  // lfield for original local gauge field
  Float * lfield = (Float*) smalloc(cname, fname, "lfield", GsiteSize * vol_node_sites * sizeof(Float));
  memcpy(lfield, (Float *)lat.GaugeField(), GsiteSize * vol_node_sites * sizeof(Float));

  int x[4], y[4];
  Float *g_offset;
  Float *l_offset;

  // If local dimension k < Slab, expand it to twice as large
  for(int k = 0; k < 4; ++k)
  {
    if(l_node_sites[k] >= Slab)
      continue;

    flag[k] = 1;

    Float * afield = (Float*) smalloc(cname, fname, "afield", GsiteSize * vol_node_sites * sizeof(Float));

    getPlusData(afield, lfield, GsiteSize * vol_node_sites, k);

    if(GJP.Bc(k) == BND_CND_GPARITY && GJP.NodeCoor(k) == GJP.Nodes(k)-1) for(int f=1;f<GsiteSize * vol_node_sites;f+=2) afield[f]*=-1; //complex conjugate links received from over G-parity boundary

    Float * bfield = lfield;
    lfield = (Float*) smalloc(cname, fname, "lfield", 2 * GsiteSize * vol_node_sites * sizeof(Float));

    int x_dir_offset[4];
    for(int i = 0; i < 4; ++i)
      x_dir_offset[i] = l_dir_offset[i];
    for(int i = k + 1; i < 4; ++i)
      x_dir_offset[i] *= 2;

    x[0] = 0;
    for(x[1] = 0; x[1] < l_node_sites[1]; ++x[1])
      for(x[2] = 0; x[2] < l_node_sites[2]; ++x[2])
        for(x[3] = 0; x[3] < l_node_sites[3]; ++x[3])
        {
          l_offset = GsiteOffset(bfield, x, l_dir_offset);
          g_offset = GsiteOffset(lfield, x, x_dir_offset);
          memcpy(g_offset, l_offset, l_dir_offset[1] * sizeof(Float));

          l_offset = GsiteOffset(afield, x, l_dir_offset);
          x[k] += l_node_sites[k];
          g_offset = GsiteOffset(lfield, x, x_dir_offset);
          memcpy(g_offset, l_offset, l_dir_offset[1] * sizeof(Float));
          x[k] -= l_node_sites[k];
        }
    sfree(cname, fname, "afield", afield);
    sfree(cname, fname, "bfield", bfield);

    l_node_sites[k] *= 2;
    for(int i = k + 1; i < 4; ++i)
      l_dir_offset[i] *= 2;
    vol_node_sites *= 2;
  }

  const int g_node_sites[4] = {
    l_node_sites[0] + 2 * Slab,
    l_node_sites[1] + 2 * Slab,
    l_node_sites[2] + 2 * Slab,
    l_node_sites[3] + 2 * Slab};

  int g_dir_offset[4];
  g_dir_offset[0] = GsiteSize;
  g_dir_offset[1] = g_dir_offset[0] * g_node_sites[0];
  g_dir_offset[2] = g_dir_offset[1] * g_node_sites[1];
  g_dir_offset[3] = g_dir_offset[2] * g_node_sites[2];

  const int g_lcl_vol = g_node_sites[0]
    * g_node_sites[1]
    * g_node_sites[2]
    * g_node_sites[3];

  Float * gfield = (Float*) smalloc(cname, fname, "gfield", GsiteSize * g_lcl_vol * sizeof(Float));

  int xsta[4] = {0, 0, 0, 0};
  int xend[4] = {l_node_sites[0],
    l_node_sites[1],
    l_node_sites[2],
    l_node_sites[3]};
  int ysta[4] = {Slab, Slab, Slab, Slab};

  //unnecessary
  int yend[4] = {l_node_sites[0] + Slab,
    l_node_sites[1] + Slab,
    l_node_sites[2] + Slab,
    l_node_sites[3] + Slab};

  // ------------------------------------------------------------------
  // Move the whole local gauge field to a new location
  // Every dimension is shifted by Slab
  // From L: 0    < x[k] < l_node_sites[k]
  // To   G: Slab < y[k] < l_node_sites[k] + Slab
  // ------------------------------------------------------------------
  x[0] = xsta[0];
  y[0] = ysta[0];
  for(x[1] = xsta[1], y[1] = ysta[1]; x[1] < xend[1]; ++x[1], ++y[1])
    for(x[2] = xsta[2], y[2] = ysta[2]; x[2] < xend[2]; ++x[2], ++y[2])
      for(x[3] = xsta[3], y[3] = ysta[3]; x[3] < xend[3]; ++x[3], ++y[3])
      {
        g_offset = GsiteOffset(gfield, y, g_dir_offset);
        l_offset = GsiteOffset(lfield, x, l_dir_offset);
        memcpy(g_offset, l_offset, l_dir_offset[1] * sizeof(Float));
      }

  Float *surf0;
  Float *surf1;
  Float *surfx;
  int SurfSize;
  int s_dir_offset[4];

  // ------------------------------------------------------------------
  // Propagate the surface slab to neighboring nodes
  // Loop over direction
  // ------------------------------------------------------------------
  for(int i = 0; i < 4; ++i)
  {
    // ------------------------------------------------------------------
    // Set offset on each direction
    // 0 < x[i] < Slab
    // 0 < x[k != i] < l_node_sites[k]
    // ------------------------------------------------------------------
    for(int k = 0; k < 4; ++k)
      s_dir_offset[k] = l_dir_offset[k];
    for(int k = i + 1; k < 4; ++k)
      s_dir_offset[k] = s_dir_offset[k] * Slab / l_node_sites[i];
    xend[i] = Slab;

    // ------------------------------------------------------------------
    // Allocate momory for the data
    // ------------------------------------------------------------------
    SurfSize = vol_node_sites * Slab / l_node_sites[i]; 
    surf0 = (Float*) smalloc(cname, fname, "surf0", GsiteSize * SurfSize * sizeof(Float));
    surf1 = (Float*) smalloc(cname, fname, "surf1", GsiteSize * SurfSize * sizeof(Float));
    if(flag[i])
      surfx = (Float*) smalloc(cname, fname, "surf1", GsiteSize * SurfSize * sizeof(Float));

    // ------------------------------------------------------------------
    // Assemble the data to propagate in the negative "i" direction
    // From L: 0 < x[i] < Slab
    //         0 < x[k != i] < l_node_sites[k]
    // To   S: 0 < x[i] < Slab
    //         0 < x[k != i] < l_node_sites[k]
    // ------------------------------------------------------------------
    x[0] = xsta[0];
    for(x[1] = xsta[1]; x[1] < xend[1]; ++x[1])
      for(x[2] = xsta[2]; x[2] < xend[2]; ++x[2])
        for(x[3] = xsta[3]; x[3] < xend[3]; ++x[3])
        {
          g_offset = GsiteOffset(surf0, x, s_dir_offset);
          l_offset = GsiteOffset(lfield, x, l_dir_offset);
          memcpy(g_offset, l_offset, s_dir_offset[1] * sizeof(Float));
        }

    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    if(GJP.Bc(i) == BND_CND_GPARITY && GJP.NodeCoor(i) == 0) for(int f=1;f<GsiteSize * SurfSize;f+=2) surf0[f]*=-1;

    if(flag[i])
    {
      getPlusData(surfx, surf0, GsiteSize * SurfSize, i);

      if(GJP.Bc(i) == BND_CND_GPARITY && GJP.NodeCoor(i) == 0) for(int f=1;f<GsiteSize * SurfSize;f+=2) surfx[f]*=-1;

      getPlusData(surf1, surfx, GsiteSize * SurfSize, i);
    }
    else
      getPlusData(surf1, surf0, GsiteSize * SurfSize, i);

    // ------------------------------------------------------------------
    // Dissemble the received data to the allocated memory 
    // From S: 0 < x[i] < Slab
    //         0 < x[k != i] < l_node_sites[k]
    // To   G: l_node_sites[i] + Slab < y[i] < l_node_sites[i] + 2 * Slab
    //         Slab < y[k != i] < l_node_sites[k] + Slab
    // ------------------------------------------------------------------
    ysta[i] = l_node_sites[i] + Slab;
    x[0] = xsta[0];
    y[0] = ysta[0];
    for(x[1] = xsta[1], y[1] = ysta[1]; x[1] < xend[1]; ++x[1], ++y[1])
      for(x[2] = xsta[2], y[2] = ysta[2]; x[2] < xend[2]; ++x[2], ++y[2])
        for(x[3] = xsta[3], y[3] = ysta[3]; x[3] < xend[3]; ++x[3], ++y[3])
        {
          l_offset = GsiteOffset(surf1, x, s_dir_offset);
          g_offset = GsiteOffset(gfield, y, g_dir_offset);
          memcpy(g_offset, l_offset, s_dir_offset[1] * sizeof(Float));
        }
    ysta[i] = Slab;

    // ------------------------------------------------------------------
    // If l_node_site[i] == Slab,  we only need to pass the same slab
    // to the opposite direcction
    // Otherwise we need to assemble the appropriate slab again
    // From L: l_node_sites[i] - Slab < x[i] < l_node_sites[i]
    //         0 < x[k != i] < l_node_sites[k]
    // To   S: 0 < x[i] < Slab
    //         0 < x[k != i] < l_node_sites[k]
    // ------------------------------------------------------------------
    if(l_node_sites[i] != Slab)
    {
      ysta[i] = l_node_sites[i];
      x[0] = xsta[0];
      y[0] = ysta[0];
      for(x[1] = xsta[1], y[1] = ysta[1]; x[1] < xend[1]; ++x[1], ++y[1])
        for(x[2] = xsta[2], y[2] = ysta[2]; x[2] < xend[2]; ++x[2], ++y[2])
          for(x[3] = xsta[3], y[3] = ysta[3]; x[3] < xend[3]; ++x[3], ++y[3])
          {
            g_offset = GsiteOffset(surf0, x, s_dir_offset);
            l_offset = GsiteOffset(gfield, y, g_dir_offset);
            memcpy(g_offset, l_offset, s_dir_offset[1] * sizeof(Float));
          }
      ysta[i] = Slab;
    }

    if(GJP.Bc(i) == BND_CND_GPARITY && GJP.NodeCoor(i) == GJP.Nodes(i)-1) for(int f=1;f<GsiteSize * SurfSize;f+=2) surf0[f]*=-1;

    if(flag[i])
    {      
      getMinusData(surfx, surf0, GsiteSize * SurfSize, i);

      if(GJP.Bc(i) == BND_CND_GPARITY && GJP.NodeCoor(i) == GJP.Nodes(i)-1) for(int f=1;f<GsiteSize * SurfSize;f+=2) surfx[f]*=-1;

      getMinusData(surf1, surfx, GsiteSize * SurfSize, i);
    }
    else
      getMinusData(surf1, surf0, GsiteSize * SurfSize, i);

    // ------------------------------------------------------------------
    // Dissemble the received data to the allocated memory 
    // From S: 0 < x[i] < Slab
    //         0 < x[k != i] < l_node_sites[k]
    // To   G: 0 < y[i] < Slab
    //         Slab < y[k != i] < l_node_sites[k] + Slab
    // ------------------------------------------------------------------
    ysta[i] = 0;
    x[0] = xsta[0];
    y[0] = ysta[0];
    for(x[1] = xsta[1], y[1] = ysta[1]; x[1] < xend[1]; ++x[1], ++y[1])
      for(x[2] = xsta[2], y[2] = ysta[2]; x[2] < xend[2]; ++x[2], ++y[2])
        for(x[3] = xsta[3], y[3] = ysta[3]; x[3] < xend[3]; ++x[3], ++y[3])
        {
          l_offset = GsiteOffset(surf1, x, s_dir_offset);
          g_offset = GsiteOffset(gfield, y, g_dir_offset);
          memcpy(g_offset, l_offset, s_dir_offset[1] * sizeof(Float));
        }
    ysta[i] = Slab;

    sfree(cname, fname, "surf0", surf0);
    sfree(cname, fname, "surf1", surf1);
    if(flag[i])
      sfree(cname, fname, "surfx", surfx);

    // ------------------------------------------------------------------
    // Propagate the cornered chunk to neighboring nodes
    // Loop over direction
    // ------------------------------------------------------------------
    for(int j = i + 1; j < 4; ++j)
    {
      // ------------------------------------------------------------------
      // 0 < x[i, j] < Slab
      // 0 < x[k != i, j] < l_node_sites[k]
      // ------------------------------------------------------------------
      for(int k = i + 1; k < 4; ++k)
        s_dir_offset[k] = l_dir_offset[k] * Slab / l_node_sites[i];
      for(int k = j + 1; k < 4; ++k)
        s_dir_offset[k] = s_dir_offset[k] * Slab / l_node_sites[j];

      xend[j] = Slab;

      Float * surf2;

      // ------------------------------------------------------------------
      // Every chunk contains two corner pieces
      // ------------------------------------------------------------------
      SurfSize = vol_node_sites * Slab  * Slab / l_node_sites[i] / l_node_sites[j]; 
      surf0 = (Float*) smalloc(cname, fname, "surf0", 2 * GsiteSize * SurfSize * sizeof(Float));
      surf1 = (Float*) smalloc(cname, fname, "surf1", 2 * GsiteSize * SurfSize * sizeof(Float));
      if(flag[j])
        surfx = (Float*) smalloc(cname, fname, "surfx", 2 * GsiteSize * SurfSize * sizeof(Float));

      // ------------------------------------------------------------------
      // Assemble the data to propagate in the negative "j" direction
      // From G: 0 < y[i] < Slab
      //         Slab < y[j] <  2 * Slab
      //         Slab < y[k != i, j] < l_node_sites[k] + Slab
      // To   S: 0 < x[i, j] < Slab
      //         0 < x[k != i,j] < l_node_sites[k]
      // Also:
      // From G: l_node_sites[i] + Slab < y[i] < l_node_sites[i] + 2 * Slab
      //         Slab < y[j] <  2 * Slab
      //         Slab < y[k != i, j] < l_node_sites[k] + Slab
      // To   S: 0 < x[i, j] < Slab
      //         0 < x[k != i,j] < l_node_sites[k]
      // ------------------------------------------------------------------
      surf2 = surf0 + GsiteSize * SurfSize;
      ysta[i] = 0;
      x[0] = xsta[0];
      y[0] = ysta[0];
      for(x[1] = xsta[1], y[1] = ysta[1]; x[1] < xend[1]; ++x[1], ++y[1])
        for(x[2] = xsta[2], y[2] = ysta[2]; x[2] < xend[2]; ++x[2], ++y[2])
          for(x[3] = xsta[3], y[3] = ysta[3]; x[3] < xend[3]; ++x[3], ++y[3])
          {
            l_offset = GsiteOffset(surf0, x, s_dir_offset);
            g_offset = GsiteOffset(gfield, y, g_dir_offset);
            memcpy(l_offset, g_offset, s_dir_offset[1] * sizeof(Float));
            y[i] += l_node_sites[i] + Slab;
            l_offset = GsiteOffset(surf2, x, s_dir_offset);
            g_offset = GsiteOffset(gfield, y, g_dir_offset);
            memcpy(l_offset, g_offset, s_dir_offset[1] * sizeof(Float));
            y[i] -= l_node_sites[i] + Slab;
          }
      ysta[i] = Slab;

      if(GJP.Bc(j) == BND_CND_GPARITY && GJP.NodeCoor(j) == 0) for(int f=1;f< 2*GsiteSize * SurfSize;f+=2) surf0[f]*=-1;

      if(flag[j])
      {
        getPlusData(surfx, surf0, 2 * GsiteSize * SurfSize, j);

	if(GJP.Bc(j) == BND_CND_GPARITY && GJP.NodeCoor(j) == 0) for(int f=1;f< 2*GsiteSize * SurfSize;f+=2) surfx[f]*=-1;

        getPlusData(surf1, surfx, 2 * GsiteSize * SurfSize, j);
      }
      else
        getPlusData(surf1, surf0, 2 * GsiteSize * SurfSize, j);

      // ------------------------------------------------------------------
      // Dissemble the received data to the allocated memory 
      // From S: 0 < x[i, j] < Slab
      //         0 < x[k != i,j] < l_node_sites[k]
      // To   G: 0 < y[i] < Slab
      //         l_node_sites[j] + Slab < y[j] < l_node_sites[j] + 2 * Slab
      //         Slab < y[k != i, j] < l_node_sites[k] + Slab
      // Also:
      // From S: 0 < x[i, j] < Slab
      //         0 < x[k != i,j] < l_node_sites[k]
      // To   G: l_node_sites[i] + Slab < y[i] < l_node_sites[i] + 2 * Slab
      //         l_node_sites[j] + Slab < y[j] < l_node_sites[j] + 2 * Slab
      //         Slab < y[k != i, j] < l_node_sites[k] + Slab
      // ------------------------------------------------------------------
      surf2 = surf1 + GsiteSize * SurfSize;
      ysta[i] = 0;
      ysta[j] = l_node_sites[j] + Slab;
      x[0] = xsta[0];
      y[0] = ysta[0];
      for(x[1] = xsta[1], y[1] = ysta[1]; x[1] < xend[1]; ++x[1], ++y[1])
        for(x[2] = xsta[2], y[2] = ysta[2]; x[2] < xend[2]; ++x[2], ++y[2])
          for(x[3] = xsta[3], y[3] = ysta[3]; x[3] < xend[3]; ++x[3], ++y[3])
          {
            l_offset = GsiteOffset(surf1, x, s_dir_offset);
            g_offset = GsiteOffset(gfield, y, g_dir_offset);
            memcpy(g_offset, l_offset, s_dir_offset[1] * sizeof(Float));
            y[i] += l_node_sites[i] + Slab;
            l_offset = GsiteOffset(surf2, x, s_dir_offset);
            g_offset = GsiteOffset(gfield, y, g_dir_offset);
            memcpy(g_offset, l_offset, s_dir_offset[1] * sizeof(Float));
            y[i] -= l_node_sites[i] + Slab;
          }
      ysta[i] = Slab;
      ysta[j] = Slab;

      // ------------------------------------------------------------------
      // If l_node_site[j] == Slab,  We only need to pass the same slab
      // to the opposite direcction
      // Otherwise we need to assemble the appropriate slab
      // Assemble the data to propagate in the positive "j" direction
      // From G: 0 < y[i] < Slab
      //         Slab < y[j] <  2 * Slab
      //         l_node_sites[j] < y[j] < l_node_sites[j] + Slab
      //         Slab < y[k != i, j] < l_node_sites[k] + Slab
      // To   S: 0 < x[i, j] < Slab
      //         0 < x[k != i,j] < l_node_sites[k]
      // Also:
      // From G: l_node_sites[i] + Slab < y[i] < l_node_sites[i] + 2 * Slab
      //         l_node_sites[j] < y[j] < l_node_sites[j] + Slab
      //         Slab < y[k != i, j] < l_node_sites[k] + Slab
      // To   S: 0 < x[i, j] < Slab
      //         0 < x[k != i,j] < l_node_sites[k]
      // ------------------------------------------------------------------
      if(l_node_sites[j] != Slab)
      {
        surf2 = surf0 + GsiteSize * SurfSize;
        ysta[i] = 0;
        ysta[j] = l_node_sites[j];
        x[0] = xsta[0];
        y[0] = ysta[0];
        for(x[1] = xsta[1], y[1] = ysta[1]; x[1] < xend[1]; ++x[1], ++y[1])
          for(x[2] = xsta[2], y[2] = ysta[2]; x[2] < xend[2]; ++x[2], ++y[2])
            for(x[3] = xsta[3], y[3] = ysta[3]; x[3] < xend[3]; ++x[3], ++y[3])
            {
              l_offset = GsiteOffset(surf0, x, s_dir_offset);
              g_offset = GsiteOffset(gfield, y, g_dir_offset);
              memcpy(l_offset, g_offset, s_dir_offset[1] * sizeof(Float));
              y[i] += l_node_sites[i] + Slab;
              l_offset = GsiteOffset(surf2, x, s_dir_offset);
              g_offset = GsiteOffset(gfield, y, g_dir_offset);
              memcpy(l_offset, g_offset, s_dir_offset[1] * sizeof(Float));
              y[i] -= l_node_sites[i] + Slab;
            }
        ysta[i] = Slab;
        ysta[j] = Slab;
      }

      if(GJP.Bc(j) == BND_CND_GPARITY && GJP.NodeCoor(j) == GJP.Nodes(j)-1) for(int f=1;f< 2*GsiteSize * SurfSize;f+=2) surf0[f]*=-1;

      if(flag[j])
      {
        getMinusData(surfx, surf0, 2 * GsiteSize * SurfSize, j);

	if(GJP.Bc(j) == BND_CND_GPARITY && GJP.NodeCoor(j) == GJP.Nodes(j)-1) for(int f=1;f< 2*GsiteSize * SurfSize;f+=2) surfx[f]*=-1;

        getMinusData(surf1, surfx, 2 * GsiteSize * SurfSize, j);
      }
      else
        getMinusData(surf1, surf0, 2 * GsiteSize * SurfSize, j);

      // ------------------------------------------------------------------
      // Dissemble the received data to the allocated memory 
      // From S: 0 < x[i, j] < Slab
      //         0 < x[k != i,j] < l_node_sites[k]
      // To   G: 0 < y[i] < Slab
      //         0 < y[j] < Slab
      //         Slab < y[k != i, j] < l_node_sites[k] + Slab
      // Also:
      // From S: 0 < x[i, j] < Slab
      //         0 < x[k != i,j] < l_node_sites[k]
      // To   G: l_node_sites[i] + Slab < y[i] < l_node_sites[i] + 2 * Slab
      //         0 < y[j] < Slab
      //         Slab < y[k != i, j] < l_node_sites[k] + Slab
      // ------------------------------------------------------------------
      surf2 = surf1 + GsiteSize * SurfSize;
      ysta[i] = 0;
      ysta[j] = 0;
      x[0] = xsta[0];
      y[0] = ysta[0];
      for(x[1] = xsta[1], y[1] = ysta[1]; x[1] < xend[1]; ++x[1], ++y[1])
        for(x[2] = xsta[2], y[2] = ysta[2]; x[2] < xend[2]; ++x[2], ++y[2])
          for(x[3] = xsta[3], y[3] = ysta[3]; x[3] < xend[3]; ++x[3], ++y[3])
          {
            l_offset = GsiteOffset(surf1, x, s_dir_offset);
            g_offset = GsiteOffset(gfield, y, g_dir_offset);
            memcpy(g_offset, l_offset, s_dir_offset[1] * sizeof(Float));
            y[i] += l_node_sites[i] + Slab;
            l_offset = GsiteOffset(surf2, x, s_dir_offset);
            g_offset = GsiteOffset(gfield, y, g_dir_offset);
            memcpy(g_offset, l_offset, s_dir_offset[1] * sizeof(Float));
            y[i] -= l_node_sites[i] + Slab;
          }
      ysta[i] = Slab;
      ysta[j] = Slab;

      // ------------------------------------------------------------------
      // Release memory and  set xend[j] = l_node_sites[j]
      // ------------------------------------------------------------------
      sfree(cname, fname, "surf0", surf0);
      sfree(cname, fname, "surf1", surf1);
      if(flag[j])
        sfree(cname, fname, "surfx", surfx);
      xend[j] = l_node_sites[j];
    }
    // Before go on to the next direction, set xend[i] = l_node_sites[i]
    // ------------------------------------------------------------------
    xend[i] = l_node_sites[i];
  }

  yend[0] = ysta[0] + GJP.XnodeSites();
  yend[1] = ysta[1] + GJP.YnodeSites();
  yend[2] = ysta[2] + GJP.ZnodeSites();
  yend[3] = ysta[3] + GJP.TnodeSites();

  int nthread = GJP.SetNthreads();
  Float tmp_action[nthread];  for(int i=0;i<nthread;i++) tmp_action[i]=0.0;

  Float timer = -dclock();

#pragma omp parallel for 
  for(int i = 0; i < GJP.VolNodeSites(); ++i)
  {
    int j = i;
    int y[4] = {0};
    y[0] = j % GJP.XnodeSites() + ysta[0]; j /= GJP.XnodeSites();
    y[1] = j % GJP.YnodeSites() + ysta[1]; j /= GJP.YnodeSites();
    y[2] = j % GJP.ZnodeSites() + ysta[2]; j /= GJP.ZnodeSites();
    y[3] = j  + ysta[3]; 
    
    int index = 0;
    Matrix clovers[6];
    for(int mu = 0; mu < 3; ++mu)
      for(int nu = mu + 1; nu < 4; ++nu)
      {
        CloverLeaf(clovers[index], y, mu, nu, gfield, g_dir_offset);
        ZeroReal(clovers[index]);
        index ++;
      }
    
    j = omp_get_thread_num();

    tmp_action[j] += ActionDensity(clovers).real();
    if(tmp_action[j] != tmp_action[j]) {
      VRB.Result(cname, fname, "NaN!!, y = (%d, %d, %d, %d), i = %d, j = %d\n", y[0], y[1], y[2], y[3], i, j);
    }
  }

  Float action = 0;
  for(int i = 0; i < nthread; ++i) {
    action += tmp_action[i];
  }

  glb_sum(&action);

  sfree(cname, fname, "gfield", gfield);
  sfree(cname, fname, "lfield", lfield);

  Float action_density = action / GJP.VolSites();

  if(result!=NULL) *result = action_density;

  // Print out results
  //----------------------------------------------------------------

  if(common_arg->filename != 0) {
    const char *fname = "run()";
    FILE *fp;
    if( (fp = Fopen(common_arg->filename, "a")) == NULL ) {
      ERR.FileA(cname,fname,common_arg->filename);
    }

    //Prints mean value of (1/2)tr(F_mu_nu F_mu_nu), in units of a^4
    Fprintf(fp, "%15e\n", action_density);
      
    Fclose(fp);
  }
#endif
}
#endif


CPS_END_NAMESPACE

