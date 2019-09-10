#include <util/gauge_field.h>

CPS_START_NAMESPACE

void gfUnitMatrix(GaugeField& gf) {
  Matrix * u = gf.getGaugeField();
  for (int i = 0; i < gf.siteSize() * 4; i++) {
    u[i].UnitMatrix();
  }
}

void gfUnitarize(GaugeField& gf) {
  const char *fname = "gfReconstruct3rdRow()";
  Matrix * u = gf.getGaugeField();
  for (int i = 0; i < gf.siteSize() * 4; i++) {
    u[i].Unitarize();
  }
}

void gfConstruct3rdRow(GaugeField& gf) {
  const char *fname = "gfConstruct3rdRow()";
  Matrix * u = gf.getGaugeField();
  for (int i = 0; i < gf.siteSize() * 4; i++) {
    u[i].Construct3rdRow();
  }
}

void gfDagger(GaugeField& gf) {
  const char *fname = "gfDagger()";
  Matrix * u = gf.getGaugeField();
  for (int i = 0; i < gf.siteSize() * 4; i++) {
    Matrix tmp = u[i];
    u[i].Dagger(tmp);
  }
}

void gfPathOrdProdPlus(Matrix & mat, GaugeField& gf, const int * x, const int* dirs, int n) {
  //char * fname = "PathOrdProd";
  //VRB.Flow(cname, fname,"(,,,%d)\n",n);

  Matrix result, result1, mat1;
  Matrix * result_mp = &result;
  Matrix * result1_mp = &result1;

  int abs_dir;
  int dir_sign;
  int link_site[4];

  const Matrix * p1;
  //Matrix m1, m2, m3;

  int i;
  for(i=0;i<4;i++)link_site[i]=x[i];

  //deal with the first link
  //------------------------------
  abs_dir = dirs[0]&3;
  //abs_dir is {0,1,2,3}
  dir_sign = dirs[0]>>2;
  //dir_sign is {0, 1}

  link_site[abs_dir] -= dir_sign;
  //if dir_sign == 1, the link is at x-n_v

  p1 = gf.unsafeGetLink(link_site, abs_dir);
  //get the first link

  link_site[abs_dir] += 1-dir_sign;
  //if dir_sign == 0, march on to the next site, if dir_sign == 1, we have
  //already moved.

  if (dir_sign) {
    result1_mp->Dagger((IFloat*)p1);
    //if dir_sign==1 the link is going backward so get its dagger
  } else {
    moveMem((IFloat*) result1_mp, (IFloat*)p1, sizeof(Matrix));
    //simply move to the cram
  }

  for(i=1;i<n;i++){
    abs_dir = dirs[i]&3;
    dir_sign = dirs[i]>>2;

    link_site[abs_dir] -= dir_sign;

    p1 = gf.unsafeGetLink(link_site, abs_dir);

    link_site[abs_dir] += 1-dir_sign;

    //put the next link on the path in mat1
    //--------------------------------------
    if (dir_sign) {
      mat1.Dagger((IFloat*)p1);
    } else {
      moveMem((IFloat*)&mat1, (IFloat*)p1, sizeof(Matrix));
    }

    if(i!=n-1) {
      mDotMEqual((IFloat*) result_mp, (IFloat*)result1_mp, (IFloat*)&mat1);
      //if not the last link on the path, just multiply to the earlier result
    } else {
      mDotMPlus((IFloat*) &mat, (IFloat*) result1_mp, (IFloat*) &mat1);
      //if the last link, multiply and add to mat.
    }

    Matrix * tmp_p = result1_mp; result1_mp=result_mp; result_mp = tmp_p;
    //swap result_mp and result1_mp;
  }
}

Float gfReTrPlaq(GaugeField &gf, const int x[4], int mu, int nu)
{
  int dirs[] = {mu, nu, mu+4, nu+4};    
  Matrix plaq;
  gfPathOrdProdPlus(plaq, gf, x, dirs, 4);
  return plaq.ReTr();
}

Float gfReTrRect(GaugeField &gf, const int x[4], int mu, int nu)
{
  int dirs[] = {mu, mu, nu, mu+4, mu+4, nu+4};    
  Matrix rect;
  gfPathOrdProdPlus(rect, gf, x, dirs, 6);
  return rect.ReTr();
}

Float gfSumReTrPlaqNode(GaugeField &gf) 
{
  Float ret = 0;
  int x[4];
#pragma omp parallel for reduction(+:ret) //FIXME: make deterministic
  for(int index = 0; index < gf.localVolume(); index++) {
    gf.coordinatesFromIndex(x, index);
    for(int mu = 0; mu < 3; mu++) {
      for(int nu = mu+1; nu < 4; nu++) {
        ret += gfReTrPlaq(gf, x, mu, nu);
      }
    }
  }
  return ret;
}

Float gfSumReTrRectNode(GaugeField &gf) 
{
  Float ret = 0;
  int x[4];
#pragma omp parallel for reduction(+:ret) //FIXME: make deterministic
  for(int index = 0; index < gf.localVolume(); index++) {
    gf.coordinatesFromIndex(x, index);
    for(int mu = 0; mu < 4; mu++) {
      for(int nu = 0; nu < 4; nu++) {
        if(mu != nu) {
          ret += gfReTrRect(gf, x, mu, nu);
        }
      }
    }
  }
  return ret;
}

//------------------------------------------------------------------
// GaugeField::Staple()
/*!
  The staple sum around the link \f$ U_\mu(x) \f$is

\f[
  \sum_{v \neq \mu} [
              U_\nu(x+\mu) U_\mu(x+\nu) U^\dagger_\nu(x)
           +  U^\dagger_\nu(x+\mu-\nu) U^\dagger_u(x-\nu) U_\nu(x-\nu) ]
\f]

  \param x The coordinates of the lattice site
  \param u The link direction
  \param stap The computed staple sum.
*/
//------------------------------------------------------------------
void gfPlaqStaple(Matrix& stap, GaugeField& gf, const int *x, int u) {
  //char * fname = "BufferedStaple()";
  //VRB.Func(cname, fname);
  int link_site[4];

  Matrix accumulate;
  accumulate.ZeroMatrix();

  for(int i=0;i<4;i++)link_site[i]=x[i];

  link_site[u]++;

  int dir[3];
  stap.ZeroMatrix();

  int u1= u+4;
  for(int v=0; v<8; v++){
    if((v&3)==u) continue;
    dir[0]=v;     //v0
    dir[1]=u1;
    dir[2]=(v+4)&7; //v1
    gfPathOrdProdPlus(accumulate, gf, link_site, dir, 3);
    //vecAddEquVec((IFloat*) &stap, (IFloat*) &m_tmp,
    //		   MATRIX_SIZE*sizeof(IFloat));
  }
  moveMem((IFloat*)&stap,  (IFloat*) &accumulate, sizeof(Matrix));
}

//------------------------------------------------------------------
// GaugeField::RectStaple()
/*! The 5-link rectangle staple sum around the link U_\mu(x) is:

\f[
 \sum_{\nu \neq \mu} \left[\right.
 U_\mu(x+\mu)  U_\nu(x+2\mu)  U^\dagger_\mu(x+\mu+\nu)
 U^\dagger_\mu(x+\nu)  U^\dagger_\nu(x)       \f]\f[
 + U_\mu(x+\mu)  U^\dagger_\nu(x+2\mu-\nu) U^\dagger_\mu(x+\mu-\nu)
 U^\dagger_\mu(x-\nu)  U_\nu(x-\nu)  \f]\f[
 + U_\nu(x+\mu)  U^\dagger_\mu(x+\nu)  U^\dagger_\mu(x-\mu+\nu)
 U^\dagger_\nu(x-\mu) U_\mu(x-\mu)      \f]\f[
 + U^\dagger_\nu(x+\mu-\nu) U^\dagger_\mu(x-\nu)  U^\dagger_\mu(x-\mu-\nu)
 U_\nu(x-\mu-\nu) U_\mu(x-\mu)  \f]\f[
 + U_\nu(x+\mu)  U_\nu(x+\mu+\nu)  U^\dagger_\mu(x+2\nu)
 U^\dagger_\nu(x+\nu)  U^\dagger_\nu(x) \f]\f[
 + U^\dagger_\nu(x+\mu-\nu) U^\dagger_\nu(x+\mu-2\nu) U^\dagger_\mu(x-2\nu)
 U_\nu(x-2\nu)  U_\nu(x-\nu)
 \left.\right]
\f]

  \param x The coordinates of the lattice site
  \param u The link direction
  \param stap The computed staple sum.
*/
//------------------------------------------------------------------
void gfRectStaple(Matrix& stap, GaugeField& gf, const int *x, int u) {
  //char * fname = "BufferedRectStaple()";
  //VRB.Func(cname, fname);

  int link_site[4];

  Matrix accumulate;
  accumulate.ZeroMatrix();

  for(int i=0;i<4;i++)link_site[i]=x[i];

  link_site[u]++;

  int dir[5];
  stap.ZeroMatrix();
  int u1= u+4;
  for(int v=0; v<8; v++){
    if((v&3)==u) continue;

    int v1= (v+4)&7;

    dir[0] = v;
    dir[1] = v;
    dir[2] = u1;
    dir[3] = v1;
    dir[4] = v1;

    gfPathOrdProdPlus(accumulate, gf, link_site, dir, 5);
    //vecAddEquVec((IFloat*) &stap, (IFloat*) &m_tmp,
    //		   MATRIX_SIZE*sizeof(IFloat));

    //dir[0] = v
    dir[1] = u1;
    //dir[2] = u1;
    //dir[3] = v1;
    dir[4] = u;

    gfPathOrdProdPlus(accumulate, gf, link_site, dir, 5);
    //vecAddEquVec((IFloat*) &stap, (IFloat*) &m_tmp,
    //		   MATRIX_SIZE*sizeof(IFloat));

    dir[0] = u;
    dir[1] = v;
    //dir[2] = u1;
    dir[3] = u1;
    dir[4] = v1;
    gfPathOrdProdPlus(accumulate, gf, link_site, dir, 5);
    //vecAddEquVec((IFloat*) &stap, (IFloat*) &m_tmp,
    //		   MATRIX_SIZE*sizeof(IFloat));

  }
  moveMem((IFloat*)&stap, (IFloat*)&accumulate, sizeof(Matrix));
}


//------------------------------------------------------------------------------
// GforceSite(Matrix& force, int *x, int mu):
// It calculates the gauge force at site x and direction mu.
//------------------------------------------------------------------------------
void gfForceSite(Matrix& force, GaugeField& gf, GaugeAction& ga, int *x, int mu) {
  const char *fname = "gfForceSite(M&,GF&,i*,i)";

//  Matrix *u_off = GaugeField()+GsiteOffset(x)+mu;

  Matrix mt1;
  gfAllStaple(mt1, gf, ga, x, mu);
  force.DotMEqual(*gf.unsafeGetLink(x, mu), mt1);
  force *= - ga.getBeta() / 3.0;

//  //----------------------------------------------------------------------------
//  //  get staple
//  //     mt1 = staple
//  //----------------------------------------------------------------------------
//  Staple(mt1, x, mu);
//  ForceFlops += 198*3*3+12+216*3;
//
//  //----------------------------------------------------------------------------
//  // mt2 = U_mu(x)
//  //----------------------------------------------------------------------------
//  Matrix mt2(*u_off);
//  // moveMem((IFloat *)mp2, (IFloat *)u_off, MATRIX_SIZE * sizeof(IFloat)) ;
//
//  //----------------------------------------------------------------------------
//  // force = -(beta*(1-8*c_1)/3)*U_mu(x)*stap
//  //----------------------------------------------------------------------------
//  force.DotMEqual(mt2, mt1);
//  force *= plaq_coeff;
//  // mDotMEqual((IFloat *)&force, (const IFloat *)mp2, (const IFloat *)mp1);
//  // tmp = plaq_coeff ;
//  // vecTimesEquFloat((IFloat *)&force, tmp, MATRIX_SIZE);
//
//  //----------------------------------------------------------------------------
//  //  get rectangle staple
//  //     mt1 = rect_stap
//  //----------------------------------------------------------------------------
//  RectStaple(mt1, x, mu);
//  ForceFlops += 198*3*18+216*3*6;
//
//  //----------------------------------------------------------------------------
//  // mt2 = -(beta*c_1/3)*U_mu(x)
//  //----------------------------------------------------------------------------
//  // mt2 = *u_off;
//  // moveMem((IFloat *)mp2, (IFloat *)u_off, MATRIX_SIZE*sizeof(IFloat));
//
//  mt2 *= rect_coeff;
//  // tmp = rect_coeff;
//  // vecTimesEquFloat((IFloat *)mp2, tmp, MATRIX_SIZE) ;
//  ForceFlops +=234;
//
//  //----------------------------------------------------------------------------
//  // force += -(beta*c_1/3)*U_mu(x)*rect_stap
//  //----------------------------------------------------------------------------
//  force.DotMPlus(mt2, mt1);
//  // mDotMPlus((IFloat *)&force, (const IFloat *)mp2, (const IFloat *)mp1);

  mt1.Dagger(force);
  force.TrLessAntiHermMatrix(mt1);
//  ForceFlops +=198+24;
}

ForceArg gfEvolveMomGforce(Matrix *mom, GaugeField& gf, GaugeAction& ga, Float dt){
  char *fname = "EvolveMomGforce(M*,F)";
//  VRB.Func(cname,fname);

#ifdef PROFILE
  Float time = -dclock();
  ForceFlops = 0;
#endif

  Float L1=0.0;
  Float L2=0.0;
  Float Linf=0.0;

#pragma omp parallel
  {
    Float pL1=0.0;
    Float pL2=0.0;
    Float pLinf=0.0;
#pragma omp for nowait
    for (int index = 0; index < gf.localVolume(); index++) {
      //  for(x[0] = 0; x[0] < GJP.XnodeSites(); ++x[0])
      //  for(x[1] = 0; x[1] < GJP.YnodeSites(); ++x[1])
      //  for(x[2] = 0; x[2] < GJP.ZnodeSites(); ++x[2])
      //  for(x[3] = 0; x[3] < GJP.TnodeSites(); ++x[3]) {

      for (int mu = 0; mu < 4; ++mu) {
        int x[4];
        gf.coordinatesFromIndex(x, index);
        //int uoff = GsiteOffset(x);

        Matrix mt0;
        Matrix mt1;
        Matrix mt2;
        Matrix *mp0 = &mt0;		// ihdot
        Matrix *mp1 = &mt1;
        Matrix *mp2 = &mt2;

        //#pragma omp critical
        gfForceSite(*mp0, gf, ga, x, mu);

        //IFloat *ihp = (IFloat *)(mom+uoff+mu);
        IFloat *ihp = (IFloat *)(mom+index*4+mu);
        IFloat *dotp = (IFloat *)mp0;
        fTimesV1PlusV2(ihp, dt, dotp, ihp, sizeof(Matrix) / sizeof(IFloat));
        Float norm = ((Matrix*)dotp)->norm();
        Float tmp = sqrt(norm);
        pL1 += tmp;
        pL2 += norm;
        pLinf = (tmp>pLinf ? tmp : pLinf);
      }
      //}
    }
#pragma omp critical
    {
      L1 += pL1;
      L2 += pL2;
      Linf = pLinf > Linf ? pLinf : Linf;
    }
  }

//  ForceFlops += gf.localVolume()*4*18*2;
#ifdef PROFILE
  time += dclock();
//  print_flops(cname,fname,ForceFlops,time);
  print_flops(cname,fname,0,time);
#endif

  glb_sum(&L1);
  glb_sum(&L2);
  glb_max(&Linf);

//  L1 /= 4.0*GJP.VolSites();
//  L2 /= 4.0*GJP.VolSites();
  L1 /= 4.0*gf.localVolume()*gf.numNodes();
  L2 /= 4.0*gf.localVolume()*gf.numNodes();


//  VRB.FuncEnd(cname,fname);
  return ForceArg(dt*L1, dt*sqrt(L2), dt*Linf);
}

Float gfGhamiltonNode(GaugeField& gf, GaugeAction& ga) {
  const char* fname = "gfGhamiltonNode()";

  VRB.Result("", fname, "Starting gfGhamiltonNode()\n");

#ifdef PROFILE
  Float time = -dclock();
  ForceFlops = 0;
#endif

  Float ret = 0.0;

  switch (ga.Gclass()) {
    case G_CLASS_NONE:
      break;

    case G_CLASS_WILSON:
      {
        Float beta = ga.getBeta();
        Float plaq_coeff = - beta / 3.0;
        ret += plaq_coeff * gfSumReTrPlaqNode(gf);
      }
      break;
      
    case G_CLASS_IMPR_RECT:
      {
        Float c1 = ga.getGaugeActionImprRect().getC1();
        Float beta = ga.getBeta();
        Float plaq_coeff = - beta * ( 1.0 - 8.0 * c1 ) / 3.0;
        Float rect_coeff = - beta * (             c1 ) / 3.0;
        ret += plaq_coeff * gfSumReTrPlaqNode(gf);
        ret += rect_coeff * gfSumReTrRectNode(gf);
      }
      break;

    default:
      ERR.NotImplemented("","","gfGhamiltonNode(GaugeField& gf, GaugeAction& ga)");
  }


#ifdef PROFILE
  time += dclock();
  print_flops(cname,fname,0,time);
#endif

  return ret;
}

CPS_END_NAMESPACE
