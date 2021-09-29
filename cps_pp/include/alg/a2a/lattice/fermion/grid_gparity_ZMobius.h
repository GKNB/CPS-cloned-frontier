#ifndef _CPS_GRID_GPARITY_MOBIUS_H
#define _CPS_GRID_GPARITY_MOBIUS_H

#ifdef USE_GRID
#include<Grid.h>

//Grid currently does not implement ZMobius for G-parity
//Do do so we have to mirror Grid's fermion class instantiations
//(I wish this were easier)

NAMESPACE_BEGIN(Grid);

typedef GparityWilsonImpl<vComplexF, FundamentalRepresentation,CoeffComplex> ZGparityWilsonImplF;  // Float
typedef GparityWilsonImpl<vComplexD, FundamentalRepresentation,CoeffComplex> ZGparityWilsonImplD;  // Double
 
//typedef GparityWilsonImpl<vComplexF, FundamentalRepresentation,CoeffComplexHalfComms> ZGparityWilsonImplFH;  // Float
//typedef GparityWilsonImpl<vComplexD, FundamentalRepresentation,CoeffComplexHalfComms> ZGparityWilsonImplDF;  // Double

typedef ZMobiusFermion<ZGparityWilsonImplF> ZGparityMobiusFermionF;
typedef ZMobiusFermion<ZGparityWilsonImplD> ZGparityMobiusFermionD;                                                                                                                                 
//typedef ZMobiusFermion<ZGparityWilsonImplFH> ZGparityMobiusFermionFH;
//typedef ZMobiusFermion<ZGparityWilsonImplDF> ZGparityMobiusFermionDF;

NAMESPACE_END(Grid);
#endif

#endif
