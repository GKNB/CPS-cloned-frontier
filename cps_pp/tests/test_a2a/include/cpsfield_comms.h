#pragma once

CPS_START_NAMESPACE

void testCyclicPermute(){
  NullObject null_obj;
  {//4D
    CPSfield<cps::ComplexD,1,FourDpolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> from(null_obj);
    CPSfield<cps::ComplexD,1,FourDpolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> tmp1(null_obj);
    CPSfield<cps::ComplexD,1,FourDpolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> tmp2(null_obj);

    from.testRandom();

    for(int dir=0;dir<4;dir++){
      int incrs[3] = { GJP.NodeSites(dir)/2, GJP.NodeSites(dir), GJP.NodeSites(dir) + GJP.NodeSites(dir)/2  };
      for(int i=0;i<3;i++){
	int incr = incrs[i];
	for(int pm=-1;pm<=1;pm+=2){
	  if(!UniqueID()) printf("Testing 4D permute in direction %c%d with increment %d\n",pm == 1 ? '+' : '-',dir,incr);
	  //permute in incr until we cycle all the way around
	  tmp1 = from;
	  CPSfield<cps::ComplexD,1,FourDpolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> *send = &tmp1;
	  CPSfield<cps::ComplexD,1,FourDpolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> *recv = &tmp2;

	  int shifted = 0;
	  printRow(from,dir,"Initial line      ");

	  int total = GJP.Nodes(dir)*GJP.NodeSites(dir);
	  int perm = 0;
	  while(shifted < total){
	    int amt = std::min(incr, total-shifted);

	    cyclicPermute(*recv,*send,dir,pm,amt);
	    shifted += amt;
	    std::ostringstream comment; comment << "After perm " << perm++ << " by incr " << amt;
	    printRow(*recv,dir,comment.str());
	    
	    if(shifted < total)
	      std::swap(send,recv);
	  }
	  printRow(*recv,dir,"Final line      ");
	  
	  CPSautoView(from_v,from,HostRead);
	  CPSautoView(recv_v,(*recv),HostRead);
	  
	  int coor[4];
	  for(coor[0]=0;coor[0]<GJP.XnodeSites();coor[0]++){
	    for(coor[1]=0;coor[1]<GJP.YnodeSites();coor[1]++){
	      for(coor[2]=0;coor[2]<GJP.ZnodeSites();coor[2]++){
		for(coor[3]=0;coor[3]<GJP.TnodeSites();coor[3]++){
		  cps::ComplexD const* orig = from_v.site_ptr(coor);
		  cps::ComplexD const* permd = recv_v.site_ptr(coor);
		  if(orig->real() != permd->real() || orig->imag() != permd->imag()){
		    printf("Error node coor (%d,%d,%d,%d) (%d,%d,%d,%d) : (%g,%g) vs (%g,%g) diff (%g,%g)\n",GJP.XnodeCoor(),GJP.YnodeCoor(),GJP.ZnodeCoor(),GJP.TnodeCoor(),coor[0],coor[1],coor[2],coor[3],orig->real(),orig->imag(),permd->real(),permd->imag(), orig->real()-permd->real(),orig->imag()-permd->imag());
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }//End 4D

  {//3D
    CPSfield<cps::ComplexD,1,SpatialPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> from(null_obj);
    CPSfield<cps::ComplexD,1,SpatialPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> tmp1(null_obj);
    CPSfield<cps::ComplexD,1,SpatialPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> tmp2(null_obj);

    from.testRandom();

    for(int dir=0;dir<3;dir++){
      for(int pm=-1;pm<=1;pm+=2){
	if(!UniqueID()) printf("Testing 3D permute in direction %c%d\n",pm == 1 ? '+' : '-',dir);
	//permute in incr until we cycle all the way around
	tmp1 = from;
	CPSfield<cps::ComplexD,1,SpatialPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> *send = &tmp1;
	CPSfield<cps::ComplexD,1,SpatialPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> *recv = &tmp2;

	int shifted = 0;
	int total = GJP.Nodes(dir)*GJP.NodeSites(dir);
	int incr = GJP.NodeSites(dir)/2;
	int perm = 0;
	while(shifted < total){
	  cyclicPermute(*recv,*send,dir,pm,incr);
	  shifted += incr;
	  if(shifted < total)
	    std::swap(send,recv);
	}
      
	CPSautoView(from_v,from,HostRead);
	CPSautoView(recv_v,(*recv),HostRead);

	int coor[3];
	for(coor[0]=0;coor[0]<GJP.XnodeSites();coor[0]++){
	  for(coor[1]=0;coor[1]<GJP.YnodeSites();coor[1]++){
	    for(coor[2]=0;coor[2]<GJP.ZnodeSites();coor[2]++){
	      cps::ComplexD const* orig = from_v.site_ptr(coor);
	      cps::ComplexD const* permd = recv_v.site_ptr(coor);
	      if(orig->real() != permd->real() || orig->imag() != permd->imag()){
		printf("Error node coor (%d,%d,%d,%d) (%d,%d,%d) : (%g,%g) vs (%g,%g) diff (%g,%g)\n",GJP.XnodeCoor(),GJP.YnodeCoor(),GJP.ZnodeCoor(),GJP.TnodeCoor(),coor[0],coor[1],coor[2],orig->real(),orig->imag(),permd->real(),permd->imag(), orig->real()-permd->real(),orig->imag()-permd->imag());
	      }
	    }
	  }
	}	
      }
    }
  }//End 3D

#ifdef USE_GRID

  {//4D
    typedef FourDSIMDPolicy<FixedFlavorPolicy<1> >::ParamType simd_params;
    simd_params sp;
    FourDSIMDPolicy<FixedFlavorPolicy<1> >::SIMDdefaultLayout(sp, Grid::vComplexD::Nsimd() );
  
    CPSfield<Grid::vComplexD,1,FourDSIMDPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> from_grid(sp);
    CPSfield<Grid::vComplexD,1,FourDSIMDPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> tmp1_grid(sp);
    CPSfield<Grid::vComplexD,1,FourDSIMDPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> tmp2_grid(sp);

    CPSfield<cps::ComplexD,1,FourDpolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> from(null_obj);
    CPSfield<cps::ComplexD,1,FourDpolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> tmp1(null_obj);
    from.testRandom();
    from_grid.importField(from);

    for(int dir=0;dir<4;dir++){
      for(int pm=-1;pm<=1;pm+=2){
	if(!UniqueID()) printf("Testing 4D permute in direction %c%d with SIMD layout\n",pm == 1 ? '+' : '-',dir);
	//permute in incr until we cycle all the way around
	tmp1_grid = from_grid;
	CPSfield<Grid::vComplexD,1,FourDSIMDPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> *send = &tmp1_grid;
	CPSfield<Grid::vComplexD,1,FourDSIMDPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> *recv = &tmp2_grid;

	int shifted = 0;
	printRow(from_grid,dir,"Initial line      ");

	int total = GJP.Nodes(dir)*GJP.NodeSites(dir);
	int incr = GJP.NodeSites(dir)/2;
	int perm = 0;
	while(shifted < total){
	  cyclicPermute(*recv,*send,dir,pm,incr);
	  shifted += incr;
	  std::ostringstream comment; comment << "After perm " << perm++ << " by incr " << incr;
	  printRow(*recv,dir,comment.str());
	        
	  if(shifted < total)
	    std::swap(send,recv);
	}
	printRow(*recv,dir,"Final line      ");

	tmp1.importField(*recv);

	CPSautoView(from_v,from,HostRead);
	CPSautoView(tmp1_v,tmp1,HostRead);
      
	int coor[4];
	for(coor[0]=0;coor[0]<GJP.XnodeSites();coor[0]++){
	  for(coor[1]=0;coor[1]<GJP.YnodeSites();coor[1]++){
	    for(coor[2]=0;coor[2]<GJP.ZnodeSites();coor[2]++){
	      for(coor[3]=0;coor[3]<GJP.TnodeSites();coor[3]++){
		cps::ComplexD const* orig = from_v.site_ptr(coor);
		cps::ComplexD const* permd = tmp1_v.site_ptr(coor);
		if(orig->real() != permd->real() || orig->imag() != permd->imag()){
		  printf("Error node coor (%d,%d,%d,%d) (%d,%d,%d,%d) : (%g,%g) vs (%g,%g) diff (%g,%g)\n",GJP.XnodeCoor(),GJP.YnodeCoor(),GJP.ZnodeCoor(),GJP.TnodeCoor(),coor[0],coor[1],coor[2],coor[3],orig->real(),orig->imag(),permd->real(),permd->imag(), orig->real()-permd->real(),orig->imag()-permd->imag());
		}
	      }
	    }
	  }
	}
      }
    }
  }

  {//3D
    typedef ThreeDSIMDPolicy<FixedFlavorPolicy<1> >::ParamType simd_params;
    simd_params sp;
    ThreeDSIMDPolicy<FixedFlavorPolicy<1> >::SIMDdefaultLayout(sp, Grid::vComplexD::Nsimd() );
  
    CPSfield<Grid::vComplexD,1,ThreeDSIMDPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> from_grid(sp);
    CPSfield<Grid::vComplexD,1,ThreeDSIMDPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> tmp1_grid(sp);
    CPSfield<Grid::vComplexD,1,ThreeDSIMDPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> tmp2_grid(sp);

    CPSfield<cps::ComplexD,1,SpatialPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> from(null_obj);
    CPSfield<cps::ComplexD,1,SpatialPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> tmp1(null_obj);
    from.testRandom();
    from_grid.importField(from);

    for(int dir=0;dir<3;dir++){
      for(int pm=-1;pm<=1;pm+=2){
	if(!UniqueID()) printf("Testing 3D permute in direction %c%d with SIMD layout\n",pm == 1 ? '+' : '-',dir);
	//permute in incr until we cycle all the way around
	tmp1_grid = from_grid;
	CPSfield<Grid::vComplexD,1,ThreeDSIMDPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> *send = &tmp1_grid;
	CPSfield<Grid::vComplexD,1,ThreeDSIMDPolicy<FixedFlavorPolicy<1> >,UVMallocPolicy> *recv = &tmp2_grid;

	int shifted = 0;
	int total = GJP.Nodes(dir)*GJP.NodeSites(dir);
	int incr = GJP.NodeSites(dir)/2;
	int perm = 0;
	while(shifted < total){
	  cyclicPermute(*recv,*send,dir,pm,incr);
	  shifted += incr;
	  if(shifted < total)
	    std::swap(send,recv);
	}
	tmp1.importField(*recv);
	
	CPSautoView(from_v,from,HostRead);
	CPSautoView(tmp1_v,tmp1,HostRead);

	int coor[3];
	for(coor[0]=0;coor[0]<GJP.XnodeSites();coor[0]++){
	  for(coor[1]=0;coor[1]<GJP.YnodeSites();coor[1]++){
	    for(coor[2]=0;coor[2]<GJP.ZnodeSites();coor[2]++){
	      cps::ComplexD const* orig = from_v.site_ptr(coor);
	      cps::ComplexD const* permd = tmp1_v.site_ptr(coor);
	      if(orig->real() != permd->real() || orig->imag() != permd->imag()){
		printf("Error node coor (%d,%d,%d,%d) (%d,%d,%d) : (%g,%g) vs (%g,%g) diff (%g,%g)\n",GJP.XnodeCoor(),GJP.YnodeCoor(),GJP.ZnodeCoor(),GJP.TnodeCoor(),coor[0],coor[1],coor[2],orig->real(),orig->imag(),permd->real(),permd->imag(), orig->real()-permd->real(),orig->imag()-permd->imag());
	      }
	    }
	  }
	}
      }

    }
  }
#endif

  if(!UniqueID()){ printf("Passed permute test\n"); fflush(stdout); }
} 

CPS_END_NAMESPACE
