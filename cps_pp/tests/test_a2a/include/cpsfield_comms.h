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


void testCshiftCconjBc(){
  std::cout << "Starting testCshiftCconjBc" << std::endl;
  size_t glb_size[4];
  for(int i=0;i<4;i++) glb_size[i] = GJP.Nodes(i)*GJP.NodeSites(i);
  int nf = GJP.Gparity()+1;

  NullObject null_obj;
  CPSfield<cps::ComplexD,9,FourDpolicy<DynamicFlavorPolicy> > field(null_obj);
  {
    CPSautoView(field_v,field,HostWrite);
#pragma omp parallel for
    for(size_t s=0;s<GJP.VolNodeSites();s++){
      int x[4];
      field_v.siteUnmap(s,x);
      for(int i=0;i<4;i++)
	x[i] += GJP.NodeCoor(i)*GJP.NodeSites(i);
      size_t glb_s = x[0] + glb_size[0]*(x[1] + glb_size[1]*(x[2] + glb_size[2]*x[3]));
      
      for(int f=0;f<nf;f++){
	cps::ComplexD * p = field_v.site_ptr(s,f);
	size_t pv = f + nf*glb_s;
	for(int i=0;i<9;i++){
	  size_t ipv = i + 9*pv;
	  p[i] = cps::ComplexD(ipv,ipv);
	}
      }
    }
  }

  {
    //Test periodic
    std::vector<int> cconj_dirs(4,0);
    bool fail = false;
    for(int mu=0;mu<4;mu++){
      for(int pmi=0;pmi<2;pmi++){
	int pm = pmi == 0 ? -1 : +1; //direction of data movement     
	CPSfield<cps::ComplexD,9,FourDpolicy<DynamicFlavorPolicy> > result = CshiftCconjBc(field,mu,pm,cconj_dirs);
	{
	  CPSautoView(result_v,result,HostRead);
	  for(size_t s=0;s<GJP.VolNodeSites();s++){
	    int x[4];
	    result_v.siteUnmap(s,x);
	    for(int i=0;i<4;i++)
	      x[i] += GJP.NodeCoor(i)*GJP.NodeSites(i);

	    if(pm == +1){ //from site x[mu]-1
	      x[mu] = (x[mu] - 1 + glb_size[mu]) % glb_size[mu];
	    }else{ //from site x[mu] + 1
	      x[mu] = (x[mu] + 1) % glb_size[mu];
	    }
	    size_t glb_s = x[0] + glb_size[0]*(x[1] + glb_size[1]*(x[2] + glb_size[2]*x[3]));
      
	    for(int f=0;f<nf;f++){
	      cps::ComplexD const* p = result_v.site_ptr(s,f);
	      size_t pv = f + nf*glb_s;
	      for(int i=0;i<9;i++){
		size_t ipv = i + 9*pv;
		cps::ComplexD expect(ipv,ipv);
		cps::ComplexD got = p[i];
		if(got.real() != expect.real() || got.imag() != expect.imag()){
		  std::cout << "FAIL " << mu << " " << pm << " " << s << std::endl;
		  fail = true;
		}
	      }
	    }
	  }
	}
      }
    }
    if(fail) ERR.General("","testCshiftCconjBc","periodic test failed");
  }


  {
    //Test cconj
    std::vector<int> cconj_dirs(4,1);
    bool fail = false;
    for(int mu=0;mu<4;mu++){
      for(int pmi=0;pmi<2;pmi++){
	int pm = pmi == 0 ? -1 : +1; //direction of data movement     
	CPSfield<cps::ComplexD,9,FourDpolicy<DynamicFlavorPolicy> > result = CshiftCconjBc(field,mu,pm,cconj_dirs);
	{
	  CPSautoView(result_v,result,HostRead);
	  for(size_t s=0;s<GJP.VolNodeSites();s++){
	    int x[4];
	    result_v.siteUnmap(s,x);
	    for(int i=0;i<4;i++)
	      x[i] += GJP.NodeCoor(i)*GJP.NodeSites(i);
	    
	    //Is this a boundary site in mu?
	    bool bnd = false;
	    if( (pm == +1 && x[mu] == 0) ||
		(pm == -1 && x[mu] == glb_size[mu]-1) )
	      bnd = true;

	    if(pm == +1){ //from site x[mu]-1
	      x[mu] = (x[mu] - 1 + glb_size[mu]) % glb_size[mu];
	    }else{ //from site x[mu] + 1
	      x[mu] = (x[mu] + 1) % glb_size[mu];
	    }
	    size_t glb_s = x[0] + glb_size[0]*(x[1] + glb_size[1]*(x[2] + glb_size[2]*x[3]));
      
	    for(int f=0;f<nf;f++){
	      cps::ComplexD const* p = result_v.site_ptr(s,f);
	      size_t pv = f + nf*glb_s;
	      for(int i=0;i<9;i++){
		size_t ipv = i + 9*pv;
		cps::ComplexD expect(ipv, (bnd ? -double(ipv) : ipv) );
		cps::ComplexD got = p[i];
		if(got.real() != expect.real() || got.imag() != expect.imag()){
		  std::cout << "FAIL " << mu << " " << pm << " " << s << " " << f << " " << i << " got " << got << " expect " << expect << std::endl;
		  fail = true;
		}
	      }
	    }
	  }
	}
      }
    }
    if(fail) ERR.General("","testCshiftCconjBc","cconj test failed");
  }


  std::cout << "testCshiftCconjBc passed" << std::endl;
}


void testCshiftCconjBcMatrix(const SIMDdims<4> &simd_dims){
  std::cout << "Starting testCshiftCconjBcMatrix" << std::endl;
  size_t glb_size[4];
  for(int i=0;i<4;i++) glb_size[i] = GJP.Nodes(i)*GJP.NodeSites(i);
  int nf = GJP.Gparity()+1;

  NullObject null_obj;  
  CPSfield<cps::ComplexD,9,FourDpolicy<OneFlavorPolicy> > field_s(null_obj);
  {
    CPSautoView(field_s_v,field_s,HostWrite);
#pragma omp parallel for
    for(size_t s=0;s<GJP.VolNodeSites();s++){
      int x[4];
      field_s_v.siteUnmap(s,x);
      for(int i=0;i<4;i++)
	x[i] += GJP.NodeCoor(i)*GJP.NodeSites(i);
      size_t glb_s = x[0] + glb_size[0]*(x[1] + glb_size[1]*(x[2] + glb_size[2]*x[3]));
      
      cps::ComplexD * p = (cps::ComplexD*)field_s_v.site_ptr(s);
      size_t pv = glb_s;
      for(int i=0;i<9;i++){
	size_t ipv = i + 9*pv;
	p[i] = cps::ComplexD(ipv,ipv);
      }
    }
  }
  CPSfield<Grid::vComplexD,9,FourDSIMDPolicy<OneFlavorPolicy> > field_v(simd_dims);
  field_v.importField(field_s);
  CPSmatrixField<CPScolorMatrix<Grid::vComplexD> > field = linearRepack<CPScolorMatrix<Grid::vComplexD> >(field_v);
  
  CPSfield<Grid::vComplexD,9,FourDSIMDPolicy<OneFlavorPolicy> > result_v(simd_dims);
  CPSfield<cps::ComplexD,9,FourDpolicy<OneFlavorPolicy> > result_s(null_obj);

  std::vector<int> cconj_dirs(4,1);
  bool fail = false;
  for(int mu=0;mu<4;mu++){
    for(int pmi=0;pmi<2;pmi++){
      int pm = pmi == 0 ? -1 : +1; //direction of data movement     
      auto result = CshiftCconjBc(field,mu,pm,cconj_dirs);
      result_v = linearUnpack(result);
      result_s.importField(result_v);

      {
	CPSautoView(result_s_v,result_s,HostRead);
	for(size_t s=0;s<GJP.VolNodeSites();s++){
	  int x[4];
	  result_v.siteUnmap(s,x);
	  for(int i=0;i<4;i++)
	    x[i] += GJP.NodeCoor(i)*GJP.NodeSites(i);
	    
	  //Is this a boundary site in mu?
	  bool bnd = false;
	  if( (pm == +1 && x[mu] == 0) ||
	      (pm == -1 && x[mu] == glb_size[mu]-1) )
	    bnd = true;

	  if(pm == +1){ //from site x[mu]-1
	    x[mu] = (x[mu] - 1 + glb_size[mu]) % glb_size[mu];
	  }else{ //from site x[mu] + 1
	    x[mu] = (x[mu] + 1) % glb_size[mu];
	  }
	  size_t glb_s = x[0] + glb_size[0]*(x[1] + glb_size[1]*(x[2] + glb_size[2]*x[3]));
      
	    
	  cps::ComplexD const* p = (cps::ComplexD*)result_s_v.site_ptr(s);
	  size_t pv = glb_s;
	  for(int i=0;i<9;i++){
	    size_t ipv = i + 9*pv;
	    cps::ComplexD expect(ipv, (bnd ? -double(ipv) : ipv) );
	    cps::ComplexD got = p[i];
	    if(got.real() != expect.real() || got.imag() != expect.imag()){
	      std::cout << "FAIL " << mu << " " << pm << " " << s << " " << i << " got " << got << " expect " << expect << std::endl;
	      fail = true;
	    }
	  }
	    
	}
      }
    }
  }
  if(fail) ERR.General("","testCshiftCconjBcMatrix","cconj test failed");

  std::cout << "testCshiftCconjBcMatrix passed" << std::endl;
}



CPS_END_NAMESPACE
