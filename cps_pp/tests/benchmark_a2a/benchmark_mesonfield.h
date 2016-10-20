#ifndef _BENCHMARK_MESONFIELD_H
#define _BENCHMARK_MESONFIELD_H

CPS_START_NAMESPACE

void randomMatrix(SpinColorFlavorMatrix &A, CPSspinColorFlavorMatrix<cps::ComplexD> &B){
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  for(int f1=0;f1<2;f1++)
	    for(int f2=0;f2<2;f2++){
	      cps::ComplexD tmp;
	      _testRandom<cps::ComplexD>::rand(&tmp,1, 3.0, -3.0);
	      A(s1,c1,f1,s2,c2,f2) = tmp;
	      B(s1,s2)(c1,c2)(f1,f2) = tmp;
	    }
}
void randomMatrix(SpinColorFlavorMatrix &A){
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  for(int f1=0;f1<2;f1++)
	    for(int f2=0;f2<2;f2++){
	      cps::ComplexD tmp;
	      _testRandom<cps::ComplexD>::rand(&tmp,1, 3.0, -3.0);
	      A(s1,c1,f1,s2,c2,f2) = tmp;
	    }
}
void randomMatrix(CPSspinColorFlavorMatrix<cps::ComplexD> &B){
  for(int s1=0;s1<4;s1++)
    for(int s2=0;s2<4;s2++)
      for(int c1=0;c1<3;c1++)
	for(int c2=0;c2<3;c2++)
	  for(int f1=0;f1<2;f1++)
	    for(int f2=0;f2<2;f2++){
	      cps::ComplexD tmp;
	      _testRandom<cps::ComplexD>::rand(&tmp,1, 3.0, -3.0);
	      B(s1,s2)(c1,c2)(f1,f2) = tmp;
	    }
}


void benchmarkTrace(const int ntests, const double tol){
  typedef CPSsquareMatrix<CPSsquareMatrix<CPSsquareMatrix<cps::ComplexD,2>,3>,4> SCFmat;
  typedef CPSsquareMatrix<cps::ComplexD,3> Cmat;
    
  //Test they give the same answer
  {
    SpinColorFlavorMatrix old_mat;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;
    randomMatrix(old_mat,new_mat);

    cps::Complex cp =  old_mat.Trace();
    cps::Complex gd = new_mat.Trace();
      
    bool fail = false;
    double rdiff = fabs(gd.real()-cp.real());
    double idiff = fabs(gd.imag()-cp.imag());
    if(rdiff > tol|| idiff > tol){
      printf("Fail: Trace Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
      fail = true;
    }
    if(fail) ERR.General("","","Trace test failed\n");
    else printf("Trace pass\n");
  }
  {
    //Benchmark it    
    SpinColorFlavorMatrix old_mat;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;
    
    //SpinFlavorTrace of SpinColorFlavorMatrix
    Float total_time_old = 0.;
    cps::ComplexD tmp_old;
    for(int iter=0;iter<ntests;iter++){
      randomMatrix(old_mat);      
      total_time_old -= dclock();
      tmp_old = old_mat.Trace();
      total_time_old += dclock();
    }
    Float total_time_new = 0.;
    cps::ComplexD tmp_new;
    for(int iter=0;iter<ntests;iter++){
      randomMatrix(new_mat);
      total_time_new -= dclock();
      tmp_new = new_mat.Trace();
      total_time_new += dclock();
    }
    printf("Trace: Avg time new code %d iters: %g secs\n",ntests,total_time_new/ntests);
    printf("Trace: Avg time old code %d iters: %g secs\n",ntests,total_time_old/ntests);
  }
}



void benchmarkSpinFlavorTrace(const int ntests, const double tol){
  typedef CPSsquareMatrix<CPSsquareMatrix<CPSsquareMatrix<cps::ComplexD,2>,3>,4> SCFmat;
  typedef CPSsquareMatrix<cps::ComplexD,3> Cmat;
    
  //Test they give the same answer
  {
    SpinColorFlavorMatrix old_mat;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;
    randomMatrix(old_mat,new_mat);

    Matrix tmp_mat_old =  old_mat.SpinFlavorTrace();
    Cmat tmp_mat_new = new_mat.SpinFlavorTrace();//TraceTwoIndices<0,2>();
      
    bool fail = false;
    for(int c1=0;c1<3;c1++)
      for(int c2=0;c2<3;c2++){
	cps::ComplexD gd = tmp_mat_old(c1,c2);
	cps::ComplexD cp = tmp_mat_new(c1,c2);
	  
	double rdiff = fabs(gd.real()-cp.real());
	double idiff = fabs(gd.imag()-cp.imag());
	if(rdiff > tol|| idiff > tol){
	  printf("Fail: SFtrace Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
	  fail = true;
	}
      }
    if(fail) ERR.General("","","SFtrace test failed\n");
    else printf("SFtrace pass\n");
  }
  {
    //Benchmark it    
    SpinColorFlavorMatrix old_mat;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;
    
    //SpinFlavorTrace of SpinColorFlavorMatrix
    Float total_time_old = 0.;
    Matrix tmp_mat_old;
    for(int iter=0;iter<ntests;iter++){
      randomMatrix(old_mat);      
      total_time_old -= dclock();
      tmp_mat_old = old_mat.SpinFlavorTrace();
      total_time_old += dclock();
    }
    Float total_time_new = 0.;
    Cmat tmp_mat_new;
    for(int iter=0;iter<ntests;iter++){
      randomMatrix(new_mat);
      total_time_new -= dclock();
      //tmp_mat_new.zero();
      //_PartialDoubleTraceImpl<Cmat,CPSspinColorFlavorMatrix<cps::ComplexD>,0,2>::doit(tmp_mat_new,new_mats[iter]);
      tmp_mat_new = new_mat.TraceTwoIndices<0,2>();
      total_time_new += dclock();
    }
    printf("SFtrace: Avg time new code %d iters: %g secs\n",ntests,total_time_new/ntests);
    printf("SFtrace: Avg time old code %d iters: %g secs\n",ntests,total_time_old/ntests);
  }
}


void benchmarkTraceProd(const int ntests, const double tol){
  typedef CPSsquareMatrix<CPSsquareMatrix<CPSsquareMatrix<cps::ComplexD,2>,3>,4> SCFmat;
  typedef CPSsquareMatrix<cps::ComplexD,3> Cmat;
    
  //Test they give the same answer
  {
    SpinColorFlavorMatrix old_mat1, old_mat2;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat1, new_mat2;
    randomMatrix(old_mat1,new_mat1);
    randomMatrix(old_mat2,new_mat2);

    cps::ComplexD tr_old = Trace(old_mat1,old_mat2);
    cps::ComplexD tr_new = Trace(new_mat1,new_mat2);
      
    bool fail = false;
    cps::ComplexD &gd = tr_new;
    cps::ComplexD &cp = tr_old;
	  
    double rdiff = fabs(gd.real()-cp.real());
    double idiff = fabs(gd.imag()-cp.imag());
    if(rdiff > tol|| idiff > tol){
      printf("Fail: Prodtrace Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
      fail = true;
    }
    if(fail) ERR.General("","","Prodtrace test failed\n");
    else printf("Prodtrace pass\n");
  }
  //Benchmark it    
  {
    SpinColorFlavorMatrix old_mat1, old_mat2;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat1, new_mat2;
    
    Float total_time_old = 0.;
    cps::ComplexD tr_old;
    for(int iter=0;iter<ntests;iter++){
      randomMatrix(old_mat1);
      randomMatrix(old_mat2);
      total_time_old -= dclock();
      tr_old = Trace(old_mat1,old_mat2);
      total_time_old += dclock();
    }
    Float total_time_new = 0.;
    cps::ComplexD tr_new;
    for(int iter=0;iter<ntests;iter++){
      randomMatrix(new_mat1);
      randomMatrix(new_mat2);
      total_time_new -= dclock();
      tr_new = Trace(new_mat1,new_mat2);
      total_time_new += dclock();
    }
    printf("Prodtrace: Avg time new code %d iters: %g secs\n",ntests,total_time_new/ntests);
    printf("Prodtrace: Avg time old code %d iters: %g secs\n",ntests,total_time_old/ntests);
  }

}


void benchmarkColorTranspose(const int ntests, const double tol){
  typedef CPSsquareMatrix<CPSsquareMatrix<CPSsquareMatrix<cps::ComplexD,2>,3>,4> SCFmat;
  typedef CPSsquareMatrix<cps::ComplexD,3> Cmat;
    
  //Test they give the same answer
  {
    SpinColorFlavorMatrix old_mat;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;
    randomMatrix(old_mat,new_mat);

    old_mat.transpose_color();
    new_mat.TransposeOnIndex<1>();
      
    bool fail = false;
    for(int s1=0;s1<4;s1++)
      for(int s2=0;s2<4;s2++)	
	for(int c1=0;c1<3;c1++)
	  for(int c2=0;c2<3;c2++)
	    for(int f1=0;f1<2;f1++)
	      for(int f2=0;f2<2;f2++){
		cps::ComplexD gd = new_mat(s1,s2)(c1,c2)(f1,f2);
		cps::ComplexD cp = old_mat(s1,c1,f1,s2,c2,f2);
	  
		double rdiff = fabs(gd.real()-cp.real());
		double idiff = fabs(gd.imag()-cp.imag());
		if(rdiff > tol|| idiff > tol){
		  printf("Fail: colortranspose Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		  fail = true;
		}
	      }
    if(fail) ERR.General("","","colortranspose test failed\n");
    else printf("colortranspose pass\n");
  }
  {
    //Benchmark it    
    SpinColorFlavorMatrix old_mat;
    CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;
    
    Float total_time_old = 0.;

    for(int iter=0;iter<ntests;iter++){
      randomMatrix(old_mat);      
      total_time_old -= dclock();
      old_mat.transpose_color();
      total_time_old += dclock();
    }
    Float total_time_new = 0.;

    for(int iter=0;iter<ntests;iter++){
      randomMatrix(new_mat);
      total_time_new -= dclock();
      new_mat.TransposeOnIndex<1>();
      total_time_new += dclock();
    }
    printf("colorTranspose: Avg time new code %d iters: %g secs\n",ntests,total_time_new/ntests);
    printf("colorTranspose: Avg time old code %d iters: %g secs\n",ntests,total_time_old/ntests);
  }
}

void multGammaLeftOld(SpinColorFlavorMatrix &M, const int whichGamma, const int i, const int mu){
  assert(whichGamma == 1 || whichGamma==2);
  static int g1[8] = {0,1,0,1,2,3,2,3};
  static int g2[8] = {1,0,3,2,1,0,3,2};

  int gg = whichGamma == 1 ? g1[i] : g2[i];
  switch(gg){
  case 0:
    M.pl(F0).gl(mu);
    break;
  case 1:
    M.pl(F0).glAx(mu);
    break;
  case 2:
    M.pl(F1).gl(mu); M *= -1.0;
    break;
  case 3:
    M.pl(F1).glAx(mu); M *= -1.0;
    break;
  default:
    ERR.General("ComputeKtoPiPiGparityBase","multGammaLeft","Invalid idx\n");
    break;
  }
}

void benchmarkmultGammaLeft(const int ntests, const double tol){
  typedef CPSsquareMatrix<CPSsquareMatrix<CPSsquareMatrix<cps::ComplexD,2>,3>,4> SCFmat;
  
  //Test they give the same answer
  for(int gamma=1;gamma<=2;gamma++)
    for(int i=0;i<8;i++)
      for(int mu=0;mu<4;mu++){
	SpinColorFlavorMatrix old_mat;
	CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;
	randomMatrix(old_mat,new_mat);

	multGammaLeftOld(old_mat,gamma,i,mu);
	ComputeKtoPiPiGparityBase::multGammaLeft(new_mat,gamma,i,mu);
      
	bool fail = false;
	for(int s1=0;s1<4;s1++)
	  for(int s2=0;s2<4;s2++)	
	    for(int c1=0;c1<3;c1++)
	      for(int c2=0;c2<3;c2++)
		for(int f1=0;f1<2;f1++)
		  for(int f2=0;f2<2;f2++){
		    cps::ComplexD gd = new_mat(s1,s2)(c1,c2)(f1,f2);
		    cps::ComplexD cp = old_mat(s1,c1,f1,s2,c2,f2);
		    
		    double rdiff = fabs(gd.real()-cp.real());
		    double idiff = fabs(gd.imag()-cp.imag());
		    if(rdiff > tol|| idiff > tol){
		      printf("Fail: multGammaLeft(%d,%d,%d) Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gamma,i,mu,gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		      fail = true;
		    }
		  }
	if(fail) ERR.General("","","multGammaLeft test %d %d %d failed\n",gamma,i,mu);
	//else printf("multGammaLeft %d %d %d pass\n",gamma,i,mu);
      }

  //Benchmark
  for(int gamma=1;gamma<=2;gamma++)
    for(int i=0;i<8;i++)
      for(int mu=0;mu<4;mu++){
	SpinColorFlavorMatrix old_mat;
	CPSspinColorFlavorMatrix<cps::ComplexD> new_mat;

	Float total_time_old = 0.;
	
	for(int iter=0;iter<ntests;iter++){
	  randomMatrix(old_mat);      
	  total_time_old -= dclock();
	  multGammaLeftOld(old_mat,gamma,i,mu);
	  total_time_old += dclock();
	}
	Float total_time_new = 0.;
	
	for(int iter=0;iter<ntests;iter++){
	  randomMatrix(new_mat);
	  total_time_new -= dclock();
	  ComputeKtoPiPiGparityBase::multGammaLeft(new_mat,gamma,i,mu);
	  total_time_new += dclock();
	}
	printf("multGammaLeft %d %d %d: Avg time new code %d iters: %g secs\n",gamma,i,mu,ntests,total_time_new/ntests);
	printf("multGammaLeft %d %d %d: Avg time old code %d iters: %g secs\n\n",gamma,i,mu,ntests,total_time_old/ntests);
      }  
}

template<typename mf_Complex, typename grid_Complex>
bool compare(const CPSspinColorFlavorMatrix<mf_Complex> &orig, const CPSspinColorFlavorMatrix<grid_Complex> &grid, const double tol){
  bool fail = false;
  
  mf_Complex gd;
  for(int sl=0;sl<4;sl++)
    for(int cl=0;cl<3;cl++)
      for(int fl=0;fl<2;fl++)
	for(int sr=0;sr<4;sr++)
	  for(int cr=0;cr<3;cr++)
	    for(int fr=0;fr<2;fr++){
	      gd = Reduce( grid(sl,sr)(cl,cr)(fl,fr) );
	      const mf_Complex &cp = orig(sl,sr)(cl,cr)(fl,fr);
	      
	      double rdiff = fabs(gd.real()-cp.real());
	      double idiff = fabs(gd.imag()-cp.imag());
	      if(rdiff > tol|| idiff > tol){
		printf("Fail: Grid (%g,%g) CPS (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		fail = true;
	      }
	    }
  return !fail;
}

template<typename mf_Complex>
bool compare(const CPSspinColorFlavorMatrix<mf_Complex> &orig, const CPSspinColorFlavorMatrix<mf_Complex> &newimpl, const double tol){
  bool fail = false;
  
  for(int sl=0;sl<4;sl++)
    for(int cl=0;cl<3;cl++)
      for(int fl=0;fl<2;fl++)
	for(int sr=0;sr<4;sr++)
	  for(int cr=0;cr<3;cr++)
	    for(int fr=0;fr<2;fr++){
	      const mf_Complex &gd = newimpl(sl,sr)(cl,cr)(fl,fr);
	      const mf_Complex &cp = orig(sl,sr)(cl,cr)(fl,fr);
	      
	      double rdiff = fabs(gd.real()-cp.real());
	      double idiff = fabs(gd.imag()-cp.imag());
	      if(rdiff > tol|| idiff > tol){
		printf("Fail: Newimpl (%g,%g) Orig (%g,%g) Diff (%g,%g)\n",gd.real(),gd.imag(), cp.real(),cp.imag(), cp.real()-gd.real(), cp.imag()-gd.imag());
		fail = true;
	      }
	    }
  return !fail;
}


template< typename mf_Complex, int SiteSize, typename FlavorPolicy, typename AllocPolicy>
void printRow(const CPSfield<mf_Complex,SiteSize,FourDpolicy,FlavorPolicy,AllocPolicy> &field, const int dir, const std::string &comment,
	       typename my_enable_if< _equal<typename ComplexClassify<mf_Complex>::type, complex_double_or_float_mark>::value, const int>::type = 0
	       ){
  int L = GJP.Nodes(dir)*GJP.NodeSites(dir);
  std::vector<mf_Complex> buf(L,0.);

  int other_dirs[3]; int aa=0;
  for(int i=0;i<4;i++)
    if(i!=dir) other_dirs[aa++] = i;

  
  if(GJP.NodeCoor(other_dirs[0]) == 0 && GJP.NodeCoor(other_dirs[1]) == 0 && GJP.NodeCoor(other_dirs[2]) == 0){
    for(int x=GJP.NodeCoor(dir)*GJP.NodeSites(dir); x < (GJP.NodeCoor(dir)+1)*GJP.NodeSites(dir); x++){
      int lcoor[4] = {0,0,0,0};
      lcoor[dir] = x - GJP.NodeCoor(dir)*GJP.NodeSites(dir);
      
      mf_Complex const* site_ptr = field.site_ptr(lcoor);
      buf[x] = *site_ptr;
    }
  }
  globalSumComplex(buf.data(),L);

  
  if(!UniqueID()){
    printf("%s: (",comment.c_str()); fflush(stdout);
    for(int x=0;x<L;x++){
      if(x % GJP.NodeSites(dir) == 0 && x!=0)
	printf(")(");
      
      printf("[%f,%f] ",buf[x].real(),buf[x].imag());
    }
    printf(")\n"); fflush(stdout);
  }
}

#ifdef USE_GRID
template< typename mf_Complex, int SiteSize, typename FlavorPolicy, typename AllocPolicy>
void printRow(const CPSfield<mf_Complex,SiteSize,FourDSIMDPolicy,FlavorPolicy,AllocPolicy> &field, const int dir, const std::string &comment,
	       typename my_enable_if< _equal<typename ComplexClassify<mf_Complex>::type, grid_vector_complex_mark>::value, const int>::type = 0
	       ){
  typedef typename mf_Complex::scalar_type ScalarComplex;
  NullObject null_obj;
  CPSfield<ScalarComplex,SiteSize,FourDpolicy,FlavorPolicy,StandardAllocPolicy> tmp(null_obj);
  tmp.importField(field);
  printRow(tmp,dir,comment);
}
#endif



void testCyclicPermute(){
  NullObject null_obj;
  CPSfield<cps::ComplexD,1,FourDpolicy,FixedFlavorPolicy<1>,StandardAllocPolicy> from(null_obj);
  CPSfield<cps::ComplexD,1,FourDpolicy,FixedFlavorPolicy<1>,StandardAllocPolicy> tmp1(null_obj);
  CPSfield<cps::ComplexD,1,FourDpolicy,FixedFlavorPolicy<1>,StandardAllocPolicy> tmp2(null_obj);

  from.testRandom();

  for(int dir=0;dir<4;dir++){
    for(int pm=0;pm<2;pm++){
      if(!UniqueID()) printf("Testing permute in direction %c%d\n",pm == 1 ? '+' : '-',dir);
      //permute in incr until we cycle all the way around
      tmp1 = from;
      CPSfield<cps::ComplexD,1,FourDpolicy,FixedFlavorPolicy<1>,StandardAllocPolicy> *send = &tmp1;
      CPSfield<cps::ComplexD,1,FourDpolicy,FixedFlavorPolicy<1>,StandardAllocPolicy> *recv = &tmp2;

      int shifted = 0;
      printRow(from,dir,"Initial line      ");

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
      
      int coor[4];
      for(coor[0]=0;coor[0]<GJP.XnodeSites();coor[0]++){
	for(coor[1]=0;coor[1]<GJP.YnodeSites();coor[1]++){
	  for(coor[2]=0;coor[2]<GJP.ZnodeSites();coor[2]++){
	    for(coor[3]=0;coor[3]<GJP.TnodeSites();coor[3]++){
	      cps::ComplexD const* orig = from.site_ptr(coor);
	      cps::ComplexD const* permd = recv->site_ptr(coor);
	      if(orig->real() != permd->real() || orig->imag() != permd->imag()){
		printf("Error node coor (%d,%d,%d,%d) (%d,%d,%d,%d) : (%g,%g) vs (%g,%g) diff (%g,%g)\n",GJP.XnodeCoor(),GJP.YnodeCoor(),GJP.ZnodeCoor(),GJP.TnodeCoor(),coor[0],coor[1],coor[2],coor[3],orig->real(),orig->imag(),permd->real(),permd->imag(), orig->real()-permd->real(),orig->imag()-permd->imag());
	      }
	    }
	  }
	}
      }
    }
  }

#ifdef USE_GRID
  typedef FourDSIMDPolicy::ParamType simd_params;
  simd_params sp;
  FourDSIMDPolicy::SIMDdefaultLayout(sp, Grid::vComplexD::Nsimd() );
  
  CPSfield<Grid::vComplexD,1,FourDSIMDPolicy,FixedFlavorPolicy<1>,Aligned128AllocPolicy> from_grid(sp);
  CPSfield<Grid::vComplexD,1,FourDSIMDPolicy,FixedFlavorPolicy<1>,Aligned128AllocPolicy> tmp1_grid(sp);
  CPSfield<Grid::vComplexD,1,FourDSIMDPolicy,FixedFlavorPolicy<1>,Aligned128AllocPolicy> tmp2_grid(sp);
  from_grid.importField(from);

  for(int dir=0;dir<4;dir++){
    for(int pm=0;pm<2;pm++){
      if(!UniqueID()) printf("Testing permute in direction %c%d with SIMD layout\n",pm == 1 ? '+' : '-',dir);
      //permute in incr until we cycle all the way around
      tmp1_grid = from_grid;
      CPSfield<Grid::vComplexD,1,FourDSIMDPolicy,FixedFlavorPolicy<1>,Aligned128AllocPolicy> *send = &tmp1_grid;
      CPSfield<Grid::vComplexD,1,FourDSIMDPolicy,FixedFlavorPolicy<1>,Aligned128AllocPolicy> *recv = &tmp2_grid;

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
      
      int coor[4];
      for(coor[0]=0;coor[0]<GJP.XnodeSites();coor[0]++){
	for(coor[1]=0;coor[1]<GJP.YnodeSites();coor[1]++){
	  for(coor[2]=0;coor[2]<GJP.ZnodeSites();coor[2]++){
	    for(coor[3]=0;coor[3]<GJP.TnodeSites();coor[3]++){
	      cps::ComplexD const* orig = from.site_ptr(coor);
	      cps::ComplexD const* permd = tmp1.site_ptr(coor);
	      if(orig->real() != permd->real() || orig->imag() != permd->imag()){
		printf("Error node coor (%d,%d,%d,%d) (%d,%d,%d,%d) : (%g,%g) vs (%g,%g) diff (%g,%g)\n",GJP.XnodeCoor(),GJP.YnodeCoor(),GJP.ZnodeCoor(),GJP.TnodeCoor(),coor[0],coor[1],coor[2],coor[3],orig->real(),orig->imag(),permd->real(),permd->imag(), orig->real()-permd->real(),orig->imag()-permd->imag());
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

void testGenericFFT(){
  bool dirs[4] = {1,1,1,0}; //3d fft

  
  CPSfermion4D<cps::ComplexD> in;
  in.testRandom();
    
  CPSfermion4D<cps::ComplexD> out1;
  CPSfermion4D<cps::ComplexD> out2;

  out1.fft(in);
  fft(out2,in,dirs);
    
  printRow(out1,0,"Out1");
  printRow(out2,0,"Out2");
    
  assert( out1.equals(out2) );


  //Code for FFT WilsonMatrix
  WilsonMatrix* buf = (WilsonMatrix*)malloc( GJP.VolNodeSites() * sizeof(WilsonMatrix) ); //here are your WilsonMatrix
  
  NullObject null_obj;
  CPSfield<cps::ComplexD,12*12,FourDpolicy,OneFlavorPolicy,StandardAllocPolicy> cpy( (cps::ComplexD*)buf,null_obj);  //create a CPSfield and copy in data

  CPSfield<cps::ComplexD,12*12,FourDpolicy,OneFlavorPolicy,StandardAllocPolicy> into(null_obj); //FFT output
  fft(into,cpy,dirs); //do the FFT

  free(buf);
  
}

template<typename mf_Complex>
void demonstrateFFTreln(const A2AArg &a2a_args){
  //Demonstrate relation between FFTW fields
  typedef _deduce_a2a_field_policies<mf_Complex> A2Apolicies;
  typedef GridA2APolicies<A2Apolicies> A2Apolicies_ext;
    
  A2AvectorW<A2Apolicies_ext> W(a2a_args);
  A2AvectorV<A2Apolicies_ext> V(a2a_args);
  W.testRandom();
  V.testRandom();

  int p1[3] = {1,1,1};
  int p5[3] = {5,1,1};

  twist<typename A2Apolicies_ext::FermionFieldType> twist_p1(p1);
  twist<typename A2Apolicies_ext::FermionFieldType> twist_p5(p5);
    
  A2AvectorVfftw<A2Apolicies_ext> Vfftw_p1(a2a_args);
  Vfftw_p1.fft(V,&twist_p1);

  A2AvectorVfftw<A2Apolicies_ext> Vfftw_p5(a2a_args);
  Vfftw_p5.fft(V,&twist_p5);

  //f5(n) = f1(n+1)
  for(int i=0;i<Vfftw_p1.getNmodes();i++)
    cyclicPermute(Vfftw_p1.getMode(i), Vfftw_p1.getMode(i), 0, -1, 1);
    
  printRow(Vfftw_p1.getMode(0),0, "T_-1 V(p1) T_-1");
  printRow(Vfftw_p5.getMode(0),0, "V(p5)          ");

  for(int i=0;i<Vfftw_p1.getNmodes();i++)
    assert( Vfftw_p1.getMode(i).equals( Vfftw_p5.getMode(i), 1e-8, true ) );

  A2AvectorWfftw<A2Apolicies_ext> Wfftw_p1(a2a_args);
  Wfftw_p1.fft(W,&twist_p1);

  A2AvectorWfftw<A2Apolicies_ext> Wfftw_p5(a2a_args);
  Wfftw_p5.fft(W,&twist_p5);

  for(int i=0;i<Wfftw_p1.getNmodes();i++)
    cyclicPermute(Wfftw_p1.getMode(i), Wfftw_p1.getMode(i), 0, -1, 1);

  printRow(Wfftw_p1.getMode(0),0, "T_-1 W(p1) T_-1");
  printRow(Wfftw_p5.getMode(0),0, "W(p5)          ");

  for(int i=0;i<Wfftw_p1.getNmodes();i++)
    assert( Wfftw_p1.getMode(i).equals( Wfftw_p5.getMode(i), 1e-8, true ) );

}

template<typename ParamType, typename mf_Complex>
struct defaultFieldParams{
  static void get(ParamType &into){}
};

template<int N, typename mf_Complex>
struct defaultFieldParams< SIMDdims<N>, mf_Complex >{
  static void get(SIMDdims<N> &into){
    SIMDpolicyBase<N>::SIMDdefaultLayout(into, mf_Complex::Nsimd(), 2);
  }
};

template<typename mf_Complex>
void testA2AvectorFFTrelnGparity(const A2AArg &a2a_args,Lattice &lat){
  assert(GJP.Gparity());

  if(lat.FixGaugeKind() == FIX_GAUGE_NONE){
    FixGaugeArg fix_gauge_arg;
    fix_gauge_arg.fix_gauge_kind = FIX_GAUGE_COULOMB_T;
    fix_gauge_arg.hyperplane_start = 0;
    fix_gauge_arg.hyperplane_step = 1;
    fix_gauge_arg.hyperplane_num = GJP.Tnodes()*GJP.TnodeSites();
    fix_gauge_arg.stop_cond = 1e-08;
    fix_gauge_arg.max_iter_num = 10000;

    CommonArg common_arg;
  
    AlgFixGauge fix_gauge(lat,&common_arg,&fix_gauge_arg);
    fix_gauge.run();
  }
  
  //Demonstrate relation between FFTW fields
  typedef _deduce_a2a_field_policies<mf_Complex> A2Apolicies;
  typedef GridA2APolicies<A2Apolicies> A2Apolicies_ext;

  typedef typename A2AvectorWfftw<A2Apolicies_ext>::FieldInputParamType FieldInputParamType;
  FieldInputParamType fp; defaultFieldParams<FieldInputParamType, mf_Complex>::get(fp);
  
  A2AvectorW<A2Apolicies_ext> W(a2a_args,fp);
  W.testRandom();

  int p_p1[3];
  GparityBaseMomentum(p_p1,+1);

  int p_m1[3];
  GparityBaseMomentum(p_m1,-1);

  //Perform base FFTs
  //twist<typename A2Apolicies_ext::FermionFieldType> twist_p1(p_p1);
  //twist<typename A2Apolicies_ext::FermionFieldType> twist_m1(p_m1);

  gaugeFixAndTwist<typename A2Apolicies_ext::FermionFieldType> twist_p1(p_p1,lat);
  gaugeFixAndTwist<typename A2Apolicies_ext::FermionFieldType> twist_m1(p_m1,lat);
  
  A2AvectorWfftw<A2Apolicies_ext> Wfftw_p1(a2a_args,fp);
  Wfftw_p1.fft(W,&twist_p1);

  A2AvectorWfftw<A2Apolicies_ext> Wfftw_m1(a2a_args,fp);
  Wfftw_m1.fft(W,&twist_m1);


  int p[3];  
  A2AvectorWfftw<A2Apolicies_ext> result(a2a_args,fp);
  A2AvectorWfftw<A2Apolicies_ext> compare(a2a_args,fp);
  
  //Get twist for first excited momentum in p1 set
  {
    memcpy(p,p_p1,3*sizeof(int));
    p[0] = 5;
    //twist<typename A2Apolicies_ext::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies_ext::FermionFieldType> twist_p(p,lat);    
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in x direction\n",p[0],p[1],p[2]);

    printRow(result.getMode(0),0,  "Result ");
    printRow(compare.getMode(0),0, "Compare");
    
    for(int i=0;i<compare.getNmodes();i++)
      assert( compare.getMode(i).equals( result.getMode(i), 1e-8, true ) );
  }
  //Get twist for first negative excited momentum in p1 set
  if(GJP.Bc(1) == BND_CND_GPARITY){
    memcpy(p,p_p1,3*sizeof(int));
    p[1] = -3;
    //twist<typename A2Apolicies_ext::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies_ext::FermionFieldType> twist_p(p,lat);    
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in y direction\n",p[0],p[1],p[2]);

    printRow(result.getMode(0),0,  "Result ");
    printRow(compare.getMode(0),0, "Compare");
    
    for(int i=0;i<compare.getNmodes();i++)
      assert( compare.getMode(i).equals( result.getMode(i), 1e-8, true ) );
  }
  //Try two directions
  if(GJP.Bc(1) == BND_CND_GPARITY){
    memcpy(p,p_p1,3*sizeof(int));
    p[0] = -3;
    p[1] = -3;
    //twist<typename A2Apolicies_ext::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies_ext::FermionFieldType> twist_p(p,lat);    
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in y direction\n",p[0],p[1],p[2]);

    printRow(result.getMode(0),0,  "Result ");
    printRow(compare.getMode(0),0, "Compare");
    
    for(int i=0;i<compare.getNmodes();i++)
      assert( compare.getMode(i).equals( result.getMode(i), 1e-7, true ) );
  }
  //Try 3 directions
  if(GJP.Bc(1) == BND_CND_GPARITY && GJP.Bc(2) == BND_CND_GPARITY){
    memcpy(p,p_p1,3*sizeof(int));
    p[0] = -3;
    p[1] = -3;
    p[2] = -3;
    //twist<typename A2Apolicies_ext::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies_ext::FermionFieldType> twist_p(p,lat);    
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in y direction\n",p[0],p[1],p[2]);

    printRow(result.getMode(0),0,  "Result ");
    printRow(compare.getMode(0),0, "Compare");
    
    for(int i=0;i<compare.getNmodes();i++)
      assert( compare.getMode(i).equals( result.getMode(i), 1e-7, true ) );
  }
  //Get twist for first excited momentum in m1 set
  {
    memcpy(p,p_m1,3*sizeof(int));
    p[0] = 3;
    //twist<typename A2Apolicies_ext::FermionFieldType> twist_p(p);
    gaugeFixAndTwist<typename A2Apolicies_ext::FermionFieldType> twist_p(p,lat);   
    compare.fft(W,&twist_p);

    result.getTwistedFFT(p, &Wfftw_p1, &Wfftw_m1);

    if(!UniqueID()) printf("Testing p=(%d,%d,%d). Should require permute of 1 in x direction\n",p[0],p[1],p[2]);

    printRow(result.getMode(0),0,  "Result ");
    printRow(compare.getMode(0),0, "Compare");
    
    for(int i=0;i<compare.getNmodes();i++)
      assert( compare.getMode(i).equals( result.getMode(i), 1e-8, true ) );
  }
  
}



template<typename T>
struct _printit{
  static void printit(const T d[], const int n){
    for(int i=0;i<n;i++){
      std::cout << d[i] << " ";
    }
    std::cout << std::endl;
  }
};

template<typename T>
struct _printit<std::complex<T> >{
  static void printit(const std::complex<T> d[], const int n){
    for(int i=0;i<n;i++){
      std::cout << '[' << d[i].real() << ',' << d[i].imag() << "] ";
    }
    std::cout << std::endl;
  }
};

  
template<typename T>
void printit(const T d[], const int n){
  _printit<T>::printit(d,n);
}



template<typename T>
void printvType(const T& v){
  typedef typename T::scalar_type S;
  int Nsimd = T::Nsimd();
  S to[Nsimd];
  vstore(v,to);
  printit(to,Nsimd);
}


template<typename T>
struct _rand{
  inline static T rand(){
    return LRG.Urand();
  }
};



template<typename T>
struct _rand<std::complex<T> >{
  inline static std::complex<T> rand(){
    return std::complex<T>(LRG.Urand(),LRG.Urand());
  }
};

template<typename T>
T randomvType(){
  T out;
  typedef typename T::scalar_type S;
  int Nsimd = T::Nsimd();
  S r[Nsimd];
  for(int i=0;i<Nsimd;i++) r[i] = _rand<S>::rand();
  vset(out,r);
  return out;
}






  
//  static void ComputeKtoPiPiGparityBase::multGammaLeft(CPSspinColorFlavorMatrix<ComplexType> &M, const int whichGamma, const int i, const int mu){

CPS_END_NAMESPACE

#endif
