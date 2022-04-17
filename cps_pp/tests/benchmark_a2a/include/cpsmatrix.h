#pragma once

CPS_START_NAMESPACE

//----------------------------------
//Benchmarks for CPSmatrix and related
//------------------------------------


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
    new_mat = new_mat.TransposeOnIndex<1>();
      
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
      new_mat = new_mat.TransposeOnIndex<1>();
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



CPS_END_NAMESPACE