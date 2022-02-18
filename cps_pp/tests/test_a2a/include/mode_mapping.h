#pragma once

CPS_START_NAMESPACE

void testModeMappingTranspose(const A2AArg &a2a_arg){
  if(!UniqueID()) printf("Starting testModeMappingTranspose\n");
  //FullyPackedIndexDilution dilA(a2a_arg);
  //TimeFlavorPackedIndexDilution dilB(a2a_arg);
  typedef ModeMapping<FullyPackedIndexDilution, TimeFlavorPackedIndexDilution> mapAB;
  typedef ModeMapping<TimeFlavorPackedIndexDilution, FullyPackedIndexDilution> mapBA;

  typename mapAB::TensorType mapAB_v;
  mapAB::compute(mapAB_v, a2a_arg);

  typename mapBA::TensorType mapBA_v;
  mapBA::compute(mapBA_v, a2a_arg);

  //FullyPackedIndexDilution  packed sc, f, t
  //TimeFlavorPackedIndexDilution   packed f,t

  int nf = GJP.Gparity() ? 2:1;
  int nt = GJP.Tnodes()*GJP.TnodeSites();

  int sizes_expect_AB[] = {12, nf, nt, nf, nt};
  int sizes_expect_BA[] = {nf, nt, 12, nf, nt};

  EXPECT_EQ(mapAB_v.size(), sizes_expect_AB[0]);
  EXPECT_EQ(mapAB_v[0].size(), sizes_expect_AB[1]);
  EXPECT_EQ(mapAB_v[0][0].size(), sizes_expect_AB[2]);
  EXPECT_EQ(mapAB_v[0][0][0].size(), sizes_expect_AB[3]);
  EXPECT_EQ(mapAB_v[0][0][0][0].size(), sizes_expect_AB[4]);

  EXPECT_EQ(mapBA_v.size(), sizes_expect_BA[0]);
  EXPECT_EQ(mapBA_v[0].size(), sizes_expect_BA[1]);
  EXPECT_EQ(mapBA_v[0][0].size(), sizes_expect_BA[2]);
  EXPECT_EQ(mapBA_v[0][0][0].size(), sizes_expect_BA[3]);
  EXPECT_EQ(mapBA_v[0][0][0][0].size(), sizes_expect_BA[4]);

  for(int sc1=0;sc1<12;sc1++){
    for(int f1=0;f1<nf;f1++){
      for(int t1=0;t1<nt;t1++){
	for(int f2=0;f2<nf;f2++){
	  for(int t2=0;t2<nt;t2++){	    
	    EXPECT_EQ(mapAB_v[sc1][f1][t1][f2][t2].size(), mapBA_v[f2][t2][sc1][f1][t1].size());
	    for(int i=0;i<mapAB_v[sc1][f1][t1][f2][t2].size();i++){
	      const std::pair<int,int> &lv = mapAB_v[sc1][f1][t1][f2][t2][i];
	      const std::pair<int,int> &rv = mapBA_v[f2][t2][sc1][f1][t1][i];
	      std::pair<int,int> rvt = {rv.second, rv.first}; //of course it will transpose the indices
	      EXPECT_EQ(lv, rvt);
	    }
	  }
	}
      }
    }
  }

 
  if(!UniqueID()) printf("Finished testModeMappingTranspose\n");
}


CPS_END_NAMESPACE
