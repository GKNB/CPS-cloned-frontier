#ifndef _COMPUTE_BK_AMA_H
#define _COMPUTE_BK_AMA_H

CPS_START_NAMESPACE

//We compute B_K for a given first kaon timeslice (t0) and a fixed K->K separation for each operator insertion time. The sink kaon timeslice is t1 = (t0 + tsep) % Lt
//The matrix is indexed as [t0][(top-t0+Lt)%Lt]
//Source momenta of the strange quark props are needed for the flavor projection.
//It is assumed that the total kaon momentum is zero, and we project onto zero momentum at the operator insertion
void GparityBK(fMatrix<double> &into, const int t0, 
	       const PropWrapper &prop_h_t0, const PropWrapper &prop_l_t0, const ThreeMomentum &p_psi_h_t0,
	       const PropWrapper &prop_h_t1, const PropWrapper &prop_l_t1, const ThreeMomentum &p_psi_h_t1,
	       const bool do_flav_project = true
	       ){
  const int Lt = GJP.TnodeSites()*GJP.Tnodes();

  const int nthread = omp_get_max_threads();
  basicComplexArray<double> tmp(Lt,nthread); //defaults to zero for all elements

  FlavorMatrix kaon_proj_t0 = getProjector(p_psi_h_t0);
  FlavorMatrix kaon_proj_t1 = getProjector(p_psi_h_t1);

#pragma omp parallel for
  for(int x=0;x<GJP.VolNodeSites();x++){
    int pos[4];
    int rem = x;
    for(int i=0;i<4;i++){ pos[i] = rem % GJP.NodeSites(i); rem /= GJP.NodeSites(i); }

    int t_glb = pos[3] + GJP.TnodeCoor() * GJP.TnodeSites();
    int tdis_glb = (t_glb - t0 + Lt) % Lt; //operator insertion time

    SpinColorFlavorMatrix prop_l_t0_site;
    prop_l_t0.siteMatrix(prop_l_t0_site,x);
    if(do_flav_project) prop_l_t0_site *= kaon_proj_t0;

    SpinColorFlavorMatrix prop_h_dag_t0_site;
    prop_h_t0.siteMatrix(prop_h_dag_t0_site,x);
    prop_h_dag_t0_site.hconj();
    
    SpinColorFlavorMatrix prop_prod_t0 = prop_l_t0_site * prop_h_dag_t0_site;

    SpinColorFlavorMatrix prop_l_t1_site;
    prop_l_t1.siteMatrix(prop_l_t1_site,x);
    if(do_flav_project) prop_l_t1_site *= kaon_proj_t1;

    SpinColorFlavorMatrix prop_h_dag_t1_site;
    prop_h_t1.siteMatrix(prop_h_dag_t1_site,x);
    prop_h_dag_t1_site.hconj();

    SpinColorFlavorMatrix prop_prod_t1 = prop_l_t1_site * prop_h_dag_t1_site;

    for(int mu=0;mu<4;mu++){
      for(int Gamma = 0; Gamma < 2; Gamma++){  //\gamma^\mu and \gamma^\mu\gamma^5
	SpinColorFlavorMatrix part1 = prop_prod_t0;
	if(Gamma == 1) part1.gl(-5);
	part1.gl(mu);
	part1.pr(F0);

	SpinColorFlavorMatrix part2 = prop_prod_t1;
	if(Gamma == 1) part2.gl(-5);
	part2.gl(mu);
	part2.pr(F0);

	tmp(tdis_glb, omp_get_thread_num()) += 2.0*Trace(part1)*Trace(part2);
	tmp(tdis_glb, omp_get_thread_num()) += -2.0*Trace(part1, part2);
      }
    }
  }
  tmp.threadSum();
  tmp.nodeSum();

  for(int tdis=0;tdis<Lt;tdis++)
    into(t0, tdis) = tmp[tdis];
}




void StandardBK(fMatrix<double> &into, const int t0, 
	       const PropWrapper &prop_h_t0, const PropWrapper &prop_l_t0,
	       const PropWrapper &prop_h_t1, const PropWrapper &prop_l_t1){
  const int Lt = GJP.TnodeSites()*GJP.Tnodes();

  const int nthread = omp_get_max_threads();
  basicComplexArray<double> tmp(Lt,nthread); //defaults to zero for all elements

#pragma omp_parallel for
  for(int x=0;x<GJP.VolNodeSites();x++){
    int pos[4];
    int rem = x;
    for(int i=0;i<4;i++){ pos[i] = rem % GJP.NodeSites(i); rem /= GJP.NodeSites(i); }

    int t_glb = pos[3] + GJP.TnodeCoor() * GJP.TnodeSites();
    int tdis_glb = (t_glb - t0 + Lt) % Lt; //operator insertion time

    WilsonMatrix prop_l_t0_site;
    prop_l_t0.siteMatrix(prop_l_t0_site,x);

    WilsonMatrix prop_h_dag_t0_site;
    prop_h_t0.siteMatrix(prop_h_dag_t0_site,x);
    prop_h_dag_t0_site.hconj();
    
    WilsonMatrix prop_prod_t0 = prop_l_t0_site * prop_h_dag_t0_site;

    WilsonMatrix prop_l_t1_site;
    prop_l_t1.siteMatrix(prop_l_t1_site,x);

    WilsonMatrix prop_h_dag_t1_site;
    prop_h_t1.siteMatrix(prop_h_dag_t1_site,x);
    prop_h_dag_t1_site.hconj();

    WilsonMatrix prop_prod_t1 = prop_l_t1_site * prop_h_dag_t1_site;

    for(int mu=0;mu<4;mu++){
      for(int Gamma = 0; Gamma < 2; Gamma++){  //\gamma^\mu and \gamma^\mu\gamma^5
	WilsonMatrix part1 = prop_prod_t0;
	if(Gamma == 1) part1.gl(-5);
	part1.gl(mu);

	WilsonMatrix part2 = prop_prod_t1;
	if(Gamma == 1) part2.gl(-5);
	part2.gl(mu);

	tmp(tdis_glb, omp_get_thread_num()) += 2.0*Trace(part1)*Trace(part2);
	tmp(tdis_glb, omp_get_thread_num()) += -2.0*Trace(part1, part2);
      }
    }
  }
  tmp.threadSum();
  tmp.nodeSum();

  for(int tdis=0;tdis<Lt;tdis++)
    into(t0, tdis) = tmp[tdis];
}




CPS_END_NAMESPACE

#endif
