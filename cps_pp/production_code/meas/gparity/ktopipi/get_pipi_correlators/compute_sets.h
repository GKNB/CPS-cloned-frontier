


//Work out the set of correlators that we can form from the meson fields provided with total momentum p_tot and other total momenta related by allowed symmetries
std::list<Correlator> getCorrelators(const std::vector<mom> &R, const mom &p_tot, const bool allow_parity_ptot, const bool allow_axis_perm_ptot){
  std::set<mom> mom_have;
  for(int i=0;i<R.size();i++) mom_have.insert(R[i]);

  std::set<mom> all_p_tot = getUniqueRelatedMom(p_tot, allow_parity_ptot, allow_axis_perm_ptot);

  std::list<Correlator> correlators;

  std::cout << "Given symmetries relate total mom " << p_tot << " to the following:";
  for(std::set<mom>::const_iterator ptotit = all_p_tot.begin(); ptotit != all_p_tot.end(); ptotit++){
    std::cout << " " << *ptotit;

    //Pi-pi operators with this total momentum
    std::vector<momPair> pion_srcsnk;
    for(int i=0;i<R.size();i++){
      mom p2 = *ptotit - R[i];
      if(mom_have.count(p2)) pion_srcsnk.push_back(momPair(R[i],p2));
    }
  
    //All the correlators that we can form with p_tot^src == p_tot^snk
    for(int i=0;i<pion_srcsnk.size();i++)
      for(int j=0;j<pion_srcsnk.size();j++)
	correlators.push_back( Correlator(pion_srcsnk[i],-pion_srcsnk[j]) );  //ptot of snk = -ptot of src
  }
  std::cout << std::endl;

  return correlators;
}


//Group correlators into sets in which the elements are related by the allowed symmetries
std::vector<std::vector<Correlator> > computeSets(std::list<Correlator> correlators, const bool allow_parity, const bool allow_axis_perm, const bool allow_aux_diag){
  std::vector<std::vector<Correlator> > sets;

  while(correlators.size() > 0){
    if(sets.size() == 0){
      sets.resize(1, std::vector<Correlator>(1,*correlators.begin()));
      correlators.erase(correlators.begin());
    }else{
      std::list<Correlator>::iterator it = correlators.begin();
      bool found = false;
      for(int s=0;s<sets.size();s++){
	if(hasReln(*it, sets[s][0], allow_parity,allow_axis_perm,allow_aux_diag)){
	  sets[s].push_back(*it);	  
	  found = true;
	  break;
	}
      }
      if(!found)
	sets.push_back(std::vector<Correlator>(1,*it));
      
      correlators.erase(it);
    }
  }
  return sets;
}


void testSets(const std::vector<std::vector<Correlator> > &sets, const std::list<Correlator> &correlators,  const bool allow_parity, const bool allow_axis_perm, const bool allow_aux_diag){
  std::set<Correlator> all_c;
  for(auto it = correlators.begin(); it != correlators.end(); it++) all_c.insert(*it);

  for(int s=0;s<sets.size();s++){
    //Ensure every trasformation of the first entry is present in the set
    std::set<Correlator> elems;
    for(int i=0;i<sets[s].size();i++) elems.insert(sets[s][i]);

    const Correlator &base = sets[s][0];
    for(int aux=0;aux<(int)allow_aux_diag+1;aux++){
      for(int symm=0;symm<12;symm++){ //filter according to allowed symms!
	if(!allow_parity && symm / 6 > 0) continue;
	if(!allow_axis_perm && symm % 6 > 0) continue;

	Correlator trans = applySymmetryOp(aux ? auxDiag(base) : base, symm);
	if(all_c.count(trans) == 0) continue; //not in original set

	if(elems.count(trans) == 0){ 
	  std::cout << "Found correlator " << trans << " related to " << base << " by symm " << aux << ":" << symm << " which is not in set " << s << std::endl;
	  exit(-1);
	}
      }
    }
  }
}



