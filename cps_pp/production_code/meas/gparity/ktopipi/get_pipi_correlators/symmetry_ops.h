inline mom parity(const mom &m){ return -m; }

inline mom axisExchange(const mom &m, const int ax1, const int ax2){
  mom out(m);
  out[ax1] = m[ax2];
  out[ax2] = m[ax1];
  return out;
}

//Map  unit   0
//     parity 1
//     Pxy    2
//     Pxz    3
//     Pyz    4

inline mom applyOp(const mom &m, const int op){
  switch(op){
  case 0:
    return m;
  case 1:
    return parity(m);
  case 2:
    return axisExchange(m,0,1);
  case 3:
    return axisExchange(m,0,2);
  case 4:
    return axisExchange(m,1,2);
  default:
    assert(0);
  }
}

inline mom applyOp(const mom &m, const std::vector<int> &op){ //apply from right! 
  mom out = m;
  for(int i=op.size()-1;i>=0;i--) out = applyOp(out, op[i]);
  return out;
}


typedef std::vector<int> Operations;

inline std::ostream & operator<<(std::ostream &os, const std::vector<int> &m){
  os << "("; 
  for(int i=0;i<m.size();i++){
    if(i>0) os << " ";
    os << m[i];
  }
  os << ")";
  return os;
}

//12 symmetry ops
  //1        (a b c)  0
  //Pxy      (b a c)  1
  //Pxz      (c b a)  2
  //Pyz      (a c b)  3
  //Pxz Pxy  (c a b)  4
  //Pxy Pxz  (b c a)  5
  //and parity partners  6..11
mom applySymmetryOp(const mom &in, const int symm){
  assert(symm < 12 && symm >=0);
  int p = symm/6;
  int q = symm % 6;

  switch(q){
  case 0:
    return applyOp(in,p);
  case 1:
    return applyOp(in,{p,2});
  case 2:
    return applyOp(in,{p,3});
  case 3:
    return applyOp(in,{p,4});
  case 4:
    return applyOp(in,{p,3,2});
  case 5:
    return applyOp(in,{p,2,3});
  }
}

std::vector<Operations> getReln(const mom &a, const mom &b, const bool allow_parity, const bool allow_axis_perm){ //Operation applies to b
  //Only 12 possibilities
  //1        (a b c)
  //Pyz      (a c b)
  //Pxy      (b a c)
  //Pxz      (c b a)
  //Pxz Pxy  (c a b)
  //Pxy Pxz  (b c a)
  //and parity partners
  
  std::vector<Operations> out;

  for(int pty=0;pty<int(allow_parity) + 1;pty++){
    mom base = pty == 1 ? parity(b) : b;
    if(base == a) out.push_back({pty});

    if(allow_axis_perm){    
      mom test = applyOp(base,2);
      if(test == a) out.push_back({2,pty});
      
      test = applyOp(base,3);
      if(test == a) out.push_back({3,pty});
      
      test = applyOp(base,4);
      if(test == a) out.push_back({4,pty});
      
      test = applyOp(base,{3,2});
      if(test == a) out.push_back({3,2,pty});
      
      test = applyOp(base,{2,3});
      if(test == a) out.push_back({2,3,pty});
    }
  }
  return out;
}

//Get the set of momenta related to p by the allowed symmetries
std::set<mom> getUniqueRelatedMom(const mom &p, const bool allow_parity, const bool allow_axis_perm){
  std::set<mom> out;

  if(p == mom({0,0,0})){
    out.insert(p);
    return out;
  }else{
    //Only 12 possibilities
    //1        (a b c)
    //Pyz      (a c b)
    //Pxy      (b a c)
    //Pxz      (c b a)
    //Pxz Pxy  (c a b)
    //Pxy Pxz  (b c a)
    //and parity partners
    
    for(int pty=0;pty<int(allow_parity) + 1;pty++){
      mom base = pty == 1 ? parity(p) : p;
      out.insert(base);

      if(allow_axis_perm){    
	mom test = applyOp(base,2);
	out.insert(test);
      
	test = applyOp(base,3);
	out.insert(test);

	test = applyOp(base,4);
	out.insert(test);

	test = applyOp(base,{3,2});
	out.insert(test);

	test = applyOp(base,{2,3});
	out.insert(test);
      }
    }
    return out;
  }
}
