typedef std::array<int,3> mom;

inline mom operator-(const mom &r){ return mom({-r[0],-r[1],-r[2]}); }
inline mom operator+(const mom &a, const mom &b){ return mom( {a[0]+b[0], a[1]+b[1], a[2]+b[2]} ); }
inline mom operator-(const mom &a, const mom &b){ return mom( {a[0]-b[0], a[1]-b[1], a[2]-b[2]} ); }
inline double mod(const mom&a){ return sqrt( double(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]) ); }

inline std::ostream & operator<<(std::ostream &os, const mom &m){
  os << "(" << m[0] << ", " << m[1] << ", " << m[2] << ")"; return os;
}
typedef std::pair<mom,mom> momPair;

inline momPair operator-(const momPair &r){ return momPair(-r.first,-r.second); }

inline std::ostream & operator<<(std::ostream &os, const momPair &m){
  os << m.first << "+" << m.second; return os;
}

//p[i+n] = p[i]
inline mom cyclicPermute(const mom &p, const int n){
  mom out;
  for(int i=0;i<3;i++){
    int ii = (i+n) % 3;
    out[ii] = p[i];
  }
  return out;
}
inline momPair cyclicPermute(const momPair &p, const int n){
  return momPair(cyclicPermute(p.first,n), cyclicPermute(p.second,n));
}
