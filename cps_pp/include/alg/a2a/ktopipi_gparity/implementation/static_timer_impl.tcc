#define CONCAT_(A,B) A ## B
#define CONCAT(A,B) CONCAT_(A,B)
#define TOSTR_(A) #A
#define TOSTR(A) TOSTR_(A)

struct CONCAT(_, TIMER){
#define ELEM(A) double A;
  TIMER_ELEMS
#undef ELEM
  void reset(){
#define ELEM(A) A = 0.;
  TIMER_ELEMS
#undef ELEM
  }
  void report(){
#define ELEM(A) print_time(TOSTR(TIMER),TOSTR(A), A);
  TIMER_ELEMS
#undef ELEM  
  }
  CONCAT(_, TIMER)(){ reset(); }
};

struct TIMER{
  static CONCAT(_, TIMER) & timer(){
    static CONCAT(_, TIMER) t;
    return t;
  }
};

#undef TIMER_ELEMS
#undef TIMER
#undef CONCAT_
#undef CONCAT
#undef TOSTR_
#undef TOSTR

