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
  
  CONCAT(_, TIMER)& operator+=(const CONCAT(_, TIMER) &r){
#define ELEM(A) this->A += r.A;
  TIMER_ELEMS;
#undef ELEM
  return *this;
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

#ifndef _CPS_TIMER_START_STOP_DEF
#define _CPS_TIMER_START_STOP_DEF

void timerStart(double &v,const std::string &descr){
  __asm__ volatile("" : : : "memory");  //prevent compiler reordering
  double s = secs_since_first_call();
  std::cout << "Timer start \"" << descr << "\": " << s << std::endl;
  v -= s;
  __asm__ volatile("" : : : "memory");
}
void timerEnd(double &v,const std::string &descr){
  __asm__ volatile("" : : : "memory");  //prevent compiler reordering
  double s = secs_since_first_call();
  std::cout << "Timer stop \"" << descr << "\": " << s << std::endl;
  v += s;
  __asm__ volatile("" : : : "memory");
}
#endif
