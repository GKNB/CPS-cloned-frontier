#ifndef GP_MEAS_ENUMS_H
#define GP_MEAS_ENUMS_H

#include<config.h>

CPS_START_NAMESPACE

enum PropPrecision { Sloppy, Exact };
inline std::string toString(const PropPrecision p){ return p == Sloppy ? "Sloppy" : "Exact"; }

enum TbcCombination { Single, CombinationF, CombinationB }; //F=(P+A)/2  B=(P-A)/2
inline std::string toString(const TbcCombination p){
  switch(p){
  case Single:
    return "Single";
  case CombinationF:
    return "CombinationF";
  case CombinationB:
    return "CombinationB";
  };
};

enum QuarkType { Light, Heavy };
inline std::string toString(const QuarkType p){ return p == Light ? "Light" : "Heavy"; }

enum MomentumOf { SrcPsiBar, SrcPsi, DaggeredProp, UndaggeredProp, Total };

CPS_END_NAMESPACE
#endif
