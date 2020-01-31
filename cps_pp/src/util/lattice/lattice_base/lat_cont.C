#include <config.h>
#include <util/lat_cont.h>
#include <util/error.h>

CPS_START_NAMESPACE
char * LatticeContainer::cname = "LatticeContainer";
LatticeContainer::LatticeContainer(){
	const char *fname="LatticeContainer()";
	size_t mat_size = GJP.VolNodeSites()*4;
	if(GJP.Gparity()) mat_size *= 2;
	gauge_p = new Matrix[mat_size];
	VRB.Result(cname,fname,"gauge_p=%p\n",gauge_p);
}
LatticeContainer::~LatticeContainer(){
	delete[] gauge_p;
}

void LatticeContainer::Get(Lattice &lat){
	const char *fname="Get(Lattice&)";
	str_ord = lat.StrOrd();
	VRB.Result(cname,fname,"str_ord=%d\n",str_ord);
	lat.CopyGaugeField(gauge_p);
}
void LatticeContainer::Set(Lattice &lat){
	const char *fname="Get(Lattice&)";
	VRB.Result(cname,fname,"str_ord=%d lat.StrOrd()=%d\n",str_ord,lat.StrOrd());
	if (str_ord != lat.StrOrd())
	ERR.General(cname,"Set()","Storage ordering of LatticeContainer(%d) doesn't agree with lattice ordering(%d)", str_ord,lat.StrOrd());
	lat.GaugeField(gauge_p);
}
CPS_END_NAMESPACE
