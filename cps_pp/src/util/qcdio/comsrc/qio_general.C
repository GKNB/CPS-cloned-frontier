#ifdef USE_QIO
#include <config.h>
#include <util/qio_general.h>

CPS_START_NAMESPACE
using namespace std;


// the helper functions are outside the class...

int qio_node_number( const int x[])
{
  // node number of node at global-lattice-point x[]=x,y,z,t  

  #ifdef DEBUG_NodeNumber
  printf("UID: %i, called qio_node_number with x: [%i, %i, %i, %i], calculating...\n", UniqueID(),x[0],x[1],x[2],x[3]);
  #endif // DEBUG_NodeNumber


  //convert x in xGrid

  int xGrid[4]={0,0,0,0};
  for(int ii(0); ii < 4; ++ii)
    {
      int div = x[ii];
      while ( div >= GJP.NodeSites(ii) )
	{
	  ++xGrid[ii];
	  div -= GJP.NodeSites(ii);
	}
    }

  //now get nodeNumber from xGrid
  // assuming that QMP-node numbering is the same as CPS... !!!

  int tmp(xGrid[0]);
  int vol(GJP.Nodes(0));
  
  for( int ii(1); ii <4; ++ii)
    {
      tmp += vol*xGrid[ii];
      vol *= GJP.Nodes(ii);
    }
  

  #ifdef DEBUG_NodeNumber
  printf("UID: %i, called qio_node_number with x: [%i, %i, %i, %i], returned: %i\n", UniqueID(),x[0],x[1],x[2],x[3],tmp);
  #endif // DEBUG_NodeNumber
  
  return tmp;
  
}


int  qio_node_index( const int x[])
{
  
  // returns local index of global coordinates


  #ifdef DEBUG_NodeIndex
  printf("UID: %i, called qio_node_index with x: [%i, %i, %i, %i] calculating...\n",UniqueID(),x[0],x[1],x[2],x[3]);
  #endif // DEBUG_NodeIndex

  
  int xLoc[4];
  for(int ii(0); ii < 4; ++ii)
    xLoc[ii] = x[ii] - GJP.NodeCoor(ii)*GJP.NodeSites(ii);
  
  // now convert local coordinate into local index

  
  int tmp(xLoc[0]);
  int vol(GJP.NodeSites(0));

  for( int ii(1); ii < 4; ++ii)
    {
      tmp += vol * xLoc[ii];
      vol *= GJP.NodeSites(ii);
    }

  if( tmp >= GJP.VolNodeSites() )
    { 
      printf("ERROR QIO: qio_node_index got index >= local volume");
      exit(-1);
    }


  #ifdef DEBUG_NodeIndex
  printf("UID: %i, called qio_node_index with x: [%i, %i, %i, %i] returned index %i\n",UniqueID(),x[0],x[1],x[2],x[3],tmp);
  #endif // DEBUG_NodeIndex

  return tmp;

}


void qio_get_coords( int x[], int node, int index)
{

  #ifdef DEBUG_GetCoords
  printf("UID: %i, called qio_get_coords with node: %i, index %i; calc...\n", UniqueID(), node, index);
  #endif // DEBUG_GetCoords

  // returns x[]: global(?) coordinates on this node

  //if( node != UniqueID() )
  if( node != QMP_get_node_number() )
    { 
      printf("ERROR QIO: node number mismatch\n");
 
      exit(-1); 
    }

  x[0] = GJP.XnodeCoor()*GJP.NodeSites(0);
  x[1] = GJP.YnodeCoor()*GJP.NodeSites(1);
  x[2] = GJP.ZnodeCoor()*GJP.NodeSites(2);
  x[3] = GJP.TnodeCoor()*GJP.NodeSites(3);

  int xLoc[4]={0,0,0,0};

  for( int ii(0); ii < index; ++ii)
    { 
      if( ++xLoc[0] == GJP.NodeSites(0) ) 
	{ 
	 xLoc[0]=0; 
	 if( ++xLoc[1] == GJP.NodeSites(1) )
	   {
	     xLoc[1]=0;
	     if( ++xLoc[2] == GJP.NodeSites(2) )
	       {
		 xLoc[2]=0;
		 if( ++xLoc[3] == GJP.NodeSites(3) )
		   {
		     printf("ERROR in QIO: with index/global-coor-conver\n");
		     exit(-1);
		   }
	       }
	   }
	}
    }
      
	
  for(int ii(0); ii < 4; ++ii)
    x[ii] += xLoc[ii];
          

  #ifdef DEBUG_GetCoords
  printf("UID: %i, called qio_get_coords with node: %i, index %i; returns x: [%i, %i, %i, %i]\n", UniqueID(), node, index, x[0], x[1], x[2], x[3]);
  #endif // DEBUG_GetCoords


}


int qio_num_sites(int node)
{
  // number of sites on node
  // this is the same on all nodes !!!

  return GJP.VolNodeSites();
}


// now start the qio_init-class functions


//void qio_init::qio_setLayout( QIO_Layout *layout)
void qio_init::qio_setLayout()
{
  static int lattice_size[QIO_RW_DIMENSION];

  
  lattice_size[0] = GJP.Xnodes()*GJP.XnodeSites();
  lattice_size[1] = GJP.Ynodes()*GJP.YnodeSites();
  lattice_size[2] = GJP.Znodes()*GJP.ZnodeSites();
  lattice_size[3] = GJP.Tnodes()*GJP.TnodeSites();
  

  // put the helper outside class...

  layout.node_number     = qio_node_number;
  layout.node_index      = qio_node_index;
  layout.get_coords      = qio_get_coords;
  layout.num_sites       = qio_num_sites;
  layout.latsize         = lattice_size;         // lattice size x,y,z,t
  layout.latdim          = QIO_RW_DIMENSION;     // lattice dimension for QIO
  layout.volume          = GJP.VolSites();       // lattice volume
  layout.sites_on_node   = GJP.VolNodeSites();   // local volume
  layout.this_node       = QMP_get_node_number();// number of this node
  layout.number_of_nodes = NumNodes();           // total number of nodes

}


 
CPS_END_NAMESPACE
#endif