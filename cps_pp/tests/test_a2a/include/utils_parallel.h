#pragma once

CPS_START_NAMESPACE

void testMPIAllReduceQueued(){
  std::cout << "Starting testMPIAllReduceQueued" << std::endl;
  MPIallReduceQueued &q = MPIallReduceQueued::globalInstance();
  q.setVerbose(true);
  
  int nodes = GJP.TotalNodes();
  for(int i=0;i<3;i++){
    std::vector<double> v(10),w(10);
    for(int i=0;i<10;i++){ v[i] = 1.0; w[i] = 2.0; }

    auto h = q.enqueue(v.data(),10,MPI_DOUBLE);
    std::cout << "Main thread waiting" << std::endl;
    h->wait();
    std::cout << "Main thread task has completed" << std::endl;
    
    for(int i=0;i<10;i++) assert( v[i] == 1.0 * nodes );


    auto h2 = q.enqueue(v.data(),10,MPI_DOUBLE);
    auto h3 = q.enqueue(w.data(),10,MPI_DOUBLE);

    std::cout << "Main thread waiting" << std::endl;
    //should be ok to wait out of order
    h3->wait();
    h2->wait();

    std::cout << "Main thread task has completed" << std::endl;
    
    for(int i=0;i<10;i++) assert( v[i] == 1.0 * nodes * nodes );
    for(int i=0;i<10;i++) assert( w[i] == 2.0 * nodes );
    
  }

  q.setVerbose(false);    
  std::cout << "testMPIAllReduceQueued passed" << std::endl;
}


CPS_END_NAMESPACE
