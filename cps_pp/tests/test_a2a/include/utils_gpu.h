#pragma once

CPS_START_NAMESPACE

void testhostDeviceMirroredContainer(){
  std::cout << "Starting testhostDeviceMirroredContainer" << std::endl;
  size_t bs = 100*sizeof(double);
  hostDeviceMirroredContainer<double> buf(100);
  double* tmp_gpu = (double*)device_alloc_check(bs);
  double* tmp_host = (double*)malloc(bs);
  device_memset(tmp_gpu,0,bs);
  memset(tmp_host,0,bs);
  
  //Write some data on host
  {
    double* p = buf.getHostWritePtr();
    for(int i=0;i<100;i++) *p++ = i;
  }
  {
    double const* pr = buf.getDeviceReadPtr();
    using namespace Grid;
    //Copy it to temp buffer on device
    accelerator_for(i, 100, 1, {
	tmp_gpu[i] = pr[i];
      });    	
    //Overwrite with new values
    double* pw = buf.getDeviceWritePtr();
    accelerator_for(i, 100, 1, {
	pw[i] = 100 + i;
      });    	
  }
  {
    //Host check
    bool fail = false;
    copy_device_to_host(tmp_host,tmp_gpu,bs);
    for(int i=0;i<100;i++)
      if(tmp_host[i] != i){ std::cout << "FAIL host check 1 got "<< tmp_host[i] << " expect " << i << std::endl; fail = true; }
    
    double const* p = buf.getHostReadPtr();
    for(int i=0;i<100;i++)
      if(p[i] != 100+i){ std::cout << "FAIL host check 2 got " << p[i] << " expect " << 100+i << std::endl; fail = true; }

    if(fail) ERR.General("","testhostDeviceMirroredContainer","Blocking copy test failed");
  }
  
  //Check asynchronous copies
  buf.reset();
  {
    double* p = buf.getHostWritePtr();
    for(int i=0;i<100;i++) *p++ = 3*i;
  } 
  buf.asyncHostDeviceSync();
  device_synchronize_all();
  if(!buf.deviceInSync()) ERR.General("","testhostDeviceMirroredContainer","Expect device in sync");
  
  {
    double const* pr = buf.getDeviceReadPtr();
    using namespace Grid;
    //Copy it to temp buffer on device
    accelerator_for(i, 100, 1, {
	tmp_gpu[i] = pr[i];
      });    	
    //Overwrite with new values
    double* pw = buf.getDeviceWritePtr();
    accelerator_for(i, 100, 1, {
	pw[i] = 100 + 3*i;
      });    	
  }
  {
    //Host check
    bool fail = false;
    copy_device_to_host(tmp_host,tmp_gpu,bs);
    for(int i=0;i<100;i++)
      if(tmp_host[i] != 3*i){ std::cout << "FAIL async host check 1 got "<< tmp_host[i] << " expect " << 3*i << std::endl; fail = true; }

    buf.asyncHostDeviceSync();
    device_synchronize_all();
    if(!buf.hostInSync()) ERR.General("","testhostDeviceMirroredContainer","Expect host in sync");
    
    double const* p = buf.getHostReadPtr();
    for(int i=0;i<100;i++)
      if(p[i] != 100+3*i){ std::cout << "FAIL async host check 2 got " << p[i] << " expect " << 100+3*i << std::endl; fail = true; }

    if(fail) ERR.General("","testhostDeviceMirroredContainer","Async copy test failed");
  }
  
  std::cout << "testhostDeviceMirroredContainer passed" << std::endl;
}



CPS_END_NAMESPACE
