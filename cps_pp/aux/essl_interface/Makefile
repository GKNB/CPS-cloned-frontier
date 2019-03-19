CC=/opt/ibmcmp/vacpp/bg/12.1/bin/bgxlc++_r
CCtest=/opt/bgclang/mpi/bgclang/bin/mpiclang++11
CXXFLAGS=-I/opt/ibmmath/essl/5.1/include/
LDFLAGS=-L. -lessl_interface -L/opt/ibmmath/essl/5.1/lib64 -lesslbg -L/opt/ibmcmp/xlf/bg/14.1/bglib64 -lxlopt -lxlf90_r -lxlfmath -lxl
lib:
	$(CC) $(CXXFLAGS) -c essl_interface.C -o essl_interface.o 
	ar rc libessl_interface.a essl_interface.o
	ranlib libessl_interface.a
test: essl_interface.o
	$(CC) test.C $(LDFLAGS) -o test.x
clean:
	rm essl_interface.o test.x