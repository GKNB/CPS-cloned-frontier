# $Id: Makefile.users.in,v 1.6 2011-03-24 16:20:52 chulwoo Exp $
# Makefile.users.  Generated from Makefile.users.in by configure.
#   This is the makefile for all the commands
#   that are common for all testing applications.
#----------------------------------------------------------------------


SRCDIR = /Users/tblum/src/cps++/lanczos-single/cps_pp/obj/..
BUILDDIR = /Users/tblum/src/cps++/lanczos-single/cps_pp/obj
QOS = 
QOSLIB = ${QOS}/quser/gcc-lib-user///
CC = mpicc -m64
CXX = mpic++
AS  = as
LDFLAGS =  -L/Users/tblum/qmp-2.3.1//lib -lqmp -L/usr/local/qio-2.3.8/lib -lqio -llime -L/Users/tblum/xianyi-OpenBLAS-e552452 -lopenblas /usr/local/fftw/lib/libfftw3.a

me = $(notdir $(PWD))
BIN = NOARCH.x

#VPATH :=$(SRCDIR)/tests/$(me)



#
# include list for the Columbia code
#
INCLIST = -I${BUILDDIR} -I${SRCDIR}/include  -I/Users/tblum/qmp-2.3.1/include    -I/usr/local/qio-2.3.8//include -I/usr/local/fftw/include

CFLAGS= -g -O2 -Wall -msse -msse2 -msse3
CXXFLAGS= -g -O2 -Wall -msse -msse2 -msse3
ASFLAGS= 
DFLAGS +=  -DUSE_SSE -DUSE_QMP -DUSE_QIO -DUSE_FFTW
DFLAGS +=  -DHAVE_QIO_GET_HYPER_SPACETIME

#
# Libraries for the Columbia code
#
# (These are for the scalar version of the code)
#
#

.PHONY: cps clean


LIBLIST =\
  $(BUILDDIR)/cps.a \

#
#  targets
#


all:$(BIN)

.SUFFIXES:
.SUFFIXES:  .o .C .S .c

CSRC :=$(wildcard *.c)
CCSRC :=$(wildcard *.C)
SSRC :=$(wildcard *.S)

COBJ=$(CSRC:.c=.o)
CCOBJ=$(CCSRC:.C=.o)
SOBJ=$(SSRC:.S=.o)

OBJS_SRC = $(SOBJ) $(CCOBJ) $(COBJ)
OBJS := $(notdir $(OBJS_SRC))

$(BIN):  $(OBJS) $(LIBLIST)
	@echo OBJS = $(OBJS)
	$(CXX) $(OBJS) $(LIBLIST) $(LDFLAGS) -o $(BIN)

.c.o:
	$(CC) -o $@ $(CFLAGS) $(DFLAGS) -c $(INCLIST) $<
.C.o:
	$(CXX) -o $@ $(CXXFLAGS) $(DFLAGS) -c $(INCLIST) $<
.S.o:
	$(AS) -o $@ $(ASFLAGS) -c $(INCLIST) $<

cps:
	$(MAKE) -C $(BUILDDIR)

clean:
	rm -f *.dat *.o  $(BIN)
	rm -f ../regressions/*$(me).dat
	rm -f ../regressions/*$(me).checklog
