# Makefile for SPDP compression filter for HDF5
# http://cs.txstate.edu/~burtscher/research/SPDP/

# Change this HDF5_INSTALL variable to point
# to your base hdf5 installation directory.
HDF5_INSTALL = /path/to/hdf5

# If you have parallel hdf5 with MPI, change
# this MPI_INSTALL variable to point to your
# base MPI installation directory. Otherwise
# leave it blank as is.
MPI_INSTALL =

CFLAGS = -fPIC -O3
CC = gcc
MAJOR = 1
MINOR = 0
NAME1 = H5Zspdp
VERSION = $(MAJOR).$(MINOR)
ifdef MPI_INSTALL
INCLUDES = -I$(MPI_INSTALL)/include -I$(HDF5_INSTALL)/include
else
INCLUDES = -I$(HDF5_INSTALL)/include
endif

all: lib$(NAME1).so.$(VERSION)

$(NAME1).o: $(NAME1).c
	$(CC) $(CFLAGS) $(INCLUDES) -c $(NAME1).c

lib$(NAME1).so.$(VERSION): $(NAME1).o
	$(CC) -shared -Wl,-soname,lib$(NAME1).so.$(MAJOR) $^ -o $@

clean:
	$(RM) *.o

distclean:
	$(RM) *.o *.so*
