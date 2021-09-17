# Patched from the Nvidia makefiles and common.mk
# Stripped a lot of "rubbish" and force source, object and target to
# be simply in the current dir.
#
# Add source files here
TARGETS    := matrixprod-cublas matrixprod-cublas-um
HELPERS    := matrix_blas_helpers vector_blas_helpers
HELPERS_UM := matrix_blas_helpers-um vector_blas_helpers-um
MMPROD     := matrixprod-cublas ${HELPERS}
MMPROD_UM  := matrixprod-cublas-um ${HELPERS_UM}
CUFILES	   := $(TARGETS:=.cu) $(HELPERS:=.cu) $(HELPERS_UM:=.cu)

DEBUG := 0
USEGLLIB := 0
USECUDPP := 0
USECUBLAS := 1
USECUFFT := 0

# Use @ here to suppress all output :
# VERBOSE := @
VERBOSE :=

all : $(TARGETS)

include ../Common/common.mk  # gory stuff with the compiler settings and rules

GPUARCH := -arch=$(MINCAP)

# Needs a separate line as we need to link cublas and the extra helpers objectfile:
matrixprod-cublas: $(MMPROD:=.o)
	$(NVLINK) $(MMPROD:=.o) $(LIB) -lcublas $(GPUARCH) -o $@

# Needs a separate line as we need to link cublas and the extra helpers objectfile:
matrixprod-cublas-um: $(MMPROD_UM:=.o)
	$(NVLINK) $(MMPROD_UM:=.o) $(LIB) -lcublas $(GPUARCH) -o $@

$(OBJS) : Makefile

clean : tidy
	$(VERBOSE)rm -f $(OBJS) gmon.out
	$(VERBOSE)rm -f $(TARGETS) *.o $(TARGETS:=.linkinfo) $(HELPERS:=.linkinfo) $(HELPERS_UM:=.linkinfo)

allclean: clean
