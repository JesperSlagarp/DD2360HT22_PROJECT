# From https://x.momo86.net/?p=29

CXX=g++
CXXFLAGS=-std=c++11 -I./include -O3 -g -Xcompiler -Wall
CXXFLAGS2=-std=c++11 -DFLOAT -I./include -O3 -g -Xcompiler -Wall

NVCC=nvcc
ARCH=sm_89
NVCCFLAGS= -I./include -arch=$(ARCH) -std=c++11 -O3 -g -Xcompiler -Wall --compiler-bindir=$(CXX)
NVCCFLAGS2= -DFLOAT -I./include -arch=$(ARCH) -std=c++11 -O3 -g -Xcompiler -Wall --compiler-bindir=$(CXX)

SRCDIR=src
SRCS=$(shell find $(SRCDIR) -name '*.cu' -o -name '*.cpp')

OBJDIR=src
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))

OBJS_float=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS_float:=$(subst .cpp,_float.o,$(OBJS_float))
OBJS_float:=$(subst .cu,_float.o,$(OBJS_float))

BIN := ./bin
TARGET=sputniPIC.out

all: dir $(BIN)/$(TARGET) float

dir: ${BIN}

${BIN}:
	mkdir -p $(BIN)

$(BIN)/$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

float: $(OBJS_float)
	$(NVCC) $(NVCCFLAGS) $+ -o $(BIN)/$(TARGET).float

$(SRCDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

$(SRCDIR)/%_float.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS2) $< -c -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(NVCC) $(CXXFLAGS) $< -c -o $@

$(OBJDIR)/%_float.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(NVCC) $(CXXFLAGS2) $< -c -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(OBJS_float)
	rm -rf $(BIN)/$(TARGET)
	rm -rf $(BIN)/$(TARGET).float
