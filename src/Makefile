CUDA_HOME = /usr/local/cuda
SHLIB_LD = $(CUDA_HOME)/bin/nvcc
SHLIB = GSVA.so

FP32_FLAG = -DUSE_FP32
FP32_ARCH = -arch=sm_120
NVCC_FLAGS = -g -lineinfo

CU_OBJECTS = $(patsubst cuda/%.cu,cuda/%.o,$(wildcard cuda/*.cu))
C_OBJECTS  = $(patsubst %.c,%.o,$(wildcard *.c))
OBJECTS    = $(CU_OBJECTS) $(C_OBJECTS)

PKG_LIBS = -cudart shared

all: $(SHLIB)

cuda/%.o: cuda/%.cu
	$(CUDA_HOME)/bin/nvcc -std=c++17 $(FP32_FLAG) $(FP32_ARCH) $(NVCC_FLAGS) -c $< -o $@ $(ALL_CPPFLAGS) -I. -Xcompiler "$(CPICFLAGS) $(CFLAGS)"

%.o: %.c
	$(CC) $(ALL_CPPFLAGS) $(ALL_CFLAGS) -c $< -o $@

$(SHLIB): $(OBJECTS)
	$(SHLIB_LD) $(SHLIB_LDFLAGS) -o $@ $(OBJECTS) $(LIBR0) $(LDFLAGS) $(PKG_LIBS) $(LIBS)

clean:
	rm -f *.o cuda/*.o $(SHLIB)

.PHONY: all clean
