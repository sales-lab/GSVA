# nvcc compiler
ifeq ($(CUDACXX),)
  ifneq ($(CUDA_HOME),)
    NVCC := $(CUDA_HOME)/bin/nvcc
  else ifneq ($(CUDA_PATH),)
    NVCC := $(CUDA_PATH)/bin/nvcc
  else
    NVCC := $(shell command -v nvcc 2>/dev/null)
  endif
  ifeq ($(NVCC),)
    NVCC := $(shell ls /usr/local/cuda/bin/nvcc 2>/dev/null | head -n1)
  endif
  ifeq ($(NVCC),)
    NVCC := $(shell ls /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -n1)
  endif
  ifeq ($(NVCC),)
    ifneq ($(CONDA_PREFIX),)
      NVCC := $(CONDA_PREFIX)/bin/nvcc
    endif
  endif
else
  NVCC := $(CUDACXX)
endif

ifeq ($(wildcard $(NVCC)),)
  $(error CUDA compiler (nvcc) not found at '$(NVCC)'. Set CUDA_HOME, CUDA_PATH, or CUDACXX, or ensure nvcc is in your PATH.)
endif

CUDA_HOME    := $(dir $(dir $(NVCC)))
CUDA_VERSION := $(shell $(NVCC) --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')

# GPU architecture
# Override by setting the GPU_ARCH variable, e.g. GPU_ARCH=86
NVIDIA_SMI := $(shell command -v nvidia-smi 2>/dev/null)

ifeq ($(origin GPU_ARCH), user)
  GPU_COMPUTE_CAPABILITY := $(GPU_ARCH)
  $(info [CUDA] GPU arch: sm_$(GPU_ARCH) (user override))
else ifneq ($(NVIDIA_SMI),)
  GPU_COMPUTE_CAPABILITY := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/\.//g' | sort -n | head -n1)
  GPU_COMPUTE_CAPABILITY := $(strip $(GPU_COMPUTE_CAPABILITY))
  ifeq ($(GPU_COMPUTE_CAPABILITY),)
    $(error nvidia-smi is available but returned no GPUs. Set GPU_ARCH explicitly, e.g. GPU_ARCH=86)
  else
    $(info [CUDA] GPU arch: sm_$(GPU_COMPUTE_CAPABILITY) (auto-detected))
  endif
else
  $(error nvidia-smi not found. Install the NVIDIA driver or set GPU_ARCH explicitly, e.g. GPU_ARCH=86)
endif

GPU_ARCH_FLAGS := --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]

$(info [CUDA] nvcc $(CUDA_VERSION) from $(NVCC))

SHLIB         = GSVA.so
SHLIB_LD      = $(NVCC)
FP32_FLAG     = -DUSE_FP32
NVCC_FLAGS    = -g -lineinfo

CU_OBJECTS = $(patsubst cuda/%.cu,cuda/%.o,$(wildcard cuda/*.cu))
C_OBJECTS  = $(patsubst %.c,%.o,$(wildcard *.c))
OBJECTS    = $(CU_OBJECTS) $(C_OBJECTS)

all: $(SHLIB)

cuda/%.o: cuda/%.cu
	$(NVCC) -std=c++17 $(FP32_FLAG) $(GPU_ARCH_FLAGS) $(NVCC_FLAGS) -c $< -o $@ $(ALL_CPPFLAGS) -I. -Xcompiler "$(CPICFLAGS) $(CFLAGS)"

%.o: %.c
	$(CC) $(ALL_CPPFLAGS) $(ALL_CFLAGS) -c $< -o $@

$(SHLIB): $(OBJECTS)
	$(SHLIB_LD) $(SHLIB_LDFLAGS) -o $@ $(OBJECTS) $(LIBR0) $(LDFLAGS) $(PKG_LIBS) $(LIBS)

clean:
	rm -f *.o cuda/*.o $(SHLIB)

.PHONY: all clean
