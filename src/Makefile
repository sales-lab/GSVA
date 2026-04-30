-include Makeconf

SHLIB = GSVA.so
OBJECTS = $(CU_OBJS) $(C_OBJS)

all: $(SHLIB)

$(SHLIB): $(OBJECTS)
	$(SHLIB_LINK) -o $@ $(OBJECTS) $(ALL_LIBS)

cuda/%.o: cuda/%.cu
	$(NVCC) -std=c++17 $(FP32_FLAG) $(GPU_ARCH_FLAGS) $(NVCC_FLAGS) -c $< -o $@ $(HAVE_CUDA_DEF) $(ALL_CPPFLAGS) -I. -Xcompiler "$(CPICFLAGS) $(CFLAGS)"

clean:
	rm -f *.o cuda/*.o $(SHLIB)

.PHONY: all clean
