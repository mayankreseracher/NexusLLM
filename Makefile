UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

CXX ?= g++
CXXFLAGS ?= -O3 -Wall -Wextra -std=c++20 -Iinclude -fPIC -pthread

ifeq ($(UNAME_S),Darwin)
    TARGET_LIB = libnexus_engine.dylib
    LDFLAGS_SHARED = -dynamiclib
    ifeq ($(UNAME_M),arm64)
        SIMD_FLAGS = -march=armv8.5-a
    else
        SIMD_FLAGS = -mavx2 -mfma
    endif
    LDFLAGS_BENCH = -L. -lnexus_engine -Wl,-rpath,@loader_path
    LIBS =
else
    # Linux (Ubuntu, Debian, RHEL, CentOS, Rocky, Fedora, Arch, Alpine)
    TARGET_LIB = libnexus_engine.so
    LDFLAGS_SHARED = -shared
    ifeq ($(UNAME_M),x86_64)
        SIMD_FLAGS = -mavx2 -mfma
    else ifneq ($(filter arm% aarch64%,$(UNAME_M)),)
        SIMD_FLAGS = -march=armv8-a+simd
    else
        SIMD_FLAGS =
    endif
    LDFLAGS_BENCH = -L. -lnexus_engine -Wl,-rpath,'$$ORIGIN'
    LIBS = -ldl -pthread
endif

CXXFLAGS += $(SIMD_FLAGS)

SRC = src/memory_tier.cpp src/model_loader.cpp src/paged_cache.cpp src/quantization.cpp src/moe_router.cpp src/transformer_engine.cpp src/nexus_engine.cpp src/nexus_abi.cpp src/nexus_backend.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET_LIB) nexus_bench

$(TARGET_LIB): $(SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS_SHARED) $(SRC) -o $@ $(LIBS)

nexus_bench: src/main_bench.cpp $(TARGET_LIB)
	$(CXX) $(CXXFLAGS) src/main_bench.cpp $(LDFLAGS_BENCH) -o $@ $(LIBS)

clean:
	rm -f $(OBJ) libnexus_engine.so libnexus_engine.dylib nexus_bench
