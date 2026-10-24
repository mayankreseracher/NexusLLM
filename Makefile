CXX ?= g++
CXXFLAGS ?= -O3 -Wall -Wextra -std=c++20 -Iinclude -fPIC -mavx2 -mfma -pthread

SRC = src/memory_tier.cpp src/model_loader.cpp src/paged_cache.cpp src/quantization.cpp src/moe_router.cpp src/nexus_engine.cpp src/nexus_abi.cpp
OBJ = $(SRC:.cpp=.o)

all: libnexus_engine.so nexus_bench

libnexus_engine.so: $(SRC)
	$(CXX) $(CXXFLAGS) -shared $(SRC) -o $@

nexus_bench: src/main_bench.cpp libnexus_engine.so
	$(CXX) $(CXXFLAGS) src/main_bench.cpp -L. -lnexus_engine -o $@

clean:
	rm -f $(OBJ) libnexus_engine.so nexus_bench
