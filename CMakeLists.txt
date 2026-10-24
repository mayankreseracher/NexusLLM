cmake_minimum_required(VERSION 3.20)
project(nexus_engine C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

if(MSVC)
    add_compile_options(/O2 /W3 /arch:AVX2)
else()
    add_compile_options(-O3 -Wall -Wextra -mavx2 -mfma -pthread)
endif()

# Core dynamic library
add_library(nexus_engine SHARED
    src/memory_tier.cpp
    src/model_loader.cpp
    src/paged_cache.cpp
    src/quantization.cpp
    src/moe_router.cpp
    src/nexus_engine.cpp
    src/nexus_abi.cpp
)

target_include_directories(nexus_engine PUBLIC include)

# Standalone benchmark executable
add_executable(nexus_bench src/main_bench.cpp)
target_link_libraries(nexus_bench PRIVATE nexus_engine)
