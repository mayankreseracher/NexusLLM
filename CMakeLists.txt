cmake_minimum_required(VERSION 3.20)
project(nexus_engine C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Architecture & OS Specific Tuning
if(MSVC)
    add_compile_options(/O2 /W3 /arch:AVX2)
elseif(APPLE)
    # macOS (Apple Silicon M-Series & Intel)
    add_compile_options(-O3 -Wall -Wextra -pthread)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        message(STATUS "[Nexus Engine] Detected Apple Silicon (ARM64) - Enabling UMA Zero-Copy & NEON Acceleration")
        add_compile_options(-march=armv8.5-a)
    else()
        message(STATUS "[Nexus Engine] Detected macOS x86_64 - Enabling AVX2")
        add_compile_options(-mavx2 -mfma)
    endif()
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")
elseif(UNIX)
    # Linux (Ubuntu, Debian, RHEL, CentOS, Rocky, Fedora, Arch, Alpine)
    add_compile_options(-O3 -Wall -Wextra -fPIC -pthread)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
        message(STATUS "[Nexus Engine] Detected Linux x86_64 - Enabling AVX2 & FMA")
        add_compile_options(-mavx2 -mfma)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        message(STATUS "[Nexus Engine] Detected Linux aarch64 - Enabling ARM NEON")
        add_compile_options(-march=armv8-a+simd)
    endif()
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

# Core dynamic library
add_library(nexus_engine SHARED
    src/memory_tier.cpp
    src/model_loader.cpp
    src/paged_cache.cpp
    src/quantization.cpp
    src/moe_router.cpp
    src/transformer_engine.cpp
    src/nexus_engine.cpp
    src/nexus_abi.cpp
)

target_include_directories(nexus_engine PUBLIC include)

if(UNIX AND NOT APPLE)
    target_link_libraries(nexus_engine PRIVATE dl pthread)
endif()

# Standalone benchmark executable
add_executable(nexus_bench src/main_bench.cpp)
target_link_libraries(nexus_bench PRIVATE nexus_engine)

if(APPLE)
    set_target_properties(nexus_bench PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
elseif(UNIX)
    set_target_properties(nexus_bench PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()