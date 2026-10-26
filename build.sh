#!/usr/bin/env bash
set -e

# ==============================================================================
# NexusLLM Cross-Platform Build Script
# Native capabilities for macOS (Apple Silicon & Intel) and all Linux distros
# ==============================================================================

BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
NC="\033[0m"

echo -e "${BOLD}${BLUE}================================================================${NC}"
echo -e "${BOLD}${BLUE}   NexusLLM Engine: Cross-Platform Native Compiler Pipeline    ${NC}"
echo -e "${BOLD}${BLUE}   Apple Silicon UMA Zero-Copy + Linux HugePages + SIMD Vector ${NC}"
echo -e "${BOLD}${BLUE}================================================================${NC}"

OS="$(uname -s)"
ARCH="$(uname -m)"

echo -e "${CYAN}[Hardware Detection]${NC} Operating System: ${BOLD}${OS}${NC} | Architecture: ${BOLD}${ARCH}${NC}"

# Detect distribution if on Linux
DISTRO="Generic Linux"
if [ "$OS" = "Darwin" ]; then
    DISTRO="macOS Darwin"
    if [ "$ARCH" = "arm64" ]; then
        echo -e "${GREEN}[Platform] Detected Apple Silicon (M1/M2/M3/M4) Unified Memory Architecture (UMA).${NC}"
        echo -e "${GREEN}[Platform] Zero-copy RAM-to-compute sharing enabled. Zero PCIe transfer stall.${NC}"
    else
        echo -e "${YELLOW}[Platform] Detected Intel macOS x86_64.${NC}"
    fi
elif [ "$OS" = "Linux" ]; then
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO="${NAME:-Linux}"
    fi
    echo -e "${GREEN}[Platform] Linux Distribution: ${BOLD}${DISTRO}${NC}"

    # Package manager advice if compiler is missing
    if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
        echo -e "${YELLOW}[Notice] C++20 compiler not found. Install instructions for ${DISTRO}:${NC}"
        if command -v apt-get >/dev/null 2>&1; then
            echo -e "  sudo apt update && sudo apt install -y build-essential cmake"
        elif command -v dnf >/dev/null 2>&1; then
            echo -e "  sudo dnf groupinstall -y 'Development Tools' && sudo dnf install -y cmake gcc-c++"
        elif command -v pacman >/dev/null 2>&1; then
            echo -e "  sudo pacman -S --needed base-devel cmake"
        elif command -v apk >/dev/null 2>&1; then
            echo -e "  apk add --no-cache g++ cmake make linux-headers"
        elif command -v zypper >/dev/null 2>&1; then
            echo -e "  sudo zypper install -t pattern devel_basis && sudo zypper install cmake"
        fi
        exit 1
    fi
fi

# Select compiler
if command -v g++ >/dev/null 2>&1; then
    CXX="g++"
elif command -v clang++ >/dev/null 2>&1; then
    CXX="clang++"
else
    echo -e "\033[0;31m[Error] No C++ compiler (g++ or clang++) found in PATH.\033[0m"
    exit 1
fi

echo -e "${CYAN}[Compiler]${NC} Using: ${BOLD}${CXX}${NC}"

# Determine compiler & linker flags based on platform and raw performance capabilities
CXXFLAGS="-O3 -Wall -Wextra -std=c++20 -Iinclude -fPIC -pthread"

if [ "$OS" = "Darwin" ]; then
    SHARED_OUT="libnexus_engine.dylib"
    LDFLAGS_SHARED="-dynamiclib"
    LDFLAGS_LIBS=""
    LDFLAGS_BENCH="-L. -lnexus_engine -Wl,-rpath,@loader_path"
    if [ "$ARCH" = "arm64" ]; then
        SIMD_FLAGS="-march=armv8.5-a"
    else
        SIMD_FLAGS="-mavx2 -mfma"
    fi
else
    # Linux
    SHARED_OUT="libnexus_engine.so"
    LDFLAGS_SHARED="-shared"
    LDFLAGS_LIBS="-ldl -pthread"
    LDFLAGS_BENCH="-L. -lnexus_engine -Wl,-rpath,\$ORIGIN"
    if [ "$ARCH" = "x86_64" ]; then
        SIMD_FLAGS="-mavx2 -mfma"
    elif [[ "$ARCH" =~ ^(aarch64|arm64) ]]; then
        SIMD_FLAGS="-march=armv8-a+simd"
    else
        SIMD_FLAGS=""
    fi
fi

echo -e "${CYAN}[SIMD Config]${NC} Vector flags: ${BOLD}${SIMD_FLAGS:-None (Scalar Fallback)}${NC}"

# Source files
SOURCES="src/memory_tier.cpp src/model_loader.cpp src/paged_cache.cpp src/quantization.cpp src/moe_router.cpp src/transformer_engine.cpp src/nexus_engine.cpp src/nexus_abi.cpp src/nexus_backend.cpp"

echo -e "\n${BOLD}[1/2] Compiling shared library: ${SHARED_OUT}...${NC}"
$CXX $CXXFLAGS $SIMD_FLAGS $LDFLAGS_SHARED $SOURCES -o "$SHARED_OUT" $LDFLAGS_LIBS
echo -e "${GREEN}✓ Successfully built ${SHARED_OUT}${NC}"

echo -e "\n${BOLD}[2/2] Compiling benchmark executable: nexus_bench...${NC}"
$CXX $CXXFLAGS $SIMD_FLAGS src/main_bench.cpp $LDFLAGS_BENCH -o nexus_bench $LDFLAGS_LIBS
echo -e "${GREEN}✓ Successfully built nexus_bench${NC}"

echo -e "\n${BOLD}${GREEN}================================================================${NC}"
echo -e "${BOLD}${GREEN}   Build Completed Successfully!                                ${NC}"
echo -e "${BOLD}${GREEN}================================================================${NC}"
echo -e "Shared Library : ${BOLD}${SHARED_OUT}${NC}"
echo -e "CLI Benchmark  : ${BOLD}./nexus_bench${NC}"
echo -e "Python Module  : ${BOLD}python -m pip install -e .${NC} or import directly."
echo -e ""
echo -e "To run the benchmark now: ${BOLD}./nexus_bench${NC}"
