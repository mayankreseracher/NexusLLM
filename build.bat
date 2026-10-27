@echo off
echo ===================================================
echo Building NexusLLM High-Throughput Inference Engine
echo ===================================================

where zig >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [Build] Found Zig compiler. Compiling native DLL and benchmark...
    zig c++ -O3 -shared -std=c++20 -Iinclude -DNEXUS_BUILD_DLL src/memory_tier.cpp src/model_loader.cpp src/paged_cache.cpp src/quantization.cpp src/moe_router.cpp src/transformer_engine.cpp src/forward_pass.cpp src/nexus_engine.cpp src/nexus_abi.cpp src/nexus_backend.cpp -o nexus_engine.dll
    zig c++ -O3 -std=c++20 -Iinclude src/main_bench.cpp -L. -lnexus_engine -o nexus_bench.exe
    echo [Build] Success! Generated nexus_engine.dll and nexus_bench.exe
    exit /b 0
)

where cl >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [Build] Found MSVC cl.exe. Compiling...
    cl /O2 /LD /std:c++20 /Iinclude /DNEXUS_BUILD_DLL src\*.cpp /Fe:nexus_engine.dll
    cl /O2 /std:c++20 /Iinclude src\main_bench.cpp nexus_engine.lib /Fe:nexus_bench.exe
    echo [Build] Success!
    exit /b 0
)

echo [Build Error] No supported C++ compiler found (Install Zig, Clang, or Visual Studio C++).
exit /b 1
