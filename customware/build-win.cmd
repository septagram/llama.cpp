@echo off
rem Native build for this machine: Ryzen 7 3800X (Zen 2), RTX 4060 Ti (sm_89) + RTX 2060 SUPER (sm_75).
rem clang-cl for C/C++ (Zen 2 tuned), VS2019 cl as the nvcc host compiler (CUDA 12.3 does not accept VS2026).
rem GGML_CCACHE=OFF: ggml would auto-pick the sccache on PATH, which cannot parse nvcc command lines.
rem clang 21 rejects a cast in the VS2019 (14.28) STL <functional>, so clang-cl compiles against the VS2026 (14.51)
rem STL and everything links against the 14.51 libs (newer libs accept the cl 14.28 CUDA host objects).
rem nvcc keeps the 14.28 headers through INCLUDE. configure always runs; it is cheap on an existing build dir.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=C:\Program Files\LLVM\bin;C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
set "VS26_MSVC=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231"
set "LIB=%VS26_MSVC%\lib\x64;%LIB%"
set "CW_STL=-fms-compatibility-version=19.51 -imsvc \"%VS26_MSVC%\include\""
cd /d "%~dp0.."
cmake -S . -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl ^
  -DCMAKE_C_FLAGS="/DWIN32 /D_WINDOWS /clang:-march=znver2 %CW_STL%" ^
  -DCMAKE_CXX_FLAGS="/DWIN32 /D_WINDOWS /EHsc /clang:-march=znver2 %CW_STL%" ^
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75-real;89-real" ^
  -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON ^
  -DGGML_OPENMP=ON -DGGML_CCACHE=OFF ^
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON ^
  -DBUILD_SHARED_LIBS=OFF ^
  || exit /b 1
cmake --build build --config Release --target llama-server llama-bench llama-batched-bench llama-perplexity -j 14 %*
