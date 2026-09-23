@echo off
rem builds the small standalone probes in customware/tools with the same clang-cl setup as build-win.cmd
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "VS26_MSVC=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231"
set "LIB=%VS26_MSVC%\lib\x64;%LIB%"
cd /d "%~dp0"
"C:\Program Files\LLVM\bin\clang-cl.exe" /O2 /arch:AVX2 /clang:-march=znver2 -fms-compatibility-version=19.51 -imsvc "%VS26_MSVC%\include" membw.c /Fe:membw.exe /link advapi32.lib
nvcc -O2 -arch=sm_89 pinprobe.cu -o pinprobe.exe -Xcompiler "/MD" 2>&1 | findstr /v "warning"
