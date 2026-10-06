@echo off
rem Build one or more targets with the same environment as build-win.cmd, without reconfiguring.
rem usage: customware\build-target.cmd <target> [<target>...]
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=C:\Program Files\LLVM\bin;C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
set "VS26_MSVC=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231"
set "LIB=%VS26_MSVC%\lib\x64;%LIB%"
cd /d "%~dp0.."
cmake --build build --config Release -j 14 --target %*
