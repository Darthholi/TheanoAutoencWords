REM configuration of paths
set SCISOFT=D:\Apps
REM %~dp0

REM add tdm gcc stuff
set PATH=%SCISOFT%\TDM-GCC-64\bin;%SCISOFT%\TDM-GCC-64\x86_64-w64-mingw32\bin;%PATH%

REM return a shell
cmd.exe /k