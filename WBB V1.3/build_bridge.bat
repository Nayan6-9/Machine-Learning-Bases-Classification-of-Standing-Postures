@echo off
setlocal
REM ============================================================
REM  Build WiiBoardBridge.exe WITHOUT Visual Studio.
REM  Needs only: WiiBoardBridge.cs + WiimoteLib.dll in this folder.
REM  Get WiimoteLib.dll from your WiiBalanceWalker folder.
REM ============================================================

REM --- locate the .NET Framework C# compiler (ships with Windows) ---
set "CSC=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if not exist "%CSC%" set "CSC=%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe"
if not exist "%CSC%" (
  echo [X] csc.exe not found. Install .NET Framework 4.x, or run this from a
  echo     "Developer Command Prompt for VS".
  pause & exit /b 1
)

if not exist "WiimoteLib.dll" (
  echo [X] WiimoteLib.dll not found in this folder.
  echo     Copy it here from your WiiBalanceWalker folder, then run again.
  pause & exit /b 1
)

echo Using compiler: %CSC%
"%CSC%" /nologo /platform:x86 /reference:WiimoteLib.dll /out:WiiBoardBridge.exe WiiBoardBridge.cs

if exist "WiiBoardBridge.exe" (
  echo.
  echo [OK] Built WiiBoardBridge.exe
  echo      Double-click it, then press the board's FRONT button to connect.
) else (
  echo.
  echo [X] Build failed. If the error mentions platform/bitness, edit this file
  echo     and remove "/platform:x86", then run again.
)
pause
