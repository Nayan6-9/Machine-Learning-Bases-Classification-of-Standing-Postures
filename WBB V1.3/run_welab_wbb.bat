@echo off
REM ============================================================
REM  WELAB - Wii Balance Board Posture app  (double-click to run)
REM  First time / no board: pick "Demo (no board)" then Connect.
REM ============================================================
cd /d "%~dp0"
python wbb_gui.py
if errorlevel 1 (
  echo.
  echo Could not start. Is Python installed and on PATH?
  echo Try:  py wbb_gui.py
  pause
)
