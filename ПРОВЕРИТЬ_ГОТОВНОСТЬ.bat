@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto PATH_ERROR
if not exist ".venv\Scripts\python.exe" goto MISSING_VENV
if not exist "final_preflight.py" goto MISSING_SCRIPT
if not exist "data\Benign_Only.zip" goto MISSING_DATA
if not exist "data\PCAPs_1.zip" goto MISSING_PCAP1
if not exist "data\PCAPs_2.zip" goto MISSING_PCAP2
if not exist "data\PCAPs_3.zip" goto MISSING_PCAP3

echo Starting final readiness check...
echo Project folder: %CD%
".venv\Scripts\python.exe" "final_preflight.py" --data-dir "data" --out-dir "final_preflight"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" goto READY
echo PREFLIGHT BLOCKED OR FAILED. Open final_preflight\PREFLIGHT_REPORT.txt
goto FINISH

:READY
echo PREFLIGHT FINISHED. Open final_preflight\PREFLIGHT_REPORT.txt
goto FINISH

:PATH_ERROR
echo ERROR: Could not enter the folder containing this BAT file.
set "RC=1"
goto FINISH

:MISSING_VENV
echo ERROR: .venv\Scripts\python.exe was not found.
set "RC=1"
goto FINISH

:MISSING_SCRIPT
echo ERROR: final_preflight.py was not found in this folder.
set "RC=1"
goto FINISH

:MISSING_DATA
echo ERROR: data\Benign_Only.zip was not found.
set "RC=1"
goto FINISH

:MISSING_PCAP1
echo ERROR: data\PCAPs_1.zip was not found.
set "RC=1"
goto FINISH

:MISSING_PCAP2
echo ERROR: data\PCAPs_2.zip was not found.
set "RC=1"
goto FINISH

:MISSING_PCAP3
echo ERROR: data\PCAPs_3.zip was not found.
set "RC=1"
goto FINISH

:FINISH
echo.
echo Do not delete the final_preflight folder; send it for review.
pause
exit /b %RC%
