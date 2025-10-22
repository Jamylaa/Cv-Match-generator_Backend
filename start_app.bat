@echo off
echo Starting CV Matching Application...
echo.
echo Starting MongoDB...
start cmd /k "python start_mongodb.py"
echo.
echo Starting Backend API...
start cmd /k "python start_server.py"
echo.
echo Starting Frontend...
cd cv-matching-frontend
start cmd /k "npm start"
echo.
echo CV Matching Application is running!
echo Backend API: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Press any key to stop all services...
pause > nul
echo Stopping services...
taskkill /f /im cmd.exe /fi "windowtitle eq C:\Windows\system32\cmd.exe*"
echo Application stopped.
pause



