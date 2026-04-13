@echo off
netstat -ano | findstr :8501 >nul 2>&1
if %errorlevel%==0 goto open_browser
start /min "" streamlit run dashboard.py
timeout /t 3 /nobreak >nul
:open_browser
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --app=http://localhost:8501 --window-size=1400,900
