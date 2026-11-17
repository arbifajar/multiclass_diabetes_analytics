@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher 'py' tidak ditemukan. Install Python terlebih dahulu.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Membuat virtual environment...
  py -m venv .venv
  if errorlevel 1 goto :error
)
echo Memasang paket yang diperlukan...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo Menjalankan Dashboard Streamlit...
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto :error
goto :end
:error
echo.
echo Terjadi masalah. Periksa pesan error di atas.
pause
:end
