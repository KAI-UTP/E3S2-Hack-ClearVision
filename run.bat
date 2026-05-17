@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting Blind Assist server...
echo Open http://localhost:8000 on your phone (same WiFi network)
echo.
python app.py
pause
