@echo off
cd /d "c:\Users\jrahu\Downloads\Bridging Vision and Language Generative AI-Based Image Captioning for Enhanced Accessibility and Understanding\imagecaption"
echo Starting app.py on port 8501...
start "App1 - Port 8501" cmd /k streamlit run app.py --server.port 8501
timeout /t 2
echo Starting app2.py on port 8502...
start "App2 - Port 8502" cmd /k streamlit run app2.py --server.port 8502
echo.
echo Both apps started!
echo app.py: http://localhost:8501
echo app2.py: http://localhost:8502
pause
