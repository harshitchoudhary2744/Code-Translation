@echo off
start "Backend" powershell -NoExit -Command "cd D:\code-translator\backend; ..\venv\Scripts\python.exe -m uvicorn main:app --port 8000"
start "Frontend" powershell -NoExit -Command "cd D:\code-translator\frontend; npm run dev"
timeout /t 10 >nul
start http://localhost:5173