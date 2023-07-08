@echo off
setlocal

set "pythonPath=C:/Users/artad/artroom/artroom_backend/python.exe"
set "startDir=C:/Users/artad/Documents/GitHub/ArtroomAI/artroom-frontend/"
set "outputFile=%cd%/compiled_files.txt"

for %%F in ("%startDir%/*.py") do (
    echo Processing: %%F
    "%pythonPath%" -OO -m py_compile "%%F"
    move "%%~dpF__pycache__\%%~nF.cpython-310.opt-2.pyc" "%%~dpF%%~nF.cpython-310.opt-2.pyc"
    rmdir /S /Q "%%~dpF__pycache__"
    echo "%%~dpF%%~nF.cpython-310.opt-2.pyc" >> "%outputFile%"
)

endlocal
pause
