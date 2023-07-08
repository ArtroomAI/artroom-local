@echo off
setlocal

set "pythonPath=C:\Users\artad\artroom\artroom_backend\python.exe"
set "startDir=C:\Users\artad\Documents\GitHub\ArtroomAI\artroom-frontend"

for %%F in ("%startDir%\*.py") do (
    echo Processing: %%F
    "%pythonPath%" -OO -m py_compile "%%F"
    for %%G in ("%%~dpF__pycache__\%%~nF.cpython-310.opt-2.pyc") do (
        move "%%G" "%%~dpnF.pyc"
    )
)

set "pythonPath=C:\Users\artad\artroom\artroom_backend\python.exe"
set "startDir=C:\Users\artad\Documents\GitHub\ArtroomAI\artroom-frontend\artroom_helpers"

for %%F in ("%startDir%\*.py") do (
    echo Processing: %%F
    "%pythonPath%" -OO -m py_compile "%%F"
    for %%G in ("%%~dpF__pycache__\%%~nF.cpython-310.opt-2.pyc") do (
        move "%%G" "%%~dpnF.pyc"
    )
)


endlocal
pause
