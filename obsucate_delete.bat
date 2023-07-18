@echo off
setlocal

set "startDir=%cd%"

for /R "%startDir%" %%F in (*.pyc) do (
    echo Deleting: %%F
    del /F "%%F"
)

endlocal
pause
