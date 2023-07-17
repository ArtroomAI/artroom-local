@echo off

title Artroom Installer

REM Grab command line arguments and use default values if none are provided

REM Default values
set "URL="
set "ZIP_FILE="
set "DOWNLOAD_FOLDER=%UserProfile%\Downloads"
set "DESTINATION_FOLDER=%UserProfile%\artroom"
set "GPU_TYPE=Unknown"

REM Detect GPU type
wmic path win32_VideoController get name | findstr /i "AMD" >nul
if %errorlevel% equ 0 (
	set "GPU_TYPE=AMD"
) 

wmic path win32_VideoController get name | findstr /i "NVIDIA" >nul
if %errorlevel% equ 0 (
	set "GPU_TYPE=NVIDIA"
) 


if "%GPU_TYPE%"=="Unknown" (
    echo Unable to automatically detect your GPU type.
    echo Please select your GPU type manually: 
    echo [1] NVIDIA
    echo [2] AMD
    choice /c 12 /n /m "Enter your choice:"
    if errorlevel 2 (
        set "GPU_TYPE=AMD"
    ) else (
        set "GPU_TYPE=NVIDIA"
    )
)

REM Define URL and ZIP_FILE based on GPU type
if "%GPU_TYPE%"=="AMD" (
    set "URL=https://pub-060d7c8cf5e64af8b884ebb86d34de1a.r2.dev/artroom_backend_amd.zip"
    set "ZIP_FILE=artroom_backend_amd.zip"
) else if "%GPU_TYPE%"=="NVIDIA" (
    set "URL=https://pub-060d7c8cf5e64af8b884ebb86d34de1a.r2.dev/artroom_backend_nvidia.zip"
    set "ZIP_FILE=artroom_backend_nvidia.zip"
)

REM Resolve the full path of the download folder
for %%I in ("%DOWNLOAD_FOLDER%") do set "DOWNLOAD_FOLDER=%%~fI"
set "ZIP_PATH=%DOWNLOAD_FOLDER%\%ZIP_FILE%"

echo GPU Type: %GPU_TYPE%
echo Downloading Artroom for %GPU_TYPE% , please wait
echo If the window freezes, please try clicking into it and pressing enter. This is a common Windows error. 

REM Use curl to download file with progress bar
curl -o "%ZIP_PATH%" --progress-bar --retry 3 --retry-delay 5 --connect-timeout 30 "%URL%"
if %errorlevel% neq 0 (
    echo An error occurred while downloading the file. Please check your internet connection and try again.
    exit /b
)

REM Check if DESTINATION_FOLDER exists and create if not
if not exist "%DESTINATION_FOLDER%" (
    mkdir "%DESTINATION_FOLDER%"
)

echo Unzipping %ZIP_FILE%

REM Remove existing 'artroom_backend' directory if it exists
if exist "%DESTINATION_FOLDER%\artroom_backend" (
    rd /s /q "%DESTINATION_FOLDER%\artroom_backend"
)

REM Extracting zip using VBScript
echo Set objShell = CreateObject("Shell.Application") > %temp%\temp.vbs
echo Set FilesInZip=objShell.NameSpace("%ZIP_PATH%").Items >> %temp%\temp.vbs
echo objShell.NameSpace("%DESTINATION_FOLDER%").CopyHere(FilesInZip) >> %temp%\temp.vbs
cscript //nologo %temp%\temp.vbs

if %errorlevel% neq 0 (
    echo An error occurred while extracting the file. Please check your system and try again.
    exit /b
)

echo Artroom Successfully downloaded! Please close this window and restart Artroom.

REM Clean up the zip file and the temp vbs script if needed
REM del %ZIP_PATH%
REM del %temp%\temp.vbs

pause
