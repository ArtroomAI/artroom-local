@echo off

title Artroom Installer

REM Grab command line arguments and use default values if none are provided

REM Default values
set "URL="
set "ZIP_FILE="
set "DOWNLOAD_FOLDER=%UserProfile%\Downloads"

REM If DESTINATION_FOLDER is provided as an argument
if not "%~1"=="" (
    set "DESTINATION_FOLDER=%~1"
) else (
    set "DESTINATION_FOLDER=%UserProfile%\artroom"
)

REM If GPU_TYPE is provided as the second argument
if not "%~2"=="" (
    set "GPU_TYPE=%~2"
) else (
    set "GPU_TYPE=Unknown"
)

REM Detect GPU type only if not provided
if "%GPU_TYPE%"=="Unknown" (
    wmic path win32_VideoController get name | findstr /i "AMD" >nul
    if %errorlevel% equ 0 (
        set "GPU_TYPE=AMD"
    ) 

    wmic path win32_VideoController get name | findstr /i "NVIDIA" >nul
    if %errorlevel% equ 0 (
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

@echo off

REM Using 7za.exe for faster extraction
.\7z.exe e "%ZIP_PATH%" -o"%DESTINATION_FOLDER%\artroom_backend" -y

if %errorlevel% neq 0 (
    echo An error occurred while extracting the file. Please check your system and try again.
    exit /b
)

echo Artroom Successfully downloaded! Please close this window and restart Artroom.

REM Clean up the zip file and the temp vbs script if needed
del %ZIP_PATH%
REM del %temp%\temp.vbs

pause