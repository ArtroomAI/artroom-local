@echo off

set install_ref_file=%UserProfile%\AppData\Local\artroom_install.log

if exist %install_ref_file% (
    set /p install_path_raw=<%install_ref_file%
) else (
    echo Cant find %install_ref_file%, defaulting to user profile area
    set install_path_raw=%UserProfile%
)

set install_path=%install_path_raw%\artroom

echo raw install path: %install_path_raw%
echo install path: %install_path%
echo install ref path: %install_ref_file%

set install_conda=%install_path%\miniconda3\Scripts\conda.exe
set install_activate=%install_path%\miniconda3\condabin\activate.bat
set install_ldm=%install_path%\miniconda3\envs\artroom-ldm

echo ------------------------------
echo Setting up Artroom App. Please wait until installer finished before using the Artroom App.
echo Create artroom Directory...

mkdir %install_path% > artroomlog.txt & type artroomlog.txt

echo ------------------------------
echo Installing CONDA Environment...
echo If it freezes at any point, please try pressing enter or trying again.
echo (Doesn't always happen, but sometimes could get hung up on something)


set command=%install_conda% info --envs
FOR /F "tokens=2" %%F IN ('%command%') DO SET var=%%F

if exist %var% (
    echo miniconda3 already found in Environment PATH
) else (
    echo Installing localized miniconda
    echo Setting up CONDA Environment...
)