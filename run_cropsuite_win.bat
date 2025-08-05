@echo off
setlocal enabledelayedexpansion
echo =====================
echo    CropSuite Setup
echo =====================

set "VENV_DIR=.cropsuite-venv"

if not exist ".cropsuite-venv\" (
ping -n 1 8.8.8.8 >nul 2>nul
if %errorlevel% neq 0 (
    echo No internet connection detected.
    echo Checking if virtual environment exists...
        
    :: If venv exists, activate and start CropSuite_GUI.py
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        echo Virtual environment found. Activating it...
        call "%VENV_DIR%\Scripts\activate.bat"
        echo Starting CropSuite_GUI.py...
        call python CropSuite_GUI.py
    ) else (
        echo Virtual environment not found. Please check your setup.
    )
    goto :eof
)

python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed. Installing Python 3.13 using winget...
    winget install --id Python.Python.3.13 --source winget
) else (
    for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do (
        set "FULL_VERSION=%%i"
    )
    for /f "tokens=1,2 delims=." %%a in ("!FULL_VERSION!") do (
        set "PYTHON_MAJOR=%%a"
        set "PYTHON_MINOR=%%b"
    )
    set "PYTHON_VERSION=!PYTHON_MAJOR!.!PYTHON_MINOR!"
    echo Detected Python version: !PYTHON_VERSION!

    if "%PYTHON_VERSION%" neq "3.9" if "%PYTHON_VERSION%" neq "3.10" if "%PYTHON_VERSION%" neq "3.11" if "%PYTHON_VERSION%" neq "3.12" if "%PYTHON_VERSION%" neq "3.13" (
        echo Incorrect Python version %PYTHON_VERSION%. Installing Python 3.13 using winget...
        winget install --id Python.Python.3.13 --source winget
    ) else (
        echo Python %PYTHON_VERSION% is already installed.
    )
)

if not exist "%VENV_DIR%" (
    echo Creating a virtual environment in %VENV_DIR%...
    python -m venv "%VENV_DIR%"

call "%VENV_DIR%\Scripts\activate.bat"

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing required packages...
python -m pip install numpy==1.26.4 ^
                            scipy==1.15.2 ^
                            matplotlib==3.10.1 ^
                            rasterio==1.4.3 ^
                            xarray==2025.3.1 ^
                            numba==0.61.2 ^
                            rio-cogeo==5.4.1 ^
                            cartopy==0.24.1 ^
                            dask==2025.3.0 ^
                            "dask[distributed]" ^
                            netCDF4==1.7.2 ^
                            pillow==11.1.0 ^
                            psutil==7.0.0 ^
                            pyproj==3.7.1 ^
                            scikit-image==0.25.2 ^
                            tk
			    pyzipper
)
call "%VENV_DIR%\Scripts\activate.bat"

set "TARGET_FILE=%USERPROFILE%\.local\share\cartopy\shapefiles\natural_earth\physical\ne_110m_lakes.shp"

IF NOT EXIST "%TARGET_FILE%" (
set "CULTURAL_DIR=%USERPROFILE%\.local\share\cartopy\shapefiles\natural_earth\cultural"
set "PHYSICAL_DIR=%USERPROFILE%\.local\share\cartopy\shapefiles\natural_earth\physical"

mkdir "%CULTURAL_DIR%" 2>nul
mkdir "%PHYSICAL_DIR%" 2>nul

call :download_and_extract "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_boundary_lines_land.zip" "%CULTURAL_DIR%"
call :download_and_extract "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_boundary_lines_land.zip" "%CULTURAL_DIR%"
call :download_and_extract "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_boundary_lines_land.zip" "%CULTURAL_DIR%"

call :download_and_extract "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip" "%PHYSICAL_DIR%"
call :download_and_extract "https://naciscdn.org/naturalearth/10m/physical/ne_10m_ocean.zip" "%PHYSICAL_DIR%"
call :download_and_extract "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip" "%PHYSICAL_DIR%"
call :download_and_extract "https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip" "%PHYSICAL_DIR%"
call :download_and_extract "https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_lake_centerlines.zip" "%PHYSICAL_DIR%"
call :download_and_extract "https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip" "%PHYSICAL_DIR%"
call :download_and_extract "https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip" "%PHYSICAL_DIR%"
call :download_and_extract "https://naciscdn.org/naturalearth/110m/physical/ne_110m_ocean.zip" "%PHYSICAL_DIR%"
call :download_and_extract "https://naciscdn.org/naturalearth/110m/physical/ne_110m_lakes.zip" "%PHYSICAL_DIR%"
)
)
call python CropSuite_GUI.py

goto :eof

:download_and_extract
setlocal
set URL=%~1
set TARGET_DIR=%~2
for %%F in ("%URL%") do set FILENAME=%%~nxF
echo Downloading %FILENAME%...
curl -L -s -o "%TARGET_DIR%\%FILENAME%" "%URL%"
echo Extracting %FILENAME%...
powershell -Command "Expand-Archive -Path '%TARGET_DIR%\%FILENAME%' -DestinationPath '%TARGET_DIR%' -Force"
del "%TARGET_DIR%\%FILENAME%"
endlocal