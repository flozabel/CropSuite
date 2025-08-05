#!/bin/bash
set -e

echo "====================="
echo "   CropSuite Setup   "
echo "====================="
VENV_DIR=".cropsuite-venv"
if [ ! -d ".cropsuite-venv" ]; then

# ---------------------------------------------------
# Check for internet connection
# ---------------------------------------------------
echo "Checking for internet connection..."
if curl -s --head http://www.google.com | grep "200 OK" > /dev/null; then
    ONLINE=true
    echo "Internet connection detected."
else
    ONLINE=false
    echo "No internet connection. Skipping installations and downloads."
fi

# ---------------------------------------------------
# Check if Python is usable
# ---------------------------------------------------
echo "Checking Python..."
if ! python3 --version > /dev/null 2>&1; then
    echo "Python not found or not properly installed."
    if [ "$ONLINE" = false ]; then
        echo "Cannot install Python without internet. Exiting."
        exit 1
    fi
    echo "Attempting to install Python 3.13 using Homebrew..."

    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    brew install python@3.13
    echo "Python installation complete."
    echo "Re-running setup..."
    exec "$0"
    exit 0
fi

# ---------------------------------------------------
# Confirm usable Python version
# ---------------------------------------------------
PY_VER=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo "$PY_VER" | cut -d. -f1)
MINOR=$(echo "$PY_VER" | cut -d. -f2)
SHORT_VER="$MAJOR.$MINOR"

echo "Detected Python version: $SHORT_VER"

if [[ ! "$SHORT_VER" =~ ^(3\.9|3\.10|3\.11|3\.12|3\.13)$ ]]; then
    echo "WARNING: Python $SHORT_VER may be incompatible."
    echo "Recommended: 3.9 - 3.13"
    read -p "Press Enter to continue anyway..."
fi

# ---------------------------------------------------
# Create and activate a virtual environment
# ---------------------------------------------------


if [ ! -d "$VENV_DIR" ]; then
    echo "Creating a virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------
# Only do the following if online
# ---------------------------------------------------
if [ "$ONLINE" = true ]; then

    echo
    echo "Upgrading pip..."
    python -m pip install --upgrade pip

    echo
    echo "Installing required packages..."
    pip install \
        numpy==1.26.4 \
        scipy==1.15.2 \
        matplotlib==3.10.1 \
        rasterio==1.4.3 \
        xarray==2025.3.1 \
        numba==0.61.2 \
        rio-cogeo==5.4.1 \
        cartopy==0.24.1 \
        dask==2025.3.0 \
        "dask[distributed]" \
        netCDF4==1.7.2 \
        pillow==11.1.0 \
        psutil==7.0.0 \
        pyproj==3.7.1 \
        scikit-image==0.25.2 \
        tk \
				pyzipper

    echo
    echo "======================================="
    echo "   Downloading Cartopy Shapefiles..."
    echo "======================================="

    CULTURAL_DIR="$HOME/.local/share/cartopy/shapefiles/natural_earth/cultural"
    PHYSICAL_DIR="$HOME/.local/share/cartopy/shapefiles/natural_earth/physical"

    mkdir -p "$CULTURAL_DIR"
    mkdir -p "$PHYSICAL_DIR"

    download_and_extract() {
        local url=$1
        local target_dir=$2
        local filename=$(basename "$url")

        echo "Downloading $filename..."
        curl -s -L -o "$target_dir/$filename" "$url"

        echo "Extracting $filename..."
        unzip -o "$target_dir/$filename" -d "$target_dir" > /dev/null
        rm "$target_dir/$filename"
    }

    # Cultural shapefiles
    download_and_extract "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_boundary_lines_land.zip" "$CULTURAL_DIR"
    download_and_extract "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_boundary_lines_land.zip" "$CULTURAL_DIR"
    download_and_extract "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_boundary_lines_land.zip" "$CULTURAL_DIR"

    # Physical shapefiles
    download_and_extract "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip" "$PHYSICAL_DIR"
    download_and_extract "https://naciscdn.org/naturalearth/10m/physical/ne_10m_ocean.zip" "$PHYSICAL_DIR"
    download_and_extract "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip" "$PHYSICAL_DIR"
    download_and_extract "https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip" "$PHYSICAL_DIR"
    download_and_extract "https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_lake_centerlines.zip" "$PHYSICAL_DIR"
    download_and_extract "https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip" "$PHYSICAL_DIR"
    download_and_extract "https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip" "$PHYSICAL_DIR"
    download_and_extract "https://naciscdn.org/naturalearth/110m/physical/ne_110m_ocean.zip" "$PHYSICAL_DIR"
    download_and_extract "https://naciscdn.org/naturalearth/110m/physical/ne_110m_lakes.zip" "$PHYSICAL_DIR"

    echo "Cartopy shapefiles downloaded and installed."
else
    echo "Offline mode: Skipping package installation and shapefile download."
fi

fi
source "$VENV_DIR/bin/activate"
echo
echo "Setup complete."
echo "Launching CropSuite GUI..."
python CropSuite_GUI.py